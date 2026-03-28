import copy
import math
import os
import random
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    _HAS_TORCHMETRICS = True
except Exception:
    FrechetInceptionDistance = None
    _HAS_TORCHMETRICS = False

from models.ct_model_utils import UNet


# Config
@dataclass
class Config:
    data_root: str = "./data"
    output_dir: str = "storage/ct_fashionmnist_"

    image_size: int = 32
    in_channels: int = 1
    batch_size: int = 128
    num_workers: int = 4

    base_channels: int = 64
    channel_mults: Tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    time_emb_dim: int = 256
    dropout: float = 0.0

    sigma_data: float = 0.5 # scaling of cskip and couot
    eps: float = 1e-3 # epsilon
    sigma_max: float = 50.0 # T

    epochs: int = 100
    lr: float = 2e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0

    ema_decay: float = 0.99

    loss_type: str = "l1"   # "l1" or "l2"
    use_karras_schedule: bool = True
    num_scales: int = 40 # N
    rho: float = 7.0 # rho of karras

    amp: bool = True
    seed: int = 42
    log_every: int = 100
    sample_every_epochs: int = 1
    save_every_epochs: int = 5
    num_sample_images: int = 16

    sample_steps: int = 1

    # ---- metrics / FID ----
    metrics_filename: str = "training_metrics.json"
    fid_every_epochs: int = 5
    fid_num_real: int = 2048
    fid_num_fake: int = 2048
    fid_batch_size: int = 128

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# Utils

    
def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



# Consistency model with boundary condition

class ConsistencyModel(nn.Module):
    def __init__(self, backbone: nn.Module, sigma_data: float = 0.5, eps: float = 1e-3):
        super().__init__()
        self.backbone = backbone
        self.sigma_data = sigma_data
        self.eps = eps

    def c_skip(self, sigma: Tensor) -> Tensor:
        return (self.sigma_data ** 2) / (((sigma - self.eps) ** 2) + self.sigma_data ** 2)

    def c_out(self, sigma: Tensor) -> Tensor:
        return self.sigma_data * (sigma - self.eps) / torch.sqrt(self.sigma_data ** 2 + sigma ** 2)

    def c_in(self, sigma: Tensor) -> Tensor:
        return 1.0 / torch.sqrt(self.sigma_data ** 2 + sigma ** 2)

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        sigma = sigma.float().clamp(min=self.eps)

        cskip = self.c_skip(sigma).view(-1, 1, 1, 1)
        cout = self.c_out(sigma).view(-1, 1, 1, 1)
        cin = self.c_in(sigma).view(-1, 1, 1, 1)

        raw = self.backbone(cin * x, sigma)
        return cskip * x + cout * raw


# EMA target

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.ema_model.state_dict().items():
            src = msd[k]
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(src.detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(src)

    @torch.no_grad()
    def __call__(self, x: Tensor, sigma: Tensor) -> Tensor:
        return self.ema_model(x, sigma)


# Data
def build_dataloader(cfg: Config) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset = datasets.FashionMNIST(
        root=cfg.data_root,
        train=True,
        download=True,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )


# Noise schedule helpers
def build_karras_sigmas(
    sigma_min: float,
    sigma_max: float,
    num_scales: int,
    rho: float,
    device: torch.device,
) -> Tensor:
    ramp = torch.linspace(0, 1, num_scales, device=device)
    inv_rho_min = sigma_min ** (1.0 / rho)
    inv_rho_max = sigma_max ** (1.0 / rho)
    sigmas = (inv_rho_max + ramp * (inv_rho_min - inv_rho_max)) ** rho
    return sigmas


def sample_adjacent_sigmas(
    bsz: int,
    sigmas: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Pick adjacent pair sigma_i > sigma_{i+1}.
    Returns:
        sigma_hi, sigma_lo
    """
    idx = torch.randint(0, len(sigmas) - 1, (bsz,), device=sigmas.device)
    sigma_hi = sigmas[idx]
    sigma_lo = sigmas[idx + 1]
    return sigma_hi, sigma_lo


# Sampling helper
@torch.no_grad()
def ct_multistep_sample(
    model: ConsistencyModel,
    shape: Tuple[int, int, int, int],
    sigma_max: float,
    sigma_min: float,
    steps: int,
    rho: float,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """
    Stochastic multistep consistency sampling.

    Returns:
        final_samples: (B, C, H, W)
        all_states:    (steps, B, C, H, W)
    """
    model.eval()

    sigmas = build_karras_sigmas(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        num_scales=steps,
        rho=rho,
        device=device,
    )

    # initial noisy sample at sigma_max
    x_hat = sigma_max * torch.randn(shape, device=device)

    states = []

    # first denoising step
    sigma_batch = torch.full((shape[0],), sigmas[0].item(), device=device)
    x = model(x_hat, sigma_batch)
    states.append(x.detach().cpu())

    # stochastic multistep refinement
    for sigma in sigmas[1:]:
        z = torch.randn_like(x)
        noise_std = torch.sqrt(torch.clamp(sigma**2 - sigma_min**2, min=0.0))
        x_hat = x + noise_std * z

        sigma_batch = torch.full((shape[0],), sigma.item(), device=device)
        x = model(x_hat, sigma_batch)
        states.append(x.detach().cpu())

    all_states = torch.stack(states, dim=0)
    return x.clamp(-1.0, 1.0), all_states


@torch.no_grad()
def save_samples(
    model: ConsistencyModel,
    save_path: str,
    num_images: int,
    image_size: int,
    in_channels: int,
    sigma_max: float,
    sigma_min: float,
    steps: int,
    rho: float,
    device: str,
) -> None:
    samples, _ = ct_multistep_sample(
        model=model,
        shape=(num_images, in_channels, image_size, image_size),
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        steps=steps,
        rho=rho,
        device=torch.device(device),
    )
    samples = (samples + 1.0) / 2.0
    utils.save_image(samples, save_path, nrow=int(math.sqrt(num_images)))


def save_metrics_json(metrics: list, path: str) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    os.replace(tmp_path, path)


def to_fid_uint8(x: Tensor) -> Tensor:
    """
    Convert model images from [-1, 1] grayscale/RGB to uint8 BCHW in [0,255],
    and replicate grayscale to 3 channels for Inception/FID.
    """
    x = x.detach().clamp(-1.0, 1.0)
    x = (x + 1.0) / 2.0
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    x = (255.0 * x).clamp(0, 255).to(torch.uint8)
    return x


@torch.no_grad()
def compute_fid(
    model: ConsistencyModel,
    real_loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> float:
    if not _HAS_TORCHMETRICS:
        print("torchmetrics not installed: skipping FID.")
        return None

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    model.eval()

    # real images
    num_real = 0
    for x_real, _ in real_loader:
        x_real = x_real.to(device, non_blocking=True)
        x_real = to_fid_uint8(x_real)
        fid.update(x_real, real=True)
        num_real += x_real.size(0)
        if num_real >= cfg.fid_num_real:
            break

    # fake images
    num_fake = 0
    while num_fake < cfg.fid_num_fake:
        bsz = min(cfg.fid_batch_size, cfg.fid_num_fake - num_fake)
        x_fake, _ = ct_multistep_sample(
            model=model,
            shape=(bsz, cfg.in_channels, cfg.image_size, cfg.image_size),
            sigma_max=cfg.sigma_max,
            sigma_min=cfg.eps,
            steps=cfg.sample_steps,
            rho=cfg.rho,
            device=device,
        )
        x_fake = to_fid_uint8(x_fake)
        fid.update(x_fake, real=False)
        num_fake += x_fake.size(0)

    return float(fid.compute().item())


def save_checkpoint(
    path: str,
    epoch: int,
    model: ConsistencyModel,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema_model": ema.ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(cfg),
        },
        path,
    )


# Train
def train(cfg: Config) -> None:
    seed_everything(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    metrics_path = os.path.join(cfg.output_dir, cfg.metrics_filename)
    metrics_history = []

    device = torch.device(cfg.device)
    loader = build_dataloader(cfg)

    backbone = UNet(
        in_channels=cfg.in_channels,
        base_channels=cfg.base_channels,
        channel_mults=cfg.channel_mults,
        num_res_blocks=cfg.num_res_blocks,
        time_emb_dim=cfg.time_emb_dim,
        dropout=cfg.dropout,
    ).to(device)

    model = ConsistencyModel(
        backbone=backbone,
        sigma_data=cfg.sigma_data,
        eps=cfg.eps,
    ).to(device)

    ema = ModelEMA(model, decay=cfg.ema_decay)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    if cfg.use_karras_schedule:
        train_sigmas = build_karras_sigmas(
            sigma_min=cfg.eps,
            sigma_max=cfg.sigma_max,
            num_scales=cfg.num_scales,
            rho=cfg.rho,
            device=device,
        )
    else:
        train_sigmas = torch.linspace(cfg.sigma_max, cfg.eps, cfg.num_scales, device=device)

    print(f"Training on: {device}")
    print(f"Saving to:   {cfg.output_dir}")

    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        running_loss = 0.0
        epoch_loss_sum = 0.0
        epoch_num_steps = 0

        for step, (x0, _) in enumerate(pbar, start=1):
            global_step += 1
            x0 = x0.to(device, non_blocking=True)
            bsz = x0.size(0)

            sigma_hi, sigma_lo = sample_adjacent_sigmas(bsz, train_sigmas)

            noise = torch.randn_like(x0)
            x_hi = x0 + sigma_hi.view(-1, 1, 1, 1) * noise
            x_lo = x0 + sigma_lo.view(-1, 1, 1, 1) * noise

            with torch.no_grad():
                target = ema(x_lo, sigma_lo)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                pred = model(x_hi, sigma_hi)
                if cfg.loss_type == "l2":
                    loss = F.mse_loss(pred, target)
                else:
                    loss = F.l1_loss(pred, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)

            loss_value = float(loss.item())
            running_loss += loss_value
            epoch_loss_sum += loss_value
            epoch_num_steps += 1

            # store per-iteration training loss
            metrics_history.append({
                "type": "train_iteration",
                "epoch": epoch,
                "step_in_epoch": step,
                "global_step": global_step,
                "loss": loss_value,
            })

            if step % cfg.log_every == 0:
                avg_loss = running_loss / cfg.log_every
                pbar.set_postfix(loss=f"{avg_loss:.5f}")
                running_loss = 0.0
                save_metrics_json(metrics_history, metrics_path)

        epoch_avg_loss = epoch_loss_sum / max(epoch_num_steps, 1)

        epoch_metrics = {
            "type": "train_epoch",
            "epoch": epoch,
            "global_step": global_step,
            "avg_loss": float(epoch_avg_loss),
        }

        if epoch % cfg.save_every_epochs == 0 or epoch == cfg.epochs:
            save_checkpoint(
                path=os.path.join(cfg.output_dir, f"checkpoint_epoch_{epoch:03d}.pt"),
                epoch=epoch,
                model=model,
                ema=ema,
                optimizer=optimizer,
                cfg=cfg,
            )

        if epoch % cfg.sample_every_epochs == 0 or epoch == cfg.epochs:
            save_samples(
                model=ema.ema_model,
                save_path=os.path.join(cfg.output_dir, f"samples_epoch_{epoch:03d}.png"),
                num_images=cfg.num_sample_images,
                image_size=cfg.image_size,
                in_channels=cfg.in_channels,
                sigma_max=cfg.sigma_max,
                sigma_min=cfg.eps,
                steps=cfg.sample_steps,
                rho=cfg.rho,
                device=cfg.device,
            )

        # compute and store FID at epoch level
        if epoch % cfg.fid_every_epochs == 0 or epoch == cfg.epochs:
            fid_value = compute_fid(
                model=ema.ema_model,
                real_loader=loader,
                cfg=cfg,
                device=device,
            )
            epoch_metrics["fid"] = fid_value
            print(f"Epoch {epoch}: avg_loss={epoch_avg_loss:.6f}, fid={fid_value}")

        metrics_history.append(epoch_metrics)
        save_metrics_json(metrics_history, metrics_path)

    torch.save(
        {
            "model": model.state_dict(),
            "ema_model": ema.ema_model.state_dict(),
            "config": asdict(cfg),
        },
        os.path.join(cfg.output_dir, "ct_model_final.pt"),
    )

    print("Done.")




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    for field_name, field in Config.__dataclass_fields__.items():
        default = getattr(Config(), field_name)
        if isinstance(default, tuple):
            parser.add_argument(f"--{field_name}", type=type(default[0]), nargs="+", default=list(default))
        elif isinstance(default, bool):
            parser.add_argument(f"--{field_name}", action="store_true", default=default)
        else:
            parser.add_argument(f"--{field_name}", type=type(default), default=default)
    args = parser.parse_args()
    cfg = Config(**{k: tuple(v) if isinstance(getattr(Config(), k), tuple) else v
                    for k, v in vars(args).items()})
    train(cfg)