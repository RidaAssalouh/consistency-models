import math
import os
import random
import copy
from dataclasses import dataclass, asdict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm

from models.ct_model_utils import UNet

@dataclass
class TeacherConfig:
    data_root: str = "./data"
    output_dir: str = "storage_ablette/diffusion_teacher_fashionmnist"

    image_size: int = 32
    in_channels: int = 1
    batch_size: int = 128
    num_workers: int = 4

    base_channels: int = 64
    channel_mults: Tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    time_emb_dim: int = 256
    dropout: float = 0.0

    sigma_data: float = 0.5
    eps: float = 1e-3
    sigma_max: float = 50.0
    rho: float = 7.0

    epochs: int = 100
    lr: float = 2e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0

    amp: bool = True
    ema_decay: float = 0.99
    seed: int = 42

    log_every: int = 100
    save_every_epochs: int = 10
    sample_every_epochs: int = 5
    num_sample_images: int = 16
    sample_steps: int = 32

    sigma_sample_mode: str = "karras"   # "loguniform" or "karras"
    num_karras_scales: int = 40

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_sd = model.state_dict()
        for k, v in self.ema_model.state_dict().items():
            src = model_sd[k]
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(src.detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(src)


# Teacher model: noise predictor under x_sigma = x0 + sigma * eps

class DiffusionTeacher(nn.Module):
    def __init__(self, backbone: nn.Module, sigma_data: float = 0.5):
        super().__init__()
        self.backbone = backbone
        self.sigma_data = sigma_data

    def c_in(self, sigma: Tensor) -> Tensor:
        return 1.0 / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma: Tensor) -> Tensor:
        # preconditioned noise prediction scaling
        return sigma / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        sigma = sigma.float()
        cin = self.c_in(sigma).view(-1, 1, 1, 1)
        cout = self.c_out(sigma).view(-1, 1, 1, 1)
        raw = self.backbone(cin * x, sigma)
        return cout * raw

    @torch.no_grad()
    def predict_x0(self, x_sigma: Tensor, sigma: Tensor) -> Tensor:
        eps_hat = self(x_sigma, sigma)
        return x_sigma - sigma.view(-1, 1, 1, 1) * eps_hat

def build_dataloader(cfg: TeacherConfig) -> DataLoader:
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


def sample_training_sigmas(
    batch_size: int,
    cfg: TeacherConfig,
    device: torch.device,
) -> Tensor:
    if cfg.sigma_sample_mode == "karras":
        grid = build_karras_sigmas(
            sigma_min=cfg.eps,
            sigma_max=cfg.sigma_max,
            num_scales=cfg.num_karras_scales,
            rho=cfg.rho,
            device=device,
        )
        idx = torch.randint(0, grid.numel(), (batch_size,), device=device)
        return grid[idx]

    if cfg.sigma_sample_mode == "loguniform":
        u = torch.rand(batch_size, device=device)
        log_sigma = math.log(cfg.eps) + u * (math.log(cfg.sigma_max) - math.log(cfg.eps))
        return torch.exp(log_sigma)

    raise ValueError("Unknown sigma_sample_mode: {}".format(cfg.sigma_sample_mode))



@torch.no_grad()
def teacher_euler_sample(
    model: DiffusionTeacher,
    shape: Tuple[int, int, int, int],
    sigma_max: float,
    sigma_min: float,
    steps: int,
    rho: float,
    device: torch.device,
) -> Tensor:
    model.eval()

    sigmas = build_karras_sigmas(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        num_scales=steps,
        rho=rho,
        device=device,
    )

    x = sigma_max * torch.randn(shape, device=device)

    for i in range(len(sigmas) - 1):
        sigma_hi = sigmas[i]
        sigma_lo = sigmas[i + 1]

        sigma_batch_hi = torch.full((shape[0],), sigma_hi.item(), device=device)
        eps_hat = model(x, sigma_batch_hi)
        x0_hat = x - sigma_hi * eps_hat

        x = x0_hat + sigma_lo * eps_hat

    sigma_batch_last = torch.full((shape[0],), sigmas[-1].item(), device=device)
    x0_hat = model.predict_x0(x, sigma_batch_last)
    return x0_hat.clamp(-1.0, 1.0)


@torch.no_grad()
def save_samples(
    model: DiffusionTeacher,
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
    samples = teacher_euler_sample(
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


def save_checkpoint(
    path: str,
    epoch: int,
    model: DiffusionTeacher,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    cfg: TeacherConfig,
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



def train_teacher(cfg: TeacherConfig) -> None:
    seed_everything(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

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

    model = DiffusionTeacher(
        backbone=backbone,
        sigma_data=cfg.sigma_data,
    ).to(device)

    ema = ModelEMA(model, decay=cfg.ema_decay)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    print("Training diffusion teacher on: {}".format(device))
    print("Saving to: {}".format(cfg.output_dir))

    global_step = 0
    best_epoch_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        epoch_loss_sum = 0.0
        epoch_num_steps = 0

        pbar = tqdm(loader, desc="Epoch {}/{}".format(epoch, cfg.epochs))

        for step, (x0, _) in enumerate(pbar, start=1):
            global_step += 1

            x0 = x0.to(device, non_blocking=True)
            bsz = x0.size(0)

            sigma = sample_training_sigmas(
                batch_size=bsz,
                cfg=cfg,
                device=device,
            )

            noise = torch.randn_like(x0)
            x_sigma = x0 + sigma.view(-1, 1, 1, 1) * noise

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                eps_hat = model(x_sigma, sigma)
                loss = F.mse_loss(eps_hat, noise)

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

            if step % cfg.log_every == 0:
                avg_loss = running_loss / cfg.log_every
                pbar.set_postfix(loss="{:.5f}".format(avg_loss))
                running_loss = 0.0

        epoch_avg_loss = epoch_loss_sum / max(epoch_num_steps, 1)
        print("Epoch {} | avg_loss = {:.6f}".format(epoch, epoch_avg_loss))

        if epoch_avg_loss < best_epoch_loss:
            best_epoch_loss = epoch_avg_loss
            save_checkpoint(
                path=os.path.join(cfg.output_dir, "teacher_best.pt"),
                epoch=epoch,
                model=model,
                ema=ema,
                optimizer=optimizer,
                cfg=cfg,
            )

        if epoch % cfg.save_every_epochs == 0 or epoch == cfg.epochs:
            save_checkpoint(
                path=os.path.join(cfg.output_dir, "teacher_epoch_{:03d}.pt".format(epoch)),
                epoch=epoch,
                model=model,
                ema=ema,
                optimizer=optimizer,
                cfg=cfg,
            )

        if epoch % cfg.sample_every_epochs == 0 or epoch == cfg.epochs:
            save_samples(
                model=ema.ema_model,
                save_path=os.path.join(cfg.output_dir, "samples_epoch_{:03d}.png".format(epoch)),
                num_images=cfg.num_sample_images,
                image_size=cfg.image_size,
                in_channels=cfg.in_channels,
                sigma_max=cfg.sigma_max,
                sigma_min=cfg.eps,
                steps=cfg.sample_steps,
                rho=cfg.rho,
                device=cfg.device,
            )

    torch.save(
        {
            "model": model.state_dict(),
            "ema_model": ema.ema_model.state_dict(),
            "config": asdict(cfg),
        },
        os.path.join(cfg.output_dir, "teacher_final.pt"),
    )

    print("Done.")


if __name__ == "__main__":
    cfg = TeacherConfig()
    train_teacher(cfg)
    