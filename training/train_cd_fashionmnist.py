import copy
import math
import os
import json
import random
from dataclasses import dataclass, asdict
from typing import Tuple

import numpy as np
from scipy.linalg import sqrtm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm

from models.cd_model_utils import UNet
from models.teacher_utils import UNet as TeacherUNet


# Config
@dataclass
class CDConfig:
    data_root: str = "./data"
    output_dir: str = "../outputs/cd_fashionmnist_l1"

    teacher_ckpt_path: str = "consistency_models_ckpts/teacher_epoch_060.pt"
    teacher_use_ema: bool = True

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
    num_scales: int = 40

    epochs: int = 100
    lr: float = 2e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0

    # Two separate EMA decays:
    #   ema_decay_target : decay for the CD target network θ⁻  (Algorithm 2, Eq. 8)
    #   ema_decay_eval   : decay for the evaluation / sampling EMA (separate)
    ema_decay_target: float = 0.99
    ema_decay_eval: float = 0.999

    loss_type: str = "l1"
    amp: bool = True
    seed: int = 42
    log_every: int = 100
    save_every_epochs: int = 10
    sample_every_epochs: int = 1
    num_sample_images: int = 16

    sample_steps: int = 1
    metrics_filename: str = "training_metrics.json"
    fid_every_epochs: int = 5
    fid_num_real: int = 2048
    fid_num_fake: int = 2048
    fid_batch_size: int = 128

    extractor_ckpt: str = "consistency_models_ckpts/fashion_extractor.pt"
    extractor_feat_dim: int = 512

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# Utils

def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_metrics_json(metrics: list, path: str) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    os.replace(tmp_path, path)


class DiffusionTeacher(nn.Module):
    def __init__(self, backbone: nn.Module, sigma_data: float = 0.5):
        super().__init__()
        self.backbone = backbone
        self.sigma_data = sigma_data

    def c_in(self, sigma: Tensor) -> Tensor:
        return 1.0 / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma: Tensor) -> Tensor:
        return sigma / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        """Returns eps_hat (predicted noise)."""
        sigma = sigma.float()
        cin  = self.c_in(sigma).view(-1, 1, 1, 1)
        cout = self.c_out(sigma).view(-1, 1, 1, 1)
        raw  = self.backbone(cin * x, sigma)
        return cout * raw

    @torch.no_grad()
    def predict_x0(self, x_sigma: Tensor, sigma: Tensor) -> Tensor:
        """Tweedie denoising: x0_hat = x_sigma - sigma * eps_hat."""
        eps_hat = self(x_sigma, sigma)
        return x_sigma - sigma.view(-1, 1, 1, 1) * eps_hat


class ConsistencyModel(nn.Module):
    def __init__(self, backbone: nn.Module, sigma_data: float = 0.5, eps: float = 1e-3):
        super().__init__()
        self.backbone = backbone
        self.sigma_data = sigma_data
        self.eps = eps

    def c_skip(self, sigma: Tensor) -> Tensor:
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma: Tensor) -> Tensor:
        return sigma * self.sigma_data / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        sigma  = sigma.float().clamp(min=self.eps)
        c_skip = self.c_skip(sigma).view(-1, 1, 1, 1)
        c_out  = self.c_out(sigma).view(-1, 1, 1, 1)
        F_out  = self.backbone(x, sigma)
        return c_skip * x + c_out * F_out


# Feature extractor
class FashionMNISTFeatureExtractor(nn.Module):
    NUM_CLASSES = 10

    def __init__(self, feat_dim: int = 512):
        super().__init__()
        self.feat_dim = feat_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, feat_dim),
            nn.LayerNorm(feat_dim),
        )
        self.classifier = nn.Linear(feat_dim, self.NUM_CLASSES)

    def forward_features(self, x: Tensor) -> Tensor:
        return F.normalize(self.embedding(self.encoder(x)), dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.forward_features(x))


# EMA — used only for the evaluation / sampling model
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
def build_dataloader(cfg: CDConfig) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = datasets.FashionMNIST(
        root=cfg.data_root, train=True, download=True, transform=transform,
    )
    return DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )



def build_karras_sigmas(
    sigma_min: float, sigma_max: float,
    num_scales: int, rho: float, device: torch.device,
) -> Tensor:
    ramp = torch.linspace(0, 1, num_scales, device=device)
    inv_rho_min = sigma_min ** (1.0 / rho)
    inv_rho_max = sigma_max ** (1.0 / rho)
    sigmas = (inv_rho_max + ramp * (inv_rho_min - inv_rho_max)) ** rho
    return sigmas  # descending


def sample_adjacent_sigmas(bsz: int, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
    # sigmas is descending → sigmas[idx] > sigmas[idx+1]
    idx = torch.randint(0, len(sigmas) - 1, (bsz,), device=sigmas.device)
    sigma_hi = sigmas[idx]       # higher noise level  (t_{n+1})
    sigma_lo = sigmas[idx + 1]   # lower noise level   (t_n)
    return sigma_hi, sigma_lo


@torch.no_grad()
def teacher_transport_to_lower_sigma(
    teacher: DiffusionTeacher,
    x_hi: Tensor,
    sigma_hi: Tensor,
    sigma_lo: Tensor,
    eps_floor: float,
) -> Tensor:
    sigma_hi = sigma_hi.float().clamp(min=eps_floor)
    sigma_lo = sigma_lo.float().clamp(min=eps_floor)

    # teacher.forward returns eps_hat (noise prediction)
    eps_hat = teacher(x_hi, sigma_hi)

    # Euler step: x_lo = x_hi + (sigma_lo - sigma_hi) * eps_hat
    x_lo = x_hi + (sigma_lo - sigma_hi).view(-1, 1, 1, 1) * eps_hat
    return x_lo


# Sampling
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
    model.eval()
    sigmas = build_karras_sigmas(
        sigma_min=sigma_min, sigma_max=sigma_max,
        num_scales=steps, rho=rho, device=device,
    )

    x_hat = sigma_max * torch.randn(shape, device=device)
    states = []

    sigma_batch = torch.full((shape[0],), sigmas[0].item(), device=device)
    x = model(x_hat, sigma_batch)
    states.append(x.detach().cpu())

    for sigma in sigmas[1:]:
        z = torch.randn_like(x)
        noise_std = torch.sqrt(torch.clamp(sigma ** 2 - sigma_min ** 2, min=0.0))
        x_hat = x + noise_std * z
        sigma_batch = torch.full((shape[0],), sigma.item(), device=device)
        x = model(x_hat, sigma_batch)
        states.append(x.detach().cpu())

    return x.clamp(-1.0, 1.0), torch.stack(states, dim=0)


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
        sigma_max=sigma_max, sigma_min=sigma_min,
        steps=steps, rho=rho, device=torch.device(device),
    )
    samples = (samples + 1.0) / 2.0
    utils.save_image(samples, save_path, nrow=int(math.sqrt(num_images)))


# FID helpers

def load_feature_extractor(cfg: CDConfig, device: torch.device) -> FashionMNISTFeatureExtractor:
    extractor = FashionMNISTFeatureExtractor(feat_dim=cfg.extractor_feat_dim).to(device)
    if not os.path.exists(cfg.extractor_ckpt):
        raise FileNotFoundError(f"Feature extractor checkpoint not found: {cfg.extractor_ckpt}")
    extractor.load_state_dict(torch.load(cfg.extractor_ckpt, map_location=device))
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad_(False)
    print(f"Loaded feature extractor from: {cfg.extractor_ckpt}")
    return extractor


@torch.no_grad()
def extract_features_from_loader(extractor, loader, device, max_samples) -> np.ndarray:
    feats, seen = [], 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        feats.append(extractor.forward_features(x).cpu().numpy())
        seen += x.size(0)
        if seen >= max_samples:
            break
    return np.concatenate(feats, axis=0)[:max_samples]


@torch.no_grad()
def extract_features_from_generated_batches(extractor, batches, device, max_samples) -> np.ndarray:
    feats, seen = [], 0
    for x in batches:
        x = x.to(device, non_blocking=True)
        feats.append(extractor.forward_features(x).cpu().numpy())
        seen += x.size(0)
        if seen >= max_samples:
            break
    return np.concatenate(feats, axis=0)[:max_samples]


def frechet_distance(mu1, cov1, mu2, cov2) -> float:
    diff = mu1 - mu2
    covmean, _ = sqrtm(cov1 @ cov2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov1 + cov2 - 2.0 * covmean))


def compute_domain_fid_from_features(real_feats, fake_feats) -> float:
    mu_r, mu_f = real_feats.mean(0), fake_feats.mean(0)
    cov_r = np.cov(real_feats, rowvar=False)
    cov_f = np.cov(fake_feats, rowvar=False)
    return frechet_distance(mu_r, cov_r, mu_f, cov_f)


@torch.no_grad()
def compute_domain_fid(model, extractor, real_feats, cfg, device) -> float:
    generated_batches, generated = [], 0
    while generated < cfg.fid_num_fake:
        bsz = min(cfg.fid_batch_size, cfg.fid_num_fake - generated)
        x_fake, _ = ct_multistep_sample(
            model=model,
            shape=(bsz, cfg.in_channels, cfg.image_size, cfg.image_size),
            sigma_max=cfg.sigma_max, sigma_min=cfg.eps,
            steps=cfg.sample_steps, rho=cfg.rho, device=device,
        )
        generated_batches.append(x_fake.cpu())
        generated += bsz
    fake_feats = extract_features_from_generated_batches(
        extractor, generated_batches, device, cfg.fid_num_fake
    )
    return compute_domain_fid_from_features(real_feats, fake_feats)


# Checkpointing

def save_checkpoint(path, epoch, student, ema_eval, optimizer, cfg):
    torch.save({
        "epoch": epoch,
        "model": student.state_dict(),
        "ema_model": ema_eval.ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(cfg),
    }, path)


def load_teacher(cfg: CDConfig, device: torch.device) -> DiffusionTeacher:
    backbone = TeacherUNet(
        in_channels=cfg.in_channels,
        base_channels=cfg.base_channels,
        channel_mults=cfg.channel_mults,
        num_res_blocks=cfg.num_res_blocks,
        time_emb_dim=cfg.time_emb_dim,
        dropout=cfg.dropout,
    ).to(device)

    teacher = DiffusionTeacher(backbone=backbone, sigma_data=cfg.sigma_data).to(device)

    ckpt = torch.load(cfg.teacher_ckpt_path, map_location=device)
    if cfg.teacher_use_ema:
        if "ema_model" not in ckpt:
            raise KeyError("Teacher checkpoint does not contain 'ema_model'.")
        teacher.load_state_dict(ckpt["ema_model"])
        print(f"Loaded EMA teacher from: {cfg.teacher_ckpt_path}")
    else:
        if "model" not in ckpt:
            raise KeyError("Teacher checkpoint does not contain 'model'.")
        teacher.load_state_dict(ckpt["model"])
        print(f"Loaded raw teacher from: {cfg.teacher_ckpt_path}")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


# Main training loop 
def train_cd(cfg: CDConfig) -> None:
    seed_everything(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    metrics_path = os.path.join(cfg.output_dir, cfg.metrics_filename)
    metrics_history = []

    device = torch.device(cfg.device)
    loader = build_dataloader(cfg)

    # ── Feature extractor + real features for FID ─────────
    extractor = load_feature_extractor(cfg, device=device)
    real_feats = extract_features_from_loader(
        extractor=extractor, loader=loader, device=device, max_samples=cfg.fid_num_real,
    )
    print(f"Precomputed real features: {real_feats.shape}")

    teacher = load_teacher(cfg, device=device)

    student_backbone = UNet(
        in_channels=cfg.in_channels,
        base_channels=cfg.base_channels,
        channel_mults=cfg.channel_mults,
        num_res_blocks=cfg.num_res_blocks,
        time_emb_dim=cfg.time_emb_dim,
        dropout=cfg.dropout,
    ).to(device)
    student_backbone.load_state_dict(teacher.backbone.state_dict())

    student = ConsistencyModel(
        backbone=student_backbone, sigma_data=cfg.sigma_data, eps=cfg.eps,
    ).to(device)

    cm_target = copy.deepcopy(student).eval()
    for p in cm_target.parameters():
        p.requires_grad_(False)

    ema_eval = ModelEMA(student, decay=cfg.ema_decay_eval)

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))


    train_sigmas = build_karras_sigmas(
        sigma_min=cfg.eps, sigma_max=cfg.sigma_max,
        num_scales=cfg.num_scales, rho=cfg.rho, device=device,
    )

    print(f"Training CD on: {device}")
    print(f"Teacher: {cfg.teacher_ckpt_path}")
    print(f"Saving to: {cfg.output_dir}")

    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        student.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        running_loss = 0.0
        epoch_loss_sum = 0.0
        epoch_num_steps = 0

        for step, (x0, _) in enumerate(pbar, start=1):
            global_step += 1
            x0 = x0.to(device, non_blocking=True)
            bsz = x0.size(0)

            # Sample adjacent sigma pair: sigma_hi > sigma_lo
            sigma_hi, sigma_lo = sample_adjacent_sigmas(bsz, train_sigmas)

            # x_{t_{n+1}} ~ N(x0, sigma_hi² I)
            noise = torch.randn_like(x0)
            x_hi = x0 + sigma_hi.view(-1, 1, 1, 1) * noise

            # FIX #2: Euler ODE step — teacher returns eps_hat, not x0
            with torch.no_grad():
                x_lo_hat = teacher_transport_to_lower_sigma(
                    teacher=teacher,
                    x_hi=x_hi,
                    sigma_hi=sigma_hi,
                    sigma_lo=sigma_lo,
                    eps_floor=cfg.eps,
                )

            # Consistency distillation loss (Algorithm 2 / Eq. 7)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                # Online network: f_θ(x_{t_{n+1}}, t_{n+1})
                pred = student(x_hi, sigma_hi)

                # FIX #5: target network θ⁻ = cm_target (not ema_eval)
                with torch.no_grad():
                    target = cm_target(x_lo_hat, sigma_lo)

                if cfg.loss_type == "l2":
                    loss = F.mse_loss(pred, target)
                else:
                    loss = F.l1_loss(pred, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            # FIX #5a: update target network θ⁻  (Algorithm 2, Eq. 8)
            # θ⁻ ← stopgrad(µ·θ⁻ + (1−µ)·θ)
            mu = cfg.ema_decay_target
            with torch.no_grad():
                for p_tgt, p_src in zip(cm_target.parameters(), student.parameters()):
                    p_tgt.data.mul_(mu).add_(p_src.data, alpha=1.0 - mu)
                for b_tgt, b_src in zip(cm_target.buffers(), student.buffers()):
                    b_tgt.copy_(b_src)

            # FIX #5b: update ema_eval (separate slower EMA for sampling)
            ema_eval.update(student)

            loss_value = float(loss.item())
            running_loss += loss_value
            epoch_loss_sum += loss_value
            epoch_num_steps += 1

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
                epoch=epoch, student=student, ema_eval=ema_eval,
                optimizer=optimizer, cfg=cfg,
            )

        if epoch % cfg.sample_every_epochs == 0 or epoch == cfg.epochs:
            save_samples(
                model=ema_eval.ema_model,
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

        if epoch % cfg.fid_every_epochs == 0 or epoch == cfg.epochs:
            domain_fid = compute_domain_fid(
                model=ema_eval.ema_model, extractor=extractor,
                real_feats=real_feats, cfg=cfg, device=device,
            )
            epoch_metrics["domain_fid"] = domain_fid
            print(f"Epoch {epoch}: avg_loss={epoch_avg_loss:.6f}, domain_fid={domain_fid:.4f}")
        else:
            print(f"Epoch {epoch}: avg_loss={epoch_avg_loss:.6f}")

        metrics_history.append(epoch_metrics)
        save_metrics_json(metrics_history, metrics_path)

    torch.save({
        "model": student.state_dict(),
        "ema_model": ema_eval.ema_model.state_dict(),
        "config": asdict(cfg),
    }, os.path.join(cfg.output_dir, "cd_model_final.pt"))

    print("Done.")



def load_latest_student_checkpoint(cfg: CDConfig, device: torch.device) -> ConsistencyModel:
    ckpt_files = sorted([
        f for f in os.listdir(cfg.output_dir)
        if f.startswith("checkpoint_epoch_") and f.endswith(".pt")
    ])
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint_epoch_*.pt found in {cfg.output_dir}")

    latest_path = os.path.join(cfg.output_dir, ckpt_files[-1])
    print(f"Loading latest student checkpoint: {latest_path}")

    backbone = UNet(
        in_channels=cfg.in_channels, base_channels=cfg.base_channels,
        channel_mults=cfg.channel_mults, num_res_blocks=cfg.num_res_blocks,
        time_emb_dim=cfg.time_emb_dim, dropout=cfg.dropout,
    ).to(device)

    student = ConsistencyModel(
        backbone=backbone, sigma_data=cfg.sigma_data, eps=cfg.eps,
    ).to(device)

    ckpt = torch.load(latest_path, map_location=device)
    key = "ema_model" if "ema_model" in ckpt else "model"
    student.load_state_dict(ckpt[key])
    print(f"Using '{key}' weights.")
    student.eval()
    return student


@torch.no_grad()
def save_grid(samples: Tensor, path: str, nrow: int = 8) -> None:
    utils.save_image((samples.clamp(-1.0, 1.0) + 1.0) / 2.0, path, nrow=nrow)



if __name__ == "__main__":
    cfg = CDConfig()
    train_cd(cfg)