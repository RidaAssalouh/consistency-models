import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.cd_model_utils import UNet as UNet_CD
from models.ct_model_utils import UNet as UNet_CT
from models.teacher_utils import UNet as UNet_Teacher

# Config
@dataclass
class SlerpConfig:
    save_dir: str = "./interpolation_results"

    image_size: int = 32
    in_channels: int = 1

    base_channels: int = 64
    channel_mults: Tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    time_emb_dim: int = 256
    dropout: float = 0.0

    sigma_data: float = 0.5
    eps: float = 1e-3
    sigma_max: float = 50.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


    num_interpolation_points: int = 9
    num_pairs: int = 1

    sample_steps: int = 1


# Utils
def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def denorm_to_01(x: Tensor) -> Tensor:
    return ((x.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)


# Consistency models

class ConsistencyModel_CT(nn.Module):
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


class ConsistencyModel_CD(nn.Module):
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
        sigma = sigma.float().clamp(min=self.eps)
        c_skip = self.c_skip(sigma).view(-1, 1, 1, 1)
        c_out = self.c_out(sigma).view(-1, 1, 1, 1)
        F_out = self.backbone(x, sigma)
        return c_skip * x + c_out * F_out


# Checkpoint loaders

def build_cd_model_from_checkpoint(ckpt_path: str, device: torch.device) -> ConsistencyModel_CD:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt.get("config", {})

    backbone = UNet_CD(
        in_channels=cfg_dict.get("in_channels", 1),
        base_channels=cfg_dict.get("base_channels", 64),
        channel_mults=tuple(cfg_dict.get("channel_mults", [1, 2, 4])),
        num_res_blocks=cfg_dict.get("num_res_blocks", 2),
        time_emb_dim=cfg_dict.get("time_emb_dim", 256),
        dropout=cfg_dict.get("dropout", 0.0),
    ).to(device)

    model = ConsistencyModel_CD(
        backbone=backbone,
        sigma_data=cfg_dict.get("sigma_data", 0.5),
        eps=cfg_dict.get("eps", 1e-3),
    ).to(device)

    state_dict = ckpt.get("ema_model") or ckpt.get("model")
    if state_dict is None:
        raise KeyError(f"Could not find 'ema_model' or 'model' in checkpoint: {ckpt_path}")

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def build_ct_model_from_checkpoint(ckpt_path: str, device: torch.device) -> ConsistencyModel_CT:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt.get("config", {})

    backbone = UNet_CT(
        in_channels=cfg_dict.get("in_channels", 1),
        base_channels=cfg_dict.get("base_channels", 64),
        channel_mults=tuple(cfg_dict.get("channel_mults", [1, 2, 4])),
        num_res_blocks=cfg_dict.get("num_res_blocks", 2),
        time_emb_dim=cfg_dict.get("time_emb_dim", 256),
        dropout=cfg_dict.get("dropout", 0.0),
    ).to(device)

    model = ConsistencyModel_CT(
        backbone=backbone,
        sigma_data=cfg_dict.get("sigma_data", 0.5),
        eps=cfg_dict.get("eps", 1e-3),
    ).to(device)

    state_dict = ckpt.get("ema_model") or ckpt.get("model")
    if state_dict is None:
        raise KeyError(f"Could not find 'ema_model' or 'model' in checkpoint: {ckpt_path}")

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# Teacher model

@dataclass
class TeacherConfig:
    data_root: str = "./data"
    output_dir: str = "consistency_models_ckpts/teacher_epoch_060.pt"

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

    sigma_sample_mode: str = "karras"
    num_karras_scales: int = 40

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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
        sigma = sigma.float()
        cin = self.c_in(sigma).view(-1, 1, 1, 1)
        cout = self.c_out(sigma).view(-1, 1, 1, 1)
        raw = self.backbone(cin * x, sigma)
        return cout * raw

    def predict_x0(self, x_t: Tensor, sigma: Tensor) -> Tensor:
        eps_hat = self(x_t, sigma)
        return x_t - sigma.view(-1, 1, 1, 1) * eps_hat


def load_teacher(cfg: TeacherConfig, device: torch.device) -> DiffusionTeacher:
    backbone = UNet_Teacher(
        in_channels=cfg.in_channels,
        base_channels=cfg.base_channels,
        channel_mults=cfg.channel_mults,
        num_res_blocks=cfg.num_res_blocks,
        time_emb_dim=cfg.time_emb_dim,
        dropout=cfg.dropout,
    ).to(device)

    teacher = DiffusionTeacher(backbone=backbone, sigma_data=cfg.sigma_data).to(device)
    ckpt = torch.load(cfg.output_dir, map_location=device)
    if "ema_model" not in ckpt:
        raise KeyError("Teacher checkpoint does not contain 'ema_model'.")
    teacher.load_state_dict(ckpt["ema_model"])
    print(f"Loaded EMA teacher from: {cfg.output_dir}")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


# SLERP

def slerp(z1: Tensor, z2: Tensor, alpha: float, eps: float = 1e-8) -> Tensor:
    z1_norm = z1.norm(dim=-1, keepdim=True).clamp_min(eps)
    z2_norm = z2.norm(dim=-1, keepdim=True).clamp_min(eps)
    cos_psi = (z1 * z2).sum(dim=-1, keepdim=True) / (z1_norm * z2_norm)
    cos_psi = cos_psi.clamp(-1.0, 1.0)
    psi = torch.acos(cos_psi)
    sin_psi = torch.sin(psi)
    small_angle = sin_psi.abs() < 1e-6
    coeff1 = torch.sin((1.0 - alpha) * psi) / sin_psi.clamp_min(eps)
    coeff2 = torch.sin(alpha * psi) / sin_psi.clamp_min(eps)
    z = coeff1 * z1 + coeff2 * z2
    z_lin = (1.0 - alpha) * z1 + alpha * z2
    return torch.where(small_angle, z_lin, z)


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
    return (inv_rho_max + ramp * (inv_rho_min - inv_rho_max)) ** rho


@torch.no_grad()
def generate_from_noise(
    model,
    z: Tensor,
    sigma_max: float,
    sigma_min: float,
    steps: int,
    rho: float,
) -> Tensor:
    device = z.device
    shape = z.shape

    sigmas = build_karras_sigmas(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        num_scales=steps,
        rho=rho,
        device=device,
    )

    # First step: denoise from the provided noise vector
    sigma_batch = torch.full((shape[0],), sigmas[0].item(), device=device)
    x = model(z, sigma_batch)

    for sigma in sigmas[1:]:
        noise_std = torch.sqrt(torch.clamp(sigma ** 2 - sigma_min ** 2, min=0.0))
        x_hat = x + noise_std * torch.randn_like(x)
        sigma_batch = torch.full((shape[0],), sigma.item(), device=device)
        x = model(x_hat, sigma_batch)

    return x.clamp(-1.0, 1.0)


def sample_noise_pairs(
    num_pairs: int,
    image_shape: Tuple[int, int, int],
    sigma_max: float,
    device: torch.device,
) -> List[Tuple[Tensor, Tensor]]:
    """
    Sample fixed noise pairs once, to be reused across all models.
    Returns a list of (z1, z2) tuples, each of shape (1, C, H, W).
    """
    c, h, w = image_shape
    pairs = []
    for _ in range(num_pairs):
        z1 = sigma_max * torch.randn(1, c, h, w, device=device)
        z2 = sigma_max * torch.randn(1, c, h, w, device=device)
        pairs.append((z1, z2))
    return pairs


@torch.no_grad()
def interpolate_pair(
    model,
    z1: Tensor,
    z2: Tensor,
    sigma_max: float,
    sigma_min: float,
    steps: int,
    rho: float,
    num_points: int,
) -> Tensor:
    """
    Interpolate between a fixed (z1, z2) pair using multistep sampling.
    Returns (num_points, C, H, W).
    """
    z1_flat = z1.view(1, -1)
    z2_flat = z2.view(1, -1)
    c, h, w = z1.shape[1], z1.shape[2], z1.shape[3]

    alphas = torch.linspace(0.0, 1.0, num_points, device=z1.device)
    images = []
    for alpha in alphas:
        z_alpha_flat = slerp(z1_flat, z2_flat, float(alpha.item()))
        z_alpha = z_alpha_flat.view(1, c, h, w)
        x_alpha = generate_from_noise(
            model, z_alpha,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            steps=steps,
            rho=rho,
        )
        images.append(x_alpha[0].cpu())
    return torch.stack(images, dim=0) 




LINE = "#8172b3"
ACCENT = "#55a868"


def plot_comparison(
    all_rows: List[List[Tensor]],
    model_labels: List[str],
    num_points: int,
    save_path: str,
) -> None:

    num_models = len(model_labels)
    num_pairs = len(all_rows[0])

    cell = 1.5                          
    label_width = 1.4                       
    pair_gap = 0.35                      

    fig_w = label_width + num_points * cell
    fig_h = num_models * num_pairs * cell + (num_pairs - 1) * pair_gap
    fig = plt.figure(figsize=(fig_w, fig_h))



    total_row_units = num_models * num_pairs + (num_pairs - 1)
    heights = []
    for p in range(num_pairs):
        heights.extend([1.0] * num_models)
        if p < num_pairs - 1:
            heights.append(pair_gap / cell)  


    gs = gridspec.GridSpec(
        len(heights), num_points,
        figure=fig,
        left=label_width / fig_w,
        right=0.99,
        top=0.93,
        bottom=0.02,
        hspace=0.0,
        wspace=0.04,
        height_ratios=heights,
    )

    # α column labels at the very top
    alphas = torch.linspace(0.0, 1.0, num_points)
    for j, alpha in enumerate(alphas):
        ax = fig.add_subplot(gs[0, j])
        ax.set_title(f"α={alpha:.2f}", fontsize=12.5, pad=3)
        ax.axis("off") 

    model_colors = ["#4c72b0", "#c44e52", ACCENT, LINE]

    def gs_row(pair_idx: int, model_idx: int) -> int:
        """Map (pair, model) to the gridspec row index."""
        return pair_idx * (num_models + 1) + model_idx

    for p in range(num_pairs):
        for m, label in enumerate(model_labels):
            row_images = all_rows[m][p]         # (num_points, C, H, W)
            row_images = denorm_to_01(row_images)
            gs_r = gs_row(p, m)

            for j in range(num_points):
                ax = fig.add_subplot(gs[gs_r, j])
                ax.imshow(row_images[j, 0].numpy(), cmap="gray", vmin=0, vmax=1,
                          interpolation="nearest")
                ax.axis("off")

                if j == 0:
                    ax.text(
                        -0.25, 0.5, label,
                        transform=ax.transAxes,
                        fontsize=15, rotation=45,
                        ha="right", va="center",
                        color=model_colors[m % len(model_colors)],
                    )


    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


# Main
def run_one_checkpoint(
    model_type: str,
    ckpt_path: str,
    cfg: SlerpConfig,
    noise_pairs: List[Tuple[Tensor, Tensor]],
    steps,
) -> List[Tensor]:

    device = torch.device(cfg.device)

    if model_type == "cd":
        model = build_cd_model_from_checkpoint(ckpt_path, device)
    else:
        model = build_ct_model_from_checkpoint(ckpt_path, device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt.get("config", {})
    sigma_max = float(cfg_dict.get("sigma_max", cfg.sigma_max))
    sigma_min = float(cfg_dict.get("eps", cfg.eps))
    rho      = float(cfg_dict.get("rho", 7.0))

    rows = []
    for z1, z2 in noise_pairs:
        row = interpolate_pair(
            model=model,
            z1=z1.to(device),
            z2=z2.to(device),
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            steps=steps,
            rho=rho,
            num_points=cfg.num_interpolation_points,
        )
        rows.append(row)

    return rows


def main():
    cfg = SlerpConfig()

    device = torch.device(cfg.device)

    checkpoints = [
        ("./consistency_models_ckpts/ct_model_final_l1.pt", "ct", r"CT ($\ell_1$)"),
        ("./consistency_models_ckpts/ct_model_final_l2.pt", "ct", r"CT ($\ell_2$)"),
        ("./consistency_models_ckpts/cd_model_final_l2.pt", "cd", r"CD ($\ell_2$)"),
        ("./consistency_models_ckpts/cd_model_final_l1.pt", "cd", r"CD ($\ell_1$)"),
    ]

    first_ckpt = torch.load(checkpoints[0][0], map_location="cpu")
    first_cfg = first_ckpt.get("config", {})
    sigma_max = float(first_cfg.get("sigma_max", cfg.sigma_max))
    image_size = int(first_cfg.get("image_size", cfg.image_size))
    in_channels = int(first_cfg.get("in_channels", cfg.in_channels))
    sample_steps = int(cfg.sample_steps)

    noise_pairs = sample_noise_pairs(
        num_pairs=cfg.num_pairs,
        image_shape=(in_channels, image_size, image_size),
        sigma_max=sigma_max,
        device=device,
    )

    all_rows = []      
    model_labels = []

    for ckpt_path, model_type, label in checkpoints:
        print(f"\nRunning SLERP interpolation: {label}  ({ckpt_path})")
        rows = run_one_checkpoint(model_type, ckpt_path, cfg, noise_pairs, sample_steps)
        all_rows.append(rows)
        model_labels.append(label)

    plot_comparison(
        all_rows=all_rows,
        model_labels=model_labels,
        num_points=cfg.num_interpolation_points,
        save_path=str(Path(cfg.save_dir) / "slerp_comparison.png"),
    )


if __name__ == "__main__":
    main()