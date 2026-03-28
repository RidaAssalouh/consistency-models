import math
import os
import json
import random
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import utils, datasets, transforms
from torch.utils.data import DataLoader
from scipy.linalg import sqrtm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import matplotlib.ticker as ticker

from models.cd_model_utils import UNet


@dataclass
class EvalCDConfig:
    ckpt_path: str = "consistency_models_ckpts/cd_model_final_l2.pt"
    output_dir: str = "figures/cd_fashionmnist_l2/eval"

    image_size: int = 32
    in_channels: int = 1

    num_images: int = 64
    nrow: int = 8

    sigma_data: float = 0.5
    eps: float = 1e-3
    sigma_max: float = 50.0
    rho: float = 7.0

    min_steps: int = 1
    max_steps: int = 20

    base_channels: int = 64
    channel_mults: Tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    time_emb_dim: int = 256
    dropout: float = 0.0

    use_ema: bool = True
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    data_root: str = "./data"
    fid_num_real: int = 2048
    fid_num_fake: int = 2048
    fid_batch_size: int = 128
    num_workers: int = 4

    extractor_ckpt: str = "consistency_models_ckpts/fashion_extractor.pt"
    extractor_feat_dim: int = 512


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_image_grid(samples: Tensor, path: str, nrow: int = 8) -> None:
    samples = samples.clamp(-1.0, 1.0)
    samples = (samples + 1.0) / 2.0
    utils.save_image(samples, path, nrow=nrow)




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
        sigma = sigma.float().clamp(min=self.eps)
        c_skip = self.c_skip(sigma).view(-1, 1, 1, 1)
        c_out = self.c_out(sigma).view(-1, 1, 1, 1)
        f_out = self.backbone(x, sigma)
        return c_skip * x + c_out * f_out

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


@torch.no_grad()
def cd_multistep_sample(
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
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        num_scales=steps,
        rho=rho,
        device=device,
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

def build_eval_dataloader(cfg: EvalCDConfig) -> DataLoader:
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
        batch_size=cfg.fid_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )


def load_feature_extractor(cfg: EvalCDConfig, device: torch.device) -> FashionMNISTFeatureExtractor:
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
def extract_features_from_loader(
    extractor: FashionMNISTFeatureExtractor,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
) -> np.ndarray:
    feats = []
    seen = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        feats.append(extractor.forward_features(x).cpu().numpy())
        seen += x.size(0)
        if seen >= max_samples:
            break
    return np.concatenate(feats, axis=0)[:max_samples]


@torch.no_grad()
def extract_features_from_generated_batches(
    extractor: FashionMNISTFeatureExtractor,
    batches: List[Tensor],
    device: torch.device,
    max_samples: int,
) -> np.ndarray:
    feats = []
    seen = 0
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


def compute_domain_fid_from_features(real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
    mu_r, mu_f = real_feats.mean(0), fake_feats.mean(0)
    cov_r = np.cov(real_feats, rowvar=False)
    cov_f = np.cov(fake_feats, rowvar=False)
    return frechet_distance(mu_r, cov_r, mu_f, cov_f)


@torch.no_grad()
def compute_domain_fid_vs_steps(
    model: ConsistencyModel,
    extractor: FashionMNISTFeatureExtractor,
    real_feats: np.ndarray,
    cfg: EvalCDConfig,
    device: torch.device,
    steps: int,
) -> float:
    generated_batches = []
    generated = 0

    while generated < cfg.fid_num_fake:
        bsz = min(cfg.fid_batch_size, cfg.fid_num_fake - generated)
        x_fake, _ = cd_multistep_sample(
            model=model,
            shape=(bsz, cfg.in_channels, cfg.image_size, cfg.image_size),
            sigma_max=cfg.sigma_max,
            sigma_min=cfg.eps,
            steps=steps,
            rho=cfg.rho,
            device=device,
        )
        generated_batches.append(x_fake.cpu())
        generated += bsz

    fake_feats = extract_features_from_generated_batches(
        extractor=extractor,
        batches=generated_batches,
        device=device,
        max_samples=cfg.fid_num_fake,
    )

    return compute_domain_fid_from_features(real_feats, fake_feats)




def build_model_from_cfg(cfg: EvalCDConfig) -> ConsistencyModel:
    backbone = UNet(
        in_channels=cfg.in_channels,
        base_channels=cfg.base_channels,
        channel_mults=cfg.channel_mults,
        num_res_blocks=cfg.num_res_blocks,
        time_emb_dim=cfg.time_emb_dim,
        dropout=cfg.dropout,
    )
    model = ConsistencyModel(
        backbone=backbone,
        sigma_data=cfg.sigma_data,
        eps=cfg.eps,
    )
    return model


def maybe_override_cfg_from_checkpoint(cfg: EvalCDConfig, ckpt_cfg: Dict[str, Any]) -> EvalCDConfig:
    fields_to_copy = [
        "image_size",
        "in_channels",
        "sigma_data",
        "eps",
        "sigma_max",
        "rho",
        "base_channels",
        "channel_mults",
        "num_res_blocks",
        "time_emb_dim",
        "dropout",
    ]
    for k in fields_to_copy:
        if k in ckpt_cfg:
            setattr(cfg, k, ckpt_cfg[k])
    if isinstance(cfg.channel_mults, list):
        cfg.channel_mults = tuple(cfg.channel_mults)
    return cfg


def load_cd_model(cfg: EvalCDConfig, device: torch.device) -> ConsistencyModel:
    ckpt = torch.load(cfg.ckpt_path, map_location=device)

    if "config" in ckpt:
        cfg = maybe_override_cfg_from_checkpoint(cfg, ckpt["config"])

    model = build_model_from_cfg(cfg).to(device)

    state_key = "ema_model" if (cfg.use_ema and "ema_model" in ckpt) else "model"
    if state_key not in ckpt:
        raise KeyError(f"Checkpoint does not contain '{state_key}'.")

    model.load_state_dict(ckpt[state_key], strict=True)
    model.eval()

    print(f"Loaded checkpoint: {cfg.ckpt_path}")
    print(f"Using weights: {state_key}")
    return model




@torch.no_grad()
def evaluate_multistep_sampling(cfg: EvalCDConfig) -> None:
    seed_everything(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    device = torch.device(cfg.device)
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
    if "config" in ckpt:
        cfg = maybe_override_cfg_from_checkpoint(cfg, ckpt["config"])

    model = load_cd_model(cfg, device)

    extractor = load_feature_extractor(cfg, device)
    real_loader = build_eval_dataloader(cfg)
    real_feats = extract_features_from_loader(
        extractor=extractor,
        loader=real_loader,
        device=device,
        max_samples=cfg.fid_num_real,
    )
    print(f"Precomputed real features: {real_feats.shape}")

    summary: List[Dict[str, Any]] = []
    metrics_vs_steps: List[Dict[str, Any]] = []

    for steps in range(cfg.min_steps, cfg.max_steps + 1):
        samples, states = cd_multistep_sample(
            model=model,
            shape=(cfg.num_images, cfg.in_channels, cfg.image_size, cfg.image_size),
            sigma_max=cfg.sigma_max,
            sigma_min=cfg.eps,
            steps=steps,
            rho=cfg.rho,
            device=device,
        )

        samples_path = os.path.join(cfg.output_dir, f"samples_{steps:02d}_steps.png")
        save_image_grid(samples.cpu(), samples_path, nrow=cfg.nrow)

        traj = states[:, 0]
        traj_path = os.path.join(cfg.output_dir, f"trajectory_first_sample_{steps:02d}_steps.png")
        save_image_grid(traj, traj_path, nrow=min(steps, 5))

        domain_fid = compute_domain_fid_vs_steps(
            model=model,
            extractor=extractor,
            real_feats=real_feats,
            cfg=cfg,
            device=device,
            steps=steps,
        )

        summary.append({
            "steps": steps,
            "samples_path": samples_path,
            "trajectory_path": traj_path,
            "domain_fid": domain_fid,
        })

        metrics_vs_steps.append({
            "steps": steps,
            "domain_fid": domain_fid,
        })

        print(f"[done] steps={steps:02d} -> {samples_path} | domain_fid={domain_fid:.6f}")

    with open(os.path.join(cfg.output_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(cfg.output_dir, "metrics_vs_steps.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_vs_steps, f, indent=2)

    print(f"\nSaved all outputs to: {cfg.output_dir}")




BG = "#ffffff"
TEXT = "#222222"
SPINE = "#888888"
GRID = "#d9d9d9"
LINE = "#8172b3"
ACCENT = "#55a868"
ANNOT_BG = "#f7f7f7"

def _apply_elegant_style(ax) -> None:
    ax.set_facecolor(BG)

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    for side in ["left", "bottom"]:
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(0.7)

    ax.tick_params(axis="both", colors=TEXT, labelsize=7.5, width=0.5, length=2.5)

    ax.grid(True, which="major", axis="both",
            color=GRID, linewidth=0.5, alpha=0.5)

    ax.minorticks_on()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    ax.grid(True, which="minor", axis="both",
            color=GRID, linewidth=0.3, alpha=0.25)

    ax.set_axisbelow(True)



# plot domain FID from training_metrics.json
def plot_domain_fid_vs_epoch_from_training_metrics(
    metrics_json: str,
    save_path: str,
    title: str = "",
) -> None:
    with open(metrics_json, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    points = []
    for entry in metrics:
        if entry.get("type") != "train_epoch":
            continue
        if "domain_fid" not in entry:
            continue
        points.append((int(entry["epoch"]), float(entry["domain_fid"])))

    if len(points) == 0:
        raise ValueError("No 'train_epoch' entries with 'domain_fid' found in metrics JSON.")

    points = sorted(points, key=lambda x: x[0])

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    best_idx = int(np.argmin(ys))
    best_x = xs[best_idx]
    best_y = ys[best_idx]

    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    _apply_elegant_style(ax)

    ax.plot(
        xs,
        ys,
        color=LINE,
        linewidth=1.5,
        marker="o",
        markersize=4,
        markeredgewidth=0.6,
        markeredgecolor="white",
        markerfacecolor=LINE,
        zorder=3,
        solid_capstyle="round",
        label="Domain FID",
    )

    ax.scatter(
        [best_x],
        [best_y],
        s=60,
        color=ACCENT,
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
        label="Best epoch",
    )



    ax.set_title(title, fontsize=7, fontweight="bold", color=TEXT, pad=10)
    ax.set_xlabel("Epoch", fontsize=8, color=TEXT, labelpad=6)
    ax.set_ylabel("Domain FID", fontsize=8, color=TEXT, labelpad=6)

    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    leg = ax.legend(
        frameon=True,
        fontsize=7.5,
        framealpha=0.95,
        edgecolor=GRID,
        fancybox=False,
        loc="upper right",
        handlelength=1.8,
        handleheight=0.9,
    )
    leg.get_frame().set_linewidth(0.6)

    plt.tight_layout(pad=0.8)
    plt.savefig(save_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved epoch plot to: {save_path}")

@torch.no_grad()
def evaluate_multistep_sampling_grid(cfg: EvalCDConfig) -> None:
    seed_everything(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    device = torch.device(cfg.device)
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
    if "config" in ckpt:
        cfg = maybe_override_cfg_from_checkpoint(cfg, ckpt["config"])

    model = load_cd_model(cfg, device)

    summary: List[Dict[str, Any]] = []

    metrics_path = os.path.join(cfg.output_dir, "metrics_vs_steps.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"metrics_vs_steps.json not found at: {metrics_path}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    fid_by_steps = {}
    for entry in metrics:
        if "steps" in entry and "domain_fid" in entry:
            fid_by_steps[int(entry["steps"])] = float(entry["domain_fid"])

    steps_list = [1, 2, 4, 8, 16]
    missing = [s for s in steps_list if s not in fid_by_steps]
    if missing:
        raise ValueError(f"Missing FID values for steps: {missing}")

    horizontal_panels = []

    for steps in steps_list:
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        samples, states = cd_multistep_sample(
            model=model,
            shape=(cfg.num_images, cfg.in_channels, cfg.image_size, cfg.image_size),
            sigma_max=cfg.sigma_max,
            sigma_min=cfg.eps,
            steps=steps,
            rho=cfg.rho,
            device=device,
        )

        samples_vis = (samples.clamp(-1.0, 1.0) + 1.0) / 2.0

        grid = utils.make_grid(
            samples_vis,
            nrow=int(math.sqrt(cfg.num_images)),
            padding=0
        )
        horizontal_panels.append((steps, grid))

        samples_path = os.path.join(cfg.output_dir, f"samples_{steps:02d}_steps.png")
        utils.save_image(grid, samples_path)

        traj = states[:, 0]
        traj_path = os.path.join(cfg.output_dir, f"trajectory_first_sample_{steps:02d}_steps.png")
        save_image_grid(traj, traj_path, nrow=min(steps, 5))

        summary.append({
            "steps": steps,
            "samples_path": samples_path,
            "trajectory_path": traj_path,
            "domain_fid": fid_by_steps[steps],
        })

        print(f"[done] steps={steps:02d} -> {samples_path} | domain_fid={fid_by_steps[steps]:.6f}")

    ncols = len(horizontal_panels)
    fig, axes = plt.subplots(1, ncols, figsize=(2.6 * ncols, 3.2))
    if ncols == 1:
        axes = [axes]

    for ax, (steps, grid) in zip(axes, horizontal_panels):
        img = grid.permute(1, 2, 0).cpu().numpy()
        if img.shape[-1] == 1:
            ax.imshow(img.squeeze(), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        else:
            ax.imshow(img, vmin=0.0, vmax=1.0, interpolation="nearest")

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
            spine.set_linewidth(0.8)

        ax.set_title(f"Number of sampling steps = {steps}", fontsize=8, pad=5)
        ax.set_xlabel(f"FID = {fid_by_steps[steps]:.3f}", fontsize=7.5, labelpad=4)

    fig.suptitle(
        "",
        fontsize=9,
        fontweight="bold",
        y=0.97,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93], pad=0.8)

    strip_path_png = os.path.join(cfg.output_dir, "cd_progression_strip_with_fid.png")
    strip_path_pdf = os.path.join(cfg.output_dir, "cd_progression_strip_with_fid.pdf")
    plt.savefig(strip_path_png, dpi=400, bbox_inches="tight", facecolor="white")
    plt.savefig(strip_path_pdf, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\nSaved progression strip to: {strip_path_png}")
    print(f"Saved progression strip to: {strip_path_pdf}")

    with open(os.path.join(cfg.output_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved all outputs to: {cfg.output_dir}")


if __name__ == "__main__":
    cfg = EvalCDConfig()
    evaluate_multistep_sampling_grid(cfg)