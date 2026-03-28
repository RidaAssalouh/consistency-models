import json
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import sqrtm
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    HAS_TORCHMETRICS = True
except Exception:
    FrechetInceptionDistance = None
    HAS_TORCHMETRICS = False

from models.ct_model_utils import UNet

def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_image_grid(samples: Tensor, path: str, nrow: int = 8) -> None:
    samples = samples.clamp(-1.0, 1.0)
    samples = (samples + 1.0) / 2.0
    utils.save_image(samples, path, nrow=nrow)



@dataclass
class EvalConfig:
    ckpt_path: str = "consistency_models_ckpts/ct_model_final_l2.pt"
    output_dir: str = "figures/ct_fashionmnist_l2/eval"
    data_root: str = "./data"

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
    rho: float = 7.0

    num_images: int = 64
    use_ema: bool = True
    seed: int = 0


    min_steps: int = 1
    max_steps: int = 20

    fid_num_real: int = 2048
    fid_num_fake: int = 2048
    fid_batch_size: int = 128
    num_workers: int = 4

    extractor_ckpt: str = "consistency_models_ckpts/fashion_extractor.pt"
    extractor_epochs: int = 10
    extractor_feat_dim: int = 512

    pr_k: int = 3 

    also_compute_inception_fid: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ConsistencyModel(nn.Module):
    def __init__(self, backbone: nn.Module, sigma_data: float = 0.5, eps: float = 1e-3):
        super().__init__()
        self.backbone = backbone
        self.sigma_data = sigma_data
        self.eps = eps

    def c_skip(self, sigma): return (self.sigma_data**2) / ((sigma - self.eps)**2 + self.sigma_data**2)
    def c_out(self, sigma):  return self.sigma_data * (sigma - self.eps) / torch.sqrt(self.sigma_data**2 + sigma**2)
    def c_in(self, sigma):   return 1.0 / torch.sqrt(self.sigma_data**2 + sigma**2)

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        sigma = sigma.float().clamp(min=self.eps)
        cskip = self.c_skip(sigma).view(-1, 1, 1, 1)
        cout  = self.c_out(sigma).view(-1, 1, 1, 1)
        cin   = self.c_in(sigma).view(-1, 1, 1, 1)
        return cskip * x + cout * self.backbone(cin * x, sigma)



class FashionMNISTFeatureExtractor(nn.Module):
    """
    Lightweight CNN trained on Fashion-MNIST classification.
    After training, the classification head is removed and the
    penultimate representation is used as a feature vector for FID.

    Input : (B, 1, H, W)  — native grayscale, no upsampling needed.
    Output: (B, feat_dim) — L2-normalised embedding.
    """
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
        h = self.encoder(x)
        h = self.embedding(h)
        return F.normalize(h, dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.forward_features(x))


def train_feature_extractor(cfg: EvalConfig) -> FashionMNISTFeatureExtractor:
    device = torch.device(cfg.device)
    extractor = FashionMNISTFeatureExtractor(feat_dim=cfg.extractor_feat_dim).to(device)

    if os.path.exists(cfg.extractor_ckpt):
        extractor.load_state_dict(torch.load(cfg.extractor_ckpt, map_location=device))
        extractor.eval()
        print("Loaded feature extractor from: {}".format(cfg.extractor_ckpt))
        return extractor

    print("Training feature extractor for {} epochs...".format(cfg.extractor_epochs))

    train_tf = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    train_ds = datasets.FashionMNIST(cfg.data_root, train=True,  download=True, transform=train_tf)
    val_ds   = datasets.FashionMNIST(cfg.data_root, train=False, download=True, transform=transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]))

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    optim  = torch.optim.AdamW(extractor.parameters(), lr=1e-3, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.extractor_epochs)

    for epoch in range(1, cfg.extractor_epochs + 1):
        extractor.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = extractor(x)
            loss = F.cross_entropy(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() * x.size(0)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += x.size(0)
        sched.step()

        extractor.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_correct += (extractor(x).argmax(1) == y).sum().item()
                val_total   += x.size(0)

        print("  Epoch {:2d}/{} | loss={:.4f} | train_acc={:.1f}% | val_acc={:.1f}%".format(
            epoch, cfg.extractor_epochs,
            total_loss / total,
            100 * correct / total,
            100 * val_correct / val_total,
        ))

    os.makedirs(os.path.dirname(cfg.extractor_ckpt) or ".", exist_ok=True)
    torch.save(extractor.state_dict(), cfg.extractor_ckpt)
    print("Saved feature extractor to: {}".format(cfg.extractor_ckpt))
    extractor.eval()
    return extractor



@torch.no_grad()
def extract_features(
    extractor: FashionMNISTFeatureExtractor,
    loader_or_tensors,
    device: torch.device,
    max_samples: int,
) -> np.ndarray:
    extractor.eval()
    all_feats = []
    seen = 0

    if isinstance(loader_or_tensors, DataLoader):
        for x, _ in loader_or_tensors:
            x = x.to(device)
            all_feats.append(extractor.forward_features(x).cpu().numpy())
            seen += x.size(0)
            if seen >= max_samples:
                break
    else:
        for x in loader_or_tensors:
            x = x.to(device)
            all_feats.append(extractor.forward_features(x).cpu().numpy())
            seen += x.size(0)
            if seen >= max_samples:
                break

    feats = np.concatenate(all_feats, axis=0)[:max_samples]
    return feats


def frechet_distance(mu1: np.ndarray, cov1: np.ndarray,
                     mu2: np.ndarray, cov2: np.ndarray) -> float:
    diff = mu1 - mu2
    covmean, _ = sqrtm(cov1 @ cov2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(cov1 + cov2 - 2 * covmean))
    return fid


def compute_domain_fid(real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
    mu_r, cov_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu_f, cov_f = fake_feats.mean(0), np.cov(fake_feats, rowvar=False)
    return frechet_distance(mu_r, cov_r, mu_f, cov_f)



def compute_precision_recall(
    real_feats: np.ndarray,
    fake_feats: np.ndarray,
    k: int = 3,
) -> Tuple[float, float]:
    def knn_radii(feats: np.ndarray, k: int) -> np.ndarray:
        """For each point, return the distance to its k-th nearest neighbour."""
        sq = np.sum(feats**2, axis=1, keepdims=True)
        dists = sq + sq.T - 2 * feats @ feats.T
        np.fill_diagonal(dists, np.inf)
        sorted_dists = np.sort(dists, axis=1)
        return np.sqrt(np.maximum(sorted_dists[:, k - 1], 0.0))

    def in_manifold(query: np.ndarray, ref: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """Returns boolean mask: True if query[i] is inside any ball of ref."""
        sq_q = np.sum(query**2, axis=1, keepdims=True)
        sq_r = np.sum(ref**2,   axis=1, keepdims=True)
        dists = sq_q + sq_r.T - 2 * query @ ref.T
        dists = np.sqrt(np.maximum(dists, 0.0))
        return (dists <= radii[None, :]).any(axis=1)

    real_radii = knn_radii(real_feats, k)
    fake_radii = knn_radii(fake_feats, k)

    precision = float(in_manifold(fake_feats, real_feats, real_radii).mean())
    recall    = float(in_manifold(real_feats, fake_feats, fake_radii).mean())
    return precision, recall


def build_eval_dataloader(cfg: EvalConfig) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = datasets.FashionMNIST(cfg.data_root, train=True, download=True, transform=transform)
    return DataLoader(
        dataset,
        batch_size=cfg.fid_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )



def build_karras_sigmas(
    sigma_min: float, sigma_max: float, num_scales: int, rho: float, device: torch.device
) -> Tensor:
    ramp = torch.linspace(0, 1, num_scales, device=device)
    inv_rho_min = sigma_min ** (1.0 / rho)
    inv_rho_max = sigma_max ** (1.0 / rho)
    return (inv_rho_max + ramp * (inv_rho_min - inv_rho_max)) ** rho


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
    sigmas = build_karras_sigmas(sigma_min, sigma_max, steps, rho, device)

    x_hat = sigma_max * torch.randn(shape, device=device)
    states = []

    sigma_batch = torch.full((shape[0],), sigmas[0].item(), device=device)
    x = model(x_hat, sigma_batch)
    states.append(x.detach().cpu())

    for sigma in sigmas[1:]:
        z = torch.randn_like(x)
        noise_std = torch.sqrt(torch.clamp(sigma**2 - sigma_min**2, min=0.0))
        x_hat = x + noise_std * z
        sigma_batch = torch.full((shape[0],), sigma.item(), device=device)
        x = model(x_hat, sigma_batch)
        states.append(x.detach().cpu())

    return x.clamp(-1.0, 1.0), torch.stack(states, dim=0)


@torch.no_grad()
def save_grid(samples: Tensor, path: str, nrow: int = 8) -> None:
    samples = (samples.clamp(-1.0, 1.0) + 1.0) / 2.0
    utils.save_image(samples, path, nrow=nrow)


@torch.no_grad()
def save_progress_grid(all_states: Tensor, path: str) -> None:
    steps, batch = all_states.shape[:2]
    flat = all_states.reshape(steps * batch, *all_states.shape[2:])
    flat = (flat.clamp(-1.0, 1.0) + 1.0) / 2.0
    utils.save_image(flat, path, nrow=batch)


def to_inception_uint8(x: Tensor) -> Tensor:
    """Upsamples grayscale 32px → 299px RGB uint8 for InceptionV3."""
    x = x.detach().clamp(-1.0, 1.0)
    x = (x + 1.0) / 2.0
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    return (255.0 * x).clamp(0, 255).to(torch.uint8)


@torch.no_grad()
def compute_inception_fid(
    model: ConsistencyModel,
    real_loader: DataLoader,
    cfg: EvalConfig,
    device: torch.device,
    steps: int,
) -> Optional[float]:
    if not HAS_TORCHMETRICS:
        return None
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    seen = 0
    for x_real, _ in real_loader:
        fid_metric.update(to_inception_uint8(x_real.to(device)), real=True)
        seen += x_real.size(0)
        if seen >= cfg.fid_num_real:
            break

    generated = 0
    while generated < cfg.fid_num_fake:
        bsz = min(cfg.fid_batch_size, cfg.fid_num_fake - generated)
        x_fake, _ = ct_multistep_sample(
            model=model,
            shape=(bsz, cfg.in_channels, cfg.image_size, cfg.image_size),
            sigma_max=cfg.sigma_max, sigma_min=cfg.eps,
            steps=steps, rho=cfg.rho, device=device,
        )
        fid_metric.update(to_inception_uint8(x_fake), real=False)
        generated += bsz

    return float(fid_metric.compute().item())


def load_ct_model(cfg: EvalConfig) -> ConsistencyModel:
    device = torch.device(cfg.device)
    backbone = UNet(
        in_channels=cfg.in_channels, base_channels=cfg.base_channels,
        channel_mults=cfg.channel_mults, num_res_blocks=cfg.num_res_blocks,
        time_emb_dim=cfg.time_emb_dim, dropout=cfg.dropout,
    ).to(device)
    model = ConsistencyModel(backbone, sigma_data=cfg.sigma_data, eps=cfg.eps).to(device)

    ckpt = torch.load(cfg.ckpt_path, map_location=device)
    key = "ema_model" if cfg.use_ema else "model"
    if key not in ckpt:
        raise KeyError("Checkpoint does not contain '{}'.".format(key))
    model.load_state_dict(ckpt[key])
    print("Loaded {} from: {}".format(key, cfg.ckpt_path))
    model.eval()
    return model


@torch.no_grad()
def evaluate_multistep_sampling_grid(cfg: EvalConfig) -> None:
    seed_everything(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    device = torch.device(cfg.device)
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu")

    model = load_ct_model(cfg)

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

        samples, states = ct_multistep_sample(
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

    strip_path_png = os.path.join(cfg.output_dir, "ct_progression_strip_with_fid.png")
    plt.savefig(strip_path_png, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\nSaved progression strip to: {strip_path_png}")

    with open(os.path.join(cfg.output_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved all outputs to: {cfg.output_dir}")



def main() -> None:
    cfg = EvalConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device)

    model = load_ct_model(cfg)

    extractor = train_feature_extractor(cfg)

    real_loader = build_eval_dataloader(cfg)

    print("Extracting real image features...")
    real_feats = extract_features(extractor, real_loader, device, cfg.fid_num_real)
    print("  Real features: {}".format(real_feats.shape))

    results = []

    for steps in range(cfg.min_steps, cfg.max_steps + 1):
        print("\nEvaluating steps = {}".format(steps))

        final_samples, all_states = ct_multistep_sample(
            model=model,
            shape=(cfg.num_images, cfg.in_channels, cfg.image_size, cfg.image_size),
            sigma_max=cfg.sigma_max, sigma_min=cfg.eps,
            steps=steps, rho=cfg.rho, device=device,
        )

        tag = "steps{:02d}_{}".format(steps, "ema" if cfg.use_ema else "raw")
        save_grid(final_samples.cpu(),
                  os.path.join(cfg.output_dir, "ct_samples_{}.png".format(tag)),
                  nrow=int(math.sqrt(cfg.num_images)))
        save_progress_grid(all_states,
                           os.path.join(cfg.output_dir, "ct_progress_{}.png".format(tag)))

        fake_batches = []
        generated = 0
        while generated < cfg.fid_num_fake:
            bsz = min(cfg.fid_batch_size, cfg.fid_num_fake - generated)
            x_fake, _ = ct_multistep_sample(
                model=model,
                shape=(bsz, cfg.in_channels, cfg.image_size, cfg.image_size),
                sigma_max=cfg.sigma_max, sigma_min=cfg.eps,
                steps=steps, rho=cfg.rho, device=device,
            )
            fake_batches.append(x_fake.cpu())
            generated += bsz

        fake_feats = extract_features(extractor, [b.to(device) for b in fake_batches],
                                      device, cfg.fid_num_fake)
        print("  Fake features: {}".format(fake_feats.shape))

        domain_fid = compute_domain_fid(real_feats, fake_feats)
        print("  Domain FID : {:.4f}".format(domain_fid))

        precision, recall = compute_precision_recall(real_feats, fake_feats, k=cfg.pr_k)
        print("  Precision  : {:.4f}".format(precision))
        print("  Recall     : {:.4f}".format(recall))

        inception_fid = None
        if cfg.also_compute_inception_fid:
            inception_fid = compute_inception_fid(model, real_loader, cfg, device, steps)
            if inception_fid is not None:
                print("  Inception FID: {:.4f}".format(inception_fid))
            else:
                print("  Inception FID: skipped (torchmetrics not installed)")

        result = {
            "steps": steps,
            "domain_fid": domain_fid,
            "precision": precision,
            "recall": recall,
            "inception_fid": inception_fid,
        }
        results.append(result)

        json_path = os.path.join(cfg.output_dir, "metrics_vs_steps.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    best = min(results, key=lambda r: r["domain_fid"])
    print("\n── Best by Domain FID ──")
    print("  Steps     : {}".format(best["steps"]))
    print("  Domain FID: {:.4f}".format(best["domain_fid"]))
    print("  Precision : {:.4f}".format(best["precision"]))
    print("  Recall    : {:.4f}".format(best["recall"]))
    print("\nSaved metrics to: {}".format(json_path))



import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial"]
rcParams["mathtext.fontset"] = "dejavusans"
rcParams["axes.unicode_minus"] = False

LINE     = "#1A6B3C"
LINE2    = "#2563EB"
LINE3    = "#9333EA"
ACCENT   = "#D76213"
GRID     = "#DCDCDC"
GRID_MIN = "#EFEFEF"
SPINE    = "#AAAAAA"
TEXT     = "#222222"
ANNOT_BG = "#FAFAFA"


def _apply_elegant_style(ax: plt.Axes) -> None:
    ax.figure.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(which="major", color=GRID,     linewidth=0.6, linestyle="-")
    ax.grid(which="minor", color=GRID_MIN, linewidth=0.3, linestyle="-")
    ax.minorticks_on()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color(SPINE)
    ax.tick_params(axis="both", which="both", labelsize=7, length=3,
                   width=0.7, colors=TEXT, direction="out")


def plot_metrics_vs_steps(
    json_path: str,
    save_dir: str = ".",
    title_prefix: str = "Consistency Model — Fashion-MNIST",
) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    results = sorted(results, key=lambda x: x["steps"])
    steps     = np.array([r["steps"]      for r in results])
    dom_fids  = np.array([r["domain_fid"] for r in results])
    precisions = np.array([r["precision"] for r in results])
    recalls   = np.array([r["recall"]     for r in results])

    inc_fids_raw = [r.get("inception_fid") for r in results]
    has_inception = any(v is not None for v in inc_fids_raw)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    _apply_elegant_style(ax)
    ax.plot(steps, dom_fids, color=LINE, linewidth=1.5, marker="o", markersize=4,
            markeredgewidth=0.6, markeredgecolor="white", markerfacecolor=LINE,
            zorder=3, label="Domain FID", solid_capstyle="round")
    best_idx = int(np.argmin(dom_fids))
    ax.scatter([steps[best_idx]], [dom_fids[best_idx]], s=60, color=ACCENT,
               edgecolors="white", linewidths=0.8, zorder=5,
               label="Best ({} steps)".format(steps[best_idx]))
    ax.annotate(
        "FID = {:.2f} ".format(dom_fids[best_idx]),
        xy=(steps[best_idx], dom_fids[best_idx]),
        xytext=(14, 40), textcoords="offset points", fontsize=7, color=TEXT,
        arrowprops=dict(arrowstyle="-", color=SPINE, lw=0.7,
                        connectionstyle="arc3,rad=0.0"),
        bbox=dict(boxstyle="round,pad=0.35", facecolor=ANNOT_BG,
                  edgecolor=GRID, linewidth=0.7),
    )
    ax.set_title("{} — Domain FID vs Steps".format(title_prefix),
                 fontsize=9, fontweight="bold", color=TEXT, pad=10)
    ax.set_xlabel("Sampling steps", fontsize=8, color=TEXT, labelpad=6)
    ax.set_ylabel("Domain FID ↓", fontsize=8, color=TEXT, labelpad=6)
    ax.set_xticks(steps)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    leg = ax.legend(frameon=True, fontsize=7.5, framealpha=0.95, edgecolor=GRID,
                    fancybox=False, loc="upper right", handlelength=1.8)
    leg.get_frame().set_linewidth(0.6)
    plt.tight_layout(pad=0.8)
    out = os.path.join(save_dir, "domain_fid_vs_steps.png")
    plt.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved: {}".format(out))

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    _apply_elegant_style(ax)
    ax.plot(steps, precisions, color=LINE2, linewidth=1.5, marker="o", markersize=4,
            markeredgewidth=0.6, markeredgecolor="white", markerfacecolor=LINE2,
            zorder=3, label="Precision (fidelity)")
    ax.plot(steps, recalls, color=LINE3, linewidth=1.5, marker="s", markersize=4,
            markeredgewidth=0.6, markeredgecolor="white", markerfacecolor=LINE3,
            zorder=3, label="Recall (diversity)")
    ax.set_title("{} — Precision & Recall vs Steps".format(title_prefix),
                 fontsize=9, fontweight="bold", color=TEXT, pad=10)
    ax.set_xlabel("Sampling steps", fontsize=8, color=TEXT, labelpad=6)
    ax.set_ylabel("Score ↑", fontsize=8, color=TEXT, labelpad=6)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(steps)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    leg = ax.legend(frameon=True, fontsize=7.5, framealpha=0.95, edgecolor=GRID,
                    fancybox=False, loc="lower right", handlelength=1.8)
    leg.get_frame().set_linewidth(0.6)
    plt.tight_layout(pad=0.8)
    out = os.path.join(save_dir, "precision_recall_vs_steps.png")
    plt.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved: {}".format(out))

    if has_inception:
        inc_fids = np.array([v if v is not None else float("nan") for v in inc_fids_raw])
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        _apply_elegant_style(ax)
        ax.plot(steps, inc_fids, color="#6B1A1A", linewidth=1.5, marker="o", markersize=4,
                markeredgewidth=0.6, markeredgecolor="white",
                zorder=3, label="Inception FID")
        ax.set_title("{} — Inception FID vs Steps".format(title_prefix),
                     fontsize=9, fontweight="bold", color=TEXT, pad=10)
        ax.set_xlabel("Sampling steps", fontsize=8, color=TEXT, labelpad=6)
        ax.set_ylabel("Inception FID ↓", fontsize=8, color=TEXT, labelpad=6)
        ax.set_xticks(steps)
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        leg = ax.legend(frameon=True, fontsize=7.5, framealpha=0.95, edgecolor=GRID,
                        fancybox=False, loc="upper right", handlelength=1.8)
        leg.get_frame().set_linewidth(0.6)
        plt.tight_layout(pad=0.8)
        out = os.path.join(save_dir, "inception_fid_vs_steps.png")
        plt.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("Saved: {}".format(out))


if __name__ == "__main__":
    cfg = EvalConfig()
    evaluate_multistep_sampling_grid(cfg)