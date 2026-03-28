import os
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class EvalConfig:
    data_root: str = "consistency_models/data"
    classifier_ckpt: str = "consistency_models/consistency_models_ckpts/fashionmnist_classifier_best.pt"

    ct_ckpt: str = "consistency_models/consistency_models_ckpts/ct_model_final_l2.pt"
    cd_ckpt: str = "consistency_models/consistency_models_ckpts/cd_model_final_l2.pt"

    save_dir: str = "consistency_models/coverage_eval_outputs"

    num_generated: int = 10000
    batch_size: int = 256
    num_workers: int = 4

    device: str = "cuda" if torch.cuda.is_available() else "cpu"




class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, 3, 1, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + x
        return self.act(out)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_ch, out_ch, 3, 1, 1),
            ResidualBlock(out_ch),
            nn.MaxPool2d(2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class FashionClassifier(nn.Module):
    def __init__(self, num_classes: int = 10, embedding_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(1, 64, 3, 1, 1),
            ResidualBlock(64),
        )
        self.stage1 = DownBlock(64, 128)
        self.stage2 = DownBlock(128, 256)
        self.stage3 = nn.Sequential(
            ConvBNAct(256, 256, 3, 1, 1),
            ResidualBlock(256),
            ConvBNAct(256, 256, 3, 1, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(256, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        z = self.embedding(x)
        return z

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.forward_features(x))




from models.ct_model_utils import UNet as CTUNet
from models.cd_model_utils import UNet as CDUNet



class CTConsistencyModel(nn.Module):
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


class CDConsistencyModel(nn.Module):
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



CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def denorm_from_gen_space(x: Tensor) -> Tensor:
    return (x + 1.0) / 2.0


def renorm_for_classifier(x_gen: Tensor) -> Tensor:
    """
    x_gen: tensor in [-1,1], shape (B,1,32,32)
    classifier expects normalized 28x28 FashionMNIST.
    """
    x = denorm_from_gen_space(x_gen).clamp(0.0, 1.0)
    x = F.interpolate(x, size=(28, 28), mode="bilinear", align_corners=False)
    x = (x - 0.2860) / 0.3530
    return x


@torch.no_grad()
def generate_one_step(
    model: nn.Module,
    num_samples: int,
    batch_size: int,
    sigma_max: float,
    image_size: int,
    in_channels: int,
    device: torch.device,
) -> Tensor:
    model.eval()
    outputs = []
    total = 0

    while total < num_samples:
        bsz = min(batch_size, num_samples - total)
        x_T = sigma_max * torch.randn(bsz, in_channels, image_size, image_size, device=device)
        sigma = torch.full((bsz,), sigma_max, device=device)
        x_gen = model(x_T, sigma).clamp(-1.0, 1.0)
        outputs.append(x_gen.cpu())
        total += bsz

    return torch.cat(outputs, dim=0)


@torch.no_grad()
def predict_labels(classifier: nn.Module, images_for_classifier: Tensor, batch_size: int, device: torch.device) -> Tensor:
    classifier.eval()
    preds = []
    for i in range(0, images_for_classifier.size(0), batch_size):
        batch = images_for_classifier[i:i + batch_size].to(device)
        logits = classifier(batch)
        pred = logits.argmax(dim=1)
        preds.append(pred.cpu())
    return torch.cat(preds, dim=0)


def relative_histogram_from_labels(labels: Tensor, num_classes: int = 10) -> np.ndarray:
    counts = torch.bincount(labels, minlength=num_classes).float()
    rel = counts / counts.sum()
    return rel.numpy()


def plot_histograms(ct_hist: np.ndarray, cd_hist: np.ndarray, save_path: str) -> None:
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
    })

    x = np.arange(len(CLASS_NAMES))
    width = 0.36

    fig, ax = plt.subplots(figsize=(15, 7))

    ct_color = "#4C72B0"
    cd_color = "#DD8452"

    bars_ct = ax.bar(
        x - width / 2,
        ct_hist,
        width=width,
        label="CT one-step",
        color=ct_color,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.92,
        zorder=3,
    )

    bars_cd = ax.bar(
        x + width / 2,
        cd_hist,
        width=width,
        label="CD one-step",
        color=cd_color,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.92,
        zorder=3,
    )

    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35, zorder=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=28, ha="right")
    ax.set_ylabel("Relative occurrence")
    ax.set_title("Label distribution of one-step generated FashionMNIST samples", pad=16)

    ymax = max(ct_hist.max(), cd_hist.max())
    ax.set_ylim(0, ymax * 1.18)

    def annotate_bars(bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.004,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                rotation=0,
            )

    annotate_bars(bars_ct)
    annotate_bars(bars_cd)

    leg = ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        borderpad=0.8,
    )
    leg.get_frame().set_edgecolor("#D0D0D0")
    leg.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def load_classifier(cfg: EvalConfig, device: torch.device) -> nn.Module:
    ckpt = torch.load(cfg.classifier_ckpt, map_location=device)

    embedding_dim = ckpt.get("embedding_dim", 128)
    num_classes = ckpt.get("num_classes", 10)

    model = FashionClassifier(num_classes=num_classes, embedding_dim=embedding_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_ct_model(cfg: EvalConfig, device: torch.device) -> nn.Module:
    ckpt = torch.load(cfg.ct_ckpt, map_location=device)
    c = ckpt["config"]

    backbone = CTUNet(
        in_channels=c["in_channels"],
        base_channels=c["base_channels"],
        channel_mults=tuple(c["channel_mults"]),
        num_res_blocks=c["num_res_blocks"],
        time_emb_dim=c["time_emb_dim"],
        dropout=c["dropout"],
    ).to(device)

    model = CTConsistencyModel(
        backbone=backbone,
        sigma_data=c["sigma_data"],
        eps=c["eps"],
    ).to(device)

    key = "ema_model" if "ema_model" in ckpt else "model"
    model.load_state_dict(ckpt[key])
    model.eval()
    return model


def load_cd_model(cfg: EvalConfig, device: torch.device) -> nn.Module:
    ckpt = torch.load(cfg.cd_ckpt, map_location=device)
    c = ckpt["config"]

    backbone = CDUNet(
        in_channels=c["in_channels"],
        base_channels=c["base_channels"],
        channel_mults=tuple(c["channel_mults"]),
        num_res_blocks=c["num_res_blocks"],
        time_emb_dim=c["time_emb_dim"],
        dropout=c["dropout"],
    ).to(device)

    model = CDConsistencyModel(
        backbone=backbone,
        sigma_data=c["sigma_data"],
        eps=c["eps"],
    ).to(device)

    key = "ema_model" if "ema_model" in ckpt else "model"
    model.load_state_dict(ckpt[key])
    model.eval()
    return model


def main() -> None:
    cfg = EvalConfig()
    os.makedirs(cfg.save_dir, exist_ok=True)

    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    classifier = load_classifier(cfg, device)
    ct_model = load_ct_model(cfg, device)
    cd_model = load_cd_model(cfg, device)

    ct_ckpt = torch.load(cfg.ct_ckpt, map_location="cpu")
    cd_ckpt = torch.load(cfg.cd_ckpt, map_location="cpu")

    ct_conf = ct_ckpt["config"]
    cd_conf = cd_ckpt["config"]

    print("Generating CT one-step samples...")
    ct_samples = generate_one_step(
        model=ct_model,
        num_samples=cfg.num_generated,
        batch_size=cfg.batch_size,
        sigma_max=ct_conf["sigma_max"],
        image_size=ct_conf["image_size"],
        in_channels=ct_conf["in_channels"],
        device=device,
    )

    print("Classifying CT samples...")
    ct_samples_for_clf = renorm_for_classifier(ct_samples)
    ct_pred = predict_labels(classifier, ct_samples_for_clf, cfg.batch_size, device)
    ct_hist = relative_histogram_from_labels(ct_pred, num_classes=10)

    print("Generating CD one-step samples...")
    cd_samples = generate_one_step(
        model=cd_model,
        num_samples=cfg.num_generated,
        batch_size=cfg.batch_size,
        sigma_max=cd_conf["sigma_max"],
        image_size=cd_conf["image_size"],
        in_channels=cd_conf["in_channels"],
        device=device,
    )

    print("Classifying CD samples...")
    cd_samples_for_clf = renorm_for_classifier(cd_samples)
    cd_pred = predict_labels(classifier, cd_samples_for_clf, cfg.batch_size, device)
    cd_hist = relative_histogram_from_labels(cd_pred, num_classes=10)

    np.save(os.path.join(cfg.save_dir, "ct_hist.npy"), ct_hist)
    np.save(os.path.join(cfg.save_dir, "cd_hist.npy"), cd_hist)

    fig_path = os.path.join(cfg.save_dir, "label_coverage_histogram_ct_vs_cd.png")
    plot_histograms(ct_hist, cd_hist, fig_path)

    print("\nRelative occurrences:")
    for i, name in enumerate(CLASS_NAMES):
        print(
            f"{i:2d} | {name:12s} | "
            f"ct={ct_hist[i]:.4f} | cd={cd_hist[i]:.4f}"
        )

    print(f"\nSaved histogram to: {fig_path}")


if __name__ == "__main__":
    main()