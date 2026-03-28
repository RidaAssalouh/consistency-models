from __future__ import annotations

from pathlib import Path
import random

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from torch import Tensor


SEED = 50
SIGMA_MAX = 3.0
SIGMA_DATA = 0.5
EPSILON = 0.002

N_SAMPLES = 5_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DENOISE_LAMBDAS = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]
MULTISTEP_SIGMAS = [3.0, 2.0, 1.0, 0.5,0.01]

CKPT_DIR = Path("toy_dataset")
OUT_DIR = Path("toy_dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)



def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fourier_embedding(t: Tensor, embed_dim: int) -> Tensor:
    assert embed_dim % 2 == 0
    half = embed_dim // 2
    freqs = torch.arange(half, device=t.device, dtype=t.dtype) / half
    log_t = torch.log(t.clamp(min=1e-5))
    angles = log_t[:, None] * (2 ** (freqs[None, :] * 4.0))
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def _broadcast_time_like_x(t: Tensor, x: Tensor) -> Tensor:
    while t.ndim < x.ndim:
        t = t.unsqueeze(-1)
    return t


def bc_score(real, fake, bins=80):
    lo = min(real.min(), fake.min()) - 0.3
    hi = max(real.max(), fake.max()) + 0.3
    b = np.linspace(lo, hi, bins)
    hr, _, _ = np.histogram2d(real[:, 0], real[:, 1], bins=[b, b], density=True)
    hf, _, _ = np.histogram2d(fake[:, 0], fake[:, 1], bins=[b, b], density=True)
    hr /= hr.sum() + 1e-8
    hf /= hf.sum() + 1e-8
    return float(np.sqrt(hr * hf).sum())


def get_data(
    n: int = 10_000,
    noise: float = 0.03,
    device: torch.device | None = None,
) -> Tensor:
    X, _ = make_moons(n_samples=n, noise=noise, random_state=42)
    X = X - X.mean(0)
    X = X / X.std(0)
    return torch.tensor(X, dtype=torch.float32, device=device)


def lambda_to_ckpt_name(lmbda: float) -> str:
    return f"student_lambda_{lmbda:.4g}.pt"



def edm_preconditioning_student(
    x: Tensor,
    model_out: Tensor,
    t: Tensor,
    sigma_data: float,
    epsilon: float,
) -> Tensor:
    t_exp = _broadcast_time_like_x(t, x)
    sd2 = sigma_data ** 2
    t_shift = t_exp - epsilon
    c_skip = sd2 / (t_shift ** 2 + sd2)
    c_out = sigma_data * t_shift / (sd2 + t_shift ** 2).sqrt()
    return c_skip * x + c_out * model_out



class ResBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.net(x))


class StudentMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        embed_dim: int = 32,
        num_blocks: int = 4,
        sigma_data: float = 0.5,
        epsilon: float = 1e-3,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.sigma_data = sigma_data
        self.epsilon = epsilon
        self.input_proj = nn.Linear(2 + embed_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_proj = nn.Linear(hidden_dim, 2)
        self.act = nn.SiLU()

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        emb = fourier_embedding(t, self.embed_dim)
        h = torch.cat([x_t, emb], dim=-1)
        h = self.act(self.input_proj(h))
        for block in self.blocks:
            h = block(h)
        h = self.act(h)
        return edm_preconditioning_student(
            x_t, self.output_proj(h), t, self.sigma_data, self.epsilon
        )


@torch.no_grad()
def consistency_sample(
    model: StudentMLP,
    n_samples: int,
    sigma_max: float = 80.0,
    device: torch.device | None = None,
) -> Tensor:
    device = device or next(model.parameters()).device
    model.eval()
    x = torch.randn(n_samples, 2, device=device) * sigma_max
    t = torch.full((n_samples,), sigma_max, device=device)
    return model(x, t)


@torch.no_grad()
def consistency_sample_multistep(
    model: StudentMLP,
    n_samples: int,
    sigmas: list[float],
    epsilon: float,
    device: torch.device | None = None,
) -> Tensor:
    device = device or next(model.parameters()).device
    model.eval()

    x = torch.randn(n_samples, 2, device=device) * sigmas[0]

    for i, sigma in enumerate(sigmas):
        t = torch.full((n_samples,), sigma, device=device)
        x = model(x, t)

        if i < len(sigmas) - 1:
            sigma_next = sigmas[i + 1]
            variance = max(sigma_next ** 2 - epsilon ** 2, 0.0)
            noise_scale = variance ** 0.5
            x = x + noise_scale * torch.randn_like(x)

    return x



BG = "white"
TEXT = "#222222"
GRID = "#D9DDE3"
LINE = "#2F6DB3"
LINE2 = "#C05A2B"
ACCENT = "#2AA876"
ACCENT2 = "#7A4FB7"


def _apply_elegant_style(ax):
    ax.set_facecolor(BG)

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=7.5,
        colors=TEXT,
        length=3,
        width=0.7,
        pad=2,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        colors=TEXT,
        length=2,
        width=0.5,
    )

    ax.grid(True, which="major", color=GRID, linewidth=0.7, alpha=0.7)
    ax.grid(True, which="minor", color=GRID, linewidth=0.45, alpha=0.35)
    ax.set_axisbelow(True)


def plot_bc_vs_lambda_elegant(
    lambdas,
    bc_1step,
    bc_multistep,
    save_path,
    title="BC score vs denoising penalty",
):
    lambdas = np.asarray(lambdas, dtype=float)
    bc_1step = np.asarray(bc_1step, dtype=float)
    bc_multistep = np.asarray(bc_multistep, dtype=float)

    if not (len(lambdas) == len(bc_1step) == len(bc_multistep)):
        raise ValueError("Input arrays must have the same length.")

    order = np.argsort(lambdas)
    lambdas = lambdas[order]
    bc_1step = bc_1step[order]
    bc_multistep = bc_multistep[order]

    pos = lambdas[lambdas > 0]
    if len(pos) == 0:
        raise ValueError("Need at least one positive lambda for log-scale plotting.")

    min_pos = pos.min()
    lambda_display = lambdas.copy()
    lambda_display[lambda_display == 0] = min_pos / 3.0

    best1_idx = int(np.argmax(bc_1step))
    bestm_idx = int(np.argmax(bc_multistep))

    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    _apply_elegant_style(ax)

    ax.plot(
        lambda_display,
        bc_1step,
        color=LINE,
        linewidth=1.5,
        marker="o",
        markersize=4,
        markeredgewidth=0.6,
        markeredgecolor="white",
        markerfacecolor=LINE,
        zorder=3,
        solid_capstyle="round",
        label="1-step BC",
    )

    ax.plot(
        lambda_display,
        bc_multistep,
        color=LINE2,
        linewidth=1.5,
        marker="o",
        markersize=4,
        markeredgewidth=0.6,
        markeredgecolor="white",
        markerfacecolor=LINE2,
        zorder=3,
        solid_capstyle="round",
        label="Multi-step BC",
    )

    ax.scatter(
        [lambda_display[best1_idx]],
        [bc_1step[best1_idx]],
        s=60,
        color=ACCENT,
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
        label="Best 1-step",
    )

    ax.scatter(
        [lambda_display[bestm_idx]],
        [bc_multistep[bestm_idx]],
        s=60,
        color=ACCENT2,
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
        label="Best multi-step",
    )

    ax.set_xscale("log")
    ax.set_title(title, fontsize=7, fontweight="bold", color=TEXT, pad=10)
    ax.set_xlabel(r"Denoising penalty $\lambda$", fontsize=8, color=TEXT, labelpad=6)
    ax.set_ylabel("BC score", fontsize=8, color=TEXT, labelpad=6)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    ax.set_xticks(lambda_display)
    xticklabels = []
    for lam in lambdas:
        if lam == 0:
            xticklabels.append("0")
        elif lam >= 1:
            xticklabels.append(f"{lam:g}")
        else:
            xticklabels.append(f"{lam:.0e}")
    ax.set_xticklabels(xticklabels)

    leg = ax.legend(
        frameon=True,
        fontsize=7.2,
        framealpha=0.95,
        edgecolor=GRID,
        fancybox=False,
        loc="best",
        handlelength=1.8,
        handleheight=0.9,
    )
    leg.get_frame().set_linewidth(0.6)

    plt.tight_layout(pad=0.8)
    plt.savefig(save_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved elegant BC plot to: {save_path}")



def main():
    seed_everything(SEED)

    data = get_data(device=DEVICE)
    data_np = data.cpu().numpy()

    all_lambdas = []
    all_bc_1step = []
    all_bc_multistep = []

    for lmbda in DENOISE_LAMBDAS:
        ckpt_path = CKPT_DIR / lambda_to_ckpt_name(lmbda)

        if not ckpt_path.exists():
            print(f"[warn] Missing checkpoint: {ckpt_path}")
            continue

        model = StudentMLP(
            hidden_dim=128,
            embed_dim=32,
            num_blocks=4,
            sigma_data=SIGMA_DATA,
            epsilon=EPSILON,
        ).to(DEVICE)

        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()

        s1 = consistency_sample(
            model,
            N_SAMPLES,
            sigma_max=SIGMA_MAX,
            device=DEVICE,
        ).cpu().numpy()

        sm = consistency_sample_multistep(
            model,
            N_SAMPLES,
            sigmas=MULTISTEP_SIGMAS,
            epsilon=EPSILON,
            device=DEVICE,
        ).cpu().numpy()

        bc_1 = bc_score(data_np, s1, bins=80)
        bc_m = bc_score(data_np, sm, bins=80)

        print(
            f"lambda={lmbda:.4g} | "
            f"BC 1-step={bc_1:.6f} | "
            f"BC multi-step={bc_m:.6f}"
        )

        all_lambdas.append(float(lmbda))
        all_bc_1step.append(float(bc_1))
        all_bc_multistep.append(float(bc_m))

    all_lambdas = np.array(all_lambdas, dtype=float)
    all_bc_1step = np.array(all_bc_1step, dtype=float)
    all_bc_multistep = np.array(all_bc_multistep, dtype=float)

    np.savez(
        OUT_DIR / "bc_results_from_checkpoints.npz",
        lambdas=all_lambdas,
        bc_1step=all_bc_1step,
        bc_multistep=all_bc_multistep,
    )
    print(f"Saved arrays to: {OUT_DIR / 'bc_results_from_checkpoints.npz'}")

    plot_bc_vs_lambda_elegant(
        lambdas=all_lambdas,
        bc_1step=all_bc_1step,
        bc_multistep=all_bc_multistep,
        save_path=OUT_DIR / "bc_vs_lambda_elegant.png",
        title="BC score vs denoising penalty",
    )


if __name__ == "__main__":
    main()