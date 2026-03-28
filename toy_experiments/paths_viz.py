from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import torch
import torch.nn as nn
from torch import Tensor
from sklearn.datasets import make_moons


CKPT_PATHS = [
    "consistency_models/consistency_models_ckpts/student_ct_toy.pt",
    "consistency_models/consistency_models_ckpts/student_cd_toy.pt",
]
OUT_DIR = "make_moons_results_/flow_line_geometry"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_DATA = 10000
DATA_NOISE = 0.05

XMIN, XMAX = -2.8, 2.8
YMIN, YMAX = -2.2, 2.2

STREAM_GRID_N = 260
STREAM_DENSITY = 2.2
T_STREAM = 6.0

SIGMA_DATA = 1.0
EPSILON = 0.002



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


def edm_preconditioning_student(
    x: Tensor, model_out: Tensor, t: Tensor, sigma_data: float, epsilon: float
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
        sigma_data: float = 1.0,
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


def get_data(n: int = 10000, noise: float = 0.05, device=None) -> Tensor:
    X, _ = make_moons(n_samples=n, noise=noise, random_state=42)
    X = X - X.mean(0)
    X = X / X.std(0)
    return torch.tensor(X, dtype=torch.float32, device=device)



def infer_architecture_from_state_dict(state: dict) -> tuple[int, int, int]:
    hidden_dim = state["input_proj.weight"].shape[0]
    embed_dim = state["input_proj.weight"].shape[1] - 2
    block_ids = sorted(
        {
            int(k.split(".")[1])
            for k in state.keys()
            if k.startswith("blocks.")
        }
    )
    num_blocks = max(block_ids) + 1
    return hidden_dim, embed_dim, num_blocks


@torch.no_grad()
def vector_field(model: nn.Module, pts: Tensor, t: float) -> Tensor:
    t_batch = torch.full((pts.shape[0],), float(t), device=pts.device)
    fx = model(pts, t_batch)
    return fx - pts


def make_grid(xmin, xmax, ymin, ymax, n):
    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, n)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    return xx, yy, pts


def load_model_from_ckpt(ckpt_path: str) -> nn.Module:
    try:
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=DEVICE)

    hidden_dim, embed_dim, num_blocks = infer_architecture_from_state_dict(state)

    model = StudentMLP(
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        sigma_data=SIGMA_DATA,
        epsilon=EPSILON,
    ).to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def title_from_ckpt_path(path: str) -> str:
    low = path.lower()
    if "student_cd" in low or "/cd_" in low or "_cd_" in low:
        return "Stream field CD"
    if "student_ct" in low or "/ct_" in low or "_ct_" in low:
        return "Stream field CT"
    return "Stream field"



def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    data = get_data(n=N_DATA, noise=DATA_NOISE, device=DEVICE).cpu().numpy()

    xx, yy, pts_np = make_grid(XMIN, XMAX, YMIN, YMAX, STREAM_GRID_N)
    pts = torch.tensor(pts_np, dtype=torch.float32, device=DEVICE)

    fields = []
    speeds = []

    for ckpt_path in CKPT_PATHS:
        model = load_model_from_ckpt(ckpt_path)
        v = vector_field(model, pts, T_STREAM).cpu().numpy()
        u = v[:, 0].reshape(xx.shape)
        w = v[:, 1].reshape(yy.shape)
        speed = np.sqrt(u**2 + w**2)
        fields.append((u, w))
        speeds.append(speed)

    vmin = min(s.min() for s in speeds)
    vmax = max(s.max() for s in speeds)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")

    fig = plt.figure(figsize=(14.8, 6.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045], wspace=0.08)

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
    ]
    cax = fig.add_subplot(gs[0, 2])

    for ax, ckpt_path, (u, w), speed in zip(axes, CKPT_PATHS, fields, speeds):
        ax.scatter(
            data[:, 0],
            data[:, 1],
            s=3,
            alpha=0.08,
            color="black",
            rasterized=True,
            zorder=1,
        )

        ax.streamplot(
            xx,
            yy,
            u,
            w,
            density=STREAM_DENSITY,
            color=speed,
            cmap=cmap,
            norm=norm,
            linewidth=1.0,
            arrowsize=0.85,
            maxlength=5.0,
            zorder=2,
        )

        ax.set_title(f"{title_from_ckpt_path(ckpt_path)}", fontsize=15)
        ax.set_xlim(XMIN, XMAX)
        ax.set_ylim(YMIN, YMAX)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.12)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"$\|f_\theta(x,t)-x\|$", rotation=90, labelpad=10, fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    out_path = os.path.join(OUT_DIR, "flow_field_stream_compare_t6.png")
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()