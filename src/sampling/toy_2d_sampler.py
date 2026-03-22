from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from ..models import ConsistencyMLP, DiffusionDenoiserMLP


@dataclass
class ToySamplerConfig:
    sigma_min: float = 0.002
    sigma_max: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def sample_initial_noise(
    num_samples: int,
    sigma_max: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Start from x_T ~ N(0, sigma_max^2 I).
    """
    return sigma_max * torch.randn(num_samples, 2, device=device)


@torch.no_grad()
def sample_diffusion_teacher(
    teacher: DiffusionDenoiserMLP,
    num_samples: int,
    sigma_max: float,
    num_steps: int = 100,
    sigma_min: float = 0.002,
) -> torch.Tensor:
    """
    Simple deterministic PF-ODE style sampling with repeated denoising updates.

    This is a toy sampler for the 2D experiment, not a faithful large-scale image sampler.
    """
    device = next(teacher.parameters()).device
    teacher.eval()

    x = sample_initial_noise(
        num_samples=num_samples,
        sigma_max=sigma_max,
        device=device,
    )

    # linearly decreasing time grid from sigma_max to sigma_min
    t_grid = torch.linspace(sigma_max, sigma_min, num_steps, device=device)

    for i in range(len(t_grid) - 1):
        t_cur = torch.full((num_samples,), t_grid[i], device=device)
        t_next = torch.full((num_samples,), t_grid[i + 1], device=device)

        x0_hat = teacher(x, t_cur)

        drift = (x - x0_hat) / t_cur[:, None].clamp_min(1e-8)
        dt = (t_next - t_cur)[:, None]
        x = x + dt * drift

    # final denoise
    t_final = torch.full((num_samples,), sigma_min, device=device)
    x = teacher(x, t_final)
    return x


@torch.no_grad()
def sample_consistency_one_step(
    model: ConsistencyMLP,
    num_samples: int,
    sigma_max: float,
) -> torch.Tensor:
    """
    One-step consistency sampling:
        x_T ~ N(0, sigma_max^2 I)
        x_hat = f_theta(x_T, T)
    """
    device = next(model.parameters()).device
    model.eval()

    x = sample_initial_noise(
        num_samples=num_samples,
        sigma_max=sigma_max,
        device=device,
    )
    t = torch.full((num_samples,), sigma_max, device=device)
    return model(x, t)


@torch.no_grad()
def sample_consistency_multi_step(
    model: ConsistencyMLP,
    num_samples: int,
    sigma_max: float,
    sigma_min: float,
    num_steps: int = 4,
) -> torch.Tensor:
    """
    Multi-step consistency sampling.

    At each stage:
      1. denoise with the consistency model at current noise level,
      2. if not final step, inject fresh Gaussian noise to the next lower scale.

    This is the standard practical spirit of consistency multi-step sampling.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}.")

    device = next(model.parameters()).device
    model.eval()

    x = sample_initial_noise(
        num_samples=num_samples,
        sigma_max=sigma_max,
        device=device,
    )

    if num_steps == 1:
        t = torch.full((num_samples,), sigma_max, device=device)
        return model(x, t)

    t_grid = torch.linspace(sigma_max, sigma_min, num_steps, device=device)

    for i in range(num_steps):
        t_cur = torch.full((num_samples,), t_grid[i], device=device)
        x = model(x, t_cur)

        if i < num_steps - 1:
            t_next = t_grid[i + 1]
            noise = torch.randn_like(x)
            x = x + t_next * noise

    return x


def save_2d_scatter(
    points: torch.Tensor,
    path: str | Path,
    title: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    points_np = points.detach().cpu().numpy()

    plt.figure(figsize=(5, 5))
    plt.scatter(points_np[:, 0], points_np[:, 1], s=5, alpha=0.7)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")

    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_2d_comparison_grid(
    real_data: torch.Tensor,
    diffusion_samples: torch.Tensor,
    cd_one_step: torch.Tensor,
    ct_one_step: torch.Tensor,
    cd_multi_step: torch.Tensor,
    ct_multi_step: torch.Tensor,
    path: str | Path,
) -> None:
    real_np = real_data.detach().cpu().numpy()
    diff_np = diffusion_samples.detach().cpu().numpy()
    cd1_np = cd_one_step.detach().cpu().numpy()
    ct1_np = ct_one_step.detach().cpu().numpy()
    cdm_np = cd_multi_step.detach().cpu().numpy()
    ctm_np = ct_multi_step.detach().cpu().numpy()

    all_pts = torch.cat(
        [
            real_data,
            diffusion_samples,
            cd_one_step,
            ct_one_step,
            cd_multi_step,
            ct_multi_step,
        ],
        dim=0,
    ).cpu()

    x_min = float(all_pts[:, 0].min()) - 0.3
    x_max = float(all_pts[:, 0].max()) + 0.3
    y_min = float(all_pts[:, 1].min()) - 0.3
    y_max = float(all_pts[:, 1].max()) + 0.3

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    panels = [
        (real_np, "Real data"),
        (diff_np, "Diffusion samples"),
        (cd1_np, "Consistency CD - 1 step"),
        (ct1_np, "Consistency CT - 1 step"),
        (cdm_np, "Consistency CD - multi-step"),
        (ctm_np, "Consistency CT - multi-step"),
    ]

    for ax, (pts, title) in zip(axes.flat, panels):
        ax.scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.7)
        ax.set_title(title)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()