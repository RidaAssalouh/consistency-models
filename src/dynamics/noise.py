"""Noise-level utilities for consistency models."""

from __future__ import annotations

import torch


def build_noise_levels(
    epsilon: float,
    T: float,
    num_levels: int,
    rho: float = 7.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build the discretized noise levels t_1, ..., t_N used in the paper.

    The discretization follows:
        t_i = (epsilon^(1/rho) + ((i - 1) / (N - 1)) * (T^(1/rho) - epsilon^(1/rho)))^rho

    where:
        - t_1 = epsilon
        - t_N = T

    Args:
        epsilon: Minimum noise level.
        T: Maximum noise level.
        num_levels: Number N of discretization boundaries.
        rho: Schedule curvature parameter (paper uses rho = 7).
        device: Optional torch device.

    Returns:
        Tensor of shape (num_levels,) containing [t_1, ..., t_N].
    """
    if num_levels < 2:
        raise ValueError(f"num_levels must be at least 2, got {num_levels}.")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}.")
    if T <= epsilon:
        raise ValueError(f"T must be > epsilon, got T={T}, epsilon={epsilon}.")

    step = torch.linspace(0.0, 1.0, num_levels, device=device)
    noise_levels = (
        epsilon ** (1.0 / rho)
        + step * (T ** (1.0 / rho) - epsilon ** (1.0 / rho))
    ) ** rho
    return noise_levels


def add_noise(x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample x_t = x + t * z with z ~ N(0, I).

    Args:
        x: Clean data, shape (B, ...).
        t: Noise levels, shape (B,) or scalar.

    Returns:
        x_t: Noisy samples.
        z: Gaussian noise used to construct x_t.
    """
    z = torch.randn_like(x)

    if t.ndim == 0:
        t = t[None]

    while t.ndim < x.ndim:
        t = t.unsqueeze(-1)

    x_t = x + t * z
    return x_t, z