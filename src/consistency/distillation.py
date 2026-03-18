"""Utilities for building consistency-distillation training batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from src.dynamics.ode_solver import ODESolver, SolverMethod


Tensor = torch.Tensor


@dataclass
class CDBatch:
    """Container for all minibatch objects needed by consistency distillation."""

    x_clean: Tensor
    indices: Tensor
    t_n: Tensor
    t_nplus1: Tensor
    noise: Tensor
    x_tnplus1: Tensor
    x_tn_hat: Tensor


def _expand_like(t: Tensor, x: Tensor) -> Tensor:
    """Expand a scalar or (B,) tensor so it broadcasts against x of shape (B, ...)."""
    while t.ndim < x.ndim:
        t = t.unsqueeze(-1)
    return t


def sample_interval_indices(
    batch_size: int,
    num_levels: int,
    device: torch.device,
) -> Tensor:
    """Sample interval indices uniformly.

    Mathematical convention:
        n in {1, ..., N-1}

    Python convention:
        idx in {0, ..., N-2}

    so that:
        t_n     = noise_levels[idx]
        t_nplus1 = noise_levels[idx + 1]

    Args:
        batch_size: Batch size B.
        num_levels: Number N of noise-level boundaries.
        device: Torch device.

    Returns:
        Tensor of shape (B,) with values in {0, ..., N-2}.
    """
    if num_levels < 2:
        raise ValueError(f"num_levels must be at least 2, got {num_levels}.")
    return torch.randint(0, num_levels - 1, (batch_size,), device=device)


def sample_noisy_next_state(
    x: Tensor,
    t_nplus1: Tensor,
) -> tuple[Tensor, Tensor]:
    """Sample x_{t_{n+1}} = x + t_{n+1} z with z ~ N(0, I).

    This is equivalent to:
        x_{t_{n+1}} ~ N(x, t_{n+1}^2 I)

    Args:
        x: Clean batch, shape (B, ...).
        t_nplus1: Noise levels at the next boundary, shape (B,).

    Returns:
        x_tnplus1: Noisy states, shape (B, ...).
        noise: Standard Gaussian noise z, shape (B, ...).
    """
    if x.ndim < 2:
        raise ValueError(f"x must have shape (B, ...), got {tuple(x.shape)}.")
    if t_nplus1.ndim != 1:
        raise ValueError(
            f"t_nplus1 must have shape (B,), got {tuple(t_nplus1.shape)}."
        )
    if t_nplus1.shape[0] != x.shape[0]:
        raise ValueError(
            f"Batch mismatch: x has batch {x.shape[0]}, "
            f"but t_nplus1 has shape {tuple(t_nplus1.shape)}."
        )

    noise = torch.randn_like(x)
    t_view = _expand_like(t_nplus1, x)
    x_tnplus1 = x + t_view * noise
    return x_tnplus1, noise


@torch.no_grad()
def construct_previous_state(
    x_tnplus1: Tensor,
    t_nplus1: Tensor,
    t_n: Tensor,
    teacher: nn.Module,
    solver: ODESolver | None = None,
    solver_method: SolverMethod = "euler",
) -> Tensor:
    """Construct x_hat_{t_n}^phi from x_{t_{n+1}}.

    This computes one solver step:
        x_hat_{t_n}^phi = SolverStep(x_{t_{n+1}}, t_{n+1} -> t_n; Phi)

    where Phi is represented by `teacher`.

    Args:
        x_tnplus1: States at t_{n+1}, shape (B, ...).
        t_nplus1: Starting noise levels, shape (B,).
        t_n: Target noise levels, shape (B,).
        teacher: Teacher vector field Phi(x, t).
        solver: Optional instantiated ODESolver.
        solver_method: Solver method if `solver` is not provided.

    Returns:
        Approximate states at t_n, shape (B, ...).
    """
    ode_solver = solver if solver is not None else ODESolver(method=solver_method)
    return ode_solver.step(
        x=x_tnplus1,
        t_from=t_nplus1,
        t_to=t_n,
        field=teacher,
    )


def sample_cd_training_batch(
    x: Tensor,
    noise_levels: Tensor,
    teacher: nn.Module,
    solver: ODESolver | None = None,
    solver_method: SolverMethod = "euler",
) -> CDBatch:
    """

    Implements the first part of Consistency Distillation algorithm:
        - sample x ~ D                      [x is already the sampled minibatch]
        - sample n uniformly
        - sample x_{t_{n+1}} ~ N(x, t_{n+1}^2 I)
        - construct x_hat_{t_n}^phi with the ODE solver and teacher field

    Args:
        x: Clean batch sampled from the dataset, shape (B, ...).
        noise_levels: 1D tensor [t_1, ..., t_N], shape (N,).
        teacher: Teacher vector field Phi(x, t).
        solver: Optional ODESolver instance.
        solver_method: Solver method used if `solver` is None.

    Returns:
        CDBatch containing all intermediate objects needed for the CD loss.
    """
    if x.ndim < 2:
        raise ValueError(f"x must have shape (B, ...), got {tuple(x.shape)}.")
    if noise_levels.ndim != 1:
        raise ValueError(
            f"noise_levels must be a 1D tensor of shape (N,), got {tuple(noise_levels.shape)}."
        )
    if len(noise_levels) < 2:
        raise ValueError("noise_levels must contain at least two boundaries.")

    batch_size = x.shape[0]
    device = x.device
    noise_levels = noise_levels.to(device=device, dtype=x.dtype)

    indices = sample_interval_indices(
        batch_size=batch_size,
        num_levels=len(noise_levels),
        device=device,
    )

    t_n = noise_levels[indices]
    t_nplus1 = noise_levels[indices + 1]

    x_tnplus1, noise = sample_noisy_next_state(
        x=x,
        t_nplus1=t_nplus1,
    )

    x_tn_hat = construct_previous_state(
        x_tnplus1=x_tnplus1,
        t_nplus1=t_nplus1,
        t_n=t_n,
        teacher=teacher,
        solver=solver,
        solver_method=solver_method,
    )

    return CDBatch(
        x_clean=x,
        indices=indices,
        t_n=t_n,
        t_nplus1=t_nplus1,
        noise=noise,
        x_tnplus1=x_tnplus1,
        x_tn_hat=x_tn_hat,
    )


__all__ = [
    "CDBatch",
    "construct_previous_state",
    "sample_cd_training_batch",
    "sample_interval_indices",
    "sample_noisy_next_state",
]