from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Protocol

import torch
import torch.nn as nn


Tensor = torch.Tensor
SolverMethod = Literal["euler", "heun"]


class VectorField(Protocol):
    """Protocol for ODE vector fields."""

    def __call__(self, x: Tensor, t: Tensor) -> Tensor:
        """Return the ODE drift evaluated at state x and time/noise level t."""
        ...


def _validate_state_and_times(x: Tensor, t_from: Tensor, t_to: Tensor) -> None:
    """Validate basic input assumptions for solver steps."""
    if x.ndim < 2:
        raise ValueError(
            f"`x` must have shape (B, ...), got shape {tuple(x.shape)}."
        )

    if t_from.ndim > 1:
        raise ValueError(
            f"`t_from` must be a scalar or a 1D tensor of shape (B,), "
            f"got shape {tuple(t_from.shape)}."
        )

    if t_to.ndim > 1:
        raise ValueError(
            f"`t_to` must be a scalar or a 1D tensor of shape (B,), "
            f"got shape {tuple(t_to.shape)}."
        )

    batch_size = x.shape[0]

    if t_from.ndim == 1 and t_from.shape[0] != batch_size:
        raise ValueError(
            f"`t_from` has shape {tuple(t_from.shape)} but batch size is {batch_size}."
        )

    if t_to.ndim == 1 and t_to.shape[0] != batch_size:
        raise ValueError(
            f"`t_to` has shape {tuple(t_to.shape)} but batch size is {batch_size}."
        )


def _ensure_tensor_on_x_device(t: Tensor | float, x: Tensor) -> Tensor:
    """Convert a float/scalar tensor into a tensor on the same device/dtype as x."""
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, device=x.device, dtype=x.dtype)
    else:
        t = t.to(device=x.device, dtype=x.dtype)
    return t


def _expand_like(t: Tensor, x: Tensor) -> Tensor:
    """Expand a scalar or (B,) tensor so it can broadcast against x of shape (B, ...)."""
    while t.ndim < x.ndim:
        t = t.unsqueeze(-1)
    return t


def _normalize_time_input(t: Tensor | float, x: Tensor) -> Tensor:
    """Normalize time input to a tensor on x's device/dtype."""
    t = _ensure_tensor_on_x_device(t, x)

    if t.ndim == 0:
        return t
    if t.ndim == 1:
        return t

    raise ValueError(
        f"Time tensor must be scalar or 1D of shape (B,), got shape {tuple(t.shape)}."
    )


def euler_step(
    x: Tensor,
    t_from: Tensor | float,
    t_to: Tensor | float,
    field: VectorField | nn.Module,
) -> Tensor:
    """Take one explicit Euler step from t_from to t_to.

    Computes:
        x_next = x + (t_to - t_from) * field(x, t_from)

    Args:
        x: Current state, shape (B, ...).
        t_from: Starting time / noise level, scalar or shape (B,).
        t_to: Target time / noise level, scalar or shape (B,).
        field: Vector field Phi(x, t) returning a tensor of the same shape as x.

    Returns:
        Next state approximation at t_to, shape (B, ...).
    """
    t_from = _normalize_time_input(t_from, x)
    t_to = _normalize_time_input(t_to, x)
    _validate_state_and_times(x, t_from, t_to)

    drift = field(x, t_from)
    if drift.shape != x.shape:
        raise ValueError(
            f"`field(x, t_from)` must return shape {tuple(x.shape)}, "
            f"got {tuple(drift.shape)}."
        )

    dt = _expand_like(t_to - t_from, x)
    return x + dt * drift


def heun_step(
    x: Tensor,
    t_from: Tensor | float,
    t_to: Tensor | float,
    field: VectorField | nn.Module,
) -> Tensor:
    """Take one Heun (RK2 / explicit trapezoid) step from t_from to t_to.

    Computes:
        k1 = field(x, t_from)
        x_pred = x + (t_to - t_from) * k1
        k2 = field(x_pred, t_to)
        x_next = x + (t_to - t_from) * 0.5 * (k1 + k2)

    Args:
        x: Current state, shape (B, ...).
        t_from: Starting time / noise level, scalar or shape (B,).
        t_to: Target time / noise level, scalar or shape (B,).
        field: Vector field Phi(x, t) returning a tensor of the same shape as x.

    Returns:
        Next state approximation at t_to, shape (B, ...).
    """
    t_from = _normalize_time_input(t_from, x)
    t_to = _normalize_time_input(t_to, x)
    _validate_state_and_times(x, t_from, t_to)

    dt = _expand_like(t_to - t_from, x)

    k1 = field(x, t_from)
    if k1.shape != x.shape:
        raise ValueError(
            f"`field(x, t_from)` must return shape {tuple(x.shape)}, "
            f"got {tuple(k1.shape)}."
        )

    x_pred = x + dt * k1

    k2 = field(x_pred, t_to)
    if k2.shape != x.shape:
        raise ValueError(
            f"`field(x_pred, t_to)` must return shape {tuple(x.shape)}, "
            f"got {tuple(k2.shape)}."
        )

    return x + dt * 0.5 * (k1 + k2)


def get_solver_step(method: SolverMethod) -> Callable[[Tensor, Tensor | float, Tensor | float, VectorField | nn.Module], Tensor]:
    """Return a solver step function from its string name."""
    if method == "euler":
        return euler_step
    if method == "heun":
        return heun_step
    raise ValueError(f"Unknown ODE solver method: {method!r}.")


@dataclass(frozen=True)
class ODESolver:
    """Thin wrapper around one-step ODE solvers.

    Example:
        solver = ODESolver(method="heun")
        x_next = solver.step(x, t_from=t_np1, t_to=t_n, field=teacher_field)
    """

    method: SolverMethod = "euler"

    @property
    def step_fn(
        self,
    ) -> Callable[[Tensor, Tensor | float, Tensor | float, VectorField | nn.Module], Tensor]:
        """Return the configured step function."""
        return get_solver_step(self.method)

    def step(
        self,
        x: Tensor,
        t_from: Tensor | float,
        t_to: Tensor | float,
        field: VectorField | nn.Module,
    ) -> Tensor:
        """Advance one solver step from t_from to t_to."""
        return self.step_fn(x=x, t_from=t_from, t_to=t_to, field=field)


@torch.no_grad()
def integrate_fixed_grid(
    x: Tensor,
    time_grid: Tensor,
    field: VectorField | nn.Module,
    method: SolverMethod = "euler",
    return_all: bool = False,
) -> Tensor | list[Tensor]:
    """Integrate across a fixed sequence of time points.

    This is a convenience utility for repeated one-step integration over a grid:
        time_grid = [t_0, t_1, ..., t_K]

    and performs:
        x_{k+1} = step(x_k, t_k -> t_{k+1})

    Args:
        x: Initial state, shape (B, ...).
        time_grid: 1D tensor of shape (K + 1,).
        field: Vector field Phi(x, t).
        method: Solver method name.
        return_all: If True, return all intermediate states including the initial one.

    Returns:
        If return_all=False: final state at time_grid[-1].
        If return_all=True: list [x_0, x_1, ..., x_K].
    """
    if time_grid.ndim != 1:
        raise ValueError(
            f"`time_grid` must be 1D, got shape {tuple(time_grid.shape)}."
        )
    if len(time_grid) < 2:
        raise ValueError("`time_grid` must contain at least two time points.")

    step_fn = get_solver_step(method)
    states: list[Tensor] = [x] if return_all else []

    x_curr = x
    for i in range(len(time_grid) - 1):
        t_from = time_grid[i]
        t_to = time_grid[i + 1]
        x_curr = step_fn(x=x_curr, t_from=t_from, t_to=t_to, field=field)
        if return_all:
            states.append(x_curr)

    return states if return_all else x_curr


__all__ = [
    "ODESolver",
    "SolverMethod",
    "VectorField",
    "euler_step",
    "heun_step",
    "get_solver_step",
    "integrate_fixed_grid",
]