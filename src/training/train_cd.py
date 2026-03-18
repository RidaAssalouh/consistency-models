"""Training utilities for consistency distillation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.consistency.distillation import sample_cd_training_batch
from src.loss.distillation_loss import (
    DistillationLossOutput,
    constant_loss_weight,
    distillation_loss,
)
from src.consistency.ema import EMA
from src.dynamics.ode_solver import ODESolver


Tensor = torch.Tensor


@dataclass
class CDTrainStepOutput:
    """Structured output for one consistency-distillation optimization step."""

    loss: float
    metrics: dict[str, float]


def _to_float_metrics(metrics: dict[str, Tensor | float | int]) -> dict[str, float]:
    """Convert scalar metric values to plain Python floats."""
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            out[key] = float(value.detach().item())
        else:
            out[key] = float(value)
    return out


def _extract_inputs(batch: Any, device: torch.device) -> Tensor:
    """Extract input tensor x from a dataloader batch.

    Supported batch formats:
    - x
    - (x, y)
    - {"x": x}
    - {"image": x}
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)

    if isinstance(batch, (tuple, list)):
        if len(batch) == 0:
            raise ValueError("Received an empty tuple/list batch.")
        return batch[0].to(device)

    if isinstance(batch, dict):
        if "x" in batch:
            return batch["x"].to(device)
        if "image" in batch:
            return batch["image"].to(device)
        raise ValueError(
            f"Unsupported dict batch keys: {list(batch.keys())}. "
            "Expected one of: 'x', 'image'."
        )

    raise TypeError(f"Unsupported batch type: {type(batch)!r}.")


def train_cd_step(
    model: nn.Module,
    ema: EMA,
    teacher: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: Tensor,
    noise_levels: Tensor,
    solver: ODESolver,
    loss_weight_fn=constant_loss_weight,
    distance: str = "l2",
    grad_clip_norm: float | None = None,
    lpips_model: nn.Module | None = None,
) -> CDTrainStepOutput:
    """Run one optimization step of consistency distillation.

    Args:
        model: Trainable consistency model.
        ema: EMA wrapper containing the target model.
        teacher: Teacher vector field Phi(x, t).
        optimizer: Optimizer for the online model.
        x: Clean batch, shape (B, ...).
        noise_levels: 1D tensor [t_1, ..., t_N].
        solver: ODE solver used to construct x_hat_{t_n}.
        loss_weight_fn: Weighting function lambda(t).
        distance: Distillation distance type.
        grad_clip_norm: Optional gradient clipping threshold.
        lpips_model: Optional pre-instantiated LPIPS module.

    Returns:
        CDTrainStepOutput containing scalar loss and metrics.
    """
    model.train()
    teacher.eval()
    ema.eval()  #stopgrad

    optimizer.zero_grad(set_to_none=True)

    cd_batch = sample_cd_training_batch(
        x=x,
        noise_levels=noise_levels,
        teacher=teacher,
        solver=solver,
    )

    loss_out: DistillationLossOutput = distillation_loss(
        online_model=model,
        ema_model=ema.ema_model,
        batch=cd_batch,
        loss_weight_fn=loss_weight_fn,
        distance=distance,
        lpips_model=lpips_model,
    )

    loss_out.loss.backward()

    grad_norm = None
    if grad_clip_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

    optimizer.step()
    ema.update(model)

    metrics = _to_float_metrics(loss_out.metrics)
    if grad_norm is not None:
        metrics["grad_norm"] = float(grad_norm.item())

    return CDTrainStepOutput(
        loss=float(loss_out.loss.detach().item()),
        metrics=metrics,
    )


@torch.no_grad()
def evaluate_cd_epoch(
    model: nn.Module,
    ema: EMA,
    teacher: nn.Module,
    dataloader: DataLoader,
    noise_levels: Tensor,
    device: torch.device,
    solver: ODESolver,
    loss_weight_fn=constant_loss_weight,
    distance: str = "l2",
    lpips_model: nn.Module | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate the EMA model with the CD objective on a validation loader."""
    model.eval()
    teacher.eval()
    ema.eval()

    metric_sums: dict[str, float] = {}
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = _extract_inputs(batch, device=device)

        cd_batch = sample_cd_training_batch(
            x=x,
            noise_levels=noise_levels.to(device=device, dtype=x.dtype),
            teacher=teacher,
            solver=solver,
        )

        loss_out = distillation_loss(
            online_model=model,
            ema_model=ema.ema_model,
            batch=cd_batch,
            loss_weight_fn=loss_weight_fn,
            distance=distance,
            lpips_model=lpips_model,
        )

        metrics = _to_float_metrics(loss_out.metrics)
        for key, value in metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + value

        num_batches += 1

    if num_batches == 0:
        raise ValueError("Validation dataloader produced zero batches.")

    return {key: value / num_batches for key, value in metric_sums.items()}


def train_cd_epoch(
    model: nn.Module,
    ema: EMA,
    teacher: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    noise_levels: Tensor,
    device: torch.device,
    solver: ODESolver,
    loss_weight_fn=constant_loss_weight,
    distance: str = "l2",
    grad_clip_norm: float | None = None,
    lpips_model: nn.Module | None = None,
    log_every: int = 100,
) -> dict[str, float]:
    """Train for one epoch of consistency distillation."""
    metric_sums: dict[str, float] = {}
    num_steps = 0

    for step_idx, batch in enumerate(dataloader):
        x = _extract_inputs(batch, device=device)
        noise_levels_batch = noise_levels.to(device=device, dtype=x.dtype)

        step_out = train_cd_step(
            model=model,
            ema=ema,
            teacher=teacher,
            optimizer=optimizer,
            x=x,
            noise_levels=noise_levels_batch,
            solver=solver,
            loss_weight_fn=loss_weight_fn,
            distance=distance,
            grad_clip_norm=grad_clip_norm,
            lpips_model=lpips_model,
        )

        for key, value in step_out.metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + value

        num_steps += 1

        if log_every > 0 and (step_idx + 1) % log_every == 0:
            print(
                f"[train] step={step_idx + 1} "
                f"loss={step_out.loss:.6f} "
                f"t_n={step_out.metrics.get('t_n_mean', float('nan')):.4f} "
                f"t_nplus1={step_out.metrics.get('t_nplus1_mean', float('nan')):.4f}"
            )

    if num_steps == 0:
        raise ValueError("Training dataloader produced zero batches.")

    return {key: value / num_steps for key, value in metric_sums.items()}


def fit_cd(
    model: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    noise_levels: Tensor,
    device: torch.device,
    num_epochs: int,
    ema_decay: float = 0.999,
    solver_method: str = "euler",
    loss_weight_fn=constant_loss_weight,
    distance: str = "l2",
    grad_clip_norm: float | None = None,
    lpips_model: nn.Module | None = None,
    log_every: int = 100,
) -> tuple[nn.Module, EMA, list[dict[str, float]]]:
    """Fit a consistency model with consistency distillation.

    Returns:
        model: Trained online model.
        ema: EMA wrapper containing the target model.
        history: Per-epoch metrics.
    """
    model = model.to(device)
    teacher = teacher.to(device)
    teacher.eval()

    ema = EMA(model=model, decay=ema_decay, update_buffers=True, device=device)
    solver = ODESolver(method=solver_method)

    history: list[dict[str, float]] = []

    for epoch in range(num_epochs):
        train_metrics = train_cd_epoch(
            model=model,
            ema=ema,
            teacher=teacher,
            optimizer=optimizer,
            dataloader=train_loader,
            noise_levels=noise_levels,
            device=device,
            solver=solver,
            loss_weight_fn=loss_weight_fn,
            distance=distance,
            grad_clip_norm=grad_clip_norm,
            lpips_model=lpips_model,
            log_every=log_every,
        )

        epoch_metrics = {f"train/{k}": v for k, v in train_metrics.items()}

        if val_loader is not None:
            val_metrics = evaluate_cd_epoch(
                model=model,
                ema=ema,
                teacher=teacher,
                dataloader=val_loader,
                noise_levels=noise_levels,
                device=device,
                solver=solver,
                loss_weight_fn=loss_weight_fn,
                distance=distance,
                lpips_model=lpips_model,
            )
            epoch_metrics.update({f"val/{k}": v for k, v in val_metrics.items()})

        history.append(epoch_metrics)

        summary = " ".join(f"{k}={v:.6f}" for k, v in epoch_metrics.items())
        print(f"[epoch {epoch + 1}/{num_epochs}] {summary}")

    return model, ema, history