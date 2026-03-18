"""Loss utilities for consistency distillation.

This module implements the second part of Algorithm 2 (Consistency Distillation):

    - online prediction at (x_{t_{n+1}}, t_{n+1})
    - target prediction at (x_hat_{t_n}^phi, t_n) using the EMA model
    - weighted discrepancy between the two

The ODE-based construction of x_hat_{t_n}^phi is handled upstream by
`src.consistency.distillation`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from src.consistency.distillation import CDBatch

try:
    import lpips
except ImportError:
    lpips = None


Tensor = torch.Tensor
Reduction = Literal["mean", "sum", "none"]
DistanceType = Literal["l1", "l2" , "lpips"]


@dataclass
class DistillationLossOutput:
    """Structured output for consistency distillation loss."""

    loss: Tensor
    online_pred: Tensor
    target_pred: Tensor
    per_example_loss: Tensor
    weights: Tensor
    metrics: dict[str, Tensor]


def _flatten_per_example(x: Tensor) -> Tensor:
    """Flatten all non-batch dimensions."""
    return x.reshape(x.shape[0], -1)


def _reduce_per_example(loss_tensor: Tensor, reduction: Reduction) -> Tensor:
    """Reduce a tensor of shape (B,) according to the requested reduction."""
    if reduction == "mean":
        return loss_tensor.mean()
    if reduction == "sum":
        return loss_tensor.sum()
    if reduction == "none":
        return loss_tensor
    raise ValueError(f"Unknown reduction: {reduction!r}.")


def _lp_distance_per_example(
    x: Tensor,
    y: Tensor,
    p: DistanceType = "l2",
) -> Tensor:
    """Compute per-example discrepancy between x and y for non-LPIPS distances."""
    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape, got {tuple(x.shape)} and {tuple(y.shape)}."
        )

    diff = x - y
    flat = _flatten_per_example(diff)

    if p == "l2":
        return (flat ** 2).mean(dim=1)

    if p == "l1":
        return flat.abs().mean(dim=1)

    raise ValueError(f"Unknown non-LPIPS distance type: {p!r}.")


def _validate_lpips_inputs(x: Tensor, y: Tensor) -> None:
    """Validate tensor shapes for LPIPS."""
    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape for LPIPS, got {tuple(x.shape)} and {tuple(y.shape)}."
        )
    if x.ndim != 4:
        raise ValueError(
            f"LPIPS expects image tensors of shape (B, C, H, W), got {tuple(x.shape)}."
        )
    if x.shape[1] not in (1, 3):
        raise ValueError(
            f"LPIPS expects 1 or 3 channels, got C={x.shape[1]}."
        )


def _prepare_for_lpips(x: Tensor) -> Tensor:
    """Prepare tensors for LPIPS.

    LPIPS is designed for image tensors, typically in [-1, 1].
    If the tensor has 1 channel, we replicate it to 3 channels.
    """
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return x


class LPIPSDistance(nn.Module):
    """Thin wrapper around the LPIPS perceptual distance.

    Returns one scalar distance per example, shape (B,).
    """

    def __init__(self, net: str = "alex") -> None:
        super().__init__()
        if lpips is None:
            raise ImportError(
                "LPIPS is not installed. Install it with `pip install lpips`."
            )
        self.metric = lpips.LPIPS(net=net)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        _validate_lpips_inputs(x, y)
        x = _prepare_for_lpips(x)
        y = _prepare_for_lpips(y)

        values = self.metric(x, y)
        return values.reshape(values.shape[0])


def constant_loss_weight(t: Tensor) -> Tensor:
    """Default constant weighting lambda(t) = 1."""
    return torch.ones_like(t)


def inverse_time_loss_weight(
    t: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Example weighting lambda(t) = 1 / (t + eps)."""
    return 1.0 / (t + eps)


def get_loss_weight_fn(name: str):
    if name == "constant":
        return constant_loss_weight
    if name == "inverse_time":
        return inverse_time_loss_weight
    raise ValueError(f"Unknown weight function: {name}")

def get_loss_weight_fn(name: str):
    if name == "constant":
        return constant_loss_weight
    if name == "inverse_time":
        return inverse_time_loss_weight
    raise ValueError(f"Unknown weight function: {name}")

@torch.no_grad()
def compute_distillation_target(
    ema_model: nn.Module,
    batch: CDBatch,
) -> Tensor:
    """Compute the stop-gradient target prediction."""
    return ema_model(batch.x_tn_hat, batch.t_n)


def compute_online_prediction(
    online_model: nn.Module,
    batch: CDBatch,
) -> Tensor:
    """Compute the online prediction."""
    return online_model(batch.x_tnplus1, batch.t_nplus1)


def distillation_loss(
    online_model: nn.Module,
    ema_model: nn.Module,
    batch: CDBatch,
    loss_weight_fn=constant_loss_weight,
    distance: DistanceType = "l2",
    reduction: Reduction = "mean",
    lpips_model: nn.Module | None = None,
) -> DistillationLossOutput:
    """Compute the consistency distillation loss.

    Args:
        online_model: Trainable consistency model f_theta.
        ema_model: EMA target model f_{theta^-}.
        batch: CDBatch produced upstream by consistency/distillation.py.
        loss_weight_fn: Function lambda(t) returning per-example weights of shape (B,).
        distance: Distance type: l1, l2, or lpips.
        reduction: Reduction over the batch.
        lpips_model: Pre-instantiated LPIPS module.

    Returns:
        DistillationLossOutput with loss, predictions, weights, and metrics.
    """
    online_pred = compute_online_prediction(online_model=online_model, batch=batch)
    target_pred = compute_distillation_target(ema_model=ema_model, batch=batch)

    if online_pred.shape != target_pred.shape:
        raise ValueError(
            "online and target predictions must have the same shape, "
            f"got {tuple(online_pred.shape)} and {tuple(target_pred.shape)}."
        )

    if distance == "lpips":
        lpips_metric = lpips_model if lpips_model is not None else LPIPSDistance()
        lpips_metric = lpips_metric.to(device=online_pred.device)
        per_example_distance = lpips_metric(online_pred, target_pred)
    else:
        per_example_distance = _lp_distance_per_example(
            x=online_pred,
            y=target_pred,
            p=distance
        )

    weights = loss_weight_fn(batch.t_nplus1)
    if weights.ndim != 1:
        raise ValueError(
            f"loss_weight_fn must return shape (B,), got {tuple(weights.shape)}."
        )
    if weights.shape[0] != per_example_distance.shape[0]:
        raise ValueError(
            f"Weight shape mismatch: got {tuple(weights.shape)} for "
            f"{per_example_distance.shape[0]} examples."
        )

    weighted_per_example = weights * per_example_distance
    loss = _reduce_per_example(weighted_per_example, reduction=reduction)

    metrics = {
        "loss": loss.detach() if isinstance(loss, torch.Tensor) else torch.tensor(loss),
        "unweighted_distance_mean": per_example_distance.mean().detach(),
        "weighted_distance_mean": weighted_per_example.mean().detach(),
        "weight_mean": weights.mean().detach(),
        "t_n_mean": batch.t_n.mean().detach(),
        "t_nplus1_mean": batch.t_nplus1.mean().detach(),
    }

    return DistillationLossOutput(
        loss=loss,
        online_pred=online_pred,
        target_pred=target_pred,
        per_example_loss=weighted_per_example,
        weights=weights,
        metrics=metrics,
    )


__all__ = [
    "DistillationLossOutput",
    "LPIPSDistance",
    "compute_distillation_target",
    "compute_online_prediction",
    "constant_loss_weight",
    "distillation_loss",
    "inverse_time_loss_weight",
    "get_loss_weight_fn"
]