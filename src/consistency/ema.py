"""Exponential Moving Average (EMA) utilities for model parameters.

In consistency distillation, the EMA model corresponds to θ^- in the paper:
    θ^- ← μ θ^- + (1 - μ) θ

This module provides:
- functional EMA update
- helper class to manage EMA model lifecycle
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn


Tensor = torch.Tensor


@torch.no_grad()
def update_ema_parameters(
    ema_model: nn.Module,
    model: nn.Module,
    decay: float,
) -> None:
    """Update EMA parameters in-place.

    Performs:
        θ^- ← decay * θ^- + (1 - decay) * θ

    Args:
        ema_model: Target EMA model (θ^-).
        model: Source model (θ).
        decay: EMA decay μ in [0, 1).
    """
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"EMA decay must be in [0, 1), got {decay}.")

    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())

    if ema_params.keys() != model_params.keys():
        raise ValueError("EMA model and source model must have identical parameter structure.")

    for name, param in model_params.items():
        ema_param = ema_params[name]

        if not param.requires_grad:
            continue

        ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)


@torch.no_grad()
def update_ema_buffers(
    ema_model: nn.Module,
    model: nn.Module,
) -> None:
    """Copy buffers (e.g., BatchNorm stats) from model to EMA model.

    EMA is typically applied only to parameters, not buffers.
    Buffers are copied directly.

    Args:
        ema_model: EMA model.
        model: Source model.
    """
    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())

    if ema_buffers.keys() != model_buffers.keys():
        raise ValueError("EMA model and source model must have identical buffer structure.")

    for name, buf in model_buffers.items():
        ema_buffers[name].data.copy_(buf.data)


@dataclass
class EMAConfig:
    """Configuration for EMA behavior."""

    decay: float = 0.999
    update_buffers: bool = True


class EMA:
    """Helper class managing an EMA copy of a model.

    Example:
        ema = EMA(model, decay=0.999)

        for step in training:
            ...
            ema.update(model)

        # use ema.ema_model for evaluation / target
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        update_buffers: bool = True,
        device: torch.device | None = None,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}.")

        self.decay = decay
        self.update_buffers_flag = update_buffers

        # Create a deep copy for EMA model
        self.ema_model = copy.deepcopy(model)

        # EMA model is not trainable
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

        if device is not None:
            self.ema_model.to(device)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA model from current model."""
        update_ema_parameters(
            ema_model=self.ema_model,
            model=model,
            decay=self.decay,
        )

        if self.update_buffers_flag:
            update_ema_buffers(
                ema_model=self.ema_model,
                model=model,
            )

    def state_dict(self) -> dict:
        """Return state dict of EMA model."""
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Load EMA model state."""
        self.ema_model.load_state_dict(state_dict)

    def to(self, device: torch.device) -> None:
        """Move EMA model to device."""
        self.ema_model.to(device)

    def eval(self) -> None:
        """Set EMA model to eval mode."""
        self.ema_model.eval()

    def train(self) -> None:
        """(Usually not needed) set EMA model to train mode."""
        self.ema_model.train()