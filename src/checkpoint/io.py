"""Simple checkpoint I/O helpers."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    path: os.PathLike,
    ema_state: dict | None = None,
    extra_state: dict | None = None,
) -> None:
    """Persist a lightweight checkpoint to disk.

    Args:
        model: Model to save.
        optimizer: Optimizer state to save.
        epoch: Current epoch number.
        path: Directory where checkpoint will be written.
        ema_state: Optional EMA model state_dict.
        extra_state: Optional extra metadata/state.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }

    if ema_state is not None:
        payload["ema_state"] = ema_state

    if extra_state is not None:
        payload["extra_state"] = extra_state

    torch.save(payload, path / f"checkpoint_epoch_{epoch}.pt")