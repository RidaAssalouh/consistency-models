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
) -> None:
    """Persist a lightweight checkpoint to disk."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(payload, path / f"checkpoint_epoch_{epoch}.pt")
