"""Evaluation routines for consistency experiments."""

from __future__ import annotations

from typing import Iterable

import torch

from ..config import ConsistencyConfig
from ..loss.distillation_loss import consistency_loss
from ..models import ConsistencyModel


def evaluate_loss(
    model: ConsistencyModel,
    dataloader: Iterable[torch.Tensor],
    config: ConsistencyConfig,
) -> float:
    """Tune this helper once we have a validation set."""

    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch, _ in dataloader:
            batch = batch.to(config.training.device)
            pred = model(batch)
            loss = consistency_loss(pred, batch)
            total_loss += float(loss)
            count += 1
    return total_loss / max(1, count)
