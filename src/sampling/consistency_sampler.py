"""Placeholder sampler as a compatibility shim."""

from __future__ import annotations

import torch

from ..config import ConsistencyConfig
from ..models import ConsistencyModel


def consistency_sample(
    model: ConsistencyModel,
    config: ConsistencyConfig,
    num_steps: int = 5,
) -> torch.Tensor:
    """Dummy sampler that repeatedly feeds noise through the model."""

    model.eval()
    noise = torch.randn(
        1,
        config.model.input_channels,
        28,
        28,
        device=config.training.device,
    )
    with torch.no_grad():
        sample = noise
        for _ in range(num_steps):
            sample = model(sample)
    # TODO: Replace with the consistency sampling procedure described in the paper.
    return sample
