from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml


@dataclass
class DataConfig:
    """Dataset-related parameters."""

    dataset_name: str = "mnist"
    batch_size: int = 128
    num_workers: int = 4


@dataclass
class ModelConfig:
    """Model shape and architecture knobs."""

    hidden_dim: int = 256
    input_channels: int = 1
    output_dims: int = 1


@dataclass
class TrainingConfig:
    """Hyper-parameters for training."""

    epochs: int = 1
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval: int = 100
    checkpoint_dir: str = "checkpoints"


@dataclass
class ConsistencyConfig:
    """Root config that bundles dataset, model, and training overrides."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _load_subconfig(dataclass_type: Any, overrides: Dict[str, Any]) -> Any:
    field_names = set(dataclass_type.__dataclass_fields__)
    filtered = {k: v for k, v in overrides.items() if k in field_names}
    return dataclass_type(**filtered)


def load_config(path: Optional[os.PathLike] = None) -> ConsistencyConfig:
    """Load the config from a YAML file while keeping defaults for missing fields."""

    if path is None:
        return ConsistencyConfig()
    raw = yaml.safe_load(Path(path).read_text()) or {}
    data = _load_subconfig(DataConfig, raw.get("data", {}))
    model = _load_subconfig(ModelConfig, raw.get("model", {}))
    training = _load_subconfig(TrainingConfig, raw.get("training", {}))
    return ConsistencyConfig(data=data, model=model, training=training)


def save_config(config: ConsistencyConfig, path: os.PathLike) -> None:
    """Persist the configuration in YAML form for reproducibility."""

    yaml.safe_dump(dataclasses.asdict(config), Path(path).open("w"))
