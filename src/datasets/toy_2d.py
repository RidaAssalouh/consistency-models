from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Toy2DDataConfig:
    n_samples: int = 20000
    noise: float = 0.05
    batch_size: int = 256
    num_workers: int = 0
    val_fraction: float = 0.1
    seed: int = 42
    normalize: bool = True


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-8)
    x_std = (x - mean) / std
    return x_std, mean, std


def make_moons_tensors(
    n_samples: int = 20000,
    noise: float = 0.05,
    val_fraction: float = 0.1,
    seed: int = 42,
    normalize: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Build a 2D make_moons dataset and return train/val tensors.

    Returns:
        {
            "x_train": (N_train, 2),
            "x_val":   (N_val, 2),
            "mean":    (1, 2),
            "std":     (1, 2),
        }
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0,1), got {val_fraction}.")

    x, _ = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    x = x.astype(np.float32)

    if normalize:
        x, mean, std = _standardize(x)
    else:
        mean = np.zeros((1, 2), dtype=np.float32)
        std = np.ones((1, 2), dtype=np.float32)

    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_val = int(val_fraction * n_samples)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    x_train = torch.from_numpy(x[train_idx])
    x_val = torch.from_numpy(x[val_idx])
    mean_t = torch.from_numpy(mean.astype(np.float32))
    std_t = torch.from_numpy(std.astype(np.float32))

    return {
        "x_train": x_train,
        "x_val": x_val,
        "mean": mean_t,
        "std": std_t,
    }


def build_toy_dataloaders(
    config: Toy2DDataConfig,
) -> tuple[DataLoader, DataLoader, dict[str, torch.Tensor]]:
    """
    Returns:
        train_loader, val_loader, stats
    where stats contains mean/std tensors for possible de-normalization.
    """
    tensors = make_moons_tensors(
        n_samples=config.n_samples,
        noise=config.noise,
        val_fraction=config.val_fraction,
        seed=config.seed,
        normalize=config.normalize,
    )

    train_ds = TensorDataset(tensors["x_train"])
    val_ds = TensorDataset(tensors["x_val"])

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )

    stats = {
        "mean": tensors["mean"],
        "std": tensors["std"],
    }
    return train_loader, val_loader, stats


def denormalize_points(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Map normalized data back to original coordinates.
    """
    return x * std.to(x.device) + mean.to(x.device)