from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from ..checkpoint.io import save_checkpoint
from ..models import DiffusionDenoiserMLP


@dataclass
class Diffusion2DTrainConfig:
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    sigma_min: float = 0.002
    sigma_max: float = 1.0
    checkpoint_dir: str = "checkpoints/diffusion_2d"
    checkpoint_every: int = 50


def sample_lognormal_sigmas(
    batch_size: int,
    sigma_min: float,
    sigma_max: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Sample noise levels in [sigma_min, sigma_max] on a log scale.
    """
    log_min = torch.log(torch.tensor(sigma_min, device=device))
    log_max = torch.log(torch.tensor(sigma_max, device=device))
    u = torch.rand(batch_size, device=device)
    return torch.exp(log_min + u * (log_max - log_min))


def diffusion_denoising_loss(
    model: nn.Module,
    x0: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
) -> torch.Tensor:
    """
    Train D_phi(xt, t) to predict x0 under xt = x0 + t z.
    """
    device = x0.device
    batch_size = x0.shape[0]

    t = sample_lognormal_sigmas(
        batch_size=batch_size,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        device=device,
    )
    z = torch.randn_like(x0)
    xt = x0 + t[:, None] * z

    x0_hat = model(xt, t)
    loss = F.mse_loss(x0_hat, x0)
    return loss


@torch.no_grad()
def evaluate_diffusion_loss(
    model: nn.Module,
    dataloader: DataLoader,
    sigma_min: float,
    sigma_max: float,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0

    for (x0,) in dataloader:
        x0 = x0.to(device)
        loss = diffusion_denoising_loss(
            model=model,
            x0=x0,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        total += float(loss.item())
        count += 1

    return total / max(count, 1)


def train_diffusion_2d(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Diffusion2DTrainConfig,
    model: DiffusionDenoiserMLP | None = None,
) -> tuple[DiffusionDenoiserMLP, dict[str, list[float]]]:
    device = torch.device(config.device)

    if model is None:
        model = DiffusionDenoiserMLP(
            input_dim=2,
            output_dim=2,
            hidden_dim=128,
            time_embedding_dim=64,
            num_hidden_layers=4,
        )

    model = model.to(device)
    optimizer = Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0

        for (x0,) in train_loader:
            x0 = x0.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = diffusion_denoising_loss(
                model=model,
                x0=x0,
                sigma_min=config.sigma_min,
                sigma_max=config.sigma_max,
            )
            loss.backward()
            optimizer.step()

            running += float(loss.item())
            n_batches += 1

        train_loss = running / max(n_batches, 1)
        val_loss = evaluate_diffusion_loss(
            model=model,
            dataloader=val_loader,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"[Diffusion][Epoch {epoch:04d}/{config.epochs:04d}] "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )

        if epoch % config.checkpoint_every == 0 or epoch == config.epochs:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                path=config.checkpoint_dir,
            )

    return model, history