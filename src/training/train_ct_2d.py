from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from ..checkpoint.io import save_checkpoint
from ..consistency.ema import EMA
from ..dynamics.noise import build_noise_levels
from ..models import ConsistencyMLP, DiffusionDenoiserMLP


LossType = Literal["l1", "l2"]


@dataclass
class Consistency2DTrainConfig:
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    sigma_min: float = 0.002
    sigma_max: float = 1.0
    rho: float = 7.0
    num_scales: int = 40

    ema_decay: float = 0.999
    loss_type: LossType = "l2"

    checkpoint_dir: str = "checkpoints/consistency_2d"
    checkpoint_every: int = 50


def pairwise_consistency_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    loss_type: LossType = "l2",
) -> torch.Tensor:
    if loss_type == "l2":
        return ((x - y) ** 2).sum(dim=-1).mean()
    if loss_type == "l1":
        return (x - y).abs().sum(dim=-1).mean()
    raise ValueError(f"Unknown loss_type: {loss_type}")


@torch.no_grad()
def teacher_pf_ode_drift(
    teacher: DiffusionDenoiserMLP,
    x_t: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    VE-style probability flow ODE drift from a denoiser D(x_t, t) ≈ x0.

    Since x_t = x0 + t z, one has approximately:
        score(x_t, t) ≈ -(x_t - x0_hat) / t^2

    For VE SDE, the PF-ODE drift is:
        dx/dt = - t * score(x,t)
              ≈ (x_t - x0_hat) / t

    This is the clean toy-form we use here.
    """
    x0_hat = teacher(x_t, t)
    drift = (x_t - x0_hat) / t[:, None].clamp_min(1e-8)
    return drift


@torch.no_grad()
def euler_step_teacher(
    teacher: DiffusionDenoiserMLP,
    x: torch.Tensor,
    t_from: torch.Tensor,
    t_to: torch.Tensor,
) -> torch.Tensor:
    """
    One backward Euler step along the teacher PF-ODE from t_from to t_to.
    """
    drift = teacher_pf_ode_drift(teacher=teacher, x_t=x, t=t_from)
    dt = (t_to - t_from)[:, None]
    return x + dt * drift


@torch.no_grad()
def heun_step_teacher(
    teacher: DiffusionDenoiserMLP,
    x: torch.Tensor,
    t_from: torch.Tensor,
    t_to: torch.Tensor,
) -> torch.Tensor:
    """
    One backward Heun step along the teacher PF-ODE from t_from to t_to.
    """
    dt = (t_to - t_from)[:, None]

    d1 = teacher_pf_ode_drift(teacher=teacher, x_t=x, t=t_from)
    x_euler = x + dt * d1
    d2 = teacher_pf_ode_drift(teacher=teacher, x_t=x_euler, t=t_to)

    return x + 0.5 * dt * (d1 + d2)


def build_discrete_time_grid(
    sigma_min: float,
    sigma_max: float,
    num_scales: int,
    rho: float,
    device: torch.device,
) -> torch.Tensor:
    return build_noise_levels(
        epsilon=sigma_min,
        T=sigma_max,
        num_levels=num_scales,
        rho=rho,
        device=device,
    )


def sample_adjacent_time_indices(
    batch_size: int,
    num_scales: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample indices n and n+1 from the discrete schedule.

    Returned indices are valid for 0-based tensors:
        i_low in {0, ..., N-2}
        i_high = i_low + 1
    """
    i_low = torch.randint(
        low=0,
        high=num_scales - 1,
        size=(batch_size,),
        device=device,
    )
    i_high = i_low + 1
    return i_low, i_high


def consistency_distillation_loss(
    model: ConsistencyMLP,
    ema_model: ConsistencyMLP,
    teacher: DiffusionDenoiserMLP,
    x0: torch.Tensor,
    noise_levels: torch.Tensor,
    loss_type: LossType = "l2",
    solver: Literal["euler", "heun"] = "heun",
) -> torch.Tensor:
    """
    CD:
      - sample x_{t_{n+1}} = x0 + t_{n+1} z
      - transport it to t_n with teacher PF-ODE
      - match:
            f_theta(x_{t_{n+1}}, t_{n+1})
            f_theta_minus(x_hat_{t_n}, t_n)
    """
    device = x0.device
    batch_size = x0.shape[0]

    i_low, i_high = sample_adjacent_time_indices(
        batch_size=batch_size,
        num_scales=noise_levels.shape[0],
        device=device,
    )

    t_low = noise_levels[i_low]
    t_high = noise_levels[i_high]

    z = torch.randn_like(x0)
    x_high = x0 + t_high[:, None] * z

    if solver == "euler":
        x_low_teacher = euler_step_teacher(
            teacher=teacher,
            x=x_high,
            t_from=t_high,
            t_to=t_low,
        )
    elif solver == "heun":
        x_low_teacher = heun_step_teacher(
            teacher=teacher,
            x=x_high,
            t_from=t_high,
            t_to=t_low,
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    pred_online = model(x_high, t_high)
    with torch.no_grad():
        pred_target = ema_model(x_low_teacher, t_low)

    return pairwise_consistency_distance(
        pred_online,
        pred_target,
        loss_type=loss_type,
    )


def consistency_training_loss(
    model: ConsistencyMLP,
    ema_model: ConsistencyMLP,
    x0: torch.Tensor,
    noise_levels: torch.Tensor,
    loss_type: LossType = "l2",
) -> torch.Tensor:
    """
    Independent CT:
      - sample shared z
      - construct x_{t_n} and x_{t_{n+1}} using same clean sample and same z
      - match:
            f_theta(x_{t_{n+1}}, t_{n+1})
            f_theta_minus(x_{t_n}, t_n)
    """
    device = x0.device
    batch_size = x0.shape[0]

    i_low, i_high = sample_adjacent_time_indices(
        batch_size=batch_size,
        num_scales=noise_levels.shape[0],
        device=device,
    )

    t_low = noise_levels[i_low]
    t_high = noise_levels[i_high]

    z = torch.randn_like(x0)
    x_low = x0 + t_low[:, None] * z
    x_high = x0 + t_high[:, None] * z

    pred_online = model(x_high, t_high)
    with torch.no_grad():
        pred_target = ema_model(x_low, t_low)

    return pairwise_consistency_distance(
        pred_online,
        pred_target,
        loss_type=loss_type,
    )


@torch.no_grad()
def evaluate_consistency_model(
    model: ConsistencyMLP,
    ema_model: ConsistencyMLP,
    val_loader: DataLoader,
    noise_levels: torch.Tensor,
    mode: Literal["cd", "ct"],
    loss_type: LossType,
    teacher: DiffusionDenoiserMLP | None = None,
) -> float:
    model.eval()
    ema_model.eval()

    total = 0.0
    count = 0

    for (x0,) in val_loader:
        x0 = x0.to(noise_levels.device)

        if mode == "cd":
            if teacher is None:
                raise ValueError("teacher must be provided for mode='cd'.")
            loss = consistency_distillation_loss(
                model=model,
                ema_model=ema_model,
                teacher=teacher,
                x0=x0,
                noise_levels=noise_levels,
                loss_type=loss_type,
                solver="heun",
            )
        elif mode == "ct":
            loss = consistency_training_loss(
                model=model,
                ema_model=ema_model,
                x0=x0,
                noise_levels=noise_levels,
                loss_type=loss_type,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        total += float(loss.item())
        count += 1

    return total / max(count, 1)


def train_consistency_distillation_2d(
    train_loader: DataLoader,
    val_loader: DataLoader,
    teacher: DiffusionDenoiserMLP,
    config: Consistency2DTrainConfig,
    model: ConsistencyMLP | None = None,
) -> tuple[ConsistencyMLP, EMA, dict[str, list[float]]]:
    device = torch.device(config.device)

    if model is None:
        model = ConsistencyMLP(
            input_dim=2,
            output_dim=2,
            hidden_dim=128,
            time_embedding_dim=64,
            num_hidden_layers=4,
        )

    model = model.to(device)
    teacher = teacher.to(device)
    teacher.eval()
    teacher.requires_grad_(False)

    ema = EMA(model, decay=config.ema_decay, update_buffers=True, device=device)
    optimizer = Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    noise_levels = build_discrete_time_grid(
        sigma_min=config.sigma_min,
        sigma_max=config.sigma_max,
        num_scales=config.num_scales,
        rho=config.rho,
        device=device,
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
            loss = consistency_distillation_loss(
                model=model,
                ema_model=ema.ema_model,
                teacher=teacher,
                x0=x0,
                noise_levels=noise_levels,
                loss_type=config.loss_type,
                solver="heun",
            )
            loss.backward()
            optimizer.step()
            ema.update(model)

            running += float(loss.item())
            n_batches += 1

        train_loss = running / max(n_batches, 1)
        val_loss = evaluate_consistency_model(
            model=model,
            ema_model=ema.ema_model,
            val_loader=val_loader,
            noise_levels=noise_levels,
            mode="cd",
            loss_type=config.loss_type,
            teacher=teacher,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"[Consistency-CD][Epoch {epoch:04d}/{config.epochs:04d}] "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )

        if epoch % config.checkpoint_every == 0 or epoch == config.epochs:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                path=config.checkpoint_dir,
            )

        model.train()

    return model, ema, history


def train_consistency_training_2d(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Consistency2DTrainConfig,
    model: ConsistencyMLP | None = None,
) -> tuple[ConsistencyMLP, EMA, dict[str, list[float]]]:
    device = torch.device(config.device)

    if model is None:
        model = ConsistencyMLP(
            input_dim=2,
            output_dim=2,
            hidden_dim=128,
            time_embedding_dim=64,
            num_hidden_layers=4,
        )

    model = model.to(device)

    ema = EMA(model, decay=config.ema_decay, update_buffers=True, device=device)
    optimizer = Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    noise_levels = build_discrete_time_grid(
        sigma_min=config.sigma_min,
        sigma_max=config.sigma_max,
        num_scales=config.num_scales,
        rho=config.rho,
        device=device,
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
            loss = consistency_training_loss(
                model=model,
                ema_model=ema.ema_model,
                x0=x0,
                noise_levels=noise_levels,
                loss_type=config.loss_type,
            )
            loss.backward()
            optimizer.step()
            ema.update(model)

            running += float(loss.item())
            n_batches += 1

        train_loss = running / max(n_batches, 1)
        val_loss = evaluate_consistency_model(
            model=model,
            ema_model=ema.ema_model,
            val_loader=val_loader,
            noise_levels=noise_levels,
            mode="ct",
            loss_type=config.loss_type,
            teacher=None,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"[Consistency-CT][Epoch {epoch:04d}/{config.epochs:04d}] "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )

        if epoch % config.checkpoint_every == 0 or epoch == config.epochs:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                path=config.checkpoint_dir,
            )

        model.train()

    return model, ema, history