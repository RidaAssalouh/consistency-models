from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_time_shape(t: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Convert scalar or shape-(B,) time tensor into shape (B, 1).
    """
    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=torch.float32, device=device)

    t = t.to(device=device, dtype=torch.float32)

    if t.ndim == 0:
        t = t.expand(batch_size)

    if t.ndim != 1:
        raise ValueError(f"Expected t to have shape () or (B,), got {tuple(t.shape)}.")

    if t.shape[0] != batch_size:
        raise ValueError(f"Batch size mismatch: x has batch {batch_size}, t has shape {tuple(t.shape)}.")

    return t.unsqueeze(-1)  # (B, 1)


class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard sinusoidal embedding for scalar time/noise levels.
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim < 2:
            raise ValueError("embedding_dim must be >= 2.")
        self.embedding_dim = embedding_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: shape (B, 1)

        Returns:
            emb: shape (B, embedding_dim)
        """
        if t.ndim != 2 or t.shape[1] != 1:
            raise ValueError(f"Expected t of shape (B,1), got {tuple(t.shape)}.")

        half_dim = self.embedding_dim // 2
        device = t.device

        # frequencies from large to small scales
        freq_exponent = -math.log(10000.0) * torch.arange(
            half_dim, device=device, dtype=torch.float32
        ) / max(half_dim - 1, 1)
        freqs = torch.exp(freq_exponent)  # (half_dim,)

        args = t * freqs.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


class TimeMLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, time_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.time_proj = nn.Linear(time_dim, out_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.linear(x) + self.time_proj(t_emb)
        return self.act(h)


class BaseTimeConditionedMLP(nn.Module):
    """
    Shared backbone for small 2D toy models.

    Input:
        x: (B, 2)
        t: scalar or (B,)
    Output:
        y: (B, 2)
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 2,
        hidden_dim: int = 128,
        time_embedding_dim: int = 64,
        num_hidden_layers: int = 3,
    ) -> None:
        super().__init__()

        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1.")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.time_embedding_dim = time_embedding_dim
        self.num_hidden_layers = num_hidden_layers

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [TimeMLPBlock(hidden_dim, hidden_dim, time_embedding_dim) for _ in range(num_hidden_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected x of shape (B,{self.input_dim}), got {tuple(x.shape)}."
            )

        t = _ensure_time_shape(t, batch_size=x.shape[0], device=x.device)
        t_emb = self.time_embed(t)

        h = self.input_proj(x)
        h = F.silu(h)

        for block in self.blocks:
            h = h + block(h, t_emb)

        y = self.output_proj(h)
        return y


class DiffusionDenoiserMLP(BaseTimeConditionedMLP):
    """
    Teacher diffusion model for the toy setting.

    Trained to predict x0 from xt = x0 + t z.
    """
    pass


class ConsistencyMLP(BaseTimeConditionedMLP):
    """
    Consistency model for the toy setting.

    Also outputs a 2D point directly.
    """
    pass