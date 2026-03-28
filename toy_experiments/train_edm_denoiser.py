"""
EDM-style denoiser trained on sklearn make_moons.
Sampling via Euler ODE solver (deterministic probability flow ODE).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from torch import Tensor


def build_noise_levels(
    epsilon: float,
    T: float,
    num_levels: int,
    rho: float = 7.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    if num_levels < 2:
        raise ValueError(f"num_levels must be at least 2, got {num_levels}.")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}.")
    if T <= epsilon:
        raise ValueError(f"T must be > epsilon, got T={T}, epsilon={epsilon}.")

    step = torch.linspace(0.0, 1.0, num_levels, device=device)
    noise_levels = (
        epsilon ** (1.0 / rho)
        + step * (T ** (1.0 / rho) - epsilon ** (1.0 / rho))
    ) ** rho
    return noise_levels


def add_noise(x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.randn_like(x)
    if t.ndim == 0:
        t = t[None]
    while t.ndim < x.ndim:
        t = t.unsqueeze(-1)
    x_t = x + t * z
    return x_t, z




def fourier_embedding(t: Tensor, embed_dim: int) -> Tensor:
    assert embed_dim % 2 == 0
    half = embed_dim // 2
    freqs = torch.arange(half, device=t.device, dtype=t.dtype)
    freqs = freqs / half
    log_t = torch.log(t.clamp(min=1e-5)) 
    angles = log_t[:, None] * (2 ** (freqs[None, :] * 4.0))
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  
    return emb


def edm_preconditioning_teacher(
    x: Tensor,
    model_out: Tensor,
    t: Tensor,
    sigma_data: float = 0.5,
) -> Tensor:
    """
    EDM-style skip + output scaling.

    D(x, t) = c_skip(t)*x + c_out(t)*F(x, t)

    where:
        c_skip = sigma_data^2 / (t^2 + sigma_data^2)
        c_out  = t * sigma_data / sqrt(t^2 + sigma_data^2)
    """
    sd2 = sigma_data ** 2
    t2 = t ** 2  
    c_skip = sd2 / (t2 + sd2)               
    c_out  = t * sigma_data / (t2 + sd2).sqrt()  


    c_skip = c_skip[:, None]
    c_out  = c_out[:, None]

    return c_skip * x + c_out * model_out



class ResBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.net(x))


class TeacherMLP(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 256,
        embed_dim: int = 32,
        num_blocks: int = 6,
        sigma_data: float = 0.5,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.sigma_data = sigma_data

        self.input_proj = nn.Linear(2 + embed_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_proj = nn.Linear(hidden_dim, 2)
        self.act = nn.SiLU()

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError(f"`x` must have shape (B, 2), got {tuple(x.shape)}.")
        if t.ndim != 1:
            raise ValueError(f"`t` must have shape (B,), got {tuple(t.shape)}.")
        if x.shape[0] != t.shape[0]:
            raise ValueError(
                f"Batch mismatch: x has batch {x.shape[0]}, t has batch {t.shape[0]}."
            )

        emb = fourier_embedding(t, self.embed_dim)
        h = torch.cat([x, emb], dim=-1)
        h = self.act(self.input_proj(h))

        for block in self.blocks:
            h = block(h)

        h = self.act(h)
        model_out = self.output_proj(h)

        return edm_preconditioning_teacher(
            x=x,
            model_out=model_out,
            t=t,
            sigma_data=self.sigma_data,
        )



def edm_loss(model: TeacherMLP, x0: Tensor, sigma_data: float = 0.5) -> Tensor:

    B = x0.shape[0]
    device = x0.device

    # Sample log-t uniformly (EDM recommendation)
    log_t = torch.empty(B, device=device).uniform_(math.log(SIGMA_MIN), math.log(SIGMA_MAX))
    t = log_t.exp()  

    x_t, _ = add_noise(x0, t)
    x_hat = model(x_t, t)


    sd2 = sigma_data ** 2
    lam = (t ** 2 + sd2) / (t * sigma_data) ** 2  
    loss = lam[:, None] * (x_hat - x0) ** 2      
    return loss.mean()



@torch.no_grad()
def euler_ode_sample(
    model: TeacherMLP,
    n_samples: int,
    num_steps: int = 100,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: torch.device | None = None,
) -> Tensor:

    device = device or next(model.parameters()).device
    model.eval()

    
    ts = build_noise_levels(
        epsilon=sigma_min,
        T=sigma_max,
        num_levels=num_steps + 1,
        rho=rho,
        device=device,
    ).flip(0)

    x = torch.randn(n_samples, 2, device=device) * sigma_max

    for i in range(num_steps):
        t_cur = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t_cur  

        t_batch = t_cur.expand(n_samples)
        d_cur = model(x, t_batch)          
        score = (x - d_cur) / t_cur
        x = x + score * dt               
    return x



def get_data(n: int = 10_000, noise: float = 0.05, device: torch.device | None = None) -> Tensor:
    X, _ = make_moons(n_samples=n, noise=noise, random_state=42)
    X = X - X.mean(0)
    X = X / X.std(0)
    return torch.tensor(X, dtype=torch.float32, device=device)


# Config

SIGMA_MIN   = 0.002
SIGMA_MAX   = 4.0
SIGMA_DATA  = 0.5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN_DIM  = 256
EMBED_DIM   = 32
NUM_BLOCKS  = 6

LR          = 3e-4
BATCH_SIZE  = 512
N_STEPS     = 20_000
LOG_EVERY   = 500

SAMPLE_STEPS = 100
N_SAMPLES    = 5000



def train() -> TeacherMLP:
    data = get_data(device=DEVICE)
    model = TeacherMLP(
        hidden_dim=HIDDEN_DIM,
        embed_dim=EMBED_DIM,
        num_blocks=NUM_BLOCKS,
        sigma_data=SIGMA_DATA,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_STEPS)

    losses = []
    print(f"Training on {DEVICE}  |  {sum(p.numel() for p in model.parameters()):,} params")
    print(f"{'step':>8}  {'loss':>10}")

    for step in range(1, N_STEPS + 1):
        idx = torch.randint(len(data), (BATCH_SIZE,), device=DEVICE)
        x0 = data[idx]

        loss = edm_loss(model, x0, sigma_data=SIGMA_DATA)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        losses.append(loss.item())

        if step % LOG_EVERY == 0 or step == 1:
            avg = sum(losses[-LOG_EVERY:]) / len(losses[-LOG_EVERY:])
            print(f"{step:>8}  {avg:>10.5f}")

    return model, losses



def plot(model: TeacherMLP, losses: list[float], data: Tensor) -> None:
    samples = euler_ode_sample(
        model,
        n_samples=N_SAMPLES,
        num_steps=SAMPLE_STEPS,
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
        device=DEVICE,
    ).cpu().numpy()

    data_np = data.cpu().numpy()

    def bc_score(real, fake, bins=80):
        lo = min(real.min(), fake.min()) - 0.3
        hi = max(real.max(), fake.max()) + 0.3
        b  = np.linspace(lo, hi, bins)
        hr, _, _ = np.histogram2d(real[:,0], real[:,1], bins=[b,b], density=True)
        hf, _, _ = np.histogram2d(fake[:,0], fake[:,1], bins=[b,b], density=True)
        hr /= hr.sum() + 1e-8; hf /= hf.sum() + 1e-8
        return float(np.sqrt(hr * hf).sum())

    score1 = bc_score(data_np, samples)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("EDM Denoiser", fontsize=13)

    # Training data
    ax = axes[0]
    ax.scatter(data_np[:, 0], data_np[:, 1], s=3, alpha=0.3, color="#1D9E75", rasterized=True)
    ax.set_title("Training data")
    ax.set_aspect("equal"); ax.axis("off")

    # Generated samples
    ax = axes[1]
    ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.6, color="#D85A30", rasterized=True)
    ax.set_title(f"{SAMPLE_STEPS}-step samples (BC score={score1:.4f})")
    ax.set_aspect("equal"); ax.axis("off")

    # Loss curve
    ax = axes[2]
    window = max(1, len(losses) // 200)
    smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
    ax.plot(smooth, linewidth=1.2, color="#534AB7")
    ax.set_title("Training loss")
    ax.set_xlabel("step"); ax.set_ylabel("EDM loss")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("make_moons_results_/denoiser_results.png", dpi=150, bbox_inches="tight")
    print("Saved make_moons_results_/denoiser_results.png")
    plt.show()


# Main

if __name__ == "__main__":
    model, losses = train()
    data = get_data(device=DEVICE)
    plot(model, losses, data)
    torch.save(model.state_dict(), "storage/denoiser.pt")
    print("Saved denoiser.pt")