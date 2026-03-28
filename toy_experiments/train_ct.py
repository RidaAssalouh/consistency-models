from __future__ import annotations

import copy
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from torch import Tensor



def fourier_embedding(t: Tensor, embed_dim: int) -> Tensor:
    assert embed_dim % 2 == 0
    half = embed_dim // 2
    freqs = torch.arange(half, device=t.device, dtype=t.dtype) / half
    log_t = torch.log(t.clamp(min=1e-5))
    angles = log_t[:, None] * (2 ** (freqs[None, :] * 4.0))
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def _broadcast_time_like_x(t: Tensor, x: Tensor) -> Tensor:
    while t.ndim < x.ndim:
        t = t.unsqueeze(-1)
    return t


def build_noise_levels(
    epsilon: float, T: float, num_levels: int,
    rho: float = 7.0, device=None,
) -> Tensor:
    if num_levels < 2:
        raise ValueError(f"num_levels must be >= 2, got {num_levels}.")
    step = torch.linspace(0.0, 1.0, num_levels, device=device)
    return (
        epsilon ** (1.0 / rho)
        + step * (T ** (1.0 / rho) - epsilon ** (1.0 / rho))
    ) ** rho



def edm_preconditioning_student(
    x: Tensor, model_out: Tensor, t: Tensor,
    sigma_data: float, epsilon: float,
) -> Tensor:
    t_exp = _broadcast_time_like_x(t, x)
    sd2 = sigma_data ** 2
    t_shift = t_exp - epsilon
    c_skip = sd2 / (t_shift ** 2 + sd2)
    c_out  = sigma_data * t_shift / (sd2 + t_shift ** 2).sqrt()
    return c_skip * x + c_out * model_out


class ResBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.net(x))


class StudentMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        embed_dim: int = 32,
        num_blocks: int = 4,
        sigma_data: float = 1.0,   # ← FIXED
        epsilon: float = 1e-3,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.sigma_data = sigma_data
        self.epsilon = epsilon
        self.input_proj = nn.Linear(2 + embed_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_proj = nn.Linear(hidden_dim, 2)
        self.act = nn.SiLU()

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        emb = fourier_embedding(t, self.embed_dim)
        h = torch.cat([x_t, emb], dim=-1)
        h = self.act(self.input_proj(h))
        for block in self.blocks:
            h = block(h)
        h = self.act(h)
        return edm_preconditioning_student(
            x_t, self.output_proj(h), t, self.sigma_data, self.epsilon
        )


# Step schedules

def schedule_N(k: int, K: int, N_0: int = 2, N_end: int = 200) -> int:
    return min(N_end, N_0 + int((N_end - N_0) * k / K))


def schedule_mu(k: int, K: int, mu_0: float = 0.9, mu_end: float = 0.999) -> float:
    return min(mu_end, mu_0 + (mu_end - mu_0) * k / K)


# Pseudo-Huber loss

def pseudo_huber(a: Tensor, b: Tensor, c: float = 0.001) -> Tensor:
    diff_sq = (a - b).pow(2).sum(dim=-1)
    return (torch.sqrt(diff_sq + c ** 2) - c).mean()


# CT loss

def consistency_training_loss(
    student: StudentMLP,
    ema_student: StudentMLP,
    x0: Tensor,
    noise_levels: Tensor,
    lam: float = 1.0,
) -> Tensor:
    B      = x0.shape[0]
    N      = len(noise_levels)
    device = x0.device

    n_idx = torch.randint(0, N - 1, (B,), device=device)
    t_n   = noise_levels[n_idx]
    t_n1  = noise_levels[n_idx + 1]

    z = torch.randn_like(x0)
    x_t_n1 = x0 + t_n1[:, None] * z
    x_t_n  = x0 + t_n [:, None] * z

    pred_online = student(x_t_n1, t_n1)
    with torch.no_grad():
        pred_target = ema_student(x_t_n, t_n)

    return lam * pseudo_huber(pred_online, pred_target)


# EMA update

def ema_update(ema_model: nn.Module, online_model: nn.Module, mu: float) -> None:
    with torch.no_grad():
        for p_ema, p_online in zip(ema_model.parameters(), online_model.parameters()):
            p_ema.data.mul_(mu).add_(p_online.data, alpha=1.0 - mu)


# Sampling  

@torch.no_grad()
def consistency_sample(
    model: StudentMLP,
    n_samples: int,
    sigma_max: float,
    device=None,
) -> Tensor:
    device = device or next(model.parameters()).device
    model.eval()
    x = torch.randn(n_samples, 2, device=device) * sigma_max
    t = torch.full((n_samples,), sigma_max, device=device)
    return model(x, t)


@torch.no_grad()
def consistency_sample_multistep(
    model: StudentMLP,
    n_samples: int,
    sigma_max: float,
    sigma_min: float,
    n_steps: int = 5,
    device=None,
) -> Tensor:
    device = device or next(model.parameters()).device
    model.eval()

    
    rho = 7.0
    steps = torch.linspace(0, 1, n_steps, device=device)
    sigmas = (
        sigma_max ** (1.0 / rho)
        + steps * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
    ) ** rho                              

    z   = torch.randn(n_samples, 2, device=device) * sigmas[0]
    t   = torch.full((n_samples,), sigmas[0].item(), device=device)
    x0  = model(z, t)

    for sig in sigmas[1:]:
        eps  = torch.randn_like(x0)
        x_noisy = x0 + sig.item() * eps
        t   = torch.full((n_samples,), sig.item(), device=device)
        x0  = model(x_noisy, t)

    return x0


# Data

def get_data(n: int = 10_000, noise: float = 0.05, device=None) -> Tensor:
    X, _ = make_moons(n_samples=n, noise=noise, random_state=42)
    X = X - X.mean(0)
    X = X / X.std(0)
    return torch.tensor(X, dtype=torch.float32, device=device)


# Config

SIGMA_MIN  = 0.002
SIGMA_MAX  = 4.0      
SIGMA_DATA = 1.0    
EPSILON    = 0.002
RHO        = 7.0

N_0        = 4
N_END      = 200       
MU_0       = 0.98
MU_END     = 0.999

LR         = 3e-4
BATCH_SIZE = 512
N_STEPS    = 50_000     
LOG_EVERY  = 5_000

N_SAMPLES  = 5_000
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MS_STEPS   = 5         



def train_ct(data: Tensor) -> tuple[StudentMLP, list[float]]:
    model = StudentMLP(
        hidden_dim=256, embed_dim=32, num_blocks=6,
        sigma_data=SIGMA_DATA, epsilon=EPSILON,
    ).to(DEVICE)

    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    losses = []
    print(f"\nConsistency Training (CT) on {DEVICE}")
    print(f"Model params : {sum(p.numel() for p in model.parameters()):,}")
    print(f"σ ∈ [{SIGMA_MIN}, {SIGMA_MAX}]  |  σ_data={SIGMA_DATA}")
    print(f"N: {N_0}→{N_END}  |  µ: {MU_0}→{MU_END}\n")
    print(f"{'step':>8}  {'loss':>10}  {'N(k)':>6}  {'µ(k)':>8}")

    for k in range(1, N_STEPS + 1):
        N_k  = schedule_N(k, N_STEPS, N_0, N_END)
        mu_k = schedule_mu(k, N_STEPS, MU_0, MU_END)

        noise_levels = build_noise_levels(
            epsilon=SIGMA_MIN, T=SIGMA_MAX,
            num_levels=N_k, rho=RHO, device=DEVICE,
        )

        idx  = torch.randint(len(data), (BATCH_SIZE,), device=DEVICE)
        x0   = data[idx]

        loss = consistency_training_loss(
            student=model, ema_student=ema_model,
            x0=x0, noise_levels=noise_levels,
        )

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema_update(ema_model, model, mu=mu_k)

        losses.append(loss.item())

        if k % LOG_EVERY == 0 or k == 1:
            avg = sum(losses[-LOG_EVERY:]) / len(losses[-LOG_EVERY:])
            print(f"{k:>8}  {avg:>10.6f}  {N_k:>6}  {mu_k:>8.4f}")

    return ema_model, losses


# Plot

def plot(data: Tensor, losses: list[float],
         s1: np.ndarray, sm: np.ndarray) -> None:
    data_np = data.cpu().numpy()

    # Compute 2-D histogram overlap (Bhattacharyya) for quantitative score
    def bc_score(real, fake, bins=80):
        lo = min(real.min(), fake.min()) - 0.3
        hi = max(real.max(), fake.max()) + 0.3
        b  = np.linspace(lo, hi, bins)
        hr, _, _ = np.histogram2d(real[:,0], real[:,1], bins=[b,b], density=True)
        hf, _, _ = np.histogram2d(fake[:,0], fake[:,1], bins=[b,b], density=True)
        hr /= hr.sum() + 1e-8; hf /= hf.sum() + 1e-8
        return float(np.sqrt(hr * hf).sum())

    score_1  = bc_score(data_np, s1)
    score_ms = bc_score(data_np, sm)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Consistency Training in Isolation ", fontsize=13)

    axes[0].scatter(*data_np.T, s=3, alpha=0.3, color="#1D9E75", rasterized=True)
    axes[0].set_title("Training data")

    axes[1].scatter(*s1.T, s=6, alpha=0.6, color="#D85A30", rasterized=True)
    axes[1].set_title(f"CT — 1-step  (BC score = {score_1:.3f})")

    axes[2].scatter(*sm.T, s=6, alpha=0.6, color="#534AB7", rasterized=True)
    axes[2].set_title(f"CT — {MS_STEPS}-step  (BC score = {score_ms:.3f})")

    window = max(1, len(losses) // 200)
    smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
    axes[3].plot(smooth, linewidth=1.2, color="#534AB7")
    axes[3].set_title("CT training loss")
    axes[3].set_xlabel("step"); axes[3].set_ylabel("pseudo-Huber loss")
    axes[3].grid(True, alpha=0.3)

    for ax in axes[:3]:
        ax.set_aspect("equal"); ax.axis("off")

    plt.tight_layout()
    out = "make_moons_results_/ct_results_v3.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.show()


# Main

if __name__ == "__main__":
    data = get_data(device=DEVICE)
    model, losses = train_ct(data)

    s1 = consistency_sample(
        model, N_SAMPLES, sigma_max=SIGMA_MAX, device=DEVICE
    ).cpu().numpy()

    sm = consistency_sample_multistep(
        model, N_SAMPLES,
        sigma_max=SIGMA_MAX,
        sigma_min=SIGMA_MIN * 10, 
        n_steps=MS_STEPS,
        device=DEVICE,
    ).cpu().numpy()

    print(f"\n1-step   std: {s1.std(0).round(3)}")
    print(f"{MS_STEPS}-step   std: {sm.std(0).round(3)}")
    print(f"Data     std: {data.cpu().numpy().std(0).round(3)}")

    plot(data, losses, s1, sm)
    torch.save(model.state_dict(), "student_ct_v3.pt")
    print("Saved student_ct_v3.pt")