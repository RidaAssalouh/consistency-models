from __future__ import annotations

import copy
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from torch import Tensor
import random


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
    epsilon: float,
    T: float,
    num_levels: int,
    rho: float = 7.0,
    device: torch.device | None = None,
) -> Tensor:
    if num_levels < 2:
        raise ValueError(f"num_levels must be >= 2, got {num_levels}.")
    step = torch.linspace(0.0, 1.0, num_levels, device=device)
    return (
        epsilon ** (1.0 / rho)
        + step * (T ** (1.0 / rho) - epsilon ** (1.0 / rho))
    ) ** rho


def add_noise(x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    z = torch.randn_like(x)
    t_ = t
    if t_.ndim == 0:
        t_ = t_[None]
    while t_.ndim < x.ndim:
        t_ = t_.unsqueeze(-1)
    return x + t_ * z, z


def edm_preconditioning_teacher(
    x: Tensor, model_out: Tensor, t: Tensor, sigma_data: float = 0.5
) -> Tensor:
    sd2 = sigma_data ** 2
    t2 = t ** 2
    c_skip = (sd2 / (t2 + sd2))[:, None]
    c_out  = (t * sigma_data / (t2 + sd2).sqrt())[:, None]
    return c_skip * x + c_out * model_out


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
        emb = fourier_embedding(t, self.embed_dim)
        h = torch.cat([x, emb], dim=-1)
        h = self.act(self.input_proj(h))
        for block in self.blocks:
            h = block(h)
        h = self.act(h)
        return edm_preconditioning_teacher(x, self.output_proj(h), t, self.sigma_data)


class StudentMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        embed_dim: int = 32,
        num_blocks: int = 4,
        sigma_data: float = 0.5,
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



@torch.no_grad()
def euler_ode_step(
    teacher: TeacherMLP,
    x_t_next: Tensor,
    t_next: Tensor,
    t_cur: Tensor,
) -> Tensor:
    B = x_t_next.shape[0]
    t_next_b = t_next.expand(B)
    denoised = teacher(x_t_next, t_next_b)
    score    = (x_t_next - denoised) / t_next_b[:, None]
    dt       = (t_cur - t_next_b)[:, None]
    return x_t_next + dt * score



def consistency_distillation_loss(
    student: StudentMLP,
    ema_student: StudentMLP,
    teacher: TeacherMLP,
    x0: Tensor,
    noise_levels: Tensor,
    lam: float = 1.0,
) -> Tensor:

    B      = x0.shape[0]
    N      = len(noise_levels) 
    device = x0.device
    n_idx = torch.randint(0, N - 1, (B,), device=device)
    t_n      = noise_levels[n_idx] 
    t_n1     = noise_levels[n_idx + 1] 
    x_t_n1, _ = add_noise(x0, t_n1)
    x_hat_t_n = _euler_step_batched(teacher, x_t_n1, t_n1, t_n)
    pred_online = student(x_t_n1, t_n1)
    with torch.no_grad():
        pred_target = ema_student(x_hat_t_n, t_n)

    loss = lam * pseudo_huber(pred_online, pred_target)
    return loss


def _euler_step_batched(
    teacher: TeacherMLP,
    x_t_next: Tensor,
    t_next: Tensor,
    t_cur: Tensor,
) -> Tensor:
    with torch.no_grad():
        denoised = teacher(x_t_next, t_next)
    t_next_ = t_next[:, None] 
    score   = (x_t_next - denoised) / t_next_
    dt      = (t_cur - t_next)[:, None]
    return x_t_next + dt * score



def ema_update(ema_model: nn.Module, online_model: nn.Module, mu: float) -> None:
    with torch.no_grad():
        for p_ema, p_online in zip(ema_model.parameters(), online_model.parameters()):
            p_ema.data.mul_(mu).add_(p_online.data, alpha=1.0 - mu)


@torch.no_grad()
def consistency_sample(
    model: StudentMLP,
    n_samples: int,
    sigma_max: float = 80.0,
    device: torch.device | None = None,
) -> Tensor:
    """
    Single-step generation: x ~ N(0, σ_max²I)  →  f_θ(x, σ_max).
    The consistency model maps any (x_t, t) directly to x_0.
    """
    device = device or next(model.parameters()).device
    model.eval()
    x = torch.randn(n_samples, 2, device=device) * sigma_max
    t = torch.full((n_samples,), sigma_max, device=device)
    return model(x, t)


@torch.no_grad()
def consistency_sample_multistep(
    model: StudentMLP,
    n_samples: int,
    sigmas: list[float],
    epsilon: float,
    device: torch.device | None = None,
) -> Tensor:
    device = device or next(model.parameters()).device
    model.eval()

    x = torch.randn(n_samples, 2, device=device) * sigmas[0]

    for i, sigma in enumerate(sigmas):
        t = torch.full((n_samples,), sigma, device=device)
        x = model(x, t)

        if i < len(sigmas) - 1:
            sigma_next = sigmas[i + 1]
            noise_scale = (sigma_next**2 - epsilon**2) ** 0.5
            x = x + noise_scale * torch.randn_like(x)

    return x


def get_data(n: int = 10_000, noise: float = 0.03, device: torch.device | None = None) -> Tensor:
    X, _ = make_moons(n_samples=n, noise=noise, random_state=42)
    X = X - X.mean(0)
    X = X / X.std(0)
    return torch.tensor(X, dtype=torch.float32, device=device)

def pseudo_huber(a: Tensor, b: Tensor, c: float = 0.1) -> Tensor:
    diff = (a - b).pow(2).sum(dim=-1)           # (B,)
    return (torch.sqrt(diff + c**2) - c).mean()



SIGMA_MIN  = 0.002
SEED = 50
SIGMA_MAX  = 4.0
SIGMA_DATA = 0.5
EPSILON    = 0.002
RHO        = 7.0
N_LEVELS   = 20

EMA_MU     = 0.99 
LR         =  0.0009514840198718756
BATCH_SIZE = 512
N_STEPS    = 30_000
LOG_EVERY  = 1_000

N_SAMPLES  = 5_000
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEACHER_CKPT = "storage/denoiser.pt" 




def train_cd(teacher: TeacherMLP, data: Tensor) -> tuple[StudentMLP, list[float]]:
    seed_everything(SEED)
    noise_levels = build_noise_levels(
        epsilon=SIGMA_MIN, T=SIGMA_MAX, num_levels=N_LEVELS,
        rho=RHO, device=DEVICE,
    )

    student = StudentMLP(
        hidden_dim=256, embed_dim=32, num_blocks=6,
        sigma_data=SIGMA_DATA, epsilon=EPSILON,
    ).to(DEVICE)

    ema_student = copy.deepcopy(student)
    for p in ema_student.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(student.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_STEPS)

    losses = []
    print(f"\nConsistency Distillation on {DEVICE}")
    print(f"Student params : {sum(p.numel() for p in student.parameters()):,}")
    print(f"Noise levels   : {N_LEVELS}  |  σ ∈ [{SIGMA_MIN}, {SIGMA_MAX}]")
    print(f"EMA µ          : {EMA_MU}  |  batch: {BATCH_SIZE}  |  steps: {N_STEPS}\n")
    print(f"{'step':>8}  {'loss':>10}")

    for step in range(1, N_STEPS + 1):
        idx = torch.randint(len(data), (BATCH_SIZE,), device=DEVICE)
        x0  = data[idx]

        loss = consistency_distillation_loss(
            student      = student,
            ema_student  = ema_student,
            teacher      = teacher,
            x0           = x0,
            noise_levels = noise_levels,
        )

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        scheduler.step()

        ema_update(ema_student, student, mu=EMA_MU)

        losses.append(loss.item())

        if step % LOG_EVERY == 0 or step == 1:
            avg = sum(losses[-LOG_EVERY:]) / len(losses[-LOG_EVERY:])
            print(f"{step:>8}  {avg:>10.5f}")

    return ema_student, losses





def load_teacher(ckpt_path: str, device: torch.device) -> TeacherMLP:
    teacher = TeacherMLP(hidden_dim=256, embed_dim=32, num_blocks=6, sigma_data=SIGMA_DATA)
    if Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location=device)
        teacher.load_state_dict(state)
        print(f"Loaded teacher from {ckpt_path}")
    else:
        print(f"[warn] {ckpt_path} not found — using random teacher weights for demo.")
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher




def plot(
    data: Tensor,
    losses: list[float],
    samples_1step: np.ndarray,
    samples_multi: np.ndarray,
) -> None:
    data_np = data.cpu().numpy()
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Consistency Distillation", fontsize=13)

    kw_data   = dict(s=3,  alpha=0.3, color="#1D9E75", rasterized=True)
    kw_sample = dict(s=6,  alpha=0.6, color="#D85A30", rasterized=True)
    kw_multi  = dict(s=6,  alpha=0.6, color="#534AB7", rasterized=True)

    axes[0].scatter(*data_np.T, **kw_data)
    axes[0].set_title("Training data")

    def bc_score(real, fake, bins=80):
        lo = min(real.min(), fake.min()) - 0.3
        hi = max(real.max(), fake.max()) + 0.3
        b  = np.linspace(lo, hi, bins)
        hr, _, _ = np.histogram2d(real[:,0], real[:,1], bins=[b,b], density=True)
        hf, _, _ = np.histogram2d(fake[:,0], fake[:,1], bins=[b,b], density=True)
        hr /= hr.sum() + 1e-8; hf /= hf.sum() + 1e-8
        return float(np.sqrt(hr * hf).sum())

    score_1  = bc_score(data_np, samples_1step)
    score_m = bc_score(data_np, samples_multi)

    axes[1].scatter(*samples_1step.T, **kw_sample)
    axes[1].set_title(f"CD — 1-step (BC score = {score_1:.4f})")

    axes[2].scatter(*samples_multi.T, **kw_multi)
    axes[2].set_title(f"CD —  3-step (BC score = {score_m:.4f})")

    window = max(1, len(losses) // 200)
    smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
    axes[3].plot(smooth, linewidth=1.2, color="#534AB7")
    axes[3].set_title("CD training loss")
    axes[3].set_xlabel("step"); axes[3].set_ylabel("pseudo-Huber loss")
    axes[3].grid(True, alpha=0.3)

    for ax in axes[:3]:
        ax.set_aspect("equal"); ax.axis("off")

    plt.tight_layout()
    plt.savefig("make_moons_results_/cd_results.png", dpi=150, bbox_inches="tight")
    print("Saved make_moons_results_/cd_results.png")
    plt.show()


if __name__ == "__main__":
    teacher = load_teacher(TEACHER_CKPT, DEVICE)
    data    = get_data(device=DEVICE)
    student, losses = train_cd(teacher, data)
    s1 = consistency_sample(student, N_SAMPLES, sigma_max=SIGMA_MAX, device=DEVICE).cpu().numpy()
    sm = consistency_sample_multistep(student, N_SAMPLES, sigmas=[4.0, 1.0, 0.01], device=DEVICE, epsilon=EPSILON).cpu().numpy()
    print(f"\n1-step  samples: {s1.shape}")
    print(f"3-step  samples: {sm.shape}")
    plot(data, losses, s1, sm)
    torch.save(student.state_dict(), "storage/student_cd.pt")
    print("Saved student_cd.pt")