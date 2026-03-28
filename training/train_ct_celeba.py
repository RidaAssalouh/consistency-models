"""
Consistency Training on CelebA 64x64
python train_consistency.py --data_root ./data --epochs 100 --loss_type l2
"""

import os
import math
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

import numpy as np

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)



class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(-math.log(10_000) * torch.arange(half) / half)
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Swish(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_log = torch.log(t / 4) * 0.25
        args  = t_log[:, None] * self.freqs[None]
        emb   = torch.cat([args.sin(), args.cos()], dim=-1) 
        return self.mlp(emb) 


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act   = Swish()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.t_proj(self.act(t_emb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.norm  = nn.GroupNorm(32, ch)
        self.qkv   = nn.Conv2d(ch, ch * 3, 1)
        self.proj  = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h   = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.heads, C // self.heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]   # (B, heads, head_ch, HW)
        scale = (C // self.heads) ** -0.5
        attn  = torch.einsum("bhci,bhcj->bhij", q, k) * scale
        attn  = attn.softmax(dim=-1)
        out   = torch.einsum("bhij,bhcj->bhci", attn, v)
        out   = out.reshape(B, C, H, W)
        return x + self.proj(out)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, use_attn=False, dropout=0.1):
        super().__init__()
        self.res   = ResBlock(in_ch, out_ch, t_dim, dropout)
        self.attn  = SelfAttention(out_ch) if use_attn else nn.Identity()
        self.down  = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.res(x, t_emb)
        x = self.attn(x)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, t_dim, use_attn=False, dropout=0.1):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.res  = ResBlock(in_ch + skip_ch, out_ch, t_dim, dropout)
        self.attn = SelfAttention(out_ch) if use_attn else nn.Identity()

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, t_emb)
        x = self.attn(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=128, ch_mult=(1, 2, 3, 4),
                 attn_res=(16,), dropout=0.1):
        super().__init__()
        t_dim = base_ch * 4
        self.time_emb = TimeEmbedding(base_ch)

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        chs = [base_ch * m for m in ch_mult] 
        res = 64
        self.downs = nn.ModuleList()
        in_c = base_ch
        self.skip_chs = []
        for i, out_c in enumerate(chs):
            use_attn = (res in attn_res)
            self.downs.append(DownBlock(in_c, out_c, t_dim, use_attn, dropout))
            self.skip_chs.append(out_c)
            in_c = out_c
            res //= 2

        self.mid_res1 = ResBlock(in_c, in_c, t_dim, dropout)
        self.mid_attn = SelfAttention(in_c)
        self.mid_res2 = ResBlock(in_c, in_c, t_dim, dropout)

        self.ups = nn.ModuleList()
        for i, skip_c in reversed(list(enumerate(chs))):
            out_c = chs[i - 1] if i > 0 else base_ch
            use_attn = ((res * 2) in attn_res)
            self.ups.append(UpBlock(in_c, skip_c, out_c, t_dim, use_attn, dropout))
            in_c = out_c
            res *= 2

        self.out_norm = nn.GroupNorm(32, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 3, padding=1)
        self.act = Swish()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)

        h = self.in_conv(x)
        skips = []
        for block in self.downs:
            h, skip = block(h, t_emb)
            skips.append(skip)

        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, t_emb)

        for block, skip in zip(self.ups, reversed(skips)):
            h = block(h, skip, t_emb)

        h = self.act(self.out_norm(h))
        return self.out_conv(h)



class ConsistencyModel(nn.Module):
    def __init__(self, unet: UNet, eps: float = 0.002, sigma_data: float = 0.5):
        super().__init__()
        self.unet       = unet
        self.eps        = eps
        self.sigma_data = sigma_data

    def c_skip(self, t: torch.Tensor) -> torch.Tensor:
        sd2 = self.sigma_data ** 2
        return sd2 / ((t - self.eps) ** 2 + sd2)

    def c_out(self, t: torch.Tensor) -> torch.Tensor:
        sd  = self.sigma_data
        return sd * (t - self.eps) / torch.sqrt(torch.tensor(sd ** 2, device=t.device) + t ** 2)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        cs = self.c_skip(t)[:, None, None, None]
        co = self.c_out(t)[:, None, None, None]
        return cs * x + co * self.unet(x, t)



def N_schedule(k: int, K: int, s0: int, s1: int) -> int:
    inner = (k / K) * ((s1 + 1) ** 2 - s0 ** 2) + s0 ** 2
    return int(math.floor(math.sqrt(inner) - 1) + 1)


def mu_schedule(k: int, K: int, s0: int, s1: int, mu0: float) -> float:
    Nk = N_schedule(k, K, s0, s1)
    return math.exp(s0 * math.log(mu0) / Nk)


def get_time_steps(N: int, eps: float, T: float, device) -> torch.Tensor:
    rho = 7.0
    steps = torch.arange(1, N + 1, device=device, dtype=torch.float32)
    t = (eps ** (1 / rho) + (steps - 1) / (N - 1) * (T ** (1 / rho) - eps ** (1 / rho))) ** rho
    return t   # shape (N,)


class LossFunction:
    def __init__(self, loss_type: str, device, lpips_net: str = "alex"):
        self.loss_type = loss_type
        if loss_type == "lpips":
            if not LPIPS_AVAILABLE:
                raise ImportError("pip install lpips  to use LPIPS loss.")
            self.lpips_fn = lpips.LPIPS(net=lpips_net).to(device)
            self.lpips_fn.eval()
            for p in self.lpips_fn.parameters():
                p.requires_grad_(False)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "l2":
            return F.mse_loss(pred, target)
        elif self.loss_type == "l1":
            return F.l1_loss(pred, target)
        elif self.loss_type == "lpips":
            return self.lpips_fn(pred, target).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


@torch.no_grad()
def update_ema(online: nn.Module, target: nn.Module, mu: float):
    """θ⁻ ← stopgrad(μ * θ⁻ + (1-μ) * θ)"""
    for p_on, p_tg in zip(online.parameters(), target.parameters()):
        p_tg.data.mul_(mu).add_(p_on.data, alpha=1.0 - mu)


@torch.no_grad()
def sample_images(model: ConsistencyModel, n: int, T: float, device) -> torch.Tensor:
    """x̂ = f_θ(x̂_T, T),  x̂_T ~ N(0, T²I)"""
    model.eval()
    x_T = torch.randn(n, 3, 64, 64, device=device) * T
    t   = torch.full((n,), T, device=device)
    out = model(x_T, t).clamp(-1, 1)
    model.train()
    return out


def get_celeba_loader(data_root: str, batch_size: int, num_workers: int, max_images: int = 60000):
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = datasets.CelebA(
        root=data_root,
        split="train",
        target_type="attr",
        transform=transform,
        download=False,
    )

    subset = torch.utils.data.Subset(dataset, indices=range(min(max_images, len(dataset))))
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    log.info(f"CelebA training set: {len(subset):,} images (capped from {len(dataset):,})")
    return loader


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    ckpt_dir    = Path(args.ckpt_dir);   ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir  = Path(args.sample_dir); sample_dir.mkdir(parents=True, exist_ok=True)

    loader = get_celeba_loader(args.data_root, args.batch_size, args.num_workers, args.max_images)

    unet_online = UNet(
        in_ch=3,
        base_ch=args.base_ch,
        ch_mult=tuple(args.ch_mult),
        attn_res=(16,),
        dropout=args.dropout,
    )
    unet_target = UNet(
        in_ch=3,
        base_ch=args.base_ch,
        ch_mult=tuple(args.ch_mult),
        attn_res=(16,),
        dropout=args.dropout,
    )

    online_model = ConsistencyModel(unet_online, eps=args.eps, sigma_data=args.sigma_data).to(device)
    target_model = ConsistencyModel(unet_target, eps=args.eps, sigma_data=args.sigma_data).to(device)

    target_model.load_state_dict(online_model.state_dict())
    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad_(False)

    num_params = sum(p.numel() for p in online_model.parameters()) / 1e6
    log.info(f"Model parameters: {num_params:.2f}M")

    try:
        from torch.optim import RAdam
        optimizer = RAdam(online_model.parameters(), lr=args.lr, weight_decay=0.0)
        log.info("Using RAdam optimizer")
    except ImportError:
        optimizer = torch.optim.Adam(online_model.parameters(), lr=args.lr)
        log.info("RAdam not available; falling back to Adam")

    loss_fn = LossFunction(args.loss_type, device, lpips_net=args.lpips_net)
    log.info(f"Loss: {args.loss_type}")

    iters_per_epoch = len(loader)
    K_total         = args.epochs * iters_per_epoch

    log.info(f"Total training steps K = {K_total:,}  ({args.epochs} epochs × {iters_per_epoch} iters)")
    log.info(f"Schedule: s0={args.s0}, s1={args.s1}, mu0={args.mu0}, N_target={args.N}")

    global_step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        online_model.load_state_dict(ckpt["online"])
        target_model.load_state_dict(ckpt["target"])

        old_loss = ckpt.get("args", {}).get("loss_type", None)

        if old_loss == args.loss_type:
            optimizer.load_state_dict(ckpt["optimizer"])
            log.info(f"Resumed model + optimizer from {args.resume} (same loss: {old_loss})")
        else:
            log.warning(
                f"Checkpoint loss was {old_loss}, current loss is {args.loss_type}. "
                f"Model weights were loaded, but optimizer was reset."
            )

        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        log.info(f"Resumed from {args.resume} (starting at epoch {start_epoch})")

    for epoch in range(start_epoch, args.epochs + 1):
        online_model.train()
        epoch_loss = 0.0

        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(device)
            B = x.shape[0]

            Nk = N_schedule(global_step, K_total, args.s0, args.s1)
            Nk = max(2, min(Nk, args.N))
            mu = mu_schedule(global_step, K_total, args.s0, args.s1, args.mu0)

            t_steps = get_time_steps(Nk, args.eps, args.T, device)
            n       = torch.randint(0, Nk - 1, (B,), device=device)

            t_n1 = t_steps[n + 1] 
            t_n  = t_steps[n]

            z         = torch.randn_like(x)
            x_tn1     = x + t_n1[:, None, None, None] * z
            x_tn      = x + t_n[:, None, None, None]  * z

            pred_online = online_model(x_tn1, t_n1)

            with torch.no_grad():
                pred_target = target_model(x_tn, t_n)

            loss = loss_fn(pred_online, pred_target)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(online_model.parameters(), args.grad_clip)
            optimizer.step()

            update_ema(online_model, target_model, mu)

            epoch_loss  += loss.item()
            global_step += 1

            if (batch_idx + 1) % args.log_every == 0:
                log.info(
                    f"Epoch [{epoch}/{args.epochs}] "
                    f"Step [{batch_idx+1}/{iters_per_epoch}] "
                    f"Loss: {loss.item():.5f}  N(k)={Nk}  μ={mu:.5f}"
                )

        avg_loss = epoch_loss / iters_per_epoch
        log.info(f"── Epoch {epoch} done | avg loss: {avg_loss:.5f}")

        samples = sample_images(online_model, args.n_samples, args.T, device)
        samples = (samples + 1) / 2   # [-1,1] → [0,1] for saving
        save_path = sample_dir / f"samples_epoch_{epoch:04d}.png"
        save_image(samples, save_path, nrow=int(math.sqrt(args.n_samples)))
        log.info(f"Saved samples → {save_path}")

        if epoch % 10 == 0:
            ckpt_path = ckpt_dir / f"ckpt_epoch_{epoch:04d}.pt"
            torch.save({
                "epoch":       epoch,
                "global_step": global_step,
                "online":      online_model.state_dict(),
                "target":      target_model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "args":        vars(args),
            }, ckpt_path)
            log.info(f"Saved checkpoint → {ckpt_path}")

    log.info("Training complete.")



def parse_args():
    parser = argparse.ArgumentParser(description="Consistency Training on CelebA 64×64")

    parser.add_argument("--data_root",   type=str,   default="./data",
                        help="Root directory for CelebA download / cache")
    parser.add_argument("--num_workers", type=int,   default=4)

    parser.add_argument("--max_images", type=int, default=60000,
                    help="Maximum number of CelebA training images to use")

    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-4,
                        help="Learning rate for RAdam")
    parser.add_argument("--grad_clip",   type=float, default=1.0,
                        help="Gradient clipping norm (0 = disabled)")
    parser.add_argument("--device",      type=str,   default="cuda")

    parser.add_argument("--T",           type=float, default=80.0,
                        help="Maximum noise level T")
    parser.add_argument("--eps",         type=float, default=0.002,
                        help="Minimum noise level ε (boundary condition)")
    parser.add_argument("--N",           type=int,   default=50,
                        help="Target number of discretisation steps N")
    parser.add_argument("--sigma_data",  type=float, default=0.5,
                        help="σ_data for skip-connection scaling")

    parser.add_argument("--s0",          type=int,   default=2,
                        help="Initial discretisation steps s₀")
    parser.add_argument("--s1",          type=int,   default=150,
                        help="Target discretisation steps s₁ (≥ N recommended)")
    parser.add_argument("--mu0",         type=float, default=0.95,
                        help="Initial EMA decay rate μ₀")

    parser.add_argument("--loss_type",   type=str,   default="l2",
                        choices=["l1", "l2", "lpips"],
                        help="Consistency loss metric d(·,·)")
    parser.add_argument("--lpips_net",   type=str,   default="alex",
                        choices=["alex", "vgg", "squeeze"],
                        help="Backbone for LPIPS (only used if --loss_type lpips)")

    parser.add_argument("--base_ch",     type=int,   default=128,
                        help="Base channel count for U-Net")
    parser.add_argument("--ch_mult",     type=int,   nargs="+", default=[1, 2, 3, 4],
                        help="Channel multipliers per resolution level")
    parser.add_argument("--dropout",     type=float, default=0.1)

    parser.add_argument("--ckpt_dir",    type=str,   default="./checkpoints")
    parser.add_argument("--sample_dir",  type=str,   default="./samples")
    parser.add_argument("--n_samples",   type=int,   default=16,
                        help="Number of images to generate per epoch")
    parser.add_argument("--log_every",   type=int,   default=100,
                        help="Log loss every N batches")
    parser.add_argument("--resume",      type=str,   default=None,
                        help="Path to checkpoint to resume from")

    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    log.info("=" * 60)
    log.info("Consistency Training — CelebA 64×64")
    log.info("=" * 60)
    for k, v in vars(args).items():
        log.info(f"  {k:20s}: {v}")
    log.info("=" * 60)

    train(args)