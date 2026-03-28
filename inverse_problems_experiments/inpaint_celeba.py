import os
import math
import argparse
import logging
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

import numpy as np

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
    def __init__(self, in_ch, out_ch, t_dim, dropout=0.1):
        super().__init__()
        self.norm1  = nn.GroupNorm(32, in_ch)
        self.conv1  = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2  = nn.GroupNorm(32, out_ch)
        self.dropout= nn.Dropout(dropout)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip   = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act    = Swish()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.t_proj(self.act(t_emb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.heads = heads
        self.norm  = nn.GroupNorm(32, ch)
        self.qkv   = nn.Conv2d(ch, ch * 3, 1)
        self.proj  = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h   = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.heads, C // self.heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        scale = (C // self.heads) ** -0.5
        attn  = torch.einsum("bhci,bhcj->bhij", q, k) * scale
        attn  = attn.softmax(dim=-1)
        out   = torch.einsum("bhij,bhcj->bhci", attn, v)
        out   = out.reshape(B, C, H, W)
        return x + self.proj(out)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, use_attn=False, dropout=0.1):
        super().__init__()
        self.res  = ResBlock(in_ch, out_ch, t_dim, dropout)
        self.attn = SelfAttention(out_ch) if use_attn else nn.Identity()
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb):
        x    = self.res(x, t_emb)
        x    = self.attn(x)
        skip = x
        x    = self.down(x)
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
        t_dim    = base_ch * 4
        self.time_emb = TimeEmbedding(base_ch)
        self.in_conv  = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        chs = [base_ch * m for m in ch_mult]
        res = 64
        self.downs    = nn.ModuleList()
        self.skip_chs = []
        in_c = base_ch
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
            out_c    = chs[i - 1] if i > 0 else base_ch
            use_attn = ((res * 2) in attn_res)
            self.ups.append(UpBlock(in_c, skip_c, out_c, t_dim, use_attn, dropout))
            in_c = out_c
            res *= 2

        self.out_norm = nn.GroupNorm(32, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 3, padding=1)
        self.act      = Swish()

    def forward(self, x, t):
        t_emb = self.time_emb(t)
        h     = self.in_conv(x)
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

    def c_skip(self, t):
        sd2 = self.sigma_data ** 2
        return sd2 / ((t - self.eps) ** 2 + sd2)

    def c_out(self, t):
        sd = self.sigma_data
        return sd * (t - self.eps) / torch.sqrt(torch.tensor(sd ** 2, device=t.device) + t ** 2)

    def forward(self, x, t):
        cs = self.c_skip(t)[:, None, None, None]
        co = self.c_out(t)[:, None, None, None]
        return cs * x + co * self.unet(x, t)


def make_mask_center(B, H, W, mask_ratio=0.5, device="cpu"):
    mask = torch.zeros(B, 1, H, W, device=device)
    h_size = int(H * mask_ratio)
    w_size = int(W * mask_ratio)
    y0 = (H - h_size) // 2
    x0 = (W - w_size) // 2
    mask[:, :, y0:y0 + h_size, x0:x0 + w_size] = 1.0
    return mask


def make_mask_random(B, H, W, min_ratio=0.2, max_ratio=0.6, device="cpu"):
    mask = torch.zeros(B, 1, H, W, device=device)
    for b in range(B):
        h_size = random.randint(int(H * min_ratio), int(H * max_ratio))
        w_size = random.randint(int(W * min_ratio), int(W * max_ratio))
        y0 = random.randint(0, H - h_size)
        x0 = random.randint(0, W - w_size)
        mask[b, :, y0:y0 + h_size, x0:x0 + w_size] = 1.0
    return mask


def make_mask_half(B, H, W, side="left", device="cpu"):
    mask = torch.zeros(B, 1, H, W, device=device)
    mid  = W // 2
    if side == "left":
        mask[:, :, :, :mid] = 1.0
    else:
        mask[:, :, :, mid:] = 1.0
    return mask


def get_mask(mask_type, B, H, W, half_side="left", device="cpu"):
    if mask_type == "center":
        return make_mask_center(B, H, W, device=device)
    elif mask_type == "random":
        return make_mask_random(B, H, W, device=device)
    elif mask_type == "half":
        return make_mask_half(B, H, W, side=half_side, device=device)
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")


def get_inpaint_timesteps(N, eps, T, device):
    rho  = 7.0
    i    = torch.arange(1, N + 1, device=device, dtype=torch.float64)
    t    = (T ** (1 / rho) + (i - 1) / (N - 1) * (eps ** (1 / rho) - T ** (1 / rho))) ** rho
    return t.float()


@torch.no_grad()
def inpaint(model, y_clean, mask, N, eps, T, device):
    model.eval()
    B = y_clean.shape[0]
    t_steps = get_inpaint_timesteps(N, eps, T, device)
    y = y_clean * (1.0 - mask)
    t1   = t_steps[0]
    noise = torch.randn_like(y)
    x     = y + t1 * noise
    t_vec = torch.full((B,), t1.item(), device=device)
    x     = model(x, t_vec).clamp(-1, 1)
    x = y * (1.0 - mask) + x * mask
    for n in range(1, N):
        tn = t_steps[n]
        std   = torch.sqrt(torch.clamp(tn ** 2 - eps ** 2, min=0.0))
        noise = torch.randn_like(x)
        x_noisy = x + std * noise
        t_vec = torch.full((B,), tn.item(), device=device)
        x     = model(x_noisy, t_vec).clamp(-1, 1)
        x = y * (1.0 - mask) + x * mask
    return x


def get_celeba_loader(data_root, batch_size, num_workers, n_images):
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(64),
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
    n = min(n_images, len(dataset))
    subset = Subset(dataset, indices=range(n))
    log.info(f"Loaded {n} CelebA images for inpainting")
    return DataLoader(subset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True, drop_last=False)


def save_comparison_grid(originals, masked_inp, inpainted, save_path, triplets_per_row=2):
    orig = (originals + 1) / 2
    msk  = (masked_inp + 1) / 2
    inp  = (inpainted + 1) / 2
    B = orig.shape[0]
    tiles = []
    for i in range(B):
        tiles.append(orig[i])
        tiles.append(msk[i])
        tiles.append(inp[i])
    tiles = torch.stack(tiles, dim=0)
    images_per_row = 3 * triplets_per_row
    grid = make_grid(
        tiles,
        nrow=images_per_row,
        padding=2,
        pad_value=0.5,
    )
    save_image(grid, save_path)
    log.info(f"Saved grid → {save_path}")


def load_model(ckpt_path, device, base_ch=128,
               ch_mult=(1, 2, 3, 4), eps=0.002,
               sigma_data=0.5):
    unet  = UNet(in_ch=3, base_ch=base_ch, ch_mult=ch_mult,
                 attn_res=(16,), dropout=0.0)
    model = ConsistencyModel(unet, eps=eps, sigma_data=sigma_data).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    if "online" in ckpt:
        state = ckpt["online"]
        log.info(f"Loaded online model from checkpoint (epoch {ckpt.get('epoch', '?')})")
    else:
        state = ckpt
        log.info("Loaded raw state dict from checkpoint")
    model.load_state_dict(state)
    model.eval()
    return model


def run(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(
        args.ckpt, device,
        base_ch=args.base_ch,
        ch_mult=tuple(args.ch_mult),
        eps=args.eps,
        sigma_data=args.sigma_data,
    )
    log.info(f"Model loaded from {args.ckpt}")
    loader = get_celeba_loader(args.data_root, args.batch_size,
                               args.num_workers, args.n_images)
    all_orig, all_masked, all_inpainted = [], [], []
    total = 0
    for batch_idx, (images, _) in enumerate(loader):
        images = images.to(device)
        B, C, H, W = images.shape
        mask = get_mask(args.mask_type, B, H, W,
                        half_side=args.half_side, device=device)
        y_masked = images * (1.0 - mask)
        log.info(
            f"Batch {batch_idx + 1}/{len(loader)} | "
            f"images {total + 1}–{total + B} | mask={args.mask_type}"
        )
        result = inpaint(
            model=model,
            y_clean=images,
            mask=mask,
            N=args.N_steps,
            eps=args.eps,
            T=args.T,
            device=device,
        )
        all_orig.append(images.cpu())
        all_masked.append(y_masked.cpu())
        all_inpainted.append(result.cpu())
        total += B
        save_comparison_grid(
            images.cpu(),
            y_masked.cpu(),
            result.cpu(),
            save_path=out_dir / f"inpaint_batch_{batch_idx + 1:04d}.png",
            triplets_per_row=2,
        )
    all_orig      = torch.cat(all_orig,      dim=0)
    all_masked    = torch.cat(all_masked,    dim=0)
    all_inpainted = torch.cat(all_inpainted, dim=0)
    all_orig      = all_orig[4:]
    all_masked    = all_masked[4:]
    all_inpainted = all_inpainted[4:]
    save_comparison_grid(
        all_orig,
        all_masked,
        all_inpainted,
        save_path=out_dir / "inpaint_all.png",
        triplets_per_row=2,
    )
    log.info(f"Done. All results in {out_dir}/")


def parse_args():
    p = argparse.ArgumentParser(
        description="Zero-Shot Inpainting with Consistency Models (Algorithm 4)"
    )
    p.add_argument("--ckpt",       type=str, required=True)
    p.add_argument("--data_root",  type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./inpainting_results")
    p.add_argument("--n_images",   type=int, default=32)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers",type=int, default=4)
    p.add_argument("--mask_type",  type=str, default="center",
                   choices=["center", "random", "half"])
    p.add_argument("--half_side",  type=str, default="left",
                   choices=["left", "right"])
    p.add_argument("--N_steps",    type=int,   default=40)
    p.add_argument("--T",          type=float, default=80.0)
    p.add_argument("--eps",        type=float, default=0.002)
    p.add_argument("--base_ch",    type=int,   default=128)
    p.add_argument("--ch_mult",    type=int,   nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--sigma_data", type=float, default=0.5)
    p.add_argument("--device",     type=str,   default="cuda")
    p.add_argument("--grid_nrow",  type=int,   default=2)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    log.info("=" * 60)
    log.info("Zero-Shot Inpainting — CelebA 64×64")
    log.info("=" * 60)
    for k, v in vars(args).items():
        log.info(f"  {k:20s}: {v}")
    log.info("=" * 60)
    run(args)