import math
import random
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

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
        half  = dim // 2
        freqs = torch.exp(-math.log(10_000) * torch.arange(half) / half)
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), Swish(), nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t):
        t_log = torch.log(t / 4) * 0.25
        args  = t_log[:, None] * self.freqs[None]
        emb   = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.mlp(emb)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, dropout=0.1):
        super().__init__()
        self.norm1   = nn.GroupNorm(32, in_ch)
        self.conv1   = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj  = nn.Linear(t_dim, out_ch)
        self.norm2   = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2   = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip    = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act     = Swish()

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
        attn = torch.einsum("bhci,bhcj->bhij", q, k) * (C // self.heads) ** -0.5
        attn = attn.softmax(dim=-1)
        out  = torch.einsum("bhij,bhcj->bhci", attn, v).reshape(B, C, H, W)
        return x + self.proj(out)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, use_attn=False, dropout=0.1):
        super().__init__()
        self.res  = ResBlock(in_ch, out_ch, t_dim, dropout)
        self.attn = SelfAttention(out_ch) if use_attn else nn.Identity()
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.res(x, t_emb)
        x = self.attn(x)
        return self.down(x), x


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, t_dim, use_attn=False, dropout=0.1):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.res  = ResBlock(in_ch + skip_ch, out_ch, t_dim, dropout)
        self.attn = SelfAttention(out_ch) if use_attn else nn.Identity()

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.attn(self.res(x, t_emb))


class UNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=128, ch_mult=(1, 2, 3, 4),
                 attn_res=(16,), dropout=0.1):
        super().__init__()
        t_dim         = base_ch * 4
        self.time_emb = TimeEmbedding(base_ch)
        self.in_conv  = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        chs           = [base_ch * m for m in ch_mult]
        res, in_c     = 64, base_ch

        self.downs, self.skip_chs = nn.ModuleList(), []
        for out_c in chs:
            self.downs.append(DownBlock(in_c, out_c, t_dim, res in attn_res, dropout))
            self.skip_chs.append(out_c)
            in_c, res = out_c, res // 2

        self.mid_res1 = ResBlock(in_c, in_c, t_dim, dropout)
        self.mid_attn = SelfAttention(in_c)
        self.mid_res2 = ResBlock(in_c, in_c, t_dim, dropout)

        self.ups = nn.ModuleList()
        for i, skip_c in reversed(list(enumerate(chs))):
            out_c = chs[i - 1] if i > 0 else base_ch
            self.ups.append(UpBlock(in_c, skip_c, out_c, t_dim, (res * 2) in attn_res, dropout))
            in_c, res = out_c, res * 2

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
        h = self.mid_res2(self.mid_attn(self.mid_res1(h, t_emb)), t_emb)
        for block, skip in zip(self.ups, reversed(skips)):
            h = block(h, skip, t_emb)
        return self.out_conv(self.act(self.out_norm(h)))


class ConsistencyModel(nn.Module):
    def __init__(self, unet, eps=0.002, sigma_data=0.5):
        super().__init__()
        self.unet, self.eps, self.sigma_data = unet, eps, sigma_data

    def c_skip(self, t):
        sd2 = self.sigma_data ** 2
        return sd2 / ((t - self.eps) ** 2 + sd2)

    def c_out(self, t):
        sd = self.sigma_data
        return sd * (t - self.eps) / torch.sqrt(torch.tensor(sd**2, device=t.device) + t**2)

    def forward(self, x, t):
        cs = self.c_skip(t)[:, None, None, None]
        co = self.c_out(t)[:, None, None, None]
        return cs * x + co * self.unet(x, t)

Q = torch.tensor([
    [0.4471, -0.8204,  0.3563],
    [0.8780,  0.4785,  0.0000],
    [0.1705, -0.3129, -0.9343],
], dtype=torch.float32)

OMEGA = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32)


def apply_A(x: torch.Tensor, Q_mat: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, 3)
    y_flat = x_flat @ Q_mat                            
    return y_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2) 


def apply_A_inv(y: torch.Tensor, Q_mat: torch.Tensor) -> torch.Tensor:
    """
    Inverse transform  A^{-1}: y ↦ x
    x[i,j,k] = Σ_l  y[i,j,l] * Q[k,l]  = Σ_l  y[i,j,l] * Q^T[l,k]

    Q orthogonal → Q^{-1} = Q^T
    y : (B, 3, H, W)
    returns x : (B, 3, H, W)
    """
    B, C, H, W = y.shape
    y_flat = y.permute(0, 2, 3, 1).reshape(B, H * W, 3)
    x_flat = y_flat @ Q_mat.T                     
    return x_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2) 


def rgb_to_grayscale_3ch(x: torch.Tensor) -> torch.Tensor:
    """
    Convert (B, 3, H, W) RGB image in [-1,1] to 3-channel grayscale in [-1,1].
    Each channel = 0.2989R + 0.5870G + 0.1140B  (all 3 channels identical).
    """
    x01   = (x + 1) / 2                          
    gray  = 0.2989 * x01[:, 0] + 0.5870 * x01[:, 1] + 0.1140 * x01[:, 2]
    gray  = gray.unsqueeze(1)     
    gray3 = gray.expand(-1, 3, -1, -1)   
    return gray3 * 2 - 1   



def get_timesteps(N: int, eps: float, T: float, device) -> torch.Tensor:
    rho = 7.0
    i   = torch.arange(1, N + 1, device=device, dtype=torch.float64)
    t   = (T ** (1 / rho) + (i - 1) / (N - 1) * (eps ** (1 / rho) - T ** (1 / rho))) ** rho
    return t.float()



@torch.no_grad()
def colorize(
    model:   ConsistencyModel,
    x_rgb:   torch.Tensor,
    N:       int,
    eps:     float,
    T:       float,
    device,
) -> torch.Tensor:

    model.eval()
    B = x_rgb.shape[0]

    Q_dev   = Q.to(device)
    omega   = OMEGA.to(device)[None, :, None, None] 

    t_steps = get_timesteps(N, eps, T, device)  

    y_rgb   = rgb_to_grayscale_3ch(x_rgb)

    Ay      = apply_A(y_rgb, Q_dev)
    Ay_kept = Ay * (1.0 - omega)
    y       = apply_A_inv(Ay_kept, Q_dev)

    t1  = t_steps[0]
    x   = y + t1 * torch.randn_like(y)

    t_vec = torch.full((B,), t1.item(), device=device)
    x     = model(x, t_vec).clamp(-1, 1)

    Ax = apply_A(x, Q_dev)
    x  = apply_A_inv(Ay * (1.0 - omega) + Ax * omega, Q_dev).clamp(-1, 1)

    for n in range(1, N):
        tn  = t_steps[n]
        std = torch.sqrt(torch.clamp(tn**2 - eps**2, min=0.0))

        x_noisy = x + std * torch.randn_like(x)

        t_vec = torch.full((B,), tn.item(), device=device)
        x     = model(x_noisy, t_vec).clamp(-1, 1)

        Ax = apply_A(x, Q_dev)
        x  = apply_A_inv(Ay * (1.0 - omega) + Ax * omega, Q_dev).clamp(-1, 1)

    return x   



def get_celeba_loader(data_root: str, batch_size: int,
                      num_workers: int, n_images: int) -> DataLoader:
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = datasets.CelebA(
        root=data_root, split="train",
        target_type="attr", transform=transform, download=False,
    )
    n      = min(n_images, len(dataset))
    subset = Subset(dataset, indices=range(n))
    log.info(f"Loaded {n} CelebA images for colorization")
    return DataLoader(subset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True, drop_last=False)



def save_comparison_grid(
    originals:  torch.Tensor,   
    grayscales: torch.Tensor,   
    colorized:  torch.Tensor,   
    save_path:  Path,
    nrow:       int = 8,
):
    """
    Each triple: original (color) | grayscale | colorized
    """
    orig = (originals  + 1) / 2
    gray = (grayscales + 1) / 2
    col  = (colorized  + 1) / 2

    triples = []
    for i in range(orig.shape[0]):
        triples.extend([orig[i], gray[i], col[i]])

    grid = make_grid(torch.stack(triples), nrow=nrow * 3, padding=2, pad_value=0.5)
    save_image(grid, save_path)
    log.info(f"Saved grid → {save_path}")



def load_model(ckpt_path: str, device, base_ch=128,
               ch_mult=(1, 2, 3, 4), eps=0.002, sigma_data=0.5) -> ConsistencyModel:
    unet  = UNet(in_ch=3, base_ch=base_ch, ch_mult=ch_mult,
                 attn_res=(16,), dropout=0.0)
    model = ConsistencyModel(unet, eps=eps, sigma_data=sigma_data).to(device)

    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt["online"] if "online" in ckpt else ckpt
    epoch = ckpt.get("epoch", "?") if "online" in ckpt else "?"
    model.load_state_dict(state)
    model.eval()
    log.info(f"Loaded checkpoint (epoch {epoch}) from {ckpt_path}")
    return model



def run(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model  = load_model(args.ckpt, device,
                        base_ch=args.base_ch, ch_mult=tuple(args.ch_mult),
                        eps=args.eps, sigma_data=args.sigma_data)
    loader = get_celeba_loader(args.data_root, args.batch_size,
                               args.num_workers, args.n_images)

    all_orig, all_gray, all_col = [], [], []
    total = 0

    for batch_idx, (images, _) in enumerate(loader):
        images = images.to(device)
        B      = images.shape[0]
        grays  = rgb_to_grayscale_3ch(images)

        log.info(
            f"Batch {batch_idx + 1}/{len(loader)} | "
            f"images {total + 1}–{total + B}"
        )

        result = colorize(
            model=model, x_rgb=images,
            N=args.N_steps, eps=args.eps, T=args.T, device=device,
        )

        all_orig.append(images.cpu())
        all_gray.append(grays.cpu())
        all_col.append(result.cpu())
        total += B

        save_comparison_grid(
            images, grays, result,
            save_path=out_dir / f"colorize_batch_{batch_idx + 1:04d}.png",
            nrow=min(B, args.grid_nrow),
        )

    save_comparison_grid(
        torch.cat(all_orig),
        torch.cat(all_gray),
        torch.cat(all_col),
        save_path=out_dir / "colorize_all.png",
        nrow=min(total, args.grid_nrow),
    )
    log.info(f"Done. All results saved to {out_dir}/")



def parse_args():
    p = argparse.ArgumentParser(
        description="Zero-Shot Colorization with Consistency Models (Algorithm 4)"
    )

    p.add_argument("--ckpt",        type=str, required=True,
                   help="Path to consistency model checkpoint (.pt)")
    p.add_argument("--data_root",   type=str, required=True,
                   help="Root directory of CelebA dataset")
    p.add_argument("--output_dir",  type=str, default="./colorization_results",
                   help="Where to save result grids")

    p.add_argument("--n_images",    type=int, default=32,
                   help="Total number of CelebA images to colorize")
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--N_steps",     type=int,   default=40,
                   help="Number of denoising steps N in Algorithm 4")
    p.add_argument("--T",           type=float, default=80.0,
                   help="Maximum noise level T (must match training)")
    p.add_argument("--eps",         type=float, default=0.002,
                   help="Minimum noise level ε (must match training)")

    p.add_argument("--base_ch",     type=int,   default=128)
    p.add_argument("--ch_mult",     type=int,   nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--sigma_data",  type=float, default=0.5)

    p.add_argument("--device",      type=str,   default="cuda")
    p.add_argument("--grid_nrow",   type=int,   default=2,
                   help="Number of image triples per row in the output grid")
    p.add_argument("--seed",        type=int,   default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    log.info("=" * 60)
    log.info("Zero-Shot Colorization — CelebA 64×64")
    log.info("=" * 60)
    for k, v in vars(args).items():
        log.info(f"  {k:20s}: {v}")
    log.info("=" * 60)

    run(args)