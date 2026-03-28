"""
python sample_multistep_celeba_ct.py \
    --ckpt /Data/rida.assalouh/celeba_ct_checkpoints_l1/ckpt_epoch_0240.pt \
    --outdir /Data/rida.assalouh/celeba_ct_samples_multistep \
    --num_images 16 \
    --batch_size 16 \
    --num_steps 5 \
    --save_grid 
"""

import os
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid


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
        args = t_log[:, None] * self.freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
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
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = Swish()

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
        self.norm = nn.GroupNorm(32, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.heads, C // self.heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        scale = (C // self.heads) ** -0.5
        attn = torch.einsum("bhci,bhcj->bhij", q, k) * scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhij,bhcj->bhci", attn, v)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, use_attn=False, dropout=0.1):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, t_dim, dropout)
        self.attn = SelfAttention(out_ch) if use_attn else nn.Identity()
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.res(x, t_emb)
        x = self.attn(x)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, t_dim, use_attn=False, dropout=0.1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.res = ResBlock(in_ch + skip_ch, out_ch, t_dim, dropout)
        self.attn = SelfAttention(out_ch) if use_attn else nn.Identity()

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, t_emb)
        x = self.attn(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=128, ch_mult=(1, 2, 3, 4), attn_res=(16,), dropout=0.1):
        super().__init__()
        t_dim = base_ch * 4
        self.time_emb = TimeEmbedding(base_ch)
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        chs = [base_ch * m for m in ch_mult]
        res = 64
        self.downs = nn.ModuleList()
        in_c = base_ch
        for out_c in chs:
            use_attn = (res in attn_res)
            self.downs.append(DownBlock(in_c, out_c, t_dim, use_attn, dropout))
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
        self.unet = unet
        self.eps = eps
        self.sigma_data = sigma_data

    def c_skip(self, t: torch.Tensor) -> torch.Tensor:
        sd2 = self.sigma_data ** 2
        return sd2 / ((t - self.eps) ** 2 + sd2)

    def c_out(self, t: torch.Tensor) -> torch.Tensor:
        sd = self.sigma_data
        return sd * (t - self.eps) / torch.sqrt(
            torch.tensor(sd ** 2, device=t.device, dtype=t.dtype) + t ** 2
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        cs = self.c_skip(t)[:, None, None, None]
        co = self.c_out(t)[:, None, None, None]
        return cs * x + co * self.unet(x, t)


def get_time_steps(num_steps: int, eps: float, T: float, rho: float, device: torch.device) -> torch.Tensor:
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1.")

    if num_steps == 1:
        return torch.tensor([T], device=device, dtype=torch.float32)

    idx = torch.arange(num_steps, device=device, dtype=torch.float32)
    t = (T ** (1 / rho) + idx / (num_steps - 1) * (eps ** (1 / rho) - T ** (1 / rho))) ** rho
    return t


def build_model_from_ckpt_args(ckpt_args: dict) -> ConsistencyModel:
    base_ch = ckpt_args.get("base_ch", 128)
    ch_mult = tuple(ckpt_args.get("ch_mult", [1, 2, 3, 4]))
    dropout = ckpt_args.get("dropout", 0.1)
    eps = ckpt_args.get("eps", 0.002)
    sigma_data = ckpt_args.get("sigma_data", 0.5)

    unet = UNet(
        in_ch=3,
        base_ch=base_ch,
        ch_mult=ch_mult,
        attn_res=(16,),
        dropout=dropout,
    )
    model = ConsistencyModel(unet, eps=eps, sigma_data=sigma_data)
    return model


def load_model(ckpt_path: str, device: torch.device, use_target: bool = False):
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {})

    model = build_model_from_ckpt_args(ckpt_args)
    state_key = "target" if use_target else "online"
    model.load_state_dict(ckpt[state_key], strict=True)
    model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model, ckpt_args


@torch.no_grad()
def multistep_generate(
    model: ConsistencyModel,
    num_images: int,
    batch_size: int,
    image_size: int,
    T: float,
    eps: float,
    rho: float,
    num_steps: int,
    device: torch.device,
    save_intermediate_steps: bool = False,
):
    sigmas = get_time_steps(num_steps, eps=eps, T=T, rho=rho, device=device)

    all_final = []
    all_intermediates = [] if save_intermediate_steps else None

    for start in range(0, num_images, batch_size):
        bsz = min(batch_size, num_images - start)

        x = torch.randn(bsz, 3, image_size, image_size, device=device) * sigmas[0]

        batch_steps = [] if save_intermediate_steps else None

        for i in range(len(sigmas)):
            t_cur = torch.full((bsz,), sigmas[i].item(), device=device)
            x_hat = model(x, t_cur).clamp(-1, 1)

            if save_intermediate_steps:
                batch_steps.append(x_hat.detach().cpu())

            if i < len(sigmas) - 1:
                t_next = sigmas[i + 1]
                noise_scale = torch.sqrt(torch.clamp(t_next ** 2 - eps ** 2, min=0.0))
                z = torch.randn_like(x_hat)
                x = x_hat + noise_scale * z
            else:
                x = x_hat

        all_final.append(x.cpu())

        if save_intermediate_steps:
            all_intermediates.append(batch_steps)

    final_images = torch.cat(all_final, dim=0)

    return final_images, all_intermediates, sigmas.cpu()


def save_individual_images(images: torch.Tensor, outdir: Path, prefix: str = "sample"):
    outdir.mkdir(parents=True, exist_ok=True)
    images_01 = (images + 1) / 2
    images_01 = images_01.clamp(0, 1)

    for i in range(images_01.shape[0]):
        save_image(images_01[i], outdir / f"{prefix}_{i:05d}.png")


def save_grid_image(images: torch.Tensor, path: Path, nrow: int = None):
    images_01 = (images + 1) / 2
    images_01 = images_01.clamp(0, 1)

    if nrow is None:
        nrow = int(math.sqrt(images_01.shape[0]))
        nrow = max(1, nrow)

    grid = make_grid(images_01, nrow=nrow, padding=2)
    save_image(grid, path)


def save_intermediate_grids(intermediates, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    num_steps = len(intermediates[0])
    for step_idx in range(num_steps):
        step_imgs = torch.cat([batch_steps[step_idx] for batch_steps in intermediates], dim=0)
        save_grid_image(step_imgs, outdir / f"step_{step_idx+1:02d}.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Multistep generation for CelebA consistency model")

    parser.add_argument(
        "--ckpt",
        type=str,
        default="/Data/rida.assalouh/celeba_ct_checkpoints_lpips/ckpt_epoch_0070.pt",
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./celeba_ct_multistep_samples",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=64,
        help="Total number of images to generate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Sampling batch size",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=5,
        help="Number of multistep sampling steps",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=7.0,
        help="EDM rho schedule parameter",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--use_target",
        action="store_true",
        help="Use EMA target network instead of online network",
    )
    parser.add_argument(
        "--save_grid",
        action="store_true",
        help="Save a single grid with all final generated images",
    )
    parser.add_argument(
        "--save_individual",
        action="store_true",
        help="Save every final generated image separately",
    )
    parser.add_argument(
        "--save_intermediate_grids",
        action="store_true",
        help="Save one grid per intermediate denoising step",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Loading checkpoint...")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Device: {device}")
    print("=" * 80)

    model, ckpt_args = load_model(args.ckpt, device=device, use_target=args.use_target)

    T = float(ckpt_args.get("T", 80.0))
    eps = float(ckpt_args.get("eps", 0.002))
    image_size = 64

    print("Sampling configuration:")
    print(f"  use_target      : {args.use_target}")
    print(f"  num_images      : {args.num_images}")
    print(f"  batch_size      : {args.batch_size}")
    print(f"  num_steps       : {args.num_steps}")
    print(f"  T               : {T}")
    print(f"  eps             : {eps}")
    print(f"  rho             : {args.rho}")
    print(f"  outdir          : {outdir}")

    final_images, intermediates, sigmas = multistep_generate(
        model=model,
        num_images=args.num_images,
        batch_size=args.batch_size,
        image_size=image_size,
        T=T,
        eps=eps,
        rho=args.rho,
        num_steps=args.num_steps,
        device=device,
        save_intermediate_steps=args.save_intermediate_grids,
    )

    print("Sigma schedule used:")
    print(sigmas.tolist())

    if args.save_grid or (not args.save_grid and not args.save_individual):
        save_grid_image(final_images, outdir / "generated_grid.png")
        print(f"Saved grid -> {outdir / 'generated_grid.png'}")

    if args.save_individual:
        save_individual_images(final_images, outdir / "individual_images")
        print(f"Saved individual images -> {outdir / 'individual_images'}")

    if args.save_intermediate_grids and intermediates is not None:
        save_intermediate_grids(intermediates, outdir / "intermediate_grids")
        print(f"Saved intermediate grids -> {outdir / 'intermediate_grids'}")

    print("Done.")


if __name__ == "__main__":
    main()