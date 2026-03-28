import copy
import math
import os
import json
import random
from dataclasses import dataclass, asdict
from typing import Tuple

import numpy as np
from scipy.linalg import sqrtm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm



class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        freqs = torch.exp(-scale * torch.arange(half_dim, device=t.device))
        emb = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb



class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, 3 * channels, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x).view(b, c, h * w)
        q, k, v = self.qkv(x).chunk(3, dim=1)
        head_dim = c // self.num_heads
        q = q.view(b, self.num_heads, head_dim, h * w)
        k = k.view(b, self.num_heads, head_dim, h * w)
        v = v.view(b, self.num_heads, head_dim, h * w)
        attn = torch.einsum("bhdl,bhdm->bhlm", q * (head_dim ** -0.5), k).softmax(dim=-1)
        out = torch.einsum("bhlm,bhdm->bhdl", attn, v).reshape(b, c, h * w)
        return x_in + self.proj(out).view(b, c, h, w)



class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, 4 * time_emb_dim),
            nn.SiLU(),
            nn.Linear(4 * time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skip_channels = []

        curr_ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            stage = nn.ModuleList()
            for _ in range(num_res_blocks):
                stage.append(ResidualBlock(curr_ch, out_ch, time_emb_dim, dropout))
                curr_ch = out_ch
                if mult >= 4:
                    stage.append(AttentionBlock(curr_ch))
                self.skip_channels.append(curr_ch)
            self.down_blocks.append(stage)
            self.downsamples.append(
                Downsample(curr_ch) if i < len(channel_mults) - 1 else nn.Identity()
            )

        self.mid_block1 = ResidualBlock(curr_ch, curr_ch, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(curr_ch)
        self.mid_block2 = ResidualBlock(curr_ch, curr_ch, time_emb_dim, dropout)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        reversed_mults = list(reversed(channel_mults))
        reversed_skips = list(reversed(self.skip_channels))

        for i, mult in enumerate(reversed_mults):
            out_ch = base_channels * mult
            stage = nn.ModuleList()
            for _ in range(num_res_blocks):
                skip_ch = reversed_skips.pop(0)
                stage.append(ResidualBlock(curr_ch + skip_ch, out_ch, time_emb_dim, dropout))
                curr_ch = out_ch
                if mult >= 4:
                    stage.append(AttentionBlock(curr_ch))
            self.up_blocks.append(stage)
            self.upsamples.append(
                Upsample(curr_ch) if i < len(reversed_mults) - 1 else nn.Identity()
            )

        self.final_norm = nn.GroupNorm(8, curr_ch)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(curr_ch, in_channels, 3, padding=1)

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        t_emb = self.time_mlp(torch.log(sigma.clamp(min=1e-12)))
        x = self.init_conv(x)

        skips = []
        for stage, down in zip(self.down_blocks, self.downsamples):
            for block in stage:
                if isinstance(block, ResidualBlock):
                    x = block(x, t_emb)
                    skips.append(x)
                else:
                    x = block(x)
            x = down(x)

        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        for stage, up in zip(self.up_blocks, self.upsamples):
            for block in stage:
                if isinstance(block, ResidualBlock):
                    skip = skips.pop()
                    x = torch.cat([x, skip], dim=1)
                    x = block(x, t_emb)
                else:
                    x = block(x)
            x = up(x)

        return self.final_conv(self.final_act(self.final_norm(x)))
    

__all__ = ["UNet"]