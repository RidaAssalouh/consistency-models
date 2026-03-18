# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common layers for defining score networks. Converted from JAX/Flax to PyTorch."""

import math
import functools
from typing import Any, Sequence, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def get_act(config):
    """Get activation function from config."""
    name = config.model.nonlinearity.lower()
    if name == "elu":
        return nn.ELU()
    elif name == "relu":
        return nn.ReLU()
    elif name == "lrelu":
        return nn.LeakyReLU(negative_slope=0.2)
    elif name == "swish":
        return nn.SiLU()
    else:
        raise NotImplementedError(f"Activation '{name}' not implemented.")


# ---------------------------------------------------------------------------
# Convolution helpers
# ---------------------------------------------------------------------------
# JAX uses NHWC layout; PyTorch uses NCHW.
# All modules here work in NCHW; callers must transpose if needed.

def _variance_scaling_init(tensor: torch.Tensor, scale: float, mode: str):
    """Mimics JAX's variance_scaling initializer."""
    scale = max(scale, 1e-10)
    fan_in = tensor[0].numel()   # for Conv2d weight shape (out, in, kH, kW)
    fan_out = tensor.shape[0] * tensor[0][0].numel()
    if mode == "fan_in":
        n = fan_in
    elif mode == "fan_avg":
        n = (fan_in + fan_out) / 2.0
    else:
        raise ValueError(f"Unknown mode: {mode}")
    limit = math.sqrt(3 * scale / n)
    nn.init.uniform_(tensor, -limit, limit)

    
def default_init(scale: float = 1.0):
    """Returns an in-place weight initializer using fan_avg uniform scaling.

    Matches the original JAX default_init(scale). Returns a callable
    ``fn(tensor) -> None`` suitable for passing as a kernel_init argument.
    """
    def init_fn(tensor: torch.Tensor) -> None:
        _variance_scaling_init(tensor, scale=scale, mode="fan_avg")
    return init_fn

def ncsn_conv(in_planes, out_planes, kernel_size=3, stride=1, bias=True,
              dilation=1, init_scale=1.0):
    """Conv with NCSNv1/v2 (PyTorch-style variance_scaling fan_in) init."""
    init_scale = max(init_scale, 1e-10)
    padding = dilation * (kernel_size - 1) // 2
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                     stride=stride, padding=padding, dilation=dilation,
                     bias=bias)
    _variance_scaling_init(conv.weight, scale=init_scale / 3, mode="fan_in")
    if bias:
        nn.init.zeros_(conv.bias)
    return conv


def ncsn_conv1x1(in_planes, out_planes, stride=1, bias=True, dilation=1,
                 init_scale=1.0):
    return ncsn_conv(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=bias, dilation=dilation, init_scale=init_scale)


def ncsn_conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1,
                 init_scale=1.0):
    return ncsn_conv(in_planes, out_planes, kernel_size=3, stride=stride,
                     bias=bias, dilation=dilation, init_scale=init_scale)


def ddpm_conv(in_planes, out_planes, kernel_size=3, stride=1, bias=True,
              dilation=1, init_scale=1.0):
    """Conv with DDPM (variance_scaling fan_avg) init."""
    init_scale = max(init_scale, 1e-10)
    padding = dilation * (kernel_size - 1) // 2
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                     stride=stride, padding=padding, dilation=dilation,
                     bias=bias)
    _variance_scaling_init(conv.weight, scale=init_scale, mode="fan_avg")
    if bias:
        nn.init.zeros_(conv.bias)
    return conv


def ddpm_conv1x1(in_planes, out_planes, stride=1, bias=True, dilation=1,
                 init_scale=1.0):
    return ddpm_conv(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=bias, dilation=dilation, init_scale=init_scale)


def ddpm_conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1,
                 init_scale=1.0):
    return ddpm_conv(in_planes, out_planes, kernel_size=3, stride=stride,
                     bias=bias, dilation=dilation, init_scale=init_scale)


# ---------------------------------------------------------------------------
# NCSNv1 / NCSNv2 blocks
# ---------------------------------------------------------------------------

class CRPBlock(nn.Module):
    """CRPBlock for RefineNet. Used in NCSNv2."""

    def __init__(self, features: int, n_stages: int, act=nn.ReLU()):
        super().__init__()
        self.act = act
        self.n_stages = n_stages
        self.convs = nn.ModuleList(
            [ncsn_conv3x3(features, features, stride=1, bias=False)
             for _ in range(n_stages)]
        )

    def forward(self, x):
        x = self.act(x)
        path = x
        for conv in self.convs:
            path = F.max_pool2d(path, kernel_size=5, stride=1, padding=2)
            path = conv(path)
            x = path + x
        return x


class CondCRPBlock(nn.Module):
    """Noise-conditional CRPBlock for RefineNet. Used in NCSNv1."""

    def __init__(self, features: int, n_stages: int, normalizer, act=nn.ReLU()):
        super().__init__()
        self.act = act
        self.n_stages = n_stages
        self.norms = nn.ModuleList([normalizer() for _ in range(n_stages)])
        self.convs = nn.ModuleList(
            [ncsn_conv3x3(features, features, stride=1, bias=False)
             for _ in range(n_stages)]
        )

    def forward(self, x, y):
        x = self.act(x)
        path = x
        for norm, conv in zip(self.norms, self.convs):
            path = norm(path, y)
            path = F.avg_pool2d(path, kernel_size=5, stride=1, padding=2)
            path = conv(path)
            x = path + x
        return x


class RCUBlock(nn.Module):
    """RCUBlock for RefineNet. Used in NCSNv2."""

    def __init__(self, features: int, n_blocks: int, n_stages: int,
                 act=nn.ReLU()):
        super().__init__()
        self.act = act
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        # Each block has n_stages conv layers; in_ch == out_ch == features
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                ncsn_conv3x3(features, features, stride=1, bias=False)
                for _ in range(n_stages)
            ])
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            residual = x
            for conv in block:
                x = self.act(x)
                x = conv(x)
            x = x + residual
        return x


class CondRCUBlock(nn.Module):
    """Noise-conditional RCUBlock for RefineNet. Used in NCSNv1."""

    def __init__(self, features: int, n_blocks: int, n_stages: int,
                 normalizer, act=nn.ReLU()):
        super().__init__()
        self.act = act
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.norms = nn.ModuleList([
            nn.ModuleList([normalizer() for _ in range(n_stages)])
            for _ in range(n_blocks)
        ])
        self.convs = nn.ModuleList([
            nn.ModuleList([
                ncsn_conv3x3(features, features, stride=1, bias=False)
                for _ in range(n_stages)
            ])
            for _ in range(n_blocks)
        ])

    def forward(self, x, y):
        for norms, convs in zip(self.norms, self.convs):
            residual = x
            for norm, conv in zip(norms, convs):
                x = norm(x, y)
                x = self.act(x)
                x = conv(x)
            x = x + residual
        return x


class MSFBlock(nn.Module):
    """MSFBlock for RefineNet. Used in NCSNv2."""

    def __init__(self, in_planes: Sequence[int], features: int,
                 shape: Sequence[int], interpolation: str = "bilinear"):
        super().__init__()
        self.shape = shape           # (H, W) target spatial size
        self.interpolation = interpolation
        self.convs = nn.ModuleList(
            [ncsn_conv3x3(in_ch, features, stride=1, bias=True)
             for in_ch in in_planes]
        )

    def forward(self, xs):
        # xs: list of (N, C_i, H_i, W_i)
        sums = torch.zeros(
            xs[0].shape[0], self.convs[0].out_channels, *self.shape,
            device=xs[0].device, dtype=xs[0].dtype
        )
        for conv, x in zip(self.convs, xs):
            h = conv(x)
            mode = "bilinear" if self.interpolation == "bilinear" else "nearest"
            h = F.interpolate(h, size=self.shape, mode=mode,
                              align_corners=False if mode == "bilinear" else None)
            sums = sums + h
        return sums


class CondMSFBlock(nn.Module):
    """Noise-conditional MSFBlock for RefineNet. Used in NCSNv1."""

    def __init__(self, in_planes: Sequence[int], features: int,
                 shape: Sequence[int], normalizer,
                 interpolation: str = "bilinear"):
        super().__init__()
        self.shape = shape
        self.interpolation = interpolation
        self.norms = nn.ModuleList([normalizer() for _ in in_planes])
        self.convs = nn.ModuleList(
            [ncsn_conv3x3(in_ch, features, stride=1, bias=True)
             for in_ch in in_planes]
        )

    def forward(self, xs, y):
        sums = torch.zeros(
            xs[0].shape[0], self.convs[0].out_channels, *self.shape,
            device=xs[0].device, dtype=xs[0].dtype
        )
        for norm, conv, x in zip(self.norms, self.convs, xs):
            h = norm(x, y)
            h = conv(h)
            mode = "bilinear" if self.interpolation == "bilinear" else "nearest"
            h = F.interpolate(h, size=self.shape, mode=mode,
                              align_corners=False if mode == "bilinear" else None)
            sums = sums + h
        return sums


class RefineBlock(nn.Module):
    """RefineBlock for building NCSNv2 RefineNet."""

    def __init__(self, in_planes: Sequence[int], features: int,
                 output_shape: Sequence[int], act=nn.ReLU(),
                 interpolation: str = "bilinear",
                 start: bool = False, end: bool = False):
        super().__init__()
        self.start = start
        self.end = end
        self.act = act

        self.rcu_blocks = nn.ModuleList([
            RCUBlock(in_ch, n_blocks=2, n_stages=2, act=act)
            for in_ch in in_planes
        ])

        if not start:
            self.msf = MSFBlock(in_planes, features, output_shape, interpolation)

        self.crp = CRPBlock(features, n_stages=2, act=act)

        n_out_blocks = 3 if end else 1
        self.rcu_block_output = RCUBlock(
            features, n_blocks=n_out_blocks, n_stages=2, act=act
        )

    def forward(self, xs):
        hs = [rcu(x) for rcu, x in zip(self.rcu_blocks, xs)]
        h = hs[0] if self.start else self.msf(hs)
        h = self.crp(h)
        h = self.rcu_block_output(h)
        return h


class CondRefineBlock(nn.Module):
    """Noise-conditional RefineBlock for building NCSNv1 RefineNet."""

    def __init__(self, in_planes: Sequence[int], features: int,
                 output_shape: Sequence[int], normalizer,
                 act=nn.ReLU(), interpolation: str = "bilinear",
                 start: bool = False, end: bool = False):
        super().__init__()
        self.start = start
        self.end = end

        self.rcu_blocks = nn.ModuleList([
            CondRCUBlock(in_ch, n_blocks=2, n_stages=2,
                         normalizer=normalizer, act=act)
            for in_ch in in_planes
        ])

        if not start:
            self.msf = CondMSFBlock(in_planes, features, output_shape,
                                    normalizer, interpolation)

        self.crp = CondCRPBlock(features, n_stages=2,
                                normalizer=normalizer, act=act)

        n_out_blocks = 3 if end else 1
        self.rcu_block_output = CondRCUBlock(
            features, n_blocks=n_out_blocks, n_stages=2,
            normalizer=normalizer, act=act
        )

    def forward(self, xs, y):
        hs = [rcu(x, y) for rcu, x in zip(self.rcu_blocks, xs)]
        h = hs[0] if self.start else self.msf(hs, y)
        h = self.crp(h, y)
        h = self.rcu_block_output(h, y)
        return h


# ---------------------------------------------------------------------------
# ResNet backbone (NCSNv2)
# ---------------------------------------------------------------------------

class ConvMeanPool(nn.Module):
    """Conv followed by 2x2 average (non-overlapping) downsampling."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                              stride=1, padding=kernel_size // 2, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        # Average the four strided sub-grids
        x = (x[:, :, ::2, ::2] + x[:, :, 1::2, ::2] +
             x[:, :, ::2, 1::2] + x[:, :, 1::2, 1::2]) / 4.0
        return x


class MeanPoolConv(nn.Module):
    """2x2 average downsampling followed by Conv."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                              stride=1, padding=kernel_size // 2, bias=bias)

    def forward(self, x):
        x = (x[:, :, ::2, ::2] + x[:, :, 1::2, ::2] +
             x[:, :, ::2, 1::2] + x[:, :, 1::2, 1::2]) / 4.0
        x = self.conv(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block for NCSNv2."""

    def __init__(self, in_ch: int, out_ch: int, normalization,
                 resample: Optional[str] = None, act=nn.ELU(),
                 dilation: int = 1):
        super().__init__()
        self.act = act
        self.resample = resample
        self.dilation = dilation
        self.norm1 = normalization()
        self.norm2 = normalization()

        if resample == "down":
            self.conv1 = ncsn_conv3x3(in_ch, in_ch, dilation=dilation)
            if dilation > 1:
                self.conv2 = ncsn_conv3x3(in_ch, out_ch, dilation=dilation)
                self.shortcut = ncsn_conv3x3(in_ch, out_ch, dilation=dilation)
            else:
                self.conv2 = ConvMeanPool(in_ch, out_ch)
                self.shortcut = ConvMeanPool(in_ch, out_ch, kernel_size=1)
        elif resample is None:
            if dilation > 1:
                self.conv1 = ncsn_conv3x3(in_ch, out_ch, dilation=dilation)
                self.conv2 = ncsn_conv3x3(out_ch, out_ch, dilation=dilation)
                self.shortcut = (
                    nn.Identity() if in_ch == out_ch
                    else ncsn_conv3x3(in_ch, out_ch, dilation=dilation)
                )
            else:
                self.conv1 = ncsn_conv3x3(in_ch, out_ch)
                self.conv2 = ncsn_conv3x3(out_ch, out_ch)
                self.shortcut = (
                    nn.Identity() if in_ch == out_ch
                    else ncsn_conv1x1(in_ch, out_ch)
                )
        else:
            raise ValueError(f"Unknown resample mode: {resample}")

    def forward(self, x):
        h = self.act(self.norm1(x))
        if self.resample == "down":
            h = self.conv1(h)
            h = self.act(self.norm2(h))
            h = self.conv2(h)
        else:
            h = self.conv1(h)
            h = self.act(self.norm2(h))
            h = self.conv2(h)
        return h + self.shortcut(x)


class ConditionalResidualBlock(nn.Module):
    """Noise-conditional residual block for NCSNv1."""

    def __init__(self, in_ch: int, out_ch: int, normalization,
                 resample: Optional[str] = None, act=nn.ELU(),
                 dilation: int = 1):
        super().__init__()
        self.act = act
        self.resample = resample
        self.dilation = dilation
        self.norm1 = normalization()
        self.norm2 = normalization()

        if resample == "down":
            self.conv1 = ncsn_conv3x3(in_ch, in_ch, dilation=dilation)
            if dilation > 1:
                self.conv2 = ncsn_conv3x3(in_ch, out_ch, dilation=dilation)
                self.shortcut = ncsn_conv3x3(in_ch, out_ch, dilation=dilation)
            else:
                self.conv2 = ConvMeanPool(in_ch, out_ch)
                self.shortcut = ConvMeanPool(in_ch, out_ch, kernel_size=1)
        elif resample is None:
            if dilation > 1:
                self.conv1 = ncsn_conv3x3(in_ch, out_ch, dilation=dilation)
                self.conv2 = ncsn_conv3x3(out_ch, out_ch, dilation=dilation)
                self.shortcut = (
                    nn.Identity() if in_ch == out_ch
                    else ncsn_conv3x3(in_ch, out_ch, dilation=dilation)
                )
            else:
                self.conv1 = ncsn_conv3x3(in_ch, out_ch)
                self.conv2 = ncsn_conv3x3(out_ch, out_ch)
                self.shortcut = (
                    nn.Identity() if in_ch == out_ch
                    else ncsn_conv1x1(in_ch, out_ch)
                )
        else:
            raise ValueError(f"Unknown resample mode: {resample}")

    def forward(self, x, y):
        h = self.act(self.norm1(x, y))
        if self.resample == "down":
            h = self.conv1(h)
            h = self.act(self.norm2(h, y))
            h = self.conv2(h)
        else:
            h = self.conv1(h)
            h = self.act(self.norm2(h, y))
            h = self.conv2(h)
        return h + self.shortcut(x)


# ---------------------------------------------------------------------------
# DDPM blocks
# ---------------------------------------------------------------------------

def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int,
                           max_positions: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embedding (same as DDPM / transformers)."""
    assert timesteps.ndim == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32,
                                 device=timesteps.device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:   # zero-pad for odd dimensions
        emb = F.pad(emb, (0, 1))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class NIN(nn.Module):
    """Network-in-Network: a learned 1x1 linear projection (no spatial conv)."""

    def __init__(self, in_dim: int, num_units: int, init_scale: float = 0.1):
        super().__init__()
        self.W = nn.Parameter(torch.empty(in_dim, num_units))
        self.b = nn.Parameter(torch.zeros(num_units))
        init_scale = max(init_scale, 1e-10)
        # fan_avg uniform
        limit = math.sqrt(3 * init_scale / ((in_dim + num_units) / 2.0))
        nn.init.uniform_(self.W, -limit, limit)

    def forward(self, x):
        # x: (N, C, H, W) — apply projection over channel dim at each spatial pos
        # Equivalent to einsum "nchw,cd->ndhw"
        return torch.einsum("nchw,cd->ndhw", x, self.W) + self.b[None, :, None, None]


class AttnBlock(nn.Module):
    """Channel-wise self-attention block."""

    def __init__(self, channels: int, normalize):
        super().__init__()
        self.norm = normalize()
        self.q = NIN(channels, channels)
        self.k = NIN(channels, channels)
        self.v = NIN(channels, channels)
        self.proj = NIN(channels, channels, init_scale=0.0)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h)   # (B, C, H, W)
        k = self.k(h)
        v = self.v(h)

        # Attention weights: (B, H, W, H, W)
        w = torch.einsum("bchw,bcHW->bhwHW", q, k) * (C ** -0.5)
        w = w.reshape(B, H, W, H * W)
        w = torch.softmax(w, dim=-1)
        w = w.reshape(B, H, W, H, W)

        h_out = torch.einsum("bhwHW,bcHW->bchw", w, v)
        h_out = self.proj(h_out)
        return x + h_out


class Upsample(nn.Module):
    def __init__(self, channels: int, with_conv: bool = False):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = ddpm_conv3x3(channels, channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels: int, with_conv: bool = False):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = ddpm_conv3x3(channels, channels, stride=2)

    def forward(self, x):
        if self.with_conv:
            return self.conv(x)
        return F.avg_pool2d(x, kernel_size=2, stride=2)


class ResnetBlockDDPM(nn.Module):
    """ResNet block used in DDPM."""

    def __init__(self, in_ch: int, act, normalize, out_ch: Optional[int] = None,
                 conv_shortcut: bool = False, dropout: float = 0.5):
        super().__init__()
        out_ch = out_ch if out_ch is not None else in_ch
        self.act = act
        self.conv_shortcut = conv_shortcut

        self.norm1 = normalize()
        self.conv1 = ddpm_conv3x3(in_ch, out_ch)
        self.norm2 = normalize()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = ddpm_conv3x3(out_ch, out_ch, init_scale=0.0)

        # Time embedding projection
        self.temb_proj = nn.Linear(out_ch, out_ch)   # applied after act(temb)
        _variance_scaling_init(self.temb_proj.weight, scale=1.0, mode="fan_avg")
        nn.init.zeros_(self.temb_proj.bias)

        # Shortcut
        if in_ch != out_ch:
            if conv_shortcut:
                self.shortcut = ddpm_conv3x3(in_ch, out_ch)
            else:
                self.shortcut = NIN(in_ch, out_ch)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, temb=None, train=True):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        if temb is not None:
            # temb: (B, C) -> add (B, C, 1, 1)
            h = h + self.temb_proj(self.act(temb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.dropout(h) if train else h
        h = self.conv2(h)
        return self.shortcut(x) + h