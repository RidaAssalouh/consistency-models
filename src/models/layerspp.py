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

"""Layers for defining NCSN++. Converted from JAX/Flax to PyTorch.

Layout note: all tensors are NCHW throughout.
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import layers
from src.models import up_or_down_sampling

# Aliases matching the original
conv1x1 = layers.ddpm_conv1x1
conv3x3 = layers.ddpm_conv3x3
NIN = layers.NIN
default_init = layers.default_init

def _variance_scaling_init(tensor: torch.Tensor, scale: float = 1.0):
    """fan_avg uniform init — mirrors layers.default_init."""
    scale = max(scale, 1e-10)
    fan_in = tensor[0].numel()
    fan_out = tensor.shape[0] * tensor[0][0].numel() if tensor.ndim >= 2 else fan_in
    n = (fan_in + fan_out) / 2.0
    limit = math.sqrt(3 * scale / n)
    nn.init.uniform_(tensor, -limit, limit)


def _num_groups(channels: int) -> int:
    """Match the original: min(C // 4, 32)."""
    return min(channels // 4, 32)


# ---------------------------------------------------------------------------
# Gaussian Fourier projection
# ---------------------------------------------------------------------------

class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size: int = 256, scale: float = 1.0):
        super().__init__()
        # Fixed random frequencies — not trained (stop_gradient in original).
        self.register_buffer(
            "W", torch.randn(embedding_size) * scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,)  →  x_proj: (N, embedding_size)
        x_proj = x[:, None] * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# ---------------------------------------------------------------------------
# Combine (skip-connection merge)
# ---------------------------------------------------------------------------

class Combine(nn.Module):
    """Combine information from skip connections.

    Args:
        in_ch:  channels of the skip-connection tensor x.
        out_ch: channels of the main-path tensor y (and output for 'cat' on x).
        method: 'cat' (concatenate) or 'sum'.
    """

    def __init__(self, in_ch: int, out_ch: int, method: str = "cat"):
        super().__init__()
        assert method in ("cat", "sum"), f"Unknown method: {method}"
        self.method = method
        # Project x to out_ch before combining.
        self.conv = conv1x1(in_ch, out_ch)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        if self.method == "cat":
            return torch.cat([h, y], dim=1)   # NCHW: concat on channel dim
        else:
            return h + y


# ---------------------------------------------------------------------------
# Attention block (NCSN++ / DDPM++)
# ---------------------------------------------------------------------------

class AttnBlockpp(nn.Module):
    """Channel-wise self-attention block (DDPM++).

    Args:
        channels:     Number of input (and output) channels.
        skip_rescale: If True, divide output by sqrt(2).
        init_scale:   Init scale for the final NIN projection.
    """

    def __init__(self, channels: int, skip_rescale: bool = False,
                 init_scale: float = 0.0):
        super().__init__()
        self.skip_rescale = skip_rescale
        self.norm = nn.GroupNorm(num_groups=_num_groups(channels),
                                 num_channels=channels)
        self.q = NIN(channels, channels)
        self.k = NIN(channels, channels)
        self.v = NIN(channels, channels)
        self.proj = NIN(channels, channels, init_scale=init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h)   # (B, C, H, W)
        k = self.k(h)
        v = self.v(h)

        w = torch.einsum("bchw,bcHW->bhwHW", q, k) * (C ** -0.5)
        w = w.reshape(B, H, W, H * W)
        w = torch.softmax(w, dim=-1)
        w = w.reshape(B, H, W, H, W)
        h_out = torch.einsum("bhwHW,bcHW->bchw", w, v)
        h_out = self.proj(h_out)

        out = x + h_out
        if self.skip_rescale:
            out = out / math.sqrt(2.0)
        return out


# ---------------------------------------------------------------------------
# Upsample
# ---------------------------------------------------------------------------

class Upsample(nn.Module):
    """Upsample by 2×, optionally with FIR anti-aliasing.

    Args:
        in_ch:      Input channel count.
        out_ch:     Output channel count (defaults to in_ch).
        with_conv:  Apply a conv after upsampling.
        fir:        Use FIR-filtered upsampling (StyleGAN2 style).
        fir_kernel: 1-D FIR kernel coefficients.
    """

    def __init__(self, in_ch: int, out_ch: Optional[int] = None,
                 with_conv: bool = False, fir: bool = False,
                 fir_kernel: Tuple[int, ...] = (1, 3, 3, 1)):
        super().__init__()
        out_ch = out_ch or in_ch
        self.fir = fir
        self.with_conv = with_conv
        self.fir_kernel = fir_kernel
        self.out_ch = out_ch
        self.in_ch = in_ch

        if with_conv:
            if fir:
                self.conv = up_or_down_sampling.Conv2d(
                    in_ch=in_ch,
                    fmaps=out_ch,
                    kernel=3,
                    up=True,
                    resample_kernel=fir_kernel,
                    use_bias=True,
                    kernel_init=lambda t: _variance_scaling_init(t, 1.0),
                )
            else:
                self.conv = conv3x3(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if not self.fir:
            h = F.interpolate(x, scale_factor=2, mode="nearest")
            if self.with_conv:
                h = self.conv(h)
        else:
            if not self.with_conv:
                h = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = self.conv(x)

        assert h.shape == (B, self.out_ch, H * 2, W * 2)
        return h


# ---------------------------------------------------------------------------
# Downsample
# ---------------------------------------------------------------------------

class Downsample(nn.Module):
    """Downsample by 2×, optionally with FIR anti-aliasing.

    Args:
        in_ch:      Input channel count.
        out_ch:     Output channel count (defaults to in_ch).
        with_conv:  Apply a strided conv instead of pooling.
        fir:        Use FIR-filtered downsampling (StyleGAN2 style).
        fir_kernel: 1-D FIR kernel coefficients.
    """

    def __init__(self, in_ch: int, out_ch: Optional[int] = None,
                 with_conv: bool = False, fir: bool = False,
                 fir_kernel: Tuple[int, ...] = (1, 3, 3, 1)):
        super().__init__()
        out_ch = out_ch or in_ch
        self.fir = fir
        self.with_conv = with_conv
        self.fir_kernel = fir_kernel
        self.out_ch = out_ch
        self.in_ch = in_ch

        if with_conv:
            if fir:
                self.conv = up_or_down_sampling.Conv2d(
                    in_ch=in_ch,
                    fmaps=out_ch,
                    kernel=3,
                    down=True,
                    resample_kernel=fir_kernel,
                    use_bias=True,
                    kernel_init=lambda t: _variance_scaling_init(t, 1.0),
                )
            else:
                self.conv = conv3x3(in_ch, out_ch, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if not self.fir:
            if self.with_conv:
                x = self.conv(x)
            else:
                x = F.avg_pool2d(x, kernel_size=2, stride=2)
        else:
            if not self.with_conv:
                x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
            else:
                x = self.conv(x)

        assert x.shape == (B, self.out_ch, H // 2, W // 2)
        return x


# ---------------------------------------------------------------------------
# ResnetBlockDDPMpp
# ---------------------------------------------------------------------------

class ResnetBlockDDPMpp(nn.Module):
    """ResBlock adapted from DDPM (NCSN++ variant).

    Args:
        in_ch:        Input channel count.
        act:          Activation module (e.g. nn.SiLU()).
        temb_dim:     Time-embedding dimension (None → no conditioning).
        out_ch:       Output channels (defaults to in_ch).
        conv_shortcut: Use conv3x3 for shortcut when channels differ
                       (otherwise NIN).
        dropout:      Dropout probability.
        skip_rescale: Divide output by sqrt(2) when True.
        init_scale:   Init scale for the final conv.
    """

    def __init__(self, in_ch: int, act: nn.Module,
                 temb_dim: Optional[int] = None,
                 out_ch: Optional[int] = None,
                 conv_shortcut: bool = False,
                 dropout: float = 0.1,
                 skip_rescale: bool = False,
                 init_scale: float = 0.0):
        super().__init__()
        out_ch = out_ch or in_ch
        self.act = act
        self.skip_rescale = skip_rescale

        self.norm1 = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.conv1 = conv3x3(in_ch, out_ch)

        if temb_dim is not None:
            self.temb_proj = nn.Linear(temb_dim, out_ch)
            default_init(1.0)(self.temb_proj.weight)
            nn.init.zeros_(self.temb_proj.bias)
        else:
            self.temb_proj = None

        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv3x3(out_ch, out_ch, init_scale=init_scale)

        if in_ch != out_ch:
            if conv_shortcut:
                self.shortcut = conv3x3(in_ch, out_ch)
            else:
                self.shortcut = NIN(in_ch, out_ch)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None,
                train: bool = True) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        if temb is not None and self.temb_proj is not None:
            h = h + self.temb_proj(self.act(temb))[:, :, None, None]

        h = self.act(self.norm2(h))
        if train:
            h = self.dropout(h)
        h = self.conv2(h)

        out = self.shortcut(x) + h
        if self.skip_rescale:
            out = out / math.sqrt(2.0)
        return out


# ---------------------------------------------------------------------------
# ResnetBlockBigGANpp
# ---------------------------------------------------------------------------

class ResnetBlockBigGANpp(nn.Module):
    """ResBlock adapted from BigGAN (NCSN++ variant).

    Args:
        in_ch:        Input channel count.
        act:          Activation module.
        temb_dim:     Time-embedding dimension (None → no conditioning).
        out_ch:       Output channels (defaults to in_ch).
        up:           Upsample spatial dims by 2×.
        down:         Downsample spatial dims by 2×.
        dropout:      Dropout probability.
        fir:          Use FIR-filtered re-sampling.
        fir_kernel:   1-D FIR kernel coefficients.
        skip_rescale: Divide output by sqrt(2) when True.
        init_scale:   Init scale for the final conv.
    """

    def __init__(self, in_ch: int, act: nn.Module,
                 temb_dim: Optional[int] = None,
                 out_ch: Optional[int] = None,
                 up: bool = False, down: bool = False,
                 dropout: float = 0.1,
                 fir: bool = False,
                 fir_kernel: Tuple[int, ...] = (1, 3, 3, 1),
                 skip_rescale: bool = True,
                 init_scale: float = 0.0):
        super().__init__()
        out_ch = out_ch or in_ch
        self.act = act
        self.up = up
        self.down = down
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.skip_rescale = skip_rescale

        self.norm1 = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.conv1 = conv3x3(in_ch, out_ch)

        if temb_dim is not None:
            self.temb_proj = nn.Linear(temb_dim, out_ch)
            default_init(1.0)(self.temb_proj.weight)
            nn.init.zeros_(self.temb_proj.bias)
        else:
            self.temb_proj = None

        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv3x3(out_ch, out_ch, init_scale=init_scale)

        # Shortcut is always a 1×1 conv when channels differ OR when resampling.
        if in_ch != out_ch or up or down:
            self.shortcut = conv1x1(in_ch, out_ch)
        else:
            self.shortcut = nn.Identity()

    def _resample(self, t: torch.Tensor) -> torch.Tensor:
        if self.up:
            if self.fir:
                return up_or_down_sampling.upsample_2d(t, self.fir_kernel, factor=2)
            else:
                return up_or_down_sampling.naive_upsample_2d(t, factor=2)
        elif self.down:
            if self.fir:
                return up_or_down_sampling.downsample_2d(t, self.fir_kernel, factor=2)
            else:
                return up_or_down_sampling.naive_downsample_2d(t, factor=2)
        return t

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None,
                train: bool = True) -> torch.Tensor:
        h = self.act(self.norm1(x))

        h = self._resample(h)
        x = self._resample(x)

        h = self.conv1(h)

        if temb is not None and self.temb_proj is not None:
            h = h + self.temb_proj(self.act(temb))[:, :, None, None]

        h = self.act(self.norm2(h))
        if train:
            h = self.dropout(h)
        h = self.conv2(h)

        out = self.shortcut(x) + h
        if self.skip_rescale:
            out = out / math.sqrt(2.0)
        return out


