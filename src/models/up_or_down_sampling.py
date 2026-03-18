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

"""Layers for up/down-sampling images. Ported from JAX/Flax to PyTorch.

Many functions are originally ported from https://github.com/NVlabs/stylegan2.

Layout note:
  The original JAX code operates in NHWC.
  This PyTorch version operates in NCHW throughout.
  The Conv2d module and all helpers work in NCHW.
"""

import math
import numpy as np
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------

def _setup_kernel(k: Sequence) -> np.ndarray:
    """Normalize a 1-D or 2-D FIR kernel."""
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    k /= np.sum(k)
    assert k.ndim == 2 and k.shape[0] == k.shape[1]
    return k


# ---------------------------------------------------------------------------
# Naive up/down-sample (no FIR, used as building blocks)
# ---------------------------------------------------------------------------

def naive_upsample_2d(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """Nearest-neighbour upsample in NCHW layout."""
    N, C, H, W = x.shape
    x = x.view(N, C, H, 1, W, 1)
    x = x.expand(N, C, H, factor, W, factor)
    return x.contiguous().view(N, C, H * factor, W * factor)


def naive_downsample_2d(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """Average downsample in NCHW layout."""
    N, C, H, W = x.shape
    x = x.view(N, C, H // factor, factor, W // factor, factor)
    return x.mean(dim=(3, 5))


# ---------------------------------------------------------------------------
# Core upfirdn primitive
# ---------------------------------------------------------------------------

def upfirdn_2d(
    x: torch.Tensor,
    k: np.ndarray,
    upx: int, upy: int,
    downx: int, downy: int,
    padx0: int, padx1: int,
    pady0: int, pady1: int,
) -> torch.Tensor:
    """Pad → upsample (insert zeros) → FIR filter → downsample.

    Args:
        x:    (N, C, H, W)  — NCHW layout (differs from original NHWC).
        k:    2-D FIR kernel array, shape (kH, kW).
        upx, upy:   integer upsampling factors along W and H axes.
        downx, downy: integer downsampling factors.
        padx0/padx1: left/right padding (negative → crop).
        pady0/pady1: top/bottom padding (negative → crop).

    Returns:
        (N, C, outH, outW) tensor.
    """
    assert x.ndim == 4
    N, C, inH, inW = x.shape
    kernelH, kernelW = k.shape

    # 1. Upsample: insert (up-1) zeros between each pixel.
    if upx > 1 or upy > 1:
        x = x.view(N, C, inH, 1, inW, 1)
        x = F.pad(x, [0, upx - 1, 0, 0, 0, upy - 1])   # pad last 3 dim pairs
        x = x.contiguous().view(N, C, inH * upy, inW * upx)

    # 2. Pad / crop along H and W.
    #    F.pad takes (left, right, top, bottom) for 4-D input.
    pad_left  = max(padx0, 0)
    pad_right = max(padx1, 0)
    pad_top   = max(pady0, 0)
    pad_bot   = max(pady1, 0)
    if pad_left or pad_right or pad_top or pad_bot:
        x = F.pad(x, [pad_left, pad_right, pad_top, pad_bot])

    # Crop if original pads were negative.
    h_start = max(-pady0, 0)
    h_end   = x.shape[2] - max(-pady1, 0)
    w_start = max(-padx0, 0)
    w_end   = x.shape[3] - max(-padx1, 0)
    x = x[:, :, h_start:h_end, w_start:w_end]

    # 3. Convolve with FIR kernel (applied channel-wise via groups).
    #    Flip kernel to implement correlation-as-convolution.
    k_tensor = torch.tensor(k[::-1, ::-1].copy(), dtype=x.dtype, device=x.device)
    # Shape: (1, 1, kH, kW) → broadcast over all C channels using groups=C
    k_tensor = k_tensor.view(1, 1, kernelH, kernelW).expand(C, 1, kernelH, kernelW)
    x = F.conv2d(x, k_tensor, padding=0, groups=C)

    # 4. Downsample (throw away pixels).
    if downx > 1 or downy > 1:
        x = x[:, :, ::downy, ::downx]

    return x


def _simple_upfirdn_2d(
    x: torch.Tensor,
    k: np.ndarray,
    up: int = 1,
    down: int = 1,
    pad0: int = 0,
    pad1: int = 0,
) -> torch.Tensor:
    """Wrapper that applies the same factor / padding on both spatial axes.

    Operates purely in NCHW — no data_format argument needed.
    """
    assert x.ndim == 4
    return upfirdn_2d(
        x, k,
        upx=up, upy=up,
        downx=down, downy=down,
        padx0=pad0, padx1=pad1,
        pady0=pad0, pady1=pad1,
    )


# ---------------------------------------------------------------------------
# Fused upsample-conv and conv-downsample (StyleGAN2 style)
# ---------------------------------------------------------------------------

def upsample_conv_2d(
    x: torch.Tensor,
    w: torch.Tensor,
    k: Optional[Sequence] = None,
    factor: int = 2,
    gain: float = 1.0,
) -> torch.Tensor:
    """Fused upsample → conv (NCHW).

    Args:
        x: (N, C_in, H, W)
        w: (C_out, C_in, kH, kW)  — standard PyTorch Conv weight layout.
        k: FIR kernel (1-D or 2-D). Defaults to [1]*factor (nearest-neighbour).
        factor: integer upsampling factor.
        gain: signal scaling factor.

    Returns:
        (N, C_out, H*factor, W*factor)
    """
    assert isinstance(factor, int) and factor >= 1
    assert w.ndim == 4
    outC, inC, convH, convW = w.shape
    assert convH == convW

    if k is None:
        k = [1] * factor
    k_np = _setup_kernel(k) * (gain * (factor ** 2))
    p = (k_np.shape[0] - factor) - (convW - 1)

    # For conv_transpose2d we need weight shape (C_in, C_out, kH, kW).
    # Flip the kernel spatially to match the original JAX w[::-1, ::-1] trick.
    w_t = w.flip([2, 3]).permute(1, 0, 2, 3).contiguous()

    # Transposed convolution (stride = factor, no padding yet).
    x = F.conv_transpose2d(x, w_t, stride=factor, padding=0)

    return _simple_upfirdn_2d(
        x, k_np,
        pad0=(p + 1) // 2 + factor - 1,
        pad1=p // 2 + 1,
    )


def conv_downsample_2d(
    x: torch.Tensor,
    w: torch.Tensor,
    k: Optional[Sequence] = None,
    factor: int = 2,
    gain: float = 1.0,
) -> torch.Tensor:
    """Fused conv → downsample (NCHW).

    Args:
        x: (N, C_in, H, W)
        w: (C_out, C_in, kH, kW)
        k: FIR kernel. Defaults to [1]*factor (average pooling).
        factor: integer downsampling factor.
        gain: signal scaling factor.

    Returns:
        (N, C_out, H//factor, W//factor)
    """
    assert isinstance(factor, int) and factor >= 1
    assert w.ndim == 4
    _outC, _inC, convH, convW = w.shape
    assert convH == convW

    if k is None:
        k = [1] * factor
    k_np = _setup_kernel(k) * gain
    p = (k_np.shape[0] - factor) + (convW - 1)

    x = _simple_upfirdn_2d(x, k_np, pad0=(p + 1) // 2, pad1=p // 2)
    return F.conv2d(x, w, stride=factor, padding=0)


# ---------------------------------------------------------------------------
# Stand-alone up/downsample with FIR
# ---------------------------------------------------------------------------

def upsample_2d(
    x: torch.Tensor,
    k: Optional[Sequence] = None,
    factor: int = 2,
    gain: float = 1.0,
) -> torch.Tensor:
    """Upsample with FIR anti-aliasing filter (NCHW).

    Args:
        x: (N, C, H, W)
        k: FIR kernel. Defaults to [1]*factor.
        factor: integer upsampling factor.
        gain: signal scaling factor.

    Returns:
        (N, C, H*factor, W*factor)
    """
    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k_np = _setup_kernel(k) * (gain * (factor ** 2))
    p = k_np.shape[0] - factor
    return _simple_upfirdn_2d(
        x, k_np,
        up=factor,
        pad0=(p + 1) // 2 + factor - 1,
        pad1=p // 2,
    )


def downsample_2d(
    x: torch.Tensor,
    k: Optional[Sequence] = None,
    factor: int = 2,
    gain: float = 1.0,
) -> torch.Tensor:
    """Downsample with FIR anti-aliasing filter (NCHW).

    Args:
        x: (N, C, H, W)
        k: FIR kernel. Defaults to [1]*factor.
        factor: integer downsampling factor.
        gain: signal scaling factor.

    Returns:
        (N, C, H//factor, W//factor)
    """
    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k_np = _setup_kernel(k) * gain
    p = k_np.shape[0] - factor
    return _simple_upfirdn_2d(
        x, k_np,
        down=factor,
        pad0=(p + 1) // 2,
        pad1=p // 2,
    )


# ---------------------------------------------------------------------------
# Conv2d module (StyleGAN2-style, with optional fused up/downsample)
# ---------------------------------------------------------------------------

class Conv2d(nn.Module):
    """Conv2d with optional fused FIR up- or down-sampling (StyleGAN2).

    Operates in NCHW layout.

    Args:
        in_ch:          Number of input channels.
        fmaps:          Number of output channels.
        kernel:         Square kernel size (must be odd and >= 1).
        up:             If True, apply fused upsample-then-conv.
        down:           If True, apply fused conv-then-downsample.
        resample_kernel: 1-D FIR kernel coefficients (default [1,3,3,1]).
        use_bias:       Whether to add a learnable bias.
        kernel_init:    Optional callable ``f(tensor) -> None`` for weight init.
                        Defaults to PyTorch's default (kaiming_uniform).
    """

    def __init__(
        self,
        in_ch: int,
        fmaps: int,
        kernel: int,
        up: bool = False,
        down: bool = False,
        resample_kernel: Tuple[int, ...] = (1, 3, 3, 1),
        use_bias: bool = True,
        kernel_init=None,
    ):
        super().__init__()
        assert not (up and down), "Cannot set both up=True and down=True."
        assert kernel >= 1 and kernel % 2 == 1, "Kernel must be odd and >= 1."

        self.fmaps = fmaps
        self.kernel = kernel
        self.up = up
        self.down = down
        self.resample_kernel = resample_kernel
        self.use_bias = use_bias

        # Weight shape: (out_ch, in_ch, kH, kW) — standard PyTorch layout.
        self.weight = nn.Parameter(torch.empty(fmaps, in_ch, kernel, kernel))
        if kernel_init is not None:
            kernel_init(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(fmaps))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if self.up:
            x = upsample_conv_2d(x, w, k=self.resample_kernel)
        elif self.down:
            x = conv_downsample_2d(x, w, k=self.resample_kernel)
        else:
            padding = self.kernel // 2
            x = F.conv2d(x, w, padding=padding)

        if self.use_bias:
            x = x + self.bias[None, :, None, None]
        return x