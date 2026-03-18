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

"""Normalization layers. Converted from JAX/Flax to PyTorch.

Layout note: all tensors are NCHW throughout.
  - JAX reduces over axes (1, 2) which are H, W in NHWC.
  - PyTorch reduces over axes (2, 3) which are H, W in NCHW.
"""

import functools
import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_normalization(config, conditional: bool = False):
    """Return a normalization class (or partial) from the config."""
    norm = config.model.normalization
    if conditional:
        if norm == "InstanceNorm++":
            return functools.partial(
                ConditionalInstanceNorm2dPlus,
                num_classes=config.model.num_classes,
            )
        else:
            raise NotImplementedError(f"{norm} not implemented for conditional use.")
    else:
        if norm == "InstanceNorm":
            return InstanceNorm2d
        elif norm == "InstanceNorm++":
            return InstanceNorm2dPlus
        elif norm == "VarianceNorm":
            return VarianceNorm2d
        elif norm == "GroupNorm":
            return nn.GroupNorm
        else:
            raise ValueError(f"Unknown normalization: {norm}")


# ---------------------------------------------------------------------------
# VarianceNorm2d
# ---------------------------------------------------------------------------

class VarianceNorm2d(nn.Module):
    """Variance normalization for images (NCHW).

    Divides each sample by its per-channel spatial standard deviation, then
    scales by a learned parameter initialised near 1.
    """

    def __init__(self, num_features: int, bias: bool = False):
        super().__init__()
        # scale initialised as 1 + N(0, 0.02)
        self.scale = nn.Parameter(torch.randn(1, num_features, 1, 1) * 0.02 + 1.0)
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Variance over H, W — keep batch and channel dims.
        variance = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        h = x / (variance + 1e-5).sqrt()
        h = h * self.scale
        if self.bias is not None:
            h = h + self.bias
        return h


# ---------------------------------------------------------------------------
# InstanceNorm2d
# ---------------------------------------------------------------------------

class InstanceNorm2d(nn.Module):
    """Instance normalization for images (NCHW).

    Mean and variance computed over H×W for each (sample, channel).
    Learnable per-channel scale (γ) and optional bias (β).
    """

    def __init__(self, num_features: int, bias: bool = True):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        variance = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        h = (x - mean) / (variance + 1e-5).sqrt()
        h = h * self.scale
        if self.bias is not None:
            h = h + self.bias
        return h


# ---------------------------------------------------------------------------
# InstanceNorm2dPlus
# ---------------------------------------------------------------------------

class InstanceNorm2dPlus(nn.Module):
    """InstanceNorm++ as proposed in the original NCSN paper (NCHW).

    In addition to standard per-channel instance normalisation, the
    normalised channel means are also fed back in via a learned α, giving
    the network access to global colour/brightness statistics.
    """

    def __init__(self, num_features: int, bias: bool = True):
        super().__init__()
        # α and γ both initialised as 1 + N(0, 0.02)
        self.alpha = nn.Parameter(torch.randn(1, num_features, 1, 1) * 0.02 + 1.0)
        self.gamma = nn.Parameter(torch.randn(1, num_features, 1, 1) * 0.02 + 1.0)
        if bias:
            self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        else:
            self.beta = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # means: (N, C) — spatial mean per (sample, channel)
        means = x.mean(dim=(2, 3))                         # (N, C)
        m = means.mean(dim=1, keepdim=True)                # (N, 1)
        v = means.var(dim=1, keepdim=True, unbiased=False) # (N, 1)
        means_plus = (means - m) / (v + 1e-5).sqrt()      # (N, C)

        # Standard instance norm over H×W.
        variance = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        h = (x - means[:, :, None, None]) / (variance + 1e-5).sqrt()

        # Add normalised channel-mean signal, scaled by α.
        h = h + means_plus[:, :, None, None] * self.alpha
        h = h * self.gamma
        if self.beta is not None:
            h = h + self.beta
        return h


# ---------------------------------------------------------------------------
# ConditionalInstanceNorm2dPlus
# ---------------------------------------------------------------------------

class ConditionalInstanceNorm2dPlus(nn.Module):
    """Conditional InstanceNorm++ as in the original NCSN paper (NCHW).

    γ, α (and optionally β) are produced by an embedding lookup on the
    class label y, rather than being global learned scalars.

    Args:
        num_features: Number of input channels.
        num_classes:  Number of conditioning classes.
        bias:         Whether to learn a per-class bias β.
    """

    def __init__(self, num_features: int, num_classes: int = 10,
                 bias: bool = True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias

        if bias:
            # Embedding outputs [γ | α | β], each of size num_features.
            # γ and α initialised near 1; β initialised at 0.
            embed_weight = torch.cat([
                torch.randn(num_classes, 2 * num_features) * 0.02 + 1.0,
                torch.zeros(num_classes, num_features),
            ], dim=1)
        else:
            # Embedding outputs [γ | α], each of size num_features.
            embed_weight = torch.randn(num_classes, 2 * num_features) * 0.02 + 1.0

        self.embed = nn.Embedding(num_classes,
                                  num_features * (3 if bias else 2))
        with torch.no_grad():
            self.embed.weight.copy_(embed_weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W)
            y: (N,) integer class labels
        """
        # Spatial mean per (sample, channel).
        means = x.mean(dim=(2, 3))                         # (N, C)
        m = means.mean(dim=1, keepdim=True)                # (N, 1)
        v = means.var(dim=1, keepdim=True, unbiased=False) # (N, 1)
        means_plus = (means - m) / (v + 1e-5).sqrt()      # (N, C)

        # Standard instance norm.
        variance = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        h = (x - means[:, :, None, None]) / (variance + 1e-5).sqrt()

        # Class-conditional parameters from embedding.
        emb = self.embed(y)   # (N, C*2 or C*3)
        if self.bias:
            gamma, alpha, beta = emb.chunk(3, dim=1)       # each (N, C)
            h = h + means_plus[:, :, None, None] * alpha[:, :, None, None]
            out = gamma[:, :, None, None] * h + beta[:, :, None, None]
        else:
            gamma, alpha = emb.chunk(2, dim=1)             # each (N, C)
            h = h + means_plus[:, :, None, None] * alpha[:, :, None, None]
            out = gamma[:, :, None, None] * h

        return out