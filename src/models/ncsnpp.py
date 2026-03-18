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

"""NCSN++ model. Converted from JAX/Flax to PyTorch.

Layout note: all feature tensors are NCHW throughout.
  - Skip-connection concatenation is on dim=1 (channel axis).
  - Attention-resolution checks use h.shape[2] (H) instead of h.shape[1].
  - GroupNorm num_groups = min(C // 4, 32) everywhere.
"""

import math
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ml_collections

from src.models import layers, layerspp, normalization

ResnetBlockDDPM  = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine          = layerspp.Combine
conv3x3          = layerspp.conv3x3
conv1x1          = layerspp.conv1x1
get_act          = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


def _num_groups(channels: int) -> int:
    return min(channels // 4, 32)


def _variance_scaling_init(tensor: torch.Tensor, scale: float = 1.0):
    scale = max(scale, 1e-10)
    fan_in  = tensor[0].numel()
    fan_out = tensor.shape[0] * tensor[0][0].numel() if tensor.ndim >= 2 else fan_in
    limit   = math.sqrt(3 * scale / ((fan_in + fan_out) / 2.0))
    nn.init.uniform_(tensor, -limit, limit)


# ---------------------------------------------------------------------------
# NCSNpp
# ---------------------------------------------------------------------------

class NCSNpp(nn.Module):
    """NCSN++ model (PyTorch, NCHW)."""

    def __init__(self, config: ml_collections.ConfigDict):
        super().__init__()
        self.config = config
        self.act = get_act(config)

        # ---- unpack config ----
        nf               = config.model.nf
        ch_mult          = config.model.ch_mult
        num_res_blocks   = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout          = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv
        num_resolutions  = len(ch_mult)

        conditional       = config.model.conditional
        fir               = config.model.fir
        fir_kernel        = config.model.fir_kernel
        skip_rescale      = config.model.skip_rescale
        resblock_type     = config.model.resblock_type.lower()
        progressive       = config.model.progressive.lower()
        progressive_input = config.model.progressive_input.lower()
        embedding_type    = config.model.embedding_type.lower()
        init_scale        = config.model.init_scale
        combine_method    = config.model.progressive_combine.lower()

        assert progressive       in ["none", "output_skip", "residual"]
        assert progressive_input in ["none", "input_skip", "residual"]
        assert embedding_type    in ["fourier", "positional"]

        self.conditional       = conditional
        self.progressive       = progressive
        self.progressive_input = progressive_input
        self.embedding_type    = embedding_type
        self.skip_rescale      = skip_rescale
        self.resblock_type     = resblock_type
        self.num_resolutions   = num_resolutions
        self.num_res_blocks    = num_res_blocks
        self.attn_resolutions  = attn_resolutions
        self.nf                = nf
        self.ch_mult           = ch_mult
        self.init_scale        = init_scale
        self.double_heads      = config.model.double_heads

        # ---- time embedding ----
        if embedding_type == "fourier":
            self.time_embed = layerspp.GaussianFourierProjection(
                embedding_size=nf, scale=config.model.fourier_scale
            )
            temb_channels = nf * 2
        else:  # positional
            self.time_embed = None   # get_timestep_embedding is a function
            temb_channels = nf

        if conditional:
            self.temb_dense0 = nn.Linear(temb_channels, nf * 4)
            self.temb_dense1 = nn.Linear(nf * 4, nf * 4)
            _variance_scaling_init(self.temb_dense0.weight)
            _variance_scaling_init(self.temb_dense1.weight)
            nn.init.zeros_(self.temb_dense0.bias)
            nn.init.zeros_(self.temb_dense1.bias)
            temb_out = nf * 4
        else:
            self.temb_dense0 = None
            self.temb_dense1 = None
            temb_out = None

        # ---- block factories (captured as instance state) ----
        def make_attn(in_ch):
            return layerspp.AttnBlockpp(
                channels=in_ch, init_scale=init_scale, skip_rescale=skip_rescale
            )

        def make_upsample(in_ch, out_ch=None):
            return layerspp.Upsample(
                in_ch=in_ch, out_ch=out_ch,
                with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel,
            )

        def make_pyramid_upsample(in_ch, out_ch=None):
            with_conv = (progressive == "residual")
            return layerspp.Upsample(
                in_ch=in_ch, out_ch=out_ch,
                with_conv=with_conv, fir=fir, fir_kernel=fir_kernel,
            )

        def make_downsample(in_ch, out_ch=None):
            return layerspp.Downsample(
                in_ch=in_ch, out_ch=out_ch,
                with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel,
            )

        def make_pyramid_downsample(in_ch, out_ch=None):
            with_conv = (progressive_input == "residual")
            return layerspp.Downsample(
                in_ch=in_ch, out_ch=out_ch,
                with_conv=with_conv, fir=fir, fir_kernel=fir_kernel,
            )

        def make_resblock_ddpm(in_ch, out_ch=None):
            return ResnetBlockDDPM(
                in_ch=in_ch, act=self.act, temb_dim=temb_out,
                out_ch=out_ch, dropout=dropout,
                init_scale=init_scale, skip_rescale=skip_rescale,
            )

        def make_resblock_biggan(in_ch, out_ch=None, up=False, down=False):
            return ResnetBlockBigGAN(
                in_ch=in_ch, act=self.act, temb_dim=temb_out,
                out_ch=out_ch, up=up, down=down,
                dropout=dropout, fir=fir, fir_kernel=fir_kernel,
                init_scale=init_scale, skip_rescale=skip_rescale,
            )

        # ---- build encoder (downsampling path) ----
        # We need to track channel widths to build all submodules upfront.
        self.input_conv = conv3x3(config.model.input_channels, nf)

        # Track channel sizes as we "simulate" a forward pass for graph building.
        # hs_channels[i] is the channel count of the i-th entry pushed onto hs.
        hs_ch: list[int] = [nf]

        self.down_blocks      = nn.ModuleList()
        self.down_attn        = nn.ModuleList()   # None entries handled via list
        self.down_sample      = nn.ModuleList()
        self.down_pyramid_ds  = nn.ModuleList()
        self.down_combiners   = nn.ModuleList()

        cur_ch = nf
        input_pyramid_ch = config.model.input_channels  # channels of x

        for i_level in range(num_resolutions):
            level_res_blocks = nn.ModuleList()
            level_attn       = nn.ModuleList()
            target_ch = nf * ch_mult[i_level]

            for i_block in range(num_res_blocks):
                if resblock_type == "ddpm":
                    rb = make_resblock_ddpm(cur_ch, out_ch=target_ch)
                else:
                    rb = make_resblock_biggan(cur_ch, out_ch=target_ch)
                level_res_blocks.append(rb)
                cur_ch = target_ch

                if cur_ch in attn_resolutions:
                    level_attn.append(make_attn(cur_ch))
                else:
                    level_attn.append(nn.Identity())

                hs_ch.append(cur_ch)

            self.down_blocks.append(level_res_blocks)
            self.down_attn.append(level_attn)

            if i_level != num_resolutions - 1:
                # Downsample
                if resblock_type == "ddpm":
                    self.down_sample.append(make_downsample(cur_ch))
                else:
                    self.down_sample.append(
                        make_resblock_biggan(cur_ch, down=True)
                    )

                # Progressive input
                if progressive_input == "input_skip":
                    pds = make_pyramid_downsample(input_pyramid_ch)
                    self.down_pyramid_ds.append(pds)
                    # combiner merges input_pyramid (input_pyramid_ch) with h (cur_ch)
                    self.down_combiners.append(
                        Combine(input_pyramid_ch, cur_ch, method=combine_method)
                    )
                    # After combine("cat"), channel count doubles
                    if combine_method == "cat":
                        cur_ch = cur_ch + cur_ch  # pyramid projected to cur_ch then cat
                        # Actually Combine projects input_pyramid to y.shape[1]=cur_ch,
                        # then cats → 2*cur_ch. Let's track correctly:
                        # conv1x1(input_pyramid_ch → cur_ch) then cat with h(cur_ch)
                        cur_ch = cur_ch  # stays cur_ch after combiner projects+cats?
                        # Re-check: Combine.forward: h=conv(x), cat([h,y]) along dim=1
                        # x=input_pyramid (input_pyramid_ch), y=h (cur_ch)
                        # conv projects x → cur_ch, then cat → 2*cur_ch
                        cur_ch = cur_ch * 2

                elif progressive_input == "residual":
                    pds = make_pyramid_downsample(input_pyramid_ch, out_ch=cur_ch)
                    self.down_pyramid_ds.append(pds)
                    self.down_combiners.append(nn.Identity())  # placeholder
                    # input_pyramid_ch matches cur_ch after this
                    input_pyramid_ch = cur_ch
                else:
                    self.down_pyramid_ds.append(nn.Identity())
                    self.down_combiners.append(nn.Identity())

                hs_ch.append(cur_ch)

        # ---- middle ----
        self.mid_block1 = (make_resblock_ddpm(cur_ch)
                           if resblock_type == "ddpm"
                           else make_resblock_biggan(cur_ch))
        self.mid_attn   = make_attn(cur_ch)
        self.mid_block2 = (make_resblock_ddpm(cur_ch)
                           if resblock_type == "ddpm"
                           else make_resblock_biggan(cur_ch))

        # ---- build decoder (upsampling path) ----
        self.up_blocks     = nn.ModuleList()
        self.up_attn       = nn.ModuleList()
        self.up_sample     = nn.ModuleList()
        # Progressive output modules
        self.up_pyramid_us   = nn.ModuleList()
        self.up_pyramid_conv = nn.ModuleList()

        pyramid_ch = None   # track channels of the progressive pyramid tensor

        for i_level in reversed(range(num_resolutions)):
            target_ch = nf * ch_mult[i_level]
            level_res_blocks = nn.ModuleList()
            level_attn       = nn.ModuleList()

            for i_block in range(num_res_blocks + 1):
                skip_ch = hs_ch.pop()
                in_ch   = cur_ch + skip_ch   # concat along channel dim
                if resblock_type == "ddpm":
                    rb = make_resblock_ddpm(in_ch, out_ch=target_ch)
                else:
                    rb = make_resblock_biggan(in_ch, out_ch=target_ch)
                level_res_blocks.append(rb)
                cur_ch = target_ch

            if cur_ch in attn_resolutions:
                level_attn.append(make_attn(cur_ch))
            else:
                level_attn.append(nn.Identity())

            self.up_blocks.append(level_res_blocks)
            self.up_attn.append(level_attn)

            # Progressive output
            if progressive != "none":
                if i_level == num_resolutions - 1:
                    if progressive == "output_skip":
                        self.up_pyramid_us.append(nn.Identity())
                        self.up_pyramid_conv.append(
                            conv3x3(cur_ch, config.model.input_channels,
                                    init_scale=init_scale)
                        )
                        pyramid_ch = config.model.input_channels
                    else:  # residual
                        self.up_pyramid_us.append(nn.Identity())
                        self.up_pyramid_conv.append(conv3x3(cur_ch, cur_ch))
                        pyramid_ch = cur_ch
                else:
                    if progressive == "output_skip":
                        self.up_pyramid_us.append(
                            make_pyramid_upsample(pyramid_ch)
                        )
                        self.up_pyramid_conv.append(
                            conv3x3(cur_ch, config.model.input_channels,
                                    init_scale=init_scale)
                        )
                        pyramid_ch = config.model.input_channels
                    else:  # residual
                        self.up_pyramid_us.append(
                            make_pyramid_upsample(pyramid_ch, out_ch=cur_ch)
                        )
                        self.up_pyramid_conv.append(nn.Identity())
                        pyramid_ch = cur_ch
            else:
                self.up_pyramid_us.append(nn.Identity())
                self.up_pyramid_conv.append(nn.Identity())

            if i_level != 0:
                if resblock_type == "ddpm":
                    self.up_sample.append(make_upsample(cur_ch))
                else:
                    self.up_sample.append(
                        make_resblock_biggan(cur_ch, up=True)
                    )
            else:
                self.up_sample.append(nn.Identity())

        assert not hs_ch, f"hs_ch not fully consumed: {hs_ch}"

        # ---- output head ----
        out_ch = config.model.input_channels
        if progressive != "output_skip":
            self.out_norm = nn.GroupNorm(_num_groups(cur_ch), cur_ch)
            if self.double_heads:
                self.out_conv = conv3x3(cur_ch, out_ch * 2, init_scale=init_scale)
            else:
                self.out_conv = conv3x3(cur_ch, out_ch, init_scale=init_scale)
        else:
            self.out_norm = None
            self.out_conv = None

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, time_cond: torch.Tensor,
                train: bool = True) -> torch.Tensor:
        act = self.act

        # Time embedding
        if self.embedding_type == "fourier":
            temb = self.time_embed(time_cond)
        else:
            temb = layers.get_timestep_embedding(time_cond, self.nf)

        if self.conditional:
            temb = self.temb_dense0(temb)
            temb = self.temb_dense1(act(temb))
        else:
            temb = None

        # ---- encoder ----
        input_pyramid = x if self.progressive_input != "none" else None
        input_pyramid_idx = 0   # index into down_pyramid_ds / down_combiners

        hs = [self.input_conv(x)]

        for i_level in range(self.num_resolutions):
            res_blocks = self.down_blocks[i_level]
            attn_mods  = self.down_attn[i_level]

            for i_block in range(self.num_res_blocks):
                h = res_blocks[i_block](hs[-1], temb, train)
                h = attn_mods[i_block](h)
                hs.append(h)

            if i_level != self.num_resolutions - 1:
                ds_mod = self.down_sample[i_level]
                if self.resblock_type == "ddpm":
                    h = ds_mod(hs[-1])
                else:
                    h = ds_mod(hs[-1], temb, train)

                if self.progressive_input == "input_skip":
                    input_pyramid = self.down_pyramid_ds[input_pyramid_idx](input_pyramid)
                    h = self.down_combiners[input_pyramid_idx](input_pyramid, h)
                    input_pyramid_idx += 1
                elif self.progressive_input == "residual":
                    input_pyramid = self.down_pyramid_ds[input_pyramid_idx](input_pyramid)
                    input_pyramid_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / math.sqrt(2.0)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid

                hs.append(h)

        # ---- middle ----
        h = hs[-1]
        h = self.mid_block1(h, temb, train)
        h = self.mid_attn(h)
        h = self.mid_block2(h, temb, train)

        pyramid = None
        up_idx  = 0   # index into up_pyramid_us / up_pyramid_conv / up_sample

        # ---- decoder ----
        # up_blocks are stored in reversed(range(num_resolutions)) order.
        for i_level_rev, (res_blocks, attn_mods) in enumerate(
            zip(self.up_blocks, self.up_attn)
        ):
            i_level = self.num_resolutions - 1 - i_level_rev

            for i_block in range(self.num_res_blocks + 1):
                h = res_blocks[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb, train
                )

            h = attn_mods[0](h)

            # Progressive output
            if self.progressive != "none":
                pyr_us   = self.up_pyramid_us[i_level_rev]
                pyr_conv = self.up_pyramid_conv[i_level_rev]

                if i_level == self.num_resolutions - 1:
                    # First decoder level — initialise pyramid.
                    norm_h = nn.functional.group_norm(
                        h, _num_groups(h.shape[1])
                    ) if self.progressive == "output_skip" else h
                    if self.progressive == "output_skip":
                        norm_h = act(F.group_norm(h, _num_groups(h.shape[1])))
                        pyramid = pyr_conv(norm_h)
                    else:  # residual
                        norm_h = act(F.group_norm(h, _num_groups(h.shape[1])))
                        pyramid = pyr_conv(norm_h)
                else:
                    if self.progressive == "output_skip":
                        pyramid = pyr_us(pyramid)
                        norm_h  = act(F.group_norm(h, _num_groups(h.shape[1])))
                        pyramid = pyramid + pyr_conv(norm_h)
                    else:  # residual
                        pyramid = pyr_us(pyramid)
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / math.sqrt(2.0)
                        else:
                            pyramid = pyramid + h
                        h = pyramid

            if i_level != 0:
                up_mod = self.up_sample[i_level_rev]
                if self.resblock_type == "ddpm":
                    h = up_mod(h)
                else:
                    h = up_mod(h, temb, train)

        assert not hs, "Skip-connection stack not fully consumed."

        # ---- output ----
        if self.progressive == "output_skip" and not self.double_heads:
            return pyramid
        else:
            h = act(self.out_norm(h))
            return self.out_conv(h)


# ---------------------------------------------------------------------------
# JointNCSNpp
# ---------------------------------------------------------------------------

class JointNCSNpp(nn.Module):
    """Joint NCSN++ model — two independent NCSNpp heads sharing the config."""

    def __init__(self, config: ml_collections.ConfigDict):
        super().__init__()
        self.denoiser  = NCSNpp(config)
        self.distiller = NCSNpp(config)

    def forward(self, x: torch.Tensor, time_cond: torch.Tensor,
                train: bool = True):
        return (
            self.denoiser(x, time_cond, train),
            self.distiller(x, time_cond, train),
        )