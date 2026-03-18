# tests/test_ncsnpp.py
"""Small-scale pytest tests for NCSNpp and JointNCSNpp.

All configs use tiny dimensions so tests run on CPU in seconds.
"""

import pytest
import torch
import ml_collections

from src.models.ncsnpp import NCSNpp, JointNCSNpp


# ---------------------------------------------------------------------------
# Config factories
# ---------------------------------------------------------------------------

def _base_config():
    """Minimal config shared by all tests."""
    c = ml_collections.ConfigDict()
    c.model = ml_collections.ConfigDict()

    # Architecture
    c.model.nf = 8                        # base channel count (tiny)
    c.model.ch_mult = (1, 2)             # 2 resolutions
    c.model.num_res_blocks = 1
    c.model.attn_resolutions = (4,)      # attention at 4×4 spatial
    c.model.dropout = 0.0
    c.model.resamp_with_conv = True
    c.model.input_channels = 1           # grayscale

    # Conditioning
    c.model.conditional = True
    c.model.embedding_type = "positional"
    c.model.fourier_scale = 16.0         # only used if embedding_type=="fourier"
    c.model.nonlinearity = "swish"

    # Normalisation / init
    c.model.normalization = "GroupNorm"
    c.model.init_scale = 0.0
    c.model.skip_rescale = True

    # ResBlock type
    c.model.resblock_type = "biggan"
    c.model.fir = False
    c.model.fir_kernel = (1, 3, 3, 1)

    # Progressive connections
    c.model.progressive = "none"
    c.model.progressive_input = "none"
    c.model.progressive_combine = "sum"

    c.model.double_heads = False

    return c


def _fourier_config():
    c = _base_config()
    c.model.embedding_type = "fourier"
    return c


def _ddpm_resblock_config():
    c = _base_config()
    c.model.resblock_type = "ddpm"
    return c


def _fir_config():
    c = _base_config()
    c.model.fir = True
    return c


def _output_skip_config():
    c = _base_config()
    c.model.progressive = "output_skip"
    return c


def _progressive_input_skip_config():
    c = _base_config()
    c.model.progressive_input = "input_skip"
    c.model.progressive_combine = "cat"
    return c


def _progressive_residual_config():
    c = _base_config()
    c.model.progressive = "residual"
    c.model.progressive_input = "residual"
    return c


def _double_heads_config():
    c = _base_config()
    c.model.double_heads = True
    return c


def _unconditional_config():
    c = _base_config()
    c.model.conditional = False
    return c


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH   = 2
C_IN    = 1
H = W   = 8   # small spatial size; 8→4 with ch_mult=(1,2)

@pytest.fixture
def toy_batch():
    x         = torch.randn(BATCH, C_IN, H, W)
    time_cond = torch.rand(BATCH)   # continuous noise level in [0, 1)
    return x, time_cond


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run(config, x, time_cond, train=True):
    model = NCSNpp(config)
    model.eval() if not train else model.train()
    with torch.no_grad() if not train else torch.enable_grad():
        return model(x, time_cond, train=train)


# ---------------------------------------------------------------------------
# NCSNpp tests
# ---------------------------------------------------------------------------

class TestNCSNppOutputShape:

    def test_default_output_shape(self, toy_batch):
        x, t = toy_batch
        out = _run(_base_config(), x, t)
        assert out.shape == (BATCH, C_IN, H, W)

    def test_fourier_embedding(self, toy_batch):
        x, t = toy_batch
        out = _run(_fourier_config(), x, t)
        assert out.shape == (BATCH, C_IN, H, W)

    def test_ddpm_resblock(self, toy_batch):
        x, t = toy_batch
        out = _run(_ddpm_resblock_config(), x, t)
        assert out.shape == (BATCH, C_IN, H, W)

    def test_fir_resampling(self, toy_batch):
        x, t = toy_batch
        out = _run(_fir_config(), x, t)
        assert out.shape == (BATCH, C_IN, H, W)

    def test_output_skip_progressive(self, toy_batch):
        x, t = toy_batch
        out = _run(_output_skip_config(), x, t)
        assert out.shape == (BATCH, C_IN, H, W)

    def test_progressive_input_skip(self, toy_batch):
        x, t = toy_batch
        out = _run(_progressive_input_skip_config(), x, t)
        assert out.shape == (BATCH, C_IN, H, W)

    def test_progressive_residual(self, toy_batch):
        x, t = toy_batch
        out = _run(_progressive_residual_config(), x, t)
        assert out.shape == (BATCH, C_IN, H, W)

    def test_double_heads_output_shape(self, toy_batch):
        x, t = toy_batch
        out = _run(_double_heads_config(), x, t)
        # double_heads → output channels = input_channels * 2
        assert out.shape == (BATCH, C_IN * 2, H, W)

    def test_unconditional(self, toy_batch):
        x, t = toy_batch
        out = _run(_unconditional_config(), x, t)
        assert out.shape == (BATCH, C_IN, H, W)


class TestNCSNppDtype:

    def test_output_is_float32(self, toy_batch):
        x, t = toy_batch
        out = _run(_base_config(), x, t)
        assert out.dtype == torch.float32

    def test_float16_input(self, toy_batch):
        """Model stays float32 internally; input cast should not crash."""
        x, t = toy_batch
        model = NCSNpp(_base_config())
        # Cast model to float16
        model = model.half()
        x_half = x.half()
        t_half = t.half()
        with torch.no_grad():
            out = model(x_half, t_half, train=False)
        assert out.dtype == torch.float16


class TestNCSNppGradients:

    def test_gradients_flow(self, toy_batch):
        x, t = toy_batch
        x = x.requires_grad_(True)
        model = NCSNpp(_base_config())
        out = model(x, t, train=True)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_no_nan_in_output(self, toy_batch):
        x, t = toy_batch
        out = _run(_base_config(), x, t)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestNCSNppTrainEvalMode:

    def test_train_mode(self, toy_batch):
        x, t = toy_batch
        model = NCSNpp(_base_config())
        model.train()
        out = model(x, t, train=True)
        assert out.shape == (BATCH, C_IN, H, W)

    def test_eval_mode(self, toy_batch):
        x, t = toy_batch
        model = NCSNpp(_base_config())
        model.eval()
        with torch.no_grad():
            out = model(x, t, train=False)
        assert out.shape == (BATCH, C_IN, H, W)

    def test_eval_is_deterministic(self, toy_batch):
        """In eval mode with dropout=0, two forward passes must be identical."""
        x, t = toy_batch
        model = NCSNpp(_base_config())
        model.eval()
        with torch.no_grad():
            out1 = model(x, t, train=False)
            out2 = model(x, t, train=False)
        assert torch.allclose(out1, out2)


class TestNCSNppBatchSize:

    @pytest.mark.parametrize("batch_size", [1, 3, 4])
    def test_various_batch_sizes(self, batch_size):
        x = torch.randn(batch_size, C_IN, H, W)
        t = torch.rand(batch_size)
        out = _run(_base_config(), x, t)
        assert out.shape == (batch_size, C_IN, H, W)


# ---------------------------------------------------------------------------
# JointNCSNpp tests
# ---------------------------------------------------------------------------

class TestJointNCSNpp:

    def test_returns_two_tensors(self, toy_batch):
        x, t = toy_batch
        model = JointNCSNpp(_base_config())
        with torch.no_grad():
            denoised, distilled = model(x, t, train=False)
        assert denoised.shape  == (BATCH, C_IN, H, W)
        assert distilled.shape == (BATCH, C_IN, H, W)

    def test_heads_are_independent(self, toy_batch):
        """The two heads should produce different outputs (separate weights)."""
        x, t = toy_batch
        model = JointNCSNpp(_base_config())
        with torch.no_grad():
            denoised, distilled = model(x, t, train=False)
        # With randomly initialised weights the outputs should differ.
        assert not torch.allclose(denoised, distilled)

    def test_gradients_both_heads(self, toy_batch):
        x, t = toy_batch
        x = x.requires_grad_(True)
        model = JointNCSNpp(_base_config())
        d, s = model(x, t, train=True)
        (d.sum() + s.sum()).backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_no_nan_joint(self, toy_batch):
        x, t = toy_batch
        model = JointNCSNpp(_base_config())
        with torch.no_grad():
            d, s = model(x, t, train=False)
        for out in (d, s):
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()