"""HC source-contract regressions, including intermediate rounding seams."""
import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.runtime.models import qwen_hc_prefill as hc


@pytest.mark.parametrize("hidden", [2560, 5120])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_combine_publishes_dtype_before_grouped_norm(hidden, dtype):
    mx.random.seed(22)
    x = mx.random.normal((1, 9, 4 * hidden)).astype(dtype)
    branch = (mx.random.normal((1, 9, hidden)) * .07).astype(dtype)
    inject = mx.ones((1, 9, 4), dtype=dtype)
    weight = mx.ones((4 * hidden,), dtype=mx.float32)
    combined = hc.eager_combine(x, branch, inject)
    got = hc.eager_norm(combined, weight, hidden, 1e-6)
    rounded = np.array(combined.astype(mx.float32)).reshape(1, 9, 4, hidden)
    expected = rounded / np.sqrt(np.mean(rounded ** 2, axis=-1, keepdims=True) + 1e-6)
    np.testing.assert_allclose(np.array(got.astype(mx.float32)),
        np.array(mx.array(expected.reshape(x.shape)).astype(dtype).astype(mx.float32)), atol=.04, rtol=.01)
    unrounded = np.array(x.astype(mx.float32)).reshape(1, 9, 4, hidden) + np.array(branch.astype(mx.float32))[..., None, :]
    assert np.count_nonzero(rounded != unrounded) > hidden
    assert hc.refusal(x, weight, hidden, 1e-6) is None


def test_prefill_admission_and_policy_receipt(monkeypatch):
    policy = FlashNextPolicy(hc_prefill_kernels=True, gdn_prefill_chunk=32)
    assert policy.environment()[hc.ENV] == "1"
    assert policy.as_dict()["hc_prefill_kernels"] is True
    assert "hc_prefill_kernels" not in FlashNextPolicy().as_dict()
    x = mx.zeros((1, 8, 10240), dtype=mx.bfloat16)
    assert hc.refusal(x, mx.ones((10240,)), 2560, 1e-6) == "prefill_rows"
    monkeypatch.setenv(hc.ENV, "1")
    assert hc.try_norm(x, mx.ones((10240,)), 2560, 1e-6) is None


def test_combined_norm_is_consumed_at_actual_decoder_seam(monkeypatch):
    from types import SimpleNamespace

    from mlx2.runtime.models.qwen4_exp import DecoderLayer
    events = []
    x = mx.zeros((1, 9, 4), dtype=mx.bfloat16)
    branch = mx.ones((1, 9, 1), dtype=mx.bfloat16)
    inject = mx.ones((1, 9, 4), dtype=mx.bfloat16)
    normalized = x + 2
    class Connection:
        hidden_size = 1
        hc_norm = SimpleNamespace(weight=mx.ones((4,)), eps=1e-6)
        def __call__(self, value):
            return branch, value, inject
        def _composed(self, value, glue, raw, *, normed):
            assert normed is normalized
            events.append("used_combined_norm")
            return branch, value, inject
    layer = SimpleNamespace(ple=None, attn_hyper_connection=Connection(),
        mlp_hyper_connection=Connection(), is_linear=True,
        linear_attn=lambda *a: branch, mlp=lambda *a: branch, training=False)
    monkeypatch.setattr(hc, "try_combine_norm", lambda *a: (x, normalized))
    with hc.scope(True):
        DecoderLayer._forward(layer, x, mx.zeros((1, 9), dtype=mx.int32))
    assert events == ["used_combined_norm"]


def test_speculative_windows_never_enter_hc_prefill_scope():
    from types import SimpleNamespace

    from mlx2.runtime.models.qwen4_exp import DecoderLayer
    layer = SimpleNamespace(training=False, _forward=lambda *a: hc.active())
    x = mx.zeros((1, 16, 10240), dtype=mx.bfloat16)
    assert not DecoderLayer.__call__(layer, x, None, cache=SimpleNamespace(speculating=True))
    assert DecoderLayer.__call__(layer, x, None, cache=SimpleNamespace(speculating=False))
    assert not hc.active()


def test_hc_arithmetic_gets_a_distinct_apcv2_identity():
    from mlx2.runtime.prefill_plan import apc_prefill_fingerprint, execution_identity
    identity = execution_identity(hc={"schema": "mlx2.hc-prefill.v1", "law": "rounded-combine-f32-norm-storage-gate-mix"})
    assert apc_prefill_fingerprint("reference", identity) != "reference"
