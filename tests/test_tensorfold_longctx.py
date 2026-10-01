"""CPU tests for the TensorFold 0.6.1 long-context intake (Flash-Next).

The Metal arithmetic is gated by scripts/check_qwen4_qsa_scores.py (bytes and
argpartition ids against the stock chain) and the full-model A/B; here: the
policy fields (opt-in, receipt-neutral, strict), the diagnostics key, and the
scores admission and its counted fallback.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx2.adapters.flash_next import tensorfold_longctx_diagnostics
from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.runtime.models import qwen4_exp as Q
from mlx2.runtime.models import qwen4_qsa_scores as S
from qsa_oracle import tiny_args

FIELDS = {
    "qsa_fused_scores": "MLX_QWEN4_QSA_FUSED_SCORES",
}


@pytest.fixture(autouse=True)
def _reset():
    S.set_enabled(False)
    S.status(reset=True)
    yield
    S.set_enabled(False)
    S.status(reset=True)


def test_policy_fields_are_opt_in_and_receipt_neutral():
    default = FlashNextPolicy()
    env = default.environment()
    for name, variable in FIELDS.items():
        assert getattr(default, name) is False
        assert name not in default.as_dict()
        assert variable not in env
    assert tensorfold_longctx_diagnostics(default) == {}


@pytest.mark.parametrize("name", sorted(FIELDS))
def test_policy_field_enables_its_switch_only(name):
    policy = FlashNextPolicy.from_mapping({name: True})
    env = policy.environment()
    assert env[FIELDS[name]] == "1"
    for other, variable in FIELDS.items():
        if other != name:
            assert variable not in env
    assert policy.as_dict()[name] is True
    report = tensorfold_longctx_diagnostics(policy)["tensorfold_longctx"]
    assert report["selected"] == [name]
    with pytest.raises(ValueError, match=name):
        FlashNextPolicy.from_mapping({name: 1})


def test_scores_env_flag_parsing():
    assert S._parse_flag(None) is False
    assert S._parse_flag("1") is True
    assert S._parse_flag("off") is False
    with pytest.raises(ValueError):
        S._parse_flag("sometimes")


def _qp(batch=1, rows=1, heads=4, dims=128, blocks=600, dtype=mx.bfloat16):
    return (
        mx.zeros((batch, rows, heads, dims), dtype=dtype),
        mx.zeros((batch, blocks, dims), dtype=dtype),
    )


def test_scores_admission_is_strict(monkeypatch):
    # off the GPU every call is refused before any geometry check
    assert S.supported(*_qp()) == "device"
    monkeypatch.setattr(S.mx, "default_device", lambda: S.mx.gpu)
    monkeypatch.setattr(S.mx.metal, "is_available", lambda: True)
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    assert S.supported(*_qp()) is None
    assert S.supported(*_qp(rows=8)) is None
    assert S.supported(*_qp(rows=9)) == "rows"  # 36 matrix rows > 32
    assert S.supported(*_qp(dims=64)) == "head_dim"
    assert S.supported(*_qp(blocks=128)) == "blocks"  # N <= K: not the steel GEMM
    assert S.supported(*_qp(dtype=mx.int32)) == "dtype"
    assert S.supported(*_qp(batch=4)) is None
    q, p = _qp(batch=2)
    assert S.supported(q, p[:1]) == "batch"
    monkeypatch.setenv("MLX_ENABLE_TF32", "1")
    assert S.supported(*_qp()) == "tf32"  # the stock GEMM would be NAX/TF32


def test_scores_offset_forms():
    assert S.offset_supported(4096, 1) is None
    assert S.offset_supported(4096, 2) == "offset"
    assert S.offset_supported(mx.array([7, 9], dtype=mx.int32), 2) is None
    assert S.offset_supported(mx.array([7], dtype=mx.int32), 2) == "offset"
    assert S.offset_supported(mx.array([7.0]), 1) == "offset"
    assert S.offset_supported(None, 1) == "offset"


def test_stock_scores_is_the_indexer_chain():
    rng = np.random.default_rng(3)
    q = mx.array(rng.normal(size=(1, 2, 4, 128)).astype(np.float32)).astype(mx.bfloat16)
    pooled = mx.array(rng.normal(size=(1, 200, 128)).astype(np.float32)).astype(mx.bfloat16)
    q_pos = mx.arange(798, 800)[None, :]
    valid = (mx.arange(200) * 4 + 3)[None, None, :] <= q_pos[..., None]
    got = np.array(S.stock_scores(q, pooled, valid, 128))
    ref = np.einsum(
        "lhd,nd->lnh",
        np.array(q.astype(mx.float32))[0],
        np.array(pooled.astype(mx.float32))[0],
    )
    ref = np.maximum(ref, 0).sum(-1) / np.float32(np.sqrt(128))
    ref = np.where(np.array(valid)[0], ref, -np.inf)
    np.testing.assert_allclose(got[0], ref, rtol=1e-5, atol=1e-5)
    assert np.isneginf(got[0, 0, 199]) and np.isfinite(got[0, 1, 199])


def _indexer_selection(enabled):
    S.set_enabled(enabled)
    args = tiny_args()
    mx.random.seed(11)
    indexer = Q.QSAIndexer(args, 0)
    hidden = mx.random.normal((1, 12, args.hidden_size))
    return indexer(hidden, None, None)


def test_indexer_counts_the_fallback_and_keeps_the_stock_selection():
    stock = _indexer_selection(False)
    assert S.status()["counts"] == {}
    fused = _indexer_selection(True)
    assert stock.kind == fused.kind == "explicit"
    assert np.array_equal(np.array(stock.raw_block_ids), np.array(fused.raw_block_ids))
    # CPU: refused and counted, the MLX chain ran
    assert S.status()["counts"] == {"fallback_device": 1}
