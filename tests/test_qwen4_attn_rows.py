"""CPU tests for the fused Qwen4 attention rows (omlx #4052 port).

The Metal kernels themselves are gated by scripts/check_qwen4_attn_rows.py
(bit-exactness on real Flash-Next layers); here: the policy switch, the
transcribed SDPA plan table, admission and fallback accounting, and the exact
dense short-circuit the fused rows use below the indexer budget.
"""

import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.runtime.models import qwen4_attn_rows as R
from mlx2.runtime.models import qwen4_exp as Q
from qsa_oracle import tiny_args


@pytest.fixture(autouse=True)
def _reset():
    R.set_enabled(False)
    R.status(reset=True)
    yield
    R.set_enabled(False)
    R.status(reset=True)


def test_policy_default_on_and_off_is_absent_from_receipts():
    default = FlashNextPolicy()
    assert default.attn_fused_rows is True
    assert default.as_dict()["attn_fused_rows"] is True
    assert default.environment()["MLX_QWEN4_ATTN_FUSED_ROWS"] == "1"
    off = FlashNextPolicy.from_mapping({"attn_fused_rows": False})
    assert "attn_fused_rows" not in off.as_dict()
    assert "MLX_QWEN4_ATTN_FUSED_ROWS" not in off.environment()
    with pytest.raises(ValueError, match="attn_fused_rows"):
        FlashNextPolicy.from_mapping({"attn_fused_rows": 1})


def test_env_flag_parsing():
    assert R._parse_flag(None) is False
    assert R._parse_flag("1") is True
    assert R._parse_flag("off") is False
    with pytest.raises(ValueError):
        R._parse_flag("maybe")


@pytest.mark.parametrize(
    "n, rows, expected",
    [
        (1, 1, (1, None)),
        (1023, 1, (1, None)),
        (1024, 1, (2, 64)),  # N > 1024 is strict in select_sdpa_blocks
        (1025, 1, (2, 128)),
        (8192, 1, (2, 128)),
        (8193, 1, (2, 256)),
        (32768, 1, (2, 256)),
        (32769, 1, (2, 512)),
        (65536, 1, (2, 512)),
        (65537, 1, (2, 1024)),
        (262144, 1, (2, 1024)),
        (1023, 2, (1, None)),
        (40000, 2, (2, 512)),
        (40000, 3, None),  # 3 x GQA 12 > 32: MLX's unfused fallback
        (2, 3, None),
    ],
)
def test_sdpa_plan_m5_max(monkeypatch, n, rows, expected):
    monkeypatch.setattr(R, "_device_class", lambda: "s")
    monkeypatch.delenv("MLX_SDPA_BLOCKS", raising=False)
    assert R.sdpa_plan(n, rows, 24, 2, 256) == expected


def test_sdpa_plan_ultra_and_refusals(monkeypatch):
    monkeypatch.delenv("MLX_SDPA_BLOCKS", raising=False)
    monkeypatch.setattr(R, "_device_class", lambda: "d")
    assert R.sdpa_plan(1024, 1, 24, 2, 256) == (2, 128)
    assert R.sdpa_plan(16384, 1, 24, 2, 256) == (2, 512)
    assert R.sdpa_plan(65536, 1, 24, 2, 256) == (2, 1024)
    assert R.sdpa_plan(4096, 1, 24, 2, 128) is None  # head dim not transcribed
    monkeypatch.setenv("MLX_SDPA_BLOCKS", "96")
    assert R.sdpa_plan(4096, 1, 24, 2, 256) is None
    monkeypatch.delenv("MLX_SDPA_BLOCKS")
    monkeypatch.setattr(R, "_device_class", lambda: "g")
    assert R.sdpa_plan(4096, 1, 24, 2, 256) is None


def test_log2f_matches_float32():
    assert R._log2f(1e7) == pytest.approx(float(np.log2(np.float32(1e7))), abs=2e-6)


def _attention(**overrides):
    args = tiny_args(**overrides)
    attn = Q.Attention(args, 3)
    attn.eval()
    mx.eval(attn.parameters())
    return args, attn


def test_cpu_keeps_mlx_path_and_counts_the_refusal():
    """On the CPU nothing is admitted: identical outputs, a counted fallback."""
    args, attn = _attention()
    attn.set_dtype(mx.bfloat16)
    rng = np.random.default_rng(0)
    outs = {}
    for arm in ("off", "on"):
        R.set_enabled(arm == "on")
        cache = Q.QSAKVCache(attn.indexer.summary_identity)
        x = mx.array(rng.standard_normal((1, 6, args.hidden_size)).astype(np.float32)).astype(mx.bfloat16)
        prefill = attn(x, None, cache)
        steps = []
        for _ in range(3):
            step = mx.ones((1, 1, args.hidden_size), dtype=mx.bfloat16)
            steps.append(attn(step, None, cache))
        outs[arm] = mx.concatenate([prefill] + steps, axis=1)
        rng = np.random.default_rng(0)
    assert mx.array_equal(outs["off"], outs["on"]).item()
    counts = R.status()["counts"]
    assert counts.get("fallback_device", 0) >= 3
    assert not any(k.startswith(("prep_rows", "sdpa_")) for k in counts)


def test_disabled_counts_nothing():
    args, attn = _attention()
    cache = Q.QSAKVCache(attn.indexer.summary_identity)
    attn(mx.ones((1, 2, args.hidden_size)), None, cache)
    assert R.status()["counts"] == {}


def test_dense_shortcircuit_mask_is_exact():
    """Below the budget the explicit selection's mask equals the causal mask
    cell for cell, so the fused rows may skip the selection."""
    from mlx2.runtime.models.base import create_causal_mask

    args, attn = _attention(indexer_budget=64)
    indexer = attn.indexer
    rng = np.random.default_rng(3)
    hidden = mx.array(rng.standard_normal((1, 21, args.hidden_size)).astype(np.float32))
    caches = []
    for _ in range(2):
        cache = Q.QSAKVCache(indexer.summary_identity)
        attn(hidden, None, cache)
        caches.append(cache)
    for width in (1, 2, 3):
        x = mx.array(rng.standard_normal((1, width, args.hidden_size)).astype(np.float32))
        offset = caches[0].offset
        causal = create_causal_mask(width, offset=offset)[None, None]
        explicit = indexer(x, causal, caches[0])
        short = indexer(x, causal, caches[1], dense_shortcircuit=True)
        assert explicit.kind == "explicit"
        assert short.kind == "implicit_all"
        assert mx.array_equal(explicit.dense_mask(), causal).item()
        assert mx.array_equal(short.dense_mask(), causal).item()
        assert mx.array_equal(caches[0].index_keys, caches[1].index_keys).item()
        for cache in caches:
            cache.update_and_fetch(
                mx.zeros((1, attn.num_kv_heads, width, attn.head_dim)),
                mx.zeros((1, attn.num_kv_heads, width, attn.head_dim)),
            )


def test_dense_shortcircuit_keeps_selection_past_budget():
    args, attn = _attention(indexer_budget=8)
    indexer = attn.indexer
    cache = Q.QSAKVCache(indexer.summary_identity)
    attn(mx.ones((1, 40, args.hidden_size)), None, cache)
    selection = indexer(mx.ones((1, 1, args.hidden_size)), None, cache, dense_shortcircuit=True)
    assert selection.kind == "explicit"


def test_prep_admission_reasons(monkeypatch):
    args, attn = _attention()
    monkeypatch.setattr(R, "metal_ready", lambda: True)
    qg = mx.zeros((1, 1, attn.num_heads * 2 * attn.head_dim), dtype=mx.bfloat16)
    kf = mx.zeros((1, 1, attn.num_kv_heads * attn.head_dim), dtype=mx.bfloat16)
    w = mx.ones((attn.head_dim,), dtype=mx.bfloat16)
    common = dict(heads=attn.num_heads, kv_heads=attn.num_kv_heads, head_dim=attn.head_dim, rope=attn.rope)
    # tiny head_dim is not the transcribed layout
    assert R.prep_qk_supported(qg, kf, w, w, **common) == "head_dim"
    assert R.prep_qk_supported(qg.astype(mx.float32), kf, w, w, **common) == "dtype"
    big = dict(common, head_dim=256, heads=2, kv_heads=1)
    qg256 = mx.zeros((1, 1, 2 * 2 * 256), dtype=mx.bfloat16)
    kf256 = mx.zeros((1, 1, 256), dtype=mx.bfloat16)
    w256 = mx.ones((256,), dtype=mx.bfloat16)
    assert R.prep_qk_supported(qg256, kf256, w256, w256, **big) is None
    assert R.prep_qk_supported(qg256, kf256, w256.astype(mx.float32), w256, **big) == "norm_weight_dtype"
    assert R.prep_qk_supported(qg256, kf256, w256, w256, **dict(big, rope=nn.RoPE(64, traditional=True))) == "rope_layout"
    assert R.prep_qk_supported(qg256, kf256, w256, w256, **dict(big, rope=object())) == "rope_type"


def test_sdpa_admission_reasons(monkeypatch):
    monkeypatch.setattr(R, "metal_ready", lambda: True)
    monkeypatch.setattr(R, "_device_class", lambda: "s")
    monkeypatch.delenv("MLX_SDPA_BLOCKS", raising=False)
    q = mx.zeros((1, 24, 1, 256), dtype=mx.bfloat16)
    k = mx.zeros((1, 2, 3000, 256), dtype=mx.bfloat16)
    assert R.sdpa_supported(q, k, k, None) is None
    assert R.sdpa_supported(q, k, k, mx.ones((1, 1, 1, 3000), dtype=mx.bool_)) is None
    assert R.sdpa_supported(q, k, k, mx.zeros((1, 1, 1, 3000))) == "mask_dtype"
    assert R.sdpa_supported(q, k, k, "causal") == "mask_kind"
    assert R.sdpa_supported(q, k, k, mx.ones((1, 24, 1, 3000), dtype=mx.bool_)) == "mask_heads"
    assert R.sdpa_supported(mx.zeros((2, 24, 1, 256), dtype=mx.bfloat16), k, k, None) == "batch"
    assert R.sdpa_supported(mx.zeros((1, 24, 3, 256), dtype=mx.bfloat16), k, k, None) == "plan"
    assert R.sdpa_supported(q, k.astype(mx.float16), k, None) == "kv_dtype"


def test_mask_and_index_q_admission(monkeypatch):
    args, attn = _attention(indexer_budget=8)
    indexer = attn.indexer
    cache = Q.QSAKVCache(indexer.summary_identity)
    attn(mx.ones((1, 40, args.hidden_size)), None, cache)
    selection = indexer(mx.ones((1, 1, args.hidden_size)), None, cache)
    assert R.qsa_mask_supported(selection) == "device"
    monkeypatch.setattr(R, "metal_ready", lambda: True)
    assert selection.kind == "explicit"
    assert R.qsa_mask_supported(selection) is None
    implicit = Q.QSASelection(kind="implicit_all", batch=1, length=1, block_size=4,
                              physical_width=4, n_blocks=1)
    assert R.qsa_mask_supported(implicit) == "kind"
    q_pos = mx.array([[5]], dtype=mx.int32)
    qk = mx.zeros((1, 1, 3 * 128), dtype=mx.bfloat16)
    w = mx.ones((128,), dtype=mx.bfloat16)
    assert R.index_q_supported(qk, w, q_pos, 64, 2, 128, None) is None
    assert R.index_q_supported(qk, w, q_pos, 64, 2, 128, (mx.ones((32,)), 1.0)) == "scaled_rope"
    assert R.index_q_supported(qk, w, q_pos, 64, 2, 64, None) == "head_dim"
    assert R.index_q_supported(qk, w.astype(mx.float32), q_pos, 64, 2, 128, None) == "dtype"
