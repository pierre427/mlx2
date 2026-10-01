"""CPU plumbing of the row-exact attention window (W2-B; omlx #4041/#4105).

The Metal kernels are Metal-only; their bit-exactness against R one-token
forwards on real Flash-Next layers is ``scripts/check_row_exact_window.py``.
Here: the generated sources, the per-row plan runs, admission (checked before
any state changes), the deferred-SDPA bookkeeping, and that the route still
equals MTP-off with the window switched on (CPU: nothing admitted).
"""

import mlx.core as mx
import numpy as np
import pytest

from qsa_oracle import tiny_args

from mlx2.runtime import row_exact_verify as REV
from mlx2.runtime.models import qwen4_attn_rows as AR
from mlx2.runtime.models import qwen4_attn_window as AW
from mlx2.runtime.models import qwen4_exp as Q


@pytest.fixture(autouse=True)
def _reset():
    AR.set_enabled(False)
    AR.status(reset=True)
    previous = AW.set_enabled(True)
    yield
    AR.set_enabled(False)
    AR.status(reset=True)
    AW.set_enabled(previous)


def test_env_flag_defaults_off():
    assert AW._parse_flag(None) is False
    assert AW._parse_flag("0") is False and AW._parse_flag("off") is False
    assert AW._parse_flag("1") is True
    with pytest.raises(ValueError):
        AW._parse_flag("maybe")


def test_policy_field_follows_the_row_exact_route_and_is_receipt_neutral():
    from mlx2.adapters.flash_next_policy import FlashNextPolicy

    default = FlashNextPolicy()
    assert "row_exact_window_kernels" not in default.as_dict()
    env = default.environment()
    assert AW.ENV_NAME not in env and "MLX_QWEN4_HC_ROW_EXACT" not in env
    chosen = FlashNextPolicy.from_mapping({"row_exact_verify": True})
    assert chosen.as_dict()["row_exact_window_kernels"] is True
    env = chosen.environment()
    assert env[AW.ENV_NAME] == "1" and env["MLX_QWEN4_HC_ROW_EXACT"] == "1"
    off = FlashNextPolicy.from_mapping({"row_exact_verify": True, "row_exact_window_kernels": False})
    assert "row_exact_window_kernels" not in off.as_dict()
    assert AW.ENV_NAME not in off.environment()
    with pytest.raises(ValueError, match="row_exact_window_kernels"):
        FlashNextPolicy.from_mapping({"row_exact_window_kernels": "yes"})


@pytest.mark.parametrize("masked", [False, True])
def test_sources_index_rows_and_keys_per_row(masked):
    one = AW._one_pass_source(masked)
    two = AW._two_pass_source(masked)
    for source in (one, two):
        assert "const int N = n_first[0] + row;" in source
        assert "keys_shape[2]" not in source
        assert "@" not in source
        assert ("mask_base[0]" in source) == masked
        assert ("mp[0]" in source) == masked
    assert "threadgroup_position_in_grid.z) % BLOCKS" in two
    assert "threadgroup_position_in_grid.z) / BLOCKS" in two
    assert "thread_position_in_threadgroup.z" not in two


def test_plan_runs_split_at_every_plan_switch(monkeypatch):
    monkeypatch.setattr(AR, "_device_class", lambda: "s")
    monkeypatch.delenv("MLX_SDPA_BLOCKS", raising=False)
    plans = AW.row_plans(1020, 8, 24, 2, 256)
    assert AW.plan_runs(plans) == [(0, 4, (1, None)), (4, 5, (2, 64)), (5, 8, (2, 128))]
    plans = AW.row_plans(8190, 5, 24, 2, 256)
    assert AW.plan_runs(plans) == [(0, 3, (2, 128)), (3, 5, (2, 256))]
    assert AW.plan_runs(AW.row_plans(10, 3, 24, 2, 256)) == [(0, 3, (1, None))]


def test_window_supported_refuses_off_metal_and_bad_shapes(monkeypatch):
    q = mx.zeros((1, 24, 3, 256), dtype=mx.bfloat16)
    kv = mx.zeros((1, 2, 12, 256), dtype=mx.bfloat16)
    assert AW.window_supported(q, kv, kv, 10) == "device"
    monkeypatch.setattr(AR, "metal_ready", lambda: True)
    monkeypatch.setattr(AR, "_device_class", lambda: "s")
    monkeypatch.delenv("MLX_SDPA_BLOCKS", raising=False)
    assert AW.window_supported(q, kv, kv, 10) is None
    assert AW.window_supported(q, kv, kv, 11) == "key_count"
    assert AW.window_supported(q.astype(mx.float32), kv, kv, 10) == "dtype"
    wide = mx.zeros((1, 24, AW.WINDOW_MAX_ROWS + 1, 256), dtype=mx.bfloat16)
    assert AW.window_supported(wide, kv, kv, 10) == "rows"
    monkeypatch.setattr(AR, "_device_class", lambda: "g")
    assert AW.window_supported(q, kv, kv, 10) == "plan"


def test_flat_mask_accepts_only_one_full_row():
    assert AW._flat_mask(None, 5).tolist() == [True] * 5
    row = mx.array([[[[True, False, True]]]])
    assert AW._flat_mask(row, 3).tolist() == [True, False, True]
    assert AW._flat_mask(row, 4) is None
    assert AW._flat_mask(mx.ones((1, 1, 2, 3), dtype=mx.bool_), 3) is None
    assert AW._flat_mask(mx.zeros((1, 1, 1, 3)), 3) is None  # float mask


def _attention(**overrides):
    args = tiny_args(**overrides)
    attn = Q.Attention(args, 3)
    attn.eval()
    mx.eval(attn.parameters())
    return args, attn


def _cache_with(attn, args, n):
    cache = Q.QSAKVCache(attn.indexer.summary_identity)
    if n:
        x = mx.array(np.random.default_rng(0).standard_normal((1, n, args.hidden_size)).astype(np.float32))
        attn(x, None, cache)
    return cache


def test_dense_admission_reasons_before_any_state_change():
    args, attn = _attention()
    x = mx.zeros((1, 3, args.hidden_size), dtype=mx.bfloat16)
    cache = _cache_with(attn, args, 4)
    assert AW.dense_admission(attn, x, cache, 3) == "attn_fused_rows_off"
    AR.set_enabled(True)
    assert AW.dense_admission(attn, x, cache, 3) is None
    budget = attn.indexer.block_topk * attn.indexer.compress_ratio
    far = budget - cache.offset + attn.indexer.compress_ratio
    assert AW.dense_admission(attn, mx.zeros((1, far, args.hidden_size), dtype=mx.bfloat16), cache, far) in (
        "past_budget", "rows")
    cache._mtp_shared_topk = mx.zeros((1, 1), dtype=mx.uint32)
    assert AW.dense_admission(attn, x, cache, 3) == "shared_topk"
    cache._mtp_shared_topk = None
    assert AW.dense_admission(attn, x.astype(mx.float32), cache, 3) == "rows_layout"
    batched = Q.QSAKVCache.merge([cache])
    assert AW.dense_admission(attn, x, batched, 3) is None
    batched._max_left_pad = (batched.left_padding, 2, [2])
    batched._unpadded_ref = None
    assert AW.dense_admission(attn, x, batched, 3) == "left_padding_nonzero"
    two = Q.QSAKVCache.merge([cache, _cache_with(attn, args, 4)])
    assert AW.dense_admission(attn, x, two, 3) == "batch_rows"
    assert AW.dense_admission(attn, x, object(), 3) == "cache_type"


def test_dense_window_declines_off_metal_without_touching_the_cache():
    args, attn = _attention()
    attn.set_dtype(mx.bfloat16)
    AR.set_enabled(True)
    cache = _cache_with(attn, args, 4)
    before = (cache.offset, cache.index_keys.shape[1])
    x = mx.ones((1, 3, args.hidden_size), dtype=mx.bfloat16)
    projected = attn._project_segmented_qsa(x)
    record = REV.Window(3)
    assert AW.dense_window(attn, x, cache, projected, record) is None
    assert (cache.offset, cache.index_keys.shape[1]) == before
    assert record.stages == {"attention_window_dense_decline": {"prep": 1}}


def _calls(first, rows, *, masked=True, kv=2, heads=4, d=8):
    k = mx.zeros((1, kv, first + rows - 1, d))
    out = AW.DeferredSDPA()
    for j in range(rows):
        n = first + j
        mask = mx.ones((1, 1, 1, n), dtype=mx.bool_) if masked else None
        out.append(mx.zeros((1, heads, 1, d)), k[:, :, :n], k[:, :, :n], 0.5, mask,
                   mx.full((1, 1, heads, d), float(j)))
    return out


def test_run_deferred_batches_consecutive_rows(monkeypatch):
    seen = {}

    def window(queries, keys, values, scale, *, n_first, gate, masks=None):
        seen.update(rows=queries.shape[2], n_first=n_first, keys=keys.shape[2],
                    masks=None if masks is None else masks.shape, gate=gate.shape)
        return mx.broadcast_to(mx.arange(queries.shape[2]).reshape(1, -1, 1), (1, queries.shape[2], 32))

    monkeypatch.setattr(AW, "window_supported", lambda *a: None)
    monkeypatch.setattr(AW, "window_sdpa_gate", window)
    monkeypatch.setattr(AR, "sdpa_gate", lambda *a, **k: pytest.fail("ran per row"))
    record = REV.Window(4)
    outs = AW.run_deferred(_calls(10, 4), record)
    assert seen == {"rows": 4, "n_first": 10, "keys": 13, "masks": (10 + 11 + 12 + 13,), "gate": (1, 4, 32)}
    assert [int(o[0, 0, 0].item()) for o in outs] == [0, 1, 2, 3]
    assert record.stages == {"attention_sdpa": {"window_deferred": 4}}
    AW.run_deferred(_calls(10, 3, masked=False), REV.Window(3))
    assert seen["masks"] is None


def test_run_deferred_runs_rows_alone_when_they_do_not_form_a_window(monkeypatch):
    ran = []
    monkeypatch.setattr(AR, "sdpa_gate", lambda q, k, v, s, mask=None, gate=None: ran.append(k.shape[2]) or q)
    monkeypatch.setattr(AW, "window_sdpa_gate", lambda *a, **k: pytest.fail("batched"))
    calls = _calls(10, 3)
    calls.calls[1] = calls.calls[1][:4] + (mx.ones((1, 1, 2, 11), dtype=mx.bool_),) + calls.calls[1][5:]
    record = REV.Window(3)
    assert len(AW.run_deferred(calls, record)) == 3
    assert ran == [10, 11, 12]
    assert record.stages == {"attention_window_deferred_decline": {"mask_layout": 1}}
    ran.clear()
    gap = _calls(10, 2)
    gap.calls[1] = (gap.calls[1][0], gap.calls[0][1], gap.calls[0][2]) + gap.calls[1][3:]
    AW.run_deferred(gap, REV.Window(2))
    assert ran == [10, 10]
    ran.clear()
    AW.run_deferred(_calls(10, 1), REV.Window(1))
    assert ran == [10]
    assert AW.run_deferred(AW.DeferredSDPA(), REV.Window(1)) == []


def test_route_with_the_window_on_still_matches_mtp_off_on_cpu():
    from test_qwen4_row_exact_verify import _greedy, _quantized_tiny_model

    from mlx2.runtime.models.qwen4_row_exact import install

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        AR.set_enabled(True)
        model = _quantized_tiny_model()
        model.eval()
        prompt = [1, 7, 3, 9, 2, 8, 4, 6, 5, 11, 13, 2]
        off = _greedy(model, prompt, mtp=False)
        handle = install(model)
        handle.enable(True)
        on = _greedy(model, prompt, mtp=True)
        assert on == off
        stages = handle.status()["stages"]
        # The tiny model is float32 (and off Metal the prep is refused too):
        # declined before any state change, per-row forwards as before.
        assert set(stages["attention_window_dense_decline"]) == {"rows_layout"}
        assert "window_dense" not in stages["attention"]
    finally:
        mx.set_default_device(previous)
