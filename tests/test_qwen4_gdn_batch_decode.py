"""Batched one-token fused GDN decode (omlx #4106 GDN half): admission,
routing, counters and policy on the CPU.

Metal bit-exactness lives in ``scripts/check_qwen4_gdn_batch_decode.py``.  On
the CPU an admitted batched step records "Metal runtime unavailable" in the
batch counters and runs the stock chain; any other reason is a geometry
refusal the GPU would also make.
"""

from collections import Counter

import mlx.core as mx
import pytest

from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.runtime.generate import BatchGenerator
from mlx2.runtime.models import qwen4_exp
from mlx2.runtime.models.cache import ArraysCache
from mlx2.runtime.models.qwen4_fused_gdn import (
    BATCH_DECODE_MAX_ROWS,
    CONV_DIM,
    NUM_VALUE_HEADS,
    VALUE_DIM,
    admit_batch_rollback_spans,
    admit_qwen4_fused_gdn_batch_decode,
    admit_qwen4_fused_gdn_decode,
)
from test_gdn_mask_geometry import production_gdn_qwen4  # noqa: F401 - fixture

ADMITTED_ON_CPU = "Metal runtime unavailable"


def _operands(rows, *, state_rows=None, dtype=mx.bfloat16):
    state_rows = rows if state_rows is None else state_rows
    return dict(
        qkv=mx.zeros((rows, 1, CONV_DIM), dtype),
        z=mx.zeros((rows, 1, VALUE_DIM), dtype),
        b=mx.zeros((rows, 1, NUM_VALUE_HEADS), dtype),
        a=mx.zeros((rows, 1, NUM_VALUE_HEADS), dtype),
        conv_state=mx.zeros((state_rows, 3, CONV_DIM), dtype),
        recurrent_state=mx.zeros((state_rows, NUM_VALUE_HEADS, 128, 128), mx.float32),
        conv_weight=mx.zeros((CONV_DIM, 4, 1), dtype),
        A_log=mx.zeros((NUM_VALUE_HEADS,), mx.float32),
        dt_bias=mx.zeros((NUM_VALUE_HEADS,), dtype),
        norm_weight=mx.zeros((128,), dtype),
    )


def _admit(rows, **overrides):
    kwargs = dict(
        _operands(rows), mask=None, spans=(), speculating=False, training=False,
        sharded=False, num_key_heads=16, num_value_heads=48, key_head_dim=128,
        value_head_dim=128, conv_kernel=4, gate_activation="sigmoid",
    )
    kwargs.update(overrides)
    return admit_qwen4_fused_gdn_batch_decode(**kwargs)


@pytest.mark.parametrize("rows", [2, 3, 4, 8, 16, BATCH_DECODE_MAX_ROWS])
def test_batch_admission_accepts_every_width_up_to_the_cap(rows):
    assert _admit(rows).accepted
    assert _admit(rows, spans=[1] * rows).accepted


def test_batch_admission_refusals_are_named():
    assert _admit(1).reason == "single row (B=1 kernel)"
    too_wide = BATCH_DECODE_MAX_ROWS + 1
    assert _admit(too_wide).reason == f"batch of {too_wide} rows > {BATCH_DECODE_MAX_ROWS}"
    two_tokens = _operands(2)
    two_tokens["qkv"] = mx.zeros((2, 2, CONV_DIM), mx.bfloat16)
    assert _admit(2, **two_tokens).reason == "2-token slab"
    assert _admit(2, speculating=True).reason == "speculative rollback"
    assert _admit(2, training=True).reason == "training"
    assert _admit(2, sharded=True).reason == "distributed sharding"
    assert _admit(2, spans=None).reason == "rollback geometry not describable"
    assert _admit(2, mask=mx.ones((2, 1), mx.bool_)).reason == "masked batched decode"
    assert _admit(2, spans=[1, 0]).reason == "padded batched rollback geometry"
    assert _admit(3, spans=[1, 1]).reason == "rollback spans for 2 rows, batch of 3"
    # A lane whose state rows do not match the activations (a join that has
    # not merged its state yet) is a shape refusal, not a launch.
    stale = _admit(3, **_operands(3, state_rows=2))
    assert not stale.accepted and stale.reason.startswith("conv_state shape")
    assert _admit(2, gate_activation="swish").reason == "output gate 'swish'"
    wrong = _operands(2)
    wrong["recurrent_state"] = wrong["recurrent_state"].astype(mx.bfloat16)
    assert _admit(2, **wrong).reason == "recurrent_state must be float32"


def test_batch_span_predicate_matches_the_one_row_rule():
    assert admit_batch_rollback_spans((), None, 4, 1) is None
    assert admit_batch_rollback_spans([1, 1, 1], mx.ones((3, 1), mx.bool_), 3, 1) is None


def test_one_row_admission_is_unchanged_by_the_shared_operand_checks():
    kwargs = dict(
        _operands(1), mask=None, spans=(), speculating=False, training=False,
        sharded=False, num_key_heads=16, num_value_heads=48, key_head_dim=128,
        value_head_dim=128, conv_kernel=4, gate_activation="sigmoid",
    )
    assert admit_qwen4_fused_gdn_decode(**kwargs).accepted
    kwargs.update(_operands(2))
    refusal = admit_qwen4_fused_gdn_decode(**kwargs)
    assert refusal.reason == f"qkv shape (2, 1, {CONV_DIM}), expected (1, 1, {CONV_DIM})"


def _gdn_layers(model):
    return [m for _, m in model.named_modules() if isinstance(m, qwen4_exp.GatedDeltaNet)]


def _decode(model, prompts, max_tokens, *, stagger=0):
    """Batched decode where lanes finish at different steps (``max_tokens``)
    and, with ``stagger``, the last prompt joins that many steps late."""
    gen = BatchGenerator(model, prefill_step_size=64, completion_batch_size=8,
                         prefill_batch_size=8)
    tokens = Counter()
    try:
        first = prompts[:-1] if stagger else prompts
        gen.insert(first, max_tokens=max_tokens[: len(first)])
        qwen4_exp.qwen4_fused_gdn_stats(model, reset=True)
        steps = 0
        for _ in range(400):
            if stagger and steps == stagger:
                gen.insert([prompts[-1]], max_tokens=[max_tokens[-1]])
            responses = gen.next()[1]
            steps += 1
            for response in responses:
                tokens[response.uid] += 1
            if sum(tokens.values()) == sum(max_tokens) and steps > stagger:
                break
    finally:
        gen.close()
    return qwen4_exp.qwen4_fused_gdn_stats(model), tokens


def _prompts(n):
    return [[(5 * i + 3 * j + 1) % 60 + 2 for i in range(12 + 3 * j)] for j in range(n)]


def test_default_off_keeps_the_existing_batched_refusal(production_gdn_qwen4):
    model = production_gdn_qwen4
    assert all(m.fused_gdn_batch_decode_mode == "off" for m in _gdn_layers(model))
    stats, _ = _decode(model, _prompts(3), [5, 5, 5])
    assert "batch_decode" not in stats
    shape = f"qkv shape (3, 1, {CONV_DIM}), expected (1, 1, {CONV_DIM})"
    assert stats["decode_fallback_reasons"].get(shape, 0) > 0, stats


def test_row_exact_admits_every_batched_step_while_lanes_finish_and_join(
    production_gdn_qwen4,
):
    model = production_gdn_qwen4
    for layer in _gdn_layers(model):
        layer.set_fused_gdn_batch_decode_mode("row_exact")
    try:
        stats, tokens = _decode(model, _prompts(4), [3, 7, 5, 6], stagger=2)
    finally:
        for layer in _gdn_layers(model):
            layer.set_fused_gdn_batch_decode_mode("off")
    batch = stats["batch_decode"]
    assert batch["modes"] == ["row_exact"]
    # Every batched one-token step reached the launch gate; on the CPU that
    # is the runtime refusal, never a geometry one.
    assert set(batch["fallback_reasons"]) == {ADMITTED_ON_CPU}, batch
    assert batch["fallbacks"] > 0
    # B>1 steps no longer book the B=1 shape refusal.
    assert not any(r.startswith("qkv shape") for r in stats["decode_fallback_reasons"])
    assert sorted(tokens.values()) == [3, 5, 6, 7]


def test_row_exact_output_matches_stock_on_cpu(production_gdn_qwen4):
    """On the CPU the batched route falls back, so outputs must be the stock
    chain's exactly; the counters record the admitted attempt."""
    model = production_gdn_qwen4
    layer = _gdn_layers(model)[0]
    mx.random.seed(3)
    hidden = layer.hidden_size
    x = (mx.random.normal((3, 1, hidden)) * 0.5).astype(mx.bfloat16)
    conv = (mx.random.normal((3, 3, CONV_DIM)) * 0.5).astype(mx.bfloat16)
    state = mx.random.normal((3, NUM_VALUE_HEADS, 128, 128)) * 0.05
    outs = {}
    for mode in ("off", "row_exact"):
        layer.set_fused_gdn_batch_decode_mode(mode)
        cache = ArraysCache(2)
        cache[0], cache[1] = conv, state
        outs[mode] = (layer(x, cache=cache), cache[0], cache[1])
        mx.eval(outs[mode])
    layer.set_fused_gdn_batch_decode_mode("off")
    for p, q in zip(outs["off"], outs["row_exact"]):
        assert mx.array_equal(p, q).item()
    assert layer.fused_gdn_batch_decode_fallback_reasons == {ADMITTED_ON_CPU: 1}


def test_row_exact_launch_updates_cache_and_counters(production_gdn_qwen4, monkeypatch):
    """With a stand-in kernel the route commits states, advances the cache
    once and counts calls and rows; the stock decode kill switch wins."""
    layer = _gdn_layers(production_gdn_qwen4)[0]
    launched = []

    def fake(qkv, z, b, a, conv_state, *args, threadgroup_y, **kwargs):
        launched.append((int(qkv.shape[0]), threadgroup_y))
        return (mx.zeros(z.shape, z.dtype), conv_state + 1, args[3] + 1)

    monkeypatch.setattr(qwen4_exp, "fused_gdn_runtime_supported", lambda: True)
    monkeypatch.setattr(qwen4_exp, "probe_qwen4_fused_gdn_decode", lambda dtype: 32)
    monkeypatch.setattr(qwen4_exp, "qwen4_fused_gdn_batch_decode", fake)
    layer.set_fused_gdn_batch_decode_mode("row_exact")
    try:
        cache = ArraysCache(2, left_padding=[0, 0, 0, 0])
        cache[0] = mx.zeros((4, 3, CONV_DIM), mx.bfloat16)
        cache[1] = mx.zeros((4, NUM_VALUE_HEADS, 128, 128), mx.float32)
        x = mx.zeros((4, 1, layer.hidden_size), mx.bfloat16)
        out = layer(x, cache=cache)
        mx.eval(out)
        assert launched == [(4, 32)]
        assert out.shape == (4, 1, layer.hidden_size)
        assert cache[0].max().item() == 1 and cache[1].max().item() == 1
        assert cache.left_padding.tolist() == [-1, -1, -1, -1]
        assert (layer.fused_gdn_batch_decode_calls, layer.fused_gdn_batch_decode_rows) == (1, 4)
        stats = qwen4_exp.qwen4_fused_gdn_stats(production_gdn_qwen4)
        assert stats["batch_decode"]["calls"] == 1 and stats["batch_decode"]["rows"] == 4
        layer.set_fused_gdn_decode_mode("stock")
        layer(x, cache=cache)
        assert len(launched) == 1
    finally:
        layer.set_fused_gdn_decode_mode("fused")
        layer.set_fused_gdn_batch_decode_mode("off")


def test_batch_mode_setter_rejects_unknown_modes(production_gdn_qwen4):
    layer = _gdn_layers(production_gdn_qwen4)[0]
    with pytest.raises(ValueError, match="batch decode mode"):
        layer.set_fused_gdn_batch_decode_mode("multi_row")


def test_policy_field_enters_environment_and_receipts_only_when_selected():
    default = FlashNextPolicy()
    assert default.as_dict()["fused_gdn_batch_decode"] == "row_exact"
    assert default.environment()["MLX_QWEN4_FUSED_GDN_BATCH_DECODE"] == "row_exact"
    off = FlashNextPolicy.from_mapping({"fused_gdn_batch_decode": "off"})
    assert "fused_gdn_batch_decode" not in off.as_dict()
    assert "MLX_QWEN4_FUSED_GDN_BATCH_DECODE" not in off.environment()
    with pytest.raises(ValueError, match="fused_gdn_batch_decode"):
        FlashNextPolicy.from_mapping({"fused_gdn_batch_decode": "on"})
