"""CPU evidence for the adapter-owned Qwen hybrid row-exact target executor.

Tiny Qwen3.8 27B (dense) and Qwen3.6 35B-A3B (sparse MoE) layouts on the CPU
device.  Every verify row's logits and taps must be bit-identical to serial
one-token decode from the same cache; a partial commit must leave the state
k ordinary decode steps would; abort must restore exactly; any fallback must
fail the launch closed.  The stock wide forward is the falsifier: it is shown
NOT bit-identical on the same inputs, so these comparisons can fail.  No
qualification is claimed (real-artifact Metal evidence is pending).
"""

import copy

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from test_qwen38_dflash2_cpu import (
    PROMPTS,
    drain,
    ordinary_greedy,
    tiny_draft_args,
    tiny_target,
)
from test_qwen_progressive_external_portability_cpu import tiny_qwen36

from mlx2.runtime import row_exact_verify as REV
from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows
from mlx2.runtime.models import qwen_hybrid_row_exact as QHR
from mlx2.runtime.models.base import create_causal_mask
from mlx2.runtime.models.cache import ArraysCache, KVCache
from mlx2.runtime.models.row_exact_components import (
    RowExactLinearMixin,
    row_exact_subclass,
)
from mlx2.runtime.progressive_external_verify import (
    PROGRESSIVE_FIDELITY_EXACT,
    PROGRESSIVE_TARGET_PROTOCOL_VERSION,
    execute_progressive_target_rows,
    progressive_target_protocol_binding,
)
from mlx2.runtime.segmented_plain_kv import SegmentedBatchKVCache

LAYERS = (1, 6)
FACTORIES = [
    pytest.param(tiny_target, id="qwen38-27b-dense"),
    pytest.param(tiny_qwen36, id="qwen36-35b-moe"),
]
# Physical widths 1, 2, 4 with token widths 1..4, uniform and ragged.
LENGTHS = [
    [1], [2], [3], [4],
    [1, 1], [2, 1], [3, 4], [4, 4],
    [4, 1, 3, 2], [1, 2, 1, 1], [4, 4, 4, 4],
]


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _bases(model, lanes):
    bases = []
    for lane in range(lanes):
        cache = model.make_cache()
        mx.eval(model(mx.array([[1 + lane, 2, 3 + 2 * lane, 5, 8 + lane]]), cache=cache))
        bases.append(cache)
    return bases


def _tokens(lengths):
    width = max(lengths)
    return [
        [(7 * lane + 3 * position + 11) % 120 for position in range(width)]
        for lane in range(len(lengths))
    ]


def _serial(model, cache, tokens):
    logits, taps = [], []
    for token in tokens:
        row_logits, row_taps = model.forward_with_taps(mx.array([[token]]), cache, LAYERS)
        mx.eval(row_logits, row_taps)
        logits.append(row_logits)
        taps.append(row_taps)
    return mx.concatenate(logits, axis=1), mx.concatenate(taps, axis=1)


def _next_logits(model, cache):
    value = model(mx.array([[9]]), cache=copy.deepcopy(cache))
    mx.eval(value)
    return np.asarray(value)


def _launch(model, bases, lengths, *, compact=False):
    owner = HybridVerifyRows([copy.deepcopy(base) for base in bases])
    transaction = owner.begin(lengths)
    if compact:
        from mlx2.runtime.progressive_external_verify import ProgressiveTargetRows

        logits, features, receipt = execute_progressive_target_rows(
            model, mx.array(_tokens(lengths)), transaction.caches, LAYERS,
            protocol=PROGRESSIVE_TARGET_PROTOCOL_VERSION,
            fidelity=PROGRESSIVE_FIDELITY_EXACT, input_lengths=lengths,
        )
        result = ProgressiveTargetRows(logits, features, receipt["protocol"],
                                      receipt["row_exact"], receipt["adapter_detail"])
    else:
        result = model.progressive_external_target_rows(
            mx.array(_tokens(lengths)), transaction.caches, LAYERS
        )
    mx.eval(result.logits, result.features)
    return owner, transaction, result


def test_qwen36_rejected_moe_window_retains_the_exact_per_row_route(monkeypatch):
    model = tiny_qwen36()
    model.configure_target_verify_row_exact(True)
    blocks = [
        module
        for _, module in model.named_modules()
        if callable(getattr(module, "row_exact_window_selected", None))
    ]
    assert blocks
    for block in blocks:
        block.set_moe_window_consumers(("row_exact",))
        block.set_moe_routed_decode_mode("gate_up_down_shared")
        monkeypatch.setattr(
            block,
            "row_exact_window_refusal",
            lambda _x: "gate: bits 8 != 4",
        )
    bases = _bases(model, 1)
    _owner, transaction, result = _launch(model, bases, [3])
    transaction.abort()
    assert result.row_exact is True, result.detail
    assert result.detail["moe_route"] == "per_row_one_token_block"
    status = model.external_execution_receipt["target_verify_row_exact"]
    assert status["failures"] == {}
    assert status["routes"]["moe_window_fallback"] == {
        "gate: bits 8 != 4": len(blocks)
    }


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("lengths", LENGTHS, ids=lambda v: "x".join(map(str, v)))
@pytest.mark.parametrize("compact", [False, True], ids=["rectangular", "valid-rows"])
def test_rows_taps_commit_and_abort_match_serial_ordinary_decode(factory, lengths, compact):
    model = factory()
    model.configure_target_verify_row_exact(True)
    bases = _bases(model, len(lengths))
    tokens = _tokens(lengths)

    owner, transaction, result = _launch(model, bases, lengths, compact=compact)
    assert result.protocol == PROGRESSIVE_TARGET_PROTOCOL_VERSION
    assert result.row_exact is True, result.detail
    assert result.detail["gdn_route"] == "serial_one_token"
    for lane, valid in enumerate(lengths):
        logits, taps = _serial(model, copy.deepcopy(bases[lane]), tokens[lane][:valid])
        np.testing.assert_array_equal(
            np.asarray(result.logits[lane : lane + 1, :valid]), np.asarray(logits)
        )
        np.testing.assert_array_equal(
            np.asarray(result.features[lane : lane + 1, :valid]), np.asarray(taps)
        )

    # Commit k of n rows per lane: the state is k ordinary decode steps.
    accepted = [1 + (lane % valid) for lane, valid in enumerate(lengths)]
    committed = transaction.commit(accepted)
    for lane, kept in enumerate(accepted):
        reference = copy.deepcopy(bases[lane])
        _serial(model, reference, tokens[lane][:kept])
        np.testing.assert_array_equal(
            _next_logits(model, committed[lane]), _next_logits(model, reference)
        )
        for cache in committed[lane]:
            if isinstance(cache, ArraysCache):
                assert not cache.speculating and not cache._rollbacks

    # Abort restores every lane exactly.
    owner, transaction, result = _launch(model, bases, lengths, compact=compact)
    assert result.row_exact is True
    transaction.abort()
    for lane, row in enumerate(owner.rows):
        np.testing.assert_array_equal(
            _next_logits(model, row), _next_logits(model, bases[lane])
        )


@pytest.mark.parametrize("factory", FACTORIES)
def test_stock_wide_forward_is_the_falsifier(factory):
    model = factory()
    bases = _bases(model, 1)
    tokens = _tokens([4])[0]
    serial_logits, serial_taps = _serial(model, copy.deepcopy(bases[0]), tokens)

    owner = HybridVerifyRows([copy.deepcopy(bases[0])])
    transaction = owner.begin([4])
    wide_logits, wide_taps = model.forward_with_taps(
        mx.array([tokens]), transaction.caches, LAYERS
    )
    mx.eval(wide_logits, wide_taps)
    transaction.abort()
    # The comparison above can fail: the stock wide law changes the bits.
    assert not np.array_equal(np.asarray(wide_logits), np.asarray(serial_logits))
    assert not np.array_equal(np.asarray(wide_taps), np.asarray(serial_taps))

    model.configure_target_verify_row_exact(True)
    _owner, transaction, result = _launch(model, bases, [4])
    transaction.abort()
    np.testing.assert_array_equal(np.asarray(result.logits), np.asarray(serial_logits))
    np.testing.assert_array_equal(np.asarray(result.features), np.asarray(serial_taps))


@pytest.mark.parametrize("factory", FACTORIES)
def test_protocol_and_hook_exist_only_while_selected(factory):
    model = factory()
    lm_head = model.language_model.lm_head
    assert not hasattr(model, "progressive_external_target_rows")
    assert not hasattr(model, "progressive_external_verify_protocol")
    assert model.external_execution_receipt is None
    with pytest.raises(ValueError, match="row-exact progressive target protocol"):
        progressive_target_protocol_binding(model, PROGRESSIVE_FIDELITY_EXACT)
    with pytest.raises(ValueError, match="boolean"):
        model.configure_target_verify_row_exact("yes")

    route = model.configure_target_verify_row_exact(True)
    assert model.configure_target_verify_row_exact(True) is route
    assert model.progressive_external_verify_protocol == (
        PROGRESSIVE_TARGET_PROTOCOL_VERSION
    )
    assert callable(model.progressive_external_target_rows)
    binding = progressive_target_protocol_binding(model, PROGRESSIVE_FIDELITY_EXACT)
    assert binding["target_protocol"] == PROGRESSIVE_TARGET_PROTOCOL_VERSION
    receipt = model.external_execution_receipt["target_verify_row_exact"]
    assert receipt["selected"] is True and receipt["qualified"] is False
    assert receipt["observed_used"] is False
    assert receipt["gpu_evidence"] == QHR.GPU_EVIDENCE
    assert route.static_refusals == {}
    assert type(lm_head) is not nn.Linear

    # A plain (non-transaction) B1 call is exact as well and is receipted.
    cache = model.make_cache()
    mx.eval(model(mx.array([[1, 2, 3]]), cache=cache))
    reference = copy.deepcopy(cache)
    result = model.progressive_external_target_rows(mx.array([[4, 5]]), cache, LAYERS)
    receipt = model.external_execution_receipt["target_verify_row_exact"]
    assert receipt["observed_used"] is True
    assert receipt["row_exact_launches"] == 1 and receipt["nonexact_launches"] == 0
    assert receipt["routes"]["attention"] == {"serial_b1s1": 2 * 2}
    # The selected instance's forward_with_taps takes the same law (body-only
    # prefill excepted), so the serial reference below is receipted too.
    serial_logits, _ = _serial(model, reference, [4, 5])
    np.testing.assert_array_equal(np.asarray(result.logits), np.asarray(serial_logits))
    np.testing.assert_array_equal(_next_logits(model, cache), _next_logits(model, reference))
    receipt = model.external_execution_receipt["target_verify_row_exact"]
    assert receipt["row_exact_launches"] == 3 and receipt["nonexact_launches"] == 0
    wide, _ = model.forward_with_taps(mx.array([[6, 7, 8]]), copy.deepcopy(cache), LAYERS)
    last, _ = model.forward_with_taps(
        mx.array([[6, 7, 8]]), copy.deepcopy(cache), LAYERS, last_logits_only=True
    )
    np.testing.assert_array_equal(np.asarray(last), np.asarray(wide[:, -1:]))
    body = model.prefill_body(mx.array([[6, 7, 8]]), copy.deepcopy(cache), LAYERS)
    assert model.external_execution_receipt["target_verify_row_exact"][
        "row_exact_launches"
    ] == 5
    assert body.shape[1] == 3

    model.configure_target_verify_row_exact(False)
    assert not hasattr(model, "progressive_external_target_rows")
    assert not hasattr(model, "progressive_external_verify_protocol")
    assert "forward_with_taps" not in model.__dict__
    assert model.external_execution_receipt is None
    assert type(lm_head) is nn.Linear


@pytest.mark.parametrize("factory", FACTORIES)
def test_fallback_stage_fails_the_launch_closed(factory, monkeypatch):
    model = factory()
    model.configure_target_verify_row_exact(True)
    bases = _bases(model, 2)
    monkeypatch.setattr(QHR, "_ordinary_b1_mask", lambda cache: None)
    _owner, transaction, result = _launch(model, bases, [3, 2])
    transaction.abort()
    assert result.row_exact is False
    # ``reason`` is the first failure by name; the stage failure is recorded.
    assert result.detail["reason"] in (
        "attention_layer_not_engaged", "attention_offset_not_host"
    )
    assert result.detail["failure_count"] >= 2
    _owner, transaction, _ = _launch(model, bases[:1], [2])
    transaction.abort()
    owner = HybridVerifyRows([copy.deepcopy(bases[0])])
    transaction = owner.begin([2])
    with pytest.raises(RuntimeError, match="fell off its exact path"):
        execute_progressive_target_rows(
            model,
            mx.array([[4, 5]]),
            transaction.caches,
            LAYERS,
            protocol=PROGRESSIVE_TARGET_PROTOCOL_VERSION,
            fidelity=PROGRESSIVE_FIDELITY_EXACT,
        )
    transaction.abort()
    receipt = model.external_execution_receipt["target_verify_row_exact"]
    assert receipt["nonexact_launches"] == 3
    assert receipt["failures"]["attention_offset_not_host"] > 0


@pytest.mark.parametrize("factory", FACTORIES)
def test_replaced_or_lane_prepared_projection_fails_closed(factory):
    model = factory()
    model.configure_target_verify_row_exact(True)
    bases = _bases(model, 1)
    head = model.language_model.lm_head
    swapped = type(head)
    # A non-exact projection installed after selection (here: the stock
    # multi-row Linear) must not be credited.
    head.__class__ = nn.Linear
    _owner, transaction, result = _launch(model, bases, [3])
    transaction.abort()
    assert result.row_exact is False
    assert result.detail["reason"] == "class_swap_replaced_after_install"
    head.__class__ = swapped
    object.__setattr__(head, "_mlx2_row_exact_lane", object())
    _owner, transaction, result = _launch(model, bases, [3])
    transaction.abort()
    assert result.row_exact is False
    assert result.detail["reason"] == "lane_prepared_projection"
    head.__dict__.pop("_mlx2_row_exact_lane")
    _owner, transaction, result = _launch(model, bases, [3])
    transaction.abort()
    assert result.row_exact is True


def test_static_refusals_fail_every_launch():
    model = tiny_target()
    model.language_model.args.tie_word_embeddings = True
    route = model.configure_target_verify_row_exact(True)
    assert route.static_refusals == {"tied_embedding_head": 1}
    _owner, transaction, result = _launch(model, _bases(model, 1), [2])
    transaction.abort()
    assert result.row_exact is False
    assert result.detail["reason"] == "tied_embedding_head"


@pytest.mark.parametrize("width", [2, 4, 8])
def test_shared_dense_projection_component_matches_serial_b1s1(width):
    mx.random.seed(17)
    layer = nn.Linear(13, 19, bias=True)
    values = mx.random.normal((width, 1, 13))
    expected = mx.concatenate(
        [layer(values[index : index + 1]) for index in range(width)], axis=0
    )
    original = type(layer)
    layer.__class__ = row_exact_subclass(RowExactLinearMixin, original)
    record = REV.Window(width)
    with REV.window(record):
        actual = layer(values)
    mx.eval(expected, actual)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert record.exact
    assert record.stages["projections"] in (
        {"matvec_rows": 1}, {"per_row_canary": 1}, {"per_row_biased": 1}
    )


def _ordinary_mask(cache):
    return create_causal_mask(1, offset=cache.offset, left_padding=mx.array([0]))


@pytest.mark.parametrize("lengths", [[3, 3], [3, 1, 2, 3], [3] * 8])
def test_qwen3next_attention_serial_rows_preserve_ragged_lineages(lengths):
    model = tiny_qwen36(seed=37)
    model.configure_target_verify_row_exact(True)
    attention = model.layers[3].self_attn
    hidden = mx.random.normal((len(lengths), 3, 32))
    bases = []
    for _ in lengths:
        cache = model.make_cache()[3]
        mx.eval(attention(mx.random.normal((1, 1, 32)), _ordinary_mask(cache), cache))
        bases.append(cache)
    expected = []
    for lane, valid in enumerate(lengths):
        cache = copy.deepcopy(bases[lane])
        expected.append(
            mx.concatenate(
                [
                    attention(
                        hidden[lane : lane + 1, p : p + 1], _ordinary_mask(cache), cache
                    )
                    for p in range(valid)
                ],
                axis=1,
            )
        )
    rows = [copy.deepcopy(cache) for cache in bases]
    segmented = SegmentedBatchKVCache(rows)
    segmented.prepare(lengths=lengths, right_padding=[3 - n for n in lengths])
    record = REV.Window(sum(lengths))
    with REV.window(record):
        actual = attention(hidden, segmented.make_mask(3), segmented)
    mx.eval(actual, *expected)
    for lane, valid in enumerate(lengths):
        np.testing.assert_array_equal(
            np.asarray(actual[lane : lane + 1, :valid]), np.asarray(expected[lane])
        )
    assert record.exact
    assert record.stages["attention"] == {"serial_b1s1": sum(lengths)}
    assert [row.offset for row in rows] == [
        base.offset + valid for base, valid in zip(bases, lengths)
    ]


def test_qwen3next_attention_window_keeps_serial_kv_growth(monkeypatch):
    from mlx2.runtime.models import qwen4_attn_rows as AR
    from mlx2.runtime.models import qwen4_attn_window as AW

    model = tiny_qwen36(seed=39)
    route = model.configure_target_verify_row_exact(True)
    route.set_attention_window_row_exact(True)
    attention = model.layers[3].self_attn
    valid = 2
    qg = mx.zeros(
        (1, valid, attention.num_attention_heads * attention.head_dim * 2)
    )
    k_flat = mx.zeros(
        (1, valid, attention.num_key_value_heads * attention.head_dim)
    )
    v_flat = mx.zeros_like(k_flat)
    cache = KVCache()
    cache.keys = mx.zeros(
        (1, attention.num_key_value_heads, 256, attention.head_dim)
    )
    cache.values = mx.zeros_like(cache.keys)
    cache.offset = 255

    monkeypatch.setattr(AR, "prep_qk_supported", lambda *a, **k: None)
    monkeypatch.setattr(
        AR,
        "prep_qk",
        lambda *a, **k: (
            mx.zeros(
                (1, attention.num_attention_heads, valid, attention.head_dim)
            ),
            mx.zeros(
                (1, attention.num_key_value_heads, valid, attention.head_dim)
            ),
        ),
    )
    monkeypatch.setattr(AW, "window_supported", lambda *a, **k: None)
    seen = {}

    def window(q, keys, values, scale, *, n_first, gate, masks=None):
        seen.update(
            offset=cache.offset,
            capacity=cache.keys.shape[2],
            keys=keys.shape[2],
            n_first=n_first,
            masks=masks,
        )
        return mx.zeros(
            (1, valid, attention.num_attention_heads * attention.head_dim)
        )

    monkeypatch.setattr(AW, "window_sdpa_gate", window)
    output, reason = attention._row_exact_attention_window(
        qg, k_flat, v_flat, cache, valid
    )
    mx.eval(output)
    assert reason is None
    assert cache.offset == 257
    # Two ordinary one-token appends grow 256 -> 512.  One bulk append from
    # offset 255 would truncate to 255 then add 256, leaving capacity 511.
    assert seen == {
        "offset": 257,
        "capacity": 512,
        "keys": 257,
        "n_first": 256,
        "masks": None,
    }


def test_qwen3next_attention_window_declines_before_mutation_and_never_falls_back_after(
    monkeypatch,
):
    from mlx2.runtime.models import qwen4_attn_rows as AR
    from mlx2.runtime.models import qwen4_attn_window as AW

    model = tiny_qwen36(seed=40)
    route = model.configure_target_verify_row_exact(True)
    route.set_attention_window_row_exact(True)
    attention = model.layers[3].self_attn
    qg = mx.zeros(
        (1, 2, attention.num_attention_heads * attention.head_dim * 2)
    )
    k_flat = mx.zeros((1, 2, attention.num_key_value_heads * attention.head_dim))
    v_flat = mx.zeros_like(k_flat)
    cache = KVCache()
    monkeypatch.setattr(AR, "prep_qk_supported", lambda *a, **k: None)
    monkeypatch.setattr(
        AR,
        "prep_qk",
        lambda *a, **k: (
            mx.zeros((1, attention.num_attention_heads, 2, attention.head_dim)),
            mx.zeros((1, attention.num_key_value_heads, 2, attention.head_dim)),
        ),
    )
    monkeypatch.setattr(AW, "window_supported", lambda *a, **k: "plan")
    assert attention._row_exact_attention_window(qg, k_flat, v_flat, cache, 2) == (
        None,
        "plan",
    )
    assert cache.offset == 0 and cache.keys is None

    monkeypatch.setattr(AW, "window_supported", lambda *a, **k: None)
    monkeypatch.setattr(
        AW,
        "window_sdpa_gate",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("kernel failed")),
    )
    with pytest.raises(RuntimeError, match="kernel failed"):
        attention._row_exact_attention_window(qg, k_flat, v_flat, cache, 2)
    assert cache.offset == 2  # owning transaction must abort; no serial fallback


def test_qwen3next_attention_window_counts_mixed_ragged_rows(monkeypatch):
    model = tiny_qwen36(seed=41)
    route = model.configure_target_verify_row_exact(True)
    route.set_attention_window_row_exact(True)
    attention = model.layers[3].self_attn
    hidden = mx.zeros((2, 2, 32))
    caches = [model.make_cache()[3], model.make_cache()[3]]

    def candidate(q, k, v, cache, valid):
        if valid < 2:
            return None
        for position in range(valid):
            cache.update_and_fetch(
                mx.zeros((1, attention.num_key_value_heads, 1, attention.head_dim)),
                mx.zeros((1, attention.num_key_value_heads, 1, attention.head_dim)),
            )
        return (
            mx.zeros(
                (1, valid, attention.num_attention_heads * attention.head_dim)
            ),
            None,
        )

    monkeypatch.setattr(attention, "_row_exact_attention_window", candidate)
    record = REV.Window(3)
    with REV.window(record):
        output = attention._row_exact_serial_attention(hidden, caches, (2, 1))
    mx.eval(output)
    assert [cache.offset for cache in caches] == [2, 1]
    assert record.stages["attention"] == {
        "window_b1s1": 2,
        "serial_b1s1": 1,
    }
    assert record.stages["attention_layers"] == {
        "mixed_window_serial_b1s1": 1
    }


def test_gdn_fused_verify_refused_by_its_canary_takes_the_serial_route(monkeypatch):
    from mlx2.runtime.models.qwen36_35b import GatedDeltaNet

    model = tiny_qwen36()
    model.configure_target_verify_row_exact(True)
    layers = [m for _, m in model.named_modules() if isinstance(m, GatedDeltaNet)]
    for layer in layers:
        layer.set_fused_gdn_verify_mode("row_exact")
    # Pretend a fused kernel could run: the layer's own verify path refuses
    # this tiny geometry, so the decode-parity canary must fail closed.
    monkeypatch.setattr(QHR, "_fused_runtime_supported", lambda: True)
    bases = _bases(model, 1)
    _owner, transaction, result = _launch(model, bases, [3])
    transaction.abort()
    assert result.row_exact is True
    assert result.detail["gdn_route"] == "serial_one_token"
    stages = model.external_execution_receipt["target_verify_row_exact"]["routes"]
    assert stages["gdn_canary"] == {"verify_differs_from_decode": len(layers)}
    # Canary refusal is diagnostic, not serving use or a serving fallback.
    assert all(layer.fused_gdn_verify_calls == 0 for layer in layers)
    assert all(layer.fused_gdn_verify_fallbacks == 0 for layer in layers)
    logits, _ = _serial(model, copy.deepcopy(bases[0]), _tokens([3])[0])
    np.testing.assert_array_equal(np.asarray(result.logits), np.asarray(logits))


def _stock_core_stand_in(self, qkv, z, b, a, mask, cache):
    """The stock multi-token core: not the decode law on CPU."""
    out = self._recurrent_core(qkv, z, b, a, mask, cache, dtype=qkv.dtype)
    return out.reshape(qkv.shape[0], qkv.shape[1], -1)


def _serial_equivalent_stand_in(self, qkv, z, b, a, mask, cache):
    """A "kernel" that reproduces decode exactly (one-token steps inside)."""
    scratch = QHR._scratch(cache[0], cache[1])
    cores, states = [], []
    for position in range(qkv.shape[1]):
        cut = slice(position, position + 1)
        route, core = QHR.ordinary_one_token_step(
            self, qkv[:, cut], z[:, cut], b[:, cut], a[:, cut], scratch, qkv.dtype
        )
        assert route == "stock_decode"
        cores.append(core)
        states.append([scratch[0], scratch[1]])
    cache.record_rollback(
        qkv.shape[1], lambda m, s=states: list(s[int(m) - 1]), [cache[0], cache[1]]
    )
    cache[0], cache[1] = states[-1]
    cache.advance(qkv.shape[1])
    return mx.concatenate(cores, axis=1)


def _install_qwen38_batch_verify_stand_in(monkeypatch, *, exact):
    """Install a CPU stand-in with the batched kernel's exact ABI."""

    from mlx2.runtime.models import qwen38_fused_gdn as Q38

    active = []
    original = Q38.GatedDeltaNet._try_fused_batch_verify

    def invoke(layer, *args, **kwargs):
        active.append(layer)
        try:
            return original(layer, *args, **kwargs)
        finally:
            active.pop()

    def build(
        qkv, z, b, a, conv_state, conv_weight, A_log, dt_bias,
        recurrent_state, norm_weight, norm_eps, row_steps, **kwargs,
    ):
        del conv_weight, A_log, dt_bias, norm_weight, norm_eps, kwargs
        layer = active[-1]
        steps = qkv.shape[1]
        lane_outputs = []
        lane_finals = []
        lane_state_snapshots = []
        lane_conv_snapshots = []
        for lane, valid in enumerate(row_steps):
            scratch = QHR._scratch(
                conv_state[lane:lane + 1], recurrent_state[lane:lane + 1]
            )
            outputs = []
            states = []
            for position in range(valid):
                cut = slice(position, position + 1)
                route, value = QHR.ordinary_one_token_step(
                    layer,
                    qkv[lane:lane + 1, cut],
                    z[lane:lane + 1, cut],
                    b[lane:lane + 1, cut],
                    a[lane:lane + 1, cut],
                    scratch,
                    qkv.dtype,
                    probe=True,
                )
                assert route == "stock_decode"
                outputs.append(value)
                states.append((scratch[0], scratch[1]))
            zero = mx.zeros_like(outputs[0])
            lane_outputs.append(
                mx.concatenate(outputs + [zero] * (steps - valid), axis=1)
            )
            lane_finals.append(states[-1])
            state_snaps = []
            conv_snaps = []
            for position in range(steps - 1):
                if position < valid - 1:
                    conv_value, state_value = states[position]
                else:
                    # Deliberately invalid padding: zero/full rollback must
                    # not gather an unwritten intermediate snapshot.
                    conv_value = mx.full_like(conv_state[lane:lane + 1], mx.nan)
                    state_value = mx.full_like(
                        recurrent_state[lane:lane + 1], mx.nan
                    )
                conv_snaps.append(conv_value)
                state_snaps.append(state_value)
            lane_conv_snapshots.append(mx.stack(conv_snaps, axis=1))
            lane_state_snapshots.append(mx.stack(state_snaps, axis=1))
        conv = mx.concatenate([value[0] for value in lane_finals], axis=0)
        state = mx.concatenate([value[1] for value in lane_finals], axis=0)
        if not exact:
            state = state + 1
        return (
            mx.concatenate(lane_outputs, axis=0),
            conv,
            state,
            mx.concatenate(lane_state_snapshots, axis=0),
            mx.concatenate(lane_conv_snapshots, axis=0),
        )

    monkeypatch.setattr(QHR, "_fused_runtime_supported", lambda: True)
    monkeypatch.setattr(Q38.GatedDeltaNet, "_try_fused_batch_verify", invoke)
    monkeypatch.setattr(Q38.kernels, "fused_gdn_runtime_supported", lambda: True)
    monkeypatch.setattr(Q38.kernels, "served_silu_refusal", lambda: None)
    monkeypatch.setattr(
        Q38.verify_kernels,
        "admit_qwen4_fused_gdn_batch_verify",
        lambda **kwargs: Q38.verify_kernels.FusedGdnAdmission(True, "eligible"),
    )
    monkeypatch.setattr(
        Q38.verify_kernels,
        "probe_qwen4_fused_gdn_batch_verify",
        lambda *args, **kwargs: 32,
    )
    monkeypatch.setattr(
        Q38.verify_kernels, "qwen4_fused_gdn_batch_verify", build
    )


@pytest.mark.parametrize("canary_passes", [True, False])
def test_qwen38_batched_fused_gdn_ast_route_is_exact_and_canary_gated(
    canary_passes, monkeypatch
):
    """B4 ragged AST engagement is exact; a failed canary stays serial."""

    from mlx2.runtime.models import qwen38_fused_gdn as Q38
    from mlx2.runtime.row_exact.canary import CANARIES

    CANARIES.reset()
    _install_qwen38_batch_verify_stand_in(
        monkeypatch, exact=canary_passes
    )
    model = tiny_target()
    for _, module in model.named_modules():
        if isinstance(module, Q38.GatedDeltaNet):
            module.set_fused_gdn_enabled(True)
    model.configure_target_verify_row_exact(True)
    lengths = [4, 3, 2, 1]
    accepted = [2, 3, 1, 1]
    bases = _bases(model, 4)
    tokens = _tokens(lengths)
    _owner, transaction, result = _launch(model, bases, lengths)
    assert result.row_exact is True, result.detail
    routes = model.external_execution_receipt["target_verify_row_exact"]["routes"]
    layers = [
        module
        for _, module in model.named_modules()
        if isinstance(module, Q38.GatedDeltaNet)
    ]
    if canary_passes:
        assert result.detail["gdn_route"] == "fused_batch_verify"
        assert routes["gdn"] == {
            "fused_batch_verify_decode_parity": len(layers) * 16
        }
        assert "gdn_canary" not in routes
        assert all(
            layer.fused_gdn_counters["batch_verify_calls"] == 1
            and layer.fused_gdn_counters["batch_verify_tokens"] == 10
            for layer in layers
        )
    else:
        assert result.detail["gdn_route"] == "serial_one_token"
        assert routes["gdn_canary"] == {
            "batch_verify_differs_from_decode": len(layers)
        }
        assert all(
            layer.fused_gdn_counters["batch_verify_calls"] == 0
            and layer.fused_gdn_counters["batch_verify_rows"] == 0
            and layer.fused_gdn_counters["batch_verify_tokens"] == 0
            for layer in layers
        )
    for lane, valid in enumerate(lengths):
        logits, taps = _serial(
            model, copy.deepcopy(bases[lane]), tokens[lane][:valid]
        )
        np.testing.assert_array_equal(
            np.asarray(result.logits[lane:lane + 1, :valid]),
            np.asarray(logits),
        )
        np.testing.assert_array_equal(
            np.asarray(result.features[lane:lane + 1, :valid]),
            np.asarray(taps),
        )
    committed = transaction.commit(accepted)
    for lane, kept in enumerate(accepted):
        reference = copy.deepcopy(bases[lane])
        _serial(model, reference, tokens[lane][:kept])
        np.testing.assert_array_equal(
            _next_logits(model, committed[lane]), _next_logits(model, reference)
        )
    CANARIES.reset()


@pytest.mark.parametrize(
    "stand_in, canary_passes",
    [(_stock_core_stand_in, False), (_serial_equivalent_stand_in, True)],
    ids=["stock-core-rejected", "decode-equivalent-admitted"],
)
def test_gdn_fused_verify_runs_only_under_a_passing_canary(
    stand_in, canary_passes, monkeypatch
):
    """A stand-in fused verify exercises the canary-gated route: a stand-in
    that is not the decode law is rejected (serial route), one that is
    admitted (fused route).  Either way the launch is bit-identical to
    serial decode, partial commit included."""
    from mlx2.runtime.models import qwen38_fused_gdn as Q38
    from mlx2.runtime.row_exact.canary import CANARIES

    original = Q38.GatedDeltaNet._try_fused_decode

    def fused(self, qkv, z, b, a, mask, cache, *, account=True):
        if qkv.shape[1] > 1 and bool(getattr(cache, "speculating", False)):
            out = stand_in(self, qkv, z, b, a, mask, cache)
            if account:
                self.fused_gdn_counters["verify_calls"] += 1
            return self.out_proj(out)
        return original(self, qkv, z, b, a, mask, cache, account=account)

    monkeypatch.setattr(Q38.GatedDeltaNet, "_try_fused_decode", fused)
    monkeypatch.setattr(
        Q38.GatedDeltaNet,
        "row_exact_fused_verify",
        lambda self, lanes, steps: "verify" if lanes == 1 and steps > 1 else None,
    )
    monkeypatch.setattr(QHR, "_fused_runtime_supported", lambda: True)
    CANARIES.reset()
    model = tiny_target()
    model.configure_target_verify_row_exact(True)
    bases = _bases(model, 1)
    tokens = _tokens([4])[0]
    _owner, transaction, result = _launch(model, bases, [4])
    assert result.row_exact is True, result.detail
    routes = model.external_execution_receipt["target_verify_row_exact"]["routes"]
    verdicts = [
        entry["exact"]
        for key, entry in CANARIES._results.items()
        if "qwen_hybrid_gdn_verify_vs_decode" in key
    ]
    assert verdicts == [canary_passes]
    if canary_passes:
        assert result.detail["gdn_route"] == "fused_verify"
        assert routes["gdn"] == {"fused_verify_decode_parity": 6 * 4}
        assert "gdn_canary" not in routes
    else:
        assert result.detail["gdn_route"] == "serial_one_token"
        assert routes["gdn_canary"] == {"verify_differs_from_decode": 6}
    logits, taps = _serial(model, copy.deepcopy(bases[0]), tokens)
    np.testing.assert_array_equal(np.asarray(result.logits), np.asarray(logits))
    np.testing.assert_array_equal(np.asarray(result.features), np.asarray(taps))
    committed = transaction.commit([2])[0]
    reference = copy.deepcopy(bases[0])
    _serial(model, reference, tokens[:2])
    np.testing.assert_array_equal(
        _next_logits(model, committed), _next_logits(model, reference)
    )
    CANARIES.reset()


# -- end to end: AST on the external DFlash2 engine ---------------------------
def _ast_batch(target, *, lanes, monkeypatch, oracle=True, planner=False, auto=False):
    from mlx2.adapters.proposal_path_sources import (
        SourceContinuation,
        build_continuation_drafter,
    )
    from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    mx.random.seed(8)
    draft = DFlash2DraftModel(tiny_draft_args()).bind(target)
    mx.eval(draft.parameters())
    target.configure_target_verify_row_exact(True)
    wrapper = build_continuation_drafter(
        target,
        draft,
        {"sources": ["external"], "limit": 15, "ast_planner": planner},
        target_revision="a" * 64,
        draft_revision="b" * 64,
        tokenizer_revision="a" * 64,
        session_revision="c" * 64,
    )
    references = {}

    def greedy(history, depth):
        key = tuple(history)
        if key not in references:
            references[key] = ordinary_greedy(target, list(history), 16)
        return tuple(references[key][:depth])

    def four_paths(context, _limit):
        # The ordinary greedy continuation ranks second, so a non-primary
        # confidence-tree branch must win; the others diverge at depth 1.
        exact = greedy([*context.history, context.anchor], context.depth)

        def changed(delta):
            tokens = list(exact)
            if len(tokens) > 1:
                tokens[1] = (tokens[1] + delta) % 120
            return tuple(tokens)

        confidence = (0.9, 0.8) + (0.1,) * max(0, context.depth - 2)
        confidence = confidence[: context.depth]
        return (
            SourceContinuation(changed(1), confidence, 4.0, "forced"),
            SourceContinuation(exact, confidence, 3.0, "forced"),
            SourceContinuation(changed(2), confidence, 2.0, "forced"),
            SourceContinuation(changed(3), confidence, 1.0, "forced"),
        )

    if oracle:
        monkeypatch.setitem(wrapper.providers, "external", four_paths)
    packing = (
        {
            "completion_batch_size": lanes,
            "progressive_cross_session_pack_rows": 8,
            "progressive_cross_session_pack_mode": "ordered_live_frontier",
        }
        if lanes > 1
        else {"completion_batch_size": 1}
    )
    return ExternalDraftBatchGenerator(
        target,
        draft_model=wrapper,
        binding="qwen-hybrid-ast-cpu",
        num_draft=7,
        prefill_step_size=4,
        ready_drain="all",
        continuation_pool=wrapper.policy,
        continuation_pool_ast_bridge=True,
        progressive_verification_tile=6 if auto else 3,
        **({"progressive_verification_tile_policy": "auto",
            "progressive_terminal_input_widths": target.progressive_external_exact_terminal_input_widths}
           if auto else {}),
        multilane_draft_cap=4,
        progressive_alternative_branches=4,
        progressive_alternative_strategy="confidence_tree",
        progressive_alternative_depth_policy="confidence_tail",
        **packing,
    )


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("auto", [False, True])
def test_adaptive_ast_mixed_budgets_retire_and_preserve_ordinary_output(factory, monkeypatch, auto):
    target = factory()
    batch = _ast_batch(target, lanes=3, monkeypatch=monkeypatch, oracle=False, planner=True, auto=auto)
    limits = [1, 7, 23]
    uids = batch.insert(PROMPTS[:3], max_tokens=limits,
                        sampling_configs=[{"sampling_temp": 0}] * 3)
    plans = []
    original = batch.ast_execution_planner.source_budgets

    def observe(work):
        result = original(work)
        plans.append(result)
        return result

    monkeypatch.setattr(batch.ast_execution_planner, "source_budgets", observe)
    got, final = drain(batch)
    target.configure_target_verify_row_exact(False)
    for uid, prompt, limit in zip(uids, PROMPTS, limits):
        assert got[uid] == ordinary_greedy(target, prompt, limit)
        receipt = final[uid].speculative_receipt["ast_execution_plan"]
        assert receipt["qualified"] is False
        assert receipt["binding"]
    assert plans
    assert any(len(p) == 1 and p[0].uid == uids[-1] for p in plans)
    if auto:
        assert max(batch.ast_execution_planner.tiles) == 6
        assert all(p.depth <= 4 for group in plans if len(group) > 1 for p in group)
        assert "depth_not_above_tile" not in final[uids[-1]].speculative_receipt["progressive_verification"].get(
            "ordered_alternatives", {}).get("primary_fallback_reasons", {})
    assert final[uids[-1]].speculative_receipt["ast_execution_plan"]["cost_samples"] > 0


def test_compaction_scope_skips_padding_without_escaping_its_forward():
    from mlx2.runtime import row_exact_verify as rev
    target = tiny_qwen36()
    target.configure_target_verify_row_exact(True)
    bases = _bases(target, 4)
    _, tx, _ = _launch(target, bases, [3, 3, 3, 4], compact=True)
    routes = target.external_execution_receipt["target_verify_row_exact"]["routes"]
    counts = routes["valid_row_compaction"]
    assert counts["projections_padding_skipped"] > 0
    assert counts["moe_padding_skipped"] > 0
    assert rev._VALID_ROWS.get() is None
    tx.abort()


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize(
    "lanes, oracle",
    [(1, True), (3, True), (3, False)],
    ids=["b1-oracle-paths", "b3-oracle-paths", "b3-random-drafter-packed"],
)
def test_ast_engine_greedy_equals_ordinary_decode(factory, lanes, oracle, monkeypatch):
    """Exact AST end to end: greedy output equals plain ordinary greedy decode.

    The oracle provider ranks the true continuation second among four
    confidence-tree paths (alternatives must win); the random tiny drafter
    leaves lanes in step, so cross-session packing and lane-exact cohort
    launches engage."""
    target = factory()
    batch = _ast_batch(target, lanes=lanes, monkeypatch=monkeypatch, oracle=oracle)
    prompts = PROMPTS[:lanes]
    uids = batch.insert(
        prompts, max_tokens=[14] * lanes, sampling_configs=[{"sampling_temp": 0}] * lanes
    )
    got, final = drain(batch)
    # Every target forward of the run (progressive, lane-exact cohort,
    # standard verify, plain rounds) went through the executor exactly.
    route_receipt = target.external_execution_receipt["target_verify_row_exact"]
    assert route_receipt["nonexact_launches"] == 0
    assert route_receipt["row_exact_launches"] > 0
    target.configure_target_verify_row_exact(False)
    for uid, prompt in zip(uids, prompts):
        assert got[uid] == ordinary_greedy(target, prompt, 14)
        receipt = final[uid].speculative_receipt
        assert receipt["progressive_verification"]["observed_used"] is True
        assert receipt["continuation_pool"]["ast_observed_used"] is True
        if oracle:
            assert receipt["accepted"] > 0
    stats = batch.scheduler_stats
    assert stats["external_progressive_verify_launches"] > 0
    assert stats["external_progressive_target_executor_row_exact_launches"] == (
        stats["external_progressive_target_executor_adapter_owned_launches"]
    )
    assert stats["external_progressive_target_executor_row_exact_launches"] > 0
    assert stats["external_progressive_alternative_branches"] > 0
    if not oracle:
        assert stats["external_progressive_cross_session_ast_packs"] > 0
        assert stats["external_progressive_verify_batched_rounds"] > 0
    route = target.__dict__.get("_mlx2_qwen_hybrid_row_exact")
    assert route is None  # deselected above, after the run
