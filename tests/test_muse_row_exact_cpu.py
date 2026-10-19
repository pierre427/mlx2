"""CPU transaction gates for Muse's opt-in row-exact target verifier."""

from __future__ import annotations

import copy

import mlx.core as mx
import pytest

from mlx2.adapters.muse_glimmer_config import ModelArgs
from mlx2.runtime.models.muse_glimmer import Model
from mlx2.runtime.progressive_external_verify import (
    PROGRESSIVE_FIDELITY_EXACT,
    PROGRESSIVE_TARGET_PROTOCOL_VERSION,
    execute_progressive_target_rows,
)
from mlx2.runtime.segmented_rotating_kv import SegmentedKVRows


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def model_and_cache():
    mx.random.seed(18)
    model = Model(
        ModelArgs(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=128,
            sliding_window=4,
            max_position_embeddings=2048,
        )
    )
    model.eval()
    mx.eval(model.parameters())
    cache = model.make_cache()
    for token in (1, 2, 3, 4, 5, 6):
        mx.eval(model(mx.array([[token]]), cache=cache))
    return model, cache


def assert_cache_equal(actual, expected):
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        assert type(got) is type(want)
        assert got.meta_state == want.meta_state
        assert got.offset == want.offset
        if got.keys is None:
            assert want.keys is None and want.values is None
            continue
        if hasattr(got, "_temporal_order"):
            got_values = (got._temporal_order(got.keys), got._temporal_order(got.values))
            want_values = (
                want._temporal_order(want.keys),
                want._temporal_order(want.values),
            )
        else:
            got_values = (got.keys[..., : got.offset, :], got.values[..., : got.offset, :])
            want_values = (
                want.keys[..., : want.offset, :],
                want.values[..., : want.offset, :],
            )
        assert all(
            mx.array_equal(a, b)
            for a, b in zip(got_values, want_values, strict=True)
        )


@pytest.mark.parametrize("accepted", [0, 1, 3, 7])
def test_row_exact_matches_serial_taps_logits_and_partial_commit(accepted):
    model, base = model_and_cache()
    tokens = mx.array([[11, 12, 13, 14, 15, 16, 17]])
    serial_cache = copy.deepcopy(base)
    serial_logits, serial_taps = [], []
    for position in range(tokens.shape[1]):
        logits, taps = model.forward_with_taps(
            tokens[:, position : position + 1], serial_cache, (0, 3)
        )
        serial_logits.append(logits)
        serial_taps.append(taps)
    serial_logits = mx.concatenate(serial_logits, axis=1)
    serial_taps = mx.concatenate(serial_taps, axis=1)

    expected_cache = copy.deepcopy(base)
    for position in range(accepted):
        mx.eval(model(tokens[:, position : position + 1], cache=expected_cache))

    candidate_cache = copy.deepcopy(base)
    transaction = SegmentedKVRows([candidate_cache]).begin([tokens.shape[1]])
    model.configure_target_verify_row_exact(True)
    logits, taps = model.forward_with_taps(tokens, transaction.caches, (0, 3))
    mx.eval(logits, taps)
    committed = transaction.commit([accepted])[0]
    model.configure_target_verify_row_exact(False)

    assert mx.array_equal(logits, serial_logits)
    assert mx.array_equal(taps, serial_taps)
    assert_cache_equal(committed, expected_cache)
    assert mx.array_equal(
        model(mx.array([[19]]), cache=copy.deepcopy(committed)),
        model(mx.array([[19]]), cache=copy.deepcopy(expected_cache)),
    )
    receipt = model.external_execution_receipt
    assert receipt is None


def test_row_exact_receipt_marks_observed_use_without_qualification():
    model, cache = model_and_cache()
    model.configure_target_verify_row_exact(True)
    tx = SegmentedKVRows([copy.deepcopy(cache)]).begin([2])
    model.forward_with_taps(mx.array([[7, 8]]), tx.caches, (0, 3))
    tx.abort()
    receipt = model.external_execution_receipt["target_verify_row_exact"]
    assert receipt["implemented"] and receipt["selected"]
    assert receipt["observed_used"] and receipt["executed_forwards"] == 1
    assert receipt["physical_query_rows"] == 2
    assert not receipt["qualified"] and not receipt["performance_claim"]


@pytest.mark.parametrize("width", [1, 4])
def test_progressive_target_rows_are_adapter_owned_and_transaction_exact(width):
    model, base = model_and_cache()
    tokens = mx.array(
        [[7 + lane, 11 + lane] for lane in range(width)]
    )
    rows = [copy.deepcopy(base) for _ in range(width)]
    owner = SegmentedKVRows(rows)
    transaction = owner.begin([2] * width)
    model.configure_target_verify_row_exact(True)

    logits, taps, detail = execute_progressive_target_rows(
        model,
        tokens,
        transaction.caches,
        (0, 3),
        protocol=PROGRESSIVE_TARGET_PROTOCOL_VERSION,
        fidelity=PROGRESSIVE_FIDELITY_EXACT,
    )
    mx.eval(logits, taps)
    committed = transaction.commit([1] * width)

    assert detail["execution"] == "adapter_owned"
    assert detail["row_exact"] is True
    assert detail["adapter_detail"] == {
        "algorithm": "muse-native-b1s1-causal-prefix-v2",
        "physical_width": width,
        "token_width": 2,
        "padding_stage_rows_skipped": 0,
    }
    for lane in range(width):
        reference = copy.deepcopy(base)
        expected_logits, expected_taps = [], []
        for token in tokens[lane].tolist():
            row_logits, row_taps = model.forward_with_taps(
                mx.array([[token]]), reference, (0, 3)
            )
            expected_logits.append(row_logits)
            expected_taps.append(row_taps)
        assert mx.array_equal(
            logits[lane : lane + 1], mx.concatenate(expected_logits, axis=1)
        )
        assert mx.array_equal(
            taps[lane : lane + 1], mx.concatenate(expected_taps, axis=1)
        )
        committed_reference = copy.deepcopy(base)
        mx.eval(model(mx.array([[tokens[lane, 0].item()]]), cache=committed_reference))
        assert_cache_equal(committed[lane], committed_reference)


@pytest.mark.parametrize("lengths", [(1, 3), (3, 1, 2)])
def test_compacted_valid_rows_keep_exact_logits_taps_and_partial_cache(lengths):
    model, base = model_and_cache()
    model.configure_target_verify_row_exact(True)
    tokens = mx.array([[7 + lane, 11 + lane, 17 + lane] for lane in range(len(lengths))])
    transaction = SegmentedKVRows([copy.deepcopy(base) for _ in lengths]).begin(lengths)
    logits, taps, detail = execute_progressive_target_rows(
        model, tokens, transaction.caches, (0, 3),
        protocol=PROGRESSIVE_TARGET_PROTOCOL_VERSION,
        fidelity=PROGRESSIVE_FIDELITY_EXACT, input_lengths=lengths,
    )
    mx.eval(logits, taps)
    accepted = [max(0, length - 1) for length in lengths]
    committed = transaction.commit(accepted)
    assert detail["adapter_detail"]["padding_stage_rows_skipped"] > 0
    for lane, length in enumerate(lengths):
        reference = copy.deepcopy(base)
        expected_logits, expected_taps = [], []
        for token in tokens[lane, :length].tolist():
            out, features = model.forward_with_taps(mx.array([[token]]), reference, (0, 3))
            expected_logits.append(out)
            expected_taps.append(features)
        assert mx.array_equal(logits[lane:lane + 1, :length], mx.concatenate(expected_logits, axis=1))
        assert mx.array_equal(taps[lane:lane + 1, :length], mx.concatenate(expected_taps, axis=1))
        cache = copy.deepcopy(base)
        for token in tokens[lane, :accepted[lane]].tolist():
            mx.eval(model(mx.array([[token]]), cache=cache))
        assert_cache_equal(committed[lane], cache)


def test_progressive_target_rows_refuse_before_cache_mutation_when_deselected():
    model, base = model_and_cache()
    before = copy.deepcopy(base)
    with pytest.raises(RuntimeError, match="not selected"):
        model.progressive_external_target_rows(
            mx.array([[7, 8]]), base, (0, 3)
        )
    assert_cache_equal(base, before)


# v2 stacking: row-independent ops run as one launch over every verify row
# when the parity canary agrees with the v1 per-row form.


def _serial(model, base, tokens):
    logits, taps = [], []
    for lane in range(tokens.shape[0]):
        cache = copy.deepcopy(base)
        lane_logits, lane_taps = [], []
        for token in tokens[lane].tolist():
            row_logits, row_taps = model.forward_with_taps(
                mx.array([[token]]), cache, (0, 3)
            )
            lane_logits.append(row_logits)
            lane_taps.append(row_taps)
        logits.append(mx.concatenate(lane_logits, axis=1))
        taps.append(mx.concatenate(lane_taps, axis=1))
    return mx.concatenate(logits), mx.concatenate(taps)


def _verify(model, base, tokens):
    width, length = tokens.shape
    transaction = SegmentedKVRows([copy.deepcopy(base) for _ in range(width)]).begin(
        [length] * width
    )
    model.configure_target_verify_row_exact(True)
    try:
        logits, taps = model.forward_with_taps(tokens, transaction.caches, (0, 3))
        mx.eval(logits, taps)
    finally:
        transaction.abort()
        model.configure_target_verify_row_exact(False)
    return logits, taps


@pytest.mark.parametrize("verdict", [False, None])
@pytest.mark.parametrize("width,length", [(1, 4), (3, 3)])
def test_stacked_rows_match_serial_under_any_canary_verdict(
    monkeypatch, verdict, width, length
):
    """``None`` keeps this build's real canaries (CPU rms_norm is not
    row-invariant at every row count, so both routes run); ``False`` forces
    the v1 per-row form everywhere.  Either way every row is serial-exact."""
    from mlx2.runtime.models import muse_glimmer as MG
    from mlx2.runtime.row_exact.canary import CANARIES

    model, base = model_and_cache()
    if verdict is not None:
        monkeypatch.setattr(CANARIES, "exact", lambda *a, **k: verdict)
    MG.ROW_EXACT_ROUTES.clear()
    tokens = mx.array([[21 + lane + 3 * i for i in range(length)] for lane in range(width)])
    logits, taps = _verify(model, base, tokens)
    want_logits, want_taps = _serial(model, base, tokens)
    assert mx.array_equal(logits, want_logits)
    assert mx.array_equal(taps, want_taps)
    routes = dict(MG.ROW_EXACT_ROUTES)
    if verdict is False:
        assert not any(key.endswith(":batched") for key in routes)
        assert routes.get("muse_centered_rms_norm:per_row_canary", 0) > 0
    else:
        assert any(key.endswith(":batched") for key in routes)


def test_row_exact_rows_is_one_call_when_canary_passes(monkeypatch):
    from mlx2.runtime.models import muse_glimmer as MG
    from mlx2.runtime.row_exact.canary import CANARIES

    calls = []

    def op(value):
        calls.append(tuple(value.shape))
        return value * 2

    value = mx.ones((2, 3, 4, 8))
    MG.ROW_EXACT_ROUTES.clear()
    monkeypatch.setattr(CANARIES, "exact", lambda *a, **k: True)
    out = MG._row_exact_rows(op, value, ("t_op",))
    assert calls == [(2, 3, 4, 8)] and out.shape == value.shape
    calls.clear()
    monkeypatch.setattr(CANARIES, "exact", lambda *a, **k: False)
    MG._row_exact_rows(op, value, ("t_op",))
    assert len(calls) == 24 and set(calls) == {(1, 1, 8)}
    assert MG.ROW_EXACT_ROUTES["t_op:batched"] == 1
    assert MG.ROW_EXACT_ROUTES["t_op:per_row_canary"] == 1


@pytest.mark.parametrize("array_offsets", [False, True])
def test_batched_rope_equals_v1_per_position_rope(array_offsets):
    model, _ = model_and_cache()
    attention = next(
        layer.self_attn for layer in model.layers if layer.self_attn.use_rope
    )
    width = 3 if array_offsets else 1
    offsets = [5, 9, 130][:width]
    value = mx.random.normal((width, 2, 4, 8), key=mx.random.key(3))
    batched = attention._rope_batched(value, offsets, array_offsets)
    reference = attention._rope_per_position(value, offsets, array_offsets)
    assert mx.array_equal(batched, reference)


def test_receipt_reports_stage_routes():
    from mlx2.runtime.models import muse_glimmer as MG

    model, base = model_and_cache()
    MG.ROW_EXACT_ROUTES.clear()
    _verify(model, base, mx.array([[7, 8, 9]]))
    model.configure_target_verify_row_exact(True)
    routes = model.external_execution_receipt["target_verify_row_exact"]["routes"]
    model.configure_target_verify_row_exact(False)
    # CPU: the qmv kernel declines (device), so projections run per row.
    assert routes["projection_group:per_row"] > 0
    assert any(key.startswith("muse_qk_rms_norm:") for key in routes)
    assert any(key.startswith("rope:") for key in routes)


def test_lane_exact_projections_call_the_module_itself(monkeypatch):
    """Under an exact-mode lane law (installed only on Metal), verify rows
    take each projection's own call, the same lane launch one-token decode
    makes; the row-exact qmv (stock law) is not used for those modules."""
    from mlx2.runtime.models import muse_glimmer as MG
    from mlx2.runtime.models import row_exact_qmv as REQ

    model, base = model_and_cache()
    monkeypatch.setattr(MG, "_lane_exact", lambda module, value: True)

    def refuse(*_a, **_k):
        raise AssertionError("row_exact_qmv ran under the lane law")

    monkeypatch.setattr(REQ, "quantized_linear", refuse)
    monkeypatch.setattr(REQ, "quantized_linears", refuse)
    MG.ROW_EXACT_ROUTES.clear()
    _verify(model, base, mx.array([[7, 8, 9]]))
    routes = dict(MG.ROW_EXACT_ROUTES)
    assert routes["projection_group:lane_module"] == 2 * len(model.layers)
    assert routes["projection:lane_module"] == 2 * len(model.layers) + 1


def test_lane_exact_detects_only_exact_mode_installs():
    from types import SimpleNamespace

    from mlx2.runtime.models import muse_glimmer as MG

    model, _ = model_and_cache()
    value = mx.zeros((1, 4, 16))
    # Not lane-installed (CPU): the stock-law row-exact qmv applies.
    assert MG._lane_exact(model.layers[0].mlp.down_proj, value) is False
    fake = SimpleNamespace()
    assert MG._lane_exact(fake, value) is False
