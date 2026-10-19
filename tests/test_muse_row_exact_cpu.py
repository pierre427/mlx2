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


def test_attention_window_selection_is_explicit_and_receipted():
    model, _cache = model_and_cache()
    with pytest.raises(ValueError, match="requires target_verify_row_exact"):
        model.configure_attention_window_row_exact(True)
    model.configure_target_verify_row_exact(True)
    model.configure_attention_window_row_exact(True)
    assert all(
        layer.self_attn.attention_window_row_exact for layer in model.layers
    )
    receipt = model.external_execution_receipt["target_verify_row_exact"]
    assert receipt["attention_window_row_exact"] is True
    assert receipt["implemented"] and receipt["selected"]
    assert not receipt["qualified"] and not receipt["observed_used"]


def test_muse_attention_window_keeps_serial_kv_growth(monkeypatch):
    from mlx2.runtime.models import muse_attn_window as MW
    from mlx2.runtime.models.cache import KVCache

    model, _base = model_and_cache()
    model.configure_target_verify_row_exact(True)
    model.configure_attention_window_row_exact(True)
    attention = next(
        layer.self_attn for layer in model.layers if not layer.self_attn.is_sliding
    )
    valid = 2
    q = mx.zeros((1, attention.n_heads, valid, attention.head_dim))
    k = mx.zeros((1, attention.n_kv_heads, valid, attention.head_dim))
    v = mx.zeros_like(k)
    cache = KVCache()
    cache.keys = mx.zeros(
        (1, attention.n_kv_heads, 256, attention.head_dim)
    )
    cache.values = mx.zeros_like(cache.keys)
    cache.offset = 255

    monkeypatch.setattr(MW, "window_supported", lambda *a, **k: None)
    monkeypatch.setattr(
        attention, "_attention_window_canary", lambda **_kwargs: True
    )
    seen = {}

    def window(queries, keys, values, scale, *, n_first, masks=None):
        seen.update(
            offset=cache.offset,
            capacity=cache.keys.shape[2],
            keys=keys.shape[2],
            n_first=n_first,
            masks=masks,
        )
        return mx.zeros((1, valid, attention.n_heads * attention.head_dim))

    monkeypatch.setattr(MW, "window_sdpa", window)
    output, fetched, reason = attention._row_exact_attention_window(
        q, k, v, cache, valid
    )
    mx.eval(output, *fetched)
    assert reason is None
    assert cache.offset == 257
    assert seen == {
        "offset": 257,
        "capacity": 512,
        "keys": 257,
        "n_first": 256,
        "masks": None,
    }


def test_muse_attention_window_declines_before_mutation_and_propagates_after(
    monkeypatch,
):
    from mlx2.runtime.models import muse_attn_window as MW
    from mlx2.runtime.models.cache import KVCache

    model, _base = model_and_cache()
    model.configure_target_verify_row_exact(True)
    model.configure_attention_window_row_exact(True)
    attention = next(
        layer.self_attn for layer in model.layers if not layer.self_attn.is_sliding
    )
    valid = 2
    q = mx.zeros((1, attention.n_heads, valid, attention.head_dim))
    k = mx.zeros((1, attention.n_kv_heads, valid, attention.head_dim))
    v = mx.zeros_like(k)
    cache = KVCache()

    monkeypatch.setattr(MW, "window_supported", lambda *a, **k: "plan")
    result = attention._row_exact_attention_window(q, k, v, cache, valid)
    assert result == (None, None, "plan")
    assert cache.offset == 0 and cache.keys is None

    monkeypatch.setattr(MW, "window_supported", lambda *a, **k: None)
    monkeypatch.setattr(
        attention, "_attention_window_canary", lambda **_kwargs: True
    )
    monkeypatch.setattr(
        MW,
        "window_sdpa",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("kernel failed")),
    )
    with pytest.raises(RuntimeError, match="kernel failed"):
        attention._row_exact_attention_window(q, k, v, cache, valid)
    assert cache.offset == 2


def test_muse_attention_stage_accounting_and_fail_closed_cpu_fallback():
    from mlx2.runtime.models import muse_glimmer as MG

    model, base = model_and_cache()
    model.configure_target_verify_row_exact(True)
    model.configure_attention_window_row_exact(True)
    MG.ROW_EXACT_ROUTES.clear()
    transaction = SegmentedKVRows([copy.deepcopy(base)]).begin([2])
    model.forward_with_taps(mx.array([[7, 8]]), transaction.caches, (0, 3))
    transaction.abort()
    routes = model.external_execution_receipt["target_verify_row_exact"]["routes"]
    # CPU refuses the Metal window before mutation; every layer keeps its
    # ordinary serial one-token attention and records why it did so.
    assert routes["attention:serial_b1s1"] == 2 * len(model.layers)
    assert routes["attention_layers:serial_b1s1"] == len(model.layers)
    assert routes["attention_window_fallback:device"] > 0


def test_selected_segmented_window_returns_attention_only_before_batch_gate():
    model, _base = model_and_cache()
    attention = next(
        layer.self_attn for layer in model.layers if not layer.self_attn.is_sliding
    )
    attention.attention_window_row_exact = True
    seen = {}

    class Segmented:
        def row_exact_attention(self, queries, keys, values, **kwargs):
            seen.update(kwargs)
            return queries

    q = mx.zeros((3, attention.n_heads, 2, attention.head_dim))
    k = mx.zeros((3, attention.n_kv_heads, 2, attention.head_dim))
    output, gated = attention._row_exact_attention(q, k, k, Segmented())
    assert output is q
    assert gated is False
    assert callable(seen["window_attention"])
    assert "serial_transform" not in seen


def test_row_exact_attention_applies_one_gate_after_full_batch_concat(monkeypatch):
    from mlx2.runtime.models import muse_glimmer as MG

    model, _base = model_and_cache()
    attention = next(
        layer.self_attn for layer in model.layers if not layer.self_attn.is_sliding
    )
    batch, rows = 3, 2
    q = mx.zeros((batch, rows, attention.n_heads * attention.head_dim))
    k = mx.zeros((batch, rows, attention.n_kv_heads * attention.head_dim))
    v = mx.zeros_like(k)
    gate = mx.arange(batch * rows * attention.n_heads * attention.head_dim)
    gate = gate.reshape(batch, rows, -1).astype(mx.float32) / 17
    attention_rows = mx.arange(
        batch * attention.n_heads * rows * attention.head_dim
    ).reshape(batch, attention.n_heads, rows, attention.head_dim).astype(mx.float32)
    seen = {}

    monkeypatch.setattr(MG, "_row_exact_group", lambda modules, value: (q, k, v, gate))
    monkeypatch.setattr(attention, "_row_exact_rope", lambda qv, kv, cache: (qv, kv))
    monkeypatch.setattr(
        attention,
        "_row_exact_attention",
        lambda qv, kv, vv, cache: (attention_rows, False),
    )

    def project(module, value, *, call=None):
        assert module is attention.o_proj and call is None
        seen["project_input"] = value
        return value

    monkeypatch.setattr(MG, "_row_exact_project", project)
    output = attention(
        mx.zeros((batch, rows, model.args.hidden_size)),
        cache=object(),
        row_exact=True,
    )
    expected = attention_rows.transpose(0, 2, 1, 3).reshape(batch, rows, -1)
    expected = expected * mx.sigmoid(gate)
    assert mx.array_equal(seen["project_input"], expected)
    assert mx.array_equal(output, expected)


def test_muse_d128_window_uses_each_serial_row_plan():
    from mlx2.runtime.models import muse_attn_window as MW

    assert MW.plan_signature(100, 7, 32, 2, device_class="s") == (
        (0, 7, (1, None)),
    )
    # Every row keeps the GQA16 plan of its serial call.  The grouped window
    # never substitutes a GQA32 plan merely because it contains two rows.
    assert MW.plan_signature(1023, 4, 32, 2, device_class="s") == (
        (0, 1, (1, None)),
        (1, 2, (2, 64)),
        (2, 4, (2, 128)),
    )
    assert MW.plan_signature(8192, 3, 32, 2, device_class="s") == (
        (0, 1, (2, 128)),
        (1, 3, (2, 256)),
    )
    assert MW.plan_signature(32768, 2, 32, 2, device_class="s") == (
        (0, 1, (2, 256)),
        (1, 2, (2, 512)),
    )
    assert MW.plan_signature(65536, 2, 32, 2, device_class="s") == (
        (0, 1, (2, 512)),
        (1, 2, (2, 1024)),
    )
    assert MW.plan_signature(9000, 2, 4, 2, device_class="d") == (
        (0, 2, (2, 256)),
    )
    assert MW.MLX_SDPA_DISPATCH_SHA256 == (
        "575e29a36b2d0c653f17b4f55062b644243d2e3dbc362f6725686bef959dd9dd"
    )
    assert MW.MLX_SDPA_VECTOR_SHA256 == (
        "2100a4d1eaa8a524c5147c82c771cad75197495c72daffa03e7ea4c259aebf10"
    )
    assert len(MW.source_identity()) == 64


def test_muse_d128_window_admission_is_separate_from_qwen_d256(monkeypatch):
    from mlx2.runtime.models import muse_attn_window as MW

    monkeypatch.setattr(MW.mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(MW, "_device_class", lambda: "s")
    q = MW.ShapeOnly((1, 32, 2, 128), mx.bfloat16)
    kv = MW.ShapeOnly((1, 2, 65, 128), mx.bfloat16)
    assert MW.window_supported(q, kv, kv, 64) is None
    q256 = MW.ShapeOnly((1, 32, 2, 256), mx.bfloat16)
    kv256 = MW.ShapeOnly((1, 2, 65, 256), mx.bfloat16)
    assert MW.window_supported(q256, kv256, kv256, 64) == "head_dim"


@pytest.mark.parametrize("masked", [False, True])
def test_muse_d128_window_source_uses_physical_n_and_row_masks(masked):
    from mlx2.runtime.models import muse_attn_window as MW

    one = MW._one_pass_source(masked)
    two = MW._two_pass_source(masked)
    for source in (one, two):
        assert "const int N = n_first[0] + row;" in source
        assert "keys_shape[2]" not in source
        assert "@" not in source
        assert ("mask_base[0]" in source) is masked
        assert ("row * n_first[0]" in source) is masked
        assert ("row * (row - 1)" in source) is masked
    assert "threadgroup_position_in_grid.z) % BLOCKS" in two
    assert "threadgroup_position_in_grid.z) / BLOCKS" in two
    assert "thread_position_in_threadgroup.z" not in two


def test_muse_d128_window_launches_plan_runs_with_each_run_first_n(monkeypatch):
    from mlx2.runtime.models import muse_attn_window as MW

    monkeypatch.setattr(MW, "_device_class", lambda: "s")
    calls = []

    def kernel(name):
        def run(*, inputs, output_shapes, output_dtypes, **kwargs):
            first = int(inputs[4].item()) if "chains" in name else None
            calls.append((name, first, tuple(output_shapes)))
            return [
                mx.zeros(shape, dtype=dtype)
                for shape, dtype in zip(output_shapes, output_dtypes)
            ]
        return run

    monkeypatch.setattr(MW, "_kernel", kernel)
    queries = mx.zeros((1, 32, 4, 128), dtype=mx.bfloat16)
    keys = mx.zeros((1, 2, 1026, 128), dtype=mx.bfloat16)
    output = MW.window_sdpa(
        queries, keys, keys, 128 ** -0.5, n_first=1023
    )
    mx.eval(output)
    assert output.shape == (1, 4, 32 * 128)
    chain_calls = [(name, first) for name, first, _ in calls if "chains" in name]
    assert chain_calls == [
        ("one_chains", 1023),
        ("two_chains", 1024),
        ("two_chains", 1025),
    ]


def test_muse_attention_window_canary_key_rejects_stale_context_and_source(
    monkeypatch,
):
    from mlx2.runtime.models import muse_attn_window as MW
    from mlx2.runtime.row_exact.canary import CANARIES

    model, _base = model_and_cache()
    attention = next(
        layer.self_attn for layer in model.layers if not layer.self_attn.is_sliding
    )
    keys = []

    def exact(key, probe, *, rows):
        keys.append(key)
        return True

    monkeypatch.setattr(CANARIES, "exact", exact)
    common = {
        "rows": 2,
        "dtype": mx.bfloat16,
        "masks": None,
        "layout": ("KVCache", 256, 128),
    }
    assert attention._attention_window_canary(n_first=64, **common)
    assert attention._attention_window_canary(n_first=65, **common)
    changed_layout = dict(common, layout=("KVCache", 512, 128))
    assert attention._attention_window_canary(n_first=65, **changed_layout)
    original = MW.source_identity()
    monkeypatch.setattr(MW, "source_identity", lambda: "changed-source")
    assert attention._attention_window_canary(n_first=65, **common)
    assert keys[0] != keys[1] != keys[2]
    assert keys[1] != keys[3]
    assert original in keys[0]
    assert "changed-source" in keys[3]
    assert 64 in keys[0] and 65 in keys[1]
    assert ("KVCache", 256, 128) in keys[0]


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
        "algorithm": "muse-native-b1s1-causal-prefix-v4",
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
