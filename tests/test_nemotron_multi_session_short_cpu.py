"""CPU parity for the adapter-owned multi-session Nemotron AST executor."""

import copy

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows
from mlx2.runtime.models.cache import ArraysCache
from mlx2.runtime.progressive_external_verify import (
    PROGRESSIVE_FIDELITY_EXACT,
    PROGRESSIVE_TARGET_PROTOCOL_VERSION,
    execute_progressive_target_rows,
)
from tests.test_nemotron35_progressive_cpu import (
    close_caches,
    run_cross_session_ast,
)
from tests.test_nemotron_external_taps_cpu import assert_state_equal, tiny_model

CAPTURES = (0, 1, 4)


def _selected_model():
    mx.set_default_device(mx.cpu)
    model = tiny_model()
    model.set_short_segment_policy("layerwise_short_exact_routed_batch")
    for layer in model.layers:
        if layer.block_type == "M":
            layer.mixer.set_conv_policy("fused_short")
            layer.mixer.set_ssm_policy("fused_short")
            layer.mixer.set_projection_policy("short_row_exact")
    return model


def _selected_moe_model(monkeypatch):
    seed = tiny_model()
    args = copy.copy(seed.args)
    args.hybrid_override_pattern = ["E", "M", "*", "E", "M"]
    args.moe_latent_size = 4
    args.moe_intermediate_size = 6
    args.moe_shared_expert_intermediate_size = 7
    args.n_group = 1
    args.n_routed_experts = 4
    args.n_shared_experts = 1
    args.topk_group = 1
    args.num_experts_per_tok = 2
    args.norm_topk_prob = True
    args.routed_scaling_factor = 1.0
    model = type(seed)(args)
    model.set_short_segment_policy("layerwise_short_exact_routed_batch")
    for layer in model.layers:
        if layer.block_type == "M":
            layer.mixer.set_conv_policy("fused_short")
            layer.mixer.set_ssm_policy("fused_short")
            layer.mixer.set_projection_policy("short_row_exact")
        elif layer.block_type == "E":
            monkeypatch.setattr(
                layer.mixer, "_routed_batch_geometry_refusal", lambda _rows: None
            )
    return model


def _prefilled_rows(model, count):
    rows = []
    for index in range(count):
        cache = model.make_cache()
        model(mx.array([[1 + index, 2 + index, 3 + index]]), cache)
        mx.eval([entry.state for entry in cache])
        rows.append(cache)
    return rows


def _reference(model, rows, tokens, lengths):
    outputs = []
    features = []
    committed = []
    for index, length in enumerate(lengths):
        owner = HybridVerifyRows([copy.deepcopy(rows[index])])
        transaction = owner.begin([length])
        logits, taps = model.forward_with_taps(
            tokens[index : index + 1, :length], transaction.caches, CAPTURES
        )
        mx.eval(logits, taps)
        outputs.append(np.asarray(logits))
        features.append(np.asarray(taps))
        committed.append(transaction.commit([length]))
    return outputs, features, [row[0] for row in committed]


@pytest.mark.parametrize(
    "lengths",
    [
        (4, 4),
        (4, 3),
        (4, 4, 4, 4),
        (4, 3, 2, 4),
        (4, 1, 2, 4),
        (4, 4, 4, 4, 4, 4, 4, 4),
    ],
)
def test_multi_session_short_exact_matches_private_b1_rows(lengths):
    model = _selected_model()
    rows = _prefilled_rows(model, len(lengths))
    reference_rows = copy.deepcopy(rows)
    width = max(lengths)
    tokens = mx.array(
        [
            [4 + index + position for position in range(width)]
            for index in range(len(lengths))
        ]
    )
    expected_logits, expected_features, expected_caches = _reference(
        model, reference_rows, tokens, lengths
    )

    transaction = HybridVerifyRows(rows).begin(lengths)
    logits, features, detail = execute_progressive_target_rows(
        model,
        tokens,
        transaction.caches,
        CAPTURES,
        protocol=PROGRESSIVE_TARGET_PROTOCOL_VERSION,
        fidelity=PROGRESSIVE_FIDELITY_EXACT,
    )
    mx.eval(logits, features)
    actual_caches = transaction.commit(lengths)

    assert detail["execution"] == "adapter_owned"
    assert detail["row_exact"] is True
    assert detail["adapter_detail"] == {
        "route": "multi_session_mamba_short_exact",
        "sessions": len(lengths),
        "logical_rows": sum(lengths),
        "physical_width": width,
        "ragged": len(set(lengths)) > 1,
        "moe_lattice": False,
        "moe_lattice_layers": 0,
        "moe_chunks_per_layer": 0,
        "moe_chunk_plan": "",
        "moe_chunk_row_cap": 16,
    }
    for index, length in enumerate(lengths):
        np.testing.assert_allclose(
            np.asarray(logits[index : index + 1, :length]),
            expected_logits[index],
            atol=2e-6,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(features[index : index + 1, :length]),
            expected_features[index],
            atol=2e-6,
            rtol=1e-5,
        )
        assert_state_equal(actual_caches[index], expected_caches[index])

    status = model.short_segment_status()["counts"]
    assert status["multi_session_short_exact"] == 1
    assert status["multi_session_sessions"] == len(lengths)
    assert status["multi_session_rows"] == sum(lengths)
    assert status["multi_session_ragged"] == int(len(set(lengths)) > 1)


def test_multi_session_short_abort_restores_every_recurrent_and_kv_row():
    model = _selected_model()
    rows = _prefilled_rows(model, 8)
    before = copy.deepcopy(rows)
    transaction = HybridVerifyRows(rows).begin([4] * 8)
    logits, features, _detail = execute_progressive_target_rows(
        model,
        mx.array(
            [
                [4, 5, 6, 7],
                [5, 6, 7, 7],
                [6, 7, 7, 7],
                [7, 8, 9, 10],
                [8, 9, 10, 11],
                [9, 10, 11, 12],
                [10, 11, 12, 13],
                [11, 12, 13, 14],
            ]
        ),
        transaction.caches,
        CAPTURES,
        protocol=PROGRESSIVE_TARGET_PROTOCOL_VERSION,
        fidelity=PROGRESSIVE_FIDELITY_EXACT,
    )
    mx.eval(logits, features)
    transaction.abort()
    for actual, expected in zip(rows, before, strict=True):
        assert_state_equal(actual, expected)
        assert not any(
            entry.speculating for entry in actual if isinstance(entry, ArraysCache)
        )


def test_b8s4_executor_chunks_each_moe_lattice_and_matches_b1(monkeypatch):
    model = _selected_moe_model(monkeypatch)
    lengths = (4,) * 8
    rows = _prefilled_rows(model, 8)
    reference_rows = copy.deepcopy(rows)
    tokens = mx.array(
        [[4 + index + position for position in range(4)] for index in range(8)]
    )
    expected_logits, expected_features, expected_caches = _reference(
        model, reference_rows, tokens, lengths
    )
    transaction = HybridVerifyRows(rows).begin(lengths)
    logits, features, detail = execute_progressive_target_rows(
        model,
        tokens,
        transaction.caches,
        CAPTURES,
        protocol=PROGRESSIVE_TARGET_PROTOCOL_VERSION,
        fidelity=PROGRESSIVE_FIDELITY_EXACT,
    )
    mx.eval(logits, features)
    actual_caches = transaction.commit(lengths)

    assert detail["adapter_detail"] == {
        "route": "multi_session_mamba_short_exact",
        "sessions": 8,
        "logical_rows": 32,
        "physical_width": 4,
        "ragged": False,
        "moe_lattice": True,
        "moe_lattice_layers": 2,
        "moe_chunks_per_layer": 2,
        "moe_chunk_plan": "16+16",
        "moe_chunk_row_cap": 16,
    }
    for index in range(8):
        np.testing.assert_allclose(
            np.asarray(logits[index : index + 1]),
            expected_logits[index],
            atol=2e-6,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(features[index : index + 1]),
            expected_features[index],
            atol=2e-6,
            rtol=1e-5,
        )
        assert_state_equal(actual_caches[index], expected_caches[index])
    for layer in model.layers:
        if layer.block_type == "E":
            status = layer.mixer.routed_batch_status()
            assert status["counts"]["lattice_chunked_calls"] == 1
            assert status["last_lattice"]["physical_chunk_rows"] == [16, 16]


def test_mixed_moe_failure_after_cache_advances_aborts_every_row(monkeypatch):
    model = _selected_moe_model(monkeypatch)
    rows = _prefilled_rows(model, 8)
    before = copy.deepcopy(rows)
    base_offsets = [row[1].offset for row in rows]
    second_moe = [layer for layer in model.layers if layer.block_type == "E"][1]
    observed = {"advanced": False}

    def fail_after_prior_layers(_session_hidden):
        assert all(row[1].offset > base for row, base in zip(rows, base_offsets))
        assert all(row[0]._rollbacks for row in rows)
        observed["advanced"] = True
        raise RuntimeError("injected failure after mamba and kv advances")

    monkeypatch.setattr(second_moe.mixer, "routed_lattice_exact", fail_after_prior_layers)
    transaction = HybridVerifyRows(rows).begin([4] * 8)
    with pytest.raises(RuntimeError, match="after mamba and kv advances"):
        execute_progressive_target_rows(
            model,
            mx.array(
                [[4 + index + position for position in range(4)] for index in range(8)]
            ),
            transaction.caches,
            CAPTURES,
            protocol=PROGRESSIVE_TARGET_PROTOCOL_VERSION,
            fidelity=PROGRESSIVE_FIDELITY_EXACT,
        )
    assert observed["advanced"] is True
    transaction.abort()
    for actual, expected in zip(rows, before, strict=True):
        assert_state_equal(actual, expected)
        assert not any(
            entry.speculating for entry in actual if isinstance(entry, ArraysCache)
        )


def test_mixed_executor_refuses_unsafe_moe_padding_before_cache_advance(monkeypatch):
    from mlx2.runtime.models import switch_layers

    model = _selected_moe_model(monkeypatch)
    rows = _prefilled_rows(model, 2)
    before = copy.deepcopy(rows)
    monkeypatch.setattr(switch_layers, "_RHS_PAD_POLICY", "always")
    transaction = HybridVerifyRows(rows).begin([4, 4])
    with pytest.raises(ValueError, match="moe rhs pad policy always"):
        execute_progressive_target_rows(
            model,
            mx.array([[4, 5, 6, 7], [5, 6, 7, 8]]),
            transaction.caches,
            CAPTURES,
            protocol=PROGRESSIVE_TARGET_PROTOCOL_VERSION,
            fidelity=PROGRESSIVE_FIDELITY_EXACT,
        )
    transaction.abort()
    for actual, expected in zip(rows, before, strict=True):
        assert_state_equal(actual, expected)


def test_selected_executor_drives_four_session_eight_row_ast_pack():
    baseline_batch, baseline = run_cross_session_ast()
    selected_batch, selected = run_cross_session_ast(8, execute=False)
    try:
        selected_batch.model.set_short_segment_policy(
            "layerwise_short_exact_routed_batch"
        )
        for layer in selected_batch.model.layers:
            if layer.block_type == "M":
                layer.mixer.set_conv_policy("fused_short")
                layer.mixer.set_ssm_policy("fused_short")
                layer.mixer.set_projection_policy("short_row_exact")
        selected_batch._round(selected)
        for actual, expected in zip(selected, baseline, strict=True):
            assert actual.history == expected.history
            assert actual.anchor == expected.anchor
            assert actual.rng.snapshot() == expected.rng.snapshot()
            close_caches(actual.cache, expected.cache)
        stats = selected_batch.scheduler_stats
        assert stats["external_progressive_target_executor_adapter_owned_launches"] > 0
        assert (
            stats["external_progressive_target_executor_adapter_owned_launches"]
            == stats["external_progressive_target_executor_row_exact_launches"]
        )
        assert stats["external_progressive_target_executor_nonexact_launches"] == 0
        status = selected_batch.model.short_segment_status()["counts"]
        assert status["multi_session_short_exact"] > 0
        assert status["multi_session_sessions"] >= 4
    finally:
        baseline_batch.close()
        selected_batch.close()


def test_multi_session_short_refuses_unselected_and_unbounded_shapes():
    model = _selected_model()
    rows = _prefilled_rows(model, 2)
    transaction = HybridVerifyRows(rows).begin([3, 3])
    model.set_short_segment_policy("auto")
    with pytest.raises(AttributeError, match="unselected"):
        _executor = model.progressive_external_target_rows
    with pytest.raises(ValueError, match="requires the selected"):
        model._progressive_external_target_rows(
            mx.array([[4, 5, 6], [5, 6, 7]]), transaction.caches, CAPTURES
        )
    transaction.abort()

    model = _selected_model()
    rows = _prefilled_rows(model, 2)
    transaction = HybridVerifyRows(rows).begin([17, 17])
    with pytest.raises(ValueError, match="token width 1..16"):
        model.progressive_external_target_rows(
            mx.array([[4] * 17, [5] * 17]),
            transaction.caches,
            CAPTURES,
        )
    transaction.abort()

