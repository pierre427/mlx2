"""Explicit primary-only AST keeps stock arithmetic and exact state ownership."""

import copy

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.external_draft_policy import (
    normalize_progressive_external_policy,
    progressive_external_terminal_input_widths,
    resolve_progressive_external_tile,
)
from mlx2.adapters.proposal_path_sources import build_continuation_drafter
from mlx2.runtime import row_exact_verify as REV
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator, HostDraftRow
from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows
from mlx2.runtime.progressive_external_verify import progressive_target_protocol_binding
from mlx2.runtime.proposal_providers import PointMassLaw
from test_ast_terminal_ordinary_cpu import _committed_digest
from test_qwen38_dflash2_cpu import PROMPTS, tiny_pair


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _policy():
    return {
        "num_draft": 7,
        "progressive_verification_tile": 4,
        "progressive_verification_fidelity": "near_tie_bounded",
        "progressive_alternative_branches": 1,
        "continuation_pool": {
            "sources": ["external"],
            "limit": 1,
            "ast_planner": True,
        },
    }


def _batch(tile=4, fidelity="near_tie_bounded"):
    target, draft = tiny_pair()
    if fidelity == "exact":
        target.configure_target_verify_row_exact(True)
    policy = normalize_progressive_external_policy(
        {
            **_policy(),
            "progressive_verification_tile": tile,
            "progressive_verification_fidelity": fidelity,
        },
        num_draft=7,
    )
    resolved = resolve_progressive_external_tile(
        policy, num_draft=7, target=target, max_lanes=4
    )
    wrapper = build_continuation_drafter(
        target,
        draft,
        policy["continuation_pool"],
        target_revision="a" * 64,
        draft_revision="b" * 64,
        tokenizer_revision="a" * 64,
        session_revision="c" * 64,
    )
    binding = progressive_target_protocol_binding(target, fidelity)
    batch = ExternalDraftBatchGenerator(
        target,
        draft_model=wrapper,
        binding="primary-near-tie-ast",
        num_draft=7,
        completion_batch_size=4,
        ready_drain="all",
        continuation_pool=wrapper.policy,
        continuation_pool_ast_bridge=True,
        progressive_verification_tile=resolved,
        **(
            {
                "progressive_verification_tile_policy": "auto",
                "progressive_terminal_input_widths": progressive_external_terminal_input_widths(
                    target, max_lanes=4, num_draft=7, fidelity=fidelity
                ),
            }
            if tile == "auto"
            else {}
        ),
        progressive_verification_fidelity=fidelity,
        progressive_target_protocol=binding["target_protocol"],
        progressive_protocol_binding=binding,
        cost_per_commit_handoff=False,
    )
    uids = batch.insert(
        [*PROMPTS, [11, 22, 33, 44]],
        max_tokens=[20] * 4,
        sampling_configs=[{"sampling_temp": 0}] * 4,
    )
    lanes = [batch.lanes[uid] for uid in uids]
    for lane in lanes:
        while lane.anchor is None:
            batch._prefill(lane)
        lane.ready.clear()
    return target, batch, lanes


def test_exact_primary_ast_control_uses_existing_exact_target_contract():
    target, batch, lanes = _batch("auto", "exact")
    try:
        assert batch.progressive_verification_fidelity == "exact"
        assert batch.primary_near_tie_ast is False
        assert batch.ast_target_tree_executor_provider is None
        assert "near_tie" not in batch.ast_execution_planner.profile.backend
        assert callable(target.progressive_external_target_rows)
        batch._decode_phase(lanes)
        assert all(lane.external_rounds > 0 for lane in lanes)
        assert batch.ast_execution_planner.cost_samples > 0
    finally:
        batch.close()


@pytest.mark.parametrize(
    "change",
    [
        {"progressive_alternative_branches": 2},
        {
            "continuation_pool": {
                "sources": ["external"],
                "limit": 2,
                "ast_planner": True,
            }
        },
        {"continuation_pool": {"sources": ["external"], "limit": 1}},
    ],
)
def test_unsupported_primary_bridge_combinations_remain_refused(change):
    with pytest.raises(ValueError):
        normalize_progressive_external_policy({**_policy(), **change}, num_draft=7)


@pytest.mark.parametrize("tile", [4, "auto"])
def test_primary_ast_real_draft_and_b4_stock_verification(monkeypatch, tile):
    target, batch, lanes = _batch(tile)
    assert not hasattr(target, "progressive_external_target_rows")
    planner = batch.ast_execution_planner
    assert planner.profile.backend == "padded_stock_near_tie_bounded"
    assert planner.profile.valid_row_costs == ()
    assert planner.profile.cost_authority.startswith("stock_unmeasured_")
    assert planner.tiles == ((1, 2, 3, 4, 5, 6) if tile == "auto" else (4,))
    assert batch.cost_per_commit_handoff is None
    geometry = batch._continuation_ast_bridge_geometry(4)
    assert geometry["sessions"] == geometry["physical_path_rows"] == 4
    forward = target.forward_with_taps
    shapes = []

    def observed(tokens, caches, layers):
        assert REV.current() is None
        shapes.append(tuple(tokens.shape))
        return forward(tokens, caches, layers)

    monkeypatch.setattr(target, "forward_with_taps", observed)
    batch._decode_phase(lanes)
    assert any(
        width == 4 and span >= (1 if tile == "auto" else 2) for width, span in shapes
    ), shapes
    assert planner.cost_samples > 0
    for lane in lanes:
        assert lane.proposed > 0 and lane.external_rounds > 0 and not lane.ordinary
        receipt = lane.ready[-1].speculative_receipt
        assert receipt["approximate"] is True
        progressive = receipt["progressive_verification"]
        assert progressive["qualified"] is False
        assert progressive["observed_used"] is True
        assert progressive["target_row_executor"]["adapter_owned_launches"] == 0
        assert progressive["target_row_executor"]["legacy_launches"] > 0
    assert not batch._last_alternative_blocks
    assert batch.progressive_topology_latch is None
    batch.close()


@pytest.mark.parametrize("tile", [4, "auto"])
def test_stock_batched_accept_reject_state_continuation_and_abort(monkeypatch, tile):
    target, batch, lanes = _batch(tile)
    if tile == "auto":
        from mlx2.runtime.ast_planner import LAUNCH_FIT_MIN_SAMPLES, LaunchPlan

        # Synthetic observed costs select a full terminal slab. This tests the
        # real automatic dispatch and transaction law, not a timing claim.
        planner = batch.ast_execution_planner
        for _ in range(LAUNCH_FIT_MIN_SAMPLES):
            for width in (2, 4):
                lengths = (width,) * 4
                contexts = tuple(len(lane.history) for lane in lanes)
                cost = planner.profile.cost(lengths, contexts)
                plan = LaunchPlan(
                    tuple(range(4)),
                    width,
                    lengths,
                    cost,
                    cost,
                    cost,
                    1,
                    planner.profile.backend,
                    contexts,
                )
                planner.observe_launch(plan, 0.01 + width * 0.0001)
        for lane in lanes:
            for _ in range(8):
                planner.observe_commit(lane.uid, 15, 15, censored=True)
    originals = [copy.deepcopy(lane.cache) for lane in lanes]
    serial = [copy.deepcopy(row) for row in originals]
    anchors = [lane.anchor for lane in lanes]
    proposals = [[] for _ in lanes]
    for _ in range(3):
        tx = HybridVerifyRows(serial).begin([1] * 4)
        logits, features = target.forward_with_taps(
            mx.array(anchors)[:, None], tx.caches, batch.layers
        )
        mx.eval(logits, features)
        anchors = mx.argmax(logits[:, -1, :], axis=-1).tolist()
        for tokens, anchor in zip(proposals, anchors, strict=True):
            tokens.append(anchor)
        tx.commit([1] * 4)
    proposals[0][0] = (proposals[0][0] + 1) % batch._target_args.vocab_size
    proposals[1][1] = (proposals[1][1] + 1) % batch._target_args.vocab_size
    blocks = [
        HostDraftRow(tokens, [PointMassLaw(token) for token in tokens], 4)
        for tokens in proposals
    ]
    reference = [copy.deepcopy(row) for row in originals]
    tx = HybridVerifyRows(reference).begin([4] * 4)
    inputs = mx.array(
        [[lane.anchor, *tokens] for lane, tokens in zip(lanes, proposals)]
    )
    expected_logits, expected_features = target.forward_with_taps(
        inputs, tx.caches, batch.layers
    )
    mx.eval(expected_logits, expected_features)
    greedy = mx.argmax(expected_logits, axis=-1).tolist()
    accepted = [
        next((i for i, (a, b) in enumerate(zip(tokens, law)) if a != b), 3)
        for tokens, law in zip(proposals, greedy, strict=True)
    ]
    assert accepted[:2] == [0, 1] and accepted[2:] == [3, 3]
    tx.commit([n + 1 for n in accepted])
    forward = target.forward_with_taps
    observed_logits = []

    def observed(*args):
        assert REV.current() is None
        result = forward(*args)
        observed_logits.append(result[0])
        return result

    monkeypatch.setattr(target, "forward_with_taps", observed)
    decisions, features, rows, _rngs, _stats, physical = batch._progressive_verify_many(
        lanes, blocks
    )
    assert physical["physical_launches"] == 1 and physical["max_cohort_width"] == 4
    np.testing.assert_array_equal(
        np.asarray(observed_logits[0]), np.asarray(expected_logits)
    )
    for index, (decision, row, expected) in enumerate(zip(decisions, rows, reference)):
        assert decision.accepted == accepted[index]
        assert _committed_digest(row) == _committed_digest(expected)
        np.testing.assert_array_equal(
            np.asarray(features[index, : accepted[index] + 1]),
            np.asarray(expected_features[index, : accepted[index] + 1]),
        )
        token = decision.emitted[-1]
        actual_next = target(mx.array([[token]]), cache=row)
        expected_next = target(mx.array([[token]]), cache=expected)
        np.testing.assert_array_equal(
            np.asarray(actual_next), np.asarray(expected_next)
        )
        assert _committed_digest(row) == _committed_digest(expected)
    before = [_committed_digest(lane.cache) for lane in lanes]

    def failed(*args):
        forward(*args)
        raise RuntimeError("injected primary AST failure")

    monkeypatch.setattr(target, "forward_with_taps", failed)
    with pytest.raises(RuntimeError, match="injected primary AST failure"):
        batch._progressive_verify_many(lanes, blocks)
    assert [_committed_digest(lane.cache) for lane in lanes] == before
    batch.close()


def test_auto_near_tie_geometry_is_separate_bounded_and_source_bound():
    from types import SimpleNamespace

    target, _draft = tiny_pair()
    policy = {**_policy(), "progressive_verification_tile": "auto"}
    assert (
        resolve_progressive_external_tile(
            policy, num_draft=7, target=target, max_lanes=4
        )
        == 6
    )
    assert 5 in progressive_external_terminal_input_widths(
        target, max_lanes=4, num_draft=7, fidelity="near_tie_bounded"
    )
    declaration = target.progressive_external_approximate_protocols
    binding = progressive_target_protocol_binding(target, "near_tie_bounded")
    assert "src/mlx2/adapters/external_draft_policy.py" in binding["source_hashes"]
    # Exact geometry cannot stand in for absent approximate geometry.
    foreign = SimpleNamespace(
        progressive_external_exact_tile_widths=tuple(range(1, 16)),
        progressive_external_exact_terminal_input_widths=tuple(range(2, 17)),
        progressive_external_exact_max_lanes=4,
    )
    with pytest.raises(ValueError, match="stock rectangular"):
        resolve_progressive_external_tile(
            policy, num_draft=7, target=foreign, max_lanes=4
        )
    with pytest.raises(ValueError, match="hard capacities"):
        resolve_progressive_external_tile(
            policy, num_draft=7, target=target, max_lanes=5
        )
    altered = copy.deepcopy(declaration)
    protocol = binding["target_protocol"]
    altered[protocol]["execution_geometry"]["state"] = "inplace"
    with pytest.raises(ValueError, match="transactional"):
        resolve_progressive_external_tile(
            policy,
            num_draft=7,
            target=SimpleNamespace(progressive_external_approximate_protocols=altered),
            max_lanes=4,
        )
