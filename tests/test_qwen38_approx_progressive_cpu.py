"""Focused contracts for the Qwen3.8-only approximate tile candidate.

These tests establish admission, refusal, identity, and receipt behavior only.
They do not qualify the approximate protocol.
"""

import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.external_draft_policy import (
    ExternalDraftAdapterMixin,
    normalize_progressive_external_policy,
    progressive_external_identity,
)
from mlx2.adapters.qwen36_35b import (
    EXTERNAL_POLICY_KEYS as QWEN36_EXTERNAL_POLICY_KEYS,
    Qwen3635BA3BAdapter,
)
from mlx2.adapters.qwen38_27b import (
    EXTERNAL_POLICY_KEYS as QWEN38_EXTERNAL_POLICY_KEYS,
    Qwen3827BAdapter,
)
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator, LaneFailure
from mlx2.runtime.sample_utils import LaneRNG
from mlx2.runtime.progressive_external_verify import (
    PROGRESSIVE_APPROXIMATE_QUALIFICATION_GATE_SCHEMA,
    PROGRESSIVE_APPROXIMATE_TARGET_PROTOCOL_VERSION,
    PROGRESSIVE_EXTERNAL_VERIFY_VERSION,
    PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED,
    ProgressiveTargetRows,
    progressive_target_protocol_binding,
)
from test_qwen38_dflash2_cpu import tiny_pair
from test_qwen_progressive_external_portability_cpu import tiny_qwen36


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _policy():
    return normalize_progressive_external_policy(
        {
            "draft_model": "/draft",
            "num_draft": 3,
            "progressive_verification_tile": 2,
            "progressive_verification_fidelity": (
                PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED
            ),
        },
        num_draft=3,
        minimum_proposals=1,
    )


def _candidate():
    target, draft = tiny_pair()
    binding = progressive_target_protocol_binding(
        target, PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED
    )
    batch = ExternalDraftBatchGenerator(
        target,
        draft_model=draft,
        binding="qwen38-approx-progressive-test",
        num_draft=3,
        completion_batch_size=1,
        progressive_verification_tile=2,
        progressive_verification_fidelity=PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED,
        progressive_target_protocol=(
            PROGRESSIVE_APPROXIMATE_TARGET_PROTOCOL_VERSION
        ),
        progressive_protocol_binding=binding,
    )
    return batch, binding


def test_qwen38_is_the_only_qwen_external_policy_allowlist():
    candidate_keys = {
        "progressive_verification_tile",
        "progressive_multilane_draft_cap",
        "progressive_verification_fidelity",
    }
    assert candidate_keys <= QWEN38_EXTERNAL_POLICY_KEYS
    assert candidate_keys.isdisjoint(QWEN36_EXTERNAL_POLICY_KEYS)
    assert all(
        key not in Qwen3827BAdapter.default_route_execution_policy["external_draft"]
        for key in candidate_keys
    )
    assert all(
        key not in Qwen3635BA3BAdapter.default_route_execution_policy.get(
            "external_draft", {}
        )
        for key in candidate_keys
    )


def test_qwen36_has_no_approximate_protocol_and_fails_closed():
    model = tiny_qwen36()
    assert not hasattr(model, "progressive_external_approximate_protocols")
    with pytest.raises(ValueError, match="explicit target protocol declaration"):
        progressive_target_protocol_binding(
            model, PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED
        )


def test_qwen36_refuses_candidate_policy_before_artifact_loading(tmp_path):
    with pytest.raises(ValueError, match="Qwen3.6 external draft policy has unknown keys"):
        Qwen3635BA3BAdapter(
            str(tmp_path / "target"),
            execution_policy={
                "draft_model": str(tmp_path / "draft"),
                "progressive_verification_tile": 2,
                "progressive_verification_fidelity": (
                    PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED
                ),
            },
        )


def test_exact_and_approximate_protocol_gates_are_disjoint():
    target, draft = tiny_pair()
    binding = progressive_target_protocol_binding(
        target, PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED
    )
    with pytest.raises(ValueError, match="row-exact B1 state-promotion"):
        ExternalDraftBatchGenerator(
            target,
            draft_model=draft,
            binding="must-not-alias-exact",
            num_draft=3,
            progressive_verification_tile=2,
        )
    with pytest.raises(ValueError, match="exact source-hashed protocol binding"):
        ExternalDraftBatchGenerator(
            target,
            draft_model=draft,
            binding="must-bind-sources",
            num_draft=3,
            progressive_verification_tile=2,
            progressive_verification_fidelity=(
                PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED
            ),
            progressive_target_protocol=(
                PROGRESSIVE_APPROXIMATE_TARGET_PROTOCOL_VERSION
            ),
            progressive_protocol_binding={**binding, "source_hashes": {}},
        )


def test_route_identity_and_execution_config_bind_protocol_fidelity_and_sources():
    target, _draft = tiny_pair()
    binding = progressive_target_protocol_binding(
        target, PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED
    )
    policy = _policy()
    identity = json.loads(progressive_external_identity(policy, binding))
    assert identity["algorithm"] == PROGRESSIVE_EXTERNAL_VERIFY_VERSION
    assert identity["target_protocol"] == (
        PROGRESSIVE_APPROXIMATE_TARGET_PROTOCOL_VERSION
    )
    assert identity["fidelity"] == PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED
    assert identity["source_hashes"] == binding["source_hashes"]
    assert "src/mlx2/adapters/qwen38_27b.py" in identity["source_hashes"]
    assert len(identity["source_hashes"]) >= 6

    adapter = Qwen3827BAdapter.__new__(Qwen3827BAdapter)
    adapter.external_policy = policy
    adapter._progressive_external_binding = binding
    config = ExternalDraftAdapterMixin._external_execution_config(
        adapter, max_lanes=1, prefill_step=128
    )
    assert config["progressive_verification_fidelity"] == (
        PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED
    )
    assert config["progressive_target_protocol"] == (
        PROGRESSIVE_APPROXIMATE_TARGET_PROTOCOL_VERSION
    )
    assert config["progressive_protocol_binding"] == binding


def test_receipt_is_truthfully_approximate_and_source_bound():
    batch, binding = _candidate()
    try:
        lane = SimpleNamespace(
            relaxed_accepts=0,
            verify_span_hist={},
            verify_accept_hist={},
            verify_accept_span_hist={},
            progressive_verify_rounds=0,
            progressive_verify_launches=0,
            progressive_verify_target_rows=0,
            progressive_verify_rejections=0,
            progressive_verify_full_tiles=0,
            processors=[],
        )
        receipt = batch._verification_receipt(lane)
        assert receipt["verification"] == (
            "approximate_progressive_near_tie_bounded"
        )
        assert receipt["approximate"] is True
        progressive = receipt["progressive_verification"]
        assert progressive["qualified"] is False
        assert progressive["selected"] is True
        assert progressive["observed_used"] is False
        assert progressive["target_protocol"] == (
            PROGRESSIVE_APPROXIMATE_TARGET_PROTOCOL_VERSION
        )
        assert progressive["fidelity"] == PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED
        assert progressive["source_hashes"] == binding["source_hashes"]
        assert progressive["qualification_gate_schema"] == (
            PROGRESSIVE_APPROXIMATE_QUALIFICATION_GATE_SCHEMA
        )
    finally:
        batch.close()


def test_one_round_records_real_approximate_engagement():
    batch, binding = _candidate()
    target = batch.model
    draft = batch.draft
    prompt = [1, 2, 3, 4]
    cache = target.make_cache()
    target(mx.array([prompt[:-1]]), cache=cache)
    expected_anchor = prompt[-1]
    proposal = []
    token = expected_anchor
    logits = None
    for _ in range(3):
        logits = target(mx.array([[token]]), cache=cache)
        token = int(mx.argmax(logits[0, -1]).item())
        proposal.append(token)
    assert logits is not None
    vocab_size = int(logits.shape[-1])
    laws = []
    for token in proposal:
        law = np.zeros(vocab_size, dtype=np.float64)
        law[token] = 1.0
        laws.append(law)
    real = draft.draft_distributions

    def forced(anchors, hidden, caches, count, rngs, temps, **kwargs):
        real(anchors, hidden, caches, count, rngs, temps, **kwargs)
        assert count == 3 and anchors == [expected_anchor]
        return [proposal], [laws]

    draft.draft_distributions = forced
    try:
        uid = batch.insert(
            [prompt],
            max_tokens=[8],
            lane_rngs=[LaneRNG(919)],
            sampling_configs=[{"sampling_temp": 0}],
        )[0]
        lane = batch.lanes[uid]
        while lane.anchor is None:
            batch._prefill(lane)
        batch._round([lane])
        receipt = lane.ready[-1].speculative_receipt
        assert receipt["verification"] == (
            "approximate_progressive_near_tie_bounded"
        )
        assert receipt["approximate"] is True
        progressive = receipt["progressive_verification"]
        assert progressive["observed_used"] is True
        assert progressive["target_launches"] >= 1
        assert progressive["target_rows"] >= 1
        assert progressive["source_hashes"] == binding["source_hashes"]
        executor = progressive["target_row_executor"]
        assert executor["legacy_launches"] >= 1
        assert executor["adapter_owned_launches"] == 0
        assert executor["last"]["execution"] == "legacy_forward_with_taps"
    finally:
        batch.close()


def test_adapter_owned_target_executor_is_observed_in_lane_receipt():
    batch, _binding = _candidate()
    target = batch.model
    original = target.forward_with_taps

    def target_rows(tokens, caches, layers):
        logits, features = original(tokens, caches, layers)
        return ProgressiveTargetRows(
            logits=logits,
            features=features,
            protocol=PROGRESSIVE_APPROXIMATE_TARGET_PROTOCOL_VERSION,
            row_exact=True,
            detail={"path": "cpu-test-adapter-owner"},
        )

    object.__setattr__(target, "progressive_external_target_rows", target_rows)
    try:
        uid = batch.insert(
            [[1, 2, 3, 4]],
            max_tokens=[8],
            lane_rngs=[LaneRNG(919)],
            sampling_configs=[{"sampling_temp": 0}],
        )[0]
        lane = batch.lanes[uid]
        while lane.anchor is None:
            batch._prefill(lane)
        batch._round([lane])
        receipt = lane.ready[-1].speculative_receipt
        executor = receipt["progressive_verification"]["target_row_executor"]
        assert executor["adapter_owned_launches"] >= 1
        assert executor["legacy_launches"] == 0
        assert executor["row_exact_launches"] >= 1
        assert executor["last"]["adapter_detail"] == {
            "path": "cpu-test-adapter-owner"
        }
        assert executor["last"]["physical_width"] == 1
        assert batch.scheduler_stats[
            "external_progressive_target_executor_adapter_owned_launches"
        ] >= 1
    finally:
        batch.close()


def test_approximate_progressive_candidate_is_greedy_only():
    batch, _binding = _candidate()
    try:
        lane = SimpleNamespace(uid=7, sampling={"sampling_temp": 0.8})
        with pytest.raises(LaneFailure, match="greedy-only"):
            batch._progressive_verify(lane, None)
    finally:
        batch.close()


def test_fidelity_cannot_select_itself_without_a_tile():
    with pytest.raises(ValueError, match="requires progressive_verification_tile"):
        normalize_progressive_external_policy(
            {
                "progressive_verification_fidelity": (
                    PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED
                )
            },
            num_draft=3,
        )
