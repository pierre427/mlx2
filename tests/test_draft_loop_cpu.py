"""Confidence-gated self-MTP draft looping: exact output, observable gate."""

import mlx.core as mx
import pytest

from mlx2.runtime.draft_loop import DraftLoopPolicy

from test_batched_mtp import _tiny_qwen4_model


def test_policy_validation_fails_closed():
    assert DraftLoopPolicy.from_value(None) is None
    policy = DraftLoopPolicy.from_value({"stage": 3, "threshold": -0.5})
    assert policy.gates(9) and not policy.gates(3)
    assert policy.ends(9) == (3, 6, 9) and policy.ends(7) == (3, 6, 7)
    assert [policy.decision(d, 9) for d in range(1, 10)] == [
        None, None, 0, None, None, 3, None, None, None]
    assert policy.receipt()["qualified"] is False
    explicit = DraftLoopPolicy.from_value({"boundaries": [3, 7], "threshold": -0.5})
    assert explicit.ends(7) == (3, 7) and explicit.ends(9) == (3, 7, 9)
    assert explicit.ends(5) == (3, 5)
    # Explicit boundaries own the ceiling: a lane configured at 3 may reach 7.
    assert explicit.limit(3) == 7 and explicit.limit(9) == 9 and explicit.limit(2) == 2
    assert explicit.applies(3) and not explicit.applies(2)
    capped = DraftLoopPolicy.from_value({"boundaries": [3, 7], "threshold": -0.4, "max_width": 1})
    assert capped.max_width == 1 and capped.receipt()["max_width"] == 1
    assert explicit.decision(3, 7) == 0 and explicit.decision(7, 7) is None
    assert explicit.decision(7, 9) == 3
    assert explicit.receipt()["boundaries"] == [3, 7]
    wide = DraftLoopPolicy.from_value(
        {"by_width": {"1": [3, 9], "2": [3, 7]}, "threshold": -0.4, "cohort": "any"}
    )
    assert wide.for_width(1).boundaries == (3, 9) and wide.for_width(2).boundaries == (3, 7)
    assert wide.for_width(2).cohort == "any" and wide.for_width(3) is None
    assert wide.limit(3) == 9 and wide.limit(2) == 2
    assert wide.receipt()["by_width"] == {"1": [3, 9], "2": [3, 7]}
    with pytest.raises(ValueError, match="for_width"):
        wide.ends(9)
    assert capped.for_width(1) is capped and capped.for_width(2) is None
    for bad in (
        {"by_width": {"1": [3, 9]}, "threshold": -0.4, "max_width": 1},
        {"by_width": {"0": [3, 9]}, "threshold": -0.4},
        {"by_width": {"1": [3]}, "threshold": -0.4},
        {"by_width": [3, 9], "threshold": -0.4},
        {"boundaries": [3, 7], "threshold": -0.4, "cohort": "all"},
        {"boundaries": [3], "threshold": -0.5},
        {"boundaries": [3, 3], "threshold": -0.5},
        {"boundaries": [0, 3], "threshold": -0.5},
        {"boundaries": [3, 7], "stage": 3, "threshold": -0.5},
        {"boundaries": "3,7", "threshold": -0.5},
        {"boundaries": [3, 7], "threshold": -0.5, "max_width": 0},
        {"stage": 0, "threshold": -0.5},
        {"stage": True, "threshold": -0.5},
        {"stage": 3, "threshold": 0.5},
        {"stage": 3, "threshold": float("nan")},
        {"stage": 3},
        {"stage": 3, "threshold": -0.5, "extra": 1},
        [3, -0.5],
    ):
        with pytest.raises(ValueError):
            DraftLoopPolicy.from_value(bad)


def _run(model, prompts, max_tokens, self_mtp=None, temps=None):
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.sample_utils import LaneRNG

    kwargs = {}
    if self_mtp is not None:
        kwargs["self_mtp"] = {
            "persistent": True,
            "segment_aware_live_tip": True,
            "segment_aware_cohort_size": len(prompts),
            **self_mtp,
        }
    gen = BatchGenerator(
        model,
        completion_batch_size=len(prompts),
        prefill_batch_size=len(prompts),
        prefill_step_size=32,
        **kwargs,
    )
    insert = {}
    if self_mtp is not None:
        insert = {
            "lane_rngs": [LaneRNG(17 + i) for i in range(len(prompts))],
            "self_mtp_configs": [{"sampling_temp": t} for t in (temps or [0.0] * len(prompts))],
        }
    uids = gen.insert(prompts, max_tokens=[max_tokens] * len(prompts), **insert)
    outputs = {uid: [] for uid in uids}
    receipts = {}
    for _ in range(400):
        _, responses = gen.next()
        for r in responses:
            outputs[r.uid].append(r.token)
            receipt = getattr(r, "mtp_receipt", None)
            if receipt:
                receipts[r.uid] = receipt
        if all(len(v) >= max_tokens for v in outputs.values()):
            break
    stats = dict(gen.scheduler_stats)
    gen.close()
    return [outputs[uid] for uid in uids], receipts, stats


@pytest.fixture
def cpu_model():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(922)
        yield _tiny_qwen4_model()
    finally:
        mx.set_default_device(previous)


@pytest.mark.parametrize("threshold", [-1e9, -0.5, -1e-9])
def test_greedy_output_matches_ordinary_at_any_threshold(cpu_model, threshold):
    prompts = [[1, 7, 3, 9, 2, 8, 4], [5, 2, 6, 1, 9]]
    ordinary, _, _ = _run(cpu_model, prompts, 18)
    looped, receipts, _ = _run(
        cpu_model, prompts, 18,
        self_mtp={"num_draft": 6, "draft_loop": {"stage": 2, "threshold": threshold}},
    )
    assert looped == ordinary
    assert receipts, "no self-MTP receipt"
    for receipt in receipts.values():
        loop = receipt["draft_loop"]
        assert loop["qualified"] is False and loop["stage"] == 2
        assert loop["observed_used"] is True and loop["decisions"] > 0
        if threshold == -1e9:
            # Every boundary passes: lanes draft their full num_draft.
            assert loop["extensions"] == loop["decisions"]
        if threshold == -1e-9:
            # A random tiny head is never certain: every lane stops at stage 1.
            assert loop["extensions"] == 0


def test_without_policy_lanes_carry_no_loop_receipt(cpu_model):
    _, receipts, _ = _run(cpu_model, [[1, 7, 3, 9, 2, 8, 4]], 10, self_mtp={"num_draft": 3})
    assert receipts and all("draft_loop" not in r for r in receipts.values())


def test_stopping_early_verifies_fewer_rows(cpu_model):
    prompt = [[1, 7, 3, 9, 2, 8, 4]]
    _, deep, _ = _run(
        cpu_model, prompt, 16,
        self_mtp={"num_draft": 6, "draft_loop": {"stage": 2, "threshold": -1e9}},
    )
    _, shallow, _ = _run(
        cpu_model, prompt, 16,
        self_mtp={"num_draft": 6, "draft_loop": {"stage": 2, "threshold": -1e-9}},
    )
    (deep,) = deep.values()
    (shallow,) = shallow.values()
    assert shallow["draft_loop"]["extensions"] == 0
    assert deep["draft_loop"]["extensions"] == deep["draft_loop"]["decisions"] > 0


def test_sampled_lanes_gate_on_the_drafted_tokens(cpu_model):
    """Sampled lanes run the same gate; verification keeps them valid."""
    outputs, receipts, _ = _run(
        cpu_model, [[1, 7, 3, 9, 2, 8, 4]], 14,
        self_mtp={"num_draft": 6, "draft_loop": {"stage": 3, "threshold": -2.0}},
        temps=[0.8],
    )
    assert len(outputs[0]) >= 14
    (receipt,) = receipts.values()
    assert receipt["draft_loop"]["decisions"] > 0


def test_explicit_boundaries_match_ordinary_and_stop_at_the_tile(cpu_model):
    prompts = [[1, 7, 3, 9, 2, 8, 4]]
    ordinary, _, _ = _run(cpu_model, prompts, 18)
    looped, receipts, _ = _run(
        cpu_model, prompts, 18,
        self_mtp={"num_draft": 5, "draft_loop": {"boundaries": [2, 5], "threshold": -1e9}},
    )
    assert looped == ordinary
    (receipt,) = receipts.values()
    loop = receipt["draft_loop"]
    assert loop["boundaries"] == [2, 5] and loop["decisions"] > 0
    assert loop["extensions"] == loop["decisions"]
    # Every round decided once (at depth 2) and drafted on to 5.
    assert receipt["stats"]["verify_span_hist"]


def test_explicit_boundaries_extend_past_the_configured_depth(cpu_model):
    """Configured at the first boundary, a lane drafts on to the last one."""
    prompts = [[1, 7, 3, 9, 2, 8, 4]]
    ordinary, _, _ = _run(cpu_model, prompts, 18)
    looped, receipts, _ = _run(
        cpu_model, prompts, 18,
        self_mtp={"num_draft": 2, "draft_loop": {"boundaries": [2, 5], "threshold": -1e9}},
    )
    assert looped == ordinary
    (receipt,) = receipts.values()
    assert receipt["num_draft"] == 2
    loop = receipt["draft_loop"]
    assert loop["decisions"] > 0 and loop["extensions"] == loop["decisions"]
    spans = {int(k) for k in receipt["stats"]["verify_span_hist"]}
    assert max(spans) == 6  # 5 drafts + the anchor row


def test_max_width_keeps_wider_cohorts_on_fixed_depth(cpu_model):
    prompts = [[1, 7, 3, 9, 2, 8, 4], [5, 2, 6, 1, 9]]
    ordinary, _, _ = _run(cpu_model, prompts, 14)
    looped, receipts, _ = _run(
        cpu_model, prompts, 14,
        self_mtp={"num_draft": 2,
                  "draft_loop": {"boundaries": [2, 5], "threshold": -1e9, "max_width": 1}},
    )
    assert looped == ordinary
    for receipt in receipts.values():
        assert receipt["draft_loop"]["max_width"] == 1
        spans = {int(k) for k in receipt["stats"]["verify_span_hist"]}
        assert max(spans) <= 3


def test_two_lane_cohort_extends_together_and_matches_ordinary(cpu_model):
    """Cohort rule ``any``: one passing lane extends both; output stays exact."""
    prompts = [[1, 7, 3, 9, 2, 8, 4], [5, 2, 6, 1, 9]]
    ordinary, _, _ = _run(cpu_model, prompts, 16)
    for threshold in (-1e9, -1e-9):
        looped, receipts, _ = _run(
            cpu_model, prompts, 16,
            self_mtp={"num_draft": 2, "draft_loop": {
                "by_width": {"2": [2, 5]}, "threshold": threshold, "cohort": "any"}},
        )
        assert looped == ordinary
        loops = [r["draft_loop"] for r in receipts.values()]
        assert all(l["cohort"] == "any" and l["by_width"] == {"2": [2, 5]} for l in loops)
        decisions = sum(l["decisions"] for l in loops)
        extensions = sum(l["extensions"] for l in loops)
        assert decisions > 0
        # Lanes decide together: either every lane extends or none does.
        assert extensions in (0, decisions) if threshold == -1e-9 else extensions == decisions
        spans = {int(k) for r in receipts.values() for k in r["stats"]["verify_span_hist"]}
        assert max(spans) == (6 if threshold == -1e9 else 3)


def test_by_width_leaves_unlisted_widths_on_fixed_depth(cpu_model):
    prompts = [[1, 7, 3, 9, 2, 8, 4], [5, 2, 6, 1, 9]]
    _, receipts, _ = _run(
        cpu_model, prompts, 12,
        self_mtp={"num_draft": 2, "draft_loop": {
            "by_width": {"1": [2, 5]}, "threshold": -1e9, "cohort": "any"}},
    )
    for receipt in receipts.values():
        spans = {int(k) for k in receipt["stats"]["verify_span_hist"]}
        assert max(spans) <= 3
