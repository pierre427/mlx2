"""Real native executor trains only uncensored homogeneous factorized rounds."""

import copy

import pytest
from test_batched_mtp import _tiny_qwen4_model
from test_copy_draft_self_mtp import CYCLE, _copying_model

from mlx2.runtime.adaptive_policy import CohortAdaptiveMTPDepth
from mlx2.runtime.generate import BatchGenerator


@pytest.mark.parametrize("change", [
    {"committed": -1}, {"elapsed_seconds": float("nan")},
    {"elapsed_seconds": True}, {"proposed": 3}, {"accepted": 0},
    {"landing_accepted": None}, {"landing_depths": (1, 3)},
])
def test_invalid_factorized_evidence_changes_no_learned_state(change):
    policy = CohortAdaptiveMTPDepth(2, cost_model="factorized", min_samples_per_depth=1)
    policy.select(width=2)
    before = copy.deepcopy(policy.__dict__)
    options = {"proposed": 4, "accepted": 1, "width": 2, "committed": 3,
               "elapsed_seconds": .1, "landing_depths": (2, 2), "landing_accepted": (1, 0)}
    options.update(change)
    with pytest.raises(ValueError):
        policy.observe(**options)
    assert policy._factorized.__dict__ == before["_factorized"].__dict__
    assert policy.diagnostics() == _diagnostics(before)
    assert policy.trace == before["trace"]


def _diagnostics(state):
    policy = CohortAdaptiveMTPDepth.__new__(CohortAdaptiveMTPDepth)
    policy.__dict__.update(state)
    return policy.diagnostics()


def _execute(monkeypatch, *, segmented=False, copying=False, cost_model="factorized"):
    model = _copying_model() if copying else _tiny_qwen4_model()
    width = 1 if copying else 2
    native = {"num_draft": 2, "persistent": True}
    if segmented:
        monkeypatch.setenv("MLX_LM_TRUE_BATCHED_SEGMENTED_MTP", "0")
        native.update(segment_aware_live_tip=True, segment_aware_cohort_size=width,
                      segment_aware_async_qsa_promotion=False)
    batch = BatchGenerator(model, completion_batch_size=width, prefill_batch_size=width,
                           prefill_step_size=32, self_mtp=native,
                           adaptive_mtp_depth={"cost_model": cost_model, "probe_interval": 100,
                                               "min_samples_per_depth": 1},
                           copy_draft={"enabled": True, "max_span": 8} if copying else None)
    observations = []
    original = CohortAdaptiveMTPDepth.observe

    def observe(policy, proposed, accepted, **kwargs):
        factor = policy._factorized
        before = None if factor is None else factor.samples
        result = original(policy, proposed, accepted, **kwargs)
        observations.append(dict(kwargs, proposed=proposed, accepted=accepted,
                                 trained=0 if factor is None else factor.samples - before))
        return result

    monkeypatch.setattr(CohortAdaptiveMTPDepth, "observe", observe)
    prompts = [CYCLE * 3] if copying else [[1, 7, 3, 9, 2], [4, 5, 6, 2, 8]]
    budgets = [18] if copying else [7, 10]
    uids = batch.insert(prompts, max_tokens=budgets)
    terminal, outputs = {}, {uid: [] for uid in uids}
    try:
        for _ in range(100):
            before = len(observations)
            _, responses = batch.next()
            has_terminal = any(row.finish_reason for row in responses)
            for observation in observations[before:]:
                observation["terminal_output"] = has_terminal
            for row in responses:
                outputs[row.uid].append(row.token)
                if row.finish_reason:
                    terminal[row.uid] = row
            if len(terminal) == width:
                break
        assert [len(outputs[uid]) for uid in uids] == budgets
        return observations, terminal, dict(batch.scheduler_stats)
    finally:
        batch.close()


def test_real_native_rounds_train_and_terminal_censoring_does_not(monkeypatch):
    observed, terminal, _ = _execute(monkeypatch)
    trained = [row for row in observed if row["trained"]]
    assert trained
    assert any(row["width"] == 2 for row in trained)
    for row in trained:
        assert not row["terminal_output"]
        assert row["width"] == row["observed_compute_width"]
        assert sum(row["landing_depths"]) == row["proposed"]
        assert sum(row["landing_accepted"]) == row["accepted"]
    terminal_rounds = [row for row in observed if row["terminal_output"]]
    assert terminal_rounds and all(not row["trained"] for row in terminal_rounds)
    assert all(row.mtp_receipt["adaptive_depth"]["cost_model"]["cost_model"] == "factorized"
               for row in terminal.values())


def test_real_segmented_physical_width_mismatch_is_excluded(monkeypatch):
    observed, _terminal, _stats = _execute(monkeypatch, segmented=True)
    mixed = [row for row in observed if row["width"] > row["observed_compute_width"]]
    assert mixed
    assert all(row["width"] == 2 and row["observed_compute_width"] == 1 for row in mixed)
    assert all(not row["trained"] and "landing_depths" not in row for row in mixed)


def test_real_copy_spans_do_not_train_native_head_cost(monkeypatch):
    observed, terminal, _stats = _execute(monkeypatch, copying=True)
    assert any(row.mtp_receipt["copy_draft"]["copy_rounds"] > 0 for row in terminal.values())
    copies = [row for row in observed if "elapsed_seconds" not in row]
    assert copies
    assert all(row["trained"] == 0 and "landing_depths" not in row for row in copies)


def test_real_default_goodput_route_has_no_factorized_training(monkeypatch):
    observed, terminal, _stats = _execute(monkeypatch, cost_model="goodput")
    assert observed and all(row["trained"] == 0 for row in observed)
    assert all("cost_model" not in row.mtp_receipt["adaptive_depth"]["cost_model"]
               for row in terminal.values())
