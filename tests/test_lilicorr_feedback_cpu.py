"""Verified-label capture and isolated CPU worker lifecycle, not qualification."""

import time
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from test_lilicorr_cpu import config, initialized_head
from test_lilicorr_training import example

from mlx2.runtime.lilicorr_feedback import (
    LiLiCorrFeedbackManager,
    LiLiCorrFeedbackPolicy,
)


def manager(path, **overrides):
    drafter = SimpleNamespace(config=config(), lilicorr=initialized_head())
    return LiLiCorrFeedbackManager(
        drafter,
        {
            "directory": str(path),
            "min_examples": 1,
            "train_every": 1,
            "steps": 5,
            **overrides,
        },
        target_revision="a" * 40,
        draft_revision="b" * 64,
        binding="c" * 64,
    )


def capture(value):
    fields = example()
    args = [
        mx.array(fields[name])[None]
        for name in (
            "candidate_ids",
            "token_embeddings",
            "candidate_log_probs",
            "pass_hidden",
            "anchor_hidden",
        )
    ]
    return value.capture_lattice(*args, mx.array([True]))[0]


def test_capture_is_frozen_and_only_commit_supplies_censored_labels(tmp_path):
    value = manager(tmp_path)
    payload = capture(value)
    assert not value.buffer.examples
    assert not payload["candidate_ids"].flags.writeable
    assert value.submit_verified(payload, [1, 3, 5], first_rejected_position=0)
    assert value.buffer.examples[0].teacher_columns == (1,)
    assert value.receipt()["live_head_changed"] is False
    with pytest.raises(ValueError, match="already consumed"):
        value.submit_verified(payload, [1])
    with pytest.raises(ValueError, match="revision/capture"):
        value.submit_verified({**payload, "binding": "wrong"}, [1])
    value.close()


def test_pending_capture_budget_and_closed_manager_do_not_export(tmp_path):
    value = manager(tmp_path, max_bytes=1)
    assert capture(value) is None
    assert value.stats["dropped"] == 1
    value.close()
    assert capture(value) is None


def test_real_isolated_cpu_training_exports_shadow_without_changing_live_head(tmp_path):
    from mlx.utils import tree_flatten

    value = manager(tmp_path)
    before = {
        name: np.array(tensor)
        for name, tensor in tree_flatten(value.drafter.lilicorr.parameters())
    }
    value.submit_verified(capture(value), [1, 3, 5])
    try:
        value.settle_round()
        assert value.stats["training_started"] == 1
        deadline = time.monotonic() + 20
        while value.child is not None and time.monotonic() < deadline:
            time.sleep(0.05)
            value._poll()
        assert value.child is None
        assert value.stats["training_completed"] == 1, value.receipt()
        assert value.latest_shadow["final_loss"] < value.latest_shadow["initial_loss"]
        assert not value.latest_shadow["selected"]
        assert value.latest_shadow["device"] == "CPU"
        assert mx.default_device() == mx.cpu
        for name, tensor in tree_flatten(value.drafter.lilicorr.parameters()):
            np.testing.assert_array_equal(np.array(tensor), before[name])
    finally:
        value.close()


@pytest.mark.parametrize(
    "override",
    [
        {"steps": 0},
        {"max_bytes": 65 << 20},
        {"min_examples": 33},
        {"learning_rate": float("nan")},
        {"unknown": True},
        {"max_artifacts": 0},
    ],
)
def test_invalid_policy_is_rejected(override):
    with pytest.raises(ValueError):
        LiLiCorrFeedbackPolicy.from_value({"directory": "unused", **override})


def test_capture_ticket_authenticates_contents_and_rejects_unissued_ids(tmp_path):
    import copy

    value = manager(tmp_path)
    payload = capture(value)
    changed = copy.deepcopy(payload)
    changed["candidate_ids"][0, 0] = 8
    with pytest.raises(ValueError, match="contents changed"):
        value.submit_verified(changed, [1])
    with pytest.raises(ValueError, match="unissued"):
        value.submit_verified({**payload, "nonce": payload["nonce"] + 1}, [1])
    assert value.submit_verified(
        copy.deepcopy(payload), [1], request_id="private-request", round_id=7
    )
    assert value.last_committed_trace["round_id"] == 7
    assert value.last_committed_trace["request_id_sha256"] != "private-request"
    value.close()


def test_evicted_capture_cannot_be_replayed(tmp_path):
    value = manager(tmp_path)
    payload = capture(value)
    value._issued.pop(payload["nonce"])
    with pytest.raises(ValueError, match="expired"):
        value.submit_verified(payload, [1])
    value.close()


def test_training_memory_guard_skips_worker_before_export(tmp_path):
    value = manager(tmp_path, max_training_bytes=1)
    value.submit_verified(capture(value), [1])
    value.settle_round()
    assert value.child is None
    assert value.stats["training_budget_skipped"] == 1
    assert not value.directory.exists()
    assert value.receipt()["live_head_changed"] is False
    value.close()


def test_spawn_failures_and_orphaned_jobs_are_bounded(tmp_path, monkeypatch):
    import mlx2.runtime.lilicorr_feedback as module

    value = manager(tmp_path, max_artifacts=2)

    def fail(*args, **kwargs):
        raise OSError("injected spawn failure")

    monkeypatch.setattr(module.subprocess, "Popen", fail)
    for _ in range(5):
        value.submit_verified(capture(value), [1])
        value.settle_round()
    assert value.child is None
    assert value.stats["training_failed"] == 5
    jobs = list(value.directory.glob("shadow-*"))
    assert len(jobs) == 2
    assert all((job / "result.json").is_file() for job in jobs)
    value.close()


def test_malformed_result_clears_finished_worker_and_preserves_other_manager(tmp_path):
    value = manager(tmp_path, max_artifacts=1)
    other = manager(tmp_path)
    other.directory.mkdir(parents=True)
    sentinel = other.directory / "shadow-unrelated"
    sentinel.mkdir()
    value.directory.mkdir(parents=True)
    job = value.directory / "shadow-malformed"
    job.mkdir()
    (job / "result.json").write_text("{broken")
    value.child = SimpleNamespace(poll=lambda: 0, returncode=0)
    value.child_directory = job
    value._poll()
    assert value.child is None
    assert value.stats["training_failed"] == 1
    assert "JSONDecodeError" in value.last_error
    assert sentinel.exists()
    value.close()
    other.close()


def test_worker_rechecks_memory_before_constructing_head(tmp_path):
    import json

    from mlx2.runtime.lilicorr_feedback_worker import run

    value = manager(tmp_path)
    value.submit_verified(capture(value), [1])
    value.settle_round()
    while value.child is not None:
        time.sleep(0.05)
        value._poll()
    job = next(value.directory.glob("shadow-*")) / "job.json"
    data = json.loads(job.read_text())
    data["policy"]["max_training_bytes"] = 1
    job.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="memory exceeds"):
        run(job)
    value.close()


def test_serialization_failures_bound_artifacts_while_manager_stays_open(
    tmp_path, monkeypatch
):
    value = manager(tmp_path, max_artifacts=2)

    def fail(*args, **kwargs):
        raise OSError("injected serialization failure")

    monkeypatch.setattr(mx, "save_safetensors", fail)
    for _ in range(5):
        value.submit_verified(capture(value), [1])
        value.settle_round()
    assert value.child is None and value.child_directory is None
    assert not value.closed
    assert value.stats["training_failed"] == 5
    jobs = list(value.directory.glob("shadow-*"))
    assert len(jobs) == 2
    assert all((job / "result.json").is_file() for job in jobs)
    value.close()


@pytest.mark.parametrize("num_draft", [2, 3])
def test_full_acceptance_commits_feedback_at_every_depth(tmp_path, num_draft):
    """A fully accepted round is a positive LiLiCoRR label, not a failure.

    The caller passed ``first_rejected_position = accepted`` whenever
    ``accepted < len(emitted)``, which a full acceptance satisfies because
    ``emitted`` ends with the bonus token.  At num_draft == block_size - 1
    that position equals the lattice's slot count, the buffer refused it, and
    every fully accepted round was counted as a feedback failure.
    """
    from test_lilicorr_serving_cpu import pair
    from test_standard_xpress_serving_cpu import reference

    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    model, draft = pair()
    slots = draft.config.block_size - 1
    assert num_draft <= slots
    feedback = LiLiCorrFeedbackManager(
        draft,
        # Bounds keep the shadow trainer from launching in this test.
        {"directory": str(tmp_path), "min_examples": 128, "max_examples": 128,
         "train_every": 1024},
        target_revision="a" * 40,
        draft_revision="b" * 64,
        binding="c" * 64,
    )
    draft.feedback_manager = feedback
    boundaries = []
    submit = feedback.submit_verified

    def spy(payload, teacher_tokens, **kwargs):
        boundaries.append(kwargs["first_rejected_position"])
        return submit(payload, teacher_tokens, **kwargs)

    feedback.submit_verified = spy
    generator = ExternalDraftBatchGenerator(
        model, draft_model=draft, binding="tiny-lilicorr-v1",
        num_draft=num_draft, prefill_step_size=32,
    )
    prompt = [1, 2, 3, 4, 5]
    expected = reference(model, prompt, 12)
    original = draft.draft_distributions

    def target_greedy(anchors, hidden, cache, proposal_length, *args, **kwargs):
        # Keep the real lattice capture; propose the target's own continuation
        # so the target accepts every drafted token.
        original(anchors, hidden, cache, proposal_length, *args, **kwargs)
        start = len(generator.lanes[0].history) + 1 - len(prompt)
        want = expected[start : start + proposal_length]
        eye = np.eye(draft.config.vocab_size)
        return [list(want)], [[eye[token] for token in want]]

    draft.draft_distributions = target_greedy
    try:
        generator.insert([prompt], max_tokens=[12])
        lane = generator.lanes[0]
        while lane.anchor is None:
            generator._prefill(lane)
        generator._round([lane])
        assert [response.token for response in lane.ready] == expected[: num_draft + 1]
        assert generator.scheduler_stats.get("external_feedback_failures", 0) == 0, (
            generator.last_feedback_error
        )
        assert feedback.stats["committed"] == 1
        assert not feedback._issued  # the capture ticket was consumed
        # No draft was rejected, so no boundary inside the lattice is claimed.
        assert boundaries == [None]
    finally:
        feedback.close()
