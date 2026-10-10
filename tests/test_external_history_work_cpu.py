"""History work scales with processor demand, not the retained prompt."""

from collections import Counter
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.external_speculative import (
    ExternalDraftBatchGenerator,
    HostDraftRow,
    TreeDraftRow,
)
from mlx2.runtime.sample_utils import make_presence_penalty
from mlx2.runtime.speculative_sampling import RequestRNG
from mlx2.serving import minimum_tokens_processor


class CountedHistory(list):
    def __init__(self, values):
        super().__init__(values)
        self.copied = 0

    def __add__(self, other):
        self.copied += len(self)
        return super().__add__(other)

    def __iter__(self):
        self.copied += len(self)
        return super().__iter__()

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if isinstance(key, slice):
            self.copied += len(value)
        return value


def executor():
    batch = ExternalDraftBatchGenerator.__new__(ExternalDraftBatchGenerator)
    batch.mx = mx
    batch.stops = set()
    batch.fly_verification = SimpleNamespace(enabled=False)
    batch.tree_gates = {"logprobs_on_request": True}
    batch.scheduler_stats = Counter()
    return batch


def lane(history, processors=()):
    return SimpleNamespace(
        uid=1, history=history, anchor=3, processors=list(processors),
        sampling={"sampling_temp": 0.0, "emit_logprobs": True},
        rng=RequestRNG(3),
    )


@pytest.mark.parametrize("reachable", [False, True])
def test_target_law_does_not_upload_unused_history(monkeypatch, reachable):
    batch = executor()
    logits = mx.array([0.0, 1.0, 2.0])
    calls = []
    original = mx.array

    def tracked(value, *args, **kwargs):
        calls.append(value)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(mx, "array", tracked)
    processors = [] if reachable else [lambda *_: pytest.fail("unreachable row")]
    result = batch._target_law(
        lane(list(range(4096)), processors), logits, list(range(4096)),
        reachable, greedy_token=True,
    )
    assert result == 2
    assert calls == []


def test_linear_verification_does_not_copy_unprocessed_prompt():
    batch = executor()
    history = CountedHistory([0, 1] * 2048)
    current = lane(history)
    block = HostDraftRow([1, 1], [np.array([0.0, 1.0, 0.0])] * 2)
    logits = mx.array([[[0.0, 2.0, 0.0]] * 3])
    result = batch._verify([current], [block], logits)[0]
    assert result.emitted == [1, 1, 1]
    assert history.copied == 0


@pytest.mark.parametrize("kind", ["none", "length", "window"])
def test_batched_tree_history_work_is_bounded(kind):
    batch = executor()
    history = CountedHistory([0, 1] * 2048)
    processors = {
        "none": [],
        "length": [minimum_tokens_processor(mx, [0], len(history), 2)],
        "window": [make_presence_penalty(0.5, context_size=8)],
    }[kind]
    current = lane(history, processors)
    block = TreeDraftRow([1, 2, 1], [-1, -1, 0])
    fence, _ = batch._launch_tree_laws(current, block, mx.zeros((1, 4, 4)))
    mx.eval(*fence)
    assert history.copied <= (4 * 8 if kind == "window" else 0)


@pytest.mark.parametrize("context_size", [0, 1, 3, 50])
@pytest.mark.parametrize("start", [None, 0, 4, 7, 8, 20, -2])
def test_batched_tree_windows_match_full_history_reference(context_size, start):
    batch = executor()
    processors = [make_presence_penalty(1.5, context_size, generation_start=start)]
    reference = lane([0, 1, 2, 0, 1, 2, 0], processors)
    batched = lane(list(reference.history), processors)
    block = TreeDraftRow([1, 2, 1], [-1, -1, 0])
    logits = mx.array([[[0.1, 0.4, 0.3, 0.2]] * 4])
    expected = batch._verify_tree(reference, block, logits)
    fence, state = batch._launch_tree_laws(batched, block, logits)
    mx.eval(*fence)
    actual = batch._verify_tree_batched(batched, block, state)
    assert (actual.emitted, actual.commit_rows) == (expected.emitted, expected.commit_rows)
    assert batched.rng.snapshot() == reference.rng.snapshot()
    for actual_row, expected_row in zip(actual.response_logprobs, expected.response_logprobs):
        np.testing.assert_array_equal(np.asarray(actual_row), np.asarray(expected_row))
