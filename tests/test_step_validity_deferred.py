"""MLX2_STEP_VALIDITY=deferred: a corrupt sampled row is dropped one forward later.

The default (``current``) drops the lane before the next forward; ``deferred``
lets that forward run so the next graph is built while the GPU finishes the
previous one.  Either way the corrupt token never enters history or a response
and the lane failure is reported identically.
"""
import mlx.core as mx
import pytest

from mlx2.runtime.generate import GenerationBatch, StopSequenceMatcher


@pytest.fixture(autouse=True)
def _cpu_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


class _TwoLaneModel:
    def __init__(self):
        self.inputs = []

    def __call__(self, inputs, cache):
        del cache
        self.inputs.append(inputs.tolist())
        rows = []
        for token in inputs[:, 0].tolist():
            rows.append([-float("inf")] * 3 if token == 8 else [0.0, 1.0, 2.0])
        return mx.array(rows, dtype=mx.float32)[:, None, :]


def _batch(model):
    return GenerationBatch(
        model,
        uids=[1, 2],
        inputs=mx.array([7, 8], dtype=mx.uint32),
        prompt_cache=[],
        tokens=[[], []],
        samplers=[],
        fallback_sampler=lambda rows: mx.argmax(rows, axis=-1),
        logits_processors=[],
        stop_matchers=[StopSequenceMatcher(), StopSequenceMatcher()],
        max_tokens=[3, 3],
    )


def test_deferred_drops_bad_lane_after_one_forward_never_in_history_or_response(monkeypatch):
    monkeypatch.setenv("MLX2_STEP_VALIDITY", "deferred")
    model = _TwoLaneModel()
    batch = _batch(model)  # the constructor ran the prompt-tail forward: [[7], [8]]

    # Lane 2's row was all -inf, so its sample (0) is corrupt.  The deferral's
    # cost: that row is fed to this step's forward; the lane is then dropped
    # at the tail, before its response or history entry exists.
    first = batch.next()
    assert [response.uid for response in first] == [1]
    assert [response.token for response in first] == [2]
    assert batch.uids == [1]
    assert batch.tokens == [[7, 2]]
    assert batch.take_lane_failures() == [
        {"uid": 2, "reason": "sampled token 0 has non-finite log probability"}
    ]
    assert model.inputs == [[[7], [8]], [[2], [0]]]
    # The surviving lane continues alone.
    second = batch.next()
    assert [response.uid for response in second] == [1]
    assert model.inputs[-1] == [[2]]


def test_current_default_drops_bad_lane_before_next_forward(monkeypatch):
    monkeypatch.delenv("MLX2_STEP_VALIDITY", raising=False)
    model = _TwoLaneModel()
    batch = _batch(model)
    first = batch.next()
    assert [response.uid for response in first] == [1]
    assert batch.tokens == [[7, 2]]
    assert batch.take_lane_failures() == [
        {"uid": 2, "reason": "sampled token 0 has non-finite log probability"}
    ]
    assert model.inputs == [[[7], [8]], [[2]]]
