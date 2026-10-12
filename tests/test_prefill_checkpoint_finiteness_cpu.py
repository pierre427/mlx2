"""Prompt chunks must be finite before recurrent state becomes publishable."""

import mlx.core as mx
import pytest
from route_harness import collect, make_engine, patch_host, tiny_qwen38_mtp

from mlx2.runtime.generate import PromptProcessingBatch


class _CheckpointRecorder:
    def __init__(self):
        self.calls = []
        self.state = []

    def state_checkpoint(self, positions, force=False):
        self.calls.append((tuple(positions), bool(force)))

    def prepare(self, **_kwargs):
        return None

    def finalize(self):
        return None


def _batch(model, recorder, *, uids):
    batch = object.__new__(PromptProcessingBatch)
    batch.uids = list(uids)
    batch.tokens = [[] for _ in uids]
    batch.prompt_cache = [recorder]
    batch.prefill_step_size = 2
    batch.model = model
    batch.decode_active = False
    return batch


class _CorruptSecondChunk:
    def __init__(self):
        self.calls = 0

    def __call__(self, tokens, *, cache):
        self.calls += 1
        logits = mx.zeros((tokens.shape[0], tokens.shape[1], 4), mx.float32)
        if self.calls == 2:
            logits = logits + mx.array(float("nan"), mx.float32)
        return logits


def test_nonfinite_chunk_never_records_or_forces_its_checkpoint():
    recorder = _CheckpointRecorder()
    batch = _batch(_CorruptSecondChunk(), recorder, uids=[17])

    with pytest.raises(RuntimeError, match="uid\\(s\\) \\[17\\].*publication was refused"):
        batch.prompt([[1, 2, 3, 4]])

    assert recorder.calls == [((2,), False)]


class _PaddedOutput:
    def __call__(self, tokens, *, cache):
        output = mx.zeros((2, tokens.shape[1], 3), mx.float32)
        # The second lane has one real token. Its right-padding rows are not
        # cache-committed outputs and must not fail the lane.
        if tokens.shape[1] > 1:
            padded = mx.full((tokens.shape[1] - 1, 3), float("nan"), mx.float32)
            output = mx.concatenate(
                [output[:1], mx.concatenate([output[1:2, :1], padded[None]], axis=1)],
                axis=0,
            )
        return output


def test_right_padding_outputs_do_not_authorize_or_reject_a_lane():
    recorder = _CheckpointRecorder()
    batch = _batch(_PaddedOutput(), recorder, uids=[20, 21])

    batch.prompt([[1, 2], [3]])

    assert recorder.calls[-1] == ((2, 1), True)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nan_and_positive_infinity_are_rejected(bad):
    recorder = _CheckpointRecorder()

    class Model:
        def __call__(self, tokens, *, cache):
            return mx.array([[[0.0, bad, -float("inf")]]], mx.float32)

    batch = _batch(Model(), recorder, uids=[30])
    with pytest.raises(RuntimeError, match="publication was refused"):
        batch.prompt([[1]])
    assert recorder.calls == []


def test_negative_infinity_mask_is_valid_when_a_finite_logit_exists():
    recorder = _CheckpointRecorder()

    class Model:
        def __call__(self, tokens, *, cache):
            return mx.array([[[0.0, -float("inf"), -float("inf")]]], mx.float32)

    batch = _batch(Model(), recorder, uids=[40])
    batch.prompt([[1]])
    assert recorder.calls[-1] == ((1,), True)


def test_all_masked_row_is_rejected():
    recorder = _CheckpointRecorder()

    class Model:
        def __call__(self, tokens, *, cache):
            return mx.full((1, 1, 3), -float("inf"), mx.float32)

    batch = _batch(Model(), recorder, uids=[50])
    with pytest.raises(RuntimeError, match="publication was refused"):
        batch.prompt([[1]])
    assert recorder.calls == []


@pytest.mark.parametrize(
    "output",
    [None, object(), mx.zeros((1, 1, 1, 1), mx.float32)],
)
def test_missing_or_unfamiliar_output_cannot_authorize_checkpoint(output):
    recorder = _CheckpointRecorder()

    class Model:
        def __call__(self, tokens, *, cache):
            return output

    batch = _batch(Model(), recorder, uids=[51])
    with pytest.raises(RuntimeError, match="validation projection|validation shape"):
        batch.prompt([[1]])
    assert recorder.calls == []


def test_unvalidated_checkpoint_bypass_is_explicitly_test_only():
    recorder = _CheckpointRecorder()

    class Model:
        _test_only_allow_unvalidated_prefill_checkpoint = True

        def __call__(self, tokens, *, cache):
            return None

    batch = _batch(Model(), recorder, uids=[52])
    batch.prompt([[1]])
    assert recorder.calls[-1] == ((1,), True)


def test_serving_drops_only_corrupt_prefill_lane_and_keeps_worker_and_peer(
    monkeypatch,
):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    real_call = type(model).__call__
    marker = vocab - 3
    observed_prefill_widths = []

    def prefill_forward(inputs, cache=None, **kwargs):
        observed_prefill_widths.append(int(inputs.shape[0]))
        logits = real_call(model, inputs, cache=cache, **kwargs)
        corrupt = mx.any(inputs == marker, axis=1).reshape((-1, 1, 1))
        return mx.where(corrupt, mx.array(float("nan"), logits.dtype), logits)

    object.__setattr__(model, "prefill_forward", prefill_forward)
    engine = make_engine(
        model,
        vocab,
        mtp=False,
        max_lanes=2,
        coalesce_window_ms=50,
    )
    try:
        bad_prompt = [2, marker] + [3 + (i % 13) for i in range(38)]
        good_prompt = [2, 4] + [5 + (i % 13) for i in range(38)]
        bad = engine.submit(
            {"tokens": bad_prompt, "max_tokens": 3, "temperature": 0}
        )
        good = engine.submit(
            {"tokens": good_prompt, "max_tokens": 3, "temperature": 0}
        )

        failed = collect(bad)
        served = collect(good)

        assert failed["status"] == 500
        assert "cache checkpoint publication was refused" in failed["error"]
        assert "error" not in served, served
        assert served["finish"] == "length"
        assert 2 in observed_prefill_widths
        assert engine.error is None
        assert engine.counts["prefill_output_corruption_events"] == 1
        assert engine.counts["prefill_output_corruption_failed_requests"] == 1
    finally:
        engine.close()
        mx.set_default_device(previous)
