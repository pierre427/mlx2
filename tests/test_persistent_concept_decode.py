"""Request-scoped concept memory survives ordinary decode and stays B=1."""

import mlx.core as mx
import pytest

from mlx2.runtime.generate import PromptProcessingBatch, StopSequenceMatcher


class RecordingModel:
    def __init__(self):
        self.calls = []

    def __call__(self, inputs, *, cache, **kwargs):
        self.calls.append(kwargs)
        logits = mx.zeros((*inputs.shape, 8), dtype=mx.float32)
        return logits


def _persistent(memory):
    return {
        "deep_concept_memory": memory,
        "_mlx2_persistent_decode_inputs": {"deep_concept_memory": memory},
    }


def _prompt_batch(model, *, uids=(1,), inputs=None):
    uids = list(uids)
    return PromptProcessingBatch(
        model=model,
        uids=uids,
        caches=[[] for _ in uids],
        tokens=[[] for _ in uids],
        stop_matchers=[StopSequenceMatcher() for _ in uids],
        max_tokens=[3 for _ in uids],
        prefill_inputs=inputs,
    )


def test_persistent_memory_is_forwarded_from_prefill_through_decode():
    model = RecordingModel()
    memory = {"request": "a"}
    prompt = _prompt_batch(model, inputs=[_persistent(memory)])

    generation = prompt.generate([[1, 2]])
    generation.next()

    assert len(model.calls) == 3
    assert model.calls == [
        {"deep_concept_memory": memory},
        {"deep_concept_memory": memory},
        {"deep_concept_memory": memory},
    ]
    assert generation.has_persistent_inputs
    assert prompt.persistent_inputs == []


def test_ordinary_decode_has_no_persistent_kwargs():
    model = RecordingModel()
    generation = _prompt_batch(model, inputs=[None]).generate([[1, 2]])
    generation.next()

    assert model.calls == [{}, {}, {}]
    assert not generation.has_persistent_inputs


def test_persistent_decode_refuses_batched_prefill_and_generation_merge():
    memory = {"request": "a"}
    with pytest.raises(RuntimeError, match="isolated B=1"):
        _prompt_batch(
            RecordingModel(),
            uids=(1, 2),
            inputs=[_persistent(memory), None],
        )

    model = RecordingModel()
    ordinary = _prompt_batch(model, uids=(1,), inputs=[None]).generate([[1]])
    persistent = _prompt_batch(
        model, uids=(2,), inputs=[_persistent(memory)]
    ).generate([[2]])
    with pytest.raises(RuntimeError, match="cannot share a generation batch"):
        ordinary.extend(persistent)


def test_reserved_decode_payload_rejects_arbitrary_model_kwargs():
    with pytest.raises(ValueError, match="only deep_concept_memory"):
        _prompt_batch(
            RecordingModel(),
            inputs=[
                {
                    "_mlx2_persistent_decode_inputs": {
                        "deep_concept_memory": {},
                        "pixel_values": [],
                    }
                }
            ],
        )
