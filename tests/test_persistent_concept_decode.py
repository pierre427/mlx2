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


def test_capsule_schedule_advances_then_stops_injecting():
    model = RecordingModel()
    schedule = mx.arange(8, dtype=mx.float32).reshape(2, 4)
    memory = {"request": "capsule", "decode_values": schedule}
    generation = _prompt_batch(
        model, inputs=[_persistent(memory)]
    ).generate([[1, 2]])
    generation.next()
    generation.next()

    assert model.calls[0]["deep_concept_memory"] is memory
    assert mx.array_equal(
        model.calls[1]["deep_concept_memory"]["values"], schedule[0:1]
    ).item()
    assert mx.array_equal(
        model.calls[2]["deep_concept_memory"]["values"], schedule[1:2]
    ).item()
    assert model.calls[3] == {}


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


def test_batch_generator_serializes_capsule_lane_before_ordinary_lane():
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel

    args = TextModelArgs(
        model_type="qwen3_5",
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        vocab_size=64,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
        full_attention_interval=4,
        mtp_num_hidden_layers=0,
        partial_rotary_factor=0.5,
        rope_parameters=None,
        max_position_embeddings=128,
    )
    mx.random.seed(9)
    model = TextModel(args)
    model.eval()
    mx.eval(model.parameters())
    keys = mx.ones((1, 32), dtype=mx.float32)
    values = mx.eye(32, dtype=mx.float32)[:1]
    memory = {
        "keys": keys,
        "values": values,
        "decode_values": mx.eye(32, dtype=mx.float32)[:2],
        "layer": 2,
        "temperature": 0.1,
        "gate": 0.1,
    }
    generator = BatchGenerator(
        model,
        completion_batch_size=2,
        prefill_batch_size=2,
        prefill_step_size=8,
        prefill_batch_window=1,
    )
    persistent_uid = generator.insert(
        [[1, 2]],
        max_tokens=[2],
        prefill_inputs=[_persistent(memory)],
    )[0]
    ordinary_uid = generator.insert([[3, 4]], max_tokens=[1])[0]

    generated = []
    for _ in range(20):
        _prompt, responses = generator.next()
        generated.extend(responses)
        if {response.uid for response in generated if response.finish_reason} == {
            persistent_uid,
            ordinary_uid,
        }:
            break

    assert {response.uid for response in generated if response.finish_reason} == {
        persistent_uid,
        ordinary_uid,
    }
    assert all(response.execution_width == 1 for response in generated)
