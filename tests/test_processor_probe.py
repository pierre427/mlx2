import math

import mlx.core as mx
import numpy as np

from mlx2.runtime.processor_probe import isolated_logits_processor
from mlx2.runtime.sample_utils import make_logits_processors, make_transformed_logprobs
from mlx2.serving import minimum_tokens_processor


def test_function_probe_preserves_immutable_closure_references_and_copies_mutable_state():
    calls = []
    array_module = math

    def processor(_tokens, logits):
        calls.append(array_module.pi)
        return logits

    isolated = isolated_logits_processor(processor)
    result = isolated(mx.array([1], dtype=mx.uint32), mx.zeros((1, 4)))

    mx.eval(result)
    assert calls == []


def test_every_stateless_serving_processor_declares_an_identical_probe():
    processors = {
        "sampling_transform": make_transformed_logprobs(
            0.7, top_p=0.8, top_k=4, min_p=0.05
        ),
        "minimum_tokens": minimum_tokens_processor(mx, [0, 7], 3, 4),
    }
    built = make_logits_processors(
        logit_bias={3: 1.25},
        repetition_penalty=1.1,
        repetition_context_size=8,
        presence_penalty=0.25,
        presence_context_size=8,
        frequency_penalty=0.125,
        frequency_context_size=8,
    )
    processors.update(dict(zip(
        ("logit_bias", "repetition_penalty", "presence_penalty", "frequency_penalty"),
        built,
        strict=True,
    )))
    assert set(processors) == {
        "sampling_transform",
        "minimum_tokens",
        "logit_bias",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
    }

    tokens = mx.array([1, 3, 1], dtype=mx.uint32)
    base = mx.arange(8, dtype=mx.float32)[None]
    for name, processor in processors.items():
        isolated = isolated_logits_processor(processor)
        assert isolated is processor, name
        if name == "sampling_transform":
            expected = processor(mx.array(base))
            actual = isolated(mx.array(base))
        else:
            expected = processor(tokens, mx.array(base))
            actual = isolated(tokens, mx.array(base))
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), rtol=0, atol=0, equal_nan=True
        )
