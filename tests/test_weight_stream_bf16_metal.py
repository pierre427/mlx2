"""Real-Metal gate: bf16 unquantized expert streaming is bit-identical.

Run under the GPU lock with ``MLX2_RUN_METAL_TESTS=1``.  CPU ``gather_mm`` is
float32-only, so the bf16 tables of the Laguna S 2.1 and Nemotron 3.5 Super
releases are checked here: tiny real models with those vendors' per-expert
bf16 checkpoints, resident versus streamed at a capacity that evicts, for an
unsorted (short) and a sorted (long) gather.  Logits must match bit for bit.
"""

import os

import mlx.core as mx
import numpy as np
import pytest

from tests.test_weight_stream_unquantized_auto_cpu import (
    LONG_PROMPT,
    SHORT_PROMPT,
    _laguna,
    _laguna_checkpoint,
    _logits,
    _nemotron,
    _nemotron_checkpoint,
    _resident,
    _streamed,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX2_RUN_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="set MLX2_RUN_METAL_TESTS=1 (under the GPU lock) for the real-Metal gate",
)


@pytest.fixture(autouse=True)
def _gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


@pytest.mark.parametrize(
    ("build", "checkpoint", "streams", "row"),
    [
        (_laguna, _laguna_checkpoint, 6, 32 * 64 * 2),
        (_nemotron, _nemotron_checkpoint, 4, 24 * 16 * 2),
    ],
    ids=["laguna-s21", "nemotron-h-latent"],
)
def test_bf16_streamed_forward_is_bit_identical_on_metal(tmp_path, build, checkpoint,
                                                         streams, row):
    root, weight_map, records = checkpoint(tmp_path, mx.bfloat16)
    resident = _resident(build(), root, weight_map)
    expected = {len(p): _logits(resident, p) for p in (SHORT_PROMPT, LONG_PROMPT)}
    model = build()
    loaded = _streamed(model, root, weight_map, records, top_k=2,
                       budget=streams * 2 * row)
    try:
        assert loaded.manager.plan.expert_bytes == row
        for prompt in (SHORT_PROMPT, LONG_PROMPT, SHORT_PROMPT):
            assert np.array_equal(_logits(model, prompt), expected[len(prompt)])
        assert loaded.manager.counters()["stream_evictions_total"] > 0
    finally:
        loaded.manager.close()
