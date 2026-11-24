"""Real-Metal gate: Nemotron 3.6 Academia GDP on the Metal GDN kernel.

Run under the GPU lock with ``MLX2_RUN_METAL_TESTS=1``.  The CPU suite
(test_nemotron36_academia_cpu.py) checks the model against a NumPy port of
llama.cpp on the ops recurrence; this one checks that the Metal kernel path
(rectangular Dk=64 x Dv=32 state, Householder sub-steps flattened into time)
gives the same logits as that CPU ops path, synthetic weights only.
"""

import os

import mlx.core as mx
import numpy as np
import pytest

from test_nemotron36_academia_cpu import config, hf_weights, mlx_model, nmse, split, tokens

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX2_RUN_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="set MLX2_RUN_METAL_TESTS=1 (under the GPU lock) for the real-Metal gate",
)


def _logits(cfg, weights, chunks, device):
    previous = mx.default_device()
    mx.set_default_device(device)
    try:
        model = mlx_model(cfg, weights)
        cache = model.make_cache()
        rows = []
        for chunk in chunks:
            out = model(mx.array([chunk]), cache=cache)
            mx.eval(out)
            rows.append(np.array(out[0], dtype=np.float64))
        return np.concatenate(rows)
    finally:
        mx.set_default_device(previous)


@pytest.mark.parametrize("mode", ["per_loop", "last_loop"])
@pytest.mark.parametrize("householder", [1, 3])
def test_metal_kernel_matches_cpu_ops(mode, householder):
    cfg = config(recurrent_kv_cache_mode=mode, n_recurrent_loops=3,
                 linear_num_householder=householder)
    weights = hf_weights(cfg, seed=11)
    prompt = tokens(21, seed=13)
    chunks = split(prompt[:16], 8) + [[t] for t in prompt[16:]]
    cpu = _logits(cfg, weights, chunks, mx.cpu)
    gpu = _logits(cfg, weights, chunks, mx.gpu)
    assert nmse(cpu, gpu) < 1e-8
    assert (cpu.argmax(-1) == gpu.argmax(-1)).mean() > 0.95
