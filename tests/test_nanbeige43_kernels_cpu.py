"""Synthetic candidate contracts; no model qualification or GPU speed claim."""

import copy
import hashlib
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from mlx2.adapters import nanbeige43_fused as fused
from mlx2.adapters import nanbeige43_kernels as kernels
from mlx2.runtime.models.cache import KVCache
from mlx2.runtime.models.standard_decoder import Model, ModelArgs


@pytest.fixture(autouse=True)
def cpu_receipt(record_property):
    mx.set_default_device(mx.cpu)
    root = Path(__file__).resolve().parents[1]
    record_property(
        "source_revision",
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
    )
    record_property(
        "working_source_sha256", kernels.kernel_contract(None)["source_sha256"]
    )


def tiny(record_property):
    mx.random.seed(4276)
    model = Model(
        ModelArgs(
            model_type="llama",
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=64,
            rms_norm_eps=1e-5,
            vocab_size=32,
            max_position_embeddings=512,
            rope_theta=120000000,
            tie_word_embeddings=False,
        )
    )
    nn.quantize(model, bits=8, group_size=64)
    mx.eval(model.parameters())
    digest = hashlib.sha256()
    for name, value in tree_flatten(model.parameters()):
        digest.update(name.encode())
        digest.update(np.asarray(value.astype(mx.float32)).tobytes())
    record_property(
        "checkpoint", "synthetic-nanbeige43-kernel-q8g64@sha256:" + digest.hexdigest()
    )
    return model


@pytest.mark.parametrize(
    "bad",
    [
        [],
        {"unknown": True},
        {"projection_min_rows": True},
        {"projection_min_rows": 33, "projection_max_rows": 32},
        {"q8_tile_rows": 8},
        {"projection_split_k": 3},
        {"compiled_blocks": 1},
    ],
)
def test_policy_refuses_bad_values(bad, record_property):
    record_property("checkpoint", "synthetic-nanbeige43-policy@v1")
    with pytest.raises((ValueError, TypeError)):
        kernels.validate_kernel_policy(bad)


def test_q8_source_has_byte_addressing_and_final_output_only(record_property):
    record_property("checkpoint", "synthetic-nanbeige43-q8-source@v1")
    source, names = kernels.q8_swiglu_source(256, 16)
    assert "(K / 4) + k / 4] >> ((k % 4) * 8)) & 255u" in source
    assert "(K / 8)" not in source
    assert "constexpr int TM = 16" in source
    assert "nanbeige_sigmoid<bfloat>" in source
    assert names == ["X", "M", "W0", "S0", "B0", "W1", "S1", "B1"]
    assert "OUT[size_t(row) * N + col] = value" in source


def test_pure_compiled_blocks_consecutive_calls(record_property):
    ordinary = tiny(record_property)
    candidate = copy.deepcopy(ordinary)
    fused.install(ordinary)
    fused.install(candidate, kernel_policy={"compiled_blocks": True})
    caches = [ordinary.make_cache(), candidate.make_cache()]
    for tokens in ([[1, 2, 3]], [[4]], [[5, 6]]):
        x = mx.array(tokens)
        a, b = ordinary(x, caches[0]), candidate(x, caches[1])
        mx.eval(a, b)
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-5)
        assert [c.offset for c in caches[0]] == [c.offset for c in caches[1]]
    counts = fused.status(candidate)["scheduled_kernel_counts"]
    assert counts["compiled_mlp_calls"] == 6
    assert counts["compiled_projection_calls"] == 6


def test_rope_append_two_offsets_matches_reference(record_property):
    record_property("checkpoint", "synthetic-nanbeige43-rope-cache@seed4276-v1")
    mx.random.seed(4276)
    rope = nn.RoPE(64, base=120000000, traditional=False)
    a, b = KVCache(), KVCache()
    first = mx.random.normal((1, 1, 3, 64))
    a.update_and_fetch(rope(first), first)
    b.update_and_fetch(rope(first), first)
    for length in (2, 1):
        q = mx.random.normal((1, 2, length, 64))
        k, v = (mx.random.normal((1, 1, length, 64)) for _ in range(2))
        expected_q = rope(q, offset=a.offset)
        expected_k, expected_v = a.update_and_fetch(rope(k, offset=a.offset), v)
        actual_q, actual_k, actual_v = kernels.rope_append(q, k, v, rope, b)
        mx.eval(expected_q, expected_k, expected_v, actual_q, actual_k, actual_v)
        for expected, actual in [
            (expected_q, actual_q),
            (expected_k, actual_k),
            (expected_v, actual_v),
        ]:
            np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))
        assert a.offset == b.offset


def test_rope_append_declines_before_mutation(record_property):
    record_property("checkpoint", "synthetic-nanbeige43-rope-refusal@v1")
    cache = KVCache()
    q = mx.zeros((1, 2, 1, 64))
    k = mx.zeros((1, 1, 1, 64))
    assert kernels.rope_append(q, k, k, nn.RoPE(64), cache) is None
    assert cache.offset == 0 and cache.keys is None
    cache.update_and_fetch(mx.zeros((1, 1, 256, 64)), mx.zeros((1, 1, 256, 64)))
    old = cache.keys
    assert kernels.rope_append(q, k, k, nn.RoPE(64), cache) is None
    assert cache.offset == 256 and cache.keys is old


def test_compiled_rope_policy_model_continuation(record_property):
    ordinary = tiny(record_property)
    candidate = copy.deepcopy(ordinary)
    fused.install(ordinary)
    fused.install(candidate, kernel_policy={"rope_kv_append": True})
    a, b = ordinary.make_cache(), candidate.make_cache()
    for tokens in ([[1, 2, 3]], [[4]], [[5, 6]]):
        x = mx.array(tokens)
        y, z = ordinary(x, a), candidate(x, b)
        mx.eval(y, z)
        np.testing.assert_allclose(np.asarray(y), np.asarray(z), rtol=1e-5, atol=1e-5)
        assert [c.offset for c in a] == [c.offset for c in b]
    assert fused.status(candidate)["scheduled_kernel_counts"]["rope_append_calls"] == 4
