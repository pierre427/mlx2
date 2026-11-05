"""Packed-KV consumers for the selected Source laws; synthetic CPU contracts only.

No checkpoint, GPU equivalence, speed or qualification claim. GPU bitwise
checks for the fused codec live in test_deepseek_v41_packed_fused_metal.py.
"""

import mlx.core as mx
import numpy as np
import pytest
from test_deepseek_v41_cpu import Weights, config

from mlx2.adapters.deepseek_v41 import execution_policy as gguf_policy
from mlx2.adapters.deepseek_v41_execution import (
    CONTEXT_GATE_TOKENS,
    execution_receipt,
    model_options,
    validate_csa_policy,
)
from mlx2.adapters.deepseek_v41_mlx import execution_policy as native_policy
from mlx2.adapters.deepseek_v41_source import source_policy
from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
from mlx2.runtime.models.deepseek_v41_packed_cache import (
    DeepSeekV41PackedLayerCache,
    stage_retained_pool,
)
from mlx2.runtime.models.deepseek_v41_packed_qat import PackedQAT, decode, produce


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    def forbidden(*args, **kwargs):
        raise AssertionError("synthetic CPU test attempted Metal")
    monkeypatch.setattr(mx.fast, "metal_kernel", forbidden)
    yield
    mx.set_default_device(previous)


def _bits(x):
    return np.array(x).view(np.uint32)


def _config():
    return config(head_dim=32, index_head_dim=32, simulate_quantization=True,
                  activation_dtype="bfloat16", sliding_window=3,
                  index_topk=512, candidate_topk_blocks=2048)


def _model(c, **policies):
    return DeepSeekV41Model(c, Weights(c), identity="synthetic@packed-consumers",
                            index_policy="finite_all_retained", selected_kv_policy="decode_reuse",
                            **policies)


@pytest.mark.parametrize("math", ["all_retained_broadcast", "omlx_nax_bf16_online_v1"])
@pytest.mark.parametrize("candidate", ["reference", "candidate_only"])
def test_packed_kv_under_selected_attention_math_matches_reference_storage_bits(monkeypatch, math, candidate):
    c = _config()
    ref = _model(c, source_attention_math_policy=math)
    packed = _model(c, source_attention_math_policy=math, kv_storage_policy="native_packed",
                    candidate_score_policy=candidate)
    assert packed.packed_codec_policy == "reference"
    assert ":packed_codec=" not in packed.identity
    rc, pc = ref.make_cache(), packed.make_cache()
    def forbidden(self):
        raise AssertionError("packed consumer materialized a complete floating cache plane")
    for chunk in ([1, 2, 3, 4, 5], [6], [7, 8, 9, 10, 11, 12, 13], [14], [15, 16]):
        tokens = mx.array([chunk])
        expected = ref(tokens, rc, logits_to_keep=0)
        with monkeypatch.context() as patch:
            patch.setattr(DeepSeekV41PackedLayerCache, "pooled", property(forbidden))
            patch.setattr(DeepSeekV41PackedLayerCache, "index_keys", property(forbidden))
            actual = packed(tokens, pc, logits_to_keep=0)
        np.testing.assert_array_equal(_bits(actual), _bits(expected))
        for a, b in zip(rc, pc):
            for av, bv in zip(a.state, (b.keys, b.buf_kv, b.buf_gate, b.pooled, b.index_keys, b.hash_tail)):
                if av is None:
                    assert bv is None
                else:
                    np.testing.assert_array_equal(np.array(av), np.array(bv))
    counts = packed.scheduled_kv_calls
    assert counts["packed_retained_staged_rows"] > 0
    assert counts["packed_retained_staging_max_rows"] <= 512
    if math == "all_retained_broadcast":
        assert counts["source_all_retained_broadcast_attention"] == ref.scheduled_kv_calls["source_all_retained_broadcast_attention"]
    assert counts["decode_gather_build"] > 0
    assert sum(s.nbytes for s in pc) < sum(s.nbytes for s in rc)


def _marker(shared, indices, count, length, start):
    shared["_index_generation"] = 3
    shared["_all_retained_chronological"] = (id(indices), count, length, start, 3)


@pytest.mark.parametrize("count,length,policy,stale,staged", [
    (512, 2, "finite_all_retained", False, True),
    (513, 2, "finite_all_retained", False, False),
    (8, 1, "finite_all_retained", False, False),
    (8, 2, "reference", False, False),
    (8, 2, "finite_all_retained", True, False),
])
def test_retained_staging_is_bounded_and_requires_trusted_marker(count, length, policy, stale, staged):
    from types import SimpleNamespace

    rng = np.random.default_rng(count)
    pool = produce(mx.array(rng.normal(size=(1, count, 32)).astype(np.float32)), "compressed_fp4")
    indices = mx.broadcast_to(mx.arange(count, dtype=mx.uint32)[None, None], (1, length, count))
    shared = {}
    _marker(shared, indices, count, length, 40)
    if stale:
        shared["_index_generation"] = 4
    model = SimpleNamespace(index_policy=policy, scheduled_kv_calls={})
    dense, remaining = stage_retained_pool(model, pool, indices, length, 40, shared)
    if staged:
        assert remaining is None
        np.testing.assert_array_equal(_bits(dense), _bits(decode(pool)))
        assert model.scheduled_kv_calls["packed_retained_staging_max_rows"] == count
    else:
        assert dense is None and remaining is pool and not model.scheduled_kv_calls


def test_fused_qat_selects_salted_exact_codec_and_fails_closed_off_gpu():
    c = _config()
    reference = _model(c, kv_storage_policy="native_packed", source_attention_math_policy="all_retained_broadcast")
    fused = _model(c, kv_storage_policy="native_packed", source_attention_math_policy="all_retained_broadcast",
                   qat_policy="fused_fp32")
    from mlx2.runtime.models.deepseek_v41_packed_fused import (
        CODEC,
        implementation_identity,
    )

    assert fused.packed_codec_policy == CODEC
    assert ":packed_codec=" + CODEC + ":packed_codec_implementation=" + implementation_identity() in fused.identity
    assert reference.identity != fused.identity
    cache = fused.make_cache()
    assert {state.codec for state in cache} == {CODEC}
    assert {state.codec for state in reference.make_cache()} == {"reference"}
    # Restored state keeps the codec through its identity, not its bytes.
    restored = DeepSeekV41PackedLayerCache.from_state(cache[1].state, cache[1].meta_state, xp=mx)
    assert restored.codec == CODEC
    assert execution_receipt(fused)["packed_codec"] == CODEC
    assert "packed_codec" not in execution_receipt(_model(c))
    value = produce(mx.ones((1, 32), mx.float32), "fp8")
    with pytest.raises(ValueError, match="owned Metal GPU"):
        decode(PackedQAT(value.codes, value.scales, "fp8", CODEC))
    with pytest.raises(ValueError, match="owned Metal GPU"):
        produce(mx.ones((1, 32), mx.float32), "fp8", codec=CODEC)


@pytest.mark.parametrize("parse", [gguf_policy, native_policy, source_policy])
@pytest.mark.parametrize("limit,packed", [(None, False), (4096, False), (CONTEXT_GATE_TOKENS, False),
                                          (CONTEXT_GATE_TOKENS + 1, True), (262144, True), (1048576, True)])
def test_context_gate_resolves_to_concrete_storage_and_scoring(parse, limit, packed):
    request = {"deepseek_kv_storage": "context_gated_v1"}
    if limit is not None:
        request["deepseek_context_limit"] = limit
    result = parse(request)
    options = model_options(result)
    assert options["kv_storage_policy"] == ("native_packed" if packed else "reference")
    assert options["candidate_score_policy"] == ("candidate_only" if packed else "reference")
    # Resolution is idempotent, so repeated parsing cannot change the route.
    assert model_options(parse({k: v for k, v in result.items() if k in request or k.startswith("deepseek_kv")
                                or k == "deepseek_candidate_score"})) == options


def test_context_gate_keeps_explicit_candidate_scoring_at_short_context():
    policy = {"deepseek_kv_storage": "context_gated_v1", "deepseek_candidate_score": "candidate_only"}
    validate_csa_policy(policy)
    assert policy == {"deepseek_kv_storage": "reference", "deepseek_candidate_score": "candidate_only"}
    with pytest.raises(ValueError, match="integer context"):
        validate_csa_policy({"deepseek_kv_storage": "context_gated_v1", "deepseek_context_limit": "65537"})


def test_selected_source_bundle_composes_with_packed_candidate_long_context():
    from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter

    selected = dict(DeepSeekV41SourceAdapter.default_route_execution_policy["ordinary"])
    for request in ({"deepseek_kv_storage": "native_packed", "deepseek_candidate_score": "candidate_only",
                     "deepseek_context_limit": 262144},
                    {"deepseek_kv_storage": "context_gated_v1", "deepseek_context_limit": 262144}):
        result = source_policy({**selected, **request})
        assert result["deepseek_qat"] == "fused_fp32"
        assert result["deepseek_source_attention_math"] == "all_retained_broadcast"
        assert model_options(result) == {"kv_storage_policy": "native_packed", "candidate_score_policy": "candidate_only",
                                         "ced_policy": "reference", "ced_state_only": False,
                                         "context_limit": 262144}
    for math in ("source_short_decode_joint_f32_v1", "source_short_decode_grouped_f32_v1"):
        with pytest.raises(ValueError, match="broadcast or oMLX"):
            source_policy({**selected, "deepseek_kv_storage": "native_packed", "deepseek_source_attention_math": math})
