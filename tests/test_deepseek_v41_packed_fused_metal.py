"""Real-Metal gate for the V4.1 fused packed-QAT codec and its fused-QAT route.

Run with ``MLX2_RUN_METAL_TESTS=1`` in an owned GPU window. Synthetic tensors
and a small synthetic model only: no checkpoint, speed or qualification claim.
"""

import os

import mlx.core as mx
import numpy as np
import pytest
from test_deepseek_v41_cpu import Weights, config

from mlx2.runtime.models import deepseek_v41_packed_fused as fused
from mlx2.runtime.models import deepseek_v41_packed_qat as codec
from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX2_RUN_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="set MLX2_RUN_METAL_TESTS=1 (owned GPU window) for the real-Metal gate",
)


@pytest.fixture(autouse=True)
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


def _same(a, b):
    mx.eval(a, b)
    assert np.array(a).tobytes() == np.array(b).tobytes()


@pytest.mark.parametrize("kind", ["fp8", "index_fp4", "compressed_fp4"])
@pytest.mark.parametrize("rows,dim,scale", [(1, 512, 1.0), (129, 512, 3e-3), (37, 128, 2.0**-120),
                                            (4096, 128, 11.0), (3, 32, 2.0**60)])
def test_fused_codec_matches_reference_codes_scales_and_values(kind, rows, dim, scale):
    fused.first_use_check()
    rng = np.random.default_rng(rows * dim)
    values = (rng.standard_t(3, size=(1, rows, dim)) * scale).astype(np.float32)
    values[0, 0, :8] = [0., -0., 1e-30, -1e-30, 448., -448., 6., -6.]
    x = mx.array(values)
    want = codec.produce(x, kind)
    got = codec.produce(x, kind, codec=fused.CODEC)
    _same(got.codes, want.codes)
    _same(got.scales, want.scales)
    _same(codec.decode(got), codec.reference_decode(want))
    ids = mx.array(rng.integers(0, rows, size=(1, 3, 17)).astype(np.uint32))
    _same(codec.gather_decode(got, ids), codec.gather_decode(want, ids))


def _config():
    return config(head_dim=32, index_head_dim=32, simulate_quantization=True,
                  activation_dtype="bfloat16", sliding_window=3,
                  index_topk=512, candidate_topk_blocks=2048)


def _model(c, **policies):
    model = DeepSeekV41Model(c, Weights(c), identity="synthetic@packed-fused-metal",
                             qat_policy="fused_fp32", index_policy="finite_all_retained",
                             selected_kv_policy="decode_reuse",
                             source_attention_math_policy="all_retained_broadcast", **policies)
    model.pointwise_policy = "fused"
    model.attention_epilogue_policy = "fused"
    return model


@pytest.mark.parametrize("candidate", ["reference", "candidate_only"])
def test_selected_fused_route_with_packed_kv_matches_reference_storage_bits(candidate):
    c = _config()
    ref = _model(c)
    packed = _model(c, kv_storage_policy="native_packed", candidate_score_policy=candidate)
    assert packed.packed_codec_policy == fused.CODEC
    rc, pc = ref.make_cache(), packed.make_cache()
    for chunk in ([1, 2, 3, 4, 5], [6], [7, 8, 9, 10, 11, 12, 13], [14], [15, 16]):
        tokens = mx.array([chunk])
        _same(packed(tokens, pc, logits_to_keep=0), ref(tokens, rc, logits_to_keep=0))
        for a, b in zip(rc, pc):
            for av, bv in zip(a.state, (b.keys, b.buf_kv, b.buf_gate, b.pooled, b.index_keys, b.hash_tail)):
                if av is None:
                    assert bv is None
                else:
                    _same(bv, av)
    assert packed.scheduled_kv_calls["packed_retained_staged_rows"] > 0
    assert fused.status()["first_use_checked"] is True


def test_production_width_fused_route_crosses_retained_bound_with_exact_bits():
    # Production D512/indexD128/W128 QAT widths, eight synthetic layers; the
    # ratio-one pool crosses 512 rows, leaving retained staging for scoring.
    c = config(num_hidden_layers=8, compress_ratios=(0, 2, 2, 1, 1, 1, 1, 1),
               kv_source_layer_ids=(1, 3), index_source_layer_ids=(1, 3, 5, 7),
               candidate_source_layer_id=3, head_dim=512, index_head_dim=128, sliding_window=128,
               qk_rope_head_dim=64, index_topk=512, candidate_topk_blocks=2048, candidate_block_size=8,
               simulate_quantization=True, activation_dtype="bfloat16")
    ref = _model(c)
    packed = _model(c, kv_storage_policy="native_packed", candidate_score_policy="candidate_only")
    a, b = ref.make_cache(), packed.make_cache()
    for length in (129, 131, 257, 3):
        ids = mx.array([[i % c.vocab_size for i in range(length)]])
        _same(packed(ids, b, logits_to_keep=1), ref(ids, a, logits_to_keep=1))
        _same(packed(mx.array([[1]]), b), ref(mx.array([[1]]), a))
    for u, v in zip(a, b):
        for left, right in zip(u.state, (v.keys, v.buf_kv, v.buf_gate, v.pooled, v.index_keys, v.hash_tail)):
            if left is None:
                assert right is None
            else:
                _same(right, left)
    assert a[3].pooled.shape[1] > 512
    assert packed.scheduled_kv_calls["packed_retained_staged_rows"] > 0
