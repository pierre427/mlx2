"""Original GQA7/D128 split-K attention over an implicit prefix/tree mask.

This is a separate numerical law, not a transcription of native vector SDPA.
The committed prefix is unmasked; only the <=32 node ancestor matrix is read.
Empty split partitions contribute exactly zero mass. See provenance/nanbeige43-tree.json.
"""

import hashlib
import math
from functools import lru_cache
from pathlib import Path

import mlx.core as mx

LAW = "nanbeige43-gqa7-d128-implicit-tree-split-k-v1"

_PARTIAL = r"""
    uint lane = thread_index_in_simdgroup;
    uint index = threadgroup_position_in_grid.x;
    uint split = threadgroup_position_in_grid.y;
    uint row = index % W;
    uint head = (index / W) % HQ;
    uint batch = index / (W * HQ);
    uint kvhead = head / 7;
    const int MAX_PREFIX = int(K_shape[2]) - W;
    const int SPAN = max(256, (int(K_shape[2]) + 127) / 128);
    const int SPLITS = (int(K_shape[2]) + SPAN - 1) / SPAN;
    int prefix = prefixes[batch];
    int left = MAX_PREFIX - prefix;
    int begin = split * SPAN;
    int end = min(begin + SPAN, MAX_PREFIX + W);
    float qq[4], acc[4];
    for (uint f=0; f<4; ++f) {
        qq[f] = float(Q[batch*Q_strides[0] + head*Q_strides[1] + row*Q_strides[2] + (lane+f*32)*Q_strides[3]]);
        acc[f] = 0.0f;
    }
    float mm = -INFINITY, zz = 0.0f;
    for (int pos=begin; pos<end; ++pos) {
        bool visible = pos >= left && (pos < MAX_PREFIX || ancestors[(batch*W+row)*W+pos-MAX_PREFIX]);
        if (!visible) continue;
        float score = 0.0f;
        for (uint f=0; f<4; ++f)
            score += qq[f] * float(K[batch*K_strides[0]+kvhead*K_strides[1]+pos*K_strides[2]+(lane+f*32)*K_strides[3]]);
        score = simd_sum(score) * SCALE;
        float next_m = max(mm, score);
        float old_scale = metal::exp(mm - next_m);
        float weight = metal::exp(score - next_m);
        for (uint f=0; f<4; ++f)
            acc[f] = acc[f]*old_scale + weight*float(V[batch*V_strides[0]+kvhead*V_strides[1]+pos*V_strides[2]+(lane+f*32)*V_strides[3]]);
        zz = zz*old_scale + weight;
        mm = next_m;
    }
    uint slot = index * SPLITS + split;
    if (lane == 0) { stats[slot*2] = mm; stats[slot*2+1] = zz; }
    for (uint f=0; f<4; ++f) partial[slot*128+lane+f*32] = acc[f];
"""
_COMBINE = r"""
    uint lane = thread_index_in_simdgroup;
    uint index = threadgroup_position_in_grid.x;
    const int SPLITS = int(stats_shape[0]) / COUNT;
    float mm = -INFINITY;
    for (uint s=0; s<SPLITS; ++s) {
        uint slot = index*SPLITS+s;
        if (stats[slot*2+1] > 0.0f) mm = max(mm, stats[slot*2]);
    }
    float denom = 0.0f, acc[4];
    for (uint f=0; f<4; ++f) acc[f] = 0.0f;
    for (uint s=0; s<SPLITS; ++s) {
        uint slot=index*SPLITS+s;
        float zz=stats[slot*2+1];
        if (zz == 0.0f) continue;
        float weight=metal::exp(stats[slot*2]-mm);
        denom += weight*zz;
        for (uint f=0; f<4; ++f) acc[f] += weight*partial[slot*128+lane+f*32];
    }
    for (uint f=0; f<4; ++f) Y[index*128+lane+f*32] = T(acc[f]/denom);
"""


def source_identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@lru_cache(maxsize=8)
def _kernels(scale):
    return (
        mx.fast.metal_kernel(name="nanbeige43_tree_partial", input_names=["Q", "K", "V", "prefixes", "ancestors"], output_names=["partial", "stats"], source=_PARTIAL, ensure_row_contiguous=False, header=f"constant constexpr float SCALE = {float(scale):.17g}f;"),
        mx.fast.metal_kernel(name="nanbeige43_tree_combine", input_names=["partial", "stats"], output_names=["Y"], source=_COMBINE),
    )


def validate(q, k, v, prefixes, ancestors):
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("implicit tree attention needs Q/K/V rank four")
    b, h, w, d = q.shape
    if not 1 <= b <= 4 or not 1 <= w <= 32 or d != 128 or h != 56 or k.shape[:2] != (b, 8):
        raise ValueError("implicit tree attention requires B1..4, W1..32, Q56/KV8/D128")
    if q.dtype != k.dtype or k.dtype != v.dtype or q.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise ValueError("implicit tree attention operands must share a floating dtype")
    if len(prefixes) != b or any(type(p) is not int or p < 0 for p in prefixes):
        raise ValueError("implicit tree prefixes must be nonnegative host lengths")
    if k.shape[2] != max(prefixes) + w or ancestors.shape != (b, w, w) or ancestors.dtype != mx.bool_:
        raise ValueError("implicit tree key/ancestor geometry mismatch")
    return b, h, w, d


def attend(q, k, v, *, prefixes, ancestors, scale, dispatch_trace=None):
    """Return [B,56,W,128], avoiding any device W*context mask on GPU."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("implicit tree scale must be positive and finite")
    b, h, w, d = validate(q, k, v, prefixes, ancestors)
    maximum = max(prefixes)
    if mx.default_device() == mx.cpu:
        masks = []
        for i, prefix in enumerate(prefixes):
            head = mx.concatenate([mx.zeros((w, maximum-prefix), dtype=mx.bool_), mx.ones((w, prefix), dtype=mx.bool_)], axis=1)
            masks.append(mx.concatenate([head, ancestors[i]], axis=1))
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mx.stack(masks)[:, None])
    # Bound scratch at 128 splits. Each partition owns only initialized slots.
    span = max(256, (maximum + w + 127) // 128)
    splits = (maximum + w + span - 1) // span
    count = b*h*w
    partial_kernel, combine_kernel = _kernels(scale)
    constants = [("T", q.dtype), ("W", w), ("HQ", h)]
    combine_constants = [("T", q.dtype), ("COUNT", count)]
    if dispatch_trace is not None:
        dispatch_trace.append({"prefixes": list(prefixes), "partial_template": [(key, str(value)) for key, value in constants],
                               "combine_template": [(key, str(value)) for key, value in combine_constants],
                               "splits": splits, "span": span, "scratch_bytes": count*splits*(d+2)*4,
                               "factory_cache": _kernels.cache_info()._asdict(),
                               "observation_scope": "actual dispatch templates and Python factory cache; not measured Metal compiler events"})
    partial, stats = partial_kernel(inputs=[q, k, v, mx.array(prefixes, dtype=mx.int32), ancestors], template=constants, grid=(count*32, splits, 1), threadgroup=(32, 1, 1), output_shapes=[(count*splits, d), (count*splits, 2)], output_dtypes=[mx.float32, mx.float32])
    out = combine_kernel(inputs=[partial, stats], template=combine_constants, grid=(count*32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[q.shape], output_dtypes=[q.dtype])
    return out[0]
