# SPDX-License-Identifier: MIT
"""Native IQ1_S row-invariant groups, with no selected-weight concatenation.

Reuse decoded weights across independent token accumulators, retaining each
lane's original k order and the original slot-order final sum. Six expert
bindings keep a gate/up group below 28MB for the admitted artifact. See the
oMLX, DwarfStar and local FlashNext entries in deepseek-v41-adapter.json.
"""

from functools import lru_cache

import numpy as np

from .reader import GGUFError
from .v41_metal import BF16_HEADER

BROADCAST = r"""
inline float v41_iq1_broadcast(device const uchar *row, uint k, uint lane) {
    device const uchar *p=row+(k/256)*50;
    uint i=k%256, group=i/32, j=(i%32)/8;
    uint qh=lane==0 ? uint(p[34+2*group])|(uint(p[35+2*group])<<8) : 0;
    qh=simd_broadcast(qh,0);
    float d=lane==0 ? gguf_half(p) : 0.0f;
    d=simd_broadcast(d,0);
    uint index=(lane%8)==0 ? uint(p[2+4*group+j])|(((qh>>(3*j))&7)<<8) : 0;
    index=simd_shuffle(index,lane&~7u);
    char value=as_type<char>(uchar((iq1_grid[index]>>((i%8)*8))&255));
    float delta=(qh&0x8000) ? -0.125f : 0.125f;
    return d*float(2*((qh>>12)&7)+1)*(float(value)+delta);
}
"""


def plan(ids, experts, reuse, group_experts=6):
    ids = np.asarray(ids)
    if (ids.dtype != np.uint32 or ids.ndim != 2 or not 1 <= ids.shape[0] <= 1024
            or not 1 <= ids.shape[1] <= 6 or not 1 <= experts <= 384
            or int(ids.max()) >= experts or reuse not in (1, 2, 4) or group_experts not in (6, 10)):
        raise GGUFError("invalid row-exact IQ1 routing geometry")
    order = np.argsort(ids.ravel(), kind="stable").astype(np.uint32)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size, dtype=np.uint32)
    offsets = np.concatenate(([0], np.bincount(ids.ravel(), minlength=experts).cumsum()))
    groups = []
    for first in range(0, experts, group_experts):
        last = min(first + group_experts, experts)
        begin, end = int(offsets[first]), int(offsets[last])
        if begin == end:
            continue
        tiles = [(row-begin, expert-first, min(reuse, int(offsets[expert+1])-row))
                 for expert in range(first, last)
                 for row in range(int(offsets[expert]), int(offsets[expert+1]), reuse)]
        groups.append((first, last, order[begin:end], np.asarray(tiles, np.uint32)))
    return inverse, groups


def kernel_spec(kind, count, broadcast=False):
    if kind not in ("gate", "down") or not 1 <= count <= 10:
        raise GGUFError("invalid row-exact IQ1 binding")

    def choose(name):
        return "("+"".join(f"expert=={i}?{name}{i}:" for i in range(count-1))+f"{name}{count-1})"

    def value(name):
        return f"v41_iq1_broadcast({name},k,lane)" if broadcast else f"gguf_value({name},k,19)"

    common = r"""
const uint lane=thread_index_in_simdgroup, row=threadgroup_position_in_grid.y*SIMDS+simdgroup_index_in_threadgroup;
const uint tile=threadgroup_position_in_grid.z;
const uint first=tiles[3*tile], expert=tiles[3*tile+1], count=tiles[3*tile+2];
"""
    if kind == "gate":
        names = ["x", "order", "tiles", "scores"] + [f"g{i}" for i in range(count)] + [f"u{i}" for i in range(count)]
        source = common + f"device const uchar *g={choose('g')}+size_t(row)*ROW_BYTES, *u={choose('u')}+size_t(row)*ROW_BYTES;\n"
        source += r"""
float sg[ROWS],su[ROWS];
for(uint r=0;r<ROWS;r++) {sg[r]=0.0f;su[r]=0.0f;}
for(uint k=lane;k<K;k+=32) {
""" + f"float wg={value('g')}, wu={value('u')};\n" + r"""
    for(uint r=0;r<ROWS;r++) if(r<count) {
        uint token=order[first+r]/TOPK;
        float a=x[size_t(token)*K+k];
        sg[r]+=a*wg; su[r]+=a*wu;
    }
}
for(uint r=0;r<ROWS;r++) {
    float gsum=simd_sum(sg[r]), usum=simd_sum(su[r]);
    if(lane==0 && r<count) {
        gsum=min(v41_bf16(gsum),10.0f); usum=clamp(v41_bf16(usum),-10.0f,10.0f);
        y[size_t(first+r)*N+row]=v41_bf16((gsum/(1.0f+exp(-gsum)))*usum*scores[order[first+r]]);
    }
}
"""
    else:
        names = ["x", "tiles"] + [f"w{i}" for i in range(count)]
        source = common + f"device const uchar *w={choose('w')}+size_t(row)*ROW_BYTES;\n"
        source += r"""
float sums[ROWS];for(uint r=0;r<ROWS;r++) sums[r]=0.0f;
for(uint k=lane;k<K;k+=32) {
"""+f"float weight={value('w')};\n"+r"""
    for(uint r=0;r<ROWS;r++) if(r<count) sums[r]+=x[size_t(first+r)*K+k]*weight;
}
for(uint r=0;r<ROWS;r++) {
    float sum=simd_sum(sums[r]);
    if(lane==0 && r<count) y[size_t(first+r)*N+row]=v41_bf16(sum);
}
"""
    return names, source


@lru_cache(maxsize=40)
def kernel(kind, count, broadcast):
    import mlx.core as mx

    names, source = kernel_spec(kind, count, broadcast)
    return mx.fast.metal_kernel(name=f"mlx2_v41_exact_{kind}_{count}_{int(broadcast)}_v1",
                               input_names=names, output_names=["y"], header=BF16_HEADER+BROADCAST,
                               source=source, ensure_row_contiguous=True)


@lru_cache(maxsize=1)
def combine_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(name="mlx2_v41_exact_combine_v1", input_names=["rows", "inverse"],
                               output_names=["y"], source=r"""
uint row=thread_position_in_grid.x, token=thread_position_in_grid.y;
if(row>=N) return;
float total=0.0f;
for(uint slot=0;slot<TOPK;slot++) total+=rows[size_t(inverse[token*TOPK+slot])*N+row];
y[size_t(token)*N+row]=total;
""", ensure_row_contiguous=True)


def _moe(x, ids, scores, gate, up, down, *, reuse=1, broadcast=False, counters=None, group_experts=6, simd_groups=1):
    import mlx.core as mx

    count = len(gate)
    if not count or len(up) != count or len(down) != count:
        raise GGUFError("row-exact expert planes disagree")
    _, hidden, width = gate[0].shape
    if (any(w.ggml_type != 19 or w.shape != (1, hidden, width) for w in (*gate, *up))
            or any(w.ggml_type != 19 or w.shape != (1, width, hidden) for w in down)
            or x.ndim != 2 or x.shape[-1] != width or scores.shape != ids.shape
            or ids.shape[0] != x.shape[0] or width % 256 or hidden % 256):
        raise GGUFError("invalid row-exact IQ1 planes")
    if simd_groups not in (1,4):
        raise GGUFError("row-exact SIMD group bound")
    inverse, groups = plan(ids, count, reuse, group_experts)
    projected = []
    for begin, end, order, tiles in groups:
        tile_array = mx.array(tiles)
        mid = kernel("gate", end-begin, broadcast)(
            inputs=[x.astype(mx.float32), mx.array(order), tile_array, scores.reshape(-1).astype(mx.float32)]
                   +[w.data for w in gate[begin:end]]+[w.data for w in up[begin:end]],
            template=[("TOPK", ids.shape[1]), ("K", width), ("N", hidden), ("ROWS", reuse), ("SIMDS", simd_groups), ("ROW_BYTES", gate[0].row_bytes)],
            grid=(32*simd_groups, hidden//simd_groups, len(tiles)), threadgroup=(32*simd_groups, 1, 1),
            output_shapes=[(len(order), hidden)], output_dtypes=[mx.float32])[0]
        projected.append(kernel("down", end-begin, broadcast)(
            inputs=[mid, tile_array]+[w.data for w in down[begin:end]],
            template=[("K", hidden), ("N", width), ("ROWS", reuse), ("SIMDS", simd_groups), ("ROW_BYTES", down[0].row_bytes)],
            grid=(32*simd_groups, width//simd_groups, len(tiles)), threadgroup=(32*simd_groups, 1, 1),
            output_shapes=[(len(order), width)], output_dtypes=[mx.float32])[0])
    rows = projected[0] if len(projected) == 1 else mx.concatenate(projected)
    if counters is not None:
        counters["exact_calls"] = counters.get("exact_calls", 0)+1
        counters["exact_groups"] = counters.get("exact_groups", 0)+len(groups)
        counters["exact_tiles"] = counters.get("exact_tiles", 0)+sum(len(g[3]) for g in groups)
        counters["exact_activation_concat_bytes"] = counters.get("exact_activation_concat_bytes", 0)+(rows.nbytes if len(groups)>1 else 0)
    return combine_kernel()(inputs=[rows, mx.array(inverse)],
                            template=[("N", width), ("TOPK", ids.shape[1])],
                            grid=(((width+255)//256)*256, ids.shape[0], 1), threadgroup=(256, 1, 1),
                            output_shapes=[x.shape], output_dtypes=[mx.float32])[0]


@lru_cache(maxsize=24)
def self_check(reuse=1, broadcast=False, group_experts=6, simd_groups=1):
    import mlx.core as mx
    from dataclasses import replace
    from .packed_metal import PackedMatrix
    from .reader import GGUFTensor
    from .v41_metal import moe as reference

    rng = np.random.default_rng(41196)
    matrices, planes = [], []
    for kind in ("gate", "up", "down"):
        raw = rng.integers(0, 256, (11*256, 50), np.uint8)
        raw[:, :2] = np.frombuffer(np.float16(.03125).tobytes(), np.uint8)
        tensor = GGUFTensor(kind, (256, 256, 11), 19, 0, raw.nbytes)
        matrices.append(PackedMatrix(mx.array(raw.ravel()), tensor))
        planes.append(tuple(PackedMatrix(mx.array(row.ravel()), replace(tensor, shape=(256,256,1), nbytes=row.nbytes)) for row in raw.reshape(11,256,50)))
    for tokens in (1, 2, 6, 10, 15):
        ids = np.asarray([rng.choice(11, 6, replace=False) for _ in range(tokens)], np.uint32)
        x = mx.array(rng.normal(size=(tokens,256)).astype(np.float32))
        scores = mx.array(rng.uniform(.01,.4,size=ids.shape).astype(np.float32))
        expected = reference(x, mx.array(ids), scores, *matrices)
        actual = _moe(x, ids, scores, *planes, reuse=reuse, broadcast=broadcast, group_experts=group_experts, simd_groups=simd_groups)
        mx.eval(expected, actual)
        if np.array(expected).tobytes() != np.array(actual).tobytes():
            raise GGUFError(f"row-exact IQ1 gate failed: rows={tokens}, reuse={reuse}, broadcast={broadcast}")


def moe(*args, reuse=1, broadcast=False, group_experts=6, simd_groups=1, **kwargs):
    self_check(reuse, broadcast, group_experts, simd_groups)
    return _moe(*args, reuse=reuse, broadcast=broadcast, group_experts=group_experts, simd_groups=simd_groups, **kwargs)
