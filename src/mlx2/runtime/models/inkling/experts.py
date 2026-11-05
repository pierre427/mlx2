# SPDX-License-Identifier: Apache-2.0
"""Resident affine q2/g128 expert groups with private per-row reductions.

Mined from mlx2 qwen36_moe_grouped at c744e90d5; see
provenance/inkling-optimization-20261103.json. The planner is shared across
gate/up and down. No concatenated or dequantized full expert bank is made.

The CH=4 M5 Max actual-weight A/B (95eee09fc, ops-ab-r2) lost against the
batched ordinary path. The exact-tree control at 12974ec80 selected CH=1:
CH=2/4 cost more at 4..32 rows, with identical outputs. Keep this off in the
ordinary profile absent an end-to-end win there. In AST it replaces per-node
exact gathers while preserving each row's reduction. The shared expert plan
and fused gate/up dispatch remain; register hoisting across rows stays off.
"""

from functools import lru_cache

import mlx.core as mx

from .. import qwen4_routed_decode as rd
from .. import qwen36_moe_grouped as grouped

# q2's fast gather assigns one packed word (16 values) to each lane.
# Match the four-value affine fold before accumulating the next block.
HEADER = (
    r"""
using namespace metal;
constant constexpr int BITS=2, GS=128, FAST=1;
constant constexpr int PACK_FACTOR=16, BYTES_PER_PACK=4;
constant constexpr int PACKS_PER_THREAD=1, VALUES_PER_THREAD=16;
constant constexpr int BLOCK_SIZE=512, SCALE_STEP_PER_THREAD=8;
constant constexpr int NWF=16;
template<typename T>
inline float load_vector(const device T* x, thread float* xt) {
    float sum=0;
    for (int i=0;i<16;i+=4) {
        sum+=float(x[i])+float(x[i+1])+float(x[i+2])+float(x[i+3]);
        xt[i]=float(x[i]); xt[i+1]=float(x[i+1])/4.0f;
        xt[i+2]=float(x[i+2])/16.0f; xt[i+3]=float(x[i+3])/64.0f;
    }
    return sum;
}
inline void wconv(const device uint8_t* w, thread float* wf) {
    for(int i=0;i<4;++i) {
        wf[4*i]=float(w[i]&3); wf[4*i+1]=float(w[i]&12);
        wf[4*i+2]=float(w[i]&48); wf[4*i+3]=float(w[i]&192);
    }
}
inline float qdot_f(const thread float* wf,const thread float* x,
                   float scale,float bias,float sum) {
    float value=0;
    for(int i=0;i<16;i+=4)
        value+=x[i]*wf[i]+x[i+1]*wf[i+1]+x[i+2]*wf[i+2]+x[i+3]*wf[i+3];
    return scale*value+sum*bias;
}
"""
    + rd.SIGMOID
)


@lru_cache(maxsize=2)
def _kernel(kind):
    gate = kind == "gate_up"
    return mx.fast.metal_kernel(
        name="inkling_q2_grouped_" + kind,
        input_names=(
            ["x", "wg", "sg", "bg", "wu", "su", "bu", "groups"]
            if gate
            else ["x", "w", "scales", "biases", "groups"]
        ),
        output_names=["y"],
        header=HEADER,
        source=grouped.GATE_UP_HOIST_SOURCE if gate else grouped.DOWN_HOIST_SOURCE,
        ensure_row_contiguous=True,
    )


def supported(x, indices, sm):
    if mx.default_device() != mx.gpu or x.dtype != mx.bfloat16:
        return False
    rows, hidden = x.size // x.shape[-1], x.shape[-1]
    if not 1 <= rows <= 64 or hidden % 512 or indices.size != rows * 6:
        return False
    for layer in (sm.gate_proj, sm.up_proj, sm.down_proj):
        if (
            getattr(layer, "bits", None),
            getattr(layer, "group_size", None),
            getattr(layer, "mode", None),
        ) != (2, 128, "affine"):
            return False
        if layer.scales.dtype != x.dtype or layer.biases.dtype != x.dtype:
            return False
    return sm.gate_proj.weight.shape[1] % 512 == 0 and sm.scales_trivial()


def routed(x, indices, sm, *, chunk=1):
    """Return per-slot values; each member's chain is independent of its peers."""
    if not supported(x, indices, sm):
        raise ValueError("unsupported Inkling grouped q2 shape")
    rows, hidden = x.size // x.shape[-1], x.shape[-1]
    topk, inter = indices.shape[-1], sm.gate_proj.weight.shape[1]
    assignments = rows * topk
    padded = (assignments + 31) // 32 * 32
    plan = grouped._plan_kernel()(
        inputs=[indices.reshape(-1).astype(mx.uint32)],
        template=[("B", assignments), ("BP", padded), ("MAXR", rows)],
        grid=(padded, 1, 1),
        threadgroup=(padded, 1, 1),
        output_shapes=[(assignments * (2 + rows),)],
        output_dtypes=[mx.int32],
    )[0]
    common = [("T", x.dtype), ("NSG", 4), ("MAXR", rows), ("CH", min(chunk, rows))]
    h = _kernel("gate_up")(
        inputs=[
            x.reshape(rows, hidden),
            *rd.expert_operands(sm.gate_proj),
            *rd.expert_operands(sm.up_proj),
            plan,
        ],
        template=[*common, ("K", hidden), ("NI", inter), ("RPS", 2), ("TOPK", topk)],
        grid=(32, inter // 2, assignments),
        threadgroup=(32, 4, 1),
        output_shapes=[(assignments, inter)],
        output_dtypes=[x.dtype],
    )[0]
    out = _kernel("down")(
        inputs=[h, *rd.expert_operands(sm.down_proj), plan],
        template=[*common, ("K", inter), ("N", hidden), ("RPS", 4)],
        grid=(32, hidden // 4, assignments),
        threadgroup=(32, 4, 1),
        output_shapes=[(assignments, hidden)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(rows, topk, hidden)
