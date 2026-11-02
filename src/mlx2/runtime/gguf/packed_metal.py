"""Native packed GGML Metal reference kernels, not affine MLX QMM.

Weights remain packed on device. Only products/selected-expert activations are
materialized. This first implementation is deliberately selectable, unqualified
and not a performance claim; tiled prefill and cooperative block reuse need
measured follow-up. No external engine is invoked.
"""

from __future__ import annotations

import math
from functools import lru_cache

from .iq1_codebook import IQ1_GRID
from .iq2_codebook import IQ2_GRID
from .k_blocks import SIZES as K_FORMATS
from .k_metal import HEADER as K_HEADER
from .reader import FORMATS, GGUFError, GGUFTensor

# Original Metal transcription of the documented block semantics. Codebook is
# mined under MIT; provenance/deepseek-v4-gguf-packed.json and provenance/deepseek-v4-gguf-packed.NOTICE.
HEADER = (
    K_HEADER + "constant ulong iq2_grid[256] = {"
    + ",".join(f"0x{x:016x}ul" for x in IQ2_GRID)
    + "};\n"
    + "constant ulong iq1_grid[2048] = {" + ",".join(f"0x{x:016x}ul" for x in IQ1_GRID) + "};\n"
    + r"""
inline float gguf_half(device const uchar *p) {
    const ushort bits = ushort(p[0]) | (ushort(p[1]) << 8);
    return float(as_type<half>(bits));
}
inline uint gguf_u32(device const uchar *p) {
    return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24);
}
inline float gguf_value(device const uchar *row, uint k, uint format) {
    if(format==2 || (format>=11 && format<=14)) return gguf_k_value(row,k,format);
    if (format == 8) {
        device const uchar *p = row + (k / 32) * 34;
        return gguf_half(p) * float(as_type<char>(p[2 + k % 32]));
    }
    if (format == 10) {
        device const uchar *p = row + (k / 256) * 84;
        uint i=k%256, group=i/16, l=i%16;
        uint q=(uint(p[16+32*(group/8)+16*(group&1)+l]) >> (((group/2)&3)*2))&3;
        uint sc=p[group];
        return gguf_half(p+80)*float(sc&15)*float(q)-gguf_half(p+82)*float(sc>>4);
    }
    if (format == 19) {
        device const uchar *p = row + (k / 256) * 50;
        uint i=k%256, group=i/32, j=(i%32)/8, lane=i%8;
        uint qh=uint(p[34+2*group]) | (uint(p[35+2*group])<<8);
        uint index=uint(p[2+4*group+j]) | (((qh>>(3*j))&7)<<8);
        char value=as_type<char>(uchar((iq1_grid[index]>>(lane*8))&255));
        float delta=(qh&0x8000) ? -0.125f : 0.125f;
        return gguf_half(p)*float(2*((qh>>12)&7)+1)*(float(value)+delta);
    }
    device const uchar *p = row + (k / 256) * 66;
    uint i=k%256, group=i/32, j=(i%32)/8, lane=i%8;
    uint indexes=gguf_u32(p+2+group*8), aux=gguf_u32(p+6+group*8);
    uint seven=(aux>>(7*j))&127;
    uint signs=seven|((popcount(seven)&1)<<7);
    ulong grid=iq2_grid[(indexes>>(8*j))&255];
    float dl=gguf_half(p)*(0.5f+float(aux>>28))*0.25f;
    return dl*float((grid>>(8*lane))&255)*((signs&(1<<lane)) ? -1.0f : 1.0f);
}
"""
)

DENSE = r"""
const uint lane=thread_index_in_simdgroup;
const uint row=threadgroup_position_in_grid.y;
const uint batch_group=threadgroup_position_in_grid.z;
const uint group=batch_group%GROUPS;
const uint token=batch_group;
device const uchar *wr=w+(size_t(group)*N+row)*ROW_BYTES;
float sum=0.0f;
for(uint k=lane;k<K;k+=32) sum += float(x[size_t(token)*K+k])*gguf_value(wr,k,FORMAT);
sum=simd_sum(sum);
if(lane==0) y[size_t(token)*N+row]=sum;
"""

GATE_UP = r"""
const uint lane=thread_index_in_simdgroup;
const uint row=threadgroup_position_in_grid.y;
const uint slot=threadgroup_position_in_grid.z;
const uint expert=ids[slot];
if(expert>=EXPERTS) { if(lane==0) mid[size_t(slot)*M+row]=NAN; return; }
const uint token=slot/TOPK;
device const uchar *g=gate+(size_t(expert)*M+row)*ROW_BYTES;
device const uchar *u=up+(size_t(expert)*M+row)*ROW_BYTES;
float sg=0.0f,su=0.0f;
for(uint k=lane;k<K;k+=32) {
    float a=float(x[size_t(token)*K+k]);
    sg+=a*gguf_value(g,k,FORMAT); su+=a*gguf_value(u,k,FORMAT);
}
sg=simd_sum(sg); su=simd_sum(su);
if(lane==0) {
    // DeepSeek V4's clip is asymmetric for the gate, symmetric for up.
    sg=min(sg,10.0f); su=clamp(su,-10.0f,10.0f);
    mid[size_t(slot)*M+row]=(sg/(1.0f+exp(-sg)))*su;
}
"""

DOWN = r"""
const uint lane=thread_index_in_simdgroup;
const uint row=threadgroup_position_in_grid.y;
const uint token=threadgroup_position_in_grid.z;
float total=0.0f;
for(uint slot=0;slot<TOPK;slot++) {
    uint index=token*TOPK+slot, expert=ids[index];
    if(expert>=EXPERTS) { if(lane==0) y[size_t(token)*N+row]=NAN; return; }
    device const uchar *wr=w+(size_t(expert)*N+row)*ROW_BYTES;
    float sum=0.0f;
    for(uint k=lane;k<M;k+=32) sum+=mid[size_t(index)*M+k]*gguf_value(wr,k,FORMAT);
    sum=simd_sum(sum);
    total+=scores[index]*sum;
}
if(lane==0) y[size_t(token)*N+row]=total;
"""


@lru_cache(maxsize=3)
def _kernel(kind):
    import mlx.core as mx

    names = {
        "dense": (["x", "w"], ["y"], DENSE),
        "gate_up": (["x", "ids", "gate", "up"], ["mid"], GATE_UP),
        "down": (["mid", "ids", "scores", "w"], ["y"], DOWN),
    }
    ins, outs, source = names[kind]
    return mx.fast.metal_kernel(
        name=f"mlx2_gguf_packed_{kind}_v1",
        input_names=ins,
        output_names=outs,
        header=HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


def row_bytes(ggml_type: int, width: int) -> int:
    if ggml_type not in (8, 10, 16, 19) and ggml_type not in K_FORMATS:
        raise GGUFError("unsupported packed Metal quantization format")
    _, block, size = FORMATS[ggml_type]
    if width < 1 or width % block:
        raise GGUFError("packed matrix row is not block-aligned")
    return width // block * size


class PackedMatrix:
    """An explicit packed tensor with GGUF-axis preserving shape.

    Creation is not model admission. Upload must occur inside the runtime's
    owned GPU window. Grouped tensors are not implicitly flattened to dense.
    """

    def __init__(self, data, tensor: GGUFTensor):
        import mlx.core as mx

        self.tensor = tensor
        self.shape = tensor.numpy_shape
        self.ggml_type = tensor.ggml_type
        self.row_bytes = row_bytes(tensor.ggml_type, tensor.shape[0])
        if data.dtype != mx.uint8 or data.size != tensor.nbytes or data.ndim != 1:
            raise GGUFError(
                "packed device buffer must be flat uint8 with exact tensor extent"
            )
        self.data = data

    @classmethod
    def upload(cls, file, name):
        import mlx.core as mx
        import numpy as np

        tensor = file.tensors[name]
        # This is a packed copy, never an expanded float/affine conversion.
        view = file.packed_view(name)
        try:
            data = mx.array(np.frombuffer(view, dtype=np.uint8))
            mx.eval(data)
        finally:
            view.release()
            file.release_mapping()
        return cls(data, tensor)

    def __call__(self, x):
        import mlx.core as mx

        if len(self.shape) != 2:
            raise GGUFError(
                "dense call requires rank2; grouped/expert weights need explicit axes"
            )
        n, k = self.shape
        if x.ndim < 1 or x.shape[-1] != k:
            raise GGUFError("packed dense input dimension mismatch")
        leading = x.shape[:-1]
        b = math.prod(leading)
        template = [
            ("K", k),
            ("N", n),
            ("ROW_BYTES", self.row_bytes),
            ("FORMAT", self.ggml_type),
            ("GROUPS", 1),
        ]
        out = _kernel("dense")(
            inputs=[x.reshape(b, k).astype(mx.float32), self.data],
            template=template,
            grid=(32, n, b),
            threadgroup=(32, 1, 1),
            output_shapes=[(b, n)],
            output_dtypes=[mx.float32],
        )[0]
        return out.reshape(*leading, n)


def packed_moe(
    x, ids, scores, gate: PackedMatrix, up: PackedMatrix, down: PackedMatrix
):
    """Fused clipped gate/up + routed down/combine. IDs must be caller-validated.

    Caller owns routing law and ensures IDs are within [0, experts); no CPU
    synchronization of routing tensors is introduced here. Only fp32 output is
    returned; adapter decides its residual arithmetic and rounding contract.
    """
    import mlx.core as mx

    if len(gate.shape) != 3 or gate.shape != up.shape or gate.ggml_type != up.ggml_type:
        raise GGUFError("matching expert gate/up rank3 required")
    experts, m, k = gate.shape
    if down.shape != (experts, k, m) or x.ndim != 2 or x.shape[-1] != k:
        raise GGUFError("expert/down/input axes do not match")
    if (
        ids.ndim != 2
        or scores.shape != ids.shape
        or ids.shape[0] != x.shape[0]
        or ids.dtype != mx.uint32
    ):
        raise GGUFError(
            "routing IDs/scores must have matching [tokens,topk] shape and uint32 IDs"
        )
    b, topk = ids.shape
    if not 1 <= topk <= experts:
        raise GGUFError("invalid topk")
    common = [("K", k), ("M", m), ("N", k), ("TOPK", topk), ("EXPERTS", experts)]
    mid = _kernel("gate_up")(
        inputs=[x.astype(mx.float32), ids, gate.data, up.data],
        template=common + [("ROW_BYTES", gate.row_bytes), ("FORMAT", gate.ggml_type)],
        grid=(32, m, b * topk),
        threadgroup=(32, 1, 1),
        output_shapes=[(b * topk, m)],
        output_dtypes=[mx.float32],
    )[0]
    return _kernel("down")(
        inputs=[mid, ids, scores.astype(mx.float32), down.data],
        template=common + [("ROW_BYTES", down.row_bytes), ("FORMAT", down.ggml_type)],
        grid=(32, k, b),
        threadgroup=(32, 1, 1),
        output_shapes=[(b, k)],
        output_dtypes=[mx.float32],
    )[0]


def grouped_linear(x, weight: PackedMatrix):
    """Explicit [groups,out,in] projection, preserving grouped GGUF axes."""
    import mlx.core as mx

    if len(weight.shape) != 3 or x.ndim < 2:
        raise GGUFError(
            "grouped projection needs rank3 weights and [...,groups,in] input"
        )
    groups, n, k = weight.shape
    if x.shape[-2:] != (groups, k):
        raise GGUFError("grouped projection input axes mismatch")
    leading = x.shape[:-2]
    b = math.prod(leading)
    template = [
        ("K", k),
        ("N", n),
        ("ROW_BYTES", weight.row_bytes),
        ("FORMAT", weight.ggml_type),
        ("GROUPS", groups),
    ]
    y = _kernel("dense")(
        inputs=[x.reshape(b * groups, k).astype(mx.float32), weight.data],
        template=template,
        grid=(32, n, b * groups),
        threadgroup=(32, 1, 1),
        output_shapes=[(b * groups, n)],
        output_dtypes=[mx.float32],
    )[0]
    return y.reshape(*leading, groups, n)


@lru_cache(maxsize=1)
def _embedding_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="mlx2_gguf_packed_embedding_v1",
        input_names=["ids", "w"],
        output_names=["out"],
        header=HEADER,
        ensure_row_contiguous=True,
        source=r"""
        uint k=thread_position_in_grid.x;
        uint token=thread_position_in_grid.y;
        if(k>=K) return;
        uint row=ids[token];
        out[size_t(token)*K+k]=row<VOCAB ? gguf_value(w+size_t(row)*ROW_BYTES,k,FORMAT) : NAN;
    """,
    )


def embedding(ids, weight: PackedMatrix):
    """Packed token gather; no vocabulary-sized dequantized embedding table."""
    import mlx.core as mx

    if len(weight.shape) != 2 or ids.dtype != mx.uint32:
        raise GGUFError("embedding requires rank2 packed weights and uint32 IDs")
    vocab, k = weight.shape
    template = [
        ("K", k),
        ("VOCAB", vocab),
        ("ROW_BYTES", weight.row_bytes),
        ("FORMAT", weight.ggml_type),
    ]
    out = _embedding_kernel()(
        inputs=[ids.reshape(-1), weight.data],
        template=template,
        grid=(k, ids.size, 1),
        threadgroup=(min(k, 256), 1, 1),
        output_shapes=[(ids.size, k)],
        output_dtypes=[mx.float32],
    )[0]
    return out.reshape(*ids.shape, k)
