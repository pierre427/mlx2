# SPDX-License-Identifier: Apache-2.0
"""Unselected IQ1_S decoder experiments inspired by Splash's prepared decoder.

Modified mechanisms: a generated uint32 grid and block-scale/payload reuse.
The existing MIT-noticed row-exact source supplies routing and all tensor math.
No provider selects this module. See deepseek-v41-iq1-splash-kernels provenance.
"""

from functools import lru_cache

import numpy as np

from . import v41_row_exact as base
from .iq1_codebook import IQ1_GRID
from .reader import GGUFError

REVISION = "mlx2.v41-iq1-splash-decoder-experiment.v1"
VARIANTS = ("nibble", "hoist")


def nibble_grid():
    """Encode each original eight-value ternary grid entry in one uint32."""
    result = []
    for word in IQ1_GRID:
        value = 0
        for lane in range(8):
            byte = (word >> (8*lane)) & 255
            signed = byte if byte < 128 else byte-256
            if signed not in (-1, 0, 1):
                raise GGUFError("IQ1 grid is not ternary")
            value |= (signed+1) << ((lane % 4)*8+(lane//4)*4)
        result.append(value)
    return tuple(result)


NIBBLE_GRID = nibble_grid()
# Reuse the existing scale conversion and BF16 rounding verbatim, without its
# unused large IQ1/IQ2 tables or generic per-format decoder.
_HALF = "inline float gguf_half"+base.BF16_HEADER.split("inline float gguf_half", 1)[1].split("inline uint gguf_u32", 1)[0]
_BF16 = "inline float v41_bf16"+base.BF16_HEADER.split("inline float v41_bf16", 1)[1]
HEADER = ("constant uint v41_iq1_nibbles[2048] = {"+
          ",".join(f"0x{value:08x}u" for value in NIBBLE_GRID)+"};\n"+_HALF+_BF16+r"""
inline float v41_iq1_scaled(float d, uint qh, uint low, uint lane) {
    const uint j=lane/8, index=low|(((qh>>(3*j))&7)<<8);
    const uint shift=(lane%4)*8+((lane%8)/4)*4;
    const int value=int((v41_iq1_nibbles[index]>>shift)&15u)-1;
    const float delta=(qh&0x8000) ? -0.125f : 0.125f;
    return d*float(2*((qh>>12)&7)+1)*(float(value)+delta);
}
inline float v41_iq1_value(device const uchar *row, uint k) {
    device const uchar *p=row+(k/256)*50;
    const uint i=k%256, group=i/32, lane=i%32;
    const uint qh=uint(p[34+2*group])|(uint(p[35+2*group])<<8);
    return v41_iq1_scaled(gguf_half(p),qh,uint(p[2+4*group+lane/8]),lane);
}
""")


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise GGUFError("experimental decoder source anchor changed")
    return source.replace(old, new, 1)


def kernel_spec(kind, count, variant="nibble"):
    if variant not in VARIANTS:
        raise GGUFError("unsupported experimental IQ1 decoder")
    names, source = base.kernel_spec(kind, count, False)
    planes = ("g", "u") if kind == "gate" else ("w",)
    if variant == "nibble":
        for plane in planes:
            source = _replace_once(source, f"gguf_value({plane},k,19)", f"v41_iq1_value({plane},k)")
        return names, source
    prefix = "for(uint block=0;block<K/256;block++) {\n"
    for plane in planes:
        prefix += (f"device const uchar *p{plane}={plane}+size_t(block)*50;\n"
                   f"const float d{plane}=gguf_half(p{plane});\n"
                   f"uint qh{plane}=uint(p{plane}[34])|(uint(p{plane}[35])<<8);\n"
                   f"uint low{plane}=uint(p{plane}[2+lane/8]);\n")
    prefix += "for(uint group=0;group<8;group++) {\nconst uint k=block*256+group*32+lane;\n"
    loaded = ("float wg=gguf_value(g,k,19), wu=gguf_value(u,k,19);\n" if kind == "gate"
              else "float weight=gguf_value(w,k,19);\n")
    values = ("float wg=v41_iq1_scaled(dg,qhg,lowg,lane), wu=v41_iq1_scaled(du,qhu,lowu,lane);\n"
              if kind == "gate" else "float weight=v41_iq1_scaled(dw,qhw,loww,lane);\n")
    values += "if(group+1<8) {\n"
    for plane in planes:
        values += (f"qh{plane}=uint(p{plane}[34+2*(group+1)])|(uint(p{plane}[35+2*(group+1)])<<8);\n"
                   f"low{plane}=uint(p{plane}[2+4*(group+1)+lane/8]);\n")
    values += "}\n"
    source = _replace_once(source, "for(uint k=lane;k<K;k+=32) {\n"+loaded, prefix+values)
    # Close only the added outer block loop; the original inner accumulator
    # body, reduction and epilogue remain byte-for-byte source fragments.
    source = _replace_once(source, "\n}\nfor(uint r=0;r<ROWS;r++) {", "\n}\n}\nfor(uint r=0;r<ROWS;r++) {")
    return names, source


@lru_cache(maxsize=40)
def kernel(kind, count, variant):
    import mlx.core as mx
    names, source = kernel_spec(kind, count, variant)
    return mx.fast.metal_kernel(name=f"mlx2_v41_iq1_splash_{variant}_{kind}_{count}_v1",
        input_names=names, output_names=["y"], header=HEADER, source=source,
        ensure_row_contiguous=True)


def _moe(x, ids, scores, gate, up, down, *, variant="nibble", reuse=None,
         broadcast=False, counters=None, group_experts=10, simd_groups=4):
    """Bounded original row dispatcher with only its gate/down kernel changed."""
    import mlx.core as mx
    if variant not in VARIANTS or broadcast is not False:
        raise GGUFError("experimental IQ1 requires a declared scalar decoder")
    count = len(gate)
    if not count or len(up) != count or len(down) != count:
        raise GGUFError("experimental IQ1 expert planes disagree")
    _, hidden, width = gate[0].shape
    if (any(w.ggml_type != 19 or w.shape != (1, hidden, width) for w in (*gate, *up))
            or any(w.ggml_type != 19 or w.shape != (1, width, hidden) for w in down)
            or x.ndim != 2 or x.shape[-1] != width or scores.shape != ids.shape
            or ids.shape[0] != x.shape[0] or width % 256 or hidden % 256
            or simd_groups not in (1, 4)):
        raise GGUFError("invalid experimental IQ1 geometry")
    reuse = (1 if x.shape[0] == 1 else 4) if reuse is None else reuse
    inverse, groups = base.plan(ids, count, reuse, group_experts)
    projected = []
    for begin, end, order, tiles in groups:
        tile_array = mx.array(tiles)
        mid = kernel("gate", end-begin, variant)(
            inputs=[x.astype(mx.float32), mx.array(order), tile_array, scores.reshape(-1).astype(mx.float32)]
                   +[w.data for w in gate[begin:end]]+[w.data for w in up[begin:end]],
            template=[("TOPK", ids.shape[1]), ("K", width), ("N", hidden), ("ROWS", reuse), ("SIMDS", simd_groups), ("ROW_BYTES", gate[0].row_bytes)],
            grid=(32*simd_groups, hidden//simd_groups, len(tiles)), threadgroup=(32*simd_groups, 1, 1),
            output_shapes=[(len(order), hidden)], output_dtypes=[mx.float32])[0]
        projected.append(kernel("down", end-begin, variant)(
            inputs=[mid, tile_array]+[w.data for w in down[begin:end]],
            template=[("K", hidden), ("N", width), ("ROWS", reuse), ("SIMDS", simd_groups), ("ROW_BYTES", down[0].row_bytes)],
            grid=(32*simd_groups, width//simd_groups, len(tiles)), threadgroup=(32*simd_groups, 1, 1),
            output_shapes=[(len(order), width)], output_dtypes=[mx.float32])[0])
    rows = projected[0] if len(projected) == 1 else mx.concatenate(projected)
    if counters is not None:
        key = "splash_"+variant
        counters[key+"_calls"] = counters.get(key+"_calls", 0)+1
        counters[key+"_groups"] = counters.get(key+"_groups", 0)+len(groups)
    return base.combine_kernel()(inputs=[rows, mx.array(inverse)],
        template=[("N", width), ("TOPK", ids.shape[1])],
        grid=(((width+255)//256)*256, ids.shape[0], 1), threadgroup=(256, 1, 1),
        output_shapes=[x.shape], output_dtypes=[mx.float32])[0]


@lru_cache(maxsize=24)
def self_check(variant="nibble", reuse=1, group_experts=10, simd_groups=4):
    """First-use GPU parity gate; invoked only by an explicit experimental call."""
    import mlx.core as mx
    from .packed_metal import PackedMatrix
    from .reader import GGUFTensor
    if mx.default_device() != mx.gpu:
        raise GGUFError("experimental IQ1 first use requires an owned GPU window")
    rng = np.random.default_rng(41196)
    planes = []
    for kind in ("gate", "up", "down"):
        raw = rng.integers(0, 256, (11, 256, 50), np.uint8)
        raw[:, :, :2] = np.frombuffer(np.float16(.03125).tobytes(), np.uint8)
        tensor = GGUFTensor(kind, (256, 256, 1), 19, 0, 256*50)
        planes.append(tuple(PackedMatrix(mx.array(row.ravel()), tensor) for row in raw))
    for tokens in (1, 2, 6, 10, 15):
        ids = np.asarray([rng.choice(11, 6, replace=False) for _ in range(tokens)], np.uint32)
        x = mx.array(rng.normal(size=(tokens, 256)).astype(np.float32))
        scores = mx.array(rng.uniform(.01, .4, size=ids.shape).astype(np.float32))
        params = dict(reuse=reuse, group_experts=group_experts, simd_groups=simd_groups)
        expected = base._moe(x, ids, scores, *planes, **params)
        actual = _moe(x, ids, scores, *planes, variant=variant, **params)
        mx.eval(expected, actual)
        if np.array(expected).tobytes() != np.array(actual).tobytes():
            raise GGUFError(f"experimental IQ1 parity failed: {variant}, rows={tokens}, reuse={reuse}")


def moe(x, ids, scores, gate, up, down, *, variant="nibble", reuse=None,
        broadcast=False, counters=None, group_experts=10, simd_groups=4):
    if variant not in VARIANTS or broadcast is not False:
        raise GGUFError("experimental IQ1 requires a declared scalar decoder")
    reuse = (1 if x.shape[0] == 1 else 4) if reuse is None else reuse
    self_check(variant, reuse, group_experts, simd_groups)
    return _moe(x, ids, scores, gate, up, down, variant=variant, reuse=reuse,
                broadcast=broadcast, counters=counters, group_experts=group_experts, simd_groups=simd_groups)


def metal_compile_source(kind, count=6, variant="nibble", *, reuse=1, simd_groups=4):
    """Offline compiler input, without creating an MLX or Metal device."""
    names, source = kernel_spec(kind, count, variant)
    bindings = []
    for i, name in enumerate(names):
        dtype = "uint" if name in ("order", "tiles") else "float" if name in ("x", "scores") else "uchar"
        bindings.append(f"device const {dtype}* {name} [[buffer({i})]]")
    bindings.append(f"device float* y [[buffer({len(names)})]]")
    bindings += ["uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]",
                 "uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]",
                 "uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]"]
    constants = f"constant uint K=5120,N=2304,TOPK=6,ROWS={reuse},SIMDS={simd_groups},ROW_BYTES=1000;\n"
    return "#include <metal_stdlib>\nusing namespace metal;\n"+HEADER+constants+"kernel void probe("+",".join(bindings)+") {\n"+source+"\n}\n"
