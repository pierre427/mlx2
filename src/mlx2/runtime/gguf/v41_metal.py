# SPDX-License-Identifier: MIT
"""V4.1 IQ1_S experts with explicit BF16 stores and fused gate/up/SwiGLU.

This is a separate numerical law from V4's FP32 experts. Routing scales are
applied before the middle BF16 store, as in the official V4.1 Expert.forward.
No GPU probe is run at import. See provenance/deepseek-v41-adapter.json.
"""

from functools import lru_cache

from .packed_metal import DOWN, GATE_UP, HEADER
from .reader import GGUFError

BF16_HEADER = (
    HEADER
    + r"""
inline float v41_bf16(float value) {
    uint bits=as_type<uint>(value);
    return as_type<float>((bits+0x7fffu+((bits>>16)&1u))&0xffff0000u);
}
"""
)
GATE_SOURCE = GATE_UP.replace(
    "    sg=min(sg,10.0f);",
    "    sg=v41_bf16(sg); su=v41_bf16(su);\n    sg=min(sg,10.0f);",
).replace("(sg/(1.0f+exp(-sg)))*su;", "v41_bf16((sg/(1.0f+exp(-sg)))*su*scores[slot]);")
DOWN_SOURCE = DOWN.replace("total+=scores[index]*sum;", "total+=v41_bf16(sum);")


@lru_cache(maxsize=2)
def kernel(kind):
    import mlx.core as mx

    if kind == "gate":
        names, out, source = ["x", "ids", "gate", "up", "scores"], ["mid"], GATE_SOURCE
    else:
        names, out, source = ["mid", "ids", "scores", "w"], ["y"], DOWN_SOURCE
    return mx.fast.metal_kernel(
        name="mlx2_v41_iq1_" + kind + "_v1",
        input_names=names,
        output_names=out,
        header=BF16_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


def moe(x, ids, scores, gate, up, down):
    import mlx.core as mx

    if (
        gate.ggml_type != 19
        or up.ggml_type != 19
        or down.ggml_type != 19
        or len(gate.shape) != 3
        or up.shape != gate.shape
    ):
        raise GGUFError("V4.1 experts require matching IQ1_S gate/up/down")
    experts, hidden, width = gate.shape
    if (
        down.shape != (experts, width, hidden)
        or x.ndim != 2
        or x.shape[-1] != width
        or ids.ndim != 2
        or ids.dtype != mx.uint32
        or scores.shape != ids.shape
        or ids.shape[0] != x.shape[0]
        or not 1 <= ids.shape[1] <= experts
    ):
        raise GGUFError("invalid V4.1 packed expert axes")
    tokens, topk = ids.shape
    common = [
        ("K", width),
        ("M", hidden),
        ("N", width),
        ("TOPK", topk),
        ("EXPERTS", experts),
    ]
    mid = kernel("gate")(
        inputs=[
            x.astype(mx.float32),
            ids,
            gate.data,
            up.data,
            scores.astype(mx.float32),
        ],
        template=common + [("ROW_BYTES", gate.row_bytes), ("FORMAT", 19)],
        grid=(32, hidden, tokens * topk),
        threadgroup=(32, 1, 1),
        output_shapes=[(tokens * topk, hidden)],
        output_dtypes=[mx.float32],
    )[0]
    return kernel("down")(
        inputs=[mid, ids, scores.astype(mx.float32), down.data],
        template=common + [("ROW_BYTES", down.row_bytes), ("FORMAT", 19)],
        grid=(32, width, tokens),
        threadgroup=(32, 1, 1),
        output_shapes=[(tokens, width)],
        output_dtypes=[mx.float32],
    )[0]


def row_buffer_spec(kind, count):
    """Bind native row buffers directly; preserve the reference reduction order."""
    if kind not in ("gate", "down") or type(count) is not int or not 1 <= count <= 6:
        raise GGUFError("row-buffer IQ1 decode requires gate/down and one to six experts")

    def choose(name):
        return "(" + "".join(f"expert=={i}?{name}{i}:" for i in range(count-1)) + f"{name}{count-1})"

    if kind == "gate":
        names = ["x", "ids", "scores"]+[f"gate{i}" for i in range(count)]+[f"up{i}" for i in range(count)]
        source = GATE_SOURCE.replace("gate+(size_t(expert)*M+row)*ROW_BYTES", choose("gate")+"+size_t(row)*ROW_BYTES")
        source = source.replace("up+(size_t(expert)*M+row)*ROW_BYTES", choose("up")+"+size_t(row)*ROW_BYTES")
        return names, ["mid"], source
    names = ["mid", "ids", "scores"]+[f"w{i}" for i in range(count)]
    return names, ["y"], DOWN_SOURCE.replace("w+(size_t(expert)*N+row)*ROW_BYTES", choose("w")+"+size_t(row)*ROW_BYTES")


@lru_cache(maxsize=12)
def row_buffer_kernel(kind, count):
    import mlx.core as mx

    names, outputs, source = row_buffer_spec(kind, count)
    return mx.fast.metal_kernel(name=f"mlx2_v41_iq1_rows_{kind}_{count}_v1", input_names=names,
                               output_names=outputs, header=BF16_HEADER, source=source,
                               ensure_row_contiguous=True)


def moe_rows(x, ids, scores, gate, up, down):
    """B1 IQ1 decode with <=6 separate cached expert rows, without weight copies."""
    import mlx.core as mx

    count = len(gate)
    if not 1 <= count <= 6 or len(up) != count or len(down) != count:
        raise GGUFError("row-buffer expert planes disagree")
    _, hidden, width = gate[0].shape
    if (any(w.ggml_type != 19 or w.shape != (1, hidden, width) for w in (*gate, *up))
            or any(w.ggml_type != 19 or w.shape != (1, width, hidden) for w in down)
            or x.ndim != 2 or x.shape != (1, width) or ids.ndim != 2 or ids.shape[0] != 1
            or ids.dtype != mx.uint32 or not 1 <= ids.shape[1] <= count or scores.shape != ids.shape):
        raise GGUFError("invalid B1 row-buffer expert geometry")
    topk = ids.shape[1]
    common = [("K", width), ("M", hidden), ("N", width), ("TOPK", topk), ("EXPERTS", count)]
    mid = row_buffer_kernel("gate", count)(
        inputs=[x.astype(mx.float32), ids, scores.astype(mx.float32)]+[w.data for w in (*gate, *up)],
        template=common+[("ROW_BYTES", gate[0].row_bytes), ("FORMAT", 19)],
        grid=(32, hidden, topk), threadgroup=(32, 1, 1),
        output_shapes=[(topk, hidden)], output_dtypes=[mx.float32])[0]
    return row_buffer_kernel("down", count)(
        inputs=[mid, ids, scores.astype(mx.float32)]+[w.data for w in down],
        template=common+[("ROW_BYTES", down[0].row_bytes), ("FORMAT", 19)],
        grid=(32, width, 1), threadgroup=(32, 1, 1),
        output_shapes=[(1, width)], output_dtypes=[mx.float32])[0]


def metal_compile_source(kind):
    """Standalone offline compiler input; never initializes a device."""
    source = GATE_SOURCE if kind == "gate" else DOWN_SOURCE
    if kind == "gate":
        bindings = "device const float* x [[buffer(0)]],device const uint* ids [[buffer(1)]],device const uchar* gate [[buffer(2)]],device const uchar* up [[buffer(3)]],device const float* scores [[buffer(4)]],device float* mid [[buffer(5)]],"
    else:
        bindings = "device const float* mid [[buffer(0)]],device const uint* ids [[buffer(1)]],device const float* scores [[buffer(2)]],device const uchar* w [[buffer(3)]],device float* y [[buffer(4)]],"
    attrs = "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]"
    constants = (
        "constant uint K=256,M=256,N=256,TOPK=6,EXPERTS=384,ROW_BYTES=50,FORMAT=19;\n"
    )
    return (
        "#include <metal_stdlib>\nusing namespace metal;\n"
        + BF16_HEADER
        + constants
        + "kernel void v41_"
        + kind
        + "("
        + bindings
        + attrs
        + ") {\n"
        + source
        + "\n}\n"
    )


_READY = False
_ROWS_READY = False


def self_check_rows():
    """Bounded bitwise binding gate for the explicitly selected row route."""
    global _ROWS_READY
    if _ROWS_READY:
        return
    import mlx.core as mx
    import numpy as np
    from dataclasses import replace
    from .packed_metal import PackedMatrix
    from .reader import GGUFTensor

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise GGUFError("row-buffer first-use gate requires an owned Metal GPU window")
    rng = np.random.default_rng(41017)
    for count in (1, 2, 6):
        planes, rows = [], []
        for kind in ("gate", "up", "down"):
            raw = rng.integers(0, 256, (count*256, 50), np.uint8)
            raw[:, :2] = np.frombuffer(np.float16(0.0625).tobytes(), np.uint8)
            tensor = GGUFTensor(kind, (256, 256, count), 19, 0, raw.nbytes)
            matrix = PackedMatrix(mx.array(raw.reshape(-1)), tensor)
            planes.append(matrix)
            row_tensor = replace(tensor, shape=(256, 256, 1), nbytes=raw.nbytes//count)
            rows.append(tuple(PackedMatrix(matrix.data[i*row_tensor.nbytes:(i+1)*row_tensor.nbytes], row_tensor)
                              for i in range(count)))
        x = mx.full((1, 256), 0.0625)
        ids = mx.array([list(reversed(range(count)))], mx.uint32)
        scores = mx.array([[0.5+i/4 for i in range(count)]])
        reference, actual = moe(x, ids, scores, *planes), moe_rows(x, ids, scores, *rows)
        mx.eval(reference, actual)
        if not np.array(reference).tobytes() == np.array(actual).tobytes():
            raise GGUFError("row-buffer IQ1 binding first-use gate failed")
    _ROWS_READY = True


def self_check():
    """Bounded first-use numerical gate; never checkpoint qualification.

    Called only inside a future owned GPU load. Integer/dyadic operands make
    the dot products exact, isolating IQ1 bytes, BF16 stores and routing scale
    placement from large-reduction numerical drift.
    """
    global _READY
    if _READY:
        return
    import mlx.core as mx
    import numpy as np

    from .blocks import decode_blocks
    from .packed_metal import PackedMatrix
    from .reader import GGUFTensor

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise GGUFError("V4.1 packed self-check requires an owned Metal GPU window")
    rng = np.random.default_rng(4141)
    packed = []
    decoded = []
    for kind in ("gate", "up", "down"):
        raw = rng.integers(0, 256, (2 * 256, 50), dtype=np.uint8)
        raw[:, :2] = np.frombuffer(np.float16(0.0625).tobytes(), np.uint8)
        raw = raw.reshape(-1)
        tensor = GGUFTensor(kind, (256, 256, 2), 19, 0, raw.nbytes)
        packed.append(PackedMatrix(mx.array(raw), tensor))
        decoded.append(
            decode_blocks(raw.tobytes(), 19, max_elements=2 * 256 * 256).reshape(
                2, 256, 256
            )
        )

    def bf16(x):
        bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
        return (
            (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
        ).view(np.float32)

    x = (np.arange(256, dtype=np.float32) % 5 - 2)[None]
    ids = np.array([[0, 1]], np.uint32)
    scores = np.array([[0.5, 1.0]], np.float32)
    expected = np.zeros((1, 256), np.float32)
    for slot in range(2):
        gate = bf16(x @ decoded[0][slot].T)
        up = bf16(x @ decoded[1][slot].T)
        gate = np.minimum(gate, 10)
        up = np.clip(up, -10, 10)
        mid = bf16((gate / (1 + np.exp(-gate))) * up * scores[0, slot])
        expected += bf16(mid @ decoded[2][slot].T)
    actual = moe(mx.array(x), mx.array(ids), mx.array(scores), *packed)
    mx.eval(actual)
    if not np.array_equal(np.array(actual), expected):
        raise GGUFError("V4.1 IQ1_S/BF16 first-use reference gate failed")
    _READY = True
