# SPDX-License-Identifier: Apache-2.0
"""Default-off IQ1_S/384-expert NAX prefill, derived from mlx2 V4 tiles.

FP16 operands form a distinct arithmetic profile. Preserve V4.1 BF16 stores
and scale-before-middle-store semantics. This candidate requires its own GPU
comparison and checkpoint evidence; no V4 performance result transfers here.
"""

from functools import lru_cache

from . import segmented_prefill as base
from .nax_metal import NAX_HEADER
from .packed_metal import HEADER
from .reader import GGUFError
from .v41_metal import BF16_HEADER

SCAN_SOURCE = base.SCAN_SOURCE.replace("256", "512").replace("255", "511")
GATE_SOURCE = (
    base.GATE_UP_SOURCE.replace(
        "min(sums[index],10.0f)", "min(v41_bf16(sums[index]),10.0f)"
    )
    .replace(
        "clamp(up_sums[index],-10.0f,10.0f)",
        "clamp(v41_bf16(up_sums[index]),-10.0f,10.0f)",
    )
    .replace(
        "(gate/(1.0f+metal::precise::exp(-gate)))*u;",
        "v41_bf16((gate/(1.0f+metal::precise::exp(-gate)))*u*scores[order[rowbase+row]]);",
    )
)
DOWN_SOURCE = base.DOWN_SOURCE.replace(
    "=sums[f*8+r*4+j];", "=v41_bf16(sums[f*8+r*4+j]);"
)
COMBINE_SOURCE = base.COMBINE_SOURCE.replace("scores[original]*rows[", "rows[")
HEAD = NAX_HEADER + BF16_HEADER[len(HEADER) :]


@lru_cache(maxsize=4)
def _kernel(kind):
    import mlx.core as mx

    spec = {
        "scan": (
            ["sorted_ids", "original_ids"],
            ["tiles", "tile_count", "status"],
            SCAN_SOURCE,
        ),
        "gate": (
            ["x", "order", "tiles", "tile_count", "w", "up", "scores"],
            ["y"],
            GATE_SOURCE,
        ),
        "down": (["x", "tiles", "tile_count", "w"], ["y"], DOWN_SOURCE),
        "combine": (["rows", "inverse", "scores", "status"], ["out"], COMBINE_SOURCE),
    }
    names, outputs, source = spec[kind]
    return mx.fast.metal_kernel(
        name="mlx2_v41_iq1_nax_" + kind + "_v1",
        input_names=names,
        output_names=outputs,
        source=source,
        header=HEAD,
        ensure_row_contiguous=True,
    )


def plan_capacity(rows, experts):
    if (
        type(rows) is not int
        or type(experts) is not int
        or not 1 <= experts <= 512
        or rows < 1
        or rows * experts >= 2**32
    ):
        raise GGUFError(
            "V4.1 segmented plan requires bounded positive rows and <=512 experts"
        )
    return min(rows, (rows + 15) // 16 + experts - 1)


def scratch_bytes(tokens, topk, hidden, width, experts):
    rows = tokens * topk
    return (
        rows * (hidden + width) * 4
        + rows * 6 * 4
        + plan_capacity(rows, experts) * 16
        + 8
    )


def moe(x, ids, scores, gate, up, down, *, max_scratch_bytes=512 << 20):
    import mlx.core as mx

    if (
        mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or "Apple M5" not in mx.device_info().get("device_name", "")
    ):
        raise GGUFError("V4.1 NAX prefill requires an owned Apple M5 GPU window")
    if (
        len(gate.shape) != 3
        or gate.shape != up.shape
        or any(w.ggml_type != 19 for w in (gate, up, down))
    ):
        raise GGUFError("V4.1 NAX requires matching IQ1_S expert weights")
    experts, hidden, width = gate.shape
    if (
        x.ndim != 2
        or x.shape[-1] != width
        or down.shape != (experts, width, hidden)
        or ids.ndim != 2
        or ids.dtype != mx.uint32
        or ids.shape != scores.shape
        or ids.shape[0] != x.shape[0]
        or not 1 <= ids.shape[1] <= experts
        or hidden % 256
        or width % 256
    ):
        raise GGUFError("V4.1 segmented input/expert axes mismatch")
    tokens, topk = ids.shape
    rows = ids.size
    capacity = plan_capacity(rows, experts)
    if scratch_bytes(tokens, topk, hidden, width, experts) > max_scratch_bytes:
        raise GGUFError("V4.1 segmented scratch exceeds explicit cap")
    flat = ids.reshape(-1)
    order = mx.argsort(
        flat * mx.array(rows, mx.uint32) + mx.arange(rows, dtype=mx.uint32)
    ).astype(mx.uint32)
    inverse = mx.argsort(order).astype(mx.uint32)
    tiles, count, status = _kernel("scan")(
        inputs=[flat[order], flat],
        template=[("ROWS", rows), ("EXPERTS", experts), ("MAX_TILES", capacity)],
        grid=(512, 1, 1),
        threadgroup=(512, 1, 1),
        output_shapes=[(capacity, 4), (1,), (1,)],
        output_dtypes=[mx.uint32] * 3,
    )
    mid = _kernel("gate")(
        inputs=[
            x.astype(mx.float32),
            order,
            tiles,
            count,
            gate.data,
            up.data,
            scores.reshape(-1).astype(mx.float32),
        ],
        template=[
            ("TOPK", topk),
            ("K", width),
            ("N", hidden),
            ("ROW_BYTES", gate.row_bytes),
            ("FORMAT", 19),
        ],
        grid=(((hidden + 127) // 128) * 128, capacity, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, hidden)],
        output_dtypes=[mx.float32],
    )[0]
    projected = _kernel("down")(
        inputs=[mid, tiles, count, down.data],
        template=[
            ("K", hidden),
            ("N", width),
            ("ROW_BYTES", down.row_bytes),
            ("FORMAT", 19),
        ],
        grid=(((width + 127) // 128) * 128, capacity, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, width)],
        output_dtypes=[mx.float32],
    )[0]
    return _kernel("combine")(
        inputs=[projected, inverse, scores.reshape(-1).astype(mx.float32), status],
        template=[("TOKENS", tokens), ("TOPK", topk), ("N", width)],
        grid=(((width + 255) // 256) * 256, tokens, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(tokens, width)],
        output_dtypes=[mx.float32],
    )[0]


def metal_compile_source(kind):
    """Adapt V4's standalone offline compiler wrapper to the V4.1 sources."""
    old_kind = "gate_up" if kind == "gate" else kind
    source = base.metal_compile_source(old_kind)
    source = (
        source.replace(NAX_HEADER, HEAD)
        .replace("EXPERTS=6", "EXPERTS=384")
        .replace("FORMAT=16,ROW_BYTES=66", "FORMAT=19,ROW_BYTES=50")
    )
    old = {
        "scan": base.SCAN_SOURCE,
        "gate": base.GATE_UP_SOURCE,
        "down": base.DOWN_SOURCE,
        "combine": base.COMBINE_SOURCE,
    }[kind]
    new = {
        "scan": SCAN_SOURCE,
        "gate": GATE_SOURCE,
        "down": DOWN_SOURCE,
        "combine": COMBINE_SOURCE,
    }[kind]
    source = source.replace(old, new)
    if kind == "gate":
        source = source.replace(
            "device float* y [[buffer(6)]],",
            "device const float* scores [[buffer(6)]],device float* y [[buffer(7)]],",
        )
    return source
