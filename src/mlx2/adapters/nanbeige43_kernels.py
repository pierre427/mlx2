"""Default-off Nanbeige Q8 and functional graph candidates.

Local source mining is recorded in provenance/nanbeige43-kernels.json.
A compiled functional RoPE/append graph is not a claim of one Metal launch.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from functools import lru_cache, partial
from pathlib import Path

import mlx.core as mx
from mlx import nn

from ..runtime.models import prefill_metal
from ..runtime.models.activations import swiglu
from ..runtime.models.served_exp import ServedExpGate

_DEFAULTS = {
    "q8_swiglu": False,
    "q8_four_row": False,
    "rope_kv_fused": False,
    "compiled_blocks": False,
    "rope_kv_append": False,
    "projection_min_rows": 0,
    "projection_max_rows": 128,
    "projection_split_k": None,
    "q8_tile_rows": 32,
    "q8_swiglu_min_rows": 64,
}


def validate_kernel_policy(value=None):
    if value is not None and not isinstance(value, dict):
        raise TypeError("Nanbeige kernel policy must be an object")
    value = dict(value or {})
    if set(value) - _DEFAULTS.keys():
        raise ValueError(
            f"unknown Nanbeige kernel policy: {sorted(set(value) - _DEFAULTS.keys())}"
        )
    result = {**_DEFAULTS, **value}
    for key in (
        "q8_swiglu",
        "compiled_blocks",
        "rope_kv_append",
        "q8_four_row",
        "rope_kv_fused",
    ):
        if type(result[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key in ("projection_min_rows", "projection_max_rows", "q8_swiglu_min_rows"):
        if type(result[key]) is not int:
            raise ValueError(f"{key} must be integer")
    if not 0 <= result["projection_min_rows"] <= result["projection_max_rows"] <= 128:
        raise ValueError("projection rows must satisfy 0 <= min <= max <= 128")
    if result["projection_max_rows"] == 0 or result["q8_swiglu_min_rows"] < 1:
        raise ValueError("kernel row limits must be positive")
    if type(result["q8_tile_rows"]) is not int or result["q8_tile_rows"] not in (
        16,
        32,
    ):
        raise ValueError("q8_tile_rows must be 16 or 32")
    if result["projection_split_k"] is not None and (
        type(result["projection_split_k"]) is not int
        or result["projection_split_k"] not in (1, 2, 4, 8)
    ):
        raise ValueError("projection_split_k must be null,1,2,4,8")
    return result


def kernel_contract(policy):
    result = validate_kernel_policy(policy)
    return {
        "schema": "mlx2.nanbeige43-kernels.v1",
        "policy": result,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prefill_generator_sha256": hashlib.sha256(
            Path(prefill_metal.__file__).read_bytes()
        ).hexdigest(),
        "served_exp_sha256": hashlib.sha256(
            Path(__file__)
            .resolve()
            .parents[1]
            .joinpath("runtime/models/served_exp.py")
            .read_bytes()
        ).hexdigest(),
        "lane_matmul_sha256": hashlib.sha256(
            Path(__file__)
            .resolve()
            .parents[1]
            .joinpath("runtime/lane/matmul.py")
            .read_bytes()
        ).hexdigest(),
        "physical_kernel_source_sha256": hashlib.sha256(
            Path(__file__).with_name("nanbeige43_physical.py").read_bytes()
        ).hexdigest(),
        "qualified": False,
        "physical_rope_append_fusion": "one functional custom invocation per admitted lane; contiguous preparation and full state copying remain extra costs"
        if result["rope_kv_fused"]
        else "unselected",
    }


def quantized_q8(module):
    return (
        isinstance(module, nn.QuantizedLinear)
        and module.bits == 8
        and module.group_size == 64
        and module.mode == "affine"
        and "bias" not in module
    )


def project(module, x, policy, counts):
    """Explicit narrow-row candidate; ordinary module remains the reference."""
    low, high = policy["projection_min_rows"], policy["projection_max_rows"]
    rows = x.size // x.shape[-1]
    if policy["q8_four_row"] and rows == 4:
        from .nanbeige43_physical import four_row_eligible, q8_four_row

        if four_row_eligible(x, module):
            counts["four_row_calls"] = counts.get("four_row_calls", 0) + 1
            return q8_four_row(x, module)
        counts["four_row_refusals"] = counts.get("four_row_refusals", 0) + 1
    if (
        low
        and low <= rows <= high
        and quantized_q8(module)
        and x.dtype in (mx.bfloat16, mx.float16)
    ):
        from ..runtime.lane.matmul import available, lane_matmul, prepare

        if mx.default_device() == mx.gpu and available():
            identity = tuple(id(module[key]) for key in ("weight", "scales", "biases"))
            cached = module.__dict__.get("_nanbeige_lane")
            if cached is None or cached[0] != identity:
                prepared = prepare(module)
                split = policy["projection_split_k"]
                if split is not None:
                    if prepared.backend != "mpp" or prepared.k // 64 % split:
                        raise ValueError(
                            "requested split-K is incompatible with projection"
                        )
                    prepared = replace(prepared, split_k=split)
                cached = (identity, prepared)
                object.__setattr__(module, "_nanbeige_lane", cached)
            counts["lane_calls"] += 1
            return lane_matmul(x, cached[1])
    counts["reference_projection_calls"] += 1
    return module(x)


@partial(mx.compile, shapeless=True)
def compiled_q8(x, weight, scales, biases):
    return mx.quantized_matmul(
        x,
        weight,
        scales=scales,
        biases=biases,
        transpose=True,
        group_size=64,
        bits=8,
        mode="affine",
    )


@mx.compile
def compiled_mlp(x, weight, scales, biases, down_weight, down_scales, down_biases):
    y = compiled_q8(x, weight, scales, biases)
    gate, up = mx.split(y, 2, axis=-1)
    return compiled_q8(swiglu(gate, up), down_weight, down_scales, down_biases)


_SIGMOID = r"""
template <typename U>
inline U nanbeige_sigmoid(U x) {
  auto y = 1 / (1 + metal::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
"""


def _served_silu(x):
    return swiglu(x, mx.ones_like(x))


_SILU_GATE = ServedExpGate(
    "nanbeige_q8_swiglu",
    served_name="compiled SwiGLU",
    served=_served_silu,
    header=_SIGMOID,
    body="T t = x * nanbeige_sigmoid<T>(x); y = t * T(1);",
)


def q8_swiglu_source(width, tile_rows=32):
    """Adapt the original tiled generator's packing and served activation."""
    if type(width) is not int or width < 1 or tile_rows not in (16, 32):
        raise ValueError("invalid Q8 SwiGLU tile geometry")
    source, names = prefill_metal.swiglu_source(width, (False, False))
    source = source.replace(
        "(K / 8) + k / 8] >> ((k % 8) * 4)) & 15u",
        "(K / 4) + k / 4] >> ((k % 4) * 8)) & 255u",
    )
    source = source.replace("constexpr int TM = 32", f"constexpr int TM = {tile_rows}")
    start = source.index("            const float g = float(value);")
    end = source.index("OUT[size_t(row) * N + col] = value;", start)
    source = (
        source[:start]
        + "value = bfloat(value * nanbeige_sigmoid<bfloat>(value));\nvalue = bfloat(value * up);\n"
        + source[end:]
    )
    return source, names


@lru_cache(maxsize=32)
def _q8_swiglu_kernel(width, tile_rows, spelling):
    source, names = q8_swiglu_source(width, tile_rows)
    header = prefill_metal.HEADER + _SIGMOID.replace("metal::exp", spelling)
    digest = hashlib.sha256((source + header).encode()).hexdigest()[:16]
    return mx.fast.metal_kernel(
        name="nanbeige_q8_swiglu_" + digest,
        input_names=names,
        output_names=["OUT"],
        source=source,
        header=header,
    )


def q8_swiglu(x, gate_up, tile_rows=32):
    if (
        not quantized_q8(gate_up)
        or x.dtype != mx.bfloat16
        or gate_up.scales.dtype != mx.bfloat16
        or gate_up.biases.dtype != mx.bfloat16
    ):
        raise ValueError("Q8 SwiGLU requires BF16 affine Q8 group64")
    if gate_up.weight.shape[0] % 2 or x.shape[-1] != gate_up.weight.shape[-1] * 4:
        raise ValueError("Q8 SwiGLU projection geometry differs")
    if mx.default_device() != mx.gpu or "M5" not in str(
        mx.device_info().get("device_name", "")
    ):
        raise ValueError("Q8 SwiGLU tile requires an M5 GPU")
    spelling = _SILU_GATE.spelling(mx.bfloat16)
    if spelling is None:
        raise ValueError("Q8 SwiGLU served activation canary refused")
    n, k = gate_up.weight.shape[0] // 2, x.shape[-1]
    rows = x.size // k
    if k % 64 or rows < 1:
        raise ValueError("Q8 SwiGLU requires nonempty group64 geometry")
    operands = [mx.contiguous(x.reshape(rows, k)), rows]
    for begin, end in ((0, n), (n, 2 * n)):
        operands.extend(
            mx.contiguous(gate_up[key][begin:end])
            for key in ("weight", "scales", "biases")
        )
    return _q8_swiglu_kernel(n, tile_rows, spelling)(
        inputs=operands,
        template=[("K", k)],
        grid=(128 * ((rows + tile_rows - 1) // tile_rows), (n + 63) // 64, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[mx.bfloat16],
    )[0].reshape(*x.shape[:-1], n)


@lru_cache(maxsize=16)
def _compiled_rope_append(dims, base, scale, traditional):
    # Explicit array state: no closed-over Python cache or captured integer offset.
    @mx.compile
    def run(q, k, v, keys, values, offset):
        qr = mx.fast.rope(
            q, dims, traditional=traditional, base=base, scale=scale, offset=offset
        )
        kr = mx.fast.rope(
            k, dims, traditional=traditional, base=base, scale=scale, offset=offset
        )
        return (
            qr,
            mx.slice_update(keys, kr, offset.reshape(1), axes=(2,)),
            mx.slice_update(values, v, offset.reshape(1), axes=(2,)),
        )

    return run


def rope_append(q, k, v, rope, cache):
    """Pure graph successor arrays, followed by explicit host metadata commit.

    Returns None before any mutation when this graph does not own the geometry.
    No explicit prefix concatenation is added here. Backend storage donation
    and actual copy/launch behavior still require measurement.
    """
    from ..runtime.models.cache import KVCache

    if type(cache) is not KVCache or type(rope) is not nn.RoPE:
        return None
    if type(cache.offset) is not int or cache.keys is None or cache.values is None:
        return None
    start, end = cache.offset, cache.offset + k.shape[2]
    if end > cache.keys.shape[2] or k.shape != v.shape or q.shape[0] != k.shape[0]:
        return None
    if (
        q.dtype != k.dtype
        or v.dtype != cache.values.dtype
        or k.dtype != cache.keys.dtype
    ):
        return None
    qr, keys, values = _compiled_rope_append(
        rope.dims, rope.base, rope.scale, rope.traditional
    )(q, k, v, cache.keys, cache.values, mx.array(start, mx.int32))
    cache.keys, cache.values, cache.offset = keys, values, end
    return qr, keys[..., :end, :], values[..., :end, :]


def physical_rope_append(q, k, v, rope, cache):
    """Plain owner hook; AST wrappers use their own lane/position hook."""
    from ..runtime.models.cache import KVCache
    from .nanbeige43_physical import fused_rope_append_arrays

    if type(cache) is not KVCache or type(rope) is not nn.RoPE:
        return None
    if rope.dims != 128 or rope.scale != 1.0 or rope.traditional:
        return None
    if type(cache.offset) is not int:
        return None
    positions = mx.arange(k.shape[2], dtype=mx.int32) + cache.offset
    result = fused_rope_append_arrays(
        q, k, v, cache.keys, cache.values, cache.offset, positions, base=rope.base
    )
    if result is None:
        return None
    qr, keys, values, end = result
    cache.keys, cache.values, cache.offset = keys, values, end
    return qr, keys[..., :end, :], values[..., :end, :]
