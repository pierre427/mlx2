# SPDX-License-Identifier: Apache-2.0 AND MIT
"""Ling's activation-dtype gated norm; see ling-kimi-optimizations provenance.

The shared MIT Metal template is also used by Kimi, but Ling rounds both
RMSNorm and sigmoid to the activation dtype before multiplying them.
"""
from functools import lru_cache

import mlx.core as mx

from .served_exp import ServedExpGate

_SIGMOID = """
    T sy = 1 / (1 + metal::precise::exp(metal::abs(x)));
    y = (x < 0) ? sy : 1 - sy;
"""
_GATE = ServedExpGate("ling_norm_sigmoid", served_name="mx.sigmoid",
                      served=mx.sigmoid, body=_SIGMOID,
                      kernel_exp="metal::precise::exp")


def norm_source(spelling):
    from .qwen4_fused_gdn_prefill import prefill_norm_gate_source

    source = prefill_norm_gate_source("sigmoid")
    old = """float zv = float(z[base + i]);
        float sy = 1.0f / (1.0f + metal::precise::exp(metal::abs(zv)));
        float sig = zv < 0.0f ? sy : 1.0f - sy;
        out[base + i] = T(float(normed) * sig);"""
    if source.count(old) != 1:
        raise RuntimeError("Ling norm donor template changed")
    body = _SIGMOID.replace("metal::precise::exp", spelling)
    return source.replace(old, "const T x = z[base + i]; T y;\n" + body
                          + "\n        out[base + i] = normed * y;")


@lru_cache(maxsize=2)
def _kernel(spelling):
    return mx.fast.metal_kernel(
        name="mlx2_ling_norm_gate_v1_" + ("precise" if "precise" in spelling else "fast"),
        input_names=["y", "z", "norm_w", "eps"], output_names=["out"],
        source=norm_source(spelling), ensure_row_contiguous=True,
    )


def norm_gate(out, gate, weight, eps):
    """Return None for unsupported geometry/build; caller keeps eager math."""
    if (mx.default_device() != mx.gpu or out.ndim != 4 or out.shape[-1] != 128
            or gate.shape != out.shape or weight.shape != (128,)
            or out.dtype not in (mx.bfloat16, mx.float32)
            or gate.dtype != out.dtype or weight.dtype != out.dtype):
        return None
    spelling = _GATE.spelling(out.dtype)
    if spelling is None:
        return None
    b, t, h, d = out.shape
    return _kernel(spelling)(
        inputs=[out, gate, weight, mx.array(eps, mx.float32)],
        template=[("T", out.dtype), ("HV", h), ("DV", d)],
        grid=(32, b*t, h), threadgroup=(32, 1, 1),
        output_shapes=[out.shape], output_dtypes=[out.dtype],
    )[0]
