"""Opt-in, unqualified LFM2.5-VL one-token ShortConv candidate.

The pinned mlx-vlm ShortConv remains the reference path.  This module only
replaces its projected convolution for a fully cached, single-row BF16 decode
with the known 2048-channel, three-tap, bias-free geometry.  No kernel is
constructed or imported unless the environment switch is enabled on a Metal
GPU.  See provenance/lfm25-vl.json.
"""

from __future__ import annotations

import os
import types
from dataclasses import dataclass
from functools import lru_cache


ENV = "MLX2_LFM25_FUSED_SHORTCONV"
CHANNELS = 2048
TAPS = 3


@dataclass(frozen=True)
class Admission:
    accepted: bool
    reason: str


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip().lower() in {"1", "true", "on", "yes"}


def admit(*, projected, state, weight, bias, mask, cache, gdn_sink, dtype,
          kernel_size=TAPS, groups=CHANNELS, stride=1, dilation=1, padding=0) -> Admission:
    """Host-only structural gate; never reads a device value."""
    if gdn_sink is not None:
        return Admission(False, "speculative state capture")
    if mask is not None:
        return Admission(False, "masked or padded slab")
    if cache is None or state is None:
        return Admission(False, "missing completed convolution state")
    if (kernel_size, groups, stride, dilation, padding) != (TAPS, CHANNELS, 1, 1, 0):
        return Admission(False, "depthwise convolution operator geometry")
    if getattr(cache, "lengths", None) is not None or getattr(cache, "left_padding", None) is not None:
        return Admission(False, "ragged cache metadata")
    if tuple(getattr(projected, "shape", ())) != (1, 1, 3 * CHANNELS):
        return Admission(False, "one-token B1 projection geometry")
    if tuple(getattr(state, "shape", ())) != (1, TAPS - 1, CHANNELS):
        return Admission(False, "two-row convolution state geometry")
    if tuple(getattr(weight, "shape", ())) != (CHANNELS, TAPS, 1):
        return Admission(False, "sanitized depthwise convolution weights")
    if bias is not None:
        return Admission(False, "convolution bias")
    if any(getattr(value, "dtype", None) != dtype for value in (projected, state, weight)):
        return Admission(False, "BF16 dtype mismatch")
    return Admission(True, "eligible")


_SOURCE = r"""
  const uint c = thread_position_in_grid.x;
  if (c >= uint(CHANNELS)) return;
  const T b = projected[c];
  const T gate = projected[uint(CHANNELS) + c];
  const T x = projected[2u * uint(CHANNELS) + c];
  // The donor materializes B*x in activation dtype before nn.Conv1d.
  const T bx = T(b * x);
  const device T* w = weight + size_t(c) * uint(TAPS);
  float sum = float(state[c]) * float(w[0]);
  sum += float(state[uint(CHANNELS) + c]) * float(w[1]);
  sum += float(bx) * float(w[2]);
  const T conv = T(sum);
  output[c] = T(gate * conv);
  state_out[c] = state[uint(CHANNELS) + c];
  state_out[uint(CHANNELS) + c] = bx;
"""


@lru_cache(maxsize=1)
def _kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="mlx2_lfm25_shortconv_decode_candidate",
        input_names=["projected", "state", "weight"],
        output_names=["output", "state_out"],
        source=_SOURCE,
        ensure_row_contiguous=True,
    )


def fused_projected(projected, state, weight):
    """Build a two-output Metal graph; caller owns the cache write."""
    result = _kernel()(
        inputs=[projected, state, weight],
        template=[("T", projected.dtype), ("CHANNELS", CHANNELS), ("TAPS", TAPS)],
        grid=(CHANNELS, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, 1, CHANNELS), (1, TAPS - 1, CHANNELS)],
        output_dtypes=[projected.dtype, projected.dtype],
    )
    return tuple(result)


class ShortConvCounters:
    """Cheap mechanism counters, with no timing or device synchronization."""

    def __init__(self):
        self.installed = 0
        self.engaged = 0
        self.refused = 0
        self.build_failed = 0


def install(model, counters: ShortConvCounters) -> int:
    """Patch only this loaded model's ShortConv instances, never the donor class."""
    import mlx.core as mx

    if not (enabled() and hasattr(mx, "fast") and hasattr(mx.fast, "metal_kernel")
            and hasattr(mx, "metal") and mx.metal.is_available()
            and mx.default_device() == mx.gpu):
        return 0

    conv_layers = [layer for layer in model.language_model.layers
                   if not layer.is_attention_layer]
    if any(getattr(layer.conv, "_mlx2_lfm25_shortconv_installed", False)
           for layer in conv_layers):
        raise ValueError("LFM ShortConv candidate is already installed")
    for layer in conv_layers:
        conv = layer.conv
        reference = conv._convolve_projected

        def projected_call(self, projected, mask, cache, gdn_sink, *, _reference=reference):
            state = None if cache is None else cache[0]
            admission = admit(
                projected=projected, state=state, weight=self.conv.weight,
                bias=getattr(self.conv, "bias", None), mask=mask, cache=cache, gdn_sink=gdn_sink,
                dtype=mx.bfloat16,
                kernel_size=self.L_cache, groups=self.conv.groups,
                stride=self.conv.stride, dilation=self.conv.dilation,
                padding=self.conv.padding,
            )
            if not admission.accepted:
                counters.refused += 1
                return _reference(projected, mask, cache, gdn_sink)
            try:
                out, new_state = fused_projected(projected, state, self.conv.weight)
            except (RuntimeError, ValueError, TypeError):
                counters.build_failed += 1
                return _reference(projected, mask, cache, gdn_sink)
            cache[0] = new_state
            cache.advance(1)
            counters.engaged += 1
            return out

        conv._convolve_projected = types.MethodType(projected_call, conv)
        conv._mlx2_lfm25_shortconv_installed = True
        counters.installed += 1
    return counters.installed
