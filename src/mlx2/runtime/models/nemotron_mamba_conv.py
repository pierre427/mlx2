"""Geometry-locked B=1/S=1 depthwise-convolution candidates for Nemotron-H.

The kernel covers only the convolution window update, bias, and SiLU.  The
adapter must select it explicitly; every other geometry keeps the ordinary
``nn.Conv1d`` path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx import nn

from .served_exp import ServedExpGate, adapted_helper_header, kernel_dtype_refusal


@dataclass(frozen=True)
class MambaConvAdmission:
    accepted: bool
    reason: str


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(getattr(value, "shape", ()))


def admit_fused_mamba_conv_b1(
    *, conv_input, conv_state, weight, bias, mask, training: bool, kernel_size: int
) -> MambaConvAdmission:
    if training:
        return MambaConvAdmission(False, "training")
    if mask is not None:
        return MambaConvAdmission(False, "masked input")
    if kernel_size != 4:
        return MambaConvAdmission(False, f"kernel size {kernel_size}")
    if _shape(conv_input)[:2] != (1, 1):
        return MambaConvAdmission(False, f"input shape {_shape(conv_input)}")
    channels = _shape(conv_input)[-1] if len(_shape(conv_input)) == 3 else 0
    if _shape(conv_state) != (1, 3, channels):
        return MambaConvAdmission(False, f"state shape {_shape(conv_state)}")
    if _shape(weight) != (channels, 4, 1):
        return MambaConvAdmission(False, f"weight shape {_shape(weight)}")
    if _shape(bias) != (channels,):
        return MambaConvAdmission(False, f"bias shape {_shape(bias)}")
    if not channels:
        return MambaConvAdmission(False, "empty channels")
    if any(
        value.dtype != mx.bfloat16 for value in (conv_input, conv_state, weight, bias)
    ):
        return MambaConvAdmission(False, "requires bf16 operands")
    return MambaConvAdmission(True, "eligible")


def admit_fused_mamba_conv_short(
    *, conv_input, conv_state, weight, bias, mask, training: bool, kernel_size: int
) -> MambaConvAdmission:
    """Admit one linear conversation segment of at most sixteen rows."""

    if training:
        return MambaConvAdmission(False, "training")
    if mask is not None:
        return MambaConvAdmission(False, "masked input")
    if kernel_size != 4:
        return MambaConvAdmission(False, f"kernel size {kernel_size}")
    shape = _shape(conv_input)
    if len(shape) != 3 or shape[0] != 1 or not 1 <= shape[1] <= 16:
        return MambaConvAdmission(False, f"input shape {shape}")
    channels = shape[-1]
    if _shape(conv_state) != (1, 3, channels):
        return MambaConvAdmission(False, f"state shape {_shape(conv_state)}")
    if _shape(weight) != (channels, 4, 1):
        return MambaConvAdmission(False, f"weight shape {_shape(weight)}")
    if _shape(bias) != (channels,):
        return MambaConvAdmission(False, f"bias shape {_shape(bias)}")
    if any(
        value.dtype != mx.bfloat16 for value in (conv_input, conv_state, weight, bias)
    ):
        return MambaConvAdmission(False, "requires bf16 operands")
    return MambaConvAdmission(True, "eligible")


_HEADER = """
template <typename U>
inline U mlx_sigmoid_fast(U x) {
  U e = static_cast<U>(metal::exp(metal::abs(x)));
  U y = static_cast<U>(1) / (static_cast<U>(1) + e);
  return (x < 0) ? y : (static_cast<U>(1) - y);
}
"""

_SOURCE = """
  const uint channel = thread_position_in_grid.x;
  if (channel >= uint(C)) return;
  const device T* wc = weight + size_t(channel) * 4u;
  float acc = float(state[channel]) * float(wc[0]);
  acc += float(state[uint(C) + channel]) * float(wc[1]);
  acc += float(state[2u * uint(C) + channel]) * float(wc[2]);
  acc += float(input[channel]) * float(wc[3]);
  // mlx.nn.Conv1d materializes its output before the separate bias add.
  // Preserve that BF16 boundary rather than folding the bias into the FP32
  // convolution accumulator.
  T conv = static_cast<T>(acc);
  T xb = conv + bias[channel];
  output[channel] = xb * mlx_sigmoid_fast(xb);
  for (uint row = 0; row < uint(HISTORY_ROWS); ++row) {
    const uint source_row = row + uint(HISTORY_OFFSET);
    history[row * uint(C) + channel] =
        source_row < 3u ? state[source_row * uint(C) + channel] : input[channel];
  }
"""

_KERNEL = None
_SHORT_KERNEL = None

_SILU_GATE = ServedExpGate(
    "nemotron_mamba_conv_silu",
    served_name="compiled nn.silu",
    served=nn.silu,
    header=_HEADER,
    body="    y = x * mlx_sigmoid_fast<T>(x);",
)


def fused_mamba_conv_refusal(dtype=mx.bfloat16) -> str | None:
    """Why this build cannot preserve the served SiLU spelling, or ``None``."""
    return kernel_dtype_refusal(_SILU_GATE, dtype)


def _served_header() -> tuple[str, str]:
    """Kernel header with the SiLU sigmoid spelled like the served op."""
    return adapted_helper_header(_SILU_GATE, _HEADER, "mlx_sigmoid_fast")


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="mlx2_nemotron_mamba_conv4_silu_b1" + _served_header()[1],
            input_names=["input", "state", "weight", "bias"],
            output_names=["output", "history"],
            header=_served_header()[0],
            source=_SOURCE,
            ensure_row_contiguous=True,
        )
    return _KERNEL


_SHORT_SOURCE = r"""
  const uint channel = thread_position_in_grid.x;
  if (channel >= uint(C)) return;
  const device T* wc = weight + size_t(channel) * 4u;
  T s0 = state[channel];
  T s1 = state[uint(C) + channel];
  T s2 = state[2u * uint(C) + channel];
  if constexpr (CHECKPOINTS) {
    history[channel] = s0;
    history[uint(C) + channel] = s1;
    history[2u * uint(C) + channel] = s2;
  }
  for (int step = 0; step < S; ++step) {
    T in = input[uint(step) * uint(C) + channel];
    float acc = float(s0) * float(wc[0]);
    acc += float(s1) * float(wc[1]);
    acc += float(s2) * float(wc[2]);
    acc += float(in) * float(wc[3]);
    T conv = static_cast<T>(acc);
    T xb = conv + bias[channel];
    output[uint(step) * uint(C) + channel] =
        xb * mlx_sigmoid_fast(xb);
    s0 = s1;
    s1 = s2;
    s2 = in;
    if constexpr (CHECKPOINTS) {
      const uint base = uint(step + 1) * 3u * uint(C);
      history[base + channel] = s0;
      history[base + uint(C) + channel] = s1;
      history[base + 2u * uint(C) + channel] = s2;
    }
  }
  if constexpr (!CHECKPOINTS) {
    history[channel] = s0;
    history[uint(C) + channel] = s1;
    history[2u * uint(C) + channel] = s2;
  }
"""


def _short_kernel():
    global _SHORT_KERNEL
    if _SHORT_KERNEL is None:
        _SHORT_KERNEL = mx.fast.metal_kernel(
            name="mlx2_nemotron_mamba_conv4_silu_short" + _served_header()[1],
            input_names=["input", "state", "weight", "bias"],
            output_names=["output", "history"],
            header=_served_header()[0],
            source=_SHORT_SOURCE,
            ensure_row_contiguous=True,
        )
    return _SHORT_KERNEL


def fused_mamba_conv_b1(
    conv_input, conv_state, weight, bias, *, capture_window: bool = True
):
    """Return the convolution and either rollback window or next state.

    Four rows are required only by speculative rollback: prefix zero restores
    rows 0..2 and prefix one commits rows 1..3. Ordinary decode emits rows
    1..3 directly and avoids materializing the otherwise dead oldest row.
    """
    channels = int(conv_input.shape[-1])
    history_rows = 4 if capture_window else 3
    history_offset = 0 if capture_window else 1
    output, history = _kernel()(
        inputs=[conv_input, conv_state, weight, bias],
        template=[
            ("T", conv_input.dtype),
            ("C", channels),
            ("HISTORY_ROWS", history_rows),
            ("HISTORY_OFFSET", history_offset),
        ],
        grid=(channels, 1, 1),
        threadgroup=(min(256, channels), 1, 1),
        output_shapes=[(1, 1, channels), (1, history_rows, channels)],
        output_dtypes=[conv_input.dtype, conv_input.dtype],
    )
    return output, history


def fused_mamba_conv_short(
    conv_input, conv_state, weight, bias, *, capture_checkpoints: bool = True
):
    """Run one 1..16-row linear segment and return prefix windows.

    With checkpoints enabled, ``history[:, p]`` is the exact three-row conv
    window after accepting prefix ``p`` (including prefix zero).  This is the
    adapter-private rollback tape; APCv2 still owns the committed layer
    checkpoint and segment revision.
    """

    steps = int(conv_input.shape[1])
    channels = int(conv_input.shape[-1])
    history_shape = (
        (1, steps + 1, 3, channels) if capture_checkpoints else (1, 3, channels)
    )
    output, history = _short_kernel()(
        inputs=[conv_input, conv_state, weight, bias],
        template=[
            ("T", conv_input.dtype),
            ("C", channels),
            ("S", steps),
            ("CHECKPOINTS", int(capture_checkpoints)),
        ],
        grid=(channels, 1, 1),
        threadgroup=(min(256, channels), 1, 1),
        output_shapes=[conv_input.shape, history_shape],
        output_dtypes=[conv_input.dtype, conv_input.dtype],
    )
    return output, history
