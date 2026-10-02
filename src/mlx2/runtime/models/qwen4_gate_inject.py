"""Candidate Metal fusion for Flash-Next raw gate plus residual injection.

The candidate is deliberately adapter-owned and default-off.  It consumes the
raw four-value hyper-connection projection and performs divide, MLX-compatible
precise sigmoid, scale, broadcast multiply, and residual add in one dispatch.
The ordinary MLX graph remains the fallback for every declined or failed call.

Design input: llama.cpp PR #29520 / merge commit 03a667aa304f2a8e02a9a02b2e3fb45d64bcae7f.
No upstream shader code is copied; see provenance/llama-cpp-29520-qwen4-gate-inject.json.
"""

from __future__ import annotations

import os
import threading
from collections import Counter
from dataclasses import dataclass

import mlx.core as mx

HC_COUNT = 4
HIDDEN_SIZE = 2560
STREAM_WIDTH = HC_COUNT * HIDDEN_SIZE
MAX_VERIFY_TOKENS = 17
MAX_BATCH = 32
MODEL_VALIDATION_RECEIPT = (
    "qualification/runs/qwen4-gate-inject-20260930/model-validation.json"
)
PHYSICAL_ENGAGEMENT_RECEIPT = (
    "qualification/runs/qwen4-gate-inject-20260930/"
    "candidate-dispatch-count.json"
)

from .import_env import snapshot as _import_env_snapshot

_import_env_snapshot(__name__)
# Default off.  Options sweep on the served Flash-Next artifact
# (qualification/runs/options-sweep-20261001): bit-identical everywhere, but
# while it took every dense-inject call from the HC decode kernels (which
# already fuse the inject) ordinary B1 decode lost 8.5% [-12.5, -6.5] and 32K
# ordinary 8.5%; MTP B1 -0.2%, 4 lanes +1.3% (noise).  It now yields to those
# kernels (qwen4_exp.GatedResidual.split_for_gate_inject) and serves only
# the calls they decline: verify windows wider than 8 rows, batches past 8
# lanes, multi-row "off", non-eager norms.  The narrowed route is unmeasured.
_ENABLE = os.environ.get("MLX_QWEN4_FUSED_GATE_INJECT", "0") == "1"
_KERNEL = None
_KERNEL_LOCK = threading.Lock()
_STATS = Counter()
_STATS_LOCK = threading.Lock()
_LAST_DECISION: dict | None = None

_HEADER = "\n#include <metal_stdlib>\nusing namespace metal;\n"
_SOURCE = r"""
    const uint index = thread_position_in_grid.x;
    const uint row = index / 10240u;
    const uint stream_offset = index - row * 10240u;
    const uint stream = stream_offset / 2560u;
    const uint hidden = stream_offset - stream * 2560u;

    // Preserve the storage-dtype boundaries of the five eager dispatches.
    const T scaled_gate = T(raw_gate[row * 4u + stream] / T(4));
    const T sigmoid_denominator = T(
        T(1) + metal::precise::exp(metal::abs(scaled_gate)));
    const T sigmoid_low = T(T(1) / sigmoid_denominator);
    const T sigmoid_value = scaled_gate < T(0)
        ? sigmoid_low : T(T(1) - sigmoid_low);
    const T inject = T(T(2) * sigmoid_value);
    const T product = T(branch[row * 2560u + hidden] * inject);
    out[index] = T(residual[index] + product);
"""


@dataclass(frozen=True)
class GateInjectAdmission:
    accepted: bool
    reason: str
    mode: str | None = None


def fused_gate_inject_enabled() -> bool:
    """Return the live candidate lever."""
    return _ENABLE


def set_fused_gate_inject_enabled(enabled: bool) -> bool:
    """Set the process-local candidate lever, primarily for bounded tests."""
    global _ENABLE
    _ENABLE = bool(enabled)
    return _ENABLE


def _shape(value) -> tuple[int, ...]:
    return tuple(int(extent) for extent in value.shape)


def admit_qwen4_gate_inject(
    residual,
    branch,
    raw_gate,
    *,
    training: bool = False,
) -> GateInjectAdmission:
    """Fail closed outside the measured Flash-Next decode/verify envelope."""
    if not _ENABLE:
        return GateInjectAdmission(False, "fused gate inject is not enabled")
    if training:
        return GateInjectAdmission(False, "training is unsupported")
    if residual.dtype != mx.bfloat16:
        return GateInjectAdmission(False, "residual must be bfloat16")
    if branch.dtype != mx.bfloat16 or raw_gate.dtype != mx.bfloat16:
        return GateInjectAdmission(False, "branch and raw gate must be bfloat16")
    if residual.ndim != 3 or branch.ndim != 3 or raw_gate.ndim != 3:
        return GateInjectAdmission(False, "inputs must be rank-3 [batch, tokens, width]")

    batch, tokens, width = _shape(residual)
    if width != STREAM_WIDTH:
        return GateInjectAdmission(False, f"residual width must be {STREAM_WIDTH}")
    if _shape(branch) != (batch, tokens, HIDDEN_SIZE):
        return GateInjectAdmission(False, f"branch must be [batch, tokens, {HIDDEN_SIZE}]")
    if _shape(raw_gate) != (batch, tokens, HC_COUNT):
        return GateInjectAdmission(False, f"raw gate must be [batch, tokens, {HC_COUNT}]")
    if batch < 1 or tokens < 1:
        return GateInjectAdmission(False, "empty batch or token axis")
    if tokens == 1 and batch <= MAX_BATCH:
        mode = "ordinary_b1" if batch == 1 else "batched_decode"
    elif batch == 1 and tokens <= MAX_VERIFY_TOKENS:
        mode = "self_mtp_verify"
    else:
        return GateInjectAdmission(
            False,
            "geometry must be B1 decode, batched T=1 decode, or B1 self-MTP verify "
            f"up to {MAX_VERIFY_TOKENS} tokens",
        )
    if not hasattr(mx.fast, "metal_kernel") or not mx.metal.is_available():
        return GateInjectAdmission(False, "Metal kernel unavailable")
    if mx.default_device() != mx.gpu:
        return GateInjectAdmission(False, "default device is not the Metal GPU")
    return GateInjectAdmission(True, "eligible", mode)


def _get_kernel():
    global _KERNEL
    if _KERNEL is None:
        with _KERNEL_LOCK:
            if _KERNEL is None:
                _KERNEL = mx.fast.metal_kernel(
                    name="qwen4_raw_gate_inject_bf16",
                    input_names=["residual", "branch", "raw_gate"],
                    output_names=["out"],
                    header=_HEADER,
                    source=_SOURCE,
                    ensure_row_contiguous=True,
                )
    return _KERNEL


def _launch(residual, branch, raw_gate) -> mx.array:
    size = int(residual.size)
    (out,) = _get_kernel()(
        inputs=[residual, branch, raw_gate],
        template=[("T", residual.dtype)],
        grid=(size, 1, 1),
        threadgroup=(min(256, size), 1, 1),
        output_shapes=[residual.shape],
        output_dtypes=[residual.dtype],
    )
    return out


def _record_decision(admission: GateInjectAdmission, *, error: str | None = None) -> None:
    global _LAST_DECISION
    with _STATS_LOCK:
        _STATS["attempts"] += 1
        if admission.accepted and error is None:
            _STATS["calls"] += 1
            _STATS[f"calls:{admission.mode}"] += 1
        else:
            _STATS["declines"] += 1
            reason = "runtime_error" if error is not None else admission.reason
            _STATS[f"decline:{reason}"] += 1
        _LAST_DECISION = {
            "accepted": admission.accepted and error is None,
            "reason": "runtime_error" if error is not None else admission.reason,
            "mode": admission.mode,
            "error": error,
        }


def record_gate_inject_yield(reason: str) -> None:
    """Count a call the candidate left to a faster fused path (a decline)."""
    _record_decision(GateInjectAdmission(False, reason))


def try_qwen4_gate_inject(
    residual,
    branch,
    raw_gate,
    *,
    training: bool = False,
) -> mx.array | None:
    """Return one-dispatch output, or ``None`` for the ordinary fallback."""
    if not _ENABLE:
        return None
    admission = admit_qwen4_gate_inject(
        residual, branch, raw_gate, training=training
    )
    if not admission.accepted:
        _record_decision(admission)
        return None
    try:
        out = _launch(residual, branch, raw_gate)
    except Exception as exc:  # noqa: BLE001 - failure must fall back closed
        _record_decision(admission, error=repr(exc))
        return None
    _record_decision(admission)
    return out


def qwen4_gate_inject_stats(*, reset: bool = False) -> dict:
    """Return bounded call/decline evidence for route receipts."""
    global _LAST_DECISION
    with _STATS_LOCK:
        report = {
            "enabled": _ENABLE,
            "implemented": True,
            "model_validation_passed": True,
            "qualified": False,
            "selected": _ENABLE,
            # Physical dispatch engagement is established by a separate LLDB
            # receipt; graph construction or a call counter cannot prove it.
            "observed_used": False,
            "counts": dict(_STATS),
            "last_decision": _LAST_DECISION,
            "upstream_revision": "03a667aa304f2a8e02a9a02b2e3fb45d64bcae7f",
            "model_validation_receipt": MODEL_VALIDATION_RECEIPT,
            "physical_engagement_receipt": PHYSICAL_ENGAGEMENT_RECEIPT,
        }
        if reset:
            _STATS.clear()
            _LAST_DECISION = None
    return report


def reset_qwen4_gate_inject_stats() -> None:
    qwen4_gate_inject_stats(reset=True)


def eager_qwen4_gate_inject(residual, branch, raw_gate) -> mx.array:
    """Ordinary reference graph, including the original sigmoid boundary."""
    inject = 2 * mx.sigmoid(raw_gate / HC_COUNT)
    return residual + (branch[..., None, :] * inject[..., None]).reshape(
        *residual.shape
    )
