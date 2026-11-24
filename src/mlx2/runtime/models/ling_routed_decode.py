# SPDX-License-Identifier: Apache-2.0
# The per-row quantized dot products are deepseek_v41_routed_decode's gate_up
# and down kernels over qwen4_routed_decode's QMV_HEADER (adapted from
# jundot/omlx PR #3912, Apache-2.0; MLX quantized.h, MIT, header pinned at
# 0.32.2; equality is checked against the installed MLX, 0.32.4 in the venv). See
# provenance/ling-routed-decode.json.
"""Device-routed one-token routed experts for a stacked ``SwitchGLU`` bank.

Opt-in decode route for the Ling (bailing_hybrid) MoE: ``SwitchGLU`` at one
token runs three unsorted ``gather_qmm`` calls (up, gate, down: MLX picks
``gather_qmv`` at M == 1) plus the activation. This route runs gate and up for
every selected expert in one launch and down in a second one, reading the
expert IDs from a device buffer exactly as ``gather_qmv`` does.

What this route is NOT: a route-ID readback removal. Unlike the DS4.1
streamed-expert provider (``np.asarray(ids)`` per layer, 32 ms/token of
route_id_wait), the resident ``SwitchGLU`` decode never reads routes to the
host: a sys.monitoring probe of one tiny-Ling decode forward finds zero
``item``/``tolist``/``eval`` calls (Ling 3 Flash and vendor-law variants).
Any gain here is launch and activation-read fusion only, and must be measured.

Arithmetic contract (bit-identical to the ``SwitchGLU`` reference, GPU-checked
by ``self_check`` and the A/B, never assumed):

* gate/up: MLX ``qmv_fast`` lane partition and add order (``K % 512 == 0``),
  raw F32, then cast to the activation dtype as ``gather_qmv`` writes it.
* activation: the bank's own activation module (``SwiGLU`` or Ling's
  post-activation ``ClampedSwiGLU``) applied to those casts, i.e. the same
  MLX ops on the same values. No fused epilogue, so the clamp law
  ``min(silu(g), L) * clip(u, -L, L)`` is the reference's, not a re-implementation.
* down: MLX ``qmv`` with its tail-safe loop (``K % 512 != 0``), raw F32,
  cast to the activation dtype.

The caller keeps route weights, the weighted sum, its fp32/cast order and the
shared expert unchanged. Every unsupported case declines (returns ``None``)
and the caller runs ``SwitchGLU`` as before.
"""
import threading
from collections import Counter

import mlx.core as mx

from . import deepseek_v41_routed_decode as _v41

_LOCK = threading.Lock()
_COUNTS: Counter = Counter()


def _count(key: str) -> None:
    with _LOCK:
        _COUNTS[key] += 1


def status(*, reset: bool = False) -> dict:
    """Engagement counters: ``engaged`` and ``decline:<reason>`` since reset."""
    with _LOCK:
        out = dict(_COUNTS)
        if reset:
            _COUNTS.clear()
    return out


def implementation_identity() -> str:
    """Digest of this route's source, the V4.1 kernels, the QMV header and MLX."""
    import hashlib
    from pathlib import Path

    digest = hashlib.sha256(Path(__file__).read_bytes())
    digest.update(_v41.implementation_identity().encode())
    return digest.hexdigest()


def _spec(linear):
    """``{"bits", "group_size"}`` for a 4/8-bit-style affine QuantizedSwitchLinear, else None."""
    if getattr(linear, "mode", None) != "affine" or "bias" in linear:
        return None
    if linear.get("biases") is None or linear.get("scales") is None:
        return None
    return {"bits": int(linear.bits), "group_size": int(linear.group_size)}


def _metal_ready() -> bool:
    return mx.default_device() == mx.gpu and mx.metal.is_available()


def refusal(glu, x, indices) -> "str | None":
    """Why this call cannot take the routed route, or None when it can."""
    from . import invariant_prefill as _invariant
    from . import switch_grouped_experts as _grouped

    if _invariant.active():
        return "invariant_lane"
    if _grouped.MODE != "off":
        return "grouped_experts_active"
    hidden = int(x.shape[-1])
    if (indices.ndim < 2 or int(indices.shape[-2]) != 1 or x.size != hidden
            or indices.size != int(indices.shape[-1])):
        return "not_one_row"
    top = int(indices.shape[-1])
    # The reference must be the unsorted gather_qmv: a sorted call (auto at
    # top >= 20) can take the padded gather_qmm_rhs streaming kernel instead.
    from .switch_layers import _switch_sort_decision

    do_sort, _ = _switch_sort_decision(getattr(glu, "_sort_policy", "auto"),
                                       assignments=top, token_length=1, invariant=False)
    if do_sort:
        return "sorted_reference"
    projections = tuple(getattr(glu, name, None) for name in ("gate_proj", "up_proj", "down_proj"))
    if any(p is None for p in projections):
        return "not_split_bank"  # e.g. a fused gate|up bank
    specs = [_spec(p) for p in projections]
    if any(s is None for s in specs):
        return "not_affine_bank"
    gate, up, down = projections
    inter = int(gate.weight.shape[1])
    if (int(up.weight.shape[1]) != inter or int(down.weight.shape[1]) != hidden
            or gate.input_dims != hidden or down.input_dims != inter):
        return "bank_geometry"
    if x.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return "activation_dtype"
    # gather_qmm promotes its output to (x, scales); the casts below assume
    # that promotion is x.dtype, so every table must carry x.dtype metadata.
    meta = {a.dtype for p in projections for a in (p.scales, p.biases)}
    if meta != {x.dtype}:
        return "scale_dtype"
    if indices.dtype not in (mx.uint32, mx.int32, mx.uint16, mx.int16, mx.uint8):
        return "index_dtype"
    # MLX takes gather_qmv_fast only for N % 8 == 0 (no partial-rows variant);
    # any group size >= 32 already implies it for the gate/up rows (N = inter).
    if inter % 8 or not _v41.supports(hidden, inter, top, dict(zip(("gate", "up", "down"), specs))):
        return "kernel_shape"
    if not _metal_ready():
        return "no_metal"
    return None


def switch_glu(glu, x, indices, *, count: bool = True):
    """``glu(x, indices)`` for one token through the routed kernels, or None.

    Returns ``[..., top, hidden]`` in ``x.dtype`` and route order, exactly the
    shape and dtype ``SwitchGLU.__call__`` returns for an unsorted call.
    ``count=False`` (self_check) leaves the observed-use counters alone.
    """
    reason = refusal(glu, x, indices)
    if reason is not None:
        if count:
            _count("decline:" + reason)
        return None
    gate, up, down = glu.gate_proj, glu.up_proj, glu.down_proj
    spec_gu, spec_d = _spec(gate), _spec(down)
    ids = indices.reshape(-1)
    raw_gate, raw_up = _v41.gate_up(
        x.reshape(-1), ids,
        (gate.weight, gate.scales, gate.biases), (up.weight, up.scales, up.biases), spec_gu,
    )
    # gather_qmv writes static_cast<T>(float accum): the same RNE round.
    hidden_act = glu.activation(raw_up.astype(x.dtype), raw_gate.astype(x.dtype))
    raw_down = _v41.down(hidden_act, ids, (down.weight, down.scales, down.biases), spec_d)
    if count:
        _count("engaged")
    return raw_down.astype(x.dtype).reshape(*indices.shape, int(x.shape[-1]))


def self_check(glu, x, indices) -> dict:
    """GPU bit-equality of this route against ``SwitchGLU`` for one call."""
    routed = switch_glu(glu, x, indices, count=False)
    if routed is None:
        return {"engaged": False}
    reference = glu(x, indices)
    mx.eval(routed, reference)
    bits = mx.uint16 if routed.dtype in (mx.bfloat16, mx.float16) else mx.uint32
    exact = bool(mx.array_equal(routed.view(bits), reference.view(bits)).item())
    return {"engaged": True, "exact": exact,
            "max_abs": float(mx.abs(routed.astype(mx.float32) - reference.astype(mx.float32)).max().item())}
