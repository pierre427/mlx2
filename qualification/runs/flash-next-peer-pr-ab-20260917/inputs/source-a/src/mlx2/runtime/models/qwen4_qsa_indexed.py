# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
from __future__ import annotations
import hashlib
import math
import os
import threading
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any
import mlx.core as mx
import numpy as np
from .qwen4_qsa_indexed_merge import (
    combine_indexed_partials,
    fused_merge_enabled,
    fused_merge_status,
    mlx_sequential_merge,
    record_native_gate_engaged,
)
from .qwen4_qsa_nax import compact_blocks_to_kernel_inputs, compact_token_validity

_BLOCK_SIZE = 4
_SDPA_BLOCKS = 128
_SPLIT_CANDIDATES = (128, 64, 32, 16, 8)
_HPT_LADDER = (12, 6, 3, 1)
_HPT_ALLOWED = (12, 6, 4, 3, 2, 1)
_EXACT_MLX_BUILDS = frozenset(
    {"0.32.2.dev20260829+334084ce9", "0.32.2.dev20260911+a0d69e543",
     "0.32.2.dev20260915+2a817ad94"}
)
_SDPA_VECTOR_HEADER_SHA256 = (
    "2100a4d1eaa8a524c5147c82c771cad75197495c72daffa03e7ea4c259aebf10"
)
_QUANTIZED_BITS = frozenset({4, 8})
_QUANTIZED_GROUP_SIZES = frozenset({32, 64, 128})


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    value = raw.strip().lower()
    if value in {"1", "true", "on", "yes"}:
        return True
    if value in {"0", "false", "off", "no", ""}:
        return False
    raise ValueError(f"{name} must be 0/off or 1/on; got {raw!r}")


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


_PRIVATE_DELTA_MAX_QUERY = _env_int("MLX_LM_QSA_PRIVATE_DELTA_MAX_QUERY", 9, minimum=1)


def _env_mode(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value == "auto":
        return None
    if value in {"1", "true", "on", "yes"}:
        return True
    if value in {"0", "false", "off", "no", ""}:
        return False
    raise ValueError(f"{name} must be 0/off, 1/on, or auto; got {raw!r}")


_QSA_INDEXED_ENABLED = _env_mode("MLX_QWEN4_QSA_INDEXED")
_MIN_QUERY = _env_int("MLX_QWEN4_QSA_INDEXED_MIN_QUERY", 2, minimum=1)
_MAX_QUERY = _env_int("MLX_QWEN4_QSA_INDEXED_MAX_QUERY", 8, minimum=1)
_MIN_CONTEXT = _env_int("MLX_QWEN4_QSA_INDEXED_MIN_CONTEXT", 16384)
_MAX_CONTEXT = _env_int("MLX_QWEN4_QSA_INDEXED_MAX_CONTEXT", 0)
_AUTO_MIN_CONTEXT_M3 = _env_int("MLX_QWEN4_QSA_INDEXED_AUTO_MIN_CONTEXT_M3", 16384)
_AUTO_MIN_CONTEXT_M1 = _env_int("MLX_QWEN4_QSA_INDEXED_AUTO_MIN_CONTEXT_M1", 65536)
_SPLITS_OVERRIDE = _env_int("MLX_QWEN4_QSA_INDEXED_SPLITS", 0)
_HPT_OVERRIDE = _env_int("MLX_QWEN4_QSA_INDEXED_HPT", 0)
_HPT_CANDIDATES = (
    _HPT_LADDER if _env_flag("MLX_QWEN4_QSA_INDEXED_HPT_LADDER") else (12,)
)


def _qsa_indexed_mode() -> str:
    if _QSA_INDEXED_ENABLED is None:
        return "auto"
    return "on" if _QSA_INDEXED_ENABLED else "off"


def _mlx_build_hash() -> str | None:
    version = str(getattr(mx, "__version__", "unknown"))
    marker = version.rsplit("+", 1)
    if len(marker) == 2 and marker[1]:
        return marker[1]
    return None


def _sdpa_vector_header_path() -> Path | None:
    """Return the installed MLX ``sdpa_vector.h``, or None when unlocatable."""
    core_file = getattr(mx, "__file__", None)
    if not core_file:
        return None
    return (
        Path(core_file).resolve().parent
        / "include"
        / "mlx"
        / "backend"
        / "metal"
        / "kernels"
        / "sdpa_vector.h"
    )


@lru_cache(maxsize=1)
def _sdpa_vector_header_sha256() -> str | None:
    """Digest the installed header once, or None when it is not shipped."""
    path = _sdpa_vector_header_path()
    if path is None or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sdpa_header_state() -> tuple[bool, str | None, str | None]:
    """Return (verified, observed digest, decline reason) for the source pin.

    The version allowlist only names a build string; a local rebuild can keep
    that string while shipping different ``sdpa_vector_2pass`` arithmetic, and
    the exactness contract is a clone of *that source*. Both must hold.
    """
    observed = _sdpa_vector_header_sha256()
    if observed is None:
        return (False, None, "sdpa_header_missing")
    if observed != _SDPA_VECTOR_HEADER_SHA256:
        return (False, observed, "sdpa_header_mismatch")
    return (True, observed, None)


def indexed_splits_for(u_width: int) -> int:
    """Return the requested split ceiling for a compact block-slot width."""
    width = int(u_width)
    if width < 1:
        raise ValueError("u_width must be positive")
    requested = _SPLITS_OVERRIDE
    if requested == 0:
        requested = _SPLIT_CANDIDATES[0]
    if requested not in _SPLIT_CANDIDATES:
        allowed = ", ".join(map(str, reversed(_SPLIT_CANDIDATES)))
        raise ValueError(f"indexed QSA splits must be one of {allowed}")
    return requested


def indexed_hpt_for(gqa: int) -> int:
    """Return the requested heads-per-threadgroup for a GQA fan-out."""
    heads = int(gqa)
    if heads < 1:
        raise ValueError("gqa must be positive")
    requested = _HPT_OVERRIDE
    if requested == 0:
        requested = heads
    validate_hpt(requested, heads)
    return requested


def validate_hpt(hpt: int, gqa: int) -> int:
    """Reject a heads-per-threadgroup that is not an allowed divisor of GQA."""
    value = int(hpt)
    if value not in _HPT_ALLOWED:
        allowed = ", ".join(map(str, reversed(_HPT_ALLOWED)))
        raise ValueError(f"indexed QSA heads per threadgroup must be one of {allowed}")
    if int(gqa) % value:
        raise ValueError("indexed QSA heads per threadgroup must divide GQA")
    return value


def indexed_kernel_available() -> bool:
    """Return whether the current device can dispatch a Metal custom kernel."""
    return bool(
        hasattr(mx, "fast")
        and hasattr(mx.fast, "metal_kernel")
        and hasattr(mx, "metal")
        and mx.metal.is_available()
        and (mx.default_device() == mx.gpu)
    )


def qwen4_qsa_private_delta_min_context(length: int) -> int:
    """Return the runtime threshold for a source-phased query width.

    The legacy single threshold remains an explicit qualification override.
    Without it, M=1 is effectively disabled and speculative slabs require
    64K. The previous 16K/64K defaults were invalidated by the accepted
    256-token full-model brackets; the legacy override remains available for
    explicit qualification runs.
    """
    legacy = os.environ.get("MLX_LM_QSA_PRIVATE_DELTA_MIN_CONTEXT")
    if legacy is not None:
        return max(0, int(legacy))
    name = (
        "MLX_LM_QSA_PRIVATE_DELTA_MIN_CONTEXT_M1"
        if int(length) == 1
        else "MLX_LM_QSA_PRIVATE_DELTA_MIN_CONTEXT_MN"
    )
    default = 2**31 - 1 if int(length) == 1 else 65536
    return max(0, int(os.environ.get(name, str(default))))


def qwen4_qsa_indexed_private_delta_preflight(
    *,
    length: int,
    base_tokens: int,
    head_dim: int,
    num_query_heads: int,
    num_kv_heads: int,
    block_size: int,
    selected_blocks: int,
    training: bool = False,
) -> tuple[bool, str]:
    """Side-effect-free admission before any QSA ledger or KV mutation."""
    length = int(length)
    base_tokens = int(base_tokens)
    head_dim = int(head_dim)
    nqh = int(num_query_heads)
    nkh = int(num_kv_heads)
    block_size = int(block_size)
    selected_blocks = int(selected_blocks)
    if training:
        return (False, "training")
    if length < 1 or length > _PRIVATE_DELTA_MAX_QUERY:
        return (False, "width_out_of_range")
    if base_tokens <= 0 or base_tokens % _BLOCK_SIZE:
        return (False, "unaligned_base")
    if block_size != _BLOCK_SIZE:
        return (False, "unsupported_block_size")
    if head_dim != 256:
        return (False, "unsupported_head_dim")
    if nkh < 1 or nqh % nkh or nqh // nkh != 12:
        return (False, "unsupported_gqa")
    token_width = (selected_blocks + 1) * block_size
    if token_width <= 1024 or token_width > 8192:
        return (False, "unsupported_two_pass_geometry")
    mlx_version = str(getattr(mx, "__version__", "unknown"))
    allow_unverified = (
        os.environ.get("MLX_QWEN4_QSA_INDEXED_ALLOW_UNVERIFIED_MLX") == "1"
    )
    if mlx_version not in _EXACT_MLX_BUILDS and (not allow_unverified):
        return (False, "mlx_build_unverified")
    (header_verified, _, header_reason) = _sdpa_header_state()
    if not header_verified and (not allow_unverified):
        return (False, str(header_reason))
    if not indexed_kernel_available():
        return (False, "kernel_unavailable")
    return (True, "engaged")


def qwen4_qsa_indexed_private_delta_exact_set_preflight(
    *,
    batch: int,
    length: int,
    base_tokens: int,
    head_dim: int,
    num_query_heads: int,
    num_kv_heads: int,
    block_size: int,
    selected_blocks: int,
    training: bool = False,
) -> tuple[bool, str]:
    """Admit the experimental two-row shared-base fold.

    Exact-set identity is proved later by device array operations and consumed
    by the Metal kernel without a host readback.  This host preflight only
    admits the fixed B2 geometry in which that proof is meaningful.
    """
    if not _env_flag("MLX_LM_QSA_PRIVATE_DELTA_EXACT_SET_FOLD", default=True):
        return (False, "exact_set_fold_disabled")
    if int(batch) != 2:
        return (False, "exact_set_fold_requires_b2")
    return qwen4_qsa_indexed_private_delta_preflight(
        length=length,
        base_tokens=base_tokens,
        head_dim=head_dim,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        selected_blocks=selected_blocks,
        training=training,
    )


def qwen4_qsa_indexed_private_delta_exact_set_proof(compact, *, base_tokens: int):
    """Return a device-resident per-query proof of equal ordered base sets.

    There is intentionally no ``mx.eval`` or ``.item()`` here. The predicate
    becomes another dependency of the folded Metal dispatch. Delta/tail block
    ids may differ; only selected ids below the immutable-base boundary are
    part of the proof.
    """
    if int(compact.block_ids.shape[0]) != 2:
        raise ValueError("exact-set proof requires a B2 compact selection")
    if int(base_tokens) <= 0 or int(base_tokens) % int(compact.block_size):
        raise ValueError("exact-set proof base must be positive and aligned")
    (ids, counts, _n_sel, u_width, _q_pos, _left_pad, _total) = (
        compact_blocks_to_kernel_inputs(compact)
    )
    base_block = int(base_tokens) // int(compact.block_size)
    slots = mx.arange(u_width, dtype=mx.uint32)[None, :]
    valid = (slots < counts[..., None]) & (ids < base_block)
    same_valid = mx.all(valid[0] == valid[1], axis=-1)
    same_ids = mx.all(~(valid[0] | valid[1]) | (ids[0] == ids[1]), axis=-1)
    return mx.contiguous(same_valid & same_ids)


def _selection_topk_width(selection) -> int:
    ids = getattr(selection, "raw_block_ids", None)
    if ids is None:
        return 0
    return int(ids.shape[-1])


def qsa_indexed_quantized_cache_config(cache):
    if cache is None or not hasattr(cache, "group_size"):
        return None
    key_bits = getattr(cache, "key_bits", getattr(cache, "bits", None))
    value_bits = getattr(cache, "value_bits", getattr(cache, "bits", None))
    if key_bits is None or value_bits is None:
        return None
    return {
        "group_size": int(cache.group_size),
        "key_bits": int(key_bits),
        "value_bits": int(value_bits),
        "rotate": bool(getattr(cache, "rotate", False)),
        "normalize": bool(getattr(cache, "normalize", False)),
    }


def decide_qsa_indexed_admission(
    selection, *, length: int, training: bool, layout_ok: bool, cache=None
) -> tuple[bool, str]:
    """Resolve the indexed route without evaluating arrays or changing state."""
    mode = _qsa_indexed_mode()
    if mode == "off":
        return (False, "disabled")
    if training:
        return (False, "training")
    if selection.kind != "explicit":
        return (False, "selection_not_explicit")
    topk = _selection_topk_width(selection)
    if topk and int(selection.n_blocks) <= topk:
        return (False, "dense_by_construction")
    length = int(length)
    context = int(selection.physical_width)
    if mode == "auto":
        if length < 1 or length > _MAX_QUERY:
            return (False, "width_out_of_range")
        threshold = _AUTO_MIN_CONTEXT_M1 if length == 1 else _AUTO_MIN_CONTEXT_M3
        if context < threshold:
            return (False, "auto_context_out_of_range")
        if _MAX_CONTEXT and context > _MAX_CONTEXT:
            return (False, "context_out_of_range")
    else:
        if length < _MIN_QUERY or length > _MAX_QUERY:
            return (False, "width_out_of_range")
        if context < _MIN_CONTEXT or (_MAX_CONTEXT and context > _MAX_CONTEXT):
            return (False, "context_out_of_range")
    if not layout_ok:
        return (False, "unsupported_layout")
    quantized = qsa_indexed_quantized_cache_config(cache)
    if quantized is not None:
        if quantized["group_size"] not in _QUANTIZED_GROUP_SIZES:
            return (False, "quantized_group_size_unsupported")
        if (
            quantized["key_bits"] not in _QUANTIZED_BITS
            or quantized["value_bits"] not in _QUANTIZED_BITS
        ):
            return (False, "quantized_bits_unsupported")
        if quantized["rotate"] or quantized["normalize"]:
            return (False, "quantized_transform_unsupported")
    if not indexed_kernel_available():
        return (False, "kernel_unavailable")
    return (True, "engaged")


_STATUS_LOCK = threading.Lock()
_STATUS_COUNTS = Counter()
_STATUS_WIDTHS = {"1": Counter(), "2-8": Counter(), "9-17": Counter(), ">17": Counter()}
_STATUS_LAST = None
_STATUS_CANDIDATE = None
_STATUS_GEOMETRIES = {}
_STATUS_FALLBACKS = 0
_STATUS_DEVICE_PENDING = None
_STATUS_DEVICE_PENDING_EXPECTED = 0
_STATUS_DEVICE_PENDING_WIDTHS = Counter()
_STATUS_DEVICE_PENDING_LAST = None
_STATUS_DEVICE_EXPECTED = 0
_STATUS_DEVICE_OBSERVED = 0
_STATUS_DEVICE_MISMATCHES = 0


def _timing_key(key) -> str:
    """Render a probe-ladder key as "<splits>x<heads per threadgroup>"."""
    if isinstance(key, tuple):
        return "x".join((str(int(part)) for part in key))
    return str(int(key))


def _width_bucket(width: int) -> str:
    if width == 1:
        return "1"
    if width <= 8:
        return "2-8"
    if width <= 17:
        return "9-17"
    return ">17"


def record_qsa_indexed_receipt(
    *,
    engaged: bool,
    reason: str,
    length: int,
    context: int,
    splits: int | None = None,
    hpt: int | None = None,
    candidate: tuple[int, int, int] | None = None,
    exception_class: str | None = None,
    geometry_key: str | None = None,
    candidate_timings_ms: dict | None = None,
) -> None:
    """Record bounded process evidence without evaluating device arrays."""
    _ht_t0 = 0.0
    global _STATUS_CANDIDATE, _STATUS_FALLBACKS, _STATUS_LAST
    outcome = "engaged" if engaged else "declined"
    receipt = {
        "engaged": bool(engaged),
        "reason": str(reason),
        "query_width": int(length),
        "physical_kv": int(context),
        "splits": None if splits is None else int(splits),
        "heads_per_threadgroup": None if hpt is None else int(hpt),
        "candidate": None if candidate is None else list(candidate),
        "exception_class": exception_class,
        "geometry_key": geometry_key,
        "candidate_timings_ms": None
        if candidate_timings_ms is None
        else {
            _timing_key(entry): float(elapsed)
            for (entry, elapsed) in candidate_timings_ms.items()
        },
        "fully_masked_output": "zero",
    }
    with _STATUS_LOCK:
        _STATUS_COUNTS[reason] += 1
        _STATUS_WIDTHS[_width_bucket(int(length))][outcome] += 1
        if reason in {
            "mlx_build_unverified",
            "sdpa_header_mismatch",
            "sdpa_header_missing",
            "probe_declined",
            "dispatch_raised",
            "quantized_probe_declined",
            "quantized_dispatch_raised",
        }:
            _STATUS_FALLBACKS += 1
        if candidate is not None:
            _STATUS_CANDIDATE = tuple(candidate)
            if geometry_key is not None:
                _STATUS_GEOMETRIES[geometry_key] = {
                    "candidate": list(candidate),
                    "candidate_timings_ms": receipt["candidate_timings_ms"],
                }
        _STATUS_LAST = receipt


def _device_attest_output(
    output,
    counter,
    *,
    length: int,
    context: int,
    splits: int,
    hpt: int,
    candidate: tuple[int, int, int],
    geometry_key: str,
    candidate_timings_ms: dict,
    reason: str = "engaged",
):
    """Attach one device counter and defer host credit until status readback."""
    global _STATUS_CANDIDATE, _STATUS_DEVICE_PENDING
    global _STATUS_DEVICE_PENDING_EXPECTED, _STATUS_DEVICE_PENDING_LAST
    timings = {
        _timing_key(entry): float(elapsed)
        for (entry, elapsed) in candidate_timings_ms.items()
    }
    receipt = {
        "engaged": True,
        "reason": str(reason),
        "query_width": int(length),
        "physical_kv": int(context),
        "splits": int(splits),
        "heads_per_threadgroup": int(hpt),
        "candidate": list(candidate),
        "exception_class": None,
        "geometry_key": geometry_key,
        "candidate_timings_ms": timings,
        "device_attested": False,
        "fully_masked_output": "zero",
    }
    with _STATUS_LOCK:
        _STATUS_DEVICE_PENDING = (
            counter
            if _STATUS_DEVICE_PENDING is None
            else _STATUS_DEVICE_PENDING + counter
        )
        _STATUS_DEVICE_PENDING_EXPECTED += 1
        _STATUS_DEVICE_PENDING_WIDTHS[_width_bucket(int(length))] += 1
        _STATUS_DEVICE_PENDING_LAST = receipt
        _STATUS_CANDIDATE = tuple(candidate)
        _STATUS_GEOMETRIES[geometry_key] = {
            "candidate": list(candidate),
            "candidate_timings_ms": timings,
        }
        dependency = _STATUS_DEVICE_PENDING
    return mx.depends(output, dependency)


def _reconcile_device_receipts_locked() -> None:
    global _STATUS_DEVICE_PENDING, _STATUS_DEVICE_PENDING_EXPECTED
    global _STATUS_DEVICE_PENDING_LAST, _STATUS_DEVICE_EXPECTED
    global _STATUS_DEVICE_OBSERVED, _STATUS_DEVICE_MISMATCHES, _STATUS_LAST
    if _STATUS_DEVICE_PENDING is None:
        return
    observed = int(_STATUS_DEVICE_PENDING.item())
    expected = int(_STATUS_DEVICE_PENDING_EXPECTED)
    _STATUS_DEVICE_EXPECTED += expected
    _STATUS_DEVICE_OBSERVED += observed
    receipt = dict(_STATUS_DEVICE_PENDING_LAST)
    receipt["device_attested"] = observed == expected
    receipt["device_counter_observed"] = observed
    receipt["device_counter_expected"] = expected
    if observed == expected:
        _STATUS_COUNTS[receipt["reason"]] += observed
        for bucket, count in _STATUS_DEVICE_PENDING_WIDTHS.items():
            _STATUS_WIDTHS[bucket]["engaged"] += count
    else:
        _STATUS_COUNTS["device_attestation_mismatch"] += 1
        _STATUS_DEVICE_MISMATCHES += 1
        receipt["engaged"] = False
        receipt["reason"] = "device_attestation_mismatch"
    _STATUS_LAST = receipt
    _STATUS_DEVICE_PENDING = None
    _STATUS_DEVICE_PENDING_EXPECTED = 0
    _STATUS_DEVICE_PENDING_WIDTHS.clear()
    _STATUS_DEVICE_PENDING_LAST = None


def qsa_indexed_status(*, reset: bool = False) -> dict[str, Any]:
    """Return indexed-QSA admission, candidate, and fallback evidence."""
    global _STATUS_CANDIDATE, _STATUS_FALLBACKS, _STATUS_LAST
    global _STATUS_DEVICE_EXPECTED, _STATUS_DEVICE_OBSERVED
    global _STATUS_DEVICE_MISMATCHES
    (header_verified, header_sha, _) = _sdpa_header_state()
    with _STATUS_LOCK:
        _reconcile_device_receipts_locked()
        report = {
            "enabled": _qsa_indexed_mode() != "off",
            "mode": _qsa_indexed_mode(),
            "min_query_width": _MIN_QUERY,
            "max_query_width": _MAX_QUERY,
            "min_context": _MIN_CONTEXT,
            "max_context": _MAX_CONTEXT,
            "auto_min_context_m3": _AUTO_MIN_CONTEXT_M3,
            "auto_min_context_m1": _AUTO_MIN_CONTEXT_M1,
            "splits_override": _SPLITS_OVERRIDE,
            "split_candidates": list(_SPLIT_CANDIDATES),
            "hpt_override": _HPT_OVERRIDE,
            "hpt_candidates": list(_HPT_CANDIDATES),
            "hpt_ladder_enabled": _HPT_CANDIDATES != (12,),
            "mlx_version": str(getattr(mx, "__version__", "unknown")),
            "mlx_build_hash": _mlx_build_hash(),
            "mlx_build_verified": str(getattr(mx, "__version__", "unknown"))
            in _EXACT_MLX_BUILDS,
            "mlx_build_allow_unverified": os.environ.get(
                "MLX_QWEN4_QSA_INDEXED_ALLOW_UNVERIFIED_MLX"
            )
            == "1",
            "sdpa_header_verified": header_verified,
            "sdpa_header_sha256_prefix": None
            if header_sha is None
            else header_sha[:12],
            "counts": dict(_STATUS_COUNTS),
            "query_width_counts": {
                key: dict(value) for (key, value) in _STATUS_WIDTHS.items()
            },
            "candidate": None if _STATUS_CANDIDATE is None else list(_STATUS_CANDIDATE),
            "geometry_candidates": dict(_STATUS_GEOMETRIES),
            "fallbacks": _STATUS_FALLBACKS,
            "device_attestation": {
                "expected": _STATUS_DEVICE_EXPECTED,
                "observed": _STATUS_DEVICE_OBSERVED,
                "mismatches": _STATUS_DEVICE_MISMATCHES,
                "pending": _STATUS_DEVICE_PENDING_EXPECTED,
            },
            "fused_merge": fused_merge_status(reset=reset),
            "fully_masked_output": "zero",
            "last_decision": _STATUS_LAST,
        }
        if reset:
            _STATUS_COUNTS.clear()
            for value in _STATUS_WIDTHS.values():
                value.clear()
            _STATUS_CANDIDATE = None
            _STATUS_GEOMETRIES.clear()
            _STATUS_FALLBACKS = 0
            _STATUS_LAST = None
            _STATUS_DEVICE_EXPECTED = 0
            _STATUS_DEVICE_OBSERVED = 0
            _STATUS_DEVICE_MISMATCHES = 0
    return report


def qsa_indexed_enabled() -> bool:
    """Return the live indexed-QSA switch."""
    return _qsa_indexed_mode() != "off"


def _validate_no_duplicate_blocks(ids, counts) -> None:
    """Reject a malformed compact producer before it can double-count keys."""
    ids_np = np.asarray(ids)
    counts_np = np.asarray(counts)
    for index in np.ndindex(counts_np.shape):
        count = int(counts_np[index])
        row = ids_np[index][:count].tolist()
        if len(row) != len(set(row)):
            raise ValueError("compact QSA block ids must be unique per row")


def _reference_partials(q, k, v, compact, *, scale: float, splits: int):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or (k.shape != v.shape):
        raise ValueError("indexed QSA wants matching rank-4 q/k/v tensors")
    (batch, nqh, length, dim) = map(int, q.shape)
    if k.shape[0] != batch or int(k.shape[2]) != int(compact.physical_width):
        raise ValueError("indexed QSA tensors do not match compact selection")
    nkh = int(k.shape[1])
    if nkh < 1 or nqh % nkh:
        raise ValueError("indexed QSA requires integral GQA")
    (ids, counts, _, u_width, _, _, _, physical, valid) = compact_token_validity(
        compact
    )
    _validate_no_duplicate_blocks(ids, counts)
    if splits < 1 or splits > u_width:
        raise ValueError("splits must be in [1, u_width]")
    token_width = u_width * int(compact.block_size)
    steps = math.ceil(token_width / _SDPA_BLOCKS)
    padded_width = steps * _SDPA_BLOCKS
    padding = padded_width - token_width
    physical = physical.reshape(batch, length, token_width)
    valid = valid.reshape(batch, length, token_width)
    if padding:
        physical = mx.pad(physical, [(0, 0), (0, 0), (0, padding)])
        valid = mx.pad(valid, [(0, 0), (0, 0), (0, padding)], constant_values=False)
    physical = physical.reshape(batch, length, steps, _SDPA_BLOCKS).transpose(
        0, 1, 3, 2
    )
    valid = valid.reshape(batch, length, steps, _SDPA_BLOCKS).transpose(0, 1, 3, 2)
    q_rows = q.transpose(0, 2, 1, 3).astype(mx.float32) * float(scale)
    k_by_token = k.transpose(0, 2, 1, 3)
    v_by_token = v.transpose(0, 2, 1, 3)
    gqa = nqh // nkh
    head_map = mx.arange(nqh, dtype=mx.int32) // gqa
    gather_index = physical[..., None, None]
    gathered_k = mx.take_along_axis(k_by_token[:, None, None], gather_index, axis=3)
    gathered_v = mx.take_along_axis(v_by_token[:, None, None], gather_index, axis=3)
    gathered_k = mx.take(gathered_k, head_map, axis=4).transpose(0, 1, 4, 2, 3, 5)
    gathered_v = mx.take(gathered_v, head_map, axis=4).transpose(0, 1, 4, 2, 3, 5)
    scores = mx.sum(q_rows[..., None, None, :] * gathered_k.astype(mx.float32), axis=-1)
    head_valid = valid[:, :, None]
    scores = mx.where(head_valid, scores, -mx.inf)
    part_m = mx.max(scores, axis=-1)
    live = mx.isfinite(part_m)
    safe_m = mx.where(live, part_m, mx.zeros_like(part_m))
    probabilities = mx.where(
        head_valid, mx.exp(scores - safe_m[..., None]), mx.zeros_like(scores)
    )
    part_l = mx.sum(probabilities, axis=-1)
    part_o = mx.sum(
        probabilities[..., None] * gathered_v.astype(mx.float32), axis=-2
    ).astype(q.dtype)
    return (
        part_m.transpose(0, 2, 1, 3),
        part_l.transpose(0, 2, 1, 3),
        part_o.transpose(0, 2, 1, 3, 4),
    )


def _combine_reference_sdpa_partials(m, l, o, *, output_dtype):
    return mlx_sequential_merge(m, l, o, output_dtype=output_dtype)


def qwen4_qsa_indexed_reference(q, k, v, compact, *, scale: float, splits: int):
    """MLX-ops mirror of fixed-chunk two-pass indexed attention."""
    (m, l, o) = _reference_partials(q, k, v, compact, scale=scale, splits=int(splits))
    return _combine_reference_sdpa_partials(m, l, o, output_dtype=q.dtype)


def dequantize_qsa_quantized_kv(
    q_keys, q_values, *, group_size: int, key_bits: int, value_bits: int
):
    """Match the full-cache affine dequantization boundary used by MLX."""
    keys = mx.dequantize(*q_keys, group_size=int(group_size), bits=int(key_bits))
    values = mx.dequantize(*q_values, group_size=int(group_size), bits=int(value_bits))
    return (keys, values)


_HEADER = (
    "\n#include <metal_stdlib>\n#include <metal_simdgroup>\nusing namespace metal;\n"
)
_SOURCE = "\n    // Match MLX sdpa_vector_2pass_1 on the compact token order. Splits only\n    // distribute the fixed 128 blocks; every block keeps its global index.\n    const uint lane = thread_index_in_simdgroup;\n    const uint row = threadgroup_position_in_grid.y;\n    const uint unit = threadgroup_position_in_grid.z;\n    const uint slices = GQA / HPT;\n    const uint hslice = unit % slices;\n    const uint rest = unit / slices;\n    const uint head = hslice * HPT + simdgroup_index_in_threadgroup;\n    const uint split = rest % S;\n    const uint bkv = rest / S;\n    const uint b = bkv / NKVH;\n    const uint hkv = bkv % NKVH;\n\n    const int L = dims[0];\n    const int TOT = dims[1];\n    const int U = dims[2];\n    const uint count = counts[b * L + row];\n    const uint selected = n_sel[b * L + row];\n    const int qp = qpos[b * L + row];\n    const int complete = ((qp + 1) / BS) * BS;\n    const int lpad = left_pad[b];\n    const uint base = BLOCKS / S;\n    const uint remainder = BLOCKS % S;\n    const uint block_begin = split * base + metal::min(split, remainder);\n    const uint block_count = base + (split < remainder ? 1u : 0u);\n    const uint token_width = uint(U) * BS;\n    const uint qh = hkv * GQA + head;\n    const uint elements = D / 32;\n\n    float q_values[D / 32];\n    for (uint part = 0; part < elements; ++part) {\n        const uint d = lane * elements + part;\n        const size_t q_index =\n            (size_t)b * q_strides[0] +\n            (size_t)qh * q_strides[1] +\n            (size_t)row * q_strides[2] +\n            (size_t)d * q_strides[3];\n        q_values[part] = float(scale[0]) * float(q[q_index]);\n    }\n\n    const uint slot_base = (b * L + row) * uint(U);\n    const size_t k_head =\n        (size_t)b * k_strides[0] + (size_t)hkv * k_strides[1];\n    const size_t v_head =\n        (size_t)b * v_strides[0] + (size_t)hkv * v_strides[1];\n    for (uint local_block = 0; local_block < block_count; ++local_block) {\n        const uint block_idx = block_begin + local_block;\n        float out_values[D / 32] = {0};\n        float maximum = -3.402823466e+38F;\n        float sum = 0.0f;\n\n        for (uint token = block_idx; token < token_width; token += BLOCKS) {\n            const uint slot = token / BS;\n            const uint tail = token % BS;\n            int logical = 0;\n            int physical = 0;\n            bool live = slot < count;\n            if (live) {\n                const int block = int(ids[slot_base + slot]);\n                logical = block * BS + int(tail);\n                physical = lpad + logical;\n                live = physical >= 0 && physical < TOT && logical <= qp;\n                live = live && (slot < selected || logical >= complete);\n                if (HAS_MASK && live) {\n                    const uint mask_batch = mask_shape[0] == 1 ? 0 : b;\n                    const size_t mask_index =\n                        (size_t)mask_batch * mask_strides[0] +\n                        (size_t)row * mask_strides[2] +\n                        (size_t)physical * mask_strides[3];\n                    live = mask[mask_index];\n                }\n            }\n            if (!live) continue;\n\n            float score = 0.0f;\n            for (uint part = 0; part < elements; ++part) {\n                const uint d = lane * elements + part;\n                const size_t k_index =\n                    k_head + (size_t)physical * k_strides[2] +\n                    (size_t)d * k_strides[3];\n                score += q_values[part] * float(k[k_index]);\n            }\n            score = simd_sum(score);\n            const float new_max = metal::max(maximum, score);\n            const float factor = fast::exp(maximum - new_max);\n            const float probability = fast::exp(score - new_max);\n            maximum = new_max;\n            sum = sum * factor + probability;\n            for (uint part = 0; part < elements; ++part) {\n                const uint d = lane * elements + part;\n                out_values[part] = out_values[part] * factor\n                    + probability * float(\n                        v[v_head + (size_t)physical * v_strides[2] +\n                          (size_t)d * v_strides[3]]\n                    );\n            }\n        }\n\n        const size_t state = (\n            ((size_t)(b * NQH + qh) * L + row) * BLOCKS + block_idx\n        );\n        if (lane == 0) {\n            part_m[state] = maximum;\n            part_l[state] = sum;\n        }\n        for (uint part = 0; part < elements; ++part) {\n            const uint d = lane * elements + part;\n            part_o[state * D + d] = T(out_values[part]);\n        }\n    }\n    if (unit == 0 && row == 0 && head == 0 && lane == 0)\n        engaged[0] = 1;\n"
_PRIVATE_DELTA_SOURCE = "\n    // The logical cache is [one immutable B1 base, one private suffix per\n    // row]. Keep the shipped indexed split-K traversal and reduction order;\n    // sorted ids let one accumulator traverse a straight-line base loop and\n    // then a straight-line delta loop without a source branch in either dot.\n    const uint lane = thread_index_in_simdgroup;\n    const uint row = threadgroup_position_in_grid.y;\n    const uint unit = threadgroup_position_in_grid.z;\n    const uint slices = GQA / HPT;\n    const uint hslice = unit % slices;\n    const uint rest = unit / slices;\n    const uint head = hslice * HPT + simdgroup_index_in_threadgroup;\n    const uint split = rest % S;\n    const uint bkv = rest / S;\n    const uint b = bkv / NKVH;\n    const uint hkv = bkv % NKVH;\n\n    const int L = dims[0];\n    const int TOT = dims[1];\n    const int U = dims[2];\n    const int BASE = dims[3];\n    const uint count = counts[b * L + row];\n    const uint base_slots = base_counts[b * L + row];\n    const uint selected = n_sel[b * L + row];\n    const int qp = qpos[b * L + row];\n    const int complete = ((qp + 1) / BS) * BS;\n    const int delta_length = int(delta_lengths[b]);\n    const uint base = BLOCKS / S;\n    const uint remainder = BLOCKS % S;\n    const uint block_begin = split * base + metal::min(split, remainder);\n    const uint block_count = base + (split < remainder ? 1u : 0u);\n    const uint token_width = uint(U) * BS;\n    const uint qh = hkv * GQA + head;\n    const uint elements = D / 32;\n\n    float q_values[D / 32];\n    for (uint part = 0; part < elements; ++part) {\n        const uint d = lane * elements + part;\n        const size_t q_index =\n            (size_t)b * q_strides[0] +\n            (size_t)qh * q_strides[1] +\n            (size_t)row * q_strides[2] +\n            (size_t)d * q_strides[3];\n        q_values[part] = float(scale[0]) * float(q[q_index]);\n    }\n\n    const uint slot_base = (b * L + row) * uint(U);\n    const size_t base_k_head = (size_t)hkv * base_k_strides[1];\n    const size_t base_v_head = (size_t)hkv * base_v_strides[1];\n    const size_t delta_k_head =\n        (size_t)b * delta_k_strides[0] +\n        (size_t)hkv * delta_k_strides[1];\n    const size_t delta_v_head =\n        (size_t)b * delta_v_strides[0] +\n        (size_t)hkv * delta_v_strides[1];\n    for (uint local_block = 0; local_block < block_count; ++local_block) {\n        const uint block_idx = block_begin + local_block;\n        const uint base_token_stop = metal::min(base_slots * BS, token_width);\n        float out_values[D / 32] = {0};\n        float maximum = -3.402823466e+38F;\n        float sum = 0.0f;\n\n        // Compact block ids are sorted, so every immutable-base slot precedes\n        // every private-delta slot. Keep the original token order but make\n        // each source loop straight-line: no K/V pointer choice in the dot.\n        for (uint token = block_idx; token < base_token_stop; token += BLOCKS) {\n            const uint slot = token / BS;\n            const uint tail = token % BS;\n            int logical = 0;\n            bool live = slot < count;\n            if (live) {\n                const int block = int(ids[slot_base + slot]);\n                logical = block * BS + int(tail);\n                live = logical >= 0 && logical < BASE && logical <= qp;\n                live = live && (slot < selected || logical >= complete);\n                if (HAS_MASK && live) {\n                    const uint mask_batch = mask_shape[0] == 1 ? 0 : b;\n                    const size_t mask_index =\n                        (size_t)mask_batch * mask_strides[0] +\n                        (size_t)row * mask_strides[2] +\n                        (size_t)logical * mask_strides[3];\n                    live = mask[mask_index];\n                }\n            }\n            if (!live) continue;\n\n            float score = 0.0f;\n            for (uint part = 0; part < elements; ++part) {\n                const uint d = lane * elements + part;\n                const size_t k_index =\n                    base_k_head + (size_t)logical * base_k_strides[2]\n                    + (size_t)d * base_k_strides[3];\n                score += q_values[part] * float(base_k[k_index]);\n            }\n            score = simd_sum(score);\n            const float new_max = metal::max(maximum, score);\n            const float factor = fast::exp(maximum - new_max);\n            const float probability = fast::exp(score - new_max);\n            maximum = new_max;\n            sum = sum * factor + probability;\n            for (uint part = 0; part < elements; ++part) {\n                const uint d = lane * elements + part;\n                const size_t v_index =\n                    base_v_head + (size_t)logical * base_v_strides[2]\n                    + (size_t)d * base_v_strides[3];\n                out_values[part] = out_values[part] * factor\n                    + probability * float(base_v[v_index]);\n            }\n        }\n\n        uint delta_begin = block_idx;\n        if (delta_begin < base_token_stop) {\n            delta_begin +=\n                ((base_token_stop - delta_begin + BLOCKS - 1) / BLOCKS)\n                * BLOCKS;\n        }\n        for (uint token = delta_begin; token < token_width; token += BLOCKS) {\n            const uint slot = token / BS;\n            const uint tail = token % BS;\n            int logical = 0;\n            int position = 0;\n            bool live = slot < count;\n            if (live) {\n                const int block = int(ids[slot_base + slot]);\n                logical = block * BS + int(tail);\n                position = logical - BASE;\n                live = logical >= BASE && logical < TOT && logical <= qp;\n                live = live && position >= 0 && position < delta_length;\n                live = live && (slot < selected || logical >= complete);\n                if (HAS_MASK && live) {\n                    const uint mask_batch = mask_shape[0] == 1 ? 0 : b;\n                    const size_t mask_index =\n                        (size_t)mask_batch * mask_strides[0] +\n                        (size_t)row * mask_strides[2] +\n                        (size_t)logical * mask_strides[3];\n                    live = mask[mask_index];\n                }\n            }\n            if (!live) continue;\n\n            float score = 0.0f;\n            for (uint part = 0; part < elements; ++part) {\n                const uint d = lane * elements + part;\n                const size_t k_index =\n                    delta_k_head + (size_t)position * delta_k_strides[2]\n                    + (size_t)d * delta_k_strides[3];\n                score += q_values[part] * float(delta_k[k_index]);\n            }\n            score = simd_sum(score);\n            const float new_max = metal::max(maximum, score);\n            const float factor = fast::exp(maximum - new_max);\n            const float probability = fast::exp(score - new_max);\n            maximum = new_max;\n            sum = sum * factor + probability;\n            for (uint part = 0; part < elements; ++part) {\n                const uint d = lane * elements + part;\n                const size_t v_index =\n                    delta_v_head + (size_t)position * delta_v_strides[2]\n                    + (size_t)d * delta_v_strides[3];\n                out_values[part] = out_values[part] * factor\n                    + probability * float(delta_v[v_index]);\n            }\n        }\n\n        const size_t state = (\n            ((size_t)(b * NQH + qh) * L + row) * BLOCKS + block_idx\n        );\n        if (lane == 0) {\n            part_m[state] = maximum;\n            part_l[state] = sum;\n        }\n        for (uint part = 0; part < elements; ++part) {\n            const uint d = lane * elements + part;\n            part_o[state * D + d] = T(out_values[part]);\n        }\n    }\n    if (unit == 0 && row == 0 && head == 0 && lane == 0)\n        engaged[0] = 1;\n"
_PRIVATE_DELTA_EXACT_SET_SOURCE = "\n    // Experimental B2 exact-set fold. One threadgroup owns both batch rows\n    // for one query position, KV head, split and head slice. When the ordered\n    // base-page ids match, each lane loads its K/V element once and updates\n    // two independent online-softmax accumulators. Row-local Q, masks,\n    // positions and private suffixes remain independent.\n    const uint lane = thread_index_in_simdgroup;\n    const uint row = threadgroup_position_in_grid.y;\n    const uint unit = threadgroup_position_in_grid.z;\n    const uint slices = GQA / HPT;\n    const uint hslice = unit % slices;\n    const uint rest = unit / slices;\n    const uint head = hslice * HPT + simdgroup_index_in_threadgroup;\n    const uint split = rest % S;\n    const uint hkv = rest / S;\n\n    const int L = dims[0];\n    const int TOT = dims[1];\n    const int U = dims[2];\n    const int BASE = dims[3];\n    const uint split_base = BLOCKS / S;\n    const uint remainder = BLOCKS % S;\n    const uint block_begin =\n        split * split_base + metal::min(split, remainder);\n    const uint block_count = split_base + (split < remainder ? 1u : 0u);\n    const uint token_width = uint(U) * BS;\n    const uint qh = hkv * GQA + head;\n    const uint elements = D / 32;\n    const size_t base_k_head = (size_t)hkv * base_k_strides[1];\n    const size_t base_v_head = (size_t)hkv * base_v_strides[1];\n    const bool fold_query = exact_set[row];\n\n    float q_values[2][D / 32];\n    for (uint b = 0; b < 2; ++b) {\n        for (uint part = 0; part < elements; ++part) {\n            const uint d = lane * elements + part;\n            const size_t q_index =\n                (size_t)b * q_strides[0] +\n                (size_t)qh * q_strides[1] +\n                (size_t)row * q_strides[2] +\n                (size_t)d * q_strides[3];\n            q_values[b][part] = float(scale[0]) * float(q[q_index]);\n        }\n    }\n\n    const uint count[2] = {counts[row], counts[L + row]};\n    const uint base_slots[2] = {\n        base_counts[row], base_counts[L + row]\n    };\n    const uint selected[2] = {n_sel[row], n_sel[L + row]};\n    const int qp[2] = {qpos[row], qpos[L + row]};\n    const int complete[2] = {\n        ((qp[0] + 1) / BS) * BS,\n        ((qp[1] + 1) / BS) * BS\n    };\n    const int delta_length[2] = {\n        int(delta_lengths[0]), int(delta_lengths[1])\n    };\n    const uint max_base_slots = metal::max(base_slots[0], base_slots[1]);\n    const uint base_token_stop =\n        metal::min(max_base_slots * BS, token_width);\n\n    for (uint local_block = 0; local_block < block_count; ++local_block) {\n        const uint block_idx = block_begin + local_block;\n        float out_values[2][D / 32] = {{0}};\n        float maximum[2] = {\n            -3.402823466e+38F, -3.402823466e+38F\n        };\n        float sum[2] = {0.0f, 0.0f};\n\n        for (uint token = block_idx; token < base_token_stop;\n             token += BLOCKS) {\n            const uint slot = token / BS;\n            const uint tail = token % BS;\n            int logical[2] = {0, 0};\n            bool live[2] = {false, false};\n            for (uint b = 0; b < 2; ++b) {\n                live[b] = slot < count[b] && slot < base_slots[b];\n                if (live[b]) {\n                    const uint slot_base = (b * L + row) * uint(U);\n                    const int block = int(ids[slot_base + slot]);\n                    logical[b] = block * BS + int(tail);\n                    live[b] = logical[b] >= 0 && logical[b] < BASE;\n                    live[b] = live[b] && logical[b] <= qp[b];\n                    live[b] = live[b]\n                        && (slot < selected[b] || logical[b] >= complete[b]);\n                    if (HAS_MASK && live[b]) {\n                        const uint mask_batch = mask_shape[0] == 1 ? 0 : b;\n                        const size_t mask_index =\n                            (size_t)mask_batch * mask_strides[0] +\n                            (size_t)row * mask_strides[2] +\n                            (size_t)logical[b] * mask_strides[3];\n                        live[b] = mask[mask_index];\n                    }\n                }\n            }\n            if (!live[0] && !live[1]) continue;\n\n            if (fold_query && logical[0] == logical[1]) {\n                float score[2] = {0.0f, 0.0f};\n                for (uint part = 0; part < elements; ++part) {\n                    const uint d = lane * elements + part;\n                    const size_t k_index =\n                        base_k_head +\n                        (size_t)logical[0] * base_k_strides[2] +\n                        (size_t)d * base_k_strides[3];\n                    const float key_value = float(base_k[k_index]);\n                    if (live[0]) score[0] += q_values[0][part] * key_value;\n                    if (live[1]) score[1] += q_values[1][part] * key_value;\n                }\n                if (live[0]) score[0] = simd_sum(score[0]);\n                if (live[1]) score[1] = simd_sum(score[1]);\n                float factor[2] = {0.0f, 0.0f};\n                float probability[2] = {0.0f, 0.0f};\n                for (uint b = 0; b < 2; ++b) {\n                    if (!live[b]) continue;\n                    const float new_max = metal::max(maximum[b], score[b]);\n                    factor[b] = fast::exp(maximum[b] - new_max);\n                    probability[b] = fast::exp(score[b] - new_max);\n                    maximum[b] = new_max;\n                    sum[b] = sum[b] * factor[b] + probability[b];\n                }\n                for (uint part = 0; part < elements; ++part) {\n                    const uint d = lane * elements + part;\n                    const size_t v_index =\n                        base_v_head +\n                        (size_t)logical[0] * base_v_strides[2] +\n                        (size_t)d * base_v_strides[3];\n                    const float value = float(base_v[v_index]);\n                    for (uint b = 0; b < 2; ++b) {\n                        if (live[b]) {\n                            out_values[b][part] =\n                                out_values[b][part] * factor[b]\n                                + probability[b] * value;\n                        }\n                    }\n                }\n            } else {\n                // A stale structural proof cannot corrupt the result: fall\n                // back to two row-local base reads for this token.\n                for (uint b = 0; b < 2; ++b) {\n                    if (!live[b]) continue;\n                    float score = 0.0f;\n                    for (uint part = 0; part < elements; ++part) {\n                        const uint d = lane * elements + part;\n                        const size_t k_index =\n                            base_k_head +\n                            (size_t)logical[b] * base_k_strides[2] +\n                            (size_t)d * base_k_strides[3];\n                        score += q_values[b][part] * float(base_k[k_index]);\n                    }\n                    score = simd_sum(score);\n                    const float new_max = metal::max(maximum[b], score);\n                    const float factor = fast::exp(maximum[b] - new_max);\n                    const float probability = fast::exp(score - new_max);\n                    maximum[b] = new_max;\n                    sum[b] = sum[b] * factor + probability;\n                    for (uint part = 0; part < elements; ++part) {\n                        const uint d = lane * elements + part;\n                        const size_t v_index =\n                            base_v_head +\n                            (size_t)logical[b] * base_v_strides[2] +\n                            (size_t)d * base_v_strides[3];\n                        out_values[b][part] = out_values[b][part] * factor\n                            + probability * float(base_v[v_index]);\n                    }\n                }\n            }\n        }\n\n        // Private suffixes remain row-local. Their token order and online\n        // softmax update order match the proven source-phased kernel exactly.\n        for (uint b = 0; b < 2; ++b) {\n            const uint slot_base = (b * L + row) * uint(U);\n            const size_t delta_k_head =\n                (size_t)b * delta_k_strides[0] +\n                (size_t)hkv * delta_k_strides[1];\n            const size_t delta_v_head =\n                (size_t)b * delta_v_strides[0] +\n                (size_t)hkv * delta_v_strides[1];\n            uint delta_begin = block_idx;\n            const uint row_base_stop =\n                metal::min(base_slots[b] * BS, token_width);\n            if (delta_begin < row_base_stop) {\n                delta_begin +=\n                    ((row_base_stop - delta_begin + BLOCKS - 1) / BLOCKS)\n                    * BLOCKS;\n            }\n            for (uint token = delta_begin; token < token_width;\n                 token += BLOCKS) {\n                const uint slot = token / BS;\n                const uint tail = token % BS;\n                int logical = 0;\n                int position = 0;\n                bool live = slot < count[b];\n                if (live) {\n                    const int block = int(ids[slot_base + slot]);\n                    logical = block * BS + int(tail);\n                    position = logical - BASE;\n                    live = logical >= BASE && logical < TOT;\n                    live = live && logical <= qp[b];\n                    live = live && position >= 0;\n                    live = live && position < delta_length[b];\n                    live = live\n                        && (slot < selected[b] || logical >= complete[b]);\n                    if (HAS_MASK && live) {\n                        const uint mask_batch = mask_shape[0] == 1 ? 0 : b;\n                        const size_t mask_index =\n                            (size_t)mask_batch * mask_strides[0] +\n                            (size_t)row * mask_strides[2] +\n                            (size_t)logical * mask_strides[3];\n                        live = mask[mask_index];\n                    }\n                }\n                if (!live) continue;\n\n                float score = 0.0f;\n                for (uint part = 0; part < elements; ++part) {\n                    const uint d = lane * elements + part;\n                    const size_t k_index =\n                        delta_k_head +\n                        (size_t)position * delta_k_strides[2] +\n                        (size_t)d * delta_k_strides[3];\n                    score += q_values[b][part] * float(delta_k[k_index]);\n                }\n                score = simd_sum(score);\n                const float new_max = metal::max(maximum[b], score);\n                const float factor = fast::exp(maximum[b] - new_max);\n                const float probability = fast::exp(score - new_max);\n                maximum[b] = new_max;\n                sum[b] = sum[b] * factor + probability;\n                for (uint part = 0; part < elements; ++part) {\n                    const uint d = lane * elements + part;\n                    const size_t v_index =\n                        delta_v_head +\n                        (size_t)position * delta_v_strides[2] +\n                        (size_t)d * delta_v_strides[3];\n                    out_values[b][part] = out_values[b][part] * factor\n                        + probability * float(delta_v[v_index]);\n                }\n            }\n        }\n\n        for (uint b = 0; b < 2; ++b) {\n            const size_t state =\n                ((size_t)(b * NQH + qh) * L + row) * BLOCKS + block_idx;\n            if (lane == 0) {\n                part_m[state] = maximum[b];\n                part_l[state] = sum[b];\n            }\n            for (uint part = 0; part < elements; ++part) {\n                const uint d = lane * elements + part;\n                part_o[state * D + d] = T(out_values[b][part]);\n            }\n        }\n    }\n    if (unit == 0 && row == 0 && head == 0 && lane == 0)\n        engaged[0] = 1;\n"
_QUANTIZED_SOURCE = "\n    // Keep the bf16 SDPA order while dequantizing only selected K/V values.\n    const uint lane = thread_index_in_simdgroup;\n    const uint row = threadgroup_position_in_grid.y;\n    const uint unit = threadgroup_position_in_grid.z;\n    const uint slices = GQA / HPT;\n    const uint hslice = unit % slices;\n    const uint rest = unit / slices;\n    const uint head = hslice * HPT + simdgroup_index_in_threadgroup;\n    const uint split = rest % S;\n    const uint bkv = rest / S;\n    const uint b = bkv / NKVH;\n    const uint hkv = bkv % NKVH;\n\n    const int L = dims[0];\n    const int TOT = dims[1];\n    const int U = dims[2];\n    const uint count = counts[b * L + row];\n    const uint selected = n_sel[b * L + row];\n    const int qp = qpos[b * L + row];\n    const int complete = ((qp + 1) / BS) * BS;\n    const int lpad = left_pad[b];\n    const uint base = BLOCKS / S;\n    const uint remainder = BLOCKS % S;\n    const uint block_begin = split * base + metal::min(split, remainder);\n    const uint block_count = base + (split < remainder ? 1u : 0u);\n    const uint token_width = uint(U) * BS;\n    const uint qh = hkv * GQA + head;\n    const uint elements = D / 32;\n    const uint groups = D / GROUP_SIZE;\n    const uint k_packed = D * KBITS / 32;\n    const uint v_packed = D * VBITS / 32;\n    const uint k_mask = (1u << KBITS) - 1u;\n    const uint v_mask = (1u << VBITS) - 1u;\n\n    float q_values[D / 32];\n    for (uint part = 0; part < elements; ++part) {\n        const uint d = lane * elements + part;\n        q_values[part] = float(scale[0]) * float(\n            q[((size_t)(b * NQH + qh) * L + row) * D + d]\n        );\n    }\n\n    const uint slot_base = (b * L + row) * uint(U);\n    for (uint local_block = 0; local_block < block_count; ++local_block) {\n        const uint block_idx = block_begin + local_block;\n        float out_values[D / 32] = {0};\n        float maximum = -3.402823466e+38F;\n        float sum = 0.0f;\n\n        for (uint token = block_idx; token < token_width; token += BLOCKS) {\n            const uint slot = token / BS;\n            const uint tail = token % BS;\n            int logical = 0;\n            int physical = 0;\n            bool live = slot < count;\n            if (live) {\n                const int block = int(ids[slot_base + slot]);\n                logical = block * BS + int(tail);\n                physical = lpad + logical;\n                live = physical >= 0 && physical < TOT && logical <= qp;\n                live = live && (slot < selected || logical >= complete);\n                if (HAS_MASK && live)\n                    live = mask[(size_t)(b * L + row) * TOT + physical];\n            }\n            if (!live) continue;\n\n            const size_t quant_row = ((size_t)b * NKVH + hkv) * TOT + physical;\n            float score = 0.0f;\n            for (uint part = 0; part < elements; ++part) {\n                const uint d = lane * elements + part;\n                const uint word = k_w[quant_row * k_packed + d * KBITS / 32];\n                const uint code = (word >> ((d * KBITS) & 31)) & k_mask;\n                const size_t affine = quant_row * groups + d / GROUP_SIZE;\n                const T value = k_s[affine] * code + k_b[affine];\n                score += q_values[part] * float(value);\n            }\n            score = simd_sum(score);\n            const float new_max = metal::max(maximum, score);\n            const float factor = fast::exp(maximum - new_max);\n            const float probability = fast::exp(score - new_max);\n            maximum = new_max;\n            sum = sum * factor + probability;\n            for (uint part = 0; part < elements; ++part) {\n                const uint d = lane * elements + part;\n                const uint word = v_w[quant_row * v_packed + d * VBITS / 32];\n                const uint code = (word >> ((d * VBITS) & 31)) & v_mask;\n                const size_t affine = quant_row * groups + d / GROUP_SIZE;\n                const T value = v_s[affine] * code + v_b[affine];\n                out_values[part] = out_values[part] * factor\n                    + probability * float(value);\n            }\n        }\n\n        const size_t state = (\n            ((size_t)(b * NQH + qh) * L + row) * BLOCKS + block_idx\n        );\n        if (lane == 0) {\n            part_m[state] = maximum;\n            part_l[state] = sum;\n        }\n        for (uint part = 0; part < elements; ++part) {\n            const uint d = lane * elements + part;\n            part_o[state * D + d] = T(out_values[part]);\n        }\n    }\n    if (unit == 0 && row == 0 && head == 0 && lane == 0)\n        engaged[0] = 1;\n"
_COMBINE_SOURCE = "\n    const uint lane = thread_index_in_simdgroup;\n    const uint sg = simdgroup_index_in_threadgroup;\n    const uint row = threadgroup_position_in_grid.y;\n    const uint bh = threadgroup_position_in_grid.z;\n    const int L = dims[0];\n    const uint elements = D / 32;\n    const size_t state = ((size_t)bh * L + row) * BLOCKS;\n    const device float* row_m = part_m + state;\n    const device float* row_l = part_l + state;\n    const device T* row_o = part_o + state * D;\n\n    float maximum = -3.402823466e+38F;\n    for (uint group = 0; group < BLOCKS / 32; ++group)\n        maximum = metal::max(maximum, row_m[lane + 32 * group]);\n    maximum = simd_max(maximum);\n\n    float sum = 0.0f;\n    for (uint group = 0; group < BLOCKS / 32; ++group) {\n        const uint block = lane + 32 * group;\n        sum += fast::exp(row_m[block] - maximum) * row_l[block];\n    }\n    sum = simd_sum(sum);\n\n    float values[D / 32] = {0};\n    for (uint group = 0; group < BLOCKS / 32; ++group) {\n        const uint block = sg + 32 * group;\n        const float factor = fast::exp(row_m[block] - maximum);\n        for (uint part = 0; part < elements; ++part) {\n            const uint d = lane * elements + part;\n            values[part] += factor * float(row_o[(size_t)block * D + d]);\n        }\n    }\n\n    threadgroup float transposed[32 * 32];\n    for (uint part = 0; part < elements; ++part) {\n        transposed[lane * 32 + sg] = values[part];\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n        values[part] = simd_sum(transposed[sg * 32 + lane]);\n        values[part] = sum == 0.0f ? values[part] : values[part] / sum;\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n    }\n\n    if (lane == 0) {\n        device T* row_out = out + ((size_t)bh * L + row) * D + sg * elements;\n        for (uint part = 0; part < elements; ++part)\n            row_out[part] = T(values[part]);\n    }\n"
_COMBINE_SOURCE_GATED = _COMBINE_SOURCE.replace(
    "        for (uint part = 0; part < elements; ++part)\n            row_out[part] = T(values[part]);",
    "        const uint b = bh / H;\n        const uint h = bh - b * H;\n        const device T* row_gate = output_gate\n            + (((size_t)b * L + row) * H + h) * D + sg * elements;\n        for (uint part = 0; part < elements; ++part) {\n            const T attention = T(values[part]);\n            const T gate_x = row_gate[part];\n            const T gate_y = T(1) / (T(1) + metal::exp(metal::abs(gate_x)));\n            T gate_sigmoid = gate_x < T(0) ? gate_y : T(1) - gate_y;\n            if constexpr (metal::is_same_v<T, bfloat>) {\n                if (gate_x == T(-6.84375f)) {\n                    gate_sigmoid = T(0.00106048583984375f);\n                }\n            }\n            row_out[part] = attention * gate_sigmoid;\n        }",
)


@lru_cache(maxsize=None)
def _partition_kernel():
    return mx.fast.metal_kernel(
        name="qwen4_qsa_indexed_sdpa_pass1_v4",
        input_names=[
            "q",
            "k",
            "v",
            "ids",
            "counts",
            "n_sel",
            "qpos",
            "left_pad",
            "mask",
            "scale",
            "dims",
        ],
        output_names=["part_m", "part_l", "part_o", "engaged"],
        header=_HEADER,
        source=_SOURCE,
        ensure_row_contiguous=False,
    )


@lru_cache(maxsize=None)
def _private_delta_partition_kernel():
    return mx.fast.metal_kernel(
        name="qwen4_qsa_indexed_private_delta_pass1_v2",
        input_names=[
            "q",
            "base_k",
            "base_v",
            "delta_k",
            "delta_v",
            "delta_lengths",
            "base_counts",
            "ids",
            "counts",
            "n_sel",
            "qpos",
            "mask",
            "scale",
            "dims",
        ],
        output_names=["part_m", "part_l", "part_o", "engaged"],
        header=_HEADER,
        source=_PRIVATE_DELTA_SOURCE,
        ensure_row_contiguous=False,
    )


@lru_cache(maxsize=None)
def _private_delta_exact_set_partition_kernel():
    return mx.fast.metal_kernel(
        name="qwen4_qsa_indexed_private_delta_exact_set_pass1_v1",
        input_names=[
            "q",
            "base_k",
            "base_v",
            "delta_k",
            "delta_v",
            "delta_lengths",
            "base_counts",
            "exact_set",
            "ids",
            "counts",
            "n_sel",
            "qpos",
            "mask",
            "scale",
            "dims",
        ],
        output_names=["part_m", "part_l", "part_o", "engaged"],
        header=_HEADER,
        source=_PRIVATE_DELTA_EXACT_SET_SOURCE,
        ensure_row_contiguous=False,
    )


@lru_cache(maxsize=None)
def _quantized_partition_kernel():
    return mx.fast.metal_kernel(
        name="qwen4_qsa_indexed_quantized_sdpa_pass1_v2",
        input_names=[
            "q",
            "k_w",
            "k_s",
            "k_b",
            "v_w",
            "v_s",
            "v_b",
            "ids",
            "counts",
            "n_sel",
            "qpos",
            "left_pad",
            "mask",
            "scale",
            "dims",
        ],
        output_names=["part_m", "part_l", "part_o", "engaged"],
        header=_HEADER,
        source=_QUANTIZED_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=None)
def _combine_kernel(gated: bool = False):
    return mx.fast.metal_kernel(
        name="qwen4_qsa_indexed_sdpa_pass2_gate_v1"
        if gated
        else "qwen4_qsa_indexed_sdpa_pass2_v3",
        input_names=["part_m", "part_l", "part_o", "dims", "output_gate"]
        if gated
        else ["part_m", "part_l", "part_o", "dims"],
        output_names=["out"],
        header=_HEADER,
        source=_COMBINE_SOURCE_GATED if gated else _COMBINE_SOURCE,
        ensure_row_contiguous=True,
    )


class QSAIndexedProbeDeclined(RuntimeError):
    """No candidate in the indexed Metal ladder could dispatch."""

    def __init__(self, message: str, *, reason: str = "probe_declined"):
        super().__init__(message)
        self.reason = reason


_PROBE_LOCK = threading.Lock()
_PROBE_RESULTS = {}
_PROBE_TIMINGS = {}
_QUANTIZED_PROBE_RESULTS = {}
_PRIVATE_DELTA_PROBE_RESULTS = {}
_PRIVATE_DELTA_EXACT_SET_PROBE_RESULTS = {}
_MISSING = object()
_TOPOLOGY_CAPTURE_LOCK = threading.Lock()
_TOPOLOGY_CAPTURE_COUNT = 0
_TOPOLOGY_CAPTURE_DIR = os.environ.get("MLX_QWEN4_QSA_TOPOLOGY_CAPTURE_DIR")
_TOPOLOGY_CAPTURE_LIMIT = (
    _env_int("MLX_QWEN4_QSA_TOPOLOGY_CAPTURE_COUNT", 12, minimum=1)
    if _TOPOLOGY_CAPTURE_DIR
    else 0
)


def _capture_private_delta_topology_diagnostic(
    compact,
    *,
    q,
    base_tokens: int,
    delta_width: int,
    delta_lengths,
    requested_splits: int | None,
    requested_hpt: int | None,
):
    """Host one compact selection only when the diagnostic env is set."""
    if not _TOPOLOGY_CAPTURE_DIR:
        return None
    global _TOPOLOGY_CAPTURE_COUNT
    with _TOPOLOGY_CAPTURE_LOCK:
        if _TOPOLOGY_CAPTURE_COUNT >= _TOPOLOGY_CAPTURE_LIMIT:
            return None
        capture_index = _TOPOLOGY_CAPTURE_COUNT
        _TOPOLOGY_CAPTURE_COUNT += 1
    mx.eval(compact.block_ids, compact.block_counts, delta_lengths)
    from ..qsa_topology_receipt import write_qsa_topology_diagnostic

    filename = f"qsa-topology-p{os.getpid()}-{time.time_ns()}-{capture_index:04d}.json"
    return write_qsa_topology_diagnostic(
        Path(_TOPOLOGY_CAPTURE_DIR).expanduser() / filename,
        compact,
        base_tokens=int(base_tokens),
        metadata={
            "capture_index": capture_index,
            "q_shape": list(map(int, q.shape)),
            "q_dtype": str(q.dtype),
            "base_tokens": int(base_tokens),
            "delta_width": int(delta_width),
            "delta_lengths": np.asarray(delta_lengths).astype(np.int64).tolist(),
            "physical_width": int(compact.physical_width),
            "selected_block_capacity": int(compact.block_ids.shape[-1]),
            "block_size": int(compact.block_size),
            "causal_mask_present": compact.causal_mask is not None,
            "requested_splits": requested_splits,
            "requested_hpt": requested_hpt,
        },
    )


def _candidate_ladder(splits: int | None, gqa: int, hpt: int | None = None):
    """Return the (threads, splits, hpt) grid the first use times."""
    split_candidates = _SPLIT_CANDIDATES if splits is None else (int(splits),)
    if hpt is None:
        hpt_candidates = tuple(
            (value for value in _HPT_CANDIDATES if int(gqa) % value == 0)
        )
    else:
        hpt_candidates = (validate_hpt(hpt, gqa),)
    return tuple(
        (
            (heads * 32, value, heads)
            for value in split_candidates
            for heads in hpt_candidates
        )
    )


def _measure_candidates(candidates, dispatch):
    """Compile viable candidates, then time one real dispatch for each."""
    viable = []
    for candidate in candidates:
        try:
            (output, _) = dispatch(candidate)
            mx.eval(output)
            viable.append(candidate)
        except RuntimeError:
            continue
    timings = {}
    outputs = {}
    for candidate in viable:
        try:
            started = time.perf_counter_ns()
            (output, counter) = dispatch(candidate)
            mx.eval(output)
            elapsed = time.perf_counter_ns() - started
        except RuntimeError:
            continue
        timings[candidate[1], candidate[2]] = elapsed / 1000000.0
        outputs[candidate] = (output, counter)
    if not timings:
        return (None, None, None, {})
    selected = min(
        outputs,
        key=lambda candidate: (
            timings[candidate[1], candidate[2]],
            -candidate[1],
            -candidate[2],
        ),
    )
    (output, counter) = outputs[selected]
    return (selected, output, counter, timings)


def _partition_dispatch(
    q, k, v, compact, *, scale: float, threads: int, splits: int, hpt: int
):
    (batch, nqh, length, dim) = map(int, q.shape)
    nkh = int(k.shape[1])
    gqa = nqh // nkh
    (ids, counts, n_sel, u_width, q_pos, left_pad, total) = (
        compact_blocks_to_kernel_inputs(compact)
    )
    if int(hpt) < 1 or gqa % int(hpt):
        raise ValueError("indexed QSA heads per threadgroup must divide GQA")
    required_threads = int(hpt) * 32
    if threads != required_threads:
        raise ValueError("indexed QSA pass 1 requires one SIMD group per head")
    head_slices = gqa // int(hpt)
    if compact.causal_mask is None:
        mask = mx.ones((1, 1, 1, 1), dtype=mx.bool_)
        has_mask = False
    else:
        mask = compact.causal_mask
        if (
            mask.ndim != 4
            or int(mask.shape[1]) != 1
            or int(mask.shape[2]) != length
            or (int(mask.shape[3]) != total)
            or (mask.dtype != mx.bool_)
        ):
            raise QSAIndexedProbeDeclined(
                "indexed QSA requires a rank-4 [B|1, 1, L, T] bool cache mask",
                reason="unsupported_mask_layout",
            )
        if int(mask.shape[0]) not in (1, batch):
            raise QSAIndexedProbeDeclined(
                "indexed QSA cache mask batch must be 1 or B",
                reason="unsupported_mask_layout",
            )
        if int(mask.shape[0]) == 1 and batch > 1:
            mask = mx.broadcast_to(mask, (batch, 1, length, total))
        has_mask = True
    return _partition_kernel()(
        inputs=[
            q,
            k,
            v,
            mx.contiguous(ids.astype(mx.uint32)),
            mx.contiguous(counts.astype(mx.uint32)),
            mx.contiguous(n_sel.astype(mx.uint32)),
            mx.contiguous(q_pos.astype(mx.int32)),
            mx.contiguous(left_pad.astype(mx.int32)),
            mask,
            mx.array([scale], dtype=mx.float32),
            mx.array([length, total, u_width], dtype=mx.int32),
        ],
        template=[
            ("T", q.dtype),
            ("D", dim),
            ("NQH", nqh),
            ("NKVH", nkh),
            ("GQA", gqa),
            ("BS", int(compact.block_size)),
            ("S", int(splits)),
            ("HPT", int(hpt)),
            ("BLOCKS", _SDPA_BLOCKS),
            ("HAS_MASK", int(has_mask)),
        ],
        grid=(threads, length, batch * nkh * splits * head_slices),
        threadgroup=(threads, 1, 1),
        output_shapes=[
            (batch, nqh, length, _SDPA_BLOCKS),
            (batch, nqh, length, _SDPA_BLOCKS),
            (batch, nqh, length, _SDPA_BLOCKS, dim),
            (1,),
        ],
        output_dtypes=[mx.float32, mx.float32, q.dtype, mx.uint32],
    )


def _private_delta_partition_dispatch(
    q,
    base_k,
    base_v,
    delta_k,
    delta_v,
    delta_lengths,
    compact,
    *,
    scale: float,
    threads: int,
    splits: int,
    hpt: int,
):
    (batch, nqh, length, dim) = map(int, q.shape)
    nkh = int(base_k.shape[1])
    gqa = nqh // nkh
    (ids, counts, n_sel, u_width, q_pos, left_pad, total) = (
        compact_blocks_to_kernel_inputs(compact)
    )
    if compact.left_padding is not None:
        raise ValueError("private-delta QSA requires an unpadded shared base")
    base_block = int(base_k.shape[2]) // int(compact.block_size)
    slots = mx.arange(u_width, dtype=mx.uint32)[None, None]
    base_counts = mx.sum(
        (slots < counts[..., None]) & (ids < base_block), axis=-1
    ).astype(mx.uint32)
    if int(hpt) < 1 or gqa % int(hpt):
        raise ValueError("indexed QSA heads per threadgroup must divide GQA")
    required_threads = int(hpt) * 32
    if threads != required_threads:
        raise ValueError("indexed QSA pass 1 requires one SIMD group per head")
    head_slices = gqa // int(hpt)
    if compact.causal_mask is None:
        mask = mx.ones((1, 1, 1, 1), dtype=mx.bool_)
        has_mask = False
    else:
        mask = compact.causal_mask
        if (
            mask.ndim != 4
            or int(mask.shape[1]) != 1
            or int(mask.shape[2]) != length
            or (int(mask.shape[3]) != total)
            or (mask.dtype != mx.bool_)
        ):
            raise ValueError("private-delta QSA mask must be [B|1,1,L,T] bool")
        if int(mask.shape[0]) not in (1, batch):
            raise ValueError("private-delta QSA mask batch must be 1 or B")
        if int(mask.shape[0]) == 1 and batch > 1:
            mask = mx.broadcast_to(mask, (batch, 1, length, total))
        has_mask = True
    return _private_delta_partition_kernel()(
        inputs=[
            q,
            base_k,
            base_v,
            delta_k,
            delta_v,
            mx.contiguous(delta_lengths.astype(mx.uint32)),
            mx.contiguous(base_counts),
            mx.contiguous(ids.astype(mx.uint32)),
            mx.contiguous(counts.astype(mx.uint32)),
            mx.contiguous(n_sel.astype(mx.uint32)),
            mx.contiguous(q_pos.astype(mx.int32)),
            mask,
            mx.array([scale], dtype=mx.float32),
            mx.array([length, total, u_width, int(base_k.shape[2])], dtype=mx.int32),
        ],
        template=[
            ("T", q.dtype),
            ("D", dim),
            ("NQH", nqh),
            ("NKVH", nkh),
            ("GQA", gqa),
            ("BS", int(compact.block_size)),
            ("S", int(splits)),
            ("HPT", int(hpt)),
            ("BLOCKS", _SDPA_BLOCKS),
            ("HAS_MASK", int(has_mask)),
        ],
        grid=(threads, length, batch * nkh * splits * head_slices),
        threadgroup=(threads, 1, 1),
        output_shapes=[
            (batch, nqh, length, _SDPA_BLOCKS),
            (batch, nqh, length, _SDPA_BLOCKS),
            (batch, nqh, length, _SDPA_BLOCKS, dim),
            (1,),
        ],
        output_dtypes=[mx.float32, mx.float32, q.dtype, mx.uint32],
    )


def _private_delta_exact_set_partition_dispatch(
    q,
    base_k,
    base_v,
    delta_k,
    delta_v,
    delta_lengths,
    compact,
    *,
    scale: float,
    threads: int,
    splits: int,
    hpt: int,
):
    (batch, nqh, length, dim) = map(int, q.shape)
    if batch != 2:
        raise ValueError("exact-set folded private-delta QSA requires B=2")
    nkh = int(base_k.shape[1])
    gqa = nqh // nkh
    (ids, counts, n_sel, u_width, q_pos, left_pad, total) = (
        compact_blocks_to_kernel_inputs(compact)
    )
    if compact.left_padding is not None:
        raise ValueError("exact-set folded QSA requires an unpadded base")
    base_block = int(base_k.shape[2]) // int(compact.block_size)
    slots = mx.arange(u_width, dtype=mx.uint32)[None, None]
    base_counts = mx.sum(
        (slots < counts[..., None]) & (ids < base_block), axis=-1
    ).astype(mx.uint32)
    exact_set = qwen4_qsa_indexed_private_delta_exact_set_proof(
        compact, base_tokens=int(base_k.shape[2])
    )
    if int(hpt) < 1 or gqa % int(hpt):
        raise ValueError("indexed QSA heads per threadgroup must divide GQA")
    required_threads = int(hpt) * 32
    if threads != required_threads:
        raise ValueError("indexed QSA pass 1 requires one SIMD group per head")
    head_slices = gqa // int(hpt)
    if compact.causal_mask is None:
        mask = mx.ones((1, 1, 1, 1), dtype=mx.bool_)
        has_mask = False
    else:
        mask = compact.causal_mask
        if (
            mask.ndim != 4
            or int(mask.shape[1]) != 1
            or int(mask.shape[2]) != length
            or (int(mask.shape[3]) != total)
            or (mask.dtype != mx.bool_)
        ):
            raise ValueError("exact-set folded QSA mask must be [B|1,1,L,T] bool")
        if int(mask.shape[0]) not in (1, batch):
            raise ValueError("exact-set folded QSA mask batch must be 1 or B")
        if int(mask.shape[0]) == 1:
            mask = mx.broadcast_to(mask, (batch, 1, length, total))
        has_mask = True
    return _private_delta_exact_set_partition_kernel()(
        inputs=[
            q,
            base_k,
            base_v,
            delta_k,
            delta_v,
            mx.contiguous(delta_lengths.astype(mx.uint32)),
            mx.contiguous(base_counts),
            exact_set,
            mx.contiguous(ids.astype(mx.uint32)),
            mx.contiguous(counts.astype(mx.uint32)),
            mx.contiguous(n_sel.astype(mx.uint32)),
            mx.contiguous(q_pos.astype(mx.int32)),
            mask,
            mx.array([scale], dtype=mx.float32),
            mx.array([length, total, u_width, int(base_k.shape[2])], dtype=mx.int32),
        ],
        template=[
            ("T", q.dtype),
            ("D", dim),
            ("NQH", nqh),
            ("NKVH", nkh),
            ("GQA", gqa),
            ("BS", int(compact.block_size)),
            ("S", int(splits)),
            ("HPT", int(hpt)),
            ("BLOCKS", _SDPA_BLOCKS),
            ("HAS_MASK", int(has_mask)),
        ],
        grid=(threads, length, nkh * splits * head_slices),
        threadgroup=(threads, 1, 1),
        output_shapes=[
            (batch, nqh, length, _SDPA_BLOCKS),
            (batch, nqh, length, _SDPA_BLOCKS),
            (batch, nqh, length, _SDPA_BLOCKS, dim),
            (1,),
        ],
        output_dtypes=[mx.float32, mx.float32, q.dtype, mx.uint32],
    )


def _quantized_partition_dispatch(
    q,
    q_keys,
    q_values,
    compact,
    *,
    scale: float,
    threads: int,
    splits: int,
    hpt: int,
    group_size: int,
    key_bits: int,
    value_bits: int,
):
    (batch, nqh, length, dim) = map(int, q.shape)
    nkh = int(q_keys[0].shape[1])
    gqa = nqh // nkh
    (ids, counts, n_sel, u_width, q_pos, left_pad, total) = (
        compact_blocks_to_kernel_inputs(compact)
    )
    if int(hpt) < 1 or gqa % int(hpt):
        raise ValueError("indexed QSA heads per threadgroup must divide GQA")
    required_threads = int(hpt) * 32
    if threads != required_threads:
        raise ValueError("indexed QSA pass 1 requires one SIMD group per head")
    head_slices = gqa // int(hpt)
    if compact.causal_mask is None:
        mask = mx.ones((1,), dtype=mx.bool_)
        has_mask = False
    else:
        mask = compact.causal_mask
        if (
            mask.ndim != 4
            or int(mask.shape[1]) != 1
            or int(mask.shape[2]) != length
            or (int(mask.shape[3]) != total)
            or (mask.dtype != mx.bool_)
        ):
            raise QSAIndexedProbeDeclined(
                "indexed QSA requires a rank-4 [B|1, 1, L, T] bool cache mask",
                reason="unsupported_mask_layout",
            )
        if int(mask.shape[0]) not in (1, batch):
            raise QSAIndexedProbeDeclined(
                "indexed QSA cache mask batch must be 1 or B",
                reason="unsupported_mask_layout",
            )
        if int(mask.shape[0]) == 1 and batch > 1:
            mask = mx.broadcast_to(mask, (batch, 1, length, total))
        mask = mask[:, 0]
        has_mask = True
    return _quantized_partition_kernel()(
        inputs=[
            mx.contiguous(q),
            *(mx.contiguous(x) for x in q_keys),
            *(mx.contiguous(x) for x in q_values),
            mx.contiguous(ids.astype(mx.uint32)),
            mx.contiguous(counts.astype(mx.uint32)),
            mx.contiguous(n_sel.astype(mx.uint32)),
            mx.contiguous(q_pos.astype(mx.int32)),
            mx.contiguous(left_pad.astype(mx.int32)),
            mx.contiguous(mask),
            mx.array([scale], dtype=mx.float32),
            mx.array([length, total, u_width], dtype=mx.int32),
        ],
        template=[
            ("T", q.dtype),
            ("D", dim),
            ("NQH", nqh),
            ("NKVH", nkh),
            ("GQA", gqa),
            ("BS", int(compact.block_size)),
            ("S", int(splits)),
            ("HPT", int(hpt)),
            ("BLOCKS", _SDPA_BLOCKS),
            ("HAS_MASK", int(has_mask)),
            ("GROUP_SIZE", int(group_size)),
            ("KBITS", int(key_bits)),
            ("VBITS", int(value_bits)),
        ],
        grid=(threads, length, batch * nkh * splits * head_slices),
        threadgroup=(threads, 1, 1),
        output_shapes=[
            (batch, nqh, length, _SDPA_BLOCKS),
            (batch, nqh, length, _SDPA_BLOCKS),
            (batch, nqh, length, _SDPA_BLOCKS, dim),
            (1,),
        ],
        output_dtypes=[mx.float32, mx.float32, q.dtype, mx.uint32],
    )


def _combine_sdpa_partials(m, l, o, engaged, *, output_dtype, output_gate=None):
    (batch, nqh, length, blocks) = map(int, m.shape)
    dim = int(o.shape[-1])
    if blocks != _SDPA_BLOCKS or dim % 32:
        raise ValueError("indexed QSA pass 2 has unsupported partial geometry")
    if fused_merge_enabled():
        output = combine_indexed_partials(
            m,
            l,
            o,
            output_dtype=output_dtype,
            on_fallback=_record_merge_fallback,
            output_gate=output_gate,
        )
        return (output, engaged)
    gated = output_gate is not None
    inputs = [m, l, o, mx.array([length], dtype=mx.int32)]
    if gated:
        expected = (batch, length, nqh * dim)
        if output_gate.shape != expected or output_gate.dtype != output_dtype:
            raise ValueError(
                f"QSA output gate must be {expected} with dtype {output_dtype}"
            )
        inputs.append(mx.contiguous(output_gate))
        record_native_gate_engaged()
    output = _combine_kernel(gated)(
        inputs=inputs,
        template=[
            ("T", output_dtype),
            ("D", dim),
            ("BLOCKS", _SDPA_BLOCKS),
            ("H", nqh),
        ],
        grid=(1024, length, batch * nqh),
        threadgroup=(1024, 1, 1),
        output_shapes=[(batch, nqh, length, dim)],
        output_dtypes=[output_dtype],
    )[0]
    return (output, engaged)


def _record_merge_fallback() -> None:
    """Record an indexed-QSA receipt for a fused merge fallback."""
    global _STATUS_FALLBACKS
    with _STATUS_LOCK:
        _STATUS_COUNTS["merge_fallback"] += 1
        _STATUS_FALLBACKS += 1


def qwen4_qsa_indexed_private_delta_attention(
    q,
    base_k,
    base_v,
    delta_k,
    delta_v,
    delta_lengths,
    compact,
    *,
    scale: float,
    splits: int | None = None,
    hpt: int | None = None,
):
    """Read one shared QSA prefix plus private suffixes in source phases."""
    arrays = (q, base_k, base_v, delta_k, delta_v)
    if any((value.ndim != 4 for value in arrays)):
        raise ValueError("private-delta QSA requires rank-4 q/base/delta tensors")
    if base_k.shape != base_v.shape or delta_k.shape != delta_v.shape:
        raise ValueError("private-delta QSA K/V geometries must match")
    if int(base_k.shape[0]) != 1:
        raise ValueError("private-delta QSA base must have one immutable row")
    (batch, nqh, length, dim) = map(int, q.shape)
    if int(delta_k.shape[0]) != batch:
        raise ValueError("private-delta QSA needs one suffix per query row")
    if base_k.shape[1] != delta_k.shape[1] or base_k.shape[3] != delta_k.shape[3]:
        raise ValueError("private-delta QSA base and suffix layouts differ")
    if dim != int(base_k.shape[3]) or dim != 256:
        raise QSAIndexedProbeDeclined("private-delta QSA exact mode requires D=256")
    if any((value.dtype != q.dtype for value in arrays[1:])):
        raise ValueError("private-delta QSA requires one unquantized dtype")
    if delta_lengths.ndim != 1 or int(delta_lengths.shape[0]) != batch:
        raise ValueError("private-delta QSA lengths must have shape [B]")
    base_tokens = int(base_k.shape[2])
    delta_width = int(delta_k.shape[2])
    if base_tokens % _BLOCK_SIZE:
        raise ValueError("private-delta QSA base must end on a four-token block")
    if base_tokens + delta_width != int(compact.physical_width):
        raise ValueError("private-delta QSA storage does not match logical width")
    if int(compact.block_size) != _BLOCK_SIZE:
        raise ValueError("private-delta QSA requires block size 4")
    nkh = int(base_k.shape[1])
    (preflight_ok, preflight_reason) = qwen4_qsa_indexed_private_delta_preflight(
        length=length,
        base_tokens=base_tokens,
        head_dim=dim,
        num_query_heads=nqh,
        num_kv_heads=nkh,
        block_size=int(compact.block_size),
        selected_blocks=int(compact.block_ids.shape[-1]),
    )
    if not preflight_ok:
        raise QSAIndexedProbeDeclined(
            f"private-delta QSA preflight declined: {preflight_reason}",
            reason=preflight_reason,
        )
    (_, _, _, u_width, _, _, _) = compact_blocks_to_kernel_inputs(compact)
    token_width = u_width * _BLOCK_SIZE
    if token_width <= 1024 or token_width > 8192:
        raise QSAIndexedProbeDeclined(
            "private-delta QSA requires the MLX two-pass SDPA geometry"
        )
    requested = None if splits is None else int(splits)
    if requested is not None and requested not in _SPLIT_CANDIDATES:
        allowed = ", ".join(map(str, reversed(_SPLIT_CANDIDATES)))
        raise ValueError(f"indexed QSA splits must be one of {allowed}")
    requested_hpt = None if hpt is None else validate_hpt(hpt, 12)
    if _TOPOLOGY_CAPTURE_DIR:
        _capture_private_delta_topology_diagnostic(
            compact,
            q=q,
            base_tokens=base_tokens,
            delta_width=delta_width,
            delta_lengths=delta_lengths,
            requested_splits=requested,
            requested_hpt=requested_hpt,
        )
    key = (
        str(q.dtype),
        batch,
        nqh,
        nkh,
        length,
        dim,
        u_width,
        compact.causal_mask is not None,
        requested,
        requested_hpt,
    )
    with _PROBE_LOCK:
        candidate = _PRIVATE_DELTA_PROBE_RESULTS.get(key, _MISSING)
        if candidate is False:
            raise QSAIndexedProbeDeclined(
                "private-delta indexed QSA candidate ladder was declined"
            )
        if candidate is _MISSING:

            def dispatch(attempted):
                partials = _private_delta_partition_dispatch(
                    q,
                    base_k,
                    base_v,
                    delta_k,
                    delta_v,
                    delta_lengths,
                    compact,
                    scale=scale,
                    threads=attempted[0],
                    splits=attempted[1],
                    hpt=attempted[2],
                )
                return _combine_sdpa_partials(*partials, output_dtype=q.dtype)

            (candidate, output, counter, timings) = _measure_candidates(
                _candidate_ladder(requested, 12, requested_hpt), dispatch
            )
            if candidate is None:
                _PRIVATE_DELTA_PROBE_RESULTS[key] = False
                raise QSAIndexedProbeDeclined(
                    "private-delta indexed QSA candidate ladder was declined"
                )
            _PRIVATE_DELTA_PROBE_RESULTS[key] = candidate
            _PROBE_TIMINGS[("private_delta",) + key] = timings
            return mx.depends(output, counter)
    partials = _private_delta_partition_dispatch(
        q,
        base_k,
        base_v,
        delta_k,
        delta_v,
        delta_lengths,
        compact,
        scale=scale,
        threads=candidate[0],
        splits=candidate[1],
        hpt=candidate[2],
    )
    (output, counter) = _combine_sdpa_partials(*partials, output_dtype=q.dtype)
    return mx.depends(output, counter)


def qwen4_qsa_indexed_private_delta_exact_set_attention(
    q,
    base_k,
    base_v,
    delta_k,
    delta_v,
    delta_lengths,
    compact,
    *,
    scale: float,
    splits: int | None = None,
    hpt: int | None = None,
):
    """Experimental B2 fold for equal ordered selected-base-page sets.

    This is deliberately separate from the proven private-delta entry point.
    It is the default B2 consumer only after the private-delta lane itself has
    been admitted. A device-resident exact-set proof gates each query, and the
    row-local path preserves exactness wherever that proof is false.
    """
    arrays = (q, base_k, base_v, delta_k, delta_v)
    if any((value.ndim != 4 for value in arrays)):
        raise ValueError("exact-set folded QSA requires rank-4 tensors")
    if base_k.shape != base_v.shape or delta_k.shape != delta_v.shape:
        raise ValueError("exact-set folded QSA K/V geometries must match")
    if int(base_k.shape[0]) != 1:
        raise ValueError("exact-set folded QSA base must have one row")
    (batch, nqh, length, dim) = map(int, q.shape)
    if int(delta_k.shape[0]) != batch:
        raise ValueError("exact-set folded QSA needs one suffix per row")
    if int(compact.block_ids.shape[0]) != batch:
        raise ValueError("exact-set folded QSA compact batch differs")
    if base_k.shape[1] != delta_k.shape[1] or base_k.shape[3] != delta_k.shape[3]:
        raise ValueError("exact-set folded QSA base and suffix layouts differ")
    if dim != int(base_k.shape[3]) or dim != 256:
        raise QSAIndexedProbeDeclined("exact-set folded QSA exact mode requires D=256")
    if any((value.dtype != q.dtype for value in arrays[1:])):
        raise ValueError("exact-set folded QSA requires one unquantized dtype")
    if delta_lengths.ndim != 1 or int(delta_lengths.shape[0]) != batch:
        raise ValueError("exact-set folded QSA lengths must have shape [B]")
    base_tokens = int(base_k.shape[2])
    delta_width = int(delta_k.shape[2])
    if base_tokens % _BLOCK_SIZE:
        raise ValueError("exact-set folded QSA base must align to four tokens")
    if base_tokens + delta_width != int(compact.physical_width):
        raise ValueError("exact-set folded QSA storage differs from logical width")
    nkh = int(base_k.shape[1])
    (preflight_ok, preflight_reason) = (
        qwen4_qsa_indexed_private_delta_exact_set_preflight(
            batch=batch,
            length=length,
            base_tokens=base_tokens,
            head_dim=dim,
            num_query_heads=nqh,
            num_kv_heads=nkh,
            block_size=int(compact.block_size),
            selected_blocks=int(compact.block_ids.shape[-1]),
        )
    )
    if not preflight_ok:
        raise QSAIndexedProbeDeclined(
            f"exact-set folded QSA preflight declined: {preflight_reason}",
            reason=preflight_reason,
        )
    (_, _, _, u_width, _, _, _) = compact_blocks_to_kernel_inputs(compact)
    requested = None if splits is None else int(splits)
    if requested is not None and requested not in _SPLIT_CANDIDATES:
        allowed = ", ".join(map(str, reversed(_SPLIT_CANDIDATES)))
        raise ValueError(f"indexed QSA splits must be one of {allowed}")
    requested_hpt = None if hpt is None else validate_hpt(hpt, 12)
    key = (
        str(q.dtype),
        batch,
        nqh,
        nkh,
        length,
        dim,
        u_width,
        compact.causal_mask is not None,
        requested,
        requested_hpt,
    )
    with _PROBE_LOCK:
        candidate = _PRIVATE_DELTA_EXACT_SET_PROBE_RESULTS.get(key, _MISSING)
        if candidate is False:
            raise QSAIndexedProbeDeclined(
                "exact-set folded QSA candidate ladder was declined"
            )
        if candidate is _MISSING:

            def dispatch(attempted):
                partials = _private_delta_exact_set_partition_dispatch(
                    q,
                    base_k,
                    base_v,
                    delta_k,
                    delta_v,
                    delta_lengths,
                    compact,
                    scale=scale,
                    threads=attempted[0],
                    splits=attempted[1],
                    hpt=attempted[2],
                )
                return _combine_sdpa_partials(*partials, output_dtype=q.dtype)

            (candidate, output, counter, timings) = _measure_candidates(
                _candidate_ladder(requested, 12, requested_hpt), dispatch
            )
            if candidate is None:
                _PRIVATE_DELTA_EXACT_SET_PROBE_RESULTS[key] = False
                raise QSAIndexedProbeDeclined(
                    "exact-set folded QSA candidate ladder was declined"
                )
            _PRIVATE_DELTA_EXACT_SET_PROBE_RESULTS[key] = candidate
            _PROBE_TIMINGS[("private_delta_exact_set",) + key] = timings
            return mx.depends(output, counter)
    partials = _private_delta_exact_set_partition_dispatch(
        q,
        base_k,
        base_v,
        delta_k,
        delta_v,
        delta_lengths,
        compact,
        scale=scale,
        threads=candidate[0],
        splits=candidate[1],
        hpt=candidate[2],
    )
    (output, counter) = _combine_sdpa_partials(*partials, output_dtype=q.dtype)
    return mx.depends(output, counter)


def qwen4_qsa_indexed_attention(
    q,
    k,
    v,
    compact,
    *,
    scale: float,
    splits: int | None = None,
    hpt: int | None = None,
    output_gate=None,
):
    """Dispatch indexed attention with MLX SDPA's two-pass reduction tree."""
    mlx_version = str(getattr(mx, "__version__", "unknown"))
    allow_unverified = (
        os.environ.get("MLX_QWEN4_QSA_INDEXED_ALLOW_UNVERIFIED_MLX") == "1"
    )
    if mlx_version not in _EXACT_MLX_BUILDS and (not allow_unverified):
        raise QSAIndexedProbeDeclined(
            f"indexed QSA exactness is unproven on mlx {mlx_version}",
            reason="mlx_build_unverified",
        )
    (header_verified, _, header_reason) = _sdpa_header_state()
    if not header_verified and (not allow_unverified):
        raise QSAIndexedProbeDeclined(
            f"indexed QSA exactness clones the installed sdpa_vector.h ({header_reason})",
            reason=header_reason,
        )
    if not indexed_kernel_available():
        raise QSAIndexedProbeDeclined("indexed QSA Metal runtime is unavailable")
    if q.ndim != 4 or k.ndim != 4 or k.shape != v.shape:
        raise ValueError("indexed QSA wants matching rank-4 q/k/v tensors")
    if q.shape[0] != k.shape[0] or q.shape[1] % k.shape[1]:
        raise ValueError("indexed QSA requires matching batch and integral GQA")
    if int(k.shape[2]) != int(compact.physical_width):
        raise ValueError("indexed QSA tensors do not match compact selection")
    if int(compact.block_size) != _BLOCK_SIZE:
        raise ValueError("indexed QSA requires block size 4")
    if int(q.shape[-1]) != 256:
        raise QSAIndexedProbeDeclined("indexed QSA exact mode requires D=256")
    gqa = int(q.shape[1]) // int(k.shape[1])
    if gqa != 12:
        raise QSAIndexedProbeDeclined("indexed QSA exact mode requires GQA=12")
    requested_hpt = (
        None
        if hpt is None and _HPT_OVERRIDE == 0
        else indexed_hpt_for(gqa)
        if hpt is None
        else validate_hpt(hpt, gqa)
    )
    if gqa * 32 > 1024:
        raise ValueError("indexed QSA GQA exceeds the Metal threadgroup limit")
    sdpa_blocks = os.environ.get("MLX_SDPA_BLOCKS")
    if sdpa_blocks not in (None, "", str(_SDPA_BLOCKS)):
        raise QSAIndexedProbeDeclined("indexed QSA requires MLX_SDPA_BLOCKS=128")
    (_, _, _, u_width, _, _, _) = compact_blocks_to_kernel_inputs(compact)
    token_width = u_width * _BLOCK_SIZE
    if token_width <= 1024 or token_width > 8192:
        raise QSAIndexedProbeDeclined(
            "indexed QSA requires the MLX two-pass SDPA geometry"
        )
    architecture = str(mx.device_info().get("architecture", ""))
    if architecture[-1:] not in {"s", "d"}:
        raise QSAIndexedProbeDeclined(
            "indexed QSA exact mode requires a 128-block MLX SDPA device"
        )
    requested = (
        None
        if splits is None and _SPLITS_OVERRIDE == 0
        else indexed_splits_for(u_width)
        if splits is None
        else int(splits)
    )
    if requested is not None and requested not in _SPLIT_CANDIDATES:
        allowed = ", ".join(map(str, reversed(_SPLIT_CANDIDATES)))
        raise ValueError(f"indexed QSA splits must be one of {allowed}")
    geometry_key = f"B{int(q.shape[0])}-L{int(q.shape[2])}-U{int(u_width)}-dtype{q.dtype}-mask{int(compact.causal_mask is not None)}"
    key = (
        mlx_version,
        str(q.dtype),
        int(q.shape[-1]),
        int(q.shape[1]),
        int(k.shape[1]),
        int(compact.block_size),
        int(u_width),
        int(q.shape[0]),
        int(q.shape[2]),
        int(compact.causal_mask is not None),
        requested,
        requested_hpt,
        output_gate is not None,
    )
    candidate = _PROBE_RESULTS.get(key, _MISSING)
    if candidate is False:
        raise QSAIndexedProbeDeclined("indexed QSA candidate ladder was declined")
    if candidate is _MISSING:
        with _PROBE_LOCK:
            candidate = _PROBE_RESULTS.get(key, _MISSING)
            if candidate is False:
                raise QSAIndexedProbeDeclined(
                    "indexed QSA candidate ladder was declined"
                )
            if candidate is _MISSING:

                def dispatch(attempted):
                    partials = _partition_dispatch(
                        q,
                        k,
                        v,
                        compact,
                        scale=scale,
                        threads=attempted[0],
                        splits=attempted[1],
                        hpt=attempted[2],
                    )
                    return _combine_sdpa_partials(
                        *partials, output_dtype=q.dtype, output_gate=output_gate
                    )

                (candidate, combined, counter, timings) = _measure_candidates(
                    _candidate_ladder(requested, gqa, requested_hpt), dispatch
                )
                if candidate is None:
                    _PROBE_RESULTS[key] = False
                    raise QSAIndexedProbeDeclined(
                        "indexed QSA candidate ladder was declined"
                    )
                _PROBE_RESULTS[key] = candidate
                _PROBE_TIMINGS[key] = timings
                return _device_attest_output(
                    combined,
                    counter,
                    length=int(q.shape[2]),
                    context=int(compact.physical_width),
                    splits=candidate[1],
                    hpt=candidate[2],
                    candidate=candidate,
                    geometry_key=geometry_key,
                    candidate_timings_ms=timings,
                )
    (m, l, o, counter) = _partition_dispatch(
        q,
        k,
        v,
        compact,
        scale=scale,
        threads=candidate[0],
        splits=candidate[1],
        hpt=candidate[2],
    )
    (output, counter) = _combine_sdpa_partials(
        m, l, o, counter, output_dtype=q.dtype, output_gate=output_gate
    )
    return _device_attest_output(
        output,
        counter,
        length=int(q.shape[2]),
        context=int(compact.physical_width),
        splits=candidate[1],
        hpt=candidate[2],
        candidate=candidate,
        geometry_key=geometry_key,
        candidate_timings_ms=_PROBE_TIMINGS[key],
    )


def qwen4_qsa_indexed_quantized_attention(
    q,
    q_keys,
    q_values,
    compact,
    *,
    scale: float,
    group_size: int,
    key_bits: int,
    value_bits: int,
    splits: int | None = None,
    hpt: int | None = None,
):
    """Read affine int8/int4 K/V inside the indexed SDPA kernel."""
    mlx_version = str(getattr(mx, "__version__", "unknown"))
    allow_unverified = (
        os.environ.get("MLX_QWEN4_QSA_INDEXED_ALLOW_UNVERIFIED_MLX") == "1"
    )
    if mlx_version not in _EXACT_MLX_BUILDS and (not allow_unverified):
        raise QSAIndexedProbeDeclined(
            f"indexed QSA exactness is unproven on mlx {mlx_version}",
            reason="mlx_build_unverified",
        )
    (header_verified, _, header_reason) = _sdpa_header_state()
    if not header_verified and (not allow_unverified):
        raise QSAIndexedProbeDeclined(
            f"indexed QSA exactness clones the installed sdpa_vector.h ({header_reason})",
            reason=header_reason,
        )
    if not indexed_kernel_available():
        raise QSAIndexedProbeDeclined("indexed QSA Metal runtime is unavailable")
    if q.ndim != 4 or len(q_keys) != 3 or len(q_values) != 3:
        raise ValueError("quantized indexed QSA wants packed K/V triples")
    if q_keys[0].ndim != 4 or q_values[0].ndim != 4:
        raise ValueError("quantized indexed QSA wants rank-4 packed K/V")
    if q_keys[0].shape[:3] != q_values[0].shape[:3]:
        raise ValueError("quantized indexed QSA K/V geometry must match")
    if q.shape[0] != q_keys[0].shape[0] or q.shape[1] % q_keys[0].shape[1]:
        raise ValueError("indexed QSA requires matching batch and integral GQA")
    if int(q_keys[0].shape[2]) != int(compact.physical_width):
        raise ValueError("indexed QSA tensors do not match compact selection")
    if q_keys[0].dtype != mx.uint32 or q_values[0].dtype != mx.uint32:
        raise ValueError("quantized indexed QSA packed weights must be uint32")
    if any((x.dtype != q.dtype for x in (*q_keys[1:], *q_values[1:]))):
        raise QSAIndexedProbeDeclined(
            "quantized indexed QSA requires query and affine parameter dtype match"
        )
    group_size = int(group_size)
    key_bits = int(key_bits)
    value_bits = int(value_bits)
    if group_size not in _QUANTIZED_GROUP_SIZES:
        raise QSAIndexedProbeDeclined("quantized indexed QSA group size is unsupported")
    if key_bits not in _QUANTIZED_BITS or value_bits not in _QUANTIZED_BITS:
        raise QSAIndexedProbeDeclined("quantized indexed QSA bits are unsupported")
    if int(compact.block_size) != _BLOCK_SIZE:
        raise ValueError("indexed QSA requires block size 4")
    dim = int(q.shape[-1])
    if dim != 256:
        raise QSAIndexedProbeDeclined("indexed QSA exact mode requires D=256")
    groups = dim // group_size
    expected_k_packed = dim * key_bits // 32
    expected_v_packed = dim * value_bits // 32
    if int(q_keys[0].shape[-1]) != expected_k_packed:
        raise ValueError("quantized indexed QSA key packing does not match metadata")
    if int(q_values[0].shape[-1]) != expected_v_packed:
        raise ValueError("quantized indexed QSA value packing does not match metadata")
    if any((int(x.shape[-1]) != groups for x in (*q_keys[1:], *q_values[1:]))):
        raise ValueError("quantized indexed QSA affine groups do not match metadata")
    nkh = int(q_keys[0].shape[1])
    gqa = int(q.shape[1]) // nkh
    if gqa != 12:
        raise QSAIndexedProbeDeclined("indexed QSA exact mode requires GQA=12")
    requested_hpt = (
        None
        if hpt is None and _HPT_OVERRIDE == 0
        else indexed_hpt_for(gqa)
        if hpt is None
        else validate_hpt(hpt, gqa)
    )
    sdpa_blocks = os.environ.get("MLX_SDPA_BLOCKS")
    if sdpa_blocks not in (None, "", str(_SDPA_BLOCKS)):
        raise QSAIndexedProbeDeclined("indexed QSA requires MLX_SDPA_BLOCKS=128")
    (_, _, _, u_width, _, _, _) = compact_blocks_to_kernel_inputs(compact)
    token_width = u_width * _BLOCK_SIZE
    if token_width <= 1024 or token_width > 8192:
        raise QSAIndexedProbeDeclined(
            "indexed QSA requires the MLX two-pass SDPA geometry"
        )
    architecture = str(mx.device_info().get("architecture", ""))
    if architecture[-1:] not in {"s", "d"}:
        raise QSAIndexedProbeDeclined(
            "indexed QSA exact mode requires a 128-block MLX SDPA device"
        )
    requested = (
        None
        if splits is None and _SPLITS_OVERRIDE == 0
        else indexed_splits_for(u_width)
        if splits is None
        else int(splits)
    )
    if requested is not None and requested not in _SPLIT_CANDIDATES:
        allowed = ", ".join(map(str, reversed(_SPLIT_CANDIDATES)))
        raise ValueError(f"indexed QSA splits must be one of {allowed}")
    geometry_key = f"quantized-B{int(q.shape[0])}-L{int(q.shape[2])}-U{int(u_width)}-dtype{q.dtype}-mask{int(compact.causal_mask is not None)}-g{group_size}-k{key_bits}-v{value_bits}"
    key = (
        mlx_version,
        str(q.dtype),
        dim,
        int(q.shape[1]),
        nkh,
        int(compact.block_size),
        int(u_width),
        int(q.shape[0]),
        int(q.shape[2]),
        int(compact.causal_mask is not None),
        requested,
        requested_hpt,
        group_size,
        key_bits,
        value_bits,
    )
    candidate = _QUANTIZED_PROBE_RESULTS.get(key, _MISSING)
    if candidate is False:
        raise QSAIndexedProbeDeclined(
            "quantized indexed QSA candidate ladder was declined"
        )
    if candidate is _MISSING:
        with _PROBE_LOCK:
            candidate = _QUANTIZED_PROBE_RESULTS.get(key, _MISSING)
            if candidate is False:
                raise QSAIndexedProbeDeclined(
                    "quantized indexed QSA candidate ladder was declined"
                )
            if candidate is _MISSING:

                def dispatch(attempted):
                    partials = _quantized_partition_dispatch(
                        q,
                        q_keys,
                        q_values,
                        compact,
                        scale=scale,
                        threads=attempted[0],
                        splits=attempted[1],
                        hpt=attempted[2],
                        group_size=group_size,
                        key_bits=key_bits,
                        value_bits=value_bits,
                    )
                    return _combine_sdpa_partials(*partials, output_dtype=q.dtype)

                (candidate, combined, counter, timings) = _measure_candidates(
                    _candidate_ladder(requested, gqa, requested_hpt), dispatch
                )
                if candidate is None:
                    _QUANTIZED_PROBE_RESULTS[key] = False
                    raise QSAIndexedProbeDeclined(
                        "quantized indexed QSA candidate ladder was declined"
                    )
                _QUANTIZED_PROBE_RESULTS[key] = candidate
                _PROBE_TIMINGS[key] = timings
                return _device_attest_output(
                    combined,
                    counter,
                    length=int(q.shape[2]),
                    context=int(compact.physical_width),
                    reason="engaged_quantized",
                    splits=candidate[1],
                    hpt=candidate[2],
                    candidate=candidate,
                    geometry_key=geometry_key,
                    candidate_timings_ms=timings,
                )
    (m, l, o, counter) = _quantized_partition_dispatch(
        q,
        q_keys,
        q_values,
        compact,
        scale=scale,
        threads=candidate[0],
        splits=candidate[1],
        hpt=candidate[2],
        group_size=group_size,
        key_bits=key_bits,
        value_bits=value_bits,
    )
    (output, counter) = _combine_sdpa_partials(m, l, o, counter, output_dtype=q.dtype)
    return _device_attest_output(
        output,
        counter,
        length=int(q.shape[2]),
        context=int(compact.physical_width),
        reason="engaged_quantized",
        splits=candidate[1],
        hpt=candidate[2],
        candidate=candidate,
        geometry_key=geometry_key,
        candidate_timings_ms=_PROBE_TIMINGS[key],
    )


_CHUNK_SLOTS = 64


def indexed_chunk_ranges(u_width: int) -> tuple[tuple[int, int], ...]:
    """Return fixed slot chunks independent of the split count."""

    width = int(u_width)
    if width < 1:
        raise ValueError("u_width must be positive")
    return tuple(
        (start, min(start + _CHUNK_SLOTS, width))
        for start in range(0, width, _CHUNK_SLOTS)
    )


def qwen4_qsa_indexed_private_delta_reference(
    q,
    base_k,
    base_v,
    delta_k,
    delta_v,
    compact,
    *,
    scale: float,
    splits: int,
):
    """Materialized oracle for the shared-base/private-delta kernel."""

    batch = int(q.shape[0])
    keys = mx.concatenate(
        [mx.broadcast_to(base_k, (batch, *base_k.shape[1:])), delta_k],
        axis=2,
    )
    values = mx.concatenate(
        [mx.broadcast_to(base_v, (batch, *base_v.shape[1:])), delta_v],
        axis=2,
    )
    return qwen4_qsa_indexed_reference(
        q, keys, values, compact, scale=scale, splits=splits
    )


def qwen4_qsa_indexed_quantized_reference(
    q,
    q_keys,
    q_values,
    compact,
    *,
    scale: float,
    splits: int,
    group_size: int,
    key_bits: int,
    value_bits: int,
):
    """Dequantize with MLX, then run the unchanged bf16 mirror."""

    keys, values = dequantize_qsa_quantized_kv(
        q_keys,
        q_values,
        group_size=group_size,
        key_bits=key_bits,
        value_bits=value_bits,
    )
    return qwen4_qsa_indexed_reference(
        q, keys, values, compact, scale=scale, splits=splits
    )


def indexed_split_chunk_ranges(
    u_width: int, splits: int
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Distribute fixed chunks over splits, with remainders first."""

    chunks = indexed_chunk_ranges(u_width)
    count = int(splits)
    if count < 1 or count > min(_SDPA_BLOCKS, int(u_width)):
        raise ValueError("splits must be in [1, min(128, u_width)]")
    base, remainder = divmod(len(chunks), count)
    groups = []
    start = 0
    for split in range(count):
        stop = start + base + (1 if split < remainder else 0)
        groups.append(chunks[start:stop])
        start = stop
    return tuple(groups)
