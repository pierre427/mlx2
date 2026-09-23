# SPDX-License-Identifier: Apache-2.0
# Adapted from unified; see provenance/.
from __future__ import annotations

_QSA_NAX_DECODE_LAST_DECISION = None
_QSA_STAGE1_LAST_DECISION = None
_QSA_NAX_LAST_DECISION = None
import hashlib
import json
import math
import os
import threading
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from .. import round_levers as _lv
from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import (
    ArraysCache,
    BatchKVCache,
    BatchQuantizedKVCache,
    KVCache,
    QuantizedKVCache,
    SinkWindowKVCache,
    dynamic_roll,
)
from .gated_delta import gated_delta_update
from .pipeline import PipelineMixin
from .qwen4_fused_gdn import (
    admit_qwen4_fused_gdn_decode,
    fused_gdn_runtime_supported,
    probe_qwen4_fused_gdn_decode,
    qwen4_fused_gdn_decode,
    qwen4_fused_gdn_decode_outproj,
)
from .qwen4_fused_gdn_verify import (
    admit_qwen4_fused_gdn_verify,
    probe_qwen4_fused_gdn_catchup,
    probe_qwen4_fused_gdn_replay_verify,
    probe_qwen4_fused_gdn_verify,
    qwen4_fused_gdn_catchup,
    qwen4_fused_gdn_reconstruct,
    qwen4_fused_gdn_replay_verify,
    qwen4_fused_gdn_verify,
    validate_qwen4_gdn_replay_acceptance,
)
from .qwen4_gdn_outproj import admit_qwen4_gdn_outproj
from .qwen4_qsa_nax import (
    block_sparse_layout_supported,
    compact_blocks_to_kernel_inputs,
    compact_token_validity,
    nax_kernel_available,
    nax_qsa_attention,
)
from .qwen4_qsa_indexed import (
    QSAIndexedProbeDeclined,
    dequantize_qsa_quantized_kv,
    decide_qsa_indexed_admission,
    indexed_splits_for,
    qsa_indexed_enabled,
    qsa_indexed_quantized_cache_config,
    qsa_indexed_status,
    qwen4_qsa_indexed_attention,
    qwen4_qsa_indexed_quantized_attention,
    qwen4_qsa_indexed_reference,
    record_qsa_indexed_receipt,
)
from .qwen4_qsa_indexed_merge import fused_gate_enabled, mlx_apply_output_gate
from .qwen4_qsa_stage1 import (
    qsa_stage1_score_producer,
    qsa_stage1_select,
    qsa_stage1_supported,
)
from .qwen3_5 import GatedDeltaNet as Qwen35GatedDeltaNet
from .qwen3_next import Qwen3NextSparseMoeBlock as SparseMoeBlock
from . import qwen3_next
from .qwen3_next import (
    _build_hyper_gate,
    _build_hyper_mix,
    _build_inject_apply,
    _concat_tables,
    _env_flag,
    _proj_identity,
    _proj_signature,
    _proj_table,
    _run_glue,
    compile_glue_enabled,
    check_materialization_budget,
    table_bytes,
    transform_moe_weights,
)
from .rope_utils import initialize_rope
from ..verify_sync import record_verify_sync

_QSA_SEGMENT_CAPTURE_LOCK = threading.Lock()
_QSA_SEGMENT_CAPTURE_COUNT = 0


def _capture_qsa_segment_inputs(
    q, pooled, q_pos, valid_blocks, selected, *, layer_id: Union[int, str]
) -> None:
    """Save bounded QSA selection inputs for an explicit offline experiment."""
    capture_dir = os.environ.get("MLX_QWEN4_QSA_SEGMENT_CAPTURE_DIR")
    if not capture_dir:
        return
    allowed_layers = os.environ.get("MLX_QWEN4_QSA_SEGMENT_CAPTURE_LAYERS")
    if allowed_layers and str(layer_id) not in {
        value.strip() for value in allowed_layers.split(",")
    }:
        return
    min_blocks = max(
        0, int(os.environ.get("MLX_QWEN4_QSA_SEGMENT_CAPTURE_MIN_BLOCKS", "0"))
    )
    if int(pooled.shape[1]) < min_blocks:
        return
    limit = max(1, int(os.environ.get("MLX_QWEN4_QSA_SEGMENT_CAPTURE_COUNT", "24")))
    capture_keys = os.environ.get("MLX_QWEN4_QSA_SEGMENT_CAPTURE_KEYS", "1") != "0"
    global _QSA_SEGMENT_CAPTURE_COUNT
    with _QSA_SEGMENT_CAPTURE_LOCK:
        remaining = limit - _QSA_SEGMENT_CAPTURE_COUNT
        if remaining <= 0:
            return
        values = [q_pos, valid_blocks, selected]
        if capture_keys:
            values.extend((q, pooled))
        mx.eval(*values)
        q_pos_np = np.asarray(q_pos)
        valid_np = np.asarray(valid_blocks)
        selected_np = np.asarray(selected)
        q_np = np.asarray(q.astype(mx.float32)) if capture_keys else None
        pooled_np = np.asarray(pooled.astype(mx.float32)) if capture_keys else None
        directory = Path(capture_dir)
        directory.mkdir(parents=True, exist_ok=True)
        rows = min(int(selected_np.shape[0]), remaining)
        for row in range(rows):
            _QSA_SEGMENT_CAPTURE_COUNT += 1
            safe_layer_id = str(layer_id).replace(":", "-")
            stem = (
                f"qsa-selection-{_QSA_SEGMENT_CAPTURE_COUNT:04d}-layer-{safe_layer_id}"
            )
            target = directory / f"{stem}.npz"
            temporary = directory / f".{stem}.npz.tmp"
            with temporary.open("wb") as stream:
                payload = dict(
                    valid_blocks=np.asarray(
                        [int(valid_np[row, -1].sum())], dtype=np.int64
                    ),
                    q_positions=q_pos_np[row, -1:],
                    production_selected=selected_np[row, -1:],
                    layer_id=np.asarray([str(layer_id)]),
                )
                if capture_keys:
                    payload.update(keys=pooled_np[row], queries=q_np[row, -1:, :, :])
                np.savez_compressed(stream, **payload)
            temporary.replace(target)


_RMSNORM_FAST = _env_flag("MLX_QWEN4_RMSNORM_FAST", default=True)
_RMSNORM_FAST_MAX_WIDTH = max(
    0, int(os.environ.get("MLX_QWEN4_RMSNORM_FAST_MAX_WIDTH", "8"))
)
_RMSNORM_FAST_WIDTH_OVERRIDE: Optional[int] = None


@contextmanager
def _declared_width(width: int):
    """Pin the query width the gate sees, for a caller that has split a slab."""
    global _RMSNORM_FAST_WIDTH_OVERRIDE
    previous = _RMSNORM_FAST_WIDTH_OVERRIDE
    _RMSNORM_FAST_WIDTH_OVERRIDE = int(width)
    try:
        yield
    finally:
        _RMSNORM_FAST_WIDTH_OVERRIDE = previous


_TRACE_EPOCH = 0


def _trace_flags() -> tuple:
    """Every module-tier value a compiled region in this module branches on."""
    return (_TRACE_EPOCH, bool(_RMSNORM_FAST), int(_RMSNORM_FAST_MAX_WIDTH))


_PLE_COMPILE = _env_flag("MLX_QWEN4_PLE_COMPILE", default=True)
_PLE_COMPILE_CACHE_MAX = max(
    1, int(os.environ.get("MLX_QWEN4_PLE_COMPILE_CACHE", "32"))
)
_PLE_COMPILE_STATS_LOCK = threading.Lock()
_PLE_COMPILE_STATS = {
    "builds": 0,
    "hits": 0,
    "fallbacks": 0,
    "overflow": 0,
    "skips": 0,
    "retraces": 0,
    "invalidations": 0,
}
_PLE_COMPILE_LAST_RECEIPT: Optional[dict] = None


def _record_ple_compile(event: str, **fields) -> None:
    global _PLE_COMPILE_LAST_RECEIPT
    with _PLE_COMPILE_STATS_LOCK:
        _PLE_COMPILE_STATS[event] = _PLE_COMPILE_STATS.get(event, 0) + 1
        if event != "hits":
            _PLE_COMPILE_LAST_RECEIPT = {"event": event, **fields}


def qwen4_ple_compile_status(*, reset: bool = False) -> dict:
    """Bounded receipts for the compiled PLE device chain.

    ``fallbacks`` MUST be 0 on a healthy run: every one is a signature that
    raised while tracing or replaying and was demoted to the eager chain.
    ``enabled`` is False by default -- the lever is opt-in, see the module
    comment on ``_PLE_COMPILE``.
    """
    global _PLE_COMPILE_LAST_RECEIPT
    with _PLE_COMPILE_STATS_LOCK:
        report = {
            "enabled": bool(_PLE_COMPILE),
            "cache_max": _PLE_COMPILE_CACHE_MAX,
            "counts": dict(_PLE_COMPILE_STATS),
            "last_receipt": _PLE_COMPILE_LAST_RECEIPT,
        }
        if reset:
            for key in _PLE_COMPILE_STATS:
                _PLE_COMPILE_STATS[key] = 0
            _PLE_COMPILE_LAST_RECEIPT = None
    return report


_QSA_POOLED_KEY_CACHE = _env_flag("MLX_QWEN4_QSA_POOLED_KEY_CACHE")
_QSA_APC_SUMMARIES = _env_flag("MLX_QWEN4_QSA_APC_SUMMARIES")
_QSA_SUMMARY_FORMAT_VERSION = 1
_QSA_SUMMARY_PRODUCER_VERSION = "qwen4-pooled-key-v1"
_QSA_SUMMARY_META_MARKER = "qsa_summary_v1"
_QSA_SUMMARY_IDENTITY_FIELDS = (
    "format_version",
    "model_config_hash",
    "block_size",
    "compress_ratio",
    "producer_version",
    "layer_id",
)
_QSA_SUMMARY_STATS_LOCK = threading.Lock()
_QSA_SUMMARY_STATS = Counter(
    {"hits": 0, "misses": 0, "recomputes": 0, "invalidations": 0}
)
_QSA_SUMMARY_BLOCKS = Counter({"reused": 0, "recomputed": 0, "invalidated": 0})
_QSA_SUMMARY_REASONS = Counter()
_QSA_SUMMARY_LAST_RECEIPT = None


def _qsa_summary_config_hash(args) -> str:
    payload = json.dumps(
        asdict(args), sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _qsa_summary_identity(args, layer_id) -> dict[str, Any]:
    ratio = int(args.indexer_compress_ratio)
    return {
        "format_version": _QSA_SUMMARY_FORMAT_VERSION,
        "model_config_hash": _qsa_summary_config_hash(args),
        "block_size": ratio,
        "compress_ratio": ratio,
        "producer_version": _QSA_SUMMARY_PRODUCER_VERSION,
        "layer_id": str(layer_id),
        "complete_blocks": 0,
    }


def _qsa_summary_identity_matches(stored, expected) -> bool:
    return bool(stored) and all(
        (
            stored.get(name) == expected.get(name)
            for name in _QSA_SUMMARY_IDENTITY_FIELDS
        )
    )


def _qsa_summary_with_coverage(identity, complete_blocks: int):
    if identity is None:
        return None
    updated = dict(identity)
    updated["complete_blocks"] = int(complete_blocks)
    return updated


def _record_qsa_summary(event: str, reason: str, *, blocks: int = 0) -> None:
    global _QSA_SUMMARY_LAST_RECEIPT
    if not _QSA_APC_SUMMARIES:
        return
    receipt = {"event": event, "reason": reason, "blocks": int(blocks)}
    with _QSA_SUMMARY_STATS_LOCK:
        _QSA_SUMMARY_STATS[event] += 1
        if event == "hits":
            _QSA_SUMMARY_BLOCKS["reused"] += int(blocks)
        elif event == "recomputes":
            _QSA_SUMMARY_BLOCKS["recomputed"] += int(blocks)
        elif event == "invalidations":
            _QSA_SUMMARY_BLOCKS["invalidated"] += int(blocks)
        _QSA_SUMMARY_REASONS[reason] += 1
        _QSA_SUMMARY_LAST_RECEIPT = receipt


def qsa_apc_summary_status(*, reset: bool = False) -> dict[str, Any]:
    """Return bounded APC-summary receipts for the QSA status endpoint."""
    global _QSA_SUMMARY_LAST_RECEIPT
    with _QSA_SUMMARY_STATS_LOCK:
        report = {
            "enabled": bool(_QSA_APC_SUMMARIES),
            "format_version": _QSA_SUMMARY_FORMAT_VERSION,
            "producer_version": _QSA_SUMMARY_PRODUCER_VERSION,
            "counts": dict(_QSA_SUMMARY_STATS),
            "blocks": dict(_QSA_SUMMARY_BLOCKS),
            "reasons": dict(_QSA_SUMMARY_REASONS),
            "last_receipt": _QSA_SUMMARY_LAST_RECEIPT,
        }
        if reset:
            for key in _QSA_SUMMARY_STATS:
                _QSA_SUMMARY_STATS[key] = 0
            for key in _QSA_SUMMARY_BLOCKS:
                _QSA_SUMMARY_BLOCKS[key] = 0
            _QSA_SUMMARY_REASONS.clear()
            _QSA_SUMMARY_LAST_RECEIPT = None
    return report


_QSA_SCATTER_CHOSEN = _env_flag("MLX_QWEN4_QSA_SCATTER_CHOSEN", default=True)
_PLE_VECTOR_SHIFT = _env_flag("MLX_QWEN4_PLE_VECTOR_SHIFT")
_PLE_GATHER_CONCAT = _env_flag("MLX_QWEN4_PLE_GATHER_CONCAT")
_GDN_SHAPE_STABLE_PROJECTIONS = _env_flag("MLX_QWEN4_GDN_SHAPE_STABLE_PROJECTIONS")
_GDN_FUSED_INPROJ = _env_flag("MLX_QWEN4_GDN_FUSED_INPROJ")
_GDN_FUSED_INPROJ_MAX_ROWS = 8
_GDN_INPROJ_MODULES = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
_FUSED_GDN_DECODE = _env_flag("MLX_QWEN4_FUSED_GDN_DECODE")
_FUSED_GDN_DECODE_MODES = ("stock", "fused", "fused_outproj")
_FUSED_GDN_VERIFY = _env_flag("MLX_QWEN4_FUSED_GDN_VERIFY")
_FUSED_GDN_REPLAY_ROLLBACK = _env_flag("MLX_QWEN4_FUSED_GDN_REPLAY_ROLLBACK")
_FUSED_GDN_DYNAMIC_ACCEPT = _env_flag("MLX_QWEN4_FUSED_GDN_DYNAMIC_ACCEPT")
_FUSED_GDN_CATCHUP_DEFAULT = _env_flag("MLX_QWEN4_FUSED_GDN_CATCHUP")
_FUSED_GDN_CATCHUP_SCOPE = ContextVar("qwen4_fused_gdn_catchup_scope", default=False)
_FUSED_GDN_VERIFY_MODES = ("stock", "fused")
_FUSED_GDN_REPLAY_ROLLBACK_MODES = ("snapshots", "compact")


@contextmanager
def _gdn_catchup_scope(enabled: bool):
    token = _FUSED_GDN_CATCHUP_SCOPE.set(bool(enabled))
    try:
        yield
    finally:
        _FUSED_GDN_CATCHUP_SCOPE.reset(token)


_VERIFY_FALLBACK_REASON_LIMIT = 32
_DECODE_FALLBACK_REASON_LIMIT = 32
_SHAPE_STABLE_SHORT_FORWARD = _env_flag("MLX_QWEN4_SHAPE_STABLE_SHORT_FORWARD")
_EAGER_DISPATCH = _env_flag("MLX_QWEN4_EAGER_DISPATCH")
_EAGER_DISPATCH_MAX_ROWS = max(
    1, int(os.environ.get("MLX_QWEN4_EAGER_DISPATCH_MAX_ROWS", "64"))
)
_EAGER_DISPATCH_STRIDE = max(
    1, int(os.environ.get("MLX_QWEN4_EAGER_DISPATCH_STRIDE", "2"))
)


def qwen4_eager_dispatch_status() -> dict:
    """Return the selected eager-dispatch policy and bounded engagement counts."""
    counters = _lv.counters()
    return {
        "enabled": bool(_EAGER_DISPATCH),
        "max_rows": int(_EAGER_DISPATCH_MAX_ROWS),
        "stride": int(_EAGER_DISPATCH_STRIDE),
        "forwards": int(counters["eager_dispatch_forwards"]),
        "row_declines": int(counters["eager_dispatch_row_declines"]),
        "async_evals": int(counters["eager_async_evals"]),
    }


_QSA_DENSE_SHORTCIRCUIT = _env_flag("MLX_QWEN4_QSA_DENSE_SHORTCIRCUIT")
_QSA_FUSED_PROJ = _env_flag("MLX_QWEN4_QSA_FUSED_PROJ")


def _env_auto_flag(name: str):
    raw = os.environ.get(name, "auto").strip().lower()
    if raw in {"", "auto"}:
        return None
    if raw in {"1", "true", "on", "yes"}:
        return True
    if raw in {"0", "false", "off", "no"}:
        return False
    raise ValueError(f"{name} must be auto, 0/off, or 1/on; got {raw!r}")


_QSA_NAX_KERNEL = _env_auto_flag("MLX_QWEN4_QSA_NAX_KERNEL")
_QSA_NAX_DECODE = _env_flag("MLX_QWEN4_QSA_NAX_DECODE")
_QSA_NAX_DECODE_STATS_LOCK = threading.Lock()
_QSA_NAX_DECODE_STATS = Counter()
_QSA_STAGE1_KERNEL = _env_auto_flag("MLX_QWEN4_QSA_STAGE1_KERNEL")
_QSA_STAGE1_MIN_QUERY = int(os.environ.get("MLX_QWEN4_QSA_STAGE1_MIN_QUERY", "64"))
_QSA_STAGE1_MIN_PHYSICAL_KV = int(
    os.environ.get("MLX_QWEN4_QSA_STAGE1_MIN_PHYSICAL_KV", "65024")
)
_QSA_STAGE1_STATS_LOCK = threading.Lock()
_QSA_STAGE1_STATS = Counter()


def _qsa_stage1_admission_reason(query_width: int, physical_width: int) -> str | None:
    if _QSA_STAGE1_KERNEL is False:
        return "disabled"
    if int(query_width) < _QSA_STAGE1_MIN_QUERY:
        return "query_below_min"
    if int(physical_width) < _QSA_STAGE1_MIN_PHYSICAL_KV:
        return "context_below_min"
    return None


_QSA_NAX_MIN_QUERY = int(os.environ.get("MLX_QWEN4_QSA_NAX_MIN_QUERY", "64"))
_QSA_NAX_AUTO_MIN_PHYSICAL_KV = int(
    os.environ.get("MLX_QWEN4_QSA_NAX_AUTO_MIN_PHYSICAL_KV", "16384")
)
_QSA_NAX_STATS_LOCK = threading.Lock()
_QSA_NAX_STATS = Counter()
_QSA_NAX_DEVICE_SUPPORTED = None


@dataclass(frozen=True)
class QSANAXAdmission:
    engage: bool
    reason: str


def _qsa_nax_mode() -> str:
    if _QSA_NAX_KERNEL is None:
        return "auto"
    return "on" if _QSA_NAX_KERNEL else "off"


def _qsa_nax_device_supported() -> bool:
    """Restrict automatic admission to the measured Apple M5 envelope."""
    global _QSA_NAX_DEVICE_SUPPORTED
    if _QSA_NAX_DEVICE_SUPPORTED is not None:
        return _QSA_NAX_DEVICE_SUPPORTED
    try:
        supported = "M5" in str(mx.device_info().get("device_name", ""))
    except (AttributeError, RuntimeError, TypeError):
        supported = False
    _QSA_NAX_DEVICE_SUPPORTED = supported
    return supported


def decide_qsa_nax_admission(
    selection,
    *,
    training: bool,
    layout_ok: bool,
    cache_layout_ok: bool = True,
    device_supported: bool | None = None,
    kernel_available: bool | None = None,
) -> QSANAXAdmission:
    """Resolve the NAX route without mutating QSA state or building a mask."""
    mode = _qsa_nax_mode()
    if mode == "off":
        return QSANAXAdmission(False, "explicit_off")
    if training:
        return QSANAXAdmission(False, "training")
    if not cache_layout_ok:
        return QSANAXAdmission(False, "unsupported_cache_layout")
    if selection.kind != "explicit":
        return QSANAXAdmission(False, f"selection_{selection.kind}")
    if selection.length < _QSA_NAX_MIN_QUERY:
        return QSANAXAdmission(False, "query_below_min")
    if not layout_ok:
        return QSANAXAdmission(False, "unsupported_layout")
    if mode == "auto":
        if selection.batch != 1:
            return QSANAXAdmission(False, "auto_batch_gt_one")
        if selection.physical_width < _QSA_NAX_AUTO_MIN_PHYSICAL_KV:
            return QSANAXAdmission(False, "auto_context_below_crossover")
    supported = (
        _qsa_nax_device_supported()
        if device_supported is None
        else bool(device_supported)
    )
    if not supported:
        return QSANAXAdmission(False, "unsupported_device")
    available = (
        nax_kernel_available() if kernel_available is None else bool(kernel_available)
    )
    if not available:
        return QSANAXAdmission(False, "kernel_unavailable")
    return QSANAXAdmission(True, f"engaged_{mode}")


def _record_qsa_nax_admission(selection, decision: QSANAXAdmission) -> None:
    global _QSA_NAX_LAST_DECISION
    receipt = {
        "engaged": decision.engage,
        "reason": decision.reason,
        "batch": int(selection.batch),
        "query_width": int(selection.length),
        "physical_kv": int(selection.physical_width),
    }
    with _QSA_NAX_STATS_LOCK:
        _QSA_NAX_STATS[decision.reason] += 1
        _QSA_NAX_LAST_DECISION = receipt


_QSA_GATHER_KV = _env_flag("MLX_QWEN4_QSA_GATHER_KV")
_QSA_GATHER_TILE_ROWS = max(
    1, int(os.environ.get("MLX_QWEN4_QSA_GATHER_TILE_ROWS", "1"))
)
_QSA_GATHER_MIN_CONTEXT = max(
    0, int(os.environ.get("MLX_QWEN4_QSA_GATHER_MIN_CONTEXT", "0"))
)
_QSA_GATHER_MAX_CONTEXT = max(
    0, int(os.environ.get("MLX_QWEN4_QSA_GATHER_MAX_CONTEXT", "0"))
)
_QSA_GATHER_MIN_QUERY = max(
    1, int(os.environ.get("MLX_QWEN4_QSA_GATHER_MIN_QUERY", "3"))
)
_QSA_GATHER_MAX_QUERY = max(
    1, int(os.environ.get("MLX_QWEN4_QSA_GATHER_MAX_QUERY", "8"))
)


def _record_qsa_nax_decode(*, engaged: bool, reason: str, context: int) -> None:
    global _QSA_NAX_DECODE_LAST_DECISION
    receipt = {"engaged": engaged, "reason": reason, "context": int(context)}
    with _QSA_NAX_DECODE_STATS_LOCK:
        _QSA_NAX_DECODE_STATS[reason] += 1
        _QSA_NAX_DECODE_LAST_DECISION = receipt


def _record_qsa_stage1(
    *, engaged: bool, reason: str, batch: int, query_width: int, blocks: int
) -> None:
    global _QSA_STAGE1_LAST_DECISION
    receipt = {
        "engaged": engaged,
        "reason": reason,
        "batch": int(batch),
        "query_width": int(query_width),
        "blocks": int(blocks),
    }
    with _QSA_STAGE1_STATS_LOCK:
        _QSA_STAGE1_STATS[reason] += 1
        _QSA_STAGE1_LAST_DECISION = receipt


def _table_matmul(table, x: mx.array) -> mx.array:
    (weight, scales, biases, group_size, bits, mode) = table
    if scales is None:
        return x @ weight.T
    return mx.quantized_matmul(
        x,
        weight,
        scales,
        biases,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )


def _valid_span_end(mask):
    """One past each row's last valid position, as a ``[B]`` vector.

    ``ArraysCache.make_mask`` builds either ``pos >= left_padding`` (leading
    pads, from a left-padded batched prefill) or ``pos < lengths`` (trailing
    pads, from a right-padded continuation or a ragged verify), so a row's
    valid positions are always ONE contiguous run and its end is all the PLE
    state updates need.  An all-pad row returns 0 and keeps its prior state.
    """
    length = mask.shape[1]
    if isinstance(mask, np.ndarray):
        return np.max(np.where(mask, np.arange(1, length + 1), 0), axis=1)
    return mx.max(mx.where(mask, mx.arange(1, length + 1), 0), axis=1)


def _row_tail(values, end, width):
    """Per-row trailing window of ``[prev(width), new]`` ending at ``end``.

    Both PLE states are stored as the last ``width`` entries of a
    ``width``-prefixed buffer, so row ``b``'s window is
    ``values[b, end[b] : end[b] + width]``.  With ``end == values.shape[1] -
    width`` this is exactly ``values[:, -width:]``, the unpadded update.
    """
    positions = end[:, None] + (
        np.arange(width) if isinstance(values, np.ndarray) else mx.arange(width)
    )
    if isinstance(values, np.ndarray):
        return np.take_along_axis(values, positions, axis=1)
    if values.ndim == 3:
        positions = mx.broadcast_to(
            positions[..., None], (values.shape[0], width, values.shape[2])
        )
    return mx.take_along_axis(values, positions, axis=1)


_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 11400714819323198485
_SPLITMIX_M1 = 13787848793156543929
_SPLITMIX_M2 = 10723151780598845931
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = value + _SPLITMIX_GAMMA & _MASK64
    value = (value ^ value >> 30) * _SPLITMIX_M1 & _MASK64
    value = (value ^ value >> 27) * _SPLITMIX_M2 & _MASK64
    return (value ^ value >> 31) & _MASK64


def _build_layer_multipliers(
    unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int
) -> list[int]:
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    return [
        2
        * (
            _splitmix64(base_seed + _SPLITMIX_GAMMA * (index + 1) & _MASK64)
            % half_bound
        )
        + 1
        for index in range(ngram_size)
    ]


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all((value % divisor for divisor in range(3, math.isqrt(value) + 1, 2)))


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


@dataclass
class TextModelArgs(BaseModelArgs):
    model_type: str = "qwen4_exp_text"
    hidden_size: int = 2560
    intermediate_size: int = 0
    num_hidden_layers: int = 48
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    vocab_size: int = 248320
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-06
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
    output_gate_type: Optional[str] = "sigmoid"
    linear_num_value_heads: int = 48
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    full_attention_interval: int = 4
    layer_types: Optional[List[str]] = None
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    decoder_sparse_step: int = 1
    norm_topk_prob: Optional[bool] = True
    hc_count: int = 4
    hc_lowrank: int = 320
    ple_layer_ids: List[int] = field(default_factory=list)
    ple_embed_dim: Optional[int] = None
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20000000
    make_ngram_vocab_size_divisible_by: int = 128
    seed: Optional[int] = 1234
    split_ngram_parts: int = 128
    eos_token_id: Union[int, List[int]] = 248044
    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    mtp_num_hidden_layers: int = 1
    rope_parameters: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            "type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
        }
    )
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10000000.0
    rope_scaling: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        self.seed = 1234 if self.seed is None else self.seed
        self.norm_topk_prob = (
            True if self.norm_topk_prob is None else self.norm_topk_prob
        )
        self.ple_embed_dim = (
            self.hidden_size if self.ple_embed_dim is None else self.ple_embed_dim
        )
        self.ple_layer_ids = sorted(set(self.ple_layer_ids or []))
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention"
                if (i + 1) % self.full_attention_interval
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        if self.rope_parameters:
            rope = dict(self.rope_parameters)
            if "type" not in rope and "rope_type" in rope:
                rope["type"] = rope["rope_type"]
            self.partial_rotary_factor = rope.get("partial_rotary_factor", 0.25)
            self.rope_theta = rope.get("rope_theta", 10000000.0)
            self.rope_scaling = rope
        self._validate()

    def _validate(self):
        if self.hc_count <= 1:
            raise ValueError("Qwen4-Exp requires more than one hyper-connection stream")
        if self.indexer_kv_heads != 1:
            raise ValueError("Qwen4-Exp QSA requires indexer_kv_heads=1")
        if self.indexer_budget % self.indexer_compress_ratio:
            raise ValueError(
                "indexer_budget must be divisible by indexer_compress_ratio"
            )
        ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if self.ple_embed_dim % ngram_heads:
            raise ValueError("ple_embed_dim must be divisible by the n-gram head count")
        for layer_id in self.ple_layer_ids:
            if not 1 <= layer_id <= self.num_hidden_layers:
                raise ValueError(f"invalid one-indexed PLE layer id {layer_id}")
            if self.layer_types[layer_id - 1] != "linear_attention":
                raise ValueError("PLE is only defined on linear-attention layers")


class GroupRMSNorm(nn.Module):
    """Zero-centred checkpoint RMSNorm, optionally normalised per H stream."""

    def __init__(self, dim: int, group_size: Optional[int], eps: float):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.group_size = group_size
        self.eps = eps

    def _use_fast(self, x: mx.array) -> bool:
        """Fast path only for a query width inside the accepted class.

        The width is read off the SEQUENCE axis of the caller's array, before
        the grouped reshape splits the feature axis -- after the reshape
        ``shape[-2]`` is the group count, which is not a width at all.  A
        rank-1 input has no sequence axis and counts as width 1.
        """
        if not _RMSNORM_FAST:
            return False
        if _RMSNORM_FAST_WIDTH_OVERRIDE is not None:
            width = _RMSNORM_FAST_WIDTH_OVERRIDE
        else:
            width = x.shape[-2] if x.ndim >= 2 else 1
        return width <= _RMSNORM_FAST_MAX_WIDTH

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        xf = x.astype(mx.float32)
        if self.group_size is not None:
            xf = xf.reshape(*xf.shape[:-1], -1, self.group_size)
        if self._use_fast(x):
            out = mx.fast.rms_norm(xf, None, self.eps)
        else:
            out = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + self.eps)
        if self.group_size is not None:
            out = out.reshape(*x.shape)
        return (out * self.weight.astype(mx.float32)).astype(dtype)


class RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float, activation: str):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps
        self.activation = activation

    def __call__(self, hidden_states: mx.array, gate: mx.array) -> mx.array:
        dtype = hidden_states.dtype
        x = mx.fast.rms_norm(hidden_states, self.weight, self.eps).astype(mx.float32)
        g = gate.astype(mx.float32)
        g = mx.sigmoid(g) if self.activation == "sigmoid" else nn.silu(g)
        return (x * g).astype(dtype)


class GatedDeltaNet(Qwen35GatedDeltaNet):
    def __init__(self, args: TextModelArgs):
        super().__init__(args)
        self.norm = RMSNormGated(
            args.linear_value_head_dim,
            args.rms_norm_eps,
            args.output_gate_type or args.hidden_act,
        )
        self.fused_gdn_decode_mode = "fused" if _FUSED_GDN_DECODE else "stock"
        self.fused_gdn_decode_calls = 0
        self.fused_gdn_outproj_calls = 0
        self.fused_gdn_decode_fallbacks = 0
        self.fused_gdn_decode_last_fallback = None
        object.__setattr__(self, "fused_gdn_decode_fallback_reasons", {})
        object.__setattr__(self, "_gdn_inproj_fused_cache", None)
        self.gdn_fused_inproj = _GDN_FUSED_INPROJ
        self.gdn_fused_inproj_calls = 0
        self.fused_gdn_verify_mode = "fused" if _FUSED_GDN_VERIFY else "stock"
        self.fused_gdn_verify_calls = 0
        self.fused_gdn_verify_fallbacks = 0
        self.fused_gdn_verify_last_fallback = None
        self.fused_gdn_replay_rollback_mode = (
            "compact" if _FUSED_GDN_REPLAY_ROLLBACK else "snapshots"
        )
        self.fused_gdn_replay_verify_calls = 0
        self.fused_gdn_replay_rollback_calls = 0
        self.fused_gdn_replay_rollback_tokens = 0
        self.fused_gdn_dynamic_accept = _FUSED_GDN_DYNAMIC_ACCEPT
        self.fused_gdn_replay_dynamic_rollback_calls = 0
        self.fused_gdn_replay_fallbacks = 0
        self.fused_gdn_catchup_calls = 0
        self.fused_gdn_catchup_fallbacks = 0
        self.fused_gdn_catchup_last_fallback = None
        object.__setattr__(self, "fused_gdn_verify_fallback_reasons", {})
        object.__setattr__(self, "fused_gdn_replay_fallback_reasons", {})
        object.__setattr__(
            self, "_fused_gdn_outproj_control", mx.zeros((64,), mx.uint32)
        )
        self._fused_gdn_outproj_epoch = 0

    def _normalize_qk(self, q, k):
        """Preserve Qwen4-Exp's direct L2 materialization boundaries."""
        q = q * mx.rsqrt(mx.sum(mx.square(q), axis=-1, keepdims=True) + 1e-06)
        k = k * mx.rsqrt(mx.sum(mx.square(k), axis=-1, keepdims=True) + 1e-06)
        return (q * k.shape[-1] ** (-0.5), k)

    def _gated_delta_update(self, q, k, v, a, b, state, mask, use_kernel):
        return gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=use_kernel,
            beta_input_dtype=True,
        )

    def set_fused_gdn_decode_mode(self, mode: str):
        """Select the decode implementation without touching resident arrays."""
        if mode not in _FUSED_GDN_DECODE_MODES:
            raise ValueError(
                f"unknown fused GDN decode mode {mode!r}; expected one of {_FUSED_GDN_DECODE_MODES}"
            )
        self.fused_gdn_decode_mode = mode

    def _fused_gdn_fallback(self, reason: str):
        self.fused_gdn_decode_fallbacks += 1
        self.fused_gdn_decode_last_fallback = reason
        reasons = self.fused_gdn_decode_fallback_reasons
        if reason not in reasons and len(reasons) >= _DECODE_FALLBACK_REASON_LIMIT:
            reason = "other"
        reasons[reason] = reasons.get(reason, 0) + 1
        return None

    def set_fused_gdn_verify_mode(self, mode: str):
        """Select the speculative-verify implementation; independent of decode."""
        if mode not in _FUSED_GDN_VERIFY_MODES:
            raise ValueError(
                f"unknown fused GDN verify mode {mode!r}; expected one of {_FUSED_GDN_VERIFY_MODES}"
            )
        self.fused_gdn_verify_mode = mode

    def _fused_gdn_verify_fallback(self, reason: str):
        self.fused_gdn_verify_fallbacks += 1
        self.fused_gdn_verify_last_fallback = reason
        reasons = self.fused_gdn_verify_fallback_reasons
        if reason not in reasons and len(reasons) >= _VERIFY_FALLBACK_REASON_LIMIT:
            reason = "other"
        reasons[reason] = reasons.get(reason, 0) + 1
        return None

    def set_fused_gdn_replay_rollback_mode(self, mode: str):
        """Select snapshot or compact rollback without changing verify mode."""
        if mode not in _FUSED_GDN_REPLAY_ROLLBACK_MODES:
            raise ValueError(
                f"unknown fused GDN replay rollback mode {mode!r}; "
                f"expected one of {_FUSED_GDN_REPLAY_ROLLBACK_MODES}"
            )
        self.fused_gdn_replay_rollback_mode = mode

    def set_fused_gdn_dynamic_accept(self, enabled: bool):
        """Select device-count compact rollback; applies to later verifies."""
        if type(enabled) is not bool:
            raise ValueError("fused GDN dynamic accept must be a boolean")
        self.fused_gdn_dynamic_accept = enabled

    def _fused_gdn_replay_fallback(self, reason: str):
        self.fused_gdn_replay_fallbacks += 1
        reasons = self.fused_gdn_replay_fallback_reasons
        if reason not in reasons and len(reasons) >= _VERIFY_FALLBACK_REASON_LIMIT:
            reason = "other"
        reasons[reason] = reasons.get(reason, 0) + 1
        return self._fused_gdn_verify_fallback(f"compact replay: {reason}")

    def _fused_gdn_catchup_fallback(self, reason: str):
        self.fused_gdn_catchup_fallbacks += 1
        self.fused_gdn_catchup_last_fallback = reason
        return None

    def _try_fused_verify(self, qkv, z, b, a, mask, cache, *, catchup=False):
        """Fuse a B=1 speculative verify block and record its restore points.

        The generic path recomputes the recurrence over the first ``m`` tokens
        on rejection. The normal fused path records per-token snapshots. The
        compact mode records correction inputs and reconstructs a
        single state only when a partial acceptance actually rolls back.
        """
        compact_replay = bool(
            not catchup and self.fused_gdn_replay_rollback_mode == "compact"
        )
        fallback = self._fused_gdn_catchup_fallback if catchup else (
            self._fused_gdn_replay_fallback
            if compact_replay
            else self._fused_gdn_verify_fallback
        )
        if not catchup and self.fused_gdn_verify_mode == "stock":
            return fallback("fused verify disabled") if compact_replay else None
        if cache is None or cache[0] is None or cache[1] is None:
            return fallback("uninitialized cache")
        describe = getattr(cache, "rollback_spans", None)
        if catchup:
            if not callable(describe):
                return fallback("cache lacks rollback geometry")
        elif not callable(describe) or not callable(
            getattr(cache, "record_rollback", None)
        ):
            return fallback("cache lacks rollback records")
        steps = int(qkv.shape[1])
        spans = describe(steps, mask)
        admission = admit_qwen4_fused_gdn_verify(
            qkv=qkv,
            z=z,
            b=b,
            a=a,
            conv_state=cache[0],
            recurrent_state=cache[1],
            conv_weight=self.conv1d.weight,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            norm_weight=self.norm.weight,
            mask=mask,
            spans=spans,
            speculating=bool(getattr(cache, "speculating", False)),
            catchup=bool(catchup),
            training=bool(self.training),
            sharded=self.sharding_group is not None,
            num_key_heads=self.num_k_heads,
            num_value_heads=self.num_v_heads,
            key_head_dim=self.head_k_dim,
            value_head_dim=self.head_v_dim,
            conv_kernel=self.conv_kernel_size,
            gate_activation=self.norm.activation,
        )
        if not admission.accepted:
            return fallback(admission.reason)
        if not fused_gdn_runtime_supported():
            return fallback("Metal runtime unavailable")
        try:
            probe = (
                probe_qwen4_fused_gdn_catchup
                if catchup
                else (
                    probe_qwen4_fused_gdn_replay_verify
                    if compact_replay
                    else probe_qwen4_fused_gdn_verify
                )
            )
            if compact_replay and self.fused_gdn_dynamic_accept:
                threadgroup_y = probe(qkv.dtype, steps, dynamic_accept=True)
            else:
                threadgroup_y = probe(qkv.dtype, steps)
            if threadgroup_y is None:
                return fallback("Metal kernel probe declined")
            kernel = qwen4_fused_gdn_catchup if catchup else (
                qwen4_fused_gdn_replay_verify
                if compact_replay
                else qwen4_fused_gdn_verify
            )
            outputs = kernel(
                qkv,
                z,
                b,
                a,
                cache[0],
                self.conv1d.weight,
                self.A_log,
                self.dt_bias,
                cache[1],
                self.norm.weight,
                self.norm.eps,
                threadgroup_y=threadgroup_y,
            )
            (out, conv_state, recurrent_state) = outputs[:3]
            if not catchup:
                if compact_replay:
                    (replay_keys, replay_corrections, replay_decay) = outputs[3:]
                else:
                    (state_snapshots, conv_snapshots) = outputs[3:]
        except Exception as exc:
            return fallback(f"Metal kernel dispatch failed: {type(exc).__name__}")
        if not catchup:
            per_row_fn = None
            if compact_replay:
                initial_conv = cache[0]
                initial_state = cache[1]
                keep = self.conv_kernel_size - 1

                def _rollback(
                    m,
                    conv=initial_conv,
                    state=initial_state,
                    raw_qkv=qkv,
                    keys=replay_keys,
                    corrections=replay_corrections,
                    decay=replay_decay,
                    nk=keep,
                    ty=threadgroup_y,
                ):
                    restored_conv = mx.contiguous(
                        mx.concatenate([conv, raw_qkv], axis=1)[:, m : m + nk]
                    )
                    restored_state = qwen4_fused_gdn_reconstruct(
                        state,
                        keys,
                        corrections,
                        decay,
                        m,
                        threadgroup_y=ty,
                    )
                    self.fused_gdn_replay_rollback_calls += 1
                    self.fused_gdn_replay_rollback_tokens += int(m)
                    return [restored_conv, restored_state]

                if self.fused_gdn_dynamic_accept:
                    _rollback, per_row_fn = self._dynamic_replay_rollback(
                        initial_conv,
                        initial_state,
                        qkv,
                        replay_keys,
                        replay_corrections,
                        replay_decay,
                        threadgroup_y,
                    )

            else:

                def _rollback(m, conv=conv_snapshots, state=state_snapshots):
                    return [mx.contiguous(conv[:, m - 1]), state[:, m - 1]]

            if per_row_fn is None:
                cache.record_rollback(steps, _rollback, [cache[0], cache[1]])
            else:
                cache.record_rollback(
                    steps, _rollback, [cache[0], cache[1]], per_row_fn=per_row_fn
                )
        cache[0] = conv_state
        cache[1] = recurrent_state
        cache.advance(steps)
        if catchup:
            self.fused_gdn_catchup_calls += 1
            self.fused_gdn_catchup_last_fallback = None
        else:
            self.fused_gdn_verify_calls += 1
            self.fused_gdn_verify_last_fallback = None
            if compact_replay:
                self.fused_gdn_replay_verify_calls += 1
        return self.out_proj(out)

    def _dynamic_replay_rollback(
        self, conv, state, raw_qkv, keys, corrections, decay, threadgroup_y
    ):
        """Rollback closures that take the accepted prefix as a device count.

        ``rows(lengths)`` accepts a host list or an ``mx.array`` (one count per
        row) and rebuilds every row in one reconstruct dispatch; the conv
        window is a per-row gather, so an array count is never read back.
        ``fn(m)`` keeps the host-int contract (``ExactRollbackBoundary`` and
        fan-out replay it with an int) by routing through the same kernel.
        """
        keep = self.conv_kernel_size - 1
        tape_steps = int(decay.shape[1])
        combined = mx.concatenate([conv, raw_qkv], axis=1)

        def rows(lengths):
            if isinstance(lengths, mx.array):
                ends = lengths.reshape(-1).astype(mx.int32)
                tokens = None
            else:
                lengths = [int(value) for value in lengths]
                ends = mx.array(lengths, dtype=mx.int32)
                tokens = sum(lengths)
            ends = mx.broadcast_to(ends, (combined.shape[0],))
            restored_conv = mx.contiguous(_row_tail(combined, ends, keep))
            restored_state = qwen4_fused_gdn_reconstruct(
                state, keys, corrections, decay, ends, threadgroup_y=threadgroup_y
            )
            self.fused_gdn_replay_rollback_calls += 1
            self.fused_gdn_replay_dynamic_rollback_calls += 1
            if tokens is not None:
                self.fused_gdn_replay_rollback_tokens += tokens
            return [restored_conv, restored_state]

        def fn(m):
            if isinstance(m, mx.array):
                return rows(m)
            return rows([validate_qwen4_gdn_replay_acceptance(m, tape_steps)])

        return fn, rows

    def _try_fused_decode(self, qkv, z, b, a, mask, cache):
        if cache is not None and qkv.shape[1] > 1:
            speculating = bool(getattr(cache, "speculating", False))
            if speculating or _FUSED_GDN_CATCHUP_SCOPE.get():
                return self._try_fused_verify(
                    qkv, z, b, a, mask, cache, catchup=not speculating
                )
        if self.fused_gdn_decode_mode == "stock":
            return None
        if cache is None or cache[0] is None or cache[1] is None:
            return self._fused_gdn_fallback("uninitialized cache")
        describe = getattr(cache, "rollback_spans", None)
        spans = describe(int(qkv.shape[1]), mask) if callable(describe) else ()
        admission = admit_qwen4_fused_gdn_decode(
            qkv=qkv,
            z=z,
            b=b,
            a=a,
            conv_state=cache[0],
            recurrent_state=cache[1],
            conv_weight=self.conv1d.weight,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            norm_weight=self.norm.weight,
            mask=mask,
            spans=spans,
            speculating=bool(getattr(cache, "speculating", False)),
            training=bool(self.training),
            sharded=self.sharding_group is not None,
            num_key_heads=self.num_k_heads,
            num_value_heads=self.num_v_heads,
            key_head_dim=self.head_k_dim,
            value_head_dim=self.head_v_dim,
            conv_kernel=self.conv_kernel_size,
            gate_activation=self.norm.activation,
        )
        if not admission.accepted:
            return self._fused_gdn_fallback(admission.reason)
        if not fused_gdn_runtime_supported():
            return self._fused_gdn_fallback("Metal runtime unavailable")
        try:
            threadgroup_y = probe_qwen4_fused_gdn_decode(qkv.dtype)
            if threadgroup_y is None:
                return self._fused_gdn_fallback("Metal kernel probe declined")
            if self.fused_gdn_decode_mode == "fused_outproj":
                outproj_admission = admit_qwen4_gdn_outproj(self.out_proj, z)
                if not outproj_admission.accepted:
                    return self._fused_gdn_fallback(outproj_admission.reason)
                self._fused_gdn_outproj_epoch += 1
                (out, conv_state, recurrent_state) = qwen4_fused_gdn_decode_outproj(
                    qkv,
                    z,
                    b,
                    a,
                    cache[0],
                    self.conv1d.weight,
                    self.A_log,
                    self.dt_bias,
                    cache[1],
                    self.norm.weight,
                    self.norm.eps,
                    self.out_proj.weight,
                    self.out_proj.scales,
                    self.out_proj.biases,
                    self._fused_gdn_outproj_control,
                    self._fused_gdn_outproj_epoch,
                    output_dim=self.hidden_size,
                    output_group_size=self.out_proj.group_size,
                )
            else:
                (out, conv_state, recurrent_state) = qwen4_fused_gdn_decode(
                    qkv,
                    z,
                    b,
                    a,
                    cache[0],
                    self.conv1d.weight,
                    self.A_log,
                    self.dt_bias,
                    cache[1],
                    self.norm.weight,
                    self.norm.eps,
                    threadgroup_y=threadgroup_y,
                )
        except Exception as exc:
            return self._fused_gdn_fallback(
                f"Metal kernel dispatch failed: {type(exc).__name__}"
            )
        cache[0] = conv_state
        cache[1] = recurrent_state
        cache.advance(1)
        self.fused_gdn_decode_calls += 1
        if self.fused_gdn_decode_mode == "fused_outproj":
            self.fused_gdn_outproj_calls += 1
        self.fused_gdn_decode_last_fallback = None
        if self.fused_gdn_decode_mode == "fused_outproj":
            return out
        return self.out_proj(out)

    def set_gdn_fused_inproj(self, enabled: bool) -> bool:
        """Live-toggle the fused input-projection table for this layer."""
        self.gdn_fused_inproj = bool(enabled)
        return self.gdn_fused_inproj

    def _fused_inproj_table(self):
        """Lazy ``(table, split_points)`` for the four input projections.

        Returns ``None`` -- and stays returning ``None`` for these arrays --
        whenever the quartet is not concatenable: a non-quantized or biased
        projection, a mixed ``group_size``/``bits``/``mode``, a differing K
        or storage dtype, a sharded layer, or a layer whose split modules
        were already replaced by ``qwen3_5``'s in-place rewrite.
        """
        if self.sharding_group is not None:
            return None
        try:
            modules = [getattr(self, name) for name in _GDN_INPROJ_MODULES]
        except AttributeError:
            return None
        signatures = [_proj_signature(m) for m in modules]
        if any((s is None for s in signatures)):
            # A wrapped projection (a LoRA adapter, say) has no resident
            # weight to key on or concatenate. Drop any table built before
            # it was wrapped: the wrapper must run, and the table is dead.
            object.__setattr__(self, "_gdn_inproj_fused_cache", None)
            return None
        key = tuple((part for m in modules for part in _proj_identity(m)))
        cached = self._gdn_inproj_fused_cache
        if cached is not None and all(
            (new is old for (new, old) in zip(key, cached[0]))
        ):
            return cached[1]
        entry = None
        base = signatures[0]
        if (
            base is not None
            and base[0] == "quantized"
            and all((s == base for s in signatures))
        ):
            parts = [_proj_table(m) for m in modules]
            widths = [m["weight"].shape[0] for m in modules]
            k_packed = parts[0][0].shape[1]
            uniform = all(
                (
                    part[0].shape[1] == k_packed
                    and part[0].dtype == parts[0][0].dtype
                    and (part[1].dtype == parts[0][1].dtype)
                    and ((part[2] is None) == (parts[0][2] is None))
                    and (part[2] is None or part[2].dtype == parts[0][2].dtype)
                    for part in parts
                )
            )
            if uniform:
                check_materialization_budget(
                    sum((table_bytes(part) for part in parts)),
                    "GDN fused input projection",
                )
                bounds = []
                total = 0
                for width in widths[:-1]:
                    total += width
                    bounds.append(total)
                entry = (_concat_tables(parts, axis=0), tuple(bounds))
        object.__setattr__(self, "_gdn_inproj_fused_cache", (key, entry))
        return entry

    def _fused_input_projections(self, inputs: mx.array):
        """The four projection outputs from one matmul, or ``None``."""
        if not self.gdn_fused_inproj or self.training:
            return None
        if inputs.shape[0] * inputs.shape[1] > _GDN_FUSED_INPROJ_MAX_ROWS:
            return None
        entry = self._fused_inproj_table()
        if entry is None:
            return None
        (table, bounds) = entry
        self.gdn_fused_inproj_calls += 1
        return tuple(mx.split(_table_matmul(table, inputs), bounds, axis=-1))

    def _input_projections(self, inputs: mx.array):
        if not _GDN_SHAPE_STABLE_PROJECTIONS or inputs.shape[1] <= 1:
            fused = self._fused_input_projections(inputs)
            if fused is not None:
                return fused
            return super()._input_projections(inputs)
        per_token = [
            self._input_projections(inputs[:, i : i + 1])
            for i in range(inputs.shape[1])
        ]
        return tuple(
            (
                mx.concatenate([token[projection] for token in per_token], axis=1)
                for projection in range(4)
            )
        )


def qwen4_fused_gdn_stats(
    model: nn.Module,
    *,
    reset: bool = False,
    modules=None,
) -> dict[str, Any]:
    """Return graph-selection and fallback counters without host synchronization.

    ``fused_calls``/``fallbacks``/``last_fallbacks`` describe the single-token
    decode path; ``verify_*`` describes speculative verify, while ``replay_*``
    separately receipts compact-tape selection and actual partial rollback.

    ``last_fallbacks``/``verify_last_fallbacks`` are snapshots of each layer's
    *most recent* decline, which the next admitted call clears;
    ``decode_fallback_reasons``/``verify_fallback_reasons`` are the durable
    per-reason histograms of every decline since the layer was built (or since
    the last ``reset``), so a run that declines and then succeeds still
    receipts why it declined.

    ``reset`` zeroes every counter this function reports, on the model's
    layers, after reading them.
    """
    stats = {
        "fused_calls": 0,
        "fused_outproj_calls": 0,
        "fallbacks": 0,
        "last_fallbacks": {},
        "decode_fallback_reasons": {},
        "verify_calls": 0,
        "verify_fallbacks": 0,
        "verify_last_fallbacks": {},
        "verify_fallback_reasons": {},
        "replay_verify_calls": 0,
        "replay_rollback_calls": 0,
        "replay_rollback_tokens": 0,
        "replay_fallbacks": 0,
        "replay_fallback_reasons": {},
        "catchup_calls": 0,
        "catchup_fallbacks": 0,
        "catchup_last_fallbacks": {},
    }
    if modules is None:
        modules = (module for _, module in model.named_modules())
    for module in modules:
        if not isinstance(module, GatedDeltaNet):
            continue
        stats["fused_calls"] += module.fused_gdn_decode_calls
        stats["fused_outproj_calls"] += module.fused_gdn_outproj_calls
        stats["fallbacks"] += module.fused_gdn_decode_fallbacks
        reason = module.fused_gdn_decode_last_fallback
        if reason is not None:
            stats["last_fallbacks"][reason] = stats["last_fallbacks"].get(reason, 0) + 1
        stats["verify_calls"] += module.fused_gdn_verify_calls
        stats["verify_fallbacks"] += module.fused_gdn_verify_fallbacks
        reason = module.fused_gdn_verify_last_fallback
        if reason is not None:
            stats["verify_last_fallbacks"][reason] = (
                stats["verify_last_fallbacks"].get(reason, 0) + 1
            )
        durable = stats["decode_fallback_reasons"]
        for reason, count in module.fused_gdn_decode_fallback_reasons.items():
            durable[reason] = durable.get(reason, 0) + count
        durable = stats["verify_fallback_reasons"]
        for reason, count in module.fused_gdn_verify_fallback_reasons.items():
            durable[reason] = durable.get(reason, 0) + count
        stats["replay_verify_calls"] += module.fused_gdn_replay_verify_calls
        stats["replay_rollback_calls"] += module.fused_gdn_replay_rollback_calls
        stats["replay_rollback_tokens"] += module.fused_gdn_replay_rollback_tokens
        if module.fused_gdn_dynamic_accept:
            # Reported only when selected, so default diagnostics are unchanged.
            stats["replay_dynamic_rollback_calls"] = (
                stats.get("replay_dynamic_rollback_calls", 0)
                + module.fused_gdn_replay_dynamic_rollback_calls
            )
        stats["replay_fallbacks"] += module.fused_gdn_replay_fallbacks
        durable = stats["replay_fallback_reasons"]
        for reason, count in module.fused_gdn_replay_fallback_reasons.items():
            durable[reason] = durable.get(reason, 0) + count
        stats["catchup_calls"] += module.fused_gdn_catchup_calls
        stats["catchup_fallbacks"] += module.fused_gdn_catchup_fallbacks
        reason = module.fused_gdn_catchup_last_fallback
        if reason is not None:
            stats["catchup_last_fallbacks"][reason] = (
                stats["catchup_last_fallbacks"].get(reason, 0) + 1
            )
        if reset:
            module.fused_gdn_decode_calls = 0
            module.fused_gdn_outproj_calls = 0
            module.fused_gdn_decode_fallbacks = 0
            module.fused_gdn_decode_last_fallback = None
            module.fused_gdn_decode_fallback_reasons.clear()
            module.fused_gdn_verify_calls = 0
            module.fused_gdn_verify_fallbacks = 0
            module.fused_gdn_verify_last_fallback = None
            module.fused_gdn_verify_fallback_reasons.clear()
            module.fused_gdn_replay_verify_calls = 0
            module.fused_gdn_replay_rollback_calls = 0
            module.fused_gdn_replay_rollback_tokens = 0
            module.fused_gdn_replay_dynamic_rollback_calls = 0
            module.fused_gdn_replay_fallbacks = 0
            module.fused_gdn_replay_fallback_reasons.clear()
            module.fused_gdn_catchup_calls = 0
            module.fused_gdn_catchup_fallbacks = 0
            module.fused_gdn_catchup_last_fallback = None
    return stats


class Qwen4ArraysCache(ArraysCache):
    """Four-state PLE+GDN cache with one atomic speculative rollback."""

    # Slot 3 is the PLE n-gram token history. The model reads None there as
    # an all-EOS history (``_ple_embedding``), not zeros, so a cold row that
    # joins warm rows must be seeded with EOS. ``make_cache`` sets the id.
    PLE_HISTORY_SLOT = 3

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        instance._ple_rollback = None
        instance.ple_history_fill = None
        return instance

    def _adopt_empty_fill(self, caches):
        if self.ple_history_fill is None:
            self.ple_history_fill = next(
                (
                    c.ple_history_fill
                    for c in caches
                    if getattr(c, "ple_history_fill", None) is not None
                ),
                None,
            )

    def _empty_slot(self, slot, shape, dtype):
        if slot != self.PLE_HISTORY_SLOT:
            return super()._empty_slot(slot, shape, dtype)
        if self.ple_history_fill is None:
            raise RuntimeError(
                "Qwen4ArraysCache: a cold row joined warm rows but no PLE history fill id is known; zeros would seed token-0 history instead of the EOS history the model assumes for an empty slot."
            )
        return mx.full(shape, self.ple_history_fill, dtype=dtype)

    def start_speculation(self, rollback_window=None):
        self._ple_rollback = None
        super().start_speculation(rollback_window)

    def stop_speculation(self):
        self._ple_rollback = None
        super().stop_speculation()

    def _clear_staged_rollback(self):
        """Drop a PLE half staged by a forward that never reached GDN.

        ``ArraysCache._invalidate_rollbacks`` calls this on every membership
        change; without it the stale closure would restore old-batch tensors
        into the new membership on the next trim.
        """
        self._ple_rollback = None

    def _refuse_pending_ple(self, who: str):
        """A staged-but-unrecorded PLE half makes the span unrestorable.

        The rewind walks RECORDS, so a forward whose PLE half never reached
        ``record_rollback`` is invisible to it: with older records on the
        stack a trim would silently take its tokens out of those instead of
        this forward, restoring the wrong state. Fail here instead.
        """
        if self._ple_rollback is not None:
            raise RuntimeError(
                f"{who}: a Qwen4 forward staged a PLE rollback that GatedDeltaNet never recorded, so that span's PLE and GDN halves cannot be restored together. The two stage on the same geometry test, so this means the forward was interrupted between them (qwen4_exp.py, qwen3_5.py)."
            )

    def is_trimmable(self):
        return super().is_trimmable() and self._ple_rollback is None

    def trim(self, n):
        self._refuse_pending_ple("Qwen4ArraysCache.trim")
        return super().trim(n)

    def preflight_ragged_trim(self, n, *, validate: bool = True):
        self._refuse_pending_ple("Qwen4ArraysCache.trim_ragged")
        return super().preflight_ragged_trim(n, validate=validate)

    def stage_ple_rollback(self, num_tokens, fn, snapshot, *, per_row_fn=None):
        if self._ple_rollback is not None:
            raise RuntimeError(
                "Qwen4 PLE rollback was staged twice without a GDN record in between. PLE and GDN roll back as ONE record, so a forward that stages the PLE half must reach GatedDeltaNet's record_rollback in the same forward."
            )
        self._ple_rollback = (num_tokens, fn, snapshot, per_row_fn)

    def record_rollback(self, num_tokens, fn, snapshot, *, per_row_fn=None):
        staged = self._ple_rollback
        self._ple_rollback = None
        if staged is None:
            return super().record_rollback(
                num_tokens, fn, snapshot, per_row_fn=per_row_fn
            )
        (ple_tokens, ple_fn, ple_snapshot, ple_per_row) = staged
        if ple_tokens != num_tokens:
            raise RuntimeError(
                f"Qwen4 PLE/GDN rollback span mismatch: {ple_tokens} != {num_tokens}"
            )

        def combined(m):
            return list(fn(m)) + list(ple_fn(m))

        rows = None
        if per_row_fn is not None and ple_per_row is not None:

            def rows(lengths):
                return list(per_row_fn(lengths)) + list(ple_per_row(lengths))

        return super().record_rollback(
            num_tokens, combined, list(snapshot) + list(ple_snapshot), per_row_fn=rows
        )

    def extract(self, idx):
        cache = type(self)(len(self.cache))
        cache.ple_history_fill = self.ple_history_fill
        cache.cache = [
            None if value is None else mx.contiguous(value[idx : idx + 1])
            for value in self.cache
        ]
        if idx < len(self._checkpoints):
            cache._checkpoints = [list(self._checkpoints[idx])]
        cache._rollback_invalid_reason = "extract() left the batch behind"
        return cache


class GatedResidual(nn.Module):
    def __init__(self, args: TextModelArgs, use_combine: bool = True):
        super().__init__()
        self.hc_count = args.hc_count
        self.hidden_size = args.hidden_size
        hc_hidden = self.hc_count * self.hidden_size
        self.hc_norm = GroupRMSNorm(hc_hidden, args.hidden_size, args.rms_norm_eps)
        self.input_mix_weight_down = nn.Linear(hc_hidden, args.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(args.hc_lowrank, hc_hidden, bias=False)
        if use_combine:
            self.block_inject_weight = nn.Linear(hc_hidden, self.hc_count, bias=False)

    def __call__(self, hyper_input: mx.array):
        if _GDN_SHAPE_STABLE_PROJECTIONS and hyper_input.shape[1] > 1:
            with _declared_width(hyper_input.shape[1]):
                tokens = [
                    self(hyper_input[:, index : index + 1])
                    for index in range(hyper_input.shape[1])
                ]
            if isinstance(tokens[0], tuple):
                return tuple(
                    (
                        mx.concatenate([token[field] for token in tokens], axis=1)
                        for field in range(len(tokens[0]))
                    )
                )
            return mx.concatenate(tokens, axis=1)
        glue = compile_glue_enabled()
        normed = self.hc_norm(hyper_input)
        gate_input = self.input_mix_weight_down(normed)
        weights = None
        if glue:
            weights = _run_glue(
                ("hyper_gate", self.hc_count),
                lambda: _build_hyper_gate(self.hc_count),
                gate_input,
            )
        if weights is None:
            weights = nn.silu(gate_input / self.hc_count)
        weights = mx.sigmoid(self.input_mix_weight_up(weights))
        weights = weights.reshape(*weights.shape[:-1], self.hc_count, self.hidden_size)
        streams = normed.reshape(*normed.shape[:-1], self.hc_count, self.hidden_size)
        mixed = None
        if glue:
            mixed = _run_glue(("hyper_mix",), _build_hyper_mix, weights, streams)
        if mixed is None:
            mixed = mx.mean(weights * streams, axis=-2)
        if not hasattr(self, "block_inject_weight"):
            return mixed
        inject = 2 * mx.sigmoid(self.block_inject_weight(normed) / self.hc_count)
        return (mixed, hyper_input, inject)


class ShardedEmbedding(nn.Module):
    """Row-sharded embedding that never concatenates the 128 PLE tensors.

    MLX safetensor arrays remain file-backed until selected.  Grouping the row
    ids by checkpoint shard therefore transfers only requested rows instead of
    materialising the ~102.4 GB BF16 table.
    """

    def __init__(self, vocab_size: int, dims: int, num_shards: int):
        super().__init__()
        if vocab_size % num_shards:
            raise ValueError("Qwen4-Exp PLE vocabulary must split evenly")
        self.vocab_size = vocab_size
        self.dims = dims
        self.num_shards = num_shards
        self.rows_per_shard = vocab_size // num_shards
        for index in range(num_shards):
            setattr(self, f"shard_{index}", nn.Embedding(self.rows_per_shard, dims))

    def lookup_numpy(self, indices: np.ndarray) -> mx.array:
        """Gather already-hosted row ids without a redundant MLX sync."""
        shape = indices.shape
        flat = np.asarray(indices, dtype=np.int64).reshape(-1)
        shard_ids = flat // self.rows_per_shard
        if _PLE_GATHER_CONCAT:
            pieces = []
            ordering = []
            for shard_index in np.unique(shard_ids):
                positions = np.flatnonzero(shard_ids == shard_index)
                local = flat[positions] - int(shard_index) * self.rows_per_shard
                pieces.append(
                    getattr(self, f"shard_{int(shard_index)}")(
                        mx.array(local, dtype=mx.int64)
                    )
                )
                ordering.append(positions)
            permutation = np.concatenate(ordering)
            inverse = np.empty_like(permutation)
            inverse[permutation] = np.arange(permutation.size)
            output = mx.take(mx.concatenate(pieces, axis=0), mx.array(inverse), axis=0)
            return output.reshape(*shape, self.dims)
        output = None
        for shard_index in np.unique(shard_ids):
            positions = np.flatnonzero(shard_ids == shard_index)
            local = flat[positions] - int(shard_index) * self.rows_per_shard
            values = getattr(self, f"shard_{int(shard_index)}")(
                mx.array(local, dtype=mx.int64)
            )
            if output is None:
                output = mx.zeros((flat.size, self.dims), dtype=values.dtype)
            output = output.at[mx.array(positions)].add(values)
        return output.reshape(*shape, self.dims)

    def __call__(self, indices: mx.array) -> mx.array:
        mx.eval(indices)
        return self.lookup_numpy(np.asarray(indices, dtype=np.int64))


class NGramEmbedding(nn.Module):
    def __init__(
        self,
        args: TextModelArgs,
        embedding_dim: int,
        layer_idx: int,
        ple_layer_index: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.ngram_size = args.ngram_size
        self.context_len = args.ngram_size - 1
        self.heads_per_ngram = args.heads_per_ngram
        self.ngram_heads = self.context_len * self.heads_per_ngram
        self.eos_token_id = (
            args.eos_token_id[0]
            if isinstance(args.eos_token_id, list)
            else args.eos_token_id
        )
        sizes = [
            _find_nth_prime_after(args.ngram_vocab_size_base - 1, i + 1)
            for i in range(self.ngram_heads)
        ]
        offsets = np.cumsum([0] + sizes[:-1], dtype=np.int64)
        total = sum(sizes)
        divisor = args.make_ngram_vocab_size_divisible_by
        padded = math.ceil(total / divisor) * divisor
        multipliers = _build_layer_multipliers(
            args.vocab_size, args.ngram_size, ple_layer_index, args.seed
        )
        self.layer_multipliers = mx.array(multipliers, dtype=mx.int64)
        self.ngram_heads_vocab_sizes = mx.array(sizes, dtype=mx.int64)
        self.ngram_heads_offsets = mx.array(offsets, dtype=mx.int64)
        self._np_constants = None
        self._np_constants_src = None
        self.ngram_embedding = ShardedEmbedding(
            padded, embedding_dim // self.ngram_heads, args.split_ngram_parts
        )
        self.hash_backend = os.getenv("MLX_QWEN4_PLE_HASH_BACKEND", "cpu")
        if self.hash_backend not in {"cpu", "routed_cpu", "metal", "metal_prefill"}:
            raise ValueError(
                "MLX_QWEN4_PLE_HASH_BACKEND must be cpu, routed_cpu, metal, or metal_prefill"
            )
        self.metal_hash_min_tokens = int(
            os.getenv("MLX_QWEN4_PLE_METAL_MIN_TOKENS", "1024")
        )
        if self.metal_hash_min_tokens < 1:
            raise ValueError("MLX_QWEN4_PLE_METAL_MIN_TOKENS must be positive")
        self._metal_hash_kernel = None

    def _hash_constants_numpy(self):
        """Host copies of the hash constants, tracking the live mx arrays.

        Keyed by object identity (the sources are kept referenced, so an id
        can never be recycled): any ``load_weights``/``update`` swap of the
        underlying arrays invalidates the snapshot on the next call.
        Concurrent prefetch snapshots are benign under the supported
        lifecycle (constants are immutable after load); a concurrent live
        ``load_weights`` would need synchronization - the two cache
        assignments below are not jointly atomic.
        """
        src = (
            self.layer_multipliers,
            self.ngram_heads_vocab_sizes,
            self.ngram_heads_offsets,
        )
        cached_src = self._np_constants_src
        if cached_src is None or any(
            (new is not old for (new, old) in zip(src, cached_src))
        ):
            self._np_constants = tuple(
                (np.asarray(value, dtype=np.int64) for value in src)
            )
            self._np_constants_src = src
        return self._np_constants

    def _shift_history_vectorized(self, history: np.ndarray) -> list:
        """Vectorized per-token history shift with per-segment EOS resets."""
        (batch, length) = history.shape
        positions = np.arange(length, dtype=np.int64)
        eos_at = np.where(history == self.eos_token_id, positions[None, :], -1)
        previous_eos = np.concatenate(
            [
                np.full((batch, 1), -1, dtype=np.int64),
                np.maximum.accumulate(eos_at, axis=1)[:, :-1],
            ],
            axis=1,
        )
        shifted = [history.copy()]
        for shift in range(1, self.ngram_size):
            rolled = np.full_like(history, self.eos_token_id)
            rolled[:, shift:] = history[:, :-shift]
            valid = positions[None, :] - shift > previous_eos
            shifted.append(np.where(valid, rolled, self.eos_token_id))
        return shifted

    def _ngram_ids_numpy(
        self,
        input_ids: mx.array,
        cache: Optional[ArraysCache] = None,
        mask: Optional[mx.array] = None,
        previous: Optional[np.ndarray] = None,
        record_sync: bool = True,
    ) -> np.ndarray:
        if record_sync:
            record_verify_sync("qwen4.ple.ids_eval")
        mx.eval(input_ids, mask)
        tokens = np.asarray(input_ids, dtype=np.int64)
        (batch, seq_len) = tokens.shape
        if previous is not None:
            previous = np.asarray(previous, dtype=np.int64)
            if previous.shape[-1] < self.context_len:
                pad = np.full(
                    (batch, self.context_len - previous.shape[-1]),
                    self.eos_token_id,
                    dtype=np.int64,
                )
                previous = np.concatenate([pad, previous], axis=-1)
        elif cache is not None and cache[3] is not None:
            previous = np.asarray(cache[3], dtype=np.int64)
        else:
            previous = np.full(
                (batch, self.context_len), self.eos_token_id, dtype=np.int64
            )
        if mask is not None:
            record_verify_sync("qwen4.ple.mask_asarray")
            mask_array = np.asarray(mask)
            tokens = np.where(mask_array, tokens, self.eos_token_id)
        else:
            mask_array = None
        history = np.concatenate([previous, tokens], axis=-1)
        if cache is not None:
            tail = (
                history[:, -self.context_len :]
                if mask is None
                else _row_tail(history, _valid_span_end(mask_array), self.context_len)
            )
            cache[3] = mx.array(tail, dtype=mx.int64)
        if _PLE_VECTOR_SHIFT:
            shifted = self._shift_history_vectorized(history)
        else:
            shifted = []
            for shift in range(self.ngram_size):
                out = np.full_like(history, self.eos_token_id)
                if shift == 0:
                    out = history.copy()
                else:
                    for b in range(batch):
                        segment_start = 0
                        for pos in range(history.shape[1]):
                            if pos - segment_start >= shift:
                                out[b, pos] = history[b, pos - shift]
                            if history[b, pos] == self.eos_token_id:
                                segment_start = pos + 1
                shifted.append(out)
        (multipliers, sizes, offsets) = self._hash_constants_numpy()
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            with np.errstate(over="ignore"):
                mixed = shifted[0] * multipliers[0]
                for position in range(1, ngram):
                    mixed = np.bitwise_xor(
                        mixed, shifted[position] * multipliers[position]
                    )
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            blocks.append(
                np.remainder(mixed[..., None], sizes[start:end]) + offsets[start:end]
            )
        return np.concatenate(blocks, axis=-1)[:, -seq_len:]

    def _ngram_ids_metal(
        self,
        input_ids: mx.array,
        cache: Optional[ArraysCache] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        if self.ngram_size != 3 or not mx.metal.is_available():
            return mx.array(
                self._ngram_ids_numpy(input_ids, cache, mask), dtype=mx.int64
            )
        (batch, seq_len) = input_ids.shape
        if cache is not None and cache[3] is not None:
            previous = cache[3]
        else:
            previous = mx.full(
                (batch, self.context_len), self.eos_token_id, dtype=mx.int64
            )
        tokens = input_ids.astype(mx.int64)
        if mask is not None:
            tokens = mx.where(mask, tokens, self.eos_token_id)
        history = mx.concatenate([previous, tokens], axis=-1)
        if cache is not None:
            cache[3] = mx.contiguous(
                history[:, -self.context_len :]
                if mask is None
                else _row_tail(history, _valid_span_end(mask), self.context_len)
            )
        if self._metal_hash_kernel is None:
            self._metal_hash_kernel = mx.fast.metal_kernel(
                name="qwen4_ple_ngram3_hash",
                input_names=["history", "multipliers", "sizes", "offsets"],
                output_names=["out"],
                source="\n                    uint elem = thread_position_in_grid.x;\n                    uint head = elem % HEADS;\n                    uint token = (elem / HEADS) % SEQ_LEN;\n                    uint batch = elem / (HEADS * SEQ_LEN);\n                    uint history_pos = token + 2;\n                    uint history_base = batch * (SEQ_LEN + 2);\n\n                    long current = history[history_base + history_pos];\n                    long previous_1 = history[history_base + history_pos - 1];\n                    long previous_2 = history[history_base + history_pos - 2];\n                    if (previous_1 == EOS_TOKEN) {\n                        previous_2 = EOS_TOKEN;\n                    }\n\n                    ulong mixed = ulong(current) * ulong(multipliers[0]);\n                    mixed ^= ulong(previous_1) * ulong(multipliers[1]);\n                    if (head >= HEADS_PER_NGRAM) {\n                        mixed ^= ulong(previous_2) * ulong(multipliers[2]);\n                    }\n                    long remainder = long(mixed) % sizes[head];\n                    if (remainder < 0) {\n                        remainder += sizes[head];\n                    }\n                    out[elem] = remainder + offsets[head];\n                ",
            )
        total = batch * seq_len * self.ngram_heads
        return self._metal_hash_kernel(
            inputs=[
                history,
                self.layer_multipliers,
                self.ngram_heads_vocab_sizes,
                self.ngram_heads_offsets,
            ],
            template=[
                ("HEADS", self.ngram_heads),
                ("HEADS_PER_NGRAM", self.heads_per_ngram),
                ("SEQ_LEN", seq_len),
                ("EOS_TOKEN", self.eos_token_id),
            ],
            grid=(total, 1, 1),
            threadgroup=(min(256, total), 1, 1),
            output_shapes=[(batch, seq_len, self.ngram_heads)],
            output_dtypes=[mx.int64],
            stream=mx.gpu,
        )[0]

    @property
    def file_backed(self) -> bool:
        return getattr(self.ngram_embedding, "is_file_backed", False)

    def ngram_ids(
        self,
        input_ids: mx.array,
        cache: Optional[ArraysCache] = None,
        mask: Optional[mx.array] = None,
    ):
        if self.hash_backend == "metal" or (
            self.hash_backend == "metal_prefill"
            and input_ids.shape[1] >= self.metal_hash_min_tokens
        ):
            return self._ngram_ids_metal(input_ids, cache, mask)
        return mx.array(self._ngram_ids_numpy(input_ids, cache, mask), dtype=mx.int64)

    def prefetch_prompt_chunk(
        self, chunk_tokens: np.ndarray, previous: np.ndarray
    ) -> None:
        """Warm the NVMe rows an upcoming prompt chunk will gather.

        ``previous`` holds the up-to-context_len prompt tokens right before
        the chunk. Hashing and preads run on the embedding's prefetch pool;
        the call returns immediately and mutates no cache state. No-op for
        resident embeddings.

        Stage 1: this warms the page cache, so the chunk's foreground
        lookup re-reads the same rows warm (~1 us/row). TODO: hand the
        prefetched row bytes to the next chunk's lookup directly to also
        skip the warm re-read; needs a keyed handoff between the generate
        loop and the forward pass.
        """
        if not self.file_backed:
            return
        chunk = np.asarray(chunk_tokens, dtype=np.int64)
        prev = np.asarray(previous, dtype=np.int64)
        if chunk.ndim == 1:
            chunk = chunk[None]
        if prev.ndim == 1:
            prev = prev[None]
        if chunk.size == 0:
            return

        def hash_and_warm():
            ids = self._ngram_ids_numpy(mx.array(chunk), None, previous=prev)
            self.ngram_embedding.prefetch_rows(ids)

        self.ngram_embedding.submit_prefetch(hash_and_warm)

    def prefetch_positions(self, previous_tokens, tokens) -> bool:
        """Lever (a): hash ``tokens`` (host ints: the verify slab positions
        whose ids are already known, following ``previous_tokens``) and read
        + dequantize their rows on the prefetch pool. The foreground lookup
        consumes staged rows and falls back to its normal read for anything
        not staged, so the result never depends on the prefetch landing.
        No-op for resident tables."""
        if not self.file_backed:
            return False
        prefetch = getattr(self.ngram_embedding, "prefetch_dequant_rows", None)
        if prefetch is None:
            return False
        toks = np.asarray(tokens, dtype=np.int64).reshape(1, -1)
        prev = np.asarray(previous_tokens, dtype=np.int64).reshape(1, -1)
        if toks.size == 0:
            return False

        def hash_and_dequant():
            ids = self._ngram_ids_numpy(
                mx.array(toks), None, previous=prev, record_sync=False
            )
            prefetch(ids)

        self.ngram_embedding.submit_prefetch(hash_and_dequant)
        _lv.bump("ple_prefetch_submitted")
        return True

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[ArraysCache] = None,
        mask: Optional[mx.array] = None,
    ):
        table = self.ngram_embedding
        device_verify = (
            self.file_backed
            and input_ids.shape == (1, 3)
            and mx.metal.is_available()
            and table.verify_device_available
            and table.verify_status["device_prepared"]
        )
        if device_verify:
            table.record_verify_route("device")
            ids = self._ngram_ids_metal(input_ids, cache, mask)
            return table.lookup_verify_device(ids).reshape(*input_ids.shape, -1)
        if (
            self.file_backed
            or self.hash_backend == "routed_cpu"
            or (
                self.hash_backend == "metal_prefill"
                and input_ids.shape[1] < self.metal_hash_min_tokens
            )
        ):
            if self.file_backed:
                table.record_verify_route("fallback")
            ids = self._ngram_ids_numpy(input_ids, cache, mask)
            return self.ngram_embedding.lookup_numpy(ids).reshape(*input_ids.shape, -1)
        return self.ngram_embedding(self.ngram_ids(input_ids, cache, mask)).reshape(
            *input_ids.shape, -1
        )


class PLELayer(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int, ple_layer_index: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.hc_count = args.hc_count
        hc_hidden = args.hidden_size * args.hc_count
        self.ple_embedding = NGramEmbedding(
            args, args.ple_embed_dim, layer_idx, ple_layer_index
        )
        self.key_proj = nn.Linear(args.ple_embed_dim, hc_hidden, bias=False)
        self.value_proj = nn.Linear(args.ple_embed_dim, args.hidden_size, bias=False)
        self.norm_key = GroupRMSNorm(hc_hidden, args.hidden_size, args.rms_norm_eps)
        self.norm_query = GroupRMSNorm(hc_hidden, args.hidden_size, args.rms_norm_eps)
        self.norm_conv = GroupRMSNorm(hc_hidden, args.hidden_size, args.rms_norm_eps)
        self.short_conv_state_len = (args.ple_conv_kernel_size - 1) * args.ngram_size
        self.conv1d = nn.Conv1d(
            hc_hidden,
            hc_hidden,
            args.ple_conv_kernel_size,
            dilation=args.ngram_size,
            groups=hc_hidden,
            bias=False,
        )

    def _conv_state_tail(self, conv_input, mask):
        """The persistent conv state cut at each row's own last valid position.

        The conv is causal, so the branch OUTPUT at a valid position is
        already pad-free; the persistent tail is not.  Cutting at the padded
        width instead makes a short row carry pads in its conv state forever.
        """
        return mx.contiguous(
            conv_input[:, -self.short_conv_state_len :, :]
            if mask is None
            else _row_tail(conv_input, _valid_span_end(mask), self.short_conv_state_len)
        )

    def _short_conv(self, x: mx.array, cache: Optional[ArraysCache], mask=None):
        """Eager short conv WITH the cache write, for direct-state callers.

        ``__call__`` goes through ``_device_chain`` instead, which returns the
        new state rather than assigning it -- a cache mutation is exactly the
        side effect ``mx.compile`` cannot trace.  Both share
        ``_conv_state_tail`` so the tail has one definition.
        """
        state = cache[2] if cache is not None else None
        if state is None:
            state = mx.zeros(
                (x.shape[0], self.short_conv_state_len, x.shape[-1]), x.dtype
            )
        conv_input = mx.concatenate([state, x], axis=1)
        if cache is not None:
            cache[2] = self._conv_state_tail(conv_input, mask)
        return nn.silu(self.conv1d(conv_input))[:, -x.shape[1] :, :]

    def _device_chain(self, hidden, embeddings, mask, state, write_state: bool):
        """The device half of the PLE forward, as ONE pure array function.

        Everything from ``key_proj`` through the residual ``gated + conv``,
        with no host round trip, no cache mutation and no Python-level state:
        the gather has already happened and its rows arrive as ``embeddings``.
        That purity is the whole point -- it is what lets ``mx.compile`` trace
        the chain, and the ONLY thing the compiled path changes.  Returns
        ``[out, normed]``, plus the new conv state when ``write_state``.

        Split at the SIGMOID rather than written straight through, because the
        sigmoid is the one op a fused span computes differently -- see the
        ``_PLE_COMPILE`` comment.  The eager path calls the two halves back to
        back, so this function is unchanged arithmetic; the compiled path
        traces the halves separately and keeps ``mx.sigmoid`` between them,
        where it stays the standalone primitive.
        """
        (gate, value) = self._chain_gate(hidden, embeddings)
        return self._chain_tail(
            tuple(hidden.shape), mx.sigmoid(gate), value, mask, state, write_state
        )

    def _chain_gate(self, hidden, embeddings):
        """``key_proj``/``value_proj``/norms/gate arithmetic, up to the sigmoid.

        Ends one op EARLY on purpose: the caller applies ``mx.sigmoid``.
        """
        key = self.norm_key(self.key_proj(embeddings)).reshape(
            *hidden.shape[:-1], self.hc_count, self.hidden_size
        )
        value = self.value_proj(embeddings)
        query = self.norm_query(hidden).reshape(
            *hidden.shape[:-1], self.hc_count, self.hidden_size
        )
        gate = mx.sum(key * query, axis=-1, keepdims=True) / math.sqrt(self.hidden_size)
        gate = mx.sign(gate) * mx.sqrt(mx.maximum(mx.abs(gate), 1e-06))
        return (gate, value)

    def _chain_tail(self, out_shape, sigmoid, value, mask, state, write_state: bool):
        """Everything after the sigmoid: gate product, norm, conv, residual.

        ``out_shape`` rather than ``hidden`` so the compiled tail carries no
        buffer it does not read; shapes are static within a traced signature.
        """
        gated = (sigmoid * value[..., None, :]).reshape(*out_shape)
        normed = self.norm_conv(gated)
        if mask is not None:
            gated = mx.where(mask[..., None], gated, 0)
            normed = mx.where(mask[..., None], normed, 0)
        if state is None:
            state = mx.zeros(
                (normed.shape[0], self.short_conv_state_len, normed.shape[-1]),
                normed.dtype,
            )
        conv_input = mx.concatenate([state, normed], axis=1)
        conv = nn.silu(self.conv1d(conv_input))[:, -normed.shape[1] :, :]
        out = gated + conv
        if not write_state:
            return [out, normed]
        return [out, normed, self._conv_state_tail(conv_input, mask)]

    def _chain_params(self):
        """Strong references to every weight the traced chain closes over.

        ``mx.compile`` bakes a captured array into the graph as a CONSTANT and
        does not notice when the attribute is later rebound -- measured: after
        ``norm_key.weight *= 2`` the cached graph still answered with the old
        gain, and disagreed with eager.  Frozen inference never does that, but
        ``set_dtype``, quantization, a LoRA merge and training all do, so the
        entry is keyed on the identity of these objects rather than trusted.
        Holding them keeps the ids from being recycled under us; the graph
        holds them anyway, so this costs no memory.
        """
        params = []
        for module in (
            self.key_proj,
            self.value_proj,
            self.conv1d,
            self.norm_key,
            self.norm_query,
            self.norm_conv,
        ):
            for name in ("weight", "scales", "biases"):
                value = getattr(module, name, None)
                if value is not None:
                    params.append(value)
        return tuple(params)

    def _compiled_chain(
        self, signature, has_mask: bool, has_state: bool, write_state: bool
    ):
        """Return the traced chain for ``signature``, or ``None`` for eager.

        ``None`` is the fail-closed answer and is cached as such, so a
        signature that raised once is never retried -- the eager chain is
        always a correct substitute, and a compile failure must cost one
        receipt, not one exception per round.
        """
        cache = getattr(self, "_ple_compile_cache", None)
        flags = _trace_flags()
        if cache is None or getattr(self, "_ple_compile_flags", None) != flags:
            if cache:
                _record_ple_compile("invalidations", flags=repr(flags))
            cache = {}
            self._ple_compile_cache = cache
            self._ple_compile_flags = flags
        params = self._chain_params()
        entry = cache.get(signature)
        if entry is not None:
            (cached_params, compiled) = entry
            if len(cached_params) == len(params) and all(
                (a is b for (a, b) in zip(cached_params, params))
            ):
                _record_ple_compile("hits")
                return compiled
            _record_ple_compile("retraces", signature=repr(signature))
            del cache[signature]
        if len(cache) >= _PLE_COMPILE_CACHE_MAX:
            _record_ple_compile("overflow", signature=repr(signature))
            return None
        out_shape = tuple(signature[0])
        try:

            def head(hidden, embeddings):
                return list(self._chain_gate(hidden, embeddings))

            if has_mask and has_state:

                def tail(sig, value, mask, state, _os=out_shape, _ws=write_state):
                    return self._chain_tail(_os, sig, value, mask, state, _ws)
            elif has_mask:

                def tail(sig, value, mask, _os=out_shape, _ws=write_state):
                    return self._chain_tail(_os, sig, value, mask, None, _ws)
            elif has_state:

                def tail(sig, value, state, _os=out_shape, _ws=write_state):
                    return self._chain_tail(_os, sig, value, None, state, _ws)
            else:

                def tail(sig, value, _os=out_shape, _ws=write_state):
                    return self._chain_tail(_os, sig, value, None, None, _ws)

            compiled = (mx.compile(head), mx.compile(tail))
        except Exception as exc:
            compiled = None
            _record_ple_compile("fallbacks", signature=repr(signature), error=repr(exc))
        else:
            _record_ple_compile("builds", signature=repr(signature))
        cache[signature] = (params, compiled)
        return compiled

    def _run_device_chain(self, hidden, embeddings, mask, state, write_state: bool):
        if not _PLE_COMPILE:
            return self._device_chain(hidden, embeddings, mask, state, write_state)
        if mx.default_device() != mx.gpu:
            _record_ple_compile("skips", reason="non_metal_device")
            return self._device_chain(hidden, embeddings, mask, state, write_state)
        if mx.float32 in (hidden.dtype, embeddings.dtype):
            _record_ple_compile("skips", reason="float32_activations")
            return self._device_chain(hidden, embeddings, mask, state, write_state)
        has_mask = mask is not None
        has_state = state is not None
        signature = (
            tuple(hidden.shape),
            tuple(embeddings.shape),
            tuple(state.shape) if has_state else None,
            has_mask,
            bool(write_state),
            str(hidden.dtype),
            str(embeddings.dtype),
            _trace_flags(),
        )
        compiled = self._compiled_chain(signature, has_mask, has_state, write_state)
        if compiled is None:
            return self._device_chain(hidden, embeddings, mask, state, write_state)
        (compiled_head, compiled_tail) = compiled
        try:
            (gate, value) = compiled_head(hidden, embeddings)
            sigmoid = mx.sigmoid(gate)
            args = tuple((a for a in (sigmoid, value, mask, state) if a is not None))
            return compiled_tail(*args)
        except Exception as exc:
            self._ple_compile_cache[signature] = (self._chain_params(), None)
            _record_ple_compile("fallbacks", signature=repr(signature), error=repr(exc))
            return self._device_chain(hidden, embeddings, mask, state, write_state)

    def __call__(self, hidden: mx.array, input_ids: mx.array, cache=None, mask=None):
        if mask is None and isinstance(cache, ArraysCache):
            if cache.lengths is not None or cache.left_padding is not None:
                mask = cache.make_mask(input_ids.shape[1])
        previous_conv = cache[2] if cache is not None else None
        previous_tokens = cache[3] if cache is not None else None
        embeddings = self.ple_embedding(input_ids, cache, mask)
        write_state = cache is not None
        outputs = self._run_device_chain(
            hidden, embeddings, mask, previous_conv, write_state
        )
        (out, normed) = (outputs[0], outputs[1])
        if write_state:
            cache[2] = outputs[2]
        spans = (
            cache.rollback_spans(input_ids.shape[1], mask)
            if isinstance(cache, Qwen4ArraysCache) and cache.speculating
            else None
        )
        if spans is not None:
            state_len = self.short_conv_state_len
            conv_base = previous_conv
            if conv_base is None:
                conv_base = mx.zeros(
                    (normed.shape[0], state_len, normed.shape[-1]), normed.dtype
                )
            conv_input = mx.concatenate([conv_base, normed], axis=1)
            context_len = self.ple_embedding.context_len
            token_base = previous_tokens
            if token_base is None:
                token_base = mx.full(
                    (input_ids.shape[0], context_len),
                    self.ple_embedding.eos_token_id,
                    dtype=mx.int64,
                )
            staged_ids = input_ids.astype(mx.int64)
            if mask is not None:
                staged_ids = mx.where(mask, staged_ids, self.ple_embedding.eos_token_id)
            token_history = mx.concatenate([token_base, staged_ids], axis=1)

            def _ple_rollback(
                m, ci=conv_input, th=token_history, sl=state_len, cl=context_len
            ):
                return [
                    mx.contiguous(ci[:, m : m + sl, :]),
                    mx.contiguous(th[:, m : m + cl]),
                ]

            def _ple_rollback_rows(
                lengths, ci=conv_input, th=token_history, sl=state_len, cl=context_len
            ):
                ends = mx.array(list(lengths))
                return [
                    mx.contiguous(_row_tail(ci, ends, sl)),
                    mx.contiguous(_row_tail(th, ends, cl)),
                ]

            cache.stage_ple_rollback(
                input_ids.shape[1],
                _ple_rollback,
                [previous_conv, previous_tokens],
                per_row_fn=_ple_rollback_rows,
            )
        return out


_ROPE_POSITION_FREQS: Dict[tuple, mx.array] = {}


def _apply_rope_positions(x: mx.array, positions: mx.array, dims: int, base: float):
    """Transformers-compatible non-traditional partial RoPE at arbitrary positions."""
    if dims == 0:
        return x
    freqs = _ROPE_POSITION_FREQS.get((dims, base))
    if freqs is None:
        freqs = mx.exp(-math.log(base) * mx.arange(0, dims, 2) / dims)
        _ROPE_POSITION_FREQS[dims, base] = freqs
    angles = positions[..., None].astype(mx.float32) * freqs
    (cos, sin) = (mx.cos(angles), mx.sin(angles))
    (rope, tail) = (x[..., :dims], x[..., dims:])
    half = dims // 2
    (left, right) = (rope[..., :half], rope[..., half:])
    rotated = mx.concatenate(
        [left * cos - right * sin, right * cos + left * sin], axis=-1
    )
    return mx.concatenate([rotated.astype(x.dtype), tail], axis=-1)


_QSA_CYCLE_STATE = (
    ("_mtp_share_topk", False),
    ("_mtp_shared_topk", None),
    ("_mtp_shared_topk_n_blocks", None),
    ("_qsa_pooled_keys", None),
    ("_qsa_pooled_ratio", None),
)

# Process-lifetime observability for the correctness-critical amendment below.
# These are deliberately scalar host counters: they neither retain cache/model
# objects nor add a device synchronization to the decode path.
_QSA_MTP_AMEND_STATS = {
    "calls": 0,
    "noops": 0,
    "amendments": 0,
    "failures": 0,
    "appended_blocks": 0,
    "max_appended_blocks": 0,
}
_QSA_MTP_AMEND_STATS_LOCK = threading.Lock()


def qsa_mtp_amendment_status(*, reset: bool = False) -> Dict[str, int]:
    """Return bounded counters for shared-QSA blocks closed mid-draft."""
    with _QSA_MTP_AMEND_STATS_LOCK:
        status = dict(_QSA_MTP_AMEND_STATS)
        if reset:
            for key in _QSA_MTP_AMEND_STATS:
                _QSA_MTP_AMEND_STATS[key] = 0
    return status


def _record_qsa_mtp_amendment(
    *, noop: bool = False, failure: bool = False, appended_blocks: int = 0
):
    """Atomically publish one amendment attempt without retaining device state."""
    appended_blocks = int(appended_blocks)
    with _QSA_MTP_AMEND_STATS_LOCK:
        _QSA_MTP_AMEND_STATS["calls"] += 1
        if failure:
            _QSA_MTP_AMEND_STATS["failures"] += 1
        elif noop:
            _QSA_MTP_AMEND_STATS["noops"] += 1
        elif appended_blocks > 0:
            _QSA_MTP_AMEND_STATS["amendments"] += 1
            _QSA_MTP_AMEND_STATS["appended_blocks"] += appended_blocks
            _QSA_MTP_AMEND_STATS["max_appended_blocks"] = max(
                _QSA_MTP_AMEND_STATS["max_appended_blocks"], appended_blocks
            )
        else:
            # Keep the outcome partition coherent if a future refusal site
            # records an attempt without explicitly labelling it first.
            _QSA_MTP_AMEND_STATS["failures"] += 1


def _extend_mtp_shared_topk(cache, shared_topk, n_blocks):
    """Append compressed groups that become complete inside one draft loop."""
    if shared_topk is None:
        return None
    captured = getattr(cache, "_mtp_shared_topk_n_blocks", None)
    if captured is None:
        _record_qsa_mtp_amendment(failure=True)
        raise RuntimeError("QSA shared top-k is missing its captured block count")
    captured = int(captured)
    n_blocks = int(n_blocks)
    if n_blocks < captured:
        _record_qsa_mtp_amendment(failure=True)
        raise RuntimeError("QSA shared top-k outlived a block-grid rewind")
    if n_blocks == captured:
        _record_qsa_mtp_amendment(noop=True)
        return shared_topk
    appended_count = n_blocks - captured
    appended = mx.broadcast_to(
        mx.arange(captured, n_blocks, dtype=mx.uint32)[None],
        (int(shared_topk.shape[0]), n_blocks - captured),
    )
    shared_topk = mx.concatenate([shared_topk, appended], axis=-1)
    cache._mtp_shared_topk = mx.contiguous(shared_topk)
    cache._mtp_shared_topk_n_blocks = n_blocks
    _record_qsa_mtp_amendment(appended_blocks=appended_count)
    return cache._mtp_shared_topk


def _qsa_summary_persistable(cache) -> bool:
    pooled = getattr(cache, "_qsa_pooled_keys", None)
    identity = getattr(cache, "_qsa_summary_identity", None)
    return bool(
        _QSA_APC_SUMMARIES
        and pooled is not None
        and (identity is not None)
        and (pooled.ndim == 3)
        and (pooled.shape[1] > 0)
        and (int(identity.get("complete_blocks", -1)) == pooled.shape[1])
    )


def _qsa_summary_state(cache, base):
    return (*base, cache._qsa_pooled_keys) if _qsa_summary_persistable(cache) else base


def _qsa_summary_restore_state(cache, value, base_length: int):
    if len(value) == base_length:
        cache._qsa_pending_pooled = None
        return value
    if len(value) == base_length + 1:
        cache._qsa_pending_pooled = value[-1]
        return value[:-1]
    raise ValueError(
        f"Invalid {type(cache).__name__} state: expected {base_length} legacy fields or {base_length + 1} fields with a QSA summary."
    )


def _qsa_summary_meta(cache, base):
    if not _qsa_summary_persistable(cache):
        return tuple(base)
    identity = cache._qsa_summary_identity
    return tuple(base) + (
        _QSA_SUMMARY_META_MARKER,
        str(identity["format_version"]),
        identity["model_config_hash"],
        str(identity["block_size"]),
        str(identity["compress_ratio"]),
        identity["producer_version"],
        identity["layer_id"],
        str(identity["complete_blocks"]),
    )


def _qsa_summary_split_meta(value):
    value = tuple(value)
    if _QSA_SUMMARY_META_MARKER not in value:
        return (value, None, None)
    marker = value.index(_QSA_SUMMARY_META_MARKER)
    (base, summary) = (value[:marker], value[marker:])
    if len(summary) != 8:
        return (base, None, "summary_identity_mismatch")
    try:
        version = int(summary[1])
        identity = {
            "format_version": version,
            "model_config_hash": summary[2],
            "block_size": int(summary[3]),
            "compress_ratio": int(summary[4]),
            "producer_version": summary[5],
            "layer_id": summary[6],
            "complete_blocks": int(summary[7]),
        }
    except (TypeError, ValueError):
        return (base, None, "summary_identity_mismatch")
    if version != _QSA_SUMMARY_FORMAT_VERSION:
        return (base, None, "summary_identity_mismatch")
    return (base, identity, None)


def _qsa_summary_finish_restore(cache, identity, reason=None):
    pooled = getattr(cache, "_qsa_pending_pooled", None)
    cache._qsa_pending_pooled = None
    cache._qsa_pooled_keys = None
    cache._qsa_pooled_ratio = None
    cache._qsa_summary_restored = False
    geometry_limit = None
    if identity is not None and identity.get("compress_ratio", 0) > 0:
        ratio = identity["compress_ratio"]
        if hasattr(cache, "_idx") and hasattr(cache, "max_left_padding"):
            geometry_limit = (
                max(0, int(cache._idx) - int(cache.max_left_padding())) // ratio
            )
        elif isinstance(getattr(cache, "offset", None), int):
            geometry_limit = max(0, int(cache.offset)) // ratio
    valid = bool(
        _QSA_APC_SUMMARIES
        and reason is None
        and (identity is not None)
        and (pooled is not None)
        and (pooled.ndim == 3)
        and (identity["complete_blocks"] == pooled.shape[1])
        and (identity["block_size"] == identity["compress_ratio"])
        and (geometry_limit is None or identity["complete_blocks"] <= geometry_limit)
    )
    if valid:
        cache._qsa_pooled_keys = pooled
        cache._qsa_pooled_ratio = identity["compress_ratio"]
        cache._qsa_summary_identity = identity
        cache._qsa_summary_restored = True
        return
    dropped = 0 if pooled is None or pooled.ndim < 2 else int(pooled.shape[1])
    cache._qsa_summary_identity = None
    if pooled is not None and _QSA_APC_SUMMARIES:
        _record_qsa_summary(
            "invalidations", reason or "summary_identity_mismatch", blocks=dropped
        )


def _qsa_summary_rebound(cache, keep: int, reason: str) -> None:
    pooled = getattr(cache, "_qsa_pooled_keys", None)
    if pooled is None:
        return
    old = int(pooled.shape[1])
    keep = max(0, min(int(keep), old))
    if keep == 0:
        cache._qsa_pooled_keys = None
        cache._qsa_pooled_ratio = None
    elif keep < old:
        cache._qsa_pooled_keys = mx.contiguous(pooled[:, :keep])
    cache._qsa_summary_identity = _qsa_summary_with_coverage(
        getattr(cache, "_qsa_summary_identity", None), keep
    )
    if old > keep:
        _record_qsa_summary("invalidations", reason, blocks=old - keep)


def _init_qsa_summary_state(cache, identity=None) -> None:
    cache._qsa_summary_identity = (
        None if identity is None else _qsa_summary_with_coverage(identity, 0)
    )
    cache._qsa_summary_restored = False
    cache._qsa_pending_pooled = None


def _qsa_join_ledger_width(index_keys, cursor: int, who: str) -> int:
    """Width of a lane's raw-key ledger for a join, refused if it runs short.

    A lane has two lengths and they are not the same quantity: the KV cursor
    (``_idx`` / ``size()``) and the width of ``index_keys``. The joins slice
    the ledger with the KV quantity, which is exact only while the two agree.
    Where they diverge the join used to fail at ``concatenate`` with a bare
    shape message, or, single-lane, return a ledger the cursor does not
    describe. Say which quantity disagreed instead. An ABSENT ledger is width
    0 and is refused on the same rule: it is legitimate only on a lane that
    holds no KV either.
    """
    width = 0 if index_keys is None else index_keys.shape[1]
    if width < cursor:
        raise RuntimeError(
            f"{who}: the QSA raw-key ledger holds {width} positions but the cursor is at {cursor}. A join reads each lane by its ledger width, not by its KV offset; a short ledger means a shared-top-k draft cycle was joined without rewinding the drafted span."
            + (
                " This lane has no ledger at all: zero-filling it would give the joined row raw keys the lane never wrote, and the width the join stamps would then satisfy the next forward's desync check."
                if index_keys is None
                else ""
            )
        )
    return width


def _qsa_merge_summaries(caches, logical_lengths):
    """Pooled rows for a join, and the identity that describes them.

    The in-memory pooled-key cache and its durable APC identity are separate
    levers.  Runtime-only joins may return ``(pooled, None)``: the blocks are
    exact and reusable in this process even though no persistence provenance
    is being claimed.

    Coverage and provenance are two quantities, not one.  A join that can
    carry no pooled BLOCK still knows exactly who produced the rows it
    joined, and that is the zero-coverage state the rest of this file keeps
    everywhere else: identity retained, ``complete_blocks`` 0.  So ``(None,
    identity@0)`` means "nothing reusable, provenance intact" and ``(None,
    None)`` is reserved for a real provenance loss -- the joined rows
    disagree about who made them, or none of them ever had an identity.
    """
    if not (_QSA_POOLED_KEY_CACHE or _QSA_APC_SUMMARIES) or not caches:
        return (None, None)
    identity = None
    if _QSA_APC_SUMMARIES:
        identity = getattr(caches[0], "_qsa_summary_identity", None)
        if identity is None:
            return (None, None)
        for cache in caches[1:]:
            if not _qsa_summary_identity_matches(
                getattr(cache, "_qsa_summary_identity", None), identity
            ):
                return (None, None)
    zero = _qsa_summary_with_coverage(identity, 0)
    ratio = getattr(caches[0], "_qsa_pooled_ratio", None)
    if not ratio:
        return (None, zero)
    complete = min((int(length) // ratio for length in logical_lengths))
    if complete == 0:
        return (None, zero)
    rows = []
    for cache in caches:
        pooled = getattr(cache, "_qsa_pooled_keys", None)
        if (
            pooled is None
            or pooled.shape[1] < complete
            or getattr(cache, "_qsa_pooled_ratio", None) != ratio
        ):
            return (None, zero)
        rows.append(mx.contiguous(pooled[:, :complete]))
    return (
        mx.concatenate(rows, axis=0),
        _qsa_summary_with_coverage(identity, complete),
    )


def _qsa_to_quantized(
    self,
    group_size: int = 64,
    bits: int = 4,
    *,
    key_bits: Optional[int] = None,
    value_bits: Optional[int] = None,
    rotate: bool = False,
    normalize: bool = False,
):
    """Convert QSA attention K/V without dropping the raw index-key ledger.

    The quantized twins are defined below the float classes. Name lookup is
    intentionally delayed until this method is called, after module import is
    complete. QSA's auxiliary state remains native: only attention K/V is
    packed.
    """
    kwargs = dict(
        group_size=group_size,
        bits=bits,
        key_bits=key_bits,
        value_bits=value_bits,
        rotate=rotate,
    )
    if isinstance(self, BatchQSAKVCache):
        if normalize:
            raise NotImplementedError(
                "Batched QSA KVarN caches need a per-row scale contract."
            )
        return BatchQSAQuantizedKVCache.from_unquantized(self, **kwargs)
    return QSAQuantizedKVCache.from_unquantized(self, normalize=normalize, **kwargs)


class BatchQSAKVCache(BatchKVCache):
    """Batched QSA cache retaining raw indexer keys beside attention KV."""

    to_quantized = _qsa_to_quantized
    _RAGGED_TRIM_AUX_ARRAYS = (("index_keys", 1),)
    _QSA_CYCLE_FIELDS = _QSA_CYCLE_STATE

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.index_keys = None
        instance._max_left_pad = None
        for name, blank in cls._QSA_CYCLE_FIELDS:
            setattr(instance, name, blank)
        _init_qsa_summary_state(instance)
        return instance

    def __init__(
        self, left_padding: List[int], attention_backend=None, summary_identity=None
    ):
        super().__init__(left_padding, attention_backend=attention_backend)
        _init_qsa_summary_state(self, summary_identity)

    def max_left_padding(self) -> int:
        """Host copy of ``left_padding.max()``, keyed by array identity.

        ``left_padding`` is rebound only at membership boundaries (merge,
        filter, extend, finalize) and mx arrays are immutable, so an identity
        miss is exactly the set of events that can change the maximum.  The
        pooled-key bound below needs this every forward and must not sync.
        """
        padding = self.left_padding
        cached = self._max_left_pad
        if cached is None or cached[0] is not padding:
            record_verify_sync("qwen4.qsa.max_left_padding_item")
            self._max_left_pad = (padding, int(padding.max().item()))
        return self._max_left_pad[1]

    def release_qsa_cycle(
        self,
        who: str,
        *,
        rows=None,
        keep_pooled=True,
        cursor_final=True,
        keep_shared=False,
    ):
        """The ONE exit for armed QSA state. Every lifecycle method routes here.

        This class has now been bitten three times by armed state outliving
        the geometry it was computed against (the live ``trim()`` desync, the
        aborted draft cycle, the filtered block ids), so the rules below are
        stated once and derived from the CURRENT geometry -- never from which
        method called:

        * **The MTP shared top-k dies at every membership or cycle boundary.**
          It is a cycle-local set of block ids with no geometry-independent
          meaning: ``filter`` can shrink the physical grid under it (dropping
          the common left padding), which leaves ids past ``n_blocks`` that
          ``_QSA_SCATTER_CHOSEN`` would scatter out of bounds. The only
          exception is a prepare/finalize pair inside the same ragged MTP
          draft cycle: that pair changes physical padding but not the rows or
          their logical block ids, so ``keep_shared=True`` preserves the set.
        * **The pooled keys are re-bounded, not dropped.** They are indexed by
          LOGICAL block per row, so they survive anything that does not change
          a row's own logical content -- and the bound that expresses this,
          ``(cursor - max left padding) // ratio``, is read off the live state
          after the caller has updated it. That covers a rewind (offsets
          shrink), a filter (rows subset, then re-bound) and ``finalize`` (the
          right-padding roll only ever invalidates blocks past the shortest
          row). ``keep_pooled=False`` is for the one case where the cache stops
          being the same cache at all: a ``state`` restore.

        Post-condition: with the cycle released the raw-key ledger must span
        exactly the cursor. Only a live shared-top-k cycle may run it short.
        ``cursor_final=False`` defers that check either while such a cycle is
        still live, or for the ``state`` setter where ``_idx`` remains the
        allocated buffer width until ``meta_state`` lands the real one.
        """
        (pooled, ratio) = (self._qsa_pooled_keys, self._qsa_pooled_ratio)
        share_topk = self._mtp_share_topk
        shared_topk = self._mtp_shared_topk
        shared_topk_n_blocks = self._mtp_shared_topk_n_blocks
        for name, blank in self._QSA_CYCLE_FIELDS:
            setattr(self, name, blank)
        if keep_shared:
            self._mtp_share_topk = share_topk
            self._mtp_shared_topk = shared_topk
            self._mtp_shared_topk_n_blocks = shared_topk_n_blocks
        if keep_pooled and pooled is not None:
            if rows is not None:
                pooled = mx.contiguous(pooled[rows])
            if not ratio:
                raise RuntimeError("QSA pooled keys are cached without a ratio")
            keep = max(0, self._idx - self.max_left_padding()) // ratio
            if keep and pooled.shape[0] == self.offset.shape[0]:
                self._qsa_pooled_keys = (
                    pooled
                    if keep >= pooled.shape[1]
                    else mx.contiguous(pooled[:, :keep])
                )
                self._qsa_pooled_ratio = ratio
                self._qsa_summary_identity = _qsa_summary_with_coverage(
                    self._qsa_summary_identity, self._qsa_pooled_keys.shape[1]
                )
                if keep < pooled.shape[1]:
                    _record_qsa_summary(
                        "invalidations", who, blocks=pooled.shape[1] - keep
                    )
            elif pooled.shape[1]:
                self._qsa_summary_identity = _qsa_summary_with_coverage(
                    self._qsa_summary_identity, 0
                )
                _record_qsa_summary("invalidations", who, blocks=pooled.shape[1])
        if cursor_final:
            self._reconcile_index_ledger(who)

    def _reconcile_index_ledger(self, who: str):
        """``len(index_keys) == cursor``, or say exactly why not.

        A ledger WIDER than the cursor is the normal aftermath of a rewind and
        is simply cut. A ledger NARROWER than the cursor means a shared-top-k
        cycle skipped raw-key appends for KV that is still in the cache: the
        13-vs-12 shape of the live desync this class shipped once already.
        """
        if self.index_keys is None:
            return
        width = self.index_keys.shape[1]
        if width > self._idx:
            self.index_keys = mx.contiguous(self.index_keys[:, : self._idx])
        elif width < self._idx:
            raise RuntimeError(
                f"{who}: the QSA raw-key ledger holds {width} positions but the cursor is at {self._idx}. A shared-top-k draft cycle left un-ledgered KV behind and was ended without rewinding the drafted span."
            )

    def trim(self, n):
        n = super().trim(n)
        self.release_qsa_cycle("BatchQSAKVCache.trim")
        return n

    def trim_ragged(self, n, *, validate: bool = True):
        drops = super().trim_ragged(n, validate=validate)
        self.release_qsa_cycle("BatchQSAKVCache.trim_ragged")
        return drops

    def prepare(self, *args, **kwargs):
        super().prepare(*args, **kwargs)
        self.release_qsa_cycle("BatchQSAKVCache.prepare")

    def prepare_self_mtp_step(self, *args, **kwargs):
        if not self._mtp_share_topk:
            return self.prepare(*args, **kwargs)
        super().prepare(*args, **kwargs)
        self.release_qsa_cycle(
            "BatchQSAKVCache.prepare_self_mtp_step",
            cursor_final=False,
            keep_shared=True,
        )

    def last_valid_query(self, values: mx.array) -> mx.array:
        """Gather each row's final non-padding query from ``[B, L, ...]``."""
        if values.ndim < 2 or values.shape[0] != self.offset.shape[0]:
            raise ValueError("QSA query values must have shape [batch, length, ...]")
        length = values.shape[1]
        if length == 0:
            raise ValueError("QSA query values cannot have zero length")
        padding = self._right_padding
        if padding is None:
            return values[:, -1]
        invalid = mx.any((padding < 0) | (padding >= length))
        if bool(invalid.item()):
            raise ValueError("QSA right padding must leave one valid query per row")
        rows = mx.arange(values.shape[0], dtype=mx.int32)
        positions = length - padding.astype(mx.int32) - 1
        return values[rows, positions]

    def update_index_keys(self, keys: mx.array):
        self.index_keys = (
            keys
            if self.index_keys is None
            else mx.concatenate([self.index_keys[:, : self._idx], keys], axis=1)
        )
        return self.index_keys

    @property
    def state(self):
        base = (*BatchKVCache.state.fget(self), self.index_keys)
        return _qsa_summary_state(self, base)

    @state.setter
    def state(self, value):
        value = _qsa_summary_restore_state(self, value, 5)
        BatchKVCache.state.fset(self, value[:4])
        self.index_keys = value[4]
        self._max_left_pad = None
        self.release_qsa_cycle(
            "BatchQSAKVCache.state", keep_pooled=False, cursor_final=False
        )

    @property
    def meta_state(self):
        return _qsa_summary_meta(self, BatchKVCache.meta_state.fget(self))

    @meta_state.setter
    def meta_state(self, value):
        (base, identity, reason) = _qsa_summary_split_meta(value)
        BatchKVCache.meta_state.fset(self, base)
        _qsa_summary_finish_restore(self, identity, reason)
        if value:
            self._reconcile_index_ledger("BatchQSAKVCache.meta_state")

    @property
    def nbytes(self):
        summary = 0 if self._qsa_pooled_keys is None else self._qsa_pooled_keys.nbytes
        return (
            super().nbytes
            + (0 if self.index_keys is None else self.index_keys.nbytes)
            + summary
        )

    def _finalize(self, *, keep_shared=False):
        padding = self._right_padding
        if padding is not None and self.index_keys is not None:
            self.index_keys = dynamic_roll(self.index_keys, padding, axis=1)
        super().finalize()
        self.release_qsa_cycle(
            "BatchQSAKVCache.finalize",
            cursor_final=not keep_shared,
            keep_shared=keep_shared,
        )

    def finalize(self):
        self._finalize()

    def finalize_self_mtp_step(self):
        if not self._mtp_share_topk:
            return self.finalize()
        self._finalize(keep_shared=True)

    def filter(self, batch_indices):
        min_left_pad = min(
            int(self.left_padding[batch_indices].min().item()), self._idx
        )
        if self.index_keys is not None:
            self.index_keys = self.index_keys[batch_indices]
            if min_left_pad > 0:
                self.index_keys = self.index_keys[:, min_left_pad:]
        super().filter(batch_indices)
        self.release_qsa_cycle("BatchQSAKVCache.filter", rows=batch_indices)

    def extend(self, other):
        (index_a, index_b) = (self.index_keys, other.index_keys)
        (idx_a, idx_b) = (self._idx, other._idx)
        who = "BatchQSAKVCache.extend"
        _qsa_join_ledger_width(index_a, idx_a, who)
        _qsa_join_ledger_width(index_b, idx_b, who)
        if index_a is None and index_b is None:
            merged_index = None
        else:
            populated = index_a if index_a is not None else index_b
            (dims, dtype) = (populated.shape[-1], populated.dtype)
            max_idx = max(idx_a, idx_b)

            def pad(index, idx, batch):
                if index is None:
                    index = mx.zeros((batch, 0, dims), dtype=dtype)
                else:
                    index = index[:, :idx]
                return mx.pad(index, [(0, 0), (max_idx - index.shape[1], 0), (0, 0)])

            merged_index = mx.concatenate(
                [
                    pad(index_a, idx_a, self.offset.shape[0]),
                    pad(index_b, idx_b, other.offset.shape[0]),
                ]
            )
        super().extend(other)
        self.index_keys = merged_index
        self._max_left_pad = None
        self.release_qsa_cycle("BatchQSAKVCache.extend")

    def extract(self, idx):
        self._reconcile_index_ledger("BatchQSAKVCache.extract")
        cache = QSAKVCache(self._qsa_summary_identity)
        padding = self.left_padding[idx].item()
        end = self._idx
        if self._right_padding is not None:
            end -= int(self._right_padding[idx].item())
        if self.keys is not None:
            cache.keys = mx.contiguous(self.keys[idx : idx + 1, :, padding:end])
            cache.values = mx.contiguous(self.values[idx : idx + 1, :, padding:end])
            cache.offset = cache.keys.shape[2]
        if self.index_keys is not None:
            cache.index_keys = mx.contiguous(
                self.index_keys[idx : idx + 1, padding:end]
            )
        if self._qsa_pooled_keys is not None and self._qsa_pooled_ratio:
            complete = min(
                cache.offset // self._qsa_pooled_ratio, self._qsa_pooled_keys.shape[1]
            )
            if complete:
                cache._qsa_pooled_keys = mx.contiguous(
                    self._qsa_pooled_keys[idx : idx + 1, :complete]
                )
                cache._qsa_pooled_ratio = self._qsa_pooled_ratio
                cache._qsa_summary_identity = _qsa_summary_with_coverage(
                    self._qsa_summary_identity, complete
                )
        return cache

    @classmethod
    def merge(cls, caches):
        lengths = [cache.size() for cache in caches]
        width = max(lengths)
        padding = [width - length for length in lengths]
        batch = cls(padding)
        (pooled, identity) = _qsa_merge_summaries(caches, lengths)
        if identity is not None:
            batch._qsa_summary_identity = identity
        if width == 0:
            return batch
        base = BatchKVCache.merge(caches)
        batch.keys = base.keys
        batch.values = base.values
        batch.offset = base.offset
        batch.left_padding = base.left_padding
        batch._idx = base._idx
        populated = next(
            (cache.index_keys for cache in caches if cache.index_keys is not None), None
        )
        if populated is not None:
            (dims, dtype) = (populated.shape[-1], populated.dtype)
            rows = []
            for cache, length in zip(caches, lengths):
                _qsa_join_ledger_width(
                    cache.index_keys, length, "BatchQSAKVCache.merge"
                )
                values = cache.index_keys
                if values is None:
                    values = mx.zeros((1, 0, dims), dtype=dtype)
                else:
                    values = values[:, :length]
                rows.append(
                    mx.pad(values, [(0, 0), (width - values.shape[1], 0), (0, 0)])
                )
            batch.index_keys = mx.concatenate(rows)
        if pooled is not None:
            batch._qsa_pooled_keys = pooled
            batch._qsa_pooled_ratio = int(caches[0]._qsa_pooled_ratio)
            batch._qsa_summary_identity = identity
        return batch


class QSAKVCache(KVCache):
    """KV cache with the raw, pre-pooling indexer keys QSA also requires."""

    _QSA_CYCLE_FIELDS = _QSA_CYCLE_STATE
    to_quantized = _qsa_to_quantized

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.index_keys = None
        for name, blank in cls._QSA_CYCLE_FIELDS:
            setattr(instance, name, blank)
        _init_qsa_summary_state(instance)
        return instance

    def __init__(self, summary_identity=None):
        super().__init__()
        _init_qsa_summary_state(self, summary_identity)

    def update_index_keys(self, keys: mx.array):
        self.index_keys = (
            keys
            if self.index_keys is None
            else mx.concatenate([self.index_keys[:, : self.offset], keys], axis=1)
        )
        return self.index_keys

    def release_qsa_cycle(self, who: str, *, keep_pooled: bool = True):
        """Single-sequence twin of ``BatchQSAKVCache.release_qsa_cycle``.

        A rewind ends any MTP draft cycle.  A stale shared top-k would make
        the next uncycled ``mtp_step`` skip its raw-key append and desync
        ``index_keys`` from the KV offset; ``mtp_start_cycle`` re-arms it.
        A block mean is a closed window over ``ratio`` tokens, so every block
        fully inside the trimmed offset stays exact and only the tail is cut.
        """
        self._mtp_share_topk = False
        self._mtp_shared_topk = None
        self._mtp_shared_topk_n_blocks = None
        if self.index_keys is not None:
            width = self.index_keys.shape[1]
            if width < self.offset:
                raise RuntimeError(
                    f"{who}: the QSA raw-key ledger holds {width} positions but the offset is {self.offset}. A shared-top-k draft cycle left un-ledgered KV behind and was ended without rewinding the drafted span."
                )
            if width > self.offset:
                self.index_keys = mx.contiguous(self.index_keys[:, : self.offset])
        if self._qsa_pooled_keys is not None:
            keep = 0 if not keep_pooled else self.offset // self._qsa_pooled_ratio
            _qsa_summary_rebound(self, keep, who)

    def trim(self, n):
        n = super().trim(n)
        self.release_qsa_cycle("QSAKVCache.trim")
        return n

    @classmethod
    def merge(cls, caches):
        return BatchQSAKVCache.merge(caches)

    @property
    def state(self):
        return _qsa_summary_state(self, (self.keys, self.values, self.index_keys))

    @state.setter
    def state(self, value):
        value = _qsa_summary_restore_state(self, value, 3)
        (self.keys, self.values, self.index_keys) = value
        self.offset = 0 if self.keys is None else self.keys.shape[2]
        self._mtp_share_topk = False
        self._mtp_shared_topk = None
        self._mtp_shared_topk_n_blocks = None
        self._qsa_pooled_keys = None
        self._qsa_pooled_ratio = None
        self._qsa_summary_identity = None
        self._qsa_summary_restored = False

    @property
    def meta_state(self):
        return _qsa_summary_meta(self, KVCache.meta_state.fget(self))

    @meta_state.setter
    def meta_state(self, value):
        (base, identity, reason) = _qsa_summary_split_meta(value)
        KVCache.meta_state.fset(self, base)
        _qsa_summary_finish_restore(self, identity, reason)

    @property
    def nbytes(self):
        summary = 0 if self._qsa_pooled_keys is None else self._qsa_pooled_keys.nbytes
        return (
            super().nbytes
            + (0 if self.index_keys is None else self.index_keys.nbytes)
            + summary
        )


def _copy_qsa_auxiliary_state(source, destination):
    """Copy QSA-only state without conflating it with packed attention KV."""
    destination.index_keys = source.index_keys
    for name, blank in _QSA_CYCLE_STATE:
        setattr(destination, name, getattr(source, name, blank))
    destination._qsa_summary_identity = getattr(source, "_qsa_summary_identity", None)
    destination._qsa_summary_restored = getattr(source, "_qsa_summary_restored", False)
    destination._qsa_pending_pooled = None
    if hasattr(destination, "_max_left_pad"):
        destination._max_left_pad = None


def _check_qsa_quantization_boundary(cache, cursor: int):
    """Quantization is a cache-format boundary, never a live draft boundary."""
    if cache._mtp_share_topk or cache._mtp_shared_topk is not None:
        raise RuntimeError(
            "QSA KV cannot change representation during an armed MTP cycle."
        )
    if cache.index_keys is not None and cache.index_keys.shape[1] < cursor:
        raise RuntimeError(
            "QSA raw-key ledger is shorter than the KV cursor at quantization."
        )


class QSAQuantizedKVCache(QSAKVCache):
    """Quantized attention K/V plus QSA's native raw index-key ledger."""

    _validate_config = QuantizedKVCache._validate_config
    _validate_state_geometry = QuantizedKVCache._validate_state_geometry
    _channel_scale = staticmethod(QuantizedKVCache._channel_scale)

    def __init__(
        self,
        group_size: int = 64,
        bits: int = 8,
        *,
        key_bits: Optional[int] = None,
        value_bits: Optional[int] = None,
        rotate: bool = False,
        normalize: bool = False,
        summary_identity=None,
    ):
        QuantizedKVCache.__init__(
            self,
            group_size=group_size,
            bits=bits,
            key_bits=key_bits,
            value_bits=value_bits,
            rotate=rotate,
            normalize=normalize,
        )
        _init_qsa_summary_state(self, summary_identity)

    @classmethod
    def from_unquantized(
        cls,
        source,
        group_size: int = 64,
        bits: int = 4,
        *,
        key_bits: Optional[int] = None,
        value_bits: Optional[int] = None,
        rotate: bool = False,
        normalize: bool = False,
    ):
        _check_qsa_quantization_boundary(source, source.offset)
        cache = cls(
            group_size=group_size,
            bits=bits,
            key_bits=key_bits,
            value_bits=value_bits,
            rotate=rotate,
            normalize=normalize,
        )
        if source.keys is not None:
            plain = KVCache()
            plain.keys = mx.contiguous(source.keys[..., : source.offset, :])
            plain.values = mx.contiguous(source.values[..., : source.offset, :])
            plain.offset = source.offset
            packed = KVCache.to_quantized(
                plain,
                group_size=group_size,
                bits=bits,
                key_bits=key_bits,
                value_bits=value_bits,
                rotate=rotate,
                normalize=normalize,
            )
            cache.keys = packed.keys
            cache.values = packed.values
            cache.key_scale = packed.key_scale
            cache.value_scale = packed.value_scale
        cache.offset = source.offset
        _copy_qsa_auxiliary_state(source, cache)
        if cache.index_keys is not None:
            cache.index_keys = mx.contiguous(cache.index_keys[:, : cache.offset])
        return cache

    def update_and_fetch(self, keys, values):
        return QuantizedKVCache.update_and_fetch(self, keys, values)

    def keys_and_values(self):
        return QuantizedKVCache.keys_and_values(self)

    def trim(self, n):
        n = QuantizedKVCache.trim(self, n)
        self.release_qsa_cycle("QSAQuantizedKVCache.trim")
        return n

    @classmethod
    def merge(cls, caches):
        return BatchQSAQuantizedKVCache.merge(caches)

    def to_quantized(
        self,
        group_size: int = 64,
        bits: int = 4,
        *,
        key_bits: Optional[int] = None,
        value_bits: Optional[int] = None,
        rotate: bool = False,
        normalize: bool = False,
    ):
        key_bits = bits if key_bits is None else key_bits
        value_bits = bits if value_bits is None else value_bits
        current = (
            self.group_size,
            self.key_bits,
            self.value_bits,
            self.rotate,
            self.normalize,
        )
        requested = (group_size, key_bits, value_bits, rotate, normalize)
        if current != requested:
            raise NotImplementedError(
                "Re-quantizing an already packed QSA cache is not supported."
            )
        return self

    @property
    def state(self):
        base = (QuantizedKVCache.state.fget(self), self.index_keys)
        return _qsa_summary_state(self, base)

    @state.setter
    def state(self, value):
        value = _qsa_summary_restore_state(self, value, 2)
        QuantizedKVCache.state.fset(self, value[0])
        self.index_keys = value[1]
        self.offset = 0
        for name, blank in _QSA_CYCLE_STATE:
            setattr(self, name, blank)
        self._qsa_summary_identity = None
        self._qsa_summary_restored = False

    @property
    def meta_state(self):
        return _qsa_summary_meta(self, QuantizedKVCache.meta_state.fget(self))

    @meta_state.setter
    def meta_state(self, value):
        (base, identity, reason) = _qsa_summary_split_meta(value)
        QuantizedKVCache.meta_state.fset(self, base)
        self.release_qsa_cycle("QSAQuantizedKVCache.meta_state", keep_pooled=False)
        _qsa_summary_finish_restore(self, identity, reason)

    @property
    def nbytes(self):
        packed = 0
        if self.keys is not None:
            packed = sum((x.nbytes for x in (*self.keys, *self.values)))
        summary = 0 if self._qsa_pooled_keys is None else self._qsa_pooled_keys.nbytes
        return (
            packed
            + (0 if self.index_keys is None else self.index_keys.nbytes)
            + summary
        )


class BatchQSAQuantizedKVCache(BatchQSAKVCache):
    """Ragged batch QSA cache with packed attention K/V and native ledger."""

    _quantize = BatchQuantizedKVCache._quantize

    def __init__(
        self,
        left_padding: List[int],
        group_size: int = 64,
        bits: int = 8,
        *,
        key_bits: Optional[int] = None,
        value_bits: Optional[int] = None,
        rotate: bool = False,
        summary_identity=None,
    ):
        BatchQuantizedKVCache.__init__(
            self,
            left_padding,
            group_size=group_size,
            bits=bits,
            key_bits=key_bits,
            value_bits=value_bits,
            rotate=rotate,
        )
        _init_qsa_summary_state(self, summary_identity)
        BatchKVCache._configure_attention_backend(self, "sdpa")

    @classmethod
    def from_unquantized(
        cls,
        source,
        group_size: int = 64,
        bits: int = 4,
        *,
        key_bits: Optional[int] = None,
        value_bits: Optional[int] = None,
        rotate: bool = False,
    ):
        _check_qsa_quantization_boundary(source, source._idx)
        rows = int(source.offset.shape[0])
        cache = cls(
            [0] * rows,
            group_size=group_size,
            bits=bits,
            key_bits=key_bits,
            value_bits=value_bits,
            rotate=rotate,
        )
        cache.left_padding = source.left_padding
        cache.offset = source.offset
        cache._idx = source._idx
        cache._right_padding = source._right_padding
        if source.keys is not None:
            keys = mx.contiguous(source.keys[..., : source._idx, :])
            values = mx.contiguous(source.values[..., : source._idx, :])
            cache.keys = cache._quantize(keys, cache.key_bits)
            cache.values = cache._quantize(values, cache.value_bits)
        _copy_qsa_auxiliary_state(source, cache)
        if cache.index_keys is not None:
            cache.index_keys = mx.contiguous(cache.index_keys[:, : cache._idx])
        return cache

    def update_and_fetch(self, keys, values):
        return BatchQuantizedKVCache.update_and_fetch(self, keys, values)

    def keys_and_values(self):
        if self.keys is None:
            return (None, None)
        if self._idx == self.keys[0].shape[2]:
            return (self.keys, self.values)
        return (
            tuple((x[..., : self._idx, :] for x in self.keys)),
            tuple((x[..., : self._idx, :] for x in self.values)),
        )

    def trim(self, n):
        n = BatchQuantizedKVCache.trim(self, n)
        self.release_qsa_cycle("BatchQSAQuantizedKVCache.trim")
        return n

    def trim_ragged(self, n, *, validate: bool = True):
        drops = BatchQuantizedKVCache.trim_ragged(self, n, validate=validate)
        self.release_qsa_cycle("BatchQSAQuantizedKVCache.trim_ragged")
        return drops

    def preflight_ragged_trim(self, n, *, validate: bool = True):
        return BatchQuantizedKVCache.preflight_ragged_trim(self, n, validate=validate)

    def prepare(self, *args, **kwargs):
        BatchQuantizedKVCache.prepare(self, *args, **kwargs)
        self.release_qsa_cycle("BatchQSAQuantizedKVCache.prepare")

    def prepare_self_mtp_step(self, *args, **kwargs):
        if not self._mtp_share_topk:
            return self.prepare(*args, **kwargs)
        BatchQuantizedKVCache.prepare(self, *args, **kwargs)
        self.release_qsa_cycle(
            "BatchQSAQuantizedKVCache.prepare_self_mtp_step",
            cursor_final=False,
            keep_shared=True,
        )

    def _finalize(self, *, keep_shared=False):
        padding = self._right_padding
        if padding is not None and self.index_keys is not None:
            self.index_keys = dynamic_roll(self.index_keys, padding, axis=1)
        BatchQuantizedKVCache.finalize(self)
        self.release_qsa_cycle(
            "BatchQSAQuantizedKVCache.finalize",
            cursor_final=not keep_shared,
            keep_shared=keep_shared,
        )

    def finalize(self):
        self._finalize()

    def finalize_self_mtp_step(self):
        if not self._mtp_share_topk:
            return self.finalize()
        self._finalize(keep_shared=True)

    def filter(self, batch_indices):
        min_left_pad = min(
            int(self.left_padding[batch_indices].min().item()), self._idx
        )
        if self.index_keys is not None:
            self.index_keys = self.index_keys[batch_indices]
            if min_left_pad > 0:
                self.index_keys = self.index_keys[:, min_left_pad:]
        BatchQuantizedKVCache.filter(self, batch_indices)
        self.release_qsa_cycle("BatchQSAQuantizedKVCache.filter", rows=batch_indices)

    def extend(self, other):
        (index_a, index_b) = (self.index_keys, other.index_keys)
        (idx_a, idx_b) = (self._idx, other._idx)
        who = "BatchQSAQuantizedKVCache.extend"
        _qsa_join_ledger_width(index_a, idx_a, who)
        _qsa_join_ledger_width(index_b, idx_b, who)
        if index_a is None and index_b is None:
            merged_index = None
        else:
            populated = index_a if index_a is not None else index_b
            (dims, dtype) = (populated.shape[-1], populated.dtype)
            max_idx = max(idx_a, idx_b)

            def pad(index, idx, batch):
                if index is None:
                    index = mx.zeros((batch, 0, dims), dtype=dtype)
                else:
                    index = index[:, :idx]
                return mx.pad(index, [(0, 0), (max_idx - index.shape[1], 0), (0, 0)])

            merged_index = mx.concatenate(
                [
                    pad(index_a, idx_a, self.offset.shape[0]),
                    pad(index_b, idx_b, other.offset.shape[0]),
                ]
            )
        BatchQuantizedKVCache.extend(self, other)
        self.index_keys = merged_index
        self._max_left_pad = None
        self.release_qsa_cycle("BatchQSAQuantizedKVCache.extend")

    def extract(self, idx):
        self._reconcile_index_ledger("BatchQSAQuantizedKVCache.extract")
        cache = QSAQuantizedKVCache(
            group_size=self.group_size,
            bits=self.key_bits,
            key_bits=self.key_bits,
            value_bits=self.value_bits,
            rotate=self.rotate,
            summary_identity=self._qsa_summary_identity,
        )
        if self.keys is None:
            return cache
        padding = int(self.left_padding[idx].item())
        end = self._idx
        if self._right_padding is not None:
            end -= int(self._right_padding[idx].item())
        cache.keys = tuple(
            (mx.contiguous(x[idx : idx + 1, :, padding:end]) for x in self.keys)
        )
        cache.values = tuple(
            (mx.contiguous(x[idx : idx + 1, :, padding:end]) for x in self.values)
        )
        cache.offset = cache.keys[0].shape[2]
        if self.index_keys is not None:
            cache.index_keys = mx.contiguous(
                self.index_keys[idx : idx + 1, padding:end]
            )
        if self._qsa_pooled_keys is not None and self._qsa_pooled_ratio:
            complete = min(
                cache.offset // self._qsa_pooled_ratio, self._qsa_pooled_keys.shape[1]
            )
            if complete:
                cache._qsa_pooled_keys = mx.contiguous(
                    self._qsa_pooled_keys[idx : idx + 1, :complete]
                )
                cache._qsa_pooled_ratio = self._qsa_pooled_ratio
                cache._qsa_summary_identity = _qsa_summary_with_coverage(
                    self._qsa_summary_identity, complete
                )
        return cache

    @classmethod
    def merge(cls, caches):
        base = BatchQuantizedKVCache.merge(caches)
        batch = cls(
            [int(x) for x in base.left_padding.tolist()],
            group_size=base.group_size,
            bits=base.key_bits,
            key_bits=base.key_bits,
            value_bits=base.value_bits,
            rotate=base.rotate,
        )
        batch.keys = base.keys
        batch.values = base.values
        batch.offset = base.offset
        batch.left_padding = base.left_padding
        batch._idx = base._idx
        lengths = [cache.size() for cache in caches]
        width = max(lengths)
        populated = next(
            (cache.index_keys for cache in caches if cache.index_keys is not None), None
        )
        if populated is not None:
            (dims, dtype) = (populated.shape[-1], populated.dtype)
            rows = []
            for cache, length in zip(caches, lengths):
                _qsa_join_ledger_width(
                    cache.index_keys, length, "BatchQSAQuantizedKVCache.merge"
                )
                values = cache.index_keys
                if values is None:
                    values = mx.zeros((1, 0, dims), dtype=dtype)
                else:
                    values = values[:, :length]
                rows.append(
                    mx.pad(values, [(0, 0), (width - values.shape[1], 0), (0, 0)])
                )
            batch.index_keys = mx.concatenate(rows)
        (pooled, identity) = _qsa_merge_summaries(caches, lengths)
        if identity is not None:
            batch._qsa_summary_identity = identity
        if pooled is not None:
            batch._qsa_pooled_keys = pooled
            batch._qsa_pooled_ratio = int(caches[0]._qsa_pooled_ratio)
        return batch

    def to_quantized(
        self,
        group_size: int = 64,
        bits: int = 4,
        *,
        key_bits: Optional[int] = None,
        value_bits: Optional[int] = None,
        rotate: bool = False,
        normalize: bool = False,
    ):
        key_bits = bits if key_bits is None else key_bits
        value_bits = bits if value_bits is None else value_bits
        current = (self.group_size, self.key_bits, self.value_bits, self.rotate)
        requested = (group_size, key_bits, value_bits, rotate)
        if normalize or current != requested:
            raise NotImplementedError(
                "Re-quantizing an already packed batched QSA cache is not supported."
            )
        return self

    @property
    def state(self):
        base = (BatchQuantizedKVCache.state.fget(self), self.index_keys)
        return _qsa_summary_state(self, base)

    @state.setter
    def state(self, value):
        value = _qsa_summary_restore_state(self, value, 2)
        BatchQuantizedKVCache.state.fset(self, value[0])
        self.index_keys = value[1]
        self._max_left_pad = None
        for name, blank in _QSA_CYCLE_STATE:
            setattr(self, name, blank)
        self._qsa_summary_identity = None
        self._qsa_summary_restored = False
        BatchKVCache._configure_attention_backend(self, "sdpa")

    @property
    def meta_state(self):
        return _qsa_summary_meta(self, BatchQuantizedKVCache.meta_state.fget(self))

    @meta_state.setter
    def meta_state(self, value):
        (base, identity, reason) = _qsa_summary_split_meta(value)
        BatchQuantizedKVCache.meta_state.fset(self, base)
        BatchKVCache._configure_attention_backend(self, "sdpa")
        _qsa_summary_finish_restore(self, identity, reason)
        self._reconcile_index_ledger("BatchQSAQuantizedKVCache.meta_state")

    @property
    def nbytes(self):
        packed = 0
        if self.keys is not None:
            packed = sum((x.nbytes for x in (*self.keys, *self.values)))
        summary = 0 if self._qsa_pooled_keys is None else self._qsa_pooled_keys.nbytes
        return (
            packed
            + (0 if self.index_keys is None else self.index_keys.nbytes)
            + summary
        )


@dataclass(frozen=True)
class QSACompactBlocks:
    """Sorted, prefix-packed block ids, the shape a gather kernel wants.

    Ids are LOGICAL.  Row ``b``'s physical block start is
    ``left_padding[b] + block_id * block_size``, or just ``block_id *
    block_size`` without left padding.  ``[tail_start, tail_stop)`` is the
    incomplete block no selection names, and may be empty.  ``causal_mask``
    stays attached because blocks alone are NOT causally complete; see
    ``QSASelection.dense_mask``.
    """

    block_ids: mx.array
    block_counts: mx.array
    tail_start: mx.array
    tail_stop: mx.array
    left_padding: Optional[mx.array]
    block_size: int
    physical_width: int
    causal_mask: Optional[mx.array]

    @property
    def block_valid(self) -> mx.array:
        """Derived, never stored: a second tensor could desynchronize."""
        return mx.arange(self.block_ids.shape[-1]) < self.block_counts[..., None]


def _compact_qsa_block_ids(
    selected_block_ids: mx.array, selected_is_valid: mx.array, *, n_blocks: int
):
    """Return sorted, prefix-packed logical ids and their counts.

    ``argpartition`` gives no order, and invalid slots sit anywhere -- with
    one valid block the first eight slots are invalid -- so a prefix scan of
    the raw ids would be wrong.  Keying invalid slots at ``n_blocks``, past
    every real id, sorts them to the end instead.
    """
    if selected_block_ids.ndim != 3 or selected_is_valid.ndim != 3:
        raise ValueError("compaction wants [B, L, K] ids and validity")
    if selected_block_ids.shape != selected_is_valid.shape:
        raise ValueError(
            f"compaction shape mismatch: {selected_block_ids.shape} ids vs {selected_is_valid.shape} validity"
        )
    width = selected_block_ids.shape[-1]
    keys = mx.where(
        selected_is_valid,
        selected_block_ids.astype(mx.int32),
        mx.array(n_blocks, dtype=mx.int32),
    )
    ids = mx.take_along_axis(selected_block_ids, mx.argsort(keys, axis=-1), axis=-1)
    counts = mx.sum(selected_is_valid.astype(mx.int32), axis=-1)
    packed = mx.arange(width) < counts[..., None]
    return (mx.where(packed, ids, mx.zeros_like(ids)), counts)


@dataclass(frozen=True)
class QSASelection:
    """What the indexer chose, before it becomes an attention mask.

    ``dense_mask()`` rebuilds today's mask array operation for operation, so
    nothing observable changes.  ``compact_blocks()`` is the gather-shaped
    view and is LAZY: the mask path never calls it and pays nothing.

    Kinds: ``explicit`` is the normal sparse selection, ``implicit_all`` is a
    dense step whose mask IS the causal mask, and ``mask_only`` is the
    ``SinkWindowKVCache`` path, which has no QSA blocks at all.
    """

    kind: str
    batch: int
    length: int
    block_size: int
    raw_block_ids: Optional[mx.array] = None
    valid_blocks: Optional[mx.array] = None
    q_positions: Optional[mx.array] = None
    token_positions: Optional[mx.array] = None
    causal_mask: Optional[mx.array] = None
    passthrough_mask: Optional[mx.array] = None
    left_padding: Optional[mx.array] = None
    offset: Union[int, mx.array] = 0
    physical_width: int = 0
    n_blocks: int = 0
    scatter_chosen: bool = False

    def __post_init__(self):
        if self.kind not in ("explicit", "implicit_all", "mask_only"):
            raise ValueError(f"unknown QSA selection kind {self.kind!r}")
        if self.kind == "mask_only":
            return
        if self.physical_width // self.block_size != self.n_blocks:
            raise ValueError("block grid does not match the physical width")
        if self.left_padding is not None and self.left_padding.shape != (self.batch,):
            raise ValueError("left padding wants one entry per row")
        if self.causal_mask is not None and self.causal_mask.shape[-2:] != (
            self.length,
            self.physical_width,
        ):
            raise ValueError("causal mask must end in [L, physical width]")
        if self.kind == "implicit_all":
            return
        if self.raw_block_ids.ndim != 3 or self.valid_blocks.ndim != 3:
            raise ValueError("explicit selection wants rank-3 ids and validity")
        if self.raw_block_ids.shape[:2] != (self.batch, self.length):
            raise ValueError("selected ids must be [B, L, K]")
        if self.raw_block_ids.shape[-1] > self.n_blocks:
            raise ValueError("more selected ids than blocks")
        if self.valid_blocks.shape[1:] != (self.length, self.n_blocks):
            raise ValueError("block validity must be [B or 1, L, N]")
        if self.q_positions.shape[-1] != self.length:
            raise ValueError("query positions must be [B or 1, L]")
        if self.token_positions.shape[-1] != self.physical_width:
            raise ValueError("token positions must span the physical width")

    def dense_mask(self) -> Optional[mx.array]:
        """Today's QSA mask, rebuilt operation for operation."""
        if self.kind == "mask_only":
            return self.passthrough_mask
        if self.kind == "implicit_all":
            return self.causal_mask
        (batch, length) = (self.batch, self.length)
        (n_blocks, total) = (self.n_blocks, self.physical_width)
        (selected, valid_blocks) = (self.raw_block_ids, self.valid_blocks)
        (q_pos, token_logical) = (self.q_positions, self.token_positions)
        if self.scatter_chosen:
            _lv.bump("qsa_scatter_chosen_calls")
            chosen = mx.put_along_axis(
                mx.zeros((batch, length, n_blocks), dtype=mx.bool_),
                selected,
                mx.array(True),
                axis=-1,
            )
        else:
            block_ids = mx.arange(n_blocks)
            chosen = mx.any(
                selected[..., None] == block_ids[None, None, None, :], axis=-2
            )
        chosen = chosen & valid_blocks
        token_block = mx.clip(token_logical // self.block_size, 0, n_blocks - 1)
        selected_tokens = mx.take_along_axis(
            chosen,
            mx.broadcast_to(token_block[:, None, :], (batch, length, total)),
            axis=-1,
        )
        complete = (q_pos + 1) // self.block_size * self.block_size
        tail = (token_logical[:, None, :] >= complete[..., None]) & (
            token_logical[:, None, :] <= q_pos[..., None]
        )
        sparse = selected_tokens | tail
        if self.left_padding is not None:
            sparse = sparse & (token_logical[:, None, :] >= 0)
        sparse = sparse[:, None, :, :]
        return sparse if self.causal_mask is None else self.causal_mask & sparse

    def _logical_coordinates(self):
        """Rebuild ``(q_pos, token_logical)`` for a kind that never derived them.

        Deliberately NOT shared with ``QSAIndexer.__call__``: the dense path
        returns before it needs coordinates, and making it compute them would
        put new work on the hot path.
        """
        if self.left_padding is None:
            return (
                mx.arange(self.offset, self.offset + self.length)[None, :],
                mx.arange(self.physical_width)[None, :],
            )
        return (
            self.offset[:, None] + mx.arange(self.length)[None, :],
            mx.arange(self.physical_width)[None, :] - self.left_padding[:, None],
        )

    def compact_blocks(self) -> Optional[QSACompactBlocks]:
        """Sorted, prefix-packed blocks for a future gather kernel.

        Lazy on purpose: sorting up to ``block_topk`` ids on every masked SDPA
        call would be new hot-path work for no change in output.
        """
        if self.kind == "mask_only":
            return None
        if self.kind == "implicit_all":
            (q_pos, _) = self._logical_coordinates()
            starts = mx.arange(self.n_blocks) * self.block_size
            valid_blocks = (starts + self.block_size - 1)[None, None, :] <= q_pos[
                ..., None
            ]
            ids = mx.broadcast_to(
                mx.arange(self.n_blocks, dtype=mx.uint32)[None, None, :],
                (self.batch, self.length, self.n_blocks),
            )
        else:
            (q_pos, valid_blocks, ids) = (
                self.q_positions,
                self.valid_blocks,
                self.raw_block_ids,
            )
        if valid_blocks.shape[0] != ids.shape[0]:
            valid_blocks = mx.broadcast_to(
                valid_blocks, (ids.shape[0],) + valid_blocks.shape[1:]
            )
        (block_ids, counts) = _compact_qsa_block_ids(
            ids, mx.take_along_axis(valid_blocks, ids, axis=-1), n_blocks=self.n_blocks
        )
        tail_stop = q_pos + 1
        tail_start = tail_stop // self.block_size * self.block_size
        if tail_stop.shape[0] != self.batch:
            shape = (self.batch, self.length)
            tail_stop = mx.broadcast_to(tail_stop, shape)
            tail_start = mx.broadcast_to(tail_start, shape)
        return QSACompactBlocks(
            block_ids=block_ids,
            block_counts=counts,
            tail_start=tail_start,
            tail_stop=tail_stop,
            left_padding=self.left_padding,
            block_size=self.block_size,
            physical_width=self.physical_width,
            causal_mask=self.causal_mask,
        )


def _gather_qsa_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    compact: QSACompactBlocks,
    *,
    scale: float,
    tile_rows: int,
) -> mx.array:
    """Exact selected-row gather followed by bounded dense SDPA.

    ``q`` is ``[B,H,L,D]`` and ``k/v`` are ``[B,HKV,T,D]``.  QSA selection
    varies by both batch row and query row, so those axes are flattened and
    tiled.  The last tile may be short.  Within a tile every row retains its
    own token indices and validity mask; padding is computational only and can
    never become an attended key.

    The helper deliberately consumes the same compact contract as NAX.  It
    does not inspect a dynamic count on the host: doing so in twelve QSA layers
    on every decode step would serialize the command stream.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("QSA gather wants rank-4 q/k/v tensors")
    if k.shape != v.shape:
        raise ValueError("QSA gather K/V shapes must match")
    (batch, heads, length, dim) = q.shape
    if k.shape[0] != batch or k.shape[2] != compact.physical_width:
        raise ValueError("QSA gather tensors do not match compact selection")
    if compact.block_ids.shape[:2] != (batch, length):
        raise ValueError("QSA gather selection must match [batch, query]")
    if tile_rows < 1:
        raise ValueError("QSA gather tile_rows must be positive")
    (_, _, _, _, _, _, _, physical, valid) = compact_token_validity(compact)
    physical = physical.reshape(batch, length, -1)
    valid = valid.reshape(batch, length, -1)
    width = physical.shape[-1]
    total = k.shape[2]
    hkv = k.shape[1]
    rows = batch * length
    gather_index = mx.broadcast_to(
        physical[:, None, :, :, None], (batch, hkv, length, width, dim)
    ).astype(mx.uint32)
    gathered_k = mx.take_along_axis(
        mx.broadcast_to(k[:, :, None, :, :], (batch, hkv, length, total, dim)),
        gather_index,
        axis=3,
    )
    gathered_v = mx.take_along_axis(
        mx.broadcast_to(v[:, :, None, :, :], (batch, hkv, length, total, dim)),
        gather_index,
        axis=3,
    )
    gathered_k = gathered_k.transpose(0, 2, 1, 3, 4).reshape(rows, hkv, width, dim)
    gathered_v = gathered_v.transpose(0, 2, 1, 3, 4).reshape(rows, hkv, width, dim)
    q_rows = q.transpose(0, 2, 1, 3).reshape(rows, heads, 1, dim)
    row_valid = valid.reshape(rows, width)
    out = mx.fast.scaled_dot_product_attention(
        q_rows, gathered_k, gathered_v, scale=scale, mask=row_valid[:, None, None, :]
    )[:, :, 0]
    out = mx.where(mx.any(row_valid, axis=-1)[:, None, None], out, mx.zeros_like(out))
    return out.reshape(batch, length, heads, dim).transpose(0, 2, 1, 3)


def qsa_dense_attention_from_selection(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    selection: QSASelection,
    cache: QSAKVCache,
    *,
    scale: float,
) -> mx.array:
    """Consume an existing QSA selection/fetched cache without mutation.

    This is the fail-closed continuation for a segmented private-delta kernel
    that declines after the index ledger and K/V append. It reproduces the
    stock dense-mask call from that point instead of re-entering ``Attention``
    and appending the same token slab twice.
    """
    return scaled_dot_product_attention(
        q, k, v, cache=cache, scale=scale, mask=selection.dense_mask()
    )


def _gather_qsa_quantized_attention(
    q,
    q_keys,
    q_values,
    compact,
    *,
    scale: float,
    tile_rows: int,
    group_size: int,
    key_bits: int,
    value_bits: int,
):
    """Use MLX's affine dequant boundary before the existing gather path."""
    (keys, values) = dequantize_qsa_quantized_kv(
        q_keys,
        q_values,
        group_size=group_size,
        key_bits=key_bits,
        value_bits=value_bits,
    )
    return _gather_qsa_attention(
        q, keys, values, compact, scale=scale, tile_rows=tile_rows
    )


def _indexed_qsa_attention_or_gather(
    q,
    k,
    v,
    compact,
    *,
    scale: float,
    splits: int | None,
    tile_rows: int,
    output_gate=None,
):
    """Run indexed QSA or fall back with the same fetched cache tensors.

    Only synchronous Metal failures can fall back. A lazy failure after this
    function returns is outside this try block, so the first probe evaluates
    its result before caching a candidate.
    """
    length = int(q.shape[2])
    context = int(compact.physical_width)
    try:
        return qwen4_qsa_indexed_attention(
            q, k, v, compact, scale=scale, splits=splits, output_gate=output_gate
        )
    except (QSAIndexedProbeDeclined, RuntimeError) as error:
        reason = (
            error.reason
            if isinstance(error, QSAIndexedProbeDeclined)
            else "dispatch_raised"
        )
        exception_class = type(error).__name__
    record_qsa_indexed_receipt(
        engaged=False,
        reason=reason,
        length=length,
        context=context,
        splits=splits,
        exception_class=exception_class,
    )
    output = _gather_qsa_attention(q, k, v, compact, scale=scale, tile_rows=tile_rows)
    return (
        mlx_apply_output_gate(output, output_gate)
        if output_gate is not None
        else output
    )


def _indexed_qsa_quantized_attention_or_gather(
    q,
    q_keys,
    q_values,
    compact,
    *,
    scale: float,
    splits: int | None,
    tile_rows: int,
    group_size: int,
    key_bits: int,
    value_bits: int,
):
    """Run packed indexed QSA or dequantize into the existing gather route."""
    length = int(q.shape[2])
    context = int(compact.physical_width)
    try:
        return qwen4_qsa_indexed_quantized_attention(
            q,
            q_keys,
            q_values,
            compact,
            scale=scale,
            splits=splits,
            group_size=group_size,
            key_bits=key_bits,
            value_bits=value_bits,
        )
    except (QSAIndexedProbeDeclined, RuntimeError) as error:
        reason = (
            error.reason
            if isinstance(error, QSAIndexedProbeDeclined)
            else "quantized_dispatch_raised"
        )
        if reason == "probe_declined":
            reason = "quantized_probe_declined"
        exception_class = type(error).__name__
    record_qsa_indexed_receipt(
        engaged=False,
        reason=reason,
        length=length,
        context=context,
        splits=splits,
        exception_class=exception_class,
    )
    return _gather_qsa_quantized_attention(
        q,
        q_keys,
        q_values,
        compact,
        scale=scale,
        tile_rows=tile_rows,
        group_size=group_size,
        key_bits=key_bits,
        value_bits=value_bits,
    )


_QSA_INDEXED_CAPTURE_LOCK = threading.Lock()
_QSA_INDEXED_CAPTURE_LIMIT = 4
_QSA_INDEXED_CAPTURE_THRESHOLD = 0.004


def _capture_qsa_causal_slots(compact, physical):
    (batch, length) = map(int, physical.shape[:2])
    if compact.causal_mask is None:
        return mx.ones(physical.shape, dtype=mx.bool_)
    causal = mx.broadcast_to(
        compact.causal_mask, (batch, 1, length, int(compact.physical_width))
    )[:, 0]
    return mx.take_along_axis(
        causal, physical.reshape(batch, length, -1), axis=-1
    ).reshape(physical.shape)


def _capture_qsa_indexed_comparison(
    q,
    k,
    v,
    compact,
    *,
    scale: float,
    splits: int | None,
    tile_rows: int,
    layer_index: int,
    call_counter: int,
    gather_would_admit: bool,
):
    """Capture indexed and gather outputs, then preserve the gather path."""
    capture_dir = Path(os.environ["MLX_QWEN4_QSA_INDEXED_CAPTURE_DIR"])
    capture_dir.mkdir(parents=True, exist_ok=True)
    (ids, counts, n_sel, u_width, q_pos, left_pad, total, physical, _) = (
        compact_token_validity(compact)
    )
    causal_slots = _capture_qsa_causal_slots(compact, physical)
    fallback_reason = None
    try:
        indexed_out = qwen4_qsa_indexed_attention(
            q, k, v, compact, scale=scale, splits=splits
        )
    except (QSAIndexedProbeDeclined, RuntimeError) as error:
        indexed_out = None
        fallback_reason = (
            error.reason
            if isinstance(error, QSAIndexedProbeDeclined)
            else "dispatch_raised"
        )
        fallback_exception_class = type(error).__name__
    gather_out = _gather_qsa_attention(
        q, k, v, compact, scale=scale, tile_rows=tile_rows
    )
    mirror_splits = (
        int(splits) if splits is not None else min(indexed_splits_for(u_width), u_width)
    )
    mirror_out = qwen4_qsa_indexed_reference(
        q, k, v, compact, scale=scale, splits=mirror_splits
    )
    values = [
        gather_out,
        mirror_out,
        ids,
        counts,
        n_sel,
        q_pos,
        left_pad,
        compact.tail_start,
        compact.tail_stop,
        causal_slots,
    ]
    if indexed_out is not None:
        values.append(indexed_out)
    mx.eval(*values)
    gather_np = np.asarray(gather_out.astype(mx.float32))
    mirror_np = np.asarray(mirror_out.astype(mx.float32))
    mirror_delta = np.abs(mirror_np - gather_np)
    indexed_delta = None
    if indexed_out is not None:
        indexed_np = np.asarray(indexed_out.astype(mx.float32))
        indexed_delta = np.abs(indexed_np - gather_np)
        flat_argmax = int(np.argmax(indexed_delta))
        argmax = np.unravel_index(flat_argmax, indexed_delta.shape)
        per_row_head_max = indexed_delta.max(axis=-1).transpose(0, 2, 1)
        per_row_head_mean = indexed_delta.mean(axis=-1).transpose(0, 2, 1)
        max_delta = float(indexed_delta.reshape(-1)[flat_argmax])
        mean_delta = float(indexed_delta.mean())
        argmax_detail = {
            "batch": int(argmax[0]),
            "head": int(argmax[1]),
            "row": int(argmax[2]),
            "channel": int(argmax[3]),
        }
    else:
        per_row_head_max = None
        per_row_head_mean = None
        max_delta = None
        mean_delta = None
        argmax_detail = None
        record_qsa_indexed_receipt(
            engaged=False,
            reason=fallback_reason,
            length=int(q.shape[2]),
            context=int(compact.physical_width),
            splits=splits,
            exception_class=fallback_exception_class,
        )
    status = qsa_indexed_status()
    candidate = status.get("candidate")
    used_splits = mirror_splits if candidate is None else int(candidate[1])
    row_axes = tuple(range(2, mirror_delta.ndim))
    ledger = {
        "layer_index": int(layer_index),
        "call_counter": int(call_counter),
        "L": int(q.shape[2]),
        "B": int(q.shape[0]),
        "k_shape": list(map(int, k.shape)),
        "physical_width": int(compact.physical_width),
        "counts_min": int(np.asarray(counts).min()),
        "counts_max": int(np.asarray(counts).max()),
        "n_sel_min": int(np.asarray(n_sel).min()),
        "n_sel_max": int(np.asarray(n_sel).max()),
        "left_pad": np.asarray(left_pad).astype(np.int64).tolist(),
        "q_pos": np.asarray(q_pos).astype(np.int64).tolist(),
        "tail_start": np.asarray(compact.tail_start).astype(np.int64).tolist(),
        "tail_stop": np.asarray(compact.tail_stop).astype(np.int64).tolist(),
        "causal_mask_present": compact.causal_mask is not None,
        "candidate": candidate,
        "requested_splits": None if splits is None else int(splits),
        "used_splits": int(used_splits),
        "u_width": int(u_width),
        "gather_would_admit": bool(gather_would_admit),
        "indexed_only_admission": not bool(gather_would_admit),
        "fallback": fallback_reason,
        "indexed_vs_gather_max_abs_fp32": max_delta,
        "indexed_vs_gather_mean_abs_fp32": mean_delta,
        "indexed_vs_gather_per_row_head_max_abs_fp32": None
        if per_row_head_max is None
        else per_row_head_max.tolist(),
        "indexed_vs_gather_per_row_head_mean_abs_fp32": None
        if per_row_head_mean is None
        else per_row_head_mean.tolist(),
        "indexed_vs_gather_argmax": argmax_detail,
        "mirror_vs_gather_max_abs_fp32": float(mirror_delta.max()),
        "mirror_vs_gather_mean_abs_fp32": float(mirror_delta.mean()),
        "mirror_vs_gather_per_batch_max_abs_fp32": mirror_delta.max(
            axis=row_axes
        ).tolist(),
    }
    with _QSA_INDEXED_CAPTURE_LOCK:
        if max_delta is not None and max_delta > _QSA_INDEXED_CAPTURE_THRESHOLD:
            saved = len(list(capture_dir.glob("mismatch-*.safetensors")))
            if saved < _QSA_INDEXED_CAPTURE_LIMIT:
                stem = f"mismatch-{saved + 1:02d}-layer-{int(layer_index):03d}-call-{int(call_counter):06d}"
                tensor_path = capture_dir / f"{stem}.safetensors"
                sidecar = capture_dir / f"{stem}.json"
                mx.save_safetensors(
                    str(tensor_path),
                    {
                        "q": q,
                        "k": k,
                        "v": v,
                        "ids": ids,
                        "counts": counts,
                        "n_sel": n_sel,
                        "q_pos": q_pos,
                        "left_pad": left_pad,
                        "total": mx.array([int(total)], dtype=mx.int32),
                        "causal_per_slot": causal_slots,
                        "indexed_output": indexed_out,
                        "gather_output": gather_out,
                        "mirror_output": mirror_out,
                    },
                )
                ledger["capture_file"] = tensor_path.name
                sidecar.write_text(json.dumps(ledger, indent=2) + "\n")
        with (capture_dir / "calls.jsonl").open("a") as stream:
            stream.write(json.dumps(ledger, separators=(",", ":")) + "\n")
    return gather_out


def _dispatch_qsa_indexed_with_optional_capture(
    q,
    k,
    v,
    compact,
    *,
    scale: float,
    splits: int | None,
    tile_rows: int,
    layer_index: int,
    call_counter: int,
    gather_would_admit: bool,
    output_gate=None,
):
    if os.environ.get("MLX_QWEN4_QSA_INDEXED_CAPTURE_DIR"):
        output = _capture_qsa_indexed_comparison(
            q,
            k,
            v,
            compact,
            scale=scale,
            splits=splits,
            tile_rows=tile_rows,
            layer_index=layer_index,
            call_counter=call_counter,
            gather_would_admit=gather_would_admit,
        )
        return (
            mlx_apply_output_gate(output, output_gate)
            if output_gate is not None
            else output
        )
    return _indexed_qsa_attention_or_gather(
        q,
        k,
        v,
        compact,
        scale=scale,
        splits=splits,
        tile_rows=tile_rows,
        output_gate=output_gate,
    )


class QSAIndexer(nn.Module):
    def __init__(self, args: TextModelArgs, layer_id=0):
        super().__init__()
        self.n_heads = args.indexer_n_heads
        self.head_dim = args.indexer_head_dim
        self.compress_ratio = args.indexer_compress_ratio
        self.block_topk = args.indexer_budget // args.indexer_compress_ratio
        self.rotary_dim = int(args.head_dim * args.partial_rotary_factor)
        self.rope_theta = args.rope_theta
        self.layer_id = layer_id
        self.summary_identity = _qsa_summary_identity(args, layer_id)
        self.index_qk_proj = nn.Linear(
            args.hidden_size,
            (args.indexer_n_heads + args.indexer_kv_heads) * args.indexer_head_dim,
            bias=False,
        )
        self.q_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

    def _pool_blocks(self, raw: mx.array, starts: mx.array) -> mx.array:
        """Mean-pool, layernorm, and rope closed key blocks.

        ``starts`` carries absolute block-start positions, so a partial
        recompute matches the full recompute value for value.
        """
        batch = raw.shape[0]
        pooled = (
            raw.reshape(batch, starts.shape[0], self.compress_ratio, self.head_dim)
            .astype(mx.float32)
            .mean(axis=2)
            .astype(raw.dtype)
        )
        pooled = self.k_layernorm(pooled)
        return _apply_rope_positions(
            pooled, starts[None, :], self.rotary_dim, self.rope_theta
        )

    def _pool_blocks_left_padded(
        self, all_raw: mx.array, starts: mx.array, left_pad
    ) -> mx.array:
        """Pool the logical blocks ``starts`` out of a left-padded ledger.

        Row ``b``'s logical block at ``start`` occupies physical columns
        ``[left_pad[b] + start, left_pad[b] + start + r)``, so gather each
        row's own columns before pooling.  The block grid is sized off the
        physical width, an upper bound on any row's own block count, so a
        padded row's trailing gathers run past the ledger and are clamped here.

        A clamped block IS pooled and scored -- what it can never do is reach
        the returned mask.  Row ``b``'s deepest query sits at logical
        ``total - 1 - left_pad[b]``, and a block clamps exactly when
        ``left_pad[b] + block_end > total - 1``, i.e. when
        ``block_end > total - 1 - left_pad[b] >= q_pos``: precisely the
        condition under which ``valid_blocks`` rejects it.  A NaN or Inf from
        garbage keys is block-local and is overwritten by the ``-inf`` in
        ``mx.where(valid_blocks, ...)``; an invalid id that ``argpartition``
        still returns (there can be fewer than ``k`` valid blocks) is dropped
        by ``chosen & valid_blocks``.
        """
        (batch, total, _) = all_raw.shape
        columns = mx.minimum(
            left_pad[:, None, None]
            + starts[None, :, None]
            + mx.arange(self.compress_ratio)[None, None, :],
            total - 1,
        )
        gathered = mx.take_along_axis(
            all_raw, columns.reshape(batch, -1)[..., None], axis=1
        )
        return self._pool_blocks(gathered, starts)

    def _dense_by_construction(self, n_blocks, shared_topk) -> bool:
        """True when ``causal_mask & sparse == causal_mask`` for the mask this
        call would build, i.e. when the selection removes no causal cell -- so
        returning ``causal_mask`` is exact.  Raw ``sparse`` may still be wider
        than causal; see the MLX_QWEN4_QSA_DENSE_SHORTCIRCUIT proof.
        """
        if shared_topk is None:
            return n_blocks <= self.block_topk
        return shared_topk.shape[-1] == n_blocks

    def _pooled_keys(self, all_raw, n_blocks, starts, cache, length, left_pad):
        ratio = self.compress_ratio

        def pool(first, last):
            if left_pad is None:
                return self._pool_blocks(
                    all_raw[:, first * ratio : last * ratio], starts[first:last]
                )
            return self._pool_blocks_left_padded(all_raw, starts[first:last], left_pad)

        if not (
            (_QSA_POOLED_KEY_CACHE or _QSA_APC_SUMMARIES)
            and type(cache)
            in (
                QSAKVCache,
                BatchQSAKVCache,
                QSAQuantizedKVCache,
                BatchQSAQuantizedKVCache,
            )
        ):
            return pool(0, n_blocks)
        if left_pad is None and all_raw.shape[1] != cache.offset + length:
            raise RuntimeError(
                f"QSA index_keys desync: {all_raw.shape[1]} raw keys != offset {cache.offset} + {length} new"
            )
        closed = n_blocks
        if left_pad is not None:
            closed = (all_raw.shape[1] - cache.max_left_padding()) // ratio
        cached = cache._qsa_pooled_keys
        count = 0 if cached is None else cached.shape[1]
        if _QSA_APC_SUMMARIES:
            expected = self.summary_identity
            stored = getattr(cache, "_qsa_summary_identity", None)
            shape_ok = bool(
                cached is None
                or (
                    cached.ndim == 3
                    and cached.shape[0] == all_raw.shape[0]
                    and (cached.shape[2] == self.head_dim)
                    and (cached.dtype == all_raw.dtype)
                )
            )
            coverage_ok = bool(
                cached is None
                or (
                    stored is not None
                    and int(stored.get("complete_blocks", -1)) == count
                )
            )
            identity_ok = cached is None or _qsa_summary_identity_matches(
                stored, expected
            )
            if cached is not None and (not (shape_ok and coverage_ok and identity_ok)):
                _qsa_summary_rebound(cache, 0, "summary_identity_mismatch")
                (cached, count) = (None, 0)
                _record_qsa_summary("misses", "summary_identity_mismatch", blocks=0)
            elif cached is None:
                _record_qsa_summary("misses", "summary_absent", blocks=0)
            elif count and cache._qsa_summary_restored:
                _record_qsa_summary(
                    "hits", "persisted_summary", blocks=min(count, n_blocks)
                )
            cache._qsa_summary_identity = _qsa_summary_with_coverage(expected, count)
            cache._qsa_summary_restored = False
        if cached is not None and (
            count > closed
            or cache._qsa_pooled_ratio != ratio
            or cached.shape[0] != all_raw.shape[0]
        ):
            if _QSA_APC_SUMMARIES:
                _qsa_summary_rebound(cache, 0, "summary_identity_mismatch")
            (cached, count) = (None, 0)
        if count == n_blocks:
            if count:
                _lv.bump("qsa_pooled_key_cache_hits", count)
            pooled = cached
        else:
            if count:
                _lv.bump("qsa_pooled_key_cache_hits", min(count, n_blocks))
            _lv.bump("qsa_pooled_key_cache_misses", max(0, n_blocks - count))
            new = pool(count, n_blocks)
            pooled = new if cached is None else mx.concatenate([cached, new], axis=1)
            if _QSA_APC_SUMMARIES and closed > count:
                _record_qsa_summary(
                    "recomputes", "new_complete_blocks", blocks=closed - count
                )
        cache._qsa_pooled_keys = (
            pooled if closed >= n_blocks else mx.contiguous(pooled[:, :closed])
        )
        cache._qsa_pooled_ratio = ratio
        if _QSA_APC_SUMMARIES:
            cache._qsa_summary_identity = _qsa_summary_with_coverage(
                self.summary_identity,
                0
                if cache._qsa_pooled_keys is None
                else cache._qsa_pooled_keys.shape[1],
            )
        return pooled

    def select_shared_suffix(
        self,
        hidden: mx.array,
        causal_mask: mx.array,
        cache,
        projected_qk: Optional[mx.array] = None,
    ):
        """Select blocks from one immutable base plus a row-private ledger.

        Decode never concatenates the long raw-key base with its suffix.  The
        base must carry complete pooled summaries; newly closed suffix blocks
        are pooled in their own coordinate space, and only the much smaller
        score vectors are joined before the unchanged top-k selection.
        """
        if not getattr(cache, "supports_shared_qsa_suffix", False):
            raise TypeError("shared-suffix selection requires its storage ABI")
        (batch, length, _) = hidden.shape
        if batch != 1:
            raise ValueError("one shared-suffix row is selected at a time")
        base = cache.base
        ratio = self.compress_ratio
        if int(base.length) % ratio:
            raise RuntimeError("shared QSA base is not block aligned")
        if base.pooled_keys is None or int(base.pooled_ratio or 0) != ratio:
            raise RuntimeError("shared QSA base lacks compatible pooled summaries")
        base_blocks = int(base.length) // ratio
        if int(base.pooled_keys.shape[1]) != base_blocks:
            raise RuntimeError("shared QSA base pooled coverage is incomplete")
        shared_topk = getattr(cache, "_mtp_shared_topk", None)
        offset = int(cache.offset)
        if shared_topk is None:
            qk = (
                projected_qk if projected_qk is not None else self.index_qk_proj(hidden)
            )
            (q, raw) = mx.split(qk, [self.n_heads * self.head_dim], axis=-1)
            raw = raw.reshape(batch, length, self.head_dim)
            suffix_raw = cache.append_index_keys(raw)
        total = offset + length
        n_blocks = total // ratio
        if shared_topk is not None:
            shared_topk = _extend_mtp_shared_topk(cache, shared_topk, n_blocks)
        if n_blocks == 0 or (
            _QSA_DENSE_SHORTCIRCUIT
            and self._dense_by_construction(n_blocks, shared_topk)
        ):
            if shared_topk is None and getattr(cache, "_mtp_share_topk", False):
                cache._mtp_shared_topk = mx.arange(n_blocks, dtype=mx.uint32)[None]
                cache._mtp_shared_topk_n_blocks = n_blocks
            return QSASelection(
                kind="implicit_all",
                batch=1,
                length=length,
                block_size=ratio,
                causal_mask=causal_mask,
                offset=offset,
                physical_width=total,
                n_blocks=n_blocks,
            )
        q_pos = mx.arange(offset, offset + length)[None, :]
        starts = mx.arange(n_blocks) * ratio
        valid_blocks = (starts + ratio - 1)[None, None, :] <= q_pos[..., None]
        if shared_topk is None:
            q = self.q_layernorm(q.reshape(batch, length, self.n_heads, self.head_dim))
            q = _apply_rope_positions(
                q, q_pos[..., None], self.rotary_dim, self.rope_theta
            )
            suffix_blocks = n_blocks - base_blocks
            if suffix_blocks < 0:
                raise RuntimeError("shared QSA selection precedes its base")
            cached_suffix = cache._suffix_pooled_keys
            cached_blocks = 0 if cached_suffix is None else int(cached_suffix.shape[1])
            if cached_blocks > suffix_blocks:
                cached_suffix = mx.contiguous(cached_suffix[:, :suffix_blocks])
                cached_blocks = suffix_blocks
            if cached_blocks < suffix_blocks:
                first = cached_blocks * ratio
                last = suffix_blocks * ratio
                new_starts = (
                    base.length + mx.arange(cached_blocks, suffix_blocks) * ratio
                )
                new = self._pool_blocks(suffix_raw[:, first:last], new_starts)
                cached_suffix = (
                    new
                    if cached_suffix is None
                    else mx.concatenate([cached_suffix, new], axis=1)
                )
            cache.set_suffix_pooled_keys(cached_suffix)

            def score(keys):
                values = mx.einsum(
                    "blhd,bnd->blnh", q.astype(mx.float32), keys.astype(mx.float32)
                )
                return mx.sum(mx.maximum(values, 0), axis=-1) / math.sqrt(self.head_dim)

            score_parts = [score(base.pooled_keys)]
            if cached_suffix is not None and cached_suffix.shape[1]:
                score_parts.append(score(cached_suffix))
            scores = (
                score_parts[0]
                if len(score_parts) == 1
                else mx.concatenate(score_parts, axis=-1)
            )
            scores = mx.where(valid_blocks, scores, -mx.inf)
            k = min(self.block_topk, n_blocks)
            selected = mx.argpartition(scores, kth=n_blocks - k, axis=-1)[..., -k:]
            if getattr(cache, "_mtp_share_topk", False):
                cache._mtp_shared_topk = mx.contiguous(selected[:, -1])
                cache._mtp_shared_topk_n_blocks = n_blocks
        else:
            selected = mx.broadcast_to(
                shared_topk[:, None, :], (1, length, shared_topk.shape[-1])
            )
        token_logical = mx.arange(total)[None, :]
        return QSASelection(
            kind="explicit",
            batch=1,
            length=length,
            block_size=ratio,
            raw_block_ids=selected,
            valid_blocks=valid_blocks,
            q_positions=q_pos,
            token_positions=token_logical,
            causal_mask=causal_mask,
            left_padding=None,
            offset=offset,
            physical_width=total,
            n_blocks=n_blocks,
            scatter_chosen=_QSA_SCATTER_CHOSEN,
        )

    def select_shared_suffix_batch(
        self,
        hidden: mx.array,
        causal_masks,
        caches,
        projected_qk: Optional[mx.array] = None,
    ):
        """Coalesce equal-width shared-base selection across request rows.

        Persistent suffix ledgers remain row-owned. Only their short raw-key
        slabs and pooled summaries form a transient batch, while the immutable
        base summaries are broadcast by descriptor and scored in one einsum.
        """
        caches = list(caches)
        causal_masks = list(causal_masks)
        (batch, length, _) = hidden.shape
        if batch < 2 or len(caches) != batch or len(causal_masks) != batch:
            raise ValueError("shared QSA batch selection requires every row")
        if not all(
            (getattr(cache, "supports_shared_qsa_suffix", False) for cache in caches)
        ):
            raise TypeError("shared QSA batch selection requires split rows")
        base = caches[0].base
        if any((cache.base is not base for cache in caches[1:])):
            raise ValueError("shared QSA batch rows do not share one base")
        offsets = [int(cache.offset) for cache in caches]
        if len(set(offsets)) != 1:
            raise ValueError("shared QSA batch fast path requires equal offsets")
        if any(
            (getattr(cache, "_mtp_shared_topk", None) is not None for cache in caches)
        ):
            raise ValueError("target shared-suffix rows cannot reuse draft top-k")
        ratio = self.compress_ratio
        if int(base.length) % ratio:
            raise RuntimeError("shared QSA base is not block aligned")
        if base.pooled_keys is None or int(base.pooled_ratio or 0) != ratio:
            raise RuntimeError("shared QSA base lacks compatible pooled summaries")
        base_blocks = int(base.length) // ratio
        if int(base.pooled_keys.shape[1]) != base_blocks:
            raise RuntimeError("shared QSA base pooled coverage is incomplete")
        qk = projected_qk if projected_qk is not None else self.index_qk_proj(hidden)
        (q, raw) = mx.split(qk, [self.n_heads * self.head_dim], axis=-1)
        raw = raw.reshape(batch, length, self.head_dim)
        suffix_raw_rows = [
            cache.append_index_keys(raw[index : index + 1])
            for (index, cache) in enumerate(caches)
        ]
        suffix_widths = {int(value.shape[1]) for value in suffix_raw_rows}
        if len(suffix_widths) != 1:
            raise ValueError("shared QSA batch fast path requires equal suffix widths")
        suffix_raw = mx.concatenate(suffix_raw_rows, axis=0)
        offset = offsets[0]
        total = offset + length
        n_blocks = total // ratio
        if n_blocks <= self.block_topk and _QSA_DENSE_SHORTCIRCUIT:
            raise RuntimeError("shared QSA batch fast path requires sparse selection")
        q_pos = mx.broadcast_to(
            mx.arange(offset, offset + length)[None, :], (batch, length)
        )
        starts = mx.arange(n_blocks) * ratio
        valid_blocks = (starts + ratio - 1)[None, None, :] <= q_pos[..., None]
        q = self.q_layernorm(q.reshape(batch, length, self.n_heads, self.head_dim))
        q = _apply_rope_positions(q, q_pos[..., None], self.rotary_dim, self.rope_theta)
        suffix_blocks = n_blocks - base_blocks
        cached_rows = [cache._suffix_pooled_keys for cache in caches]
        cached_counts = {
            0 if value is None else int(value.shape[1]) for value in cached_rows
        }
        if len(cached_counts) != 1:
            raise ValueError("shared QSA batch rows have unequal pooled coverage")
        cached_blocks = cached_counts.pop()
        if cached_blocks > suffix_blocks:
            cached_blocks = suffix_blocks
            cached_rows = [
                None if value is None else mx.contiguous(value[:, :suffix_blocks])
                for value in cached_rows
            ]
        cached = None if cached_blocks == 0 else mx.concatenate(cached_rows, axis=0)
        if cached_blocks < suffix_blocks:
            first = cached_blocks * ratio
            last = suffix_blocks * ratio
            new_starts = base.length + mx.arange(cached_blocks, suffix_blocks) * ratio
            new = self._pool_blocks(suffix_raw[:, first:last], new_starts)
            cached = new if cached is None else mx.concatenate([cached, new], axis=1)
        for index, cache in enumerate(caches):
            cache.set_suffix_pooled_keys(
                None if cached is None else cached[index : index + 1]
            )

        def score(keys):
            values = mx.einsum(
                "blhd,bnd->blnh", q.astype(mx.float32), keys.astype(mx.float32)
            )
            return mx.sum(mx.maximum(values, 0), axis=-1) / math.sqrt(self.head_dim)

        base_keys = mx.broadcast_to(
            base.pooled_keys,
            (batch, int(base.pooled_keys.shape[1]), int(base.pooled_keys.shape[2])),
        )
        score_parts = [score(base_keys)]
        if cached is not None and int(cached.shape[1]):
            score_parts.append(score(cached))
        scores = (
            score_parts[0]
            if len(score_parts) == 1
            else mx.concatenate(score_parts, axis=-1)
        )
        scores = mx.where(valid_blocks, scores, -mx.inf)
        k = min(self.block_topk, n_blocks)
        selected = mx.argpartition(scores, kth=n_blocks - k, axis=-1)[..., -k:]
        token_logical = mx.arange(total)[None, :]
        return [
            QSASelection(
                kind="explicit",
                batch=1,
                length=length,
                block_size=ratio,
                raw_block_ids=selected[index : index + 1],
                valid_blocks=valid_blocks[index : index + 1],
                q_positions=q_pos[index : index + 1],
                token_positions=token_logical,
                causal_mask=causal_masks[index],
                left_padding=None,
                offset=offset,
                physical_width=total,
                n_blocks=n_blocks,
                scatter_chosen=_QSA_SCATTER_CHOSEN,
            )
            for index in range(batch)
        ]

    def __call__(
        self,
        hidden: mx.array,
        causal_mask: mx.array,
        cache: QSAKVCache,
        projected_qk: Optional[mx.array] = None,
    ):
        (batch, length, _) = hidden.shape
        if isinstance(cache, SinkWindowKVCache):
            mask = cache.make_mask(length, return_array=True)
            return QSASelection(
                kind="mask_only",
                batch=batch,
                length=length,
                block_size=self.compress_ratio,
                passthrough_mask=None if mask is None else mask[None, None, :, :],
            )
        offset = 0 if cache is None else cache.offset
        left_pad = None
        if isinstance(offset, mx.array):
            left_pad = cache.left_padding.astype(offset.dtype)
        shared_topk = (
            getattr(cache, "_mtp_shared_topk", None) if cache is not None else None
        )
        if shared_topk is None:
            qk = (
                projected_qk if projected_qk is not None else self.index_qk_proj(hidden)
            )
            (q, raw) = mx.split(qk, [self.n_heads * self.head_dim], axis=-1)
            raw = raw.reshape(batch, length, self.head_dim)
            all_raw = raw if cache is None else cache.update_index_keys(raw)
            total = all_raw.shape[1]
            if left_pad is not None and total != cache._idx + length:
                raise RuntimeError(
                    f"QSA index_keys desync: {total} raw keys != physical index {cache._idx} + {length} new"
                )
        else:
            total = (cache._idx if left_pad is not None else offset) + length
        n_blocks = total // self.compress_ratio
        if shared_topk is not None:
            shared_topk = _extend_mtp_shared_topk(cache, shared_topk, n_blocks)
        if n_blocks == 0:
            return QSASelection(
                kind="implicit_all",
                batch=batch,
                length=length,
                block_size=self.compress_ratio,
                causal_mask=causal_mask,
                left_padding=left_pad,
                offset=offset,
                physical_width=total,
                n_blocks=n_blocks,
            )
        if _QSA_DENSE_SHORTCIRCUIT and self._dense_by_construction(
            n_blocks, shared_topk
        ):
            if shared_topk is None and getattr(cache, "_mtp_share_topk", False):
                cache._mtp_shared_topk = mx.contiguous(
                    mx.broadcast_to(
                        mx.arange(n_blocks, dtype=mx.uint32), (batch, n_blocks)
                    )
                )
                cache._mtp_shared_topk_n_blocks = n_blocks
            return QSASelection(
                kind="implicit_all",
                batch=batch,
                length=length,
                block_size=self.compress_ratio,
                causal_mask=causal_mask,
                left_padding=left_pad,
                offset=offset,
                physical_width=total,
                n_blocks=n_blocks,
            )
        if left_pad is not None:
            q_pos = offset[:, None] + mx.arange(length)[None, :]
            token_logical = mx.arange(total)[None, :] - left_pad[:, None]
        else:
            q_pos = mx.arange(offset, offset + length)[None, :]
            token_logical = mx.arange(total)[None, :]
        if shared_topk is None:
            q = self.q_layernorm(q.reshape(batch, length, self.n_heads, self.head_dim))
            q = _apply_rope_positions(
                q, q_pos[..., None], self.rotary_dim, self.rope_theta
            )
        starts = mx.arange(n_blocks) * self.compress_ratio
        valid_blocks = (starts + self.compress_ratio - 1)[None, None, :] <= q_pos[
            ..., None
        ]
        if shared_topk is None:
            pooled = self._pooled_keys(
                all_raw, n_blocks, starts, cache, length, left_pad
            )
            k = min(self.block_topk, n_blocks)
            stage1_reason = _qsa_stage1_admission_reason(
                length, n_blocks * self.compress_ratio
            )
            stage1_engaged = False
            if stage1_reason is None:
                if qsa_stage1_supported(
                    q,
                    pooled,
                    q_pos,
                    block_topk=self.block_topk,
                    compress_ratio=self.compress_ratio,
                ):
                    selected = qsa_stage1_select(
                        q,
                        pooled,
                        q_pos,
                        block_topk=self.block_topk,
                        compress_ratio=self.compress_ratio,
                    )
                    stage1_reason = f"engaged_{qsa_stage1_score_producer(q, pooled)}"
                    stage1_engaged = True
                else:
                    stage1_reason = "unsupported_geometry"
            if not stage1_engaged:
                scores = mx.einsum(
                    "blhd,bnd->blnh", q.astype(mx.float32), pooled.astype(mx.float32)
                )
                scores = mx.sum(mx.maximum(scores, 0), axis=-1) / math.sqrt(
                    self.head_dim
                )
                scores = mx.where(valid_blocks, scores, -mx.inf)
                selected = mx.argpartition(scores, kth=n_blocks - k, axis=-1)[..., -k:]
            if length >= _QSA_STAGE1_MIN_QUERY:
                _record_qsa_stage1(
                    engaged=stage1_engaged,
                    reason=stage1_reason,
                    batch=batch,
                    query_width=length,
                    blocks=n_blocks,
                )
            _capture_qsa_segment_inputs(
                q, pooled, q_pos, valid_blocks, selected, layer_id=self.layer_id
            )
            if cache is not None and getattr(cache, "_mtp_share_topk", False):
                shared = (
                    cache.last_valid_query(selected)
                    if isinstance(cache, BatchQSAKVCache)
                    else selected[:, -1]
                )
                cache._mtp_shared_topk = mx.contiguous(shared)
                cache._mtp_shared_topk_n_blocks = n_blocks
        else:
            selected = mx.broadcast_to(
                shared_topk[:, None, :], (batch, length, shared_topk.shape[-1])
            )
        return QSASelection(
            kind="explicit",
            batch=batch,
            length=length,
            block_size=self.compress_ratio,
            raw_block_ids=selected,
            valid_blocks=valid_blocks,
            q_positions=q_pos,
            token_positions=token_logical,
            causal_mask=causal_mask,
            left_padding=left_pad,
            offset=offset,
            physical_width=total,
            n_blocks=n_blocks,
            scatter_chosen=_QSA_SCATTER_CHOSEN,
        )


class Attention(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int = -1, summary_layer_id=None):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self._qsa_indexed_capture_calls = 0
        self.num_kv_heads = args.num_key_value_heads
        self.num_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.scale = args.head_dim ** (-0.5)
        self.q_proj = nn.Linear(
            args.hidden_size,
            self.num_heads * self.head_dim * 2,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, args.hidden_size, bias=args.attention_bias
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.indexer = QSAIndexer(
            args, layer_idx if summary_layer_id is None else summary_layer_id
        )
        self.rope = initialize_rope(
            int(args.head_dim * args.partial_rotary_factor),
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )
        object.__setattr__(self, "_qsa_fused_cache", None)
        self._nax_layout_ok = block_sparse_layout_supported(
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            block_size=args.indexer_compress_ratio,
        )

    def _fused_projection_table(self):
        modules = (self.q_proj, self.k_proj, self.v_proj, self.indexer.index_qk_proj)
        signatures = [_proj_signature(m) for m in modules]
        if any((s is None for s in signatures)):
            # A wrapped projection has no resident weight to key on or
            # concatenate; see ``GatedDeltaNet._fused_inproj_table``.
            object.__setattr__(self, "_qsa_fused_cache", None)
            return None
        key = tuple((part for m in modules for part in _proj_identity(m)))
        cached = self._qsa_fused_cache
        if cached is not None and all(
            (new is old for (new, old) in zip(key, cached[0]))
        ):
            return cached[1]
        table = None
        if signatures[0] is not None and all((s == signatures[0] for s in signatures)):
            parts = [_proj_table(m) for m in modules]
            check_materialization_budget(
                sum((table_bytes(part) for part in parts)), "QSA fused projection"
            )
            table = _concat_tables(parts, axis=0)
        object.__setattr__(self, "_qsa_fused_cache", (key, table))
        return table

    def _project_segmented_qsa(self, x: mx.array):
        """Project a segmented batch once before its B1 attention reductions."""
        if _QSA_FUSED_PROJ and (not self.training):
            table = self._fused_projection_table()
            if table is not None:
                width_q = self.num_heads * self.head_dim * 2
                width_kv = self.num_kv_heads * self.head_dim
                return mx.split(
                    _table_matmul(table, x),
                    [width_q, width_q + width_kv, width_q + 2 * width_kv],
                    axis=-1,
                )
        return (
            self.q_proj(x),
            self.k_proj(x),
            self.v_proj(x),
            self.indexer.index_qk_proj(x),
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array,
        cache: Optional[QSAKVCache],
        *,
        _projected=None,
        _return_pre_o=False,
        _selection=None,
        _fetched_kv=None,
    ):
        if (_selection is None) != (_fetched_kv is None):
            raise ValueError("existing QSA selection and fetched K/V must be paired")
        segmented_consumer = getattr(cache, "segmented_attention", None)
        if segmented_consumer is not None and _projected is None:
            return segmented_consumer(self, x, mask)
        (batch, length, _) = x.shape
        quantized_indexed = qsa_indexed_quantized_cache_config(cache)
        qg = k_flat = v_flat = fused_index_qk = None
        if _projected is not None:
            (qg, k_flat, v_flat, fused_index_qk) = _projected
        elif _QSA_FUSED_PROJ and (not self.training):
            table = self._fused_projection_table()
            if table is not None:
                width_q = self.num_heads * self.head_dim * 2
                width_kv = self.num_kv_heads * self.head_dim
                (qg, k_flat, v_flat, fused_index_qk) = mx.split(
                    _table_matmul(table, x),
                    [width_q, width_q + width_kv, width_q + 2 * width_kv],
                    axis=-1,
                )
        selection = (
            self.indexer(x, mask, cache, projected_qk=fused_index_qk)
            if _selection is None
            else _selection
        )
        nax_admission = decide_qsa_nax_admission(
            selection,
            training=self.training,
            layout_ok=self._nax_layout_ok,
            cache_layout_ok=quantized_indexed is None,
        )
        if length >= _QSA_NAX_MIN_QUERY:
            _record_qsa_nax_admission(selection, nax_admission)
        direct_nax = (
            _QSA_NAX_DECODE
            and length == 1
            and (quantized_indexed is None)
            and (selection.n_blocks > self.indexer.block_topk)
        )
        use_direct_nax = (
            direct_nax
            and (not self.training)
            and (selection.kind == "explicit")
            and self._nax_layout_ok
            and nax_kernel_available()
        )
        use_nax = nax_admission.engage or use_direct_nax
        if length == 1 and _QSA_NAX_DECODE:
            if use_direct_nax:
                reason = "engaged"
            elif selection.n_blocks <= self.indexer.block_topk:
                reason = "dense_by_construction"
            elif selection.kind != "explicit":
                reason = "selection_not_explicit"
            elif not self._nax_layout_ok:
                reason = "unsupported_layout"
            elif self.training:
                reason = "training"
            elif quantized_indexed is not None:
                reason = "unsupported_cache_layout"
            else:
                reason = "kernel_unavailable"
            _record_qsa_nax_decode(
                engaged=use_direct_nax, reason=reason, context=selection.physical_width
            )
        if use_nax:
            (use_indexed, indexed_reason) = (False, "nax_engaged")
        else:
            (use_indexed, indexed_reason) = decide_qsa_indexed_admission(
                selection,
                length=length,
                training=self.training,
                layout_ok=self._nax_layout_ok,
                cache=cache,
            )
        if qsa_indexed_enabled() and (not use_indexed):
            record_qsa_indexed_receipt(
                engaged=False,
                reason=indexed_reason,
                length=length,
                context=selection.physical_width,
            )
        gather_context_ok = selection.physical_width >= _QSA_GATHER_MIN_CONTEXT and (
            _QSA_GATHER_MAX_CONTEXT == 0
            or selection.physical_width <= _QSA_GATHER_MAX_CONTEXT
        )
        use_gather = (
            _QSA_GATHER_KV
            and (not use_nax)
            and (not use_indexed)
            and (
                not (
                    quantized_indexed is not None
                    and (quantized_indexed["rotate"] or quantized_indexed["normalize"])
                )
            )
            and (not self.training)
            and (selection.kind == "explicit")
            and (length >= _QSA_GATHER_MIN_QUERY)
            and (length <= _QSA_GATHER_MAX_QUERY)
            and gather_context_ok
        )
        sparse_mask = (
            None if use_nax or use_indexed or use_gather else selection.dense_mask()
        )
        if qg is None:
            qg = self.q_proj(x)
            k_flat = self.k_proj(x)
            v_flat = self.v_proj(x)
        (q, gate) = mx.split(qg.reshape(batch, length, self.num_heads, -1), 2, axis=-1)
        gate = gate.reshape(batch, length, -1)
        k = k_flat.reshape(batch, length, self.num_kv_heads, self.head_dim)
        v = v_flat.reshape(batch, length, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        offset = (
            _selection.offset
            if _fetched_kv is not None
            else 0
            if cache is None
            else cache.offset
        )
        (q, k) = (self.rope(q, offset=offset), self.rope(k, offset=offset))
        if _fetched_kv is not None:
            (k, v) = _fetched_kv
        elif cache is not None:
            (k, v) = cache.update_and_fetch(k, v)
        if use_nax:
            (ids, counts, n_sel, u_width, q_pos, left_pad, total) = (
                compact_blocks_to_kernel_inputs(selection.compact_blocks())
            )
            out = nax_qsa_attention(
                q,
                k,
                v,
                ids,
                counts,
                n_sel,
                q_pos,
                left_pad,
                scale=self.scale,
                u_width=u_width,
                total=total,
                n_kv_heads=self.num_kv_heads,
            ).astype(q.dtype)
        elif use_indexed:
            compact = selection.compact_blocks()
            indexed_gate_applied = (
                quantized_indexed is None
                and (not _return_pre_o)
                and fused_gate_enabled()
            )
            cached_width = (
                k[0].shape[2] if quantized_indexed is not None else k.shape[2]
            )
            if int(cached_width) != int(compact.physical_width):
                raise ValueError("indexed QSA tensors do not match compact selection")
            splits = None
            if quantized_indexed is None and os.environ.get(
                "MLX_QWEN4_QSA_INDEXED_CAPTURE_DIR"
            ):
                self._qsa_indexed_capture_calls += 1
            if quantized_indexed is not None:
                out = _indexed_qsa_quantized_attention_or_gather(
                    q,
                    k,
                    v,
                    compact,
                    scale=self.scale,
                    splits=splits,
                    tile_rows=_QSA_GATHER_TILE_ROWS,
                    group_size=quantized_indexed["group_size"],
                    key_bits=quantized_indexed["key_bits"],
                    value_bits=quantized_indexed["value_bits"],
                )
            else:
                out = _dispatch_qsa_indexed_with_optional_capture(
                    q,
                    k,
                    v,
                    compact,
                    scale=self.scale,
                    splits=splits,
                    tile_rows=_QSA_GATHER_TILE_ROWS,
                    layer_index=self.layer_idx,
                    call_counter=self._qsa_indexed_capture_calls,
                    gather_would_admit=_QSA_GATHER_KV
                    and (not use_nax)
                    and (not self.training)
                    and (selection.kind == "explicit")
                    and (length >= _QSA_GATHER_MIN_QUERY)
                    and (length <= _QSA_GATHER_MAX_QUERY)
                    and gather_context_ok,
                    output_gate=gate if indexed_gate_applied else None,
                )
        elif use_gather:
            if quantized_indexed is not None:
                out = _gather_qsa_quantized_attention(
                    q,
                    k,
                    v,
                    selection.compact_blocks(),
                    scale=self.scale,
                    tile_rows=_QSA_GATHER_TILE_ROWS,
                    group_size=quantized_indexed["group_size"],
                    key_bits=quantized_indexed["key_bits"],
                    value_bits=quantized_indexed["value_bits"],
                )
            else:
                out = _gather_qsa_attention(
                    q,
                    k,
                    v,
                    selection.compact_blocks(),
                    scale=self.scale,
                    tile_rows=_QSA_GATHER_TILE_ROWS,
                )
        else:
            out = scaled_dot_product_attention(
                q, k, v, cache=cache, scale=self.scale, mask=sparse_mask
            )
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        if _return_pre_o:
            return (out, gate)
        if use_indexed and indexed_gate_applied:
            return self.o_proj(out)
        return self.o_proj(out * mx.sigmoid(gate))


def _apply_inject(residual, branch, inject):
    """Broadcast one branch back into the H residual streams and add.

    Both reshapes are views on contiguous arrays, so the compiled span sees the
    stream layout without a copy and the caller gets the flat layout back.
    """
    if compile_glue_enabled():
        streams = residual.reshape(
            *branch.shape[:-1], inject.shape[-1], branch.shape[-1]
        )
        out = _run_glue(("inject_apply",), _build_inject_apply, streams, branch, inject)
        if out is not None:
            return out.reshape(*residual.shape)
    return residual + (branch[..., None, :] * inject[..., None]).reshape(
        *residual.shape
    )


class DecoderLayer(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int, summary_layer_id=None):
        super().__init__()
        self.is_linear = args.layer_types[layer_idx] == "linear_attention"
        self.linear_attn = GatedDeltaNet(args) if self.is_linear else None
        self.self_attn = (
            None if self.is_linear else Attention(args, layer_idx, summary_layer_id)
        )
        self.mlp = SparseMoeBlock(args)
        ple_index = (
            args.ple_layer_ids.index(layer_idx + 1)
            if layer_idx + 1 in args.ple_layer_ids
            else None
        )
        self.ple = (
            PLELayer(args, layer_idx, ple_index) if ple_index is not None else None
        )
        self.attn_hyper_connection = GatedResidual(args)
        self.mlp_hyper_connection = GatedResidual(args)

    def __call__(self, x, input_ids, mask=None, cache=None, ssm_mask=None):
        if self.ple is not None:
            x = x + self.ple(x, input_ids, cache, ssm_mask)
        (mixed, residual, inject) = self.attn_hyper_connection(x)
        if self.is_linear:
            branch = self.linear_attn(mixed, ssm_mask, cache)
        else:
            branch = self.self_attn(mixed, mask, cache)
        x = _apply_inject(residual, branch, inject)
        (mixed, residual, inject) = self.mlp_hyper_connection(x)
        branch = self.mlp(mixed)
        return _apply_inject(residual, branch, inject)


class Qwen4ExpTextModel(PipelineMixin, nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.hyper_connection_mixer = GatedResidual(args, use_combine=False)
        self.ssm_idx = next(
            (i for (i, layer) in enumerate(self.layers) if layer.is_linear), None
        )
        self.fa_idx = next(
            (i for (i, layer) in enumerate(self.layers) if not layer.is_linear), None
        )

    def __call__(self, inputs, cache=None, input_embeddings=None, return_hyper=False):
        if _SHAPE_STABLE_SHORT_FORWARD and cache is not None and (inputs.shape[1] > 1):
            outputs = [
                self(
                    inputs[:, index : index + 1],
                    cache,
                    None
                    if input_embeddings is None
                    else input_embeddings[:, index : index + 1],
                    return_hyper,
                )
                for index in range(inputs.shape[1])
            ]
            if return_hyper:
                return tuple(
                    (
                        mx.concatenate([output[field] for output in outputs], axis=1)
                        for field in range(2)
                    )
                )
            return mx.concatenate(outputs, axis=1)
        hidden = (
            self.embed_tokens(inputs) if input_embeddings is None else input_embeddings
        )
        hidden = mx.tile(hidden, (1, 1, self.args.hc_count))
        cache = [None] * len(self.layers) if cache is None else cache
        fa_mask = None
        if self.fa_idx is not None:
            fa_cache = cache[self.fa_idx]
            fa_mask = create_attention_mask(hidden, fa_cache, return_array=True)
            if fa_mask is not None and fa_mask.ndim == 2:
                fa_mask = fa_mask[None, None, :, :]
        ssm_mask = (
            create_ssm_mask(hidden, cache[self.ssm_idx])
            if self.ssm_idx is not None
            else None
        )
        forward_rows = hidden.shape[0] * hidden.shape[1]
        eager = _EAGER_DISPATCH and forward_rows <= _EAGER_DISPATCH_MAX_ROWS
        if eager:
            _lv.bump("eager_dispatch_forwards")
        elif _EAGER_DISPATCH:
            _lv.bump("eager_dispatch_row_declines")
        stride = _EAGER_DISPATCH_STRIDE
        last = len(self.layers) - 1
        for index, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            hidden = layer(hidden, inputs, fa_mask, layer_cache, ssm_mask)
            if eager and (index == last or (index + 1) % stride == 0):
                mx.async_eval(hidden)
                _lv.bump("eager_async_evals")
        mixed = self.hyper_connection_mixer(hidden)
        return (mixed, hidden) if return_hyper else mixed


class TextModel(nn.Module):
    apc_v2_layout = "qwen4-exp-layer-segments-v1"

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen4ExpTextModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    @property
    def layers(self):
        return self.model.pipeline_layers

    def __call__(self, inputs, cache=None, input_embeddings=None):
        hidden = self.model(inputs, cache, input_embeddings)
        return (
            self.model.embed_tokens.as_linear(hidden)
            if self.args.tie_word_embeddings
            else self.lm_head(hidden)
        )

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.is_linear:
                if layer.ple is not None:
                    cache = Qwen4ArraysCache(size=4)
                    cache.ple_history_fill = int(layer.ple.ple_embedding.eos_token_id)
                else:
                    cache = ArraysCache(size=2)
                caches.append(cache)
            else:
                caches.append(QSAKVCache(layer.self_attn.indexer.summary_identity))
        return caches

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        override = os.environ.get("MLX_QWEN4_NORM_CONVENTION")
        if override not in (None, "", "raw", "converted"):
            raise ValueError(
                f"MLX_QWEN4_NORM_CONVENTION must be 'raw' or 'converted', got {override!r}"
            )
        raw = any(
            (
                "conv1d.weight" in key and value.shape[-1] != 1
                for (key, value) in weights.items()
            )
        )
        if override:
            raw = override == "raw"
        zero_centered = (
            ".hc_norm.weight",
            ".norm_key.weight",
            ".norm_query.weight",
            ".norm_conv.weight",
            ".q_layernorm.weight",
            ".k_layernorm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            "hyper_connection_mixer.hc_norm.weight",
            "pre_fc_norm_embedding.weight",
            "pre_fc_norm_hidden.weight",
        )
        for key, value in list(weights.items()):
            if "conv1d.weight" in key and value.shape[-1] != 1:
                weights[key] = value.moveaxis(2, 1)
            if raw and any((key.endswith(suffix) for suffix in zero_centered)):
                weights[key] = value + 1.0
        if os.environ.get(
            "MLX_QWEN4_NORM_CONVENTION_UNCHECKED", ""
        ).strip().lower() not in ("1", "true", "yes"):
            self._check_norm_convention(weights, zero_centered, raw)
        return weights

    NORM_CONVENTION_MARGIN = 0.25

    @classmethod
    def _check_norm_convention(cls, weights, zero_centered, raw):
        """Require decisive evidence that the loaded gains are one-centered.

        A wrong zero-vs-one-centered guess loads cleanly and produces
        deterministic garbage (mlx-vlm #2041/#2045 class).  After `sanitize`
        has applied (or declined) the fold, our in-memory convention is
        one-centered, so the gains themselves have to say so.  Group every
        gain whose key ends in a `zero_centered` suffix by family, take each
        family's mean, and score three count-weighted aggregates over the
        summed n:

            A_one  = sum(n * |mean - 1|) / sum(n)          # what we applied
            A_zero = sum(n * |mean|) / sum(n)              # +1 fold missing
            A_alt  = sum(n * |mean + shift - 1|) / sum(n)  # opposite fold

        where ``shift`` un-applies the fold (-1 if it ran, +1 if it did not;
        for the not-folded case A_alt is identically A_zero).  The applied
        convention must beat both by the margin: ``A_one + MARGIN <= A_zero``
        and ``A_one + MARGIN <= A_alt``.

        A_one/A_zero is an absolute score against the one-centered target
        rather than a comparison of the two fold outcomes, so it fails closed
        twice: on a decisive wrong convention (``A_zero + MARGIN < A_one``)
        *and* on ambiguity (the two within MARGIN).  Ambiguity is the point --
        a norm-sparse artifact such as a standalone MTP head separates the
        hypotheses by only ~0.15, which a purely comparative guard passes
        silently, and that slice is exactly where a wrong answer is
        unrecoverable.  A_alt keeps the double-add signature (gains near 2,
        which is decisively neither zero- nor one-centered but sits far from
        both) refused as before.

        `linear_attn.norm.weight` is deliberately absent from `zero_centered`
        and so excluded here: the gated GDN norm is one-centered by
        construction in the source checkpoint and is never folded.
        """
        families = {}
        for key, value in weights.items():
            for suffix in zero_centered:
                if key.endswith(suffix):
                    families.setdefault(suffix, []).append(
                        value.astype(mx.float32).mean().item()
                    )
                    break
        if not families:
            return
        margin = cls.NORM_CONVENTION_MARGIN
        shift = -1.0 if raw else 1.0
        total = a_one = a_zero = a_alt = 0.0
        rows = []
        for suffix, means in families.items():
            count = len(means)
            mean = sum(means) / count
            a_one += count * abs(mean - 1.0)
            a_zero += count * abs(mean)
            a_alt += count * abs(mean + shift - 1.0)
            total += count
            rows.append(
                (
                    abs(mean - 1.0) - min(abs(mean), abs(mean + shift - 1.0)),
                    suffix,
                    count,
                    mean,
                )
            )
        a_one /= total
        a_zero /= total
        a_alt /= total
        if a_one + margin <= a_zero and a_one + margin <= a_alt:
            return
        rows.sort(reverse=True)
        worst = ", ".join(
            (
                f"{suffix} (n={count}, mean {mean:.3f})"
                for (_, suffix, count, mean) in rows[:4]
            )
        )
        applied = "raw (+1 offset applied)" if raw else "converted (no offset applied)"
        if a_zero + margin < a_one:
            verdict = "the stored RMSNorm gains are decisively zero-centered, so the +1 fold is missing"
        elif a_alt + margin < a_one:
            verdict = "the opposite fold fits the stored RMSNorm gains decisively better, so the offset has been applied the wrong number of times"
        else:
            verdict = "the stored RMSNorm gains do not decisively favour either convention, so the applied fold cannot be verified"
        raise ValueError(
            f"norm convention check failed: the fold trigger chose the {applied} convention, but {verdict} (A_one {a_one:.3f} vs A_zero {a_zero:.3f} vs A_alt {a_alt:.3f}; required A_one + {margin:.2f} <= both, over n={int(total)} gains in {len(families)} families). Worst families: {worst}. Set MLX_QWEN4_NORM_CONVENTION=raw|converted to force the fold (the check still runs), or MLX_QWEN4_NORM_CONVENTION_UNCHECKED=1 to skip this check entirely."
        )

    @property
    def quant_predicate(self):

        def predicate(path, _):
            if ".ple_embedding.ngram_embedding.shard_" in path:
                return {"group_size": 32, "bits": 4, "mode": "affine"}
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate


class Qwen4ExpMTP(nn.Module):
    """Depth-1 residual-linear-shared MTP head with HC scheme-A state."""

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.hc_count = args.hc_count
        hc_hidden = args.hc_count * args.hidden_size
        self.pre_fc_norm_embedding = GroupRMSNorm(
            args.hidden_size, None, args.rms_norm_eps
        )
        self.pre_fc_norm_hidden = GroupRMSNorm(
            hc_hidden, args.hidden_size, args.rms_norm_eps
        )
        self.fc_embedding = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.fc_hidden = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        mtp_args = replace(
            args, num_hidden_layers=1, layer_types=["full_attention"], ple_layer_ids=[]
        )
        self.layers = [DecoderLayer(mtp_args, 0, summary_layer_id="mtp:0")]
        self.hyper_connection_mixer = GatedResidual(mtp_args, use_combine=False)

    def fuse(self, embeddings: mx.array, hidden: mx.array) -> mx.array:
        embeddings = self.fc_embedding(self.pre_fc_norm_embedding(embeddings))
        hidden = self.pre_fc_norm_hidden(hidden).reshape(
            *hidden.shape[:-1], self.hc_count, self.hidden_size
        )
        hidden = self.fc_hidden(hidden)
        return (embeddings[..., None, :] + hidden).reshape(
            *hidden.shape[:-2], self.hc_count * self.hidden_size
        )


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)


class Model(nn.Module):
    apc_v2_layout = "qwen4-exp-layer-segments-v1"
    supports_speculative_rollback = True

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        text_args = TextModelArgs.from_dict(args.text_config)
        self.language_model = TextModel(text_args)
        if text_args.mtp_num_hidden_layers > 0:
            self.mtp = Qwen4ExpMTP(text_args)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        return self.language_model(inputs, cache, input_embeddings)

    @property
    def model(self):
        return self.language_model.model

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def logits(self, hidden):
        return (
            self.language_model.model.embed_tokens.as_linear(hidden)
            if self.language_model.args.tie_word_embeddings
            else self.language_model.lm_head(hidden)
        )

    def mtp_backbone(self, inputs, cache=None):
        """Return LM-head and scheme-A HC hiddens from one trunk forward."""
        return self.language_model.model(inputs, cache, return_hyper=True)

    @contextmanager
    def gdn_catchup_scope(self, enabled=False):
        """Limit fused GDN catch-up to the cached-tail target forward."""
        with _gdn_catchup_scope(bool(enabled) or _FUSED_GDN_CATCHUP_DEFAULT):
            yield

    def prefill_prefetch_hook(self):
        """Return a chunk prefetcher when any PLE table is NVMe-backed.

        The callable takes ``(chunk_tokens, previous_tokens)`` as numpy or
        mx int arrays ([T] or [B, T]) and warms the sidecar rows the chunk
        will gather, asynchronously. ``None`` when every table is resident.
        """
        embeddings = [
            layer.ple.ple_embedding
            for layer in self.language_model.model.layers
            if layer.ple is not None and layer.ple.ple_embedding.file_backed
        ]
        if not embeddings:
            return None

        def hook(chunk_tokens, previous):
            for embedding in embeddings:
                embedding.prefetch_prompt_chunk(chunk_tokens, previous)

        hook.context_len = max((e.context_len for e in embeddings))
        return hook

    def ple_prefetch_verify(self, previous_tokens, tokens) -> int:
        """Lever (a): stage the PLE rows the verify slab positions ``tokens``
        will gather, given the committed tail ``previous_tokens`` (host ints).
        Returns how many file-backed tables accepted the request."""
        count = 0
        for layer in self.language_model.model.layers:
            ple = layer.ple
            if ple is None or not ple.ple_embedding.file_backed:
                continue
            context_len = ple.ple_embedding.context_len
            prev = list(previous_tokens)[-context_len:] if context_len > 0 else []
            if ple.ple_embedding.prefetch_positions(prev, list(tokens)):
                count += 1
        return count

    def make_mtp_cache(self, window_size: Optional[int] = None, sink_size: int = 4):
        if window_size is None:
            return [
                QSAKVCache(layer.self_attn.indexer.summary_identity)
                for layer in self.mtp.layers
            ]
        return [SinkWindowKVCache(window_size, sink_size) for _ in self.mtp.layers]

    def mtp_end_cycle(self, mtp_cache):
        """Disarm QSA top-k sharing at the end of an MTP draft cycle.

        The paired hook for ``mtp_start_cycle``. Arming has to be undone by
        SOMETHING even when the cycle is abandoned -- an abort, an exception,
        a stream that disconnects mid-draft -- because a surviving armed flag
        makes the next forward reuse a stale index set and build a silently
        wrong mask, with no desync check on that branch to catch it. Call it
        AFTER the drafted span is rewound; called before, it reports the
        un-ledgered KV the cycle left behind instead of hiding it.
        """
        for cache in mtp_cache:
            if isinstance(cache, BatchQSAKVCache):
                cache.release_qsa_cycle("Model.mtp_end_cycle")
            elif isinstance(cache, QSAKVCache):
                cache.release_qsa_cycle("Model.mtp_end_cycle")

    def mtp_start_cycle(self, mtp_cache, share_qsa_indices: bool = False):
        """Reset optional QSA top-k sharing at an MTP draft-cycle boundary.

        Sharing is per lane under a batched head cache: the stored index set
        is ``[B, k]`` and each row reuses its own blocks.

        Ending the previous cycle first makes arming idempotent and self
        healing: a cycle that was abandoned without reaching a rewind cannot
        leak its index set into this one, and its un-ledgered KV is reported
        here rather than several forwards later.
        """
        self.mtp_end_cycle(mtp_cache)
        for cache in mtp_cache:
            if isinstance(cache, (QSAKVCache, BatchQSAKVCache)):
                cache._mtp_share_topk = bool(share_qsa_indices)
                cache._mtp_shared_topk = None
                cache._mtp_shared_topk_n_blocks = None

    def mtp_step(self, hidden, tokens, mtp_cache):
        embeddings = self.language_model.model.embed_tokens(tokens)
        multi = self.mtp.fuse(embeddings, hidden)
        cache = mtp_cache[0]
        mask = create_attention_mask(multi, cache, return_array=True)
        if mask is not None and mask.ndim == 2:
            mask = mask[None, None, :, :]
        multi = self.mtp.layers[0](multi, tokens, mask, cache, None)
        sample = self.mtp.hyper_connection_mixer(multi)
        return (self.logits(sample), multi)

    def sanitize(self, weights):
        has_mtp_weights = any(
            (
                key.startswith("mtp.")
                or key.startswith("model.mtp.")
                or key.startswith("language_model.mtp.")
                or key.startswith("model.language_model.mtp.")
                for key in weights
            )
        )
        if not (has_mtp_weights and getattr(self, "mtp", None) is not None):
            if getattr(self, "mtp", None) is not None:
                self.mtp = None
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("model.visual") or key.startswith("vision_tower"):
                continue
            if key.startswith("model.language_model.mtp."):
                key = key.replace("model.language_model.mtp.", "mtp.", 1)
            elif key.startswith("language_model.mtp."):
                key = key.replace("language_model.mtp.", "mtp.", 1)
            elif key.startswith("model.mtp."):
                key = key.removeprefix("model.")
            elif key.startswith("mtp."):
                if getattr(self, "mtp", None) is None:
                    continue
            elif key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model", 1)
            elif not key.startswith("language_model."):
                key = "language_model." + key
            sanitized[key] = value
        mlp_prefixes = [
            f"language_model.model.layers.{layer_idx}.mlp"
            for layer_idx in range(self.language_model.args.num_hidden_layers)
        ]
        if getattr(self, "mtp", None) is not None:
            mlp_prefixes.extend(
                (
                    f"mtp.layers.{layer_idx}.mlp"
                    for layer_idx in range(
                        self.language_model.args.mtp_num_hidden_layers
                    )
                )
            )
        for prefix in mlp_prefixes:
            gate_up_key = f"{prefix}.experts.gate_up_proj"
            if gate_up_key not in sanitized:
                continue
            gate_up = sanitized.pop(gate_up_key)
            if qwen3_next._MOE_FUSED_GATE_UP:
                sanitized[f"{prefix}.switch_mlp.gate_up_proj.weight"] = gate_up
            else:
                midpoint = gate_up.shape[-2] // 2
                sanitized[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[
                    ..., :midpoint, :
                ]
                sanitized[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[
                    ..., midpoint:, :
                ]
            sanitized[f"{prefix}.switch_mlp.down_proj.weight"] = sanitized.pop(
                f"{prefix}.experts.down_proj"
            )
        transform_moe_weights(
            sanitized,
            mlp_prefixes,
            fuse_gate_up=qwen3_next._MOE_FUSED_GATE_UP,
            fold_shared=qwen3_next._MOE_SHARED_IN_GATHER,
        )
        return self.language_model.sanitize(sanitized)

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate
