# Adapted from unified tests/test_verify_state_oracle.py at 1e2bc604, Apache-2.0.
import copy
import os
import struct
from collections import deque
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from unittest.mock import patch

import numpy as np
import pytest


os.environ.setdefault("MLX_ENABLE_TF32", "0")
os.environ.pop("MLX_GDN_CORE", None)

import mlx.core as mx

from mlx2.runtime.hybrid_speculative import (
    _finalize_self_mtp_cache_group,
    _prepare_self_mtp_cache_group,
    BatchedSelfMTPState,
    DetachedSelfMTPLane,
    SelfMTPCachePair,
    attach_self_mtp_lanes,
    commit_batched_self_mtp,
    prepare_self_mtp_lane,
    propose_batched_self_mtp,
)
from mlx2.runtime.models import gated_delta as gated_delta_module
from mlx2.runtime.models import qwen4_exp as qwen4_exp_module
from mlx2.runtime.models.cache import ArraysCache, _RollbackRecord
from mlx2.runtime.models.qwen4_exp import (
    BatchQSAKVCache,
    Model,
    ModelArgs,
    QSAKVCache,
    Qwen4ArraysCache,
    TextModelArgs,
)
from mlx2.runtime.sample_utils import LaneRNG


M = 4
PROMPTS = ([1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11])
RAGGED_ACCEPTS = tuple(product(range(M + 1), repeat=2))
DETAIL_ACCEPTS = (1, 3)


@dataclass(frozen=True)
class OracleAtom:
    kind: str
    dtype: str = ""
    shape: Tuple[int, ...] = ()
    payload: Any = None

    def summary(self) -> str:
        if self.kind == "array":
            return (
                f"array(dtype={self.dtype}, shape={self.shape}, "
                f"bytes={len(self.payload)})"
            )
        return f"{self.kind}({self.payload!r})"


Capture = Dict[str, OracleAtom]


def _array_atom(value: mx.array) -> OracleAtom:
    uint_name = {1: "uint8", 2: "uint16", 4: "uint32", 8: "uint64"}[value.dtype.size]
    bits = mx.view(value, getattr(mx, uint_name))
    mx.eval(bits)
    payload = np.array(bits, copy=True).tobytes()
    return OracleAtom("array", str(value.dtype), tuple(value.shape), payload)


def _host_atom(value: Any) -> OracleAtom:
    if value is None:
        return OracleAtom("none")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        payload = struct.pack("!d", value)
    elif isinstance(value, (bool, int, str, bytes)):
        payload = value
    else:
        payload = repr(value)
    return OracleAtom("host", type(value).__name__, payload=payload)


def _put(capture: Capture, path: str, value: Any) -> None:
    if isinstance(value, mx.array):
        capture[path] = _array_atom(value)
    else:
        capture[path] = _host_atom(value)


def _capture_sequence(capture: Capture, path: str, values: Any) -> None:
    if values is None:
        _put(capture, path, None)
        return
    values = list(values)
    _put(capture, f"{path}.count", len(values))
    for index, value in enumerate(values):
        item_path = f"{path}[{index}]"
        if isinstance(value, mx.array) or value is None:
            _put(capture, item_path, value)
        elif isinstance(value, (list, tuple, deque)):
            _capture_sequence(capture, item_path, value)
        else:
            _put(capture, item_path, value)


def _capture_host_mirror(capture: Capture, path: str, mirror: Any, source: Any) -> None:
    _put(capture, f"{path}.present", mirror is not None)
    if mirror is None:
        return
    _put(capture, f"{path}.matches_source", mirror[0] is source)
    _put(capture, f"{path}.source", mirror[0])
    _capture_sequence(capture, f"{path}.values", mirror[1])


def _materialize_tree(value: Any) -> None:
    arrays: List[mx.array] = []

    def visit(item: Any) -> None:
        if isinstance(item, mx.array):
            arrays.append(item)
        elif isinstance(item, (list, tuple, deque)):
            for child in item:
                visit(child)

    visit(value)
    if arrays:
        mx.eval(*arrays)


def _record_replays(record: _RollbackRecord) -> Dict[int, List[Any]]:
    replays = {m: list(record.fn(m)) for m in range(record.num_tokens + 1)}
    _materialize_tree(list(replays.values()))
    return replays


def _ragged_replay(
    replays: Mapping[int, Sequence[Any]], lengths: Sequence[int]
) -> List[Any]:
    selected = [replays[int(length)] for length in lengths]
    result: List[Any] = []
    for slot in range(len(selected[0])):
        values = [row[slot] for row in selected]
        if all(value is None for value in values):
            result.append(None)
        elif any(value is None for value in values):
            raise AssertionError("rollback replay mixes None and array rows")
        else:
            result.append(
                mx.concatenate(
                    [value[row : row + 1] for row, value in enumerate(values)]
                )
            )
    _materialize_tree(result)
    return result


def _ragged_vectors(num_tokens: int, batch_size: int) -> Tuple[Tuple[int, ...], ...]:
    if batch_size == 1:
        return ((0,), (num_tokens,))
    vectors = [
        (0, num_tokens),
        (num_tokens, 0),
        (min(1, num_tokens), max(0, num_tokens - 1)),
    ]
    return tuple(tuple(vector[:batch_size]) for vector in vectors)


def _capture_rollback_record(
    capture: Capture,
    path: str,
    record: _RollbackRecord,
    batch_size: int,
) -> None:
    _put(capture, f"{path}.num_tokens", record.num_tokens)
    _put(capture, f"{path}.replayable", record[0])
    _put(capture, f"{path}.span", record.span)
    _put(capture, f"{path}.depths.present", record.depths is not None)
    if record.depths is not None:
        _capture_sequence(capture, f"{path}.depths", record.depths)
    _capture_sequence(capture, f"{path}.snapshot", record.snapshot)

    replays = _record_replays(record)
    for m, replay in replays.items():
        _capture_sequence(capture, f"{path}.fn[{m}]", replay)

    _put(capture, f"{path}.per_row_fn.present", record.per_row_fn is not None)
    for lengths in _ragged_vectors(record.num_tokens, batch_size):
        label = ",".join(map(str, lengths))
        if record.per_row_fn is None:
            replay = _ragged_replay(replays, lengths)
            mode = "fallback"
        else:
            replay = list(record.per_row_fn(list(lengths)))
            _materialize_tree(replay)
            mode = "per_row_fn"
        _put(capture, f"{path}.ragged[{label}].mode", mode)
        _capture_sequence(capture, f"{path}.ragged[{label}].result", replay)


def _capture_checkpoints(capture: Capture, path: str, cache: ArraysCache) -> None:
    checkpoints = cache._checkpoints
    _put(capture, f"{path}.lane_count", len(checkpoints))
    for lane, entries in enumerate(checkpoints):
        lane_path = f"{path}.lane[{lane}]"
        _put(capture, f"{lane_path}.count", len(entries))
        for index, (position, snapshot) in enumerate(entries):
            entry_path = f"{lane_path}.entry[{index}]"
            _put(capture, f"{entry_path}.position", position)
            _capture_sequence(capture, f"{entry_path}.snapshot", snapshot)


def _capture_staged_ple(capture: Capture, path: str, cache: Qwen4ArraysCache) -> None:
    staged = cache._ple_rollback
    _put(capture, f"{path}.present", staged is not None)
    if staged is None:
        return
    num_tokens, fn, snapshot, per_row_fn = staged
    _put(capture, f"{path}.num_tokens", num_tokens)
    _capture_sequence(capture, f"{path}.snapshot", snapshot)
    for m in range(num_tokens + 1):
        replay = list(fn(m))
        _materialize_tree(replay)
        _capture_sequence(capture, f"{path}.fn[{m}]", replay)
    _put(capture, f"{path}.per_row_fn.present", per_row_fn is not None)
    if per_row_fn is not None:
        for lengths in _ragged_vectors(num_tokens, cache.batch_size):
            label = ",".join(map(str, lengths))
            replay = list(per_row_fn(list(lengths)))
            _materialize_tree(replay)
            _capture_sequence(
                capture,
                f"{path}.per_row_fn[{label}]",
                replay,
            )


def capture_cache_list(caches: Sequence[Any], prefix: str = "cache") -> Capture:
    capture: Capture = {}
    _put(capture, f"{prefix}.layer_count", len(caches))
    for layer, cache in enumerate(caches):
        path = f"{prefix}.layer[{layer}]"
        _put(capture, f"{path}.type", type(cache).__name__)

        if isinstance(cache, ArraysCache):
            _capture_sequence(capture, f"{path}.cache", cache.cache)
            for name in ("left_padding", "lengths"):
                _put(capture, f"{path}.{name}", getattr(cache, name))
            _capture_host_mirror(
                capture,
                f"{path}._host_lengths",
                cache._host_lengths,
                cache.lengths,
            )
            _capture_host_mirror(
                capture,
                f"{path}._host_left_padding",
                cache._host_left_padding,
                cache.left_padding,
            )
            for name in (
                "speculating",
                "_rollback_window",
                "_rollback_invalid_reason",
            ):
                _put(capture, f"{path}.{name}", getattr(cache, name))
            _capture_checkpoints(capture, f"{path}._checkpoints", cache)
            _put(capture, f"{path}._rollbacks.count", len(cache._rollbacks))
            for index, record in enumerate(cache._rollbacks):
                _capture_rollback_record(
                    capture,
                    f"{path}._rollbacks[{index}]",
                    record,
                    cache.batch_size,
                )
            if isinstance(cache, Qwen4ArraysCache):
                _capture_staged_ple(capture, f"{path}._ple_rollback", cache)

        if isinstance(cache, (QSAKVCache, BatchQSAKVCache)):
            # KVCache.trim() moves the logical cursor and deliberately leaves
            # rejected columns in the allocation tail; update_and_fetch()
            # overwrites those columns before they can become live again.
            # Comparing the full backing allocation therefore turns harmless
            # capacity residue into a false transactional-state mismatch.
            # Keep the cursor exact and compare only its readable prefix.
            live_width = (
                cache._idx if isinstance(cache, BatchQSAKVCache) else cache.offset
            )
            for name in ("keys", "values"):
                value = getattr(cache, name)
                if value is not None:
                    value = value[..., :live_width, :]
                _put(capture, f"{path}.{name}", value)
            for name in (
                "offset",
                "index_keys",
                "_qsa_pooled_keys",
                "_qsa_pooled_ratio",
                "_mtp_share_topk",
                "_mtp_shared_topk",
            ):
                _put(capture, f"{path}.{name}", getattr(cache, name))
            for name in ("left_padding", "_idx", "_right_padding"):
                if hasattr(cache, name):
                    _put(capture, f"{path}.{name}", getattr(cache, name))
            if hasattr(cache, "_max_left_pad"):
                mirror = cache._max_left_pad
                _put(capture, f"{path}._max_left_pad.present", mirror is not None)
                if mirror is not None:
                    _put(
                        capture,
                        f"{path}._max_left_pad.matches_source",
                        mirror[0] is cache.left_padding,
                    )
                    _put(capture, f"{path}._max_left_pad.source", mirror[0])
                    _put(capture, f"{path}._max_left_pad.value", mirror[1])
    return capture
