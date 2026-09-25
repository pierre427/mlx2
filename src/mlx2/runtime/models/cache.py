# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
import copy
import importlib
import inspect
import json
import math
import operator
import os
import sys
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_reduce, tree_unflatten
from .base import create_causal_mask, hadamard_size_ok, rotate_last

_ROOT_PACKAGE = __name__.split(".")[0]
_CACHE_CLASS_REGISTRY: Dict[str, type] = {}
_AMBIGUOUS_CACHE_NAMES: set = set()


def register_cache_class(cls):
    """Register a cache class so ``load_prompt_cache`` can resolve it by name.

    Subclasses of ``_BaseCache`` register themselves when defined; calling
    this directly (it also works as a class decorator) is only needed for
    duck-typed cache classes outside that hierarchy.
    """
    name = cls.__name__
    prev = _CACHE_CLASS_REGISTRY.get(name)
    if prev is not None and (
        prev.__module__ != cls.__module__ or prev.__qualname__ != cls.__qualname__
    ):
        _AMBIGUOUS_CACHE_NAMES.add(name)
    _CACHE_CLASS_REGISTRY[name] = cls
    return cls


def _cache_class_token(cls) -> str:
    """Serialized identifier for a cache class.

    Classes defined in this module serialize as their bare name (the format
    older files already use); model-local classes carry their defining module
    as ``module:ClassName`` so a fresh process can resolve them without having
    imported the model first.
    """
    name = cls.__name__
    if globals().get(name) is cls:
        return name
    return f"{cls.__module__}:{name}"


def _resolve_cache_class(token: str):
    """Inverse of ``_cache_class_token``, with a registry fallback for older
    files that recorded a model-local class as a bare name."""
    (module_name, _, name) = token.rpartition(":")
    if module_name:
        module = sys.modules.get(module_name)
        if module is None and module_name.split(".")[0] == _ROOT_PACKAGE:
            module = importlib.import_module(module_name)
        cls = getattr(module, name, None) if module is not None else None
        if cls is not None:
            return cls
    else:
        cls = globals().get(name)
        if cls is not None:
            return cls
        if name in _AMBIGUOUS_CACHE_NAMES:
            raise ValueError(
                f"Prompt-cache class name {name!r} is ambiguous: cache classes with that name exist in more than one imported module. Re-save the cache with this version of mlx-lm to record the defining module."
            )
    cls = _CACHE_CLASS_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown prompt-cache class {token!r}. If it is defined in a model file, import that module (e.g. by loading the model) before calling load_prompt_cache, or register it with register_cache_class."
        )
    return cls


def make_prompt_cache(model: nn.Module, max_kv_size: Optional[int] = None) -> List[Any]:
    """
    Construct the model's cache for use in generation.

    This function will defer the cache construction to the model if it has a
    ``make_cache`` method, otherwise it will make a default KV cache.

    Args:
        model (nn.Module): The language model.
        max_kv_size (Optional[int]): If provided, the attention KV cache is
            bounded to ``max_kv_size`` tokens via a ``RotatingKVCache``. Models
            with a ``make_cache`` method honor this only if their ``make_cache``
            accepts a ``max_kv_size`` argument (e.g. hybrid models cap their
            full-attention layers and leave recurrent layers unbounded);
            otherwise the model's own cache policy is used unchanged.
    """
    if hasattr(model, "make_cache"):
        params = inspect.signature(model.make_cache).parameters
        if max_kv_size is not None and "max_kv_size" in params:
            return model.make_cache(max_kv_size=max_kv_size)
        return model.make_cache()
    num_layers = len(model.layers)
    if max_kv_size is not None:
        return [
            RotatingKVCache(max_size=max_kv_size, keep=4) for _ in range(num_layers)
        ]
    else:
        return [KVCache() for _ in range(num_layers)]


def _dtype_registry() -> Dict[str, Any]:
    """Every mx dtype keyed by its ``str()``. Built once, on first use.

    ``str(mx.bool_)`` is ``"mlx.core.bool"``, so a name cannot be turned back
    into the attribute by string surgery.
    """
    global _DTYPE_BY_NAME
    if _DTYPE_BY_NAME is None:
        _DTYPE_BY_NAME = {
            str(value): value
            for value in (getattr(mx, name, None) for name in dir(mx))
            if isinstance(value, mx.Dtype)
        }
    return _DTYPE_BY_NAME


_DTYPE_BY_NAME: Optional[Dict[str, Any]] = None


def save_prompt_cache(file_name: str, cache: List[Any], metadata: Dict[str, str] = {}):
    """
    Save a pre-computed prompt cache to a file.

    Args:
        file_name (str): The ``.safetensors`` file name.
        cache (List[Any]): The model state.
        metadata (Dict[str, str]): Optional metadata to save along with model
            state.
    """
    cache_data = [c.state for c in cache]
    cache_info = [c.meta_state for c in cache]
    cache_data = dict(tree_flatten(cache_data))
    # A cache with nothing written reports ``None`` leaves (KVCache keys and
    # values; ArraysCache slots), which safetensors cannot hold either.  They
    # are recorded as ``null`` beside the zero-size arrays and rebuilt on load.
    empty = {
        key: None if value is None else (str(value.dtype), list(value.shape))
        for (key, value) in cache_data.items()
        if value is None or (isinstance(value, mx.array) and value.size == 0)
    }
    for key in empty:
        del cache_data[key]
    cache_classes = [_cache_class_token(type(c)) for c in cache]
    summary_provenance = []

    def collect_summary_provenance(entry, path):
        children = getattr(entry, "caches", None)
        if children is not None:
            for child_index, child in enumerate(children):
                collect_summary_provenance(child, [*path, child_index])
            return
        info = getattr(entry, "meta_state", ())
        if not isinstance(info, (list, tuple)) or "qsa_summary_v1" not in info:
            return
        identity = getattr(entry, "_qsa_summary_identity", None)
        if identity is not None:
            summary_provenance.append({"cache_path": path, **identity})

    for index, entry in enumerate(cache):
        collect_summary_provenance(entry, [index])
    cache_metadata = [cache_info, metadata, cache_classes]
    if empty or summary_provenance:
        cache_metadata.append(json.dumps(empty))
    if summary_provenance:
        cache_metadata.append(
            json.dumps(
                {
                    "format": "qsa_apc_summaries",
                    "version": 1,
                    "entries": summary_provenance,
                },
                sort_keys=True,
            )
        )
    cache_metadata = dict(tree_flatten(cache_metadata))
    mx.save_safetensors(file_name, cache_data, cache_metadata)


def load_prompt_cache(file_name, return_metadata=False):
    """
    Load a prompt cache from a file.

    Args:
        file_name (str): The ``.safetensors`` file name.
        return_metadata (bool): Whether or not to return metadata.
            Default: ``False``.

    Returns:
        List[Any] or Tuple[List[Any], Dict[str, str]]: The prompt cache and
            the metadata if requested.
    """
    (arrays, cache_metadata) = mx.load(file_name, return_metadata=True)
    cache_metadata = tree_unflatten(list(cache_metadata.items()))
    (info, metadata, classes) = cache_metadata[:3]
    arrays = dict(arrays)
    if len(cache_metadata) > 3:
        for key, placeholder in json.loads(cache_metadata[3]).items():
            if placeholder is None:
                arrays[key] = None
                continue
            # Only zero-size arrays are recorded here (safetensors cannot hold
            # them).  A corrupt or hostile header must not turn this into an
            # arbitrary allocation before any consistency check runs.
            (dtype, shape) = placeholder
            shape = tuple(int(dim) for dim in shape)
            if not shape or 0 not in shape or any(dim < 0 for dim in shape):
                raise ValueError(
                    f"prompt cache records a non-empty placeholder array {key!r} "
                    f"with shape {shape}"
                )
            arrays[key] = mx.zeros(shape, dtype=_dtype_registry()[dtype])
    arrays = tree_unflatten(list(arrays.items()))
    if isinstance(arrays, list):
        # A trailing cache whose state has no leaves at all (an empty
        # SinkWindowKVCache reports ``[]``) leaves no key behind; without
        # this the zip below would silently drop that layer.
        arrays.extend([] for _ in range(len(classes) - len(arrays)))
    cache = [
        _resolve_cache_class(c).from_state(state, meta_state)
        for (c, state, meta_state) in zip(classes, arrays, info)
    ]
    if return_metadata:
        return (cache, metadata)
    return cache


def can_trim_prompt_cache(cache: List[Any]) -> bool:
    """
    Check if model's cache can be trimmed.
    """
    # An empty list can be a disk-only APC placeholder.  Treating it as
    # trimmable by vacuous truth lets PrefixIndex select a checkpoint whose
    # payload was re-spilled while a neighboring candidate was restored.
    return bool(cache) and all((c.is_trimmable() for c in cache))


def _snap_trim_position(cache: List[Any], position: int) -> Optional[int]:
    """Largest position <= ``position`` that every cache in the list can be
    restored to, or ``None`` if some cache cannot restore any position.

    Iterates to a fixpoint: lowering the position for one cache (to its
    nearest state checkpoint) may lower it again for another.
    """
    while True:
        start = position
        for c in cache:
            snap = getattr(c, "snap_trim_position", None)
            # Third-party cache implementations may expose ``state`` and the
            # ordinary ``trim`` contract without supporting checkpoint-aware
            # restoration.  They are safe for exact all-cache trimming (the
            # fast path above), but must not be treated as arbitrary-branch
            # capable merely because they look like an MLX cache.  Failing
            # closed here also keeps APC capability inspection diagnostic-only
            # instead of letting an optional cache method crash publication.
            if not callable(snap):
                return None
            position = snap(position)
            if position is None or position < 0:
                return None
        if position == start:
            return position


def _absolute_position(c) -> int:
    """Absolute token position of a cache. ``RotatingKVCache.size()``
    saturates at ``max_size``; its ``offset`` is the true count."""
    position = getattr(c, "offset", None)
    if isinstance(position, int):
        return position
    return c.size() if hasattr(c, "size") else 0


def _thin_checkpoints(checkpoints: List, max_checkpoints: int):
    """Drop the checkpoint with the smallest gap to its predecessor
    (implicit predecessor at 0), never the newest: this roughly doubles the
    effective stride over the older history while keeping the
    end-of-prefill checkpoint exact. Entries are tuples whose first element
    is the position."""
    while len(checkpoints) > max_checkpoints:
        prev = 0
        gaps = []
        for j, entry in enumerate(checkpoints[:-1]):
            gaps.append((entry[0] - prev, j))
            prev = entry[0]
        (_, drop) = min(gaps)
        del checkpoints[drop]


def achievable_trim(cache: List[Any], num_tokens: int):
    """Dry-run of ``trim_prompt_cache(cache, num_tokens, allow_partial=True)``.

    Returns ``(position, actual_num_tokens)`` — the absolute position the
    cache would land on and the number of tokens that would actually be
    trimmed (``>= num_tokens`` when the landing snaps to an earlier state
    checkpoint) — or ``None`` if the cache cannot be trimmed at all.
    """
    if len(cache) == 0:
        return None
    size = max((_absolute_position(c) for c in cache), default=0)
    if size <= 0:
        return None
    target = max(0, size - num_tokens)
    if can_trim_prompt_cache(cache):
        return (target, size - target)
    position = _snap_trim_position(cache, target)
    if position is None:
        return None
    return (position, size - position)


def trim_prompt_cache(
    cache: List[Any], num_tokens: int, allow_partial: bool = False
) -> int:
    """
    Trim the model's cache by the given number of tokens.

    This function will trim the cache if possible (in-place) and return the
    number of tokens that were trimmed.

    Args:
        cache (List[Any]): The model's cache.
        num_tokens (int): The number of tokens to trim.
        allow_partial (bool): If the cache is not exactly trimmable (e.g. it
            mixes recurrent-state ``ArraysCache`` layers with KV layers),
            allow trimming *more* than ``num_tokens`` by restoring the
            recurrent layers to their nearest recorded state checkpoint at or
            before the requested position. The caller must use the returned
            count (which may exceed ``num_tokens``) to decide how many tokens
            to re-process. Default: ``False``.

    Returns:
        (int): The number of tokens that were trimmed.
    """
    if len(cache) == 0:
        return 0
    if can_trim_prompt_cache(cache):
        return [c.trim(num_tokens) for c in cache][0]
    if not allow_partial:
        return 0
    landing = achievable_trim(cache, num_tokens)
    if landing is None:
        return 0
    (position, actual) = landing
    for c in cache:
        c.trim_to_position(position, actual)
    return actual


class RaggedTrimUnsupported(RuntimeError):
    """A cache cannot rewind its rows by different amounts.

    Raised instead of falling back to a uniform trim: a silent uniform trim
    would rewind rows that accepted their drafts, which is the
    silent-misroute class of bug this API exists to prevent.
    """


class _RollbackRecord(tuple):
    """``(replayable, fn, snapshot)`` plus per-row depth and an optional replay.

    Still a plain 3-tuple for every existing reader. ``depths`` holds how many
    of the record's tokens each row still has applied — rows diverge after a
    ragged trim, so one scalar cannot describe the stack. ``None`` means every
    row is still at ``num_tokens``, which is the whole single-sequence and
    pre-divergence life of a record, so no per-row list is allocated then.
    Element 0 is the smallest per-row depth: the honest answer to the uniform
    "how much can I trim?" question every existing reader is asking.
    """

    def __new__(cls, num_tokens, fn, snapshot, per_row_fn=None, depths=None):
        if depths is not None and all((d == num_tokens for d in depths)):
            depths = None
        replayable = num_tokens if depths is None else min(depths)
        record = super().__new__(cls, (replayable, fn, snapshot))
        record.num_tokens = num_tokens
        record.per_row_fn = per_row_fn
        record.depths = depths
        return record

    @property
    def fn(self):
        return self[1]

    @property
    def snapshot(self):
        return self[2]

    @property
    def span(self):
        """Tokens of history this record still holds for some row."""
        return self.num_tokens if self.depths is None else max(self.depths)

    def depth_list(self, batch_size: int) -> List[int]:
        if self.depths is None:
            return [self.num_tokens] * batch_size
        return list(self.depths)

    def with_depths(self, depths: List[int]):
        return _RollbackRecord(
            self.num_tokens, self.fn, self.snapshot, self.per_row_fn, depths
        )

    def exhausted(self) -> bool:
        return self.depths is not None and max(self.depths, default=0) == 0


class ExactRollbackBoundary:
    """Public, immutable handle for one exact recurrent rollback record.

    The cache owns the rollback stack and exports only the operations a caller
    can safely use.  Consumers do not depend on ``ArraysCache._rollbacks`` or
    on the private ``_RollbackRecord`` representation.
    """

    def __init__(
        self,
        cache_type: type,
        state_size: int,
        num_tokens: int,
        replay: Callable[[int], List[Any]],
        snapshot: List[Any],
        live_state: List[Any],
    ):
        self._cache_type = cache_type
        self._state_size = int(state_size)
        self._num_tokens = int(num_tokens)
        self._replay = replay
        self._snapshot = tuple(snapshot)
        self._live_state = tuple(live_state)

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    @property
    def snapshot(self) -> tuple:
        """The frozen pre-forward state, for inspection only."""
        return self._snapshot

    @property
    def nbytes(self) -> int:
        return sum(
            (
                int(getattr(value, "nbytes", 0))
                for value in (*self._snapshot, *self._live_state)
            )
        )

    def materialize(self, accepted_tokens: int) -> "ArraysCache":
        """Build a private cache at one exact point inside the record."""
        if isinstance(accepted_tokens, bool) or not isinstance(accepted_tokens, int):
            raise TypeError("accepted_tokens must be an integer")
        if not 0 <= accepted_tokens <= self._num_tokens:
            raise ValueError(
                f"accepted_tokens {accepted_tokens} is outside 0..{self._num_tokens}"
            )
        if accepted_tokens == 0:
            state = list(self._snapshot)
        elif accepted_tokens == self._num_tokens:
            state = list(self._live_state)
        else:
            state = list(self._replay(accepted_tokens))
        if len(state) != self._state_size:
            raise RuntimeError(
                f"rollback materialized {len(state)} entries, expected {self._state_size}"
            )
        state = [None if value is None else mx.array(value) for value in state]
        values = [value for value in state if value is not None]
        if values:
            mx.eval(*values)
        cache = self._cache_type(self._state_size)
        cache.cache = state
        return cache


def _row_vector(n, batch_size: int, who: str) -> List[int]:
    """Validate a host-side per-row count vector."""
    if isinstance(n, mx.array):
        n = n.tolist()
    if isinstance(n, int):
        raise TypeError(
            f"{who} needs one count per row, not a scalar; use trim() for a uniform rewind"
        )
    drops = []
    for value in n:
        count = int(value)
        if count != value:
            raise ValueError(f"{who} got a non-integral count: {list(n)}")
        drops.append(count)
    if len(drops) != batch_size:
        raise ValueError(f"{who} got {len(drops)} counts for {batch_size} rows")
    if any((v < 0 for v in drops)):
        raise ValueError(f"{who} got a negative count: {drops}")
    return drops


def _ragged_slab_plan(cache, n, who: str, validate: bool):
    """Validate a per-row rewind of a shared-cursor (slab) batch cache.

    Returns ``(drops, uniform, residual)``. The uniform part is a plain
    cursor move; only the residual needs a per-row roll, so a batch that
    rejects the same amount on every row costs nothing extra.
    """
    drops = _row_vector(n, cache.offset.shape[0], who)
    if max(drops, default=0) == 0:
        return (drops, 0, None)
    if cache._right_padding is not None:
        raise RaggedTrimUnsupported(
            f"{who} needs finalize() first: pending right padding leaves the per-row geometry ambiguous"
        )
    if max(drops) > cache._idx:
        raise ValueError(
            f"{who} cannot drop {max(drops)} tokens from a cache of width {cache._idx}"
        )
    if validate:
        shortest = (cache.offset - mx.array(drops)).min().item()
        if shortest < 0:
            raise ValueError(f"{who} would drop more tokens than a row holds: {drops}")
    uniform = min(drops)
    return (drops, uniform, [d - uniform for d in drops])


def _roll_rows_right(x, shifts, axis: int, lo: int, hi: int, reclaim: int = 0):
    """Right-roll each row's ``[lo, hi)`` window along ``axis`` by its shift.

    The rolled-out tail cells wrap to the bottom of the window, which is
    inside every row's (now larger) left-padding region, so the valid prefix
    of each row stays contiguous and still ends at the shared cursor ``hi``.

    ``reclaim`` drops that many leading columns of the rolled window in the
    same gather, landing the result at ``[lo, hi - reclaim)``. The caller
    passes the padding every row shares after the roll, so no row loses a
    valid cell and the batch costs no second pass over K/V to shrink.
    """
    if hi <= lo:
        return x
    window = (slice(None),) * axis + (slice(lo, hi),)
    shaped = shifts.reshape((-1,) + (1,) * (axis - 1))
    if not reclaim:
        x[window] = dynamic_roll(x[window], shaped, axis)
        return x
    width = hi - lo
    expand_shifts = (...,) + (None,) * (x.ndim - axis)
    expand_indices = expand_shifts[:-1]
    positions = mx.arange(reclaim, width)[expand_indices]
    idx = (positions - shaped[expand_shifts]) % width
    kept = (slice(None),) * axis + (slice(lo, hi - reclaim),)
    x[kept] = mx.take_along_axis(x[window], idx, axis=axis)
    return x


def trim_ragged_prompt_cache(
    cache: List[Any], n, *, validate: bool = True
) -> List[int]:
    """Rewind row ``i`` of a merged cache by ``n[i]`` tokens.

    Args:
        cache (List[Any]): The model's merged (batched) cache.
        n: One non-negative count per row, host-side (list, tuple or an
            ``mx.array`` that is turned into one). Zeros are allowed and mean
            "this row accepted everything".
        validate (bool): Check that no row is asked to drop more than it
            holds. Costs one scalar device sync; pass ``False`` in a loop that
            already tracks the logical lengths on the host.

    Returns:
        (List[int]): The per-row counts that were applied.

    Raises:
        RaggedTrimUnsupported: if any entry cannot do a per-row rewind. It
            never degrades to a uniform trim.

    Note:
        The vector describes physical rewind widths only. Do not derive any
        random draw from it or from the padded verify width: keyed draws are
        not prefix stable across shape, so drawing at ``k_max`` and slicing
        to a lane's ``k_i`` makes that lane's values depend on its
        co-scheduled lanes. Draw per lane at its own depth.
    """
    if len(cache) == 0:
        return []
    unsupported = sorted(
        {
            type(c).__name__
            for c in cache
            if not getattr(c, "supports_ragged_trim", lambda: False)()
        }
    )
    if unsupported:
        raise RaggedTrimUnsupported(
            "ragged trim is not supported by: " + ", ".join(unsupported)
        )
    not_trimmable = sorted({type(c).__name__ for c in cache if not c.is_trimmable()})
    if not_trimmable:
        raise RaggedTrimUnsupported(
            "cache entries are not trimmable right now: " + ", ".join(not_trimmable)
        )
    for c in cache:
        c.preflight_ragged_trim(n, validate=validate)
    applied = [c.trim_ragged(n, validate=False) for c in cache]
    first = applied[0]
    for other in applied[1:]:
        if other != first:
            raise RuntimeError(
                f"ragged trim diverged across cache entries: {first} != {other}"
            )
    return first


def record_state_checkpoints(cache: List[Any], positions: List[int], force=False):
    """Record a recurrent-state checkpoint on every cache that supports one.

    Called by the prefill loops at chunk boundaries. ``positions`` gives the
    absolute number of tokens processed per batch lane at this boundary
    (length-1 list for unbatched caches). No-op for pure KV caches and for
    duck-typed caches that predate this hook.
    """
    for c in cache:
        record = getattr(c, "state_checkpoint", None)
        if record is not None:
            record(positions, force=force)


def _state_checkpoint_max() -> int:
    """Max recorded state checkpoints per ArraysCache (0 disables)."""
    try:
        return int(os.environ.get("MLX_LM_STATE_CHECKPOINT_MAX", "4"))
    except ValueError:
        return 4


def _state_checkpoint_stride() -> int:
    """Minimum token gap between recorded (non-forced) state checkpoints."""
    try:
        return max(1, int(os.environ.get("MLX_LM_STATE_CHECKPOINT_STRIDE", "2048")))
    except ValueError:
        return 2048


def create_attention_mask(
    N: int, offset: int, return_array: bool, window_size: Optional[int]
):
    if window_size is not None:
        return create_causal_mask(N, offset, window_size=window_size)
    elif N == 1:
        return None
    elif return_array:
        return create_causal_mask(N, offset, window_size=window_size)
    else:
        return "causal"


class _BaseCache:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_cache_class(cls)

    @property
    def state(self):
        return []

    @state.setter
    def state(self, v):
        if v is not None and v:
            raise ValueError("This cache has no state but a state was set.")

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, v):
        if v is not None and v:
            raise ValueError("This cache has no meta_state but a meta_state was set.")

    def is_trimmable(self):
        return False

    def supports_ragged_trim(self):
        """True when ``trim_ragged`` can rewind rows by different amounts."""
        return False

    def trim_ragged(self, n, *, validate: bool = True):
        """Rewind row ``i`` by ``n[i]`` tokens. See ``trim_ragged_prompt_cache``.

        The default fails loud. A uniform fallback would rewind rows that
        accepted their drafts.
        """
        raise RaggedTrimUnsupported(
            f"{type(self).__name__} cannot rewind its rows by different amounts; only a uniform trim() is defined for it"
        )

    def preflight_ragged_trim(self, n, *, validate: bool = True):
        """Run ``trim_ragged``'s entry-local checks without mutating anything.

        The group entry points call this on every entry first, so one entry
        cannot rewind while a later one rejects the same vector.
        """
        return self.trim_ragged(n, validate=validate)

    def start_speculation(self, rollback_window: Optional[int] = None):
        """Called before draft/verify (or multi-token proposal) steps begin.

        Caches that cannot trim directly but support an exact rollback (e.g.
        recurrent-state caches, see ``ArraysCache``, or sliding-window caches,
        see ``RotatingKVCache``) use this to start recording the information
        needed for exact future trims. ``rollback_window`` bounds how many
        tokens of rollback history are kept. No-op by default.
        """
        pass

    def stop_speculation(self):
        """Stop recording trim state and release any temporary rollback data."""
        pass

    def state_checkpoint(self, positions: List[int], force: bool = False):
        """Record a restorable snapshot of this cache's state at the given
        absolute per-lane positions. No-op by default: KV caches can trim to
        any position already, so only caches with irreversible state (e.g.
        ``ArraysCache``) record checkpoints.
        """
        pass

    def snap_trim_position(self, position: int) -> Optional[int]:
        """Largest absolute position <= ``position`` this cache can be
        restored to. Exactly-trimmable caches can restore any position;
        caches with irreversible state snap to a recorded checkpoint.
        Returns ``None`` if no position is restorable.
        """
        return position if self.is_trimmable() else None

    def trim_to_position(self, position: int, num_tokens: int) -> int:
        """Restore the cache to absolute position ``position`` by trimming
        ``num_tokens`` tokens from its end. Exactly-trimmable caches just
        trim; checkpointed caches restore the snapshot recorded at
        ``position`` (which ``snap_trim_position`` guaranteed exists).
        """
        return self.trim(num_tokens)

    def size(self):
        """
        Return the size (i.e. sequence length) of the cache.

        Not every cache is required to implement this, in which case the size
        will always be 0 (though the cache may not be empty).
        """
        return 0

    @property
    def nbytes(self):
        """Return the size of this cache in bytes"""
        raise NotImplementedError("Cache sub-class must implement nbytes")

    def empty(self):
        """
        Return if the cache is empty or not.
        """
        raise NotImplementedError("Cache sub-class must implement this.")

    @classmethod
    def from_state(cls, state, meta_state):
        obj = cls.__new__(cls)
        obj.state = state
        obj.meta_state = meta_state
        return obj


def _empty_quantized(B, n_kv_heads, n_steps, head_dim, group_size, bits, dtype):
    """Allocate a zero-filled quantized (packed, scales, biases) triple, matching
    the layout `mx.quantize` produces — used to grow quantized cache buffers in
    fixed-size chunks (mirrors QuantizedKVCache.update_and_fetch's init_quant).

    The packed width is ``head_dim * bits / 32``, not ``head_dim // (32 // bits)``:
    3-, 5- and 6-bit values straddle uint32 words, so the element-per-word
    division rounds the wrong way and the buffer no longer matches the
    triples `mx.quantize` writes into it."""
    shape = (B, n_kv_heads, n_steps)
    return (
        mx.zeros((*shape, head_dim * bits // (8 * mx.uint32.size)), dtype=mx.uint32),
        mx.zeros((*shape, head_dim // group_size), dtype=dtype),
        mx.zeros((*shape, head_dim // group_size), dtype=dtype),
    )


class QuantizedKVCache(_BaseCache):
    step = 256
    _RECOVERY_APPEND_ONLY_FIELDS = (("keys", -2, "offset"), ("values", -2, "offset"))
    _supported_bits = {2, 3, 4, 5, 6, 8}
    _supported_group_sizes = {32, 64, 128}

    @classmethod
    def _validate_config(cls, group_size, key_bits, value_bits):
        if key_bits not in cls._supported_bits:
            raise ValueError(f"Unsupported key bits: {key_bits}")
        if value_bits not in cls._supported_bits:
            raise ValueError(f"Unsupported value bits: {value_bits}")
        if group_size not in cls._supported_group_sizes:
            raise ValueError(f"Unsupported group size: {group_size}")

    def __init__(
        self,
        group_size: int = 64,
        bits: int = 8,
        *,
        key_bits: Optional[int] = None,
        value_bits: Optional[int] = None,
        rotate: bool = False,
        normalize: bool = False,
    ):
        self.keys = None
        self.values = None
        self.offset = 0
        self.group_size = group_size
        self.key_bits = bits if key_bits is None else key_bits
        self.value_bits = bits if value_bits is None else value_bits
        self._validate_config(group_size, self.key_bits, self.value_bits)
        self.bits = self.key_bits if self.key_bits == self.value_bits else None
        self.rotate = rotate
        self.normalize = normalize
        self.key_scale = None
        self.value_scale = None

    @staticmethod
    def _channel_scale(x, eps: float = 1e-06):
        """Per-channel scale = RMS over the sequence axis, in ``x``'s dtype.

        Shape ``(B, n_kv_heads, 1, D)`` — one positive scalar per channel,
        constant across sequence positions. RMS (not std) is used so the scale
        is well defined even for a single-token first block and so it captures
        both the mean and variance magnitude of an outlier channel."""
        scale = mx.sqrt(mx.mean(x.astype(mx.float32) ** 2, axis=-2, keepdims=True))
        return (scale + eps).astype(x.dtype)

    def update_and_fetch(self, keys, values):
        (B, n_kv_heads, num_steps, k_head_dim) = keys.shape
        v_head_dim = values.shape[-1]
        prev = self.offset
        if self.keys is None or prev + num_steps > self.keys[0].shape[-2]:
            new_steps = (self.step + num_steps - 1) // self.step * self.step
            shape = (B, n_kv_heads, new_steps)

            def init_quant(dim, bits, dtype):
                packed_dim = dim * bits // (8 * mx.uint32.size)
                return (
                    mx.zeros((*shape, packed_dim), dtype=mx.uint32),
                    mx.zeros((*shape, dim // self.group_size), dtype=dtype),
                    mx.zeros((*shape, dim // self.group_size), dtype=dtype),
                )

            def expand_quant(x):
                new_x = mx.zeros((*shape, x.shape[-1]), dtype=x.dtype)
                return mx.concatenate([x, new_x], axis=-2)

            if self.keys is not None:
                if prev % self.step != 0:
                    (self.keys, self.values) = tree_map(
                        lambda x: x[..., :prev, :], (self.keys, self.values)
                    )
                (self.keys, self.values) = tree_map(
                    expand_quant, (self.keys, self.values)
                )
            else:
                self.keys = init_quant(k_head_dim, self.key_bits, keys.dtype)
                self.values = init_quant(v_head_dim, self.value_bits, values.dtype)
        self.offset += num_steps
        if self.rotate and hadamard_size_ok(keys.shape[-1]):
            keys = rotate_last(keys)
        if self.normalize:
            if self.key_scale is None:
                self.key_scale = self._channel_scale(keys)
                self.value_scale = self._channel_scale(values)
            keys = keys / self.key_scale
            values = values / self.value_scale
        keys = mx.quantize(keys, group_size=self.group_size, bits=self.key_bits)
        values = mx.quantize(values, group_size=self.group_size, bits=self.value_bits)
        for i in range(len(self.keys)):
            self.keys[i][..., prev : self.offset, :] = keys[i]
            self.values[i][..., prev : self.offset, :] = values[i]
        return tree_map(lambda x: x[..., : self.offset, :], (self.keys, self.values))

    def keys_and_values(self):
        if self.offset == self.keys[0].shape[2]:
            (keys, values) = (self.keys, self.values)
        else:
            (keys, values) = tree_map(
                lambda x: x[..., : self.offset, :], (self.keys, self.values)
            )
        if self.normalize:
            return (keys, values, self.key_scale, self.value_scale)
        return (keys, values)

    @property
    def state(self):
        # An empty cache (a one-token prompt prefills nothing) has no state
        # yet; report it the way KVCache does rather than raise.
        if self.keys is None:
            return (self.keys, self.values)
        return self.keys_and_values()

    @state.setter
    def state(self, v):
        if len(v) == 4:
            (self.keys, self.values, self.key_scale, self.value_scale) = v
        else:
            (self.keys, self.values) = v
            self.key_scale = self.value_scale = None

    def _validate_state_geometry(self):
        for name, state, bits in (
            ("keys", self.keys, self.key_bits),
            ("values", self.values, self.value_bits),
        ):
            if state is None:
                continue
            if len(state) != 3:
                raise ValueError(f"Invalid quantized {name} state: expected 3 arrays")
            (packed, scales, biases) = state
            if packed.dtype != mx.uint32 or scales.dtype != biases.dtype:
                raise ValueError(f"Invalid quantized {name} state dtypes")
            if scales.shape != biases.shape or packed.shape[:-1] != scales.shape[:-1]:
                raise ValueError(f"Invalid quantized {name} state geometry")
            if packed.shape[-1] * 32 != scales.shape[-1] * self.group_size * bits:
                raise ValueError(
                    f"Quantized {name} state does not match {bits}-bit metadata"
                )
            if packed.shape[-2] != self.offset:
                raise ValueError(
                    f"Quantized {name} state length does not match cache offset"
                )
        if self.keys is not None and self.values is not None:
            if self.keys[0].shape[:-1] != self.values[0].shape[:-1]:
                raise ValueError("Quantized key/value batch, head, or length mismatch")

    @property
    def meta_state(self):
        if (
            self.key_bits == self.value_bits
            and (not self.rotate)
            and (not self.normalize)
        ):
            return tuple(map(str, (self.offset, self.group_size, self.key_bits)))
        fields = [2, self.offset, self.group_size, self.key_bits, self.value_bits]
        if self.rotate or self.normalize:
            fields.append(int(self.rotate))
        if self.normalize:
            fields.append(int(self.normalize))
        return tuple(map(str, fields))

    @meta_state.setter
    def meta_state(self, v):
        if len(v) == 3:
            (self.offset, self.group_size, bits) = map(int, v)
            self.key_bits = self.value_bits = self.bits = bits
            self.rotate = False
            self.normalize = False
            self._validate_config(self.group_size, self.key_bits, self.value_bits)
            self._validate_state_geometry()
            return
        if len(v) not in (5, 6, 7):
            raise ValueError(
                "Invalid QuantizedKVCache metadata: expected 3 legacy fields or 5-7 versioned fields."
            )
        vals = list(map(int, v))
        (version, self.offset, self.group_size, self.key_bits, self.value_bits) = vals[
            :5
        ]
        if version != 2:
            raise ValueError(
                f"Unsupported QuantizedKVCache metadata version: {version}"
            )
        self.rotate = bool(vals[5]) if len(vals) > 5 else False
        self.normalize = bool(vals[6]) if len(vals) > 6 else False
        self._validate_config(self.group_size, self.key_bits, self.value_bits)
        self.bits = self.key_bits if self.key_bits == self.value_bits else None
        self._validate_state_geometry()

    def is_trimmable(self):
        return True

    def size(self):
        return self.offset

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    @classmethod
    def merge(cls, caches):
        return BatchQuantizedKVCache.merge(caches)

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return tree_reduce(lambda a, x: a + x.nbytes, (self.keys, self.values), 0)


class KVCache(_BaseCache):
    step = 256
    # Buffers written only at or past ``offset``: a recovery checkpoint keeps
    # the fill level instead of an alias that would force a copy per append.
    _RECOVERY_APPEND_ONLY_FIELDS = (("keys", -2, "offset"), ("values", -2, "offset"))

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        prev = self.offset
        if self.keys is None or prev + keys.shape[2] > self.keys.shape[2]:
            (B, n_kv_heads, _, k_head_dim) = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                (self.keys, self.values) = (new_k, new_v)
        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys_and_values()

    def keys_and_values(self):
        if self.offset < self.keys.shape[2]:
            return (
                self.keys[..., : self.offset, :],
                self.values[..., : self.offset, :],
            )
        return (self.keys, self.values)

    def size(self):
        return self.offset

    @property
    def state(self):
        return (self.keys, self.values)

    @state.setter
    def state(self, v):
        same_storage = getattr(self, "keys", None) is v[0]
        previous_offset = getattr(self, "offset", None) if same_storage else None
        (self.keys, self.values) = v
        if previous_offset is not None:
            self.offset = previous_offset
        else:
            # An empty cache's state is ``(None, None)``.
            self.offset = 0 if self.keys is None else self.keys.shape[2]

    @property
    def meta_state(self):
        return (str(self.offset),)

    @meta_state.setter
    def meta_state(self, v):
        if v:
            self.offset = int(v[0])

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def to_quantized(
        self,
        group_size: int = 64,
        bits: int = 4,
        *,
        key_bits: Optional[int] = None,
        value_bits: Optional[int] = None,
        rotate: bool = False,
        normalize: bool = False,
    ) -> QuantizedKVCache:
        quant_cache = QuantizedKVCache(
            group_size=group_size,
            bits=bits,
            key_bits=key_bits,
            value_bits=value_bits,
            rotate=rotate,
            normalize=normalize,
        )
        quant_cache.offset = self.offset
        if self.keys is not None:
            keys = self.keys
            values = self.values
            if rotate and hadamard_size_ok(keys.shape[-1]):
                keys = rotate_last(keys)
            if normalize:
                quant_cache.key_scale = quant_cache._channel_scale(keys)
                quant_cache.value_scale = quant_cache._channel_scale(values)
                keys = keys / quant_cache.key_scale
                values = values / quant_cache.value_scale
            quant_cache.keys = mx.quantize(
                keys, group_size=group_size, bits=quant_cache.key_bits
            )
            quant_cache.values = mx.quantize(
                values, group_size=group_size, bits=quant_cache.value_bits
            )
        return quant_cache

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    @classmethod
    def merge(_, caches):
        return BatchKVCache.merge(caches)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class SinkWindowKVCache(_BaseCache):
    """Bounded draft-only KV with attention sinks and absolute positions.

    The target/verifier cache remains full-context.  A bounded tail beyond the
    active window is retained only while speculation is active so rejection can
    restore the exact pre-cycle state.
    """

    _ROLLBACK_WINDOW = 64

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.speculating = False
        instance._speculation_positions = []
        return instance

    def __init__(self, window_size: int, sink_size: int = 4, rollback_window: int = 64):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if sink_size < 0:
            raise ValueError("sink_size must be >= 0")
        self.window_size = int(window_size)
        self.sink_size = int(sink_size)
        self.rollback_window = max(0, int(rollback_window))
        self.keys = None
        self.values = None
        self.offset = 0
        self._positions: List[int] = []
        self._active_positions: List[int] = []

    @property
    def state(self):
        return [] if self.keys is None else (self.keys, self.values)

    @state.setter
    def state(self, value):
        if not value:
            self.keys = self.values = None
            stored = 0
        else:
            (self.keys, self.values) = value
            stored = self.keys.shape[2]
        self.window_size = max(1, stored)
        self.sink_size = 0
        self.rollback_window = self._ROLLBACK_WINDOW
        self.offset = stored
        self._positions = list(range(stored))
        self._active_positions = self._active_keep_positions(self.offset)

    @property
    def meta_state(self):
        return tuple(
            map(
                str,
                (
                    1,
                    self.window_size,
                    self.sink_size,
                    self.rollback_window,
                    self.offset,
                    *self._positions,
                ),
            )
        )

    @meta_state.setter
    def meta_state(self, value):
        values = list(map(int, value))
        if len(values) < 5 or values[0] != 1:
            raise ValueError("Invalid SinkWindowKVCache metadata")
        (
            _,
            self.window_size,
            self.sink_size,
            self.rollback_window,
            self.offset,
            *positions,
        ) = values
        if self.window_size < 1 or self.sink_size < 0 or self.rollback_window < 0:
            raise ValueError("Invalid SinkWindowKVCache configuration in metadata")
        if self.keys is None:
            if positions:
                raise ValueError("SinkWindowKVCache positions require stored state")
        elif (
            self.values is None
            or self.keys.shape[2] != len(positions)
            or self.values.shape[2] != len(positions)
        ):
            raise ValueError(
                "SinkWindowKVCache state length does not match retained positions"
            )
        if positions != sorted(set(positions)) or any(
            (position < 0 or position >= self.offset for position in positions)
        ):
            raise ValueError("Invalid SinkWindowKVCache retained positions")
        self._positions = positions
        self._active_positions = self._active_keep_positions(self.offset)
        if not all(
            (position in self._positions for position in self._active_positions)
        ):
            raise ValueError("SinkWindowKVCache state is missing an active position")
        self.speculating = False
        self._speculation_positions = []

    def _active_keep_positions(self, end: int) -> List[int]:
        sinks = list(range(min(self.sink_size, end)))
        window_start = max(0, end - self.window_size)
        return sinks + list(range(max(self.sink_size, window_start), end))

    def _stored_keep_positions(self, end: int) -> List[int]:
        sinks = list(range(min(self.sink_size, end)))
        tail_start = max(0, end - self.window_size - self.rollback_window)
        return sinks + list(range(max(self.sink_size, tail_start), end))

    def update_and_fetch(self, keys, values):
        length = keys.shape[2]
        (start, end) = (self.offset, self.offset + length)
        if self.keys is None:
            (all_keys, all_values) = (keys, values)
            all_positions = list(range(start, end))
        else:
            all_keys = mx.concatenate([self.keys, keys], axis=2)
            all_values = mx.concatenate([self.values, values], axis=2)
            all_positions = self._positions + list(range(start, end))
        position_to_index = {
            position: index for (index, position) in enumerate(all_positions)
        }
        attention_positions = self._active_positions + list(range(start, end))
        attention_indices = mx.array(
            [position_to_index[position] for position in attention_positions],
            dtype=mx.int32,
        )
        attention_keys = mx.take(all_keys, attention_indices, axis=2)
        attention_values = mx.take(all_values, attention_indices, axis=2)
        keep = [
            position
            for position in self._stored_keep_positions(end)
            if position in position_to_index
        ]
        if self.speculating:
            keep = sorted(set(keep).union(self._speculation_positions))
        indices = mx.array(
            [position_to_index[position] for position in keep], dtype=mx.int32
        )
        self.keys = mx.take(all_keys, indices, axis=2)
        self.values = mx.take(all_values, indices, axis=2)
        self._positions = keep
        self._active_positions = self._active_keep_positions(end)
        self.offset = end
        return (attention_keys, attention_values)

    def make_mask(self, length: int, window_size=None, return_array=False):
        if length == 1:
            return None
        effective = (
            self.window_size
            if window_size is None
            else min(self.window_size, int(window_size))
        )
        if effective < 1:
            raise ValueError("window_size must be >= 1")
        key_positions = self._active_positions + list(
            range(self.offset, self.offset + length)
        )
        query_positions = list(range(self.offset, self.offset + length))
        keys = mx.array(key_positions, dtype=mx.int32)[None]
        queries = mx.array(query_positions, dtype=mx.int32)[:, None]
        causal = queries >= keys
        return causal & ((keys < self.sink_size) | (keys >= queries - effective))

    def size(self):
        return self.offset

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        return 0 if self.keys is None else self.keys.nbytes + self.values.nbytes

    def is_trimmable(self):
        return True

    def start_speculation(self, rollback_window=None):
        if self.speculating:
            raise RuntimeError("SinkWindowKVCache speculation is already active")
        self.speculating = True
        self._speculation_positions = list(self._positions)

    def stop_speculation(self):
        if self.keys is not None:
            position_to_index = {
                position: index for (index, position) in enumerate(self._positions)
            }
            keep = self._stored_keep_positions(self.offset)
            if not all((position in position_to_index for position in keep)):
                raise RuntimeError(
                    "SinkWindowKVCache cannot compact after speculative rollback"
                )
            indices = mx.array(
                [position_to_index[position] for position in keep], dtype=mx.int32
            )
            self.keys = mx.take(self.keys, indices, axis=2)
            self.values = mx.take(self.values, indices, axis=2)
            self._positions = keep
            self._active_positions = self._active_keep_positions(self.offset)
        self.speculating = False
        self._speculation_positions = []

    def trim(self, n):
        if n < 0:
            raise ValueError("trim count must be non-negative")
        new_offset = max(0, self.offset - min(self.offset, n))
        keep_indices = [
            index
            for (index, position) in enumerate(self._positions)
            if position < new_offset
        ]
        if keep_indices:
            kept_positions = [self._positions[index] for index in keep_indices]
            active = self._active_keep_positions(new_offset)
            if not all((position in kept_positions for position in active)):
                raise RuntimeError(
                    "SinkWindowKVCache rollback exceeds its retained tail; increase rollback_window."
                )
            indices = mx.array(keep_indices, dtype=mx.int32)
            self.keys = mx.take(self.keys, indices, axis=2)
            self.values = mx.take(self.values, indices, axis=2)
            self._positions = kept_positions
            self._active_positions = active
        else:
            self.keys = self.values = None
            self._positions = []
            self._active_positions = []
        trimmed = self.offset - new_offset
        self.offset = new_offset
        return trimmed


class RotatingKVCache(_BaseCache):
    step = 256
    _ROLLBACK_WINDOW = 64

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.speculating = False
        instance._rollbacks = deque()
        instance._rollback_window = cls._ROLLBACK_WINDOW
        instance._checkpoints = []
        return instance

    def __init__(self, max_size, keep=0):
        self.keep = keep
        self.keys = None
        self.values = None
        self.offset = 0
        self.max_size = max_size
        self._idx = 0

    def state_checkpoint(self, positions: List[int], force: bool = False):
        max_checkpoints = _state_checkpoint_max()
        if max_checkpoints <= 0 or self.keys is None or len(positions) != 1:
            return
        position = positions[0]
        last = self._checkpoints[-1][0] if self._checkpoints else 0
        if position <= last:
            return
        if not force and position - last < _state_checkpoint_stride():
            return
        keys = self._temporal_order(self.keys)
        values = self._temporal_order(self.values)
        self._checkpoints.append((position, mx.array(keys), mx.array(values)))
        _thin_checkpoints(self._checkpoints, max_checkpoints)

    def snap_trim_position(self, position: int) -> Optional[int]:
        if self.offset < self.max_size:
            return position
        best = 0
        for p, _, _ in self._checkpoints:
            if p <= position:
                best = max(best, p)
        return best

    def trim_to_position(self, position: int, num_tokens: int) -> int:
        if self.offset < self.max_size:
            return self.trim(num_tokens)
        if position > 0:
            found = None
            for p, keys, values in reversed(self._checkpoints):
                if p == position:
                    found = (keys, values)
                    break
            if found is None:
                raise RuntimeError(
                    f"RotatingKVCache has no window checkpoint at position {position}"
                )
            self.keys = mx.array(found[0])
            self.values = mx.array(found[1])
            self.offset = position
            self._idx = self.keys.shape[2]
        else:
            self.keys = None
            self.values = None
            self.offset = 0
            self._idx = 0
        while self._checkpoints and self._checkpoints[-1][0] > position:
            self._checkpoints.pop()
        self._rollbacks.clear()
        return num_tokens

    def start_speculation(self, rollback_window: Optional[int] = None):
        self.speculating = True
        self._rollback_window = max(
            1,
            int(self._ROLLBACK_WINDOW if rollback_window is None else rollback_window),
        )
        self._rollbacks.clear()

    def stop_speculation(self):
        self.speculating = False
        self._rollbacks.clear()

    def record_rollback(self, num_tokens, snapshot, keys, values):
        """Recorded by ``update_and_fetch`` during a forward while
        ``speculating``. A sliding-window KV cache is not directly trimmable
        once it has wrapped (``offset >= max_size``): the tokens a verify step
        pushes out of the window are gone. So we stash the pre-forward window
        (``snapshot``) plus this forward's new K/V; ``trim`` rebuilds the exact
        window for any accepted prefix by restoring the snapshot and
        re-appending the first ``m`` tokens. Arrays are COPIED because the
        single-token path (``_update_in_place``) mutates buffers in place, so
        holding references would let a later forward corrupt the snapshot."""

        def copy_array(x):
            return None if x is None else mx.array(x)

        copied_snapshot = [
            copy_array(snapshot[0]),
            copy_array(snapshot[1]),
            snapshot[2],
            snapshot[3],
        ]
        self._rollbacks.append(
            (num_tokens, copied_snapshot, mx.array(keys), mx.array(values))
        )
        total = sum((r[0] for r in self._rollbacks))
        while (
            len(self._rollbacks) > 1
            and total - self._rollbacks[0][0] >= self._rollback_window
        ):
            total -= self._rollbacks.popleft()[0]

    def _trim(self, trim_size, v, append=None):
        to_cat = []
        if trim_size > 0:
            to_cat = [v[..., : self.keep, :], v[..., trim_size + self.keep :, :]]
        else:
            to_cat = [v]
        if append is not None:
            to_cat.append(append)
        return mx.concatenate(to_cat, axis=2)

    def _temporal_order(self, v):
        """
        Rearrange the cache into temporal order, slicing off the end if unused.
        """
        if self._idx == v.shape[2]:
            return v
        elif self._idx < self.offset:
            return mx.concatenate(
                [
                    v[..., : self.keep, :],
                    v[..., self._idx :, :],
                    v[..., self.keep : self._idx, :],
                ],
                axis=2,
            )
        else:
            return v[..., : self._idx, :]

    def _update_concat(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = self._temporal_order(self.keys)
            self.values = self._temporal_order(self.values)
            self._idx = self.keys.shape[2]
            trim_size = self._idx - self.max_size + 1
            self.keys = self._trim(trim_size, self.keys, keys)
            self.values = self._trim(trim_size, self.values, values)
        self.offset += keys.shape[2]
        self._idx = self.keys.shape[2]
        return (self.keys, self.values)

    def _update_in_place(self, keys, values):
        (B, n_kv_heads, S, k_head_dim) = keys.shape
        prev = self.offset
        if self.keys is None or (
            prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            v_head_dim = values.shape[3]
            new_size = min(self.step, self.max_size - prev)
            k_shape = (B, n_kv_heads, new_size, k_head_dim)
            v_shape = (B, n_kv_heads, new_size, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                (self.keys, self.values) = (new_k, new_v)
            self._idx = prev
        trim_size = self.keys.shape[2] - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size
        if self._idx == self.max_size:
            self._idx = self.keep
        self.keys[..., self._idx : self._idx + S, :] = keys
        self.values[..., self._idx : self._idx + S, :] = values
        self.offset += S
        self._idx += S
        if self.offset < self.max_size:
            return (
                self.keys[..., : self.offset, :],
                self.values[..., : self.offset, :],
            )
        return (self.keys, self.values)

    def update_and_fetch(self, keys, values):
        if self.speculating:
            self.record_rollback(
                keys.shape[2],
                [self.keys, self.values, self._idx, self.offset],
                keys,
                values,
            )
        if keys.shape[2] == 1:
            return self._update_in_place(keys, values)
        return self._update_concat(keys, values)

    def keys_and_values(self):
        if self.offset < self.keys.shape[2]:
            return (
                self.keys[..., : self.offset, :],
                self.values[..., : self.offset, :],
            )
        return (self.keys, self.values)

    def size(self):
        return min(self.offset, self.max_size)

    @property
    def state(self):
        # An empty cache (a one-token prompt prefills nothing) has no state
        # yet; report it the way KVCache does rather than raise.
        if self.keys is None:
            return (self.keys, self.values)
        if self.offset < self.keys.shape[2]:
            live = [
                self.keys[..., : self.offset, :],
                self.values[..., : self.offset, :],
            ]
        else:
            live = [self.keys, self.values]
        if self._checkpoints:
            snaps = []
            for _, k, v in self._checkpoints:
                snaps.extend([k, v])
            return [live, snaps]
        return (live[0], live[1])

    @state.setter
    def state(self, v):
        if len(v) == 2 and isinstance(v[0], list) and isinstance(v[1], list):
            (self.keys, self.values) = v[0]
            self._pending_checkpoint_snapshots = list(v[1])
        else:
            (self.keys, self.values) = v

    @property
    def meta_state(self):
        base = tuple(map(str, (self.keep, self.max_size, self.offset, self._idx)))
        if self._checkpoints:
            return base + tuple(
                ["ckptv1"] + [str(p) for (p, _, _) in self._checkpoints]
            )
        return base

    @meta_state.setter
    def meta_state(self, v):
        pending = getattr(self, "_pending_checkpoint_snapshots", None)
        self._pending_checkpoint_snapshots = None
        v = list(v)
        if "ckptv1" in v:
            i = v.index("ckptv1")
            positions = [int(p) for p in v[i + 1 :]]
            v = v[:i]
            if pending is None or len(pending) != 2 * len(positions):
                raise ValueError("RotatingKVCache checkpoint state/metadata mismatch")
            self._checkpoints = [
                (p, pending[2 * j], pending[2 * j + 1])
                for (j, p) in enumerate(positions)
            ]
        (self.keep, self.max_size, self.offset, self._idx) = map(int, v)

    def is_trimmable(self):
        return self.speculating or self.offset < self.max_size

    def trim(self, n):
        if not self.speculating:
            if self.offset >= self.max_size:
                raise RuntimeError(
                    f"Cannot trim {n} tokens from RotatingKVCache: the cache has wrapped (offset {self.offset} >= max_size {self.max_size}) and no speculative rollback is recorded."
                )
            n = min(self.offset, n)
            self.offset -= n
            self._idx -= n
            return n
        recorded = sum((r[0] for r in self._rollbacks))
        if recorded < n:
            raise RuntimeError(
                f"Cannot trim {n} tokens from RotatingKVCache: only {recorded} tokens of exact rollback are recorded (speculative window)."
            )
        trimmed = 0
        while trimmed < n:
            (num_tokens, snap, keys, values) = self._rollbacks.pop()
            take = min(n - trimmed, num_tokens)
            m = num_tokens - take
            # Restore from copies: the single-token replay below writes the
            # live buffer in place, and the record is kept for a later trim.
            (self.keys, self.values, self._idx, self.offset) = (
                None if snap[0] is None else mx.array(snap[0]),
                None if snap[1] is None else mx.array(snap[1]),
                snap[2],
                snap[3],
            )
            if m > 0:
                if m == 1:
                    self._update_in_place(keys[..., :1, :], values[..., :1, :])
                else:
                    self._update_concat(keys[..., :m, :], values[..., :m, :])
                self._rollbacks.append((m, snap, keys, values))
            trimmed += take
        return n

    def to_quantized(
        self, group_size: int = 64, bits: int = 4
    ) -> "RotatingQuantizedKVCache":
        if self.keep > 0:
            raise NotImplementedError(
                "Quantizing a RotatingKVCache with keep tokens is not supported."
            )
        quant_cache = RotatingQuantizedKVCache(
            self.max_size, keep=self.keep, group_size=group_size, bits=bits
        )
        quant_cache.offset = self.offset
        quant_cache._idx = self._idx
        if self.keys is not None:
            quant_cache.keys = mx.quantize(self.keys, group_size=group_size, bits=bits)
            quant_cache.values = mx.quantize(
                self.values, group_size=group_size, bits=bits
            )
        return quant_cache

    def make_mask(
        self, N: int, window_size: Optional[int] = None, return_array: bool = False
    ):
        if N > 1:
            window_size = window_size or self.max_size
            offset = min(self.max_size - 1, self.offset)
            if offset + N > window_size or return_array:
                return create_causal_mask(N, offset, window_size=window_size)
            else:
                return "causal"
        else:
            if window_size is None:
                return None
            if self.offset >= window_size and self.max_size > window_size:
                idx = self._idx
                if idx >= self.max_size:
                    idx = 0
                if self.offset < self.max_size:
                    mask_size = self.offset + 1
                else:
                    mask_size = self.max_size
                mask = mx.arange(mask_size) >= mask_size - window_size
                mask = mx.roll(mask, shift=idx + 1)
                return mask

    @classmethod
    def merge(_, caches):
        return BatchRotatingKVCache.merge(caches)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        total = self.keys.nbytes + self.values.nbytes
        for _, keys, values in self._checkpoints:
            total += keys.nbytes + values.nbytes
        # Retained rollback records are live buffers too (a row extracted
        # while its batch records rollback keeps them), so count them.
        for _, snapshot, keys, values in self._rollbacks:
            total += keys.nbytes + values.nbytes
            if snapshot[0] is not None:
                total += snapshot[0].nbytes + snapshot[1].nbytes
        return total


class RotatingQuantizedKVCache(_BaseCache):
    """Quantized counterpart of RotatingKVCache. `keys`/`values` are each a
    (packed, scales, biases) triple as produced by `mx.quantize`, rather than a
    single float array. Does not support `keep` (sink) tokens — batching
    (BatchRotatingKVCache) never supports them either, and mira-mlx never
    configures them, so this restriction costs nothing in practice."""

    step = 256

    def __init__(self, max_size, keep=0, group_size: int = 64, bits: int = 4):
        if keep > 0:
            raise NotImplementedError(
                "RotatingQuantizedKVCache does not support keep tokens."
            )
        self.keep = keep
        self.keys = None
        self.values = None
        self.offset = 0
        self.max_size = max_size
        self._idx = 0
        self.group_size = group_size
        self.bits = bits

    def _quantize(self, x):
        return mx.quantize(x, group_size=self.group_size, bits=self.bits)

    def _trim(self, trim_size, v, append=None):
        if trim_size > 0:
            v = tree_map(lambda a: a[..., trim_size:, :], v)
        if append is not None:
            v = tree_map(lambda a, b: mx.concatenate([a, b], axis=2), v, append)
        return v

    def _temporal_order(self, v):
        """
        Rearrange the cache into temporal order, slicing off the end if unused.
        """
        if self._idx == v[0].shape[2]:
            return v
        elif self._idx < self.offset:
            return tree_map(
                lambda a: mx.concatenate(
                    [a[..., self._idx :, :], a[..., : self._idx, :]], axis=2
                ),
                v,
            )
        else:
            return tree_map(lambda a: a[..., : self._idx, :], v)

    def _update_concat(self, keys, values):
        qkeys = self._quantize(keys)
        qvalues = self._quantize(values)
        if self.keys is None:
            self.keys = qkeys
            self.values = qvalues
        else:
            self.keys = self._temporal_order(self.keys)
            self.values = self._temporal_order(self.values)
            self._idx = self.keys[0].shape[2]
            trim_size = self._idx - self.max_size + 1
            self.keys = self._trim(trim_size, self.keys, qkeys)
            self.values = self._trim(trim_size, self.values, qvalues)
        self.offset += keys.shape[2]
        self._idx = self.keys[0].shape[2]
        return (self.keys, self.values)

    def _update_in_place(self, keys, values):
        (B, n_kv_heads, S, k_head_dim) = keys.shape
        v_head_dim = values.shape[3]
        prev = self.offset
        cur_size = self.keys[0].shape[2] if self.keys is not None else 0
        if self.keys is None or (prev >= cur_size and cur_size < self.max_size):
            new_size = min(self.step, self.max_size - prev)
            new_k = _empty_quantized(
                B,
                n_kv_heads,
                new_size,
                k_head_dim,
                self.group_size,
                self.bits,
                keys.dtype,
            )
            new_v = _empty_quantized(
                B,
                n_kv_heads,
                new_size,
                v_head_dim,
                self.group_size,
                self.bits,
                values.dtype,
            )
            if self.keys is not None:
                self.keys = tree_map(
                    lambda a, b: mx.concatenate([a, b], axis=2), self.keys, new_k
                )
                self.values = tree_map(
                    lambda a, b: mx.concatenate([a, b], axis=2), self.values, new_v
                )
            else:
                (self.keys, self.values) = (new_k, new_v)
            self._idx = prev
        trim_size = self.keys[0].shape[2] - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size
        if self._idx == self.max_size:
            self._idx = self.keep
        qkeys = self._quantize(keys)
        qvalues = self._quantize(values)
        for i in range(3):
            self.keys[i][..., self._idx : self._idx + S, :] = qkeys[i]
            self.values[i][..., self._idx : self._idx + S, :] = qvalues[i]
        self.offset += S
        self._idx += S
        if self.offset < self.max_size:
            return (
                tree_map(lambda a: a[..., : self.offset, :], self.keys),
                tree_map(lambda a: a[..., : self.offset, :], self.values),
            )
        return (self.keys, self.values)

    def update_and_fetch(self, keys, values):
        if keys.shape[2] == 1:
            return self._update_in_place(keys, values)
        return self._update_concat(keys, values)

    def size(self):
        return min(self.offset, self.max_size)

    @property
    def state(self):
        # An empty cache (a one-token prompt prefills nothing) has no state
        # yet; report it the way KVCache does rather than raise.
        if self.keys is None:
            return (self.keys, self.values)
        if self.offset < self.keys[0].shape[2]:
            return (
                tree_map(lambda a: a[..., : self.offset, :], self.keys),
                tree_map(lambda a: a[..., : self.offset, :], self.values),
            )
        else:
            return (self.keys, self.values)

    @state.setter
    def state(self, v):
        (self.keys, self.values) = v

    @property
    def meta_state(self):
        return tuple(
            map(
                str,
                (
                    self.keep,
                    self.max_size,
                    self.offset,
                    self._idx,
                    self.group_size,
                    self.bits,
                ),
            )
        )

    @meta_state.setter
    def meta_state(self, v):
        (
            self.keep,
            self.max_size,
            self.offset,
            self._idx,
            self.group_size,
            self.bits,
        ) = map(int, v)

    def is_trimmable(self):
        return self.offset < self.max_size

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        self._idx -= n
        return n

    def make_mask(
        self, N: int, window_size: Optional[int] = None, return_array: bool = False
    ):
        if N > 1:
            window_size = window_size or self.max_size
            offset = min(self.max_size - 1, self.offset)
            if offset + N > window_size or return_array:
                return create_causal_mask(N, offset, window_size=window_size)
            else:
                return "causal"
        else:
            if window_size is None:
                return None
            if self.offset >= window_size and self.max_size > window_size:
                idx = self._idx
                if idx >= self.max_size:
                    idx = 0
                if self.offset < self.max_size:
                    mask_size = self.offset + 1
                else:
                    mask_size = self.max_size
                mask = mx.arange(mask_size) >= mask_size - window_size
                mask = mx.roll(mask, shift=idx + 1)
                return mask

    @classmethod
    def merge(_, caches):
        return BatchRotatingQuantizedKVCache.merge(caches)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return tree_reduce(lambda a, x: a + x.nbytes, (self.keys, self.values), 0)


class ArraysCache(_BaseCache):
    _ROLLBACK_WINDOW = 64

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.left_padding = None
        instance.lengths = None
        instance.speculating = False
        instance._rollbacks = deque()
        instance._rollback_window = cls._ROLLBACK_WINDOW
        instance._rollback_epoch = 0
        instance._rollback_position = 0
        instance._rollback_positions = None
        instance._rollback_invalid_reason = None
        instance._host_lengths = None
        instance._host_left_padding = None
        instance._checkpoints = []
        return instance

    def __init__(self, size, left_padding: Optional[List[int]] = None):
        self.cache = [None] * size
        if left_padding:
            host_left_padding = [int(v) for v in left_padding]
            self.left_padding = mx.array(host_left_padding)
            self._host_left_padding = (self.left_padding, host_left_padding)

    def start_speculation(self, rollback_window: Optional[int] = None):
        self.speculating = True
        self._rollback_window = max(
            1,
            int(self._ROLLBACK_WINDOW if rollback_window is None else rollback_window),
        )
        self._rollbacks.clear()
        self._rollback_epoch += 1
        self._rollback_position = 0
        self._rollback_positions = None

    def stop_speculation(self):
        self.speculating = False
        self._rollbacks.clear()
        self._rollback_epoch += 1
        self._rollback_position = 0
        self._rollback_positions = None

    def rollback_marker(self):
        """Return an epoch-bound position for single-row PLD rollback."""
        if not self.speculating:
            raise RuntimeError("ArraysCache rollback recording is not active")
        if self.batch_size != 1 or (
            self._rollback_positions is not None and len(self._rollback_positions) != 1
        ):
            raise RuntimeError(
                "ArraysCache PLD rollback markers require exactly one live row"
            )
        position = (
            self._rollback_position
            if self._rollback_positions is None
            else self._rollback_positions[0]
        )
        return (self._rollback_epoch, position)

    def rewind_to_rollback_marker(self, marker):
        """Rewind to a marker without relying on retained deque totals."""
        if (
            not isinstance(marker, (tuple, list))
            or len(marker) != 2
            or (not all((isinstance(value, int) for value in marker)))
        ):
            raise ValueError(f"Invalid ArraysCache rollback marker: {marker!r}")
        (epoch, position) = marker
        if position < 0:
            raise ValueError(
                f"ArraysCache rollback marker has a negative position: {position}"
            )
        if not self.speculating or epoch != self._rollback_epoch:
            raise RuntimeError(
                "ArraysCache rollback marker belongs to an inactive or stale epoch"
            )
        if self.batch_size != 1 or (
            self._rollback_positions is not None and len(self._rollback_positions) != 1
        ):
            raise RuntimeError(
                "ArraysCache PLD rollback markers require exactly one live row"
            )
        current = (
            self._rollback_position
            if self._rollback_positions is None
            else self._rollback_positions[0]
        )
        if position > current:
            raise RuntimeError(
                f"ArraysCache rollback marker is ahead of live state: {position} > {current}"
            )
        return self.trim(current - position)

    def _host_vector(self, field, cached):
        value = getattr(self, field)
        if value is None:
            return (None, None)
        if cached is None or cached[0] is not value:
            cached = (value, [int(v) for v in value.tolist()])
        return (cached, cached[1])

    def _length_vector(self):
        (self._host_lengths, values) = self._host_vector("lengths", self._host_lengths)
        return values

    def _left_padding_vector(self):
        (self._host_left_padding, values) = self._host_vector(
            "left_padding", self._host_left_padding
        )
        return values

    def rollback_spans(self, length: int, mask=None):
        """Tokens this forward advances per row, host-side — or ``None``.

        A record credits each row a depth. Under a padded slab that depth is
        the row's own valid span, not the slab width: crediting the width lets
        a later rewind take tokens out of this record that the row never
        processed, and stop before the older record that really holds them.

        ``()`` means "unpadded, every row advanced ``length``". ``None`` means
        the geometry is not describable row-wise, and a layer must NOT stage a
        rollback for it. That covers a mask the cache metadata cannot explain,
        and a LEADING pad run: replay closures index a row's tokens from slab
        position 0, so a row whose tokens start later cannot be replayed by a
        scalar depth.
        """
        lengths = self._length_vector()
        padding = self._left_padding_vector()
        if padding is not None and max(padding) > 0:
            return None
        if lengths is None:
            return None if mask is not None else ()
        return [min(max(v, 0), length) for v in lengths]

    def _record_depths(self, num_tokens, depths):
        """Resolve a record's per-row depths, deriving them when not given."""
        if depths is None:
            depths = self.rollback_spans(num_tokens)
            if depths is None:
                raise RuntimeError(
                    "Cannot record a rollback for this forward: the padding geometry is not describable per row, so the record would credit rows tokens they never processed. The layer must check rollback_spans() and skip staging when it is None."
                )
        if not depths:
            return None
        depths = [int(v) for v in depths]
        if len(depths) != self.batch_size:
            raise RuntimeError(
                f"Rollback spans cover {len(depths)} rows but the cache holds {self.batch_size}"
            )
        return depths

    def record_rollback(
        self, num_tokens, fn, snapshot, *, per_row_fn=None, depths=None
    ):
        """Recorded by the owning layer during a forward while ``speculating``.

        Args:
            num_tokens (int): Sequence length of the forward being recorded.
            fn (callable): ``fn(m) -> list`` returning the cache entries as they
                would be had only the first ``m`` of ``num_tokens`` tokens been
                processed. Must be exact (e.g. replay the recurrence from the
                pre-forward state over the stashed per-token inputs).
            snapshot (list): The cache entries from before the forward (returned
                for a full rollback, ``m == 0``).
            per_row_fn (callable): Optional ``fn(m_list) -> list`` giving every
                row its own length in one graph. ``trim_ragged`` uses it when
                present and otherwise replays ``fn`` once per distinct length.
            depths (list): Optional per-row spans this forward advanced, as
                ``rollback_spans`` returns them. Derived from the cache's own
                metadata when omitted, so a layer never does the arithmetic.
        """
        depths = self._record_depths(num_tokens, depths)
        batch = self.batch_size
        if (
            self._rollback_positions is not None
            and len(self._rollback_positions) != batch
        ):
            raise RuntimeError(
                "ArraysCache rollback positions do not match the live batch; restart speculation after changing membership"
            )
        if self._rollback_positions is None and depths is None and self.empty():
            self._rollback_position += num_tokens
        else:
            if self._rollback_positions is None:
                self._rollback_positions = [self._rollback_position] * batch
            advances = [num_tokens] * batch if depths is None else depths
            self._rollback_positions = [
                position + advance
                for (position, advance) in zip(self._rollback_positions, advances)
            ]
        self._rollback_invalid_reason = None
        self._rollbacks.append(
            _RollbackRecord(num_tokens, fn, snapshot, per_row_fn, depths)
        )
        total = sum((r.span for r in self._rollbacks))
        while (
            len(self._rollbacks) > 1
            and total - self._rollbacks[0].span >= self._rollback_window
        ):
            total -= self._rollbacks.popleft().span

    def retire_rollbacks(self, keep: int = 1) -> int:
        """Drop all but the newest ``keep`` records once a cycle has committed.

        Each record pins a whole-batch pre-forward state, and a self-MTP
        cohort starts speculation once per membership, so the window alone
        kept ~window/(k+1) of them alive. Nothing rewinds past committed
        tokens; the newest record still serves an abort and
        ``latest_exact_rollback_boundary``. Same bookkeeping as the window's
        own pruning. Returns the number of records dropped.
        """
        keep = max(1, int(keep))
        dropped = 0
        while len(self._rollbacks) > keep:
            self._rollbacks.popleft()
            dropped += 1
        return dropped

    def latest_exact_rollback_boundary(self) -> ExactRollbackBoundary:
        """Export the newest uniform single-row rollback as a safe handle.

        Prefix fan-out needs to retain and materialize an exact recurrent
        boundary without learning the rollback stack's private representation.
        Ragged records are refused because they no longer describe one shared
        parent, and a staged Qwen4 PLE half is refused by the subclass through
        ``is_trimmable``.
        """
        if not self.speculating or not self._rollbacks:
            raise RuntimeError("the cache has no live exact rollback record")
        if not self.is_trimmable():
            raise RuntimeError("the cache has an incomplete staged rollback")
        if self.batch_size != 1:
            raise ValueError("an exact shared-prefix boundary needs one cache row")
        record = self._rollbacks[-1]
        boundary_tokens = record.num_tokens
        if record.depths is not None:
            if (
                len(record.depths) != 1
                or record.depths[0] <= 0
                or record.depths[0] > record.num_tokens
            ):
                raise ValueError(
                    f"a ragged rollback record is not one shared prefix: forward={record.num_tokens}, depths={record.depths}"
                )
            boundary_tokens = record.depths[0]
        if len(record.snapshot) != len(self.cache):
            raise RuntimeError("rollback record does not cover the full cache")
        return ExactRollbackBoundary(
            type(self),
            len(self.cache),
            boundary_tokens,
            record.fn,
            list(record.snapshot),
            list(self.cache),
        )

    def is_trimmable(self):
        return self.speculating

    def _row_capacity(self, batch_size: int) -> List[int]:
        """Tokens each row can still rewind exactly, row by row."""
        capacity = [0] * batch_size
        for record in self._rollbacks:
            for index, depth in enumerate(record.depth_list(batch_size)):
                capacity[index] += depth
        return capacity

    def _recorded_tokens(self):
        return sum((r[0] for r in self._rollbacks))

    def _rollback_budget_error(self, n, capacity=None):
        detail = ""
        if self._rollback_invalid_reason:
            detail = f" The records were dropped: {self._rollback_invalid_reason}."
        have = self._recorded_tokens() if capacity is None else capacity
        return RuntimeError(
            f"Cannot trim {n} tokens from ArraysCache: only {have} tokens of exact rollback are recorded. Recurrent state cannot be trimmed beyond the speculative window.{detail}"
        )

    def _invalidate_rollbacks(self, reason: str):
        """Drop rollback records that no longer describe the live rows.

        ``fn``/``snapshot`` capture whole-batch tensors, so any change of
        batch membership makes them un-replayable. Dropping them turns a
        later trim into a loud failure instead of a wrong-shaped restore.
        """
        self._clear_staged_rollback()
        if (
            self.speculating
            or self._rollbacks
            or self._rollback_position
            or (self._rollback_positions is not None)
        ):
            self._rollbacks.clear()
            self._rollback_invalid_reason = reason
            self._rollback_epoch += 1
            self._rollback_position = 0
            self._rollback_positions = None

    def _clear_staged_rollback(self):
        """Drop a rollback staged by an interrupted forward.

        No-op here. A subclass that stages part of a record before
        ``record_rollback`` combines it (see ``Qwen4ArraysCache``) must
        override this, or the stale half survives a membership change.
        """

    def trim(self, n):
        if n <= 0:
            return 0
        batch = self.batch_size
        capacity = self._row_capacity(batch)
        if min(capacity, default=0) < n:
            raise self._rollback_budget_error(n, min(capacity, default=0))
        self._rewind_rows([n] * batch, batch, prefer_per_row=False)
        if self._rollback_positions is None:
            self._rollback_position -= n
        elif len(self._rollback_positions) == batch:
            self._rollback_positions = [
                position - n for position in self._rollback_positions
            ]
        self._collapse_empty_rollback_positions()
        return n

    def _collapse_empty_rollback_positions(self):
        """Return a fully rewound, uninitialized batch to its lazy form."""
        if (
            self._rollback_positions is not None
            and self.empty()
            and all((position == 0 for position in self._rollback_positions))
        ):
            self._rollback_position = 0
            self._rollback_positions = None

    def supports_ragged_trim(self):
        return True

    def _rewind_rows(self, remaining: List[int], batch: int, *, prefer_per_row):
        """Walk the record stack newest-first, rewinding each row by its own count.

        Rows sit at different depths after a divergent trim, so a record is
        applied to whatever rows still owe a rewind *and* still have depth in
        it; a row that has already passed the record contributes nothing and
        keeps its live state.
        """
        stack = self._rollbacks
        index = len(stack) - 1
        while max(remaining, default=0) > 0:
            if index < 0:
                raise RuntimeError(
                    f"ArraysCache rollback stack was exhausted mid-rewind; {remaining} tokens per row were still owed"
                )
            record = stack[index]
            depths = record.depth_list(batch)
            take = [min(remaining[i], depths[i]) for i in range(batch)]
            if any(take):
                lengths = [depths[i] - take[i] for i in range(batch)]
                rows = [i for i in range(batch) if take[i] > 0]
                if prefer_per_row and record.per_row_fn is not None:
                    self.cache = self._blend_rows(
                        list(record.per_row_fn(list(lengths))), rows
                    )
                else:
                    for m in sorted({lengths[i] for i in rows}):
                        candidate = (
                            list(record.snapshot) if m == 0 else list(record.fn(m))
                        )
                        self.cache = self._blend_rows(
                            candidate, [i for i in rows if lengths[i] == m]
                        )
                stack[index] = record.with_depths(lengths)
                remaining = [r - t for (r, t) in zip(remaining, take)]
            index -= 1
        while stack and stack[-1].exhausted():
            stack.pop()

    def _blend_rows(self, candidate, rows: List[int]):
        """Take ``rows`` from ``candidate`` and every other row from the live state."""
        if len(candidate) != len(self.cache):
            raise RuntimeError(
                f"ArraysCache rollback returned {len(candidate)} entries, expected {len(self.cache)}"
            )
        batch = self.batch_size
        if len(rows) == batch:
            return list(candidate)
        keep = mx.array([index in set(rows) for index in range(batch)])
        blended = []
        for index, (live, new) in enumerate(zip(self.cache, candidate)):
            if live is None and new is None:
                blended.append(None)
                continue
            if live is None or new is None:
                raise RuntimeError(
                    f"ArraysCache entry {index} is None on one side of a ragged rollback; a per-row rewind cannot materialize 'no state' for part of a batch"
                )
            mask = keep.reshape((batch,) + (1,) * (live.ndim - 1))
            blended.append(mx.where(mask, new, live))
        return blended

    def preflight_ragged_trim(self, n, *, validate: bool = True):
        """Validate a per-row rewind without mutating anything."""
        batch = self.batch_size
        drops = _row_vector(n, batch, "ArraysCache.trim_ragged")
        if max(drops, default=0) == 0:
            return drops
        capacity = self._row_capacity(batch)
        short = [i for i in range(batch) if drops[i] > capacity[i]]
        if short:
            detail = ""
            if self._rollback_invalid_reason:
                detail = f" The records were dropped: {self._rollback_invalid_reason}."
            raise RuntimeError(
                f"Cannot rewind ArraysCache rows {short} by {[drops[i] for i in short]}: only {[capacity[i] for i in short]} tokens of exact rollback are recorded for them.{detail}"
            )
        return drops

    def trim_ragged(self, n, *, validate: bool = True):
        """Rewind row ``i`` of the recurrent state by ``n[i]`` tokens.

        Rows are independent, so a record is replayed at each distinct row
        length and every row keeps its own slice (the design's interim form
        ``a0``). A layer that stages ``per_row_fn`` gets the one-graph form
        instead. Rows with ``n[i] == 0`` keep their live state untouched.
        """
        drops = self.preflight_ragged_trim(n, validate=validate)
        if max(drops, default=0) == 0:
            return drops
        self._rewind_rows(list(drops), self.batch_size, prefer_per_row=True)
        if self._rollback_positions is None and len(set(drops)) <= 1:
            self._rollback_position -= drops[0]
        else:
            if self._rollback_positions is None:
                self._rollback_positions = [self._rollback_position] * self.batch_size
            self._rollback_positions = [
                position - drop
                for (position, drop) in zip(self._rollback_positions, drops)
            ]
        self._collapse_empty_rollback_positions()
        return drops

    def state_checkpoint(self, positions: List[int], force: bool = False):
        max_checkpoints = _state_checkpoint_max()
        if max_checkpoints <= 0 or self.empty():
            return
        if len(self._checkpoints) != len(positions):
            self._checkpoints = [[] for _ in positions]
        stride = _state_checkpoint_stride()
        for i, position in enumerate(positions):
            lane = self._checkpoints[i]
            last = lane[-1][0] if lane else 0
            if lane and position <= last:
                continue
            if not force and position - last < stride:
                continue
            snapshot = [
                None if c is None else mx.array(c[i : i + 1]) for c in self.cache
            ]
            # An unevaluated row slice keeps the whole batched state alive;
            # scheduling the copy detaches it without a host sync.
            mx.async_eval(snapshot)
            lane.append((position, snapshot))
            _thin_checkpoints(lane, max_checkpoints)

    def snap_trim_position(self, position: int) -> Optional[int]:
        if self.batch_size > 1 or len(self._checkpoints) > 1:
            return None
        best = 0
        if self._checkpoints:
            for p, _ in self._checkpoints[0]:
                if p <= position:
                    best = max(best, p)
        return best

    def trim_to_position(self, position: int, num_tokens: int) -> int:
        lane = self._checkpoints[0] if self._checkpoints else []
        if position > 0:
            found = None
            for p, snapshot in reversed(lane):
                if p == position:
                    found = snapshot
                    break
            if found is None:
                raise RuntimeError(
                    f"ArraysCache has no state checkpoint at position {position}"
                )
            self.cache = [None if a is None else mx.array(a) for a in found]
        else:
            self.cache = [None] * len(self.cache)
        while lane and lane[-1][0] > position:
            lane.pop()
        self._rollbacks.clear()
        self._rollback_epoch += 1
        self._rollback_position = 0
        self._rollback_positions = None
        return num_tokens

    @property
    def batch_size(self):
        for c in self.cache:
            if c is not None:
                return c.shape[0]
        if self.left_padding is not None:
            return self.left_padding.size
        elif self.lengths is not None:
            return self.lengths.size
        else:
            return 1

    def __setitem__(self, idx, value):
        self.cache[idx] = value

    def __getitem__(self, idx):
        return self.cache[idx]

    def _persistable_checkpoints(self):
        """Single-lane checkpoints whose snapshots are fully materialized.

        Snapshots containing ``None`` entries cannot round-trip through
        ``save_prompt_cache`` (safetensors holds arrays only), so they are
        skipped. Batched histories are never persisted (snapshot files hold
        per-request caches).
        """
        if len(self._checkpoints) != 1:
            return []
        return [
            (p, snapshot)
            for (p, snapshot) in self._checkpoints[0]
            if all((a is not None for a in snapshot))
        ]

    @property
    def state(self):
        left_padding = mx.array([]) if self.left_padding is None else self.left_padding
        lengths = mx.array([]) if self.lengths is None else self.lengths
        checkpoints = self._persistable_checkpoints()
        if checkpoints:
            cache = [list(self.cache), [list(s) for (_, s) in checkpoints]]
        else:
            cache = self.cache
        return (cache, left_padding, lengths)

    @state.setter
    def state(self, v):
        (cache, left_padding, lengths) = v
        self.left_padding = left_padding if left_padding.size > 0 else None
        self.lengths = lengths if lengths.size > 0 else None
        if (
            len(cache) == 2
            and isinstance(cache[0], list)
            and isinstance(cache[1], list)
        ):
            self.cache = list(cache[0])
            self._pending_checkpoint_snapshots = [list(s) for s in cache[1]]
        else:
            self.cache = cache

    @property
    def meta_state(self):
        checkpoints = self._persistable_checkpoints()
        if checkpoints:
            return tuple(["ckptv1"] + [str(p) for (p, _) in checkpoints])
        return ""

    @meta_state.setter
    def meta_state(self, v):
        pending = getattr(self, "_pending_checkpoint_snapshots", None)
        self._pending_checkpoint_snapshots = None
        if not v:
            return
        if v[0] != "ckptv1":
            raise ValueError(f"Unknown ArraysCache metadata version: {v[0]}")
        positions = [int(p) for p in v[1:]]
        if pending is None or len(pending) != len(positions):
            raise ValueError("ArraysCache checkpoint state/metadata mismatch")
        self._checkpoints = [list(zip(positions, pending))]

    def filter(self, batch_indices):
        """
        In-place filter to keep just the given indices in the cache.
        """
        self._invalidate_rollbacks("filter() changed the batch membership")
        host_indices = None
        if isinstance(batch_indices, (list, tuple, range)):
            try:
                host_indices = [operator.index(i) for i in batch_indices]
            except (TypeError, ValueError, OverflowError):
                pass
        old_batch = self.batch_size
        old_left_padding = self.left_padding
        old_lengths = self.lengths
        old_host_left_padding = self._host_left_padding
        old_host_lengths = self._host_lengths
        self.cache = [c[batch_indices] if c is not None else None for c in self.cache]
        if self.left_padding is not None:
            self.left_padding = self.left_padding[batch_indices]
        if self.lengths is not None:
            self.lengths = self.lengths[batch_indices]

        def filtered_mirror(value, old_value, cached):
            if value is None:
                return None
            if host_indices is None or cached is None or cached[0] is not old_value:
                return None
            try:
                if len(cached[1]) != old_batch:
                    return None
                host = [cached[1][i] for i in host_indices]
            except (IndexError, TypeError):
                return None
            return (value, host)

        self._host_left_padding = filtered_mirror(
            self.left_padding, old_left_padding, old_host_left_padding
        )
        self._host_lengths = filtered_mirror(
            self.lengths, old_lengths, old_host_lengths
        )
        if self._checkpoints:
            self._checkpoints = [self._checkpoints[i] for i in batch_indices]

    def extend(self, other):
        """
        In-place extend this cache with the other cache.
        """
        self._invalidate_rollbacks("extend() changed the batch membership")
        a_batch = self.batch_size
        b_batch = other.batch_size
        old_left_padding = self.left_padding
        old_lengths = self.lengths
        old_host_left_padding = self._host_left_padding
        old_host_lengths = self._host_lengths
        other_left_padding = other.left_padding
        other_lengths = other.lengths
        other_host_left_padding = other._host_left_padding
        other_host_lengths = other._host_lengths

        def zeros(slot, shape, dtype):
            return mx.zeros(shape, dtype=dtype)

        def cat(a, b, slot=None, fill=zeros):
            shape = dtype = None
            if a is not None:
                shape = a.shape
                dtype = a.dtype
            if b is not None:
                shape = b.shape
                dtype = b.dtype
            if shape is None:
                return None
            if a is None:
                a = fill(slot, (a_batch,) + shape[1:], dtype)
            if b is None:
                b = fill(slot, (b_batch,) + shape[1:], dtype)
            return mx.concatenate([a, b])

        self._adopt_empty_fill([self, other])
        self.cache = [
            cat(c, o, slot, self._empty_slot)
            for (slot, (c, o)) in enumerate(zip(self.cache, other.cache))
        ]
        self.left_padding = cat(self.left_padding, other.left_padding)
        self.lengths = cat(self.lengths, other.lengths)

        def extended_mirror(value, a_value, a_cached, b_value, b_cached):
            if value is None:
                return None

            def source_host(source, cached, batch):
                if source is None:
                    return [0] * batch
                if cached is None or cached[0] is not source:
                    return None
                host = cached[1]
                try:
                    valid_size = len(host) == batch
                except TypeError:
                    return None
                if not valid_size:
                    return None
                return list(host)

            a_host = source_host(a_value, a_cached, a_batch)
            b_host = source_host(b_value, b_cached, b_batch)
            if a_host is None or b_host is None:
                return None
            return (value, a_host + b_host)

        self._host_left_padding = extended_mirror(
            self.left_padding,
            old_left_padding,
            old_host_left_padding,
            other_left_padding,
            other_host_left_padding,
        )
        self._host_lengths = extended_mirror(
            self.lengths,
            old_lengths,
            old_host_lengths,
            other_lengths,
            other_host_lengths,
        )
        a_lanes = self._checkpoints or [[] for _ in range(a_batch)]
        b_lanes = other._checkpoints or [[] for _ in range(b_batch)]
        self._checkpoints = [list(l) for l in a_lanes] + [list(l) for l in b_lanes]

    def extract(self, idx):
        cache = ArraysCache(len(self.cache))
        cache.cache = [
            None if c is None else mx.contiguous(c[idx : idx + 1]) for c in self.cache
        ]
        if idx < len(self._checkpoints):
            cache._checkpoints = [list(self._checkpoints[idx])]
        cache._rollback_invalid_reason = "extract() left the batch behind"
        return cache

    def prepare(self, lengths=None, **kwargs):
        self.lengths = mx.array(lengths)
        if lengths is not None:
            self._host_lengths = (self.lengths, [int(v) for v in lengths])

    def finalize(self):
        self.lengths = None
        self.left_padding = None
        self._host_lengths = None
        self._host_left_padding = None

    def advance(self, N):
        lengths = self._length_vector()
        padding = self._left_padding_vector()
        if self.lengths is not None:
            self.lengths -= N
        if self.left_padding is not None:
            self.left_padding -= N
        if lengths is not None:
            self._host_lengths = (self.lengths, [v - N for v in lengths])
        if padding is not None:
            self._host_left_padding = (self.left_padding, [v - N for v in padding])
        metadata = tuple(
            (v for v in (self.lengths, self.left_padding) if v is not None)
        )
        if metadata:
            for index, value in enumerate(self.cache):
                if value is not None:
                    self.cache[index] = mx.depends(value, metadata)
                    break

    def _host_all_valid(self, N: int) -> bool:
        """Whether every row is valid across an ``N``-token slab, host-side.

        An unpadded prompt merged into a batch keeps ``left_padding`` at 0 and
        ``advance`` drives it negative; a verify block sets ``lengths`` to its
        own width. Both describe an all-True mask, and handing a GDN layer that
        array instead of ``None`` refuses the fused decode kernel and the
        packed kernel for no reason. Only mirrors that still describe the live
        arrays are trusted, so this never reads the device: an unmirrored
        field keeps the explicit mask.
        """
        for (value, cached, valid) in (
            (self.left_padding, self._host_left_padding, lambda v: v <= 0),
            (self.lengths, self._host_lengths, lambda v: v >= N),
        ):
            if value is None:
                continue
            if cached is None or cached[0] is not value:
                return False
            if not all((valid(v) for v in cached[1])):
                return False
        return True

    def make_mask(self, N: int):
        if self._host_all_valid(N):
            return None
        pos = mx.arange(N)
        mask = None
        if self.left_padding is not None:
            mask = pos >= self.left_padding[:, None]
        if self.lengths is not None:
            bounded = pos < self.lengths[:, None]
            mask = bounded if mask is None else mx.logical_and(mask, bounded)
        return mask

    @classmethod
    def merge(cls, caches):
        n_state = len(caches[0].cache)
        B = len(caches)
        cache = cls(n_state)
        cache._adopt_empty_fill(caches)
        if all((c.empty() for c in caches)):
            host_left_padding = [0] * B
            cache.left_padding = mx.array(host_left_padding)
            cache._host_left_padding = (cache.left_padding, host_left_padding)
            return cache
        for e in range(n_state):
            c_init = next((c[e] for c in caches if c[e] is not None), None)
            if c_init is None:
                cache[e] = None
                continue
            shape = list(c_init.shape)
            shape[0] = B
            if any(c[e] is None for c in caches):
                cache[e] = cache._empty_slot(e, tuple(shape), c_init.dtype)
            else:
                # Every row is overwritten below; no empty-slot semantics.
                cache[e] = mx.zeros(shape, c_init.dtype)
            for i in range(B):
                if caches[i][e] is None:
                    continue
                if tuple(caches[i][e].shape[1:]) != tuple(shape[1:]):
                    raise ValueError(
                        f"ArraysCache.merge: state slot {e} of lane {i} has shape {tuple(caches[i][e].shape)}, which does not fit a batched slot of {tuple(shape)}."
                    )
                cache[e][i : i + 1] = caches[i][e]
        cache._checkpoints = [
            list(c._checkpoints[0]) if len(c._checkpoints) == 1 else [] for c in caches
        ]
        cache._rollback_invalid_reason = "merge() built a new batch"
        return cache

    def empty(self):
        return self.cache[0] is None

    def _empty_slot(self, slot, shape, dtype):
        """Batched stand-in for rows whose state ``slot`` is still None.

        Zeros are the model's own reading of None for GDN/conv state. A
        subclass whose model reads None differently must override this, or a
        cold row joining warm rows is silently seeded with the wrong state.
        """
        return mx.zeros(shape, dtype=dtype)

    def _adopt_empty_fill(self, caches):
        """Carry whatever ``_empty_slot`` needs from the caches being joined."""

    @property
    def nbytes(self):
        total = sum((c.nbytes for c in self.cache if c is not None))
        for lane in self._checkpoints:
            for _, snapshot in lane:
                total += sum((a.nbytes for a in snapshot if a is not None))
        return total


class CacheList(_BaseCache):
    def __init__(self, *caches):
        self.caches = caches

    def __getitem__(self, idx):
        return self.caches[idx]

    def is_trimmable(self):
        return all((c.is_trimmable() for c in self.caches))

    def supports_ragged_trim(self):
        return all((c.supports_ragged_trim() for c in self.caches))

    def preflight_ragged_trim(self, n, *, validate: bool = True):
        for c in self.caches:
            c.preflight_ragged_trim(n, validate=validate)
        return n

    def trim_ragged(self, n, *, validate: bool = True):
        self.preflight_ragged_trim(n, validate=validate)
        applied = [c.trim_ragged(n, validate=False) for c in self.caches]
        if any((a != applied[0] for a in applied[1:])):
            raise RuntimeError("CacheList members disagreed on a ragged trim")
        return applied[0]

    def trim(self, n):
        for c in self.caches:
            m = c.trim(n)
        return m

    def start_speculation(self, rollback_window: Optional[int] = None):
        for c in self.caches:
            c.start_speculation(rollback_window)

    def stop_speculation(self):
        for c in self.caches:
            c.stop_speculation()

    def state_checkpoint(self, positions: List[int], force: bool = False):
        for c in self.caches:
            c.state_checkpoint(positions, force=force)

    def snap_trim_position(self, position: int) -> Optional[int]:
        while True:
            start = position
            for c in self.caches:
                position = c.snap_trim_position(position)
                if position is None:
                    return None
            if position == start:
                return position

    def trim_to_position(self, position: int, num_tokens: int) -> int:
        for c in self.caches:
            m = c.trim_to_position(position, num_tokens)
        return m

    @property
    def state(self):
        return [c.state for c in self.caches]

    @state.setter
    def state(self, v):
        for c, s in zip(self.caches, v):
            c.state = s

    @property
    def meta_state(self):
        return (
            [_cache_class_token(type(c)) for c in self.caches],
            [c.meta_state for c in self.caches],
        )

    @meta_state.setter
    def meta_state(self, v):
        for c, m in zip(self.caches, v[1]):
            c.meta_state = m

    def filter(self, batch_indices):
        """
        In-place filter to keep just the given indices in the cache.
        """
        for c in self.caches:
            c.filter(batch_indices)

    def extend(self, other):
        """
        In-place extend this cache with the other cache.
        """
        for c, o in zip(self.caches, other.caches):
            c.extend(o)

    @classmethod
    def merge(cls, caches):
        cache = cls()
        cache.caches = tuple(
            (
                caches[0].caches[i].merge([c.caches[i] for c in caches])
                for i in range(len(caches[0].caches))
            )
        )
        return cache

    def extract(self, idx):
        return CacheList(*(c.extract(idx) for c in self.caches))

    def prepare(self, **kwargs):
        for c in self.caches:
            c.prepare(**kwargs)

    def finalize(self):
        for c in self.caches:
            c.finalize()

    def size(self):
        return max((c.size() for c in self.caches))

    def empty(self):
        return self.caches[0].empty()

    @property
    def nbytes(self):
        return sum((c.nbytes for c in self.caches))

    @classmethod
    def from_state(cls, state, meta_state):
        obj = cls.__new__(cls)
        obj.caches = [
            _resolve_cache_class(c).from_state(s, m)
            for (s, c, m) in zip(state, *meta_state)
        ]
        return obj


def dynamic_roll(x, shifts, axis):
    n = x.shape[axis]
    expand_shifts = (...,) + (None,) * (x.ndim - axis)
    expand_indices = expand_shifts[:-1]
    idx = (mx.arange(n)[expand_indices] - shifts[expand_shifts]) % n
    rolled = mx.take_along_axis(x, idx, axis=axis)
    return rolled


@dataclass
class _BucketedAttentionGroup:
    indices: tuple[int, ...]
    index_array: mx.array
    cache: "BatchKVCache"


class BatchQuantizedKVCache(_BaseCache):
    """Continuous-batching counterpart of :class:`QuantizedKVCache`.

    Rows are right-aligned exactly like ``BatchKVCache`` while K/V remain in
    MLX's packed ``(weight, scale, bias)`` representation.  Uniform rotation
    and asymmetric key/value bit widths are supported. KVarN-normalized rows
    intentionally fail closed: their frozen per-row channel scales need a
    separate batch-attention contract before they can be merged safely.
    """

    step = 256

    def __init__(
        self,
        left_padding: List[int],
        group_size: int = 64,
        bits: int = 8,
        *,
        key_bits: Optional[int] = None,
        value_bits: Optional[int] = None,
        rotate: bool = False,
    ):
        self.keys = None
        self.values = None
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-p for p in left_padding])
        self._idx = 0
        self.group_size = group_size
        self.key_bits = bits if key_bits is None else key_bits
        self.value_bits = bits if value_bits is None else value_bits
        QuantizedKVCache._validate_config(group_size, self.key_bits, self.value_bits)
        self.bits = self.key_bits if self.key_bits == self.value_bits else None
        self.rotate = rotate
        self.normalize = False
        self.key_scale = self.value_scale = None
        self._right_padding = None

    def _quantize(self, x, bits, *, keys: bool):
        # Rotation is compensated by rotating the queries, which leaves Q.K
        # unchanged; nothing un-rotates the attention output, so values must
        # be stored as-is, exactly like QuantizedKVCache and to_quantized.
        # ``keys`` is keyword-only and required so no caller can rotate values
        # by default.
        if keys and self.rotate and hadamard_size_ok(x.shape[-1]):
            x = rotate_last(x)
        return mx.quantize(x, group_size=self.group_size, bits=bits)

    def update_and_fetch(self, keys, values):
        (B, H, steps, dk) = keys.shape
        dv = values.shape[-1]
        prev = self._idx
        if self.keys is None or prev + steps > self.keys[0].shape[2]:
            grow = (self.step + steps - 1) // self.step * self.step
            new_k = _empty_quantized(
                B, H, grow, dk, self.group_size, self.key_bits, keys.dtype
            )
            new_v = _empty_quantized(
                B, H, grow, dv, self.group_size, self.value_bits, values.dtype
            )
            if self.keys is None:
                (self.keys, self.values) = (new_k, new_v)
            else:
                if prev % self.step:
                    self.keys = tree_map(lambda x: x[..., :prev, :], self.keys)
                    self.values = tree_map(lambda x: x[..., :prev, :], self.values)
                self.keys = tree_map(
                    lambda a, b: mx.concatenate([a, b], axis=2), self.keys, new_k
                )
                self.values = tree_map(
                    lambda a, b: mx.concatenate([a, b], axis=2), self.values, new_v
                )
        qk = self._quantize(keys, self.key_bits, keys=True)
        qv = self._quantize(values, self.value_bits, keys=False)
        self._idx += steps
        self.offset += steps
        for i in range(3):
            self.keys[i][..., prev : self._idx, :] = qk[i]
            self.values[i][..., prev : self._idx, :] = qv[i]
        return tree_map(lambda x: x[..., : self._idx, :], (self.keys, self.values))

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchQuantizedKVCache"
                )
            padding = mx.array(left_padding)
            self.left_padding += padding
            self.offset -= padding
        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self):
        if self._right_padding is None:
            return
        padding = self._right_padding
        if self.keys is not None:
            self.keys = tree_map(
                lambda x: dynamic_roll(x, padding[:, None], axis=2), self.keys
            )
            self.values = tree_map(
                lambda x: dynamic_roll(x, padding[:, None], axis=2), self.values
            )
        self.offset -= padding
        self.left_padding += padding
        self._right_padding = None
        self._tie_row_metadata()

    def _tie_row_metadata(self):
        """Make evaluating K/V also evaluate the rebound per-row metadata.

        See ``BatchKVCache._tie_row_metadata``: layers whose mask is never
        built would otherwise chain one lazy node per verify round.
        """
        if self.keys is not None:
            self.keys = tree_map(
                lambda a: mx.depends(a, (self.left_padding, self.offset)), self.keys
            )

    @property
    def state(self):
        if self.keys is None:
            return (None, None, self.offset, self.left_padding)
        return (
            tree_map(lambda x: x[..., : self._idx, :], self.keys),
            tree_map(lambda x: x[..., : self._idx, :], self.values),
            self.offset,
            self.left_padding,
        )

    @state.setter
    def state(self, value):
        (self.keys, self.values, self.offset, self.left_padding) = value
        self._idx = 0 if self.keys is None else self.keys[0].shape[2]
        self._right_padding = None

    @property
    def meta_state(self):
        return tuple(
            map(
                str,
                (
                    1,
                    self._idx,
                    self.group_size,
                    self.key_bits,
                    self.value_bits,
                    int(self.rotate),
                ),
            )
        )

    @meta_state.setter
    def meta_state(self, value):
        (
            version,
            self._idx,
            self.group_size,
            self.key_bits,
            self.value_bits,
            rotate,
        ) = map(int, value)
        if version != 1:
            raise ValueError(f"Unsupported BatchQuantizedKVCache version: {version}")
        QuantizedKVCache._validate_config(
            self.group_size, self.key_bits, self.value_bits
        )
        self.bits = self.key_bits if self.key_bits == self.value_bits else None
        self.rotate = bool(rotate)
        self.normalize = False
        self.key_scale = self.value_scale = None
        self._right_padding = None

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self._idx, n)
        self._idx -= n
        self.offset -= n
        return n

    def supports_ragged_trim(self):
        return True

    def preflight_ragged_trim(self, n, *, validate: bool = True):
        who = f"{type(self).__name__}.trim_ragged"
        return _ragged_slab_plan(self, n, who, validate)

    def trim_ragged(self, n, *, validate: bool = True):
        """Per-row rewind, identical bookkeeping to ``BatchKVCache``.

        Quantization groups run along the head-dim axis, so rolling whole
        token rows along the sequence axis is exact on the packed
        ``(weight, scale, bias)`` triple.
        """
        (drops, uniform, residual) = self.preflight_ragged_trim(n, validate=validate)
        if residual is None:
            return drops
        if uniform:
            self._idx -= uniform
            self.offset -= uniform
        if max(residual) > 0:
            shifts = mx.array(residual)
            if self.keys is not None:
                roll = lambda x: _roll_rows_right(x, shifts, 2, 0, self._idx)
                self.keys = tree_map(roll, self.keys)
                self.values = tree_map(roll, self.values)
            self.left_padding = self.left_padding + shifts
            self.offset = self.offset - shifts
            self._tie_row_metadata()
        return drops

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        return create_causal_mask(
            N, offset=self._idx, left_padding=self.left_padding, **kwargs
        )

    def empty(self):
        return self.keys is None

    def size(self):
        return self._idx

    @property
    def batch_size(self):
        if self.keys is not None:
            return int(self.keys[0].shape[0])
        return int(self.left_padding.shape[0])

    def is_single_row(self):
        return self.batch_size == 1

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return tree_reduce(
            lambda total, x: total + x.nbytes, (self.keys, self.values), 0
        )

    def filter(self, batch_indices):
        if self.keys is not None:
            self.keys = tree_map(lambda x: x[batch_indices], self.keys)
            self.values = tree_map(lambda x: x[batch_indices], self.values)
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]
        if self._right_padding is not None:
            self._right_padding = self._right_padding[batch_indices]
        shift = min(int(self.left_padding.min().item()), self._idx)
        if shift > 0:
            if self.keys is not None:
                self.keys = tree_map(lambda x: x[..., shift:, :], self.keys)
                self.values = tree_map(lambda x: x[..., shift:, :], self.values)
            self._idx -= shift
            self.left_padding -= shift

    def extend(self, other):
        if (self.group_size, self.key_bits, self.value_bits, self.rotate) != (
            other.group_size,
            other.key_bits,
            other.value_bits,
            other.rotate,
        ):
            raise ValueError("Cannot extend incompatible quantized batch caches")
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return
        max_idx = max(self._idx, other._idx)
        max_size = max(
            0 if self.keys is None else self.keys[0].shape[2],
            0 if other.keys is None else other.keys[0].shape[2],
            max_idx,
        )
        ref_k = self.keys if self.keys is not None else other.keys
        ref_v = self.values if self.values is not None else other.values

        def empty_like(parts, batch):
            return tuple(
                (
                    mx.zeros((batch, *x.shape[1:2], 0, x.shape[-1]), dtype=x.dtype)
                    for x in parts
                )
            )

        def pad(cache, ref_keys, ref_values):
            if cache.keys is None:
                batch = cache.offset.shape[0]
                keys = empty_like(ref_keys, batch)
                values = empty_like(ref_values, batch)
            else:
                (keys, values) = (cache.keys, cache.values)
            left = max_idx - cache._idx
            right = max_size - keys[0].shape[2] - left
            if right < 0:
                keys = tree_map(lambda x: x[..., :right, :], keys)
                values = tree_map(lambda x: x[..., :right, :], values)
                right = 0
            if left or right:
                spec = [(0, 0), (0, 0), (left, right), (0, 0)]
                keys = tree_map(lambda x: mx.pad(x, spec), keys)
                values = tree_map(lambda x: mx.pad(x, spec), values)
            return (keys, values, cache.offset, cache.left_padding + left)

        (a, b) = (pad(self, ref_k, ref_v), pad(other, ref_k, ref_v))
        self.keys = tree_map(lambda x, y: mx.concatenate([x, y]), a[0], b[0])
        self.values = tree_map(lambda x, y: mx.concatenate([x, y]), a[1], b[1])
        self.offset = mx.concatenate([a[2], b[2]])
        self.left_padding = mx.concatenate([a[3], b[3]])
        self._idx = max_idx

    def extract(self, idx):
        cache = QuantizedKVCache(
            group_size=self.group_size,
            bits=self.key_bits,
            key_bits=self.key_bits,
            value_bits=self.value_bits,
            rotate=self.rotate,
        )
        if self.keys is None:
            return cache
        padding = int(self.left_padding[idx].item())
        end = self._idx
        if self._right_padding is not None:
            end -= int(self._right_padding[idx].item())
        cache.keys = tree_map(
            lambda x: mx.contiguous(x[idx : idx + 1, :, padding:end]), self.keys
        )
        cache.values = tree_map(
            lambda x: mx.contiguous(x[idx : idx + 1, :, padding:end]), self.values
        )
        cache.offset = cache.keys[0].shape[2]
        return cache

    @classmethod
    def merge(cls, caches):
        if not caches:
            raise ValueError("Cannot merge an empty cache list")
        config = (
            caches[0].group_size,
            caches[0].key_bits,
            caches[0].value_bits,
            caches[0].rotate,
        )
        if any((c.normalize for c in caches)):
            raise ValueError(
                "Batching normalized QuantizedKVCache is not supported safely yet"
            )
        if any(
            (
                (c.group_size, c.key_bits, c.value_bits, c.rotate) != config
                for c in caches
            )
        ):
            raise ValueError("Cannot merge incompatible QuantizedKVCache rows")
        lengths = [c.size() for c in caches]
        width = max(lengths)
        batch = cls(
            [width - n for n in lengths],
            group_size=config[0],
            bits=config[1],
            key_bits=config[1],
            value_bits=config[2],
            rotate=config[3],
        )
        batch.offset = mx.array(lengths)
        batch._idx = width
        if width == 0:
            return batch
        ref_k = next((c.keys for c in caches if c.keys is not None))
        ref_v = next((c.values for c in caches if c.values is not None))

        def merge_parts(attr, ref):
            rows = []
            for cache, length in zip(caches, lengths):
                parts = getattr(cache, attr)
                if parts is None:
                    parts = tuple(
                        (
                            mx.zeros((1, x.shape[1], 0, x.shape[-1]), dtype=x.dtype)
                            for x in ref
                        )
                    )
                pad = width - length
                spec = [(0, 0), (0, 0), (pad, 0), (0, 0)]
                rows.append(tree_map(lambda x: mx.pad(x[..., :length, :], spec), parts))
            return tree_map(lambda *xs: mx.concatenate(xs), *rows)

        batch.keys = merge_parts("keys", ref_k)
        batch.values = merge_parts("values", ref_v)
        return batch


class BatchKVCache(_BaseCache):
    step = 256

    def __init__(
        self, left_padding: List[int], attention_backend: Optional[str] = None
    ):
        """
        The BatchKV cache expects inputs to be left-padded.

        E.g. the following prompts:

            [1, 3, 5]
            [7]
            [2, 6, 8, 9]

        Should be padded like so:

            [0, 1, 3, 5]
            [0, 0, 0, 7]
            [2, 6, 8, 9]

        And ``left_padding`` specifies the amount of padding for each.
        In this case, ``left_padding = [1, 3, 0]``.
        """
        self.keys = None
        self.values = None
        self.left_padding = mx.array(left_padding)
        # A host list seeds the exact floor ``trim_ragged`` reclaims against.
        self._host_padding_floor = (
            (self.left_padding, [int(l) for l in left_padding])
            if isinstance(left_padding, (list, tuple))
            else None
        )
        self._bind_unpadded(self._padding_floor())
        self.offset = mx.array([-l for l in left_padding])
        self._idx = 0
        self._right_padding = None
        self._configure_attention_backend(attention_backend)

    def _configure_attention_backend(self, attention_backend=None):
        backend = attention_backend
        if backend is None:
            backend = os.environ.get("MLX_LM_BATCH_ATTENTION_BACKEND", "sdpa")
        backend = backend.strip().lower()
        if backend not in {"sdpa", "bucketed"}:
            raise ValueError(
                "MLX_LM_BATCH_ATTENTION_BACKEND must be 'sdpa' or 'bucketed'"
            )
        max_tax = float(os.environ.get("MLX_LM_BUCKETED_SDPA_MAX_TAX", "1.25"))
        min_dense_tax = float(
            os.environ.get("MLX_LM_BUCKETED_SDPA_MIN_DENSE_TAX", "1.25")
        )
        lookahead = int(os.environ.get("MLX_LM_BUCKETED_SDPA_LOOKAHEAD", "32"))
        if (
            not math.isfinite(max_tax)
            or not math.isfinite(min_dense_tax)
            or max_tax < 1.0
            or (min_dense_tax < 1.0)
            or (lookahead <= 0)
        ):
            raise ValueError("invalid bucketed SDPA policy configuration")
        self.attention_backend = backend
        self._bucket_max_tax = max_tax
        self._bucket_min_dense_tax = min_dense_tax
        self._bucket_lookahead = lookahead
        self._bucket_groups: Optional[List[_BucketedAttentionGroup]] = None
        self._attention_backend_stats = {
            "attempts": 0,
            "bucketed_calls": 0,
            "dense_fallbacks": 0,
            "group_builds": 0,
            "group_invalidations": 0,
            "fallback_reasons": {},
            "last_dense_traffic_tax": 1.0,
            "last_bucketed_traffic_tax": 1.0,
            "last_group_count": 0,
        }

    def set_attention_backend(self, backend: str):
        """Select the backend for this cache group.

        Existing state remains authoritative in the dense mirror. Switching
        invalidates derived groups and rebuilds them losslessly on the next
        supported decode call.
        """
        backend = backend.strip().lower()
        if backend not in {"sdpa", "bucketed"}:
            raise ValueError("attention backend must be 'sdpa' or 'bucketed'")
        if backend != self.attention_backend:
            self._invalidate_attention_groups()
            self.attention_backend = backend

    @property
    def attention_backend_metrics(self):
        metrics = dict(self._attention_backend_stats)
        metrics["fallback_reasons"] = dict(metrics["fallback_reasons"])
        metrics["backend"] = self.attention_backend
        return metrics

    def _fallback(self, reason: str):
        stats = self._attention_backend_stats
        stats["dense_fallbacks"] += 1
        reasons = stats["fallback_reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1
        return None

    def _invalidate_attention_groups(self):
        if self._bucket_groups is not None:
            self._attention_backend_stats["group_invalidations"] += 1
            self._bucket_groups = None

    def _plan_attention_groups(self, lengths: List[int]):
        remaining = list(range(len(lengths)))
        groups = []
        while remaining:
            anchor = remaining[0]
            candidates = remaining[1 : self._bucket_lookahead]
            candidates.sort(
                key=lambda index: (
                    abs(lengths[index].bit_length() - lengths[anchor].bit_length()),
                    abs(lengths[index] - lengths[anchor]) / lengths[anchor],
                    index,
                )
            )
            chosen = [anchor]
            for index in candidates:
                proposed = chosen + [index]
                useful = sum((lengths[i] for i in proposed))
                dense = len(proposed) * max((lengths[i] for i in proposed))
                if dense / useful <= self._bucket_max_tax:
                    chosen.append(index)
            chosen_set = set(chosen)
            remaining = [index for index in remaining if index not in chosen_set]
            groups.append(chosen)
        return groups

    def _build_attention_groups(self):
        if self.keys is None or self._idx <= 0:
            return self._fallback("empty_cache")
        lengths = [int(length) for length in self.offset.tolist()]
        if len(lengths) < 2:
            return self._fallback("batch_one")
        if any((length <= 0 for length in lengths)):
            return self._fallback("non_positive_length")
        useful = sum(lengths)
        dense = len(lengths) * max(lengths)
        dense_tax = dense / useful
        self._attention_backend_stats["last_dense_traffic_tax"] = dense_tax
        if dense_tax < self._bucket_min_dense_tax:
            return self._fallback("dense_tax")
        planned = self._plan_attention_groups(lengths)
        if len(planned) <= 1:
            return self._fallback("single_group")
        dense_keys = self.keys[..., : self._idx, :]
        dense_values = self.values[..., : self._idx, :]
        groups = []
        bucketed_dense = 0
        for indices in planned:
            group_lengths = [lengths[index] for index in indices]
            width = max(group_lengths)
            left_padding = [width - length for length in group_lengths]
            index_array = mx.array(indices)
            keys = mx.contiguous(
                mx.take(dense_keys, index_array, axis=0)[..., self._idx - width :, :]
            )
            values = mx.contiguous(
                mx.take(dense_values, index_array, axis=0)[..., self._idx - width :, :]
            )
            group_cache = BatchKVCache(left_padding, attention_backend="sdpa")
            group_cache.keys = keys
            group_cache.values = values
            group_cache.offset = mx.array(group_lengths)
            group_cache.left_padding = mx.array(left_padding)
            group_cache._idx = width
            groups.append(
                _BucketedAttentionGroup(tuple(indices), index_array, group_cache)
            )
            bucketed_dense += len(indices) * width
        self._bucket_groups = groups
        stats = self._attention_backend_stats
        stats["group_builds"] += 1
        stats["last_group_count"] = len(groups)
        stats["last_bucketed_traffic_tax"] = bucketed_dense / useful
        return groups

    def update_and_fetch(self, keys, values):
        (new_keys, new_values) = (keys, values)
        prev = self._idx
        if self.keys is None or prev + keys.shape[2] > self.keys.shape[2]:
            (B, n_kv_heads, _, k_head_dim) = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                (self.keys, self.values) = (new_k, new_v)
        self.offset += keys.shape[2]
        self._idx += keys.shape[2]
        self.keys[..., prev : self._idx, :] = keys
        self.values[..., prev : self._idx, :] = values
        if self._bucket_groups is not None:
            for group in self._bucket_groups:
                group.cache.update_and_fetch(
                    mx.take(new_keys, group.index_array, axis=0),
                    mx.take(new_values, group.index_array, axis=0),
                )
        return self.keys_and_values()

    def keys_and_values(self):
        if self._idx < self.keys.shape[2]:
            return (self.keys[..., : self._idx, :], self.values[..., : self._idx, :])
        return (self.keys, self.values)

    def bucketed_attention(self, queries, scale, mask, sinks=None):
        """Run exact length-shaped decode attention when capability-gated.

        Returns ``None`` for an observable dense fallback. Only single-token
        decode on this full, unquantized batch cache is supported initially.
        """
        if self.attention_backend != "bucketed":
            return None
        stats = self._attention_backend_stats
        stats["attempts"] += 1
        if queries.shape[2] != 1:
            return self._fallback("not_decode")
        if queries.shape[0] != self.offset.shape[0]:
            return self._fallback("batch_mismatch")
        if self._right_padding is not None:
            return self._fallback("right_padding")
        if isinstance(mask, str):
            return self._fallback("string_mask")
        if mask is not None and mask.dtype != mx.bool_:
            return self._fallback("additive_mask")
        groups = self._bucket_groups or self._build_attention_groups()
        if not groups:
            return None
        outputs = []
        for group in groups:
            group_queries = mx.take(queries, group.index_array, axis=0)
            group_keys = group.cache.keys[..., : group.cache._idx, :]
            group_values = group.cache.values[..., : group.cache._idx, :]
            if mask is None:
                group_mask = None
            else:
                group_mask = mx.take(mask, group.index_array, axis=0)[
                    ..., -group.cache._idx :
                ]
            output = mx.fast.scaled_dot_product_attention(
                group_queries,
                group_keys,
                group_values,
                scale=scale,
                mask=group_mask,
                sinks=sinks,
            )
            outputs.append((group.indices, output))
        rows = [None] * queries.shape[0]
        for indices, output in outputs:
            for local_index, original_index in enumerate(indices):
                rows[original_index] = output[local_index : local_index + 1]
        if any((row is None for row in rows)):
            raise RuntimeError("bucketed attention lost a batch row")
        stats["bucketed_calls"] += 1
        return mx.concatenate(rows, axis=0)

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        self._invalidate_attention_groups()
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchKVCache"
                )
            left_padding = mx.array(left_padding)
            self._unpadded_ref = None  # in-place: identity survives
            self.left_padding += left_padding
            self.offset -= left_padding
        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)
            self._host_right_padding = (
                (self._right_padding, [int(v) for v in right_padding])
                if isinstance(right_padding, (list, tuple))
                else None
            )

    def finalize(self, *, reclaim: bool = True):
        """Turn pending right padding into left padding.

        Like ``trim_ragged``, the roll also drops the left padding every row
        then shares (``reclaim=False`` keeps the physical grid, for a caller
        holding state indexed by it): a self-MTP draft head appends unequal
        accepted spans with right padding every cycle and would otherwise
        widen by the smallest of them each time.
        """
        self._invalidate_attention_groups()
        if self._right_padding is not None:
            padding = self._right_padding
            host = getattr(self, "_host_right_padding", None)
            floor = None
            if host is not None and host[0] is padding:
                floor = [f + r for (f, r) in zip(self._padding_floor(), host[1])]
            shared = min(floor) if reclaim and floor else 0
            spec = self._finalize_reclaim_spec() if shared else None
            if spec is None or shared >= self._idx:
                shared = 0
            if shared:
                self.keys = _roll_rows_right(
                    self.keys, padding, 2, 0, self._idx, shared
                )
                self.values = _roll_rows_right(
                    self.values, padding, 2, 0, self._idx, shared
                )
                # Subclass ledgers were already rolled by their own finalize.
                for name, axis in spec:
                    ledger = getattr(self, name, None)
                    if ledger is not None:
                        cut = (slice(None),) * axis + (slice(shared, None),)
                        setattr(self, name, ledger[cut])
                self._idx -= shared
                self.left_padding = self.left_padding + mx.array(
                    [r - shared for r in host[1]]
                )
            else:
                self.keys = dynamic_roll(self.keys, padding[:, None], axis=2)
                self.values = dynamic_roll(self.values, padding[:, None], axis=2)
                self._unpadded_ref = None  # in-place: identity survives
                self.left_padding += padding
            self.offset -= padding
            if floor is not None:
                self._host_padding_floor = (
                    self.left_padding,
                    [f - shared for f in floor],
                )
            self._right_padding = None
            self._host_right_padding = None
            self._tie_row_metadata()

    def _finalize_reclaim_spec(self):
        """The ledgers to shift with a reclaim, or ``None`` to skip it."""
        try:
            return self._ragged_trim_aux_spec()
        except RaggedTrimUnsupported:
            return None

    def _tie_row_metadata(self):
        """Make evaluating K/V also evaluate the rebound per-row metadata.

        Only the layer whose mask the forward builds ever reads
        ``left_padding``. On every other full-attention layer a rebinding
        with a fresh ``shifts`` buffer each verify round would otherwise
        grow an unevaluated chain (one node and one live buffer per round)
        until the next membership change, the same live-buffer exhaustion
        ``BatchRotatingKVCache`` avoids with ``mx.depends``.
        """
        if self.keys is not None:
            self.keys = mx.depends(self.keys, (self.left_padding, self.offset))

    @property
    def state(self):
        return (self.keys, self.values, self.offset, self.left_padding)

    @state.setter
    def state(self, v):
        backend = getattr(self, "attention_backend", None)
        same_storage = getattr(self, "keys", None) is v[0]
        previous_idx = getattr(self, "_idx", None) if same_storage else None
        (self.keys, self.values, self.offset, self.left_padding) = v
        if previous_idx is None:
            # An empty cache (nothing written yet) has no keys.
            previous_idx = 0 if self.keys is None else self.keys.shape[2]
        self._idx = previous_idx
        self._right_padding = None
        self._configure_attention_backend(backend)

    @property
    def meta_state(self):
        return (str(self._idx),)

    @meta_state.setter
    def meta_state(self, v):
        if v:
            self._idx = int(v[0])

    def is_trimmable(self):
        return True

    def trim(self, n):
        self._invalidate_attention_groups()
        n = min(self._idx, n)
        self._idx -= n
        self.offset -= n
        return n

    def supports_ragged_trim(self):
        return True

    def _ragged_trim_aux_spec(self):
        """The per-row ledgers a subclass keeps beside K/V, or ``None``.

        A subclass must say so explicitly: either declare
        ``_RAGGED_TRIM_AUX_ARRAYS = ((name, axis), ...)`` (``()`` when it has
        no ledger) or override ``_trim_ragged_aux``. Inheriting silently
        would leave the ledger un-rolled and desynced from the KV, the
        failure mode of the QSA draft-cycle rewind bug.
        """
        cls = type(self)
        if cls is BatchKVCache:
            return ()
        for base in cls.__mro__:
            if base is BatchKVCache:
                break
            if "_trim_ragged_aux" in base.__dict__:
                return None
            if "_RAGGED_TRIM_AUX_ARRAYS" in base.__dict__:
                return base.__dict__["_RAGGED_TRIM_AUX_ARRAYS"]
        raise RaggedTrimUnsupported(
            f"{cls.__name__} extends BatchKVCache but declares neither _RAGGED_TRIM_AUX_ARRAYS nor _trim_ragged_aux, so a ragged trim would leave any auxiliary per-row ledger out of step with the KV"
        )

    def _check_ragged_trim_aux(self, hi: int, spec):
        """Check the declared ledgers reach the cursor, before anything moves."""
        for name, axis in spec or ():
            ledger = getattr(self, name, None)
            if ledger is not None and ledger.shape[axis] < hi:
                raise RuntimeError(
                    f"{type(self).__name__}.{name} holds {ledger.shape[axis]} positions but the cursor is at {hi}: the ledger is already out of step with the KV"
                )

    def _trim_ragged_aux(self, shifts, lo: int, hi: int, spec, reclaim: int = 0):
        """Roll the declared auxiliary ledgers with the same per-row shifts."""
        for name, axis in spec or ():
            ledger = getattr(self, name, None)
            if ledger is None:
                continue
            setattr(
                self, name, _roll_rows_right(ledger, shifts, axis, lo, hi, reclaim)
            )

    def preflight_ragged_trim(self, n, *, validate: bool = True):
        """Run every entry-local check without mutating anything.

        The whole plan is validated first because the roll is in place: a
        raise after K/V moved would leave the rows rolled under the old mask
        geometry, with ``left_padding``/``offset`` never updated.
        """
        spec = self._ragged_trim_aux_spec()
        who = f"{type(self).__name__}.trim_ragged"
        (drops, uniform, residual) = _ragged_slab_plan(self, n, who, validate)
        if residual is not None and max(residual) > 0:
            self._check_ragged_trim_aux(self._idx - uniform, spec)
        return (drops, uniform, residual, spec)

    def trim_ragged(self, n, *, validate: bool = True):
        """Rewind row ``i`` by ``n[i]`` tokens, keeping every row contiguous.

        The shared write cursor cannot move per row, so the residual rewind
        is a per-row right roll of ``[0, _idx)``: the dropped cells wrap into
        the row's own (now larger) left padding, the valid prefix stays
        contiguous and still ends at the cursor, and the next slab append
        lands after every row's live data. The uniform part of the vector is
        taken as a plain cursor move first, so an all-equal rewind is exactly
        as cheap as ``trim()`` and costs no padding.

        The same gather also drops the left padding every row now shares.
        The self-MTP cohort never calls ``filter()`` while its membership is
        stable, so padding left in place would grow by the rejection spread
        every cycle, widening the attention span and every later roll with
        columns no row reads.
        """
        (drops, uniform, residual, spec) = self.preflight_ragged_trim(
            n, validate=validate
        )
        if residual is None:
            return drops
        self._invalidate_attention_groups()
        if uniform:
            self._idx -= uniform
            self.offset -= uniform
        if max(residual) > 0:
            shifts = mx.array(residual)
            floor = [p + r for (p, r) in zip(self._padding_floor(), residual)]
            # Only processed columns are removable, and a window with no valid
            # cell left keeps the plain roll.
            reclaim = min(floor)
            if reclaim >= self._idx:
                reclaim = 0
            if self.keys is not None:
                self.keys = _roll_rows_right(
                    self.keys, shifts, 2, 0, self._idx, reclaim
                )
                self.values = _roll_rows_right(
                    self.values, shifts, 2, 0, self._idx, reclaim
                )
            self._trim_ragged_aux(shifts, 0, self._idx, spec, reclaim)
            self._idx -= reclaim
            self.left_padding = self.left_padding + (
                mx.array([r - reclaim for r in residual]) if reclaim else shifts
            )
            self._host_padding_floor = (
                self.left_padding,
                [p - reclaim for p in floor],
            )
            self.offset = self.offset - shifts
            self._tie_row_metadata()
        return drops

    def _padding_floor(self) -> List[int]:
        """A host lower bound on each row's left padding, with no device read.

        The bound is keyed by the identity of the ``left_padding`` array it
        describes. Every in-place update of that array only adds padding
        (``prepare`` and ``finalize``) and every other change rebinds it, so
        a matching identity means the bound still holds, and a mismatch
        (``filter``, ``extend``, a ``state`` restore, an external rebinding)
        falls back to zero, which is always safe. The floor then only ever
        underestimates what can be reclaimed.
        """
        rows = int(self.left_padding.shape[0])
        cached = getattr(self, "_host_padding_floor", None)
        if (
            cached is not None
            and cached[0] is self.left_padding
            and len(cached[1]) == rows
        ):
            return list(cached[1])
        return [0] * rows

    def _bind_unpadded(self, host_padding):
        """Record that the current ``left_padding`` array is all <= 0.

        Keyed by identity like ``_padding_floor``, so any rebinding (here or
        by outside code) drops the proof; the in-place updates in this class
        clear it explicitly. Only an exact host list may set it.
        """
        exact = getattr(self, "_host_padding_floor", None)
        self._unpadded_ref = (
            self.left_padding
            if exact is not None
            and exact[0] is self.left_padding
            and all(p <= 0 for p in host_padding)
            else None
        )

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        # Unpadded rows need no mask array: "causal" computes the same (keys
        # are fetched as exactly ``[:_idx]``, so MLX's kL - qL offset is the
        # diagonal) and lets head_dim-256 prefill take the causal fused
        # kernel, which skips fully masked blocks; the array mask kept the
        # Qwen3.5-9B ordinary route off it (2026-09-24). Decode (N == 1),
        # windows and explicit array requests are unchanged.
        if (
            N > 1
            and not return_array
            and kwargs.get("window_size") is None
            and getattr(self, "_unpadded_ref", None) is not None
            and self._unpadded_ref is self.left_padding
        ):
            return "causal"
        return create_causal_mask(
            N, offset=self._idx, left_padding=self.left_padding, **kwargs
        )

    def filter(self, batch_indices):
        """
        In-place filter to keep just the given indices in the cache.
        """
        self._invalidate_attention_groups()
        if self.keys is not None:
            self.keys = self.keys[batch_indices]
            self.values = self.values[batch_indices]
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]
        # A surviving row can still have padding in future prefill chunks.
        # Only columns already processed by the shared cursor are removable.
        padding = [int(v) for v in self.left_padding.tolist()]
        min_left_pad = min(min(padding), self._idx)
        if min_left_pad > 0:
            if self.keys is not None:
                self.keys = self.keys[..., min_left_pad:, :]
                self.values = self.values[..., min_left_pad:, :]
            self._idx -= min_left_pad
            self.left_padding -= min_left_pad
        self._host_padding_floor = (
            self.left_padding,
            [p - min_left_pad for p in padding],
        )
        self._bind_unpadded(self._host_padding_floor[1])

    def extend(self, other):
        """
        In-place extend this cache with the other cache.
        """
        self._invalidate_attention_groups()
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return
        max_idx = max(self._idx, other._idx)
        L1 = L2 = 0
        if self.keys is not None:
            (B, H, L1, D) = self.keys.shape
            M = self.values.shape[3]
        if other.keys is not None:
            (B, H, L2, D) = other.keys.shape
            M = other.values.shape[3]
        max_size = max(L1, L2)
        populated = self if self.keys is not None else other
        key_dtype = populated.keys.dtype
        value_dtype = populated.values.dtype

        def pad(c):
            (k, v) = (c.keys, c.values)
            if k is None:
                Bc = c.offset.shape[0]
                k = mx.zeros((Bc, H, 0, D), dtype=key_dtype)
                v = mx.zeros((Bc, H, 0, M), dtype=value_dtype)
            left = max_idx - c._idx
            right = max_size - k.shape[2] - left
            if right < 0:
                k = k[..., :right, :]
                v = v[..., :right, :]
                right = 0
            if left != 0 or right != 0:
                pad = [(0, 0), (0, 0), (left, right), (0, 0)]
                k = mx.pad(k, pad)
                v = mx.pad(v, pad)
            left_padding = c.left_padding + left
            return (k, v, c.offset, left_padding)

        (self.keys, self.values, self.offset, self.left_padding) = map(
            mx.concatenate, zip(*(pad(self), pad(other)))
        )
        self._idx = max_idx

    def extract(self, idx):
        cache = KVCache()
        # A one-token prompt feeds its only token to the first decode step, so
        # the prompt boundary is captured before anything was written here.
        if self.keys is None:
            return cache
        padding = self.left_padding[idx].item()
        end = self._idx
        if self._right_padding is not None:
            end -= int(self._right_padding[idx].item())
        cache.keys = mx.contiguous(self.keys[idx : idx + 1, :, padding:end])
        cache.values = mx.contiguous(self.values[idx : idx + 1, :, padding:end])
        cache.offset = cache.keys.shape[2]
        return cache

    @classmethod
    def merge(cls, caches):
        lengths = [c.size() for c in caches]
        max_length = max(lengths)
        if max_length == 0:
            return BatchKVCache([0] * len(caches))
        padding = [max_length - l for l in lengths]
        B = len(caches)
        H = max((c.keys.shape[1] for c in caches if c.keys is not None))
        Dk = max((c.keys.shape[3] for c in caches if c.keys is not None))
        Dv = max((c.values.shape[3] for c in caches if c.values is not None))
        dt = next(iter((c.keys.dtype for c in caches if c.keys is not None)))
        keys = mx.zeros((B, H, max_length, Dk), dtype=dt)
        values = mx.zeros((B, H, max_length, Dv), dtype=dt)
        for i, (p, c) in enumerate(zip(padding, caches)):
            if c.keys is None:
                continue
            keys[i : i + 1, :, p : p + c.offset] = c.keys[..., : c.offset, :]
            values[i : i + 1, :, p : p + c.offset] = c.values[..., : c.offset, :]
        cache = cls(padding)
        cache.keys = keys
        cache.values = values
        cache.offset += keys.shape[2]
        cache._idx = keys.shape[2]
        return cache

    def size(self):
        return self._idx

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        dense = self.keys.nbytes + self.values.nbytes
        grouped = sum((group.cache.nbytes for group in self._bucket_groups or []))
        return dense + grouped


class BatchRotatingKVCache(_BaseCache):
    step = 256

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.speculating = False
        instance._rollbacks = deque()
        instance._rollback_window = RotatingKVCache._ROLLBACK_WINDOW
        # ``from_state`` skips ``__init__``; a loaded cache has no pending
        # right padding either.
        instance._lengths = None
        return instance

    def __init__(self, max_size, left_padding: List[int]):
        self.keys = None
        self.values = None
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-l for l in left_padding])
        self.max_size = max_size
        self._idx = 0
        self._offset = 0
        self.rotated = False
        self._lengths = None

    def start_speculation(self, rollback_window: Optional[int] = None):
        self.speculating = True
        self._rollback_window = max(
            1,
            int(
                RotatingKVCache._ROLLBACK_WINDOW
                if rollback_window is None
                else rollback_window
            ),
        )
        self._rollbacks.clear()

    def stop_speculation(self):
        self.speculating = False
        self._rollbacks.clear()

    def record_rollback(self, num_tokens, keys, values):

        def copy_array(x):
            return None if x is None else mx.array(x)

        snapshot = (
            copy_array(self.keys),
            copy_array(self.values),
            mx.array(self.offset),
            mx.array(self.left_padding),
            self._offset,
            self._idx,
            self.rotated,
        )
        self._rollbacks.append((num_tokens, snapshot, mx.array(keys), mx.array(values)))
        total = sum((r[0] for r in self._rollbacks))
        while (
            len(self._rollbacks) > 1
            and total - self._rollbacks[0][0] >= self._rollback_window
        ):
            total -= self._rollbacks.popleft()[0]

    def _trim(self, trim_size, v, append=None):
        if trim_size > 0:
            v = v[..., trim_size:, :]
        if append is not None:
            return mx.concatenate([v, append], axis=2)
        return v

    def _temporal_order(self):
        """
        Rearrange the cache into temporal order.
        """
        if self.rotated:
            self.keys = mx.roll(self.keys, -self._idx, axis=2)
            self.values = mx.roll(self.values, -self._idx, axis=2)
            self._idx = self.keys.shape[2]
            self.rotated = False

    def _update_concat(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self._temporal_order()
            if self.keys.shape[2] > self._idx:
                self.keys = self.keys[..., : self._idx, :]
                self.values = self.values[..., : self._idx, :]
            if self._lengths is not None:
                roll = mx.maximum(0, self.offset - self._lengths)
                self.keys = dynamic_roll(self.keys, roll[:, None], axis=2)
                self.values = dynamic_roll(self.values, roll[:, None], axis=2)
                self.left_padding += roll
                self.offset -= roll
            trim_size = self._idx - self.max_size + 1
            if trim_size > 0:
                self.left_padding -= trim_size
            self.keys = self._trim(trim_size, self.keys, keys)
            self.values = self._trim(trim_size, self.values, values)
        self.offset += keys.shape[2]
        self._offset += keys.shape[2]
        self._idx = self.keys.shape[2]
        self.keys = mx.depends(self.keys, (self.left_padding, self.offset))
        return (self.keys, self.values)

    def _update_in_place(self, keys, values):
        if self._lengths is not None:
            raise RuntimeError(
                "finalize() should be called before deocoding with BatchRotatingKVCache"
            )
        (B, n_kv_heads, S, k_head_dim) = keys.shape
        prev = self._offset
        if self.keys is None or (
            prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            v_head_dim = values.shape[3]
            new_size = min(self.step, self.max_size - prev)
            k_shape = (B, n_kv_heads, new_size, k_head_dim)
            v_shape = (B, n_kv_heads, new_size, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                (self.keys, self.values) = (new_k, new_v)
            self._idx = prev
        trim_size = self.keys.shape[2] - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size
            self.left_padding -= trim_size
        if self._idx == self.max_size:
            self.rotated = True
            self._idx = 0
        if self.rotated:
            self.left_padding -= S
        self.keys[..., self._idx : self._idx + S, :] = keys
        self.values[..., self._idx : self._idx + S, :] = values
        self._offset += S
        self.offset += S
        self._idx += S
        self.keys = mx.depends(self.keys, (self.left_padding, self.offset))
        if self._offset < self.max_size:
            return (
                self.keys[..., : self._offset, :],
                self.values[..., : self._offset, :],
            )
        return (self.keys, self.values)

    def update_and_fetch(self, keys, values):
        if self.speculating:
            self.record_rollback(keys.shape[2], keys, values)
        if keys.shape[2] == 1:
            return self._update_in_place(keys, values)
        return self._update_concat(keys, values)

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchRotatingKVCache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= left_padding
        if right_padding is not None and max(right_padding) > 0:
            self._lengths = mx.array(lengths) + self.offset

    def finalize(self):
        if self._lengths is not None:
            roll = mx.maximum(0, self.offset - self._lengths)
            self.keys = dynamic_roll(self.keys, roll[:, None], axis=2)
            self.values = dynamic_roll(self.values, roll[:, None], axis=2)
            self.left_padding += roll
            self.offset -= roll
            self._lengths = None

    @property
    def state(self):
        (k, v) = (self.keys, self.values)
        # An empty cache reports ``None`` keys and values, as KVCache does.
        if k is not None and self._offset < k.shape[2]:
            (k, v) = (k[..., : self._offset, :], v[..., : self._offset, :])
        return (k, v, self.offset, self.left_padding)

    @state.setter
    def state(self, v):
        (self.keys, self.values, self.offset, self.left_padding) = v

    @property
    def meta_state(self):
        return tuple(
            map(str, (self.max_size, self._offset, self._idx, int(self.rotated)))
        )

    @meta_state.setter
    def meta_state(self, v):
        (self.max_size, self._offset, self._idx) = map(int, v[:3])
        self.rotated = str(v[3]) in ("True", "true", "1")

    def is_trimmable(self):
        return self._offset < self.max_size

    def trim(self, n):
        n = min(self._offset, n)
        self._offset -= n
        self._idx -= n
        self.offset -= n
        return n

    def trim_ragged(self, n, *, validate: bool = True):
        raise RaggedTrimUnsupported(
            "BatchRotatingKVCache cannot rewind rows by different amounts: the ring shares one eviction cursor (``_idx``/``rotated``) and its rollback records are whole-batch snapshots, so a per-row rewind cannot restore what the shared window already evicted. The >=32K windowed MTP profile is admission-gated out of batch composition for this reason"
        )

    def to_quantized(
        self, group_size: int = 64, bits: int = 4
    ) -> "BatchRotatingQuantizedKVCache":
        quant_cache = BatchRotatingQuantizedKVCache(
            self.max_size, self.left_padding.tolist(), group_size=group_size, bits=bits
        )
        quant_cache.offset = self.offset
        quant_cache._idx = self._idx
        quant_cache._offset = self._offset
        quant_cache.rotated = self.rotated
        if self.keys is not None:
            quant_cache.keys = mx.quantize(self.keys, group_size=group_size, bits=bits)
            quant_cache.values = mx.quantize(
                self.values, group_size=group_size, bits=bits
            )
        return quant_cache

    def make_mask(
        self, N: int, window_size: Optional[int] = None, return_array: bool = False
    ):
        left_padding = self.left_padding
        window_size = window_size or self.max_size
        offset = min(self.max_size - 1, self._offset)
        rinds = mx.arange(offset + N)
        linds = mx.arange(offset, offset + N) if offset else rinds
        linds = linds[:, None]
        rinds = rinds[None]
        mask = linds >= rinds
        mask &= linds < rinds + window_size
        if (trim_size := (self._idx - self.max_size + int(N > 1))) > 0:
            left_padding = left_padding - trim_size
        rotated = N == 1 and (self.rotated or self._idx >= self.max_size)
        if rotated:
            left_padding = left_padding - 1
        mask = mask & (rinds >= mx.expand_dims(left_padding, (1, 2, 3)))
        if rotated:
            idx = self._idx
            if idx >= self.max_size:
                idx = 0
            mask = mx.roll(mask, shift=idx + 1, axis=-1)
        return mask

    def filter(self, batch_indices):
        """
        In-place filter to keep just the given indices in the cache.
        """
        self._rollbacks.clear()
        if self.keys is not None:
            self.keys = self.keys[batch_indices]
            self.values = self.values[batch_indices]
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

    def extend(self, other):
        """
        In-place extend this cache with the other cache.
        """
        self._rollbacks.clear()
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return
        if self.rotated != other.rotated or self._idx != other._idx:
            self._temporal_order()
            other._temporal_order()
        max_idx = max(self._idx, other._idx)
        L1 = L2 = 0
        if self.keys is not None:
            (B, H, L1, D) = self.keys.shape
            M = self.values.shape[3]
        if other.keys is not None:
            (B, H, L2, D) = other.keys.shape
            M = other.values.shape[3]
        max_size = max(L1, L2)
        populated = self if self.keys is not None else other
        key_dtype = populated.keys.dtype
        value_dtype = populated.values.dtype

        def pad(c):
            left = max_idx - c._idx
            (k, v) = (c.keys, c.values)
            if k is None:
                Bc = c.offset.shape[0]
                k = mx.zeros((Bc, H, 0, D), dtype=key_dtype)
                v = mx.zeros((Bc, H, 0, M), dtype=value_dtype)
            right = max_size - k.shape[2] - left
            if right < 0:
                k = k[..., :right, :]
                v = v[..., :right, :]
                right = 0
            if left != 0 or right != 0:
                pad = [(0, 0), (0, 0), (left, right), (0, 0)]
                k = mx.pad(k, pad)
                v = mx.pad(v, pad)
            left_padding = c.left_padding + left
            return (k, v, c.offset, left_padding)

        (self.keys, self.values, self.offset, self.left_padding) = map(
            mx.concatenate, zip(*(pad(self), pad(other)))
        )
        self._idx = max_idx
        self._offset = max(self._offset, other._offset)

    def _extract_rotating_state(
        self, idx, keys, values, offset, left_padding, _offset, cache_idx, rotated
    ):
        cache = RotatingKVCache(self.max_size)
        if keys is None:
            cache.offset = int(offset[idx].item())
            return cache
        mx.eval(left_padding, offset)
        padding = max(0, left_padding.tolist()[idx])
        cache.keys = keys[idx : idx + 1]
        cache.values = values[idx : idx + 1]
        cache._idx = cache_idx
        if rotated:
            cache.keys = mx.roll(cache.keys, -cache_idx, axis=2)
            cache.values = mx.roll(cache.values, -cache_idx, axis=2)
            cache._idx = self.max_size
        cache.keys = mx.contiguous(cache.keys[:, :, padding : cache._idx])
        cache.values = mx.contiguous(cache.values[:, :, padding : cache._idx])
        cache.offset = int(offset[idx].item())
        cache._idx = cache.keys.shape[2]
        return cache

    def extract(self, idx):
        cache = self._extract_rotating_state(
            idx,
            self.keys,
            self.values,
            self.offset,
            self.left_padding,
            self._offset,
            self._idx,
            self.rotated,
        )
        if self._lengths is not None and cache.keys is not None:
            pad = max(0, int((self.offset - self._lengths).tolist()[idx]))
            if pad:
                cache.keys = mx.contiguous(cache.keys[:, :, :-pad])
                cache.values = mx.contiguous(cache.values[:, :, :-pad])
                cache.offset -= pad
                cache._idx = cache.keys.shape[2]
        if self.speculating and self._rollbacks:
            cache.start_speculation(self._rollback_window)
            for num_tokens, snap, keys, values in self._rollbacks:
                snap_cache = self._extract_rotating_state(idx, *snap)
                cache._rollbacks.append(
                    (
                        num_tokens,
                        [
                            snap_cache.keys,
                            snap_cache.values,
                            snap_cache._idx,
                            snap_cache.offset,
                        ],
                        mx.contiguous(keys[idx : idx + 1]),
                        mx.contiguous(values[idx : idx + 1]),
                    )
                )
            # The row keeps its history so it stays trimmable, but each record
            # is a lazy slice of a whole-batch snapshot.  Callers schedule only
            # the row's ``state``, so schedule the history copies here, where
            # they are cut; otherwise the row pins every batch row.
            mx.async_eval(list(cache._rollbacks))
        return cache

    @classmethod
    def merge(cls, caches):
        if not all((c.max_size == caches[0].max_size for c in caches)):
            raise ValueError(
                "BatchRotatingKVCache can only merge caches with the same maximum size"
            )
        offsets = [c.offset for c in caches]
        lengths = [c.size() for c in caches]
        max_length = max(lengths)
        if max_length == 0:
            return cls(caches[0].max_size, [0] * len(caches))
        padding = [max_length - l for l in lengths]
        B = len(caches)
        H = max((c.keys.shape[1] for c in caches if c.keys is not None))
        Dk = max((c.keys.shape[3] for c in caches if c.keys is not None))
        Dv = max((c.values.shape[3] for c in caches if c.values is not None))
        dt = next(iter((c.keys.dtype for c in caches if c.keys is not None)))
        keys = mx.zeros((B, H, max_length, Dk), dtype=dt)
        values = mx.zeros((B, H, max_length, Dv), dtype=dt)
        for i, (p, l, c) in enumerate(zip(padding, lengths, caches)):
            if c.keys is None:
                continue
            keys[i : i + 1, :, p : p + l] = c._temporal_order(c.keys)[..., -l:, :]
            values[i : i + 1, :, p : p + l] = c._temporal_order(c.values)[..., -l:, :]
        cache = cls(caches[0].max_size, padding)
        cache.keys = keys
        cache.values = values
        cache.offset = mx.array(offsets)
        cache._idx = keys.shape[2]
        cache._offset = keys.shape[2]
        return cache

    def size(self):
        return min(self._offset, self.max_size)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class BatchRotatingQuantizedKVCache(_BaseCache):
    """Quantized counterpart of BatchRotatingKVCache — the class BatchGenerator's
    continuous-batching path actually calls `update_and_fetch` on once per-job
    caches are merged into a batch. `keys`/`values` are each a (packed, scales,
    biases) triple. Never receives `keep` tokens — merge() only ever takes
    RotatingQuantizedKVCache inputs, which themselves reject keep>0."""

    step = 256

    def __init__(
        self, max_size, left_padding: List[int], group_size: int = 64, bits: int = 4
    ):
        self.keys = None
        self.values = None
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-l for l in left_padding])
        self.max_size = max_size
        self.group_size = group_size
        self.bits = bits
        self._idx = 0
        self._offset = 0
        self.rotated = False
        self._lengths = None

    def _quantize(self, x):
        return mx.quantize(x, group_size=self.group_size, bits=self.bits)

    def _trim(self, trim_size, v, append=None):
        if trim_size > 0:
            v = tree_map(lambda a: a[..., trim_size:, :], v)
        if append is not None:
            v = tree_map(lambda a, b: mx.concatenate([a, b], axis=2), v, append)
        return v

    def _temporal_order(self):
        """
        Rearrange the cache into temporal order.
        """
        if self.rotated:
            self.keys = tree_map(lambda a: mx.roll(a, -self._idx, axis=2), self.keys)
            self.values = tree_map(
                lambda a: mx.roll(a, -self._idx, axis=2), self.values
            )
            self._idx = self.keys[0].shape[2]
            self.rotated = False

    def _update_concat(self, keys, values):
        qkeys = self._quantize(keys)
        qvalues = self._quantize(values)
        if self.keys is None:
            self.keys = qkeys
            self.values = qvalues
        else:
            self._temporal_order()
            if self.keys[0].shape[2] > self._idx:
                self.keys = tree_map(lambda a: a[..., : self._idx, :], self.keys)
                self.values = tree_map(lambda a: a[..., : self._idx, :], self.values)
            if self._lengths is not None:
                roll = mx.maximum(0, self.offset - self._lengths)
                self.keys = tree_map(
                    lambda a: dynamic_roll(a, roll[:, None], axis=2), self.keys
                )
                self.values = tree_map(
                    lambda a: dynamic_roll(a, roll[:, None], axis=2), self.values
                )
                self.left_padding += roll
                self.offset -= roll
            trim_size = self._idx - self.max_size + 1
            if trim_size > 0:
                self.left_padding -= trim_size
            self.keys = self._trim(trim_size, self.keys, qkeys)
            self.values = self._trim(trim_size, self.values, qvalues)
        self.offset += keys.shape[2]
        self._offset += keys.shape[2]
        self._idx = self.keys[0].shape[2]
        self.keys = tree_map(
            lambda a: mx.depends(a, (self.left_padding, self.offset)), self.keys
        )
        return (self.keys, self.values)

    def _update_in_place(self, keys, values):
        if self._lengths is not None:
            raise RuntimeError(
                "finalize() should be called before decoding with BatchRotatingQuantizedKVCache"
            )
        (B, n_kv_heads, S, k_head_dim) = keys.shape
        v_head_dim = values.shape[3]
        prev = self._offset
        cur_size = self.keys[0].shape[2] if self.keys is not None else 0
        if self.keys is None or (prev >= cur_size and cur_size < self.max_size):
            new_size = min(self.step, self.max_size - prev)
            new_k = _empty_quantized(
                B,
                n_kv_heads,
                new_size,
                k_head_dim,
                self.group_size,
                self.bits,
                keys.dtype,
            )
            new_v = _empty_quantized(
                B,
                n_kv_heads,
                new_size,
                v_head_dim,
                self.group_size,
                self.bits,
                values.dtype,
            )
            if self.keys is not None:
                self.keys = tree_map(
                    lambda a, b: mx.concatenate([a, b], axis=2), self.keys, new_k
                )
                self.values = tree_map(
                    lambda a, b: mx.concatenate([a, b], axis=2), self.values, new_v
                )
            else:
                (self.keys, self.values) = (new_k, new_v)
            self._idx = prev
        trim_size = self.keys[0].shape[2] - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size
            self.left_padding -= trim_size
        if self._idx == self.max_size:
            self.rotated = True
            self._idx = 0
        if self.rotated:
            self.left_padding -= S
        qkeys = self._quantize(keys)
        qvalues = self._quantize(values)
        for i in range(3):
            self.keys[i][..., self._idx : self._idx + S, :] = qkeys[i]
            self.values[i][..., self._idx : self._idx + S, :] = qvalues[i]
        self._offset += S
        self.offset += S
        self._idx += S
        self.keys = tree_map(
            lambda a: mx.depends(a, (self.left_padding, self.offset)), self.keys
        )
        if self._offset < self.max_size:
            return (
                tree_map(lambda a: a[..., : self._offset, :], self.keys),
                tree_map(lambda a: a[..., : self._offset, :], self.values),
            )
        return (self.keys, self.values)

    def update_and_fetch(self, keys, values):
        if keys.shape[2] == 1:
            return self._update_in_place(keys, values)
        return self._update_concat(keys, values)

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchRotatingQuantizedKVCache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= left_padding
        if right_padding is not None and max(right_padding) > 0:
            self._lengths = mx.array(lengths) + self.offset

    def finalize(self):
        if self._lengths is not None:
            roll = mx.maximum(0, self.offset - self._lengths)
            self.keys = tree_map(
                lambda a: dynamic_roll(a, roll[:, None], axis=2), self.keys
            )
            self.values = tree_map(
                lambda a: dynamic_roll(a, roll[:, None], axis=2), self.values
            )
            self.left_padding += roll
            self.offset -= roll
            self._lengths = None

    @property
    def state(self):
        (k, v) = (self.keys, self.values)
        if self._offset < k[0].shape[2]:
            k = tree_map(lambda a: a[..., : self._offset, :], k)
            v = tree_map(lambda a: a[..., : self._offset, :], v)
        return (k, v, self.offset, self.left_padding)

    @state.setter
    def state(self, v):
        (self.keys, self.values, self.offset, self.left_padding) = v

    @property
    def meta_state(self):
        return tuple(
            map(
                str,
                (
                    self.max_size,
                    self._offset,
                    self._idx,
                    int(self.rotated),
                    self.group_size,
                    self.bits,
                ),
            )
        )

    @meta_state.setter
    def meta_state(self, v):
        (self.max_size, self._offset, self._idx) = map(int, v[:3])
        self.rotated = str(v[3]) in ("True", "true", "1")
        (self.group_size, self.bits) = map(int, v[4:6])

    def is_trimmable(self):
        return self._offset < self.max_size

    def trim(self, n):
        n = min(self._offset, n)
        self._offset -= n
        self._idx -= n
        self.offset -= n
        return n

    def trim_ragged(self, n, *, validate: bool = True):
        raise RaggedTrimUnsupported(
            "BatchRotatingQuantizedKVCache cannot rewind rows by different amounts: it shares BatchRotatingKVCache's single ring cursor"
        )

    def make_mask(
        self, N: int, window_size: Optional[int] = None, return_array: bool = False
    ):
        left_padding = self.left_padding
        window_size = window_size or self.max_size
        offset = min(self.max_size - 1, self._offset)
        rinds = mx.arange(offset + N)
        linds = mx.arange(offset, offset + N) if offset else rinds
        linds = linds[:, None]
        rinds = rinds[None]
        mask = linds >= rinds
        mask &= linds < rinds + window_size
        if (trim_size := (self._idx - self.max_size + int(N > 1))) > 0:
            left_padding = left_padding - trim_size
        rotated = N == 1 and (self.rotated or self._idx >= self.max_size)
        if rotated:
            left_padding = left_padding - 1
        mask = mask & (rinds >= mx.expand_dims(left_padding, (1, 2, 3)))
        if rotated:
            idx = self._idx
            if idx >= self.max_size:
                idx = 0
            mask = mx.roll(mask, shift=idx + 1, axis=-1)
        return mask

    def filter(self, batch_indices):
        """
        In-place filter to keep just the given indices in the cache.
        """
        if self.keys is not None:
            self.keys = tree_map(lambda a: a[batch_indices], self.keys)
            self.values = tree_map(lambda a: a[batch_indices], self.values)
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

    def extend(self, other):
        """
        In-place extend this cache with the other cache.
        """
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return
        if self.rotated != other.rotated or self._idx != other._idx:
            self._temporal_order()
            other._temporal_order()
        max_idx = max(self._idx, other._idx)
        L1 = L2 = 0
        if self.keys is not None:
            L1 = self.keys[0].shape[2]
        if other.keys is not None:
            L2 = other.keys[0].shape[2]
        max_size = max(L1, L2)
        ref = self.keys if self.keys is not None else other.keys
        ref_v = self.values if self.values is not None else other.values

        def pad(c):
            left = max_idx - c._idx
            if c.keys is None:
                Bc = c.offset.shape[0]
                k = tree_map(
                    lambda a: mx.zeros(
                        (Bc, *a.shape[1:2], 0, a.shape[-1]), dtype=a.dtype
                    ),
                    ref,
                )
                v = tree_map(
                    lambda a: mx.zeros(
                        (Bc, *a.shape[1:2], 0, a.shape[-1]), dtype=a.dtype
                    ),
                    ref_v,
                )
            else:
                (k, v) = (c.keys, c.values)
            right = max_size - k[0].shape[2] - left
            if right < 0:
                k = tree_map(lambda a: a[..., :right, :], k)
                v = tree_map(lambda a: a[..., :right, :], v)
                right = 0
            if left != 0 or right != 0:
                k = tree_map(
                    lambda a: mx.pad(a, [(0, 0), (0, 0), (left, right), (0, 0)]), k
                )
                v = tree_map(
                    lambda a: mx.pad(a, [(0, 0), (0, 0), (left, right), (0, 0)]), v
                )
            left_padding = c.left_padding + left
            return (k, v, c.offset, left_padding)

        (pa, pb) = (pad(self), pad(other))
        self.keys = tree_map(lambda a, b: mx.concatenate([a, b], axis=0), pa[0], pb[0])
        self.values = tree_map(
            lambda a, b: mx.concatenate([a, b], axis=0), pa[1], pb[1]
        )
        self.offset = mx.concatenate([pa[2], pb[2]])
        self.left_padding = mx.concatenate([pa[3], pb[3]])
        self._idx = max_idx
        self._offset = max(self._offset, other._offset)

    def extract(self, idx):
        mx.eval(self.left_padding, self.offset)
        cache = RotatingQuantizedKVCache(
            self.max_size, group_size=self.group_size, bits=self.bits
        )
        if self.keys is None:
            return cache
        padding = max(0, self.left_padding.tolist()[idx])
        offset = self.offset.tolist()[idx]
        cache.keys = tree_map(lambda a: a[idx : idx + 1], self.keys)
        cache.values = tree_map(lambda a: a[idx : idx + 1], self.values)
        cache._idx = self._idx
        if self.rotated:
            cache.keys = tree_map(lambda a: mx.roll(a, -self._idx, axis=2), cache.keys)
            cache.values = tree_map(
                lambda a: mx.roll(a, -self._idx, axis=2), cache.values
            )
            cache._idx = self.max_size
        cache.keys = tree_map(
            lambda a: mx.contiguous(a[:, :, padding : cache._idx]), cache.keys
        )
        cache.values = tree_map(
            lambda a: mx.contiguous(a[:, :, padding : cache._idx]), cache.values
        )
        cache.offset = offset
        cache._idx = cache.keys[0].shape[2]
        if self._lengths is not None:
            pad = max(0, int((self.offset - self._lengths).tolist()[idx]))
            if pad:
                cache.keys = tree_map(
                    lambda a: mx.contiguous(a[:, :, :-pad]), cache.keys
                )
                cache.values = tree_map(
                    lambda a: mx.contiguous(a[:, :, :-pad]), cache.values
                )
                cache.offset -= pad
                cache._idx = cache.keys[0].shape[2]
        return cache

    @classmethod
    def merge(cls, caches):
        if not all((c.max_size == caches[0].max_size for c in caches)):
            raise ValueError(
                "BatchRotatingQuantizedKVCache can only merge caches with the same maximum size"
            )
        if not all(
            (
                (c.group_size, c.bits) == (caches[0].group_size, caches[0].bits)
                for c in caches
            )
        ):
            raise ValueError(
                "BatchRotatingQuantizedKVCache can only merge caches with the same group_size/bits"
            )
        offsets = [c.offset for c in caches]
        lengths = [c.size() for c in caches]
        max_length = max(lengths)
        if max_length == 0:
            return cls(
                caches[0].max_size,
                [0] * len(caches),
                group_size=caches[0].group_size,
                bits=caches[0].bits,
            )
        padding = [max_length - l for l in lengths]
        B = len(caches)
        ref = next((c.keys for c in caches if c.keys is not None))
        H = ref[0].shape[1]
        def alloc_like(ref_tuple):
            return tuple(
                (
                    mx.zeros((B, H, max_length, a.shape[-1]), dtype=a.dtype)
                    for a in ref_tuple
                )
            )

        keys = alloc_like(ref)
        values = alloc_like(next((c.values for c in caches if c.values is not None)))
        for i, (p, l, c) in enumerate(zip(padding, lengths, caches)):
            if c.keys is None:
                continue
            ok = c._temporal_order(c.keys)
            ov = c._temporal_order(c.values)
            for j in range(3):
                keys[j][i : i + 1, :, p : p + l] = ok[j][..., -l:, :]
                values[j][i : i + 1, :, p : p + l] = ov[j][..., -l:, :]
        cache = cls(
            caches[0].max_size,
            padding,
            group_size=caches[0].group_size,
            bits=caches[0].bits,
        )
        cache.keys = keys
        cache.values = values
        cache.offset = mx.array(offsets)
        cache._idx = max_length
        cache._offset = max_length
        return cache

    def size(self):
        return min(self._offset, self.max_size)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return tree_reduce(lambda a, x: a + x.nbytes, (self.keys, self.values), 0)


class TokenBuffer:
    """A simple token buffer that can be efficiently appended to in a similar
    fashion to the KVCache.

    Perhaps these could share some logic in the future.
    """

    step = 256

    def __init__(self, tokens=[]):
        self._buffer = mx.array(tokens, dtype=mx.int32)
        self._size = len(tokens)

    def update_and_fetch(self, tokens):
        start = self._size
        end = start + len(tokens)
        new_size = (end + self.step - 1) // self.step * self.step
        if new_size > self._buffer.size:
            self._buffer = mx.concatenate(
                [self._buffer, mx.zeros(new_size - self._buffer.size, dtype=mx.int32)]
            )
        self._buffer[start:end] = tokens
        self._size = end
        return self._buffer[:end]

    @property
    def state(self):
        return self._buffer

    @property
    def tokens(self):
        return self._buffer[: self._size]


@dataclass
class PromptTrieResult:
    model: Any
    exact: Optional[List[int]]
    shorter: Optional[List[int]]
    longer: Optional[List[int]]
    common_prefix: int


class PromptTrie:
    def __init__(self):
        self._trie = {}

    def add(self, model: Any, tokens: List[int], value: Any):
        if model not in self._trie:
            self._trie[model] = {}
        current = self._trie[model]
        for tok in tokens:
            if tok not in current:
                current[tok] = {}
            current = current[tok]
        prev = current.get("__value__", None)
        current["__value__"] = value
        return prev

    def get(self, model: Any, tokens: List[int]):
        current = self._trie[model]
        for tok in tokens:
            current = current[tok]
        return current["__value__"]

    def pop(self, model: Any, tokens: List[int]):
        path = [self._trie[model]]
        for tok in tokens:
            path.append(path[-1][tok])
        value = path[-1].pop("__value__")
        for i in range(len(tokens), 0, -1):
            node = path[i]
            parent = path[i - 1]
            tok = tokens[i - 1]
            if len(node) > 0:
                break
            del parent[tok]
        return value

    def pop_prefixes(self, model: Any, tokens: List[int], predicate=None):
        values = []
        current = self._trie[model]
        for i, tok in enumerate(tokens):
            if "__value__" in current and (
                predicate is None or predicate(i, current["__value__"])
            ):
                values.append((i, current.pop("__value__")))
            current = current[tok]
        return values

    def search(self, model: Any, tokens: List[int]) -> PromptTrieResult:
        if model not in self._trie:
            return PromptTrieResult(model, None, None, None, 0)
        current = self._trie[model]
        if not tokens and "__value__" in current:
            return PromptTrieResult(model, [], None, None, 0)
        last_index = -1
        index = 0
        while index < len(tokens) and tokens[index] in current:
            current = current[tokens[index]]
            if "__value__" in current:
                last_index = index
            index += 1
        if last_index == len(tokens) - 1 >= 0:
            return PromptTrieResult(model, tokens, None, None, 0)
        shorter = None
        # A one-token stored prefix (the committed boundary of a two-token
        # prompt) is a shorter match too.  Reporting it only as a "longer"
        # path made untrimmable hybrid caches miss it.
        if last_index >= 0:
            shorter = tokens[: last_index + 1]
        longer = None
        common_prefix = index
        if index > 0:
            best = None
            path = []
            pop_path = object()
            stack = [(current, None, False)]
            while stack:
                item = stack.pop()
                if item is pop_path:
                    path.pop()
                    continue
                (current, tok, append_token) = item
                if append_token:
                    path.append(tok)
                if "__value__" in current:
                    if best is None or len(path) < len(best):
                        best = list(path)
                elif best is None or len(path) < len(best):
                    for tok in current:
                        stack.append(pop_path)
                        stack.append((current[tok], tok, True))
            longer = tokens[:index] + best
        return PromptTrieResult(model, None, shorter, longer, common_prefix)


def _mark_prompt_cache_restored(prompt_cache):
    stack = list(prompt_cache)
    while stack:
        entry = stack.pop()
        if isinstance(entry, CacheList):
            stack.extend(entry.caches)
            continue
        if isinstance(entry, (list, tuple)):
            stack.extend(entry)
            continue
        if (
            hasattr(entry, "_qsa_summary_restored")
            and getattr(entry, "_qsa_pooled_keys", None) is not None
        ):
            entry._qsa_summary_restored = True


def _copy_prompt_cache_for_restore(prompt_cache):
    if not getattr(prompt_cache, "_cow_frozen", False):
        restored = copy.deepcopy(prompt_cache)
        _mark_prompt_cache_restored(restored)
        return restored
    try:
        from ..cow_cache import (
            COWCacheStale,
            COWFrozenPromptCache,
            record_fallback_deepcopy,
            restore_prompt_cache,
        )
    except ImportError:
        COWFrozenPromptCache = ()
    if isinstance(prompt_cache, COWFrozenPromptCache):
        started = __import__("time").perf_counter_ns()
        telemetry = prompt_cache.cow_owner.telemetry
        try:
            restored = restore_prompt_cache(prompt_cache)
        except COWCacheStale:
            raise
        except Exception:
            restored = copy.deepcopy(list(prompt_cache))
            record_fallback_deepcopy(
                prompt_cache, telemetry, __import__("time").perf_counter_ns() - started
            )
        _mark_prompt_cache_restored(restored)
        return restored
    restored = copy.deepcopy(prompt_cache)
    _mark_prompt_cache_restored(restored)
    return restored


class PrefixIndex:
    @dataclass
    class CacheEntry:
        prompt_cache: List[Any]
        nbytes: int
        cache_type: str
        sidecar: Any = None

    class CacheOrder:
        def __init__(self, ordering: List[str] = ["assistant", "user", "system"]):
            self._ordering = ordering
            self._lrus = {k: deque() for k in ordering}

        def __len__(self):
            return sum((len(lru) for lru in self._lrus.values()))

        def push(self, model: Any, tokens: List[Any], cache_type: str = "assistant"):
            self._lrus[cache_type].append((model, tokens))

        def remove(self, model: Any, tokens: List[Any]):
            for cache_type in self._ordering:
                try:
                    self._lrus[cache_type].remove((model, tokens))
                    break
                except ValueError:
                    pass

        def pop(self):
            i = 0
            while i + 1 < len(self._ordering):
                lru_a = self._lrus[self._ordering[i]]
                lru_b = self._lrus[self._ordering[i + 1]]
                if lru_a and len(lru_a) >= len(lru_b):
                    return lru_a.popleft()
                i += 1
            return lru_b.popleft()

    def __init__(
        self,
        max_size: int = 10,
        max_bytes: int = 1 << 63,
        max_tokens: Optional[int] = None,
    ):
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.max_size = max_size
        self.max_bytes = max_bytes
        self.max_tokens = max_tokens
        self.overlength_rejections = 0
        self._trie = PromptTrie()
        self._lru = PrefixIndex.CacheOrder()
        self._n_bytes = 0
        self._n_bytes_by_type = {k: 0 for k in self._lru._ordering}

    def __len__(self):
        return len(self._lru)

    @property
    def nbytes(self):
        return self._n_bytes

    @property
    def max_entry_tokens(self):
        return max(
            (len(tokens) for lru in self._lru._lrus.values() for (_, tokens) in lru),
            default=0,
        )

    @staticmethod
    def _exact_entry_serves(cache_entry, tokens: List[int]) -> bool:
        """Whether an exact entry can land strictly inside its own prompt."""
        if can_trim_prompt_cache(cache_entry.prompt_cache):
            return True
        landing = achievable_trim(cache_entry.prompt_cache, 1)
        return landing is not None and landing[1] < len(tokens)

    def _proper_prefix_result(self, model: Any, tokens: List[int]):
        """The deepest stored proper prefix of ``tokens``, as a shorter match.

        ``PromptTrie.search`` reports no shorter prefix alongside an exact
        match, so an exact entry that cannot serve its own prompt (an
        untrimmable hybrid cache) would otherwise hide every shorter one.
        """
        nearest = self._trie.search(model, tokens[:-1])
        shorter = nearest.exact if nearest.exact is not None else nearest.shorter
        if not shorter:
            # Match ``search``, which never reports an empty shorter prefix.
            shorter = None
        return PromptTrieResult(model, None, shorter, None, 0)

    def fetch_nearest_cache(self, model: Any, tokens: List[int]):
        result = self._trie.search(model, tokens)
        if result.exact is not None and len(tokens) == 0:
            cache_entry = self._trie.get(result.model, result.exact)
            return (_copy_prompt_cache_for_restore(cache_entry.prompt_cache), [])
        if result.exact is not None:
            cache_entry = self._trie.get(result.model, result.exact)
            if can_trim_prompt_cache(cache_entry.prompt_cache):
                cache = _copy_prompt_cache_for_restore(cache_entry.prompt_cache)
                trim_prompt_cache(cache, 1)
                return (cache, tokens[-1:])
            landing = achievable_trim(cache_entry.prompt_cache, 1)
            if landing is not None and landing[1] < len(tokens):
                cache = _copy_prompt_cache_for_restore(cache_entry.prompt_cache)
                trimmed = trim_prompt_cache(cache, 1, allow_partial=True)
                landed = len(tokens) - trimmed
                if 0 < landed <= len(tokens) - 1:
                    return (cache, tokens[landed:])
            result = self._proper_prefix_result(model, tokens)
        short_length = len(result.shorter) if result.shorter is not None else 0
        if result.longer is not None and result.common_prefix > short_length:
            cache_entry = self._trie.get(result.model, result.longer)
            prefix = min(len(tokens) - 1, result.common_prefix)
            num_to_trim = len(result.longer) - prefix
            if can_trim_prompt_cache(cache_entry.prompt_cache):
                cache = _copy_prompt_cache_for_restore(cache_entry.prompt_cache)
                trim_prompt_cache(cache, num_to_trim)
                return (cache, tokens[prefix:])
            landing = achievable_trim(cache_entry.prompt_cache, num_to_trim)
            if landing is not None:
                landed = len(result.longer) - landing[1]
                if 0 < landed <= len(tokens) - 1 and landed > short_length:
                    cache = _copy_prompt_cache_for_restore(cache_entry.prompt_cache)
                    trimmed = trim_prompt_cache(cache, num_to_trim, allow_partial=True)
                    landed = len(result.longer) - trimmed
                    return (cache, tokens[landed:])
        if short_length > 0:
            cache_entry = self._trie.get(result.model, result.shorter)
            return (
                _copy_prompt_cache_for_restore(cache_entry.prompt_cache),
                tokens[short_length:],
            )
        return (None, tokens)

    def insert_cache(
        self,
        model: Any,
        tokens: List[int],
        prompt_cache: List[Any],
        *,
        cache_type: str = "assistant",
        sidecar: Any = None,
        prune_prefixes=True,
        removed_entries=None,
    ):
        """Insert a cache and optionally report every entry removed by it.

        ``removed_entries`` receives ``(reason, model, tokens, entry)`` after
        each trie removal.  APCv2 uses these exact mutations to release owners
        without scanning the whole trie before and after publication.
        """
        if self.max_tokens is not None and len(tokens) > self.max_tokens:
            self.overlength_rejections += 1
            return False
        sidecar_nbytes = int(getattr(sidecar, "nbytes", 0))
        entry = PrefixIndex.CacheEntry(
            prompt_cache,
            sum((c.nbytes for c in prompt_cache)) + sidecar_nbytes,
            cache_type,
            sidecar,
        )
        self._n_bytes += entry.nbytes
        self._n_bytes_by_type[cache_type] += entry.nbytes
        prev = self._trie.add(model, tokens, entry)
        if prev is not None:
            self._n_bytes -= prev.nbytes
            self._n_bytes_by_type[prev.cache_type] -= prev.nbytes
            self._lru.remove(model, tokens)
            if removed_entries is not None:
                removed_entries.append(("replaced", model, tokens, prev))
        self._lru.push(model, tokens, cache_type)
        if prune_prefixes and can_trim_prompt_cache(prompt_cache) and sidecar is None:
            for prefix_len, entry in self._trie.pop_prefixes(
                model, tokens,
                predicate=prune_prefixes if callable(prune_prefixes) else None,
            ):
                self._n_bytes -= entry.nbytes
                self._n_bytes_by_type[entry.cache_type] -= entry.nbytes
                self._lru.remove(model, tokens[:prefix_len])
                if removed_entries is not None:
                    removed_entries.append(("subsumed", model, tokens[:prefix_len], entry))
        if len(self._lru) > self.max_size:
            (model, tokens) = self._lru.pop()
            entry = self._trie.pop(model, tokens)
            self._n_bytes -= entry.nbytes
            self._n_bytes_by_type[entry.cache_type] -= entry.nbytes
            if removed_entries is not None:
                removed_entries.append(("size_limit", model, tokens, entry))
        while self._n_bytes > self.max_bytes:
            (model, tokens) = self._lru.pop()
            entry = self._trie.pop(model, tokens)
            self._n_bytes -= entry.nbytes
            self._n_bytes_by_type[entry.cache_type] -= entry.nbytes
            if removed_entries is not None:
                removed_entries.append(("byte_limit", model, tokens, entry))
        return True

    def trim_to(
        self, *, n_sequences: Optional[int] = None, n_bytes: Optional[int] = None
    ):
        n_sequences = max(0, n_sequences) if n_sequences is not None else 1 << 63
        n_bytes = max(0, n_bytes) if n_bytes is not None else 1 << 63
        while len(self._lru) > n_sequences:
            (model, tokens) = self._lru.pop()
            entry = self._trie.pop(model, tokens)
            self._n_bytes -= entry.nbytes
            self._n_bytes_by_type[entry.cache_type] -= entry.nbytes
        while self._n_bytes > n_bytes:
            (model, tokens) = self._lru.pop()
            entry = self._trie.pop(model, tokens)
            self._n_bytes -= entry.nbytes
            self._n_bytes_by_type[entry.cache_type] -= entry.nbytes

    def stats_by_type(self):
        result = {}
        for cache_type in self._lru._ordering:
            result[cache_type] = {
                "n_sequences": len(self._lru._lrus[cache_type]),
                "n_bytes": self._n_bytes_by_type[cache_type],
            }
        return result
