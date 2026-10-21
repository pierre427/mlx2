"""Tree-tile target execution for attention-only (mlx-lm style) models.

Original mlx2 implementation.  An AST tile is the anchor row plus proposal
nodes, each attending to the committed prefix and its own ancestors.  A model
built the mlx-lm way (decoder layers whose attention applies
``rope(x, offset=cache.offset)`` and takes a mask from
``create_attention_mask -> cache.make_mask``) can verify such a tile with its
own forward, unchanged, if for that one forward

* RoPE sees one position per row (``offset + depth``, rows as the batch axis
  with an offset per row: the same per-element law as one call per row), and
* each cache's ``make_mask`` returns the prefix-plus-ancestor mask, with the
  sliding-window bound applied to prefix keys for windowed layers.

Commit keeps the accepted root path by compaction: a full ``KVCache`` moves
the path's rows to the front of the window in place; a ``RotatingKVCache``
(whose multi-row update leaves ``[prefix, window]`` in temporal order) keeps
its prefix plus the path's rows.  Abort restores the pre-forward descriptors.
Any other cache class (recurrent state) is refused: those families need a
parent-indexed state kernel (the Qwen hybrids have TensorFold's).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import mlx.core as mx

from .models.cache import KVCache, RotatingKVCache
from .target_tree_executor import TargetTreeDispatchMetrics

PROTOCOL = "attention_tree_v1"
_ABSENT = object()
MAX_TILE_ROWS = 32


def _depths(parents):
    depths = []
    for row, parent in enumerate(parents):
        parent = int(parent)
        if parent < 0:
            depths.append(0)
        elif parent >= row:
            raise ValueError("tree parents must precede their children")
        else:
            depths.append(depths[parent] + 1)
    return depths


def _ancestors(parents):
    width = len(parents)
    rows = [[False] * width for _ in range(width)]
    for row, parent in enumerate(parents):
        rows[row][row] = True
        node = int(parent)
        while node >= 0:
            rows[row][node] = True
            node = int(parents[node])
    return rows


def _prefix_columns(cache):
    """Committed key columns a multi-row update leaves before the window."""

    if type(cache) is KVCache:
        return int(cache.offset)
    if type(cache) is RotatingKVCache:
        if cache.keys is None:
            return 0
        return min(int(cache.offset), int(cache.max_size) - 1)
    raise ValueError(f"attention tree tiles do not support {type(cache).__name__}")


def attention_modules(model):
    """The decoder layers' RoPE-bearing attention modules."""

    core = getattr(model, "model", model)
    modules = []
    for layer in core.layers:
        for name in ("self_attn", "attention", "attn"):
            module = getattr(layer, name, None)
            if module is not None and getattr(module, "rope", None) is not None:
                modules.append(module)
                break
        else:
            raise ValueError("attention tree tiles need a RoPE attention module per layer")
    return modules


def validate_caches(cache):
    for item in cache:
        if type(item) not in (KVCache, RotatingKVCache):
            raise ValueError(f"attention tree tiles do not support {type(item).__name__}")
        if type(item) is RotatingKVCache and (item.keep or item.speculating):
            raise ValueError("attention tree tiles need an unpinned, non-speculating window")


class _RowRope:
    """RoPE with one position per row for the duration of a tile forward."""

    def __init__(self, rope, depths):
        self.rope = rope
        self.depths = mx.array(depths, dtype=mx.int32)

    def __call__(self, x, offset=0):
        batch, heads, rows, dim = x.shape
        if batch != 1 or rows != int(self.depths.shape[0]):
            return self.rope(x, offset=offset)
        rows_first = x.transpose(2, 1, 0, 3)                 # [rows, heads, 1, dim]
        rotated = self.rope(rows_first, offset=self.depths + int(offset))
        return rotated.transpose(2, 1, 0, 3)


class _TileScope:
    """Install row RoPE and tree masks on one lane's modules and caches."""

    def __init__(self, model, cache, parents):
        self.model, self.cache, self.parents = model, cache, list(parents)
        self.depths = _depths(self.parents)
        self.ancestors = _ancestors(self.parents)
        self.installed_rope = []
        self.installed_mask = []

    def _mask_for(self, item):
        start = int(item.offset)
        prefix = _prefix_columns(item)
        width = len(self.parents)

        def make_mask(n, return_array=False, window_size=None):
            if n != width:
                raise ValueError("tree tile mask requested for a different row count")
            rows = []
            for row in range(width):
                position = start + self.depths[row]
                if window_size is None:
                    head = [True] * prefix
                else:
                    head = [position - (start - prefix + column) < window_size
                            for column in range(prefix)]
                rows.append(head + self.ancestors[row])
            return mx.array(rows, dtype=mx.bool_)

        return make_mask

    def __enter__(self):
        for module in attention_modules(self.model):
            original = module.rope
            # A module child lives in the Module dict; a plain RoPE object in
            # the instance __dict__, which must be put back, not deleted.
            shadowed = module.__dict__.get("rope", _ABSENT)
            object.__setattr__(module, "rope", _RowRope(original, self.depths))
            self.installed_rope.append((module, shadowed))
        for item in self.cache:
            item.make_mask = self._mask_for(item)
            self.installed_mask.append(item)
        return self

    def __exit__(self, *_exc):
        for module, shadowed in self.installed_rope:
            if shadowed is _ABSENT:
                object.__delattr__(module, "rope")
            else:
                object.__setattr__(module, "rope", shadowed)
        for item in self.installed_mask:
            item.__dict__.pop("make_mask", None)
        return False


def _snapshot(cache):
    return [(item, item.keys, item.values, int(item.offset), getattr(item, "_idx", None))
            for item in cache]


def _restore(snapshot):
    for item, keys, values, offset, idx in snapshot:
        item.keys, item.values, item.offset = keys, values, offset
        if idx is not None:
            item._idx = idx


def _compact(snapshot, width, path):
    keep = len(path)
    rows = None if list(path) == list(range(keep)) else mx.array(list(path), dtype=mx.int32)
    for item, _keys, _values, start, _idx in snapshot:
        if type(item) is KVCache:
            if rows is not None:
                window_k = mx.take(item.keys[..., start:start + width, :], rows, axis=2)
                window_v = mx.take(item.values[..., start:start + width, :], rows, axis=2)
                item.keys[..., start:start + keep, :] = window_k
                item.values[..., start:start + keep, :] = window_v
            item.offset = start + keep
        elif width == 1:
            continue                                     # in-place one-row step: kept as is
        else:
            prefix = int(item._idx) - width
            if rows is not None or keep != width:
                take = rows if rows is not None else mx.arange(keep)
                item.keys = mx.concatenate(
                    [item.keys[..., :prefix, :], mx.take(item.keys[..., prefix:, :], take, axis=2)], axis=2)
                item.values = mx.concatenate(
                    [item.values[..., :prefix, :], mx.take(item.values[..., prefix:, :], take, axis=2)], axis=2)
            item.offset = start + keep
            item._idx = prefix + keep


class AttentionTreeTransaction:
    def __init__(self, cache, snapshot, width, *, protocol, token):
        self.cache, self.snapshot, self.width = cache, snapshot, int(width)
        self.closed = False
        self.executor_cached = False
        self.physical_packed = False
        self.provider_protocol, self.provider_revision_token = protocol, token

    def commit_paths(self, paths):
        if self.closed or len(paths) != 1:
            raise RuntimeError("attention tree transaction is single-lane")
        try:
            _compact(self.snapshot, self.width, list(paths[0]))
        finally:
            self.closed = True
        return [self.cache]

    def commit(self, *, accepted_lengths):
        if len(accepted_lengths) != 1:
            raise RuntimeError("attention tree transaction is single-lane")
        return self.commit_paths([list(range(int(accepted_lengths[0])))])

    def abort(self):
        if not self.closed:
            _restore(self.snapshot)
            self.closed = True


class AttentionTreeCohortTransaction:
    def __init__(self, transactions, *, protocol, token):
        self.transactions = tuple(transactions)
        self.closed = False
        self.executor_cached = False
        self.physical_packed = False
        self.provider_protocol, self.provider_revision_token = protocol, token

    def commit_paths(self, paths):
        if self.closed or len(paths) != len(self.transactions):
            raise RuntimeError("attention tree cohort path count mismatch")
        caches = []
        try:
            for transaction, path in zip(self.transactions, paths):
                caches.extend(transaction.commit_paths([path]))
        finally:
            self.closed = True
        return caches

    def commit(self, *, accepted_lengths):
        return self.commit_paths([list(range(int(n))) for n in accepted_lengths])

    def abort(self):
        if not self.closed:
            for transaction in self.transactions:
                transaction.abort()
            self.closed = True


def tile_forward(model, tokens, parents, cache, capture_layers):
    """Verify one lane's tile: ``(logits [1, W, V], features [1, W, F], txn)``."""

    if not 1 <= len(tokens) <= MAX_TILE_ROWS or len(tokens) != len(parents):
        raise ValueError("attention tree tile rows must match parents and fit 32 rows")
    if int(parents[0]) != -1 or any(int(p) < 0 for p in parents[1:]):
        raise ValueError("attention tree tile must be rooted at its anchor row")
    validate_caches(cache)
    offsets = {int(item.offset) for item in cache}
    if len(offsets) != 1:
        raise ValueError("attention tree tile requires one committed boundary")
    snapshot = _snapshot(cache)
    try:
        with _TileScope(model, cache, parents):
            logits, features = model.forward_with_taps(
                mx.array([[int(token) for token in tokens]], dtype=mx.uint32),
                cache, capture_layers,
            )
    except BaseException:
        _restore(snapshot)
        raise
    return logits, features, snapshot


class AttentionTreeExecutor:
    """Target-tree provider for attention-only models with standard caches."""

    protocol = PROTOCOL
    maximum_cohort_width = 4

    def __init__(self):
        source = Path(__file__).read_bytes()
        self.source_revision_token = hashlib.sha256(source).hexdigest()
        self.source_identity = (("module", "mlx2.runtime.attention_tree_executor"),
                                ("sha256", self.source_revision_token))
        self.dispatches = 0

    @classmethod
    def create(cls):
        return cls()

    def validate_revision(self):
        return None

    def validate_cohort(self, width, configured_limit):
        if type(width) is not int or not 1 <= width <= self.maximum_cohort_width or width > configured_limit:
            raise ValueError("attention tree cohort exceeds its bound")

    def forward(self, model, inputs, parents, cache, layers, *, cached):
        self.validate_cohort(1, 1)
        logits, features, snapshot = tile_forward(model, inputs, parents, cache, layers)
        self.dispatches += 1
        return logits, features, AttentionTreeTransaction(
            cache, snapshot, len(inputs), protocol=self.protocol, token=self.source_revision_token,
        )

    def forward_many(self, model, inputs, parents, caches, layers, *, cached, configured_limit):
        self.validate_cohort(len(caches), configured_limit)
        if len({len(row) for row in inputs}) != 1:
            raise ValueError("attention tree cohort requires uniform tile widths")
        outputs, transactions = [], []
        try:
            for row, tree, cache in zip(inputs, parents, caches, strict=True):
                logits, features, snapshot = tile_forward(model, row, tree, cache, layers)
                outputs.append((logits, features))
                transactions.append(AttentionTreeTransaction(
                    cache, snapshot, len(row), protocol=self.protocol,
                    token=self.source_revision_token,
                ))
        except BaseException:
            for transaction in transactions:
                transaction.abort()
            raise
        self.dispatches += 1
        return (
            mx.concatenate([logits for logits, _ in outputs], axis=0),
            mx.concatenate([features for _, features in outputs], axis=0),
            AttentionTreeCohortTransaction(
                transactions, protocol=self.protocol, token=self.source_revision_token,
            ),
        )

    def dispatch_metrics(self, width, transaction, *, cached):
        self.validate_cohort(width, self.maximum_cohort_width)
        return TargetTreeDispatchMetrics(
            width=width, physical_packed=False, executor_cached=False,
            cached=bool(cached), provider_protocol=self.protocol,
        )

    def receipt(self):
        return {
            "provider": self.protocol,
            "source_identity": dict(self.source_identity),
            "source_revision_token": self.source_revision_token,
            "dispatches": self.dispatches,
            "qualified": False,
        }


__all__ = [
    "AttentionTreeExecutor",
    "AttentionTreeTransaction",
    "attention_modules",
    "tile_forward",
]
