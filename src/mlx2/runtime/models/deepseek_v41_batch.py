# SPDX-License-Identifier: MIT
"""Model-owned ordinary batch: private attention, jointly executed FFN/head.

The numerical profile differs from width-one GEMV. No qualification or official
CUDA parity follows from this implementation. B1 remains a separate model route.
"""

import json
from contextlib import nullcontext
from copy import copy
from dataclasses import dataclass
from itertools import pairwise

import mlx.core as mx
import numpy as np

from ..cache_operations import (
    CacheOperationsContract,
    CachePlaneDeclaration,
    CheckpointPublicationCondition,
)
from .cache import register_cache_class
from .deepseek_v41_cache import DeepSeekV41LayerCache
from .deepseek_v41_packed_cache import DeepSeekV41PackedLayerCache

REVISION = "mlx2.v41.private-attention-joint-ffn-head.v1"
ROW_CLASSES = {
    c.__name__: c for c in (DeepSeekV41LayerCache, DeepSeekV41PackedLayerCache)
}


def _clone(row):
    snap = row.snapshot()
    return type(row).from_state(snap.arrays, snap.metadata, xp=row.xp)


@dataclass(frozen=True)
class BatchSnapshot:
    arrays: tuple
    metadata: tuple


class DeepSeekV41BatchLayerCache:
    cache_operations_contract = CacheOperationsContract(
        family="deepseek_v41",
        topology="checkpointed_hybrid",
        state_revision=REVISION,
        cache_layout="private-native-rows",
        numerics="joint-ffn-head-profile",
        planes=(
            CachePlaneDeclaration(
                "attention_ring", ("state",), ("identity", "max_width")
            ),
        ),
        exact_prefix=True,
        restores_at_checkpoints=True,
        rollback=None,
        publication_conditions=(
            CheckpointPublicationCondition(
                "pending", "truthy", "batch forward is incomplete"
            ),
        ),
    )

    def __init__(self, rows, identity, max_width):
        rows = list(rows)
        if (
            type(max_width) is not int
            or max_width not in (2, 4)
            or not 1 <= len(rows) <= max_width
            or type(identity) is not str
            or not identity
            or type(rows[0]) not in ROW_CLASSES.values()
        ):
            raise ValueError("V4.1 batch row count or identity is invalid")
        topology = tuple(rows[0].meta_state[:9])
        if any(
            type(r) not in ROW_CLASSES.values()
            or type(r) is not type(rows[0])
            or tuple(r.meta_state[:9]) != topology
            or r.pending
            for r in rows
        ):
            raise ValueError(
                "V4.1 batch requires independent compatible committed rows"
            )
        if len({id(r) for r in rows}) != len(rows):
            raise ValueError("batch rows must have independent owners")
        self.rows, self.identity, self.max_width = rows, identity, max_width
        self._in_forward = False
        self._closed = False
        self._remaining = None
        self._left = None

    @property
    def pending(self):
        return self._in_forward or any(r.pending for r in self.rows)

    @property
    def offset(self):
        return (
            self.rows[0].offset
            if len(self.rows) == 1
            else mx.array([r.offset for r in self.rows], mx.int32)
        )

    @property
    def left_padding(self):
        return mx.zeros((len(self.rows),), mx.int32)

    @property
    def host_lengths(self):
        return [r.offset for r in self.rows]

    @property
    def host_left_padding(self):
        return [0] * len(self.rows)

    @property
    def nbytes(self):
        return sum(r.nbytes for r in self.rows)

    @property
    def state(self):
        return tuple(tuple(r.state) for r in self.rows)

    @state.setter
    def state(self, values):
        if self.pending or self._closed:
            raise ValueError("cannot replace active/closed batch state")
        if len(values) != len(self.rows):
            raise ValueError("batch state row mismatch")
        for row, value in zip(self.rows, values):
            row.state = value

    @property
    def meta_state(self):
        return (
            REVISION,
            self.identity,
            str(self.max_width),
            type(self.rows[0]).__name__,
            json.dumps([r.meta_state for r in self.rows], separators=(",", ":")),
        )

    @classmethod
    def from_state(cls, state, meta_state):
        if (
            len(meta_state) != 5
            or meta_state[0] != REVISION
            or meta_state[3] not in ROW_CLASSES
        ):
            raise ValueError("invalid V4.1 batch cache revision/class")
        if meta_state[2] not in ("2", "4"):
            raise ValueError("invalid V4.1 batch width")
        metadata = json.loads(meta_state[4])
        row_cls = ROW_CLASSES[meta_state[3]]
        if not 1 <= len(metadata) <= int(meta_state[2]) or len(metadata) != len(state):
            raise ValueError("batch state/metadata row mismatch")
        rows = [
            row_cls.from_state(value, tuple(meta))
            for value, meta in zip(state, metadata)
        ]
        return cls(rows, meta_state[1], int(meta_state[2]))

    def snapshot(self):
        self.validate_state()
        self.cache_operations_contract.validate_publication(self)
        rows = [_clone(r) for r in self.rows]
        value = type(self)(rows, self.identity, self.max_width)
        return BatchSnapshot(value.state, value.meta_state)

    def restore(self, snapshot):
        value = type(self).from_state(snapshot.arrays, snapshot.metadata)
        if (
            value.identity != self.identity
            or value.max_width != self.max_width
            or len(value.rows) != len(self.rows)
        ):
            raise ValueError("batch snapshot execution identity mismatch")
        self.rows = value.rows
        self._in_forward = False
        self._remaining = self._left = None

    @classmethod
    def merge(cls, caches):
        if not caches or any(
            type(c) is not cls or c.pending or c._closed for c in caches
        ):
            raise ValueError("V4.1 batch merge needs committed batch owners")
        first = caches[0]
        if any(
            c.identity != first.identity or c.max_width != first.max_width
            for c in caches
        ):
            raise ValueError("V4.1 batch merge execution identity mismatch")
        count = sum(len(c.rows) for c in caches)
        if count > first.max_width:
            raise ValueError("V4.1 physical batch exceeds admitted width")
        return cls(
            [_clone(r) for c in caches for r in c.rows], first.identity, first.max_width
        )

    def extract(self, index):
        if (
            self.pending
            or self._closed
            or type(index) is not int
            or not 0 <= index < len(self.rows)
        ):
            raise ValueError("invalid batch row index")
        return type(self)([_clone(self.rows[index])], self.identity, self.max_width)

    def fork(self):
        return type(self).merge([self])

    def filter(self, keep):
        keep = list(keep)
        if (
            self.pending
            or self._closed
            or len(set(keep)) != len(keep)
            or any(type(i) is not int or not 0 <= i < len(self.rows) for i in keep)
        ):
            raise ValueError("invalid batch row filter")
        self.rows = [self.rows[i] for i in keep]
        if self._remaining is not None:
            self._remaining = [self._remaining[i] for i in keep]
        if self._left is not None:
            self._left = [self._left[i] for i in keep]
        if not keep:
            self._closed = True

    def extend(self, other):
        merged = type(self).merge([self, other])
        self.rows = merged.rows
        self._remaining = self._left = None

    def prepare(self, *, lengths=None, left_padding=None, right_padding=None):
        if self.pending or self._closed:
            raise ValueError("cannot prepare a closed/active batch")
        count = len(self.rows)

        def vector(value, label):
            if value is None:
                return None
            result = list(value)
            if len(result) != count or any(type(n) is not int or n < 0 for n in result):
                raise ValueError("invalid batch " + label)
            return result

        remaining = vector(lengths, "lengths")
        left = vector(left_padding, "left padding")
        right = vector(right_padding, "right padding")
        if right is not None and remaining is None:
            raise ValueError("right padding requires logical lengths")
        if (
            right is not None
            and len({length + pad for length, pad in zip(remaining, right)}) != 1
        ):
            raise ValueError("ragged rows do not share a padded span")
        self._remaining, self._left = remaining, left

    def layout(self, width):
        return [
            (
                min((self._left or [0] * len(self.rows))[i], width),
                min(
                    width - min((self._left or [0] * len(self.rows))[i], width),
                    (self._remaining or [width] * len(self.rows))[i],
                ),
            )
            for i in range(len(self.rows))
        ]

    def advance(self, width):
        layout = self.layout(width)
        if self._remaining is not None:
            self._remaining = [
                n - count for n, (_, count) in zip(self._remaining, layout)
            ]
        if self._left is not None:
            self._left = [max(0, n - width) for n in self._left]

    def finalize(self):
        if (
            self.pending
            or self._closed
            or (self._remaining is not None and any(self._remaining))
            or (self._left is not None and any(self._left))
        ):
            raise ValueError("V4.1 batch prefill has incomplete logical rows")
        self._remaining = self._left = None
        self.validate_state()

    def validate_state(self):
        if self._closed:
            raise ValueError("batch cache is closed")
        for row in self.rows:
            row.validate_state()

    def is_trimmable(self):
        return False

    def empty(self):
        return all(r.offset == 0 for r in self.rows)

    def size(self):
        return max((r.offset for r in self.rows), default=0)

    def close(self):
        self.rows = []
        self._closed = True
        self._remaining = self._left = None


class DeepSeekV41BatchedModel:
    """Per-layer request-private attention plus actual multirow FFN/router/head."""

    def __init__(self, base, *, width=2):
        if type(width) is not int or width not in (2, 4):
            raise ValueError("ordinary physical width must be2 or4")
        self.base, self.args, self.weights = (
            base,
            base.args,
            base.weights,
        )
        self.max_width = width
        self.identity = base.identity + ":" + REVISION + ":width=" + str(width)
        self.context_limit = base.context_limit
        self.batch_execution_profile = {
            "revision": REVISION,
            "maximum_physical_width": width,
            "attention_physical_width": 1,
            "router_ffn_head": "joint_request_rows",
            "qualification": "unqualified",
            "numerical_equivalence": "not_assumed_bitwise",
        }
        self.batch_counters = {
            "forwards": 0,
            "joint_ffn_calls": 0,
            "joint_head_calls": 0,
            "maximum_live_requests": 0,
            "maximum_joint_ffn_requests": 0,
        }

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def layers(self):
        # Parking replaces this list; retaining the original would pin weights.
        return self.base.layers

    def make_cache(self):
        return [
            DeepSeekV41BatchLayerCache([r], self.identity, self.max_width)
            for r in self.base.make_cache()
        ]

    def named_modules(self):
        yield "", self

    def eval(self):
        self.base.eval()
        return self

    def preload(self):
        self.base.preload()
        return self

    def parameters(self):
        return self.base.parameters()

    def __call__(self, inputs, cache=None, *, logits_to_keep=None):
        self.base._weight_executions += 1
        try:
            execution = getattr(self.weights, "execution_scope", None)
            with execution() if execution is not None else nullcontext():
                return self._forward_impl(inputs, cache, logits_to_keep=logits_to_keep)
        finally:
            self.base._weight_executions -= 1

    def _forward_impl(self, inputs, cache=None, *, logits_to_keep=None):
        if logits_to_keep is not None and (
            type(logits_to_keep) is not int or logits_to_keep < 0
        ):
            raise ValueError("invalid V4.1 logits_to_keep")
        if (
            inputs.ndim != 2
            or not 1 <= inputs.shape[0] <= self.max_width
            or not 0 < inputs.shape[1] <= 1024
            or inputs.dtype not in (mx.int32, mx.uint32, mx.int64)
        ):
            raise ValueError(
                "V4.1 batch requires integer B1..bound tokens and rows1..1024"
            )
        if cache is None:
            cache = self.make_cache()
            if inputs.shape[0] > 1:
                fresh = [self.make_cache() for _ in range(inputs.shape[0])]
                cache[:] = [
                    DeepSeekV41BatchLayerCache.merge([r[i] for r in fresh])
                    for i in range(len(self.layers))
                ]
        if len(cache) != len(self.layers) or any(
            type(c) is not DeepSeekV41BatchLayerCache
            or c.identity != self.identity
            or c.max_width != self.max_width
            or len(c.rows) != inputs.shape[0]
            or c.pending
            for c in cache
        ):
            raise ValueError("V4.1 batch cache geometry/execution mismatch")
        layouts = [c.layout(inputs.shape[1]) for c in cache]
        if any(v != layouts[0] for v in layouts):
            raise ValueError("batch logical padding differs across layers")
        layout = layouts[0]
        m, c = self.base, self.args
        # Descriptor-only transaction: global stores append only past the saved
        # visible frontier; local/tail planes replace. Prefix payloads are never
        # rewritten. Merge/extract own separate buffers across requests.
        rows = [
            [copy(layer.rows[i]) for layer in cache] for i in range(inputs.shape[0])
        ]
        ids = np.asarray(inputs)
        hidden = {}
        pre = {}
        hashes = {}
        tails = {}
        shared = {}
        for i, (start, count) in enumerate(layout):
            if not count:
                continue
            tokens = inputs[i : i + 1, start : start + count]
            m.validate_forward(tokens, rows[i], 0)
            if m.hasher is not None:
                hashes[i], tail = m.hasher(
                    ids[i, start : start + count], np.asarray(rows[i][0].hash_tail)
                )
                tails[i] = mx.array(tail, mx.int32)
            with m._weight_scope():
                x = m.round(self.weights.embedding(tokens.astype(mx.uint32)))
                hidden[i] = mx.broadcast_to(
                    x[:, :, None], (1, count, c.hc_mult, c.hidden_size)
                )
                pre[i] = mx.broadcast_to(
                    mx.array([1.0] + [0.0] * (c.hc_mult - 1)), hidden[i].shape[:-1]
                )
                mx.eval(hidden[i], pre[i])
            shared[i] = {}
        for layer in cache:
            layer._in_forward = True
        try:
            for layer, block in enumerate(self.layers):
                with m._weight_scope():
                    stages = {}
                    flat = []
                    lengths = []
                    for i, current_hidden in hidden.items():
                        if layer in c.engram_layer_ids:
                            # EngramRows owns one bounded prefetch window. Consume
                            # each lane before allowing the next lane to replace it.
                            m.engram_tables[layer].prefetch(
                                hashes[i][:, c.engram_layer_ids.index(layer)]
                            )
                            current_hidden = m.engram(
                                current_hidden,
                                layer,
                                hashes[i][:, c.engram_layer_ids.index(layer)],
                            )
                        h, x, next_pre, post, comb = block.attention_half(
                            current_hidden, pre[i], rows[i][layer], rows[i], shared[i]
                        )
                        stages[i] = (h, post, comb)
                        pre[i] = next_pre
                        flat.append(x.reshape(-1, c.hidden_size))
                        lengths.append((i, x.shape[1]))
                    if not flat:
                        continue
                    combined = mx.concatenate(flat, 0)
                    routed_parts = []
                    shared_parts = []
                    boundaries = np.cumsum([0] + [length for _, length in lengths])
                    # Existing providers admit at most1024 routed rows.
                    for first in range(0, combined.shape[0], 1024):
                        x = combined[first : first + 1024]
                        indices, scores = block.router(x)
                        routed_parts.append(
                            self.weights.moe(x, indices, scores, block.prefix[:-1])
                        )
                        shared_parts.append(block.shared_expert(x))
                        requests = int(
                            sum(
                                left < first + x.shape[0] and right > first
                                for left, right in pairwise(boundaries)
                            )
                        )
                        self.batch_counters["maximum_joint_ffn_requests"] = max(
                            self.batch_counters["maximum_joint_ffn_requests"], requests
                        )
                        if requests > 1:
                            self.batch_counters["joint_ffn_calls"] += 1
                    routed, common = (
                        mx.concatenate(routed_parts),
                        mx.concatenate(shared_parts),
                    )
                    offset = 0
                    for i, length in lengths:
                        h, post, comb = stages[i]
                        hidden[i] = block.finish(
                            h,
                            routed[offset : offset + length][None],
                            common[offset : offset + length][None],
                            post,
                            comb,
                        )
                        offset += length
                    # Complete every provider lease before exiting its window.
                    mx.eval(
                        list(hidden.values()),
                        list(pre.values()),
                        [r.state for lane in rows for r in lane],
                        shared,
                    )
            for i, tail in tails.items():
                rows[i][0].hash_tail = tail
            normalized = []
            for i, (start, count) in enumerate(layout):
                if count:
                    value = m.norm(m.collapse(hidden[i], pre[i]), m.norm_weight)
                    value = mx.concatenate(
                        (
                            mx.zeros((1, start, c.hidden_size)),
                            value,
                            mx.zeros(
                                (1, inputs.shape[1] - start - count, c.hidden_size)
                            ),
                        ),
                        1,
                    )
                else:
                    value = mx.zeros((1, inputs.shape[1], c.hidden_size))
                normalized.append(value)
            value = mx.concatenate(normalized, 0)
            from ..prefill_checkpoint_rows import select_checkpoint_rows

            value = (
                select_checkpoint_rows(value)
                if logits_to_keep is None
                else value[:, -logits_to_keep:]
                if logits_to_keep
                else value
            )
            with m._weight_scope():
                logits = m.linear(value, "output.weight", rounded=False)
                mx.eval(logits)
            if len(hidden) > 1:
                self.batch_counters["joint_head_calls"] += 1
            mx.eval(logits, [r.state for lane in rows for r in lane])
            for lane in rows:
                for row in lane:
                    row.validate_state()
            for layer, plane in enumerate(cache):
                plane.rows = [lane[layer] for lane in rows]
                plane.advance(inputs.shape[1])
            self.batch_counters["forwards"] += 1
            self.batch_counters["maximum_live_requests"] = max(
                self.batch_counters["maximum_live_requests"], len(hidden)
            )
            return logits
        finally:
            for plane in cache:
                plane._in_forward = False

    def forward_with_taps(self, *args, **kwargs):
        raise ValueError("ordinary physical batch excludes draft target taps")

    def prefill_body(self, *args, **kwargs):
        raise ValueError("ordinary physical batch requires its full-depth prefill")

    def prefill_forward(self, inputs, cache=None, **kwargs):
        return self(inputs, cache, **kwargs)


register_cache_class(DeepSeekV41BatchLayerCache)
