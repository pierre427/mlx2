# SPDX-License-Identifier: MIT
"""GLM-5.3 ``glm53_kv_append=rotate_v1``: MLA appends that never copy the plane.

Pipelined decode (``async_eval`` of step N+1 before step N completes) keeps
step N holding the latent (1,1,T,512) and raw indexer (1,1,T,256) buffers
while step N+1's ``SliceUpdate`` is encoded. MLX writes in place only into a
buffer nothing else holds, so the reference ``KVCache``/``BatchKVCache``
append copies the whole plane every token: ~3.7 ms per token from 2K to 32K
on GLM-5.3-Flash-REAP50-Q3_K_M.gguf@5a865918 (scripts/research/
glm53_kv_update_bench.py at bceda4ded, M5 Max, MLX 0.32.4).

``rotate_v1`` keeps ``RING_BUFFERS`` full-capacity buffers per plane. Append
t writes buffer t % K with one slice update: first the rows that buffer
missed since it was last current (copied from the current buffer; K - 1 rows
in steady decode), then the new rows. That buffer becomes the plane's
``keys``/``values``. It was last written K steps ago, so the step in flight
holds it at most as a read, and MLX updates it in place. Measured with the
bench (pipelined p50 at 2K/8K/32K, 600 steps): K=1 (the reference law)
0.68/1.44/4.43 ms with up to 624 MB per-step extra peak; K=2 0.44/0.45/0.50
ms with ~0 extra peak; K=3 and K=4 the same as K=2. K=2 is the default; a
loop with deeper pipelining than the bench may need K=3 (MLX falls back to a
copy, never to a wrong value, if a buffer is still held).

Readers see one contiguous buffer ``keys[..., :offset]``, so every consumer
(gathered and mask attention, pool extends, APCv2 ``state``, trims, extract,
merge) runs the reference code on it. Capacity is reserved in ``BASE_BLOCK``
rows (or the plane's ``reserve`` target), so growth allocates once per 2048
tokens instead of every 256. Memory: K x the indexed planes.

Engagement is a class swap of an exact ``KVCache``/``BatchKVCache`` plane by
the GLM-5.3 model under the policy. ``Glm53MLACache.meta_state`` serializes
the canonical class token, so an APCv2 payload names ``KVCache`` or
``BatchKVCache`` as before and a restore comes back as the plain class. Any
outside operation that replaces the buffer or moves the write cursor
(``state`` or ``meta_state`` set, ragged trim, filter, finalize, reserve)
reseeds the ring: the other slots keep their storage but are marked empty and
catch up in full on their next turn.

An evaluated alternative, the native paged arena (native/paged_kv), cannot
hold these planes: its storage is float16/bfloat16 only, it exposes no
mx.array view, and its reads are fp16 dense attention or copy-out gathers.
"""

from __future__ import annotations

import mlx.core as mx

from ..cache_operations import KV_OPERATIONS
from .cache import BatchKVCache, KVCache
from .glm53_kv_append_geometry import (  # noqa: F401 (KV_APPEND_POLICIES re-exported)
    BASE_BLOCK,
    KV_APPEND_POLICIES,
    RING_BUFFERS,
)

_SCHEDULED = {
    "planes_engaged": 0,
    "ring_writes": 0,  # appends into an existing buffer (in place when nothing holds it)
    "ring_catchup_rows": 0,  # rows a buffer copied from the current one before its append
    "ring_grows": 0,  # appends that needed a new buffer (capacity, geometry or first use)
    "ring_reseeds": 0,  # an outside operation replaced the buffer or moved the cursor
    "declined": 0,  # ineligible plane (right padding, bucketed groups): reference append
}


def _capacity(end, reservation):
    return -(-max(end, reservation) // BASE_BLOCK) * BASE_BLOCK


class _RingPlane:
    """Mixin over an exact KV plane class; ``_LEVEL`` names its write cursor.

    ``_ring`` is a list of ``[keys, values, valid_rows]`` slots; the current
    slot is ``_ring_cur`` and holds the plane's ``keys``/``values``.
    """

    _LEVEL = "offset"
    _PARENT = None

    def _level(self):
        return self.__dict__[self._LEVEL]

    def _ring_eligible(self):
        return True

    def _advance(self, n):
        self.offset += n

    def _ring_slots(self):
        d = self.__dict__
        ring = d.get("_ring")
        current = d.get("keys")
        if (
            ring is not None
            and len(ring) == RING_BUFFERS
            and current is d.get("_ring_ref")
            and self._level() == d.get("_ring_seen")
        ):
            return ring
        # Keep the other slots' storage (no fresh allocation), with no valid rows.
        spare = [[s[0], s[1], 0] for s in (ring or ()) if s[0] is not None and s[0] is not current]
        ring = [[current, d.get("values"), 0 if current is None else self._level()]]
        ring += spare[: RING_BUFFERS - 1]
        ring += [[None, None, 0] for _ in range(RING_BUFFERS - len(ring))]
        _SCHEDULED["ring_reseeds"] += d.get("_ring") is not None
        d["_ring"], d["_ring_cur"] = ring, 0
        d["_ring_ref"], d["_ring_seen"] = current, self._level()
        return ring

    def update_and_fetch(self, keys, values):
        if not self._ring_eligible():
            _SCHEDULED["declined"] += 1
            return self._PARENT.update_and_fetch(self, keys, values)
        ring = self._ring_slots()
        d = self.__dict__
        level, n = self._level(), keys.shape[2]
        end = level + n
        cur = d["_ring_cur"]
        nxt = (cur + 1) % RING_BUFFERS
        src_k, src_v = ring[cur][0], ring[cur][1]
        if src_k is not None:
            # The first append fixes each plane's dtype, as the reference buffer does.
            keys = keys if keys.dtype == src_k.dtype else keys.astype(src_k.dtype)
            values = values if values.dtype == src_v.dtype else values.astype(src_v.dtype)
        buf_k, buf_v, valid = ring[nxt]
        valid = min(valid, level)
        fits = (
            buf_k is not None
            and buf_k.shape[2] >= end
            and buf_k.shape[:2] == keys.shape[:2]
            and buf_k.shape[3] == keys.shape[3]
            and buf_v.shape[3] == values.shape[3]
            and buf_k.dtype == keys.dtype
            and buf_v.dtype == values.dtype
        )
        if fits:
            if valid < level:
                keys = mx.concatenate([src_k[..., valid:level, :], keys], axis=2)
                values = mx.concatenate([src_v[..., valid:level, :], values], axis=2)
            # In place when nothing holds buf_k/buf_v (the point of the ring).
            buf_k[..., valid:end, :] = keys
            buf_v[..., valid:end, :] = values
            _SCHEDULED["ring_writes"] += 1
            _SCHEDULED["ring_catchup_rows"] += level - valid
        else:
            cap = _capacity(end, d.get("_reserve_tokens", 0))
            built = []
            for src, rows in ((src_k, keys), (src_v, values)):
                pieces = [src[..., :level, :]] if src is not None and level else []
                pieces.append(rows)
                if cap > end:
                    pieces.append(mx.zeros((*rows.shape[:2], cap - end, rows.shape[3]), rows.dtype))
                built.append(pieces[0] if len(pieces) == 1 else mx.concatenate(pieces, axis=2))
            buf_k, buf_v = built
            _SCHEDULED["ring_grows"] += 1
        ring[nxt] = [buf_k, buf_v, end]
        d["keys"], d["values"] = buf_k, buf_v
        d["_ring_cur"], d["_ring_ref"] = nxt, buf_k
        self._advance(n)
        d["_ring_seen"] = self._level()
        return self.keys_and_values()

    def trim(self, n):
        trimmed = self._PARENT.trim(self, n)
        d = self.__dict__
        ring = d.get("_ring")
        if ring is not None and d.get("_ring_ref") is d.get("keys"):
            # Rows below the new cursor are untouched in every slot.
            level = self._level()
            for slot in ring:
                slot[2] = min(slot[2], level)
            d["_ring_seen"] = level
        return trimmed

    @property
    def nbytes(self):
        d = self.__dict__
        seen, total = set(), 0
        slots = d.get("_ring") or []
        for a in (d.get("keys"), d.get("values"), *(x for slot in slots for x in slot[:2])):
            if a is not None and id(a) not in seen:
                seen.add(id(a))
                total += a.nbytes
        return total


class Glm53RingKVCache(_RingPlane, KVCache):
    """``KVCache`` over rotating buffers; state/meta_state are the reference ones."""

    _PARENT = KVCache
    # Same planes and fields as KVCache; declared on this exact class because
    # inherited declarations are not proof (cache_operations._declared_contract).
    cache_operations_contract = KV_OPERATIONS


class Glm53RingBatchKVCache(_RingPlane, BatchKVCache):
    """``BatchKVCache`` over rotating buffers (the serving plane after merge)."""

    _PARENT = BatchKVCache
    _LEVEL = "_idx"
    # No per-row ledger beside K/V. A ragged trim rolls the current buffer
    # into a new array, so the ring reseeds on its next append.
    _RAGGED_TRIM_AUX_ARRAYS = ()

    def _ring_eligible(self):
        d = self.__dict__
        return d.get("_right_padding") is None and d.get("_bucket_groups") is None

    def _advance(self, n):
        # As BatchKVCache.update_and_fetch advances its cursors.
        self.offset += n
        self._idx += n


_RING_CLASS = {KVCache: Glm53RingKVCache, BatchKVCache: Glm53RingBatchKVCache}
_CANONICAL = {Glm53RingKVCache: KVCache, Glm53RingBatchKVCache: BatchKVCache}


def engage(planes, policy="rotate_v1"):
    """Swap each exact KVCache/BatchKVCache plane in ``planes`` to its ring class, in place."""
    if policy != "rotate_v1":
        raise ValueError(f"unsupported GLM-5.3 KV append policy {policy!r}")
    for plane in planes:
        cls = _RING_CLASS.get(type(plane))
        if cls is not None:
            plane.__class__ = cls
            _rebind_cow_hook(plane, cls)
            _SCHEDULED["planes_engaged"] += 1


def _rebind_cow_hook(plane, cls):
    """An APCv2 COW branch carries an instance ``update_and_fetch`` hook bound
    to the plain class's method (cow_cache._attach_mutation_hooks); after the
    class swap it would keep running the reference append. Re-wrap the ring
    method with the same first-write token."""
    d = plane.__dict__
    if "update_and_fetch" not in d.get("_cow_hooked_methods", ()):
        return
    tokens = d.get("_cow_segment_tokens") or {}
    token = tokens.get("attention_kv") or next(iter(tokens.values()), None)
    method = cls.update_and_fetch.__get__(plane)

    def hooked(*args, __method=method, __token=token, **kwargs):
        if __token is not None:
            __token.note_mutation(f"{cls.__name__}.update_and_fetch")
        return __method(*args, **kwargs)

    plane.update_and_fetch = hooked


def is_ring(plane):
    return isinstance(plane, _RingPlane)


def canonical_type(plane):
    """The class a plane serializes as: ring planes name their reference class."""
    return _CANONICAL.get(type(plane), type(plane))


def route_status():
    return {
        "policies": {
            "reference": {"implemented": True, "qualified": False, "selected_by_default": True},
            "rotate_v1": {"implemented": True, "qualified": False, "selected_by_default": False,
                          "ring_buffers": RING_BUFFERS, "base_block": BASE_BLOCK},
        },
        "scheduled_calls": dict(_SCHEDULED),
    }
