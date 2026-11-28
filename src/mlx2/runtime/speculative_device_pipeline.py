"""Depth-2 pipelined device rounds (``device_round.pipeline``, spec-round DC-5).

Original mlx2 implementation; design in
docs/research/2026-11-25-gpu-resident-spec-rounds-design.md section 4.9 (DC-5).
Default off: only ``speculative_device_round`` calls into this module, and only
when the ``pipeline`` switch is on (it needs ``device_commit``).

Each poll serves the one live lane's device round as follows: dispatch R if no
round is in flight; dispatch R' from R's device anchor and its device-committed
caches; read R's packet and book R; keep R' in flight. The host read of R
overlaps R' on the device, and every token leaves in the same poll as in the
serial device-commit round (DC-4), of which this is an exact re-schedule:

* R commits on the device at dispatch with the serial round's ``consumed``,
  computed on the device: ``min(accepted + 1, first stop + 1)``;
* R' drafts with no RNG (the greedy pair walk never reads its uniforms) and
  books its ``k`` draft uniforms before its verify draws, so the lane RNG
  sequence is the serial one;
* R' is dispatched only when its depth is ``num_draft`` whatever R accepts;
* an undo record taken just before each dispatch restores the lane exactly
  while only that round wrote since (its KV rows land at or past the previous
  device ``pos``, its ring rows in slots invisible there). R' is discarded
  (restored, never booked) whenever the serial engine would not run it next:
  R finishes the lane or fails, the next poll's cohort or round differs, or
  the lane leaves the device form (decline, ordinary/parked/forced round,
  removal). Discarding is exact because R' is exactly that next round.
* counters the dispatch bumps (target forward, lane execution counts) are
  deferred to the booking, so a discarded round leaves no trace.

Declared presence penalties (``presence_window``, e.g. Raon's vendor 1.5) are
served under the ``processors`` switch (DC-6): a following round's processor
history is a ``speculative_device_presence.DeviceHistory`` (the round ahead's
unread tokens stay on the device), the same law as the serial round's.

Under ``device_round.fixed_shape`` (slice 1a-5) every dispatched round
writes the menu width ``num_draft + 1`` (drafts padded on the device, the
fixed-shape plan): ``Inflight.rows`` is the rows written (the read base, the
settle of the round behind, ``book_round``), while ``Inflight.count`` stays
the drafted count (draft uniforms, the follower budget check, the packet).

Out of scope (the serial DC-4 round serves them, counted under
``external_device_round_pipeline_skips``): any other logits processor and
proposal composition (they read the host history), round timing, more than one
live lane, and budget ends within reach of a following round.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

VERSION = "speculative-device-pipeline-v1"

COUNTERS = (
    "external_device_round_pipeline_rounds",
    "external_device_round_pipeline_dispatches",
    "external_device_round_pipeline_followers",
    "external_device_round_pipeline_discards",
    "external_device_round_pipeline_aborts",
    "external_device_round_pipeline_dispatch_failures",
    "external_device_round_pipeline_follower_declines",
)
SKIPS = "external_device_round_pipeline_skips"
_LANE_COUNTERS = ("cohort_target_execution",)
_COMPOSITION = "<composition_stats>"  # the composed drafter's counters (owner.draft)


def _bump(stats, key, amount=1):
    stats[key] = stats.get(key, 0) + amount


# -- counter deltas ------------------------------------------------------------------

def _delta(after, before):
    out = {}
    for key, value in after.items():
        old = before.get(key)
        if isinstance(value, dict):
            nested = _delta(value, old if isinstance(old, dict) else {})
            if nested:
                out[key] = nested
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if old is None:
                if value:
                    out[key] = value
            elif isinstance(old, (int, float)) and value != old:
                out[key] = value - old
    return out


def _apply(target, delta):
    for key, value in delta.items():
        if isinstance(value, dict):
            _apply(target.setdefault(key, {}), value)
        else:
            target[key] = target.get(key, 0) + value


# -- the undo record -----------------------------------------------------------------

@dataclass
class _Record:
    lane: Any
    planes: list
    host: dict
    refs: tuple
    history: int
    stats: dict
    # Journal lengths (append-only lists restored by truncation, never copied).
    ready: int = 0
    lane_counters: dict = field(default_factory=dict)


def _host_copy(lane):
    from .external_speculative import _LANE_JOURNAL, _LANE_PLANES
    from .processor_probe import copy_sharing, rollback_shared_memo

    fields = vars(lane)
    shared = rollback_shared_memo(fields.get("processors"))
    # ``ready`` is the response queue serving drains between polls: a journal
    # (restored by truncation), so a restore never re-publishes a response.
    return copy_sharing(
        {k: v for k, v in fields.items()
         if k not in _LANE_PLANES | _LANE_JOURNAL | {"ready"}}, shared
    )


def record(owner, lane):
    """Everything a dispatch may change, without pinning a K/V buffer."""
    from .drafters.device_ring_kv import DeviceRingKVCache
    from .models.cache import ArraysCache
    from .models.device_offset_kv import DeviceOffsetKVCache

    planes = []
    for cache in lane.cache:
        if type(cache) is DeviceOffsetKVCache:
            planes.append(("kv", cache, cache.pos, int(cache.bound), int(cache.offset),
                           cache._exact, cache._last_write, cache.read_base))
        elif type(cache) is ArraysCache:
            planes.append(("rda", cache, list(cache.cache), cache.lengths, cache.left_padding))
        else:
            raise RuntimeError(f"a pipelined lane has no undo form for {type(cache).__name__}")
    for ring in lane.draft_cache:
        if type(ring) is not DeviceRingKVCache:
            raise RuntimeError("a pipelined lane's draft context must be device rings")
        # ``slot_pos`` is written in place (``slot_pos[slots] = ...``): keep an
        # alias of its current buffer, which the next write then copies.
        planes.append(("ring", ring, owner.mx.stop_gradient(ring.slot_pos), ring.pos,
                       ring.pending_count, int(ring.bound), ring.keys is None))
    return _Record(
        lane=lane, planes=planes, host=_host_copy(lane),
        refs=(lane.history, lane.ready, lane.cache, lane.draft_cache, lane.tail),
        history=len(lane.history), stats=owner._snapshot_scheduler_stats(),
        ready=len(lane.ready),
    )


def refresh(rec):
    """Re-take the host half after the round ahead of this one is booked."""
    rec.host = _host_copy(rec.lane)
    rec.history = len(rec.lane.history)
    rec.ready = len(rec.lane.ready)
    rec.refs = (rec.lane.history, rec.lane.ready, *rec.refs[2:])


def restore(owner, rec, *, host=False, stats=False, settle=True):
    """Undo a dispatch.

    A dispatch changes no host field of the lane but ``tail`` (its counters
    are deferred), so an abort or a discard restores the cache planes and the
    tail only: host changes made since (``disable_speculation``, the
    scheduler's service epoch) stay. ``host=True`` (a failed booking) also
    puts back the host fields captured at this poll's entry.
    """
    lane = rec.lane
    for plane in rec.planes:
        kind, cache = plane[0], plane[1]
        if kind == "kv":
            _, _, pos, bound, offset, exact, last, read_base = plane
            cache.pos, cache.bound, cache.offset, cache._exact = pos, bound, offset, exact
            cache._last_write, cache.read_base = last, read_base
        elif kind == "rda":
            _, _, state, lengths, left_padding = plane
            if cache.speculating:
                cache.stop_speculation()
            cache.cache = list(state)
            cache.lengths, cache.left_padding = lengths, left_padding
        else:
            _, _, slot_pos, pos, pending, bound, empty = plane
            cache.slot_pos, cache.pos, cache.pending_count, cache.bound = slot_pos, pos, pending, bound
            if empty:
                # The undone round wrote the ring's first rows: none were
                # committed before it.
                cache.keys = cache.values = None
    history, ready, cache, draft_cache, tail = rec.refs
    if host:
        lane.__dict__.clear()
        lane.__dict__.update(copy.copy(rec.host))
        del history[rec.history:]
        # Responses already drained by serving stay drained; ones booked since
        # the record are dropped.
        while len(ready) > rec.ready:
            ready.pop()
        lane.history, lane.ready = history, ready
    lane.cache, lane.draft_cache, lane.tail = cache, draft_cache, tail
    if stats:
        owner.scheduler_stats = rec.stats
    if settle:
        settle_exact(lane)


def settle_exact(lane):
    """Tighten every KV bound to the lane's booked length (nothing in flight)."""
    from .models.device_offset_kv import DeviceOffsetKVCache

    length = getattr(lane, "device_commit_length", None)
    if length is None:
        raise RuntimeError("a device-commit lane lost its committed length")
    for cache in lane.cache:
        if type(cache) is DeviceOffsetKVCache and cache.device_mode:
            cache.settle(int(length), 0)


# -- eligibility ----------------------------------------------------------------------

def skip_reason(owner, cohort):
    if len(cohort) != 1:
        return "width"
    lane = cohort[0]
    if lane.processors:
        # DC-6: declared presence penalties run on the device history; any
        # other processor reads the host history and keeps the serial round.
        from .speculative_device_presence import presence_contracts

        device_round = getattr(owner, "device_round", None)
        if (presence_contracts(lane.processors) is None or device_round is None
                or not device_round._processors_ok(lane)):
            return "processors"
    composed = getattr(owner.draft, "draft_lazy_composed", False)
    if composed or getattr(owner.draft, "requires_proposal_score_states", False):
        # DC-6 ``lagged_composition``: a lazily composed drafter is pipelined,
        # its following round planned from the lagged host state.
        policy = getattr(getattr(owner, "device_round", None), "policy", None)
        if not (composed and getattr(policy, "lagged_composition", False)):
            return "composition"
    if getattr(owner, "round_timing", False):
        return "round_timing"
    if (getattr(owner.model, "device_commit_kv_position_free", False) is not True
            and getattr(owner.model, "device_commit_kv_device_position", False) is not True):
        # A following round is verified while its KV bound still has slack,
        # and ``DeviceOffsetKVCache.offset`` is that bound: a layer that takes
        # its RoPE position from ``offset`` would rotate at the wrong position.
        # A target declares either no positional encoding over its device KV
        # (Raon's NoPE latent) or that every RoPE layer reads the device fill
        # (``DeviceOffsetKVCache.rope_offset``: Qwen3.8's full attention).
        return "position_encoding"
    if float(lane.sampling.get("sampling_temp", 0)) != 0.0:
        return "sampled"
    live = [candidate for candidate in owner.lanes.values() if not candidate.cancelled]
    if len(live) != 1 or live[0] is not lane:
        return "live_lanes"
    return None


def _follow_depth(owner):
    depth = int(owner.num_draft)
    return min(depth, int(getattr(owner.draft, "num_draft", depth)))


# -- dispatch and booking ---------------------------------------------------------------

@dataclass
class Inflight:
    lane: Any
    rec: _Record
    plan: Any
    logits: Any
    features: Any
    transaction: Any
    commit: Any
    depth: int
    count: int  # drafted tokens (RNG draws, budget, packet)
    rows: int   # verify rows written (read base, settle, book_round): count + 1, or the menu width
    stats_delta: dict
    lane_delta: dict
    discarded: bool = False


def _stop_aware_count(owner, plan, drafts, count):
    """``min(accepted + 1, first stop + 1)`` on the device: the serial ``consumed``."""
    mx = owner.mx
    accepted = plan.accepted.reshape(())
    index = mx.arange(count + 1, dtype=mx.int32)
    padded = mx.concatenate([drafts.reshape(-1)[:count].astype(mx.int32),
                             mx.zeros((1,), dtype=mx.int32)])
    emitted = mx.where(index < accepted, padded, plan.last.reshape(()))
    stops = sorted(int(t) for t in owner.stops)
    used = accepted + 1
    if stops:
        hit = mx.any(emitted[:, None] == mx.array(stops, dtype=mx.int32)[None, :], axis=-1)
        hit = hit & (index <= accepted)
        first = mx.where(mx.any(hit), mx.argmax(hit.astype(mx.int32)), accepted)
        used = mx.minimum(used, first + 1)
    return used.reshape(1).astype(mx.int32)


def dispatch(device_round, owner, lane, depth, anchor=None, inflight_rows=None, previous=None):
    """Queue one device round of ``depth`` drafts; nothing is read.

    ``inflight_rows``: the verify rows of the round still in flight ahead of
    this one (None: none), for the shared read base (``lagged_width``).
    ``previous``: that round (``Inflight``), whose unread tokens a processed
    lane's history needs (DC-6 ``DeviceHistory``).
    """
    from .hybrid_verify_rows import DeviceCommitTransaction
    from .speculative_device_commit import lagged_base, set_read_base

    mx = owner.mx
    rec = record(owner, lane)
    before = {name: copy.deepcopy(getattr(lane, name, None) or {}) for name in _LANE_COUNTERS}
    composed = bool(getattr(owner.draft, "draft_lazy_composed", False))
    # A composed drafter books ``composition_stats`` when it plans: deferred
    # to this round's booking like the other counters.
    composition_before = dict(owner.draft.composition_stats) if composed else None
    try:
        input_vocab = owner._target_token_vocab_size
        if anchor is not None:
            anchor_array = mx.clip(anchor.reshape(1).astype(mx.int32), 0, input_vocab - 1)
        elif getattr(device_round, "fixed_shape", False):
            # The same ops as a device anchor (a no-op on a booked token): one
            # graph whether or not a round is in flight ahead of this one.
            anchor_array = mx.clip(mx.array([int(lane.anchor)], dtype=mx.int32), 0, input_vocab - 1)
        else:
            anchor_array = mx.array([int(lane.anchor)], dtype=mx.int32)
        history = extra = None
        pad_block = (getattr(device_round, "fixed_shape", False) and depth <= int(owner.num_draft)
                     and getattr(owner.draft, "draft_lazy_pad", False))
        if lane.processors:
            from .speculative_device_presence import DeviceHistory, presence_contracts, presence_store

            # ``fixed_shape``: the window's host part has one shape every round.
            store = presence_store(lane) if getattr(device_round, "fixed_shape", False) else None
            if previous is not None:
                history = DeviceHistory(mx, lane.history, lane.anchor,
                                        previous.plan.row.drafts, previous.commit - 1,
                                        previous.plan.last, store=store,
                                        stats=owner.scheduler_stats)
            elif pad_block and presence_contracts(lane.processors):
                # A round with nothing in flight ahead of it (the first after
                # entry, one near a budget end) builds the following round's
                # graph (slice 1a-5, one round kind).
                history = DeviceHistory.first(mx, lane.history, lane.anchor, owner.num_draft,
                                              store=store, stats=owner.scheduler_stats)
            else:
                history = list(lane.history)
            extra = {"logits_processors": lane.processors, "processor_history": history}
        if composed:
            # The source is planned from the host state: exact for a first
            # round, one round behind for a following one (DC-6).
            extra = dict(extra or {})
            extra.update(planning_anchor=int(lane.anchor), planning_history=list(lane.history),
                         proposal_score_state=lane.proposal_score_counts,
                         proposal_trust_state=lane.proposal_trust_states, cohort_width=1)
        if pad_block:
            # The serial fixed-shape round's drafter graph (``DeviceRound._pad_depth``;
            # a pipelined round is full depth, so nothing is padded off).
            extra = dict(extra or {})
            extra["pad_to"] = int(owner.num_draft)
        drafts = owner.draft.draft_lazy(anchor_array, lane.tail, lane.draft_cache, depth, None,
                                        **(extra or {}))
        composition = device_round._composition_row(owner.draft) if composed else None
        drafts = drafts.reshape(-1).astype(mx.int32)
        count = int(drafts.shape[0])
        if pad_block:
            # The proposals are the first ``depth`` (a composed backend's cap included).
            count = depth if not composed else min(depth, int(owner.draft.backend_proposal_length))
            if count < int(drafts.shape[0]):
                # Deferred with the dispatch counters; the lane count at booking.
                _bump(owner.scheduler_stats, "external_device_round_fixed_shape_padded_drafts")
        fixed = getattr(device_round, "fixed_shape", False)
        menu = int(owner.num_draft)
        if fixed and count > menu:
            raise RuntimeError("a pipelined round drafted past the fixed-shape menu")
        rows = menu + 1 if fixed else count + 1
        drafted = int(drafts.shape[0])  # ``count``, or the menu block when padded
        padded = (mx.concatenate([drafts, mx.zeros((rows - 1 - drafted,), dtype=mx.int32)])
                  if rows - 1 > drafted else drafts)
        inputs = mx.concatenate([anchor_array, mx.clip(padded, 0, input_vocab - 1)])[None]
        transaction = DeviceCommitTransaction(lane.cache, [rows], lagged=True)
        set_read_base(lane, lagged_base(lane, inflight_rows))
        logits, features = owner._cohort_target_forward(
            inputs, transaction, [lane], site="standard_verify"
        )
        row = SimpleNamespace(lane=lane, count=count, host=None, drafts=drafts, q=None,
                              ratio=False, seed=None, source="device_round", composition=composition)
        if lane.processors and not isinstance(history, list):
            row.device_history = history  # verify rows read the same device history
        if fixed:
            plan = device_round._lane_plan_fixed(owner, row, logits[0, :rows])
            # Deferred with the other dispatch counters (stats_delta); the
            # lane count is taken at booking, so a discarded round leaves none.
            _bump(owner.scheduler_stats, "external_device_round_fixed_shape_rounds")
        else:
            plan = device_round._lane_plan(owner, row, logits[0, : count + 1])
        commit = _stop_aware_count(owner, plan, drafts, count)
        transaction.commit_on_device(commit)
        for ring in lane.draft_cache:
            ring.pending_count = commit
        lane.tail = features[0:1]
        # Under ``fixed_shape`` the round also schedules its own commit side, so
        # a round's encoded work does not depend on whether one follows it.
        mx.async_eval(plan.packet, commit,
                      *((lane.tail, *transaction.device_state(),
                         *(a for ring in lane.draft_cache for a in getattr(ring, "state", ())
                           if a is not None)) if fixed else ()))
    except BaseException:
        restore(owner, rec, stats=True, settle=False)
        for name, value in before.items():
            setattr(lane, name, value)
        if composition_before is not None:
            owner.draft.composition_stats.clear()
            owner.draft.composition_stats.update(composition_before)
        raise
    stats_delta = _delta(owner.scheduler_stats, rec.stats)
    owner.scheduler_stats = rec.stats
    lane_delta = {}
    for name, value in before.items():
        lane_delta[name] = _delta(getattr(lane, name, None) or {}, value)
        setattr(lane, name, value)
    if composition_before is not None:
        lane_delta[_COMPOSITION] = _delta(owner.draft.composition_stats, composition_before)
        owner.draft.composition_stats.clear()
        owner.draft.composition_stats.update(composition_before)
    _bump(owner.scheduler_stats, "external_device_round_pipeline_dispatches")
    return Inflight(lane, rec, plan, logits, features, transaction, commit, depth, count, rows,
                    stats_delta, lane_delta)


def book(device_round, owner, inflight, following):
    """Read ``inflight``'s packet and book it; returns the round left in flight."""
    from .external_speculative import HostDraftRow
    from .proposal_providers import PointMassLaw

    import numpy as np

    mx = owner.mx
    lane = inflight.lane
    mx.eval(inflight.plan.packet, inflight.commit)
    if following is not None:
        # The serial engine never runs a round after a finish: discard the
        # following round before this one books anything (its record holds
        # the lane's host fields as they are now).
        packet = [int(v) for v in np.asarray(inflight.plan.packet).tolist()]
        accepted, last = packet[0], packet[2]
        emitted = [*packet[3:3 + inflight.count][:accepted], last]
        for j, token in enumerate(emitted):
            if token in owner.stops:
                emitted = emitted[: j + 1]
                break
        if (any(token in owner.stops for token in emitted)
                or lane.generated + len(emitted) >= lane.maximum):
            restore(owner, following.rec, settle=False)
            following.discarded = True
            following = None
            _bump(owner.scheduler_stats, "external_device_round_pipeline_discards")
    # Serial order of the lane RNG: this round's draft uniforms, then its
    # verify draws (``_finish``).
    if getattr(owner.draft, "draft_lazy_consumes_rng", False):
        for _ in range(inflight.count):
            lane.rng.uniform()
    decisions = device_round._finish(owner, [inflight.plan], inflight.logits)
    decision = decisions[0]
    used = min(decision.accepted + 1, len(decision.emitted))
    if int(inflight.commit.item()) != used:
        raise RuntimeError("pipelined device commit differs from the booked count")
    if following is not None and (any(token in owner.stops for token in decision.emitted)
                                  or lane.generated + len(decision.emitted) >= lane.maximum):
        raise RuntimeError("a pipelined round finished after its packet read said it would not")
    _apply(owner.scheduler_stats, inflight.stats_delta)
    for name, delta in inflight.lane_delta.items():
        if name == _COMPOSITION:
            _apply(owner.draft.composition_stats, delta)
            continue
        counts = getattr(lane, name, None)
        if counts is None:
            counts = {}
            setattr(lane, name, counts)
        _apply(counts, delta)
    stats = owner.scheduler_stats
    for key in ("external_device_round_rounds", "external_device_round_lazy_draft_lanes",
                "external_device_round_host_reads", "external_device_round_decide_rounds",
                "external_device_round_device_commit_rounds",
                "external_device_round_pipeline_rounds"):
        _bump(stats, key)
    stats["draft_max_width"] = max(stats.get("draft_max_width", 0), 1)
    for key in ("lazy_draft_rounds", "rounds", "device_commit_rounds", "pipeline_rounds"):
        device_round._lane_count(lane, key)
    if getattr(device_round, "fixed_shape", False):
        device_round._lane_count(lane, "fixed_shape_rounds")
        if inflight.count < int(inflight.plan.row.drafts.shape[0]):
            device_round._lane_count(lane, "fixed_shape_padded_drafts")
    tokens = inflight.plan.drafts_host
    blocks = [HostDraftRow(tokens, [PointMassLaw(t) for t in tokens], 1)]
    for key, value in (inflight.plan.row.composition or {}).items():
        # ``_propose``'s composed-row attributes: ``_commit`` learns the lagged
        # score and trust feedback from them.
        setattr(blocks[0], key, value)
    transaction = inflight.transaction
    transaction.expected_used = used
    transaction.settle_length = int(lane.device_commit_length) + used
    transaction.settle_pending = 0 if following is None else following.rows
    from .speculative_device_commit import book_round

    book_round(lane, inflight.rows, used)
    owner._commit(
        [lane], decisions, inflight.features, blocks=blocks, transaction=transaction,
        deferred_context=False, deferred_context_uids=set(),
    )
    if following is not None:
        refresh(following.rec)
    return following


def abort(owner):
    """Discard the round in flight (never read, never published)."""
    inflight = getattr(owner, "_device_inflight", None)
    owner._device_inflight = None
    if inflight is None:
        return False
    restore(owner, inflight.rec)
    _bump(owner.scheduler_stats, "external_device_round_pipeline_aborts")
    return True


def serve(device_round, owner, cohort, depth):
    """Serve this poll's round pipelined; False leaves it to the serial round."""
    inflight = getattr(owner, "_device_inflight", None)
    reason = skip_reason(owner, cohort)
    if (reason is None and depth != _follow_depth(owner)
            and not (getattr(device_round, "fixed_shape", False) and 1 <= depth <= int(owner.num_draft)
                     and getattr(owner.draft, "draft_lazy_pad", False))):
        # A budget end is near: the serial round decides it. Under ``fixed_shape``
        # a padding drafter keeps it pipelined (the menu block, one round kind).
        reason = "short_depth"
    if (reason is None and getattr(device_round, "fixed_shape", False)
            and depth > int(owner.num_draft)):
        # A round deeper than the fixed-shape menu (unreachable while depths are
        # capped at num_draft): the serial round counts it as wider_than_menu.
        reason = "wider_than_menu"
    if reason is not None:
        abort(owner)
        skips = owner.scheduler_stats.setdefault(SKIPS, {})
        skips[reason] = skips.get(reason, 0) + len(cohort)
        return False
    lane = cohort[0]
    if inflight is not None and (inflight.lane is not lane or inflight.depth != depth):
        abort(owner)
        inflight = None
    owner._device_inflight = None
    if inflight is not None:
        # The failure path's host restore: the lane as it is at this poll.
        refresh(inflight.rec)
    stats = owner._snapshot_scheduler_stats()
    composition = (dict(owner.draft.composition_stats)
                   if getattr(owner.draft, "draft_lazy_composed", False) else None)
    owner._open = True
    following = None
    try:
        if inflight is None:
            inflight = dispatch(device_round, owner, lane, depth)
        if lane.maximum - lane.generated - (inflight.count + 1) - 1 >= _follow_depth(owner):
            try:
                following = dispatch(device_round, owner, lane, depth, anchor=inflight.plan.last,
                                     inflight_rows=inflight.rows, previous=inflight)
                _bump(owner.scheduler_stats, "external_device_round_pipeline_followers")
            except Exception as error:  # noqa: BLE001 - restored; the next serial round re-raises it
                from .proposal_composition import LazyDecline

                following = None
                # A lagged plan that picked another source (copy, MTP, trusted
                # PLD) is a planned decline, not a failure (DC-6).
                _bump(owner.scheduler_stats,
                      "external_device_round_pipeline_follower_declines"
                      if isinstance(error, LazyDecline)
                      else "external_device_round_pipeline_dispatch_failures")
        following = book(device_round, owner, inflight, following)
    except BaseException:
        # Newest first: the following round's record sits on top of this one's.
        if following is not None and not following.discarded:
            restore(owner, following.rec, settle=False)
        if inflight is not None:
            restore(owner, inflight.rec, host=True)
        owner.scheduler_stats = stats
        if composition is not None:
            # ``book`` applied this round's deferred composition counts.
            owner.draft.composition_stats.clear()
            owner.draft.composition_stats.update(composition)
        raise
    finally:
        owner._open = False
    owner._device_inflight = following
    return True


__all__ = ["COUNTERS", "SKIPS", "VERSION", "abort", "serve"]
