"""Presence-penalty windows on the device for pipelined rounds (spec-round DC-6).

Original mlx2 implementation; design in
docs/research/2026-11-25-gpu-resident-spec-rounds-design.md section 4.9 (DC-6).
Default off: used only by ``speculative_device_pipeline`` when a lane's
processors all declare the presence contract (``presence_window``) and the
``device_round.processors`` switch is on.

A pipelined round R' is dispatched before round R is read, so the host does not
know R's committed tokens. The presence penalty (``sample_utils.
make_presence_penalty``) needs only the last ``context_size`` tokens of the
history at or after ``generation_start``. R' sees the history

    H + [anchor_R] + drafts_R[:consumed_R - 1] + [last_R] + drafts_R'[:j]

where H (booked through R-1) and anchor_R are host-known and the rest is on
the device. ``DeviceHistory`` lays that sequence out with absolute positions,
masks the drafts R did not commit, and selects the window on the device; the
penalty is then ``logits - penalty`` on exactly the window's tokens, the same
float32 operation the host processor's ``logits[:, tokens] -= penalty``
performs, so the processed law is bitwise the serial round's.

Readers: ``speculative_device_round._row_value`` (verify rows, one at a time),
``speculative_device_round._fixed_values`` (all menu rows at once, slice 1a-5,
through ``apply_rows``) and DFlash2's ``_draft_lazy_processed``.
"""

from __future__ import annotations


def presence_contracts(processors):
    """``[(penalty, context_size, generation_start)]`` per processor, or None
    when any processor is not a declared presence penalty."""
    contracts = []
    for processor in processors or ():
        window = getattr(processor, "presence_window", None)
        if window is None or len(window) != 3:
            return None
        _penalty, context, start = window
        # The host slices ``tokens[start:][-context:]`` with Python semantics;
        # negative values count from the end there, which the absolute-position
        # window does not model: such lanes keep the serial round.
        if int(context) < 0 or (start is not None and int(start) < 0):
            return None
        contracts.append(window)
    return contracts or None


def presence_store(lane):
    """The lane's fixed-shape presence state (``DeviceHistory(store=...)``), created on use.

    Shared, not copied, by the lane's recovery snapshots
    (``ExternalDraftBatchGenerator._freeze_lane``); ``DeviceHistory._seen``
    validates the covered prefix against the history on every use, so a
    restored lane rebuilds what no longer matches."""
    store = getattr(lane, "device_presence_store", None)
    if store is None:
        store = {}
        lane.device_presence_store = store
    return store


class DeviceHistory:
    """The processor history of a round dispatched before the round ahead of it is read."""

    def __init__(self, mx, host_history, anchor, prev_drafts, prev_kept, last, *, store=None,
                 stats=None):
        self.mx = mx
        self.stats = stats  # the engine's scheduler stats: seen-map rebuilds and gaps
        # ``store`` (``fixed_shape``, slice 1a-5): a lane-owned dict that makes the
        # host part one shape every round (``_host_part``); None keeps the exact
        # host window, whose length grows with the history.
        self.store = store
        self.host = [int(t) for t in host_history]  # a copy: the lane's list grows later
        self.anchor = int(anchor)
        self.prev_drafts = prev_drafts.reshape(-1).astype(mx.int32)
        # Drafts of R that R committed (consumed_R - 1), device int32 [1].
        self.prev_kept = prev_kept.reshape(1).astype(mx.int32)
        self.last = last.reshape(1).astype(mx.int32)
        # Host part (tokens, positions) per window: it does not depend on the
        # tail, so it is built once per round, not per verify row and draft.
        self._host_parts = {}

    @classmethod
    def first(cls, mx, host_history, anchor, menu, *, store=None, stats=None):
        """The history of a round with no unread round ahead of it (slice 1a-5).

        ``H + [anchor] + tail``, laid out as a following round's history so a
        ``fixed_shape`` round builds one graph whether or not a round is in
        flight: ``menu`` masked previous drafts and ``last = anchor`` at the
        anchor's own position (``prev_kept = -1``). The duplicate shares the
        anchor's position, and the penalty marks distinct tokens, so the
        window and the law are the host processor's."""
        import numpy as np

        anchor = int(anchor)
        # Host-built constants and ``0 - 1`` (a following round's ``commit - 1``):
        # the same ops in the round as a following round's history.
        return cls(mx, host_history, anchor, mx.array(np.zeros((int(menu),), dtype=np.int32)),
                   mx.array([0], dtype=mx.int32) - 1, mx.array([anchor], dtype=mx.int32),
                   store=store, stats=stats)

    # -- the host part -------------------------------------------------------------
    def _host_part(self, contract, vocab):
        """``(tokens, positions, valid, seen)``: the window's host-known part.

        Without a store: ``host[first:] + [anchor]``, exact, its length growing
        until the window is full (whole history: always). With a store
        (``fixed_shape``) the shape is the same every round:

        * ``context > 0``: the same tokens padded to ``context + 1`` slots
          (``valid`` false, position -1, below any window start);
        * ``context == 0`` (the whole history from ``start``): only the anchor,
          plus ``seen``, a ``[vocab]`` bool map of ``host[start:]`` kept on the
          device in the store and advanced by one fixed-size scatter of the
          tokens booked since its last use (``_seen``).

        The penalty marks distinct tokens, so the masks are the same sets."""
        import numpy as np

        mx = self.mx
        _penalty, context, start = contract
        start = 0 if start is None else int(start)
        n_host = len(self.host)
        first = max(start, n_host - context) if context > 0 else start
        first = min(max(first, 0), n_host)
        key = (first, context, start, vocab)
        cached = self._host_parts.get(key)
        if cached is not None:
            return cached
        if self.store is None:
            tokens = mx.array(self.host[first:] + [self.anchor], dtype=mx.int32)
            cached = (tokens, mx.arange(first, n_host + 1, dtype=mx.int32),
                      mx.ones(tokens.shape, dtype=mx.bool_), None)
        elif context > 0:
            held = self.host[first:] + [self.anchor]
            pad = int(context) + 1 - len(held)
            cached = (mx.array(np.array(held + [0] * pad, dtype=np.int32)),
                      mx.array(np.array(list(range(first, n_host + 1)) + [-1] * pad, dtype=np.int32)),
                      mx.array(np.array([True] * len(held) + [False] * pad)), None)
        else:
            cached = (mx.array(np.array([self.anchor], dtype=np.int32)),
                      mx.array(np.array([n_host], dtype=np.int32)),
                      mx.array(np.array([True])), self._seen(start, vocab))
        self._host_parts[key] = cached
        return cached

    def _seen(self, start, vocab):
        """``[vocab]`` bool: the tokens of ``host[start:]``, from the lane's store.

        The store keeps the covered host prefix (a copy) and an int32 count map.
        A history that extends the covered prefix adds its new tokens in chunks
        of ``chunk`` slots (padding adds 0), always at least one chunk, so a
        round adds one fixed-shape scatter. Any other history (a restore, a new
        request) rebuilds the map from scratch."""
        import numpy as np

        mx = self.mx
        # A round books at most menu + 1 tokens; fixed per lane at its first use,
        # so every later update is the same scatter.
        chunk = int(self.store.setdefault("chunk", int(self.prev_drafts.shape[0]) + 2))
        key = ("seen", start, vocab)
        n_host = len(self.host)
        entry = self.store.get(key)
        covered = 0
        if entry is not None:
            prefix, counts = entry
            if len(prefix) <= n_host and self.host[: len(prefix)] == prefix:
                covered = len(prefix)
            else:
                entry = None
        # Clipped as the exact window's scatter clips (a drafter vocabulary
        # smaller than a booked target id marks ``vocab - 1``); never out of range.
        def ids(tokens):
            return mx.array(np.clip(np.array(tokens, dtype=np.int64), 0, vocab - 1).astype(np.int32))

        if entry is None:
            counts = mx.zeros((vocab,), dtype=mx.int32)
            begin = min(max(start, 0), n_host)
            if begin < n_host:  # the rebuild: one scatter of the whole window
                counts = counts.at[ids(self.host[begin:])].add(1)
            covered = n_host
            self._bump("external_device_round_presence_seen_rebuilds")
        new = self.host[max(covered, start):]
        if len(new) > chunk:
            # A gap wider than one round (a host round in between): a
            # different update graph this round, counted.
            self._bump("external_device_round_presence_seen_multi_chunk")
        while True:
            part, new = new[:chunk], new[chunk:]
            pad = chunk - len(part)
            counts = counts.at[ids(part + [0] * pad)].add(
                mx.array(np.array([1] * len(part) + [0] * pad, dtype=np.int32)))
            if not new:
                break
        self.store[key] = (self.host, counts)  # self.host is already a private copy
        return counts > 0

    def _bump(self, key):
        if self.stats is not None:
            self.stats[key] = self.stats.get(key, 0) + 1

    def _window(self, contract, tail, vocab):
        mx = self.mx
        _penalty, context, start = contract
        start = 0 if start is None else int(start)
        n_host = len(self.host)
        k = int(self.prev_drafts.shape[0])
        j = 0 if tail is None else int(tail.shape[0])
        kept = self.prev_kept
        host_tokens, host_pos, host_valid, seen = self._host_part(contract, vocab)
        draft_pos = n_host + 1 + mx.arange(k, dtype=mx.int32)
        last_pos = n_host + 1 + kept
        parts = [host_tokens, self.prev_drafts, self.last]
        pos = [host_pos, draft_pos, last_pos]
        valid = [host_valid, mx.arange(k, dtype=mx.int32) < kept, mx.ones((1,), dtype=mx.bool_)]
        if j:
            parts.append(tail.reshape(-1).astype(mx.int32))
            pos.append(last_pos + 1 + mx.arange(j, dtype=mx.int32))
            valid.append(mx.ones((j,), dtype=mx.bool_))
        tokens = mx.concatenate(parts)
        positions = mx.concatenate(pos)
        total = n_host + 2 + kept + j  # sequence length, device int32 [1]
        lo = mx.array([start], dtype=mx.int32)
        if context > 0:
            lo = mx.maximum(lo, total - context)
        return tokens, mx.concatenate(valid) & (positions >= lo), seen

    def apply(self, processors, tail, logits):
        """The lane's presence processors on ``self + tail`` (lazy, no host read)."""
        mx = self.mx
        contracts = presence_contracts(processors)
        if contracts is None:
            raise ValueError("a device history serves declared presence penalties only")
        value = logits
        vocab = int(value.shape[-1])
        for contract in contracts:
            tokens, mask, seen = self._window(contract, tail, vocab)
            # Even a masked, unbooked proposal supplies a scatter index.
            tokens = mx.clip(tokens, 0, vocab - 1)
            hits = mx.zeros((vocab,), dtype=mx.int32).at[tokens].add(mask.astype(mx.int32)) > 0
            if seen is not None:
                hits = hits | seen
            # ``logits[:, tokens] -= penalty`` on the window's distinct tokens.
            value = mx.where(hits[None, :], value - contract[0], value)
        return value


    def apply_rows(self, processors, drafts, count, values):
        """``apply`` for every verify row of a fixed-shape round at once (slice 1a-5).

        ``values`` is ``[rows, vocab]`` float32; row ``j`` takes the tail
        ``drafts[:min(j, count)]`` (``drafts`` padded to ``rows - 1``). Each row's
        window is the one ``apply`` selects for that tail, built from one token
        layout and a ``[rows, layout]`` mask instead of one graph per row; the
        penalty is the same ``logits - penalty`` on the same distinct tokens.
        """
        mx = self.mx
        contracts = presence_contracts(processors)
        if contracts is None:
            raise ValueError("a device history serves declared presence penalties only")
        rows, vocab = int(values.shape[0]), int(values.shape[-1])
        tail = drafts.reshape(-1).astype(mx.int32)
        menu = int(tail.shape[0])
        if rows != menu + 1 or not 0 <= int(count) <= menu:
            raise ValueError("apply_rows needs rows == len(drafts) + 1 and 0 <= count <= len(drafts)")
        depth = mx.minimum(mx.arange(rows, dtype=mx.int32), int(count))  # tail tokens per row
        for contract in contracts:
            _penalty, context, start = contract
            start = 0 if start is None else int(start)
            n_host = len(self.host)
            k = int(self.prev_drafts.shape[0])
            kept = self.prev_kept
            host_tokens, host_pos, host_valid, seen = self._host_part(contract, vocab)
            last_pos = n_host + 1 + kept
            tokens = mx.concatenate([host_tokens, self.prev_drafts, self.last, tail])
            positions = mx.concatenate([host_pos, n_host + 1 + mx.arange(k, dtype=mx.int32),
                                        last_pos, last_pos + 1 + mx.arange(menu, dtype=mx.int32)])
            fixed = mx.concatenate([host_valid, mx.arange(k, dtype=mx.int32) < kept,
                                    mx.ones((1,), dtype=mx.bool_)])
            head = int(fixed.shape[0])
            tail_valid = mx.arange(menu, dtype=mx.int32)[None, :] < depth[:, None]
            valid = mx.concatenate([mx.broadcast_to(fixed[None, :], (rows, head)), tail_valid], axis=1)
            total = n_host + 2 + kept + depth  # each row's sequence length
            lo = mx.full((rows,), start, dtype=mx.int32)
            if context > 0:
                lo = mx.maximum(lo, total - context)
            mask = valid & (positions[None, :] >= lo[:, None])
            row_ids = mx.broadcast_to(mx.arange(rows, dtype=mx.int32)[:, None], mask.shape)
            tokens = mx.clip(tokens, 0, vocab - 1)
            hits = mx.zeros((rows, vocab), dtype=mx.int32).at[
                row_ids.reshape(-1), mx.broadcast_to(tokens[None, :], mask.shape).reshape(-1)
            ].add(mask.astype(mx.int32).reshape(-1)) > 0
            if seen is not None:
                hits = hits | seen[None, :]
            values = mx.where(hits, values - contract[0], values)
        return values


__all__ = ["DeviceHistory", "presence_contracts", "presence_store"]
