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


class DeviceHistory:
    """The processor history of a round dispatched before the round ahead of it is read."""

    def __init__(self, mx, host_history, anchor, prev_drafts, prev_kept, last):
        self.mx = mx
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
    def first(cls, mx, host_history, anchor, menu):
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
                   mx.array([0], dtype=mx.int32) - 1, mx.array([anchor], dtype=mx.int32))

    def _window(self, contract, tail):
        mx = self.mx
        _penalty, context, start = contract
        start = 0 if start is None else int(start)
        n_host = len(self.host)
        # Host positions below this can never reach the window.
        first = max(start, n_host - context) if context > 0 else start
        first = min(max(first, 0), n_host)
        k = int(self.prev_drafts.shape[0])
        j = 0 if tail is None else int(tail.shape[0])
        kept = self.prev_kept
        cached = self._host_parts.get(first)
        if cached is None:
            cached = self._host_parts[first] = (
                mx.array(self.host[first:] + [self.anchor], dtype=mx.int32),
                mx.arange(first, n_host + 1, dtype=mx.int32),
            )
        host_tokens, host_pos = cached
        draft_pos = n_host + 1 + mx.arange(k, dtype=mx.int32)
        last_pos = n_host + 1 + kept
        parts = [host_tokens, self.prev_drafts, self.last]
        pos = [host_pos, draft_pos, last_pos]
        valid = [mx.ones(host_tokens.shape, dtype=mx.bool_),
                 mx.arange(k, dtype=mx.int32) < kept, mx.ones((1,), dtype=mx.bool_)]
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
        return tokens, mx.concatenate(valid) & (positions >= lo)

    def apply(self, processors, tail, logits):
        """The lane's presence processors on ``self + tail`` (lazy, no host read)."""
        mx = self.mx
        contracts = presence_contracts(processors)
        if contracts is None:
            raise ValueError("a device history serves declared presence penalties only")
        value = logits
        vocab = int(value.shape[-1])
        for contract in contracts:
            tokens, mask = self._window(contract, tail)
            hits = mx.zeros((vocab,), dtype=mx.int32).at[tokens].add(mask.astype(mx.int32)) > 0
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
            first = max(start, n_host - context) if context > 0 else start
            first = min(max(first, 0), n_host)
            k = int(self.prev_drafts.shape[0])
            kept = self.prev_kept
            cached = self._host_parts.get(first)
            if cached is None:
                cached = self._host_parts[first] = (
                    mx.array(self.host[first:] + [self.anchor], dtype=mx.int32),
                    mx.arange(first, n_host + 1, dtype=mx.int32),
                )
            host_tokens, host_pos = cached
            last_pos = n_host + 1 + kept
            tokens = mx.concatenate([host_tokens, self.prev_drafts, self.last, tail])
            positions = mx.concatenate([host_pos, n_host + 1 + mx.arange(k, dtype=mx.int32),
                                        last_pos, last_pos + 1 + mx.arange(menu, dtype=mx.int32)])
            fixed = mx.concatenate([mx.ones(host_tokens.shape, dtype=mx.bool_),
                                    mx.arange(k, dtype=mx.int32) < kept, mx.ones((1,), dtype=mx.bool_)])
            head = int(fixed.shape[0])
            tail_valid = mx.arange(menu, dtype=mx.int32)[None, :] < depth[:, None]
            valid = mx.concatenate([mx.broadcast_to(fixed[None, :], (rows, head)), tail_valid], axis=1)
            total = n_host + 2 + kept + depth  # each row's sequence length
            lo = mx.full((rows,), start, dtype=mx.int32)
            if context > 0:
                lo = mx.maximum(lo, total - context)
            mask = valid & (positions[None, :] >= lo[:, None])
            row_ids = mx.broadcast_to(mx.arange(rows, dtype=mx.int32)[:, None], mask.shape)
            hits = mx.zeros((rows, vocab), dtype=mx.int32).at[
                row_ids.reshape(-1), mx.broadcast_to(tokens[None, :], mask.shape).reshape(-1)
            ].add(mask.astype(mx.int32).reshape(-1)) > 0
            values = mx.where(hits, values - contract[0], values)
        return values


__all__ = ["DeviceHistory", "presence_contracts"]
