"""Next-token gating for the block-sparse kernels (experimental, opt-in, default off, unqualified).

Pierre (2026-11-19): "can the predicted next token be what opens up the gates for the next forward?"

Same-step gating (the existing law) decides step t's tile set inside step t's forward: the tile-gate kernel
reads the probe logits of an early gate layer, so the layers before it run with every tile open and the set
is known on the GPU only, mid-forward.  Next-token gating decides step t+1's set as the last op of step t,
from what step t has produced: its final hidden state (probe law) or its sampled token (token-prior law).
The set is then known one token ahead, so it can be handed to a prefetch hook (paging), and the gate
decision leaves the forward.

Modes (:data:`GATE_MODES`):

* ``same_step``          the existing law (the reference): the probe at gate layer v gives step t's set; layers
                         below v run with every tile open.
* ``next_token``         the predicted set, after hysteresis, gates every gated layer, layers below v included
                         (there is no probe in the forward).  Approximate: a tile the prediction missed stays
                         closed for that step, and the layers below v see other states than same-step's.
* ``next_token_verify``  exact against same-step: layers below v stay dense, the probe at v runs as in
                         same-step, and the gated layers use the same-step set.  The prediction only feeds the
                         prefetch hook (the tile set one token ahead).  With every tile resident this is
                         same-step plus the step-end op; it pays off only when tiles are paged.  A layer that
                         ran on the predicted (resident) set before the missing tiles arrived is repaired by
                         :func:`verified`: the rows whose predicted set differs from the same-step set are
                         recomputed with it and merged, bit-exact for that call because the per-row kernels
                         are row-invariant (a row's output depends only on its own open row and input).
                         Exactness covers one layer call with the same input; it cannot hold if the layers
                         below v were gated by the prediction, since v's state (and so the verify set) would
                         then differ from same-step's.  ``exact=False`` recomputes only rows that missed an
                         open.  Checked on the fp32 CPU emulator; the Metal check is written, not run.

Guards:

* hysteresis: a tile opens at once and closes only after ``hysteresis`` steps without a prediction;
* fail open: a probe box opens when its logit is above ``-fail_open_margin`` (sure-closed only), which is
  the same kernel with the bias shifted, so no new launch;
* prefill: the decode set starts as the union over the whole prompt (age 0).  The prefill's last op is
  :meth:`NextTokenGate.advance` from the last prompt position, like every decode step, so the union
  survives ``hysteresis`` steps beside the first prediction;
* speculative decoding: the verify pass of k draft positions uses the union of their predicted sets
  (:meth:`NextTokenGate.speculative_union`, one shared row, the batch-union law).

Copies: the state is one int32 age array [sequences, tiles] (FN: 416 tiles), replaced by one fused
elementwise update per step; no host synchronization; the prefetch hook receives the device array itself.
The recompute pass of verify mode runs on a ``topen`` whose rows that need no recompute are all zero, so
those rows read no weight (an all-closed row skips every chunk and group), and the merge is one ``where``.

States: implemented; not qualified; not selected by any route; never observed-used in serving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import mlx.core as mx
import numpy as np

from .ops import BlockSparseUnsupported, GateTables, tile_gates, union_open

GATE_MODES = ("same_step", "next_token", "next_token_verify")
DEFAULT_GATE_MODE = "same_step"
REQUEST_FIELD = "mlx2_amoeba_gate_mode"
RECEIPT_SCHEMA = "mlx2.amoeba-blocks-gate-mode.v1"
_CLOSED_AGE = 1 << 20                     # "never predicted"; ages saturate here (no int32 overflow)


class GateModeUnavailable(ValueError):
    """A request asked for a gate mode this route cannot run (HTTP 409, fail closed)."""

    http_status = 409
    code = "amoeba_gate_mode_unavailable"


def check_request_gate_mode(route_blocks, requested):
    """The gate mode a request may run, or None when it names none.

    ``route_blocks`` is the route's block-sparse installation (``adapter.amoeba_blocks``) or None.  A route
    without amoeba_blocks refuses every gate mode, including ``same_step``: it has no gates at all."""
    if requested is None:
        return None
    if not isinstance(requested, str) or requested not in GATE_MODES:
        raise ValueError(f"{REQUEST_FIELD} must be one of {GATE_MODES}")
    if route_blocks is None:
        raise GateModeUnavailable(f"{REQUEST_FIELD}={requested!r}: this route has no amoeba_blocks kernels")
    modes = tuple(getattr(route_blocks, "gate_modes", (DEFAULT_GATE_MODE,)))
    if requested not in modes:
        raise GateModeUnavailable(f"{REQUEST_FIELD}={requested!r}: this route runs {modes}")
    return requested


def receipt_fields(requested: str | None, stats: dict | None = None) -> dict:
    """Route receipt fields for a route with amoeba_blocks.

    ``amoeba_gate_mode_requested`` is the request's field (None: not asked).  ``amoeba_gate_mode`` is the mode
    the route reports it executed (``stats["executed_mode"]``, from the installation's own counters), or None
    when nothing was observed: a requested mode is never echoed as if it ran."""
    if requested is not None and requested not in GATE_MODES:
        raise ValueError(f"unknown gate mode {requested!r}")
    stats = dict(stats or {})
    executed = stats.pop("executed_mode", None)
    if executed is not None and executed not in GATE_MODES:
        raise ValueError(f"unknown executed gate mode {executed!r}")
    out = {"amoeba_gate_mode_requested": requested, "amoeba_gate_mode": executed,
           "amoeba_gate_observed": executed is not None, "amoeba_gate_schema": RECEIPT_SCHEMA,
           "amoeba_gate_qualified": False}
    if stats:
        out["amoeba_gate_stats"] = stats
    return out


APPROXIMATE_MODES = ("next_token",)      # cached state differs from same-step's: its own APCv2 scope


# ---------------------------------------------------------------------------
# Prefetch hook
# ---------------------------------------------------------------------------


class PrefetchHook(Protocol):
    def next_tiles(self, rows: list[int], topen: mx.array) -> None:
        """Called as the last op of step t with step t+1's open set (uint8 [len(rows), tiles], a device
        array; the hook decides whether and when to read it back)."""


class NullPrefetch:
    def next_tiles(self, rows, topen):
        return None


@dataclass
class RecordingPrefetch:
    """Keeps every call (rows, the device array itself); for tests and the bench."""

    calls: list = field(default_factory=list)

    def next_tiles(self, rows, topen):
        self.calls.append((list(rows), topen))


# ---------------------------------------------------------------------------
# Token prior: sampled token -> box open bits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenPrior:
    """uint8 [vocab, boxes]: box b opens for step t+1 when step t sampled a token whose fit-row open rate
    for b is above the threshold (calibrated on fit rows only).  Lookup is a row gather."""

    table: mx.array

    @classmethod
    def from_probs(cls, probs, threshold: float = 0.5) -> "TokenPrior":
        p = np.asarray(probs, np.float32)
        if p.ndim != 2:
            raise BlockSparseUnsupported("token prior: probabilities [vocab, boxes]")
        return cls(mx.array((p > threshold).astype(np.uint8)))

    @property
    def boxes(self) -> int:
        return int(self.table.shape[1])

    def box_open(self, tokens) -> mx.array:
        return self.table[tokens.reshape(-1).astype(mx.int32)]


# ---------------------------------------------------------------------------
# Per-sequence state
# ---------------------------------------------------------------------------


@dataclass
class NextTokenConfig:
    hysteresis: int = 0               # steps of disuse before a predicted tile closes (0: the prediction)
    fail_open_margin: float = 0.0     # probe law: open when logit > -margin (0: the plain sign rule)
    source: str = "probe"             # "probe" (step t's final hidden state) | "token" (step t's sampled token)

    def __post_init__(self):
        if self.hysteresis < 0 or self.hysteresis >= _CLOSED_AGE:
            raise ValueError("hysteresis must be in [0, 2**20)")
        if self.fail_open_margin < 0:
            raise ValueError("fail_open_margin must be >= 0 (fail open, never fail closed)")
        if self.source not in ("probe", "token"):
            raise ValueError("source: probe | token")
        if self.source == "token" and self.fail_open_margin:
            raise ValueError("fail_open_margin applies to the probe law; build the TokenPrior with a lower "
                             "threshold to fail open on the token law")


def fail_open_tables(tables: GateTables, margin: float) -> GateTables:
    """The same tables with every probe bias raised by ``margin`` (box open when logit > -margin)."""
    if margin == 0:
        return tables
    if tables.probe_b is None:
        raise BlockSparseUnsupported("fail-open margin needs probe tables")
    return GateTables(tables.probe_w, tables.probe_b + mx.array(np.float32(margin)), tables.tile_bits,
                      tables.always, tables.boxes)


class NextTokenGate:
    """Next-token gate state for ``seqs`` sequences (one row each).

    Per step: :meth:`current` is step t's open set (uint8 [seqs, tiles]); after step t's forward,
    :meth:`advance` (the last op of the step) predicts step t+1's set and calls the prefetch hook."""

    def __init__(self, tables: GateTables, seqs: int, cfg: NextTokenConfig | None = None, *,
                 prior: TokenPrior | None = None, prefetch: PrefetchHook | None = None,
                 gate_fn: Callable = tile_gates, prefill_tables: GateTables | None = None):
        """``tables``: the next-token probe (fit to predict step t+1's same-step set from step t's final
        hidden state; the same-step probe itself also works when it reads a compatible state) and the tile
        graph.  ``gate_fn`` is the gate law (default the fused kernel; :func:`reference_tile_gates` on CPU).
        ``prefill_tables``: the same-step probe for :meth:`prefill`'s gate-layer prompt states (default
        ``tables``, which is right only when both probes read the same state)."""
        self.cfg = cfg or NextTokenConfig()
        self.gate_fn = gate_fn
        if self.cfg.source == "token":
            if prior is None:
                raise BlockSparseUnsupported("source=token needs a TokenPrior")
            if prior.boxes != tables.boxes:
                raise BlockSparseUnsupported("token prior box count does not match the tile graph")
        self.tables = tables
        self.prefill_tables = prefill_tables or tables
        if self.prefill_tables.tiles != tables.tiles:
            raise BlockSparseUnsupported("prefill and prediction tables name different tile graphs")
        self.predict_tables = fail_open_tables(tables, self.cfg.fail_open_margin) if self.cfg.source == "probe" \
            else tables
        self.prior = prior
        self.prefetch = prefetch or NullPrefetch()
        self.seqs = int(seqs)
        self.age = mx.full((self.seqs, tables.tiles), _CLOSED_AGE, mx.int32)
        self.always = tables.always.astype(mx.bool_)[None]
        self.steps = 0
        self.predicted_steps = 0

    # ---- state
    def _open(self, age) -> mx.array:
        return mx.logical_or(age <= self.cfg.hysteresis, self.always).astype(mx.uint8)

    def current(self) -> mx.array:
        """Step t's open set, uint8 [seqs, tiles] (core always open)."""
        return self._open(self.age)

    def reset(self, rows) -> None:
        """Forget rows (a finished or replaced sequence): every gated tile closed until the next prefill."""
        m = np.zeros((self.seqs, 1), bool)
        m[list(rows)] = True
        self.age = mx.where(mx.array(m), mx.array(np.int32(_CLOSED_AGE)), self.age)

    def _set_rows(self, rows, topen_rows, steps) -> None:
        """age := 0 where predicted, else min(age + steps, closed), for ``rows`` only (others unchanged)."""
        rows = list(rows)
        if len(set(rows)) != len(rows):
            raise BlockSparseUnsupported("rows must be distinct")
        if rows == list(range(self.seqs)):
            pred = topen_rows.astype(mx.bool_)
        else:
            idx = mx.array(np.asarray(rows, np.int32))
            pred = mx.zeros((self.seqs, self.tables.tiles), mx.int32).at[idx].add(topen_rows.astype(mx.int32)) > 0
        sel = np.zeros((self.seqs, 1), bool)
        sel[rows] = True
        aged = mx.minimum(self.age + np.int32(steps), np.int32(_CLOSED_AGE))
        new = mx.where(pred, mx.array(np.int32(0)), aged)
        self.age = mx.where(mx.array(sel), new, self.age)

    # ---- the prompt
    def prefill(self, row: int, h_prompt=None, *, box_open=None) -> mx.array:
        """Prefill rule: the decode set starts as the union over the whole prompt (age 0).

        ``h_prompt`` [T, d] are the gate layer's hidden states of the prompt (the same-step probe,
        ``prefill_tables``, through the gate kernel's union output) or ``box_open`` uint8 [T, boxes] (router or
        token law).  Then call :meth:`advance` from the last prompt position (the prefill's last op)."""
        _, un = self.gate_fn(h_prompt, self.prefill_tables, box_open=box_open, union=True)
        u = union_open(un)                                   # uint8 [1, tiles]
        self._set_rows([row], u, _CLOSED_AGE)                # the union opens; everything else closes
        return u

    # ---- one step
    def predict(self, *, h_last=None, tokens=None) -> mx.array:
        """Step t+1's predicted open set before hysteresis, uint8 [seqs, tiles], from step t's final hidden
        state ``h_last`` [seqs, d] (probe law) or its sampled tokens [seqs] (token law)."""
        if self.cfg.source == "probe":
            if h_last is None:
                raise BlockSparseUnsupported("source=probe needs step t's final hidden state")
            return self.gate_fn(h_last, self.predict_tables)[0]
        if tokens is None:
            raise BlockSparseUnsupported("source=token needs step t's sampled tokens")
        n = int(tokens.size)
        return self.gate_fn(mx.zeros((n, 1), mx.bfloat16), self.tables, box_open=self.prior.box_open(tokens))[0]

    def advance(self, *, h_last=None, tokens=None, rows=None, steps: int = 1) -> mx.array:
        """The last op of step t: predict, apply hysteresis, hand step t+1's set to the prefetch hook.

        ``rows`` (default every row) are the sequences that decoded this step; ``steps`` is how many
        tokens they advanced (1, or the accepted length after a speculative verify)."""
        rows = list(range(self.seqs)) if rows is None else list(rows)
        steps = int(min(max(int(steps), 1), _CLOSED_AGE))
        p = self.predict(h_last=h_last, tokens=tokens)
        if int(p.shape[0]) != len(rows):
            raise BlockSparseUnsupported(f"{p.shape[0]} predictions for {len(rows)} rows")
        self._set_rows(rows, p, steps)
        self.steps += int(steps)
        self.predicted_steps += 1
        nxt = self.current()
        self.prefetch.next_tiles(rows, nxt if rows == list(range(self.seqs))
                                 else nxt[mx.array(np.asarray(rows, np.int32))])
        return nxt

    # ---- speculative decoding
    def speculative_union(self, row: int, draft_topen=None) -> mx.array:
        """Verify-pass open set for one sequence's draft positions: the union of its current set and the
        per-draft predictions ``draft_topen`` (uint8 [k, tiles], e.g. the token law on the draft tokens),
        as one shared row (uint8 [1, tiles]; ``per_row=False`` over the verify rows)."""
        cur = self.current()[row:row + 1]
        if draft_topen is None:
            return cur
        return mx.maximum(cur, mx.max(draft_topen, axis=0, keepdims=True)).astype(mx.uint8)


# ---------------------------------------------------------------------------
# Verify and recompute
# ---------------------------------------------------------------------------


def recompute_rows(predicted, verified, *, exact: bool = True):
    """(redo bool [rows], topen_redo uint8 [rows, tiles]).

    exact: a row is redone when its predicted set differs from the verified (same-step) set at all;
    otherwise only when it missed an open.  ``topen_redo`` keeps the verified set on redone rows and is all
    zero elsewhere, so those rows read no weight in the recompute pass."""
    p = predicted.astype(mx.bool_)
    v = verified.astype(mx.bool_)
    diff = mx.not_equal(p, v) if exact else mx.logical_and(v, mx.logical_not(p))
    redo = mx.any(diff, axis=1)
    return redo, mx.where(redo[:, None], verified, mx.zeros_like(verified)).astype(mx.uint8)


def verified(mlp: Callable, x, predicted, verified_open, *, exact: bool = True):
    """Run ``mlp(x, topen)`` (a per-row block-sparse call, e.g. ``block_mlp(..., per_row=True)[0]``) on the
    predicted set, recompute the rows the verify set disagrees with, merge.  Returns ``(y, redo)``.

    Bit-exact against ``mlp(x, verified_open)`` under the per-row law (a row depends on its own open row
    only): redone rows come from a call with their verified row; the others had predicted == verified."""
    if predicted.shape != verified_open.shape:
        raise BlockSparseUnsupported("predicted and verified open sets differ in shape")
    y_p = mlp(x, predicted)
    redo, t_redo = recompute_rows(predicted, verified_open, exact=exact)
    y_r = mlp(x, t_redo)
    n = int(redo.shape[0])
    for j in range(1, y_p.ndim):                      # the leading dims that index the open rows
        if int(np.prod(y_p.shape[:j])) == n:
            r = redo.reshape(*y_p.shape[:j], *((1,) * (y_p.ndim - j)))
            return mx.where(r, y_r, y_p), redo
    raise BlockSparseUnsupported("cannot align the output rows with the open rows")


def reference_tile_gates(h, tables: GateTables, *, box_open=None, union: bool = False):
    """The tile-gate law with MLX ops (any device; CPU tests): box open when ``h . w_b + b_b > 0`` (fp32;
    the kernel's lane-strided sum may round differently on a near-zero logit), tile open when ``always`` or
    any closure-folded member box is open.  Same outputs as :func:`ops.tile_gates`."""
    if box_open is None:
        if tables.probe_w is None:
            raise BlockSparseUnsupported("no probes: pass box_open (router law)")
        h2 = h.reshape(-1, int(h.shape[-1])).astype(mx.float32)
        raw = (h2 @ tables.probe_w.T + tables.probe_b) > 0
    else:
        raw = box_open.astype(mx.bool_)
    nb, w = tables.boxes, tables.words
    bits = tables.tile_bits.reshape(tables.tiles, w)
    shift = mx.array(np.arange(nb, dtype=np.uint32) % 32)
    word = mx.array(np.arange(nb, dtype=np.int32) // 32)
    member = ((bits[:, word] >> shift[None]) & np.uint32(1)) > 0                # [tiles, boxes]
    hit = (raw.astype(mx.float32) @ member.T.astype(mx.float32)) > 0
    topen = mx.logical_or(hit, tables.always.astype(mx.bool_)[None]).astype(mx.uint8)
    return topen, (mx.max(topen, axis=0).astype(mx.uint32) if union else None)
