# SPDX-License-Identifier: MIT
"""Per-verify expert-load ledger and miss-priced verify costs for embedded DSpark.

Why this exists. On the streamed-expert Source checkpoint, embedded DSpark
accepted 95-98% of proposals and still lost to ordinary decode in every
2026-10-28 arm (docs/DEEPSEEK_V41.md). A verify of ``w`` rows routes each row
to its own six experts per layer; rows that leave the resident set turn into
SSD reads. Neither acceptance length nor tok/s says how many expert loads a
verify width costs, so this ledger counts them per round.

Counters (bounded integers; no device sync, no allocation beyond one small
tuple per round). The target expert provider's ``stats`` already counts
``cache_hits``/``cache_misses``/``bytes_read``/``demand_wait_ns``. The draft
head owns a separate expert cache, so between two consecutive draft calls of
one request the target provider only sees that round's verify (and its taps).
The ledger attributes that delta to the verify width ``proposals + 1`` of the
round that just closed. The final round of a request is never closed (no next
draft call) and is therefore not counted; the prefill before the first draft
call is not counted either.

Opt-in pricing (``deepseek_dspark_verify_schedule: survival_miss_v1``). The
survival schedule picks k from relative verify costs that describe compute
only. ``miss_priced_costs`` adds the measured marginal expert-read cost of each
extra verify row, in units of one plain round:

    cost'(w) = cost(w) + (w - 1) * slope_bytes / bandwidth / plain_round_s

``slope_bytes`` is the per-extra-row miss bytes, from EWMAs of miss bytes per
round at each observed width against the width-one (plain) EWMA;
``bandwidth`` is the request's own miss bytes over its demand-read wait
(provider lifetime fallback); ``plain_round_s`` is the EWMA wall time of
width-one rounds. Until all three exist the costs are returned unchanged
(``priced=False``), i.e. survival_v1 behaviour. Width-one evidence is
refreshed by a plain probe after 32 consecutive speculative rounds, and after
32 consecutive width-one rounds one round is verified unpriced so a request
priced out by a cold cache can recover once its experts become resident. Every committed token remains
the target's greedy choice; only the verify length changes.

Implemented, unqualified, not selected by default. Provenance:
provenance/deepseek-v41-dspark-verify-ledger.json.
"""
import math
import time

COUNTERS = ("cache_hits", "cache_misses", "bytes_read", "demand_wait_ns")
EWMA = 0.25
MIN_BANDWIDTH_EVIDENCE_BYTES = 64 << 20
PLAIN_PROBE_ROUNDS = 2
REPROBE_INTERVAL = 32
MAX_WIDTH = 6


def _ewma(previous, value):
    return value if previous is None else EWMA * value + (1 - EWMA) * previous


def _empty_width():
    return {"rounds": 0, "expert_acquisitions": 0, "expert_loads": 0, "miss_bytes": 0,
            "demand_wait_ns": 0, "committed_tokens": 0}


class VerifyExpertLedger:
    """Request-local; ``stats`` is the target provider's live counter dict (or None)."""

    def __init__(self, stats, *, timed=False):
        self._stats = stats
        self.timed = bool(timed)
        self.pending = None  # (width, counters, started_ns or None)
        self.by_width = {}
        self.ewma_bytes = {}
        self.plain_round_s = None
        self.closed_rounds = 0
        self.speculative_since_probe = 0
        self.plain_since_speculative = 0

    def _counters(self):
        values = self._stats if isinstance(self._stats, dict) else {}
        return tuple(int(values.get(key, 0) or 0) for key in COUNTERS)

    def close(self, committed_tokens):
        """Close the round opened by the previous draft call; returns its delta or None."""
        if self.pending is None:
            return None
        width, before, started = self.pending
        self.pending = None
        after = self._counters()
        hits, misses, read, wait = (max(0, a - b) for a, b in zip(after, before))
        entry = self.by_width.setdefault(width, _empty_width())
        entry["rounds"] += 1
        entry["expert_acquisitions"] += hits + misses
        entry["expert_loads"] += misses
        entry["miss_bytes"] += read
        entry["demand_wait_ns"] += wait
        entry["committed_tokens"] += max(0, int(committed_tokens))
        self.ewma_bytes[width] = _ewma(self.ewma_bytes.get(width), float(read))
        if width == 1 and started is not None:
            elapsed = max(0, time.perf_counter_ns() - started) / 1e9
            if elapsed > 0:
                self.plain_round_s = _ewma(self.plain_round_s, elapsed)
        self.closed_rounds += 1
        return {"width": width, "expert_acquisitions": hits + misses, "expert_loads": misses,
                "miss_bytes": read, "demand_wait_ns": wait}

    def open(self, width):
        if type(width) is not int or not 1 <= width <= MAX_WIDTH:
            raise ValueError("DSpark verify ledger width must be 1..6")
        self.pending = (width, self._counters(), time.perf_counter_ns() if self.timed else None)
        if width > 1:
            self.speculative_since_probe += 1
            self.plain_since_speculative = 0
        else:
            self.speculative_since_probe = 0
            self.plain_since_speculative += 1

    def needs_plain_probe(self):
        """True while width-one evidence is missing or stale (survival_miss_v1 only)."""
        plain = self.by_width.get(1, {}).get("rounds", 0)
        return plain < PLAIN_PROBE_ROUNDS or self.speculative_since_probe >= REPROBE_INTERVAL

    def needs_speculative_probe(self):
        """After a priced-out stretch, verify one unpriced round so the slope can recover."""
        return self.plain_since_speculative >= REPROBE_INTERVAL and self.slope_bytes() is not None

    def slope_bytes(self):
        base = self.ewma_bytes.get(1)
        if base is None:
            return None
        weighted = total = 0.0
        for width, value in self.ewma_bytes.items():
            if width < 2:
                continue
            rounds = self.by_width[width]["rounds"]
            weighted += rounds * max(0.0, value - base) / (width - 1)
            total += rounds
        return weighted / total if total else None

    def bandwidth(self, fallback=None):
        read = sum(e["miss_bytes"] for e in self.by_width.values())
        wait = sum(e["demand_wait_ns"] for e in self.by_width.values()) / 1e9
        if read >= MIN_BANDWIDTH_EVIDENCE_BYTES and wait > 0:
            return read / wait, "request"
        values = fallback if isinstance(fallback, dict) else {}
        read, wait = int(values.get("bytes_read", 0) or 0), int(values.get("demand_wait_ns", 0) or 0) / 1e9
        if read >= MIN_BANDWIDTH_EVIDENCE_BYTES and wait > 0:
            return read / wait, "provider_lifetime"
        return None, None

    def receipt(self):
        rows = sum(width * e["rounds"] for width, e in self.by_width.items())
        rounds = sum(e["rounds"] for e in self.by_width.values())
        loads = sum(e["expert_loads"] for e in self.by_width.values())
        return {"closed_rounds": rounds, "verify_rows": rows,
                "expert_loads": loads,
                "expert_loads_per_verify": loads / rounds if rounds else None,
                "miss_bytes": sum(e["miss_bytes"] for e in self.by_width.values()),
                "by_width": {str(w): dict(self.by_width[w]) for w in sorted(self.by_width)},
                "plain_round_seconds": self.plain_round_s, "slope_bytes_per_extra_row": self.slope_bytes()}


def miss_priced_costs(costs, ledger, *, fallback_stats=None):
    """Return (costs', receipt). Unpriced evidence returns the costs unchanged."""
    costs = [float(v) for v in costs]
    if (len(costs) != MAX_WIDTH or any(not math.isfinite(v) or v <= 0 for v in costs)
            or any(b < a for a, b in zip(costs, costs[1:]))):
        raise ValueError("DSpark verify costs must be six positive nondecreasing numbers")
    slope = ledger.slope_bytes()
    bandwidth, source = ledger.bandwidth(fallback_stats)
    plain = ledger.plain_round_s
    if slope is None or plain is None or plain <= 0:
        return costs, {"priced": False, "reason": "missing_width_evidence"}
    if slope > 0 and bandwidth is None:
        return costs, {"priced": False, "reason": "read_bandwidth_unknown"}
    per_row = 0.0 if slope <= 0 else slope / bandwidth / plain
    priced = [c + (w - 1) * per_row for w, c in enumerate(costs, 1)]
    return priced, {"priced": True, "per_extra_row_cost": per_row, "slope_bytes": slope,
                    "bytes_per_second": bandwidth, "bandwidth_source": source, "plain_round_seconds": plain}
