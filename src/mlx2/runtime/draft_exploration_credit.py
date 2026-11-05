# SPDX-License-Identifier: Apache-2.0
"""Request-local elapsed-time credit for speculative recovery probes.

Mechanism after IronMLX's DFlash2 cost policies; see
``provenance/ironmlx-draft-credit-20261104.json``.  The owning controller
supplies only clean ordinary intervals, and charges *all* probe intervals,
including the first deferred-context catch-up.  Credit is an estimate-based
admission rule, not a hard wall-time bound on a probe that is already running.
Actual overspend remains as debt.  Mandatory context/round and cohort probes
are accounted separately; they can exceed the credit budget.
"""
from __future__ import annotations

import math


class DraftExplorationCredit:
    """One lane and timing regime's bounded credit and cumulative accounting."""

    def __init__(self, share: float, max_rounds: int):
        if isinstance(share, bool) or not isinstance(share, (float, int)):
            raise TypeError("reprobe credit share must be a finite number in (0, 1]")
        if not math.isfinite(share) or not 0 < share <= 1:
            raise ValueError("reprobe credit share must be a finite number in (0, 1]")
        if type(max_rounds) is not int or max_rounds < 1:
            raise ValueError("reprobe credit max rounds must be positive")
        self.share = float(share)
        self.max_rounds = max_rounds
        self.balance_ns = 0.0
        self.reserve_ns = 0.0
        self.ordinary_rounds = 0
        self.earned_ns = 0.0
        self.spent_ns = 0.0
        self.credit_probe_excess_ns = 0.0
        self.forced_probe_excess_ns = 0.0
        self.credit_probes = 0
        self.forced_probes = 0
        self.resets = 0
        self._probe_reason: str | None = None

    def reset(self):
        """Forget transferable timing/credit, retaining only audit totals."""
        self.balance_ns = self.reserve_ns = 0.0
        self.ordinary_rounds = 0
        self._probe_reason = None
        self.resets += 1

    def set_reserve(self, value: float):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("reprobe credit reserve must be finite and positive")
        self.reserve_ns = value
        self.balance_ns = min(self.balance_ns, value)

    def earn(self, elapsed_ns: int):
        if type(elapsed_ns) is not int or elapsed_ns <= 0:
            raise ValueError("ordinary credit interval must be positive integer nanoseconds")
        earned = self.share * elapsed_ns
        self.earned_ns += earned
        self.balance_ns += earned
        if self.reserve_ns > 0:
            self.balance_ns = min(self.balance_ns, self.reserve_ns)
        self.ordinary_rounds += 1

    def reason(self, *, context_bound: bool) -> str | None:
        if context_bound:
            return "deferred_context_bound"
        if self.ordinary_rounds >= self.max_rounds:
            return "ordinary_round_bound"
        if self.reserve_ns > 0 and self.balance_ns >= self.reserve_ns:
            return "time_credit"
        return None

    def begin(self, reason: str):
        self._probe_reason = reason
        self.ordinary_rounds = 0
        if reason == "time_credit":
            self.credit_probes += 1
        else:
            self.forced_probes += 1

    def charge(self, elapsed_ns: int, ordinary_ns: float):
        if type(elapsed_ns) is not int or elapsed_ns <= 0:
            raise ValueError("probe interval must be positive integer nanoseconds")
        if not math.isfinite(ordinary_ns) or ordinary_ns < 0:
            raise ValueError("ordinary probe estimate must be finite and nonnegative")
        excess = max(0.0, elapsed_ns - ordinary_ns)
        self.balance_ns -= excess
        self.spent_ns += excess
        if self._probe_reason == "time_credit":
            self.credit_probe_excess_ns += excess
        else:
            self.forced_probe_excess_ns += excess

    def receipt(self):
        return {
            "share": self.share,
            "max_ordinary_rounds": self.max_rounds,
            "balance_ns": self.balance_ns,
            "probe_reserve_ns": self.reserve_ns,
            "ordinary_rounds": self.ordinary_rounds,
            "earned_ns": self.earned_ns,
            "probe_excess_ns": self.spent_ns,
            "credit_probe_excess_ns": self.credit_probe_excess_ns,
            "forced_probe_excess_ns": self.forced_probe_excess_ns,
            "credit_probes": self.credit_probes,
            "forced_probes": self.forced_probes,
            "regime_resets": self.resets,
        }
