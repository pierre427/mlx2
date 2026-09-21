"""Host-only adaptive policies for continuous serving.

These controllers intentionally own no model or cache state.  Callers sample
them only at closed scheduler/verification boundaries, then apply one decision
to the whole physical cohort.  This keeps policy experimentation out of the
transactional cache machinery.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

_COUNTER_MAX = (1 << 63) - 1


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _bump(counters: dict[str, int], key: str, amount: int = 1) -> None:
    counters[key] = min(_COUNTER_MAX, int(counters.get(key, 0)) + int(amount))


@dataclass
class DecodeTimeFairness:
    """Bound prefill stalls and repay their wall-time cost with decode work."""

    enabled: bool = False
    fair_share: float = 0.5
    stall_target_ms: float = 500.0
    fallback_cap: int = 512
    floor: int = 64
    grid: int = 64
    debt_seconds: float = 0.0
    best_prefill_tokens_per_second: float = 0.0
    counters: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.fair_share) or self.fair_share < 0:
            raise ValueError("decode fair_share must be finite and nonnegative")
        if not math.isfinite(self.stall_target_ms) or self.stall_target_ms <= 0:
            raise ValueError("decode stall_target_ms must be finite and positive")
        if min(self.fallback_cap, self.floor, self.grid) < 1:
            raise ValueError("decode fairness token limits must be positive")
        for name in (
            "prefill_chunks",
            "debt_deferrals",
            "debt_repayments",
            "cap_clamps",
        ):
            self.counters.setdefault(name, 0)

    def cap(self, configured: int, *, contended: bool) -> int:
        configured = max(1, int(configured))
        if not self.enabled or not contended:
            return configured
        value = self.stall_bound(configured)
        if value < configured:
            _bump(self.counters, "cap_clamps")
        return value

    def stall_bound(self, configured: int) -> int:
        """Largest grid-aligned chunk expected to finish within the stall target.

        Pure: it neither consults ``enabled`` nor bumps counters, so the
        one-slice contention rule of :class:`PrefillOrder` can bound a slice
        with the same measurement while decode fairness itself is off.
        """
        configured = max(1, int(configured))
        if self.best_prefill_tokens_per_second > 0:
            value = int(
                self.best_prefill_tokens_per_second * self.stall_target_ms / 1000.0
            )
            value = max(self.grid, (value // self.grid) * self.grid)
        else:
            value = self.fallback_cap
        return max(self.floor, min(configured, value))

    def may_prefill(self, *, contended: bool) -> bool:
        if not self.enabled or not contended:
            if not contended:
                self.debt_seconds = 0.0
            return True
        if self.debt_seconds > 0:
            _bump(self.counters, "debt_deferrals")
            return False
        return True

    def observe_decode(self, seconds: float) -> None:
        paid = min(self.debt_seconds, max(0.0, float(seconds)))
        self.debt_seconds -= paid
        if self.debt_seconds < 1e-12:
            self.debt_seconds = 0.0
        if paid:
            _bump(self.counters, "debt_repayments")

    def observe_prefill(self, tokens: int, seconds: float, *, contended: bool) -> None:
        seconds = max(0.0, float(seconds))
        tokens = max(0, int(tokens))
        if tokens and seconds:
            # Contention only depresses a sample; a running maximum avoids a
            # feedback loop where smaller measured chunks cause smaller caps.
            self.best_prefill_tokens_per_second = max(
                self.best_prefill_tokens_per_second, tokens / seconds
            )
        if self.enabled and contended and seconds:
            self.debt_seconds += seconds * self.fair_share
            _bump(self.counters, "prefill_chunks")


@dataclass(frozen=True)
class PrefillCandidate:
    """One request with prefill work left, as :class:`PrefillOrder` sees it.

    ``uid`` is the scheduler's monotonic insertion id and therefore the
    arrival order; ``queued_at`` is informational only because the MTP path
    rewrites it to the time of the latest prefill slice.
    """

    uid: int
    remaining: int
    cached: int = 0
    queued_at: float = 0.0
    bypassed: int = 0


_PREFILL_ORDERS = ("srpt",)


@dataclass
class PrefillOrder:
    """Choose which queued prompt receives the next prefill service.

    Disabled (the default) this is exactly the ordering main already applied
    on the self-MTP path: shortest remaining prompt, then deepest APC reuse,
    then queue position, with no bypass bound (the age deadline and the
    short-prompt interleave in ``BatchGenerator`` bound starvation instead).
    The ordinary path stays FIFO unless enabled.

    Enabled (``order="srpt"``) both paths use shortest-remaining-first and a
    request that later arrivals have overtaken ``max_bypass`` consecutive
    times is served first (oldest first among such requests).  Bypass counts
    reset whenever the request is served.  Design reference: Splash
    ``Scheduler::nextPrefill`` (bounded overtakes) and ``prefillBudget``
    (a peer that fits one slice makes a long prefill contended).
    """

    enabled: bool = False
    order: str = "srpt"
    max_bypass: int = 3
    one_slice_contention: bool = False
    counters: dict[str, int] = field(default_factory=dict)
    _bypassed: dict[int, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("prefill_scheduling enabled must be boolean")
        if self.order not in _PREFILL_ORDERS:
            raise ValueError(
                f"prefill_scheduling order must be one of {list(_PREFILL_ORDERS)}"
            )
        self.max_bypass = _positive_integer(
            self.max_bypass, name="prefill_scheduling max_bypass"
        )
        if not isinstance(self.one_slice_contention, bool):
            raise ValueError("prefill_scheduling one_slice_contention must be boolean")
        if self.one_slice_contention and not self.enabled:
            raise ValueError("one_slice_contention requires prefill_scheduling")
        if self.enabled:
            for name in ("bypasses", "bypass_forced", "one_slice_clamps"):
                self.counters.setdefault(name, 0)

    @classmethod
    def from_value(
        cls, value: Mapping[str, Any] | PrefillOrder | None
    ) -> PrefillOrder:
        """Parse the server-owned ``prefill_scheduling`` object; absent = off."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("prefill_scheduling must be an object")
        allowed = {"order", "max_bypass", "one_slice_contention"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"unknown prefill_scheduling settings: {sorted(unknown)}"
            )
        return cls(
            enabled=True,
            order=value.get("order", "srpt"),
            max_bypass=value.get("max_bypass", 3),
            one_slice_contention=value.get("one_slice_contention", True),
        )

    def as_dict(self) -> dict[str, bool | int | str]:
        return {
            "order": self.order,
            "max_bypass": self.max_bypass,
            "one_slice_contention": self.one_slice_contention,
        }

    def candidate(
        self, uid: int, remaining: int, cached: int = 0, queued_at: float = 0.0
    ) -> PrefillCandidate:
        return PrefillCandidate(
            uid=int(uid),
            remaining=int(remaining),
            cached=int(cached),
            queued_at=float(queued_at),
            bypassed=self._bypassed.get(int(uid), 0),
        )

    def select(self, candidates: Sequence[PrefillCandidate]) -> int:
        """Return the index into ``candidates`` of the request to serve next."""
        if not candidates:
            raise ValueError("prefill order needs at least one candidate")
        if self.enabled:
            overdue = [
                index
                for index, candidate in enumerate(candidates)
                if candidate.bypassed >= self.max_bypass
            ]
            if overdue:
                _bump(self.counters, "bypass_forced")
                return min(overdue, key=lambda index: (candidates[index].uid, index))
        return min(
            range(len(candidates)),
            key=lambda index: (
                candidates[index].remaining,
                -candidates[index].cached,
                index,
            ),
        )

    def commit(
        self,
        served: Sequence[int],
        candidates: Sequence[PrefillCandidate],
        *,
        pending: Sequence[int] | None = None,
    ) -> None:
        """Record one service decision.

        Every unserved candidate that arrived before the youngest served
        request was overtaken once more; served requests reset.  ``pending``
        (uids still waiting for prefill) prunes counts of requests that were
        admitted, finished, or removed through any other path.
        """
        if not self.enabled or not served:
            return
        served = {int(uid) for uid in served}
        youngest = max(served)
        for candidate in candidates:
            if candidate.uid in served:
                self._bypassed.pop(candidate.uid, None)
            elif candidate.uid < youngest:
                self._bypassed[candidate.uid] = candidate.bypassed + 1
                _bump(self.counters, "bypasses")
        if pending is not None:
            keep = {int(uid) for uid in pending}
            for uid in [uid for uid in self._bypassed if uid not in keep]:
                del self._bypassed[uid]

    def note_one_slice_clamp(self) -> None:
        _bump(self.counters, "one_slice_clamps")


@dataclass
class CohortAdaptiveMTPDepth:
    """One cost-aware draft-depth decision for an entire physical cohort.

    Depth zero is an exact ordinary target round over the still-owned target
    state.  Goodput is learned independently for bounded width buckets at
    closed verification boundaries.  Concurrent buckets use infrequent,
    deterministic one-round probes; width one remains at the model's qualified
    depth so adaptive B1 is the fixed-depth correctness oracle.  Acceptance is
    retained as a fast safety valve, including the bounded park/re-entry path.
    """

    max_depth: int
    ewma_alpha: float = 0.25
    shrink_gate: float = 0.35
    grow_gate: float = 0.80
    loss_rounds: int = 3
    gain_rounds: int = 3
    park_rounds: int = 4
    goodput_alpha: float = 0.25
    goodput_hysteresis: float = 0.05
    min_samples_per_depth: int = 3
    goodput_window: int = 8
    probe_interval: int = 16
    stale_rounds: int = 128
    current_depth: int | None = None
    acceptance_ewma: float | None = None
    bad_streak: int = 0
    good_streak: int = 0
    parked_remaining: int = 0
    counters: dict[str, int] = field(default_factory=dict)
    trace: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=64), repr=False
    )
    _active_trace: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _buckets: dict[str, dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )
    _active_bucket: str = field(default="", init=False, repr=False)

    _WIDTH_BUCKETS = (
        (1, 1, "1"),
        (2, 2, "2"),
        (3, 4, "3-4"),
        (5, 8, "5-8"),
        (9, 16, "9-16"),
        (17, _COUNTER_MAX, "17+"),
    )

    def __post_init__(self) -> None:
        self.max_depth = _positive_integer(
            self.max_depth, name="adaptive MTP max_depth"
        )
        self.ewma_alpha = _finite_number(
            self.ewma_alpha, name="adaptive MTP ewma_alpha"
        )
        self.shrink_gate = _finite_number(
            self.shrink_gate, name="adaptive MTP shrink_gate"
        )
        self.grow_gate = _finite_number(
            self.grow_gate, name="adaptive MTP grow_gate"
        )
        if not 0 < self.ewma_alpha <= 1:
            raise ValueError("adaptive MTP ewma_alpha must be in (0, 1]")
        if not 0 <= self.shrink_gate < self.grow_gate <= 1:
            raise ValueError("adaptive MTP gates must satisfy 0 <= shrink < grow <= 1")
        self.loss_rounds = _positive_integer(
            self.loss_rounds, name="adaptive MTP loss_rounds"
        )
        self.gain_rounds = _positive_integer(
            self.gain_rounds, name="adaptive MTP gain_rounds"
        )
        self.park_rounds = _positive_integer(
            self.park_rounds, name="adaptive MTP park_rounds"
        )
        self.goodput_alpha = _finite_number(
            self.goodput_alpha, name="adaptive MTP goodput_alpha"
        )
        self.goodput_hysteresis = _finite_number(
            self.goodput_hysteresis, name="adaptive MTP goodput_hysteresis"
        )
        if not 0 < self.goodput_alpha <= 1:
            raise ValueError("adaptive MTP goodput_alpha must be in (0, 1]")
        if not 0 <= self.goodput_hysteresis < 1:
            raise ValueError(
                "adaptive MTP goodput_hysteresis must be in [0, 1)"
            )
        self.min_samples_per_depth = _positive_integer(
            self.min_samples_per_depth,
            name="adaptive MTP min_samples_per_depth",
        )
        self.goodput_window = _positive_integer(
            self.goodput_window, name="adaptive MTP goodput_window"
        )
        self.probe_interval = _positive_integer(
            self.probe_interval, name="adaptive MTP probe_interval"
        )
        self.stale_rounds = _positive_integer(
            self.stale_rounds, name="adaptive MTP stale_rounds"
        )
        if self.current_depth is not None and (
            isinstance(self.current_depth, bool)
            or not isinstance(self.current_depth, int)
        ):
            raise ValueError("adaptive MTP current_depth must be an integer")
        self.current_depth = (
            self.max_depth if self.current_depth is None else self.current_depth
        )
        if not 0 <= self.current_depth <= self.max_depth:
            raise ValueError("adaptive MTP current_depth is out of range")
        for name in (
            "boundaries",
            "depth_changes",
            "parks",
            "reentries",
            "probes",
            "cost_probes",
            "cost_depth_changes",
            "depth_decreases_concurrent",
            "depth_recoveries_alone",
        ):
            self.counters.setdefault(name, 0)

    @classmethod
    def width_bucket(cls, width: int) -> str:
        width = _positive_integer(width, name="adaptive MTP width")
        return next(
            label
            for low, high, label in cls._WIDTH_BUCKETS
            if low <= width <= high
        )

    def _bucket(self, label: str) -> dict[str, Any]:
        if label not in self._buckets:
            self._buckets[label] = {
                "chosen_depth": (
                    int(self.current_depth) if not self._buckets else self.max_depth
                ),
                "rounds": 0,
                "probes": 0,
                "probe_cursor": 0,
                "goodput": {},
                "samples": {},
                "estimate_updates": {},
                "pending_committed": {},
                "pending_elapsed": {},
                "pending_samples": {},
                "last_sample": {},
                "acceptance_ewma": None,
                "bad_streak": 0,
                "good_streak": 0,
                "parked_remaining": 0,
            }
        return self._buckets[label]

    def _sync_public_state(self, state: Mapping[str, Any]) -> None:
        self.acceptance_ewma = state["acceptance_ewma"]
        self.bad_streak = int(state["bad_streak"])
        self.good_streak = int(state["good_streak"])
        self.parked_remaining = int(state["parked_remaining"])

    def _set_chosen_depth(
        self,
        state: dict[str, Any],
        depth: int,
        *,
        width: int,
        cost_change: bool = False,
    ) -> None:
        depth = max(0, min(self.max_depth, int(depth)))
        old = int(state["chosen_depth"])
        state["chosen_depth"] = depth
        if old != depth:
            self.current_depth = depth
            self._changed(old, depth, width=width)
            if cost_change:
                _bump(self.counters, "cost_depth_changes")

    def diagnostics(self) -> dict[str, Any]:
        """Return bounded JSON diagnostics suitable for status and receipts."""
        buckets = {}
        for _, _, label in self._WIDTH_BUCKETS:
            state = self._buckets.get(label)
            if state is None:
                continue
            rounds = int(state["rounds"])
            probes = int(state["probes"])
            buckets[label] = {
                "chosen_depth": int(state["chosen_depth"]),
                "rounds": rounds,
                "probes": probes,
                "probe_fraction": probes / rounds if rounds else 0.0,
                "goodput_tokens_per_second": {
                    str(depth): float(value)
                    for depth, value in sorted(state["goodput"].items())
                },
                "samples": {
                    str(depth): int(value)
                    for depth, value in sorted(state["samples"].items())
                },
                "estimate_updates": {
                    str(depth): int(value)
                    for depth, value in sorted(state["estimate_updates"].items())
                },
                "pending_samples": {
                    str(depth): int(value)
                    for depth, value in sorted(state["pending_samples"].items())
                    if value
                },
            }
        return {"active_bucket": self._active_bucket, "buckets": buckets}

    def _classify_transition(self, old: int, new: int, *, width: int) -> None:
        if new < old and width > 1:
            _bump(self.counters, "depth_decreases_concurrent")
        elif new > old and width == 1:
            _bump(self.counters, "depth_recoveries_alone")

    def _changed(self, old: int, new: int, *, width: int) -> None:
        if new == old:
            return
        _bump(self.counters, "depth_changes")
        self._classify_transition(old, new, width=width)

    def select(self, *, admitted_cap: int | None = None, width: int = 1) -> int:
        """Choose at a closed boundary; never returns lane-specific depths."""
        width = _positive_integer(width, name="adaptive MTP width")
        label = self.width_bucket(width)
        previous_label = self._active_bucket
        self._active_bucket = label
        state = self._bucket(label)
        if width == 1:
            # Width one is the fixed-depth correctness anchor. Cost learning
            # remains diagnostic here; neither acceptance nor a previous
            # concurrent choice may change the qualified depth.
            state["chosen_depth"] = self.max_depth
        elif previous_label == label and int(self.current_depth) != int(
            state["chosen_depth"]
        ):
            # Preserve the public current_depth seam used by deterministic
            # CPU tests and operators forcing a bounded re-entry at a closed
            # boundary. Cross-bucket changes remain owned by learned state.
            state["chosen_depth"] = int(self.current_depth)
        state["rounds"] = min(_COUNTER_MAX, int(state["rounds"]) + 1)
        cap = (
            self.max_depth
            if admitted_cap is None
            else max(0, min(self.max_depth, int(admitted_cap)))
        )
        chosen = int(state["chosen_depth"])
        if int(self.current_depth) != chosen:
            old = int(self.current_depth)
            self.current_depth = chosen
            self._changed(old, chosen, width=width)
        depth = min(chosen, cap)
        probe_kind = None
        if int(state["parked_remaining"]) > 0:
            state["parked_remaining"] -= 1
            if int(state["parked_remaining"]) > 0:
                depth = 0
            else:
                if cap == 0:
                    # Admission, not the adaptive policy, owns this boundary.
                    # Keep the probe pending until it can actually execute.
                    state["parked_remaining"] = 1
                    depth = 0
                else:
                    depth = min(1, cap)
                    self._set_chosen_depth(state, depth, width=width)
                    _bump(self.counters, "reentries")
                    _bump(self.counters, "probes")
                    probe_kind = "reentry"
        elif width > 1 and int(state["rounds"]) % self.probe_interval == 0:
            candidates = [
                candidate
                for candidate in range(min(self.max_depth, cap) + 1)
                if candidate != int(state["chosen_depth"])
            ]
            under_sampled = [
                candidate
                for candidate in candidates
                if int(state["samples"].get(candidate, 0))
                < self.min_samples_per_depth
            ]
            stale = [
                candidate
                for candidate in candidates
                if int(state["rounds"])
                - int(state["last_sample"].get(candidate, 0))
                >= self.stale_rounds
            ]
            eligible = under_sampled or stale
            if eligible:
                if under_sampled:
                    # Finish the shallowest cold cell first. In particular,
                    # K=0 gets enough samples to decide by boundary
                    # min_samples_per_depth * probe_interval.
                    depth = min(eligible)
                else:
                    cursor = int(state["probe_cursor"])
                    depth = eligible[cursor % len(eligible)]
                    state["probe_cursor"] = cursor + 1
                state["probes"] = min(_COUNTER_MAX, int(state["probes"]) + 1)
                _bump(self.counters, "probes")
                _bump(self.counters, "cost_probes")
                probe_kind = "cost"
        _bump(self.counters, "boundaries")
        self._active_trace = {
            "boundary": int(self.counters["boundaries"]),
            "width": width,
            "cohort_width": width,
            "width_bucket": label,
            "admitted_cap": cap,
            "selected_depth": depth,
            "current_before_observation": int(self.current_depth),
            "probe_kind": probe_kind,
        }
        self.trace.append(self._active_trace)
        self._sync_public_state(state)
        return depth

    def observe(
        self,
        proposed: int,
        accepted: int,
        *,
        width: int = 1,
        observed_compute_width: int | None = None,
        committed: int | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        proposed, accepted = int(proposed), int(accepted)
        width = _positive_integer(width, name="adaptive MTP width")
        if proposed < 0 or not 0 <= accepted <= proposed:
            raise ValueError("invalid adaptive MTP observation")
        label = self.width_bucket(width)
        if (
            self._active_trace is not None
            and int(self._active_trace["cohort_width"]) != width
        ):
            raise ValueError(
                "adaptive MTP observation cohort width must match selection width"
            )
        if observed_compute_width is None:
            observed_compute_width = width
        observed_compute_width = _positive_integer(
            observed_compute_width, name="adaptive MTP observed_compute_width"
        )
        self._active_bucket = label
        state = self._bucket(label)
        probe_kind = (
            None if self._active_trace is None else self._active_trace.get("probe_kind")
        )
        selected_depth = (
            int(self.current_depth)
            if self._active_trace is None
            else int(self._active_trace["selected_depth"])
        )
        if (committed is None) != (elapsed_seconds is None):
            raise ValueError("adaptive MTP goodput requires committed and elapsed_seconds")
        cost_changed = False
        estimate_updated = False
        pooled_goodput = None
        if committed is not None and elapsed_seconds is not None:
            committed = int(committed)
            elapsed_seconds = _finite_number(
                elapsed_seconds, name="adaptive MTP elapsed_seconds"
            )
            if committed < 0 or elapsed_seconds <= 0:
                raise ValueError(
                    "adaptive MTP committed must be nonnegative and elapsed_seconds positive"
                )
            state["samples"][selected_depth] = min(
                _COUNTER_MAX, int(state["samples"].get(selected_depth, 0)) + 1
            )
            state["pending_committed"][selected_depth] = min(
                _COUNTER_MAX,
                int(state["pending_committed"].get(selected_depth, 0)) + committed,
            )
            state["pending_elapsed"][selected_depth] = (
                float(state["pending_elapsed"].get(selected_depth, 0.0))
                + elapsed_seconds
            )
            state["pending_samples"][selected_depth] = min(
                _COUNTER_MAX,
                int(state["pending_samples"].get(selected_depth, 0)) + 1,
            )
            state["last_sample"][selected_depth] = int(state["rounds"])
            old_goodput = state["goodput"].get(selected_depth)
            pending_samples = int(state["pending_samples"][selected_depth])
            flush_at = (
                self.min_samples_per_depth
                if old_goodput is None
                else self.goodput_window
            )
            estimate_updated = pending_samples >= flush_at
            if estimate_updated:
                pooled_goodput = (
                    int(state["pending_committed"][selected_depth])
                    / float(state["pending_elapsed"][selected_depth])
                )
                state["goodput"][selected_depth] = (
                    pooled_goodput
                    if old_goodput is None
                    else self.goodput_alpha * pooled_goodput
                    + (1.0 - self.goodput_alpha) * old_goodput
                )
                state["estimate_updates"][selected_depth] = min(
                    _COUNTER_MAX,
                    int(state["estimate_updates"].get(selected_depth, 0)) + 1,
                )
                state["pending_committed"][selected_depth] = 0
                state["pending_elapsed"][selected_depth] = 0.0
                state["pending_samples"][selected_depth] = 0
            current = int(state["chosen_depth"])
            estimates = state["goodput"]
            if estimate_updated and current in estimates:
                best = max(estimates, key=lambda depth: (estimates[depth], depth))
                if (
                    best != current
                    and estimates[best]
                    > estimates[current] * (1.0 + self.goodput_hysteresis)
                ):
                    self._set_chosen_depth(
                        state, best, width=width, cost_change=True
                    )
                    cost_changed = True
        if self._active_trace is not None:
            self._active_trace.update(
                width=width,
                cohort_width=width,
                width_bucket=label,
                observed_compute_width=observed_compute_width,
                proposed=proposed,
                accepted=accepted,
                acceptance=(accepted / proposed if proposed else None),
                committed=committed,
                elapsed_seconds=elapsed_seconds,
                goodput=(
                    None
                    if committed is None or elapsed_seconds is None
                    else committed / elapsed_seconds
                ),
                estimate_updated=estimate_updated,
                pooled_goodput=pooled_goodput,
                published_goodput=state["goodput"].get(selected_depth),
                pending_samples=int(
                    state["pending_samples"].get(selected_depth, 0)
                ),
            )
        if proposed == 0:
            if self._active_trace is not None:
                self._active_trace.update(
                    current_after_observation=int(self.current_depth),
                    chosen_after_observation=int(state["chosen_depth"]),
                )
            self._sync_public_state(state)
            return
        if probe_kind == "cost":
            # Cost probes update only their (bucket, depth) goodput cell. They
            # cannot advance an acceptance streak or mutate the stable depth.
            if self._active_trace is not None:
                self._active_trace.update(
                    current_after_observation=int(self.current_depth),
                    chosen_after_observation=int(state["chosen_depth"]),
                )
            self._sync_public_state(state)
            return
        ratio = accepted / proposed
        state["acceptance_ewma"] = (
            ratio
            if state["acceptance_ewma"] is None
            else self.ewma_alpha * ratio
            + (1.0 - self.ewma_alpha) * state["acceptance_ewma"]
        )
        if width == 1:
            state["bad_streak"] = state["good_streak"] = 0
            self._set_chosen_depth(state, self.max_depth, width=width)
            self._sync_public_state(state)
            if self._active_trace is not None:
                self._active_trace.update(
                    acceptance_ewma=self.acceptance_ewma,
                    current_after_observation=int(self.current_depth),
                    chosen_after_observation=int(state["chosen_depth"]),
                )
            return
        if state["acceptance_ewma"] < self.shrink_gate:
            state["bad_streak"] += 1
            state["good_streak"] = 0
        elif state["acceptance_ewma"] >= self.grow_gate:
            state["good_streak"] += 1
            state["bad_streak"] = 0
        else:
            state["bad_streak"] = state["good_streak"] = 0
        # A one-round cost probe informs the cost model only.  Letting its
        # acceptance mutate the stable depth would defeat the probe bound.
        if not cost_changed and state["bad_streak"] >= self.loss_rounds:
            old = int(state["chosen_depth"])
            if old > 1:
                self._set_chosen_depth(state, old - 1, width=width)
            else:
                self._set_chosen_depth(state, 0, width=width)
                # One extra countdown boundary performs the re-entry; the
                # configured number remains the exact count of K=0 rounds.
                state["parked_remaining"] = self.park_rounds + 1
                _bump(self.counters, "parks")
            state["bad_streak"] = 0
        elif (
            not cost_changed
            and state["good_streak"] >= self.gain_rounds
            and int(state["chosen_depth"]) < self.max_depth
        ):
            old = int(state["chosen_depth"])
            candidate = min(self.max_depth, max(1, old + 1))
            estimates = state["goodput"]
            if (
                old not in estimates
                or candidate not in estimates
                or estimates[candidate]
                > estimates[old] * (1.0 + self.goodput_hysteresis)
            ):
                self._set_chosen_depth(state, candidate, width=width)
            state["good_streak"] = 0
        self._sync_public_state(state)
        if self._active_trace is not None:
            self._active_trace.update(
                acceptance_ewma=self.acceptance_ewma,
                current_after_observation=int(self.current_depth),
                chosen_after_observation=int(state["chosen_depth"]),
            )


@dataclass(frozen=True)
class AdaptiveMTPDepthPolicy:
    """Validated serving selection for the cohort-wide depth controller."""

    enabled: bool = False
    ewma_alpha: float = 0.25
    shrink_gate: float = 0.35
    grow_gate: float = 0.80
    loss_rounds: int = 3
    gain_rounds: int = 3
    park_rounds: int = 4
    goodput_alpha: float = 0.25
    goodput_hysteresis: float = 0.05
    min_samples_per_depth: int = 3
    goodput_window: int = 8
    probe_interval: int = 16
    stale_rounds: int = 128

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("adaptive MTP enabled must be boolean")
        # Reuse the executable controller's validation so the serving contract
        # cannot accept a policy that later fails after model allocation.
        CohortAdaptiveMTPDepth(max_depth=1, **self.controller_kwargs())

    @classmethod
    def from_value(
        cls, value: bool | Mapping[str, Any] | AdaptiveMTPDepthPolicy | None
    ) -> AdaptiveMTPDepthPolicy:
        if value is None or value is False:
            return cls()
        if value is True:
            return cls(enabled=True)
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("adaptive_mtp_depth must be a boolean or object")
        allowed = {
            "enabled",
            "ewma_alpha",
            "shrink_gate",
            "grow_gate",
            "loss_rounds",
            "gain_rounds",
            "park_rounds",
            "goodput_alpha",
            "goodput_hysteresis",
            "min_samples_per_depth",
            "goodput_window",
            "probe_interval",
            "stale_rounds",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown adaptive MTP settings: {sorted(unknown)}")
        return cls(**dict(value))

    def controller_kwargs(self) -> dict[str, float | int]:
        return {
            "ewma_alpha": self.ewma_alpha,
            "shrink_gate": self.shrink_gate,
            "grow_gate": self.grow_gate,
            "loss_rounds": self.loss_rounds,
            "gain_rounds": self.gain_rounds,
            "park_rounds": self.park_rounds,
            "goodput_alpha": self.goodput_alpha,
            "goodput_hysteresis": self.goodput_hysteresis,
            "min_samples_per_depth": self.min_samples_per_depth,
            "goodput_window": self.goodput_window,
            "probe_interval": self.probe_interval,
            "stale_rounds": self.stale_rounds,
        }

    def as_dict(self) -> dict[str, bool | float | int]:
        values = {
            "enabled": self.enabled,
            "ewma_alpha": self.ewma_alpha,
            "shrink_gate": self.shrink_gate,
            "grow_gate": self.grow_gate,
            "loss_rounds": self.loss_rounds,
            "gain_rounds": self.gain_rounds,
            "park_rounds": self.park_rounds,
        }
        if self.enabled:
            values.update(
                {
                    "goodput_alpha": self.goodput_alpha,
                    "goodput_hysteresis": self.goodput_hysteresis,
                    "min_samples_per_depth": self.min_samples_per_depth,
                    "goodput_window": self.goodput_window,
                    "probe_interval": self.probe_interval,
                    "stale_rounds": self.stale_rounds,
                }
            )
        return values


@dataclass(frozen=True)
class MTPOrdinaryHandoffPolicy:
    """Validated one-way native-MTP to ordinary-batcher width policy.

    The threshold is intentionally static and qualification-bound. Native-MTP
    goodput is measured at decode boundaries, while the benchmark's ordinary
    aggregate rate includes HTTP, admission and prefill; comparing those two
    timing domains would not be a valid adaptive decision.
    """

    enabled: bool = False
    max_mtp_width: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("MTP ordinary handoff enabled must be boolean")
        if self.max_mtp_width is not None:
            _positive_integer(
                self.max_mtp_width, name="MTP ordinary handoff max_mtp_width"
            )
        if self.enabled and self.max_mtp_width is None:
            raise ValueError(
                "enabled MTP ordinary handoff requires max_mtp_width"
            )

    @classmethod
    def from_value(
        cls, value: bool | Mapping[str, Any] | MTPOrdinaryHandoffPolicy | None
    ) -> MTPOrdinaryHandoffPolicy:
        if value is None or value is False:
            return cls()
        if value is True:
            raise ValueError(
                "MTP ordinary handoff requires an object with enabled true "
                "and max_mtp_width"
            )
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("mtp_ordinary_handoff must be an object")
        allowed = {"enabled", "max_mtp_width"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"unknown MTP ordinary handoff settings: {sorted(unknown)}"
            )
        if value and value.get("enabled") is not True:
            raise ValueError(
                "mtp_ordinary_handoff settings require enabled: true"
            )
        return cls(**dict(value))

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_mtp_width": self.max_mtp_width,
        }

    def decision(
        self,
        *,
        width: int,
        width_locked: bool,
        adaptive: CohortAdaptiveMTPDepth | None,
    ) -> dict[str, Any] | None:
        """Return bounded static-threshold evidence for a handoff."""
        del adaptive
        if not self.enabled:
            return None
        width = _positive_integer(width, name="MTP ordinary handoff width")
        if width <= self.max_mtp_width:
            return None
        return {
            "reason": (
                "segmented_width_lock"
                if width_locked
                else "static_width_threshold"
            ),
            "bucket": CohortAdaptiveMTPDepth.width_bucket(width),
            "max_mtp_width": self.max_mtp_width,
        }
