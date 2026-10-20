# SPDX-License-Identifier: Apache-2.0
"""Request-owned cost-per-committed-token speculative handoff.

The controller in this module owns policy only.  Executors remain responsible
for observing a closed committed boundary and for moving exact target state to
their ordinary decoder.  State is intentionally lane-local: timing evidence is
never shared across requests, tenants, draft providers, or execution widths.

The feature is default-off for ordinary speculative routes.  A selected AST
planner or explicitly selected AST physical target installs the generic
reversible controller unless the caller supplies or disables it explicitly.
A selected controller first
collects ordinary target samples, then speculative samples in the same timing
domain and physical width.  A sustained speculative loss produces one terminal
decision.  It can never re-enter speculation for that request.

Target cost grows with context for both routes, so a terminal decision may
only use the speculative samples that directly follow an ordinary baseline
(``speculative_samples + loss_decisions - 1``, the fewest that can decide).
A sustained loss against an older baseline does not decide: it discards both
windows and asks for a fresh ordinary probe at the current context.  A
speculator that keeps winning pays no extra probes.

``reversible`` selects a per-session park instead of the one-way decision.
A parked session keeps its draft context exact (the executor runs target-only
rounds and defers the draft append), keeps refreshing its ordinary baseline,
and after ``reprobe_tokens`` parked tokens runs ``speculative_samples``
speculative rounds.  If those beat the fresh ordinary cost by the same margin
that parks it, speculation re-engages; otherwise the session re-parks with the
wait doubled up to ``reprobe_max_tokens``.  With ``pipelined_park`` a parked
executor overlaps each anchor-only round with the host read of the previous
one, as the PLD plain-round pipeline does.

Evidence can exclude one-time costs that are not the route's steady price.
Reversible control uses ``settle_rounds`` for each kind (the first
speculative round ingests the whole prompt into the draft plane; the first
rounds after an APCv2 restore pay the lazy restore) and the first round of
every re-probe (it consumes the deferred draft context).  The ordinary
baseline is the median per-round cost, so one stalled round cannot set it.
The terminal controller remains byte-for-byte backward compatible by default;
an adapter may explicitly set ``one_way_ordinary_settle_rounds`` when its
first ordinary probe rounds include compilation or lazy-restore work that the
direct ordinary decoder would not keep paying.  Speculative startup remains
charged because it is part of the selected route's per-request cost.
Reversible evidence compares the
session's own wall time per committed token, whatever cohort width the round
ran at, because that is what the session waits for.

A decision is not a handoff.  ``decision_reached`` records the terminal
policy decision; ``handoff_triggered`` is set only by the executor that
actually moves the lane to ordinary decode (``mark_handoff_performed``).  A
lane whose decision lands on its final token finishes without a handoff.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _unit_interval(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in (0, 1]")
    return value


@dataclass(frozen=True)
class CostPerCommitHandoffSettings:
    """Validated, conservative and default-off decision settings."""

    enabled: bool = False
    ordinary_samples: int = 4
    speculative_samples: int = 4
    loss_ratio: float = 0.97
    loss_decisions: int = 3
    window: int = 16
    reversible: bool = False
    reprobe_tokens: int = 32
    reprobe_max_tokens: int = 512
    deferred_context_cap: int = 256
    settle_rounds: int = 1
    one_way_ordinary_settle_rounds: int = 0
    pipelined_park: bool = True
    loss_basis: str = "rolling_window"

    _KEYS = frozenset(
        {
            "enabled",
            "ordinary_samples",
            "speculative_samples",
            "loss_ratio",
            "loss_decisions",
            "window",
            "reversible",
            "reprobe_tokens",
            "reprobe_max_tokens",
            "deferred_context_cap",
            "settle_rounds",
            "one_way_ordinary_settle_rounds",
            "pipelined_park",
            "loss_basis",
        }
    )

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("cost-per-commit handoff enabled must be boolean")
        if type(self.reversible) is not bool:
            raise ValueError("cost-per-commit reversible must be boolean")
        if type(self.pipelined_park) is not bool:
            raise ValueError("cost-per-commit pipelined_park must be boolean")
        if self.loss_basis not in {"rolling_window", "current_round"}:
            raise ValueError(
                "cost-per-commit loss_basis must be rolling_window or current_round"
            )
        if self.reversible and self.loss_basis != "rolling_window":
            raise ValueError(
                "cost-per-commit current_round loss_basis is only valid for one-way handoff"
            )
        for name in (
            "ordinary_samples",
            "speculative_samples",
            "loss_decisions",
            "window",
            "reprobe_tokens",
            "reprobe_max_tokens",
            "deferred_context_cap",
        ):
            _positive_int(getattr(self, name), f"cost-per-commit {name}")
        _unit_interval(self.loss_ratio, "cost-per-commit loss_ratio")
        if self.window < max(self.ordinary_samples, self.speculative_samples):
            raise ValueError(
                "cost-per-commit window must cover both sample minima"
            )
        for name in ("settle_rounds", "one_way_ordinary_settle_rounds"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"cost-per-commit {name} must be a non-negative integer"
                )
        if self.reprobe_max_tokens < self.reprobe_tokens:
            raise ValueError(
                "cost-per-commit reprobe_max_tokens must be at least reprobe_tokens"
            )

    @classmethod
    def from_value(
        cls,
        value: bool | Mapping[str, Any] | "CostPerCommitHandoffSettings" | None,
    ) -> "CostPerCommitHandoffSettings | None":
        if value is None or value is False:
            return None
        if isinstance(value, cls):
            return value if value.enabled else None
        if value is True:
            return cls(enabled=True)
        if not isinstance(value, Mapping):
            raise ValueError("cost-per-commit handoff must be an object or boolean")
        unknown = set(value) - cls._KEYS
        if unknown:
            raise ValueError(
                f"unknown cost-per-commit handoff settings: {sorted(unknown)}"
            )
        if value.get("enabled", True) is not True:
            if set(value) - {"enabled"}:
                raise ValueError(
                    "disabled cost-per-commit handoff cannot carry settings"
                )
            return None
        return cls(**{"enabled": True, **dict(value)})

    def as_dict(self) -> dict[str, Any]:
        out = {
            "enabled": self.enabled,
            "ordinary_samples": self.ordinary_samples,
            "speculative_samples": self.speculative_samples,
            "loss_ratio": self.loss_ratio,
            "loss_decisions": self.loss_decisions,
            "window": self.window,
        }
        if not self.reversible:
            # The terminal (pre-reversible) form stays byte-identical, so
            # existing receipts and qualification records keep matching.
            result = (
                {**out, "loss_basis": "current_round"}
                if self.loss_basis == "current_round"
                else out
            )
            if self.one_way_ordinary_settle_rounds:
                result = {
                    **result,
                    "one_way_ordinary_settle_rounds": (
                        self.one_way_ordinary_settle_rounds
                    ),
                }
            return result
        return {
            **out,
            "reversible": True,
            "reprobe_tokens": self.reprobe_tokens,
            "reprobe_max_tokens": self.reprobe_max_tokens,
            "deferred_context_cap": self.deferred_context_cap,
            "settle_rounds": self.settle_rounds,
            "pipelined_park": self.pipelined_park,
        }

    @classmethod
    def ast_self_configuring_defaults(cls) -> "CostPerCommitHandoffSettings":
        """Conservative model-independent defaults for an explicit AST route.

        These values only control how quickly a request obtains its own
        ordinary and speculative measurements. They do not describe model
        geometry or select a tensor executor. The reversible controller keeps
        both state planes exact and can therefore park and later re-probe the
        selected AST route without an adapter-owned tuning declaration.
        """

        return cls(
            enabled=True,
            # Three ordinary rounds give the reversible controller a median
            # baseline that one stalled observation cannot poison.  One
            # speculative observation can then protect a short request: a
            # false park re-probes after bounded exact ordinary work, while
            # later speculative observations can still overturn a false keep.
            # The old 2+2 samples plus two discarded settle rounds let a
            # 16-token request finish before a severe loss could be parked.
            ordinary_samples=3,
            speculative_samples=1,
            loss_ratio=0.97,
            loss_decisions=1,
            window=4,
            reversible=True,
            reprobe_tokens=16,
            reprobe_max_tokens=64,
            deferred_context_cap=64,
            settle_rounds=0,
            pipelined_park=True,
        )


class CostPerCommitLaneState:
    """One request's handoff evidence: a terminal decision, or (reversible)
    its own park / re-probe / re-engage cycle."""

    def __init__(
        self,
        settings: CostPerCommitHandoffSettings,
        *,
        lane_uid: Hashable,
        owner_scope: Hashable,
    ) -> None:
        if not isinstance(settings, CostPerCommitHandoffSettings):
            raise TypeError("validated cost-per-commit settings are required")
        if not settings.enabled:
            raise ValueError("cost-per-commit handoff settings are disabled")
        try:
            hash((lane_uid, owner_scope))
        except TypeError as error:
            raise TypeError("lane uid and owner scope must be hashable") from error
        self.settings = settings
        self.lane_uid = lane_uid
        self.owner_scope = owner_scope
        self._ordinary: dict[int, deque[tuple[int, int]]] = {}
        self._speculative: dict[int, deque[tuple[int, int]]] = {}
        self._loss_streak: dict[int, int] = {}
        # Speculative samples taken since this width's ordinary baseline was
        # last sampled; bounds how stale a baseline a decision may use.
        self._speculative_since_ordinary: dict[int, int] = {}
        self._ordinary_refreshes = 0
        # Cohort probe accounting (see cohort_needs_ordinary_probe): forced
        # depth-zero rounds while this lane's own baseline was complete, the
        # current consecutive run of them, and speculative rounds granted to
        # blocked peers while this lane still lacked its baseline.
        self._probe_blocked_rounds = 0
        self._probe_blocked_streak = 0
        self._probe_deferred_rounds = 0
        self._decision: dict[str, Any] | None = None
        self._handoff_performed = False
        self._cancelled = False
        self._invalid_samples = 0
        self._observations = 0
        self._valid_samples = 0
        self._mixed_parked_rounds = 0
        # Reversible park state (unused by the one-way controller).
        self._parked = False
        self._probing = False
        self._parked_tokens = 0
        self._reprobe_after = settings.reprobe_tokens
        self._probe: deque[tuple[int, int]] = deque(maxlen=settings.window)
        self._parks = 0
        self._reengagements = 0
        self._reprobes = 0
        self._failed_reprobes = 0
        self._events: deque[dict[str, Any]] = deque(maxlen=16)
        self._settled = {"ordinary": 0, "speculative": 0}
        self._settle_skipped = 0
        self._probe_catchup_pending = False
        self._session_seeded = False
        self._session_seed_active = False

    def seed_reversible_park(self, prior: Mapping[str, Any]) -> None:
        """Start this exact lane parked from a generator-local slow-route prior.

        The prior carries token-free timing only and never crosses a generator
        binding.  Ordinary parked rounds rebuild this lane's own baseline;
        after ``reprobe_tokens`` it exercises the normal exact speculative
        re-probe, so the shared hint cannot permanently suppress a route.
        """
        if not self.settings.reversible:
            raise ValueError("a session park prior requires reversible control")
        if self._observations or self._parked or self._decision is not None:
            raise RuntimeError("a session park prior requires a fresh lane state")
        fields = {}
        for name in (
            "ordinary_ns_per_committed_token",
            "speculative_ns_per_committed_token",
            "speculative_to_ordinary_cost_ratio",
        ):
            value = prior.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("invalid session cost prior")
            value = float(value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("invalid session cost prior")
            fields[name] = value
        self._parked = True
        self._parks = 1
        self._session_seeded = True
        self._session_seed_active = True
        self._events.append(
            {
                "event": "seed_park",
                "observation": 0,
                "reason": "generator_session_measured_slow",
                **fields,
                "reprobe_after_tokens": self._reprobe_after,
            }
        )

    @property
    def session_prior_eligible(self) -> bool:
        """Whether a generator may seed this untouched reversible state."""
        return (
            self.settings.reversible
            and not self._observations
            and not self._parked
            and self._decision is None
            and not self._cancelled
        )

    @property
    def session_prior_seeded(self) -> bool:
        return self._session_seeded

    @property
    def session_prior_active(self) -> bool:
        return self._session_seed_active

    def revoke_session_prior_for_width_growth(self, execution_width: int) -> bool:
        """Drop a B1 seed's policy state before executing at a wider width.

        Cumulative counters and events remain intact so receipts keep an exact
        history of the B1 rounds that already ran.  Only the measurements and
        state that could steer a future route decision are reset.
        """
        if type(execution_width) is not int or execution_width <= 1:
            raise ValueError("session prior revocation requires width greater than one")
        if not self._session_seed_active:
            return False
        self._ordinary.clear()
        self._speculative.clear()
        self._loss_streak.clear()
        self._speculative_since_ordinary.clear()
        self._decision = None
        self._handoff_performed = False
        self._parked = False
        self._probing = False
        self._parked_tokens = 0
        self._reprobe_after = self.settings.reprobe_tokens
        self._probe.clear()
        self._settled = {"ordinary": 0, "speculative": 0}
        self._probe_catchup_pending = False
        self._probe_blocked_streak = 0
        self._session_seed_active = False
        self._event(
            "seed_revoke",
            reason="execution_width_grew",
            execution_width=execution_width,
        )
        return True

    def _baseline(self, samples) -> float | None:
        """Ordinary cost: the median round under reversible evidence."""
        if not self.settings.reversible:
            return self._cost(samples)
        per_token = sorted(elapsed / count for elapsed, count in samples if count > 0)
        if not per_token:
            return None
        middle = len(per_token) // 2
        return (
            per_token[middle]
            if len(per_token) % 2
            else (per_token[middle - 1] + per_token[middle]) / 2
        )

    @staticmethod
    def _cost(samples: deque[tuple[int, int]]) -> float | None:
        committed = sum(count for _elapsed, count in samples)
        return (
            None
            if committed <= 0
            else sum(elapsed for elapsed, _count in samples) / committed
        )

    def _width_key(self, execution_width: int) -> int:
        # Reversible evidence is the session's own wait per committed token.
        return 0 if self.settings.reversible else int(execution_width)

    @property
    def speculation_paused(self) -> bool:
        """Parked and not re-probing: the executor runs target-only rounds."""
        return self._parked and not self._probing and not self._cancelled

    def needs_ordinary_probe(self, execution_width: int) -> bool:
        if self._decision is not None or self._cancelled or self._parked:
            return False
        samples = self._ordinary.get(self._width_key(execution_width), ())
        return len(samples) < self.settings.ordinary_samples

    @property
    def handed_off(self) -> bool:
        return self._decision is not None

    def mark_handoff_performed(self) -> None:
        """Record that the executor moved this lane to ordinary decode."""
        if self._decision is None:
            raise RuntimeError("cost-per-commit handoff requires a decision")
        self._handoff_performed = True

    def cancel(self) -> None:
        self._cancelled = True
        self._ordinary.clear()
        self._speculative.clear()
        self._loss_streak.clear()
        self._speculative_since_ordinary.clear()

    def observe(
        self,
        *,
        kind: str,
        elapsed_ns: int,
        committed_tokens: int,
        execution_width: int,
        lane_uid: Hashable,
        owner_scope: Hashable,
    ) -> dict[str, Any] | None:
        """Observe one closed boundary and return a new terminal decision.

        Identity must match exactly.  A mismatch raises instead of allowing a
        caller to mix another lane's or tenant's cost evidence into this lane.
        Invalid timing samples are ignored and counted; they never trigger a
        handoff.
        """
        if lane_uid != self.lane_uid or owner_scope != self.owner_scope:
            raise ValueError("cost-per-commit handoff owner identity mismatch")
        if kind not in {"ordinary", "speculative"}:
            raise ValueError("cost-per-commit sample kind must be ordinary or speculative")
        if self._decision is not None or self._cancelled:
            return None
        self._observations += 1
        if (
            type(elapsed_ns) is not int
            or type(committed_tokens) is not int
            or type(execution_width) is not int
            or elapsed_ns <= 0
            or committed_tokens <= 0
            or execution_width <= 0
            or not math.isfinite(float(elapsed_ns))
        ):
            self._invalid_samples += 1
            return None
        settle_rounds = self.settings.settle_rounds if self.settings.reversible else (
            self.settings.one_way_ordinary_settle_rounds
            if kind == "ordinary"
            else 0
        )
        if self._settled[kind] < settle_rounds:
            # One-time warm-up rounds are not the route's steady price.
            self._settled[kind] += 1
            self._settle_skipped += 1
            return None
        self._valid_samples += 1
        execution_width = self._width_key(execution_width)
        if self._parked:
            return self._observe_parked(kind, elapsed_ns, committed_tokens)
        table = self._ordinary if kind == "ordinary" else self._speculative
        samples = table.setdefault(
            execution_width, deque(maxlen=self.settings.window)
        )
        samples.append((elapsed_ns, committed_tokens))
        if kind == "ordinary":
            self._speculative_since_ordinary[execution_width] = 0
            return None
        ordinary = self._ordinary.get(execution_width)
        if ordinary is None or len(ordinary) < self.settings.ordinary_samples:
            return None
        since = self._speculative_since_ordinary.get(execution_width, 0) + 1
        self._speculative_since_ordinary[execution_width] = since
        if len(samples) < self.settings.speculative_samples:
            return None
        ordinary_cost = self._baseline(ordinary)
        speculative_cost = self._cost(samples)
        if ordinary_cost is None or speculative_cost is None:
            return None
        latest_cost = elapsed_ns / committed_tokens
        comparison_cost = (
            latest_cost
            if self.settings.loss_basis == "current_round"
            else speculative_cost
        )
        losing = comparison_cost >= ordinary_cost / self.settings.loss_ratio
        streak = self._loss_streak.get(execution_width, 0)
        streak = streak + 1 if losing else 0
        self._loss_streak[execution_width] = streak
        if streak < self.settings.loss_decisions:
            return None
        if since > (
            self.settings.speculative_samples + self.settings.loss_decisions - 1
        ):
            # The baseline predates the speculative window.  Re-measure both
            # routes at the current context before any one-way decision.
            del self._ordinary[execution_width]
            samples.clear()
            self._loss_streak[execution_width] = 0
            self._speculative_since_ordinary[execution_width] = 0
            self._ordinary_refreshes += 1
            return None
        if self.settings.reversible:
            return self._park(execution_width, ordinary_cost, speculative_cost, streak)
        self._decision = {
            "reason": "cost_per_committed_token",
            "one_way": True,
            "execution_width": execution_width,
            "ordinary_ns_per_committed_token": ordinary_cost,
            "speculative_ns_per_committed_token": comparison_cost,
            "speculative_to_ordinary_cost_ratio": comparison_cost / ordinary_cost,
            "loss_ratio_threshold": self.settings.loss_ratio,
            "loss_decisions": streak,
            "speculative_samples_since_ordinary": since,
        }
        if self.settings.loss_basis == "current_round":
            self._decision.update(
                {
                    "speculative_cost_basis": "current_round",
                    "speculative_window_ns_per_committed_token": speculative_cost,
                }
            )
        return dict(self._decision)

    def _event(self, event: str, **fields: Any) -> dict[str, Any]:
        record = {"event": event, "observation": self._observations, **fields}
        self._events.append(record)
        return dict(record)

    def _park(self, width, ordinary_cost, speculative_cost, streak):
        self._parked = True
        self._probing = False
        self._parked_tokens = 0
        self._parks += 1
        self._loss_streak[width] = 0
        self._speculative.pop(width, None)
        self._speculative_since_ordinary[width] = 0
        return self._event(
            "park",
            reason="cost_per_committed_token",
            ordinary_ns_per_committed_token=ordinary_cost,
            speculative_ns_per_committed_token=speculative_cost,
            speculative_to_ordinary_cost_ratio=speculative_cost / ordinary_cost,
            loss_decisions=streak,
            reprobe_after_tokens=self._reprobe_after,
        )

    def _observe_parked(self, kind, elapsed_ns, committed_tokens):
        width = 0
        if kind == "ordinary":
            ordinary = self._ordinary.setdefault(
                width, deque(maxlen=self.settings.window)
            )
            ordinary.append((elapsed_ns, committed_tokens))
            if self._probing:
                return None
            return self._advance_parked_tokens(committed_tokens)
        if not self._probing:
            # A speculative round while paused (a cohort peer forced it) is
            # not evidence for this session's re-probe.
            return None
        if self._probe_catchup_pending:
            # The probe's first round consumes the deferred draft context.
            self._probe_catchup_pending = False
            self._settle_skipped += 1
            return None
        self._probe.append((elapsed_ns, committed_tokens))
        if len(self._probe) < self.settings.speculative_samples:
            return None
        ordinary_cost = self._baseline(self._ordinary.get(width, ()))
        probe_cost = self._cost(self._probe)
        self._probing = False
        if ordinary_cost is None or probe_cost is None:
            return None
        ratio = probe_cost / ordinary_cost
        if probe_cost <= ordinary_cost * self.settings.loss_ratio:
            self._parked = False
            self._parked_tokens = 0
            self._reprobe_after = self.settings.reprobe_tokens
            self._reengagements += 1
            self._speculative[width] = deque(self._probe, maxlen=self.settings.window)
            self._speculative_since_ordinary[width] = 0
            self._loss_streak[width] = 0
            return self._event(
                "reengage",
                ordinary_ns_per_committed_token=ordinary_cost,
                speculative_ns_per_committed_token=probe_cost,
                speculative_to_ordinary_cost_ratio=ratio,
            )
        self._failed_reprobes += 1
        self._parked_tokens = 0
        self._reprobe_after = min(
            2 * self._reprobe_after, self.settings.reprobe_max_tokens
        )
        return self._event(
            "repark",
            ordinary_ns_per_committed_token=ordinary_cost,
            speculative_ns_per_committed_token=probe_cost,
            speculative_to_ordinary_cost_ratio=ratio,
            reprobe_after_tokens=self._reprobe_after,
        )

    def observe_parked_progress(self, *, committed_tokens, lane_uid, owner_scope):
        """Advance a mixed cohort's park clock without fabricating timing.

        A parked anchor can share a target invocation with speculative peers.
        Its latency includes their speculation and is not an ordinary-route
        sample. Only the committed-token reprobe clock advances.
        """
        if lane_uid != self.lane_uid or owner_scope != self.owner_scope:
            raise ValueError("cost-per-commit handoff owner identity mismatch")
        if type(committed_tokens) is not int or committed_tokens <= 0:
            raise ValueError("parked progress requires positive committed tokens")
        if not self.speculation_paused:
            return None
        self._mixed_parked_rounds += 1
        return self._advance_parked_tokens(committed_tokens)

    def _advance_parked_tokens(self, committed_tokens):
        self._parked_tokens += committed_tokens
        if (
            self._parked_tokens >= self._reprobe_after
            and len(self._ordinary.get(0, ())) >= self.settings.ordinary_samples
        ):
            self._probing = True
            self._probe.clear()
            self._probe_catchup_pending = True
            self._reprobes += 1
            return self._event("reprobe", parked_tokens=self._parked_tokens)
        return None

    def receipt(self) -> dict[str, Any]:
        def costs(table):
            return {
                str(width): {
                    "samples": len(samples),
                    "committed_tokens": sum(count for _elapsed, count in samples),
                    "ns_per_committed_token": self._cost(samples),
                }
                for width, samples in sorted(table.items())
            }

        return {
            "implemented": True,
            "qualified": False,
            "selected": True,
            "defaulted": False,
            "observed_used": self._observations > 0,
            "observed_sampling": self._valid_samples > 0,
            "decision_reached": self._decision is not None,
            "handoff_triggered": self._handoff_performed,
            "one_way": not self.settings.reversible,
            "scope": "request_lane",
            "reversible": (
                {
                    "state": (
                        "probing" if self._parked and self._probing
                        else "parked" if self._parked
                        else "speculating"
                    ),
                    "parks": self._parks,
                    "reprobes": self._reprobes,
                    "reengagements": self._reengagements,
                    "failed_reprobes": self._failed_reprobes,
                    "parked_tokens": self._parked_tokens,
                    "mixed_parked_rounds": self._mixed_parked_rounds,
                    "settle_skipped_rounds": self._settle_skipped,
                    "reprobe_after_tokens": self._reprobe_after,
                    "events": [dict(event) for event in self._events],
                    "session_seeded": self._session_seeded,
                    "session_seed_active": self._session_seed_active,
                    "session_seed_revoked": (
                        self._session_seeded and not self._session_seed_active
                    ),
                }
                if self.settings.reversible
                else None
            ),
            "settings": self.settings.as_dict(),
            "ordinary": costs(self._ordinary),
            "speculative": costs(self._speculative),
            "decision": None if self._decision is None else dict(self._decision),
            "cancelled": self._cancelled,
            "invalid_samples": self._invalid_samples,
            "ordinary_refreshes": self._ordinary_refreshes,
            "probe_blocked_rounds": self._probe_blocked_rounds,
            "probe_deferred_rounds": self._probe_deferred_rounds,
            "observations": self._observations,
            "valid_samples": self._valid_samples,
        }


def cohort_needs_ordinary_probe(states, execution_width: int) -> bool:
    """Decide whether this physical round is a cohort-wide ordinary probe.

    Native and external batching cannot mix verify depths inside one physical
    cohort, so a lane that lacks its ordinary baseline (a late join, or a
    stale-baseline refresh) forces the whole cohort to depth zero.  Executors
    record that probe only for the lanes that need it; established lanes keep
    their windows but collect nothing.

    The forced run is bounded so continuous arrivals cannot starve
    established lanes: once an established lane has been held at depth zero
    for ``ordinary_samples`` consecutive rounds, the next round is
    speculative for the whole cohort.  Executors do not record that round for
    lanes still missing a baseline.  Hence an established lane runs
    speculative at least once every ``ordinary_samples + 1`` rounds, and a
    late lane (same settings) completes its baseline within
    ``ordinary_samples + 1`` rounds of joining.

    Call this exactly once per physical round: it records the round's probe
    accounting on each lane state.  The result depends only on that call
    sequence, so decisions stay deterministic.
    """
    width = int(execution_width)
    live = [
        state
        for state in states
        if state is not None
        and state._decision is None
        and not state._cancelled
    ]
    needing = [state for state in live if state.needs_ordinary_probe(width)]
    blocked = [state for state in live if not state.needs_ordinary_probe(width)]
    for state in needing:
        state._probe_blocked_streak = 0
    if needing and not any(
        state._probe_blocked_streak >= state.settings.ordinary_samples
        for state in blocked
    ):
        for state in blocked:
            state._probe_blocked_streak += 1
            state._probe_blocked_rounds += 1
        return True
    for state in blocked:
        state._probe_blocked_streak = 0
    for state in needing:
        state._probe_deferred_rounds += 1
    return False


__all__ = [
    "CostPerCommitHandoffSettings",
    "CostPerCommitLaneState",
    "cohort_needs_ordinary_probe",
]
