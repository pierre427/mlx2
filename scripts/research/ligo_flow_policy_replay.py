#!/usr/bin/env python3
"""Offline shadow replay for admission, deadline and row-pack research.

The replay compares three host-only policies over one chronological trace:

* ``baseline`` uses FIFO admission and nominal complete-round prices.
* ``residual_bound`` keeps FIFO but adds a one-sided residual reserve.
* ``wait_pack`` adds bounded stage-pressure admission and orders legal prefill
  offers by progress per predicted complete-round millisecond.

Every arm still uses mlx2's final-extent, step-rounded cohort projection as a
hard memory authority and ``choose_paged_pack`` as the legal pack filter.  The
time and memory values are declared proxies; this script makes no device-speed,
route-qualification or production-selection claim.  Observed output length,
cancellation time and round residuals are outcomes and are never policy inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from mlx2.runtime.batch_admission import AdmissionState, LinearStateCost
from mlx2.runtime.paged_pack_scheduler import (
    PrefillOffer,
    PrefillOption,
    ReservedRows,
    choose_paged_pack,
)

SCHEMA = "mlx2.ligo-flow-policy-replay.v1"
ARMS = ("baseline", "residual_bound", "wait_pack")


@dataclass(frozen=True)
class TraceRequest:
    """One request with separate declared inputs and observed outcomes."""

    uid: int
    arrival_ms: float
    prompt_tokens: int
    max_output_tokens: int
    actual_output_tokens: int
    warm_prefix_tokens: int = 0
    draft_tokens: int = 0
    accepted_draft_tokens: int = 0
    cancel_at_ms: float | None = None
    cost_epoch: str = "normal"
    observed_residual_ms: float = 5.0
    cohort_id: str | None = None
    cohort_size: int = 1

    def __post_init__(self) -> None:
        integer_fields = (
            self.uid,
            self.prompt_tokens,
            self.max_output_tokens,
            self.actual_output_tokens,
            self.warm_prefix_tokens,
            self.draft_tokens,
            self.accepted_draft_tokens,
            self.cohort_size,
        )
        if any(type(value) is not int for value in integer_fields):
            raise TypeError("trace counts must be integers")
        if self.uid < 0 or self.prompt_tokens < 1 or self.max_output_tokens < 1:
            raise ValueError("uid must be nonnegative and request sizes positive")
        if not 0 <= self.actual_output_tokens <= self.max_output_tokens:
            raise ValueError("actual output must fit the declared maximum")
        if not 0 <= self.warm_prefix_tokens <= self.prompt_tokens:
            raise ValueError("warm prefix must lie inside the prompt")
        if not 0 <= self.accepted_draft_tokens <= self.draft_tokens:
            raise ValueError("accepted draft tokens must fit the draft")
        if self.cohort_size < 1 or (self.cohort_id is None) != (self.cohort_size == 1):
            raise ValueError("only named multi-request cohorts declare cohort_size")
        numeric = (self.arrival_ms, self.observed_residual_ms)
        if any(not math.isfinite(value) or value < 0 for value in numeric):
            raise ValueError("trace times must be finite and nonnegative")
        if self.cancel_at_ms is not None and (
            not math.isfinite(self.cancel_at_ms) or self.cancel_at_ms < self.arrival_ms
        ):
            raise ValueError("cancellation must be finite and follow arrival")
        if not self.cost_epoch:
            raise ValueError("cost_epoch is required")

    @property
    def projected_units(self) -> int:
        return self.prompt_tokens + self.max_output_tokens

    @property
    def prefill_tokens(self) -> int:
        return max(1, self.prompt_tokens - self.warm_prefix_tokens)


@dataclass(frozen=True)
class ReplayConfig:
    max_lanes: int = 4
    memory_budget_units: int = 3584
    fixed_state_units: int = 32
    state_units_per_token: int = 1
    allocation_step_tokens: int = 64
    prefill_options: tuple[int, ...] = (32, 64, 128)
    row_capacity: int = 160
    free_pages: int = 64
    page_tokens: int = 64
    decode_deadline_ms: float = 55.0
    nominal_fixed_ms: float = 8.0
    nominal_row_ms: float = 0.40
    residual_alpha: float = 0.20
    residual_min_samples: int = 3
    residual_fallback_ms: float = 20.0
    wait_force_ms: float = 600.0
    wait_stage_limits: tuple[int, int, int] = (4, 2, 1)
    max_rounds: int = 5000

    def __post_init__(self) -> None:
        if min(
            self.max_lanes,
            self.memory_budget_units,
            self.allocation_step_tokens,
            self.row_capacity,
            self.free_pages,
            self.page_tokens,
            self.residual_min_samples,
            self.max_rounds,
        ) < 1:
            raise ValueError("positive replay limits required")
        if not self.prefill_options or tuple(sorted(set(self.prefill_options))) != self.prefill_options:
            raise ValueError("prefill options must be sorted unique positive rows")
        if self.prefill_options[0] < 1 or self.prefill_options[-1] > self.row_capacity:
            raise ValueError("prefill options must fit row capacity")
        if len(self.wait_stage_limits) != 3 or min(self.wait_stage_limits) < 1:
            raise ValueError("three positive WAIT-like stage limits are required")
        if not 0 < self.residual_alpha < 1:
            raise ValueError("residual_alpha must lie in (0, 1)")


@dataclass
class Job:
    request: TraceRequest
    attempt_id: str
    state: str = "future"
    remaining_prefill: int = 0
    remaining_output: int = 0
    admitted_at_ms: float | None = None
    first_token_at_ms: float | None = None
    terminal_at_ms: float | None = None
    emitted_tokens: int = 0

    def __post_init__(self) -> None:
        self.remaining_prefill = self.request.prefill_tokens
        self.remaining_output = self.request.actual_output_tokens


def policy_request_view(request: TraceRequest) -> dict:
    """Return fields an online policy is allowed to inspect."""

    return {
        "uid": request.uid,
        "arrival_ms": request.arrival_ms,
        "prompt_tokens": request.prompt_tokens,
        "max_output_tokens": request.max_output_tokens,
        "warm_prefix_tokens": request.warm_prefix_tokens,
        "draft_tokens": request.draft_tokens,
        "cost_epoch": request.cost_epoch,
        "cohort_id": request.cohort_id,
        "cohort_size": request.cohort_size,
    }


def default_trace() -> tuple[TraceRequest, ...]:
    """Mixed heavy-tail, cancellation, warm/cold and atomic-cohort trace."""

    return (
        TraceRequest(1, 0, 640, 192, 160, draft_tokens=2,
                     accepted_draft_tokens=1, observed_residual_ms=6),
        TraceRequest(2, 0, 160, 32, 8, observed_residual_ms=5),
        TraceRequest(3, 0, 192, 48, 12, warm_prefix_tokens=128,
                     observed_residual_ms=4),
        TraceRequest(4, 25, 256, 64, 20, draft_tokens=1,
                     accepted_draft_tokens=1, observed_residual_ms=5),
        TraceRequest(5, 50, 704, 256, 220, cost_epoch="interference",
                     observed_residual_ms=28),
        TraceRequest(6, 60, 512, 128, 100, cancel_at_ms=190,
                     cost_epoch="interference", observed_residual_ms=30),
        TraceRequest(7, 75, 128, 24, 6, warm_prefix_tokens=96,
                     observed_residual_ms=4),
        TraceRequest(8, 90, 224, 40, 10, cohort_id="pair-a", cohort_size=2,
                     cost_epoch="interference", observed_residual_ms=26),
        TraceRequest(9, 90, 224, 40, 14, cohort_id="pair-a", cohort_size=2,
                     cost_epoch="interference", observed_residual_ms=27),
        TraceRequest(10, 120, 144, 32, 8, warm_prefix_tokens=112,
                     cost_epoch="interference", observed_residual_ms=25),
        TraceRequest(11, 145, 768, 320, 280, draft_tokens=2,
                     accepted_draft_tokens=1, cost_epoch="interference",
                     observed_residual_ms=32),
        TraceRequest(12, 170, 176, 48, 12, observed_residual_ms=5),
        TraceRequest(13, 210, 208, 56, 16, warm_prefix_tokens=128,
                     observed_residual_ms=4),
        TraceRequest(14, 240, 288, 80, 30, observed_residual_ms=6),
    )


def validate_trace(trace: Iterable[TraceRequest]) -> tuple[TraceRequest, ...]:
    trace = tuple(trace)
    if not trace:
        raise ValueError("trace must contain requests")
    if any(type(item) is not TraceRequest for item in trace):
        raise TypeError("trace must contain exact TraceRequest values")
    if len({item.uid for item in trace}) != len(trace):
        raise ValueError("request uids must be unique")
    cohorts: dict[str, list[TraceRequest]] = defaultdict(list)
    for item in trace:
        if item.cohort_id is not None:
            cohorts[item.cohort_id].append(item)
    for cohort_id, members in cohorts.items():
        sizes = {item.cohort_size for item in members}
        if len(sizes) != 1 or sizes.pop() != len(members):
            raise ValueError(f"atomic cohort {cohort_id!r} is incomplete")
    return tuple(sorted(trace, key=lambda item: (item.arrival_ms, item.uid)))


class ResidualEnvelope:
    """Finite-sample, one-sided chronological residual reserve."""

    def __init__(self, config: ReplayConfig):
        self.config = config
        self.samples: dict[str, list[float]] = defaultdict(list)

    def bound(self, epoch: str) -> tuple[float, bool]:
        samples = self.samples[epoch]
        if len(samples) < self.config.residual_min_samples:
            return self.config.residual_fallback_ms, False
        ordered = sorted(samples)
        rank = math.ceil((len(ordered) + 1) * (1 - self.config.residual_alpha))
        if rank > len(ordered):
            return self.config.residual_fallback_ms, False
        return max(0.0, ordered[rank - 1]), True

    def observe(self, epoch: str, residual_ms: float) -> None:
        if math.isfinite(residual_ms):
            self.samples[epoch].append(float(residual_ms))


class ProxyPackPrice:
    """Declared offline time proxy implementing the live pack-price protocol."""

    profile_id = "offline-proxy:not-measured:not-qualified:v1"

    def __init__(self, replay: Replay, bounded: bool):
        self.replay = replay
        self.bounded = bounded

    def nominal_ms(
        self,
        reserved: tuple[ReservedRows, ...],
        prefill: tuple[int, PrefillOption] | None,
    ) -> float:
        rows = sum(item.rows for item in reserved)
        if prefill is not None:
            rows += prefill[1].rows
        if rows == 0:
            return 0.0
        return self.replay.config.nominal_fixed_ms + self.replay.config.nominal_row_ms * rows

    def estimate_ms(
        self,
        reserved: tuple[ReservedRows, ...],
        prefill: tuple[int, PrefillOption] | None,
    ) -> float:
        value = self.nominal_ms(reserved, prefill)
        if self.bounded and value:
            epoch = self.replay.action_epoch(reserved, prefill)
            value += self.replay.residuals.bound(epoch)[0]
        return value


class Replay:
    def __init__(
        self,
        arm: str,
        trace: Iterable[TraceRequest],
        config: ReplayConfig,
        *,
        logical_run_id: str,
    ) -> None:
        if arm not in ARMS:
            raise ValueError(f"unknown replay arm {arm!r}")
        self.arm = arm
        self.trace = validate_trace(trace)
        self.config = config
        self.logical_run_id = logical_run_id
        self.jobs = {
            item.uid: Job(item, f"{logical_run_id}:{arm}:{item.uid}:attempt-1")
            for item in self.trace
        }
        self.now_ms = 0.0
        self.events: list[dict] = []
        self.decisions: list[dict] = []
        self.residuals = ResidualEnvelope(config)
        self.cost = LinearStateCost(
            config.fixed_state_units,
            config.state_units_per_token,
            allocation_step_units=config.allocation_step_tokens,
        )
        self.price = ProxyPackPrice(self, bounded=arm != "baseline")
        self.stats = Counter()
        self.queue_waits: list[float] = []
        self.ttfts: list[float] = []
        self.latencies: list[float] = []
        self.peak_projected_memory = 0.0

    def event(self, job: Job, kind: str, **detail) -> None:
        self.events.append({
            "sequence": len(self.events),
            "time_ms": round(self.now_ms, 6),
            "logical_request_id": job.request.uid,
            "attempt_id": job.attempt_id,
            "kind": kind,
            **detail,
        })

    def active(self) -> list[Job]:
        return [job for job in self.jobs.values() if job.state in {"prefill", "decode"}]

    def projected_memory(self, jobs: Iterable[Job]) -> float:
        states = [
            AdmissionState(
                job.request.uid,
                job.request.projected_units,
                job.request.warm_prefix_tokens,
                metadata={"phase": job.state},
            )
            for job in jobs
        ]
        return self.cost.cohort_bytes(states)

    def _arrivals_and_cancellations(self) -> None:
        for job in self.jobs.values():
            if job.state == "future" and job.request.arrival_ms <= self.now_ms:
                job.state = "queued"
                self.event(job, "eligible", policy_view=policy_request_view(job.request))
        for job in self.jobs.values():
            cancel = job.request.cancel_at_ms
            if cancel is None or cancel > self.now_ms:
                continue
            if job.state == "queued":
                job.state = "cancelled"
                job.terminal_at_ms = self.now_ms
                self.event(job, "terminal_cancelled", phase="queued")
            elif job.state in {"prefill", "decode"}:
                job.state = "cancelled"
                job.terminal_at_ms = self.now_ms
                self.event(job, "state_released", reason="cancelled")
                self.event(job, "terminal_cancelled", phase="active")

    def _queued_groups(self) -> list[list[Job]]:
        queued = [job for job in self.jobs.values() if job.state == "queued"]
        by_cohort: dict[str, list[Job]] = defaultdict(list)
        groups: list[list[Job]] = []
        for job in queued:
            if job.request.cohort_id is None:
                groups.append([job])
            else:
                by_cohort[job.request.cohort_id].append(job)
        for members in by_cohort.values():
            if len(members) == members[0].request.cohort_size:
                groups.append(members)
        return groups

    @staticmethod
    def _stage(request: TraceRequest) -> int:
        if request.max_output_tokens <= 64:
            return 0
        if request.max_output_tokens <= 192:
            return 1
        return 2

    def _wait_order(self, groups: list[list[Job]]) -> list[list[Job]]:
        active_stages = Counter(self._stage(job.request) for job in self.active())

        def key(group: list[Job]):
            oldest = min(job.request.arrival_ms for job in group)
            forced = self.now_ms - oldest >= self.config.wait_force_ms
            stage = max(self._stage(job.request) for job in group)
            pressure = active_stages[stage]
            projected = max(job.request.projected_units for job in group)
            return (not forced, pressure, projected, oldest, min(job.request.uid for job in group))

        return sorted(groups, key=key)

    def _admit(self) -> None:
        while len(self.active()) < self.config.max_lanes:
            groups = self._queued_groups()
            if not groups:
                return
            groups = sorted(
                groups,
                key=lambda group: (
                    min(job.request.arrival_ms for job in group),
                    min(job.request.uid for job in group),
                ),
            )
            if self.arm == "wait_pack":
                groups = self._wait_order(groups)
            selected = None
            active = self.active()
            for group in groups:
                if len(active) + len(group) > self.config.max_lanes:
                    if self.arm != "wait_pack":
                        return
                    continue
                if self.arm == "wait_pack":
                    stage = max(self._stage(job.request) for job in group)
                    stage_active = sum(
                        self._stage(job.request) == stage for job in active
                    )
                    oldest = min(job.request.arrival_ms for job in group)
                    forced = self.now_ms - oldest >= self.config.wait_force_ms
                    if (
                        stage_active + len(group) > self.config.wait_stage_limits[stage]
                        and not forced
                    ):
                        continue
                solo = self.projected_memory(group)
                if solo > self.config.memory_budget_units:
                    for job in group:
                        job.state = "refused"
                        job.terminal_at_ms = self.now_ms
                        self.event(job, "terminal_refused", reason="hard_memory_authority")
                    selected = []
                    break
                projected = self.projected_memory(active + group)
                if projected <= self.config.memory_budget_units:
                    selected = group
                    break
                if self.arm != "wait_pack":
                    return
            if selected is None:
                return
            if not selected:
                continue
            projected = self.projected_memory(active + selected)
            if projected > self.config.memory_budget_units:
                self.stats["hard_memory_violations"] += 1
                raise AssertionError("shadow policy bypassed final-extent memory authority")
            self.peak_projected_memory = max(self.peak_projected_memory, projected)
            for job in selected:
                job.state = "prefill"
                job.admitted_at_ms = self.now_ms
                wait = self.now_ms - job.request.arrival_ms
                self.queue_waits.append(wait)
                self.event(
                    job,
                    "admitted",
                    projected_cohort_memory_units=projected,
                    authority="LinearStateCost.cohort_bytes(final_extent)",
                )
                self.event(job, "state_created")

    def action_epoch(
        self,
        reserved: tuple[ReservedRows, ...],
        prefill: tuple[int, PrefillOption] | None,
    ) -> str:
        uids = [item.lane_id for item in reserved]
        if prefill is not None:
            uids.append(prefill[0])
        epochs = {self.jobs[uid].request.cost_epoch for uid in uids}
        return "+".join(sorted(epochs)) if epochs else "idle"

    def _offers(self, prefill: list[Job]) -> tuple[PrefillOffer, ...]:
        def options(job: Job) -> tuple[PrefillOption, ...]:
            rows = sorted({
                min(job.remaining_prefill, size)
                for size in self.config.prefill_options
                if min(job.remaining_prefill, size) > 0
            })
            return tuple(
                PrefillOption(value, math.ceil(value / self.config.page_tokens))
                for value in rows
            )

        ordered = sorted(prefill, key=lambda job: (job.admitted_at_ms, job.request.uid))
        if self.arm == "wait_pack":
            def score(job: Job) -> tuple[float, float, int]:
                largest = options(job)[-1]
                proxy_ms = self.config.nominal_fixed_ms + self.config.nominal_row_ms * largest.rows
                age = self.now_ms - job.request.arrival_ms
                progress_per_ms = largest.rows / proxy_ms
                return (-progress_per_ms - age / 1_000_000.0, job.request.arrival_ms, job.request.uid)

            ordered.sort(key=score)
        return tuple(PrefillOffer(job.request.uid, options(job)) for job in ordered)

    def _round(self) -> None:
        decode = sorted(
            (job for job in self.active() if job.state == "decode"),
            key=lambda job: job.request.uid,
        )
        prefill = [job for job in self.active() if job.state == "prefill"]
        reserved = tuple(
            ReservedRows(
                job.request.uid,
                "verify" if job.request.draft_tokens else "decode",
                1 + job.request.draft_tokens,
                0,
                self.config.decode_deadline_ms,
            )
            for job in decode
        )
        offers = self._offers(prefill)
        decision = choose_paged_pack(
            reserved,
            offers,
            price=self.price,
            row_capacity=self.config.row_capacity,
            free_pages=self.config.free_pages,
            permit_candidate=True,
        )
        if not decision.accepted:
            raise RuntimeError(f"mandatory replay action is infeasible: {decision.reason}")
        selected = (
            None if decision.prefill_lane_id is None
            else (decision.prefill_lane_id, PrefillOption(
                decision.prefill_rows,
                math.ceil(decision.prefill_rows / self.config.page_tokens),
            ))
        )
        nominal = self.price.nominal_ms(reserved, selected)
        epoch = self.action_epoch(reserved, selected)
        selected_uids = [item.lane_id for item in reserved]
        if selected is not None:
            selected_uids.append(selected[0])
        residual = max(
            (self.jobs[uid].request.observed_residual_ms for uid in selected_uids),
            default=0.0,
        )
        actual_ms = nominal + residual
        reserve, calibrated = self.residuals.bound(epoch)
        if self.arm != "baseline" and not calibrated:
            self.stats["uncalibrated_rounds"] += 1
        self.residuals.observe(epoch, actual_ms - nominal)
        rows = sum(item.rows for item in reserved) + decision.prefill_rows
        self.stats["charged_rows"] += rows
        self.stats["legal_actions_considered"] += 1 + sum(len(offer.options) for offer in offers)
        self.stats[f"pack_{decision.reason}"] += 1
        if reserved:
            self.stats["deadline_rounds"] += 1
            if actual_ms > self.config.decode_deadline_ms:
                self.stats["deadline_misses"] += 1
                overshoot = actual_ms - self.config.decode_deadline_ms
                self.stats["deadline_overshoot_milli_ms"] += round(1000 * overshoot)
        finish_ms = self.now_ms + actual_ms
        self.decisions.append({
            "round": len(self.decisions),
            "start_ms": round(self.now_ms, 6),
            "arm": self.arm,
            "reason": decision.reason,
            "mandatory_decode_uids": [item.lane_id for item in reserved],
            "prefill_uid": decision.prefill_lane_id,
            "prefill_rows": decision.prefill_rows,
            "charged_rows": rows,
            "nominal_ms": round(nominal, 6),
            "residual_reserve_ms": 0.0 if self.arm == "baseline" else round(reserve, 6),
            "estimated_ms": round(float(decision.estimated_ms or 0.0), 6),
            "observed_complete_round_ms": round(actual_ms, 6),
            "cost_epoch": epoch,
            "price_profile": self.price.profile_id,
            "proxy": True,
        })
        # State transitions and token publication occur at the closed round
        # boundary, not at its submission time.
        self.now_ms = finish_ms
        if selected is not None:
            job = self.jobs[selected[0]]
            job.remaining_prefill -= selected[1].rows
            if job.remaining_prefill == 0:
                job.state = "decode"
                self.event(job, "prefill_complete")
        for job in decode:
            emitted = min(
                job.remaining_output,
                1 + job.request.accepted_draft_tokens,
            )
            if emitted:
                if job.first_token_at_ms is None:
                    job.first_token_at_ms = finish_ms
                    self.ttfts.append(finish_ms - job.request.arrival_ms)
                job.remaining_output -= emitted
                job.emitted_tokens += emitted
                self.event(job, "tokens_committed", count=emitted)
                self.event(job, "tokens_emitted", count=emitted)
                self.stats["emitted_tokens"] += emitted
            if job.remaining_output == 0:
                job.state = "completed"
                job.terminal_at_ms = finish_ms
                self.latencies.append(finish_ms - job.request.arrival_ms)
                self.event(job, "state_released", reason="completed")
                self.event(job, "terminal_completed")

    def _advance_to_next_event(self) -> bool:
        future = [
            job.request.arrival_ms
            for job in self.jobs.values()
            if job.state == "future"
        ]
        cancellations = [
            job.request.cancel_at_ms
            for job in self.jobs.values()
            if job.state in {"queued", "prefill", "decode"}
            and job.request.cancel_at_ms is not None
            and job.request.cancel_at_ms > self.now_ms
        ]
        times = [value for value in future + cancellations if value is not None]
        if not times:
            return False
        self.now_ms = min(times)
        return True

    def run(self) -> dict:
        for _ in range(self.config.max_rounds):
            self._arrivals_and_cancellations()
            self._admit()
            if self.active():
                self._round()
                continue
            terminal = {"completed", "cancelled", "refused"}
            if all(job.state in terminal for job in self.jobs.values()):
                break
            if not self._advance_to_next_event():
                raise RuntimeError("replay stalled with nonterminal requests")
        else:
            raise RuntimeError("replay exceeded max_rounds")
        audit = audit_events(self.events)
        statuses = Counter(job.state for job in self.jobs.values())
        completed_tokens = sum(
            job.emitted_tokens for job in self.jobs.values() if job.state == "completed"
        )
        deadline_rounds = self.stats["deadline_rounds"]
        metrics = {
            "eligible_requests": len(self.jobs),
            "completed_requests": statuses["completed"],
            "cancelled_requests": statuses["cancelled"],
            "refused_requests": statuses["refused"],
            "completion_fraction": statuses["completed"] / len(self.jobs),
            "makespan_ms": round(self.now_ms, 6),
            "completed_contract_tokens": completed_tokens,
            "completed_tokens_per_second": (
                0.0 if self.now_ms == 0 else 1000.0 * completed_tokens / self.now_ms
            ),
            "deadline_rounds": deadline_rounds,
            "deadline_misses": self.stats["deadline_misses"],
            "deadline_miss_fraction": (
                0.0 if not deadline_rounds else self.stats["deadline_misses"] / deadline_rounds
            ),
            "deadline_overshoot_ms": self.stats["deadline_overshoot_milli_ms"] / 1000.0,
            "ttft_p99_ms": percentile(self.ttfts, 0.99),
            "latency_p99_ms": percentile(self.latencies, 0.99),
            "max_queue_wait_ms": max(self.queue_waits, default=0.0),
            "peak_projected_memory_units": self.peak_projected_memory,
            "hard_memory_violations": self.stats["hard_memory_violations"],
            "charged_rows": self.stats["charged_rows"],
            "committed_tokens_per_charged_row": (
                0.0 if not self.stats["charged_rows"]
                else completed_tokens / self.stats["charged_rows"]
            ),
            "uncalibrated_rounds": self.stats["uncalibrated_rounds"],
            "pack_decisions": {
                key.removeprefix("pack_"): value
                for key, value in sorted(self.stats.items())
                if key.startswith("pack_")
            },
        }
        return {
            "arm": self.arm,
            "metrics": metrics,
            "conservation": audit,
            "decisions": self.decisions,
            "events": self.events,
        }


def percentile(values: Iterable[float], probability: float) -> float | None:
    values = sorted(float(value) for value in values)
    if not values:
        return None
    index = max(0, min(len(values) - 1, math.ceil(probability * len(values)) - 1))
    return round(values[index], 6)


def audit_events(events: Iterable[dict]) -> dict:
    """Independent lifecycle/token conservation reducer over immutable events."""

    by_uid: dict[int, list[dict]] = defaultdict(list)
    violations = []
    for expected, event in enumerate(events):
        if event.get("sequence") != expected:
            violations.append("event sequence is not contiguous")
        by_uid[event["logical_request_id"]].append(event)
    state_residual = token_residual = terminal_residual = 0
    for uid, rows in by_uid.items():
        attempts = {row.get("attempt_id") for row in rows}
        if len(attempts) != 1 or None in attempts:
            violations.append(f"request {uid} has ambiguous attempt identity")
        eligible = sum(row["kind"] == "eligible" for row in rows)
        terminals = sum(row["kind"].startswith("terminal_") for row in rows)
        created = sum(row["kind"] == "state_created" for row in rows)
        released = sum(row["kind"] == "state_released" for row in rows)
        committed = sum(row.get("count", 0) for row in rows if row["kind"] == "tokens_committed")
        emitted = sum(row.get("count", 0) for row in rows if row["kind"] == "tokens_emitted")
        terminal_residual += eligible - terminals
        state_residual += created - released
        token_residual += committed - emitted
        if eligible != 1 or terminals != 1:
            violations.append(f"request {uid} lacks one eligible/terminal event")
        if created not in (0, 1) or released != created:
            violations.append(f"request {uid} state ownership is unbalanced")
    residuals = {
        "eligible_minus_terminal": terminal_residual,
        "created_minus_released_minus_live": state_residual,
        "committed_minus_emitted_minus_buffered_minus_discarded": token_residual,
    }
    if any(residuals.values()):
        violations.append("nonzero conservation residual")
    return {"passed": not violations, "residuals": residuals, "violations": violations}


def compare_arm(candidate: dict, baseline: dict) -> dict:
    c = candidate["metrics"]
    b = baseline["metrics"]
    miss_reduction = (
        0.0 if b["deadline_miss_fraction"] == 0
        else 1.0 - c["deadline_miss_fraction"] / b["deadline_miss_fraction"]
    )
    if b["completed_tokens_per_second"] == 0:
        throughput_gain = 0.0 if c["completed_tokens_per_second"] == 0 else 1.0
    else:
        throughput_gain = (
            c["completed_tokens_per_second"] / b["completed_tokens_per_second"] - 1.0
        )
    hard_gates = {
        "conservation": candidate["conservation"]["passed"],
        "hard_memory": c["hard_memory_violations"] == 0,
        "no_hidden_refusal_gain": c["refused_requests"] <= b["refused_requests"],
        "completion_not_lower": c["completed_requests"] >= b["completed_requests"],
        "queue_wait_bounded": c["max_queue_wait_ms"] <= max(
            b["max_queue_wait_ms"] * 1.25,
            b["max_queue_wait_ms"] + 100.0,
        ),
    }
    useful_effect = miss_reduction >= 0.20 or throughput_gain >= 0.05
    no_material_throughput_harm = throughput_gain >= -0.05
    return {
        "deadline_miss_relative_reduction": miss_reduction,
        "completed_token_throughput_gain": throughput_gain,
        "hard_gates": hard_gates,
        "helpful_on_this_proxy_trace": (
            all(hard_gates.values()) and useful_effect and no_material_throughput_harm
        ),
        "kill_reasons": [name for name, passed in hard_gates.items() if not passed]
        + ([] if no_material_throughput_harm else ["throughput_regression_gt_5pct"])
        + ([] if useful_effect else ["no_predeclared_useful_effect"]),
    }


def build_receipt(
    trace: Iterable[TraceRequest] | None = None,
    config: ReplayConfig | None = None,
    *,
    logical_run_id: str = "ligo-flow-default",
    attempt_id: str | None = None,
    source_revision: str | None = None,
) -> dict:
    trace = validate_trace(default_trace() if trace is None else trace)
    config = config or ReplayConfig()
    if attempt_id is None:
        digest = hashlib.sha256(
            json.dumps([asdict(item) for item in trace], sort_keys=True).encode()
        ).hexdigest()[:16]
        attempt_id = f"{logical_run_id}:{digest}"
    arms = {
        arm: Replay(arm, trace, config, logical_run_id=logical_run_id).run()
        for arm in ARMS
    }
    comparisons = {
        arm: compare_arm(arms[arm], arms["baseline"])
        for arm in ARMS if arm != "baseline"
    }
    positive_outputs = [
        item.actual_output_tokens for item in trace if item.actual_output_tokens
    ]
    return {
        "schema": SCHEMA,
        "logical_run_id": logical_run_id,
        "attempt_id": attempt_id,
        "source_revision": source_revision,
        "status": "passed" if all(
            result["conservation"]["passed"]
            and result["metrics"]["hard_memory_violations"] == 0
            for result in arms.values()
        ) else "failed",
        "implemented": True,
        "qualified": False,
        "selected": False,
        "selected_by_default": False,
        "observed_used": False,
        "replay_observed_used": True,
        "route_observed_used": False,
        "performance_claim": False,
        "scope": (
            "deterministic CPU shadow replay; time, memory and charged rows are "
            "declared proxies, not device measurements"
        ),
        "policy_contract": {
            "future_outcomes_hidden": [
                "actual_output_tokens",
                "accepted_draft_tokens",
                "cancel_at_ms",
                "observed_residual_ms",
            ],
            "hard_memory_authority": "LinearStateCost.cohort_bytes(final_extent)",
            "hard_pack_authority": "choose_paged_pack",
            "unpriced_action": "refuse",
            "atomic_cohort": "all_members_or_none",
        },
        "kill_criteria": {
            "correctness": "zero conservation, memory-authority, or atomic-cohort violations",
            "coverage": "no benefit obtained by increasing refusals or reducing completions",
            "fairness": "max queue wait no more than max(1.25x baseline, baseline + 100 ms)",
            "utility": "at least 20% relative deadline-miss reduction or 5% completed-token throughput gain",
            "harm": "kill above 5% completed-token throughput regression",
        },
        "config": asdict(config),
        "trace": {
            "requests": [asdict(item) for item in trace],
            "features": {
                "heavy_tail": bool(positive_outputs)
                and max(positive_outputs) >= 10 * min(positive_outputs),
                "cancellations": sum(item.cancel_at_ms is not None for item in trace),
                "warm_requests": sum(item.warm_prefix_tokens > 0 for item in trace),
                "atomic_cohorts": len({item.cohort_id for item in trace if item.cohort_id}),
                "cost_epochs": sorted({item.cost_epoch for item in trace}),
            },
        },
        "arms": arms,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--logical-run-id", default="ligo-flow-default")
    parser.add_argument("--attempt-id")
    parser.add_argument("--source-revision")
    args = parser.parse_args()
    receipt = build_receipt(
        logical_run_id=args.logical_run_id,
        attempt_id=args.attempt_id,
        source_revision=args.source_revision,
    )
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
