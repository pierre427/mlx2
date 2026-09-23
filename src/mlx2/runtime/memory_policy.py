# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
import math
import os
import platform
from dataclasses import dataclass, replace
from typing import (
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)


@dataclass(frozen=True)
class SelfMTPLaneAdmission:
    """One cycle-boundary memory decision for batched self-MTP.

    ``modes`` and ``draft_depths`` are aligned with the caller's lane order.
    An MTP lane has depth 1 or 2, a plain lane has depth 0, and a queued lane
    has depth ``None``.  The decision is immutable so the exact budget checked
    for a cycle can be logged or asserted without later membership changes
    rewriting it.
    """

    modes: Tuple[Literal["self_mtp", "plain", "queue"], ...]
    draft_depths: Tuple[Optional[int], ...]
    stage: Literal["full", "fewer_lanes", "lower_k", "plain", "queue"]
    estimated_gib: float
    usable_gib: float
    primary_rows: int = 0
    speculative_rows: int = 0
    optional_reclaim_wait: bool = False

    @property
    def mtp_indices(self) -> Tuple[int, ...]:
        return tuple((i for (i, mode) in enumerate(self.modes) if mode == "self_mtp"))


class SelfMTPLaneAdmissionController:
    """Fail-closed memory/context policy for the M=(k+1)N verify forward.

    The production PLE-offload operating point is about 72.5 GiB resident on
    the 128 GiB calibration host, leaving about 55.5 GiB free.  A 20 GiB hard
    margin (16 GiB service reserve plus 4 GiB for the driver) leaves 35.5 GiB
    for lane cache growth and the verify transient.  The linear envelope below
    is calibrated so that this operating point admits N=16 around 1K and N=4
    around 16K:

      cache/context share: 0.44 GiB per 1K tokens per lane
      k=2 verify transient: 1.76 GiB per lane

    Reducing k from 2 to 1 halves only the verify transient.  Plain M=N keeps
    one third of the k=2 verify transient.  This is deliberately an envelope,
    not a claim about allocator internals.  Inputs that cannot be measured are
    queued rather than guessed.

    Memory is not the only ceiling.  The M=(k+1)N verify forward saturates the
    GPU near a fixed lane count; past it, aggregate throughput falls even when
    memory permits more lanes.  A dense Qwen3.8-27B (~18 GiB resident) leaves
    ~95 GiB free, so the memory envelope alone would admit N~40 at short
    context -- 2.5x past the measured throughput peak (N=16: 270 t/s agg;
    N=40: 52 t/s).  ``SATURATION_LANE_CAP`` bounds the admitted subset so the
    N~16 knee holds regardless of how much memory is free.  At Flash-Next's
    72.5 GiB PLE operating point the memory envelope already caps near 16 at
    1K, so this cap is a no-op there and only binds when free memory is large.
    ``transient_gib_per_lane`` is the peak *transient* device memory of one
    lane's M=(k+1) verify forward: the activation working set that is
    allocated and released inside the forward, measured above the resident
    state that survives it, and therefore excluding both the model weights
    and the K/V + recurrent cache (including that cache's own growth during
    the forward).  ``TRANSIENT_SCALE`` rescales it for other depths.  It is
    charged per lane and is context-independent.

    The k=2 figure is calibrated on Flash-Next (MoE, 6B active) at 1.76, and
    a dense Qwen3.8-27B measures ~3.1 GiB/lane, so it is configurable per
    adapter for dense deployments at long context where it dominates.  The
    dense 3.1 was then copied onto three small-active MoE adapters it was
    never measured on.  Direct measurement on an M3 Pro (see
    ``MOE_TRANSIENT_GIB_PER_LANE`` and provenance/lane-transient-moe.json)
    shows that figure is wrong by two orders of magnitude for that class and
    large enough, on a small host, to consume the whole lane budget and
    silently disable the self-MTP route it is meant to size.

    The hard margin is *not* an absolute.  16 GiB and 4 GiB are the values of
    a rule on the 128 GiB calibration host, not the rule itself.  Held fixed
    they scale catastrophically downward: the same M3 Pro has only ~11.5 GiB
    of advisory residue left after a 16 GiB model, so a flat 20 GiB reserve
    refuses every request before a single lane is costed.  3fa14e7 made them
    host-scaled; ``host_scaled_reserves`` below re-derives the basis, capped
    at the calibrated absolutes on larger hosts and floored by
    ``MIN_SERVICE_RESERVE_GIB`` / ``MIN_DRIVER_ALLOWANCE_GIB``.

    3fa14e7 keyed both figures off **physical** unified memory and argued
    that Metal's ``max_recommended_working_set_size`` must not enter, because
    the advisory is a per-process ceiling already applied once by
    ``available_execution_bytes``, so scaling the reserve by it too would
    subtract the OS margin twice.  The premise is right and the conclusion
    does not follow: it is the *physical* rule that double-charges, and by a
    margin that grows as the host shrinks.

    ``available_execution_bytes`` hands ``decide``
    ``min(host available, advisory - in use)``.  The second term is where the
    OS margin already lives.  ``host - advisory`` is exactly what macOS has
    withheld from this process before we reserve anything, and that share is
    **not** a constant fraction of the host:

      128 GiB host -> 112.00 GiB advisory: 16.00 GiB withheld, 12.5%
       36 GiB host ->  28.08 GiB advisory:  7.92 GiB withheld, 22.0%

    (both measured, 2026-09-19).  A reserve of 12.5% of physical RAM is
    therefore a second copy of the OS margin *sized as if macOS always
    withheld 12.5%*.  On the calibration host that happens to be true, so the
    two charges together withhold 32 GiB = 25% of the machine.  On the M3 Pro
    macOS has already taken 22% and the physical rule charges another 12.5%
    on top, withholding 34.5% of a host that has a third of the RAM -- and it
    charges it against the term that binds there, the advisory residue, which
    a resident model drives toward zero.  Muse-Glimmer-30B + DFlash2 at
    21.56 GiB resident leaves 6.52 GiB of residue; a 5.625 GiB reserve takes
    86% of it and the request 429s with the machine two-thirds empty.

    The rule below charges the quota **once**.  The host's non-lane share is
    ``NON_LANE_HOST_QUOTA_FRACTION`` of physical RAM -- 25%, read off the
    calibration host, where the advisory's 16 GiB and the reserve's 16 GiB
    make it up between them -- and the service reserve supplies only the part
    the advisory has not already collected:

        service = clamp(floor, 16.0, advisory - 0.75 * host)

    128 GiB: ``112 - 96 = 16.0``, bit for bit, so nothing calibrated there
    moves.  36 GiB: ``28.08 - 27.0 = 1.08``, i.e. macOS has already collected
    all but 1.08 GiB of the quota, and the floor below takes over.

    The driver allowance is a different quantity and keeps a different basis.
    It is the Metal/IOGPU driver's own working set, which is charged *inside*
    our process budget and is not withheld by the advisory at all, so it
    scales with the advisory rather than with the host:
    ``advisory * 4.0 / 112.0`` -- again exactly 4.0 on the calibration host,
    and 1.003 on the M3.

    With only one of the two readings available the other is imputed at the
    calibration host's ratio (``ADVISORY_RATIO`` = 0.875).  Under that
    imputation ``service = 0.875h - 0.75h = 0.125h`` and
    ``driver = 0.875h * 4/112 = 0.03125h``: 3fa14e7's physical rule *is* this
    rule evaluated at a 0.875 advisory ratio.  It is not a different policy,
    it is this policy with a hardcoded assumption about macOS, and the
    assumption is what fails on a small host.

    ``decide`` is stateless on purpose.  The server calls it at every decode
    cycle boundary with fresh free memory and current per-lane contexts, so
    lane joins/leaves and cache growth are always reflected in the next plan.
    """

    BASE_RESIDENT_GIB = 72.5
    HOST_MEMORY_GIB = 128.0
    SERVICE_RESERVE_GIB = 16.0
    DRIVER_ALLOWANCE_GIB = 4.0
    # Metal's max_recommended_working_set_size on the calibration host,
    # measured 2026-09-19: 112.0 GiB of 128.0, i.e. macOS withholds 16.0 GiB
    # from this process before admission reserves anything.
    CALIBRATION_ADVISORY_GIB = 112.0
    ADVISORY_RATIO = CALIBRATION_ADVISORY_GIB / HOST_MEMORY_GIB  # 0.875
    # The share of physical RAM withheld from lane use in total, counting
    # what the advisory already withholds: (16.0 + (128.0 - 112.0)) / 128.0.
    # Read off the calibration host, charged once.
    NON_LANE_HOST_QUOTA_FRACTION = (
        SERVICE_RESERVE_GIB + (HOST_MEMORY_GIB - CALIBRATION_ADVISORY_GIB)
    ) / HOST_MEMORY_GIB  # 0.25
    # Floors.  The quota above protects the rest of the machine; these floors
    # protect us from ourselves once the advisory has collected the quota.
    # The advisory is a recommendation, not a hard cap -- exceeding it does
    # not fail an allocation, it starts swapping -- and lane_gib is an
    # envelope with error.  3.0 GiB covers the largest unmodelled excursion
    # we have measured (peak-minus-steady across the MoE transient sweep in
    # provenance/lane-transient-moe.json tops out at 0.281 GiB/lane on the
    # cold-allocator forward; prefill temporaries on a 20 GiB MoE at 16K are
    # the next term) with roughly an order of magnitude of margin, and it is
    # also what macOS idles at in wired plus compressed pages.  0.75 GiB is
    # the driver working set that does not shrink to nothing on a small
    # machine.  Both are unchanged from 3fa14e7.
    MIN_SERVICE_RESERVE_GIB = 3.0
    MIN_DRIVER_ALLOWANCE_GIB = 0.75
    CACHE_GIB_PER_1K_TOKENS = 0.44
    TRANSIENT_SCALE = {0: 1.0 / 3.0, 1: 0.8, 2: 1.0, 3: 1.25, 4: 1.55}
    K2_TRANSIENT_GIB_PER_LANE = 1.76
    # Measured 2026-09-19 on an M3 Pro (36 GiB, Metal advisory 28.08 GiB) with
    # mx.get_peak_memory()/get_active_memory() around real verify forwards, on
    # two small-active MoE models: Qwen3.6-35B-A3B (256 experts, 8 active,
    # 20 GiB q4) and North-Mini-Code-1.0 (128 experts, 8 active, 16 GiB q4).
    # Sweep k in {0,1,2} x context in {1K,4K,16K} x lanes in {1,2,4}, 3 reps.
    # The k=2 per-lane transient stayed between 0.015 and 0.071 GiB across all
    # 36 measured cells -- flat in context (a 16x context range moved it by
    # under 2x) and sublinear in batch -- and the largest value seen anywhere,
    # including the cold-allocator spike on the first forward after a cache
    # flush, was 0.281 GiB/lane.  0.35 is 1.25x that worst observation and
    # about 5x the worst steady-state cell.  Raw numbers in
    # provenance/lane-transient-moe.json.  This replaces the dense-27B 3.1
    # that had been copied onto MoE adapters; the dense 3.1 and Flash-Next's
    # own 1.76 are untouched (Flash-Next's 127 GB artifact is not on the M3,
    # so it could not be re-measured and was not changed).
    MOE_TRANSIENT_GIB_PER_LANE = 0.35
    SATURATION_LANE_CAP = 16
    RESIDENT_GROWTH_HORIZON_TOKENS = 1024
    READMIT_MARGIN_GIB = 4.0
    READMIT_MAX_HOLDS = 8

    @staticmethod
    def _reading(value: Optional[float]) -> Optional[float]:
        """A probe result, or ``None`` for "unknown" (never an error)."""
        if value is None:
            return None
        candidate = float(value)
        if not math.isfinite(candidate) or candidate <= 0:
            return None
        return candidate

    @classmethod
    def host_scaled_reserves(
        cls,
        host_memory_gib: Optional[float] = None,
        advisory_gib: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Return ``(service_reserve_gib, driver_allowance_gib)`` for a host.

        ``host_memory_gib`` is *physical* unified memory and ``advisory_gib``
        is Metal's ``max_recommended_working_set_size``; both are best-effort
        probes and either may be ``None``.  See the class docstring for why
        the rule needs both: the service reserve supplies only the part of
        the host's non-lane quota that the advisory has not already taken.

        With neither reading this returns the 128 GiB calibration, so an
        unmeasurable host degrades to what shipped rather than to a guess.
        With one reading the other is imputed at ``ADVISORY_RATIO``, which
        reproduces 3fa14e7's physical-RAM rule exactly.
        """
        host = cls._reading(host_memory_gib)
        advisory = cls._reading(advisory_gib)
        if host is None and advisory is None:
            (host, advisory) = (cls.HOST_MEMORY_GIB, cls.CALIBRATION_ADVISORY_GIB)
        elif advisory is None:
            advisory = cls.ADVISORY_RATIO * host
        elif host is None:
            host = advisory / cls.ADVISORY_RATIO
        # An advisory above physical RAM is not a reading we can reason
        # about; fall back to the imputed one rather than hand the service
        # term a bonus for it.
        advisory = min(advisory, host)
        # Charge the host's non-lane quota once: the advisory has already
        # taken ``host - advisory`` of it.
        already_withheld = host - advisory
        shortfall = cls.NON_LANE_HOST_QUOTA_FRACTION * host - already_withheld
        service = max(
            cls.MIN_SERVICE_RESERVE_GIB,
            min(cls.SERVICE_RESERVE_GIB, shortfall),
        )
        driver = max(
            cls.MIN_DRIVER_ALLOWANCE_GIB,
            min(
                cls.DRIVER_ALLOWANCE_GIB,
                advisory * cls.DRIVER_ALLOWANCE_GIB / cls.CALIBRATION_ADVISORY_GIB,
            ),
        )
        return (service, driver)

    def __init__(
        self,
        *,
        host_memory_gib: Optional[float] = None,
        advisory_gib: Optional[float] = None,
        service_reserve_gib: Optional[float] = None,
        driver_allowance_gib: Optional[float] = None,
        transient_gib_per_lane: float = K2_TRANSIENT_GIB_PER_LANE,
        stream_reserve_gib: float = 0.0,
        saturation_lane_cap: Optional[int] = SATURATION_LANE_CAP,
        verification_row_cap: Optional[int] = None,
        cache_estimator: Optional[Callable[[int], int]] = None,
    ):
        (service_floor, driver_floor) = self.host_scaled_reserves(
            host_memory_gib, advisory_gib
        )
        if service_reserve_gib is None:
            service_reserve_gib = service_floor
        if driver_allowance_gib is None:
            driver_allowance_gib = driver_floor
        service_reserve_gib = float(service_reserve_gib)
        driver_allowance_gib = float(driver_allowance_gib)
        # An explicit override may raise the reserve but never lower it below
        # what this host needs; the floor is host-scaled instead of a flat
        # 16 GiB so a small Mac is configurable at all.
        if not math.isfinite(service_reserve_gib) or service_reserve_gib <= 0:
            raise ValueError("self-MTP service reserve must be finite and positive")
        if service_reserve_gib < service_floor:
            raise ValueError(
                "self-MTP service reserve must be at least the host-scaled "
                f"floor of {service_floor:.4g} GiB"
            )
        if not math.isfinite(driver_allowance_gib) or driver_allowance_gib < 0:
            raise ValueError("self-MTP driver allowance must be non-negative")
        if driver_allowance_gib < driver_floor:
            raise ValueError(
                "self-MTP driver allowance must be at least the host-scaled "
                f"floor of {driver_floor:.4g} GiB"
            )
        if not math.isfinite(transient_gib_per_lane) or transient_gib_per_lane <= 0:
            raise ValueError("self-MTP transient GiB per lane must be positive")
        stream_reserve_gib = float(stream_reserve_gib or 0.0)
        if not math.isfinite(stream_reserve_gib) or stream_reserve_gib < 0:
            raise ValueError("streamed weight reserve must be non-negative")
        if saturation_lane_cap is not None and (
            isinstance(saturation_lane_cap, bool)
            or not isinstance(saturation_lane_cap, int)
            or saturation_lane_cap < 1
        ):
            raise ValueError(
                "self-MTP saturation lane cap must be a positive int or None"
            )
        if verification_row_cap is not None and (
            isinstance(verification_row_cap, bool)
            or not isinstance(verification_row_cap, int)
            or verification_row_cap < 1
        ):
            raise ValueError("verification row cap must be a positive int or None")
        self.cache_estimator = cache_estimator
        self.host_memory_gib = host_memory_gib
        self.advisory_gib = advisory_gib
        self.service_reserve_gib = float(service_reserve_gib)
        self.driver_allowance_gib = float(driver_allowance_gib)
        self.transient_gib_per_lane = float(transient_gib_per_lane)
        self.stream_reserve_gib = stream_reserve_gib
        self.saturation_lane_cap = saturation_lane_cap
        self.verification_row_cap = verification_row_cap

    @property
    def hard_reserve_gib(self) -> float:
        """Service + driver + the streamed weight cache ceiling.

        ``stream_reserve_gib`` is the streaming manager's **enforced** ceiling,
        not an estimate of its working set, and it is subtracted exactly once
        here -- before any lane is costed -- so a lane admitted at cycle N
        cannot be starved by expert cache growth at cycle N+1.  A streamed
        model's resident charge is therefore ``R_fixed + B_stream``: its
        non-streamable remainder plus this ceiling, never its file size.

        It composes with, rather than replaces, the other two terms.  The
        service and driver reserves protect the rest of the host and key off
        physical RAM (or, if the in-flight advisory re-keying lands, off the
        Metal advisory); this term protects admission from our own cache and
        is sized against ``max_recommended_working_set_size`` by
        ``mlx2.runtime.weight_stream``.  Whichever way the other two are
        derived, adding a third independent reservation here is still correct.
        This branch assumes ``main``; if the advisory re-keying lands first,
        nothing here needs to change.
        """
        return (
            self.service_reserve_gib
            + self.driver_allowance_gib
            + self.stream_reserve_gib
        )

    def lane_gib(
        self,
        context_tokens: int,
        draft_depth: int,
        cache_gib: float = 0.0,
        *,
        resident_cache: bool = False,
        pending_gib: float = 0.0,
    ) -> float:
        """Conservative incremental cost for one lane.

        Before preparation, ``cache_gib`` is a projected/retained cache floor
        and the full cache plus transient must fit in currently free memory.
        At a cycle boundary an active lane's cache is already resident and is
        therefore already absent from the live free-memory measurement.  In
        that case only near-term growth plus the next verify transient is
        incremental.  Charging the resident cache a second time can queue the
        sole lane after prefill, where it can never release the allocation
        required to admit itself again.

        With a measured cache, near-term growth is
        ``RESIDENT_GROWTH_HORIZON_TOKENS`` at the larger of the measured and
        envelope per-token rates; a joining lane also pays the measured cache
        (for example an APC restore scaled to its post-prepare context).  The
        envelope is calibrated on full-attention growth.  A hybrid cache
        (recurrent state plus sparse attention) is far smaller at long
        context, and charging the envelope gap queued a 57K lane on most
        cycles.  An unmeasured cache still pays the full envelope.
        ``pending_gib`` is lazy memory that is not allocated yet; it is always
        charged in full.
        """
        if isinstance(context_tokens, bool) or not isinstance(context_tokens, int):
            raise ValueError("context_tokens must be an integer")
        if context_tokens < 0:
            raise ValueError("context_tokens must be non-negative")
        if draft_depth not in self.TRANSIENT_SCALE:
            raise ValueError(
                f"draft_depth must be one of {sorted(self.TRANSIENT_SCALE)}"
            )
        cache_gib = float(cache_gib)
        if not math.isfinite(cache_gib) or cache_gib < 0:
            raise ValueError("cache_gib must be finite and non-negative")
        if not isinstance(resident_cache, bool):
            raise ValueError("resident_cache must be a boolean")
        pending_gib = float(pending_gib)
        if not math.isfinite(pending_gib) or pending_gib < 0:
            raise ValueError("pending_gib must be finite and non-negative")
        envelope_gib = self.CACHE_GIB_PER_1K_TOKENS * (context_tokens / 1024.0)
        growth_floor = self.CACHE_GIB_PER_1K_TOKENS / 1024.0
        if self.cache_estimator is not None:
            projected = float(self.cache_estimator(context_tokens))
            future = float(self.cache_estimator(context_tokens + self.RESIDENT_GROWTH_HORIZON_TOKENS))
            if not math.isfinite(projected) or projected < 0 or not math.isfinite(future) or future < projected:
                raise ValueError("adapter cache estimator must be finite, nonnegative and monotone")
            envelope_gib = projected / (1 << 30)
            growth_floor = (future - projected) / self.RESIDENT_GROWTH_HORIZON_TOKENS / (1 << 30)
        if cache_gib > 0.0 and context_tokens > 0:
            if self.cache_estimator is not None:
                # Hybrid recurrent state is fixed, not a per-token rate. The
                # model's finite difference bounds ordinary cache growth;
                # conservatively extrapolate only observed bytes beyond that
                # complete bound (for an unexpected cache extension/view).
                gib_per_token = growth_floor + max(0.0, cache_gib - envelope_gib) / context_tokens
            else:
                gib_per_token = max(cache_gib / context_tokens, growth_floor)
            growth_gib = gib_per_token * self.RESIDENT_GROWTH_HORIZON_TOKENS
            # A complete measured warm state gives its actual storage dtype
            # and allocation capacity. Charge its entire copy plus conservative
            # growth; do not replace observed bf16 bytes with a cold fp32 bound.
            context_gib = growth_gib if resident_cache else cache_gib + growth_gib
        else:
            context_gib = envelope_gib
        transient_scale = self.TRANSIENT_SCALE[draft_depth]
        return context_gib + pending_gib + self.transient_gib_per_lane * transient_scale

    def _fit(
        self,
        indices: Sequence[int],
        contexts: Sequence[int],
        cache_gib: Sequence[float],
        resident_cache: Sequence[bool],
        draft_depth: int,
        usable_gib: float,
        max_lanes: Optional[int] = None,
        pending_gib: Optional[Sequence[float]] = None,
    ) -> Tuple[Tuple[int, ...], float]:
        if pending_gib is None:
            pending_gib = (0.0,) * len(contexts)
        ranked = sorted(
            indices,
            key=lambda i: (
                self.lane_gib(
                    contexts[i],
                    draft_depth,
                    cache_gib[i],
                    resident_cache=resident_cache[i],
                    pending_gib=pending_gib[i],
                ),
                i,
            ),
        )
        chosen = []
        used = 0.0
        for i in ranked:
            if max_lanes is not None and len(chosen) >= max_lanes:
                break
            cost = self.lane_gib(
                contexts[i],
                draft_depth,
                cache_gib[i],
                resident_cache=resident_cache[i],
                pending_gib=pending_gib[i],
            )
            if used + cost <= usable_gib:
                chosen.append(i)
                used += cost
        return (tuple(sorted(chosen)), used)

    def decide(
        self,
        context_tokens: Sequence[int],
        free_memory_gib: float,
        *,
        eligible: Optional[Sequence[bool]] = None,
        cache_gib: Optional[Sequence[float]] = None,
        resident_cache: Optional[Sequence[bool]] = None,
        max_draft: int = 2,
        pending_gib: Optional[Sequence[float]] = None,
        atomic_cohort: bool = False,
    ) -> SelfMTPLaneAdmission:
        """Return the next-cycle plan in the frozen degradation order.

        Excluded lanes route directly to plain decode.  For eligible lanes the
        controller tries, in order: the largest safe k=2 subset; if no k=2
        lane fits, the largest safe k=1 subset; if none fits, one safe plain
        lane; otherwise queue. Thus lowering k never jumps ahead of admitting
        fewer full-depth lanes. A declared indivisible cohort sets
        ``atomic_cohort``: every eligible lane must fit at one uniform depth,
        trying maximum k then successively lower k, or the whole group queues.
        """
        contexts = tuple(context_tokens)
        if eligible is None:
            eligible = (True,) * len(contexts)
        else:
            eligible = tuple(eligible)
        if len(eligible) != len(contexts):
            raise ValueError("eligible must align with context_tokens")
        if cache_gib is None:
            cache_gib = (0.0,) * len(contexts)
        else:
            cache_gib = tuple(cache_gib)
        if len(cache_gib) != len(contexts):
            raise ValueError("cache_gib must align with context_tokens")
        if resident_cache is None:
            resident_cache = (False,) * len(contexts)
        else:
            resident_cache = tuple(resident_cache)
        if len(resident_cache) != len(contexts):
            raise ValueError("resident_cache must align with context_tokens")
        if pending_gib is None:
            pending_gib = (0.0,) * len(contexts)
        else:
            pending_gib = tuple(pending_gib)
        if len(pending_gib) != len(contexts):
            raise ValueError("pending_gib must align with context_tokens")
        if max_draft not in self.TRANSIENT_SCALE:
            raise ValueError(
                f"max_draft must be a calibrated depth: {sorted(self.TRANSIENT_SCALE)}"
            )
        modes: List[Literal["self_mtp", "plain", "queue"]] = [
            "plain" if not ok else "queue" for ok in eligible
        ]
        depths: List[Optional[int]] = [0 if not ok else None for ok in eligible]
        mtp_candidates = [i for (i, ok) in enumerate(eligible) if ok]
        if not mtp_candidates:
            return SelfMTPLaneAdmission(
                tuple(modes), tuple(depths), "plain", 0.0, 0.0, len(contexts), 0
            )
        try:
            free = float(free_memory_gib)
            valid = math.isfinite(free) and free >= 0
            for i in mtp_candidates:
                self.lane_gib(
                    contexts[i],
                    max_draft,
                    cache_gib[i],
                    resident_cache=resident_cache[i],
                    pending_gib=pending_gib[i],
                )
        except (TypeError, ValueError, OverflowError):
            valid = False
            free = 0.0
        usable = max(free - self.hard_reserve_gib, 0.0) if valid else 0.0
        if not valid:
            return SelfMTPLaneAdmission(
                tuple(modes), tuple(depths), "queue", 0.0, usable, len(contexts), 0
            )
        if max_draft == 0:
            (chosen, used) = self._fit(
                mtp_candidates,
                contexts,
                cache_gib,
                resident_cache,
                0,
                usable,
                pending_gib=pending_gib,
            )
            if atomic_cohort and len(chosen) != len(mtp_candidates):
                chosen = ()
                used = 0.0
            for i in chosen:
                modes[i] = "plain"
                depths[i] = 0
            return SelfMTPLaneAdmission(
                tuple(modes),
                tuple(depths),
                "plain" if chosen else "queue",
                used,
                usable,
                len(contexts),
                0,
            )
        for depth in range(max_draft, 0, -1):
            lane_cap = self.saturation_lane_cap
            if self.verification_row_cap is not None:
                branch_cap = max(
                    0, (self.verification_row_cap - len(contexts)) // depth
                )
                lane_cap = branch_cap if lane_cap is None else min(lane_cap, branch_cap)
            (chosen, used) = self._fit(
                mtp_candidates,
                contexts,
                cache_gib,
                resident_cache,
                depth,
                usable,
                lane_cap,
                pending_gib,
            )
            if atomic_cohort and len(chosen) != len(mtp_candidates):
                continue
            if not chosen:
                continue
            for i in chosen:
                modes[i] = "self_mtp"
                depths[i] = depth
            if depth < max_draft:
                stage = "lower_k"
            elif len(chosen) == len(mtp_candidates):
                stage = "full"
            else:
                stage = "fewer_lanes"
            return SelfMTPLaneAdmission(
                tuple(modes),
                tuple(depths),
                stage,
                used,
                usable,
                len(contexts),
                len(chosen) * depth,
            )
        if atomic_cohort:
            return SelfMTPLaneAdmission(
                tuple(modes),
                tuple(depths),
                "queue",
                0.0,
                usable,
                len(contexts),
                0,
            )
        (chosen, used) = self._fit(
            mtp_candidates,
            contexts,
            cache_gib,
            resident_cache,
            0,
            usable,
            pending_gib=pending_gib,
        )
        if chosen:
            i = chosen[0]
            modes[i] = "plain"
            depths[i] = 0
            used = self.lane_gib(
                contexts[i],
                0,
                cache_gib[i],
                resident_cache=resident_cache[i],
                pending_gib=pending_gib[i],
            )
            return SelfMTPLaneAdmission(
                tuple(modes), tuple(depths), "plain", used, usable, len(contexts), 0
            )
        return SelfMTPLaneAdmission(
            tuple(modes), tuple(depths), "queue", 0.0, usable, len(contexts), 0
        )


def _system_available_memory_bytes() -> Optional[int]:
    """Return reclaimable system memory without using allocator headroom."""
    try:
        if platform.system() == "Darwin":
            # Same free + inactive + speculative pages the vm_stat parse used,
            # read through host_statistics64 instead of a subprocess per call.
            from .os_memory import host_memory_snapshot

            snapshot = host_memory_snapshot()
            if snapshot is None:
                return None
            available = snapshot.reclaimable_bytes
            return available if available > 0 else None
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return pages * page_size if pages > 0 and page_size > 0 else None
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        return None


def _current_self_mtp_free_memory_gib() -> Optional[float]:
    """Return actual available system memory for this admission boundary."""
    available = _system_available_memory_bytes()
    if available is None or available <= 0:
        return None
    return available / float(1 << 30)


def _make_self_mtp_admission_callback(
    controller: Optional[SelfMTPLaneAdmissionController] = None,
    free_memory: Callable[[], Optional[float]] = _current_self_mtp_free_memory_gib,
    *,
    max_draft: Union[int, Callable[[], int]] = 2,
    observer: Optional[Callable[[SelfMTPLaneAdmission], None]] = None,
    reclaim_memory: Optional[Callable[[], None]] = None,
    evict_unused_cache: Optional[Callable[[], bool]] = None,
) -> Callable[
    [Sequence[Tuple[int, int, int, bool, float]]], Mapping[int, Union[int, str]]
]:
    """Adapt the pure controller to the generator's cycle-boundary seam.

    The generator calls this before every proposal, when no transaction is
    open.  It owns the resulting detach/pause/plain migration; the server owns
    the policy and the live memory measurement.  Rows are
    ``(uid, logical_context_len, current_k, resident, cache_gib[, pending_gib])``
    and include paused or joining lanes, so retained cache rows participate
    before merge. Active, paused, and explicitly owned bounded-prefill
    continuations are resident: their cache is already absent from the live
    free-memory reading. A paused row reports its merge copy as pending, and a
    prefill continuation reports its remaining projected growth as pending.
    Fresh and APC joining rows pay their cache in full.
    """
    controller = controller or SelfMTPLaneAdmissionController()
    native_demoted: Dict[int, int] = {}
    # Handed-off (ordinary) rows are admitted by ``at_depth(0)`` at the same
    # boundary as the native rows, over a disjoint row set.  Each owns its
    # READMIT hysteresis, because every cycle boundary replaces the state of
    # the rows it was given and would otherwise erase the other set's holds.
    ordinary_demoted: Dict[int, int] = {}

    def preview(
        rows,
        *,
        ceiling=False,
        atomic_cohort=False,
        max_draft_override=None,
    ):
        """Side-effect-free plan for retiring optional allocations first."""
        rows = tuple(rows)
        free = free_memory() if not ceiling else None
        depth = (
            max_draft_override
            if max_draft_override is not None
            else max_draft()
            if callable(max_draft)
            else max_draft
        )
        if ceiling:
            try:
                free = controller.hard_reserve_gib + 1.0 + sum(
                    controller.lane_gib(int(row[1]), depth,
                        float(row[4]) if len(row) > 4 else 0.0,
                        resident_cache=bool(row[3]),
                        pending_gib=float(row[5]) if len(row) > 5 else 0.0)
                    for row in rows
                )
            except (TypeError, ValueError, OverflowError):
                free = None
        return controller.decide(
            [int(row[1]) for row in rows], math.nan if free is None else free,
            cache_gib=[float(row[4]) if len(row) > 4 else 0.0 for row in rows],
            resident_cache=[bool(row[3]) for row in rows],
            max_draft=depth,
            pending_gib=[float(row[5]) if len(row) > 5 else 0.0 for row in rows],
            atomic_cohort=atomic_cohort,
        )

    def _admit(
        rows,
        *,
        atomic_cohort=False,
        max_draft_override=None,
        observer_stage_override=None,
        hysteresis=None,
    ):
        demoted = hysteresis if hysteresis is not None else native_demoted
        rows = tuple(rows)
        if not rows:
            demoted.clear()
            return {}
        current_free = free_memory()
        free = math.nan if current_free is None else current_free
        contexts = [int(row[1]) for row in rows]
        cache_gib = [float(row[4]) if len(row) > 4 else 0.0 for row in rows]
        pending_gib = [float(row[5]) if len(row) > 5 else 0.0 for row in rows]
        resident = [bool(row[3]) for row in rows]
        depth_cap = (
            max_draft_override
            if max_draft_override is not None
            else max_draft()
            if callable(max_draft)
            else max_draft
        )

        def plan(free_gib):
            return controller.decide(
                contexts,
                free_gib,
                cache_gib=cache_gib,
                resident_cache=resident,
                max_draft=depth_cap,
                pending_gib=pending_gib,
                atomic_cohort=atomic_cohort,
            )

        decision = plan(free)
        # Cached allocator scratch is reclaimable. Recheck measured headroom
        # before an irreversible migration to ordinary decoding or queueing.
        # This does not credit hypothetical bytes or spend the hard reserve.
        if decision.stage != "full" and reclaim_memory is not None:
            reclaim_memory()
            current_free = free_memory()
            free = math.nan if current_free is None else current_free
            decision = plan(free)
        # Idle APC checkpoints compete with the next active cohort. Reclaim
        # only unleased checkpoints before splitting/demoting that cohort;
        # preserve the immutable owners currently borrowed by its warm rows.
        # The ceiling plan retains lane/verification caps, so capacity-only
        # limits never flush a useful cache. Admission still uses observed
        # headroom after every eviction, never estimated reclaimed bytes.
        if decision.stage != "full" and evict_unused_cache is not None:
            try:
                ceiling_free = controller.hard_reserve_gib + 1.0 + sum(
                    controller.lane_gib(context, depth_cap, cache,
                        resident_cache=is_resident, pending_gib=pending)
                    for context, cache, is_resident, pending in zip(
                        contexts, cache_gib, resident, pending_gib)
                )
                ceiling = plan(ceiling_free)
            except (TypeError, ValueError, OverflowError):
                ceiling = decision
            # A bounded pass also protects this scheduling seam from a
            # callback that reports success without consuming an entry.
            for _ in range(32):
                if (decision.modes, decision.draft_depths) == (ceiling.modes, ceiling.draft_depths):
                    break
                if not evict_unused_cache():
                    break
                if reclaim_memory is not None:
                    reclaim_memory()
                current_free = free_memory()
                free = math.nan if current_free is None else current_free
                decision = plan(free)
        uids = [int(row[0]) for row in rows]
        cycle_boundary = all(resident)
        rejoining = [
            i
            for (i, uid) in enumerate(uids)
            if cycle_boundary
            and uid in demoted
            and (demoted[uid] < controller.READMIT_MAX_HOLDS)
            and (decision.modes[i] == "self_mtp")
        ]
        others_decode = any(
            (
                mode == "self_mtp" and i not in rejoining
                for (i, mode) in enumerate(decision.modes)
            )
        )
        held = []
        if rejoining and others_decode:
            strict = plan(free - controller.READMIT_MARGIN_GIB)
            held = [i for i in rejoining if strict.modes[i] != "self_mtp"]
        if held:
            modes = list(decision.modes)
            depths = list(decision.draft_depths)
            for i in held:
                modes[i] = "queue"
                depths[i] = None
            admitted = [d for (m, d) in zip(modes, depths) if m == "self_mtp"]
            decision = replace(
                decision,
                modes=tuple(modes),
                draft_depths=tuple(depths),
                stage="fewer_lanes" if decision.stage == "full" else decision.stage,
                speculative_rows=sum(admitted),
            )
        if observer_stage_override is not None and decision.stage == "full":
            decision = replace(decision, stage=observer_stage_override)
        if observer is not None:
            observer(decision)
        actions: Dict[int, Union[int, str]] = {}
        for uid, mode, depth in zip(uids, decision.modes, decision.draft_depths):
            actions[uid] = int(depth) if mode == "self_mtp" else mode
        if cycle_boundary:
            holds = {uids[i]: demoted.get(uids[i], 0) + 1 for i in held}
            demoted.clear()
            demoted.update(
                (
                    (uid, holds.get(uid, 0))
                    for (uid, action) in actions.items()
                    if action == "queue"
                )
            )
        return actions

    def admit(rows):
        return _admit(rows, atomic_cohort=False)

    def admit_atomic(rows):
        return _admit(rows, atomic_cohort=True)

    def atomic_at_depth(depth):
        depth = int(depth)
        configured_depth = max_draft() if callable(max_draft) else max_draft
        stage_override = "lower_k" if depth < configured_depth else None

        def bounded(rows):
            return _admit(
                rows,
                atomic_cohort=True,
                max_draft_override=depth,
                observer_stage_override=stage_override,
            )

        bounded.preview = lambda rows: preview(
            rows, atomic_cohort=True, max_draft_override=depth
        )
        bounded.ceiling = lambda rows: preview(
            rows,
            ceiling=True,
            atomic_cohort=True,
            max_draft_override=depth,
        )
        bounded.reclaim = reclaim_memory
        bounded.cache_projection_bytes = controller.cache_estimator

        def defer_optional_bounded(rows):
            if observer is not None:
                observer(
                    replace(
                        preview(
                            rows,
                            atomic_cohort=True,
                            max_draft_override=depth,
                        ),
                        optional_reclaim_wait=True,
                    )
                )

        bounded.defer_optional = defer_optional_bounded
        bounded.atomic_cohort = True
        bounded.fixed_depth = depth
        return bounded

    def at_depth(depth):
        depth = int(depth)

        def bounded(rows):
            return _admit(
                rows, max_draft_override=depth, hysteresis=ordinary_demoted
            )

        bounded.preview = lambda rows: preview(
            rows, max_draft_override=depth
        )
        bounded.ceiling = lambda rows: preview(
            rows, ceiling=True, max_draft_override=depth
        )
        bounded.reclaim = reclaim_memory
        bounded.cache_projection_bytes = controller.cache_estimator
        bounded.atomic_cohort = False
        bounded.fixed_depth = depth
        return bounded

    admit.preview = preview
    admit.ceiling = lambda rows: preview(rows, ceiling=True)
    admit.atomic = admit_atomic
    admit.atomic.atomic_cohort = True
    admit.at_depth = at_depth
    admit.atomic_at_depth = atomic_at_depth
    admit.atomic.atomic_at_depth = atomic_at_depth
    admit.atomic.preview = lambda rows: preview(rows, atomic_cohort=True)
    admit.atomic.ceiling = lambda rows: preview(
        rows, ceiling=True, atomic_cohort=True
    )
    admit.reclaim = reclaim_memory
    admit.atomic.reclaim = reclaim_memory
    # Expose the adapter-qualified cache geometry to the sliced-prefill owner.
    # Keeping this in bytes avoids repeated GiB rounding while it binds one
    # final-context reservation for the lifetime of a queued continuation.
    admit.cache_projection_bytes = controller.cache_estimator
    admit.atomic.cache_projection_bytes = controller.cache_estimator
    def defer_optional(rows):
        if observer is not None:
            observer(replace(preview(rows), optional_reclaim_wait=True))
    def defer_optional_atomic(rows):
        if observer is not None:
            observer(
                replace(
                    preview(rows, atomic_cohort=True), optional_reclaim_wait=True
                )
            )
    admit.defer_optional = defer_optional
    admit.atomic.defer_optional = defer_optional_atomic
    return admit
