"""Tenant-owned, bounded-cardinality multi-drafter runtime telemetry."""

from __future__ import annotations

import math
import threading
from collections import Counter, OrderedDict

DRAFTER_LABELS = frozenset(
    {
        "prompt_lookup",
        "native_mtp",
        "xpress",
        "lilicorr",
        "dflash2",
        "dflash2_fold",
        "dspark",
        "dpara",
        "other",
    }
)


def _drafter(value):
    value = str(value)
    return value if value in DRAFTER_LABELS else "other"


def _count(value):
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


class DrafterRuntimeMetrics:
    """Aggregate terminal receipts without retaining request/session identity."""

    def __init__(self, *, max_tenants=256):
        if type(max_tenants) is not int or max_tenants < 1:
            raise ValueError("max_tenants must be positive")
        self.max_tenants = max_tenants
        self._tenants = OrderedDict()
        # Operator counters must remain monotonic even when the bounded
        # per-tenant LRU evicts an inactive tenant.
        self._global = self._empty()
        self._lock = threading.Lock()
        self.evictions = 0

    @staticmethod
    def _empty():
        return {
            "events": Counter(),
            "depths": Counter(),
            "tokens": Counter(),
            "scores": Counter(),
            "combinations": Counter(),
            "budget": Counter(),
            "states": {},
        }

    def _state(self, tenant_id):
        tenant = str(tenant_id or "default")
        state = self._tenants.get(tenant)
        if state is None:
            state = self._empty()
            self._tenants[tenant] = state
        self._tenants.move_to_end(tenant)
        while len(self._tenants) > self.max_tenants:
            self._tenants.popitem(last=False)
            self.evictions += 1
        return state

    @staticmethod
    def _observe(state, pool, provider_stats):
        for provider in provider_stats.values():
            if not isinstance(provider, dict):
                continue
            mechanism = _drafter(provider.get("mechanism", "other"))
            for event, field in (
                ("enabled", "enabled_rounds"),
                ("admitted", "admitted_rounds"),
                ("selected", "selected_rounds"),
                ("contributed", "contributed_rounds"),
                ("zero", "zero_rounds"),
                ("partial", "partial_rounds"),
                ("full", "full_rounds"),
                ("parked", "parked_rounds"),
                ("fallback", "fallback_rounds"),
            ):
                state["events"][(mechanism, event)] += _count(provider.get(field, 0))
            state["tokens"][(mechanism, "proposed")] += _count(
                provider.get("proposed_tokens", 0)
            )
            state["tokens"][(mechanism, "accepted")] += _count(
                provider.get("accepted_tokens", 0)
            )
            for kind, field in (
                ("proposal", "proposal_depths"),
                ("accepted", "accepted_depths"),
                ("first_rejected", "first_rejected_positions"),
                ("selected_rank", "selected_ranks"),
            ):
                values = provider.get(field) or {}
                if not isinstance(values, dict):
                    continue
                for raw_depth, raw_count in values.items():
                    try:
                        depth = int(raw_depth)
                        count = int(raw_count)
                    except (TypeError, ValueError):
                        continue
                    bound = 14 if kind == "selected_rank" else 15
                    label = str(depth) if 0 <= depth <= bound else "other"
                    state["depths"][(mechanism, kind, label)] += _count(count)
            score = provider.get("score_sum", 0.0)
            observations = provider.get("score_observations", 0)
            if (
                type(score) in (int, float)
                and math.isfinite(float(score))
                and type(observations) is int
                and observations >= 0
            ):
                state["scores"][(mechanism, "sum")] += float(score)
                state["scores"][(mechanism, "observations")] += observations
        combinations = pool.get("winning_combinations") or {}
        if not isinstance(combinations, dict):
            combinations = {}
        for combination, count in combinations.items():
            contributors = "multiple" if "+" in str(combination) else "one"
            state["combinations"][contributors] += _count(count)
        admission = pool.get("last_admission") or {}
        if not isinstance(admission, dict):
            admission = {}
        admission_totals = pool.get("admission_totals") or admission
        if not isinstance(admission_totals, dict):
            admission_totals = {}
        for field in (
            "lane_slots_used",
            "lane_slots_available",
            "tile_rows_used",
            "tile_rows_available",
        ):
            state["budget"][field] += _count(admission_totals.get(field, 0))
        parked = {
            str(item.get("source_id"))
            for item in admission.get("parked", ())
            if isinstance(item, dict)
        }
        admitted_values = admission.get("admitted") or ()
        admitted = (
            set(admitted_values)
            if isinstance(admitted_values, (list, tuple, set))
            else set()
        )
        priority = {"idle": 0, "parked": 1, "active": 2}
        current = {}
        for source_id, provider in provider_stats.items():
            if not isinstance(provider, dict):
                continue
            mechanism = _drafter(provider.get("mechanism", "other"))
            if source_id in parked:
                candidate = "parked"
            elif source_id in admitted:
                candidate = "active"
            else:
                candidate = "idle"
            previous = current.get(mechanism)
            if previous is None or priority[candidate] > priority[previous]:
                current[mechanism] = candidate
        # Resolve duplicate mechanisms within this receipt by priority, then
        # publish the latest observed state. Historical priority must not make
        # a parked or idle provider look active forever.
        state["states"].update(current)

    def observe_receipt(self, tenant_id, receipt):
        speculation = (receipt or {}).get("speculation") or {}
        pool = speculation.get("continuation_pool") or {}
        provider_stats = pool.get("provider_stats") or {}
        if not isinstance(provider_stats, dict):
            return
        with self._lock:
            tenant = self._state(tenant_id)
            self._observe(self._global, pool, provider_stats)
            self._observe(tenant, pool, provider_stats)

    @staticmethod
    def _merge(states):
        merged = DrafterRuntimeMetrics._empty()
        for state in states:
            for name in (
                "events",
                "depths",
                "tokens",
                "scores",
                "combinations",
                "budget",
            ):
                merged[name].update(state[name])
            merged["states"].update(state["states"])
        return merged

    def prometheus_snapshot(self, tenant_id=None):
        with self._lock:
            if tenant_id is None:
                state = self._merge([self._global])
            else:
                state = self._merge(
                    [self._tenants.get(str(tenant_id or "default"), self._empty())]
                )
            evictions = self.evictions
        return {**state, "evictions": evictions}


__all__ = ["DRAFTER_LABELS", "DrafterRuntimeMetrics"]
