"""Host-only replay of candidate streamed-expert residency policies.

Nothing here is on the serving path.  It consumes an exact v2 access trace and
compares the current independent per-projection LRU with a hybrid design:
every projection retains a minimum floor while unused bytes form one shared
overflow LRU.  The scan-resistant variant observes intermediate prefill calls
without allowing them to displace durable decode residency, then admits the
terminal pass in whole-prompt frequency order, as the live ``terminal_hot``
admission does: counts cover only the request's prefill calls, and ties keep
router order.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field


@dataclass
class ReplayStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    transition_evictions: int = 0
    phase: dict = field(default_factory=dict)
    decode_page_ins: list = field(default_factory=list)

    def note(self, phase: str, hits: int, misses: int) -> None:
        self.hits += hits
        self.misses += misses
        row = self.phase.setdefault(phase, {"hits": 0, "misses": 0, "calls": 0})
        row["hits"] += hits
        row["misses"] += misses
        row["calls"] += 1
        if phase == "decode":
            self.decode_page_ins.append(misses)

    def as_dict(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "evictions": self.evictions,
            "transition_evictions": self.transition_evictions,
            "phase": self.phase,
            "decode_first_8_page_ins": sum(self.decode_page_ins[:8]),
            "decode_first_32_page_ins": sum(self.decode_page_ins[:32]),
        }


def _phase(index: int, *, decode_start: int, terminal_start: int, serving_start: int = 0) -> str:
    if index < serving_start:
        return "load"
    if index >= decode_start:
        return "decode"
    if index >= terminal_start:
        return "terminal_prefill"
    return "intermediate_prefill"


def replay_per_projection_lru(
    sequence: Sequence[tuple[int, Sequence[int]]],
    *,
    capacities: Mapping[int, int],
    decode_start: int,
    terminal_start: int,
    serving_start: int = 0,
) -> dict:
    """Replay the engine's current independent-LRU rule in global call order."""
    resident = {layer: OrderedDict() for layer in capacities}
    stats = ReplayStats()
    for index, (layer, values) in enumerate(sequence):
        wanted = list(dict.fromkeys(int(value) for value in values))
        cache = resident[layer]
        hits = 0
        missing = []
        for value in wanted:
            if value in cache:
                hits += 1
                cache.move_to_end(value)
            else:
                missing.append(value)
        keep = set(wanted)
        while len(cache) + len(missing) > capacities[layer]:
            victim = next((value for value in cache if value not in keep), None)
            if victim is None:
                break
            del cache[victim]
            stats.evictions += 1
        for value in missing:
            cache[value] = None
        while len(cache) > capacities[layer]:
            cache.popitem(last=False)
            stats.evictions += 1
        stats.note(_phase(index, decode_start=decode_start, terminal_start=terminal_start, serving_start=serving_start), hits, len(missing))
    return {"policy": "per_projection_lru", **stats.as_dict()}


def replay_hybrid(
    sequence: Sequence[tuple[int, Sequence[int]]],
    *,
    row_bytes: Mapping[int, int],
    total_budget_bytes: int,
    floor_entries: int,
    decode_start: int,
    terminal_start: int,
    serving_start: int = 0,
    scan_resistant: bool = False,
    budget_schedule: Iterable[tuple[int, int]] = (),
) -> dict:
    """Replay per-projection floors plus one byte-budgeted shared overflow."""
    layers = sorted(row_bytes)
    floor = {layer: OrderedDict() for layer in layers}
    shared: OrderedDict[tuple[int, int], None] = OrderedDict()
    shared_bytes = 0
    minimum_bytes = sum(max(0, floor_entries) * int(row_bytes[layer]) for layer in layers)
    if minimum_bytes > total_budget_bytes:
        raise ValueError("hybrid floor exceeds total budget")
    budget = int(total_budget_bytes)
    schedule = {int(index): int(value) for index, value in budget_schedule}
    frequencies = {layer: Counter() for layer in layers}
    stats = ReplayStats()

    def trim(*, transition: bool = False, keep=frozenset()):
        nonlocal shared_bytes
        allowance = max(0, budget - minimum_bytes)
        while shared_bytes > allowance:
            victim = next((key for key in shared if key not in keep), None)
            if victim is None:
                break
            del shared[victim]
            shared_bytes -= int(row_bytes[victim[0]])
            stats.evictions += 1
            stats.transition_evictions += int(transition)

    for index, (layer, values) in enumerate(sequence):
        if index in schedule:
            budget = schedule[index]
            if budget < minimum_bytes:
                raise ValueError("scheduled budget falls below hybrid floor")
            trim(transition=True)
        phase = _phase(index, decode_start=decode_start, terminal_start=terminal_start, serving_start=serving_start)
        wanted = list(dict.fromkeys(int(value) for value in values))
        if phase in ("intermediate_prefill", "terminal_prefill"):
            for value in wanted:
                frequencies[layer][value] += 1
        hits = 0
        missing = []
        for value in wanted:
            key = (layer, value)
            if value in floor[layer] or key in shared:
                hits += 1
            else:
                missing.append(value)
        stats.note(phase, hits, len(missing))
        if scan_resistant and phase == "intermediate_prefill":
            continue
        # Terminal admission uses whole-prompt frequency while retaining only
        # rows this terminal call actually materialized. Admit cold-to-hot so
        # the hottest rows finish at the MRU end of both tiers. The stable
        # sort keeps router order on ties, as ExpertLRU does.
        admission_order = wanted
        if scan_resistant and phase == "terminal_prefill":
            admission_order = sorted(
                wanted, key=lambda value: frequencies[layer][value]
            )
        # Every admitted access refreshes the projection's protected floor.
        # A shared hit is promoted and the local LRU is demoted, so the floor
        # is a dynamic minimum residency rather than a pinned first-N set.
        for value in admission_order:
            key = (layer, value)
            if value in floor[layer]:
                floor[layer].move_to_end(value)
                continue
            if key in shared:
                del shared[key]
                shared_bytes -= int(row_bytes[layer])
            if floor_entries:
                if len(floor[layer]) >= floor_entries:
                    victim, _ = floor[layer].popitem(last=False)
                    shared[(layer, victim)] = None
                    shared_bytes += int(row_bytes[layer])
                floor[layer][value] = None
            else:
                shared[key] = None
                shared_bytes += int(row_bytes[layer])
        keep = {(layer, value) for value in wanted}
        trim(keep=keep)
        # An oversized call may temporarily exceed the shared allowance. Trim
        # after its whole working set has been served, matching the live LRU's
        # transient-overflow rule.
        trim()
    name = "hybrid_scan_resistant" if scan_resistant else "hybrid"
    return {
        "policy": name,
        "budget_bytes": total_budget_bytes,
        "floor_entries": floor_entries,
        "minimum_floor_bytes": minimum_bytes,
        "final_shared_bytes": shared_bytes,
        "final_budget_bytes": budget,
        **stats.as_dict(),
    }
