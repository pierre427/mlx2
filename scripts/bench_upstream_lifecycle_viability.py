#!/usr/bin/env python3
"""CPU simulations for upstream speculative/state/cache lifecycle proposals.

The harness intentionally changes no mlx2 production path.  It exercises four
bug shapes as deterministic host data:

* mlx-serve #729: never open an eager next-round draft at budget or pending EOS;
* Strata #652: publish only outputs actually handed to the client;
* TensorRT-LLM #19846: suspended recurrent state follows request ownership,
  including acceptance recorded after suspension;
* TensorRT-LLM #19670: SWA endpoint priority under actual bounded pressure.

The JSON distinguishes a demonstrated invariant from an optimization that mlx2
could select.  In particular, mlx2 has no eager cross-round pre-draft chain and
APCv2 evicts whole revision-bound checkpoints rather than independent SWA pages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path


def predraft_open_allowed(completion_tokens: int, max_tokens: int, pending_is_eos: bool) -> bool:
    """Pure guard from mlx-serve #729, retained only as a test oracle."""
    return int(completion_tokens) < int(max_tokens) and not bool(pending_is_eos)


def committed_prefix(prefix, verify_outputs, *, accepted: int, emitted: int):
    """State visible to the next request after a speculative terminal window."""
    outputs = tuple(verify_outputs)
    if not 0 <= emitted <= accepted + 1 <= len(outputs):
        raise ValueError("invalid speculative-window counts")
    return tuple(prefix) + outputs[:emitted]


def overcommitted_prefix(prefix, verify_outputs, *, accepted: int):
    """The Strata #652 counterexample: commits accepted+bonus regardless of delivery."""
    return tuple(prefix) + tuple(verify_outputs)[: accepted + 1]


def _digest(value) -> str:
    return hashlib.sha256(repr(value).encode()).hexdigest()


class RequestOwnedStatePool:
    """Tiny slot-owner simulation of the TensorRT-LLM #19846 repair contract."""

    def __init__(self, slots: int):
        self.slots = [None] * slots
        self.owners: dict[str, int] = {}
        self.suspended: dict[str, object] = {}
        self.late_acceptance: dict[str, int] = {}

    def attach(self, request: str, slot: int, *, context: bool = False):
        if not 0 <= slot < len(self.slots):
            raise ValueError("slot out of range")
        if context:
            self.suspended.pop(request, None)
            self.late_acceptance.pop(request, None)
            state = None
        else:
            state = self.suspended.pop(request, None)
            if state is not None and request in self.late_acceptance:
                state = {**state, "accepted": self.late_acceptance.pop(request)}
        self.owners[request] = slot
        if state is not None:
            self.slots[slot] = state

    def write(self, request: str, state):
        self.slots[self.owners[request]] = state

    def suspend(self, request: str):
        slot = self.owners.pop(request)
        state = self.slots[slot]
        self.suspended[request] = None if state is None else dict(state)

    def record_late_acceptance(self, request: str, accepted: int):
        if request in self.suspended:
            self.late_acceptance[request] = max(0, int(accepted))
        else:
            state = dict(self.slots[self.owners[request]])
            state["accepted"] = max(0, int(accepted))
            self.write(request, state)

    def read(self, request: str):
        return self.slots[self.owners[request]]


@dataclass(frozen=True)
class Page:
    name: str
    lifecycle: str
    token_start: int
    token_stop: int
    last_access: int


def swa_endpoint_priority(
    page: Page, *, reusable_prompt_tokens: int, window_tokens: int, rewind_tokens: int
) -> int:
    """Priority formula described by TensorRT-LLM #19670."""
    if min(reusable_prompt_tokens, window_tokens, rewind_tokens) < 0:
        raise ValueError("token counts must be nonnegative")
    if page.lifecycle == "sink":
        return 70
    if page.lifecycle != "swa":
        return 35
    endpoint_start = max(0, reusable_prompt_tokens - window_tokens - rewind_tokens)
    return 70 if page.token_stop > endpoint_start else 0


def evict_under_pressure(
    pages, *, keep: int, priority=None
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Evict low-priority/LRU pages until ``keep`` remain."""
    pages = list(pages)
    if not 0 <= keep <= len(pages):
        raise ValueError("invalid pressure target")
    score = priority or (lambda _page: 0)
    victims = sorted(pages, key=lambda page: (score(page), page.last_access, page.name))
    evicted = tuple(page.name for page in victims[: len(pages) - keep])
    retained = tuple(page.name for page in pages if page.name not in evicted)
    return evicted, retained


def run_simulations(*, iterations: int = 20_000) -> dict:
    verify = (101, 102, 103, 104, 105)
    prefix = tuple(range(32))
    accepted, emitted = 4, 2
    exact = committed_prefix(prefix, verify, accepted=accepted, emitted=emitted)
    old = overcommitted_prefix(prefix, verify, accepted=accepted)

    pool = RequestOwnedStatePool(2)
    pool.attach("first", 0)
    original = {"conv": (1, 2), "recurrent": (3, 4), "accepted": 1}
    pool.write("first", original)
    pool.suspend("first")
    pool.attach("second", 0, context=True)
    pool.write("second", {"conv": (9, 9), "recurrent": (8, 8), "accepted": 3})
    pool.record_late_acceptance("first", 2)
    pool.attach("first", 1)
    restored = pool.read("first")

    pages = (
        Page("sink", "sink", 0, 16, 1),
        Page("old-swa", "swa", 1024, 1088, 9),
        Page("middle-swa", "swa", 7000, 7064, 8),
        Page("endpoint-swa", "swa", 8128, 8192, 2),
        Page("global", "full", 4096, 4160, 3),
    )
    config = {"reusable_prompt_tokens": 8192, "window_tokens": 1024, "rewind_tokens": 256}
    priority = lambda page: swa_endpoint_priority(page, **config)
    lru_evicted, lru_retained = evict_under_pressure(pages, keep=3)
    policy_evicted, policy_retained = evict_under_pressure(pages, keep=3, priority=priority)

    timings = {}
    for name, operation in {
        "commit_only_emitted": lambda: committed_prefix(prefix, verify, accepted=accepted, emitted=emitted),
        "predraft_guard": lambda: predraft_open_allowed(63, 64, False),
        "endpoint_priority": lambda: [priority(page) for page in pages],
    }.items():
        samples = []
        for _ in range(5):
            start = time.perf_counter_ns()
            for _iteration in range(iterations):
                operation()
            samples.append((time.perf_counter_ns() - start) / iterations)
        timings[name + "_median_ns"] = statistics.median(samples)

    return {
        "schema": "mlx2.upstream-lifecycle-viability.v1",
        "gpu_used": False,
        "production_behavior_changed": False,
        "commit_only_emitted": {
            "exact_digest": _digest(exact),
            "overcommitted_digest": _digest(old),
            "overshoot_tokens": len(old) - len(exact),
            "mlx2_overlap": "implemented by emitted_counts and terminal commit vectors",
            "source": "Niko1221/Strata#652@cf14881f316c889d600f954a5ac6993c8981c28a",
        },
        "terminal_predraft": {
            "matrix": {
                "under_budget": predraft_open_allowed(63, 64, False),
                "at_budget": predraft_open_allowed(64, 64, False),
                "over_budget": predraft_open_allowed(65, 64, False),
                "pending_eos": predraft_open_allowed(63, 64, True),
            },
            "mlx2_overlap": "not applicable: no eager cross-round pre-draft chain",
            "source": "ddalcu/mlx-serve#729@c1dc5ca3e1e608507ac0371cc2b073ecd7dfb910",
        },
        "request_owned_recurrent_state": {
            "expected_digest": _digest({**original, "accepted": 2}),
            "restored_digest": _digest(restored),
            "exact": restored == {**original, "accepted": 2},
            "other_request_unchanged": pool.read("second")["accepted"] == 3,
            "mlx2_overlap": "committed recovery is route/revision/boundary owned, not raw slot owned",
            "source": "NVIDIA/TensorRT-LLM#19846@10fd5cf9f52d06c37934722bef98a844a92b94ed",
        },
        "swa_endpoint_priority": {
            "priorities": {page.name: priority(page) for page in pages},
            "lru_only": {"evicted": lru_evicted, "retained": lru_retained},
            "priority_then_lru": {"evicted": policy_evicted, "retained": policy_retained},
            "endpoint_retained_under_pressure": "endpoint-swa" in policy_retained,
            "mlx2_overlap": "not directly applicable: APCv2 retains whole checkpoints, not SWA pages",
            "source": "NVIDIA/TensorRT-LLM#19670@7b7305c5b98fc96efdb0fe01b82064749340f2b8",
            "related_v2_review": "#19754 requested a real pressure/eviction test beyond callback math",
        },
        "host_operation_timings": timings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    report = run_simulations(iterations=args.iterations)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
