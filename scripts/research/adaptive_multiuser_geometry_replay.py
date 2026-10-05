#!/usr/bin/env python3
"""Short multi-user/multi-turn scheduler geometry replay.

This is a deterministic host-side prototype, not a throughput benchmark.  It
feeds recurring users, growing histories, staggered arrivals and speculative
decode lanes through the same capability-aware round planner used by tests.
The receipt compares charged token rows, avoided padding and completion rounds;
actual model wall time belongs in the follow-on HTTP gate.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from mlx2.runtime.batch_geometry import (
    GeometryCapabilities,
    GeometryLane,
    GeometryPolicy,
    plan_batch_geometry,
)


@dataclass(frozen=True)
class Request:
    uid: int
    user: str
    turn: int
    arrival_round: int
    prompt_tokens: int
    output_tokens: int
    draft_tokens: int


# Eight requests, four recurring users, two turns each.  The second turn has a
# larger history.  Sizes span support-chat, retrieval and long-domain-context
# traffic while keeping the replay and the later cap4 HTTP gate short.
TRACE = (
    Request(1, "ada", 1, 0, 1152, 8, 2),
    Request(2, "ben", 1, 0, 4096, 8, 0),
    Request(3, "cy", 1, 0, 6982, 8, 2),
    Request(4, "dee", 1, 1, 1536, 8, 1),
    Request(5, "ada", 2, 3, 2176, 8, 2),
    Request(6, "ben", 2, 4, 5120, 8, 2),
    Request(7, "cy", 2, 6, 7680, 8, 0),
    Request(8, "dee", 2, 7, 2944, 8, 1),
)


def arm(name: str):
    base = {
        "max_lanes": 4,
        "token_budget": 8192,
        "prefill_chunk": 2048,
        "row_tile": 64,
    }
    if name == "always_padded":
        return GeometryCapabilities(**base), GeometryPolicy(
            packed_padding_fraction=.99, bucket_padding_fraction=.99)
    if name == "always_packed":
        return GeometryCapabilities(
            **base, packed_prefill=True, packed_verify=True
        ), GeometryPolicy(packed_padding_fraction=0, bucket_padding_fraction=.99)
    if name == "adaptive":
        return GeometryCapabilities(
            **base, packed_prefill=True, packed_verify=True, mixed_forward=True
        ), GeometryPolicy()
    raise ValueError("unknown arm")


def replay(name: str, trace=TRACE):
    caps, policy = arm(name)
    queued = {request.uid: request for request in trace}
    prefill = {}
    decode = {}
    finished = {}
    first_token = {}
    modes = Counter()
    totals = Counter()
    decisions = []
    round_id = 0
    while queued or prefill or decode:
        for uid, request in list(queued.items()):
            if request.arrival_round <= round_id:
                prefill[uid] = request.prompt_tokens
                del queued[uid]
        lanes = [
            GeometryLane(uid, "decode", 1, trace[uid - 1].draft_tokens)
            for uid in sorted(decode)
        ] + [
            GeometryLane(uid, "prefill", remaining)
            for uid, remaining in sorted(prefill.items())
        ]
        if not lanes:
            round_id += 1
            continue
        plan = plan_batch_geometry(lanes, caps, policy)
        modes[plan.geometry] += 1
        totals["real_rows"] += plan.real_rows
        totals["charged_rows"] += plan.charged_rows
        totals["decision_padding_rows"] += plan.padding_rows
        totals["verification_rows"] += plan.verification_rows
        decisions.append({"round": round_id, **plan.receipt()})

        for uid in plan.prefill_uids:
            prefill[uid] -= min(prefill[uid], caps.prefill_chunk)
            if prefill[uid] == 0:
                del prefill[uid]
                decode[uid] = trace[uid - 1].output_tokens
        for uid in plan.decode_uids:
            request = trace[uid - 1]
            # Deterministic optimistic K acceptance for the scheduler shape:
            # accepted drafts plus one target token, capped by the response.
            emitted = min(decode[uid], 1 + request.draft_tokens)
            if uid not in first_token:
                first_token[uid] = round_id
            decode[uid] -= emitted
            totals["output_tokens"] += emitted
            totals["accepted_draft_tokens"] += max(0, emitted - 1)
            if decode[uid] == 0:
                del decode[uid]
                finished[uid] = round_id
        round_id += 1
        if round_id > 1000:
            raise RuntimeError("replay did not converge")
    return {
        "arm": name,
        "rounds": round_id,
        "geometry_rounds": dict(sorted(modes.items())),
        **dict(totals),
        "first_token_round": first_token,
        "completion_round": finished,
        "decisions": decisions,
    }


def build_receipt():
    arms = {name: replay(name) for name in (
        "always_padded", "always_packed", "adaptive")}
    padded = arms["always_padded"]
    adaptive = arms["adaptive"]
    return {
        "schema": "mlx2.adaptive-multiuser-geometry-replay.v1",
        "status": "passed",
        "implemented": True,
        "qualified": False,
        "selected_by_default": False,
        "observed_used": True,
        "performance_claim": False,
        "scope": (
            "deterministic host scheduler replay; charged rows are a geometry "
            "proxy and are not model wall time"
        ),
        "traffic": {
            "users": len({request.user for request in TRACE}),
            "requests": len(TRACE),
            "turns_per_user": 2,
            "speculative_requests": sum(request.draft_tokens > 0 for request in TRACE),
            "requests_detail": [asdict(request) for request in TRACE],
        },
        "arms": arms,
        "adaptive_vs_padded": {
            "charged_rows_ratio": (
                adaptive["charged_rows"] / padded["charged_rows"]
            ),
            "charged_rows_avoided": (
                padded["charged_rows"] - adaptive["charged_rows"]
            ),
            "rounds_ratio": adaptive["rounds"] / padded["rounds"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = build_receipt()
    payload = json.dumps(receipt, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
