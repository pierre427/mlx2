#!/usr/bin/env python3
"""Run the mlx2 server with default-off self-MTP proposal tracing.

The wrapper records host-visible proposal outcomes after the runtime's existing
verification boundary.  It does not synchronize Metal, alter route selection,
or enable heterogeneous draft depths.  Set ``MLX2_ASPIRE_TRACE_PATH`` to the
JSON output path and pass ordinary ``mlx2.server`` arguments unchanged.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from mlx2 import server
from mlx2.runtime import hybrid_speculative


def _context_lengths(batch: Any) -> tuple[int, ...]:
    row_caches = getattr(batch, "row_caches", None)
    if row_caches is not None:
        return tuple(
            hybrid_speculative._self_mtp_group_offset(pair.target)
            for pair in row_caches
        )
    offset = hybrid_speculative._self_mtp_group_offset(batch.caches.target)
    return (offset,) * len(batch.lanes)


def _proposal_rows(
    proposal: Any,
    *,
    context_lengths: tuple[int, ...],
    cycle_ns: int,
    round_indices: dict[int, int],
) -> list[dict[str, int]]:
    if len(context_lengths) != len(proposal.lane_uids):
        raise RuntimeError("trace context/lane width mismatch")
    rows = []
    for uid, context, depth, accepted in zip(
        proposal.lane_uids,
        context_lengths,
        proposal.draft_depths,
        proposal.accepted_lengths,
    ):
        round_index = round_indices.get(uid, 0)
        round_indices[uid] = round_index + 1
        rows.append(
            {
                "request_id": str(uid),
                "round_index": round_index,
                "context_length": int(context),
                "batch_size": len(proposal.lane_uids),
                "draft_depth": int(depth),
                "accepted_prefix": int(accepted),
                "cycle_ns": int(cycle_ns),
            }
        )
    return rows


def main() -> None:
    output = os.environ.get("MLX2_ASPIRE_TRACE_PATH")
    if not output:
        raise SystemExit("MLX2_ASPIRE_TRACE_PATH is required")
    output_path = Path(output)
    records: list[dict[str, int]] = []
    round_indices: dict[int, int] = {}
    original = hybrid_speculative.propose_batched_self_mtp

    def traced(model, batch):
        contexts = _context_lengths(batch)
        started = time.perf_counter_ns()
        proposal = original(model, batch)
        cycle_ns = time.perf_counter_ns() - started
        records.extend(
            _proposal_rows(
                proposal,
                context_lengths=contexts,
                cycle_ns=cycle_ns,
                round_indices=round_indices,
            )
        )
        return proposal

    hybrid_speculative.propose_batched_self_mtp = traced
    started_at_ns = time.time_ns()
    try:
        server.main()
    finally:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "schema": "mlx2.async-mtp-observed-trace.v1",
                    "scope": "post-verification host-visible proposal outcomes",
                    "started_at_ns": started_at_ns,
                    "finished_at_ns": time.time_ns(),
                    "argv": sys.argv[1:],
                    "records": records,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
