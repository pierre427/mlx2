#!/usr/bin/env python3
"""Bounded two-rank MLX ring transport and ownership diagnostic.

Launch only through ``mlx.launch -n 2 --backend ring`` under the repository's
GPU ownership wrapper.  Fault modes keep every rank in the same collective
sequence, so a negative test cannot intentionally strand a peer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


def _atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(mode: str, ownership: dict) -> dict:
    import mlx.core as mx

    group = mx.distributed.init(strict=True, backend="ring")
    rank, size = group.rank(), group.size()
    if size != 2:
        raise RuntimeError("distributed serving probe requires exactly two ranks")

    cluster = hashlib.sha256(b"mlx2-two-rank-fixture-v1").digest()
    local_identity = bytearray(cluster)
    if mode == "identity_mismatch" and rank == 1:
        local_identity[0] ^= 1
    identities = mx.distributed.all_gather(mx.array(list(local_identity), dtype=mx.uint8), group=group)
    mx.eval(identities)
    identity_match = bytes(identities[:32].tolist()) == bytes(identities[32:].tolist())

    # Both ranks execute the same fixed projection. Divergence is injected
    # only after the collective input has been formed.
    x = mx.array([[1.0, 2.0, 3.0, 4.0]], dtype=mx.float32)
    weight = mx.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]], dtype=mx.float32)
    local = x @ weight
    if mode == "output_divergence" and rank == 1:
        local = local + mx.array([[0.0, 1.0]], dtype=mx.float32)
    outputs = mx.distributed.all_gather(local, group=group)
    mx.eval(outputs)
    output_match = outputs[0].tolist() == outputs[1].tolist()

    cancel = 1 if mode == "cancel" and rank == 1 else 0
    cancel_count = mx.distributed.all_sum(mx.array([cancel], dtype=mx.int32), group=group)
    mx.eval(cancel_count)
    cancellation_observed = int(cancel_count.item()) > 0

    detected = {
        "normal": identity_match and output_match and not cancellation_observed,
        "identity_mismatch": not identity_match,
        "output_divergence": identity_match and not output_match,
        "cancel": identity_match and output_match and cancellation_observed,
    }[mode]
    publishable = identity_match and output_match and not cancellation_observed
    return {
        "schema": "mlx2.distributed-ring-diagnostic.v1",
        "rank": rank,
        "world_size": size,
        "backend": "ring",
        "mode": mode,
        "ownership": ownership,
        "identity_match": identity_match,
        "output_match": output_match,
        "cancellation_observed": cancellation_observed,
        "state_publishable": publishable,
        "expected_outcome_detected": detected,
        "qualified": False,
        "performance_qualified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("normal", "identity_mismatch", "output_divergence", "cancel"), required=True)
    parser.add_argument("--cpg-task", required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    parser.add_argument("--cpg-generation", type=int, required=True)
    args = parser.parse_args()

    from qualification_gpu_ownership import require_qualification_lease

    ownership = require_qualification_lease(
        task_id=args.cpg_task,
        cpg_owner_lock=args.cpg_owner_lock,
        generation=args.cpg_generation,
    )
    result = run(args.mode, ownership)
    output = args.output_dir / args.mode / f"rank-{result['rank']}.json"
    if output.exists():
        parser.error(f"evidence already exists: {output}")
    _atomic(output, result)
    print(json.dumps({"rank": result["rank"], "mode": args.mode,
                      "passed": result["expected_outcome_detected"]}), flush=True)
    return 0 if result["expected_outcome_detected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
