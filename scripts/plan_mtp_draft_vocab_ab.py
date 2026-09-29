#!/usr/bin/env python3
"""Emit the qualification matrix for a future reduced-MTP-vocabulary GPU A/B.

This planner deliberately cannot start a model or contact a server.  It makes
the deferred GPU work concrete while keeping preparation safe to run on CPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--mlx2-revision",
        required=True,
        help="exact committed mlx2 revision to bind into the deferred run plan",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()
    if args.reps < 3:
        raise SystemExit("qualification requires at least three repetitions")
    root = args.model_path.expanduser().resolve()
    manifest = json.loads((root / "mtp_draft_vocab.json").read_text())
    cells = []
    for width in (1, 4, 8):
        for prompt in ("code-short", "agent-cached", "cold-16k", "cold-64k"):
            for arm, enabled in (("full-vocab", False), ("reduced-vocab", True)):
                cells.append(
                    {
                        "width": width,
                        "prompt": prompt,
                        "arm": arm,
                        "repetitions": args.reps,
                        "execution_policy": {"mtp_draft_vocab": enabled},
                    }
                )
    plan = {
        "schema": "mlx2.mtp-draft-vocab-ab-plan.v1",
        "status": "planned-not-executed",
        "mlx2_revision": args.mlx2_revision,
        "artifact": {
            "path": str(root),
            "vocab_size": manifest["vocab_size"],
            "token_count": manifest["token_count"],
            "ids_sha256": manifest["ids_sha256"],
        },
        "invariants": [
            "same source revision and artifact binding for both arms",
            "ordinary target and verify logits remain full-vocabulary",
            "capture route receipt plus mtp_draft_vocab proposal/bypass counters",
            "compare output hashes for greedy; compare fixed-seed quality suite for sampled",
            "report TTFT, decode tok/s, aggregate tok/s, tokens/verify-step, and peak memory",
        ],
        "cells": cells,
    }
    rendered = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
