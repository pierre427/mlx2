#!/usr/bin/env python3
"""Write a CPU-only Gemma 4 exact-prefix-cascade qualification plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.adapters.gemma4 import inspect_gemma4_artifact

    path = path.expanduser().resolve()
    record = inspect_gemma4_artifact(path)
    result = {
        "status": "static_metadata_verified",
        "path": str(path),
        "config_sha256": sha256(path / "config.json"),
        "variant": record["variant"],
        "precision": record["precision"],
        "max_context": record["max_context"],
        "full_attention_layers": record["full_attention_layers"],
        "sliding_attention_layers": record["sliding_attention_layers"],
        "source_revision": record.get("source_revision"),
    }
    provenance = path / "source-and-quantization.json"
    if provenance.is_file():
        result["source_and_quantization_sha256"] = sha256(provenance)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit")
    args = parser.parse_args(argv)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if args.source_commit is not None and args.source_commit != head:
        parser.error(
            f"source commit mismatch: expected {args.source_commit}, got {head}"
        )
    artifacts = [artifact(path) for path in args.model]
    variants = {item["variant"] for item in artifacts}
    precisions = {item["precision"] for item in artifacts}
    report = {
        "schema": "mlx2.gemma4-strategy-transfer-plan.v1",
        "status": "staged_cpu_only_gpu_authority_required",
        "source_commit": head,
        "artifacts": artifacts,
        "artifact_matrix_complete": variants == {"26b-a4b", "31b"}
        and precisions == {"bf16", "mlx-affine-8bit"},
        "implemented": {
            "longest_first_planner": True,
            "invalid_sibling_pruning": True,
            "adapter_state_contract": True,
            "exact_cache_geometry_gate": True,
            "serving_route": False,
        },
        "candidate": {
            "verification": "exact_target_draw_then_prefix_match",
            "ordering": "longest_first",
            "accepted_prefix_state": "exact_apcv2_restore_or_canonical_replay",
            "shared_prefix_authority": "apcv2",
            "recompute_common_tokens": False,
            "transactional_multirow_state_reuse": False,
            "cascade_state_apcv2_publication": False,
        },
        "required_native_cells": [
            {"context": 128, "purpose": "controlled ordering and pruning"},
            {"context": 1024, "purpose": "sliding-window boundary"},
            {"context": 2048, "purpose": "first wrapped-window restore"},
            {"context": 16384, "purpose": "mixed full/sliding exactness"},
            {"context": 32768, "purpose": "long-context memory gate"},
        ],
        "arms": [
            "ordinary_s1",
            "parallel_complete_paths",
            "longest_first_canonical_replay",
            "longest_first_exact_apcv2_prefix_reuse",
        ],
        "hard_gates": [
            "run every arm separately for dense and A4B, BF16 and 8-bit",
            "ordinary greedy and sampled transcript parity",
            "next-step logits and every full/sliding cache plane match ordinary S1",
            "common tokens are not forwarded after an exact APCv2 restore",
            "wrong cache type, offset, window or media fingerprint fails closed",
            "cascade-private state is never published to APCv2",
            "shared waiter and both GPU locks are recorded",
            "clean exact source revision, zero swapout growth and thermal state retained",
        ],
        "implemented_state": True,
        "qualified": False,
        "selected": False,
        "observed_used": False,
        "gpu_executed": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "artifacts": len(artifacts),
                "matrix_complete": report["artifact_matrix_complete"],
                "out": str(args.out),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
