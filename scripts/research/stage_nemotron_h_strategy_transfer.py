"""Write a CPU/static Nemotron-H exact-prefix qualification plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS = (
    Path.home()
    / "mlx-models"
    / "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-mlx-8Bit",
    Path.home() / "mlx-models" / "Nemotron-3-Super-120B-A12B-5bit-MTP",
    Path.home() / "mlx-models" / "TwoTower-30B-stage",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_artifact(path: Path) -> dict:
    path = path.expanduser().resolve()
    config_path = path / "config.json"
    if not config_path.is_file():
        return {"path": str(path), "status": "absent"}
    config = json.loads(config_path.read_text())
    model_type = config.get("model_type")
    result = {
        "path": str(path),
        "status": "static_metadata_verified",
        "model_type": model_type,
        "architecture": config.get("architectures", [None])[0],
        "config_sha256": sha256(config_path),
        "layers": int(config["num_hidden_layers"]),
        "hidden_size": int(config["hidden_size"]),
        "max_position_embeddings": int(config["max_position_embeddings"]),
    }
    index = path / "model.safetensors.index.json"
    if index.is_file():
        result["index_sha256"] = sha256(index)
        result["tensor_count"] = len(json.loads(index.read_text())["weight_map"])
    if model_type == "nemotron_h":
        result.update(
            strategy_boundary="eligible_target_cache_geometry",
            mtp_boundary="excluded_from_shared_prefix_reuse",
        )
    else:
        result.update(
            strategy_boundary="excluded_architecture",
            exclusion="nemotron_twotower has no mlx2 adapter or exact target-state contract",
        )
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", type=Path)
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
    artifacts = tuple(args.artifact or DEFAULT_ARTIFACTS)
    report = {
        "schema": "mlx2.nemotron-h-strategy-transfer-plan.v1",
        "status": "staged_cpu_only_native_authority_required",
        "source_commit": head,
        "artifacts": [inspect_artifact(path) for path in artifacts],
        "implemented": {
            "adapter_prefill_step": 2048,
            "generic_autoscale_when_adapter_declines": True,
            "longest_first_planner": True,
            "invalid_sibling_pruning": True,
            "b1_exact_shared_prefix_primitive": True,
            "serving_route": False,
        },
        "candidate": {
            "verification": "ordinary_tokenwise_b1_target",
            "accepted_prefix_state": "committed_hybrid_transaction",
            "shared_prefix_reuse": "request_private_target_cache_clone",
            "common_tokens_recomputed": False,
            "mtp_cache_reuse": False,
            "twotower_reuse": False,
            "apcv2_publication": False,
        },
        "required_native_cells": [128, 2048, 8192, 32768, 131072],
        "hard_gates": [
            "clean committed source revision and exact artifact identity",
            "ordinary transcript, next-logit, attention-KV and every recurrent slot parity",
            "full, partial and zero proposal acceptance plus sibling-pruning controls",
            "cold and APCv2-warm ordinary reference retained",
            "waiter and both GPU locks owned; no overlapping GPU window",
            "zero swapout growth and thermally paired arm order for performance claims",
        ],
        "qualified": False,
        "selected": False,
        "observed_used": False,
        "gpu_executed": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
