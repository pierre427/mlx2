"""Write a CPU-only Qwen4 exact-prefix-cascade qualification plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path | None) -> dict:
    if path is None:
        return {"status": "unresolved", "reason": "no Qwen4 artifact supplied"}
    path = path.expanduser().resolve()
    config_path = path / "config.json"
    if not config_path.is_file():
        raise ValueError("Qwen4 artifact has no config.json")
    config = json.loads(config_path.read_text())
    text = config.get("text_config", config)
    if config.get("model_type") != "qwen4_exp":
        raise ValueError("strategy gate requires a qwen4_exp artifact")
    return {
        "status": "static_metadata_verified",
        "path": str(path),
        "config_sha256": sha256(config_path),
        "layers": int(text["num_hidden_layers"]),
        "max_position_embeddings": int(text["max_position_embeddings"]),
        "indexer_budget": int(text["indexer_budget"]),
        "indexer_compress_ratio": int(text["indexer_compress_ratio"]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
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
    report = {
        "schema": "mlx2.qwen4-strategy-transfer-plan.v1",
        "status": "staged_cpu_only_gpu_authority_required",
        "source_commit": head,
        "artifact": artifact(args.model),
        "implemented": {
            "longest_first_planner": True,
            "invalid_sibling_pruning": True,
            "adapter_state_contract": True,
            "serving_route": False,
        },
        "candidate": {
            "verification": "exact_target_draw_then_prefix_match",
            "ordering": "longest_first",
            "accepted_prefix_state": "canonical_ordinary_replay",
            "transactional_multirow_state_reuse": False,
            "apcv2_publication": False,
        },
        "required_native_cells": [
            {
                "context": 128,
                "purpose": "controlled longest-first and prefix-reuse geometry",
            },
            {"context": 2048, "purpose": "first exact transcript/state gate"},
            {"context": 8192, "purpose": "QSA boundary below indexer budget"},
            {"context": 16384, "purpose": "QSA selected-history gate"},
            {"context": 32768, "purpose": "memory and long-context gate"},
        ],
        "arms": [
            "ordinary_s1",
            "parallel_complete_paths",
            "longest_first",
            "longest_first_full_replay",
            "longest_first_canonical_prefix_reuse",
        ],
        "hard_gates": [
            "ordinary greedy and sampled transcript parity",
            "next-step logits and every QSA/GDN/cache plane match canonical S1",
            "accepted-prefix state is request-private and never published to APCv2",
            "both GPU locks and waiter ownership recorded",
            "zero swapout growth; thermal and arm order retained",
        ],
        "qualification": False,
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
