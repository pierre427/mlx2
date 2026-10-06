#!/usr/bin/env python3
"""Write a CPU-only Agnes exact-prefix-cascade native qualification plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = (
    Path.home() / "mlx-models" / "Agnes-3.0-Flash-Preview-MLX-6bit"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict:
    path = path.expanduser().resolve()
    config_path = path / "config.json"
    if not config_path.is_file():
        raise ValueError("Agnes artifact has no config.json")
    config = json.loads(config_path.read_text())
    text = config.get("text_config")
    vision = config.get("vision_config")
    if config.get("model_type") != "agnes" or not isinstance(text, dict):
        raise ValueError("strategy gate requires an Agnes conditional-generation artifact")
    if not isinstance(vision, dict):
        raise TypeError("Agnes conditional-generation artifact lacks vision metadata")
    return {
        "status": "static_metadata_verified",
        "path": str(path),
        "config_sha256": sha256(config_path),
        "text_layers": int(text["num_hidden_layers"]),
        "text_max_position_embeddings": int(text["max_position_embeddings"]),
        "vision_depth": int(vision["depth"]),
        "text_cache_layout": "agnes-hybrid-layer-segments-v1",
        "conditional_generation_state_separate": True,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
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
        "schema": "mlx2.agnes-strategy-transfer-plan.v1",
        "status": "staged_cpu_only_gpu_authority_required",
        "source_commit": head,
        "artifact": artifact(args.model),
        "implemented": {
            "adapter_prefill_step_2048": True,
            "generic_autoscale_only_on_adapter_decline": True,
            "longest_first_planner": True,
            "invalid_sibling_pruning": True,
            "exact_text_state_gate": True,
            "serving_route": False,
        },
        "candidate": {
            "verification": "canonical_target_draw_prefix_match",
            "ordering": "longest_first",
            "accepted_prefix_state": "exact_text_decoder_hybrid_checkpoint",
            "reuse": "suffix_only_without_common_token_replay",
            "vision_or_conditional_generation_reuse": False,
            "apcv2_publication": False,
        },
        "required_native_cells": [
            {"context": 128, "purpose": "controlled exact-prefix geometry"},
            {"context": 2048, "purpose": "pinned prefill boundary"},
            {"context": 8192, "purpose": "multi-chunk hybrid-state gate"},
            {"context": 32768, "purpose": "long-context state and memory gate"},
        ],
        "arms": [
            "ordinary_s1",
            "parallel_complete_paths",
            "longest_first_full_replay",
            "longest_first_exact_hybrid_prefix_reuse",
        ],
        "hard_gates": [
            "ordinary greedy and sampled transcript parity",
            "next-step logits and all recurrent/KV planes match canonical S1",
            "reused state contains text decoder planes only",
            "vision tower and conditional-generation prefill are never replayed as text state",
            "accepted-prefix state stays request-private and is not published to APCv2",
            "shared waiter and both GPU locks recorded against an exact clean source commit",
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
