"""Write a CPU-only Muse Glimmer exact-prefix-cascade qualification plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_config(path: Path) -> tuple[Path, dict]:
    path = path.expanduser().resolve()
    config_path = path / "config.json"
    if not config_path.is_file():
        raise ValueError(f"artifact has no config.json: {path}")
    return config_path, json.loads(config_path.read_text())


def target_artifact(path: Path) -> dict:
    config_path, config = read_config(path)
    if config.get("model_type") not in {"muse_glimmer", "muse_glimmer_text"}:
        raise ValueError("strategy gate requires a Muse Glimmer target artifact")
    text = config.get("text_config", config)
    quant = config.get("quantization", config.get("quantization_config")) or {}
    return {
        "status": "static_metadata_verified",
        "role": "authoritative_text_target",
        "path": str(config_path.parent),
        "config_sha256": sha256(config_path),
        "layers": int(text["num_hidden_layers"]),
        "max_position_embeddings": int(text["max_position_embeddings"]),
        "sliding_window": int(text["sliding_window"]),
        "quantization_bits": quant.get("bits"),
    }


def proposal_artifact(path: Path) -> dict:
    config_path, config = read_config(path)
    model_type = config.get("model_type")
    architectures = tuple(config.get("architectures", ()))
    if model_type == "muse_glimmer_assistant" and architectures == (
        "MuseGlimmerAssistantModel",
    ):
        kind, implementation = "assistant", "metadata_only"
        block_size = int(config["block_size"])
    elif architectures == ("DFlash2DraftModel",) and "dflash_config" in config:
        kind, implementation = "dflash2", "external_draft_integrated"
        block_size = int(config["dflash_config"]["block_size"])
    else:
        raise ValueError("proposal artifact is neither Muse assistant nor DFlash2")
    return {
        "status": "static_metadata_verified",
        "role": "proposal_only",
        "kind": kind,
        "implementation": implementation,
        "path": str(config_path.parent),
        "config_sha256": sha256(config_path),
        "block_size": block_size,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--proposal", type=Path, action="append", default=[],
        help="Muse assistant or DFlash2 artifact; may be repeated",
    )
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
    target = target_artifact(args.model)
    proposals = [proposal_artifact(path) for path in args.proposal]
    if len({item["config_sha256"] for item in proposals}) != len(proposals):
        parser.error("duplicate proposal artifact identity")
    report = {
        "schema": "mlx2.muse-glimmer-strategy-transfer-plan.v1",
        "status": "staged_cpu_only_gpu_authority_required",
        "source_commit": head,
        "target": target,
        "proposal_sources": proposals,
        "implemented": {
            "longest_first_planner": True,
            "invalid_sibling_pruning": True,
            "adapter_state_contract": True,
            "adapter_prefill_step": 2048,
            "serving_route": False,
        },
        "candidate": {
            "verification": "exact_target_draw_then_prefix_match",
            "ordering": "longest_first",
            "accepted_prefix_state": "canonical_ordinary_replay",
            "shared_prefix_reuse": "suffix_only_after_exact_canonical_prefix",
            "transactional_multirow_state_reuse": False,
            "apcv2_publication": False,
        },
        "state_namespaces": {
            "target_text": "authoritative",
            "proposal_draft": "request_private",
            "multimodal": "excluded_from_text_route",
        },
        "required_native_cells": [
            {"context": 128, "purpose": "controlled cascade and rollback"},
            {"context": 2047, "purpose": "below rotating-window boundary"},
            {"context": 2048, "purpose": "at rotating-window boundary"},
            {"context": 2049, "purpose": "first rotating eviction boundary"},
            {"context": 32768, "purpose": "long-context global/local state gate"},
        ],
        "arms": [
            "ordinary_s1",
            "parallel_complete_paths",
            "longest_first_full_replay",
            "longest_first_canonical_prefix_reuse",
        ],
        "hard_gates": [
            "ordinary greedy and sampled transcript parity",
            "next-step logits and every global/sliding KV plane match canonical S1",
            "rotating offsets, retained windows and target/draft revisions match",
            "assistant and DFlash2 state remains proposal-only and request-private",
            "no multimodal state enters the text checkpoint namespace",
            "accepted-prefix state is never published to APCv2 before exact parity",
            "shared waiter and both GPU locks recorded against exact clean source",
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
