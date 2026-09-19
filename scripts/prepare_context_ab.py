#!/usr/bin/env python3
"""Freeze an A/B source pair and a one-run Flash-Next context ladder.

A is the current worktree with a reviewed path set restored from ``HEAD``.
B is the current worktree as-is.  Both arms otherwise use the same qualified
Flash-Next MTP configuration and frozen prompts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_context_prompts import calibrate, rendered_tokens  # noqa: E402
from run_qualification_matrix import (  # noqa: E402
    alternating_cells,
    atomic_json,
    digest,
    host_identity,
)


DEFAULT_REVERT_PATHS = (
    "src/mlx2/runtime/generate.py",
    "src/mlx2/adapters/qwen38_memory.py",
    "src/mlx2/runtime/cache_planes.py",
)


def source_digest(source_root: Path) -> str:
    digest = hashlib.sha256()
    package = source_root / "mlx2"
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_sources(
    output_dir: Path, baseline_revision: str, revert_paths=DEFAULT_REVERT_PATHS
) -> dict:
    inputs = output_dir / "inputs"
    source_a = inputs / "source-a" / "src"
    source_b = inputs / "source-b" / "src"
    for target in (source_a, source_b):
        if target.parent.exists():
            shutil.rmtree(target.parent)
        shutil.copytree(ROOT / "src", target)
    audit = []
    for relative in revert_paths:
        baseline = subprocess.check_output(
            ["git", "show", f"{baseline_revision}:{relative}"], cwd=ROOT
        )
        a_path = source_a / Path(relative).relative_to("src")
        b_path = source_b / Path(relative).relative_to("src")
        a_path.write_bytes(baseline)
        audit.append(
            {
                "path": relative,
                "a_sha256": file_sha256(a_path),
                "b_sha256": file_sha256(b_path),
                "different": a_path.read_bytes() != b_path.read_bytes(),
            }
        )
    if not all(row["different"] for row in audit):
        unchanged = [row["path"] for row in audit if not row["different"]]
        raise RuntimeError(f"requested A/B paths have no worktree delta: {unchanged}")
    return {
        "a": {"src": str(source_a), "source_sha256": source_digest(source_a)},
        "b": {"src": str(source_b), "source_sha256": source_digest(source_b)},
        "path_audit": audit,
        "worktree_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "a_path_baseline_revision": subprocess.check_output(
            ["git", "rev-parse", baseline_revision], cwd=ROOT, text=True
        ).strip(),
    }


def arm_for_variant(base_arm: dict, variant: str, output_dir: Path, source: dict) -> dict:
    arm = copy.deepcopy(base_arm)
    arm["name"] = variant
    arm["activation_key"] = f"flash-next-mtp-context-ab-{variant.lower()}"
    command = arm["activate_command"]
    separator = command.index("--")
    command[command.index("--log") + 1] = str(output_dir / f"server-{variant}.log")
    command[command.index("--state") + 1] = str(output_dir / "server.json")
    env_index = separator + 2
    if command[separator + 1] != "/usr/bin/env" or not command[env_index].startswith("PYTHONPATH="):
        raise ValueError("unexpected activation command shape")
    command[env_index] = f"PYTHONPATH={source['src']}"
    cache_index = command.index("--cache-dir", separator + 1) + 1
    command[cache_index] = str(output_dir / f"cache-{variant}")
    arm["qualification_receipt"] = {
        **arm["qualification_receipt"],
        "path": str(output_dir / f"route-{variant}.json"),
    }
    arm["source_variant"] = {
        "name": variant,
        "src": source["src"],
        "source_sha256": source["source_sha256"],
    }
    # A context-ladder cell is intentionally B1. It must prove self-MTP and
    # every mechanism that can execute on a single lane, while B2-only
    # segmented/shared/private-delta mechanisms remain bound to the aggregate
    # route qualification receipt above.
    for requirement in arm["receipt_requirements"]:
        if requirement.get("path") in {
            "mtp.route",
            "mtp.num_draft",
            "mtp.stats.draft_proposed",
        }:
            requirement["max_context_tokens"] = 131072
        if requirement.get("path") == "mtp.route":
            maximum = requirement["max_context_tokens"]
            requirement.clear()
            requirement.update(
                path="mtp.route", contains="self_mtp",
                max_context_tokens=maximum,
            )
    if variant == "A":
        arm["receipt_requirements"].extend(
            [
                {"path": "mtp", "equals": None, "min_context_tokens": 262016},
                {
                    "path": "ordinary_compute_width",
                    "gte": 1,
                    "min_context_tokens": 262016,
                },
            ]
        )
    else:
        # The peer-PR cache-plane hardening is expected to keep segmented
        # self-MTP admitted at the 262K boundary where the restored baseline
        # falls back to ordinary decode.  Bind the B cell to actual route
        # engagement rather than treating that improvement as a gate failure.
        arm["receipt_requirements"].extend(
            [
                {
                    "path": "mtp.route",
                    "contains": "self_mtp",
                    "min_context_tokens": 262016,
                },
                {
                    "path": "mtp.num_draft",
                    "equals": 2,
                    "min_context_tokens": 262016,
                },
                {
                    "path": "mtp.stats.draft_proposed",
                    "gt": 0,
                    "min_context_tokens": 262016,
                },
            ]
        )
    aggregate_only = {
        "execution.segmented_mtp.engaged",
        "execution.segmented_mtp.committed_cycles",
        "execution.segmented_mtp.private_delta_attention_calls",
        "execution.indexed_qsa.counts.engaged",
    }
    arm["counter_requirements"] = [
        requirement
        for requirement in arm["counter_requirements"]
        if requirement.get("path") not in aggregate_only
    ]
    for requirement in arm["counter_requirements"]:
        if requirement.get("path") in {
            "execution.fused_gdn.verify_calls",
            "execution.round_levers.ple_tail_prefetch_tables",
        }:
            requirement["max_context_tokens"] = 131072
    return arm


def rebase_passing_cells(source_report: Path, target_report: Path, manifest_path: Path) -> None:
    """Carry only already-passing cells across a gate-only manifest revision."""
    report = json.loads(source_report.read_text())
    manifest = json.loads(manifest_path.read_text())
    old_cells = alternating_cells(report["manifest"]["models"])
    new_cells = alternating_cells(manifest["models"])
    if old_cells != new_cells:
        raise ValueError("refusing to rebase a report across a changed cell schedule")
    if report["manifest"]["experiment"]["source_freeze"] != manifest["experiment"]["source_freeze"]:
        raise ValueError("refusing to rebase a report across changed source variants")
    previous_hash = report["manifest_sha256"]
    previous_host = report["host"]
    report["cells"] = {
        cell_id: row for cell_id, row in report["cells"].items() if row.get("passed") is True
    }
    report["manifest"] = manifest
    report["manifest_sha256"] = digest(manifest)
    report["host"] = host_identity()
    report["passed"] = False
    report.pop("completed_at", None)
    report["gate_rebase"] = {
        "source_report": str(source_report.resolve()),
        "previous_manifest_sha256": previous_hash,
        "new_manifest_sha256": report["manifest_sha256"],
        "previous_host": previous_host,
        "preserved_passing_cells": len(report["cells"]),
        "scope": "receipt and counter applicability only; cell schedule and source freeze unchanged",
    }
    atomic_json(target_report, report)


def build_manifest(base_manifest: Path, output_dir: Path, frozen: dict) -> Path:
    manifest = json.loads(base_manifest.read_text())
    model = next(row for row in manifest["models"] if row["name"] == "flash-next")
    base_arm = next(row for row in model["arms"] if row["name"] == "mtp")
    model["arms"] = [
        arm_for_variant(base_arm, "A", output_dir, frozen["a"]),
        arm_for_variant(base_arm, "B", output_dir, frozen["b"]),
    ]
    model["context"]["runs_per_cell"] = 1
    model["arm_order_claim"] = "alternating"
    prompt_dir = output_dir / "inputs" / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model["tokenizer_path"], trust_remote_code=False, local_files_only=True
    )
    for context in model["contexts"]:
        target = int(context["tokens"])
        text = calibrate(tokenizer, target, model["tokenizer_renderer"])
        prompt_path = prompt_dir / f"flash-next-{target}.txt"
        prompt_path.write_text(text)
        context.update(
            prompt_path=str(prompt_path),
            prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
            calibrated_prompt_tokens=rendered_tokens(
                tokenizer, text, model["tokenizer_renderer"]
            ),
            tokenizer_renderer=model["tokenizer_renderer"],
        )
    manifest["models"] = [model]
    manifest["experiment"] = {
        "kind": "exploratory-source-ab-context-ladder",
        "runs_per_cell": 1,
        "qualification_claim": False,
        "a": "current all-features worktree with peer-PR paths restored from HEAD",
        "b": "same current all-features worktree with peer-PR paths retained",
        "source_freeze": frozen,
    }
    output = output_dir / "context-ab.generated.json"
    atomic_json(output, manifest)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=ROOT / "qualification" / "four-model-experiments.json",
    )
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--resume-output", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-revision",
        required=True,
        help="Git revision supplying only the reviewed A-side path versions.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen = freeze_sources(args.output_dir, args.baseline_revision)
    manifest = build_manifest(args.base_manifest.resolve(), args.output_dir.resolve(), frozen)
    if args.resume_from or args.resume_output:
        if not (args.resume_from and args.resume_output):
            parser.error("--resume-from and --resume-output must be supplied together")
        rebase_passing_cells(args.resume_from, args.resume_output, manifest)
    print(manifest)


if __name__ == "__main__":
    main()
