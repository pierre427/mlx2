"""Write a CPU-only LFM2.5-VL exact-prefix qualification plan."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET = Path.home() / "mlx-models" / "LFM2.5-VL-3B"
DEFAULT_DRAFT = Path.home() / "mlx-models" / "LFM2.5-VL-3B-DSpark"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
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
    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.adapters.lfm25_vl import inspect_artifact, inspect_dspark_artifact

    target = inspect_artifact(args.target)
    draft = inspect_dspark_artifact(args.draft, target=target)
    report = {
        "schema": "mlx2.lfm25-vl-strategy-transfer-plan.v1",
        "status": "staged_cpu_only_gpu_authority_required",
        "source_commit": head,
        "artifacts": {
            "target": {
                "path": target["path"],
                "fingerprint": target["fingerprint"],
                "header_sha256": target["header_sha256"],
                "cache_layout": "lfm25-vl-hybrid-image-apcv2-v2",
            },
            "dspark": {
                "path": draft["path"],
                "fingerprint": draft["fingerprint"],
                "target_fingerprint": draft["target_fingerprint"],
                "qualified": False,
                "selected": False,
            },
        },
        "implemented": {
            "generic_prefill_autoscale_after_adapter_decline": True,
            "longest_first_planner": True,
            "invalid_sibling_pruning": True,
            "lfm_exact_hybrid_state_gate": True,
            "suffix_only_shared_prefix_plan": True,
            "serving_route": False,
        },
        "candidate": {
            "execution_domain": "autoregressive_language_only",
            "accepted_prefix_state": "exact_ordinary_hybrid_checkpoint",
            "attention_layers": 8,
            "recurrent_layers": 22,
            "media_prefill": "bound_below_language_checkpoint",
            "dspark_state": "excluded",
            "apcv2_publication": False,
        },
        "required_native_cells": [
            "ordinary text cold and checkpoint-reuse parity",
            "image cold and suffix-only branch parity",
            "ordered two-frame cold and suffix-only branch parity",
            "all eight KV and 22 ShortConv planes plus next-step logits",
            "changed-media refusal and DSpark-state refusal",
        ],
        "hard_gates": [
            "exact clean source revision and target/draft artifact identities",
            "ordinary transcript and state parity at every accepted length",
            "image/frame work remains outside the AR cascade",
            "DSpark remains offline, unqualified and unselected",
            "shared waiter and both GPU locks recorded",
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
