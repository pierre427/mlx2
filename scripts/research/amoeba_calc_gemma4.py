#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Pierre-Lamy-Calculator-NC-ND-1.0
# Copyright (c) 2026 Pierre Lamy
# Scope and terms: provenance/LICENSE-calculator-scope.json; provenance/LICENSE-calculator.txt
"""CPU-only, exact-artifact preparation for Gemma 4 resident calculator tests.

No model weights, MLX arrays, GPU dispatch or benchmark answers enter the tile.
Run the emitted commands only inside the project's normal owned GPU window.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = Path.home() / "mlx-models/gemma-4-26B-A4B-it-MLX-8bit"


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def prepare(model, out):
    from mlx2.adapters.gemma4 import inspect_gemma4_artifact, _TOPOLOGIES
    from mlx2.runtime.amoeba_det_tiles import SCHEMA, PARSER_V5

    model = Path(model).resolve()
    artifact = inspect_gemma4_artifact(model)
    layers = _TOPOLOGIES[artifact["variant"]]["layers"]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    checkpoint = model.name + "@" + artifact["fingerprint"][:8]
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    # No learned weights are needed: the gate validates the full expression and
    # its exact result. The hashed data file binds the kernel/parser contract.
    data = {
        "schema": "mlx2.calculator-resident-binding.v1",
        "parser": PARSER_V5,
        "gate": "complete_syntax_and_exact_result",
        "source_revision": revision,
        "license": "LicenseRef-Pierre-Lamy-Calculator-NC-ND-1.0",
    }
    _write(out / "resident.json", data)
    manifest = {
        "schema": SCHEMA,
        "source": {
            "fingerprint": artifact["fingerprint"],
            "checkpoint": checkpoint,
            "path": str(model),
        },
        "layers": layers,
        "tiles": [{
            "id": "calculator", "kind": "calculator", "write_mode": "force",
            "read_layer": layers - 1, "write_layer": layers - 1,
            "file": "resident.json", "parser": PARSER_V5,
            "arithmetic_backend": "gpu", "gate": {"kind": "validated_syntax"},
            "complete_literal_requests": True,
        }],
        "files": {"resident.json": hashlib.sha256((out / "resident.json").read_bytes()).hexdigest()},
        "qualification": "unqualified",
        "calculator_license": "LicenseRef-Pierre-Lamy-Calculator-NC-ND-1.0",
        "host_tiles_manifest_sha256": None,
    }
    _write(out / "manifest.json", manifest)
    _write(out / "execution-policy.json", {"amoeba_det_tiles": str(out / "manifest.json")})
    _write(out / "setup.json", {
        "checkpoint": checkpoint, "fingerprint": artifact["fingerprint"],
        "source_revision": revision, "model_path": str(model),
        "variant": artifact["variant"], "tensor_backend": "native",
        "parser": PARSER_V5, "qualification": "unqualified",
        "default_changed": False, "model_loaded": False, "gpu_used": False,
        "corpus": str(ROOT / "benchmarks/calculator_transfer/20261117"),
        "benchmark_command": [
            "python", str(ROOT / "scripts/research/amoeba_calc_27b.py"), "run",
            "--model", str(model), "--manifest", str(out / "manifest.json"),
            "--corpus", str(ROOT / "benchmarks/calculator_transfer/20261117"),
            "--out", str(out.parent / (out.name + "-served")),
            "--port", "8392", "--width", "4", "--max-tokens", "4096",
        ],
        "required_environment": {
            "PYTHONPATH": str(ROOT / "src"), "MLX2_GEMMA4_BACKEND": "native",
            "MLX_METAL_FAST_SYNCH": "0", "HF_HUB_OFFLINE": "1",
        },
        "notes": [
            "Preparation does not qualify the model or parser.",
            "Use a CPG lease and matching paired GPU locks for actual execution.",
            "The harness must include the --execution-policy JSON-file fix (30f7740ae or descendant).",
            "The frozen 128-case corpus is reused diagnostic evidence, not a new held-out benchmark.",
        ],
    })
    return out / "manifest.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    print(prepare(args.model, args.out))


if __name__ == "__main__":
    main()
