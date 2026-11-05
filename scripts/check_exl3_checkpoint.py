#!/usr/bin/env python3
"""CPU decode oracle for an upstream EXL3 checkpoint (no GPU, no serving).

Admits the checkpoint through the adapter registry (headers only), checks
that per-expert tensors stack exactly, then decodes sampled modules to dense
weights with the independent host decoder and compares them with an affine
MLX checkpoint of the same base model. A wrong tile order, codebook or
Hadamard placement reads as cosine near 0; quantization noise does not.

Hessian-weighted quantizers may leave arbitrary weights on input rows that
never see activation (layer-0 experts here), so each module is judged on the
reference's live rows (norm >= 10% of its largest row); the all-row cosine is
kept beside it. A row-shuffled reference is the negative control.

The cosine bar depends on the rate: no K-bit code beats the Gaussian
rate-distortion floor D = 2^(-2K), so the bar allows four times that
distortion, capped at 0.98 (K2 0.866, K3 0.968, K4+ 0.98). A fixed 0.98 is
unreachable at K2 (best possible cosine about 0.968); a wrong decode still
reads near 0.

Usage: check_exl3_checkpoint.py --model EXL3_DIR --reference AFFINE_DIR --out NEW_DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mlx.core as mx  # noqa: E402

from mlx2.adapters.registry import inspect_model  # noqa: E402
from mlx2.runtime.models.exl3_reference import effective_weight  # noqa: E402
from mlx2.runtime.trellis.upstream import ShardReader, load_bank, load_linear  # noqa: E402

PASS_COSINE_CAP = 0.98
DISTORTION_ALLOWANCE = 4.0
CONTROL_COSINE = 0.2
LIVE_FRACTION = 0.1
HEAD_COLUMNS = ((0, 1024), (124160, 125184))


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _artifact_name(path):
    # A Hugging Face cache snapshot is named by its repository, not its revision.
    if path.parent.name == "snapshots" and path.parent.parent.name.startswith("models--"):
        return path.parent.parent.name.removeprefix("models--").replace("--", "/", 1)
    return path.name


def _cos(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))


def _reference_name(key):
    name = key.replace("model.language_model.", "language_model.model.", 1)
    if key == "lm_head":
        return "language_model.lm_head", None
    match = re.match(r"^(.+)\.experts\.(\d+)\.(\w+_proj)$", name)
    if match:
        return f"{match[1]}.switch_mlp.{match[3]}", int(match[2])
    return name, None


class Reference:
    def __init__(self, path):
        self.path = Path(path)
        self.map = json.loads((self.path / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        self.quant = json.loads((self.path / "config.json").read_text())["quantization"]
        self.read = ShardReader(self.path, self.map)

    def weight(self, name, expert=None, rows=None):
        """Dequantized [out, in] float64 rows of one (expert) linear."""
        spec = self.quant.get(name, self.quant)
        w, s, b = (self.read(f"{name}.{part}") for part in ("weight", "scales", "biases"))
        if expert is not None:
            w, s, b = w[expert], s[expert], b[expert]
        if rows is not None:
            w, s, b = w[rows[0]:rows[1]], s[rows[0]:rows[1]], b[rows[0]:rows[1]]
        dense = mx.dequantize(
            w, s, b, group_size=spec["group_size"], bits=spec["bits"], mode=spec.get("mode", "affine")
        )
        return np.asarray(dense.astype(mx.float32)).astype(np.float64)


def pass_cosine(bits):
    return min(PASS_COSINE_CAP, float(np.sqrt(1 - DISTORTION_ALLOWANCE * 2.0 ** (-2 * bits))))


def _compare(decoded, reference, rng, bits):
    """decoded and reference are [in, out]; rows are input channels."""
    norms = np.linalg.norm(reference, axis=1)
    live = norms >= LIVE_FRACTION * norms.max()
    control = reference[rng.permutation(reference.shape[0])]
    result = {
        "cosine_all_rows": _cos(decoded, reference),
        "cosine_live_rows": _cos(decoded[live], reference[live]),
        "live_rows": int(live.sum()),
        "rows": int(live.size),
        "relative_error_live_rows": float(
            np.linalg.norm(decoded[live] - reference[live]) / np.linalg.norm(reference[live])
        ),
        "control_cosine_shuffled_rows": _cos(decoded, control),
        "finite": bool(np.isfinite(decoded).all()),
    }
    result["pass_cosine"] = pass_cosine(bits)
    result["passed"] = (
        result["finite"]
        and result["cosine_live_rows"] >= pass_cosine(bits)
        and abs(result["control_cosine_shuffled_rows"]) <= CONTROL_COSINE
    )
    return result


def _samples(plan, layers, experts):
    keys = []
    for key in sorted(plan.linears):
        match = re.search(r"\.layers\.(\d+)\.", key)
        if key == "lm_head" or (match and int(match[1]) in layers):
            keys.append(key)
    for bank in sorted(plan.banks):
        match = re.search(r"\.layers\.(\d+)\.", bank)
        if match and int(match[1]) in layers:
            spec = plan.banks[bank]
            keys.extend(spec.expert_key(e) for e in experts if e < spec.experts)
    return keys


def _bank_stacking(read, plan, rng):
    """Stacked banks must equal their per-expert tensors exactly."""
    out = []
    for key in sorted(plan.banks)[:: max(1, len(plan.banks) // 3)][:3]:
        spec = plan.banks[key]
        stacked = load_bank(read, spec)
        mx.eval(stacked)
        for expert in sorted({0, spec.experts - 1, int(rng.integers(spec.experts))}):
            single = load_linear(read, spec.expert_spec(expert))
            for part in ("trellis", "suh", "svh"):
                if not bool(mx.array_equal(stacked[part][expert], single[part]).item()):
                    raise SystemExit(f"stacked {key}[{expert}].{part} differs")
        out.append({
            "bank": key,
            "experts": spec.experts,
            "trellis": [list(stacked["trellis"].shape), str(stacked["trellis"].dtype)],
            "suh": list(stacked["suh"].shape),
            "svh": list(stacked["svh"].shape),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", default="0,1,10,20,39")
    parser.add_argument("--experts", default="0,77,255")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty evidence directory {out}")
    mx.set_default_device(mx.cpu)
    started = time.time()
    status = _git("status", "--porcelain", "--untracked-files=no")
    resolution = inspect_model(args.model)
    artifact = resolution.artifact
    plan = artifact["exl3_upstream"]
    if plan is None:
        raise SystemExit("not an upstream EXL3 checkpoint")
    reference = inspect_model(args.reference)
    model_path = Path(args.model).expanduser().resolve()
    read = ShardReader(model_path, artifact["weight_map"])
    ref = Reference(Path(args.reference).expanduser().resolve())
    rng = np.random.default_rng(20261104)
    layers = {int(x) for x in args.layers.split(",")}
    experts = [int(x) for x in args.experts.split(",")]
    banks = _bank_stacking(read, plan, rng)
    modules = []
    for key in _samples(plan, layers, experts):
        match = re.match(r"^(.+)\.experts\.(\d+)\.(\w+_proj)$", key)
        spec = (
            plan.banks[f"{match[1]}.{match[3]}"].expert_spec(int(match[2]))
            if match
            else plan.linears[key]
        )
        tensors = load_linear(read, spec)
        name, expert = _reference_name(key)
        windows = HEAD_COLUMNS if key == "lm_head" else ((0, spec.output_dims),)
        for lo, hi in windows:
            trellis = np.asarray(tensors["trellis"][:, lo // 16 : hi // 16])
            decoded = effective_weight(
                trellis,
                np.asarray(tensors["suh"]),
                np.asarray(tensors["svh"][lo:hi]),
                spec.codebook,
            )
            target = ref.weight(name, expert, rows=(lo, hi) if key == "lm_head" else None).T
            if decoded.shape != target.shape:
                raise SystemExit(f"{key}: decoded {decoded.shape} vs reference {target.shape}")
            row = {
                "module": key,
                "columns": [lo, hi],
                "codebook": spec.codebook,
                "bits_per_weight": spec.bits_per_weight,
                "reference": name if expert is None else f"{name}[{expert}]",
                **_compare(decoded, target, rng, spec.bits_per_weight),
            }
            modules.append(row)
            print(
                f"{'ok  ' if row['passed'] else 'FAIL'} {key} [{lo}:{hi}] k{spec.bits_per_weight:g} "
                f"live {row['cosine_live_rows']:.4f} all {row['cosine_all_rows']:.4f} "
                f"ctl {row['control_cosine_shuffled_rows']:+.3f}",
                flush=True,
            )
    receipt = {
        "schema": "mlx2.exl3-upstream-decode-oracle.v1",
        "checkpoint": f"{_artifact_name(model_path)}@{artifact['identity']['fingerprint']}",
        "reference_checkpoint": (
            f"{_artifact_name(Path(args.reference).expanduser().resolve())}"
            f"@{reference.artifact['identity']['fingerprint']}"
        ),
        "source_revision": _git("rev-parse", "HEAD"),
        "source_dirty": bool(status),
        "plan": plan.receipt(),
        "bank_stacking": banks,
        "thresholds": {
            "cosine_live_rows": "min(cap, sqrt(1 - allowance * 2^(-2K)))",
            "cosine_cap": PASS_COSINE_CAP,
            "distortion_allowance": DISTORTION_ALLOWANCE,
            "control_abs_cosine": CONTROL_COSINE,
            "live_row_fraction": LIVE_FRACTION,
        },
        "modules": modules,
        "passed": all(row["passed"] for row in modules),
        "failed_modules": [row["module"] for row in modules if not row["passed"]],
        "device": "cpu",
        "elapsed_s": round(time.time() - started, 1),
    }
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / ".receipt.json.tmp"
    tmp.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    tmp.replace(out / "receipt.json")
    digest = hashlib.sha256((out / "receipt.json").read_bytes()).hexdigest()
    print(f"passed={receipt['passed']} modules={len(modules)} receipt sha256 {digest}")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
