#!/usr/bin/env python3
"""Fresh-cache repeat control for the first long-context FGN QA/PPL input.

Runs E1/E2, F1/F2, then E/F and F/E on identical token IDs. This is a
direct-model numerical control, not a serving or route-qualification receipt.
Use only after obtaining the shared GPU lease and file lock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import eval_fused_group_norm_quality_v2 as gate

SCHEMA = "mlx2.fused-group-norm-repeat-control.v1"
PHASES = (
    ("E1", False), ("E2", False), ("F1", True), ("F2", True),
    ("EF_E", False), ("EF_F", True), ("FE_F", True), ("FE_E", False),
)
PAIRS = (
    ("E1", "E2"), ("F1", "F2"), ("E1", "F1"),
    ("E1", "EF_E"), ("F1", "EF_F"),
    ("F1", "FE_F"), ("E1", "FE_E"),
    ("EF_E", "EF_F"), ("FE_F", "FE_E"),
)


def compare_logprobs(a: np.ndarray, b: np.ndarray, targets: list[int]) -> dict:
    """Compare every vocab column of every scored row, in bounded row chunks."""
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != len(targets):
        raise ValueError("logprob shape or target count mismatch")
    if not np.issubdtype(a.dtype, np.floating) or not np.issubdtype(b.dtype, np.floating):
        raise ValueError("logprobs must be floating point")
    if any(t < 0 or t >= a.shape[1] for t in targets):
        raise ValueError("target outside vocabulary")
    n = len(targets)
    if n == 0:
        raise ValueError("empty scored continuation")
    exact = True
    max_abs = 0.0
    kl_ab = []
    kl_ba = []
    top1 = 0
    nll_a = []
    nll_b = []
    for i, target in enumerate(targets):
        x = np.asarray(a[i], dtype=np.float64)
        y = np.asarray(b[i], dtype=np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"non-finite logprob at row {i}")
        exact &= bool(np.array_equal(a[i], b[i]))
        max_abs = max(max_abs, float(np.max(np.abs(y - x))))
        top1 += int(np.argmax(x) == np.argmax(y))
        kl_ab.append(max(0.0, float(np.sum(np.exp(x) * (x - y)))))
        kl_ba.append(max(0.0, float(np.sum(np.exp(y) * (y - x)))))
        nll_a.append(-float(x[target]))
        nll_b.append(-float(y[target]))
    mean_a, mean_b = math.fsum(nll_a) / n, math.fsum(nll_b) / n
    return {
        "compared_positions": n, "vocab_size": a.shape[1],
        "full_matrix_equal": exact, "max_abs_logprob_delta": max_abs,
        "kl_a_to_b_mean": math.fsum(kl_ab) / n, "kl_a_to_b_max": max(kl_ab),
        "kl_b_to_a_mean": math.fsum(kl_ba) / n, "kl_b_to_a_max": max(kl_ba),
        "top1_agreement": top1 / n,
        "nll_a": mean_a, "nll_b": mean_b,
        "ppl_a": math.exp(mean_a), "ppl_b": math.exp(mean_b),
        "ppl_relative_delta": math.expm1(mean_b - mean_a),
    }


def source_digest(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def write_receipt(path: Path, receipt: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2) + "\n")


def validate_reference(reference: dict, receipt: dict, ppl: dict) -> None:
    """Bind the repeat control to the first input in the completed v2 gate."""
    if reference.get("schema") != gate.SCHEMA:
        raise ValueError("reference is not a v2 FGN quality receipt")
    matching = (
        (reference.get("source_revision"), receipt["source_revision"]),
        (reference.get("model"), receipt["model"]),
        (reference.get("model_config_sha256"), receipt["model_config_sha256"]),
        (reference.get("harness_sha256"), receipt["source_sha256"][
            "scripts/eval_fused_group_norm_quality_v2.py"]),
        (reference.get("kernel_sha256"), receipt["source_sha256"][
            "src/mlx2/runtime/models/qwen4_fused_group_norm.py"]),
        (reference["cases"][0]["prompt_sha256"], receipt["prompt_sha256"]),
        (reference["ppl_windows"][0]["context_sha256"], ppl["context_sha256"]),
        (reference["ppl_windows"][0]["continuation_sha256"],
         receipt["continuation_sha256"]),
    )
    if any(a != b for a, b in matching):
        raise ValueError("reference source or first QA/held-out input mismatch")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP")
    parser.add_argument("--context-tokens", type=int, default=16384)
    parser.add_argument("--score-tokens", type=int, default=128)
    parser.add_argument("--gen-cap", type=int, default=768)
    parser.add_argument("--out", required=True)
    parser.add_argument("--quality-receipt", default=(
        "qualification/runs/m5-fgn-quality-20260926/fgn-quality-v2.json"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args(argv)
    if args.context_tokens != 16384 or args.score_tokens != 128 or args.gen_cap < 256:
        parser.error("repeat control requires context 16384, score 128, gen-cap >=256")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=False, local_files_only=True)
    sources = gate.prepare_sources()
    qa = gate.build_qa(tokenizer, args.context_tokens, gate.CASES[:1], sources)[0]
    ppl = gate.build_ppl(tokenizer, args.context_tokens, args.score_tokens,
                         count=20, sources=sources)[0]
    receipt = {
        "schema": SCHEMA, "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {
            name: source_digest(name) for name in (
                "scripts/eval_fused_group_norm_repeat_control.py",
                "scripts/eval_fused_group_norm_quality_v2.py",
                "scripts/eval_fused_group_norm_quality.py",
                "src/mlx2/runtime/models/qwen4_fused_group_norm.py")},
        "model": args.model,
        "model_config_sha256": hashlib.sha256(
            (Path(args.model) / "config.json").read_bytes()).hexdigest(),
        "model_index_sha256": hashlib.sha256(
            (Path(args.model) / "model.safetensors.index.json").read_bytes()).hexdigest(),
        "tokenizer_json_sha256": hashlib.sha256(
            (Path(args.model) / "tokenizer.json").read_bytes()).hexdigest(),
        "context_tokens": args.context_tokens, "score_tokens": args.score_tokens,
        "gen_cap": args.gen_cap, "qa_index": qa["index"],
        "prompt_sha256": qa["prompt_sha256"],
        "heldout_context_sha256": ppl["context_sha256"],
        "continuation_sha256": ppl["continuation_sha256"],
        "input_source_sha256": {name: hashlib.sha256(data).hexdigest()
                                for name, data in sources.items()},
        "phases": [label for label, _ in PHASES], "comparisons": {},
        "scope": "fresh-cache direct-model repeat control; not serving qualification",
    }
    if args.dry_run:
        print(json.dumps(receipt, indent=2))
        return 0
    if not args.i_own_the_gpu:
        parser.error("refusing Metal without --i-own-the-gpu")
    reference_path = Path(args.quality_receipt)
    reference_bytes = reference_path.read_bytes()
    reference = json.loads(reference_bytes)
    if len(reference.get("results", [])) != len(reference.get("cases", [])):
        raise SystemExit("v2 quality reference has not finished every case")
    validate_reference(reference, receipt, ppl)
    receipt["quality_receipt_sha256"] = hashlib.sha256(reference_bytes).hexdigest()
    receipt["quality_receipt_path"] = str(reference_path.resolve())

    import mlx.core as mx

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime.models import qwen4_fused_group_norm as fgn

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise SystemExit("exclusive Metal GPU queue required")
    if fgn.fused_group_norm_enabled():
        raise SystemExit("start with production FGN lever off")
    adapter = resolve_adapter(args.model)(args.model)
    model, tokenizer = adapter.model, adapter.tokenizer
    receipt["loaded_identity"] = {
        "adapter_class": f"{type(adapter).__module__}.{type(adapter).__qualname__}",
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "tokenizer_class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
    }
    qa = gate.build_qa(tokenizer, args.context_tokens, gate.CASES[:1], sources)[0]
    ppl = gate.build_ppl(tokenizer, args.context_tokens, args.score_tokens,
                         count=20, sources=sources)[0]
    if (qa["prompt_sha256"] != receipt["prompt_sha256"] or
            ppl["continuation_sha256"] != receipt["continuation_sha256"]):
        raise SystemExit("loaded model tokenizer changed the controlled input")
    receipt["mlx_version"] = mx.__version__
    receipt["device"] = str(mx.default_device())
    receipt["runs"] = {}
    out = Path(args.out)
    with tempfile.TemporaryDirectory(prefix="fgn-repeat-") as temp:
        arrays = {}
        for label, fused in PHASES:
            qa_result, lp, score_seconds, counters = gate.run_arm(
                mx, model, tokenizer, qa, ppl, fused, fgn, args.gen_cap)
            values = np.asarray(lp, dtype=np.float32)
            if values.shape[0] != 128:
                raise RuntimeError(f"{label}: expected 128 logprob rows, got {values.shape}")
            data_path = Path(temp) / f"{label}.npy"
            np.save(data_path, values)
            arrays[label] = data_path
            receipt["runs"][label] = {
                "arm": "fused" if fused else "eager", "fresh_cache": True,
                "generated_token_ids": qa_result.pop("ids"),
                "generation": qa_result, "score_seconds": score_seconds,
                "counters": counters, "logprobs_shape": list(values.shape),
                "logprobs_dtype": str(values.dtype),
                "logprobs_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
            }
            write_receipt(out, receipt)
            del lp, values
            mx.clear_cache()
            print(f"{label}: complete", flush=True)
        for a_label, b_label in PAIRS:
            a = np.load(arrays[a_label], mmap_mode="r")
            b = np.load(arrays[b_label], mmap_mode="r")
            result = compare_logprobs(a, b, ppl["continuation"])
            aid = receipt["runs"][a_label]["generated_token_ids"]
            bid = receipt["runs"][b_label]["generated_token_ids"]
            common = 0
            while common < min(len(aid), len(bid)) and aid[common] == bid[common]:
                common += 1
            result.update(generated_ids_equal=aid == bid,
                          generated_common_prefix_tokens=common)
            receipt["comparisons"][f"{a_label}:{b_label}"] = result
            write_receipt(out, receipt)
            print(f"{a_label}:{b_label} exact={result['full_matrix_equal']} "
                  f"gen={result['generated_ids_equal']}", flush=True)
    receipt["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_receipt(out, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
