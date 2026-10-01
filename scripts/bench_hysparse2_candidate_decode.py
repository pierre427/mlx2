"""Paired end-to-end ordinary decode after candidate fine-ranking changes."""

import argparse
import importlib.util
import json
import sys
import time
from types import MethodType
from pathlib import Path

from mlx2.experimental.hysparse2.config import Config
from mlx2.experimental.hysparse2.resources import gpu_guard
from mlx2.experimental.hysparse2.train import _load_model_state, file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokens", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate", type=Path, help="Isolated candidate attention module; does not replace runtime source")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--contexts", type=int, nargs="+", default=[4096, 16384])
    p.add_argument("--decode-tokens", type=int, default=16)
    p.add_argument("--public-methods", action="store_true", help="Compare reference model public prefill/decode methods on current internals")
    args = p.parse_args()
    if args.candidate and args.public_methods:
        p.error("candidate attention cannot be combined with public-method comparison")
    if (any(length < 8 for length in args.contexts)
            or len(set(args.contexts)) != len(args.contexts)
            or not 1 <= args.decode_tokens <= 64):
        p.error("contexts must be unique and at least8; decode tokens must be1..64")
    if args.output.exists():
        p.error("use a new receipt path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    name = "mlx2.experimental.hysparse2.reference_model" if args.public_methods else "reference"
    spec = importlib.util.spec_from_file_location(name, args.reference)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[name] = reference
    spec.loader.exec_module(reference)
    candidate = None
    if args.candidate:
        candidate_spec = importlib.util.spec_from_file_location("isolated_attention_candidate", args.candidate)
        candidate = importlib.util.module_from_spec(candidate_spec)
        candidate_spec.loader.exec_module(candidate)
    c = Config(**json.loads((args.checkpoint / "state.json").read_text())["config"])
    report = {
        "schema": "mlx2.hysparse2-candidate-decode.v1",
        "completed": False,
        "serving_route_qualified": False,
        "repetitions": 1,
        "thermal_controls": False,
        "batch": 1,
        "dtype": "bfloat16",
        "decode_tokens": args.decode_tokens,
        "arm_order": ["reference", "gathered"],
        "contexts": [],
        "checkpoint_sha256": file_hash(args.checkpoint / "model.safetensors"),
        "tokens_sha256": file_hash(args.tokens),
        "reference_sha256": file_hash(args.reference),
        "candidate_sha256": file_hash(args.candidate) if args.candidate else None,
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    with gpu_guard(wait_seconds=0):
        import mlx.core as mx
        import numpy as np

        from mlx2.experimental.hysparse2 import attention
        from mlx2.experimental.hysparse2 import model as model_module
        candidate_attention = candidate.attention if candidate is not None else attention.attention

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(24 << 30)
        mx.set_cache_limit(256 << 20)
        model = model_module.Model(c)
        _load_model_state(args.checkpoint, model)
        model.set_dtype(mx.bfloat16)
        model.eval()
        mx.eval(model.parameters())
        ordinary_methods = (model.prefill, model.decode)
        report["comparison"] = "public-prefill-decode-on-current-internals" if args.public_methods else "attention"
        def install(label, function):
            model_module.attention = function
            if args.public_methods:
                model.prefill, model.decode = (
                    (MethodType(reference.Model.prefill, model), MethodType(reference.Model.decode, model))
                    if label == "reference" else ordinary_methods
                )
        values = np.load(args.tokens, allow_pickle=False)
        if (values.ndim != 1 or len(values) < max(args.contexts)
                or c.max_context < max(args.contexts) + args.decode_tokens):
            raise ValueError("need flat prompt tokens and sufficient context capacity")
        report["source_hashes"] = {
            str(path): file_hash(path)
            for path in (
                Path(__file__),
                Path(attention.__file__),
                Path(model_module.__file__),
            )
        }
        try:
            for label, function in (
                ("reference", reference.attention),
                ("gathered", candidate_attention),
            ):
                install(label, function)
                logits, cache = model.prefill(mx.array(values[:8][None]))
                model.decode(mx.argmax(logits[:, -1], axis=-1)[:, None], cache)
                del logits, cache
            mx.synchronize()
            mx.clear_cache()
            for length in args.contexts:
                row, results = {"context": length, "arms": {}}, []
                for label, function in (
                    ("reference", reference.attention),
                    ("gathered", candidate_attention),
                ):
                    install(label, function)
                    stats = (getattr(candidate, "_MATERIALIZATION_STATS", None)
                             if label == "gathered" and candidate is not None else None)
                    if stats is not None:
                        stats["evaluations"] = 0
                    print(json.dumps({"event": "arm_start", "context": length,
                                      "arm": label}), flush=True)
                    prompt = mx.array(values[:length][None])
                    mx.eval(prompt)
                    mx.synchronize()
                    mx.reset_peak_memory()
                    start = time.perf_counter()
                    logits, cache = model.prefill(prompt)
                    mx.synchronize()
                    prefill = time.perf_counter() - start
                    prefill_evaluations = stats["evaluations"] if stats is not None else None
                    print(json.dumps({"event": "prefill_complete", "context": length,
                                      "arm": label, "seconds": prefill,
                                      "periodic_materializations": prefill_evaluations}), flush=True)
                    generated = []
                    start = time.perf_counter()
                    for _ in range(args.decode_tokens):
                        token = mx.argmax(logits[:, -1], axis=-1)[:, None]
                        generated.append(token)
                        logits = model.decode(token, cache)
                    mx.eval(logits, generated)
                    mx.synchronize()
                    elapsed = time.perf_counter() - start
                    assert cache.length == length + args.decode_tokens and bool(
                        mx.all(mx.isfinite(logits)).item()
                    )
                    output = mx.concatenate(generated, axis=1).tolist()[0]
                    results.append((output, logits))
                    row["arms"][label] = {
                        "prefill_seconds": prefill,
                        "prefill_tokens_per_second": length / prefill,
                        "decode_seconds": elapsed,
                        "decode_tokens_per_second": args.decode_tokens / elapsed,
                        "peak_memory_bytes": mx.get_peak_memory(),
                        "kv_bytes": cache.resident_bytes(),
                    }
                    if stats is not None:
                        row["arms"][label]["periodic_materializations"] = {
                            "prefill": prefill_evaluations,
                            "decode": stats["evaluations"] - prefill_evaluations,
                        }
                    del cache, prompt, generated
                    mx.clear_cache()
                row["tokens_equal"] = results[0][0] == results[1][0]
                row["final_logits_max_error"] = float(
                    mx.max(mx.abs(results[0][1] - results[1][1])).item()
                )
                report["contexts"].append(row)
                save()
                print(json.dumps(row), flush=True)
                assert row["tokens_equal"] and row["final_logits_max_error"] == 0, row
                del results
        finally:
            model_module.attention = attention.attention
            model.prefill, model.decode = ordinary_methods
        report["completed"] = True
        save()


if __name__ == "__main__":
    main()
