"""Paired end-to-end ordinary decode after candidate fine-ranking changes."""

import argparse
import importlib.util
import json
import time
from pathlib import Path

from mlx2.experimental.hysparse2.config import Config
from mlx2.experimental.hysparse2.resources import gpu_guard
from mlx2.experimental.hysparse2.train import _load_model_state, file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokens", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("use a new receipt path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("reference", args.reference)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    c = Config(**json.loads((args.checkpoint / "state.json").read_text())["config"])
    report = {
        "schema": "mlx2.hysparse2-candidate-decode.v1",
        "completed": False,
        "serving_route_qualified": False,
        "repetitions": 1,
        "thermal_controls": False,
        "batch": 1,
        "dtype": "bfloat16",
        "decode_tokens": 16,
        "arm_order": ["reference", "gathered"],
        "contexts": [],
        "checkpoint_sha256": file_hash(args.checkpoint / "model.safetensors"),
        "tokens_sha256": file_hash(args.tokens),
        "reference_sha256": file_hash(args.reference),
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    with gpu_guard(wait_seconds=0):
        import mlx.core as mx
        import numpy as np

        from mlx2.experimental.hysparse2 import attention
        from mlx2.experimental.hysparse2 import model as model_module

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(24 << 30)
        mx.set_cache_limit(256 << 20)
        model = model_module.Model(c)
        _load_model_state(args.checkpoint, model)
        model.set_dtype(mx.bfloat16)
        model.eval()
        mx.eval(model.parameters())
        values = np.load(args.tokens, allow_pickle=False)
        if values.ndim != 1 or len(values) < 16384 or c.max_context < 16400:
            raise ValueError("need flat16K tokens and context capacity")
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
                ("gathered", attention.attention),
            ):
                model_module.attention = function
                logits, cache = model.prefill(mx.array(values[:8][None]))
                model.decode(mx.argmax(logits[:, -1], axis=-1)[:, None], cache)
                del logits, cache
            mx.synchronize()
            mx.clear_cache()
            for length in (4096, 16384):
                row, results = {"context": length, "arms": {}}, []
                for label, function in (
                    ("reference", reference.attention),
                    ("gathered", attention.attention),
                ):
                    model_module.attention = function
                    prompt = mx.array(values[:length][None])
                    mx.eval(prompt)
                    mx.synchronize()
                    mx.reset_peak_memory()
                    start = time.perf_counter()
                    logits, cache = model.prefill(prompt)
                    mx.synchronize()
                    prefill = time.perf_counter() - start
                    generated = []
                    start = time.perf_counter()
                    for _ in range(16):
                        token = mx.argmax(logits[:, -1], axis=-1)[:, None]
                        generated.append(token)
                        logits = model.decode(token, cache)
                    mx.eval(logits, generated)
                    mx.synchronize()
                    elapsed = time.perf_counter() - start
                    assert cache.length == length + 16 and bool(
                        mx.all(mx.isfinite(logits)).item()
                    )
                    output = mx.concatenate(generated, axis=1).tolist()[0]
                    results.append((output, logits))
                    row["arms"][label] = {
                        "prefill_seconds": prefill,
                        "prefill_tokens_per_second": length / prefill,
                        "decode_seconds": elapsed,
                        "decode_tokens_per_second": 16 / elapsed,
                        "peak_memory_bytes": mx.get_peak_memory(),
                        "kv_bytes": cache.resident_bytes(),
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
        report["completed"] = True
        save()


if __name__ == "__main__":
    main()
