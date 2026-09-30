"""One-repetition synchronized research-model ladder; no thermal controls."""

import argparse
import json
import platform
import time
from pathlib import Path

from mlx2.experimental.hysparse2.config import Config
from mlx2.experimental.hysparse2.resources import gpu_guard
from mlx2.experimental.hysparse2.train import _load_model_state, file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokens", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-revision", required=True)
    p.add_argument(
        "--lengths", type=int, nargs="+", default=[256, 1024, 4096, 8192, 16384]
    )
    p.add_argument("--decode-tokens", type=int, default=32)
    p.add_argument("--memory-limit-gib", type=float, default=24)
    args = p.parse_args()
    if args.output.exists() or args.decode_tokens < 1 or min(args.lengths) < 1:
        p.error("use a new output and positive context/decode lengths")
    state = json.loads((args.checkpoint / "state.json").read_text())
    c = Config(**state["config"])
    if max(args.lengths) + args.decode_tokens > c.max_context:
        p.error("ladder exceeds context capacity")
    receipt = {
        "schema": "mlx2.hysparse2-context-benchmark.v1",
        "source_revision": args.source_revision,
        "source_files_sha256": {
            str(path): file_hash(path)
            for path in [
                Path(__file__),
                Path("src/mlx2/experimental/hysparse2/model.py"),
                Path("src/mlx2/experimental/hysparse2/config.py"),
                Path("src/mlx2/experimental/hysparse2/train.py"),
            ]
        },
        "checkpoint_sha256": file_hash(args.checkpoint / "model.safetensors"),
        "tokens_sha256": file_hash(args.tokens),
        "host": platform.node(),
        "config": c.as_dict(),
        "parameters": c.capacity()["parameters"],
        "repetitions": 1,
        "thermal_controls": False,
        "batch": 1,
        "dtype": "bfloat16",
        "warmup": "8 prompt tokens and one decode token",
        "decode_method": "autoregressive greedy, fixed token count, no EOS stop",
        "contexts": [],
        "completed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(receipt, indent=2) + "\n")
        tmp.replace(args.output)

    save()
    with gpu_guard(wait_seconds=0):
        import mlx.core as mx
        import numpy as np

        from mlx2.experimental.hysparse2.model import Model

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(int(args.memory_limit_gib * 2**30))
        mx.set_cache_limit(256 << 20)
        model = Model(c)
        _load_model_state(args.checkpoint, model)
        model.set_dtype(mx.bfloat16)
        model.eval()
        mx.eval(model.parameters())
        values = np.load(args.tokens, allow_pickle=False)
        if values.ndim != 1 or len(values) < max(args.lengths):
            raise ValueError("insufficient flat prompt tokens")
        prompt = mx.array(values[:8][None])
        logits, cache = model.prefill(prompt)
        model.decode(mx.argmax(logits[:, -1], axis=-1)[:, None], cache)
        mx.synchronize()
        del logits, cache, prompt
        mx.clear_cache()
        for n in args.lengths:
            prompt = mx.array(values[:n][None])
            mx.eval(prompt)
            mx.synchronize()
            mx.reset_peak_memory()
            start = time.perf_counter()
            logits, cache = model.prefill(prompt)
            mx.eval(logits)
            mx.synchronize()
            prefill = time.perf_counter() - start
            finite = bool(mx.all(mx.isfinite(logits)).item())
            start = time.perf_counter()
            for _ in range(args.decode_tokens):
                token = mx.argmax(logits[:, -1], axis=-1)[:, None]
                logits = model.decode(token, cache)
            mx.eval(logits)
            mx.synchronize()
            decode = time.perf_counter() - start
            finite = finite and bool(mx.all(mx.isfinite(logits)).item())
            row = {
                "context": n,
                "prefill_seconds": prefill,
                "prefill_tokens_per_second": n / prefill,
                "decode_tokens": args.decode_tokens,
                "decode_seconds": decode,
                "decode_tokens_per_second": args.decode_tokens / decode,
                "peak_memory_bytes": mx.get_peak_memory(),
                "kv_bytes": cache.resident_bytes(),
                "finite_logits": finite,
                "self_layer_calls": cache.self_layer_calls,
                "cross_layer_calls": cache.cross_layer_calls,
            }
            receipt["contexts"].append(row)
            save()
            print(json.dumps(row), flush=True)
            if not finite:
                raise FloatingPointError("nonfinite logits")
            del prompt, logits, cache
            mx.clear_cache()
        receipt["completed"] = True
        save()


if __name__ == "__main__":
    main()
