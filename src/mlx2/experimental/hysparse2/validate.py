"""Bounded GPU checkpoint/context validation; does not qualify learned retrieval."""

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

from .config import Config
from .resources import gpu_guard


def run(args):
    import mlx.core as mx
    import numpy as np

    from .model import Model

    state = json.loads((args.checkpoint / "state.json").read_text())
    c = Config(**state["config"])
    if max(args.lengths) >= c.max_context:
        raise ValueError("context ladder exceeds checkpoint capacity")
    metadata_path = args.tokens.parent / "receipt.json"
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text())
        if (
            meta.get("schema") == "mlx2.hysparse2-tokens.v1"
            and meta["tokenizer_sha256"] != state["run"]["tokenizer_sha256"]
        ):
            raise ValueError("context tokens use a different tokenizer")
    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(int(args.memory_limit_gib * 2**30))
    mx.set_cache_limit(1 << 30)
    model = Model(c)
    model.load_weights(str(args.checkpoint / "model.safetensors"), strict=True)
    model.eval()
    values = np.load(args.tokens, mmap_mode="r", allow_pickle=False)
    if (
        values.ndim != 1
        or values.dtype != np.uint32
        or len(values) < max(args.lengths) + 1
    ):
        raise ValueError("not enough uint32 tokens for requested context ladder")
    prefix = mx.array(values[:33][None])
    full = model(prefix)[0]
    last, cache = model.prefill(prefix[:, :32])
    decoded = model.decode(prefix[:, 32:], cache)
    mx.eval(full, last, decoded)
    prefill_error = float(mx.max(mx.abs(full[:, 31:32] - last)).item())
    decode_error = float(mx.max(mx.abs(full[:, 32:] - decoded)).item())
    receipt = {
        "schema": "mlx2.hysparse2-gpu-context.v1",
        "checkpoint_step": state["step"],
        "config": asdict(c),
        "fp32_prefill_max_abs_error": prefill_error,
        "fp32_decode_max_abs_error": decode_error,
        "dtype": args.dtype,
        "contexts": [],
        "learned_retrieval_qualified": False,
    }
    if max(prefill_error, decode_error) > 1e-3:
        raise RuntimeError(
            f"full vs cached FP32 parity failed: {prefill_error}, {decode_error}"
        )
    del full, last, decoded, cache, prefix
    if args.dtype == "bfloat16":
        model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    mx.clear_cache()
    for length in args.lengths:
        mx.reset_peak_memory()
        tokens = mx.array(values[:length][None])
        started = time.perf_counter()
        logits, cache = model.prefill(tokens)
        prefill_seconds = time.perf_counter() - started
        before = cache.resident_bytes()
        started = time.perf_counter()
        decoded = model.decode(mx.array(values[length : length + 1][None]), cache)
        decode_seconds = time.perf_counter() - started
        finite = bool(mx.all(mx.isfinite(logits)).item()) and bool(
            mx.all(mx.isfinite(decoded)).item()
        )
        entry = {
            "tokens": length,
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "kv_logical_bytes": before,
            "mlx_active_memory_bytes": mx.get_active_memory(),
            "mlx_peak_memory_bytes": mx.get_peak_memory(),
            "finite_logits": finite,
            "cross_layer_calls": cache.cross_layer_calls,
        }
        receipt["contexts"].append(entry)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(entry), flush=True)
        if not finite:
            raise FloatingPointError("nonfinite long-context logits")
        del logits, decoded, cache, tokens
        mx.clear_cache()
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokens", type=Path, required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=[256, 1024, 4096])
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    p.add_argument("--memory-limit-gib", type=float, default=48)
    p.add_argument("--wait-for-gpu", type=float, default=180)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if (
        not math.isfinite(args.memory_limit_gib)
        or args.memory_limit_gib <= 0
        or not math.isfinite(args.wait_for_gpu)
        or args.wait_for_gpu < 0
    ):
        p.error("invalid GPU memory or wait limit")
    if any(n < 1 or n >= 2097152 for n in args.lengths):
        p.error("contexts must be positive and leave space for a decode token")
    if args.output.exists():
        p.error("use a new receipt path")
    with gpu_guard(wait_seconds=args.wait_for_gpu):
        run(args)


if __name__ == "__main__":
    main()
