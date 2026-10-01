"""One warm fixed-order prefill chunk comparison; no serving selection."""

import argparse
import hashlib
import json
from pathlib import Path
import time

from mlx2.experimental.hysparse2.resources import gpu_guard
from mlx2.experimental.hysparse2.train import file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokens", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384])
    p.add_argument("--chunks", type=int, nargs="+", default=[256, 128, 64])
    args = p.parse_args()
    if min(args.lengths + args.chunks) < 1:
        p.error("positive lengths/chunks required")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "mlx2.hysparse2-prefill-chunks.v1", "completed": False,
              "serving_route_qualified": False, "selected": False,
              "thermal_controls": False, "repetitions": 1,
              "arm_order": args.chunks, "cells": []}
    with gpu_guard(wait_seconds=0):
        import mlx.core as mx
        import numpy as np
        from mlx2.experimental.hysparse2.config import Config
        from mlx2.experimental.hysparse2.model import Model
        from mlx2.experimental.hysparse2.train import _load_model_state

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(12 << 30)
        mx.set_cache_limit(256 << 20)
        c = Config(**json.loads((args.checkpoint / "state.json").read_text())["config"])
        if args.chunks[0] != c.prefill_chunk or max(args.chunks) > c.prefill_chunk:
            raise ValueError("first arm must be configured chunk, others no larger")
        model = Model(c)
        _load_model_state(args.checkpoint, model)
        model.set_dtype(mx.bfloat16)
        model.eval()
        mx.eval(model.parameters())
        values = np.load(args.tokens, allow_pickle=False)
        if values.ndim != 1 or len(values) < max(args.lengths):
            raise ValueError("insufficient flat tokens")
        warm, cache = model.prefill(mx.array(values[:8][None]))
        model.decode(mx.argmax(warm[:, -1], axis=-1)[:, None], cache)
        mx.synchronize()
        del warm, cache
        for length in args.lengths:
            references = None
            for chunk in args.chunks:
                mx.clear_cache()
                tokens = mx.array(values[:length][None])
                mx.eval(tokens)
                mx.reset_peak_memory()
                started = time.perf_counter()
                if chunk == c.prefill_chunk:
                    logits, cache = model.prefill(tokens)
                else:
                    cache = model.new_cache()
                    for start in range(0, length, chunk):
                        end = min(length, start + chunk)
                        logits, cache = model.prefill(tokens[:, start:end], cache,
                                                     return_logits=end == length)
                mx.synchronize()
                prefill = time.perf_counter() - started
                outputs, logits_rows = [], [logits]
                started = time.perf_counter()
                for _ in range(4):
                    token = mx.argmax(logits[:, -1], axis=-1)[:, None]
                    outputs.append(int(token.item()))
                    logits = model.decode(token, cache)
                    logits_rows.append(logits)
                mx.synchronize()
                decode = time.perf_counter() - started
                joined = mx.concatenate(logits_rows, axis=1)
                mx.eval(joined)
                assert bool(mx.all(mx.isfinite(joined)).item())
                expected_kv = ((1 + c.cross_blocks) * (length + 4)
                               + (c.self_layers - 1) * min(length + 4, c.local_window)) * 2 * c.head_dim * 2
                assert cache.length == length + 4 and cache.resident_bytes() == expected_kv
                assert cache.self_layer_calls == c.self_layers * ((length + chunk - 1) // chunk + 4)
                assert cache.cross_layer_calls == c.cross_blocks * (1 + c.sparse_per_block) * 5
                peak = mx.get_peak_memory()
                if references is None:
                    references = joined, outputs
                error = float(mx.max(mx.abs(joined - references[0])).item())
                report["cells"].append({"context": length, "chunk": chunk,
                    "prefill_seconds": prefill, "prefill_tps": length / prefill,
                    "decode_tps": 4 / decode, "peak_gib": peak / (1 << 30),
                    "logit_error": error, "tokens_equal": outputs == references[1],
                    "kv_bytes": expected_kv, "self_calls": cache.self_layer_calls,
                    "cross_calls": cache.cross_layer_calls, "finite": True})
                print(json.dumps(report["cells"][-1]), flush=True)
                del cache, logits, logits_rows, tokens, joined
            del references
        report["parameters"] = c.capacity()["parameters"]
    root = Path(__file__).resolve().parents[1]
    report["checkpoint_sha256"] = file_hash(args.checkpoint / "model.safetensors")
    report["tokens_sha256"] = file_hash(args.tokens)
    report["source_sha256"] = {str(path.relative_to(root)): file_hash(path)
        for path in (Path(__file__).resolve(), root / "src/mlx2/experimental/hysparse2/model.py",
                     root / "src/mlx2/experimental/hysparse2/attention.py")}
    report["completed"] = True
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
