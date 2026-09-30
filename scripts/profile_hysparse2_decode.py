"""Synchronized first-anchor attribution; probe overhead is not decode speed."""

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
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference-attention", type=Path)
    p.add_argument(
        "--warm-paired",
        action="store_true",
        help="Warm each paired attention and full-cross path before timing",
    )
    p.add_argument("--lengths", type=int, nargs="+", default=[256, 1024, 4096, 16384])
    args = p.parse_args()
    if args.warm_paired and not args.reference_attention:
        p.error("--warm-paired requires --reference-attention")
    if args.output.exists():
        p.error("use a fresh receipt path")
    state = json.loads((args.checkpoint / "state.json").read_text())
    c = Config(**state["config"])
    report = {
        "schema": "mlx2.hysparse2-decode-attribution.v1",
        "contexts": [],
        "checkpoint_sha256": file_hash(args.checkpoint / "model.safetensors"),
        "target": "first cross-attention anchor only",
        "repetitions": 1,
        "completed": False,
        "production_throughput_measurement": False,
        "paired_paths_warmed": args.warm_paired,
        "script_sha256": file_hash(Path(__file__)),
        "attention_source_sha256": file_hash(
            Path("src/mlx2/experimental/hysparse2/attention.py")
        ),
    }
    reference_attention = None
    if args.reference_attention:
        spec = importlib.util.spec_from_file_location(
            "attention_reference", args.reference_attention
        )
        reference_attention = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference_attention)
        report["reference_attention_sha256"] = file_hash(args.reference_attention)
    with gpu_guard(wait_seconds=0):
        import mlx.core as mx
        import numpy as np

        from mlx2.experimental.hysparse2 import model as model_module
        from mlx2.experimental.hysparse2.attention import attention
        from mlx2.experimental.hysparse2.model import Model

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(24 << 30)
        mx.set_cache_limit(256 << 20)
        model = Model(c)
        _load_model_state(args.checkpoint, model)
        model.set_dtype(mx.bfloat16)
        model.eval()
        mx.eval(model.parameters())
        values = np.load(args.tokens, allow_pickle=False)

        def timed(fn, *positional, **keyword):
            mx.synchronize()
            start = time.perf_counter()
            result = fn(*positional, **keyword)
            mx.eval(result)
            mx.synchronize()
            return result, time.perf_counter() - start

        for length in args.lengths:
            if length + 1 > len(values) or length + 1 > c.max_context:
                raise ValueError("context exceeds available tokens/capacity")
            _, cache = model.prefill(
                mx.array(values[:length][None]), return_logits=False
            )
            token = mx.array(values[length : length + 1][None])
            mx.eval(token)
            (hidden, offset), append_seconds = timed(model._append, token, cache)
            layer = model.cross_decoder[0]
            raw, _ = layer.attention_hc.read(hidden)
            normalized = layer.attention_norm(raw)
            q = (
                layer.attention.q(normalized)
                .reshape(1, 1, c.num_heads, c.head_dim)
                .transpose(0, 2, 1, 3)
            )
            mx.eval(q)
            blocks = cache.cross_kv[0]
            common = {
                "offset": offset,
                "query_tile": c.query_tile,
                "key_tile": c.key_tile,
            }
            dense, dense_seconds = timed(attention, q, blocks, **common)
            oracle, oracle_seconds = timed(
                attention, q, blocks, **common, select=(c.local_window, c.global_tokens)
            )
            if args.warm_paired:
                for function in (attention, reference_attention.attention):
                    timed(
                        function,
                        q,
                        blocks,
                        **common,
                        select=(c.local_window, c.global_tokens),
                        block_select=(c.candidate_block_size, c.candidate_blocks),
                    )
            selected, block_seconds = timed(
                attention,
                q,
                blocks,
                **common,
                select=(c.local_window, c.global_tokens),
                block_select=(c.candidate_block_size, c.candidate_blocks),
            )
            error = float(mx.max(mx.abs(dense[0] - selected[0])).item())
            if error != 0:
                raise AssertionError("candidate selection changed the dense anchor")
            row = {
                "context": length,
                "cache_segments": len(blocks),
                "self_append_seconds": append_seconds,
                "dense_anchor_seconds": dense_seconds,
                "dense_plus_token_selection_seconds": oracle_seconds,
                "dense_plus_block_selection_seconds": block_seconds,
                "dense_anchor_max_abs_error": error,
                "support_slots": selected[1][2].shape[-1],
                "valid_support_slots": int(mx.sum(selected[1][2] <= offset).item()),
                "finite_anchor": bool(mx.all(mx.isfinite(selected[0])).item()),
            }
            if reference_attention is not None:
                original, original_seconds = timed(
                    reference_attention.attention,
                    q,
                    blocks,
                    **common,
                    select=(c.local_window, c.global_tokens),
                    block_select=(c.candidate_block_size, c.candidate_blocks),
                )
                support_equal = bool(mx.all(original[1][2] == selected[1][2]).item())
                support_set_equal = bool(
                    mx.all(
                        mx.sort(original[1][2], axis=-1)
                        == mx.sort(selected[1][2], axis=-1)
                    ).item()
                )
                original_error = float(mx.max(mx.abs(original[0] - selected[0])).item())
                row["paired_reference_block_seconds"] = original_seconds
                row["paired_reference_support_equal"] = support_equal
                row["paired_reference_support_set_equal"] = support_set_equal
                row["paired_reference_anchor_max_abs_error"] = original_error
                if args.warm_paired:
                    timed(model._cross, hidden, cache, offset)
                    try:
                        model_module.attention = reference_attention.attention
                        timed(model._cross, hidden, cache, offset)
                    finally:
                        model_module.attention = attention
                optimized_logits, optimized_cross_seconds = timed(
                    model._cross, hidden, cache, offset
                )
                try:
                    model_module.attention = reference_attention.attention
                    original_logits, original_cross_seconds = timed(
                        model._cross, hidden, cache, offset
                    )
                finally:
                    model_module.attention = attention
                logit_error = float(
                    mx.max(mx.abs(original_logits - optimized_logits)).item()
                )
                row["paired_reference_full_logits_max_abs_error"] = logit_error
                row["optimized_full_cross_seconds"] = optimized_cross_seconds
                row["reference_full_cross_seconds"] = original_cross_seconds
                if not support_set_equal or original_error != 0 or logit_error != 0:
                    report["failure"] = row
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps(row), flush=True)
                    raise AssertionError("optimized support set or full logits differ")
            report["contexts"].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
            del cache, hidden, q, blocks, dense, oracle, selected, raw, normalized
            mx.clear_cache()
        report["completed"] = True
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
