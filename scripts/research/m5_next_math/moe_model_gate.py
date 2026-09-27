#!/usr/bin/env python3
"""Research-only full-model M=3 Flash-Next MoE tile gate.

Run each arm in a fresh process under cpg_job + owned_exec. The default
--describe action is CPU/static only. This script does not admit or select a
production route; it temporarily changes Python module functions in this
process so the existing model call graph can exercise the explicit M=3 width.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import moe_tile_bench as tile_probe

MODEL_DEFAULT = Path("~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP")
SOURCE = ROOT / "src/mlx2/runtime/models/qwen4_fused_moe.py"
SOURCE_PATHS = {
    "fused_moe": SOURCE,
    "qwen3_next": ROOT / "src/mlx2/runtime/models/qwen3_next.py",
    "switch_layers": ROOT / "src/mlx2/runtime/models/switch_layers.py",
    "cache": ROOT / "src/mlx2/runtime/models/cache.py",
    "qwen4_exp": ROOT / "src/mlx2/runtime/models/qwen4_exp.py",
    "qwen3_5": ROOT / "src/mlx2/runtime/models/qwen3_5.py",
    "qwen4_ple_nvme": ROOT / "src/mlx2/runtime/models/qwen4_ple_nvme.py",
    "flash_next_adapter": ROOT / "src/mlx2/adapters/flash_next.py",
    "flash_next_policy": ROOT / "src/mlx2/adapters/flash_next_policy.py",
    "model_gate": Path(__file__).resolve(),
    "tile_source_generator": Path(tile_probe.__file__).resolve(),
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _width(hidden) -> int:
    result = 1
    for extent in hidden.shape[:-2]:
        result *= int(extent)
    return result


def _source_hashes() -> dict[str, str]:
    return {name: _hash(path) for name, path in SOURCE_PATHS.items()}


def _cache_position(entry) -> int | tuple[int, ...]:
    # Recurrent caches use a speculation-relative rollback cursor rather than
    # the KV cache's absolute size(). Both must return to their own pre-block
    # position after the three-token verify is trimmed.
    if hasattr(entry, "_rollback_position"):
        positions = entry._rollback_positions
        return tuple(int(value) for value in positions) if positions is not None else int(entry._rollback_position)
    return int(entry.size())


def _run(args) -> dict:
    owner = tile_probe._require_ownership()
    source_hashes = _source_hashes()
    sources = tile_probe._sources()
    if not args.model.is_dir() or not (args.model / "config.json").is_file():
        raise RuntimeError(f"model artifact not found: {args.model}")

    from mlx2.adapters.flash_next import FlashNextAdapter, configure_environment
    profile = configure_environment(args.model.resolve())
    import mlx.core as mx
    import numpy as np
    from mlx2.runtime.models import qwen4_fused_moe as fused
    from mlx2.runtime.models import switch_layers
    from mlx2.runtime.models.cache import make_prompt_cache

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("Metal GPU unavailable")
    if fused._DOWN_REDUCE_TILE4_SOURCE != sources[4]:
        raise RuntimeError("runtime tile4 source differs from source-bound literal")

    original_sort_threshold = switch_layers._GATHER_SORT_MIN_ASSIGNMENTS
    if args.arm != "stock":
        # The present M=3 candidate is unreachable with the current threshold:
        # 3 tokens x top-10 = 30 assignments, which would select sorted gather
        # and make _try_qwen4_fused_down return None before our hook is called.
        switch_layers._GATHER_SORT_MIN_ASSIGNMENTS = max(31, original_sort_threshold)
    adapter = FlashNextAdapter(str(args.model))
    if adapter.environment != profile:
        raise RuntimeError("adapter profile differs from pre-import execution profile")
    originals = (fused.admit_qwen4_fused_down, fused.qwen4_fused_down)
    kernel8 = None
    hook_calls = {"tile4": 0, "tile8": 0}
    try:
        model = adapter.model
        moe_modules = [
            module for _, module in model.named_modules()
            if hasattr(module, "fused_expert_dispatches")
        ]
        if not moe_modules:
            raise RuntimeError("no Qwen4 MoE modules found in loaded model")
        for module in moe_modules:
            module.set_moe_router_mode("stock")
            module.set_fused_expert_kernel_mode(
                "stock" if args.arm == "stock" else "tile4"
            )

        if args.arm != "stock":
            def candidate_admission(*positional, **keywords):
                if _width(positional[0]) == 3 and not keywords.get("candidate_token_widths"):
                    keywords["candidate_token_widths"] = (3,)
                return originals[0](*positional, **keywords)

            fused.admit_qwen4_fused_down = candidate_admission
            if args.arm == "tile8":
                kernel8 = mx.fast.metal_kernel(
                    name="research_model_qwen4_q4_down_tile8_v1",
                    input_names=[
                        "hidden", "down_weight", "down_scales", "down_biases",
                        "indices", "scores",
                    ],
                    output_names=["out"],
                    source=sources[8],
                    ensure_row_contiguous=False,
                )

            def candidate_down(
                hidden, indices, scores, weight, scales, biases, **keywords
            ):
                tokens = _width(hidden)
                if tokens != 3 or keywords.get("variant") != "tile4":
                    return originals[1](
                        hidden, indices, scores, weight, scales, biases, **keywords
                    )
                admission = candidate_admission(
                    hidden, indices, scores, weight, scales, biases,
                    num_experts=keywords.get("num_experts", fused.NUM_EXPERTS),
                    group_size=keywords.get("group_size", fused.GROUP_SIZE),
                    bits=keywords.get("bits", fused.BITS),
                    mode=keywords.get("mode", "affine"),
                )
                if not admission.accepted:
                    raise RuntimeError(f"M=3 candidate admission failed: {admission.reason}")
                hook_calls[args.arm] += 1
                if args.arm == "tile4":
                    return originals[1](
                        hidden, indices, scores, weight, scales, biases,
                        candidate_token_widths=(3,), **keywords
                    )
                output = kernel8(
                    inputs=[hidden, weight, scales, biases, indices, scores],
                    template=[("T", hidden.dtype), ("W", scales.dtype)],
                    grid=(160, fused.HIDDEN_SIZE // 8, tokens),
                    threadgroup=(160, 1, 1),
                    output_shapes=[(tokens, fused.HIDDEN_SIZE)],
                    output_dtypes=[hidden.dtype],
                )[0]
                return output.reshape(hidden.shape[:-2] + (fused.HIDDEN_SIZE,))

            fused.qwen4_fused_down = candidate_down

        tokenizer = adapter.tokenizer
        prompt = list(tokenizer.encode(args.prompt, add_special_tokens=False))
        block = list(tokenizer.encode(args.block, add_special_tokens=False))[:3]
        if len(prompt) < 4 or len(block) != 3:
            raise RuntimeError("prompt must have >=4 tokens and block >=3 tokens")

        args.out.parent.mkdir(parents=True, exist_ok=True)

        def one_run(round_index: int, *, retain: bool = True) -> dict:
            cache = list(make_prompt_cache(model))
            logits = model(mx.array([prompt], dtype=mx.uint32), cache=cache)
            mx.eval(logits)
            started_entries = []
            try:
                for entry in cache:
                    entry.start_speculation()
                    started_entries.append(entry)
                positions_before = [_cache_position(entry) for entry in cache]
                before_hook = dict(hook_calls)
                before_dispatch = sum(
                    int(module.fused_expert_dispatches.get("tile4", 0))
                    for module in moe_modules
                )
                started = time.perf_counter_ns()
                output = model(mx.array([block], dtype=mx.uint32), cache=cache)
                mx.eval(output)
                elapsed_ns = time.perf_counter_ns() - started
                array = np.asarray(output.astype(mx.float32))
                if not np.isfinite(array).all():
                    raise RuntimeError("nonfinite full-model M=3 logits")
                dispatch_delta = sum(
                    int(module.fused_expert_dispatches.get("tile4", 0))
                    for module in moe_modules
                ) - before_dispatch
                hook_delta = hook_calls[args.arm] - before_hook[args.arm] if args.arm != "stock" else 0
                if args.arm == "stock" and dispatch_delta != 0:
                    raise RuntimeError("stock arm unexpectedly engaged fused tile4")
                if args.arm != "stock" and (hook_delta < 1 or dispatch_delta < 1):
                    raise RuntimeError(
                        f"candidate did not engage: hook={hook_delta}, dispatch={dispatch_delta}"
                    )
                output_path = None
                if retain:
                    output_path = args.out.with_name(
                        f"{args.out.stem}-round{round_index}-logits.npy"
                    )
                    np.save(output_path, array, allow_pickle=False)
                trimmed = [entry.trim(len(block)) for entry in cache]
                positions_after = [_cache_position(entry) for entry in cache]
                if any(count != len(block) for count in trimmed):
                    raise RuntimeError(f"incomplete speculative trim: {trimmed}")
                if positions_after != positions_before:
                    raise RuntimeError("speculative trim did not restore cache positions")
            finally:
                for entry in reversed(started_entries):
                    entry.stop_speculation()
            return {
                "elapsed_ns": elapsed_ns,
                "logits_path": str(output_path) if output_path else None,
                "logits_sha256": _hash(output_path) if output_path else None,
                "logits_shape": list(array.shape),
                "hook_calls": hook_delta,
                "tile4_dispatches": dispatch_delta,
                "trimmed_entry_count": len(trimmed),
                "cache_entry_count": len(cache),
                "cache_positions_restored": True,
            }

        for index in range(args.warmups):
            one_run(-(index + 1), retain=False)
        rows = [one_run(index) for index in range(args.rounds)]
        if _source_hashes() != source_hashes:
            raise RuntimeError("path source changed during model gate")
        report = {
            "schema": "mlx2.m5-next-math.qwen4-m3-model-gate.v1",
            "status": "isolated_model_candidate_unqualified",
            "completed": True,
            "arm": args.arm,
            "model": str(args.model.resolve()),
            "model_config_sha256": _hash(args.model / "config.json"),
            "model_identity": adapter.identity,
            "repo_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "production_source_sha256": source_hashes["fused_moe"],
            "path_source_sha256": source_hashes,
            "execution_profile": adapter.environment,
            "tile8_source_sha256": hashlib.sha256(sources[8].encode()).hexdigest(),
            "owner": owner,
            "prompt_token_count": len(prompt),
            "prompt_token_sha256": hashlib.sha256(
                json.dumps(prompt, separators=(",", ":")).encode()
            ).hexdigest(),
            "block_token_ids": block,
            "moe_module_count": len(moe_modules),
            "router_mode": "stock",
            "sort_threshold_original": original_sort_threshold,
            "sort_threshold_used": switch_layers._GATHER_SORT_MIN_ASSIGNMENTS,
            "warmups": args.warmups,
            "rounds": rows,
            "median_verify_ms": statistics.median(
                row["elapsed_ns"] for row in rows
            ) / 1e6,
        }
        args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
        return report
    finally:
        fused.admit_qwen4_fused_down, fused.qwen4_fused_down = originals
        switch_layers._GATHER_SORT_MIN_ASSIGNMENTS = original_sort_threshold
        adapter.close()
        mx.clear_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    parser.add_argument("--arm", choices=("stock", "tile4", "tile8"), default="tile8")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--prompt",
        default="Explain why exact cache state matters during speculative decoding.",
    )
    parser.add_argument("--block", default="The answer follows from exact state.")
    args = parser.parse_args()
    if not args.run_gpu:
        print(json.dumps({
            "schema": "mlx2.m5-next-math.qwen4-m3-model-gate.v1",
            "status": "static_only",
            "model": str(args.model),
            "arm": args.arm,
            "production_source_sha256": _hash(SOURCE),
            "tile8_source_sha256": hashlib.sha256(
                tile_probe._sources()[8].encode()
            ).hexdigest(),
        }, indent=2))
        return 0
    if args.out is None or args.warmups < 0 or args.rounds < 1:
        parser.error("--run-gpu requires --out, warmups >=0, and rounds >=1")
    report = _run(args)
    print(json.dumps({
        "out": str(args.out), "arm": args.arm,
        "median_verify_ms": report["median_verify_ms"],
        "moe_module_count": report["moe_module_count"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
