#!/usr/bin/env python3
"""Research-only direct attention comparator; default invocation is CPU dry-run.

An explicit --run is required to load MLX and touch the GPU. Results are direct
attention timings, never a model-path or serving qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import subprocess
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "omlx3958_wide_sdpa.py"
DEFAULT_CONFIG = Path("~/mlx-models/Qwen3.8-27B-oQ4e-mtp/config.json")
_SPEC = importlib.util.spec_from_file_location("omlx3958_wide_sdpa", MODULE_PATH)
WIDE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(WIDE)


def plan(config: dict, *, prefix: int, widths: tuple[int, ...], dtype: str) -> list[dict]:
    text = config.get("text_config", config)
    geometry = (text.get("num_attention_heads"), text.get("num_key_value_heads"),
                text.get("head_dim"))
    if geometry != (24, 4, 256):
        raise ValueError(f"artifact is not the studied 24/4/D256 geometry: {geometry}")
    if prefix < 0:
        raise ValueError("prefix must be nonnegative")
    rows = []
    for m in widths:
        t = prefix + m
        reasons = WIDE.admission_reasons(
            (1, 24, m, 256), (1, 4, t, 256), (1, 4, t, 256),
            q_dtype=dtype, k_dtype=dtype, v_dtype=dtype,
            cache_type="KVCache", cache_offset=t, mask="causal",
        )
        rows.append({"prefix": prefix, "verify_width": m, "kv_length": t,
                     "eligible": not reasons, "reasons": reasons,
                     "matrix_rows": 8, "gqa": 6})
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                   cwd=HERE.parents[1], text=True).strip()


def _run(rows: list[dict], *, iterations: int, repetitions: int,
         dtype_name: str) -> list[dict]:
    import mlx.core as mx
    import numpy as np
    from mlx2.runtime.models.cache import KVCache

    if iterations < 1 or repetitions < 1:
        raise ValueError("iterations and repetitions must be positive")
    dtype = {"bfloat16": mx.bfloat16, "float16": mx.float16}[dtype_name]
    results = []
    for row in rows:
        if not row["eligible"]:
            raise ValueError(f"ineligible row: {row}")
        p, m = row["prefix"], row["verify_width"]
        cache = KVCache()
        if p:
            pk = mx.random.normal((1, 4, p, 256)).astype(dtype)
            pv = mx.random.normal((1, 4, p, 256)).astype(dtype)
            cache.update_and_fetch(pk, pv)
        q = mx.random.normal((1, 24, m, 256)).astype(dtype)
        new_k = mx.random.normal((1, 4, m, 256)).astype(dtype)
        new_v = mx.random.normal((1, 4, m, 256)).astype(dtype)
        k, v = cache.update_and_fetch(new_k, new_v)
        assert cache.offset == p + m
        scale = 256 ** -0.5

        def stock():
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale,
                                                         mask="causal")

        def wide():
            return WIDE.wide_attention(q, k, v, cache=cache, scale=scale,
                                       mask="causal")

        expected, actual = stock(), wide()
        mx.eval(expected, actual)
        delta = np.abs(np.asarray(expected.astype(mx.float32)) -
                       np.asarray(actual.astype(mx.float32)))
        max_abs = float(delta.max())
        if not np.isfinite(max_abs) or max_abs > 0.05:
            raise AssertionError(f"M{m} numeric parity failed: max_abs={max_abs}")
        if cache.offset != p + m:
            raise AssertionError("candidate modified cache offset")

        # A rejected verify suffix is trimmed by the existing cache contract;
        # replaying the same rows must recover the same answer.
        assert cache.trim(m) == m and cache.offset == p
        k, v = cache.update_and_fetch(new_k, new_v)
        replay = wide()
        mx.eval(replay)
        replay_delta = float(np.abs(np.asarray(actual.astype(mx.float32)) -
                                    np.asarray(replay.astype(mx.float32))).max())
        if replay_delta:
            raise AssertionError(f"M{m} rollback/replay mismatch: {replay_delta}")

        def median_ms(fn):
            for _ in range(3):
                mx.eval(fn())
            samples = []
            for _ in range(iterations):
                start = time.perf_counter_ns()
                mx.eval(fn())
                samples.append((time.perf_counter_ns() - start) / 1e6)
            return statistics.median(samples)

        paired = []
        for rep in range(repetitions):
            order = (("stock", stock), ("wide", wide)) if rep % 2 == 0 else (
                ("wide", wide), ("stock", stock))
            sample = {name + "_ms": median_ms(fn) for name, fn in order}
            sample["stock_over_wide"] = sample["stock_ms"] / sample["wide_ms"]
            paired.append(sample)
        stock_ms = statistics.median(x["stock_ms"] for x in paired)
        wide_ms = statistics.median(x["wide_ms"] for x in paired)
        results.append({**row, "stock_ms": stock_ms, "wide_ms": wide_ms,
                        "stock_over_wide": stock_ms / wide_ms,
                        "paired_repetitions": paired,
                        "max_abs": max_abs, "replay_max_abs": replay_delta,
                        "scope": "direct_attention_only"})
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prefix", type=int, default=8192)
    parser.add_argument("--widths", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--run", action="store_true", help="explicitly compile and time on GPU")
    parser.add_argument("--output", type=Path, help="optional JSON receipt path")
    args = parser.parse_args(argv)
    config = json.loads(args.model_config.read_text())
    rows = plan(config, prefix=args.prefix, widths=tuple(args.widths), dtype=args.dtype)
    receipt = {"mode": "gpu_direct_attention" if args.run else "cpu_dry_run",
               "source_head": _git_head(),
               "candidate_sha256": _sha256(MODULE_PATH),
               "model_config": str(args.model_config),
               "model_config_sha256": _sha256(args.model_config),
               "rows": rows}
    if args.run:
        receipt["rows"] = _run(rows, iterations=args.iterations,
                               repetitions=args.repetitions, dtype_name=args.dtype)
    rendered = json.dumps(receipt, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
