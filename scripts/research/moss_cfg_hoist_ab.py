#!/usr/bin/env python3
"""Byte-equality + timing A/B for the MOSS-Image CFG conditioning hoist (sglang#48688).

Loads the pipeline once.  The "recompute" arm wraps the transformer in a
proxy without ``conditioning`` that drops the reuse keywords, i.e. the
pre-hoist per-call recomputation; the "hoist" arm is the pipeline as is.
Counterbalanced (recompute, hoist, hoist, recompute); images must be
byte-identical.  Writes JSON + PNG hashes to --out (refuses a nonempty dir).
"""

from __future__ import annotations

import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src

import argparse
import hashlib
import io
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx

from mlx2.adapters.moss_image_pipeline import MOSSImagePipeline


class _Recompute:
    def __init__(self, transformer):
        self._transformer = transformer

    def __call__(self, *args, conditioning=None, context_memo=None, **kwargs):
        return self._transformer(*args, **kwargs)


def _swapouts() -> int:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    return int([line for line in out.splitlines() if "Swapouts" in line][0].split()[-1].rstrip("."))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty evidence dir {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    pipeline = MOSSImagePipeline(args.model)
    hoisted = pipeline.transformer
    recompute = _Recompute(hoisted)
    prompt = "A red ceramic teapot on a wooden table beside a window, morning light"
    rows, sw0 = [], _swapouts()
    for arm in ("recompute", "hoist", "hoist", "recompute"):
        pipeline.transformer = recompute if arm == "recompute" else hoisted
        os.environ["MLX2_MOSS_CFG_HOIST"] = "0" if arm == "recompute" else "1"
        mx.synchronize()
        start = time.perf_counter()
        image = pipeline.generate(prompt, width=args.size, height=args.size, steps=args.steps,
                                  seed=1234, guidance_scale=3.0)
        mx.synchronize()
        seconds = time.perf_counter() - start
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        rows.append({"arm": arm, "seconds": round(seconds, 3),
                     "pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest()})
        print(json.dumps(rows[-1]), flush=True)
    pipeline.transformer = hoisted
    result = {
        "model": str(args.model), "size": args.size, "steps": args.steps, "rows": rows,
        "byte_identical": len({row["pixels_sha256"] for row in rows}) == 1,
        "recompute_median_s": statistics.median(r["seconds"] for r in rows if r["arm"] == "recompute"),
        "hoist_median_s": statistics.median(r["seconds"] for r in rows if r["arm"] == "hoist"),
        "swapouts_delta": _swapouts() - sw0,
        "source_revision": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                          text=True).stdout.strip(),
    }
    (args.out / "results.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("byte_identical", "recompute_median_s", "hoist_median_s")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
