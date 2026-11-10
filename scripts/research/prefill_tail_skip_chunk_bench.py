#!/usr/bin/env python3
"""In-process per-chunk A/B for MLX2_PREFILL_TAIL_SKIP (lower noise than TTFT).

Loads one artifact through its mlx2 adapter, prefills a fixed context offset
once (tail skip off), then times one prompt chunk of --chunk rows at that
offset on a fresh deep copy of the cache, alternating modes per rep in a
counterbalanced order (ABBA...).  Each timed forward runs inside a
checkpoint-row request exactly as the ordinary prompt batch opens it, and
evaluates the logits and every cache state.  Also records the checkpoint
row's argmax per mode (must agree) and the tail-skip counters.

Usage (under the shared GPU queue):
  PYTHONPATH=$PWD/src python scripts/research/prefill_tail_skip_chunk_bench.py \
      --model ~/mlx-models/<model> --modes 0,1 --out <dir>/chunk.json --i-own-the-gpu
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _swapouts() -> int:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False).stdout
    return int([line for line in out.splitlines() if "Swapouts" in line][0].split()[-1].rstrip("."))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--modes", default="0,1")
    ap.add_argument("--offsets", default="0,16384,30720")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")
    out_path = Path(a.out)
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite {out_path}")

    import mlx.core as mx

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime import prefill_tail_skip as pts
    from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows

    adapter = resolve_adapter(a.model, mtp=False)(a.model)
    model = adapter.model
    mx.eval(model.parameters())
    mx.set_cache_limit(4 << 30)
    swap0 = _swapouts()
    tok = adapter.tokenizer
    docs = sorted((ROOT / "docs").rglob("*.md"))
    text, ids = [], []
    need = max(int(o) for o in a.offsets.split(",")) + a.chunk
    for path in docs:
        text.append(path.read_text(errors="replace"))
        if len(text) % 20 == 0:
            ids = tok.encode("\n\n".join(text), add_special_tokens=False)
            if len(ids) >= need:
                break
    ids = tok.encode("\n\n".join(text), add_special_tokens=False)[:need]
    assert len(ids) == need, len(ids)
    modes = a.modes.split(",")

    def forward(tokens, cache, mode):
        os.environ[pts.ENV] = mode
        with prefill_checkpoint_rows([tokens.shape[1]], width=tokens.shape[1]) as request:
            logits = model(tokens, cache=cache)
            mx.eval(logits, [c.state for c in cache])
        return logits, request.applied

    result = {"model": a.model, "mlx": mx.__version__, "modes": modes, "chunk": a.chunk,
              "reps": a.reps, "offsets": {}}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=False).stdout.strip()
    result["source_revision"] = revision
    for offset in (int(o) for o in a.offsets.split(",")):
        base = model.make_cache()
        for start in range(0, offset, a.chunk):
            forward(mx.array([ids[start:start + a.chunk]], mx.int32), base, "0")
        chunk = mx.array([ids[offset:offset + a.chunk]], mx.int32)
        times = {m: [] for m in modes}
        argmax = {}
        # Warm each mode once (kernel compilation), untimed.
        for m in modes:
            logits, _ = forward(chunk, copy.deepcopy(base), m)
            argmax[m] = int(mx.argmax(logits[0, -1]).item())
        for rep in range(a.reps):
            order = modes if rep % 2 == 0 else modes[::-1]
            for m in order:
                cache = copy.deepcopy(base)
                mx.eval([c.state for c in cache])
                t0 = time.perf_counter()
                _, applied = forward(chunk, cache, m)
                times[m].append(round(time.perf_counter() - t0, 5))
                del cache
        row = {"times_s": times, "median_s": {m: statistics.median(v) for m, v in times.items()},
               "checkpoint_argmax": argmax}
        off = row["median_s"][modes[0]]
        row["delta_vs_first_pct"] = {m: round(100 * (v / off - 1), 2) for m, v in row["median_s"].items()}
        result["offsets"][str(offset)] = row
        print(json.dumps({"offset": offset, **row}), flush=True)
        del base
        mx.clear_cache()
        if _swapouts() - swap0 > 20000:
            raise SystemExit("swapouts rose; aborting")
    result["tail_skip_counts"] = dict(pts.STATS)
    result["swapouts_delta"] = _swapouts() - swap0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
