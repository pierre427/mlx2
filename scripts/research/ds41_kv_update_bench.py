"""Does a DS4.1 decode-step source-pool append copy the whole buffer?

Synthetic, no model. Builds the four DS4.1 KV-source caches with the real
cache classes and geometry (layers 2/8/14 ratio 2, layer 20 ratio 1;
DeepSeekV41Config defaults: head_dim 512, index_head_dim 128), prefilled to
``T`` tokens, then runs ``--steps`` decode tokens through the real append
methods (``append_pools`` at deepseek_v41_cache.py:123-147, or
``append_pools_packed`` at deepseek_v41_packed_cache.py:114-138). Each step
reads the appended planes the way decode does: a 512-row selected gather of
the pooled plane and a full-plane index score (select() reads every index key).

Modes:
  serial     eval per step (no in-flight reference to the old buffer)
  pipelined  async_eval step N+1 before waiting on N (GenerationBatch._step
             deferred validity, runtime/generate.py:1519-1690)
  bank_cow   serial, but each step runs on a descriptor_cow_v1 fork and is
             committed back, exactly the resident-bank transaction
             (adapters/deepseek_v41_source_bank_session.py:_fork/_commit);
             the owner keeps aliasing the old buffers while the step runs
  bank_copy  same with physical_copy_v1 (from_state copies all planes)

An in-place append is flat in T; a full-buffer copy grows linearly with T.
Steps on which any plane's 256-row capacity grows are reported separately
(``growth_*``) so capacity-growth spikes do not pollute the steady p50
(``p50_all_ms`` covers every step). Under bank_copy every step "grows":
from_state trims _capacity to the live count (deepseek_v41_cache.py:186), so
each fork's first append reallocates and copies the prefix again.
Research only; writes <out>/kv-update.json and refuses a nonempty out dir.
"""

from __future__ import annotations

import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src

import argparse
import json
import time
from pathlib import Path

SOURCE_LAYERS = ((2, 2), (8, 2), (14, 2), (20, 1))  # (layer, compress ratio)
HEAD_DIM, INDEX_DIM, INDEX_HEADS, TOPK = 512, 128, 32, 512


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[2048, 8192, 32768])
    parser.add_argument("--steps", type=int, default=320,
                        help="decode steps per cell; 320 crosses at least one ratio-1 256-row growth")
    parser.add_argument("--storage", choices=("reference", "native_packed"), default="reference")
    parser.add_argument("--modes", nargs="+", default=["serial", "pipelined", "bank_cow"],
                        choices=("serial", "pipelined", "bank_cow", "bank_copy"))
    parser.add_argument("--source-revision", default="unrecorded")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu",
                        help="cpu only for a tiny logic smoke; timings are meaningless there")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to reuse a nonempty evidence directory")
    args.out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx

    if args.device == "cpu":
        mx.set_default_device(mx.cpu)

    from mlx2.runtime.models.deepseek_v41_cache import DeepSeekV41LayerCache
    from mlx2.runtime.models.deepseek_v41_packed_cache import DeepSeekV41PackedLayerCache
    from mlx2.runtime.models.deepseek_v41_packed_qat import PackedQAT, decode

    packed = args.storage == "native_packed"
    cls = DeepSeekV41PackedLayerCache if packed else DeepSeekV41LayerCache

    def pool_rows(n):
        """n new pooled/index rows in the cache's storage format."""
        if not packed:
            return (mx.random.normal((1, n, HEAD_DIM)), mx.random.normal((1, n, INDEX_DIM)))
        def codes(shape):
            return mx.random.randint(0, 256, shape).astype(mx.uint8)
        # compressed_fp4: 2 codes/byte, e4m3 scale per 16; index_fp4: e8m0 per 32.
        # Scale bytes 120..127 keep decode finite (e4m3 ~0.5..1, e8m0 2^-7..2^0).
        def scales(shape):
            return mx.random.randint(120, 128, shape).astype(mx.uint8)
        return (PackedQAT(codes((1, n, HEAD_DIM // 2)), scales((1, n, HEAD_DIM // 16)), "compressed_fp4"),
                PackedQAT(codes((1, n, INDEX_DIM // 2)), scales((1, n, INDEX_DIM // 32)), "index_fp4"))

    def append(cache, rows):
        (cache.append_pools_packed if packed else cache.append_pools)(*rows)

    def local_rows(n):
        if not packed:
            return mx.random.normal((1, 1, n, HEAD_DIM))
        return PackedQAT(mx.random.randint(0, 120, (1, 1, n, HEAD_DIM)).astype(mx.uint8),
                         mx.random.randint(120, 128, (1, 1, n, HEAD_DIM // 32)).astype(mx.uint8), "fp8")

    def advance(cache, rows):
        """Model order: local band update (offset += n), partial group, then pools."""
        (cache.update_local_packed if packed else cache.update_local)(rows)
        rem = cache.offset % cache.ratio
        cache.buf_kv = cache.buf_gate = mx.zeros((1, rem, HEAD_DIM), mx.float32)

    def build(tokens):
        caches = []
        for layer, ratio in SOURCE_LAYERS:
            c = cls(ratio, xp=mx, head_dim=HEAD_DIM, index_head_dim=INDEX_DIM, window_size=128,
                    identity="ds41-kv-update-bench", source=True)
            advance(c, local_rows(tokens))
            append(c, pool_rows(tokens // ratio))
            caches.append(c)
        mx.eval([c._pooled for c in caches], [c._index_keys for c in caches])
        return caches

    def planes(c):
        if packed:
            return (c._pooled.codes, c._pooled.scales, c._index_keys.codes, c._index_keys.scales)
        return (c._pooled, c._index_keys)

    record = {"schema": "mlx2.ds41-kv-update-bench.v1", "source_revision": args.source_revision,
              "mlx": mx.__version__, "storage": args.storage, "device": args.device,
              "source_layers": [list(v) for v in SOURCE_LAYERS], "cells": []}
    q_index = mx.random.normal((INDEX_DIM, INDEX_HEADS))
    one_row = [pool_rows(1)]
    empty = pool_rows(0)
    one_local = local_rows(1)
    mx.eval(q_index, one_row, empty, one_local)

    for tokens in args.tokens:
        ids = mx.random.randint(0, tokens // 2, (1, 1, TOPK)).astype(mx.uint32)
        mx.eval(ids)
        for mode in args.modes:
            owner = build(tokens)

            def step(caches):
                outs = []
                for c in caches:
                    advance(c, one_local)
                    complete = c.offset // c.ratio - c._pool_count
                    # Ratio-2 layers complete a group every second token; the
                    # model still calls append with zero rows on the others.
                    append(c, one_row[0] if complete else empty)
                    if packed:
                        rows = c.gather_pooled(ids)
                        keys = decode(c.packed_index_keys)[0]
                    else:
                        rows = c.pooled[0, ids[0]]
                        keys = c.index_keys[0]
                    outs.append(rows.sum() + (keys @ q_index).max())
                return mx.stack(outs).sum()

            def capacity(caches):
                return tuple(c._capacity for c in caches)

            fork = commit = revision = None
            if mode.startswith("bank_") and packed:
                print(json.dumps({"tokens": tokens, "mode": mode, "skipped":
                                  "resident bank requires reference FP32 cache state"}), flush=True)
                continue
            if mode.startswith("bank_"):
                from mlx2.adapters.deepseek_v41_source_bank_session import _commit, _fork, _revision
                policy = "descriptor_cow_v1" if mode == "bank_cow" else "physical_copy_v1"
                fork = lambda rows, policy=policy: _fork(rows, policy)  # noqa: E731
                commit, revision = _commit, _revision

            gaps, peaks, grew, error = [], [], [], None
            try:
                if mode == "pipelined":
                    pending = step(owner)
                    mx.async_eval(pending)
                    last = time.perf_counter()
                    for _ in range(args.steps):
                        before = capacity(owner)
                        nxt = step(owner)
                        grew.append(capacity(owner) != before)
                        mx.async_eval(nxt)
                        pending.item()
                        now = time.perf_counter()
                        gaps.append(now - last)
                        last = now
                        pending = nxt
                    pending.item()
                else:
                    for i in range(args.steps + 1):
                        mx.reset_peak_memory()
                        active = mx.get_active_memory()
                        before = capacity(owner)
                        t0 = time.perf_counter()
                        if fork is None:
                            out = step(owner)
                            mx.eval(out, [planes(c) for c in owner])
                        else:
                            expected = revision(owner)
                            working = fork(owner)
                            out = step(working)
                            mx.eval(out, [s.state for s in working])
                            commit(owner, working, expected)
                        elapsed = time.perf_counter() - t0
                        if i == 0:
                            continue  # warm-up, matching the pipelined priming step
                        gaps.append(elapsed)
                        peaks.append(mx.get_peak_memory() - active)
                        grew.append(capacity(owner) != before)
            except Exception as exc:  # report and continue with the other cells
                error = f"{type(exc).__name__}: {exc}"

            steady = sorted(g for g, gr in zip(gaps, grew) if not gr)
            growth = sorted(g for g, gr in zip(gaps, grew) if gr)
            plane_bytes = sum(a.nbytes for c in owner for a in planes(c))
            cell = {"tokens": tokens, "mode": mode, "storage": args.storage, "steps": len(gaps),
                    "error": error,
                    "p50_all_ms": sorted(gaps)[len(gaps) // 2] * 1e3 if gaps else None,
                    "p50_ms": steady[len(steady) // 2] * 1e3 if steady else None,
                    "p90_ms": steady[int(len(steady) * 0.9)] * 1e3 if steady else None,
                    "steady_max_ms": steady[-1] * 1e3 if steady else None,
                    "growth_steps": len(growth),
                    "growth_max_ms": growth[-1] * 1e3 if growth else None,
                    "peak_above_active_mb_p50": (sorted(peaks)[len(peaks) // 2] / 2**20) if peaks else None,
                    "source_plane_mb_total": plane_bytes / 2**20,
                    "copy_mb_per_token_if_full": sum(
                        a.nbytes for c in owner for a in planes(c)
                        if c.ratio == 1) / 2**20 + 0.5 * sum(
                        a.nbytes for c in owner for a in planes(c) if c.ratio == 2) / 2**20}
            record["cells"].append(cell)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in cell.items()}), flush=True)
            del owner
            mx.clear_cache()
    (args.out / "kv-update.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
