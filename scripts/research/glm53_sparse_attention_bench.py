# SPDX-License-Identifier: MIT
"""GLM-5.3 fused gathered attention v2 bench (docs/design/glm53-speed-round2.md 1.1, 2.1, 7).

Subcommands (run through the GPU lease, e.g. the roofline campaign's bench.sh):

``pytest``   the Metal gates in tests/test_glm53_sparse_attention_v2.py with
             MLX2_RUN_GPU_TESTS=1 (float64 bounds per arm, NaN contract,
             padding inertness, row and block invariance, DIM_SPLIT bits).
``matrix``   per-layer attention time, cold (11 distinct layers chained by
             data dependency, as in a decode step), for ``gathered_v1`` (ops:
             gather + SDPA fallback), ``fused_decode_v1`` and every fused_v2
             bench point ``SPLIT in {32, 64} x DIM_SPLIT in {2, 4}``, at decode
             (one row) and for a 64-row prompt block. Synthetic operands.
``ttft``     prefill TTFT on the checkpoint, ABAB arms of model policy
             attributes; v2 prompt values route ``sparse_prefill`` to the
             fused_v2 law (the model call site is WP5's; this harness binds
             the law the value names). Each arm also decodes a natural greedy
             continuation with top-k logprob windows; the first divergence
             is classified under the AGENTS.md near-tie contract against the
             first arm, with the needle as functional oracle.
``captured`` R2 crossover: operands captured from a real prompt (q, latent,
             ids of one MLA layer per 128-query piece), dense piece
             (``_mla_masked``) vs the fused_v2 block per position band.

The shader profiler's occupancy/spill report is a manual Xcode capture and
is not produced here.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

LAYERS = 11


def host_record(mx):
    return {"host": platform.node(), "mlx": mx.__version__, "device": str(mx.default_device()),
            "device_info": {k: v for k, v in mx.device_info().items() if isinstance(v, (int, str))}}


def receipt(args, mx, **extra):
    """Receipt header; refuses a --source-revision that is not HEAD or a dirty src/
    (glm53_policy_ab.source_record, design 7.2)."""
    from glm53_policy_ab import source_record

    return {"schema": "mlx2.glm53-sparse-attention-v2-bench.v1", **source_record(args.source_revision),
            "command": args.command, "argv": sys.argv[1:], "host": host_record(mx), **extra}


# ---------------------------------------------------------------------------
# pytest

def run_pytest(args):
    import mlx.core as mx

    env = dict(os.environ, MLX2_RUN_GPU_TESTS="1")
    cmd = [sys.executable, "-m", "pytest", "-rA", "-s", "-p", "no:cacheprovider",
           f"--junitxml={args.out / 'junit.xml'}", "tests/test_glm53_sparse_attention_v2.py"]
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    (args.out / "pytest.log").write_text(proc.stdout + proc.stderr)
    print(proc.stdout[-6000:], proc.stderr[-2000:], flush=True)
    (args.out / "receipt.json").write_text(json.dumps(receipt(args, mx, returncode=proc.returncode), indent=1))
    return proc.returncode


# ---------------------------------------------------------------------------
# matrix

def decode_ids(rng, total, rows, width=2051):
    """Decode-shaped selections: 512 random visible pools x 4 rows, then the 3-row tail."""
    import numpy as np

    ids = np.full((rows, width), -1, np.int32)
    for r in range(rows):
        pos = total - rows + r
        pools = np.sort(rng.choice(pos // 4, size=min((width - 3) // 4, pos // 4), replace=False))
        row = (pools[:, None] * 4 + np.arange(4)).reshape(-1)
        ids[r, : len(row)] = row
        ids[r, len(row) : len(row) + 3] = np.arange(pos - 2, pos + 1)
    return ids


def build_layers(mx, rng, total, rows, dense):
    import numpy as np

    layers = []
    for _ in range(LAYERS):
        latent = mx.array(rng.normal(size=(1, 1, total, 512)).astype(np.float32))
        q = mx.array(rng.normal(size=(1, 64, rows, 512)).astype(np.float32))
        ids = None if dense else mx.array(decode_ids(rng, total, rows)[None])
        layers.append((q, latent, ids))
    mx.eval(layers)
    return layers


def arm_fn(name, mx, sa, v2):
    if name == "dense_sdpa":
        return lambda q, latent, ids: mx.fast.scaled_dot_product_attention(q, latent, latent, scale=sa.SCALE)
    if name == "gathered_v1":
        return lambda q, latent, ids: sa.gathered_attention(q, *sa.gather_latent(latent, ids))
    if name == "fused_decode_v1":
        return lambda q, latent, ids: sa.fused_decode_attention(q, latent, ids)
    split, dim_split = (int(x) for x in name.removeprefix("fused_v2@").split(","))

    def fused(q, latent, ids):
        flat, total = sa._v2_rows_ids(q, latent, ids)
        length = q.shape[2]
        meta = mx.array([[total, 0, 0, total - length + r] for r in range(length)], dtype=mx.int32)
        out = v2.checked_launch(q[0].transpose(1, 0, 2), latent[0, 0], mx.zeros((1, 512), mx.float32), flat, meta,
                                split=split, dim_split=dim_split)
        return out.transpose(1, 0, 2)[None]

    return fused


def chain(fn, layers, mx):
    out = None
    for q, latent, ids in layers:
        if out is not None:
            q = mx.depends(q, out)
        out = fn(q, latent, ids)
    return out


def run_matrix(args):
    import mlx.core as mx
    import numpy as np

    from mlx2.runtime.models import glm53_sparse_attention as sa
    from mlx2.runtime.models import glm53_sparse_attention_v2 as v2

    mx.set_default_device(mx.gpu)
    v2_arms = [f"fused_v2@{s},{d}" for s in v2.SPLITS_SUPPORTED for d in v2.DIM_SPLITS_SUPPORTED]
    cells = []
    for spec in args.cells:
        total, rows, kind = spec.split(":")
        cells.append((int(total), int(rows), kind == "dense"))
    rng = np.random.default_rng(args.seed)
    result = receipt(args, mx, layers=LAYERS, reps=args.reps, cells=[])
    for total, rows, dense in cells:
        arms = (["dense_sdpa"] if dense and rows == 1 else []) + ([] if dense else ["gathered_v1"])
        if not dense and rows <= 8:
            arms.append("fused_decode_v1")
        arms += v2_arms
        layers = build_layers(mx, rng, total, rows, dense)
        fns = {a: arm_fn(a, mx, sa, v2) for a in arms}
        for a in arms:  # compile, first-use gates, warm-up
            for _ in range(3):
                mx.eval(chain(fns[a], layers, mx))
        times = {a: [] for a in arms}
        for rep in range(args.reps):
            order = arms[rep % len(arms):] + arms[: rep % len(arms)]
            for a in order:
                mx.synchronize()
                t0 = time.perf_counter()
                mx.eval(chain(fns[a], layers, mx))
                times[a].append(time.perf_counter() - t0)
        # Agreement of each arm with the first on the last layer (not a gate).
        ref = np.array(fns[arms[0]](*layers[-1]), np.float64)
        row = {"total": total, "rows": rows, "dense": dense, "arms": {}}
        for a in arms:
            t = sorted(times[a])
            got = np.array(fns[a](*layers[-1]), np.float64)
            per_layer = [x / LAYERS * 1e3 for x in t]
            row["arms"][a] = {"ms_per_layer_p50": statistics.median(per_layer),
                              "ms_per_layer_p10": per_layer[len(t) // 10],
                              "ms_per_layer_p90": per_layer[(9 * len(t)) // 10],
                              "us_per_query_layer_p50": statistics.median(per_layer) * 1e3 / rows,
                              "max_abs_vs_first_arm": float(np.nanmax(np.abs(got - ref)))}
        result["cells"].append(row)
        print(json.dumps({"total": total, "rows": rows, "dense": dense,
                          **{a: round(v["ms_per_layer_p50"], 4) for a, v in row["arms"].items()}}), flush=True)
        del layers
        mx.clear_cache()
    (args.out / "matrix.json").write_text(json.dumps(result, indent=1) + "\n")
    return 0


# ---------------------------------------------------------------------------
# model helpers (ttft, captured)

def load_model(policy_path):
    import mlx.core as mx
    from glm53_decode_replay import load

    adapter, policy = load(policy_path)
    return adapter, policy, mx


def bind_prefill_law(model):
    """Route the model's sparse_prefill call to the law its value names.

    glm53_gguf._mla_attend calls ``sparse_prefill(q, source, selected,
    counters=...)`` without a policy (the call site is WP5's); this binds
    ``prefill_law(model.sparse_prefill)`` and, for fused_v2, PREFILL_BLOCK_V2.
    """
    from mlx2.runtime.models import glm53_sparse_attention as sa

    original = sa.sparse_prefill

    def bound(q, latent, ids, *, block=sa.PREFILL_BLOCK, counters=None, policy=None):
        law = policy or sa.prefill_law(model.sparse_prefill)
        if law == "fused_v2" and block == sa.PREFILL_BLOCK:
            block = sa.PREFILL_BLOCK_V2
        return original(q, latent, ids, block=block, counters=counters, policy=law)

    sa.sparse_prefill = bound
    return original


def run_ttft(args):
    import hashlib

    from glm53_ordinary_profile import CHECKPOINT, run_pass
    from glm53_policy_ab import (
        classify_divergence,
        decode_text,
        needle_found,
        needle_prompt,
        served,
    )

    adapter, policy, mx = load_model(args.policy)
    model = adapter.model
    bind_prefill_law(model)
    arms = json.loads(args.arms_json)
    ids = needle_prompt(adapter, args.tokens)
    result = receipt(args, mx, checkpoint=CHECKPOINT, policy=policy, prompt_tokens=len(ids),
                     prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(), arms=arms, runs=[])
    keep = ("sparse_attention_fused_v2", "sparse_attention_fused_v2_rows", "sparse_prefill_fused_v2_blocks",
            "sparse_prefill_gathered_blocks", "sparse_prefill_dense_pieces", "sparse_attention_gathered",
            "sparse_attention_mask", "mla_fused", "mla_fused_declined")
    first = {}
    # Every arm starts from the loaded policy's attributes: an attribute one
    # arm sets and another omits is restored, never inherited.
    touched = sorted({attr for arm in arms for attr in arm["attrs"]})
    baseline = {attr: getattr(model, attr) for attr in touched}
    result["baseline_attrs"] = baseline
    for rep in range(args.reps):
        order = arms if rep % 2 == 0 else arms[::-1]
        for arm in order:
            for attr, value in {**baseline, **arm["attrs"]}.items():
                setattr(model, attr, value)
            if args.cooldown_c:
                admitted = subprocess.run(
                    [sys.executable, str(Path(args.cooldown_script)), str(args.cooldown_c), "600"], check=False)
                if admitted.returncode != 0:
                    raise SystemExit(f"cooldown admission failed (rc {admitted.returncode}); no arm is timed")
            before = {k: model.calls.get(k, 0) for k in keep}
            mx.clear_cache()
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            chunks, steps, out = run_pass(model, ids, 2048, args.decode, mx, window=5)
            wall = time.perf_counter() - t0
            text, eos_at = decode_text(adapter, out)
            row = {"arm": arm["name"], "rep": rep, "ttft_s": sum(chunks), "chunks_s": chunks,
                   "decode_step_p50_ms": statistics.median(steps) * 1e3 if steps else None,
                   "wall_s": wall, "metal_peak_bytes": mx.get_peak_memory(), "tokens": out,
                   "windows": run_pass.windows, "finite": run_pass.finite, "text": text, "eos_at": eos_at,
                   "served": served(out, eos_at),
                   "oracle": needle_found(text), "prefill_logits_sha256": run_pass.prefill_logits_sha256,
                   "counters": {k: model.calls.get(k, 0) - before[k] for k in keep}}
            result["runs"].append(row)
            first.setdefault(arm["name"], row)
            print(json.dumps({k: row[k] for k in ("arm", "rep", "ttft_s", "decode_step_p50_ms", "metal_peak_bytes",
                                                  "oracle", "counters")}), flush=True)
            (args.out / "ttft.json").write_text(json.dumps(result, indent=1) + "\n")
    ref = first[arms[0]["name"]]
    # Every repetition of every other arm against the reference arm's first run.
    result["classification_all"] = [
        {"arm": row["arm"], "rep": row["rep"],
         **classify_divergence(ref["served"], row["served"], ref["windows"], row["windows"], ref["oracle"], row["oracle"])}
        for row in result["runs"] if row is not ref]
    result["classification"] = {
        # Compared through the first stop token only (tokens past it are filler).
        name: classify_divergence(ref["served"], row["served"], ref["windows"], row["windows"],
                                  ref["oracle"], row["oracle"])
        for name, row in first.items() if name != arms[0]["name"]}
    result["summary"] = {name: {"ttft_s": [r["ttft_s"] for r in result["runs"] if r["arm"] == name]}
                         for name in first}
    problems = [f"{c['arm']} rep {c['rep']}: {c['verdict']}" for c in result["classification_all"]
                if c["verdict"] not in ("exact", "near_tie_equivalent")]
    problems += [f"{r['arm']} rep {r['rep']}: nonfinite logits" for r in result["runs"] if not r["finite"]]
    problems += [f"{r['arm']} rep {r['rep']}: functional oracle failed" for r in result["runs"] if not r["oracle"]]
    for r, arm in ((r, next(a for a in arms if a["name"] == r["arm"])) for r in result["runs"]):
        law = arm["attrs"].get("sparse_prefill", "")
        if law.startswith("gathered_v2_from_") and not r["counters"]["sparse_prefill_fused_v2_blocks"]:
            problems.append(f"{r['arm']} rep {r['rep']}: fused_v2 prefill not engaged")
    result["verdict"] = {"passed": not problems, "problems": problems}
    print(json.dumps({"classification": result["classification"], "summary": result["summary"],
                      "verdict": result["verdict"]}), flush=True)
    (args.out / "ttft.json").write_text(json.dumps(result, indent=1) + "\n")
    return 0 if not problems else 3


def run_captured(args):
    """R2: per band of query positions, the dense piece vs the fused_v2 block on
    operands captured from one real prompt (one MLA layer, 128-query pieces)."""
    import numpy as np
    from glm53_ordinary_profile import CHECKPOINT
    from glm53_policy_ab import needle_prompt

    from mlx2.runtime.models import glm53_sparse_attention as sa

    adapter, policy, mx = load_model(args.policy)
    model = adapter.model
    ids = needle_prompt(adapter, args.tokens)
    # A band is keyed by its threshold value n (gathered_v2_from_<n>: queries
    # with pos + 1 >= n). Its piece is the 128 queries from pos = n - 1, moved
    # to the next chunk start when they would cross a chunk boundary.
    if any(n not in sa.PREFILL_POSITIONS_V2 for n in args.bands):
        raise SystemExit(f"--bands must be threshold values from {sa.PREFILL_POSITIONS_V2}")
    starts = {}
    for n in args.bands:
        pos = n - 1
        if pos % 2048 > 2048 - 128:
            pos = (pos // 2048 + 1) * 2048
        if pos + 128 > len(ids):
            raise SystemExit(f"band {n} needs a prompt of at least {pos + 128} tokens (have {len(ids)})")
        starts[n] = pos
    captured = {}
    original = type(model)._mla_attend
    layer = {"n": 0}

    def capture(self, q, selected, source):
        # Count MLA calls per chunk; keep the args.layer-th MLA layer's operands.
        out = original(self, q, selected, source)
        index = layer["n"] % 11
        layer["n"] += 1
        if index == args.layer and selected is not None and not isinstance(source, tuple):
            total, length = source.shape[2], q.shape[2]
            first = total - length
            for n, pos in starts.items():
                if n not in captured and first <= pos and pos + 128 <= total:
                    s0 = pos - first
                    captured[n] = (q[:, :, s0 : s0 + 128], source[:, :, : pos + 128], selected[:, s0 : s0 + 128])
                    mx.eval(captured[n])
        return out

    type(model)._mla_attend = capture
    cache = model.make_cache()
    for begin in range(0, len(ids), 2048):
        piece = mx.array([ids[begin : begin + 2048]])
        with model._prompt_scope(piece.shape[1]):
            logits = model(piece, cache, logits_to_keep=1)
        mx.eval(logits, [c.state for c in cache])
    type(model)._mla_attend = original
    missing = sorted(set(args.bands) - set(captured))
    if missing:
        raise SystemExit(f"bands {missing} were not captured; no crossover is emitted on partial coverage")
    del cache, logits
    mx.clear_cache()
    result = receipt(args, mx, checkpoint=CHECKPOINT, policy=policy, prompt_tokens=len(ids), layer=args.layer,
                     bands=[])
    for band in sorted(captured):
        q, source, selected = captured[band]
        total, length = source.shape[2], q.shape[2]

        def dense(q=q, source=source, selected=selected, length=length, total=total):
            return model._mla_masked(q, source, selected, length, total)

        def fused(q=q, source=source, selected=selected, length=length):
            return sa.sparse_prefill(q, source, selected, block=length, policy="fused_v2")

        def gathered(q=q, source=source, selected=selected):
            return sa.sparse_prefill(q, source, selected, policy="gathered_v1")

        fns = {"dense": dense, "fused_v2": fused, "gathered_v1": gathered}
        for fn in fns.values():
            for _ in range(2):
                mx.eval(fn())
        times = {k: [] for k in fns}
        for rep in range(args.reps):
            for k in (list(fns) if rep % 2 == 0 else list(fns)[::-1]):
                mx.synchronize()
                t0 = time.perf_counter()
                mx.eval(fns[k]())
                times[k].append(time.perf_counter() - t0)
        ref = np.array(dense(), np.float64)
        row = {"band": band, "policy": f"gathered_v2_from_{band}", "first_pos": total - length, "total": total,
               "rows": length,
               "ms": {k: statistics.median(v) * 1e3 for k, v in times.items()},
               "max_abs_vs_dense": {k: float(np.nanmax(np.abs(np.array(fns[k](), np.float64) - ref)))
                                    for k in ("fused_v2", "gathered_v1")}}
        result["bands"].append(row)
        print(json.dumps(row), flush=True)
    bands = result["bands"]
    qualifying = [b["band"] for i, b in enumerate(bands)
                  if all(x["ms"]["fused_v2"] <= x["ms"]["dense"] for x in bands[i:])]
    result["p_star"] = min(qualifying) if qualifying else None
    # R2 decides only over its full band set; a subset is a preliminary reading.
    result["complete_r2_bands"] = sorted(b["band"] for b in bands) == list(sa.PREFILL_POSITIONS_V2)
    if not result["complete_r2_bands"]:
        result["p_star_preliminary"], result["p_star"] = result["p_star"], None
    (args.out / "captured.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"p_star": result["p_star"]}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-revision", default="unrecorded")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pytest")
    p = sub.add_parser("matrix")
    p.add_argument("--cells", nargs="+", default=["2048:1:dense", "2100:1:ids", "16384:1:ids", "32768:1:ids",
                                                  "16384:64:ids", "32768:64:ids"],
                   help="total:rows:dense|ids")
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--seed", type=int, default=5304)
    p = sub.add_parser("ttft")
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=16384)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--decode", type=int, default=24)
    p.add_argument("--arms-json", required=True, help='[{"name":..,"attrs":{model attr: value}}, ...]')
    p.add_argument("--cooldown-c", type=float, default=None)
    p.add_argument("--cooldown-script",
                   default="~/Desktop/mlx2/qualification/runs/glm53-ordinary-roofline-20261030/cooldown.py")
    p = sub.add_parser("captured")
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=8400)
    p.add_argument("--layer", type=int, default=5)
    p.add_argument("--bands", type=lambda s: [int(x) for x in s.split(",")],
                   default=[2049, 3072, 4096, 5120, 6144, 7168, 8192],
                   help="threshold values n of gathered_v2_from_<n>; 12288,16384 need --tokens >= 16512")
    p.add_argument("--reps", type=int, default=10)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to reuse a nonempty evidence directory")
    args.out.mkdir(parents=True, exist_ok=True)
    return {"pytest": run_pytest, "matrix": run_matrix, "ttft": run_ttft, "captured": run_captured}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
