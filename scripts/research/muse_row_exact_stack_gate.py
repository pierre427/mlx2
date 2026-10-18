#!/usr/bin/env python3
"""Muse row-exact verify forward: cost by stage form, and bits vs decode.

One process, the served Muse target (the external-draft AST policy loads the
drafter too, which this gate does not run), the lane-matmul law installed the
way ``mlx2.server`` installs it.  A real-text context of ``--context`` tokens
is prefilled once; each shape ``BxL`` then verifies B lanes x L draft tokens
from deep copies of that cache.

Arms (interleaved per rep, rep 0 discarded):

* ``v2``     the stacked form: row-independent stages in one launch when the
             parity canary passes (this tree's default);
* ``v1``     the same forward with every row-independent stage forced to its
             per-row form (v1's launch geometry; ``--arms`` may drop it);
* ``stock``  the ordinary multi-token forward over the same rows: one launch
             per stage, not row-exact; the lower bound for this geometry;
* ``decode`` one ordinary one-token forward (the serial reference's cost).

Per arm: median wall ms per forward (GPU synchronized), median host ms until
the forward's own ``mx.eval`` (graph construction), and for v1/v2 the route
counters.  Exactness: v1 and v2 logits and DFlash taps are compared bit for
bit with serial one-token decode of each lane; the stock arm is the falsifier
(it must differ somewhere, or the comparison cannot tell forms apart).

  python scripts/run_with_gpu_locks.py --session <s> --label muse-stack-gate \\
      --receipt <dir>/lock.json -- env PYTHONPATH=src MLX_ENABLE_TF32=0 \\
      <venv>/bin/python scripts/research/muse_row_exact_stack_gate.py \\
      --i-own-the-gpu --policy <muse-ast.json> --out <dir>/gate.json
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

TAPS = (1, 13, 25, 37, 49)


def swapouts() -> int:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False).stdout
    line = next(line for line in out.splitlines() if line.startswith("Swapouts"))
    return int(line.split(":")[1].strip(" ."))


def context_tokens(tokenizer, count: int) -> list[int]:
    """Varied real text (repository docs), never a repeated string."""
    ids: list[int] = []
    for path in sorted((ROOT / "docs").rglob("*.md")):
        ids.extend(tokenizer.encode(path.read_text(errors="replace") + "\n\n"))
        if len(ids) >= count + 4096:
            return ids
    raise SystemExit("not enough documentation text for the requested context")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-own-the-gpu", action="store_true", required=True)
    ap.add_argument("--model", default=str(Path.home() / "mlx-models/Muse-Glimmer-30B-mlx-8bit"))
    ap.add_argument("--policy", type=Path, required=True, help="Muse external-draft policy JSON")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--shapes", default="1x4,2x4,4x4,5x4,8x4")
    ap.add_argument("--arms", default="v2,v1,stock,decode")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument(
        "--lane-mode",
        default="auto",
        choices=["auto", "exact"],
        help="lane matmul law as mlx2.server --lane-matmul installs it",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists() and args.out.stat().st_size:
        raise SystemExit(f"refusing to overwrite nonempty evidence {args.out}")

    import mlx.core as mx

    from mlx2.adapters.muse_glimmer import MuseGlimmerAdapter
    from mlx2.runtime.lane import apply_policy
    from mlx2.runtime.lane import stats as lane_stats
    from mlx2.runtime.lane.policy import detect, resolve
    from mlx2.runtime.models import muse_glimmer as MG
    from mlx2.runtime.models import row_exact_qmv as REQ
    from mlx2.runtime.row_exact.canary import CANARIES
    from mlx2.runtime.segmented_rotating_kv import SegmentedKVRows

    swap0 = swapouts()
    mx.set_cache_limit(8 << 30)
    adapter = MuseGlimmerAdapter(args.model, execution_policy=json.loads(args.policy.read_text()))
    model = adapter.model
    lane_policy = resolve(
        detect(model, json.loads((Path(args.model) / "config.json").read_text())),
        family=getattr(getattr(adapter, "descriptor", None), "family", None),
        adapter=adapter.lane_policy_defaults() if callable(getattr(adapter, "lane_policy_defaults", None)) else None,
        overrides=None,
        mode=args.lane_mode,
    )
    from mlx2.serving import row_exact_target_mutation_guard

    row_exact_target_mutation_guard(adapter, lane_policy=lane_policy)
    offered = getattr(adapter, "lane_projection_groups", None)
    lane_receipt = apply_policy(model, lane_policy, declared=offered() if callable(offered) else ())
    if not model.supports_contextual_prefix_equivalence:
        raise SystemExit("policy does not select target_verify_row_exact")

    ids = context_tokens(adapter.tokenizer, args.context)
    base = model.make_cache()
    model.configure_target_verify_row_exact(False)
    for start in range(0, args.context, 2048):
        mx.eval(model(mx.array([ids[start : min(start + 2048, args.context)]]), cache=base))
    continuation = ids[args.context :]

    def lane_tokens(width, length):
        return mx.array([continuation[97 * lane : 97 * lane + length] for lane in range(width)])

    real_eval = mx.eval
    marks = {}

    def marking_eval(*values, **kw):
        marks.setdefault("host_end", time.perf_counter())
        return real_eval(*values, **kw)

    def forced_v1():
        """v1's per-row geometry for every row-independent stage."""
        saved = (MG._row_exact_rows, MG.Attention._rope_rows)

        def rows(call, value, key):
            MG.ROW_EXACT_ROUTES[f"{key[0]}:forced_per_row"] += 1
            return REQ.per_row(call, value)

        def rope(self, value, offsets, array_offsets):
            MG.ROW_EXACT_ROUTES["rope:forced_per_row"] += 1
            return self._rope_per_position(value, offsets, array_offsets)

        MG._row_exact_rows, MG.Attention._rope_rows = rows, rope
        return saved

    def run(arm, tokens):
        width, length = tokens.shape
        if arm == "decode":
            cache = copy.deepcopy(base)
            t0 = time.perf_counter()
            out = model.forward_with_taps(tokens[:1, :1], cache, TAPS)
            host_end = time.perf_counter()
            real_eval(*out)
            return out, time.perf_counter() - t0, host_end - t0
        if arm == "stock":
            caches = [copy.deepcopy(base) for _ in range(width)]
            tx = SegmentedKVRows(caches).begin([length] * width)
            t0 = time.perf_counter()
            try:
                out = model.forward_with_taps(tokens, tx.caches, TAPS)
                host_end = time.perf_counter()
                real_eval(*out)
            finally:
                tx.abort()
            return out, time.perf_counter() - t0, host_end - t0
        saved = forced_v1() if arm == "v1" else None
        caches = [copy.deepcopy(base) for _ in range(width)]
        tx = SegmentedKVRows(caches).begin([length] * width)
        model.configure_target_verify_row_exact(True)
        mx.eval = marking_eval
        marks.clear()
        try:
            t0 = time.perf_counter()
            out = model.forward_with_taps(tokens, tx.caches, TAPS)
            real_eval(*out)
            wall = time.perf_counter() - t0
        finally:
            mx.eval = real_eval
            model.configure_target_verify_row_exact(False)
            tx.abort()
            if saved is not None:
                MG._row_exact_rows, MG.Attention._rope_rows = saved
        return out, wall, marks.get("host_end", t0) - t0

    def serial(tokens):
        logits, taps = [], []
        for lane in range(tokens.shape[0]):
            cache = copy.deepcopy(base)
            rows = [model.forward_with_taps(mx.array([[t]]), cache, TAPS) for t in tokens[lane].tolist()]
            real_eval(*[a for r in rows for a in r])
            logits.append(mx.concatenate([r[0] for r in rows], axis=1))
            taps.append(mx.concatenate([r[1] for r in rows], axis=1))
        return mx.concatenate(logits), mx.concatenate(taps)

    def bits(a, b):
        if a.shape != b.shape:
            return -1
        return int(mx.sum(a.view(mx.uint16) != b.view(mx.uint16)).item())

    arms = [a for a in args.arms.split(",") if a]
    shapes = [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",") if s]
    report = {
        "schema": "mlx2.muse-row-exact-stack-gate.v1",
        "model": args.model,
        "policy": json.loads(args.policy.read_text()),
        "context": args.context,
        "algorithm": MG.TARGET_VERIFY_ROW_EXACT_VERSION,
        "lane_law": (lane_receipt or {}).get("law_id"),
        "device": mx.device_info().get("device_name"),
        "shapes": {},
    }
    for width, length in shapes:
        tokens = lane_tokens(width, length)
        want_logits, want_taps = serial(tokens)
        cell = {"rows": width * length, "arms": {}}
        samples = {arm: {"wall_ms": [], "host_ms": []} for arm in arms}
        for rep in range(args.reps + 1):
            order = arms[rep % len(arms) :] + arms[: rep % len(arms)]
            for arm in order:
                MG.ROW_EXACT_ROUTES.clear()
                (logits, taps), wall, host = run(arm, tokens)
                if rep == 0:
                    if arm in ("v1", "v2", "stock"):
                        cell["arms"].setdefault(arm, {})["bit_mismatches"] = {
                            "logits": bits(logits, want_logits),
                            "taps": bits(taps, want_taps),
                        }
                    if arm in ("v1", "v2"):
                        cell["arms"][arm]["routes"] = dict(sorted(MG.ROW_EXACT_ROUTES.items()))
                    continue
                samples[arm]["wall_ms"].append(wall * 1e3)
                samples[arm]["host_ms"].append(host * 1e3)
        for arm in arms:
            entry = cell["arms"].setdefault(arm, {})
            for key, values in samples[arm].items():
                entry[key] = {
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                }
        exact = all(
            cell["arms"][arm]["bit_mismatches"] == {"logits": 0, "taps": 0}
            for arm in ("v1", "v2")
            if arm in cell["arms"]
        )
        falsifier = cell["arms"].get("stock", {}).get("bit_mismatches")
        cell["exact"] = exact
        cell["falsifier_fired"] = None if falsifier is None else (sum(falsifier.values()) != 0)
        report["shapes"][f"{width}x{length}"] = cell
        print(
            f"{width}x{length}: exact={exact} falsifier={cell['falsifier_fired']} "
            + " ".join(
                f"{arm}={cell['arms'][arm]['wall_ms']['median']:.1f}ms"
                f"(host {cell['arms'][arm]['host_ms']['median']:.1f})"
                for arm in arms
            ),
            flush=True,
        )
    report["canaries"] = CANARIES.snapshot()
    report["lane_stats"] = lane_stats()
    report["swapouts_delta"] = swapouts() - swap0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=1, default=str))
    tmp.replace(args.out)
    ok = all(cell["exact"] for cell in report["shapes"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
