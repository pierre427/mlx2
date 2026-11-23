#!/usr/bin/env python3
"""Hy4-flash routed-expert kernel timing at verify-sized row counts (GPU micro-benchmark).

Loads ONE real layer's latent expert bank (512 experts x gate/up/down 1536x1536,
4-bit affine g64, ~1.9 GB) into mlx2's own SwitchGLU + QuantizedSwitchLinear, the
module Hy4-flash's MoE calls, and times one SwitchGLU forward for R rows x top-16
routing at R in --rows.  For each call it reports the unique experts U and the
assignments A = R*16, and the time a pure weight read would need if MLX read each
unique expert once (U * expert_bytes) or once per assignment (A * expert_bytes),
both at the machine's measured streaming read bandwidth (a large-array reduction,
measured in the same process).  Two routing laws: ``uniform`` (independent rows)
and ``overlap`` (rows draw from a shared pool sized so assignments per expert ~=
--target-overlap, the 1.5-1.8 the B=1..4 profiles measured).

``--arms`` selects expert-kernel knobs (each timed on the same routing and
compared element-wise with the ``stock`` arm's output):

  stock           the default route (floor pad 3, NAX off, grouped off)
  pad_floor2/1    MLX2_MOE_RHS_PAD_MIN_ROWS 2 / 1 (pads to MLX's streaming floor
                  only when sorted rows >= floor * num_experts)
  pad_always      MLX2_MOE_RHS_PAD_POLICY=always (every sorted gather padded to
                  4 * num_experts rows -> gather_qmm_rhs)
  pad_always_nax  pad_always + MLX2_MOE_NAX_GATHER=gather inside a prefill scope
  grouped         MLX2_MOE_GROUPED_EXPERTS=1 (needs --activation swiglu)
  read_once       switch_read_once kernels (mlx2, this branch)

Timing: ``median_ms`` is one call evaluated alone (includes the per-eval
submit/sync floor, ``sync_floor_ms``); ``chain_marginal_ms`` (``--chain K``)
is the per-call cost inside a dependent chain of K calls, the number a
forward graph pays per MoE layer.  The read-once kernel is measured at every
row count, also outside its serving admission window.

Single process, no server, model files read directly; needs the GPU lease.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mlx2.runtime.models import moe_nax_gather as NAX  # noqa: E402
from mlx2.runtime.models import switch_grouped_experts as G  # noqa: E402
from mlx2.runtime.models import switch_layers as S  # noqa: E402
from mlx2.runtime.models import switch_read_once as RO  # noqa: E402

ARMS = ("stock", "pad_floor2", "pad_floor1", "pad_always", "pad_always_nax", "grouped", "read_once")


def make_activation(kind: str, limit: float):
    if kind == "swiglu":
        return S.SwiGLU()
    from mlx2.runtime.models.hy_v4_flash import ClampedSwiGLU

    return ClampedSwiGLU(limit)


def load_layer(model_dir: Path, layer: int, activation):
    f = model_dir / f"model-experts-L{layer}.safetensors"
    w = mx.load(str(f))
    stem = f"model.layers.{layer}.mlp.switch_mlp."
    glu = S.SwitchGLU(1536, 1536, 512, bias=False, activation=activation)
    for name in ("gate_proj", "up_proj", "down_proj"):
        q = S.QuantizedSwitchLinear.__new__(S.QuantizedSwitchLinear)
        import mlx.nn as nn

        nn.Module.__init__(q)
        q.group_size, q.bits, q.mode = 64, 4, "affine"
        q.weight, q.scales, q.biases = w[stem + name + ".weight"], w[stem + name + ".scales"], w[stem + name + ".biases"]
        q.freeze()
        setattr(glu, name, q)
    mx.eval(glu.parameters())
    nbytes = sum(v.nbytes for v in w.values() if v.ndim == 3)
    return glu, nbytes / 512


def routing(rows: int, law: str, rng, target_overlap: float, k: int = 16, experts: int = 512):
    if law == "uniform":
        return np.stack([rng.choice(experts, k, replace=False) for _ in range(rows)])
    pool = max(k, min(experts, int(round(rows * k / target_overlap))))
    pool_ids = rng.choice(experts, pool, replace=False)
    return np.stack([rng.choice(pool_ids, k, replace=False) for _ in range(rows)])


def timed(fn, reps: int, warm: int = 3):
    for _ in range(warm):
        mx.eval(fn())
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(fn())
        out.append((time.perf_counter() - t) * 1e3)
    return statistics.median(out), min(out)


def read_bandwidth_gbs(gb: float = 1.5, reps: int = 15) -> float:
    a = mx.ones((int(gb * 2**30) // 2,), dtype=mx.bfloat16)
    mx.eval(a)
    med, best = timed(lambda: mx.sum(a), reps)
    del a
    mx.clear_cache()
    return (gb * 2**30) / (best / 1e3) / 1e9


@contextlib.contextmanager
def arm_scope(arm: str):
    """Set the module knobs an arm selects, restore the defaults after."""
    floor0, pol0 = S._RHS_PAD_MIN_ROWS_PER_EXPERT, S._RHS_PAD_POLICY
    nax0, grp0 = NAX.MODE, G.MODE
    stack = contextlib.ExitStack()
    try:
        if arm == "pad_floor2":
            S._RHS_PAD_MIN_ROWS_PER_EXPERT = 2
        elif arm == "pad_floor1":
            S._RHS_PAD_MIN_ROWS_PER_EXPERT = 1
        elif arm in ("pad_always", "pad_always_nax"):
            S.set_pad_policy("always")
            if arm == "pad_always_nax":
                NAX.set_mode("gather")
                stack.enter_context(NAX.prefill_scope(True))
        elif arm == "grouped":
            G.set_mode("on")
        yield
    finally:
        stack.close()
        S._RHS_PAD_MIN_ROWS_PER_EXPERT = floor0
        S.set_pad_policy(pol0)
        NAX.set_mode(nax0)
        G.set_mode(grp0)


def arm_fn(arm: str, glu, x, ind):
    if arm == "read_once":
        def f():
            y = RO.switch_glu(glu, x, ind)
            if y is None:
                raise RuntimeError(f"read_once refused: {RO.status()['reasons']}")
            return y
        return f
    return lambda: glu(x, ind)


def call(arm: str, glu, x, ind):
    if arm == "read_once":
        y = RO.switch_glu(glu, x, ind)
        if y is None:
            raise RuntimeError(f"read_once refused: {RO.status()['reasons']}")
        return y
    return glu(x, ind)


def chain_marginal(arm: str, glu, cells, rounds: int) -> float:
    """Per-call cost of a dependent chain of len(cells) calls, the sync floor removed."""
    def run(n):
        x = cells[0][0]
        for j in range(n):
            xi, ind, _ = cells[j]
            y = call(arm, glu, x, ind)
            x = (xi + 1e-3 * y.sum(axis=-2)).astype(mx.bfloat16)
        t = time.perf_counter()
        mx.eval(x)
        return (time.perf_counter() - t) * 1e3
    one, many = [], []
    for _ in range(rounds):
        one.append(run(1))
        many.append(run(len(cells)))
    return (statistics.median(many) - statistics.median(one)) / (len(cells) - 1)


def compare(y, ref) -> dict:
    a = np.array(y.astype(mx.float32))
    b = np.array(ref.astype(mx.float32))
    diff = np.abs(a - b)
    scale = np.abs(b)
    return {"bit_exact": bool(np.array_equal(a, b)), "max_abs": float(diff.max()),
            "ref_max_abs": float(scale.max()),
            "max_abs_over_ref_max": float(diff.max() / max(scale.max(), 1e-30)),
            "rel_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(b), 1e-30)),
            "mismatch_frac": float((diff > 0).mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--rows", default="1,2,4,6,7,8,12,16,24,28,32,48,64")
    ap.add_argument("--laws", default="uniform,overlap")
    ap.add_argument("--arms", default="stock", help=",".join(ARMS))
    ap.add_argument("--activation", choices=("swiglu", "clamped"), default="swiglu",
                    help="swiglu (the v1 receipt's module) or Hy4's ClampedSwiGLU(10)")
    ap.add_argument("--swiglu-limit", type=float, default=10.0)
    ap.add_argument("--target-overlap", type=float, default=1.7)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--chain", type=int, default=1,
                    help="also time a dependent chain of K calls (K distinct routings, each call's input "
                         "built from the previous output, one eval): marginal = (T(K) - T(1)) / (K - 1), "
                         "the per-layer cost inside a forward graph without the per-eval sync floor; "
                         "roofs use the chain's mean U")
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cpu", action="store_true", help="dry run on the CPU (timings meaningless)")
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    arms = [s for s in a.arms.split(",") if s]
    bad = set(arms) - set(ARMS)
    if bad:
        raise SystemExit(f"unknown arms {sorted(bad)}")
    if "stock" not in arms:
        arms.insert(0, "stock")
    mx.set_default_device(mx.cpu if a.cpu else mx.gpu)
    # Measure the read-once kernel at every row count, outside the serving
    # admission window (MIN_ROWS / MAX_ASSIGNMENTS) too.
    window = {"min_rows": RO.MIN_ROWS, "max_assignments": RO.MAX_ASSIGNMENTS}
    RO.MIN_ROWS, RO.MAX_ASSIGNMENTS = 1, 1 << 30
    bw = read_bandwidth_gbs(0.25 if a.cpu else 1.5, 3 if a.cpu else 15)
    glu, expert_bytes = load_layer(a.model, a.layer, make_activation(a.activation, a.swiglu_limit))
    rng = np.random.default_rng(a.seed)
    rows_out = []
    # Per-call sync floor: one tiny op evaluated alone, the same way each arm is timed.
    tiny = mx.zeros((16,), dtype=mx.float32)
    mx.eval(tiny)
    sync_floor, _ = timed(lambda: tiny + 1, 50)
    # Warm every arm (kernel builds, clocks) before the first timed cell.
    wx = mx.random.normal((8, 1536)).astype(mx.bfloat16)
    wi = mx.array(routing(8, "overlap", rng, a.target_overlap).astype(np.uint32))
    for arm in arms:
        with arm_scope(arm):
            timed(arm_fn(arm, glu, wx, wi), 10)
    for law in a.laws.split(","):
        for r in [int(x) for x in a.rows.split(",")]:
            idx = routing(r, law, rng, a.target_overlap)
            x = mx.random.normal((r, 1536)).astype(mx.bfloat16)
            ind = mx.array(idx.astype(np.uint32))
            mx.eval(x, ind)
            u, asg = len(np.unique(idx)), idx.size
            t_unique = u * expert_bytes / (bw * 1e9) * 1e3
            t_assign = asg * expert_bytes / (bw * 1e9) * 1e3
            cells = [(x, ind, u)]
            for _ in range(max(0, a.chain - 1)):
                ei = routing(r, law, rng, a.target_overlap)
                exi = mx.random.normal((r, 1536)).astype(mx.bfloat16)
                cells.append((exi, mx.array(ei.astype(np.uint32)), len(np.unique(ei))))
            mx.eval([c[:2] for c in cells])
            u_mean = sum(c[2] for c in cells) / len(cells)
            t_unique_p = u_mean * expert_bytes / (bw * 1e9) * 1e3
            t_assign_p = asg * expert_bytes / (bw * 1e9) * 1e3
            ref = None
            for arm in arms:
                S.moe_pad_status(reset=True)
                G.status(reset=True)
                NAX.status(reset=True)
                RO.status(reset=True)
                with arm_scope(arm):
                    fn = arm_fn(arm, glu, x, ind)
                    y = fn()
                    mx.eval(y)
                    med, best = timed(fn, a.reps)
                    med_p = None
                    if len(cells) > 1:
                        med_p = chain_marginal(arm, glu, cells, max(5, a.reps // 2))
                    pad, grp, nax, ro = S.moe_pad_status(), G.status(), NAX.status(), RO.status()
                if arm == "stock":
                    ref = y
                engaged = {"stock": True,
                           "pad_floor2": pad["padded_calls"] > 0, "pad_floor1": pad["padded_calls"] > 0,
                           "pad_always": pad["padded_calls"] > 0,
                           "pad_always_nax": pad["padded_calls"] > 0 and sum(nax.get("calls", {}).values()) > 0,
                           "grouped": grp["calls"] > 0, "read_once": ro["calls"] > 0}[arm]
                row = {"law": law, "rows": r, "arm": arm, "engaged": engaged,
                       "unique_experts": u, "assignments": asg,
                       "assign_per_expert": round(asg / u, 3), "median_ms": round(med, 4),
                       "best_ms": round(best, 4), "read_once_ms": round(t_unique, 4),
                       "read_per_assignment_ms": round(t_assign, 4),
                       "x_of_read_once": round(med / t_unique, 3),
                       "x_of_read_per_assignment": round(med / t_assign, 3),
                       "x_of_stock": None,
                       "chain_marginal_ms": None if med_p is None else round(med_p, 4),
                       "chain_read_once_ms": None if med_p is None else round(t_unique_p, 4),
                       "chain_x_of_read_once": None if med_p is None else round(med_p / t_unique_p, 3),
                       "chain_x_of_read_per_assignment": None if med_p is None else round(med_p / t_assign_p, 3),
                       "chain_x_of_stock": None,
                       "vs_stock": None if arm == "stock" else compare(y, ref),
                       "counters": {"pad_calls": pad["padded_calls"], "pad_rows": pad["pad_rows"],
                                    "pad_choices": pad["choices"], "grouped_calls": grp["calls"],
                                    "grouped_reasons": grp["reasons"],
                                    "nax_calls": nax.get("calls"),
                                    "nax_fallbacks": nax.get("fallbacks"),
                                    "read_once_calls": ro["calls"], "read_once_reasons": ro["reasons"]}}
                stock_row = next((x for x in rows_out[::-1] if x["arm"] == "stock"
                                  and x["law"] == law and x["rows"] == r), None)
                if stock_row is not None:
                    row["x_of_stock"] = round(med / stock_row["median_ms"], 3)
                    if med_p is not None:
                        row["chain_x_of_stock"] = round(med_p / stock_row["chain_marginal_ms"], 3)
                rows_out.append(row)
                brief = {k: row[k] for k in ("law", "rows", "arm", "engaged", "median_ms",
                                             "x_of_read_once", "x_of_read_per_assignment", "x_of_stock",
                                             "chain_x_of_read_once", "chain_x_of_stock")}
                if row["vs_stock"]:
                    brief["max_abs"] = row["vs_stock"]["max_abs"]
                    brief["bit_exact"] = row["vs_stock"]["bit_exact"]
                print(json.dumps(brief), flush=True)
                del y
            del ref
    receipt = {"schema": "mlx2.hy4-expert-kernel-bench.v2", "qualification": "unqualified",
               "model": str(a.model), "layer": a.layer, "expert_bytes": expert_bytes,
               "activation": a.activation, "arms": arms,
               "measured_read_bandwidth_gbs": round(bw, 1),
               "sync_floor_ms": round(sync_floor, 4), "chain": a.chain,
               "mlx2_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                             capture_output=True, text=True).stdout.strip(),
               "mlx2_dirty": bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                                                 capture_output=True, text=True).stdout.strip()),
               "mlx": mx.__version__, "device": mx.device_info().get("device_name") if not a.cpu else "cpu",
               "read_once_config": {**RO.status()["config"], "serving_window": window},
               "rows": rows_out}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(receipt, indent=1, default=str))
    print(json.dumps({"bandwidth_gbs": receipt["measured_read_bandwidth_gbs"], "expert_mb": expert_bytes / 1e6}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
