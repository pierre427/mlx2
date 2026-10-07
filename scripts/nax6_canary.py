#!/usr/bin/env python3
"""NAX sorted gather_qmm: canary sweep over affine widths (6-bit focus).

Runs the module's own one-time self-tests (``_self_test`` plain kernel vs
stock ``mx.gather_qmm`` bitwise; ``_self_test_act`` fused [gate; up] SwiGLU
vs plain + split + reference activation; ``_self_test_act_map`` row map;
``_self_test_split`` split tables vs two stock gathers + SwiGLU, and its
row-mapped form) for every requested width / group / plan, then a
real-shape bitwise check (Qwen3.6-35B-A3B experts: hidden 2048, MoE
intermediate 512, 256 experts top-8) on captured routing.

Research harness (needs Metal on an M5); run through run_with_gpu_locks.py.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

E, TOPK, HID, INTER = 256, 8, 2048, 512


def swapouts() -> int:
    for line in subprocess.check_output(["vm_stat"], text=True).splitlines():
        if line.startswith("Swapouts"):
            return int(line.split(":")[1].strip().rstrip("."))
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", default="2,3,4,5,6,8")
    ap.add_argument("--full-grid-bits", default="6",
                    help="widths canaried on the whole plan grid (others: planner plans)")
    ap.add_argument("--routing", default=str(
        ROOT / "qualification/runs/research-20261006/b1-moe-tile-headroom/capture"))
    ap.add_argument("--real-keys", type=int, default=4)
    ap.add_argument("--groups", default="32,64,128")
    ap.add_argument("--falsify", action="store_true",
                    help="drop the straddling-word half of the non-power-of-two unpack "
                         "(the canary must refuse every 3/5/6-bit instantiation)")
    a = ap.parse_args()

    import mlx.core as mx
    import numpy as np

    from mlx2.runtime.models import moe_nax_gather as nax
    from mlx2.runtime.models.activations import swiglu

    mx.set_cache_limit(4 << 30)
    if a.falsify:
        needle = "qv |= rw[wi + 1] << (32 - sh);"
        assert needle in nax._MM_HEADER
        nax._MM_HEADER = nax._MM_HEADER.replace(needle, "qv |= 0u;")
    if not nax.enabled():
        raise SystemExit("NAX host gate is closed on this device")
    swap0 = swapouts()
    t0 = time.time()

    P, SEG, DB, GX = nax.Plan, nax._SCHED_SEG, nax._SCHED_DB, nax._GX
    planner = [P(DB, 64, 64, 0, 0), P(DB, 64, 64, GX, 0), P(DB, 96, 64, GX, 0),
               P(SEG, 96, 128, GX, 0), P(SEG, 128, 128, GX, 8192)]
    grid = [P(s, bm, bk, gx, pad) for s, bm, bk, gx, pad in itertools.product(
        (SEG, DB), (64, 96, 128), (64, 128), (0, GX), (0, 8192)) if not (s == DB and bk != 64)]
    full_bits = {int(b) for b in a.full_grid_bits.split(",") if b}
    groups = {int(g) for g in a.groups.split(",") if g}

    rows = []
    for bits in [int(b) for b in a.bits.split(",")]:
        for gs in (32, 64, 128):
            if gs not in groups:
                continue
            combos = [(mx.bfloat16, p) for p in dict.fromkeys(planner + [P(SEG, 64, 64, 0, 0)])]
            if bits in full_bits and gs == 64:
                combos = [(mx.bfloat16, p) for p in dict.fromkeys(planner + grid)]
                combos += [(mx.float16, p) for p in planner]
            for dtype, plan in combos:
                fit = nax._fit_plan(plan, bits, gs)
                for align_n, align_k in ((True, True), (False, True), (True, False)):
                    K = nax._canary_k(fit, align_k)
                    if K % gs:
                        continue  # mx.quantize needs K % group == 0: no canary
                    if fit.sched == DB and (not align_n or not align_k or fit.bk != 64):
                        continue  # the entry points run these as seg
                    key = (dtype, "affine", bits, gs, fit, align_n, align_k)
                    t = time.time()
                    res = {"plain": nax._self_test(key)}
                    if align_n:
                        ek = key + (nax._EPI_SWIGLU, None)
                        res["act"] = nax._self_test_act(ek)
                        res["act_map"] = nax._self_test_act_map(ek + ("map",))
                        res["split"] = nax._self_test_split(ek + ("split",))
                        res["split_map"] = nax._self_test_split(ek + ("split", "map"))
                    rec = {"bits": bits, "gs": gs, "dtype": str(dtype).rsplit(".", 1)[-1],
                           "plan": plan.describe(), "fit": fit.describe(),
                           "align_n": align_n, "align_k": align_k, "K": K,
                           "results": res, "s": round(time.time() - t, 2)}
                    rows.append(rec)
                    if not all(res.values()):
                        print(json.dumps(rec), flush=True)
        print(json.dumps({"bits": bits, "done": len(rows), "elapsed_s": round(time.time() - t0)}), flush=True)
        if swapouts() > swap0:
            print(json.dumps({"abort": "swapouts rose"}), flush=True)
            break

    # Real-shape bitwise check on captured routing (6-bit gs64 tables).
    real = []
    routing = np.load(Path(a.routing) / "routing.npz")
    cap = json.loads((Path(a.routing) / "capture.json").read_text())
    bits_of = {int(k): v for k, v in cap["expert_bits_per_layer"].items()}
    keys = [k for k in sorted(routing.files) if bits_of[int(k[1:3])]["gate_proj"] == 6][: a.real_keys]

    def table(o, i, bits, seed):
        w = (mx.random.normal((E, o, i), key=mx.random.key(seed)) * 0.02).astype(mx.bfloat16)
        q = mx.quantize(w, group_size=64, bits=bits)
        mx.eval(*q)
        return q

    g6, u6, d6 = table(INTER, HID, 6, 11), table(INTER, HID, 6, 12), table(HID, INTER, 6, 13)
    kw = dict(group_size=64, bits=6)
    for n, key in enumerate(keys):
        for T in (2048, 8192):
            if T == 8192:
                layer = key[:3]
                parts = [routing[k] for k in sorted(routing.files) if k.startswith(layer)]
                inds_np = np.concatenate([p.reshape(-1, TOPK) for p in parts])[:T]
            else:
                inds_np = routing[key].reshape(-1, TOPK)
            inds = mx.array(inds_np.astype(np.uint32))
            x_tok = (mx.random.normal((T, 1, HID), key=mx.random.key(300 + n)) * 0.5).astype(mx.bfloat16)
            flat = inds.flatten()
            order = mx.argsort(flat)
            row_map = (order // TOPK).astype(mx.uint32)
            idx = flat[order].astype(mx.uint32)
            x_sorted = x_tok[row_map]
            M = int(idx.size)
            ref_up = mx.gather_qmm(x_sorted, *u6, rhs_indices=idx, transpose=True, sorted_indices=True, **kw)
            ref_gate = mx.gather_qmm(x_sorted, *g6, rhs_indices=idx, transpose=True, sorted_indices=True, **kw)
            ref_h = swiglu(ref_gate, ref_up)
            got_h = nax.sorted_gather_qmm_swiglu_split(x_tok, g6, u6, idx, row_map=row_map, **kw)
            ref_d = mx.gather_qmm(ref_h, *d6, rhs_indices=idx, transpose=True, sorted_indices=True, **kw)
            got_d = nax.sorted_gather_qmm(ref_h, *d6, idx, **kw)
            # The plain kernel on the up table too (gather route, K=2048).
            got_up = nax.sorted_gather_qmm(x_sorted, *u6, idx, **kw)
            rec = {
                "key": key, "T": T, "rows": M,
                "gate_up_plan": nax._fit_plan(nax._plan(M, E, HID, 2 * INTER), 6, 64).describe(),
                "down_plan": nax._fit_plan(nax._plan(M, E, INTER, HID), 6, 64).describe(),
                "swiglu_split_map_bitwise": got_h is not None and nax._bits_equal(got_h, ref_h),
                "gather_up_bitwise": got_up is not None and nax._bits_equal(got_up, ref_up),
                "gather_down_bitwise": got_d is not None and nax._bits_equal(got_d, ref_d),
            }
            real.append(rec)
            print(json.dumps(rec), flush=True)
            del x_tok, x_sorted, ref_up, ref_gate, ref_h, ref_d, got_h, got_d, got_up
            mx.clear_cache()

    swap1 = swapouts()
    fails = [r for r in rows if not all(r["results"].values())]
    report = {
        "schema": "mlx2.research.nax6-canary.v1",
        "source_sha": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "src"], text=True).strip()),
        "falsify": a.falsify,
        "instantiations": len(rows),
        "failures": fails,
        "rows": rows,
        "real_shape": real,
        "elapsed_s": time.time() - t0,
        "swapouts_before": swap0, "swapouts_after": swap1, "swap_rose": swap1 > swap0,
        "mlx_version": mx.__version__,
        "device": mx.device_info().get("device_name"),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({"instantiations": len(rows), "failures": len(fails),
                      "real_all_bitwise": all(r["swiglu_split_map_bitwise"] and r["gather_down_bitwise"]
                                              and r["gather_up_bitwise"] for r in real),
                      "swap_rose": swap1 > swap0, "elapsed_s": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
