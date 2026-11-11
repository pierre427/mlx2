#!/usr/bin/env python3
"""Time the EXL3 encoder on one real DS4.1 routed expert (3 projections, LDLQ).

prepare: dequantize one expert and build its Hessians exactly as
convert_experts.py does (CPU), saved to <cache>/L<layer>E<expert>.npz so timing
runs do not reread the checkpoint.
time:    encode the cached expert at the given word counts with the chosen
backend and Viterbi kernel, print a JSON row per run with wall time, the time
split (prep / Viterbi / host LDLQ / finish) and a sha256 of the trellis bits, so
two kernels can be compared for bit identity on the same expert.

usage: bench_encoder.py prepare --layer 5 --expert 40
       bench_encoder.py time --layer 5 --expert 40 --words 32,48,64 --backend metal [--kernel v1|v2]
       bench_encoder.py stream --words 32 --count 6 --inflight 2   (throughput, v2 only)
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))
CACHE = Path.home() / "mlx-models/.ds41-exl3-encoder-20261109/bench-inputs"


def prepare(layer, expert):
    import convert_experts as C

    ids = np.load(C.TRACE)["ids"]
    X = np.load(C.CAP / "x_by_layer" / f"L{layer:02d}.npy", mmap_mode="r")
    sel = np.where((ids[:, layer] == expert).any(-1))[0]
    cal = sel[C.EVAL_ROWS:]
    art = C.inspect_artifact(C.MODEL)
    raw = C.read_tensors(art, C.expert_names(layer, expert))
    W = {k: C.decode_mxfp4(raw[f"layers.{layer}.ffn.experts.{expert}.{p}.weight"],
                           raw[f"layers.{layer}.ffn.experts.{expert}.{p}.scale"]) for k, p in C.PROJECTIONS.items()}
    xc = C.fp8(X[cal].astype(np.float32))
    if len(cal) < C.MIN_CALIB:
        raise SystemExit("bench uses a top expert (no layer-sample mixing)")
    h_in, h_mid = C.gram(xc), C.gram(C.swiglu_mid(xc, W["gate"], W["up"]))
    CACHE.mkdir(parents=True, exist_ok=True)
    dst = CACHE / f"L{layer}E{expert}.npz"
    np.savez(dst, gate=W["gate"], up=W["up"], down=W["down"], h_in=h_in, h_mid=h_mid)
    print(json.dumps({"saved": str(dst), "shapes": {k: list(v.shape) for k, v in W.items()},
                      "calib_rows": int(len(cal))}))


def time_runs(layer, expert, words_list, backend, kernel, probe, repeat):
    import exl3_encode as E

    if kernel:
        import viterbi_metal

        viterbi_metal.KERNEL_VERSION = kernel
    f = np.load(CACHE / f"L{layer}E{expert}.npz")
    items = [(f["gate"].T, f["h_in"]), (f["up"].T, f["h_in"]), (f["down"].T, f["h_mid"])]
    for words in words_list:
        for r in range(repeat):
            t0 = time.time()
            enc = E.encode_many(items, words, seed=layer * 1000 + expert, probe=probe, backend=backend)
            dt = time.time() - t0
            print(json.dumps({"layer": layer, "expert": expert, "words": words, "bits_per_weight": words / 16,
                              "backend": backend, "kernel": kernel or "default", "probe": probe, "repeat": r,
                              "wall_s": round(dt, 2), **_timing(enc),
                              "tiles": int(sum(e["trellis"].shape[0] * e["trellis"].shape[1] for e in enc)),
                              "sha256": _sha(enc)}), flush=True)


def _timing(enc):
    """Sweep 2026-11-09 EXL-6: the CPU / v1 lockstep path stamps one whole-expert split on every
    projection; the v2 kernel streams each projection as its own unit, so report all three."""
    t = [e["stats"].get("timing") for e in enc]
    if all(x == t[0] for x in t):
        return {"timing": t[0], "timing_scope": "expert"}
    return {"timing": t, "timing_scope": "per_projection"}


def _sha(enc):
    h = hashlib.sha256()
    for e in enc:
        h.update(np.ascontiguousarray(e["trellis"]).tobytes())
        h.update(e["suh"].tobytes())
        h.update(e["svh"].tobytes())
    return h.hexdigest()


def stream_runs(layer, expert, words_list, count, inflight, probe, group="unit"):
    """Throughput: the cached expert encoded ``count`` times back to back through
    encode_experts (same seed, so every unit must hash like the single run)."""
    import exl3_encode as E

    path = CACHE / f"L{layer}E{expert}.npz"

    def load():
        f = np.load(path)
        return [(f["gate"].T, f["h_in"]), (f["up"].T, f["h_in"]), (f["down"].T, f["h_mid"])], layer * 1000 + expert

    for words in words_list:
        t0 = time.time()
        shas, timings, clock = [], [], {}
        for enc in E.encode_experts((load for _ in range(count)), words, inflight=inflight, group=group, probe=probe,
                                    clock=clock):
            shas.append(_sha(enc))
            timings.append(enc[0]["stats"]["timing"])
        dt = time.time() - t0
        print(json.dumps({"layer": layer, "expert": expert, "words": words, "bits_per_weight": words / 16,
                          "mode": "stream", "count": count, "inflight": inflight, "group": group, "probe": probe,
                          "wall_s": round(dt, 2), "per_expert_s": round(dt / count, 2),
                          "main_thread": {"gpu_wait_s": round(clock["wait_s"], 2),
                                          "prep_wait_s": round(clock["prep_wait_s"], 2), "calls": clock["calls"]},
                          "distinct_sha256": sorted(set(shas)), "unit_timing": timings}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "time", "stream"])
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--expert", type=int, default=40)
    ap.add_argument("--words", default="32,48,64")
    ap.add_argument("--backend", default="metal")
    ap.add_argument("--kernel", default="")
    ap.add_argument("--probe", type=int, default=32)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--inflight", type=int, default=2)
    ap.add_argument("--group", default="unit", choices=["unit", "job"])
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.layer, a.expert)
    elif a.cmd == "stream":
        stream_runs(a.layer, a.expert, [int(x) for x in a.words.split(",")], a.count, a.inflight, a.probe, a.group)
    else:
        time_runs(a.layer, a.expert, [int(x) for x in a.words.split(",")], a.backend, a.kernel, a.probe, a.repeat)


if __name__ == "__main__":
    main()
