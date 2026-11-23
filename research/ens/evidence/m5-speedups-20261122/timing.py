"""M5 GPU timing for the ENS speed-ups (one process, synthetic data, <8 GB, self-capped at 60 min).

  A. contributor tiles: 4 tiles one after another (contrib.train) vs batched (contrib_batched.train),
     same core, same 300 steps, batch 64 each
  B. GDN: forward+backward per step, sequential scan vs chunked (C=32, 64), GGGGA, B=32, T=512
Writes timing.json in this directory with the commit and origin/main revisions.
"""
import argparse, json, os, subprocess, sys, tempfile, time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn

HERE = Path(__file__).resolve().parent
T_START = time.time()
CAP_S = 60 * 60


def guard():
    if time.time() - T_START > CAP_S:
        raise SystemExit("self-cap reached")


def git(*a):
    return subprocess.check_output(["git", *a], text=True).strip()


res = {"commit": git("rev-parse", "HEAD"), "origin_main": git("rev-parse", "origin/main"),
       "device": str(mx.default_device()), "host": os.uname().nodename}
from research.ens import train_core, contrib, contrib_batched
work = Path(tempfile.mkdtemp(dir=HERE))
train_core.main(["--arm", "c6", "--seed", "0", "--steps", "20", "--batch", "64", "--warmup", "5",
                 "--log-every", "100", "--out", str(work / "core")])
fams = ["program.ADD.h1", "graph.6-2.h1", "fst.3-1.h1", "graph.4-2.h1"]
common = ["--core", str(work / "core"), "--kind", "mlp", "--seed", "0", "--steps", "300", "--batch", "64",
          "--log-every", "300"]
t0 = time.time()
for f in fams:
    contrib.main(common + ["--family", f, "--out", str(work / f"seq-{f}")])
    guard()
seq_s = time.time() - t0
t0 = time.time()
contrib_batched.main(common + ["--families", *fams, "--out-root", str(work), "--prefix", "bat-"])
bat_s = time.time() - t0
res["tiles"] = {"tiles": 4, "steps": 300, "batch": 64, "sequential_s": round(seq_s, 2), "batched_s": round(bat_s, 2),
                "speedup": round(seq_s / bat_s, 2)}
print(res["tiles"], flush=True)
guard()
from research.ens.hybrid import HybridConfig, HybridCore
from research.ens import longctx as LC
from research.ens.train_hybrid import batch, value_logits
rows = LC.make_rows(32, L=512, k=1, n_pairs=48, seed=0)
t, a, y = batch(rows)
gdn = {}
for C in (0, 32, 64):
    mx.random.seed(0)
    core = HybridCore(HybridConfig(d=128, heads=4, ffn=512, seq_len=512, pattern="GGGGA", gdn_chunk=C))
    g = nn.value_and_grad(core, lambda m: nn.losses.cross_entropy(value_logits(m(t), a), y, reduction="mean"))
    l, gr = g(core); mx.eval(l, gr)
    n = 3 if C == 0 else 10
    t0 = time.time()
    for _ in range(n):
        l, gr = g(core); mx.eval(l, gr)
        guard()
    gdn[str(C)] = {"s_per_step": round((time.time() - t0) / n, 4), "loss": round(l.item(), 5)}
    print("gdn_chunk", C, gdn[str(C)], flush=True)
res["gdn_fwd_bwd_B32_T512"] = gdn
res["peak_mem_gb"] = round(mx.get_peak_memory() / 1e9, 3)
res["wall_s"] = round(time.time() - T_START, 1)
(HERE / "timing.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res), flush=True)
