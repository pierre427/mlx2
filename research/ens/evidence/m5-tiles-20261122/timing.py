"""M5 retest of the tile trainer (one process, synthetic data): 4 tiles x 2000 steps, batch 64, sequential
contrib.train vs the cached + packed + compiled contrib_batched. Writes timing.json with commit and origin/main."""
import json, os, subprocess, tempfile, time
from pathlib import Path
import mlx.core as mx
HERE = Path(__file__).resolve().parent
git = lambda *a: subprocess.check_output(["git", *a], text=True).strip()
res = {"commit": git("rev-parse", "HEAD"), "origin_main": git("rev-parse", "origin/main"),
       "device": str(mx.default_device()), "host": os.uname().nodename}
from research.ens import train_core, contrib, contrib_batched
work = Path(tempfile.mkdtemp(dir=HERE))
train_core.main(["--arm", "c6", "--seed", "0", "--steps", "20", "--batch", "64", "--warmup", "5", "--log-every", "100",
                 "--out", str(work / "core")])
fams = ["program.ADD.h1", "graph.6-2.h1", "fst.3-1.h1", "graph.4-2.h1"]
common = ["--core", str(work / "core"), "--kind", "mlp", "--seed", "0", "--steps", "2000", "--batch", "64",
          "--log-every", "2000"]
t0 = time.time()
for f in fams:
    contrib.main(common + ["--family", f, "--out", str(work / f"seq-{f}")])
seq = time.time() - t0
t0 = time.time()
contrib_batched.main(common + ["--families", *fams, "--out-root", str(work), "--prefix", "bat-"])
bat = time.time() - t0
res["tiles"] = {"tiles": 4, "steps": 2000, "batch": 64, "sequential_s": round(seq, 2), "batched_s": round(bat, 2),
                "speedup": round(seq / bat, 2), "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 3)}
# parity: batched tiles must equal sequential tiles (coordinator: check parity, not just timing)
par = {}
for f in fams:
    a_ = mx.load(str(work / f"seq-{f}" / "module.safetensors"))
    b_ = mx.load(str(work / f"bat-{f}" / "module.safetensors"))
    par[f] = max(float(mx.max(mx.abs(a_[k] - b_[k])).item()) for k in a_)
res["parity_max_abs_diff"] = par
res["parity_pass"] = all(v < 1e-3 for v in par.values())
for p in work.glob("*/module.safetensors"):
    p.unlink()
(HERE / "timing.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res), flush=True)
