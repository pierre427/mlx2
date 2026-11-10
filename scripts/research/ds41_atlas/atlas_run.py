#!/usr/bin/env python3
"""Coordinator for the DS4.1 Source expert token atlas: work queue over (layer, experts) units.

Each unit's gate/up MXFP4 codes (w1, w3 + E8M0 scales) are cut from the sealed official shards
(checked_file, stamp-bound) into a small staging safetensors, run by the local M5 worker or shipped
to a remote worker over SSH (copied in, result copied back, remote copies deleted).
Resumable: experts already present in a unit json under OUT/units are skipped.

env HOSTS = comma list of remotes below (default "m2,m3"); LOCAL=1 runs the M5 worker.
usage: atlas_run.py MODEL OUT
"""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[3] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
import glob, json, os, queue, subprocess, sys, threading, time
from pathlib import Path
import numpy as np
import mlx.core as mx
from mlx2.adapters.deepseek_v41_source_artifact import inspect_artifact, checked_file

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = f"{HERE}/atlas_worker.py"
PY = sys.executable
REMOTES = {
    "m2": {"ssh": ["ssh", "-i", str(Path.home() / ".ssh/m3_node"), "-o", "BatchMode=yes", "user@169.254.68.18"],
           "py": "mlx2-0324-venv/bin/python", "tmp": "ds41_atlas_tmp", "unit": 16},
    "m3": {"ssh": ["ssh", "-o", "BatchMode=yes", "m3"], "py": "mlx2-0324-venv/bin/python", "tmp": "ds41_atlas_tmp", "unit": 8},
}
LOCAL_UNIT = 32
lock = threading.Lock()
MODEL, OUT = sys.argv[1], sys.argv[2]
LITE = f"{OUT}/lite"
STAGE = str(Path.home() / ".cache/mlx2/ds41-atlas-stage")
ART = inspect_artifact(MODEL)
log = open(f"{OUT}/atlas_run.log", "a")


def say(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    with lock:
        print(s, flush=True); log.write(s + "\n"); log.flush()


def stage(layer, experts, path):
    arrays = {}
    for e in experts:
        for part in ("w1", "w3"):
            for field, key, dt in (("weight", "q", "<u4"), ("scale", "s", np.uint8)):
                t = ART["tensors"][f"layers.{layer}.ffn.experts.{e}.{part}.{field}"]
                p = checked_file(ART["root"], t["file"], ART["seals"][t["file"]])
                with open(p, "rb") as f:
                    f.seek(t["offset"]); raw = f.read(t["size"])
                shape = t["shape"] if key == "s" else (t["shape"][0], t["shape"][1] // 4)
                arrays[f"e{e}.{part}.{key}"] = mx.array(np.frombuffer(raw, dt).reshape(shape))
    mx.save_safetensors(path, arrays)


def done_set():
    out = set()
    for p in glob.glob(f"{OUT}/units/*.json"):
        u = json.load(open(p))
        out |= {(u["layer"], e) for e in u["experts"]}
    return out


def record(u, npz, host, secs):
    name = f"L{u['layer']:02d}_{u['experts'][0]:03d}-{u['experts'][-1]:03d}_{host}"
    os.replace(npz, f"{OUT}/units/{name}.npz")
    json.dump({**u, "host": host, "seconds": round(secs, 1), "npz": f"{name}.npz",
               "checkpoint": f"DeepSeek-V4.1-Flash-1017-Source@{ART['identity']['fingerprint']}",
               "time": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(f"{OUT}/units/{name}.json", "w"))


def worker(host, q):
    cfg = REMOTES.get(host)
    if cfg:
        ssh, tmp = cfg["ssh"], cfg["tmp"]; dest = ssh[-1]
        scp = ["scp", "-q", "-o", "BatchMode=yes"] + (["-i", ssh[2]] if ssh[1] == "-i" else [])
        subprocess.run(ssh + [f"mkdir -p {tmp}"], check=True)
        if subprocess.run(ssh + [f"test -f {tmp}/embed.safetensors"]).returncode:
            say(host, "copying embedding (1.3 GB)")
            subprocess.run(scp + [f"{LITE}/embed.safetensors", f"{dest}:{tmp}/"], check=True)
        subprocess.run(scp + [WORKER, f"{dest}:{tmp}/"], check=True)
    n = cfg["unit"] if cfg else LOCAL_UNIT
    while True:
        try:
            u = q.get_nowait()
        except queue.Empty:
            return
        L = u["layer"]
        parts = [{**u, "experts": u["experts"][i:i + n]} for i in range(0, len(u["experts"]), n)]
        for k, part in enumerate(parts):
            t0 = time.time(); upath = f"{STAGE}/unit_{host}.safetensors"; npz = f"{OUT}/tmp_{host}.npz"
            try:
                stage(L, part["experts"], upath)
                if cfg:
                    subprocess.run(ssh + [f"rm -rf {tmp}/u && mkdir -p {tmp}/u"], check=True)
                    subprocess.run(scp + [f"{LITE}/layer_{L:02d}.safetensors", upath, f"{dest}:{tmp}/u/"], check=True)
                    r = subprocess.run(ssh + [f"{cfg['py']} {tmp}/atlas_worker.py {tmp}/u/layer_{L:02d}.safetensors "
                                              f"{tmp}/embed.safetensors {tmp}/u/{os.path.basename(upath)} {tmp}/u/out.npz"],
                                       capture_output=True, text=True)
                    if r.returncode:
                        raise RuntimeError(r.stderr[-400:])
                    subprocess.run(scp + [f"{dest}:{tmp}/u/out.npz", npz], check=True)
                else:
                    r = subprocess.run([PY, WORKER, f"{LITE}/layer_{L:02d}.safetensors", f"{LITE}/embed.safetensors", upath, npz],
                                       capture_output=True, text=True)
                    if r.returncode:
                        raise RuntimeError(r.stderr[-400:])
                record(part, npz, host, time.time() - t0)
                say(f"{host} L{L:02d} {part['experts'][0]}-{part['experts'][-1]} {len(part['experts'])} experts {time.time() - t0:.0f}s")
            except Exception as exc:
                say(host, "unit failed, returned to queue:", str(exc)[-300:])
                for rest in parts[k:]:
                    q.put(rest)
                return
            finally:
                if cfg:
                    subprocess.run(ssh + [f"rm -rf {tmp}/u"])


def main():
    os.makedirs(f"{OUT}/units", exist_ok=True); os.makedirs(STAGE, exist_ok=True)
    done = done_set()
    q = queue.Queue()
    total = 0
    for L in range(ART["args"].num_hidden_layers):
        es = [e for e in range(ART["args"].n_routed_experts) if (L, e) not in done]
        for i in range(0, len(es), 32):
            q.put({"layer": L, "experts": es[i:i + 32]}); total += len(es[i:i + 32])
    say(f"queue: {total} experts, {len(done)} already done")
    hosts = [h for h in os.environ.get("HOSTS", "m2,m3").split(",") if h]
    if os.environ.get("LOCAL", "1") == "1":
        hosts.append("m5")
    threads = [threading.Thread(target=worker, args=(h, q), daemon=True) for h in hosts]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    say("ATLAS UNITS DONE", len(done_set()))


if __name__ == "__main__":
    main()
