#!/usr/bin/env python3
"""Download deepseek-ai/DeepSeek-V4.1-Flash-1017 @ the pinned Source revision into one
directory whose shards are tiered across two volumes.

Files land on the internal SSD (the canonical model path the Source adapter reads) while it keeps
a free-space floor; the rest go to the external volume and are symlinked into the canonical
directory. NOTE: the Source adapter currently refuses symlinked shards (checked_file lstat/S_ISREG,
O_NOFOLLOW); serving this layout needs the tier-aware file check. Every pinned file is sha256-verified
against src/mlx2/adapters/deepseek_v41_source_pins.json; a mismatch deletes the file and fails.
Resumable: verified files are recorded in tiered-download.json and skipped.
"""
import argparse, hashlib, json, os, shutil, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINS = json.loads((ROOT / "src/mlx2/adapters/deepseek_v41_source_pins.json").read_text())
GiB = 1 << 30


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(64 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--internal", default=os.path.expanduser("~/mlx-models/DeepSeek-V4.1-Flash-1017-Source"))
    ap.add_argument("--external", default="/Volumes/T7/ds41-source")
    ap.add_argument("--internal-floor-gib", type=float, default=40)
    ap.add_argument("--external-floor-gib", type=float, default=20)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    repo, rev = PINS["repository"], PINS["revision"]
    info = HfApi().model_info(repo, revision=rev, files_metadata=True)
    sizes = {s.rfilename: (s.size or 0) for s in info.siblings}
    missing = sorted(set(PINS["files"]) - set(sizes))
    if missing:
        sys.exit(f"pinned files absent upstream: {missing}")
    inner, outer = Path(a.internal), Path(a.external)
    inner.mkdir(parents=True, exist_ok=True)
    state_path = inner / "tiered-download.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"repo": repo, "revision": rev, "files": {}}
    lock = threading.Lock()
    reserved = {"internal": 0, "external": 0}
    # small files first, then shards in order
    order = sorted(sizes, key=lambda n: (n.startswith("model-"), n))
    todo = [n for n in order if n not in state["files"]]
    print(f"{len(sizes)} files, {sum(sizes.values()) / 1e9:.1f} GB; {len(todo)} to fetch", flush=True)

    def place(n):
        size = sizes[n]
        with lock:
            if shutil.disk_usage(inner).free - reserved["internal"] - size > a.internal_floor_gib * GiB:
                reserved["internal"] += size; return "internal"
            outer.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(outer).free - reserved["external"] - size > a.external_floor_gib * GiB:
                reserved["external"] += size; return "external"
        return None

    def fetch(n, tier):
        t0 = time.time()
        d = inner if tier == "internal" else outer
        try:
            p = Path(hf_hub_download(repo, n, revision=rev, local_dir=str(d)))
            pin = PINS["files"].get(n)
            if pin:
                got = sha256(p)
                if got != pin["sha256"] or p.stat().st_size != pin["size"]:
                    p.unlink(); raise RuntimeError(f"{n}: sha256/size mismatch")
            if tier == "external":
                link = inner / n
                link.parent.mkdir(parents=True, exist_ok=True)
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(p)
            return n, tier, str(p), time.time() - t0
        finally:
            with lock:
                reserved[tier] -= sizes[n]

    pool = ThreadPoolExecutor(a.workers)
    pending, it, t0, done_bytes = set(), iter(todo), time.time(), 0
    nxt = next(it, None)
    while nxt is not None or pending:
        while nxt is not None and len(pending) < a.workers:
            tier = place(nxt)
            if tier is None:
                if not pending:
                    print("no space on either volume; waiting 60 s", flush=True); time.sleep(60)
                break
            pending.add(pool.submit(fetch, nxt, tier)); nxt = next(it, None)
        if not pending:
            continue
        fin, pending = wait(pending, timeout=60, return_when=FIRST_COMPLETED)
        for f in fin:
            n, tier, p, dt = f.result()
            state["files"][n] = {"tier": tier, "path": p, "size": sizes[n], "verified": n in PINS["files"]}
            tmp = state_path.with_suffix(".tmp"); tmp.write_text(json.dumps(state, indent=1)); tmp.replace(state_path)
            done_bytes += sizes[n]
            print(f"{time.strftime('%H:%M:%S')} {tier:8s} {n} {sizes[n] / 1e9:.1f} GB {dt:.0f}s "
                  f"total {done_bytes / 1e9:.0f} GB {done_bytes / 1e6 / (time.time() - t0):.0f} MB/s", flush=True)
    print("DS41 SOURCE DOWNLOAD DONE", json.dumps({t: sum(v["size"] for v in state["files"].values() if v["tier"] == t) / 1e9
                                                   for t in ("internal", "external")}), flush=True)


if __name__ == "__main__":
    main()
