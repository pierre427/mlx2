#!/usr/bin/env python3
"""Does re-reading a just-read Metal buffer get faster?  (expert-prefetch question)

Original mlx2 script.  Evidence for docs/hy4-flash/verify-phasing.md: an
early "touch" of layer L+1's experts can only help if the bytes it reads stay
somewhere faster than DRAM until layer L+1 runs.  Weights are already wired
(``weight_residency``: MTLResidencySet), so there is no page-in to hide; the
only candidate is the GPU's last-level / system-level cache.  This probe
measures, per buffer size, the read time of a buffer right after a 512 MB
flush read (``first``) and right after reading the same buffer (``second``),
one ``mx.eval`` each, median of ``--reps``.  ``second / first`` well below 1
means the buffer (partly) survived in cache; about 1 means it did not.

Hy4-flash-MLX-4.6bit-mlx2 routed experts are ~4 MB each (3 x [1536 x 1536]
4-bit + bf16 scales/biases; ~2 MB on its q2 layers) and a B=1 verify round
touches ~61 experts per layer, ~243 MB per layer.

Memory: flush buffer + one probe buffer (<= 768 MB).  Not publishable
statistics (no thermal gating); the receipt lists foreign GPU processes it
could see.

    PYTHONPATH="$PWD/src" .venv/bin/python scripts/probe_metal_reread.py --i-own-the-gpu --out probe.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--i-own-the-gpu", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes-mb", nargs="+", type=int, default=[4, 16, 32, 64, 128, 256])
    ap.add_argument("--flush-mb", type=int, default=512)
    ap.add_argument("--reps", type=int, default=7)
    a = ap.parse_args(argv)
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")
    if max(a.sizes_mb) + a.flush_mb > 1024:
        ap.error("keep flush + probe buffers at or under 1 GB")
    out = Path(a.out)
    if out.exists():
        ap.error(f"{out} exists; refusing to overwrite evidence")
    import mlx.core as mx

    def buf(mb, seed):
        x = mx.random.uniform(shape=(mb * (1 << 20) // 2,), key=mx.random.key(seed)).astype(mx.float16)
        mx.eval(x)
        return x

    def read(x):
        mx.synchronize()
        t = time.perf_counter()
        mx.eval(mx.sum(x))
        return time.perf_counter() - t

    flush = buf(a.flush_mb, 1)
    tiny = mx.ones((8,), dtype=mx.float16)
    mx.eval(tiny)
    for _ in range(5):
        read(tiny)
    # Launch + sync floor of one read: subtracted to get the bytes' own time.
    floor = statistics.median(read(tiny) for _ in range(31))
    rows = []
    for mb in a.sizes_mb:
        x = buf(mb, 2 + mb)
        read(x), read(flush)  # warm the kernels
        first, second = [], []
        for _ in range(a.reps):
            read(flush)
            first.append(read(x))
            second.append(read(x))
        f, s = statistics.median(first), statistics.median(second)
        fm, sm = max(f - floor, 1e-7), max(s - floor, 1e-7)
        rows.append({"size_mb": mb, "first_ms": round(f * 1e3, 4), "second_ms": round(s * 1e3, 4),
                     "floor_ms": round(floor * 1e3, 4),
                     "marginal_second_over_first": round(sm / fm, 4),
                     "marginal_first_gb_s": round(mb / 1024 / fm, 1),
                     "marginal_second_gb_s": round(mb / 1024 / sm, 1)})
        print(json.dumps(rows[-1]), flush=True)
        del x
        mx.clear_cache()
    ps = subprocess.run(["ps", "-axo", "pid,rss,comm,args"], capture_output=True, text=True).stdout
    foreign = [line.strip()[:200] for line in ps.splitlines()[1:]
               if "python" in line and "probe_metal_reread" not in line and int(line.split()[1]) > 4 << 20]
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    doc = {"schema": "mlx2.probe_metal_reread.v1", "mlx2_commit": git, "mlx": mx.__version__,
           "device_info": mx.device_info(), "finished": datetime.now(UTC).isoformat(timespec="seconds"),
           "args": vars(a), "rows": rows, "foreign_large_python_processes": foreign,
           "method": "one mx.eval(mx.sum(x)) per read (includes one command-buffer launch + sync, "
                     "the floor_ms of an 8-element read, subtracted in the marginal_* columns); "
                     "first = after a flush read, second = immediately after reading x"}
    out.write_text(json.dumps(doc, indent=1, default=str))
    return doc


if __name__ == "__main__":
    main()
