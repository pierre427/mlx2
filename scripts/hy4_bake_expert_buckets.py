#!/usr/bin/env python3
"""Bake an aMoEba box into a cut-down Hy4-flash artifact (width-bucketed routed experts), CPU and disk only.

    hy4_bake_expert_buckets.py --model SRC --mask BUNDLE --out NEW --classes 24,22,20 [--block 64]

Streams one expert layer at a time (raw safetensors bytes, no requantization; see
src/mlx2/runtime/models/hy_v4_flash_buckets.py). Unchanged files become hard links (never modified); the source
artifact is only read. A memory watchdog stops the bake before each layer when free + inactive + purgeable memory is
under --min-avail-gib or swap grew more than --max-swap-growth-mib since the start; the partial output keeps a
``.part`` file at most and is refused as a target until removed by hand.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def avail_gib() -> float:
    out = subprocess.check_output(["vm_stat"], text=True)
    page = int(out.split("page size of ")[1].split()[0])
    pages = 0
    for line in out.splitlines():
        if line.startswith(("Pages free:", "Pages inactive:", "Pages purgeable:")):
            pages += int(line.split(":")[1].strip().rstrip("."))
    return pages * page / 2**30


def swap_mib() -> float:
    out = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
    return float(out.split("used = ")[1].split("M")[0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--mask", type=Path, required=True, help="aMoEba neuron-mask bundle of --model")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--classes", required=True, help="bucket widths in blocks, e.g. 24,22,20")
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--min-avail-gib", type=float, default=16.0)
    ap.add_argument("--max-swap-growth-mib", type=float, default=1024.0)
    ap.add_argument("--min-free-disk-gib", type=float, default=50.0, help="stop before a layer when the output volume has less free space")
    a = ap.parse_args(argv)
    from mlx2.runtime import amoeba_neuron_mask as M
    from mlx2.runtime.models import hy_v4_flash_buckets as B

    s0 = swap_mib()
    t0 = time.time()

    def watchdog(stage):
        import shutil
        disk = shutil.disk_usage(a.out).free / 2**30
        if disk < a.min_free_disk_gib:
            raise SystemExit(f"bake watchdog at {stage}: {disk:.0f} GiB free on the output volume "
                             f"(min {a.min_free_disk_gib:.0f}); stopped")
        av, sw = avail_gib(), swap_mib()
        if av < a.min_avail_gib or sw - s0 > a.max_swap_growth_mib:
            raise SystemExit(f"bake watchdog at {stage}: available {av:.1f} GiB (min {a.min_avail_gib}), "
                             f"swap {s0:.0f} -> {sw:.0f} MiB (max growth {a.max_swap_growth_mib:.0f}); stopped")

    def log(msg):
        print(f"[bake {time.time() - t0:7.1f}s avail {avail_gib():5.1f} GiB swap {swap_mib():6.0f} MiB] {msg}",
              flush=True)

    classes = [int(c) for c in a.classes.split(",")]
    rec = B.bake(a.model, a.out, M.NeuronMaskBundle.load(a.mask), classes=classes, block=a.block,
                 watchdog=watchdog, log=log)
    rec["swap_mib"] = {"start": s0, "end": swap_mib()}
    (a.out / "mlx2_bake_receipt.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k not in ("linked",)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
