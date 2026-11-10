"""First-use bit gate of the GLM-5.3 HC mega kernels, plus their speed (GPU)."""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src

import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-revision", default="unrecorded")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    import mlx.core as mx
    import numpy as np

    from mlx2.runtime.models import glm53_hc_mega as mega

    record = {"source_revision": args.source_revision}
    for lanes, flat_fma in ((True, False), (True, True)):
        mega.LANES = lanes
        mega.FLAT_FMA = flat_fma
        mega._kernel.cache_clear()
        mega.self_check.cache_clear()
        try:
            result = mega.self_check()
        except ValueError as error:
            result = str(error)
        record[f"self_check_lanes{int(lanes)}_fma{int(flat_fma)}"] = result
        print(f"self_check lanes={lanes} flat_fma={flat_fma}:", result, flush=True)
    rng = np.random.default_rng(1)
    h = mx.array(rng.normal(size=(1, 1, 4, 4096)).astype(np.float32))
    mixes = mx.array(rng.normal(size=(1, 1, 24)).astype(np.float32))
    scale = mx.array(np.ones(3, np.float32))
    base = mx.array(np.zeros(24, np.float32))
    w = mx.array(np.ones(4096, np.float32))
    br = mx.array(rng.normal(size=(1, 1, 4096)).astype(np.float32))

    def chain(post_fn, pre_fn):
        x = h
        outs = []
        for _ in range(90):
            xn, _p, q, c = post_fn(x, mixes, scale, base, w, rms_eps=1e-5)
            x, z = pre_fn(br, x, q, c, rms_eps=1e-5)
            outs += [xn, z]
        return [x] + outs

    for label, fns in (("mega", (mega.post, mega.pre)), ("reference", (mega.reference_post, mega.reference_pre)),
                       ("post only", (mega.post, mega.reference_pre)), ("pre only", (mega.reference_post, mega.pre))):
        fn = lambda fns=fns: chain(*fns)
        mx.eval(fn())
        best = 1e9
        for _ in range(7):
            t0 = time.perf_counter()
            mx.eval(fn())
            best = min(best, time.perf_counter() - t0)
        record[label + "_ms_per_90_halves"] = best * 1e3
        print(f"{label:10s} {best*1e3:7.3f} ms per 90 half-layers (no mixing matmul)", flush=True)
    (args.out / "hc_mega.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
