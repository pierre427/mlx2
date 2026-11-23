"""CORE-001 on a trained core artifact: with no bus call, the core's logits are bitwise equal to its backbone
reloaded alone (the ordinary graph), at width 1, on a fixed ordinary batch, and a detach/reattach round trip
leaves the ordinary result unchanged. Writes core001.json beside the core.

  python -m research.ens.core001 <core dir>
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from research.ens import common as C
from research.ens import families as F
from research.ens.model import Backbone, BusCall


def check(core_dir) -> dict:
    core, ckpt = C.load_core(core_dir)
    core.freeze()
    rng = random.Random(0)
    fams = F.split("train")
    rows = [F.sample_row(fams[rng.randrange(len(fams))], rng) for _ in range(64)]
    tokens, q, f, a, ans, zs = C.batch(rows, "ordinary")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "bb.safetensors"
        mx.save_safetensors(str(path), dict(tree_flatten(core.backbone.parameters())))
        ordinary = Backbone(core.cfg)
        ordinary.load_weights(str(path))
    base = core(tokens)
    width1 = all(bool(mx.array_equal(core(tokens[i:i + 1]), ordinary(tokens[i:i + 1])).item()) for i in range(8))
    batch_eq = bool(mx.array_equal(base, ordinary(tokens)).item())
    msg = mx.random.normal((tokens.shape[0], core.cfg.bus_width), key=mx.random.key(1))
    from research.ens.contrib import core_site
    site = core_site(core_dir)
    mx.eval(core(tokens, BusCall(site, q, f, msg)))
    reattach = bool(mx.array_equal(core(tokens), base).item())
    res = {"core_checkpoint": ckpt, "width1_bitwise": width1, "batch_bitwise": batch_eq,
           "detach_reattach_bitwise": reattach, "pass": width1 and batch_eq and reattach}
    C.write_json(Path(core_dir) / "core001.json", res)
    return res


if __name__ == "__main__":
    r = check(sys.argv[1])
    print(json.dumps(r))
    sys.exit(0 if r["pass"] else 1)
