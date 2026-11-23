"""Read-only: per-family ordinary / native-channel accuracy on the seed 0-2 confirmation cores (CPU)."""
import json, random, mlx.core as mx
from research.ens import common as C, families as F
from research.ens.model import BusCall, value_logits
fams = F.split("train"); out = {}
for seed in (0, 1, 2):
    for arm in ("c6", "c7"):
        core, ck = C.load_core(f"evidence/confirm-20261122/seed{seed}/core-{arm}"); core.freeze()
        M = C.codebook(core.cfg.bus_width)
        for fam in fams:
            rng = random.Random(1234)
            rows = [F.sample_row(fam, rng) for _ in range(400)]
            tok, q, f, a, ans, zs = C.batch(rows, "ordinary")
            o = float(mx.mean(mx.argmax(value_logits(core(tok), a), -1) == ans).item())
            mode = "typed" if arm == "c7" else "bus"
            tok, q, f, a, ans, zs = C.batch(rows, mode)
            bus = BusCall("early", q, f, M[zs]) if mode == "bus" else None
            n = float(mx.mean(mx.argmax(value_logits(core(tok, bus), a), -1) == ans).item())
            out.setdefault(fam.fid, {})[f"s{seed}{arm}"] = (round(o, 3), round(n, 3))
cols = [f"s{s}{a}" for s in (0, 1, 2) for a in ("c6", "c7")]
print("family".ljust(22), *[c.rjust(11) for c in cols])
for k, v in out.items():
    print(k.ljust(22), *[f"{v[c][0]:.2f}/{v[c][1]:.2f}".rjust(11) for c in cols])
print(json.dumps(out))
