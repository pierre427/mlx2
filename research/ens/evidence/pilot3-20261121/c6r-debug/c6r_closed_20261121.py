"""Closed-form ingress reader on the frozen pilot-2 c7: is c6r solvable at all?"""
import random, dataclasses, mlx.core as mx
from research.ens import common as C, families as F, vocab as V
from research.ens.evaluate import evaluate
core, ck = C.load_core("evidence/pilot2-20261121/core-c7"); core.freeze()
print(ck)
E = core.backbone.embed.weight
M = C.codebook(64)
T = E[V.V0:V.V0 + 64] - E[V.NEUTRAL][None, :]
Rt = mx.linalg.solve(M, T, stream=mx.cpu)
print("fit err", float(mx.max(mx.abs(M @ Rt - T)).item()),
      "max |T|/|e_neutral|", float((mx.linalg.norm(T, axis=-1) / mx.linalg.norm(E[V.NEUTRAL])).max().item()))
for tau in (0.5, 1.5, 3.0):
    core.cfg = dataclasses.replace(core.cfg, tau=tau)
    core.ports["ingress"].reader.weight = Rt.T
    core.ports["ingress"].gate = mx.array([1.0])
    out = []
    for fid in ("program.ADD", "graph.6-2", "fst.3-1", "graph.4-2"):
        fam = F.BY_ID[fid]; rng = random.Random(5); rows = [F.sample_row(fam, rng) for _ in range(512)]
        r = evaluate(core, "c6r", "ingress", None, fam, rows)
        out.append("%s=%.3f" % (fid, r["oracle_codebook"]))
    print("tau", tau, " ".join(out))
