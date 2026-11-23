"""Loss along init -> closed-form reader (ingress, tau 1.5) on frozen c7, and the gradient's direction."""
import random, dataclasses, mlx.core as mx, mlx.nn as nn
from research.ens import common as C, families as F, vocab as V
from research.ens.model import BusCall, value_logits
from research.ens.train_core import setup_trainable, build_batch
core, ck = C.load_core("evidence/pilot2-20261121/core-c7")
core.cfg = dataclasses.replace(core.cfg, tau=1.5)
site = "ingress"; p = core.ports[site]
E = core.backbone.embed.weight; M = C.codebook(64)
Rs = mx.linalg.solve(M, E[V.V0:V.V0 + 64] - E[V.NEUTRAL][None, :], stream=mx.cpu).T
R0 = p.reader.weight * 0.01
setup_trainable(core, "c6r", site)
rng = random.Random(1); j = random.Random(2)
t, q, f, a, ans, msg = build_batch(rng, j, F.split("train"), 512, "c6r", 0.0, M)
_, _, _, _, _, clean = build_batch(random.Random(1), random.Random(2), F.split("train"), 512, "c6r", 0.0, M)
def lf(m, t, q, f, a, ans, msg):
    return nn.losses.cross_entropy(value_logits(m(t, BusCall(site, q, f, msg)), a), ans, reduction="mean")
g = nn.value_and_grad(core, lf)
z = mx.array([0])
for al in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
    p.reader.weight = R0 + al * (Rs - R0); p.gate = mx.array([1.0])
    l, gr = g(core, t, q, f, a, ans, msg)
    gR = gr["ports"][site]["reader"]["weight"]
    cos = mx.sum(-gR * (Rs - R0)) / (mx.linalg.norm(gR) * mx.linalg.norm(Rs - R0))
    acc = mx.mean(mx.argmax(value_logits(core(t, BusCall(site, q, f, msg)), a), -1) == ans).item()
    print("alpha %.2f loss %.3f acc %.3f cos(-grad, R*-R) %.3f |grad| %.2e" % (al, l.item(), acc, cos.item(), mx.linalg.norm(gR).item()))
print("|R*| %.2f |R0| %.2f" % (mx.linalg.norm(Rs).item(), mx.linalg.norm(R0).item()))
