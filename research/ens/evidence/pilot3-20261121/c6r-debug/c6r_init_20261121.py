"""c6r gradient training on the frozen pilot-2 c7 vs reader init scale (is the always-clipping init the failure?)."""
import random, dataclasses, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from research.ens import common as C, families as F
from research.ens.model import BusCall, value_logits, bounded_update
from research.ens.train_core import setup_trainable, build_batch
for site, tau, scale in (("ingress", 1.5, 1.0), ("ingress", 1.5, 0.01), ("early", 1.5, 0.01), ("ingress", 0.5, 0.01)):
    core, ck = C.load_core("evidence/pilot2-20261121/core-c7")
    core.cfg = dataclasses.replace(core.cfg, tau=tau)
    p = core.ports[site]
    p.reader.weight = p.reader.weight * scale
    setup_trainable(core, "c6r", site)
    M = C.codebook(64); fams = F.split("train"); rng = random.Random(1); j = random.Random(2)
    def lf(m, t, q, f, a, ans, msg):
        return nn.losses.cross_entropy(value_logits(m(t, BusCall(site, q, f, msg)), a), ans, reduction="mean")
    g = nn.value_and_grad(core, lf); opt = optim.Adam(learning_rate=1e-3)
    for it in range(1001):
        t, q, f, a, ans, msg = build_batch(rng, j, fams, 128, "c6r", 0.0, M)
        if it == 0:
            h = core.backbone.embed_tokens(t)[mx.arange(128), f]
            d = p.gate * p.reader(msg)
            clip = mx.mean((mx.linalg.norm(d, axis=-1) > tau * mx.linalg.norm(h, axis=-1)).astype(mx.float32)).item()
        l, gr = g(core, t, q, f, a, ans, msg); opt.update(core, gr); mx.eval(core.parameters(), opt.state)
        if it % 250 == 0:
            print(site, tau, scale, "clip@init %.2f" % clip, it, round(l.item(), 3), "gate", round(p.gate.item(), 3), flush=True)
