"""c6r on frozen c7 at ingress, tau 1.5: gradient training with and without provider jitter, plus a jitter-aware ridge reader."""
import random, dataclasses, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from research.ens import common as C, families as F, vocab as V
from research.ens.model import BusCall, value_logits
from research.ens.train_core import setup_trainable, build_batch
site = "ingress"; M = C.codebook(64); fams = F.split("train")
def fresh():
    core, _ = C.load_core("evidence/pilot2-20261121/core-c7")
    core.cfg = dataclasses.replace(core.cfg, tau=1.5)
    core.ports[site].reader.weight = core.ports[site].reader.weight * 0.01
    setup_trainable(core, "c6r", site)
    return core
def acc(core, jit):
    rng = random.Random(9); j = random.Random(8)
    t, q, f, a, ans, msg = build_batch(rng, j, fams, 1024, "c6r", 0.0, M)
    if not jit:
        msg = M[mx.array([r for r in range(1)] * 0 + [0])] if False else msg
    return msg
def run(core, t, q, f, a, msg):
    return mx.argmax(value_logits(core(t, BusCall(site, q, f, msg)), a), -1)
rng = random.Random(9); j = random.Random(8)
rows = [F.sample_row(fams[rng.randrange(len(fams))], rng) for _ in range(1024)]
tE, qE, fE, aE, ansE, zE = C.batch(rows, "bus")
clean = M[zE]; jit = C.jitter(clean, j)
for use_jitter in (False, True):
    core = fresh()
    def lf(m, t, q, f, a, ans, msg):
        return nn.losses.cross_entropy(value_logits(m(t, BusCall(site, q, f, msg)), a), ans, reduction="mean")
    g = nn.value_and_grad(core, lf); opt = optim.Adam(learning_rate=1e-3)
    r2 = random.Random(1); j2 = random.Random(2)
    for it in range(1501):
        rr = [F.sample_row(fams[r2.randrange(len(fams))], r2) for _ in range(128)]
        t, q, f, a, ans, z = C.batch(rr, "bus")
        msg = C.jitter(M[z], j2) if use_jitter else M[z]
        l, gr = g(core, t, q, f, a, ans, msg); opt.update(core, gr); mx.eval(core.parameters(), opt.state)
    print("trained jitter=%s: acc clean %.3f, acc jittered %.3f, last loss %.3f" % (
        use_jitter, mx.mean(run(core, tE, qE, fE, aE, clean) == ansE).item(),
        mx.mean(run(core, tE, qE, fE, aE, jit) == ansE).item(), l.item()), flush=True)
# ridge reader under the jitter model: E[m m^T] = s2 * (M M^T) * keep-ish + noise; solve on samples
core = fresh(); E = core.backbone.embed.weight
T = E[V.V0:V.V0 + 64] - E[V.NEUTRAL][None, :]
zz = mx.array([i % 64 for i in range(64 * 200)]); X = C.jitter(M[zz], random.Random(3)); Y = T[zz]
for lam in (1e-2, 1.0, 10.0):
    Rt = mx.linalg.solve(X.T @ X + lam * mx.eye(64), X.T @ Y, stream=mx.cpu)
    core.ports[site].reader.weight = Rt.T; core.ports[site].gate = mx.array([1.0])
    print("ridge lam %g: acc clean %.3f, acc jittered %.3f" % (lam, mx.mean(run(core, tE, qE, fE, aE, clean) == ansE).item(),
          mx.mean(run(core, tE, qE, fE, aE, jit) == ansE).item()), flush=True)
