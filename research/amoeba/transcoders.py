"""Per-layer TopK skip transcoders for a dense donor MLP.

    a      = TopK(ReLU(W_e (x/sx) + b_e))            features, (F,)
    y_hat  = sy * (W_d a + W_s (x/sx) + b_d)          predicted MLP output

x is the MLP input (post-attention RMSNorm output), y the MLP output. sx and sy
are per-layer scalars (mean vector norm / sqrt(d)) fixed from the first batch,
so every layer trains at the same scale. Decoder rows are kept unit-norm, so a
feature's activation is its contribution magnitude.

Interpretability infrastructure for box extraction, not amoeba training: its
tokens are reported separately and never charged to the amoeba.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


class Transcoder(nn.Module):
    def __init__(self, d: int, F: int, k: int):
        super().__init__()
        self.d, self.F, self.k = d, F, k
        s = d ** -0.5
        self.W_e = mx.random.normal((F, d)) * s
        self.b_e = mx.zeros((F,))
        self.W_d = self.W_e.T / mx.linalg.norm(self.W_e.T, axis=0, keepdims=True)   # (d, F) unit columns
        self.W_s = mx.zeros((d, d))
        self.b_d = mx.zeros((d,))
        self.sx = mx.array(1.0)
        self.sy = mx.array(1.0)

    def encode(self, x):
        pre = (x / self.sx) @ self.W_e.T + self.b_e
        pre = mx.maximum(pre, 0)
        thr = mx.stop_gradient(mx.topk(pre, self.k, axis=-1)[..., :1])        # k-th largest value
        return mx.where(pre >= thr, pre, 0.0)

    def decode(self, a, x):
        return self.sy * (a @ self.W_d.T + (x / self.sx) @ self.W_s.T + self.b_d)

    def __call__(self, x):
        a = self.encode(x)
        return self.decode(a, x), a

    def feature_write(self, ids):
        """Residual-space write vectors (sy * W_d[:, f]) for the given feature ids."""
        return self.sy * self.W_d[:, ids]


def normalize_decoder(tc: Transcoder):
    tc.W_d = tc.W_d / mx.maximum(mx.linalg.norm(tc.W_d, axis=0, keepdims=True), 1e-8)


def save(tcs: dict[int, Transcoder], out: Path, meta: dict):
    out.mkdir(parents=True, exist_ok=True)
    for i, tc in tcs.items():
        mx.save_safetensors(str(out / f"layer{i:02d}.safetensors"),
                            {k: v for k, v in tc.parameters().items()} | {"sx": tc.sx, "sy": tc.sy})
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


def load(out: Path) -> dict[int, Transcoder]:
    meta = json.loads((out / "meta.json").read_text())
    tcs = {}
    for i in meta["layers"]:
        tc = Transcoder(meta["d"], meta["F"], meta["k"])
        w = mx.load(str(out / f"layer{i:02d}.safetensors"))
        tc.sx, tc.sy = w.pop("sx"), w.pop("sy")
        tc.update(nn.utils.tree_unflatten(list(w.items())))
        tcs[i] = tc
    mx.eval([t.parameters() for t in tcs.values()])
    return tcs


def fvu(y, y_hat):
    yc = y - y.mean(axis=0, keepdims=True)
    return mx.sum((y - y_hat) ** 2) / mx.sum(yc ** 2)


def make_trainer(tc: Transcoder, lr: float):
    tc.freeze(keys=["sx", "sy"])
    opt = optim.Adam(learning_rate=lr, betas=[0.9, 0.999])

    def loss_fn(model, x, y):
        y_hat, a = model(x)
        return mx.mean(((y - y_hat) / model.sy) ** 2), a

    step_fn = nn.value_and_grad(tc, loss_fn)

    def step(x, y):
        (loss, a), g = step_fn(tc, x, y)
        g = dict(nn.utils.tree_flatten(g))
        # remove the gradient component parallel to each decoder column (keeps unit norm under Adam)
        Wd = tc.W_d
        gd = g["W_d"]
        g["W_d"] = gd - Wd * mx.sum(gd * Wd, axis=0, keepdims=True)
        opt.update(tc, nn.utils.tree_unflatten(list(g.items())))
        normalize_decoder(tc)
        return loss, a
    return step, opt
