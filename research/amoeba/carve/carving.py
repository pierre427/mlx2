"""Losses, scoring and the straight-through carving loop.

kl_rows, ce_rows, train_mask, topk_mask, random_same_counts: ported unchanged from
scripts/amoeba27b_lib.py at claude/amoeba-27b-80bc6a@37d75f144 (lines 387-399, 694-752).
"""
from __future__ import annotations

import time

import numpy as np

from .ops import peak_gb

CLEAN_STOP_RC = 75


class CleanStop(SystemExit):
    """The --stop-at deadline passed: state is saved, rerun with --resume to continue."""
    def __init__(self, msg):
        super().__init__(CLEAN_STOP_RC)
        self.msg = msg


def _save_ckpt(path, step, params, opt):
    from mlx.utils import tree_flatten
    flat = {"Z": np.array(params["Z"]), "next_step": np.array(step)}
    flat.update({f"opt.{k}": np.array(v) for k, v in tree_flatten(opt.state)})
    tmp = str(path) + ".tmp.npz"
    np.savez(tmp, **flat)
    import os
    os.replace(tmp, path)


def _load_ckpt(path, opt):
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    z = np.load(path)
    opt.state = tree_unflatten([(k[4:], mx.array(z[k])) for k in z.files if k.startswith("opt.")])
    return int(z["next_step"]), {"Z": mx.array(z["Z"])}

def kl_rows(lc, lm):
    """KL(clean || masked) per row, float32 logits [..., V]."""
    import mlx.core as mx
    pc = lc - mx.logsumexp(lc, axis=-1, keepdims=True)
    pm = lm - mx.logsumexp(lm, axis=-1, keepdims=True)
    return (mx.exp(pc) * (pc - pm)).sum(-1)


def masked_kl(lc, lm, w):
    """Mean KL(clean || masked) over the scored positions: w [B, P] in {0, 1} (the on-policy loss mask). With w all
    ones this is kl_rows(lc, lm).mean() exactly; positions with w = 0 contribute nothing (and no gradient)."""
    import mlx.core as mx
    k = kl_rows(lc, lm)
    if w is None or float(w.min()) == 1.0:        # w is a constant mask (never traced): the exact unmasked mean
        return k.mean()
    n = w.sum()
    return mx.where(n > 0, (k * w).sum() / mx.maximum(n, 1), mx.array(0.0, k.dtype))


def ce_rows(logits, tgt):
    import mlx.core as mx
    lse = mx.logsumexp(logits, axis=-1)
    picked = mx.take_along_axis(logits, tgt[..., None], axis=-1)[..., 0]
    return lse - picked


def train_mask(Z, fixed, lam, steps, lr, grad_fn, batches_fn, log, tag, every=10, peak_cap_gb=None,
               ckpt=None, deadline=None):
    """Carve a unit (Pierre's term, 2026-11-14: choosing neurons is carving, not training; no weight
    changes). Straight-through hard mask over the free units: m = fixed + free * ST(Z).

    grad_fn(m, micro) -> (mean loss of that micro-batch, d loss / d m) at the hard mask m.
    batches_fn(step) -> list of micro-batches (gradient accumulation). Chain rule to Z through the
    straight-through sigmoid: dZ = free * s(1 - s) * (mean_micro dm + lam / n_free)."""
    import mlx.core as mx
    import mlx.optimizers as optim
    opt = optim.Adam(learning_rate=lr)
    free = 1.0 - fixed
    nfree = max(1.0, float(free.sum()))
    params = {"Z": Z}
    start = 0
    if ckpt is not None and __import__("os").path.exists(ckpt):
        start, params = _load_ckpt(ckpt, opt)
        log(f"  {tag}: resumed at step {start} from {ckpt}")
    t0 = time.time()
    for step in range(start, steps):
        Zc = params["Z"]
        m = fixed + free * (Zc > 0).astype(mx.float32)
        micros = batches_fn(step)
        gsum, msum = None, 0.0
        for i, micro in enumerate(micros):
            main, gm = grad_fn(m, micro)
            gsum = gm if gsum is None else gsum + gm
            mx.eval(gsum)
            msum += float(main) / len(micros)
            mx.clear_cache()
            if step == 0 and i == 0:
                log(f"  {tag}: first micro-batch peak {peak_gb()} GB")
                if peak_cap_gb is not None and peak_gb() > peak_cap_gb:
                    raise SystemExit(f"{tag}: MLX peak {peak_gb()} GB over the {peak_cap_gb} GB cap after one micro-batch")
        s_ = mx.sigmoid(Zc)
        gZ = free * s_ * (1 - s_) * (gsum / len(micros) + lam / nfree)
        params = opt.apply_gradients({"Z": gZ}, params)
        mx.eval(params, opt.state)
        mx.clear_cache()
        if step % every == 0 or step == steps - 1:
            kept = float((fixed + free * (params["Z"] > 0)).mean())
            log(f"  {tag} lam={lam} step {step} main={msum:.4f} kept={kept:.4f} "
                f"({(time.time() - t0) / (step + 1 - start):.1f} s/step, peak {peak_gb()} GB)")
            if ckpt is not None:
                _save_ckpt(ckpt, step + 1, params, opt)
        if deadline is not None and time.time() >= deadline and step < steps - 1:
            if ckpt is not None:
                _save_ckpt(ckpt, step + 1, params, opt)
            raise CleanStop(f"{tag}: --stop-at deadline after step {step}; checkpoint {ckpt}")
    return params["Z"]


def topk_mask(score, frac):
    import mlx.core as mx
    L, N = score.shape
    k = max(1, int(round(frac * L * N)))
    flat = score.reshape(-1)
    thr = mx.sort(flat)[-k]
    return (score >= thr).astype(mx.float32)


def random_same_counts(mask, seed):
    import mlx.core as mx
    rng = np.random.default_rng(seed)
    m = np.array(mask)
    R = np.zeros_like(m)
    for l in range(m.shape[0]):
        k = int(m[l].sum())
        R[l, rng.choice(m.shape[1], k, replace=False)] = 1.0
    return mx.array(R)
