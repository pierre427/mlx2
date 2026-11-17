"""A Donor-shaped backend over the 27B (adapter-loaded, optionally amoeba-tiled) for the det-tile drivers.

Research tooling only. ``forward(ids, mlp_hook=, record=)`` returns ``(logits, caps, {})`` like
research.amoeba.donor.Donor, with caps[l] = (MLP input, MLP output). The hook sees each live MLP
(stock or tiled) exactly as the serving wrapper does. ``p(i, name)`` returns the *source
checkpoint's* dequantized MLP rows (the recogniser's host neurons), whatever the live MLP is.
"""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def _reference_sdpa(q, k, v, scale, mask):
    """softmax(q k^T * scale + mask) v in float32, with grouped-query heads; differentiable."""
    B, Hq, Lq, D = q.shape
    Hk, Lk = k.shape[1], k.shape[2]
    if Hq != Hk:
        k = mx.repeat(k, Hq // Hk, axis=1)
        v = mx.repeat(v, Hq // Hk, axis=1)
    s = (q.astype(mx.float32) * scale) @ k.astype(mx.float32).swapaxes(-1, -2)
    if isinstance(mask, str):
        if mask != "causal":
            raise ValueError(f"unsupported sdpa mask {mask!r}")
        qi = mx.arange(Lk - Lq, Lk)[:, None]
        s = mx.where(mx.arange(Lk)[None] <= qi, s, mx.array(-mx.inf, s.dtype))
    elif mask is not None:
        s = mx.where(mask, s, mx.array(-mx.inf, s.dtype)) if mask.dtype == mx.bool_ else s + mask.astype(s.dtype)
    return (mx.softmax(s, axis=-1) @ v.astype(mx.float32)).astype(q.dtype)


class _Rec(nn.Module):
    def __init__(self, inner, layer, owner):
        super().__init__()
        self.inner = inner
        self._layer = layer
        self._owner = owner

    def __call__(self, x, *a, **k):
        y = self.inner(x, *a, **k)
        o = self._owner
        if o._hook is not None:
            r = o._hook(self._layer, x, y)
            if r is not None:
                y = r
        if self._layer in o._record:
            o._caps[self._layer] = (x, y)
        return y


class Backend27:
    def __init__(self, path=None, policy=None, log=print, footprint_limit_gb=80.0):
        import amoeba27b_lib as A
        if path is None:
            # Pierre's per-process limit; also aborts on swap growth, low availability or critical pressure.
            A.footprint_watchdog(footprint_limit_gb, log)
        self.host = A.Host(path or A.MODEL_27B, log=log, policy=policy, wrap=False)
        h = self.host
        self.L, self.d = h.L, h.d
        self.path = h.path
        self.artifact = h.checkpoint
        self.source_fingerprint = h.fingerprint
        self.tiles = getattr(h.adapter, "amoeba_tiles", None)
        self.host_tiles_sha256 = None if self.tiles is None else self.tiles.identity["manifest_sha256"]
        self._hook, self._record, self._caps = None, set(), {}
        for i, layer in enumerate(h.layers):
            layer.mlp = _Rec(layer.mlp, i, self)
        self._rows = {}
        self._index = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
        self._q = json.loads((self.path / "config.json").read_text()).get("quantization")
        from transformers import AutoTokenizer
        from huggingface_hub import snapshot_download
        self._src_tok = AutoTokenizer.from_pretrained(snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True))
        self.w = {}

    # --- Donor interface ----------------------------------------------------------------------
    def forward(self, ids, *, mlp_hook=None, record=(), attn_probs=None, keep_probs=False, stop_after=None):
        self._hook, self._record, self._caps = mlp_hook, set(record), {}
        try:
            hidden = self.host.lm.model(ids)
            if stop_after is not None:
                # Capture-only call (Donor returns no logits here either): skip the 248k-vocab head.
                mx.eval([v for pair in self._caps.values() for v in pair])
                return None, dict(self._caps), {}
            logits = self.host.head(hidden)
            return logits, dict(self._caps), {}
        finally:
            self._hook, self._record = None, set()

    def p(self, i, name):
        """Source-checkpoint MLP rows (float32), e.g. name 'mlp.gate_proj.weight'."""
        key = f"model.language_model.layers.{i}.{name}"
        if key not in self._rows:
            base = key[: -len(".weight")]
            shard = self._index[key]
            t = mx.load(str(self.path / shard))
            w = t[key]
            if base + ".scales" in t and self._q:
                w = mx.dequantize(w, t[base + ".scales"], t.get(base + ".biases"),
                                  group_size=self._q["group_size"], bits=self._q["bits"],
                                  mode=self._q.get("mode", "affine"))
            self._rows[key] = w.astype(mx.float32)
            mx.eval(self._rows[key])
        return self._rows[key]

    @contextlib.contextmanager
    def grad_context(self):
        """GDN takes its differentiable ops path only in training mode.

        On the GPU, MLX's fused SDPA picks its vector kernel for short queries (the 5-10 token key
        prompts), and that kernel has no VJP. Inside this context short queries go through a
        reference softmax(QK^T)V instead; long queries keep the fused full-attention kernel.
        """
        fast = mx.fast
        orig = fast.scaled_dot_product_attention

        def sdpa(q, k, v, *, scale, mask=None, **kw):
            if kw.get("sinks", 0) is None:
                kw.pop("sinks")              # models.base.fast_sdpa always passes sinks=None
            # Anything else (e.g. real sinks) keeps the fused kernel; short queries then have no VJP.
            if q.shape[-2] > 8 or kw:
                return orig(q, k, v, scale=scale, mask=mask, **kw)
            return _reference_sdpa(q, k, v, scale, mask)

        # Tiled MLPs count gate opens with async_eval, which value_and_grad forbids.
        from mlx2.runtime.amoeba_tiles import gate_counting_suspended
        try:
            fast.scaled_dot_product_attention = sdpa
            self.host.train_mode(True)
            with gate_counting_suspended():
                yield
        finally:
            self.host.train_mode(False)
            fast.scaled_dot_product_attention = orig

    def reencode(self, ids):
        """Corpus ids are Qwen3-0.6B tokens: decode and re-encode with the host tokenizer."""
        return self.host.encode(self._src_tok.decode(list(ids)))
