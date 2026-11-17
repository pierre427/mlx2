"""Qwen3 dense donor in plain MLX with hooks for the amoeba experiments.

A deliberately small, readable forward (fp32 by default) with three hooks:

* ``mlp_hook(layer, x_in, y_out) -> y``: sees each MLP's input (the
  post-attention RMSNorm output) and real output, and may return a replacement
  output.  This is where transcoders, box ablation and sufficiency tests plug in.
* ``record``: layers whose MLP input/output are returned.
* ``attn_probs``: per-layer attention probabilities to reuse instead of
  recomputing them (``freeze attention to the clean run``).

The donor is never modified on disk; every intervention is a runtime hook.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx


def snapshot(repo_id: str = "Qwen/Qwen3-0.6B") -> Path:
    root = Path.home() / ".cache/huggingface/hub" / ("models--" + repo_id.replace("/", "--"))
    rev = (root / "refs/main").read_text().strip()
    return root / "snapshots" / rev


class Donor:
    def __init__(self, repo_id: str = "Qwen/Qwen3-0.6B", dtype=mx.float32):
        self.path = snapshot(repo_id)
        self.artifact = f"{repo_id}@{self.path.name[:8]}"
        c = json.loads((self.path / "config.json").read_text())
        self.cfg = c
        self.L, self.d = c["num_hidden_layers"], c["hidden_size"]
        self.nh, self.nkv, self.hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.eps, self.theta = c["rms_norm_eps"], c["rope_theta"]
        w = {}
        for shard in sorted(self.path.glob("*.safetensors")):
            w.update(mx.load(str(shard)))
        q = c.get("quantization")
        if q:                                   # MLX-quantized checkpoint: dequantize once at load
            for k in [k for k in w if k.endswith(".scales")]:
                base = k[: -len(".scales")]
                w[base + ".weight"] = mx.dequantize(w[base + ".weight"], w.pop(k), w.pop(base + ".biases"),
                                                    group_size=q["group_size"], bits=q["bits"])
            self.quantization = q
        else:
            self.quantization = None
        self.w = {k: v.astype(dtype) for k, v in w.items()}
        self.embed = self.w["model.embed_tokens.weight"]
        self.head = self.w.get("lm_head.weight", self.embed)
        mx.eval(self.w)

    def p(self, i, name):
        return self.w[f"model.layers.{i}.{name}"]

    def norm(self, x, weight):
        return mx.fast.rms_norm(x, weight, self.eps)

    def mlp(self, i, x):
        g = x @ self.p(i, "mlp.gate_proj.weight").T
        u = x @ self.p(i, "mlp.up_proj.weight").T
        return (g * mx.sigmoid(g) * u) @ self.p(i, "mlp.down_proj.weight").T

    def attention(self, i, x, probs=None):
        B, T, _ = x.shape
        q = (x @ self.p(i, "self_attn.q_proj.weight").T).reshape(B, T, self.nh, self.hd)
        k = (x @ self.p(i, "self_attn.k_proj.weight").T).reshape(B, T, self.nkv, self.hd)
        v = (x @ self.p(i, "self_attn.v_proj.weight").T).reshape(B, T, self.nkv, self.hd)
        q = self.norm(q, self.p(i, "self_attn.q_norm.weight")).transpose(0, 2, 1, 3)
        k = self.norm(k, self.p(i, "self_attn.k_norm.weight")).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        q = mx.fast.rope(q, self.hd, traditional=False, base=self.theta, scale=1.0, offset=0)
        k = mx.fast.rope(k, self.hd, traditional=False, base=self.theta, scale=1.0, offset=0)
        rep = self.nh // self.nkv
        k, v = mx.repeat(k, rep, axis=1), mx.repeat(v, rep, axis=1)
        if probs is None:
            s = (q @ k.transpose(0, 1, 3, 2)) * (self.hd ** -0.5)
            mask = mx.triu(mx.full((T, T), -mx.inf, dtype=s.dtype), k=1)
            probs = mx.softmax(s + mask, axis=-1, precise=True)
        o = (probs @ v).transpose(0, 2, 1, 3).reshape(B, T, self.nh * self.hd)
        return o @ self.p(i, "self_attn.o_proj.weight").T, probs

    def forward(self, ids, *, mlp_hook=None, record=(), attn_probs=None, keep_probs=False,
                stop_after=None):
        h = self.embed[ids]
        caps, probs_out = {}, {}
        for i in range(self.L):
            a, pr = self.attention(i, self.norm(h, self.p(i, "input_layernorm.weight")),
                                   None if attn_probs is None else attn_probs.get(i))
            if keep_probs:
                probs_out[i] = pr
            h = h + a
            x = self.norm(h, self.p(i, "post_attention_layernorm.weight"))
            y = self.mlp(i, x)
            if mlp_hook is not None:
                r = mlp_hook(i, x, y)
                if r is not None:
                    y = r
            if i in record:
                caps[i] = (x, y)
            h = h + y
            if stop_after is not None and i >= stop_after:
                return None, caps, probs_out
        logits = self.norm(h, self.w["model.norm.weight"]) @ self.head.T
        return logits, caps, probs_out

    def generate(self, prompts_ids, n_new, *, temperature=0.8, top_p=0.95, seed=0, eos=None):
        """Batched sampling with a KV cache (left-padded prompts). Returns new token lists."""
        mx.random.seed(seed)
        B = len(prompts_ids); P = max(len(p) for p in prompts_ids)
        pad = [P - len(p) for p in prompts_ids]
        ids = mx.array([[0] * a + p for a, p in zip(pad, prompts_ids)])
        valid = mx.array([[0] * a + [1] * len(p) for a, p in zip(pad, prompts_ids)]).astype(mx.bool_)
        cache = [None] * self.L
        out = [[] for _ in range(B)]
        done = [False] * B
        x_ids, pos0 = ids, 0
        keymask = valid
        for step in range(n_new + 1):
            h = self.embed[x_ids]
            T = h.shape[1]
            for i in range(self.L):
                xi = self.norm(h, self.p(i, "input_layernorm.weight"))
                q = (xi @ self.p(i, "self_attn.q_proj.weight").T).reshape(B, T, self.nh, self.hd)
                k = (xi @ self.p(i, "self_attn.k_proj.weight").T).reshape(B, T, self.nkv, self.hd)
                v = (xi @ self.p(i, "self_attn.v_proj.weight").T).reshape(B, T, self.nkv, self.hd)
                q = self.norm(q, self.p(i, "self_attn.q_norm.weight")).transpose(0, 2, 1, 3)
                k = self.norm(k, self.p(i, "self_attn.k_norm.weight")).transpose(0, 2, 1, 3)
                v = v.transpose(0, 2, 1, 3)
                q = mx.fast.rope(q, self.hd, traditional=False, base=self.theta, scale=1.0, offset=pos0)
                k = mx.fast.rope(k, self.hd, traditional=False, base=self.theta, scale=1.0, offset=pos0)
                if cache[i] is not None:
                    k = mx.concatenate([cache[i][0], k], axis=2); v = mx.concatenate([cache[i][1], v], axis=2)
                cache[i] = (k, v)
                S = k.shape[2]
                mask = mx.where(keymask[:, None, None, :], 0.0, -mx.inf)
                causal = mx.triu(mx.full((T, S), -mx.inf), k=S - T + 1)
                o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.hd ** -0.5, mask=mask + causal)
                h = h + o.transpose(0, 2, 1, 3).reshape(B, T, -1) @ self.p(i, "self_attn.o_proj.weight").T
                h = h + self.mlp(i, self.norm(h, self.p(i, "post_attention_layernorm.weight")))
            if step == n_new:
                break
            lg = (self.norm(h[:, -1], self.w["model.norm.weight"]) @ self.head.T) / temperature
            probs = mx.softmax(lg, axis=-1)
            srt = mx.sort(probs, axis=-1)[:, ::-1]
            cut = mx.take_along_axis(srt, mx.sum(mx.cumsum(srt, axis=-1) < top_p, axis=-1, keepdims=True), axis=-1)
            lg = mx.where(probs >= cut, lg, -mx.inf)
            nxt = mx.random.categorical(lg)
            mx.eval(nxt)
            for b, t in enumerate(nxt.tolist()):
                if not done[b]:
                    out[b].append(t)
                    done[b] = eos is not None and t in eos
            if all(done):
                break
            x_ids = nxt[:, None]
            pos0 = P + step
            keymask = mx.concatenate([keymask, mx.ones((B, 1), dtype=mx.bool_)], axis=1)
        return out
