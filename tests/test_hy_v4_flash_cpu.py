"""Hy4-flash (hy_v4_flash) CPU tests on a tiny random checkpoint.

The oracle is an independent float64 NumPy transcription of the reference
math (transformers PR 50903 ``modeling_hy_v4_flash.py``; DFly2 from vLLM PR
68296 ``hy_v4/nvidia/dfly*.py``), run on checkpoint-named weights, so the
sanitize/fold path is covered too.  No reference code is executed.
"""

import json
from types import SimpleNamespace
from pathlib import Path
import math
import re

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.models import hy_v4_flash as H  # noqa: E402

LT = ["deepseek_sparse_attention", "deepseek_sparse_attention", "sliding_attention",
      "sliding_attention", "deepseek_sparse_attention", "deepseek_sparse_attention"]
F32_EPS = float(np.finfo(np.float32).eps)


def tiny_config(**kw):
    text = dict(
        model_type="hy_v4_flash", vocab_size=97, hidden_size=32, intermediate_size=48,
        moe_intermediate_size=16, routed_expert_hidden_size=24, num_hidden_layers=6,
        num_attention_heads=2, swa_num_attention_heads=4, q_lora_rank=16, kv_lora_rank=12,
        qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=8, swa_qk_nope_head_dim=8,
        swa_qk_rope_head_dim=4, swa_v_head_dim=8, layer_types=LT,
        mlp_layer_types=["dense"] + ["sparse"] * 5,
        rope_parameters={"deepseek_sparse_attention": {"rope_type": "default", "rope_theta": 1e4,
                                                        "mrope_section": [1, 0, 1]},
                         "sliding_attention": {"rope_type": "default", "rope_theta": 100.0,
                                               "mrope_section": [1, 0, 1]}},
        sliding_window=8, window_size=[7, 0], rms_norm_eps=1e-5, n_routed_experts=8,
        num_experts_per_tok=3, routed_scaling_factor=2.5, compressed_indexer_num_attention_heads=2,
        compressed_indexer_head_dim=8, compressed_indexer_group_topk=4,
        compressed_indexer_compress_ratio=2, engram_layer_ids=[2, 5], engram_n_head_per_ngram=2,
        engram_n_embed_per_ngram=8, engram_max_ngram_size=3,
        engram_layer_head_vocab_sizes={"2": [[11, 13], [17, 19]], "5": [[23, 29], [31, 37]]},
        engram_pad_id=1, engram_lookup_table_size=97, engram_fp8_embedding=False, engram_seed=0,
        swiglu_limit=1.5, hc_mult=4, hc_magnitude=2.0, hc_eps=1e-6, eos_token_id=2,
    )
    text.update(kw)
    return {"model_type": "hy_v4_flash", "text_config": text, "dfly_num_hidden_layers": 2,
            "dfly_mask_token_id": 3, "dfly_target_index_layer": [4, 5],
            "dfly_target_kv_layers": [4, 5], "speculative_config": {"num_speculative_tokens": 3}}


def checkpoint(config, seed=0, scale=0.35):
    """Random weights with the checkpoint's tensor names and layouts."""
    t = config["text_config"]
    rng = np.random.default_rng(seed)
    W = {}

    def r(name, *shape, s=scale):
        W[name] = (rng.standard_normal(shape) * s).astype(np.float32)

    def ones(name, n):
        W[name] = (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)

    D, n = t["hidden_size"], t["hc_mult"]
    r("model.embed_tokens.weight", t["vocab_size"], D, s=1.0)
    r("lm_head.weight", t["vocab_size"], D)
    ones("model.norm.weight", D)
    r("model.hc_head.hc_head_fn", n, n * D, s=0.05)
    r("model.hc_head.hc_head_base", n, s=0.5)
    r("model.hc_head.hc_head_scale", 1, s=0.5)
    for i, kind in enumerate(t["layer_types"]):
        p = f"model.layers.{i}."
        sl = kind == "sliding_attention"
        Hh = t["swa_num_attention_heads"] if sl else t["num_attention_heads"]
        nope, rope, v = t["qk_nope_head_dim"], t["qk_rope_head_dim"], t["v_head_dim"]
        r(p + "self_attn.q_a_proj.weight", t["q_lora_rank"], D)
        ones(p + "self_attn.q_a_layernorm.weight", t["q_lora_rank"])
        r(p + "self_attn.q_b_proj.weight", Hh * (nope + rope), t["q_lora_rank"])
        r(p + "self_attn.kv_a_proj_with_mqa.weight", t["kv_lora_rank"] + rope, D)
        ones(p + "self_attn.kv_a_layernorm.weight", t["kv_lora_rank"])
        r(p + "self_attn.kv_b_proj.weight", Hh * (nope + v), t["kv_lora_rank"])
        r(p + "self_attn.linear_gate.weight", Hh * v, D)
        r(p + "self_attn.o_proj.weight", D, Hh * v)
        ones(p + "self_attn.post_proj_norm.weight", D)
        r(p + "self_attn.learnable_sink_param", Hh, s=1.0)
        if not sl:
            c = p + "self_attn.compressed_indexer."
            d, h = t["compressed_indexer_head_dim"], t["compressed_indexer_num_attention_heads"]
            r(c + "query_projection.weight", h * d, t["q_lora_rank"])
            r(c + "head_weight_projection.weight", h, D)
            r(c + "token_key_projection.weight", d, D)
            ones(c + "token_key_norm.weight", d)
            r(c + "key_value_projection.weight", d, d)
            r(c + "key_gate_projection.weight", d, d)
            ones(c + "key_norm.weight", d)
            r(c + "intra_group_ape", t["compressed_indexer_compress_ratio"], d)
        ones(p + "input_layernorm.weight", D)
        ones(p + "post_attention_layernorm.weight", D)
        for hc in ("hc_attn_layer", "hc_mlp_layer"):
            r(p + hc + ".hc_pre.hc_fn", 2 * n, n * D, s=0.05)
            r(p + hc + ".hc_pre.hc_base", 2 * n, s=0.5)
            r(p + hc + ".hc_pre.hc_scale", 2, s=0.5)
        if t["mlp_layer_types"][i] == "dense":
            r(p + "mlp.gate_proj.weight", t["intermediate_size"], D)
            r(p + "mlp.up_proj.weight", t["intermediate_size"], D)
            r(p + "mlp.down_proj.weight", D, t["intermediate_size"])
            ones(p + "mlp.post_fc2_norm.weight", D)
        else:
            E, I, Lt = t["n_routed_experts"], t["moe_intermediate_size"], t["routed_expert_hidden_size"]
            r(p + "mlp.gate.weight", E, D)
            r(p + "mlp.gate.e_score_correction_bias", E, s=0.1)
            r(p + "mlp.experts.gate_up_proj", E, 2 * I, Lt)
            r(p + "mlp.experts.down_proj", E, Lt, I)
            r(p + "mlp.routed_expert_down_proj.weight", Lt, D)
            r(p + "mlp.routed_expert_up_proj.weight", D, Lt)
            r(p + "mlp.shared_experts.gate_proj.weight", I, D)
            r(p + "mlp.shared_experts.up_proj.weight", I, D)
            r(p + "mlp.shared_experts.down_proj.weight", D, I)
            ones(p + "mlp.combine_output_norm.weight", D)
        if i + 1 in t["engram_layer_ids"]:
            e = p + "engram_layer."
            sizes = [s for o in t["engram_layer_head_vocab_sizes"][str(i + 1)] for s in o]
            hd = t["engram_n_embed_per_ngram"] // t["engram_n_head_per_ngram"]
            width = (t["engram_max_ngram_size"] - 1) * t["engram_n_embed_per_ngram"]
            # A non-identity normalized vocabulary: case/whitespace variants share ids.
            W[e + "lookup_table"] = (np.arange(t["engram_lookup_table_size"]) // 2).astype(np.int32)
            r(e + "multi_head_embedding.embedding.weight", sum(sizes), hd, s=1.0)
            for h in range(len(sizes)):
                ones(e + f"lookup_per_head_norms.{h}.weight", hd)
            r(e + "value_proj.weight", D, width)
            r(e + "value_proj.bias", D, s=0.1)
            for c in range(n):
                r(e + f"key_projs.{c}.weight", D, width)
                r(e + f"key_projs.{c}.bias", D, s=0.1)
                ones(e + f"norm_key.{c}.weight", D)
                ones(e + f"norm_query.{c}.weight", D)
                r(e + f"unified_per_hc_scales.{c}", D, s=0.3)
            ones(e + "unified_pre_gamma", D)
    # DFly2 layers
    q = "model.dfly_layers."
    nope, rope, v, Hh = t["qk_nope_head_dim"], t["qk_rope_head_dim"], t["v_head_dim"], t["num_attention_heads"]
    for i in range(config["dfly_num_hidden_layers"]):
        p = f"{q}{i}."
        r(p + "self_attn.q_a_proj.weight", t["q_lora_rank"], D)
        ones(p + "self_attn.q_a_layernorm.weight", t["q_lora_rank"])
        r(p + "self_attn.q_b_proj.weight", Hh * (nope + rope), t["q_lora_rank"])
        r(p + "self_attn.kv_a_proj_with_mqa.weight", t["kv_lora_rank"] + rope, D)
        ones(p + "self_attn.kv_a_layernorm.weight", t["kv_lora_rank"])
        r(p + "self_attn.kv_b_proj.weight", Hh * (nope + v), t["kv_lora_rank"])
        r(p + "self_attn.linear_gate.weight", Hh * v, D)
        r(p + "self_attn.o_proj.weight", D, Hh * v)
        ones(p + "self_attn.post_proj_norm.weight", D)
        r(p + "self_attn.learnable_sink_param", Hh, s=1.0)
        ones(p + "input_layernorm.weight", D)
        ones(p + "post_attention_layernorm.weight", D)
        r(p + "mlp.gate_proj.weight", 40, D)
        r(p + "mlp.up_proj.weight", 40, D)
        r(p + "mlp.down_proj.weight", D, 40)
    ones(q + "final_norm.weight", D)
    ones(q + "mtp_fusion.enorm.weight", D)
    ones(q + "mtp_fusion.hnorm.weight", D)
    r(q + "mtp_fusion.eh_proj.weight", D, 2 * D)
    r(q + "mtp_fusion.gate", D, s=0.5)
    ones(q + "hidden_correction.hidden_norm.weight", D)
    ones(q + "hidden_correction.embed_norm.weight", D)
    r(q + "hidden_correction.gate_proj.weight", 20, 2 * D)
    r(q + "hidden_correction.up_proj.weight", 20, 2 * D)
    r(q + "hidden_correction.down_proj.weight", D, 20)
    r(q + "confidence_head.proj.weight", 1, D)
    r(q + "confidence_head.proj.bias", 1)
    return W


def build(config, W):
    args = H.ModelArgs.from_hf_config(config)
    model = H.Model(args)
    weights = model.sanitize({k: mx.array(v) for k, v in W.items()})
    tables = model.pop_engram_tables(weights)
    model.load_weights(list(weights.items()), strict=True)
    model.bind_engram({i: H.ResidentEngramRows(w) for i, w in tables.items()})
    dfly_raw = {k: mx.array(v) for k, v in W.items() if k.startswith("model.dfly_layers.")}
    inter, corr = H.HY4FlashDFly.geometry(dfly_raw)
    drafter = H.HY4FlashDFly(args, intermediate=inter, correction=corr)
    drafter.load_weights(list(drafter.sanitize(dfly_raw).items()), strict=True)
    return model, drafter


# ------------------------------------------------------------- numpy oracle --

def rms(x, w=None, eps=1e-5):
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return y if w is None else y * w


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def rope_ref(x, pos, theta):
    """``apply_rotary_pos_emb``: rotate_half on even/odd lanes, interleaved cos."""
    d = x.shape[-1]
    inv = 1.0 / theta ** (np.arange(0, d, 2) / d)
    f = np.asarray(pos, np.float64)[:, None] * inv
    cos = np.repeat(np.cos(f), 2, axis=-1)
    sin = np.repeat(np.sin(f), 2, axis=-1)
    rot = np.stack([-x[..., 1::2], x[..., 0::2]], axis=-1).reshape(x.shape)
    shape = (len(pos),) + (1,) * (x.ndim - 2) + (d,)
    return x * cos.reshape(shape) + rot * sin.reshape(shape)


class Oracle:
    def __init__(self, config, W):
        self.t = config["text_config"]
        self.c = config
        self.W = {k: v.astype(np.float64) if v.dtype != np.int32 else v for k, v in W.items()}

    def theta(self, kind):
        return self.t["rope_parameters"][kind]["rope_theta"]

    def hc(self, streams, p):
        W, n = self.W, self.t["hc_mult"]
        flat = streams.reshape(streams.shape[0], -1)
        mixes = (flat @ W[p + "hc_fn"].T) / np.sqrt(np.mean(flat ** 2, -1, keepdims=True) + self.t["rms_norm_eps"])
        base, scale = W[p + "hc_base"], W[p + "hc_scale"]
        pre = sigmoid(mixes[:, :n] * scale[0] + base[:n]) + self.t["hc_eps"]
        post = self.t["hc_magnitude"] * sigmoid(mixes[:, n:] * scale[1] + base[n:]) + self.t["hc_eps"]
        return post, (pre[..., None] * streams).sum(1)

    def indexer_mask(self, x, q_resid, p, L):
        t, W = self.t, self.W
        c = p + "self_attn.compressed_indexer."
        ratio, d, h = t["compressed_indexer_compress_ratio"], t["compressed_indexer_head_dim"], t["compressed_indexer_num_attention_heads"]
        rd = t["qk_rope_head_dim"]
        th = self.theta("deepseek_sparse_attention")
        tk = rms(x @ W[c + "token_key_projection.weight"].T, W[c + "token_key_norm.weight"])
        G = L // ratio
        grouped = tk[: G * ratio].reshape(G, ratio, d)
        values = grouped @ W[c + "key_value_projection.weight"].T
        gates = grouped @ W[c + "key_gate_projection.weight"].T + W[c + "intra_group_ape"]
        gates = np.exp(gates - gates.max(1, keepdims=True))
        gates /= gates.sum(1, keepdims=True)
        gk = rms((values * gates).sum(1), W[c + "key_norm.weight"])
        gk = np.concatenate([gk[:, :-rd], rope_ref(gk[:, -rd:], np.arange(G) * ratio, th)], -1)
        q = (q_resid @ W[c + "query_projection.weight"].T).reshape(L, h, d)
        q = np.concatenate([q[..., :-rd], rope_ref(q[..., -rd:], np.arange(L), th)], -1)
        hw = (x @ W[c + "head_weight_projection.weight"].T) * (h ** -0.5 * d ** -0.5)
        scores = np.maximum(np.einsum("lhd,gd->lhg", q, gk), 0)
        logits = np.einsum("lh,lhg->lg", hw, scores)
        own = np.arange(L) // ratio
        logits = np.where(np.arange(G)[None] >= own[:, None], -np.inf, logits)
        k = min(t["compressed_indexer_group_topk"] - 1, G)
        recalled = np.argsort(-logits, axis=-1, kind="stable")[:, :k]
        recalled = np.concatenate([recalled, own[:, None]], -1)
        tokens = (recalled[..., None] * ratio + np.arange(ratio)).reshape(L, -1).clip(max=L - 1)
        keep = np.zeros((L, L), bool)
        np.put_along_axis(keep, tokens, True, axis=-1)
        return keep

    def attention(self, x, i, L):
        t, W = self.t, self.W
        p = f"model.layers.{i}."
        kind = t["layer_types"][i]
        sl = kind == "sliding_attention"
        Hh = t["swa_num_attention_heads"] if sl else t["num_attention_heads"]
        nope, rd, v, r = t["qk_nope_head_dim"], t["qk_rope_head_dim"], t["v_head_dim"], t["kv_lora_rank"]
        a = p + "self_attn."
        gate = (x @ W[a + "linear_gate.weight"].T).reshape(L, Hh, v)
        q_resid = rms(x @ W[a + "q_a_proj.weight"].T, W[a + "q_a_layernorm.weight"])
        q = (q_resid @ W[a + "q_b_proj.weight"].T).reshape(L, Hh, nope + rd)
        ckv = x @ W[a + "kv_a_proj_with_mqa.weight"].T
        k_pass = rms(ckv[:, :r], W[a + "kv_a_layernorm.weight"])
        th = self.theta(kind)
        q_rot = rope_ref(q[..., nope:], np.arange(L), th)
        k_rot = rope_ref(ckv[:, r:], np.arange(L), th)
        kv = (k_pass @ W[a + "kv_b_proj.weight"].T).reshape(L, Hh, nope + v)
        K = np.concatenate([kv[..., :nope], np.broadcast_to(k_rot[:, None], (L, Hh, rd))], -1)
        Q = np.concatenate([q[..., :nope], q_rot], -1)
        V = kv[..., nope:]
        s = np.einsum("ihd,jhd->hij", Q, K) * (nope + rd) ** -0.5
        allowed = np.tril(np.ones((L, L), bool))
        if sl:
            allowed &= np.arange(L)[None] > np.arange(L)[:, None] - t["sliding_window"]
        elif L > t["compressed_indexer_group_topk"] * t["compressed_indexer_compress_ratio"]:
            allowed &= self.indexer_mask(x, q_resid, p, L)
        s = np.where(allowed[None], s, -np.inf)
        sink = W[a + "learnable_sink_param"][:, None, None]
        m = np.maximum(s.max(-1, keepdims=True), sink)
        e = np.exp(s - m)
        probs = e / (e.sum(-1, keepdims=True) + np.exp(sink - m))
        out = np.einsum("hij,jhd->ihd", probs, V) * sigmoid(gate)
        return rms(out.reshape(L, -1) @ W[a + "o_proj.weight"].T, W[a + "post_proj_norm.weight"])

    def mlp(self, x, i):
        t, W = self.t, self.W
        p = f"model.layers.{i}.mlp."
        if t["mlp_layer_types"][i] == "dense":
            y = (silu(x @ W[p + "gate_proj.weight"].T) * (x @ W[p + "up_proj.weight"].T)) @ W[p + "down_proj.weight"].T
            return rms(y, W[p + "post_fc2_norm.weight"])
        scores = sigmoid(x @ W[p + "gate.weight"].T)
        choice = scores + W[p + "gate.e_score_correction_bias"]
        k = t["num_experts_per_tok"]
        idx = np.argsort(-choice, -1)[:, :k]
        w = np.take_along_axis(scores, idx, -1)
        w = w / (w.sum(-1, keepdims=True) + 1e-20) * t["routed_scaling_factor"]
        xl = x @ W[p + "routed_expert_down_proj.weight"].T
        I, lim = t["moe_intermediate_size"], t["swiglu_limit"]
        y = np.zeros_like(xl)
        for row in range(x.shape[0]):
            for j in range(k):
                e = idx[row, j]
                gu = W[p + "experts.gate_up_proj"][e] @ xl[row]
                g, u = np.minimum(gu[:I], lim), np.clip(gu[I:], -lim, lim)
                y[row] += w[row, j] * (W[p + "experts.down_proj"][e] @ (silu(g) * u))
        routed = y @ W[p + "routed_expert_up_proj.weight"].T
        sh = p + "shared_experts."
        shared = (silu(x @ W[sh + "gate_proj.weight"].T) * (x @ W[sh + "up_proj.weight"].T)) @ W[sh + "down_proj.weight"].T
        return rms(routed + shared, W[p + "combine_output_norm.weight"])

    def ngram_tokens(self, tokens):
        n, pad = self.t["engram_max_ngram_size"], self.t["engram_pad_id"]
        L = len(tokens)
        cols = [tokens]
        for off in range(1, n):
            prev = np.concatenate([np.full(off, pad), tokens])[:L]
            cols.append(np.where(np.arange(L) < off, pad, prev))
        return np.stack(cols, -1).astype(np.int64)

    def engram(self, streams, i, tokens):
        t, W = self.t, self.W
        lid = i + 1
        e = f"model.layers.{i}.engram_layer."
        table = W[e + "lookup_table"].astype(np.int64)
        heads = t["engram_n_head_per_ngram"]
        sizes = np.array([s for o in t["engram_layer_head_vocab_sizes"][str(lid)] for s in o], np.int64)
        comp_vocab = int(table.max()) + 1
        half = max(1, np.iinfo(np.int64).max // comp_vocab // 2)
        g = np.random.default_rng(t["engram_seed"] + 10007 * lid)
        mult = g.integers(0, half, size=t["engram_max_ngram_size"], dtype=np.int64) * 2 + 1
        ng = self.ngram_tokens(tokens)
        comp = np.where(ng < table.size, table[np.minimum(ng, table.size - 1)], table[t["engram_pad_id"]])
        mixed = comp * mult
        ids = []
        for order in range(2, t["engram_max_ngram_size"] + 1):
            mix = mixed[:, 0]
            for o in range(1, order):
                mix = np.bitwise_xor(mix, mixed[:, o])
            start = (order - 2) * heads
            ids.append(mix[:, None] % sizes[start:start + heads])
        ids = np.concatenate(ids, -1)
        offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        emb = W[e + "multi_head_embedding.embedding.weight"][ids + offsets]  # [L, heads, hd]
        emb = np.stack([rms(emb[:, h], W[e + f"lookup_per_head_norms.{h}.weight"], F32_EPS)
                        for h in range(len(sizes))], 1).reshape(len(tokens), -1)
        gates = []
        D = t["hidden_size"]
        for c in range(t["hc_mult"]):
            key = rms(emb @ W[e + f"key_projs.{c}.weight"].T + W[e + f"key_projs.{c}.bias"], W[e + f"norm_key.{c}.weight"], F32_EPS)
            qq = rms(streams[:, c], W[e + f"norm_query.{c}.weight"], F32_EPS)
            gg = (key * qq).sum(-1) * D ** -0.5
            gates.append(sigmoid(np.sqrt(np.maximum(np.abs(gg), 1e-6)) * np.sign(gg)))
        gates = np.stack(gates, 1)[..., None]
        value = emb @ W[e + "value_proj.weight"].T + W[e + "value_proj.bias"]
        main = W[e + "unified_pre_gamma"] * rms(value, None, F32_EPS)
        branch = np.stack([W[e + f"unified_per_hc_scales.{c}"] * main for c in range(t["hc_mult"])], 1)
        out = gates * main[:, None] + branch
        mask = (tokens != t["engram_pad_id"])[:, None, None]
        return streams + np.where(mask, out, 0)

    def hidden(self, tokens):
        t, W = self.t, self.W
        tokens = np.asarray(tokens)
        L, n = len(tokens), t["hc_mult"]
        streams = np.repeat(W["model.embed_tokens.weight"][tokens][:, None], n, 1)
        for i in range(t["num_hidden_layers"]):
            p = f"model.layers.{i}."
            if i + 1 in t["engram_layer_ids"]:
                streams = self.engram(streams, i, tokens)
            post, x = self.hc(streams, p + "hc_attn_layer.hc_pre.")
            x = self.attention(rms(x, W[p + "input_layernorm.weight"]), i, L)
            streams = post[..., None] * x[:, None] + streams
            post, x = self.hc(streams, p + "hc_mlp_layer.hc_pre.")
            x = self.mlp(rms(x, W[p + "post_attention_layernorm.weight"]), i)
            streams = post[..., None] * x[:, None] + streams
        flat = streams.reshape(L, -1)
        mixes = (flat @ W["model.hc_head.hc_head_fn"].T) / np.sqrt(np.mean(flat ** 2, -1, keepdims=True) + t["rms_norm_eps"])
        pre = sigmoid(mixes * W["model.hc_head.hc_head_scale"] + W["model.hc_head.hc_head_base"]) + t["hc_eps"]
        return rms((pre[..., None] * streams).sum(1), W["model.norm.weight"])

    def logits(self, tokens):
        return self.hidden(tokens) @ self.W["lm_head.weight"].T


TOKENS = [5, 7, 9, 11, 13, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 31, 33, 35, 37, 39]


@pytest.fixture(scope="module")
def tiny():
    config = tiny_config()
    W = checkpoint(config)
    model, drafter = build(config, W)
    return config, W, model, drafter


def test_full_sequence_matches_numpy_oracle(tiny):
    config, W, model, _ = tiny
    # 24 tokens > candidate_topk 8: the indexer selection is active; the
    # sliding window (8) is exceeded too.
    ref = Oracle(config, W).logits(TOKENS)
    got = np.array(model(mx.array([TOKENS]))[0])
    np.testing.assert_allclose(got, ref, rtol=2e-4, atol=2e-4)


def test_short_context_dense_matches_oracle(tiny):
    config, W, model, _ = tiny
    ref = Oracle(config, W).logits(TOKENS[:7])
    got = np.array(model(mx.array([TOKENS[:7]]))[0])
    np.testing.assert_allclose(got, ref, rtol=2e-4, atol=2e-4)


def test_mask_path_equals_gather_path(tiny, monkeypatch):
    _, _, model, _ = tiny
    gathered = model(mx.array([TOKENS]))
    monkeypatch.setattr(H, "GATHER_MAX_QUERY", 0)
    masked = model(mx.array([TOKENS]))
    np.testing.assert_allclose(np.array(masked), np.array(gathered), rtol=1e-4, atol=1e-4)


def test_chunked_and_stepwise_decode_match_full(tiny):
    _, _, model, _ = tiny
    full = np.array(model(mx.array([TOKENS])))
    cache = model.make_cache()
    parts = [model(mx.array([TOKENS[:9]]), cache), model(mx.array([TOKENS[9:15]]), cache)]
    parts += [model(mx.array([[tok]]), cache) for tok in TOKENS[15:]]
    np.testing.assert_allclose(np.array(mx.concatenate(parts, axis=1)), full, rtol=1e-4, atol=1e-4)


def test_trim_rollback_is_exact(tiny):
    _, _, model, _ = tiny
    cache = model.make_cache()
    model(mx.array([TOKENS[:18]]), cache)
    model(mx.array([[40, 41, 42, 43, 44]]), cache)  # a rejected block
    for c in cache:
        c.trim(5)
    after = np.array(model(mx.array([TOKENS[18:]]), cache))
    fresh = model.make_cache()
    model(mx.array([TOKENS[:18]]), fresh)
    ref = np.array(model(mx.array([TOKENS[18:]]), fresh))
    np.testing.assert_allclose(after, ref, rtol=1e-6, atol=1e-6)


def test_sliding_cache_refuses_rollback_past_window():
    cache = H.HY4FlashLayerCache(H.SLIDING, window=4)
    cache.append(mx.zeros((1, 100, 5)))
    assert cache.ring_end - cache.ring_head == 4 - 1 + H.ROLLBACK
    cache.trim(H.ROLLBACK)
    with pytest.raises(ValueError):
        cache.trim(1)


def test_engram_hash_ids_and_contributions(tiny):
    config, W, model, _ = tiny
    hasher = model.engram_hashers[1]
    ng = Oracle(config, W).ngram_tokens(np.array(TOKENS))
    ids = hasher.hash_ids(ng[None])[0]
    sizes = np.array([11, 13, 17, 19])
    assert ids.shape == (len(TOKENS), 4) and (ids >= 0).all() and (ids < sizes).all()
    # The multipliers are the reference draw over the loaded map (max id + 1).
    comp = int(W["model.layers.1.engram_layer.lookup_table"].max()) + 1
    g = np.random.default_rng(0 + 10007 * 2)
    expect = g.integers(0, np.iinfo(np.int64).max // comp // 2, size=3, dtype=np.int64) * 2 + 1
    np.testing.assert_array_equal(hasher.multipliers, expect)
    # Removing the memory changes the output (it is wired in).
    with_engram = np.array(model(mx.array([TOKENS])))
    saved_hashers, saved_ids = model.engram_hashers, model.args.engram_layer_ids
    model.engram_hashers, model.args.engram_layer_ids = {}, None
    try:
        without = np.array(model(mx.array([TOKENS])))
    finally:
        model.engram_hashers, model.args.engram_layer_ids = saved_hashers, saved_ids
    assert np.abs(with_engram - without).max() > 1e-3


def test_engram_decode_tail_matches_prefill(tiny):
    """n-grams that span a chunk boundary read the cached token tail."""
    _, _, model, _ = tiny
    full = np.array(model(mx.array([TOKENS[:12]])))
    cache = model.make_cache()
    a = model(mx.array([TOKENS[:5]]), cache)
    b = model(mx.array([TOKENS[5:6]]), cache)
    c = model(mx.array([TOKENS[6:12]]), cache)
    np.testing.assert_allclose(np.array(mx.concatenate([a, b, c], axis=1)), full, rtol=1e-5, atol=1e-5)


def e4m3_ref(x):
    x = np.asarray(x, np.float64)
    a = np.abs(x)
    e = np.floor(np.log2(np.where(a > 0, a, 1)))
    e = np.maximum(e, -6)
    step = 2.0 ** (e - 3)
    q = np.round(a / step) * step
    return np.sign(x) * np.minimum(q, 448)


def test_fp8_round_trip_matches_e4m3_rounding():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((5, 96)).astype(np.float32) * np.array([[1e-3], [1], [30], [1e2], [5]], np.float32)
    got = np.array(H.fp8_round_trip(mx.array(x)))
    scale = np.maximum(np.abs(x).max(-1, keepdims=True), 1e-12) / 448
    ref = e4m3_ref(x / scale) * scale
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-9)


def test_sanitize_folds_and_drops(tiny):
    config, W, model, _ = tiny
    weights = model.sanitize({k: mx.array(v) for k, v in W.items()})
    assert not any(k.startswith("model.dfly_layers.") for k in weights)
    assert "model.layers.1.mlp.experts.gate_proj.weight" in weights
    assert "model.layers.0.self_attn.kv_b_proj.weight" not in weights
    assert weights["model.layers.2.self_attn.embed_q.weight"].shape == (4, 12, 8)
    assert weights["model.layers.0.hc_attn_layer.hc_pre.hc_fn"].dtype == mx.float32
    again = model.sanitize(dict(weights))
    assert set(again) == set(weights)


# ------------------------------------------------------------------- DFly2 --

def dfly_oracle(config, W, model, cache, bonus, target_hidden, anchor, num_draft, path=None):
    """vLLM HYV4DFlyQueryModel transcription on the MLX target cache.

    ``path`` forces the drafts (step ``k`` then reads ``path[k - 1]``), so
    row ``k`` is the law conditioned on that prefix."""
    t = config["text_config"]
    Wd = {k: v.astype(np.float64) for k, v in W.items() if k.startswith("model.dfly_layers.")}
    E = W["model.embed_tokens.weight"].astype(np.float64)
    lm = W["lm_head.weight"].astype(np.float64)
    eps = t["rms_norm_eps"]
    Wn = num_draft + 1
    ids = np.array([bonus] + [config["dfly_mask_token_id"]] * num_draft)
    h = E[ids].copy()
    f = "model.dfly_layers.mtp_fusion."
    fused = np.concatenate([rms(h[0], Wd[f + "enorm.weight"], eps), rms(target_hidden, Wd[f + "hnorm.weight"], eps)]) @ Wd[f + "eh_proj.weight"].T
    h[0] = h[0] + Wd[f + "gate"] * fused
    pos = np.arange(anchor, anchor + Wn)
    nope, rd, v, r, Hh = t["qk_nope_head_dim"], t["qk_rope_head_dim"], t["v_head_dim"], t["kv_lora_rank"], t["num_attention_heads"]
    th = t["rope_parameters"]["deepseek_sparse_attention"]["rope_theta"]
    for i in range(config["dfly_num_hidden_layers"]):
        p = f"model.dfly_layers.{i}.self_attn."
        x = rms(h, Wd[f"model.dfly_layers.{i}.input_layernorm.weight"], eps)
        kv_cache = cache[config["dfly_target_kv_layers"][i]]
        sel_cache = cache[config["dfly_target_index_layer"][i]]
        rows_hist = np.array(kv_cache.history(anchor)[0]).astype(np.float64)
        lat_hist, pe_hist = rows_hist[:, :r], rows_hist[:, r:]
        selection = sel_cache.selection_at(anchor - 1)
        if selection is None:
            hist = np.arange(anchor)
        else:
            rows = np.array(selection[0])
            hist = np.unique(rows[(rows >= 0) & (rows < anchor)])
        q_resid = rms(x @ Wd[p + "q_a_proj.weight"].T, Wd[p + "q_a_layernorm.weight"], eps)
        q = (q_resid @ Wd[p + "q_b_proj.weight"].T).reshape(Wn, Hh, nope + rd)
        ckv = x @ Wd[p + "kv_a_proj_with_mqa.weight"].T
        lat_blk = rms(ckv[:, :r], Wd[p + "kv_a_layernorm.weight"], eps)
        pe_blk = rope_ref(ckv[:, r:], pos, th)
        lat = np.concatenate([lat_hist[hist], lat_blk])
        pe = np.concatenate([pe_hist[hist], pe_blk])
        kv = (lat @ Wd[p + "kv_b_proj.weight"].T).reshape(len(lat), Hh, nope + v)
        K = np.concatenate([kv[..., :nope], np.broadcast_to(pe[:, None], (len(lat), Hh, rd))], -1)
        Q = np.concatenate([q[..., :nope], rope_ref(q[..., nope:], pos, th)], -1)
        s = np.einsum("ihd,jhd->hij", Q, K) * (nope + rd) ** -0.5
        allowed = np.concatenate([np.ones((Wn, len(hist)), bool), np.tril(np.ones((Wn, Wn), bool))], 1)
        s = np.where(allowed[None], s, -np.inf)
        sink = Wd[p + "learnable_sink_param"][:, None, None]
        m = np.maximum(s.max(-1, keepdims=True), sink)
        e = np.exp(s - m)
        probs = e / (e.sum(-1, keepdims=True) + np.exp(sink - m))
        gate = sigmoid(x @ Wd[p + "linear_gate.weight"].T).reshape(Wn, Hh, v)
        out = np.einsum("hij,jhd->ihd", probs, kv[..., nope:]) * gate
        h = h + rms(out.reshape(Wn, -1) @ Wd[p + "o_proj.weight"].T, Wd[p + "post_proj_norm.weight"], eps)
        mp = f"model.dfly_layers.{i}.mlp."
        y = rms(h, Wd[f"model.dfly_layers.{i}.post_attention_layernorm.weight"], eps)
        h = h + (silu(y @ Wd[mp + "gate_proj.weight"].T) * (y @ Wd[mp + "up_proj.weight"].T)) @ Wd[mp + "down_proj.weight"].T
    h = rms(h, Wd["model.dfly_layers.final_norm.weight"], eps)
    c = "model.dfly_layers.hidden_correction."
    prev = bonus
    drafts, logits = [], []
    for step in range(num_draft):
        comb = np.concatenate([rms(h[step + 1], Wd[c + "hidden_norm.weight"], eps), rms(E[prev], Wd[c + "embed_norm.weight"], eps)])
        corr = h[step + 1] + (silu(comb @ Wd[c + "gate_proj.weight"].T) * (comb @ Wd[c + "up_proj.weight"].T)) @ Wd[c + "down_proj.weight"].T
        row = corr @ lm.T
        prev = int(np.argmax(row)) if path is None or step >= len(path) else int(path[step])
        drafts.append(prev)
        logits.append(row)
    return drafts, np.stack(logits)


@pytest.mark.parametrize("prompt_len", [6, 20])
def test_dfly_propose_matches_oracle(tiny, prompt_len):
    config, W, model, drafter = tiny
    model.record_layers = frozenset(drafter.index_layers)
    try:
        cache = model.make_cache()
        hidden = model.hidden(mx.array([TOKENS[:prompt_len]]), cache)
    finally:
        model.record_layers = frozenset()
    row = prompt_len - 1
    if prompt_len > 8:
        assert cache[4].selection_at(prompt_len - 1) is not None  # Index Share is exercised
    bonus = int(mx.argmax(model.logits(hidden[:, row]), -1).item())
    drafts, logits = drafter.propose(model, cache, mx.array([bonus]), hidden[:, row],
                                     anchor=prompt_len, num_draft=3)
    ref_drafts, ref_logits = dfly_oracle(config, W, model, cache, bonus,
                                         np.array(hidden[0, row]).astype(np.float64),
                                         prompt_len, 3)
    # Teacher-forced by the draft's own previous token, so compare the first
    # step's logits and every logit row given matching earlier drafts.
    np.testing.assert_allclose(np.array(logits[0, 0]), ref_logits[0], rtol=2e-4, atol=2e-4)
    assert [int(t) for t in np.array(drafts[0])] == ref_drafts


def test_dfly_greedy_equals_ordinary_greedy(tiny):
    _, _, model, drafter = tiny
    prompt = TOKENS[:14]
    cache = model.make_cache()
    out = [int(mx.argmax(model(mx.array([prompt]), cache)[0, -1]).item())]
    for _ in range(19):
        out.append(int(mx.argmax(model(mx.array([[out[-1]]]), cache)[0, -1]).item()))
    tokens, receipt = H.dfly_greedy_generate(model, drafter, prompt, max_tokens=20, num_draft=3)
    assert tokens == out
    assert receipt["verify_rounds"] >= 1 and sum(receipt["verify_accept_hist"]) == receipt["verify_rounds"]
    # Drafted depth per round: round for round with the accepts, never less
    # than what a round accepted, and at most num_draft.
    depths = receipt["verify_depth_hist"]
    assert len(depths) == 4 and sum(depths) == receipt["verify_rounds"]
    for k in range(1, 4):
        assert sum(receipt["verify_accept_hist"][k:]) <= sum(depths[k:])
    assert receipt["qualified"] is False
    assert model.record_layers == frozenset()


def test_dfly_rejects_sliding_target_layer():
    config = tiny_config()
    config["dfly_target_kv_layers"] = [2, 5]
    args = H.ModelArgs.from_hf_config(config)
    with pytest.raises(ValueError):
        H.HY4FlashDFly(args, intermediate=40, correction=20)


# ---------------------------------------------------------------- adapter --

def test_adapter_inspect_artifact_fails_closed(tmp_path):
    from safetensors.numpy import save_file

    from mlx2.adapters import hy_v4_flash as A

    config = tiny_config()
    W = checkpoint(config)
    save_file(W, str(tmp_path / "model-00001.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: "model-00001.safetensors" for k in W}}))
    (tmp_path / "config.json").write_text(json.dumps(config))
    artifact = A.inspect_artifact(tmp_path)
    assert artifact["has_dfly"] and artifact["dfly_layers"] == 2
    bad = dict(config, text_config=dict(config["text_config"], gating_type="headwise"))
    (tmp_path / "config.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="gating_type"):
        A.inspect_artifact(tmp_path)


def test_ssd_engram_rows_match_resident(tmp_path):
    from safetensors.numpy import save_file

    from mlx2.runtime.models.hy_v4_flash_engram import EngramTableRows

    rng = np.random.default_rng(1)
    table = rng.standard_normal((300, 4)).astype(np.float32)
    bf16 = (table.view(np.uint32) >> 16).astype(np.uint16)
    pad = np.zeros(3, np.float32).tobytes()
    header = json.dumps({"pad": {"dtype": "F32", "shape": [3], "data_offsets": [0, len(pad)]},
                         "t": {"dtype": "BF16", "shape": [300, 4],
                               "data_offsets": [len(pad), len(pad) + bf16.nbytes]}}).encode()
    (tmp_path / "x.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + pad + bf16.tobytes())
    reader = EngramTableRows(tmp_path / "x.safetensors", "t", cache_rows=8)
    try:
        rows = np.array([5, 299, 5, 0, 17])
        got = reader.read_host(rows)
        expect = (bf16.astype(np.uint32) << 16).view(np.float32)[rows]
        np.testing.assert_array_equal(got, expect)
        assert reader.stats["unique_rows"] == 4
        with pytest.raises(IndexError):
            reader.read_host(np.array([300]))
    finally:
        reader.close()


def test_registry_resolves_hy_v4_flash(tmp_path):
    from safetensors.numpy import save_file

    from mlx2.adapters.registry import inspect_model

    config = tiny_config()
    W = checkpoint(config)
    save_file(W, str(tmp_path / "model-00001.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: "model-00001.safetensors" for k in W}}))
    (tmp_path / "config.json").write_text(json.dumps(config))
    resolution = inspect_model(tmp_path)
    assert resolution.adapter_type.__name__ == "HYV4FlashAdapter"
    assert resolution.descriptor.metadata["qualification"] == "pending"


# -------------------------------------------------------------- converter --

def _write_artifact(tmp_path, config, W):
    from safetensors.numpy import save_file

    tables = {k for k in W if k.endswith("multi_head_embedding.embedding.weight")}
    shards = {"model-00001.safetensors": {k: v for k, v in W.items() if k not in tables},
              "model-00002.safetensors": {k: W[k] for k in tables}}
    for name, tensors in shards.items():
        save_file(tensors, str(tmp_path / name))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: n for n, t in shards.items() for k in t}}))
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


def test_load_model_reads_engram_from_ssd_and_matches(tmp_path, tiny):
    from mlx2.adapters import hy_v4_flash as A

    config, W, model, _ = tiny
    src = _write_artifact(tmp_path, config, W)
    loaded, drafter, readers = A.load_model(src, A.inspect_artifact(src), keep_dfly=True)
    try:
        np.testing.assert_array_equal(np.array(loaded(mx.array([TOKENS]))),
                                      np.array(model(mx.array([TOKENS]))))
        assert drafter is not None and readers[1].stats["unique_rows"] > 0
    finally:
        for r in readers.values():
            r.close()


def test_converter_round_trip_and_resume(tmp_path):
    import importlib.util

    from mlx2.adapters import hy_v4_flash as A

    spec = importlib.util.spec_from_file_location(
        "convert_hy_v4_flash_mlx", Path(__file__).resolve().parents[1] / "scripts" / "convert_hy_v4_flash_mlx.py")
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)
    config = tiny_config(routed_expert_hidden_size=32, moe_intermediate_size=32)
    W = checkpoint(config, seed=4, scale=0.2)
    (tmp_path / "src").mkdir()
    src = _write_artifact(tmp_path / "src", config, W)
    out = tmp_path / "out"
    conv.convert(src, out, group_size=32, expert_bits=8, trunk_bits=8, resume=False)
    artifact = A.inspect_artifact(out)
    weights = artifact["weight_map"]
    assert "model.layers.1.mlp.experts.gate_proj.scales" in weights
    assert "model.layers.0.self_attn.q_a_proj.scales" in weights
    assert "model.layers.1.mlp.gate.weight" in weights and "model.layers.1.mlp.gate.scales" not in weights
    assert weights["model.layers.1.engram_layer.multi_head_embedding.embedding.weight"].startswith("engram-")
    converted, drafter, readers = A.load_model(out, artifact, keep_dfly=True)
    try:
        reference, _ = build(config, W)
        got = np.array(converted(mx.array([TOKENS])))[0]
        ref = np.array(reference(mx.array([TOKENS])))[0]
        cos = (got * ref).sum(-1) / np.linalg.norm(got, axis=-1) / np.linalg.norm(ref, axis=-1)
        # 8-bit noise can flip a discrete expert/group pick on a random model.
        assert np.median(cos) > 0.999 and cos.min() > 0.95, cos
        assert drafter is not None
    finally:
        for r in readers.values():
            r.close()
    with pytest.raises(SystemExit):
        conv.convert(src, out, group_size=32, expert_bits=8, trunk_bits=8, resume=False)
    before = json.loads((out / "model.safetensors.index.json").read_text())
    conv.convert(src, out, group_size=32, expert_bits=8, trunk_bits=8, resume=True)
    assert json.loads((out / "model.safetensors.index.json").read_text()) == before


def test_engram_tail_survives_state_round_trip(tiny):
    _, _, model, _ = tiny
    cache = model.make_cache()
    model(mx.array([TOKENS[:9]]), cache)
    restored = [type(c).from_state(c.state, c.meta_state) for c in cache]
    a = np.array(model(mx.array([TOKENS[9:12]]), cache))
    b = np.array(model(mx.array([TOKENS[9:12]]), restored))
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


def test_unbound_engram_raises(tiny):
    _, _, model, _ = tiny
    saved = model.engram_hashers
    model.engram_hashers = {}
    try:
        with pytest.raises(ValueError, match="not bound"):
            model(mx.array([TOKENS[:4]]))
    finally:
        model.engram_hashers = saved


def test_indexer_chunked_scoring_matches_unchunked(tiny, monkeypatch):
    _, _, model, _ = tiny
    monkeypatch.setattr(H, "GATHER_MAX_QUERY", 0)
    full = np.array(model(mx.array([TOKENS])))
    monkeypatch.setattr(H, "SCORE_BUDGET", 64)
    chunked = np.array(model(mx.array([TOKENS])))
    np.testing.assert_allclose(chunked, full, rtol=1e-4, atol=1e-4)


def _engram_oracle(layer, streams, rows, precision):
    """float64 transcription of each reference's Engram precision choices."""
    f = lambda a: np.array(a.astype(mx.float32)).astype(np.float64)
    # Every Engram norm is torch.nn.RMSNorm(eps=None): fp32 epsilon in both references.
    eps = F32_EPS
    x = np.array(rows).astype(np.float64)
    if precision == "transformers":
        scale = np.maximum(np.abs(x).max(-1, keepdims=True), 1e-12) / 448
        x = e4m3_ref(x / scale) * scale
    gam = np.stack([f(n.weight) for n in layer.lookup_per_head_norms])
    x = rms(x, gam, eps).reshape(*x.shape[:-2], -1)
    s = f(streams)
    D = s.shape[-1]
    gates = []
    for c in range(layer.n):
        key = rms(x @ f(layer.key_projs[c].weight).T + f(layer.key_projs[c].bias), f(layer.norm_key[c].weight), eps)
        q = rms(s[..., c, :], f(layer.norm_query[c].weight), eps)
        g = (key * q).sum(-1) * D ** -0.5
        gates.append(sigmoid(np.sqrt(np.maximum(np.abs(g), 1e-6)) * np.sign(g)))
    gate = np.stack(gates, -1)[..., None]
    value = x @ f(layer.value_proj.weight).T + f(layer.value_proj.bias)
    main = f(layer.unified_pre_gamma) * rms(value, None, eps)
    scales = np.stack([f(v) for v in layer.unified_per_hc_scales])
    return s + (gate + scales) * main[..., None, :]


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("precision", ["transformers", "vllm"])
def test_engram_precision_profiles_bf16(precision, dtype):
    config = tiny_config(engram_fp8_embedding=True)
    args = H.ModelArgs.from_hf_config(config)
    args.engram_precision = precision
    layer = H.EngramLayer(args, 2)
    rng = np.random.default_rng(7)
    from mlx.utils import tree_flatten, tree_unflatten
    params = [(k, mx.array((rng.standard_normal(v.shape) * 0.3 + (1.0 if "norm" in k else 0.0)).astype(np.float32)).astype(dtype))
              for k, v in tree_flatten(layer.parameters()) if v.dtype != mx.int32]
    layer.update(tree_unflatten(params))
    streams = mx.array(rng.standard_normal((1, 5, 4, 32)).astype(np.float32)).astype(dtype)
    # Small rows, so an epsilon of 2**-7 instead of fp32's would show.
    rows = mx.array((rng.standard_normal((1, 5, 4, 4)) * 0.05).astype(np.float32))
    got = np.array(layer(streams, rows, mx.ones((1, 5), dtype=mx.bool_)).astype(mx.float32))
    ref = _engram_oracle(layer, streams, rows, precision)
    other = _engram_oracle(layer, streams, rows, "vllm" if precision == "transformers" else "transformers")
    err, err_other = np.abs(got - ref).max(), np.abs(got - other).max()
    assert err < 0.1, err
    if dtype == mx.float32:
        # The profiles now differ only in the E4M3 round trip and the combine
        # dtype; bf16 rounding noise is as large as that, fp32's is not.
        assert err < err_other / 3, (err, err_other)


def test_torch_rms_norm_matches_torch_eps_none_bf16():
    """torch.nn.RMSNorm(eps=None) on bf16: fp32 epsilon, weight in fp32, one rounding."""
    # All 1e-3 over 96 lanes: torch gives 0.9453 (fp32 eps); 2**-7 would give 0.0113.
    x = mx.full((1, 96), 1e-3, dtype=mx.bfloat16)
    assert abs(H.torch_rms_norm(x).astype(mx.float32)[0, 0].item() - 0.9453) < 2e-3
    rng = np.random.default_rng(3)
    x = mx.array(rng.standard_normal((64, 96)).astype(np.float32)).astype(mx.bfloat16)
    w = mx.array((1 + 0.3 * rng.standard_normal(96)).astype(np.float32)).astype(mx.bfloat16)
    xf = x.astype(mx.float32)
    inv = mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + F32_EPS)
    want = (xf * inv * w.astype(mx.float32)).astype(mx.bfloat16)
    assert mx.array_equal(H.torch_rms_norm(x, w), want).item()
    stacked = mx.broadcast_to(w, (4, 96))
    got = H.torch_rms_norm(x.reshape(16, 4, 96), stacked).reshape(64, 96)
    assert mx.array_equal(got, want).item()


def test_engram_precision_default_is_transformers():
    assert H.ModelArgs.from_hf_config(tiny_config()).engram_precision == "transformers"
    from mlx2.adapters import hy_v4_flash as A
    with pytest.raises(ValueError):
        A._parse_policy({"engram_precision": "fp64"})
    assert A._parse_policy({"engram_precision": "vllm"})[0]["engram_precision"] == "vllm"


# ------------------------------------------------- batching and AST (engine) --

def _greedy(model, prompt, n):
    cache = model.make_cache()
    out = [int(mx.argmax(model(mx.array([prompt]), cache)[0, -1]).item())]
    for _ in range(n - 1):
        out.append(int(mx.argmax(model(mx.array([[out[-1]]]), cache)[0, -1]).item()))
    return out


def _tiny_artifact(tmp_path, config, W):
    from tokenizers import Tokenizer, models

    path = _write_artifact(tmp_path, config, W)
    vocab = {f"t{i}": i for i in range(config["text_config"]["vocab_size"])}
    Tokenizer(models.WordLevel(vocab, unk_token="t0")).save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps(
        {"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "t2"}))
    return path


def _drain(batch, uids, n):
    out = {u: [] for u in uids}
    for _ in range(200):
        result = batch.next()
        responses = result[1] if isinstance(result, tuple) else result
        for r in responses:
            out[r.uid].append(int(r.token))
        if all(len(v) >= n for v in out.values()):
            return out
    raise AssertionError("scheduler stalled")


def test_batch_generator_ragged_lanes_equal_single_lane_greedy(tiny):
    from mlx2.runtime.generate import BatchGenerator

    _, _, model, _ = tiny
    prompts = [TOKENS[:10], TOKENS[:17], TOKENS[3:7]]
    batch = BatchGenerator(model, max_tokens=6, completion_batch_size=3, prefill_batch_size=2)
    uids = batch.insert(prompts, max_tokens=[6] * 3)
    out = _drain(batch, uids, 6)
    for uid, prompt in zip(uids, prompts):
        assert out[uid][:6] == _greedy(model, prompt, 6)


def test_batch_cache_merge_extract_filter_and_ragged_trim(tiny):
    _, _, model, _ = tiny
    a, b = model.make_cache(), model.make_cache()
    model(mx.array([TOKENS[:9]]), a)
    model(mx.array([TOKENS[:12]]), b)
    merged = [type(x).merge([x, y]) for x, y in zip(a, b)]
    assert merged[0].offset.tolist() == [9, 12]
    model(mx.array([[40, 41], [42, 43]]), merged)  # joint step, private histories
    for layer in merged:
        layer.trim_ragged([2, 1])
    assert [r.offset for r in merged[0].rows] == [9, 13]
    one = [layer.extract(1) for layer in merged]
    ref = model.make_cache()
    model(mx.array([TOKENS[:12] + [42]]), ref)
    np.testing.assert_allclose(np.array(model(mx.array([[44]]), one)),
                               np.array(model(mx.array([[44]]), ref)), rtol=1e-5, atol=1e-5)
    for layer in merged:
        layer.filter([0])
    assert merged[0].offset == 9


@pytest.mark.parametrize("tree,topology,tails", [
    # None reaches the engine's default topology (3x5 pattern); "drafter" is
    # the best-first search.
    (False, None, "conditioned"), (True, None, "conditioned"), (True, "3x2", "conditioned"),
    (True, "drafter", "conditioned"), (True, None, "independent"),
    (True, "drafter", "independent")])
def test_external_engine_dfly2_equals_greedy(tiny, tree, topology, tails, monkeypatch):
    from mlx2.adapters.hy_v4_flash_ast import Hy4ChainTreeExecutor
    from mlx2.runtime.drafters.hy4_dfly import Hy4DFlyDraftModel
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    _, _, model, head = tiny
    model.record_layers = frozenset(head.index_layers)
    try:
        monkeypatch.setenv("MLX2_HY4_DFLY_TREE_TAILS", tails)
        draft = Hy4DFlyDraftModel(model, head, num_draft=3, revision="test")
        assert draft.receipt_settings["tree_tails"] == tails
        executor = Hy4ChainTreeExecutor.create()
        options = dict(dynamic_singleton_tree=True, dynamic_tree_max_width=1,
                       target_tree_executor_provider=executor,
                       tree_node_budget_by_lanes={1: 7}, ast_topology=topology) if tree else {}
        batch = ExternalDraftBatchGenerator(model, draft_model=draft, binding="test", num_draft=3,
                                            prefill_step_size=4, **options)
        prompt = TOKENS[:13]
        uids = batch.insert([prompt], max_tokens=[16], sampling_configs=[{"sampling_temp": 0}])
        out = _drain(batch, uids, 16)
        assert out[uids[0]][:16] == _greedy(model, prompt, 16)
        assert draft.stats["block_drafts"] > 0
        if tree:
            assert draft.stats["tree_lattices"] > 0 and executor.receipt()["chain_forwards"] > 0
            conditioned = draft.stats["tree_block_forwards"] > 0
            assert conditioned == (tails == "conditioned")
            # One block forward per tree round; corrections only rerun the head.
            assert draft.stats["tree_block_forwards"] in (0, draft.stats["tree_lattices"])
    finally:
        model.record_layers = frozenset()


# ------------------------------------------------- DFly2 conditioned trees --

def _dfly_round(tiny, prompt_len, num_draft=3):
    from mlx2.runtime.drafters.hy4_dfly import Hy4DFlyDraftModel

    config, W, model, head = tiny
    model.record_layers = frozenset(head.index_layers)
    try:
        cache = model.make_cache()
        hidden = model.hidden(mx.array([TOKENS[:prompt_len]]), cache)
    finally:
        model.record_layers = frozenset()
    bonus = int(mx.argmax(model.logits(hidden[:, -1]), -1).item())
    draft = Hy4DFlyDraftModel(model, head, num_draft=num_draft, revision="test")

    def oracle(path):
        """fp64 logit rows ``[len(path) + 1, V]`` conditioned on ``path``."""
        _, rows = dfly_oracle(config, W, model, cache, bonus,
                              np.array(hidden[0, -1]).astype(np.float64), prompt_len,
                              num_draft, path=list(path))
        return rows

    def sequential(path):
        """MLX sequential recomputation: logits at step ``len(path)``."""
        forced = list(path)

        def choose(step, row):
            return mx.array([forced[step]]) if step < len(forced) else mx.argmax(row, -1)

        _, rows = head.propose(model, cache, mx.array([bonus]), hidden[:, -1], anchor=prompt_len,
                               num_draft=num_draft, choose=choose)
        return np.array(rows[0, len(forced)])

    return SimpleNamespace(draft=draft, cache=cache, hidden=hidden, bonus=bonus,
                           oracle=oracle, sequential=sequential, num_draft=num_draft)


def _path(tokens, parents, node):
    out = []
    while node >= 0:
        out.append(tokens[node])
        node = parents[node]
    return out[::-1]


def _log_softmax(row):
    row = np.asarray(row, dtype=np.float64)
    m = row.max()
    return row - (m + np.log(np.exp(row - m).sum()))


def _reference_best_first(oracle, depth, max_nodes, children, blocked=()):
    """Sequential best-first search, each node's law from a direct fp64
    recomputation along its path (same tie keys as the drafter)."""
    import heapq

    tokens, parents, heap = [], [], []

    def push(node, path, base):
        lp = _log_softmax(oracle(path)[len(path)])
        kept = 0
        for rank, token in enumerate(np.argsort(-lp, kind="stable")):
            if node < 0 and int(token) in blocked:
                continue
            heapq.heappush(heap, (-(base + lp[token]), node, rank, int(token), len(path) + 1))
            kept += 1
            if kept >= children:
                break

    push(-1, [], 0.0)
    while heap and len(tokens) < max_nodes:
        negative, parent, _rank, token, slot = heapq.heappop(heap)
        tokens.append(token)
        parents.append(parent)
        if slot < depth:
            push(len(tokens) - 1, _path(tokens, parents, len(tokens) - 1), -negative)
    return tokens, parents


@pytest.mark.parametrize("prompt_len,max_nodes,num_draft", [
    (6, 7, 3), (20, 7, 3), (20, 3, 3), (20, 15, 7), (6, 31, 7)])
def test_dfly_best_first_tree_nodes_are_parent_conditioned(tiny, prompt_len, max_nodes, num_draft):
    r = _dfly_round(tiny, prompt_len, num_draft=num_draft)
    draft = r.draft
    depth = min(max_nodes, r.num_draft)
    dcache = draft.make_cache()
    blocks = draft._tree_blocks([r.bonus], r.hidden, dcache, depth, [r.cache])
    record = {}
    tokens, parents = draft._best_first(blocks, [r.bonus], depth, max_nodes, [()], None,
                                        record=record)[0]
    assert len(tokens) == max_nodes and all(p < i for i, p in enumerate(parents))
    # Every node's scoring row equals the direct sequential recomputation
    # along its own path (plain MLX and the fp64 vLLM transcription).
    for node in range(len(tokens)):
        prefix = _path(tokens, parents, node)[:-1]
        row = np.array(record[(0, parents[node])])
        np.testing.assert_allclose(row, r.sequential(prefix), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(row, r.oracle(prefix)[len(prefix)], rtol=2e-4, atol=2e-4)
    assert (tokens, parents) == _reference_best_first(r.oracle, depth, max_nodes,
                                                      draft.tree_children)
    # The public entry point builds the same tree from one block forward.
    before = dict(draft.stats)
    public = draft.propose_tree([r.bonus], r.hidden[:, :0], dcache, max_nodes,
                                target_caches=[r.cache])
    assert public == [(tokens, parents)]
    assert draft.stats["tree_block_forwards"] - before["tree_block_forwards"] == 1
    # Corrections run in rounds of one batched call, at most one per depth.
    assert draft.stats["tree_correction_calls"] - before["tree_correction_calls"] <= depth


def test_dfly_best_first_respects_forbidden_root(tiny):
    r = _dfly_round(tiny, 20)
    greedy = int(np.argmax(r.oracle([])[0]))
    tokens, parents = r.draft.propose_tree([r.bonus], r.hidden, r.draft.make_cache(), 7,
                                           forbidden_token_ids=[(greedy,)],
                                           target_caches=[r.cache])[0]
    assert greedy not in [t for t, p in zip(tokens, parents) if p < 0]
    assert (tokens, parents) == _reference_best_first(r.oracle, 3, 7, r.draft.tree_children,
                                                      blocked={greedy})


@pytest.mark.parametrize("topology,max_nodes", [("3x3", 9), ("4x2", 8), ("chain", 3), ("3x3", 5)])
def test_dfly_pattern_branches_unroll_their_own_chain(tiny, topology, max_nodes):
    from mlx2.runtime.ast_tree_proposals import pattern_for_budget

    r = _dfly_round(tiny, 20)
    tokens, parents = r.draft.propose_tree([r.bonus], r.hidden, r.draft.make_cache(), max_nodes,
                                           topology=topology, target_caches=[r.cache])[0]
    pattern = [min(d, r.num_draft) for d in pattern_for_budget(topology, max_nodes)]
    root_row = r.oracle([])[0]
    roots = [t for t, p in zip(tokens, parents) if p < 0]
    assert roots == [int(t) for t in np.argsort(-root_row, kind="stable")[:len(pattern)]]
    for node in range(len(tokens)):
        if parents[node] < 0:
            continue
        prefix = _path(tokens, parents, node)[:-1]
        assert tokens[node] == int(np.argmax(r.sequential(prefix)))
        assert tokens[node] == int(np.argmax(r.oracle(prefix)[len(prefix)]))
    if topology == "chain":
        drafted, _ = r.draft.draft_distributions([r.bonus], r.hidden, r.draft.make_cache(),
                                                 max_nodes, [None], [0.0], target_caches=[r.cache])
        assert tokens == drafted[0] and parents == list(range(-1, max_nodes - 1))


def test_dfly_pattern_tree_skips_forbidden_root(tiny):
    r = _dfly_round(tiny, 20)
    greedy = int(np.argmax(r.oracle([])[0]))
    tokens, parents = r.draft.propose_tree([r.bonus], r.hidden, r.draft.make_cache(), 6,
                                           topology="2x3", forbidden_token_ids=[(greedy,)],
                                           target_caches=[r.cache])[0]
    roots = [t for t, p in zip(tokens, parents) if p < 0]
    assert greedy not in roots and len(roots) == 2


def test_tree_chains_cover_rooted_tree():
    from mlx2.adapters.hy_v4_flash_ast import chains

    parents = [-1, 0, 1, 1, 0, 4, 2]
    plan = chains(parents)
    nodes = sorted(n for _, chain in plan for n in chain)
    assert nodes == list(range(len(parents)))
    for source, chain in plan:
        assert parents[chain[0]] == source
        assert all(parents[b] == a for a, b in zip(chain, chain[1:]))


@pytest.mark.parametrize("drafter", ["embedded_dfly", "prompt_lookup"])
def test_adapter_ast_route_end_to_end(tmp_path, tiny, drafter):
    from mlx2.adapters import hy_v4_flash as A

    config, W, model, _ = tiny
    path = _tiny_artifact(tmp_path, config, W)
    adapter = A.HYV4FlashAdapter(str(path), execution_policy={"draft_model": drafter, "num_draft": 3})
    try:
        assert adapter.execution_config(max_lanes=1, prefill_step=4)["backend"] == "external_draft"
        batch = adapter.create_external_batch(completion_batch_size=1, prefill_step_size=4)
        prompt = TOKENS[:13]
        uids = batch.insert([prompt], max_tokens=[12], sampling_configs=[{"sampling_temp": 0}])
        out = _drain(batch, uids, 12)
        assert out[uids[0]][:12] == _greedy(adapter.model, prompt, 12)
        d = adapter.diagnostics()
        assert d["route"] == "external_draft" and d["ast"]["selected"]
        assert d["drafter"] == ("dfly" if drafter == "embedded_dfly" else "prompt_lookup")
        if drafter == "embedded_dfly":
            assert d["dfly2"]["observed_used"] and not d["dfly2"]["qualified"]
    finally:
        adapter.close()


def test_adapter_ordinary_route_declares_batching(tmp_path, tiny):
    from mlx2.adapters import hy_v4_flash as A
    from mlx2.contracts import Capability

    config, W, _, _ = tiny
    adapter = A.HYV4FlashAdapter(str(_tiny_artifact(tmp_path, config, W)))
    try:
        assert Capability.CONTINUOUS_BATCH in adapter.descriptor.capabilities
        assert adapter.execution_config(max_lanes=4, prefill_step=8)["segment_aware_cohort_size"] == 4
        assert adapter.diagnostics()["route"] == "ordinary"
    finally:
        adapter.close()


LONG = [int(t) for t in np.random.default_rng(5).integers(4, 90, size=100)]


def test_restored_cache_keeps_index_share_state(tiny):
    _, _, model, head = tiny
    model.record_layers = frozenset(head.index_layers)
    try:
        cache = model.make_cache()
        hidden = model.hidden(mx.array([LONG[:30]]), cache)
    finally:
        model.record_layers = frozenset()
    restored = [type(c).from_state(c.state, c.meta_state) for c in cache]
    assert restored[4].record and restored[4].selection_at(29) is not None
    args = dict(anchor=30, num_draft=3)
    a = head.propose(model, cache, mx.array([7]), hidden[:, -1], **args)
    b = head.propose(model, restored, mx.array([7]), hidden[:, -1], **args)
    np.testing.assert_allclose(np.array(a[1]), np.array(b[1]), rtol=1e-6, atol=1e-6)


def test_apc_preflight_refuses_rollback_past_sliding_window(tiny):
    from mlx2.runtime.models.cache import can_trim_prompt_cache_at

    _, _, model, _ = tiny
    cache = model.make_cache()
    model(mx.array([LONG]), cache)  # 100 > window - 1 + ROLLBACK = 71 retained rows
    assert can_trim_prompt_cache_at(cache, 20)
    assert not can_trim_prompt_cache_at(cache, 80)


def test_long_ragged_batch_exceeds_sliding_retention(tiny):
    from mlx2.runtime.generate import BatchGenerator

    _, _, model, _ = tiny
    prompts = [LONG, LONG[:13], LONG[20:67]]
    batch = BatchGenerator(model, max_tokens=8, completion_batch_size=3, prefill_batch_size=2,
                           prefill_step_size=16)
    uids = batch.insert(prompts, max_tokens=[8] * 3)
    out = _drain(batch, uids, 8)
    for uid, prompt in zip(uids, prompts):
        assert out[uid][:8] == _greedy(model, prompt, 8)


def test_external_engine_one_token_prompt_falls_back(tiny):
    from mlx2.runtime.drafters.hy4_dfly import Hy4DFlyDraftModel
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    _, _, model, head = tiny
    model.record_layers = frozenset(head.index_layers)
    try:
        draft = Hy4DFlyDraftModel(model, head, num_draft=3, revision="test")
        batch = ExternalDraftBatchGenerator(model, draft_model=draft, binding="test", num_draft=3)
        uids = batch.insert([[9]], max_tokens=[6], sampling_configs=[{"sampling_temp": 0}])
        assert _drain(batch, uids, 6)[uids[0]][:6] == _greedy(model, [9], 6)
    finally:
        model.record_layers = frozenset()


def test_adapter_ast_route_clamps_server_lane_count(tmp_path, tiny):
    from mlx2.adapters import hy_v4_flash as A

    config, W, _, _ = tiny
    adapter = A.HYV4FlashAdapter(str(_tiny_artifact(tmp_path, config, W)),
                                 execution_policy={"draft_model": "embedded_dfly"})
    try:
        # AST tiles admit one to four lanes; a larger server count clamps to 4.
        assert adapter.execution_config(max_lanes=16, prefill_step=8)["segment_aware_cohort_size"] == 4
        batch = adapter.create_external_batch(completion_batch_size=16, prefill_step_size=8)
        uids = batch.insert([LONG[:20]], max_tokens=[6], sampling_configs=[{"sampling_temp": 0}])
        assert _drain(batch, uids, 6)[uids[0]][:6] == _greedy(adapter.model, LONG[:20], 6)
    finally:
        adapter.close()


def test_rope_is_interleaved_complex_rotation():
    """Both MLA and the indexer rotate adjacent pairs (x[2i], x[2i+1]) as one
    complex number (``view_as_complex``), confirmed by Tencent for HY-V4
    (transformers#50589); a consistently split-half RoPE would still pass
    prefill-versus-decode checks, so check the layout itself."""
    args = H.ModelArgs.from_hf_config(tiny_config())
    attn = H.Attention(args, H.INDEXED, indexer=True)
    rng = np.random.default_rng(11)
    x = rng.standard_normal((1, 1, 5, 4)).astype(np.float32)
    theta = args.rope_parameters[H.INDEXED]["rope_theta"]
    inv = 1.0 / theta ** (np.arange(0, 4, 2) / 4)
    pos = np.arange(3, 8)[:, None]
    z = (x[..., 0::2] + 1j * x[..., 1::2]) * np.exp(1j * pos * inv)
    ref = np.stack([z.real, z.imag], axis=-1).reshape(x.shape)
    np.testing.assert_allclose(np.array(attn.rope(mx.array(x), 3)), ref, rtol=1e-5, atol=1e-5)
    # Indexer: rotate only the trailing rope lanes, pooled groups at first-token phase.
    d = args.compressed_indexer_head_dim
    y = rng.standard_normal((1, 3, d)).astype(np.float32)
    got = np.array(attn.compressed_indexer._rope_tail(mx.array(y), 2, scale=float(args.compressed_indexer_compress_ratio)))
    tail = y[..., -4:]
    gpos = (np.arange(2, 5) * args.compressed_indexer_compress_ratio)[:, None]
    zt = (tail[..., 0::2] + 1j * tail[..., 1::2]) * np.exp(1j * gpos * inv)
    np.testing.assert_allclose(got[..., -4:], np.stack([zt.real, zt.imag], -1).reshape(tail.shape), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(got[..., :-4], y[..., :-4])


def test_engram_ple_offload_prefill_and_decode_pools(tmp_path):
    from mlx2.runtime.models.hy_v4_flash_engram import PREFILL_ROW_THRESHOLD, EngramTableRows

    rng = np.random.default_rng(2)
    table = rng.standard_normal((2000, 4)).astype(np.float32)
    bf16 = (table.view(np.uint32) >> 16).astype(np.uint16)
    header = json.dumps({"t": {"dtype": "BF16", "shape": [2000, 4], "data_offsets": [0, bf16.nbytes]}}).encode()
    (tmp_path / "x.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + bf16.tobytes())
    reader = EngramTableRows(tmp_path / "x.safetensors", "t", cache_rows=4096)
    try:
        expect = (bf16.astype(np.uint32) << 16).view(np.float32)
        rows = np.arange(PREFILL_ROW_THRESHOLD + 7)
        np.testing.assert_array_equal(reader.read_host(rows), expect[rows])
        np.testing.assert_array_equal(reader.read_host(np.array([1999, 3])), expect[[1999, 3]])
        d = reader.diagnostics()
        assert d["offload"] == "ple_file_backed" and d["prefill_reads"] == 1 and d["decode_reads"] == 1
    finally:
        reader.close()


def test_adapter_declares_tools_and_routes_them_through_the_hy4_parser(tmp_path, tiny):
    from mlx2.adapters import hy_v4_flash as A
    from mlx2.adapters.hy_v4_flash_output import HYV4OutputParser
    from mlx2.contracts import Capability

    config, W, _, _ = tiny
    adapter = A.HYV4FlashAdapter(str(_tiny_artifact(tmp_path, config, W)))
    try:
        assert Capability.TOOLS in adapter.descriptor.capabilities
        request = {"messages": [{"role": "user", "content": "x"}],
                   "tools": [{"type": "function", "function": {"name": "now"}}]}
        parser = adapter.output_parser(request)
        assert isinstance(parser, HYV4OutputParser)
        events = parser.push(f"<tool_call{adapter.tool_suffix}>now</tool_call{adapter.tool_suffix}>", final=True)
        assert events[0]["tool_calls"][0]["function"]["name"] == "now"
        grammar = adapter.tool_constraint({**request, "tool_choice": "required"})
        assert grammar.startswith(re.escape(f"<tool_calls{adapter.tool_suffix}>"))
        assert adapter.tool_constraint(request) is None
        assert Capability.REASONING in adapter.descriptor.capabilities
        assert adapter.thinking_enabled(request) is False
        diagnostics = adapter.diagnostics()
        assert diagnostics["tool_choice_forced"]["grammar_hook_calls"] == 1
        assert diagnostics["reasoning"] == {"implemented": True, "qualified": False, "default": "no_think",
                                            "selected": False, "observed_used": False, "requests": 0}
    finally:
        adapter.close()


def _bf16_table(tmp_path, rows=64, dim=4, truncate=0):
    table = np.random.default_rng(3).standard_normal((rows, dim)).astype(np.float32)
    bf16 = (table.view(np.uint32) >> 16).astype(np.uint16)
    header = json.dumps({"t": {"dtype": "BF16", "shape": [rows, dim], "data_offsets": [0, bf16.nbytes]}}).encode()
    data = len(header).to_bytes(8, "little") + header + bf16.tobytes()
    path = tmp_path / "t.safetensors"
    path.write_bytes(data[: len(data) - truncate])
    return path


def test_engram_reader_short_shard_refused_without_fd_leak(tmp_path):
    import os
    from mlx2.runtime.models.hy_v4_flash_engram import EngramTableRows

    path = _bf16_table(tmp_path, truncate=16)
    before = len(os.listdir("/dev/fd"))
    with pytest.raises(ValueError, match="bytes"):
        EngramTableRows(path, "t")
    assert len(os.listdir("/dev/fd")) <= before


def test_engram_reader_timeout_raises_read_timeout_and_close_is_safe(tmp_path, monkeypatch):
    import time as _time
    from mlx2.runtime.io_ownership import ReadTimeout
    from mlx2.runtime.models.hy_v4_flash_engram import EngramTableRows

    reader = EngramTableRows(_bf16_table(tmp_path), "t", timeout=0.2)
    monkeypatch.setattr(reader, "_read_chunk", lambda fd, rows: _time.sleep(0.5) or [])
    with pytest.raises(ReadTimeout):
        reader.read_host(np.array([1, 2]))
    reader.close()
    with pytest.raises(RuntimeError, match="closed"):
        reader.read_host(np.array([3]))
    reader.close()


@pytest.mark.parametrize("topology,max_nodes", [(None, 7), ("3x2", 6)])
def test_dfly_tree_lanes_batch_like_single_lanes(tiny, topology, max_nodes):
    lanes = [_dfly_round(tiny, 20), _dfly_round(tiny, 13)]
    draft = lanes[0].draft
    banned = [(), (int(np.argmax(lanes[1].oracle([])[0])),)]
    single = [lane.draft.propose_tree([lane.bonus], lane.hidden, lane.draft.make_cache(), max_nodes,
                                      topology=topology, forbidden_token_ids=[banned[i]],
                                      target_caches=[lane.cache])[0]
              for i, lane in enumerate(lanes)]
    rows = []
    for lane in lanes:
        cache = draft.make_cache()
        draft.append_context(lane.hidden, cache)
        rows.append(cache)
    width = lanes[0].hidden.shape[-1]
    both = draft.propose_tree([lane.bonus for lane in lanes], mx.zeros((2, 0, width)),
                              draft.batch_caches(rows), max_nodes, topology=topology,
                              forbidden_token_ids=banned,
                              target_caches=[lane.cache for lane in lanes])
    assert both == single
