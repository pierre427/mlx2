"""Kimi-K3.1 MLX math against an independent FP32 PyTorch transcription (CPU).

The oracle transcribes the Moonshot K3 reference (moonshotai/Kimi-K3
@9f62e4e9 modeling_kimi_linear.py) with FLA's naive KDA recurrence and its
in-kernel per-head gate (fla/ops/kda/gate.py reads A_log[i_h]); MLA is dense
because the prompt is shorter than the DSA top-k. Checkpoint:
tiny-random/kimi-k3.1-pro@507bf23a377e31775c3a79b63b2c42bd6e4b9cc0. Unqualified.
"""
import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters import kimi_k31 as A
from mlx2.runtime.models import kimi_k31 as k31

TINY = Path(os.environ.get("MLX2_KIMI_K31_TINY", Path.home() / ".cache/mlx2/tiny-random/"
                           "kimi-k3.1-pro-507bf23a377e31775c3a79b63b2c42bd6e4b9cc0"))
TOKENS = [163584, 100, 2000, 30000, 7, 999, 12345, 5, 4242, 88, 161, 70000]

torch = pytest.importorskip("torch")
F = torch.nn.functional
pytestmark = pytest.mark.skipif(not (TINY / "model.safetensors").exists(),
                                reason="tiny-random/kimi-k3.1-pro not present")


def reference_logits(tensors, config, tokens):
    c = config["text_config"]
    lin = c["linear_attn_config"]
    eps = c["rms_norm_eps"]
    P = "language_model.model."
    W = {k: torch.from_numpy(np.array(v.astype(mx.float32))) for k, v in tensors.items()
         if k.startswith("language_model.") and v.dtype != mx.uint8}

    def rms(x, w, e=eps):
        return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + e))

    def situ(gt, up):
        b, lb = c["activation_situ_beta"], c["activation_situ_linear_beta"]
        return b * torch.tanh(gt / b) * torch.sigmoid(gt) * (lb * torch.tanh(up / lb))

    def mix(prefix, blocks, norm_w, proj_w):
        v = torch.stack(blocks + [prefix], dim=1)                       # [T, S, D]
        k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
        probs = (k * (norm_w * proj_w.reshape(-1))).sum(-1).softmax(-1)
        return (probs.unsqueeze(1) @ v).squeeze(1)

    def kda(p, x):
        T = x.shape[0]
        H, D, K = lin["num_heads"], lin["head_dim"], lin["short_conv_kernel_size"]

        def conv(name, y):
            w = W[p + f"self_attn.{name}.weight"]
            return F.silu(F.conv1d(y.T.unsqueeze(0), w, padding=K - 1, groups=w.shape[0])[0, :, :T].T)

        q = conv("q_conv1d", x @ W[p + "self_attn.q_proj.weight"].T).reshape(T, H, D)
        k = conv("k_conv1d", x @ W[p + "self_attn.k_proj.weight"].T).reshape(T, H, D)
        v = conv("v_conv1d", x @ W[p + "self_attn.v_proj.weight"].T).reshape(T, H, D)
        q = q * torch.rsqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6) * D ** -0.5
        k = k * torch.rsqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)
        a = (x @ W[p + "self_attn.f_a_proj.weight"].T) @ W[p + "self_attn.f_b_proj.weight"].T
        A_ = W[p + "self_attn.A_log"][:H].exp().view(H, 1)
        gate = lin["gate_lower_bound"] * torch.sigmoid(
            A_ * (a.reshape(T, H, D) + W[p + "self_attn.dt_bias"].view(H, D)))
        beta = torch.sigmoid(x @ W[p + "self_attn.b_proj.weight"].T)
        S = torch.zeros(H, D, D)                                        # [H, K, V]
        outs = []
        for t in range(T):
            S = S * gate[t].exp().unsqueeze(-1)
            u = beta[t].unsqueeze(-1) * (v[t] - torch.einsum("hk,hkv->hv", k[t], S))
            S = S + k[t].unsqueeze(-1) * u.unsqueeze(1)
            outs.append(torch.einsum("hk,hkv->hv", q[t], S))
        o = rms(torch.stack(outs), W[p + "self_attn.o_norm.weight"]) * torch.sigmoid(
            (x @ W[p + "self_attn.g_proj.weight"].T).reshape(T, H, D))
        return o.reshape(T, H * D) @ W[p + "self_attn.o_proj.weight"].T

    def mla(p, x):
        T = x.shape[0]
        H, dn, dr, dv, r = (c["num_attention_heads"], c["qk_nope_head_dim"], c["qk_rope_head_dim"],
                            c["v_head_dim"], c["kv_lora_rank"])
        q = rms(x @ W[p + "self_attn.q_a_proj.weight"].T, W[p + "self_attn.q_a_layernorm.weight"], 1e-6)
        q = (q @ W[p + "self_attn.q_b_proj.weight"].T).reshape(T, H, dn + dr)
        ckv = x @ W[p + "self_attn.kv_a_proj_with_mqa.weight"].T
        kv = rms(ckv[:, :r], W[p + "self_attn.kv_a_layernorm.weight"], 1e-6)
        kv = (kv @ W[p + "self_attn.kv_b_proj.weight"].T).reshape(T, H, dn + dv)
        k = torch.cat([kv[..., :dn], ckv[:, None, r:].expand(T, H, dr)], -1)
        s = torch.einsum("thd,shd->hts", q, k) * (dn + dr) ** -0.5
        s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
        o = torch.einsum("hts,shd->thd", s.softmax(-1), kv[..., dn:]).reshape(T, H * dv)
        o = o * torch.sigmoid(x @ W[p + "self_attn.g_proj.weight"].T)
        return o @ W[p + "self_attn.o_proj.weight"].T

    def mlp(p, x):
        return situ(x @ W[p + "gate_proj.weight"].T, x @ W[p + "up_proj.weight"].T) @ W[p + "down_proj.weight"].T

    def expert(p, e, name):
        key = p + f"block_sparse_moe.experts.{e}.{name}"
        return torch.from_numpy(k31.mxfp4_dequantize_reference(
            np.array(tensors[key + ".weight_packed"]), np.array(tensors[key + ".weight_scale"])))

    def moe(p, x):
        scores = torch.sigmoid(x @ W[p + "block_sparse_moe.gate.weight"].T)
        _, idx = torch.topk(scores + W[p + "block_sparse_moe.gate.e_score_correction_bias"],
                            c["num_experts_per_token"], dim=-1)
        wts = scores.gather(1, idx)
        wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * c["routed_scaling_factor"]
        xl = x @ W[p + "block_sparse_moe.routed_expert_down_proj.weight"].T
        y = torch.zeros_like(xl)
        for t in range(x.shape[0]):
            for slot in range(idx.shape[1]):
                w1, w2, w3 = (expert(p, int(idx[t, slot]), n) for n in ("w1", "w2", "w3"))
                y[t] += wts[t, slot] * (situ(xl[t] @ w1.T, xl[t] @ w3.T) @ w2.T)
        y = rms(y, W[p + "block_sparse_moe.routed_expert_norm.weight"])
        y = y @ W[p + "block_sparse_moe.routed_expert_up_proj.weight"].T
        return y + mlp(p + "block_sparse_moe.shared_experts.", x)

    kda_layers = {i - 1 for i in lin["kda_layers"]}
    prefix, blocks = W[P + "embed_tokens.weight"][torch.tensor(tokens)], []
    for i in range(c["num_hidden_layers"]):
        p = P + f"layers.{i}."
        h = mix(prefix, blocks, W[p + "self_attention_res_norm.weight"],
                W[p + "self_attention_res_proj.weight"]) if blocks else prefix
        if i % c["attn_res_block_size"] == 0:
            blocks, prefix = blocks + [prefix], None
        xn = rms(h, W[p + "input_layernorm.weight"])
        y = kda(p, xn) if i in kda_layers else mla(p, xn)
        prefix = y if prefix is None else prefix + y
        h = mix(prefix, blocks, W[p + "mlp_res_norm.weight"], W[p + "mlp_res_proj.weight"])
        xn = rms(h, W[p + "post_attention_layernorm.weight"])
        prefix = prefix + (mlp(p + "mlp.", xn) if i < c["first_k_dense_replace"] else moe(p, xn))
    h = mix(prefix, blocks, W[P + "output_attn_res_norm.weight"], W[P + "output_attn_res_proj.weight"])
    return rms(h, W[P + "norm.weight"]) @ W["language_model.lm_head.weight"].T


def test_whole_model_matches_independent_torch_reference():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        tensors = mx.load(str(TINY / "model.safetensors"))
        config = json.loads((TINY / "config.json").read_text())
        with torch.no_grad():
            ref = reference_logits(tensors, config, TOKENS).numpy()
        got = np.array(A.whole_model_logits(tensors, config, TOKENS, dtype=mx.float32, last_only=False))
    finally:
        mx.set_default_device(old)
    err = np.abs(got - ref).max() / np.abs(ref).max()
    print("fp32 mlx vs torch reference: max rel err", err)
    assert err < 1e-3, err
    assert (got.argmax(-1) == ref.argmax(-1)).all()
