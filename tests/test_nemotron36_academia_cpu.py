"""Nemotron 3.6 Academia runtime vs an independent NumPy port of llama.cpp.

No checkpoint exists (2026-11-23), so every check runs on a synthetic config
(the llama.cpp ``test-llama-archs`` toy: pattern ``MM*--``, 1 input, 3
recurrent, 1 output layer; GDP Dk=64, Dv=32).  The reference below is a
line-by-line float64 translation of ggml-org/llama.cpp#32766's graph
(``nemotron3-6-academia.cpp``) and #32765's CPU ``gated_delta_net`` loop; it
shares no code with the mlx model except the weight dictionary.  CPU only.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx2.runtime.models import nemotron36_academia as academia


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def config(**overrides):
    base = dict(
        model_type="nemotron3_6_academia",
        hidden_size=64,
        vocab_size=97,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        intermediate_size=96,
        rms_norm_eps=1e-5,
        hidden_act="relu2",
        n_input_layers=1,
        n_recurrent_layers=3,
        n_output_layers=1,
        n_recurrent_loops=2,
        layer_types=["linear_attention", "linear_attention", "full_attention", "mlp", "mlp"],
        recurrent_kv_cache_mode="per_loop",
        loop_conditioning_hidden_size=16,
        loop_conditioning_max_loops=4,
        loop_adapter_rank=8,
        loop_input_mixing=True,
        linear_conv_kernel_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=64,
        linear_value_head_dim=32,
        linear_num_householder=3,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------- weights

def hf_weights(cfg, seed=0):
    """Random weights under Hugging Face names (conv1d in torch [C, 1, K])."""
    model = academia.Model(academia.ModelArgs.from_dict(cfg))
    rng = np.random.default_rng(seed)
    out = {}
    for name, value in tree_flatten(model.parameters()):
        shape = tuple(value.shape)
        if name.endswith("embed_tokens.weight"):
            array = rng.normal(size=shape)
        elif name.endswith(("norm.weight",)):
            array = 1.0 + 0.1 * rng.normal(size=shape)
        elif name.endswith("A_log"):
            array = rng.uniform(-1.0, 1.0, size=shape)
        elif name.endswith(("dt_bias", "base_alpha_raw")):
            array = rng.normal(size=shape)
        elif name.endswith("conv1d.weight"):
            array = 0.5 * rng.normal(size=shape)  # mlx [C, K, 1]
            array = np.moveaxis(array, 1, 2)  # torch [C, 1, K]
        else:
            fan_in = shape[-1] if len(shape) > 1 else shape[0]
            array = rng.normal(size=shape) / math.sqrt(fan_in)
        name = name.replace("loop_conditioner.up.", "loop_conditioner.mlp.0.")
        name = name.replace("loop_conditioner.down.", "loop_conditioner.mlp.2.")
        out[name] = array.astype(np.float32)
    return out


def mlx_model(cfg, weights):
    model = academia.Model(academia.ModelArgs.from_dict(cfg))
    sanitized = model.sanitize({k: mx.array(v) for k, v in weights.items()})
    model.load_weights(list(sanitized.items()), strict=True)
    model.eval()
    return model


# ---------------------------------------------------------------- reference

def softplus(x):
    return np.logaddexp(x, 0.0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def rms(x, eps, w=None):
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return y if w is None else y * w


class Reference:
    """llama.cpp's Nemotron3_6Academia graph, one ubatch per ``forward``."""

    def __init__(self, cfg, weights, max_tokens=64):
        self.c = cfg
        self.W = {k: v.astype(np.float64) for k, v in weights.items()}
        kinds = {"linear_attention": "M", "full_attention": "*", "mlp": "-"}
        self.pattern = "".join(kinds[t] for t in cfg["layer_types"])
        n_in, n_rec, n_out = cfg["n_input_layers"], cfg["n_recurrent_layers"], cfg["n_output_layers"]
        loops = cfg["n_recurrent_loops"]
        self.physical = (list(range(n_in)) + [n_in + i for _ in range(loops) for i in range(n_rec)]
                         + [n_in + n_rec + i for i in range(n_out)])
        mode = cfg["recurrent_kv_cache_mode"]
        self.cache_layer = []
        for il, p in enumerate(self.physical):
            recurrent_attn = self.pattern[p] == "*" and n_in <= p < n_in + n_rec
            self.cache_layer.append(p if recurrent_attn and mode == "last_loop" else il)
        hd, nk = cfg["head_dim"], cfg["num_key_value_heads"]
        self.kv = {il: (np.zeros((max_tokens, nk, hd)), np.zeros((max_tokens, nk, hd)))
                   for il in set(self.cache_layer)}
        m, hk, hv = cfg["linear_num_householder"], cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
        sk, sv = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
        cd = m * hv * sv + (m + 1) * hk * sk
        self.conv = {il: np.zeros((cfg["linear_conv_kernel_dim"] - 1, cd)) for il in range(len(self.physical))}
        self.S = {il: np.zeros((hv, sk, sv)) for il in range(len(self.physical))}  # S[i][j], i over k
        self.pos = 0

    def lin(self, name, x, bias=True):
        y = x @ self.W[name + ".weight"].T
        return y + self.W[name + ".bias"] if bias else y

    def condition(self, progress, budget):
        n_cond = self.c["loop_conditioning_hidden_size"]
        half = n_cond // 2
        freq = np.exp(np.arange(half) * (-math.log(100.0) / half))

        def embed(value):
            args = freq * 100.0 * value
            return np.concatenate([np.cos(args), np.sin(args)])

        cur = np.concatenate([embed(progress), embed(budget)])
        cur = self.lin("model.loop_conditioner.mlp.0", cur)
        return self.lin("model.loop_conditioner.mlp.2", silu(cur))

    def forward(self, tokens):
        c, eps = self.c, self.c["rms_norm_eps"]
        n = len(tokens)
        cur = self.W["model.embed_tokens.weight"][tokens]
        loops, max_loops = c["n_recurrent_loops"], c["loop_conditioning_max_loops"]
        budget = min(loops, max_loops) / max_loops
        conds = [self.condition((k + 1) / loops, budget) for k in range(loops)]
        alpha = softplus(self.W["model.state_update.base_alpha_raw"])
        il = 0
        for _ in range(c["n_input_layers"]):
            cur = self.layer(cur, None, il)
            il += 1
        input_h = state = cur
        for k in range(loops):
            cond = conds[k]
            loop_input = state + input_h
            cur = loop_input
            if k > 0:
                scale = self.lin("model.loop_seam.entry_gamma", cond) + 1.0
                cur = rms(cur, eps, self.W["model.loop_seam.entry_norm.weight"])
                cur = cur * scale + self.lin("model.loop_seam.entry_beta", cond)
                if c["loop_input_mixing"]:
                    cur = cur + input_h * self.lin("model.loop_seam.input_mix", cond)
            for _ in range(c["n_recurrent_layers"]):
                cur = self.layer(cur, cond, il)
                il += 1
            beta = sigmoid(self.lin("model.state_update.proj", loop_input)) * alpha
            new_state = state + beta * (cur - state)
            gate = self.lin("model.loop_seam.exit_gate", cond) + (1.0 if k == 0 else 0.0)
            state = state + (new_state - state) * gate
        cur = state
        for _ in range(c["n_output_layers"]):
            cur = self.layer(cur, conds[-1], il)
            il += 1
        cur = rms(cur, eps, self.W["model.norm.weight"])
        self.pos += n
        head = self.W.get("lm_head.weight", self.W["model.embed_tokens.weight"])
        return cur @ head.T

    def layer(self, cur, cond, il):
        p = self.physical[il]
        pre = f"model.layers.{p}"
        residual = cur
        cur = rms(cur, self.c["rms_norm_eps"], self.W[pre + ".norm.weight"])
        gate = None
        if cond is not None:
            lc = pre + ".loop_conditioning"
            cur = cur * (self.lin(lc + ".gamma_head", cond) + 1.0) + self.lin(lc + ".beta_head", cond)
            gate = self.lin(lc + ".gate_head", cond) + 1.0
            if self.c["loop_adapter_rank"]:
                adapter = self.lin(lc + ".down_proj", cur, bias=False) * self.lin(lc + ".adapter_gate", cond)
                cur = cur + self.lin(lc + ".up_proj", adapter, bias=False)
        kind = self.pattern[p]
        if kind == "-":
            up = self.lin(pre + ".mixer.up_proj", cur, bias=False)
            cur = self.lin(pre + ".mixer.down_proj", np.maximum(up, 0.0) ** 2, bias=False)
        elif kind == "*":
            cur = self.attention(cur, il, pre + ".mixer")
        else:
            cur = self.gdp(cur, il, pre + ".mixer")
        if gate is not None:
            cur = cur * gate
        return residual + cur

    def attention(self, x, il, pre):
        c = self.c
        n, nh, nk, hd = len(x), c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        q = self.lin(pre + ".q_proj", x, bias=False).reshape(n, nh, hd)
        k = self.lin(pre + ".k_proj", x, bias=False).reshape(n, nk, hd)
        v = self.lin(pre + ".v_proj", x, bias=False).reshape(n, nk, hd)
        K, V = self.kv[self.cache_layer[il]]
        K[self.pos : self.pos + n] = k  # write this loop's rows, then read
        V[self.pos : self.pos + n] = v
        out = np.zeros((n, nh, hd))
        for t in range(n):
            last = self.pos + t + 1
            for h in range(nh):
                g = h // (nh // nk)
                s = K[:last, g] @ q[t, h] / math.sqrt(hd)
                w = np.exp(s - s.max())
                out[t, h] = (w / w.sum()) @ V[:last, g]
        return self.lin(pre + ".o_proj", out.reshape(n, nh * hd), bias=False)

    def gdp(self, x, il, pre):
        c = self.c
        m, hk, hv = c["linear_num_householder"], c["linear_num_key_heads"], c["linear_num_value_heads"]
        sk, sv = c["linear_key_head_dim"], c["linear_value_head_dim"]
        n, kd, vd = len(x), hk * sk, hv * sv
        q = self.lin(pre + ".q_proj", x, bias=False)
        k = self.lin(pre + ".k_proj", x, bias=False)
        v = self.lin(pre + ".v_proj", x, bias=False)
        z = self.lin(pre + ".z_proj", x, bias=False)
        mixed = np.concatenate([v, k, q], axis=1)  # ggml_concat(concat(v, k), q)
        conv_in = np.concatenate([self.conv[il], mixed], axis=0)
        self.conv[il] = conv_in[-(c["linear_conv_kernel_dim"] - 1):]
        w = self.W[pre + ".conv1d.weight"][:, 0, :]  # [C, K]
        width = w.shape[1]
        conv = np.stack([np.sum(conv_in[t : t + width].T * w, axis=1) for t in range(n)])
        conv = silu(conv)
        vs = conv[:, : m * vd].reshape(n, m, hv, sv)
        ks = conv[:, m * vd : m * (vd + kd)].reshape(n, m, hk, sk)
        qs = conv[:, m * (vd + kd) :].reshape(n, hk, sk)

        def l2(a):  # build_gdn_l2_norm: x * rsqrt(sum(x^2) + 1e-6)
            return a / np.sqrt(np.sum(a * a, axis=-1, keepdims=True) + 1e-6)

        ks, qs = l2(ks), l2(qs)
        a = self.lin(pre + ".a_proj", x, bias=False) + self.W[pre + ".dt_bias"]
        g = softplus(a) * (-np.exp(self.W[pre + ".A_log"]))  # converter: A = -exp(A_log)
        b = sigmoid(self.lin(pre + ".b_proj", x, bias=False)).reshape(n, m, hv)
        S = self.S[il]
        out = np.zeros((n, hv, sv))
        scale = 1.0 / math.sqrt(sk)
        for t in range(n):
            for j in range(m):
                for h in range(hv):
                    kh = h // (hv // hk)
                    if j == 0:
                        S[h] *= math.exp(g[t, h])  # ggml_pad: zero log-decay on later sub-steps
                    kv = ks[t, j, kh]
                    delta = (vs[t, j, h] - kv @ S[h]) * b[t, j, h]
                    S[h] += np.outer(kv, delta)
                    if j == m - 1:
                        out[t, h] = (qs[t, kh] @ S[h]) * scale
        out = out.reshape(n, vd) * silu(z)
        out = rms(out.reshape(n, hk, vd // hk), c["rms_norm_eps"]).reshape(n, vd)
        out = out * self.W[pre + ".norm.weight"]
        return self.lin(pre + ".out_proj", out, bias=False)


# ---------------------------------------------------------------- helpers

def tokens(n, seed=3):
    return np.random.default_rng(seed).integers(0, 97, size=n).tolist()


def run_mlx(model, chunks):
    cache = model.make_cache()
    rows = []
    for chunk in chunks:
        logits = model(mx.array([chunk]), cache=cache)
        mx.eval(logits)
        rows.append(np.array(logits[0], dtype=np.float64))
    return np.concatenate(rows)


def run_ref(cfg, weights, chunks):
    ref = Reference(cfg, weights)
    return np.concatenate([ref.forward(chunk) for chunk in chunks])


def nmse(ref, actual):
    return float(np.sum((ref - actual) ** 2) / np.sum(ref ** 2))


def split(seq, size):
    return [seq[i : i + size] for i in range(0, len(seq), size)]


# ---------------------------------------------------------------- tests

@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"linear_num_householder": 1},
        {"n_recurrent_loops": 1},
        {"n_recurrent_loops": 3, "loop_adapter_rank": 0, "loop_input_mixing": False},
        {"recurrent_kv_cache_mode": "last_loop", "n_recurrent_loops": 3},
        {"tie_word_embeddings": True, "n_recurrent_loops": 5},  # loops > max_loops
    ],
)
def test_whole_prompt_matches_reference(overrides):
    cfg = config(**overrides)
    weights = hf_weights(cfg)
    if cfg.get("tie_word_embeddings"):
        assert "lm_head.weight" not in weights
    prompt = tokens(12)
    ref = run_ref(cfg, weights, [prompt])
    got = run_mlx(mlx_model(cfg, weights), [prompt])
    assert nmse(ref, got) < 1e-9


@pytest.mark.parametrize("mode", ["per_loop", "last_loop"])
def test_chunked_prefill_and_decode_match_reference(mode):
    cfg = config(recurrent_kv_cache_mode=mode, n_recurrent_loops=3)
    weights = hf_weights(cfg, seed=1)
    prompt = tokens(17, seed=5)
    chunks = split(prompt[:16], 4) + [[t] for t in prompt[16:]]
    ref = run_ref(cfg, weights, chunks)
    got = run_mlx(mlx_model(cfg, weights), chunks)
    assert nmse(ref, got) < 1e-9


def test_cache_modes_agree_whole_prompt_and_differ_after_chunking():
    """The llama.cpp ``check_academia_cache`` assertion, on the mlx model."""
    cfg_per = config(n_recurrent_loops=3)
    cfg_last = config(n_recurrent_loops=3, recurrent_kv_cache_mode="last_loop")
    weights = hf_weights(cfg_per, seed=2)
    per, last = mlx_model(cfg_per, weights), mlx_model(cfg_last, weights)
    prompt = tokens(16, seed=7)
    np.testing.assert_allclose(run_mlx(per, [prompt]), run_mlx(last, [prompt]), rtol=0, atol=0)
    per_whole = run_mlx(per, [prompt])
    per_chunked = run_mlx(per, split(prompt, 4))
    last_chunked = run_mlx(last, split(prompt, 4))
    assert nmse(per_whole, per_chunked) < 1e-9  # per_loop is partition-invariant
    assert not np.allclose(per_chunked[-1], last_chunked[-1], rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("backend", ["sdpa", "bucketed"])
@pytest.mark.parametrize("mode", ["per_loop", "last_loop"])
def test_left_padded_batch_matches_single_rows(mode, backend):
    """Batch caches (BatchKVCache + padded ArraysCache) reproduce B=1 runs."""
    from mlx2.runtime.models.cache import ArraysCache, BatchKVCache, KVCache

    cfg = config(recurrent_kv_cache_mode=mode, n_recurrent_loops=2)
    model = mlx_model(cfg, hf_weights(cfg, seed=4))
    prompts = [tokens(40, seed=21), tokens(3, seed=22)]  # wide gap: bucketing engages
    width = max(map(len, prompts))
    padding = [width - len(p) for p in prompts]
    batch = []
    for plane in model.make_cache():
        if isinstance(plane, KVCache):
            fresh = BatchKVCache(padding, attention_backend=backend)
            if type(plane) is academia.LastLoopKVCache:
                fresh.__class__ = academia.LastLoopBatchKVCache
            batch.append(fresh)
        else:
            fresh = ArraysCache(2, left_padding=padding)
            fresh.apcv2_layer_index = plane.apcv2_layer_index
            batch.append(fresh)
    rows = mx.array([[0] * pad + p for pad, p in zip(padding, prompts)])
    prefill = model(rows, cache=batch)
    step = model(mx.array([[5], [7]]), cache=batch)
    mx.eval(prefill, step)
    for row, prompt in enumerate(prompts):
        alone = run_mlx(model, [prompt, [5 if row == 0 else 7]])
        np.testing.assert_allclose(np.array(prefill[row, -1]), alone[len(prompt) - 1], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(np.array(step[row, -1]), alone[-1], rtol=1e-4, atol=1e-4)
    if backend == "bucketed":
        calls = sum(p._attention_backend_stats["bucketed_calls"] for p in batch
                    if isinstance(p, BatchKVCache))
        assert calls > 0


@pytest.mark.parametrize("mode", ["per_loop", "last_loop"])
def test_apcv2_refuses_last_loop_prefixes(mode):
    """last_loop KV depends on the prefill partition, so APCv2 must not store it."""
    from mlx2.runtime.apc_v2 import inspect_apc_capabilities

    cfg = config(recurrent_kv_cache_mode=mode)
    model = mlx_model(cfg, hf_weights(cfg, seed=6))
    caches = [model.make_cache(), model.make_cache()]
    for cache, prompt in zip(caches, (tokens(5, seed=1), tokens(5, seed=2))):
        model(mx.array([prompt]), cache=cache)
    assert inspect_apc_capabilities(caches[0]).exact_prefix is (mode == "per_loop")
    merged = [plane.merge([caches[0][i], caches[1][i]]) for i, plane in enumerate(caches[0])]
    model(mx.array([[3], [4]]), cache=merged)
    extracted = [plane.extract(1) for plane in merged]
    assert inspect_apc_capabilities(extracted).exact_prefix is (mode == "per_loop")
    if mode == "last_loop":
        assert type(extracted[2]) is academia.LastLoopKVCache


def test_topology_and_cache_planes():
    per = academia.Model(academia.ModelArgs.from_dict(config(n_recurrent_loops=3)))
    last = academia.Model(academia.ModelArgs.from_dict(
        config(n_recurrent_loops=3, recurrent_kv_cache_mode="last_loop")))
    assert per.physical == [0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4]
    assert len(per.layers) == 5
    # per_loop: GDP 0 + (GDP, attn) x 3 loops; MLPs hold no cache.
    assert per.cache_slots == [0, 1, 2, None, 3, 4, None, 5, 6, None, None]
    assert last.cache_slots == [0, 1, 2, None, 3, 2, None, 4, 2, None, None]
    caches = last.make_cache()
    assert len(caches) == 5 and type(caches[2]) is academia.LastLoopKVCache
    assert [type(c).__name__ for c in per.make_cache()].count("KVCache") == 3
    receipt = last.layer_receipt()
    assert receipt["virtual_layers"] == 11 and receipt["gdp_state"] == [4, 32, 64]


def test_rectangular_gdp_state_shape():
    cfg = config()
    model = mlx_model(cfg, hf_weights(cfg))
    cache = model.make_cache()
    model(mx.array([tokens(5)]), cache=cache)
    conv, state = cache[0][0], cache[0][1]
    assert state.shape == (1, 4, 32, 64)  # [B, Hv, Dv, Dk]
    assert conv.shape == (1, 3, 3 * 4 * 32 + 4 * 2 * 64)


@pytest.mark.parametrize(
    "overrides",
    [
        {"hidden_act": "silu"},
        {"recurrent_kv_cache_mode": "shared"},
        {"layer_types": ["linear_attention", "full_attention"]},
        {"loop_conditioning_hidden_size": 15},
        {"linear_num_value_heads": 3},
        {"linear_num_householder": 17},
        {"n_recurrent_loops": 0},
    ],
)
def test_invalid_config_fails_closed(overrides):
    with pytest.raises(ValueError):
        academia.ModelArgs.from_dict(config(**overrides))


def test_last_loop_refuses_unsupported_plane():
    from mlx2.runtime.models.cache import QuantizedKVCache

    with pytest.raises(NotImplementedError):
        academia.rewrite_last_rows(QuantizedKVCache(), mx.zeros((1, 1, 2, 4)), mx.zeros((1, 1, 2, 4)))
