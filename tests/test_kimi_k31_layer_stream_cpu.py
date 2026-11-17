"""Kimi-K3.1 layer stream on tiny-random/kimi-k3.1-pro (CPU; unqualified).

Checkpoint: tiny-random/kimi-k3.1-pro@507bf23a377e31775c3a79b63b2c42bd6e4b9cc0,
model.safetensors sha256 ac374bbdaa96d2f4... (random weights, real K3.1 schema,
real compressed-tensors MXFP4 experts). Set MLX2_KIMI_K31_TINY to its folder;
the default is the ~/.cache/mlx2 copy. Tests skip when it is absent.
"""
import hashlib
import json
import os
import re
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters import kimi_k31 as A
from mlx2.runtime.models import kimi_k31 as k31

TINY = Path(os.environ.get("MLX2_KIMI_K31_TINY", Path.home() / ".cache/mlx2/tiny-random/"
                           "kimi-k3.1-pro-507bf23a377e31775c3a79b63b2c42bd6e4b9cc0"))
TINY_SHA256 = "ac374bbdaa96d2f421c37acfeaeefb029ae9acee6f9b2a3bbbc0ebddbc8c428f"
TOKENS = [163584, 100, 2000, 30000, 7, 999, 12345, 5, 4242, 88, 161, 70000]
REAL = Path(os.environ.get("MLX2_KIMI_K31_REAL", Path.home() / ".cache/mlx2/kimi-k3.1-pro"))
T7_Q8 = REAL / "kimi_q8.py"

pytestmark = pytest.mark.skipif(not (TINY / "model.safetensors").exists(),
                                reason="tiny-random/kimi-k3.1-pro not present")


@pytest.fixture(scope="module", autouse=True)
def cpu():
    """CPU by default; MLX2_K31_TEST_DEVICE=gpu reruns the module on the GPU path (Metal KDA kernel)."""
    old = mx.default_device()
    mx.set_default_device(mx.gpu if os.environ.get("MLX2_K31_TEST_DEVICE") == "gpu" else mx.cpu)
    yield
    mx.set_default_device(old)


@pytest.fixture(scope="module")
def tiny():
    digest = hashlib.sha256((TINY / "model.safetensors").read_bytes()).hexdigest()
    assert digest == TINY_SHA256, "unexpected tiny checkpoint"
    return mx.load(str(TINY / "model.safetensors")), json.loads((TINY / "config.json").read_text())


@pytest.fixture(scope="module")
def layout(tmp_path_factory):
    return A.write_stream_layout(TINY, tmp_path_factory.mktemp("k31"))


def _walk(stream, tokens, choose=None, full_route=False):
    st = stream.begin(tokens)
    routes = []
    for i in range(stream.cfg.num_hidden_layers):
        r = stream.route(i, st)
        routes.append(r)
        experts = None
        if r is not None:
            ids = choose(i, r) if choose else stream.available_experts(i)
            experts = stream.load_experts(i, ids)
        stream.forward_layer(i, st, experts, full_route=full_route)
    return stream.finish(st, last_only=False), routes


def test_mxfp4_dequant_is_bit_exact(tiny):
    tensors, _ = tiny
    names = [k for k in tensors if k.endswith(".weight_packed")]
    assert len(names) == 16 * 3 * 16  # 16 MoE layers x 16 experts x w1/w2/w3
    for name in names:
        packed, scale = tensors[name], tensors[name.replace("_packed", "_scale")]
        ref = k31.mxfp4_dequantize_reference(np.array(packed), np.array(scale))
        q = mx.contiguous(packed).view(mx.uint32)
        dq = mx.dequantize(q, scale, None, group_size=32, bits=4, mode="mxfp4").astype(mx.float32)
        assert np.array_equal(np.array(dq), ref.astype(np.float32)), name
        # The serving product is the MLX mxfp4 quantized matmul on those bytes.
        x = mx.random.normal((3, ref.shape[1]), key=mx.random.key(0)).astype(mx.bfloat16)
        got = mx.quantized_matmul(x, q, scale, None, transpose=True, group_size=32, bits=4,
                                  mode="mxfp4").astype(mx.float32)
        want = x.astype(mx.float32) @ mx.array(ref).T
        assert mx.allclose(got, want, rtol=2e-2, atol=2e-2 * float(mx.abs(want).max()) + 1e-6)


def test_mxfp4_real_checkpoint_expert_if_fetched():
    files = sorted((REAL / "experts").glob("layer_*/expert_*.safetensors"))
    files = [f for f in files if not f.name.startswith("._")]
    if not files:
        pytest.skip("no real Kimi-K3.1-Pro expert fetched yet")
    tensors = mx.load(str(files[0]))
    for name in [k for k in tensors if k.endswith(".weight_packed")]:
        packed, scale = tensors[name], tensors[name.replace("_packed", "_scale")]
        ref = k31.mxfp4_dequantize_reference(np.array(packed), np.array(scale))
        dq = mx.dequantize(mx.contiguous(packed).view(mx.uint32), scale, None, group_size=32,
                           bits=4, mode="mxfp4").astype(mx.float32)
        assert np.array_equal(np.array(dq), ref), name


def test_layer_stream_equals_whole_model_bf16(tiny, layout):
    tensors, config = tiny
    whole = A.whole_model_logits(tensors, config, TOKENS, last_only=False)
    streamed, routes = _walk(A.KimiK31LayerStream(layout, skeleton="bf16"), TOKENS, full_route=True)
    assert routes[0] is None and all(r is not None for r in routes[1:])
    assert mx.array_equal(streamed, whole)


def test_layer_stream_equals_whole_model_q8(tiny, layout):
    tensors, config = tiny
    whole_q8 = A.whole_model_logits(A.quantize_q8(tensors), config, TOKENS, last_only=False)
    streamed, _ = _walk(A.KimiK31LayerStream(layout, skeleton="q8"), TOKENS, full_route=True)
    assert mx.array_equal(streamed, whole_q8)
    # The tiny rule quantizes only last-dim-64 matrices (o_proj, kv_a, shared
    # down, routed up, ...); q8 must stay close to bf16.
    whole = A.whole_model_logits(tensors, config, TOKENS, last_only=False)
    a, b = streamed.reshape(-1), whole.reshape(-1)
    cosine = float((a * b).sum() / (mx.linalg.norm(a) * mx.linalg.norm(b)))
    assert cosine > 0.99, cosine


def test_q8_layout_follows_rule(layout):
    w = mx.load(str(layout / "skeleton_q8" / "layer_03.safetensors"))
    p = "language_model.model.layers.3."
    assert w[p + "self_attn.o_proj.weight"].dtype == mx.uint32         # [8, 12288]: in % 64 == 0
    assert p + "self_attn.o_proj.biases" in w
    assert w[p + "self_attn.q_a_proj.weight"].dtype == mx.bfloat16     # in = hidden 8: kept bf16
    for kept in ("block_sparse_moe.gate.weight", "self_attn.kv_b_proj.weight",
                 "self_attn.indexer.wq_b.weight", "self_attention_res_proj.weight"):
        assert w[p + kept].dtype == mx.bfloat16 and p + kept.replace(".weight", ".scales") not in w
    if T7_Q8.exists():  # read as text only; never executed
        def body(src, name):
            return re.sub(r"\s+", "", re.search(name + r" = re\.compile\(\n(.*?)\n\)", src, re.S).group(1))
        assert body(T7_Q8.read_text(), "QUANT") == body(Path(A.__file__).read_text(), "Q8_RULE")


def test_restricted_with_all_experts_equals_full(layout):
    s_full = A.KimiK31LayerStream(layout, skeleton="bf16")
    full, _ = _walk(s_full, TOKENS, full_route=True)
    s_res = A.KimiK31LayerStream(layout, skeleton="bf16")
    res, _ = _walk(s_res, TOKENS, full_route=False)
    assert all(abs(r["kept_min"] - 1.0) < 1e-6 for r in s_res.reports if r["moe"])
    diff = float(mx.abs(res - full).max())
    print("restricted-vs-full max abs diff", diff)
    assert diff <= 1e-3 * float(mx.abs(full).max()), diff


def test_restricted_subset_kept_weight_and_renormalisation(layout):
    s = A.KimiK31LayerStream(layout, skeleton="bf16")
    st = s.begin(TOKENS)
    s.forward_layer(0, st)
    r = s.route(1, st)
    agg = r["aggregate"]
    assert abs(sum(agg.values()) - len(TOKENS)) < 1e-4       # renormalised top-k per token
    present = sorted(agg, key=lambda e: -agg[e])[:4]
    rep = s.forward_layer(1, st, s.load_experts(1, present))
    want = np.where(np.isin(r["ids"], present), r["weights"], 0).sum(-1)
    assert np.allclose(rep["kept_per_token"], want, atol=1e-6)
    # Kept slots are renormalised over the present experts.
    routing = k31.Routing(mx.array(r["ids"]), mx.array(r["weights"]), mx.array(r["scores"]))
    slot_w, kept = k31.restrict(routing, present, s.cfg)
    sums = np.array(slot_w).sum(-1)
    assert np.allclose(sums[np.array(kept) > 0], 1.0, atol=1e-5)
    assert np.allclose(np.array(slot_w)[~np.isin(r["ids"], present)], 0)


def test_token_without_present_expert_gets_zero_routed_output(layout):
    s = A.KimiK31LayerStream(layout, skeleton="bf16")
    st = s.begin(TOKENS[:3])
    s.forward_layer(0, st)
    s.route(1, st)
    rep = s.forward_layer(1, st, {})
    assert rep["tokens_without_experts"] == 3 and rep["kept_mean"] == 0.0


def test_full_route_fails_closed_when_an_expert_is_missing(layout):
    s = A.KimiK31LayerStream(layout, skeleton="bf16")
    st = s.begin(TOKENS)
    s.forward_layer(0, st)
    r = s.route(1, st)
    with pytest.raises(KeyError, match="missing"):
        s.forward_layer(1, st, s.load_experts(1, [int(r["ids"][0, 0])]), full_route=True)


def test_dsa_sparse_mask_matches_bruteforce(layout):
    # index_topk 3 makes the K3.1 indexer active on the tiny prompt.
    s = A.KimiK31LayerStream(layout, skeleton="bf16", config_overrides={"index_topk": 3})
    w, quant = s.load_layer(3)
    t = 9
    x = mx.random.normal((t, s.cfg.hidden_size), key=mx.random.key(3)).astype(mx.bfloat16)
    qr = mx.random.normal((t, s.cfg.q_lora_rank), key=mx.random.key(4)).astype(mx.bfloat16)
    index = k31.dsa_index(w, x, qr, s.cfg, quant)
    keep = np.array(k31.dsa_keep(index, 0, t, 3))
    keep_chunked = np.concatenate([np.pad(np.array(k31.dsa_keep(index, r0, min(t, r0 + 2), 3)),
                                          ((0, 0), (0, t - min(t, r0 + 2))))
                                   for r0 in range(0, t, 2)])
    assert np.array_equal(keep, keep_chunked)
    q, k, iw = (np.array(index[n]) for n in ("q", "k", "w"))
    score = (np.maximum(np.einsum("thd,sd->ths", q, k), 0) * iw[..., None]).sum(1)
    for row in range(t):
        want = set(np.argsort(-score[row, :row + 1], kind="stable")[:3].tolist())
        assert set(np.nonzero(keep[row])[0].tolist()) == want
    # A prompt of index_topk tokens or fewer runs exactly dense.
    assert k31.dsa_index(w, x[:3], qr[:3], s.cfg, quant) is None
    out_sparse, sparse = k31.mla_attention(w, x, s.cfg, quant, row_chunk=4)
    assert sparse and out_sparse.shape == (t, s.cfg.hidden_size)


def test_mla_row_chunking_is_exact(layout):
    s = A.KimiK31LayerStream(layout, skeleton="bf16")
    w, quant = s.load_layer(3)
    x = mx.random.normal((10, s.cfg.hidden_size), key=mx.random.key(5)).astype(mx.bfloat16)
    one, _ = k31.mla_attention(w, x, s.cfg, quant, row_chunk=10)
    many, _ = k31.mla_attention(w, x, s.cfg, quant, row_chunk=3)
    assert mx.allclose(one.astype(mx.float32), many.astype(mx.float32), atol=1e-2)


def test_defaults_receipt_and_refusals(layout, tmp_path):
    s = A.KimiK31LayerStream(layout, skeleton="q8")
    assert A.KimiK31LayerStream.default_max_context == 32768
    assert A.KimiK31LayerStream.default_max_lanes == 1
    receipt = s.route_receipt()
    assert receipt["qualification"] == "unqualified" and receipt["qualified"] is False
    assert receipt["max_context"] == 32768 and receipt["lanes"] == 1
    assert receipt["kv_plan"]["context"] == 32768
    with pytest.raises(ValueError, match="max_context"):
        A.KimiK31LayerStream(layout, max_context=4).begin(TOKENS)
    from mlx2.adapters.registry import inspect_model
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "kimi_k31"}))
    with pytest.raises(ValueError, match="no mlx2 serving route"):
        inspect_model(tmp_path)


def test_real_config_parses_with_one_based_layer_lists():
    cfg_path = REAL / "config.json"
    if not cfg_path.exists():
        pytest.skip("real config not mounted")
    cfg = k31.K31Config.from_dict(json.loads(cfg_path.read_text()))
    assert len(cfg.kda_layers) == 69 and len(cfg.full_attn_layers) == 24
    assert 0 in cfg.kda_layers and 3 in cfg.full_attn_layers and 92 in cfg.full_attn_layers
    assert cfg.dsa_layers == cfg.full_attn_layers
    plan = A.kv_plan(cfg)
    assert plan["mla_latent_bytes"] == 24 * 576 * 2 * 32768
    assert plan["dsa_index_key_bytes"] == 24 * 128 * 2 * 32768


def test_fail_closed_paths(layout):
    s = A.KimiK31LayerStream(layout, skeleton="bf16")
    st = s.begin(TOKENS)
    s.forward_layer(0, st)
    with pytest.raises(ValueError, match="is MoE"):
        s.forward_layer(1, st, None)
    w, quant = s.load_layer(3)
    no_index = {k: v for k, v in w.items() if ".indexer." not in k}
    with pytest.raises(KeyError, match="indexer"):
        k31.mla_attention(no_index, mx.zeros((2, s.cfg.hidden_size), mx.bfloat16), s.cfg, quant)
    with pytest.raises(ValueError, match="without scales"):
        k31.linear({"x.weight": mx.zeros((8, 4), mx.uint32)}, "x", mx.zeros((1, 32)))
    cfg = json.loads((layout / "config.json").read_text())
    cfg["text_config"]["dsa_config"]["dsa_layers"] = [4]
    with pytest.raises(ValueError, match="dsa_layers"):
        k31.K31Config.from_dict(cfg)


def test_attn_res_mix_row_chunks_are_exact():
    keys = mx.random.split(mx.random.key(7), 6)
    blocks = [mx.random.normal((5000, 64), key=keys[i]).astype(mx.bfloat16) for i in range(3)]
    partial = mx.random.normal((5000, 64), key=keys[3]).astype(mx.bfloat16)
    nw, pw = mx.random.normal((64,), key=keys[4]), mx.random.normal((1, 64), key=keys[5])
    whole = k31.attn_res_mix(blocks, partial, nw, pw, 1e-5)
    pieces = mx.concatenate([k31.attn_res_mix([b[r:r + 700] for b in blocks], partial[r:r + 700],
                                              nw, pw, 1e-5) for r in range(0, 5000, 700)])
    assert mx.array_equal(whole, pieces)


def test_restricted_subset_output_matches_dense_reference(tiny, layout):
    tensors, _ = tiny
    s = A.KimiK31LayerStream(layout, skeleton="bf16")
    st = s.begin(TOKENS)
    s.forward_layer(0, st)
    r = s.route(1, st)
    present = sorted(r["aggregate"], key=lambda e: -r["aggregate"][e])[:5]
    w, quant = s.load_layer(1)
    xl = k31.linear(w, "block_sparse_moe.routed_expert_down_proj", st.pending["ffn_in"], quant)
    got = k31.routed_restricted(s.load_experts(1, present), xl, st.pending["routing"], s.cfg)[0]
    pre = "language_model.model.layers.1.block_sparse_moe.experts."
    dense = {e: {n: k31.mxfp4_dequantize_reference(np.array(tensors[f"{pre}{e}.{n}.weight_packed"]),
                                                  np.array(tensors[f"{pre}{e}.{n}.weight_scale"]))
                 for n in ("w1", "w2", "w3")} for e in present}
    xl_np = np.array(xl.astype(mx.float32))
    want = np.zeros_like(xl_np)
    b, lb = s.cfg.situ_beta, s.cfg.situ_linear_beta
    for t in range(len(TOKENS)):
        kept = [(e, sc) for e, sc in zip(r["ids"][t], r["scores"][t]) if e in present]
        total = sum(sc for _, sc in kept)
        for e, sc in kept:
            gt, up = xl_np[t] @ dense[e]["w1"].T, xl_np[t] @ dense[e]["w3"].T
            a = b * np.tanh(gt / b) / (1 + np.exp(-gt)) * lb * np.tanh(up / lb)
            want[t] += sc / total * (a @ dense[e]["w2"].T)
    got = np.array(got.astype(mx.float32))
    assert np.abs(got - want).max() <= 2e-2 * np.abs(want).max()


def test_dsa_sparse_attention_output_matches_numpy(layout):
    # Independent numpy DSA (adapted from the fresh-context reviewer's probe).
    s = A.KimiK31LayerStream(layout, skeleton="bf16", config_overrides={"index_topk": 3})
    w, quant = s.load_layer(3)
    cfg, t = s.cfg, 9
    x = mx.random.normal((t, cfg.hidden_size), key=mx.random.key(3)).astype(mx.float32)
    out, sparse = k31.mla_attention(w, x, cfg, quant, row_chunk=4)
    dense, was_sparse = k31.mla_attention(w, x, cfg, quant, dsa=False)
    assert sparse and not was_sparse

    def f(n):
        return np.array(w[n].astype(mx.float32))

    def rms(v, g, e):
        return g * v / np.sqrt((v * v).mean(-1, keepdims=True) + e)

    H, dn, dr, dv, r = (cfg.num_attention_heads, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim,
                        cfg.v_head_dim, cfg.kv_lora_rank)
    X = np.array(x)
    qr = rms(X @ f("self_attn.q_a_proj.weight").T, f("self_attn.q_a_layernorm.weight"), 1e-6)
    Q = (qr @ f("self_attn.q_b_proj.weight").T).reshape(t, H, dn + dr)
    ckv = X @ f("self_attn.kv_a_proj_with_mqa.weight").T
    kv = (rms(ckv[:, :r], f("self_attn.kv_a_layernorm.weight"), 1e-6)
          @ f("self_attn.kv_b_proj.weight").T).reshape(t, H, dn + dv)
    K = np.concatenate([kv[..., :dn], np.broadcast_to(ckv[:, None, r:], (t, H, dr))], -1)
    iq = (qr @ f("self_attn.indexer.wq_b.weight").T).reshape(t, cfg.index_heads, cfg.index_head_dim)
    ik = X @ f("self_attn.indexer.wk.weight").T
    mu, var = ik.mean(-1, keepdims=True), ik.var(-1, keepdims=True)
    ik = ((ik - mu) / np.sqrt(var + cfg.rms_norm_eps) * f("self_attn.indexer.k_norm.weight")
          + f("self_attn.indexer.k_norm.bias"))
    iw = (X @ f("self_attn.indexer.weights_proj.weight").T) / np.sqrt(cfg.index_heads * cfg.index_head_dim)
    sc = (np.maximum(np.einsum("thd,sd->ths", iq, ik), 0) * iw[..., None]).sum(1)
    mask = np.zeros((t, t), bool)
    for i in range(t):
        mask[i, np.argsort(-sc[i, :i + 1], kind="stable")[:3]] = True
    S = np.where(mask[None], np.einsum("thd,shd->hts", Q, K) / np.sqrt(dn + dr), -np.inf)
    P = np.exp(S - S.max(-1, keepdims=True))
    P /= P.sum(-1, keepdims=True)
    O = np.einsum("hts,shd->thd", P, kv[..., dn:]).reshape(t, H * dv)
    O = O / (1 + np.exp(-(X @ f("self_attn.g_proj.weight").T))) @ f("self_attn.o_proj.weight").T
    scale = np.abs(O).max()
    assert np.abs(np.array(out) - O).max() / scale < 1e-4
    assert np.abs(np.array(dense) - np.array(out)).max() / scale > 1e-2   # the mask matters


def test_moe_token_chunks_are_exact(layout):
    s = A.KimiK31LayerStream(layout, skeleton="bf16")
    st = s.begin(TOKENS)
    s.forward_layer(0, st)
    s.route(1, st)
    w, quant = s.load_layer(1)
    xl = k31.linear(w, "block_sparse_moe.routed_expert_down_proj", st.pending["ffn_in"], quant)
    routing = st.pending["routing"]
    experts = s.load_experts(1, s.available_experts(1))
    one = k31.routed_experts(experts, xl, routing, routing.weights, s.cfg)
    many = k31.routed_experts(experts, xl, routing, routing.weights, s.cfg, rows=5)
    assert mx.array_equal(one, many)


def test_expert_q2_tier_is_affine_and_approximate():
    """Derived affine-q2 expert tier: affine layout and an approximate (not garbage) output."""
    mx.random.seed(7)

    def mxfp4(shape):
        w = (mx.random.normal(shape) * 0.05).astype(mx.bfloat16)
        q, s = mx.quantize(w, group_size=32, bits=4, mode="mxfp4")
        return q, s

    e = k31.Expert(mxfp4((128, 256)), mxfp4((256, 128)), mxfp4((128, 256)))
    q = k31.expert_to_q2(e)
    for wq in (q.w1, q.w2, q.w3):
        assert len(wq) == 3 and wq[0].dtype == mx.uint32 and wq[1].dtype == mx.bfloat16
    x = mx.random.normal((5, 256)).astype(mx.bfloat16)
    a = k31.expert_matmul(x, e.w1).astype(mx.float32)
    b = k31.expert_matmul(x, q.w1).astype(mx.float32)
    cos = float(mx.sum(a * b) / (mx.linalg.norm(a.reshape(-1)) * mx.linalg.norm(b.reshape(-1))))
    assert 0.7 < cos < 0.999, cos   # approximate by construction, not garbage


def _walk_state(stream, st, full_route=True):
    for i in range(stream.cfg.num_hidden_layers):
        r = stream.route(i, st)
        experts = None if r is None else stream.load_experts(i, stream.available_experts(i))
        stream.forward_layer(i, st, experts, full_route=full_route)
    return stream.finish(st, last_only=False)


@pytest.mark.parametrize("overrides", [{}, {"index_topk": 3}], ids=["dense", "dsa_crosses_topk"])
@pytest.mark.parametrize("skeleton", ["bf16", "q8"])
def test_cached_decode_equals_uncached_forward(layout, overrides, skeleton):
    """Prompt + one-token cached steps == one uncached forward over the whole sequence.

    Covers KDA conv tails + recurrent state, MLA latent/k_pe cache, and (with
    index_topk 3) the DSA indexer key cache across the sparse threshold."""
    prompt, gen = TOKENS[:5], TOKENS[5:]
    ref_stream = A.KimiK31LayerStream(layout, skeleton=skeleton, config_overrides=overrides)
    ref = _walk_state(ref_stream, ref_stream.begin(TOKENS))            # [T, V]

    s = A.KimiK31LayerStream(layout, skeleton=skeleton, config_overrides=overrides)
    cache: dict = {}
    got = [_walk_state(s, s.begin(prompt, cache=cache))]
    pos = len(prompt)
    for t in gen:
        got.append(_walk_state(s, s.begin([t], cache=cache, offset=pos)))
        pos += 1
    got = mx.concatenate(got)
    assert got.shape == ref.shape
    scale = float(mx.abs(ref).max())
    diff = float(mx.abs(got - ref).max())
    print(skeleton, overrides, "cached-vs-uncached max abs diff", diff, "scale", scale)
    assert diff <= 2e-2 * scale, diff   # bf16 activations: a few ulps from chunked vs whole sums
    # Cache holds 576-wide latent rows for every MLA layer and KDA state for every KDA layer.
    for i, c in cache.items():
        if i in s.cfg.kda_layers:
            assert c["state"].shape[-1] == s.cfg.kda_head_dim and c["q_tail"].shape[0] == 3
        else:
            assert c["latent"].shape[0] == len(TOKENS)


def test_decode_refuses_offset_without_cache(layout):
    s = A.KimiK31LayerStream(layout, skeleton="bf16")
    with pytest.raises(ValueError):
        s.begin([5], offset=3)


@pytest.mark.parametrize("overrides", [{}, {"index_topk": 3}], ids=["dense", "dsa_crosses_topk"])
def test_cached_decode_equals_uncached_forward_fp32_strict(layout, overrides):
    """Same equivalence with FP32 activations: any cache-logic error shows up far above 1e-5."""
    prompt, gen = TOKENS[:5], TOKENS[5:]
    r = A.KimiK31LayerStream(layout, skeleton="bf16", dtype=mx.float32, config_overrides=overrides)
    ref = _walk_state(r, r.begin(TOKENS))
    s = A.KimiK31LayerStream(layout, skeleton="bf16", dtype=mx.float32, config_overrides=overrides)
    cache: dict = {}
    got = [_walk_state(s, s.begin(prompt, cache=cache))]
    for j, t in enumerate(gen):
        got.append(_walk_state(s, s.begin([t], cache=cache, offset=len(prompt) + j)))
    got = mx.concatenate(got)
    rel = float(mx.abs(got - ref).max()) / float(mx.abs(ref).max())
    print(overrides, "fp32 cached-vs-uncached rel diff", rel)
    assert rel < 1e-4, rel
    assert np.array_equal(np.array(got.argmax(-1)), np.array(ref.argmax(-1)))


def test_expert_lru_cache_is_transparent_and_evicts(layout, tmp_path):
    """NVMe read-through cache: identical output, LRU stays under budget, canonical files untouched."""
    ref_s = A.KimiK31LayerStream(layout, skeleton="bf16")
    ref = _walk_state(ref_s, ref_s.begin(TOKENS))
    canon = sorted((layout / "experts").glob("layer_*/expert_*.safetensors"))
    one = canon[0].stat().st_size
    s = A.KimiK31LayerStream(layout, skeleton="bf16")
    s.enable_expert_cache(tmp_path / "cache", max_bytes=3 * one)
    got = _walk_state(s, s.begin(TOKENS))
    assert float(mx.abs(got - ref).max()) == 0.0
    cached = list((tmp_path / "cache").glob("layer_*/expert_*.safetensors"))
    assert sum(p.stat().st_size for p in cached) <= 3 * one
    assert s.cache_stats["copies"] > 3 and s.cache_stats["evictions"] > 0
    assert sorted((layout / "experts").glob("layer_*/expert_*.safetensors")) == canon
    # A copy deleted underneath (sibling eviction) falls back to the canonical file.
    for p in cached:
        p.unlink()
    again = _walk_state(s, s.begin(TOKENS))
    assert float(mx.abs(again - ref).max()) == 0.0
