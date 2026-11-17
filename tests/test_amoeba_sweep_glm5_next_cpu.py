"""aMoEba sweep, glm5_next family (Sage-S2.1-Flash) on a tiny random checkpoint; CPU only.

The checkpoint keeps every geometry the GLM-5.3 forward fixes (64x128 KDA, 64x256 MLA, 32x128
indexer, HC4) and shrinks the rest (hidden 64, 5 layers, 12 experts). It is written in Sage's
on-disk naming (``model.language_model.*``, per-expert tensors, split conv1d, vision and MTP
tensors present but never read). transformers' ``Glm5NextTextModel`` is the independent oracle
for the name map and transforms.
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from research.amoeba.sweep import capture as C  # noqa: E402
from research.amoeba.sweep.glm5_next import Glm5NextFamily, text_config  # noqa: E402

D, L, E, VOCAB = 64, 5, 12, 97
KINDS = ["linear_attention"] * 3 + ["deepseek_sparse_attention", "linear_attention"]


def tiny_text_config():
    return {
        "model_type": "glm5_next_text", "hidden_size": D, "num_hidden_layers": L, "vocab_size": VOCAB,
        "intermediate_size": 48, "moe_intermediate_size": 32, "n_routed_experts": E, "n_shared_experts": 1,
        "num_experts_per_tok": 8, "n_group": 1, "topk_group": 1, "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid", "norm_topk_prob": True, "topk_method": "noaux_tc", "swiglu_limit": 10.0,
        "hidden_act": "silu", "first_k_dense_replace": 3, "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 2,
        "num_attention_heads": 64, "num_key_value_heads": 64, "head_dim": 0, "q_lora_rank": 32,
        "kv_lora_rank": 512, "qk_head_dim": 256, "qk_nope_head_dim": 256, "qk_rope_head_dim": 0,
        "v_head_dim": 256, "mla_use_nope": True, "attention_bias": False, "attention_dropout": 0.0,
        "index_n_heads": 32, "index_head_dim": 128, "index_topk": 2048, "index_kpool": 4,
        "index_kpool_compress": True, "index_kpool_always_select_tail": True, "indexer_types": ["full"] * L,
        "indexer_rope_interleave": True, "index_share_for_mtp_iteration": True,
        "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6, "mhc": True,
        "layer_types": KINDS,
        "linear_attn_config": {"num_heads": 64, "head_dim": 128, "short_conv_kernel_size": 4,
                               "gate_lower_bound": -5.0, "kda_layers": [0, 1, 2, 4], "full_attn_layers": [3]},
        "rms_norm_eps": 1e-5, "max_position_embeddings": 4096, "num_nextn_predict_layers": 1,
        "tie_word_embeddings": False, "moe_router_dtype": "float32", "dtype": "bfloat16", "pad_token_id": 0,
    }


def random_weights(seed=7):
    """Sage-named tensors (float32 numpy)."""
    rng = np.random.default_rng(seed)
    t = {}
    P = "model.language_model."

    def mat(name, *shape, std=None):
        std = std if std is not None else 1.0 / np.sqrt(shape[-1])
        t[name] = rng.normal(0, std, shape).astype(np.float32)

    def norm(name, n):
        t[name] = (1 + 0.1 * rng.normal(size=n)).astype(np.float32)

    mat(P + "embed_tokens.weight", VOCAB, D, std=1.0)
    mat("lm_head.weight", VOCAB, D)
    norm(P + "norm.weight", D)
    for l, kind in enumerate(KINDS):
        p = f"{P}layers.{l}."
        for half in ("attn", "ffn"):
            mat(p + f"hc_{half}_fn", 24, 4 * D, std=0.05)
            t[p + f"hc_{half}_base"] = rng.normal(0, 0.5, 24).astype(np.float32)
            t[p + f"hc_{half}_scale"] = rng.uniform(0.5, 1.5, 3).astype(np.float32)
        norm(p + "input_layernorm.weight", D)
        norm(p + "post_attention_layernorm.weight", D)
        a = p + "self_attn."
        if kind == "linear_attention":
            for x in "qkv":
                mat(a + f"{x}_proj.weight", 8192, D)
                t[a + f"{x}_conv1d.weight"] = rng.normal(0, 0.3, (8192, 1, 4)).astype(np.float32)
            t[a + "A_log"] = np.log(rng.uniform(1, 8, 64)).astype(np.float32)
            t[a + "dt_bias"] = rng.normal(-1, 0.3, 8192).astype(np.float32)
            mat(a + "b_proj.weight", 64, D)
            mat(a + "f_a_proj.weight", 128, D)
            mat(a + "f_b_proj.weight", 8192, 128)
            mat(a + "g_a_proj.weight", 128, D)
            mat(a + "g_b_proj.weight", 8192, 128)
            norm(a + "o_norm.weight", 128)
            mat(a + "o_proj.weight", D, 8192)
        else:
            mat(a + "q_a_proj.weight", 32, D)
            norm(a + "q_a_layernorm.weight", 32)
            mat(a + "q_b_proj.weight", 64 * 256, 32)
            mat(a + "kv_a_proj_with_mqa.weight", 512, D)
            norm(a + "kv_a_layernorm.weight", 512)
            mat(a + "kv_b_proj.weight", 64 * 512, 512)
            mat(a + "o_proj.weight", D, 64 * 256)
            i = a + "indexer."
            mat(i + "wq_b.weight", 32 * 128, 32)
            mat(i + "wk.weight", 128, D)
            norm(i + "k_norm.weight", 128)
            t[i + "k_norm.bias"] = rng.normal(0, 0.1, 128).astype(np.float32)
            mat(i + "weights_proj.weight", 32, D)
            t[i + "index_kpool_compress_ape"] = rng.normal(0, 0.1, (4, 128)).astype(np.float32)
            mat(i + "index_kpool_compress_gate", 128, D)
        m = p + "mlp."
        if l < 3:
            mat(m + "gate_proj.weight", 48, D)
            mat(m + "up_proj.weight", 48, D)
            mat(m + "down_proj.weight", D, 48)
        else:
            mat(m + "gate.weight", E, D)
            t[m + "gate.e_score_correction_bias"] = rng.normal(0, 0.1, E).astype(np.float32)
            for e in range(E):
                mat(m + f"experts.{e}.gate_proj.weight", 32, D)
                mat(m + f"experts.{e}.up_proj.weight", 32, D)
                mat(m + f"experts.{e}.down_proj.weight", D, 32)
            mat(m + "shared_experts.gate_proj.weight", 32, D)
            mat(m + "shared_experts.up_proj.weight", 32, D)
            mat(m + "shared_experts.down_proj.weight", D, 32)
    # never-read tensors: MTP layer and vision tower
    mat(f"{P}layers.{L}.eh_proj.weight", D, 2 * D)
    mat("model.visual.patch_embed.proj.weight", 8, 8)
    return t


def write_checkpoint(path: Path, tensors, *, dtype="F32", config_overrides=None):
    from safetensors.numpy import save_file
    path.mkdir(parents=True, exist_ok=True)
    tc = tiny_text_config()
    tc.update(config_overrides or {})
    (path / "config.json").write_text(json.dumps({"model_type": "glm5_next", "text_config": tc,
                                                  "vision_config": {"hidden_size": 8}}))
    names = sorted(tensors)
    shards = [names[i::3] for i in range(3)]           # interleaved: a layer spans every shard
    wmap = {}
    for j, group in enumerate(shards):
        fn = f"model-{j + 1:05d}-of-00003.safetensors"
        out = {}
        for n in group:
            v = tensors[n]
            if dtype == "BF16" and not stored_f32(n):
                bits = (v.astype(np.float32).view(np.uint32) + 0x7FFF + ((v.view(np.uint32) >> 16) & 1)) >> 16
                out[n] = bits.astype(np.uint16)
            else:
                out[n] = v
            wmap[n] = fn
        save_file(out, str(path / fn))
        if dtype == "BF16":
            _retag_bf16(path / fn, [n for n in group if out[n].dtype == np.uint16])
    (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wmap}))
    return path


def _retag_bf16(file, names):
    """Rewrite the header dtype of uint16-stored tensors to BF16 (same bytes)."""
    raw = file.read_bytes()
    n = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + n])
    for k in names:
        header[k]["dtype"] = "BF16"
    blob = json.dumps(header).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    file.write_bytes(len(blob).to_bytes(8, "little") + blob + raw[8 + n:])


def stored_f32(name):
    """Tensors Sage keeps in F32 (the rest is BF16)."""
    return name.endswith(("A_log", "dt_bias", "conv1d.weight", "_base", "_scale", "e_score_correction_bias"))


def bf16_round(tensors):
    out = {}
    for n, v in tensors.items():
        if stored_f32(n):
            out[n] = v
            continue
        u = v.astype(np.float32).view(np.uint32)
        bits = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint32) << 16
        out[n] = bits.view(np.float32)
    return out


@pytest.fixture(scope="module")
def weights():
    return random_weights()


@pytest.fixture(scope="module")
def ckpt_dir(tmp_path_factory, weights):
    return write_checkpoint(tmp_path_factory.mktemp("sage") / "Sage-tiny", weights)


@pytest.fixture(scope="module")
def ckpt_bf16(tmp_path_factory, weights):
    return write_checkpoint(tmp_path_factory.mktemp("sagebf") / "Sage-tiny-bf16", weights, dtype="BF16")


def family(path, **kw):
    kw.setdefault("dtype", "float32")
    kw.setdefault("experts", "routed")
    return Glm5NextFamily(path, fingerprint="feedface" * 8, **kw)


def seqs(n=6):
    rng = np.random.default_rng(3)
    return [[int(t) for t in rng.integers(1, VOCAB, size=int(rng.choice([9, 10, 12])))] for _ in range(n)]


def run(fam, ss, tmp, **kw):
    out, st = tmp / "out", tmp / "state"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "mu.npy", np.zeros((fam.L, max(fam.widths)), np.float32))
    kw.setdefault("parity", True)
    final = C.sweep(fam, ss, ["fit" if i % 2 else "eval" for i in range(len(ss))], out, st,
                    log=lambda m: None, **kw)
    return out, final


# --------------------------------------------------------------------------------- the oracle
def hf_model(tensors):
    """transformers' Glm5NextTextModel loaded from the Sage-named tensors via its own conversion."""
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    cfg = Glm5NextTextConfig(**{k: v for k, v in tiny_text_config().items() if k != "model_type"})
    cfg._attn_implementation = "eager"
    model = tf.Glm5NextTextModel(cfg).eval().float()
    sd = {}
    P = "model.language_model."
    for n, v in tensors.items():
        if not n.startswith(P) or f"layers.{L}." in n or "experts." in n.replace("shared_experts", ""):
            continue
        k = n[len(P):]
        for a, b in (("self_attn.f_a_proj", "self_attn.forget_gate.f_a_proj"),
                     ("self_attn.f_b_proj", "self_attn.forget_gate.f_b_proj"),
                     ("self_attn.dt_bias", "self_attn.forget_gate.dt_bias"),
                     ("self_attn.A_log", "self_attn.forget_gate.A_log"),
                     ("hc_attn_fn", "attn_hc.fn"), ("hc_attn_base", "attn_hc.base"),
                     ("hc_attn_scale", "attn_hc.scale"), ("hc_ffn_fn", "ffn_hc.fn"),
                     ("hc_ffn_base", "ffn_hc.base"), ("hc_ffn_scale", "ffn_hc.scale")):
            k = k.replace(a, b)
        if "_conv1d.weight" in k:
            continue
        sd[k] = torch.tensor(v)
    for l, kind in enumerate(KINDS):
        p = f"{P}layers.{l}."
        if kind == "linear_attention":
            sd[f"layers.{l}.self_attn.conv1d.weight"] = torch.tensor(np.concatenate(
                [tensors[p + f"self_attn.{x}_conv1d.weight"] for x in "qkv"], 0))
        if l >= 3:
            gu = [np.concatenate([tensors[p + f"mlp.experts.{e}.gate_proj.weight"],
                                  tensors[p + f"mlp.experts.{e}.up_proj.weight"]], 0) for e in range(E)]
            sd[f"layers.{l}.mlp.experts.gate_up_proj"] = torch.tensor(np.stack(gu))
            sd[f"layers.{l}.mlp.experts.down_proj"] = torch.tensor(
                np.stack([tensors[p + f"mlp.experts.{e}.down_proj.weight"] for e in range(E)]))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert not [m for m in missing if "e_score_correction_bias" not in m], missing
    for l in range(3, L):
        model.layers[l].mlp.gate.e_score_correction_bias.copy_(
            torch.tensor(tensors[f"{P}layers.{l}.mlp.gate.e_score_correction_bias"]))
    return model


def hf_logits(model, tensors, ids):
    import torch
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), use_cache=False)
        h = out.last_hidden_state[0].numpy()
    return h @ tensors["lm_head.weight"].T


# --------------------------------------------------------------------------------- tests
def test_ordinary_forward_matches_transformers_reference(ckpt_dir, weights):
    fam = family(ckpt_dir)
    model = hf_model(weights)
    for s in seqs(3):
        ref = hf_logits(model, weights, s)
        got = np.array(fam.reference_logits(mx.array([s], mx.int32))[0])
        np.testing.assert_allclose(got, ref, rtol=2e-3, atol=2e-3 * np.abs(ref).max())
        assert (got.argmax(-1) == ref.argmax(-1)).all()


@pytest.mark.parametrize("which", ["f32", "bf16"])
def test_sweep_final_logits_bitwise_equal_ordinary_forward(tmp_path, ckpt_dir, ckpt_bf16, which):
    fam = (family(ckpt_dir, ffn_texts=1) if which == "f32"
           else family(ckpt_bf16, dtype="bfloat16", ffn_texts=1))
    ss = seqs()
    _, final = run(fam, ss, tmp_path)
    for j, s in enumerate(ss):
        got = fam.final_logits(final.read(np.array([j])))
        ref = fam.reference_logits(mx.array([s], mx.int32))
        np.testing.assert_array_equal(np.array(got).view(np.uint32), np.array(ref).view(np.uint32))


def test_bf16_storage_matches_rounded_f32_reference(ckpt_bf16, weights):
    """BF16 on-disk tensors read back as exactly the bf16-rounded values."""
    fam = family(ckpt_bf16)                                   # dtype float32: upcast storage
    model = hf_model(bf16_round(weights))
    rounded = bf16_round(weights)
    s = seqs(1)[0]
    ref = hf_logits(model, rounded, s)
    got = np.array(fam.reference_logits(mx.array([s], mx.int32))[0])
    np.testing.assert_allclose(got, ref, rtol=2e-3, atol=2e-3 * np.abs(ref).max())


def test_packed_ffn_matches_ordinary_within_tolerance(tmp_path, ckpt_dir):
    fam = family(ckpt_dir)                                    # several texts per FFN call
    ss = seqs()
    _, final = run(fam, ss, tmp_path)
    assert fam.counters["ffn_calls"] < fam.L * len(ss)
    for j, s in enumerate(ss):
        got = np.array(fam.final_logits(final.read(np.array([j]))))
        ref = np.array(fam.reference_logits(mx.array([s], mx.int32)))
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def hf_units(model, tensors, ss):
    """Independent oracle from transformers' own forward: dense layers' ``mlp.down_proj`` input and,
    for MoE layers, the router's (topk_indices, topk_weights) seen by ``mlp.experts`` with each
    route's expert output recomputed in numpy (Glm5NextTextExperts law)."""
    import torch
    P = "model.language_model.layers."
    acts, routes = {}, {}
    hooks = []
    for l in range(L):
        mlp = model.layers[l].mlp
        if l < 3:
            hooks.append(mlp.down_proj.register_forward_pre_hook(
                lambda mod, args, l=l: acts.__setitem__(l, args[0][0].detach().numpy().copy())))
        else:
            hooks.append(mlp.experts.register_forward_pre_hook(
                lambda mod, args, l=l: routes.__setitem__(l, tuple(a.detach().numpy().copy() for a in args))))
    res = np.zeros((L, len(ss), 48))
    try:
        for j, s in enumerate(ss):
            acts.clear(); routes.clear()
            with torch.no_grad():
                model(input_ids=torch.tensor([s]), use_cache=False)
            for l in range(L):
                if l < 3:
                    res[l, j] = 0
                    continue
                x, idx, w = routes[l]
                for t in range(1, len(s)):
                    for q in range(idx.shape[1]):
                        e = idx[t, q]
                        g = tensors[f"{P}{l}.mlp.experts.{e}.gate_proj.weight"] @ x[t]
                        u = tensors[f"{P}{l}.mlp.experts.{e}.up_proj.weight"] @ x[t]
                        g, u = np.minimum(g, 10), np.clip(u, -10, 10)
                        row = tensors[f"{P}{l}.mlp.experts.{e}.down_proj.weight"] @ (g / (1 + np.exp(-g)) * u)
                        res[l, j, e] += w[t, q] * np.linalg.norm(row)
                res[l, j] /= len(s) - 1
            yield j, dict(acts), res
    finally:
        for h in hooks:
            h.remove()


def test_resp_and_mu_match_transformers_units(tmp_path, ckpt_dir, weights):
    fam = family(ckpt_dir)                                    # packed FFN, the real run's shape
    ss = seqs()
    out, _ = run(fam, ss, tmp_path)
    acts = {}
    for j, a, res in hf_units(hf_model(weights), weights, ss):
        for l, v in a.items():
            acts[(l, j)] = v
    split = ["fit" if i % 2 else "eval" for i in range(len(ss))]
    mu = np.load(out / "mu.npy")
    for l in range(fam.L):
        r = np.load(out / f"resp_L{l:02d}.npy")
        assert r.shape == (len(ss), max(fam.widths)) and r.dtype == np.float16
        if fam.kinds[l] == "dense":
            fit = np.concatenate([acts[(l, j)][1:] for j in range(len(ss)) if split[j] == "fit"])
            np.testing.assert_allclose(mu[l, :48], fit.mean(0), rtol=1e-3, atol=1e-4)
            want = np.stack([np.abs(acts[(l, j)][1:] - fit.mean(0)).mean(0) for j in range(len(ss))])
            np.testing.assert_allclose(r[:, :48].astype(np.float32), want, rtol=3e-3, atol=3e-3)
            assert not r[:, 48:].any()
        else:
            assert not mu[l].any()
            np.testing.assert_allclose(r[:, :E].astype(np.float32), res[l, :, :E], rtol=3e-3, atol=3e-3)
            assert r[:, :E].astype(np.float32).sum() > 1 and not r[:, E:].any()
    assert json.loads((out / "progress.json").read_text())["done"] == list(range(fam.L))
    assert fam.kinds == ["dense"] * 3 + ["experts"] * 2 and fam.widths == [48] * 3 + [E] * 2


def test_resume_matches_uninterrupted(tmp_path, ckpt_dir):
    ss = seqs()
    a_out, _ = run(family(ckpt_dir, ffn_texts=1), ss, tmp_path / "a", parity=False)
    fam = family(ckpt_dir, ffn_texts=1)
    out, st = tmp_path / "b" / "out", tmp_path / "b" / "state"
    out.mkdir(parents=True)
    np.save(out / "mu.npy", np.zeros((fam.L, max(fam.widths)), np.float32))
    split = ["fit" if i % 2 else "eval" for i in range(len(ss))]
    assert C.sweep(fam, ss, split, out, st, log=lambda m: None, max_layers=2) is None
    assert json.loads((out / "progress.json").read_text())["done"] == [0, 1]
    assert sorted(x.name for x in st.iterdir()) == ["L02"]
    C.sweep(family(ckpt_dir, ffn_texts=1), ss, split, out, st, log=lambda m: None, resume=True)
    for l in range(fam.L):
        np.testing.assert_array_equal(np.load(out / f"resp_L{l:02d}.npy"), np.load(a_out / f"resp_L{l:02d}.npy"))
    np.testing.assert_array_equal(np.load(out / "mu.npy"), np.load(a_out / "mu.npy"))
    np.testing.assert_array_equal(np.load(out / "nll_full.npy"), np.load(a_out / "nll_full.npy"))


def test_kill_inside_a_layer_then_resume_equals_uninterrupted(tmp_path, ckpt_dir):
    ss = seqs()
    a_out, _ = run(family(ckpt_dir, ffn_texts=1), ss, tmp_path / "a", parity=False, max_texts=2)
    fam = family(ckpt_dir, ffn_texts=1)
    inner, calls = fam.run_layer, []

    def run_layer(l, carry):
        calls.append(l)
        if l == 3 and calls.count(3) == 2:                 # second batch of the first MoE layer
            raise SystemExit(143)
        return inner(l, carry)
    fam.run_layer = run_layer
    out, st = tmp_path / "b" / "out", tmp_path / "b" / "st"
    out.mkdir(parents=True)
    np.save(out / "mu.npy", np.zeros((fam.L, max(fam.widths)), np.float32))
    split = ["fit" if i % 2 else "eval" for i in range(len(ss))]
    with pytest.raises(SystemExit):
        C.sweep(fam, ss, split, out, st, log=lambda m: None, max_texts=2)
    assert json.loads((out / "progress.json").read_text())["done"] == [0, 1, 2]
    C.sweep(family(ckpt_dir, ffn_texts=1), ss, split, out, st, log=lambda m: None, resume=True, max_texts=2)
    for l in range(fam.L):
        np.testing.assert_array_equal(np.load(out / f"resp_L{l:02d}.npy"), np.load(a_out / f"resp_L{l:02d}.npy"))
    np.testing.assert_array_equal(np.load(out / "nll_full.npy"), np.load(a_out / "nll_full.npy"))
    assert sorted(x.name for x in st.iterdir()) == [f"L{fam.L:02d}"]


def test_stacked_expert_sweep_matches_routed_reference(tmp_path, ckpt_dir):
    """The real run's expert path (stacked bank + gather_mm) against the default reference, which
    reads experts one at a time: the parity check is not the sweep compared with itself."""
    fam = family(ckpt_dir, experts="all")
    ss = seqs(3)
    _, final = run(fam, ss, tmp_path)
    assert all("all" in fam.weights._banks[l] for l in (3, 4))
    for j, s in enumerate(ss):
        got = np.array(fam.final_logits(final.read(np.array([j]))))
        ref = np.array(fam.reference_logits(mx.array([s], mx.int32)))
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)
    assert fam.weights.force_mode is None
    assert any(isinstance(k, int) for k in fam.weights._banks[3])          # routed experts were read


def test_capture_loader_options_and_budget_guard(tmp_path, ckpt_dir):
    from research.amoeba.sweep.family import load_family
    fam = load_family("glm5_next", ckpt_dir, expert_budget_gb=24, ffn_texts=1, fingerprint="ab" * 32,
                      dtype="float32", experts="routed")
    assert fam.weights.resident_limit == 24e9 and fam.ffn_texts == 1
    small = family(ckpt_dir, expert_budget_gb=0.05)            # < the 5 layers of a parity run
    with pytest.raises(RuntimeError, match="--parity-layers"):
        run(small, seqs(2), tmp_path)
    with pytest.raises(ValueError, match="stacked experts"):
        family(ckpt_dir, experts="all", expert_budget_gb=1e-5)


def test_state_dir_floor_fails_closed(tmp_path, ckpt_dir, monkeypatch):
    import shutil
    fam = family(ckpt_dir)
    ss = seqs(4)
    info = fam.prepare_layer(0, ss, tmp_path)
    assert info["boundary_gb"] >= 0
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(shutil, "disk_usage", lambda p: usage._replace(free=fam.state_floor + 10))
    with pytest.raises(SystemExit, match="floor"):
        fam.prepare_layer(0, ss, tmp_path)


def test_prefetch_ranges_cover_each_layer_text_tensors(ckpt_dir):
    fam = family(ckpt_dir)
    ranges = fam.prefetch_ranges
    assert sorted(ranges) == list(range(fam.L))
    for l in range(fam.L):
        want = sum(t.nbytes for n, t in fam.ckpt.tensors._tensors.items()
                   if n.startswith(f"model.language_model.layers.{l}."))
        got = sum(e - b for _, b, e in ranges[l])
        assert got == want                  # this checkpoint's tensors are disjoint byte ranges
        files = {f for f, _, _ in ranges[l]}
        assert files <= {str(p) for p in ckpt_dir.glob("*.safetensors")}


def test_tokenizer_keeps_control_spellings_literal():
    path = Path.home() / "Models/Sage-S2.1-Flash"
    if not (path / "tokenizer.json").exists():
        pytest.skip("Sage-S2.1-Flash not mounted")
    from research.amoeba.sweep.glm5_next import _Tok
    tok = _Tok(path)
    plain = "The KV cache grows linearly with context length."
    assert tok.encode(plain) == tok.t.encode(plain, add_special_tokens=False).ids
    controls = {t.content: i for i, t in tok.t.get_added_tokens_decoder().items()}
    text = "a model emits <think> then </think> and <|user|> or <|endoftext|> in docs"
    ids = tok.encode(text)
    assert not set(ids) & {controls[k] for k in ("<think>", "</think>", "<|user|>", "<|endoftext|>")}
    assert tok.decode(ids) == text


def test_reads_only_text_tensors_of_the_layer(tmp_path, ckpt_dir):
    fam = family(ckpt_dir)
    seen = []
    raw = fam.ckpt.raw
    fam.ckpt.raw = lambda name: (seen.append(name), raw(name))[1]
    rows = fam.ckpt.rows
    fam.ckpt.rows = lambda name, ids: (seen.append(name), rows(name, ids))[1]
    carry = fam.embed(mx.array([seqs(1)[0]], mx.int32))
    seen.clear()
    carry, _ = fam.run_layer(1, carry)
    assert seen and all(n.startswith("model.language_model.layers.1.") for n in seen)
    seen.clear()
    fam.run_layer(3, carry)
    assert all(n.startswith("model.language_model.layers.3.") for n in seen)
    assert not [n for n in seen if "experts." in n and ".shared_experts." not in n
                and int(n.split("experts.")[1].split(".")[0]) >= E]
    fam.release_layer(1)
    assert not [k for k in fam.weights._cache if k[0].startswith("blk.1.")]


def test_long_sequence_fails_closed(ckpt_dir):
    fam = family(ckpt_dir)
    fam.topk = 8
    with pytest.raises(ValueError, match="all-retained"):
        fam.embed(mx.array([list(range(1, 11))], mx.int32))
    carry = {"ids": mx.zeros((1, 10), mx.int32), "h": mx.zeros((1, 10, 4, D))}
    with pytest.raises(ValueError, match="all-retained"):
        fam.run_layer(3, carry)


def test_indexer_selection_fails_closed(ckpt_dir, monkeypatch):
    from mlx2.runtime.models import glm53_gguf
    fam = family(ckpt_dir)
    carry = fam.embed(mx.array([seqs(1)[0]], mx.int32))
    orig = glm53_gguf.Glm53GGUFModel._mla_select_cache

    def select(self, x, qr, cache, prefix, projected, total):
        sel = orig(self, x, qr, cache, prefix, projected, total)
        return mx.zeros((1, x.shape[1], 2051), mx.int32) if sel is None else sel
    monkeypatch.setattr(glm53_gguf.Glm53GGUFModel, "_mla_select_cache", select)
    with pytest.raises(RuntimeError, match="all-retained"):
        fam.run_layer(3, carry)


@pytest.mark.parametrize("key,value", [("index_kpool", 8), ("hc_mult", 2), ("qk_rope_head_dim", 64),
                                       ("first_k_dense_replace", 1), ("num_experts_per_tok", 6)])
def test_unsupported_geometry_is_refused(key, value):
    tc = tiny_text_config()
    tc[key] = value
    with pytest.raises(ValueError, match=key):
        text_config({"model_type": "glm5_next", "text_config": tc})


def test_real_sage_config_is_admitted():
    path = Path.home() / "Models/Sage-S2.1-Flash/config.json"
    if not path.exists():
        pytest.skip("Sage-S2.1-Flash not mounted")
    cfg = text_config(json.loads(path.read_text()))
    assert cfg["num_hidden_layers"] == 45 and cfg["n_routed_experts"] == 288
    assert cfg["layer_types"].count("deepseek_sparse_attention") == 11


def test_registered_in_family_table():
    from research.amoeba.sweep.family import FAMILIES
    assert FAMILIES["glm5_next"] == "research.amoeba.sweep.glm5_next:load"


def test_expert_rows_follow_the_vendor_swiglu_clamp(ckpt_dir, weights):
    """Glm5NextTextExperts: clamp(gate, max=10), clamp(up, +-10), silu(gate) * up, down; inputs
    scaled so both clamps bind."""
    fam = family(ckpt_dir)
    rng = np.random.default_rng(11)
    x = rng.normal(0, 40, (5, D)).astype(np.float32)
    ids = np.stack([rng.permutation(E)[:8] for _ in range(5)]).astype(np.int32)
    got = np.array(fam.weights.expert_rows(3, mx.array(x), mx.array(ids)))
    p = "model.language_model.layers.3.mlp.experts."
    clamped = 0
    for n in range(5):
        for j, e in enumerate(ids[n]):
            g = weights[p + f"{e}.gate_proj.weight"] @ x[n]
            u = weights[p + f"{e}.up_proj.weight"] @ x[n]
            clamped += int((g > 10).any() and (np.abs(u) > 10).any())
            g, u = np.minimum(g, 10), np.clip(u, -10, 10)
            want = weights[p + f"{e}.down_proj.weight"] @ (g / (1 + np.exp(-np.maximum(g, -80))) * u)
            np.testing.assert_allclose(got[n, j], want, rtol=1e-4, atol=1e-4)
    assert clamped > 10
