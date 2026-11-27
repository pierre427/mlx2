"""CPU tests of whole-layer weight streaming for the rolling map capture (research/amoeba/sweep/layer_swap.py) on
tiny synthetic checkpoints written in each artifact's own on-disk layout:

- kimi_linear: the hub BF16 layout (per-expert tensors, HF depthwise convs) and the MLX affine q8 layout;
- nemotron_h: the Nemotron 3.5 Lightning affine layout and the Super VL NVFP4 export layout (NVFP4 experts with
  per-expert weight_scale_2 folded into the routing, vendor FP8 dense projections, language_model. prefix, media and
  MTP tensors present but never read).

Each streamed family (one layer's every tensor read per chunk) is compared BITWISE against the same model loaded
resident the adapter's way (whole checkpoint through sanitize, load_weights strict, the adapter's installers),
scheduled layer-major by the host family: every resp_L, mu and the NLL; plus the per-run parity gate and the read
count."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "scripts", ROOT / "tests"):
    sys.path.insert(0, str(p))

from mlx import nn  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from mlx2.experimental.amoeba import child_rolling as R  # noqa: E402

V = 96


@pytest.fixture(autouse=True)
def _cpu():
    mx.set_default_device(mx.cpu)


class Tok:
    def encode(self, text, add_special_tokens=False):
        return [b % V for b in text.encode()]

    def decode(self, ids):
        return "".join(chr(97 + i % 26) for i in ids)


def _seqs(n, T, seed=3):
    rng = np.random.default_rng(seed)
    lens = [T, T, T - 3, T - 3, T, T - 5][:n]
    return [rng.integers(0, V, size=t).tolist() for t in lens]


def _randomize(model, seed, scale=0.08, keep=()):
    """Random float parameters (quantized banks are produced from them later by nn.quantize)."""
    mx.random.seed(seed)
    new = {}
    for k, v in tree_flatten(model.parameters()):
        if any(s in k for s in keep) or not mx.issubdtype(v.dtype, mx.floating):
            continue
        new[k] = (mx.random.normal(v.shape) * scale).astype(v.dtype)
    model.load_weights(list(new.items()), strict=False)
    mx.eval(model.parameters())


def _write(path, tensors, per_shard=40):
    """safetensors shards plus the HF index, tensors spread over shards in key order."""
    path.mkdir(parents=True, exist_ok=True)
    keys = sorted(tensors)
    wm = {}
    for i in range(0, len(keys), per_shard):
        name = f"model-{i // per_shard:05d}.safetensors"
        mx.save_safetensors(str(path / name), {k: tensors[k] for k in keys[i:i + per_shard]})
        wm.update({k: name for k in keys[i:i + per_shard]})
    (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    return wm


def _same_captures(a, b, L):
    for l in range(L):
        x, y = np.load(a / f"resp_L{l:02d}.npy"), np.load(b / f"resp_L{l:02d}.npy")
        assert x.shape == y.shape and np.array_equal(x.view(np.uint8), y.view(np.uint8)), l


def _check_family(tmp_path, fam, resident_host, L, chunks=2, carry="resident"):
    """The streamed family vs the resident model under the same layer-major schedule: bitwise."""
    from research.amoeba.sweep.host_family import HostFamily
    ref = HostFamily(resident_host)
    ref.layer_scope = fam.layer_scope
    seqs = _seqs(6, 12)
    fit = [True, False, True, True, False, True]
    (tmp_path / "s").mkdir()
    (tmp_path / "r").mkdir()
    loads0 = fam.loads
    rs, mus, nlls = R.run(fam, seqs, fit, tmp_path / "s", window=2, chunks=chunks, carry_mode=carry, max_texts=2,
                          log=lambda *_: None)
    assert fam.loads - loads0 == L * rs["chunks"]                 # each layer read once per chunk, experts included
    rr, mur, nllr = R.run(ref, seqs, fit, tmp_path / "r", window=2, chunks=chunks, carry_mode=carry, max_texts=2,
                          log=lambda *_: None)
    _same_captures(tmp_path / "s", tmp_path / "r", L)
    assert np.array_equal(mus.view(np.uint32), mur.view(np.uint32))
    assert np.array_equal(nlls.view(np.uint32), nllr.view(np.uint32))
    # the per-run gate: schedule vs the family's own text-major forward (layers loaded as the trunk calls them)
    par = R.parity(fam, seqs[:3], None, log=lambda *_: None)
    assert par["ok"] and all(t["bitwise_equal"] for t in par["texts"])
    # and that forward is the resident model's ordinary forward, bit for bit
    ids = mx.array(np.array([s[:9] for s in seqs[:3]], np.int32))
    got, want = fam.reference_logits(ids), resident_host.lm(ids)
    assert np.array_equal(np.array(got.astype(mx.float32)).view(np.uint32),
                          np.array(want.astype(mx.float32)).view(np.uint32))
    assert not fam._loaded                                      # nothing left bound after a pass
    return rs


# ------------------------------------------------------------------ Kimi Linear

KIMI_CPU_POLICY = {"kda": "reference", "mla": "expanded", "mla_flash": "off", "rms_law": "reference"}


def kimi_config(experts=8):
    return {
        "architectures": ["KimiLinearForCausalLM"], "model_type": "kimi_linear", "dtype": "bfloat16",
        "first_k_dense_replace": 1, "head_dim": 16, "hidden_act": "silu", "hidden_size": 64,
        "intermediate_size": 128, "kv_lora_rank": 64, "mla_use_nope": True, "model_max_length": 4096,
        "moe_intermediate_size": 64, "moe_layer_freq": 1, "moe_renormalize": True,
        "moe_router_activation_func": "sigmoid", "num_attention_heads": 4, "num_expert_group": 1,
        "num_experts": experts, "num_experts_per_token": 2, "num_hidden_layers": 4, "num_key_value_heads": 4,
        "num_nextn_predict_layers": 0, "num_shared_experts": 1, "q_lora_rank": None, "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 16, "rms_norm_eps": 1e-5, "rope_scaling": None, "rope_theta": 10000.0,
        "routed_scaling_factor": 2.446, "tie_word_embeddings": False, "topk_group": 1, "use_grouped_topk": True,
        "v_head_dim": 16, "vocab_size": V,
        "linear_attn_config": {"full_attn_layers": [4], "kda_layers": [1, 2, 3], "head_dim": 16, "num_heads": 4,
                               "short_conv_kernel_size": 4},
    }


def _kimi_model(cfg, seed):
    from mlx2.runtime.models.kimi_linear import Model, ModelArgs
    m = Model(ModelArgs.from_dict(cfg))
    _randomize(m, seed, keep=("A_log", "dt_bias"))
    # BF16 as the hub checkpoint stores it (the model is built in float32, so the streamed loads bind BF16 into
    # float32-initialised modules, as load_weights does for the real checkpoint); the routed experts stay float32
    # (CPU gather_mm runs only in float32) and A_log / dt_bias are F32 by the hub dtype law
    m.load_weights([(k, v.astype(mx.bfloat16)) for k, v in tree_flatten(m.parameters())
                    if not k.endswith((".A_log", ".dt_bias")) and ".switch_mlp." not in k], strict=False)
    mx.eval(m.parameters())
    return m


def _kimi_hub_tensors(model, cfg):
    """The hub BF16 checkpoint of ``model`` (the inverse of Model.sanitize's renames and stacking)."""
    out = {}
    E = cfg["num_experts"]
    for k, v in tree_flatten(model.parameters()):
        moe = ".mlp." in k and not k.startswith("model.layers.0.")
        h = k.replace("model.word_embeddings.", "model.embed_tokens.").replace(".attention.", ".self_attn.")
        if moe:
            h = h.replace(".mlp.", ".block_sparse_moe.")
            h = h.replace(".gate.expert_bias", ".gate.e_score_correction_bias")
            h = h.replace(".gate.gate_proj.weight", ".gate.weight")
            if ".switch_mlp." in h:
                name = h.split(".switch_mlp.")[1].split(".")[0]
                w = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}[name]
                for e in range(E):
                    out[h.split(".switch_mlp.")[0] + f".experts.{e}.{w}.weight"] = v[e]
                continue
        for n in ("q", "k", "v"):
            if f".self_attn.{n}_conv1d.conv.weight" in h:
                h = h.replace(f".{n}_conv1d.conv.weight", f".{n}_conv1d.weight")
                v = v.moveaxis(1, 2)                                   # MLX [C, K, 1] -> HF depthwise [C, 1, K]
        out[h] = v
    return out


def _kimi_q8_tensors(model):
    """The mlx-lm affine q8 checkpoint of a quantized ``model`` (the conversion's names)."""
    out = {}
    for k, v in tree_flatten(model.parameters()):
        h = k.replace("model.word_embeddings.", "model.embed_tokens.").replace(".attention.", ".self_attn.")
        h = h.replace(".gate.expert_bias", ".e_score_correction_bias")
        for part in ("weight", "scales", "biases"):
            if h.endswith(f".mlp.gate.gate_proj.{part}"):
                h = h[:-len(f"gate_proj.{part}")] + part
        for n in ("q", "k", "v"):
            h = h.replace(f".self_attn.{n}_conv1d.conv.", f".self_attn.{n}_conv.conv.")
        out[h] = v
    return out


def _q8(model):
    """MLX affine 8/64 on every projection whose input width the group divides (the tiny MLA/KDA head dims do not;
    the checkpoint's .scales then name exactly the quantized modules, as the adapter's predicate reads them)."""
    from mlx2.adapters import kimi_linear as K
    nn.quantize(model, **K.QUANTIZATION, class_predicate=lambda name, m: hasattr(m, "to_quantized")
                and getattr(m, "weight", None) is not None and m.weight.shape[-1] % 64 == 0)
    mx.eval(model.parameters())


@pytest.fixture
def _env():
    """Adapters pin their process profile into os.environ; give it back after the test."""
    import os
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _stub_tokenizers(monkeypatch):
    """The adapters' tokenizer loads (a tiny checkpoint has no real vocabulary): stubs, nothing numerical."""
    from types import SimpleNamespace

    import transformers

    from mlx2.runtime import kimi_tokenizer, tokenizer_integrity, tokenizer_utils
    monkeypatch.setattr(kimi_tokenizer.TikTokenTokenizer, "from_pretrained",
                        classmethod(lambda cls, *a, **k: SimpleNamespace(chat_template=None)))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        staticmethod(lambda *a, **k: SimpleNamespace(convert_tokens_to_ids=lambda t: 18)))
    monkeypatch.setattr(tokenizer_integrity, "repair_loaded_tokenizer", lambda tok, path: {})
    monkeypatch.setattr(tokenizer_utils, "TokenizerWrapper", lambda tok, **k: Tok())


def _kimi_adapter_resident(monkeypatch, ck, art, policy):
    """The resident model the KimiLinearAdapter itself loads (its _load, unmodified), with only its artifact
    inspection (which pins the real topology) answered by the tiny artifact."""
    from mlx2.adapters import kimi_linear as K
    (ck / "chat_template.jinja").write_text("")
    monkeypatch.setattr(K, "inspect_artifact", lambda p: art)
    _stub_tokenizers(monkeypatch)
    adapter = K.KimiLinearAdapter(str(ck), execution_policy=dict(policy))
    mx.eval(adapter.model.parameters())
    return adapter


@pytest.mark.parametrize("layout", ["hub_bf16", "mlx_affine"])
@pytest.mark.parametrize("units", ["neuron", "expert"])
def test_kimi_streamed_equals_resident(tmp_path, monkeypatch, _env, layout, units):
    """Streamed through the family's own loader (``kimi_linear.load``: its inspection, spec and policy) vs the
    resident model KimiLinearAdapter loads: every response, mu, NLL and the forward bitwise."""
    from research.amoeba.carve.host import Host
    from research.amoeba.sweep import kimi_linear as KF
    from mlx2.adapters import kimi_linear as K
    cfg = kimi_config()
    src = _kimi_model(cfg, seed=1)
    if layout == "mlx_affine":
        _q8(src)
        tensors = _kimi_q8_tensors(src)
        cfg = {**cfg, "quantization": dict(K.QUANTIZATION)}
    else:
        tensors = _kimi_hub_tensors(src, cfg)
    ck = tmp_path / "kimi"
    wm = _write(ck, tensors)
    (ck / "config.json").write_text(json.dumps(cfg))
    art = {"config": cfg, "weight_map": wm, "layout": layout, "identity": {"path": str(ck), "fingerprint": "k" * 64}}
    adapter = _kimi_adapter_resident(monkeypatch, ck, art, KIMI_CPU_POLICY)
    resident = adapter.model
    # the checkpoint round-trips the source model exactly (so the comparison is not vacuous)
    assert all(np.array_equal(np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32)))
               for (_, a), (_, b) in zip(sorted(tree_flatten(src.parameters())),
                                         sorted(tree_flatten(resident.parameters()))))
    fam = KF.load(ck, moe_units=units, log=lambda *_: None, policy=KIMI_CPU_POLICY, tokenizer=Tok())
    assert fam.spec.how == "adapter" and fam.spec.policy == adapter.policy and fam.fingerprint == "k" * 64
    host = Host(model=resident, tokenizer=Tok(), fingerprint="k" * 64, log=lambda *_: None, moe_units=units)
    assert fam.widths == host.widths and fam.kinds == [("dense" if s.kind == "dense" else
                                                         ("neurons" if s.units == "neuron" else "experts"))
                                                        if s.kind else None for s in host.sites]
    assert fam.kinds[0] == "dense" and set(fam.kinds[1:]) == {"neurons" if units == "neuron" else "experts"}
    rec = _check_family(tmp_path, fam, host, cfg["num_hidden_layers"])
    assert rec["chunks"] == 2
    # prefetch covers every streamed tensor's bytes, per layer
    ranges = fam.prefetch_ranges
    assert sorted(ranges) == list(range(cfg["num_hidden_layers"])) and all(ranges[l] for l in ranges)


def test_kimi_release_drops_derived_caches(tmp_path):
    """A dequantized copy cached by array id (_kimi_dense) must not survive its layer: a later array can reuse the
    id. Streaming two different checkpoints' layer 1 through one model would otherwise reuse a stale copy."""
    from research.amoeba.sweep import kimi_linear as KF
    from research.amoeba.sweep.layer_swap import LayerSwapFamily
    from mlx2.adapters import kimi_linear as K
    cfg = {**kimi_config(), "quantization": dict(K.QUANTIZATION)}
    src = _kimi_model(kimi_config(), seed=2)
    _q8(src)
    ck = tmp_path / "kimi"
    wm = _write(ck, _kimi_q8_tensors(src))
    (ck / "config.json").write_text(json.dumps(cfg))
    art = {"config": cfg, "weight_map": wm, "layout": "mlx_affine", "identity": {"path": str(ck), "fingerprint": "q"}}
    fam = LayerSwapFamily(KF.KimiSpec(ck, art, "test", policy=KIMI_CPU_POLICY, tokenizer=Tok()), moe_units="expert",
                          log=lambda *_: None)
    fam.begin_layer(3)                                     # a MoE layer whose quantized router goes through dense_weight
    ids = mx.array([[1, 2, 3, 4, 5]])
    carry = fam.embed(ids)
    fam.run_layer(3, carry)
    kv_b = fam._layers[3].mlp.gate.gate_proj               # the FP32 router reads its dequantized copy
    assert "_kimi_dense" in vars(kv_b)
    fam.release_layer(3)
    assert "_kimi_dense" not in vars(kv_b) and not fam._loaded


def test_kimi_refusals(tmp_path):
    from research.amoeba.sweep import kimi_linear as KF
    cfg = kimi_config()
    art = {"config": cfg, "weight_map": {}, "layout": "hub_bf16", "identity": {"path": str(tmp_path), "fingerprint": "x"}}
    with pytest.raises(SystemExit, match="precompute"):
        KF.KimiSpec(tmp_path, art, "test", policy={"fused_projections": True})
    # a checkpoint the adapter does not stage and that is not the 256-expert hub layout refuses at inspection
    for experts in (64, 180):          # 180 = the REAP hub layout: what the adapter refuses stays refused
        (tmp_path / "config.json").write_text(json.dumps({**cfg, "num_experts": experts}))
        with pytest.raises(SystemExit, match="not the 256-expert hub"):
            KF.inspect(tmp_path)


def test_layer_swap_refuses_a_missing_layer_tensor(tmp_path):
    from research.amoeba.sweep import kimi_linear as KF
    from research.amoeba.sweep.layer_swap import LayerSwapFamily
    cfg = kimi_config()
    tensors = _kimi_hub_tensors(_kimi_model(cfg, seed=3), cfg)
    tensors.pop("model.layers.2.self_attn.o_proj.weight")
    ck = tmp_path / "kimi"
    wm = _write(ck, tensors)
    art = {"config": cfg, "weight_map": wm, "layout": "hub_bf16", "identity": {"path": str(ck), "fingerprint": "x"}}
    fam = LayerSwapFamily(KF.KimiSpec(ck, art, "test", policy=KIMI_CPU_POLICY, tokenizer=Tok()), moe_units="expert",
                          log=lambda *_: None)
    fam.begin_layer(1)
    fam.release_layer(1)
    with pytest.raises(SystemExit, match="layer 2"):
        fam.begin_layer(2)
    tensors.pop("model.norm.weight")
    ck2 = tmp_path / "kimi2"
    wm2 = _write(ck2, tensors)
    art2 = {**art, "weight_map": wm2, "identity": {"path": str(ck2), "fingerprint": "x"}}
    with pytest.raises(SystemExit, match="resident"):
        LayerSwapFamily(KF.KimiSpec(ck2, art2, "test", policy=KIMI_CPU_POLICY, tokenizer=Tok()), moe_units="expert",
                        log=lambda *_: None)


KIMI48 = Path.home() / "Models/Kimi-Linear-48B-A3B-Instruct"


@pytest.mark.skipif(not (KIMI48 / "config.json").exists(), reason="Kimi-Linear-48B BF16 not on disk")
def test_kimi48_hub_bf16_is_admitted_by_the_hub_contract():
    """Header-only: the original 48B hub BF16 (256 experts) is the adapter's hub contract except the expert count."""
    from research.amoeba.sweep import kimi_linear as KF
    art, how = KF.inspect(KIMI48)
    assert how == "hub48" and art["layout"] == "hub_bf16" and art["config"]["num_experts"] == 256


# ------------------------------------------------------------------ Nemotron-H

def nemotron_text(**over):
    cfg = {"model_type": "nemotron_h", "vocab_size": V, "hidden_size": 64, "intermediate_size": 64,
           "max_position_embeddings": 4096, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
           "attention_bias": False, "mamba_num_heads": 4, "mamba_head_dim": 16, "mamba_proj_bias": False,
           "ssm_state_size": 16, "conv_kernel": 4, "n_groups": 2, "mlp_bias": False, "layer_norm_epsilon": 1e-5,
           "use_bias": False, "use_conv_bias": True, "moe_intermediate_size": 64,
           "moe_shared_expert_intermediate_size": 64, "moe_latent_size": None, "n_group": 1, "n_routed_experts": 8,
           "n_shared_experts": 1, "topk_group": 1, "num_experts_per_tok": 2, "norm_topk_prob": True,
           "routed_scaling_factor": 2.5, "time_step_limit": (0.0, float("inf")), "time_step_min": 0.001,
           "time_step_max": 0.1, "num_nextn_predict_layers": 0, "mlp_hidden_act": "relu2",
           "layers_block_type": ["mamba", "moe", "mamba", "attention", "moe", "moe"]}
    cfg.update(over)
    cfg["num_hidden_layers"] = len(cfg["layers_block_type"])
    return cfg


def _nemotron_src(model_class, text, seed):
    from mlx2.runtime.models.nemotron_h import ModelArgs
    mx.random.seed(seed)
    m = model_class(ModelArgs.from_dict(text))
    for name in ("vision_model", "vision_projector", "mtp"):
        if name in m:
            m.pop(name)
    _randomize(m, seed, scale=0.1, keep=("A_log", "dt_bias", ".D"))
    return m


def _run_nemotron(tmp_path, spec, resident, L):
    import gc
    from research.amoeba.carve.host import Host
    from research.amoeba.sweep.layer_swap import LayerSwapFamily
    fam = LayerSwapFamily(spec, moe_units="expert", log=lambda *_: None)
    host = Host(model=resident, tokenizer=Tok(), fingerprint=spec.fingerprint, log=lambda *_: None,
                moe_units="expert")
    assert fam.widths == host.widths and [s.kind for s in host.sites].count("nemotron") == 3
    _check_family(tmp_path, fam, host, L, chunks=2)
    with pytest.raises(SystemExit, match="neuron"):        # carve neuron units cover SwitchGLU only
        LayerSwapFamily(spec, moe_units="neuron", log=lambda *_: None)
    del fam
    gc.collect()


def test_nemotron_lightning_streamed_equals_resident(tmp_path, monkeypatch, _env):
    """Streamed through ``nemotron_h.spec_for`` (the family's own artifact routing) vs the resident model the
    Nemotron35LightningAdapter loads (its _init_lightning, unmodified; only the pinned inspection answered by the
    tiny artifact). The config declares an MTP head the checkpoint does not carry: the adapter's sanitize drops it,
    the family never builds it."""
    from research.amoeba.sweep import nemotron_h as NF
    from mlx2.adapters import nemotron35_lightning as NL
    from mlx2.runtime.models.nemotron_h import Model
    quant = {"group_size": 64, "bits": 8, "mode": "affine"}
    text = nemotron_text(time_step_limit=None, quantization=quant, num_nextn_predict_layers=1,
                         mtp_layers_block_type=["attention", "moe"])
    src = _nemotron_src(Model, text, seed=5)
    nn.quantize(src, **quant)
    mx.eval(src.parameters())
    ck = tmp_path / "light"
    wm = _write(ck, dict(tree_flatten(src.parameters())))
    art = {"config": text, "weight_map": wm, "has_mtp": False, "mtp_tensor_count": 0, "mtp_path": None,
           "identity": {"path": str(ck), "fingerprint": "n" * 64, "files": []}}
    monkeypatch.setattr(NL, "inspect_artifact", lambda p: art)
    monkeypatch.setattr(NF, "_adapter_of", lambda p: NL.Nemotron35LightningAdapter)
    _stub_tokenizers(monkeypatch)
    adapter = NL.Nemotron35LightningAdapter(str(ck))
    assert getattr(adapter.model, "mtp", None) is None
    spec = NF.spec_for(ck, tokenizer=Tok())
    assert spec.kind == "lightning" and spec.fingerprint == adapter.identity["fingerprint"]
    _run_nemotron(tmp_path, spec, adapter.model, text["num_hidden_layers"])


def _fp8_exact(shape, rng):
    """A bf16 weight that is exactly bf16(e4m3 * scale) (the vendor FP8 export law), and its scale."""
    scale = float(rng.uniform(0.001, 0.004))
    codes = mx.to_fp8(mx.array(rng.normal(0, 30, shape).astype(np.float32)))
    return (mx.from_fp8(codes, dtype=mx.float32) * mx.array(scale, mx.float32)).astype(mx.bfloat16), scale


def test_nemotron_super_nvfp4_export_streamed_equals_resident(tmp_path, monkeypatch, _env):
    """The Super VL NVFP4 export: streamed through ``nemotron_h.spec_for`` vs the resident model
    Nemotron35SuperAdapter loads (_init_super / _load_super_model unmodified: its NVFP4-export defaults -- NAX
    gather, vendor FP8 dense recovery, the expert-scale fold, exact attention rows). Answered by the tiny artifact:
    the pinned inspection, the pinned language config, the vendor FP8 scale table, and a tiny RADIO tower in the
    composition. The checkpoint carries the vision tower, projector and an MTP head (with NVFP4 experts of its own),
    which the adapter loads and the text map never reads."""
    from mlx2.adapters import nemotron35_super as S
    from mlx2.adapters import nemotron35_super_config as SC
    from mlx2.adapters.radio_config import RadioConfig
    from research.amoeba.sweep import nemotron_h as NF
    from mlx2.runtime.models.nemotron35_super import Model
    text = nemotron_text(moe_latent_size=64, router_compute_dtype="float32", num_nextn_predict_layers=1,
                         mtp_layers_block_type=["attention", "moe"])
    radio = RadioConfig(32, 1, 2, 64, 16, 64, 3, 7, (0, 1), (32, 32), False)

    def composition(args):
        return Model(args, radio_config=radio, projector_hidden_size=64)
    from mlx2.runtime.models.nemotron_h import ModelArgs
    mx.random.seed(7)
    src = composition(ModelArgs.from_dict(text))
    _randomize(src, 7, scale=0.1, keep=("A_log", "dt_bias", ".D", "summary_idxs"))
    src.set_dtype(mx.bfloat16)
    rng = np.random.default_rng(7)
    moe = [i for i, t in enumerate(text["layers_block_type"]) if t == "moe"]
    nv = [f"backbone.layers.{i}.mixer.{p}" for i in moe
          for p in ("switch_mlp.fc1", "switch_mlp.fc2", "shared_experts.up_proj", "shared_experts.down_proj")]
    nv += [f"mtp.layers.1.mixer.{p}" for p in ("switch_mlp.fc1", "switch_mlp.fc2")]    # the export's MTP experts
    fp8 = {}
    params = dict(tree_flatten(src.parameters()))
    for path in ["backbone.layers.0.mixer.in_proj", "backbone.layers.3.mixer.o_proj", "lm_head"]:
        w, s = _fp8_exact(params[path + ".weight"].shape, rng)
        params[path + ".weight"] = w
        fp8[path] = s
    for path in nv:
        w = params.pop(path + ".weight").astype(mx.float32)
        s2 = rng.uniform(0.5, 2.0, (w.shape[0],) if w.ndim == 3 else ()).astype(np.float32)
        q, sc = mx.quantize(w, group_size=16, bits=4, mode="nvfp4")
        params[path + ".weight"], params[path + ".scales"] = q, sc
        params[path + ".weight_scale_2"] = mx.array(s2)
    tensors = {(k if k.startswith(("vision_model.", "vision_projector.")) else "language_model." + k): v
               for k, v in params.items()}
    assert any(".mtp." in k for k in tensors) and any(k.startswith("vision_model.") for k in tensors)
    ck = tmp_path / "super"
    wm = _write(ck, tensors)
    art = {"config": {}, "weight_map": wm, "conversion": {}, "has_mtp": True, "quantization": {},
           "nvfp4_modules": nv, "eos_token_ids": (2, 11),
           "identity": {"path": str(ck), "fingerprint": "s" * 64, "files": []}}
    monkeypatch.setattr(S.Nemotron35SuperAdapter, "artifact_inspector", staticmethod(lambda p: art))
    monkeypatch.setattr(S.Nemotron35SuperAdapter, "_runtime_language_config", staticmethod(lambda c: dict(text)))
    monkeypatch.setattr(S.Nemotron35SuperAdapter, "_runtime_model_class", staticmethod(lambda: composition))
    monkeypatch.setattr(SC, "vendor_fp8_scales", lambda: dict(fp8))
    monkeypatch.setattr(NF, "_adapter_of", lambda p: S.Nemotron35SuperAdapter)
    monkeypatch.delenv("MLX2_NVFP4_DENSE", raising=False)
    monkeypatch.delenv("MLX2_NVFP4_EXPERT_SCALE", raising=False)
    _stub_tokenizers(monkeypatch)
    adapter = S.Nemotron35SuperAdapter(str(ck))
    assert adapter.moe_nax_gather == "gather" and adapter.nvfp4_expert_scale == "fold"
    assert adapter.model.mtp is not None and "vision_model" in adapter.model      # the adapter loads all of it
    spec = NF.spec_for(ck, tokenizer=Tok())
    assert spec.kind == "super" and spec.nax == "gather" and spec.fold and spec.dropped_installs == 2
    fam_keys = [spec.layer_of(k) for k in wm]
    assert fam_keys.count("skip") == sum(1 for k in wm if k.startswith("vision_") or ".mtp." in k)
    _run_nemotron(tmp_path, spec, adapter.model, text["num_hidden_layers"])


def test_nemotron_refuses_other_adapters(monkeypatch, tmp_path):
    from research.amoeba.sweep import nemotron_h as NF

    class Bolt:
        pass
    monkeypatch.setattr(NF, "_adapter_of", lambda p: Bolt)
    with pytest.raises(SystemExit, match="refused"):
        NF.spec_for(tmp_path)
    with pytest.raises(SystemExit, match="policy"):
        NF.spec_for(tmp_path, policy={"mamba_conv": "fused_short"})


SUPER = Path.home() / "mlx-models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"


@pytest.mark.skipif(not (SUPER / "config.json").exists(), reason="Super VL NVFP4 export not on disk")
def test_super_vl_spec_from_the_adapter_inspector():
    """Header-only: the real export resolves to the Super adapter's own inspection and its checkpoint defaults; every
    checkpoint key is a layer, resident, or a media/MTP skip."""
    from research.amoeba.sweep import nemotron_h as NF
    spec = NF.spec_for(SUPER, tokenizer=Tok())
    assert spec.kind == "super" and spec.fold and spec.nax == "gather" and spec.nvfp4_modules
    where = [spec.layer_of(k) for k in spec.weight_map]
    assert None not in where
    layers = {w for w in where if isinstance(w, int)}
    assert layers == set(range(88)) and "skip" in where and "resident" in where
    # Re-pinned 2026-11-27: the adapter fingerprint binds each shard's
    # (st_dev, st_ino, size, mtime, ctime); the export moved to
    # the model disk behind a symlink on 2026-11-20, so the identity
    # changed with the same bytes (was 33a6f018).
    assert spec.fingerprint[:8] == "936ae320"


# ------------------------------------------------------------------ plan, inspection, CLI

def _moe_info(weights_gb, family, L=27, d=2304):
    per = int(weights_gb * 2 ** 30 * 0.98 / L)
    adapter = {"kimi_linear": "KimiLinearAdapter", "nemotron_h": "Nemotron35SuperAdapter"}.get(family, "A")
    return {"path": "/m", "checkpoint": "big@0", "fingerprint": "0" * 64, "adapter": adapter, "kind": "moe",
            "family": family, "model_type": family, "sweep_family": family, "layers": L, "hidden": d,
            "intermediate": 9216, "vocab": 163840, "moe": {"experts": 256, "expert_intermediate": 1024,
                                                           "moe_layers": L - 1, "top_k": 8},
            "file_backed_ple": False, "weight_bytes": int(weights_gb * 2 ** 30),
            "units": {"dense_neurons": 9216, "moe_expert_units": 256 * (L - 1), "moe_neuron_units": 0,
                      "estimate": True},
            "layer_bytes": {"per_layer": [per] * L, "max_layer": per, "rest": int(weights_gb * 2 ** 30 * 0.02),
                            "experts_per_layer": [per] * L, "max_layer_experts": per, "layered": True}}


def test_plan_streams_a_whole_layer_family_with_a_read_ahead(monkeypatch):
    from mlx2.experimental.amoeba import config as K
    from mlx2.experimental.amoeba import plan as P
    monkeypatch.setattr(P, "physical_gb", lambda: 128.0)
    cfg = K.resolve("full", {}, {}, "")
    p = P.make_plan(_moe_info(92, "kimi_linear"), cfg, texts=46000)          # Kimi 48B BF16: over the 80 GB cap
    st = p["stream"]
    assert p["fits"] and st["mode"] == "rolling" and st["family"] == "kimi_linear" and st["chunks"] == 1
    assert st["window"] >= 1 and st["stream_mode"] == "sweep_family" and "refused" in st["carve"]
    too_wide = P.make_plan(_moe_info(92, "kimi_linear"), cfg, texts=46000, window=500)
    assert not too_wide["fits"] and "window" in too_wide["refusals"][0]
    fat = P.make_plan(_moe_info(92 * 30, "kimi_linear"), cfg, texts=46000)  # one layer alone over the cap
    assert not fat["fits"] and "3 streamed layers" in fat["refusals"][0]
    other = P.make_plan(_moe_info(92, "mimo_v2"), cfg, texts=46000)          # another sweep family: unchanged
    assert other["stream"]["window"] == 0 and other["stream"]["family"] == "mimo_v2"
    # registered by model_type, so an artifact of another adapter (Bolt, Nemotron 3 Super) refuses at plan time
    bolt = {**_moe_info(92, "nemotron_h"), "adapter": "Nemotron35BoltAdapter"}
    refused = P.make_plan(bolt, cfg, texts=46000, moe_units="expert")
    assert not refused["fits"] and "Nemotron35BoltAdapter" in refused["refusals"][0]
    hub48 = {**_moe_info(92, "kimi_linear"), "adapter": None}                 # no adapter: the 48B hub BF16
    assert P.make_plan(hub48, cfg, texts=46000)["fits"]
    neuron = P.make_plan(_moe_info(92, "nemotron_h"), cfg, texts=46000)      # default neuron units: refused
    assert not neuron["fits"] and "--moe-units expert" in neuron["refusals"][0]


def test_inspection_counts_hybrid_layers_from_the_block_types():
    from mlx2.experimental.amoeba import inspection as I
    types = ["mamba", "moe", "mamba", "attention", "moe", "mlp"]
    cfg = I._text_config({"model_type": "nemotron_h_omni", "llm_config": {"model_type": "nemotron_h",
                                                                          "layers_block_type": types}})
    assert cfg["num_hidden_layers"] == 6 and cfg["model_type"] == "nemotron_h"
    moe = I._moe_layers(cfg, 6, 8)
    assert moe == [False, True, False, False, True, False]
    assert [I._dense_layer(cfg, l, moe) for l in range(6)] == [False] * 5 + [True]
    pat = {"hybrid_override_pattern": "ME*E-"}
    assert I._moe_layers(pat, 5, 8) == [False, True, False, True, False]
    assert I._dense_layer(pat, 4, I._moe_layers(pat, 5, 8)) and not I._dense_layer(pat, 0, I._moe_layers(pat, 5, 8))


@pytest.mark.parametrize("mode", ["family", "streamed_only", "foreign_family", "foreign_resident",
                                  "streamed_not_bitwise"])
def test_rolling_parity_runs_a_whole_layer_family(tmp_path, monkeypatch, mode):
    from mlx2.experimental.amoeba import cli, runner
    from mlx2.experimental.amoeba import plan as P
    monkeypatch.setattr(P, "physical_gb", lambda: 128.0)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"id":"A"}\n{"id":"B"}\n')
    out = tmp_path / "out"
    streamed = mode in ("streamed_only", "streamed_not_bitwise")
    info = _moe_info(92 if streamed else 50, "mimo_v2" if mode.startswith("foreign") else "kimi_linear")
    monkeypatch.setattr(cli, "inspect_checkpoint", lambda _: info)
    monkeypatch.setattr(cli, "check_commit", lambda _: {"commit": "test"})
    cmds = []

    def capture(cmd, **kw):
        cmds.append([str(c) for c in cmd])
        dest = Path(cmd[cmd.index("--out") + 1])
        dest.mkdir()
        items = [{"id": "A"}, {"id": "B"}]
        (dest / "items.json").write_text(json.dumps(items))
        meta = {"layers": 1, "sequences": 2, "widths": [1], "unit_kinds": ["dense"], "rolling": {"window": 1},
                "parity": {"ok": True, "texts": [{"bitwise_equal": mode != "streamed_not_bitwise"}]},
                "fingerprint": info["fingerprint"],
                "weight_reads_observed": {"once_per_chunk": True, "layer_loads": 2}}
        (dest / "meta.json").write_text(json.dumps(meta))
        np.save(dest / "resp_L00.npy", np.zeros((2, 1), np.float16))
        np.save(dest / "mu.npy", np.ones((1, 1), np.float32))
        np.save(dest / "nll_full.npy", np.ones(2, np.float32))

    monkeypatch.setattr(runner, "run_child", capture)
    args = ["rolling-parity", "--model", "toy", "--out", str(out), "--cpu", "--probe-corpus", str(corpus),
            "--swap-growth-gb", "2", "--swap-waiver", "night window: the standard 2 GB guard"]
    monkeypatch.setattr(runner, "swap_used_gb", lambda: 1.0)
    if streamed:
        args.append("--streamed-only")
    if mode == "foreign_family":
        with pytest.raises(SystemExit, match="its own parity"):
            cli.main(args)
        return
    if mode == "foreign_resident":            # --rolling-resident: the host family on resident weights, as before
        assert cli.main(args + ["--rolling-resident"]) == 0
        rolling = [c for c in cmds if "mlx2.experimental.amoeba.child_rolling" in c][0]
        assert rolling[rolling.index("--family") + 1] == "host" and "--resident" in rolling
        return
    if mode == "streamed_not_bitwise":        # same streamed tree on both sides: within-tolerance is not exact
        assert cli.main(args) == 3
        assert not json.loads((out / "parity.json").read_text())["exact"]
        return
    assert cli.main(args) == 0
    rolling = [c for c in cmds if "mlx2.experimental.amoeba.child_rolling" in c]
    assert len(rolling) == 1 and rolling[0][rolling[0].index("--family") + 1] == "kimi_linear"
    assert "--stream-mode" not in rolling[0] and rolling[0][rolling[0].index("--parity-layers") + 1] == "0"
    resident = [c for c in cmds if "mlx2.experimental.amoeba.child_capture" in c]
    assert len(resident) == (0 if mode == "streamed_only" else 1)
    res = json.loads((out / "parity.json").read_text())
    assert res["family"] == "kimi_linear" and res["exact"] and res["swap_guard"]["growth_gb"] == 2.0
    if mode == "streamed_only":
        assert res["mode"] == "streamed-only"


def test_rolling_parity_needs_streamed_only_when_the_model_does_not_fit(tmp_path, monkeypatch):
    from mlx2.experimental.amoeba import cli
    from mlx2.experimental.amoeba import plan as P
    monkeypatch.setattr(P, "physical_gb", lambda: 128.0)
    monkeypatch.setattr(cli, "inspect_checkpoint", lambda _: _moe_info(92, "kimi_linear"))
    monkeypatch.setattr(cli, "check_commit", lambda _: {"commit": "test"})
    with pytest.raises(SystemExit, match="streamed-only"):
        cli.main(["rolling-parity", "--model", "toy", "--out", str(tmp_path / "o"), "--cpu"])


# ------------------------------------------------------------------ K2-V2: a dense llama through --family host

QWEN06 = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"


@pytest.mark.skipif(not (QWEN06 / "tokenizer.json").exists(), reason="a tokenizer to borrow (Qwen3-0.6B) not on disk")
def test_llama_dense_streams_through_the_host_family(tmp_path):
    """K2-V2 is model_type llama on StandardDecoderAdapter, which accepts a dense_layers streaming request: the host
    family streams it with no family of its own. A tiny llama in the same layout, streamed vs resident, bitwise."""
    import shutil
    from mlx2.runtime.models.standard_decoder import Model, ModelArgs
    from research.amoeba.sweep.host_family import HostFamily
    cfg = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 64, "intermediate_size": 128,
           "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "num_hidden_layers": 4,
           "vocab_size": 151936, "max_position_embeddings": 4096, "rms_norm_eps": 1e-5, "rope_theta": 10000000.0,
           "tie_word_embeddings": False, "attention_bias": False, "mlp_bias": False, "hidden_act": "silu",
           "rope_scaling": None, "torch_dtype": "bfloat16"}
    m = Model(ModelArgs.from_dict(cfg))
    _randomize(m, 11)
    ck = tmp_path / "k2-tiny"
    ck.mkdir()
    mx.save_safetensors(str(ck / "model.safetensors"), dict(tree_flatten(m.parameters())))
    (ck / "config.json").write_text(json.dumps(cfg))
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        shutil.copy(QWEN06 / f, ck / f)
    host, stream = R.load_streamed_host(ck, stream_mode="dense_layers", ceiling_bytes=1 << 20, resident=False,
                                        moe_units="neuron", log=lambda *_: None)
    fam = HostFamily(host, stream=stream, model_path=ck, stream_mode="dense_layers")
    assert len(fam._blocks) == cfg["num_hidden_layers"] - 1 and fam.prefetch_ranges     # layer 0 resident
    rhost, _ = R.load_streamed_host(ck, stream_mode="dense_layers", ceiling_bytes=0, resident=True,
                                    moe_units="neuron", log=lambda *_: None)
    ref = HostFamily(rhost)
    seqs = [[int(x) for x in np.random.default_rng(i).integers(0, 1000, 12)] for i in range(5)]
    fit = [True, False, True, True, False]
    (tmp_path / "s").mkdir()
    (tmp_path / "r").mkdir()
    before = stream.stats.page_ins
    R.run(fam, seqs, fit, tmp_path / "s", window=2, chunks=1, max_texts=2, log=lambda *_: None)
    assert stream.stats.page_ins - before == len(fam._blocks)
    R.run(ref, seqs, fit, tmp_path / "r", window=2, chunks=1, max_texts=2, log=lambda *_: None)
    _same_captures(tmp_path / "s", tmp_path / "r", cfg["num_hidden_layers"])
    assert R.parity(fam, seqs[:3], None, log=lambda *_: None)["ok"]
