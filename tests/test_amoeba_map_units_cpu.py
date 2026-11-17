"""Unit-site coverage of the aMoEba carve host for every AR text family we map (CPU only).

Each test builds a tiny random-weight instance from the REAL mlx2 model classes, with the real
artifact's config.json shrunk (few layers, small widths, few experts; layer-type patterns such as
hybrid Mamba/attention/MoE interleaving and dense first layers kept), then checks:

  sites     the kind per layer and the width (config intermediate size, or expert count per layer)
  engage    every site runs on a forward and every MoE site's re-implemented routing reproduces
            its block (Host.check_site_parity)
  ones      the masked forward with an all-ones core equals the base forward
  drop      one dropped expert equals zeroing that expert's down projection (no renormalisation),
            and one dropped neuron equals zeroing that down_proj input column

Nothing here loads a real checkpoint: configs are read (skipped when the artifact is absent),
weights are random. The LFM2.5-24B-A2B config is inlined (that checkpoint lives on the M3 only).
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from research.amoeba.carve import units as U  # noqa: E402
from research.amoeba.carve.host import Host  # noqa: E402

HOME = Path.home()
MODELS = HOME / "mlx-models"
T7 = Path.home() / "Models"
HF = HOME / ".cache/huggingface/hub"
V = 256


@pytest.fixture(autouse=True)
def _cpu():
    mx.set_default_device(mx.cpu)


class Tok:
    def encode(self, text, add_special_tokens=False):
        return [b % V for b in text.encode()]

    def decode(self, ids):
        return "".join(chr(97 + i % 26) for i in ids)


def _config(path):
    path = Path(path)
    if path.is_dir():
        path = path / "config.json"
    if not path.exists():
        pytest.skip(f"{path} not on disk")
    cfg = json.loads(path.read_text())
    for k in ("quantization", "quantization_config"):
        cfg.pop(k, None)
    return cfg


def _hf(repo):
    snaps = sorted((HF / f"models--{repo.replace('/', '--')}" / "snapshots").glob("*/config.json"))
    if not snaps:
        pytest.skip(f"{repo} not in the HF cache")
    return snaps[-1]


def _ids(seed=0, n=2, T=10):
    return mx.array(np.random.default_rng(seed).integers(0, V, (n, T)).astype(np.int32))


def _host(m):
    return Host(model=m, tokenizer=Tok(), fingerprint="tiny:00000000", log=lambda *_: None)


def _set(module, name, fn):
    w = np.array(getattr(module, name).astype(mx.float32))
    fn(w)
    setattr(module, name, mx.array(w).astype(getattr(module, name).dtype))


def _down(site):
    """(projection whose input is a unit, expert axis?) for zeroing: down_proj / fc2 of a switch."""
    if site.kind == "dense":
        return site.module
    sw = U._switch(site.module, site.kind)
    return sw.down_proj if getattr(sw, "down_proj", None) is not None else sw.fc2


def randomize_zeros(m, seed=0):
    """Routers some classes initialise to zeros (Nemotron, Xing) get random weights, so routing
    actually varies across tokens; loaded checkpoints overwrite them anyway."""
    rng = np.random.default_rng(seed + 1)
    upd = []
    for k, v in nn.utils.tree_flatten(m.parameters()):
        if k.endswith("weight") and v.dtype in (mx.float32, mx.bfloat16, mx.float16) and v.size and \
                not bool(mx.any(v != 0).item()):
            upd.append((k, mx.array(rng.normal(0, 0.5, v.shape).astype(np.float32)).astype(v.dtype)))
    if upd:
        m.load_weights(upd, strict=False)
    mx.eval(m.parameters())
    return m


def check(build, kinds, widths, *, tol=1e-4, parity_tol=1e-4, drop=True, seed=0):
    """``build(seed)`` -> a fresh model; asserts sites, engagement, all-ones parity and drop rules."""
    m = build(seed)
    h = _host(m)
    assert [s.kind for s in h.sites] == kinds
    assert h.widths == widths
    ids = _ids()
    sp = h.check_site_parity(ids, tol=parity_tol if parity_tol > 2e-2 else 2e-2)
    assert sp["ok"], sp
    par = h.check_ones_parity(ids)
    assert par["top1_agree"] == 1.0 and par["max_abs"] < tol, par
    if not drop:
        return h
    for want in ("moe", "dense"):
        site = next((s for s in h.sites if s.kind is not None and (s.is_moe == (want == "moe"))), None)
        if site is None:
            continue
        unit = site.n // 2
        if site.is_moe:                   # an expert the probe actually routes to
            seen = []
            h.ctl.reset()
            h.ctl.stats = lambda l, k, p, _l=site.layer: l == _l and seen.append(np.array(p["inds"]).ravel())
            mx.eval(h.logits_at(ids))
            unit = int(np.bincount(np.concatenate(seen), minlength=site.n).argmax())
        h.ctl.reset()
        core = [mx.ones((n,)) for n in h.widths]
        core[site.layer] = mx.array((np.arange(site.n) != unit).astype(np.float32))
        h.ctl.core = core
        masked = h.logits_at(ids)
        h.ctl.reset()
        m2 = build(seed)
        h2 = _host(m2)
        dp = _down(h2.sites[site.layer])
        if site.is_moe:
            _set(dp, "weight", lambda w: w.__setitem__(unit, 0.0))
            if "bias" in dp:
                _set(dp, "bias", lambda b: b.__setitem__(unit, 0.0))
        else:
            _set(dp, "weight", lambda w: w.__setitem__((slice(None), unit), 0.0))
        zeroed = h2.logits_at(ids)
        diff = float(mx.abs(masked - zeroed).max())
        assert diff < tol, (want, site.layer, diff)
        assert float(mx.abs(masked - h.logits_at(ids)).max()) > 0, "the drop changed nothing"
    return h


def _quantize(m, bits=8, group_size=64, mode="affine", pred=None):
    def ok(path, mod):
        if not hasattr(mod, "to_quantized") or getattr(mod, "weight", None) is None:
            return False
        if mod.weight.shape[-1] % group_size:
            return False
        return pred(path, mod) if pred else True
    nn.quantize(m, group_size=group_size, bits=bits, mode=mode, class_predicate=ok)
    mx.eval(m.parameters())
    return m


# --------------------------------------------------------------------------- Nemotron-H family


def _nemotron_args(cfg, pattern, **over):
    from mlx2.runtime.models.nemotron_h import ModelArgs
    cfg = dict(cfg)
    cfg.update(hidden_size=64, intermediate_size=96, moe_intermediate_size=64, moe_shared_expert_intermediate_size=64,
               n_routed_experts=8, num_experts_per_tok=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
               mamba_num_heads=4, mamba_head_dim=16, ssm_state_size=16, n_groups=2, vocab_size=V,
               num_nextn_predict_layers=0, mtp_layers_block_type=None, mtp_hybrid_override_pattern=None,
               time_step_limit=(0.0, float("inf")))
    cfg.update(over)
    cfg["layers_block_type"] = pattern
    cfg["num_hidden_layers"] = len(pattern)
    cfg.pop("hybrid_override_pattern", None)
    return ModelArgs.from_dict(cfg)


PATTERN = ["mamba", "moe", "mamba", "attention", "moe", "mlp"]
KINDS = [None, "nemotron", None, None, "nemotron", "dense"]


def test_nemotron_lightning_hybrid_sites():
    """Lightning-30B-A3B: Mamba / attention / MoE (no latent); Mamba and attention get no site."""
    from mlx2.runtime.models.nemotron_h import Model
    cfg = _config(MODELS / "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-mlx-8Bit")
    real = cfg["layers_block_type"]
    assert set(real) == {"mamba", "moe", "attention"} and cfg["moe_latent_size"] is None

    def build(seed):
        mx.random.seed(seed)
        m = randomize_zeros(Model(_nemotron_args(cfg, real[:7])))
        mx.eval(m.parameters())
        return m
    kinds = [None if t in ("mamba", "attention") else "nemotron" for t in real[:7]]
    check(build, kinds, [8 if k else 0 for k in kinds])

    def build_dense(seed):        # a '-' MLP block in the same slot: a dense site of intermediate_size
        mx.random.seed(seed)
        m = randomize_zeros(Model(_nemotron_args(cfg, PATTERN)))
        mx.eval(m.parameters())
        return m
    check(build_dense, KINDS, [0, 8, 0, 0, 8, 96])


def test_nemotron_lightning_affine_q8_widths():
    from mlx2.runtime.models.nemotron_h import Model
    cfg = _config(MODELS / "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-mlx-8Bit")
    mx.random.seed(0)
    m = _quantize(randomize_zeros(Model(_nemotron_args(cfg, PATTERN))), bits=8, group_size=32)
    h = _host(m)
    assert h.widths == [0, 8, 0, 0, 8, 96]
    assert h.sites[1].params_per_unit == 2 * 64 * 64 and h.sites[5].params_per_unit == 2 * 64
    assert h.check_site_parity(_ids())["ok"]


def _super_cfg():
    cfg = _config(MODELS / "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4")
    return cfg["llm_config"]


def test_nemotron_super_latent_sites():
    """Super-VL-120B: latent fc1/fc2 around the routed experts, grouped sigmoid router."""
    from mlx2.runtime.models.nemotron_h import Model
    cfg = _super_cfg()
    assert cfg["moe_latent_size"] and set(cfg["layers_block_type"]) == {"mamba", "moe", "attention"}
    real = cfg["layers_block_type"]
    pat = real[:8]                                    # M E M E M E M *

    def build(seed):
        mx.random.seed(seed)
        m = randomize_zeros(Model(_nemotron_args(cfg, pat, moe_latent_size=64)))
        mx.eval(m.parameters())
        return m
    kinds = [None if t in ("mamba", "attention") else "nemotron" for t in pat]
    h = check(build, kinds, [8 if k else 0 for k in kinds])
    assert h.sites[1].module.fc1_latent_proj is not None


def test_nemotron_super_nvfp4_folded_scales():
    """The NVFP4 export's experts (ScaledNVFP4SwitchLinear, weight_scale_2 folded into the routing
    weights) and shared experts (ScaledNVFP4Linear): widths, and the fold rides in _routing_weights."""
    from mlx2.runtime import nvfp4_linear as N
    from mlx2.runtime.models.nemotron_h import Model
    cfg = _super_cfg()
    pat = ["mamba", "moe", "attention", "moe"]
    mx.random.seed(0)
    m = randomize_zeros(Model(_nemotron_args(cfg, pat, moe_latent_size=64)))
    mx.eval(m.parameters())
    paths = []
    for li in (1, 3):
        mix = m.backbone.layers[li].mixer
        for name in ("fc1", "fc2"):
            paths.append(f"backbone.layers.{li}.mixer.switch_mlp.{name}")
        for name in ("up_proj", "down_proj"):
            paths.append(f"backbone.layers.{li}.mixer.shared_experts.{name}")
    floats = {p: dict(m.named_modules())[p].weight for p in paths}
    N.install_weight_scale2(m, paths)
    rng = np.random.default_rng(0)
    for p, w in floats.items():
        mod = dict(m.named_modules())[p]
        q, s = mx.quantize(w, group_size=16, bits=4, mode="nvfp4")
        mod.weight, mod.scales = q, s
        sc = mod.weight_scale_2
        mod.weight_scale_2 = mx.array(rng.uniform(0.5, 2.0, sc.shape).astype(np.float32)) if sc.ndim else mx.array(1.5)
    m.set_dtype(mx.bfloat16)
    n = N.fold_relu2_expert_scales([m.backbone.layers[li].mixer.switch_mlp for li in (1, 3)], mx.bfloat16)
    assert n == 2
    h = _host(m)
    assert [s.kind for s in h.sites] == [None, "nemotron", None, "nemotron"] and h.widths == [0, 8, 0, 8]
    assert h.sites[1].params_per_unit == 2 * 64 * 64
    sp = h.check_site_parity(_ids(), tol=2e-2)
    assert sp["ok"], sp


def test_nemotron_bolt_ragged_experts():
    """Bolt-74B-A9B (Puzzle-NAS): per-layer expert counts and widths from block_configs."""
    from mlx2.conversion.nemotron35_bolt_plan import language_config
    from mlx2.runtime.models.nemotron35_bolt import Model
    path = MODELS / "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB/config.json"
    if not path.exists():
        pytest.skip("Bolt config not on disk")
    text = language_config(json.loads(path.read_text()))
    real_counts = sorted({b["n_routed_experts"] for b in text["block_configs"] if b["block_type"] == "moe"})
    assert len(real_counts) > 1                        # ragged in the real artifact too
    pat = text["layers_block_type"][:4]                # M E M E
    assert pat == ["mamba", "moe", "mamba", "moe"]
    blocks = [{"block_type": t} for t in pat]
    blocks[1].update(n_routed_experts=6, num_experts_per_tok=2, moe_intermediate_size=64)
    blocks[3].update(n_routed_experts=10, num_experts_per_tok=3, moe_intermediate_size=128)

    def build(seed):
        mx.random.seed(seed)
        m = randomize_zeros(Model(_nemotron_args(text, pat, moe_latent_size=64, block_configs=copy.deepcopy(blocks))))
        mx.eval(m.parameters())
        return m
    h = check(build, [None, "nemotron", None, "nemotron"], [0, 6, 0, 10])
    assert h.N is None and h.sites[3].params_per_unit == 2 * 64 * 128


# --------------------------------------------------------------------------- Qwen MoE family

QWEN_TINY = dict(hidden_size=128, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                 linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
                 num_experts=8, num_experts_per_tok=2, moe_intermediate_size=64, shared_expert_intermediate_size=64,
                 vocab_size=V, mtp_num_hidden_layers=0, num_hidden_layers=4, tie_word_embeddings=False)


def _qwen_cfg(path, **over):
    cfg = _config(path)
    t = dict(cfg["text_config"])
    t.update(QWEN_TINY)
    t.update(over)
    t["layer_types"] = t["layer_types"][:4]
    cfg["text_config"] = t
    return cfg


def test_qwen4_exp_flash_next_is_qwen_next():
    """Flash-Next (qwen4_exp): every layer is Qwen3NextSparseMoeBlock, so the qwen_next kind."""
    from mlx2.runtime.models.qwen4_exp import Model, ModelArgs
    cfg = _qwen_cfg(MODELS / "Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP", hc_lowrank=16, ple_embed_dim=32,
                    ple_layer_ids=[2], ngram_vocab_size_base=1000, split_ngram_parts=4, indexer_n_heads=2,
                    indexer_head_dim=32, indexer_budget=16, mtp={"num_hidden_layers": 0})

    def build(seed):
        mx.random.seed(seed)
        m = randomize_zeros(Model(ModelArgs.from_dict(cfg)))
        return m
    check(build, ["qwen_next"] * 4, [8] * 4)


def _qwen36(path, **over):
    from mlx2.runtime.models.qwen36_35b import Model, ModelArgs
    cfg = _qwen_cfg(path, **over)

    def build(seed):
        mx.random.seed(seed)
        return randomize_zeros(Model(ModelArgs.from_dict(cfg)))
    return build


QWEN36 = [MODELS / "Sage-A1.5-MLX-Mixed-Q4-from-GGUF", MODELS / "AX-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP8-MTP",
          MODELS / "Qwen3.6-35B-A3B-Abliterated-Heretic-MLX-4bit"]


@pytest.mark.parametrize("path", QWEN36, ids=lambda p: p.name)
def test_qwen3_5_moe_sites(path):
    check(_qwen36(path), ["qwen_next"] * 4, [8] * 4)


@pytest.mark.parametrize("mode,bits,gs", [("affine", 4, 64), ("affine", 8, 64), ("mxfp8", 8, 32), ("mxfp4", 4, 32),
                                          ("nvfp4", 4, 16)])
def test_qwen3_5_moe_quantized_widths(mode, bits, gs):
    """Sage mixed q4/q8, Heretic q4, AX-Tiel mxfp8 (and the mx/nv fp4 formats other exports use):
    expert counts and per-expert params from the quantized switch, routing parity holds."""
    m = _quantize(_qwen36(QWEN36[0])(0), bits=bits, group_size=gs, mode=mode)
    h = _host(m)
    assert h.widths == [8] * 4
    assert all(s.params_per_unit == 3 * 128 * 64 for s in h.sites), [s.params_per_unit for s in h.sites]
    sp = h.check_site_parity(_ids())
    assert sp["ok"], sp


def _exl3_switch(sw, rng):
    """An Exl3SwitchGLU of sw's geometry (random trellis words; geometry, not values, is under test)."""
    from mlx2.runtime.models.exl3 import Exl3Linear, Exl3SwitchGLU
    down = sw.down_proj
    E, hid, d = down.num_experts, down.input_dims, down.output_dims

    def lin(i, o):
        t = mx.array(rng.integers(0, 2**16, (E, i // 16, o // 16, 32)).astype(np.uint16))
        return Exl3Linear(t, mx.ones((E, i), mx.float16), mx.ones((E, o), mx.float16))
    return Exl3SwitchGLU(lin(d, hid), lin(d, hid), lin(hid, d), execution="reference")


def test_qwen3_5_moe_exl3_widths():
    """Qwen3.6-35B-A3B exl3 4.00bpw: routed banks are Exl3SwitchGLU (trellis, no ``weight``); the
    width and per-expert params come from the projections' input/output dims."""
    path = _hf("UnstableLlama/Qwen3.6-35B-A3B-exl3-4.00bpw")
    m = _qwen36(path.parent, moe_intermediate_size=128)(0)
    rng = np.random.default_rng(0)
    for layer in m.language_model.model.layers:
        layer.mlp.switch_mlp = _exl3_switch(layer.mlp.switch_mlp, rng)
    sites, _ = U.install(list(m.language_model.model.layers))
    assert [(s.kind, s.n) for s in sites] == [("qwen_next", 8)] * 4
    assert all(s.params_per_unit == 3 * 128 * 128 for s in sites)


@pytest.mark.parametrize("cls", ["ScaledNVFP4Linear", "ScaledFP8Linear", "GlobalScaledNVFP4Linear"])
def test_dense_width_owned_microscaling_modules(cls):
    """Owned NVFP4 / FP8 dense modules carry packed uint32 weights but no ``bits``: the dense width
    is the unpacked down_proj input (96), not the packed column count."""
    from mlx2.runtime import nvfp4_linear as N
    from mlx2.runtime.models.nemotron_h import Model
    cfg = _config(MODELS / "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-mlx-8Bit")
    mx.random.seed(0)
    m = randomize_zeros(Model(_nemotron_args(cfg, PATTERN)))
    mlp = m.backbone.layers[5].mixer
    for name in ("up_proj", "down_proj"):
        o, i = getattr(mlp, name).weight.shape
        mod = getattr(N, cls)(i, o)
        mode, gs = ("mxfp8", 32) if cls == "ScaledFP8Linear" else ("nvfp4", 16)
        q, s = mx.quantize(getattr(mlp, name).weight, group_size=gs, bits=8 if mode == "mxfp8" else 4, mode=mode)
        mod.weight, mod.scales = q, s
        setattr(mlp, name, mod)
    h = _host(m)
    assert h.widths == [0, 8, 0, 0, 8, 96] and h.sites[5].params_per_unit == 2 * 64
    assert h.check_site_parity(_ids())["ok"]


# --------------------------------------------------------------------------- other MoE families


def _builder(model_cls, args):
    def build(seed):
        mx.random.seed(seed)
        return randomize_zeros(model_cls(args))
    return build


def test_gpt_oss_router_experts_mxfp4():
    """gpt-oss-20b: router(x) -> top-k logits -> softmax over the k -> experts (SwiGLU, biases);
    the checkpoint's experts are mxfp4."""
    from mlx2.runtime.models.gpt_oss import Model, ModelArgs
    cfg = _config(_hf("openai/gpt-oss-20b"))
    cfg.update(num_hidden_layers=4, num_local_experts=8, num_experts_per_tok=2, hidden_size=64,
               intermediate_size=64, head_dim=16, num_attention_heads=4, num_key_value_heads=2, vocab_size=V,
               sliding_window=4, layer_types=cfg["layer_types"][:4])
    args = ModelArgs.from_dict(cfg)
    check(_builder(Model, args), ["router_experts"] * 4, [8] * 4)
    m = _quantize(_builder(Model, args)(0), bits=4, group_size=32, mode="mxfp4",
                  pred=lambda p, mod: ".experts." in p)
    h = _host(m)
    assert h.widths == [8] * 4 and h.sites[0].params_per_unit == 3 * 64 * 64
    assert h.check_site_parity(_ids())["ok"]


def test_mellum_softmax_topk():
    """Mellum2.1-12B-A2.5B: softmax(gate) -> top_k -> / sum -> switch_mlp, no shared expert."""
    from mlx2.runtime.models.mellum import Model, ModelArgs
    cfg = _config(_hf("JetBrains/Mellum2.1-12B-A2.5B-Thinking"))
    cfg.update(num_hidden_layers=4, num_experts=8, num_experts_per_tok=2, hidden_size=64, intermediate_size=96,
               moe_intermediate_size=32, head_dim=16, num_attention_heads=4, num_key_value_heads=2, vocab_size=V,
               sliding_window=4, layer_types=cfg["layer_types"][:4], num_mtp_layers=0, mtp_layer_types=[],
               mtp_mlp_layer_types=[])
    check(_builder(Model, ModelArgs.from_dict(cfg)), ["softmax_topk"] * 4, [8] * 4)


def test_flint_sigmoid_shared_and_dense_first_layers_mixed_affine():
    """flint-2a247be0-oq4e: mlp_only_layers [0, 1] are dense, then sigmoid-routed MoE with an
    ungated shared expert; the oQ4e export mixes affine 4/5/6-bit projections."""
    from mlx2.runtime.models.flint import Model, ModelArgs
    cfg = _config(MODELS / "flint-2a247be0-oq4e")
    assert cfg["mlp_only_layers"] == [0, 1]
    cfg.update(num_hidden_layers=4, num_experts=8, num_experts_per_tok=2, hidden_size=64, intermediate_size=128,
               moe_intermediate_size=64, shared_expert_intermediate_size=64, head_dim=16, num_attention_heads=4,
               num_key_value_heads=2, vocab_size=V, sliding_window=4, layer_types=cfg["layer_types"][:4],
               no_rope_layer_indices=[i for i in cfg["no_rope_layer_indices"] if i < 4])
    args = ModelArgs.from_dict(cfg)
    kinds = ["dense", "dense", "sigmoid_shared", "sigmoid_shared"]
    check(_builder(Model, args), kinds, [128, 128, 8, 8])
    bits = {}

    def mixed(path, mod):            # oQ4e: per-module 4/5/6-bit affine, as in the export's overrides
        bits[path] = (4, 5, 6)[len(bits) % 3]
        return True
    m = _builder(Model, args)(0)
    nn.quantize(m, group_size=64, bits=4, class_predicate=lambda p, mod: hasattr(mod, "to_quantized") and
                getattr(mod, "weight", None) is not None and mod.weight.shape[-1] % 64 == 0 and mixed(p, mod) and
                {"group_size": 64, "bits": bits[p], "mode": "affine"})
    mx.eval(m.parameters())
    assert {getattr(l.mlp.down_proj, "bits", None) for l in m.model.layers[:2]} <= {4, 5, 6}
    h = _host(m)
    assert h.widths == [128, 128, 8, 8] and h.sites[0].params_per_unit == 3 * 64
    assert h.check_site_parity(_ids())["ok"]


def test_xing4_0_ling_shape_with_dense_first_layers():
    """Xing4.0-29B-A4B: first_k_dense_replace = 2 dense layers, then sigmoid+bias router with a
    shared expert (the ling block shape), under the mHC hyper-connection streams."""
    from mlx2.runtime.models.xing4_0 import Model, ModelArgs
    cfg = _config(T7 / "Xing4.0-29B-A4B")
    assert cfg["first_k_dense_replace"] == 2 and cfg["n_shared_experts"] == 1
    cfg.update(num_hidden_layers=4, n_routed_experts=8, num_experts_per_tok=2, hidden_size=64, intermediate_size=96,
               moe_intermediate_size=32, num_attention_heads=4, num_key_value_heads=4, vocab_size=V,
               kv_lora_rank=32, q_lora_rank=48, qk_rope_head_dim=16, v_head_dim=16, qk_nope_head_dim=16,
               num_nextn_predict_layers=0, n_group=1, topk_group=1)
    check(_builder(Model, ModelArgs.from_dict(cfg)), ["dense", "dense", "ling", "ling"], [96, 96, 8, 8])


def test_lfm2_moe_routing_method_and_dense_first_layers():
    """LFM2.5-24B-A2B (config read from the M3 checkpoint's config.json, inlined): conv/attention
    hybrid, num_dense_layers = 2 dense layers, then sigmoid+expert-bias routing() MoE."""
    from mlx2.runtime.models.lfm25_moe import Model, ModelArgs
    cfg = dict(LFM25_24B)
    cfg.update(num_hidden_layers=4, num_experts=8, num_experts_per_tok=2, hidden_size=64, intermediate_size=96,
               moe_intermediate_size=32, num_attention_heads=4, num_key_value_heads=2, vocab_size=V,
               layer_types=cfg["layer_types"][:4])
    assert cfg["layer_types"] == ["conv", "conv", "full_attention", "conv"]
    check(_builder(Model, ModelArgs.from_dict(cfg)), ["dense", "dense", "routing", "routing"], [96, 96, 8, 8])


# ssh m3 cat ~/mlx-models/LFM2.5-24B-A2B-MLX-8bit/config.json (2026-11-15; quantization: affine 8-bit g64)
LFM25_24B = {
    "architectures": ["Lfm2MoeForCausalLM"], "conv_L_cache": 3, "conv_bias": False, "hidden_size": 2048,
    "intermediate_size": 11776,
    "layer_types": ["conv", "conv", "full_attention", "conv", "conv", "conv", "full_attention", "conv", "conv",
                    "conv", "full_attention", "conv", "conv", "conv", "full_attention", "conv", "conv", "conv",
                    "full_attention", "conv", "conv", "conv", "full_attention", "conv", "conv", "conv",
                    "full_attention", "conv", "conv", "conv", "full_attention", "conv", "conv", "conv",
                    "full_attention", "conv", "conv", "conv", "full_attention", "conv"],
    "max_position_embeddings": 131072, "model_type": "lfm2_moe", "moe_intermediate_size": 1536, "norm_eps": 1e-05,
    "norm_topk_prob": True, "num_attention_heads": 32, "num_dense_layers": 2, "num_experts": 64,
    "num_experts_per_tok": 4, "num_hidden_layers": 40, "num_key_value_heads": 8,
    "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"}, "routed_scaling_factor": 1.0,
    "tie_word_embeddings": True, "use_expert_bias": True, "vocab_size": 128000,
}


# --------------------------------------------------------------------------- re-verified families


def _gemma4(path, moe):
    from mlx2.runtime.models.gemma4 import Model, ModelConfig
    cfg = _config(path)
    t = dict(cfg["text_config"])
    assert bool(t["enable_moe_block"]) == moe
    t.update(hidden_size=64, intermediate_size=96, num_attention_heads=4, num_key_value_heads=2,
             num_global_key_value_heads=1, head_dim=16, global_head_dim=32, vocab_size=V, vocab_size_per_layer_input=V,
             num_hidden_layers=6, sliding_window=4, layer_types=t["layer_types"][:6])
    if moe:
        t.update(num_experts=8, top_k_experts=2, moe_intermediate_size=32)
    cfg["text_config"] = t
    cfg["vision_config"] = None
    return _builder(Model, ModelConfig.from_dict(cfg))


@pytest.mark.parametrize("path", [T7 / "gemma-4-31B-it", MODELS / "Artemis-31B-v1.2-q8-mlx2"], ids=lambda p: p.name)
def test_gemma4_dense(path):
    check(_gemma4(path, False), ["dense"] * 6, [96] * 6)


def test_gemma4_26b_experts_beside_parallel_dense_mlp():
    check(_gemma4(MODELS / "gemma-4-26B-A4B-it-MLX-8bit", True), ["gemma4_experts"] * 6, [8] * 6)


def test_laguna_xs21_dense_first_layer():
    from mlx2.runtime.models.laguna import Model, ModelArgs
    cfg = _config(HOME / "models/Laguna-XS-2.1")
    assert cfg["mlp_only_layers"] == [0]
    n = 4
    cfg.update(num_hidden_layers=n, hidden_size=64, intermediate_size=96, num_attention_heads=4, num_key_value_heads=2,
               head_dim=16, vocab_size=V, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
               shared_expert_intermediate_size=32, sliding_window=4,
               **{k: cfg[k][:n] for k in ("layer_types", "mlp_layer_types", "gating_types") if k in cfg})
    cfg["num_attention_heads_per_layer"] = [4] * n
    check(_builder(Model, ModelArgs.from_dict(cfg)), ["dense", "laguna", "laguna", "laguna"], [96, 8, 8, 8])


def test_kimi_linear_ling_with_dense_first_layer():
    from mlx2.runtime.models.kimi_linear import Model, ModelArgs
    cfg = _config(T7 / "Kimi-Linear-48B-A3B-Instruct")
    assert cfg["first_k_dense_replace"] == 1
    lac = dict(cfg["linear_attn_config"])
    lac.update(full_attn_layers=[4], kda_layers=[1, 2, 3], head_dim=16, num_heads=4)
    cfg.update(num_hidden_layers=4, hidden_size=64, intermediate_size=96, num_attention_heads=4, num_key_value_heads=4,
               head_dim=16, vocab_size=V, num_experts=8, num_experts_per_token=2, moe_intermediate_size=32,
               kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=16, v_head_dim=16, linear_attn_config=lac)
    check(_builder(Model, ModelArgs.from_dict(cfg)), ["dense", "ling", "ling", "ling"], [96, 8, 8, 8])


@pytest.mark.parametrize("path", [HOME / "models/granite-4.2-30b", T7 / "K2-V2-bf16"], ids=lambda p: p.name)
def test_standard_decoder_dense(path):
    """granite-4.2-30b (granite) and K2-V2 (llama) through standard_decoder."""
    from mlx2.runtime.models.standard_decoder import Model, ModelArgs
    cfg = _config(path)
    cfg.update(num_hidden_layers=3, hidden_size=64, intermediate_size=96, num_attention_heads=4, num_key_value_heads=2,
               head_dim=16, vocab_size=V)
    check(_builder(Model, ModelArgs.from_dict(cfg)), ["dense"] * 3, [96] * 3)


def test_muse_glimmer_dense():
    from mlx2.adapters.muse_glimmer_config import ModelArgs
    from mlx2.runtime.models.muse_glimmer import Model
    cfg = _config(T7 / "Muse-Glimmer-30B")
    t = dict(cfg.get("text_config") or cfg)
    t.update(num_hidden_layers=4, hidden_size=64, intermediate_size=96, num_attention_heads=4, num_key_value_heads=2,
             head_dim=16, vocab_size=V, sliding_window=4, layer_types=t["layer_types"][:4],
             layer_rope_theta=t["layer_rope_theta"][:4])
    check(_builder(Model, ModelArgs.from_dict(t)), ["dense"] * 4, [96] * 4)


QWEN_DENSE = [MODELS / "Qwen3.8-27B-BF16-q8-mlx2", MODELS / "Qwen3.6-27B-Abliterated-Heretic-Uncensored-MLX-4bit",
              HOME / "models/OrcaSAQ-2-Cyber-27B-Uncensored-GGUF/OrcaSAQ-2-27B-MLX-affine-5-6-8"]


@pytest.mark.parametrize("path", QWEN_DENSE, ids=lambda p: p.name)
def test_qwen3_5_dense_27b(path):
    """Qwen3.8-27B (q8), Qwen3.6-27B (q4), OrcaSAQ-2-27B (affine 5/6/8 mixed) on the qwen38_27b classes."""
    from mlx2.runtime.models.qwen38_27b import Model, ModelArgs
    cfg = _config(path)
    t = dict(cfg["text_config"])
    t.update(hidden_size=128, intermediate_size=256, num_hidden_layers=4, head_dim=32, num_attention_heads=4,
             num_key_value_heads=2, linear_key_head_dim=32, linear_value_head_dim=32, linear_num_key_heads=2,
             linear_num_value_heads=4, vocab_size=V, mtp_num_hidden_layers=0, tie_word_embeddings=False)
    t["layer_types"] = t["layer_types"][:4]
    cfg["text_config"] = t
    args = ModelArgs.from_dict(cfg)
    check(_builder(Model, args), ["dense"] * 4, [256] * 4)
    for bits in (5, 6, 8):
        m = _quantize(_builder(Model, args)(0), bits=bits, group_size=64)
        h = _host(m)
        assert h.widths == [256] * 4 and h.sites[0].params_per_unit == 3 * 128
        assert h.check_site_parity(_ids())["ok"]


# --------------------------------------------------------------------------- the guards


def _flint_tiny():
    from mlx2.runtime.models.flint import Model, ModelArgs
    cfg = _config(MODELS / "flint-2a247be0-oq4e")
    cfg.update(num_hidden_layers=4, num_experts=8, num_experts_per_tok=2, hidden_size=64, intermediate_size=128,
               moe_intermediate_size=64, shared_expert_intermediate_size=64, head_dim=16, num_attention_heads=4,
               num_key_value_heads=2, vocab_size=V, sliding_window=4, layer_types=cfg["layer_types"][:4],
               no_rope_layer_indices=[i for i in cfg["no_rope_layer_indices"] if i < 4])
    return _builder(Model, ModelArgs.from_dict(cfg))(0)


def test_site_parity_fails_a_wrong_router_law(monkeypatch):
    """A block given the wrong routing law (softmax for Flint's renormalised sigmoid) is caught."""
    def softmax_law(mx_, self, x):
        p = mx_.softmax(self.gate(x).astype(mx_.float32), axis=-1)
        inds = U._topk(mx_, p, self.top_k)
        return inds, mx_.take_along_axis(p, inds, axis=-1), self.switch_mlp(x, inds), self.shared_expert(x)
    h = _host(_flint_tiny())
    assert h.check_site_parity(_ids())["ok"]
    monkeypatch.setitem(U._ROWS, "sigmoid_shared", softmax_law)
    r = h.check_site_parity(_ids())
    assert not r["ok"] and set(r["over_tol"]) == {2, 3}, r


def test_site_parity_reports_a_site_a_fused_path_skips():
    """A dense MLP whose forward bypasses down_proj.__call__ (a fused kernel) masks nothing: reported."""
    h = _host(_flint_tiny())
    mlp = h.layers[1].mlp
    base = type(mlp)

    def fused(self, x):
        return (nn.silu(self.gate_proj(x)) * self.up_proj(x)) @ self.down_proj["weight"].T
    mlp.__class__ = type("Fused", (base,), {"__call__": fused})
    r = h.check_site_parity(_ids())
    assert not r["ok"] and r["not_engaged"] == [1], r


def test_unknown_block_refused_and_token_mixers_skipped():
    """A mixer slot holding attention/SSM (o_proj / out_proj) is no site; an unknown FFN shape refuses."""
    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.o_proj = nn.Linear(8, 8)

    class Odd(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(8, 8)

    class Layer(nn.Module):
        def __init__(self, mixer):
            super().__init__()
            self.mixer = mixer
    sites, _ = U.install([Layer(Attn())])
    assert sites[0].kind is None and sites[0].n == 0
    with pytest.raises(SystemExit, match="unknown shape"):
        U.install([Layer(Odd())])


# --------------------------------------------------------------------------- exFAT AppleDouble shards


def _appledouble_artifact(tmp_path):
    """One real shard and the AppleDouble companion macOS writes beside it on exFAT (the T7)."""
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {"w": mx.zeros((2, 2))})
    (tmp_path / "._model.safetensors").write_bytes(b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X        " + b"\0" * 64)
    return tmp_path


def test_shard_listing_skips_appledouble(tmp_path):
    from mlx2.adapters.artifact_paths import safetensors_shards
    from mlx2.runtime.streamed_load import artifact_weight_bytes
    from mlx2.runtime.weight_stream import SafetensorsIndex
    root = _appledouble_artifact(tmp_path)
    assert [p.name for p in safetensors_shards(root)] == ["model.safetensors"]
    assert artifact_weight_bytes(root) == (root / "model.safetensors").stat().st_size
    # expert streaming reads every listed shard's header: before the fix the ._ file raised
    # StreamingUnavailable("invalid safetensors header length")
    SafetensorsIndex.from_model_path(root)


def test_ling_without_shared_experts_still_classified():
    """Review fix: bailing_hybrid sets shared_experts = None when num_shared_experts == 0; that block
    is still ``ling`` (it was before the new kinds) and _ling_rows runs it without a shared path."""
    from mlx2.runtime.models.kimi_linear import Model, ModelArgs
    cfg = _config(T7 / "Kimi-Linear-48B-A3B-Instruct")
    lac = dict(cfg["linear_attn_config"])
    lac.update(full_attn_layers=[4], kda_layers=[1, 2, 3], head_dim=16, num_heads=4)
    cfg.update(num_hidden_layers=4, hidden_size=64, intermediate_size=96, num_attention_heads=4, num_key_value_heads=4,
               head_dim=16, vocab_size=V, num_experts=8, num_experts_per_token=2, moe_intermediate_size=32,
               kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=16, v_head_dim=16, linear_attn_config=lac,
               num_shared_experts=0)
    build = _builder(Model, ModelArgs.from_dict(cfg))
    m = build(0)
    assert all(getattr(l.mlp, "shared_experts", 1) is None for l in m.model.layers[1:])
    check(build, ["dense", "ling", "ling", "ling"], [96, 8, 8, 8])


def test_mellum_softmax_topk_bf16_matches_block_selection():
    """Review fix: the router runs softmax(precise) in the gate's dtype, as the block does, so bf16
    probability ties pick the same experts (a float32 upcast diverged on ~21% of tokens at logit std 0.1).
    The expert bank is replaced by a one-hot of the selected ids, so the block's own output is exactly
    its selection x routed weight (CPU gather_mm runs only in float32)."""
    from research.amoeba.carve import units as U
    from mlx2.runtime.models.mellum import ModelArgs, SparseMoE as MoE
    cfg = _config(_hf("JetBrains/Mellum2.1-12B-A2.5B-Thinking"))
    cfg.update(num_hidden_layers=1, num_experts=64, num_experts_per_tok=8, hidden_size=64, intermediate_size=96,
               moe_intermediate_size=32, head_dim=16, num_attention_heads=4, num_key_value_heads=2, vocab_size=V,
               sliding_window=4, layer_types=cfg["layer_types"][:1], num_mtp_layers=0, mtp_layer_types=[],
               mtp_mlp_layer_types=[])
    args = ModelArgs.from_dict(cfg)
    n = args.num_experts

    class OneHot:
        def __call__(self, x, inds):
            return (inds[..., None] == mx.arange(n)).astype(mx.float32)
    mx.random.seed(0)
    blk = MoE(args)
    blk.gate.weight = (mx.random.normal(blk.gate.weight.shape) * 0.1 / 8).astype(mx.bfloat16)  # logit std ~0.1
    blk.switch_mlp = OneHot()
    x = mx.random.normal((4, 64, 64)).astype(mx.bfloat16)
    assert U.classify(blk) == "softmax_topk"
    base = blk(x).astype(mx.float32)                                     # [B, T, n]: block's selection x weight
    inds, w, rows, shared = U._softmax_topk_rows(mx, blk, x)
    ours = (rows * w[..., None]).sum(-2)
    mx.eval(base, ours)
    assert np.array_equal(np.array(base) > 0, np.array(ours) > 0)        # same experts on every token
    assert float(mx.abs(base - ours).max()) <= 1e-2                      # bf16 weights, combine order
