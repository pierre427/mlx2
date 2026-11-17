"""MiMo-V2 layer stream against Xiaomi's own PyTorch reference, tiny random configs (CPU, fp32).

The vendor ``modeling_mimo_v2.py`` / ``configuration_mimo_v2.py`` are copied
from the MiMo-V2.6-Pro-RL checkpoint (XiaomiMiMo/MiMo-V2.6-Pro-RL@a066ba00,
hashes pinned below) into a fresh temporary package and run in a separate
``python -I`` interpreter (tests/mimo_v2_torch_runner.py). The tiny checkpoints
are written in the real on-disk format (FP8 block scales, a TP-sharded fused
qkv, MXFP4 experts, BF16 elsewhere) by tests/mimo_v2_numpy_ref.py; the
reference gets the same weights dequantized by independent NumPy code.

Config A mirrors V2.6-Pro (fused qkv TP=2, full/SWA pattern starting full,
dense layer 0, SWA-only sinks, value scale, partial RoPE, n_group 1).
Config B covers the other declared variants (split projections, SWA dims that
differ from full, sinks on both kinds, n_group 4 / topk_group 2, unnormalised
top-k with routed_scaling_factor, MoE first and dense last).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import mimo_v2_numpy_ref as R  # noqa: E402

REFERENCE_DIR = Path(os.environ.get("MLX2_MIMO_V2_REFERENCE_DIR", Path.home() / "Models/MiMo-V2.6-Pro-RL"))
PINS = {"modeling_mimo_v2.py": "a8c3cb3aae473bcc15f023010547c919f15eba6546e6ed7efb61a8937b12f3ad",
        "configuration_mimo_v2.py": "773062ac9850b908eb54751b3e4dbe00e653c0e80595599409e97d0c1af2ce3e"}

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(not all((REFERENCE_DIR / f).exists() for f in PINS),
                                reason="MiMo-V2 reference modeling files not present")

BASE = {
    "model_type": "mimo_v2", "vocab_size": 256, "hidden_size": 64, "intermediate_size": 96,
    "moe_intermediate_size": 32, "hidden_act": "silu", "layernorm_epsilon": 1e-5,
    "max_position_embeddings": 4096, "tie_word_embeddings": False, "attention_bias": False,
    "attention_dropout": 0.0, "initializer_range": 0.02, "use_cache": False,
    "scoring_func": "sigmoid", "topk_method": "noaux_tc",
    "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
                            "weight_block_size": [32, 32], "mxfp4_block_size": 32, "store_dtype": "mxfp4"},
}
CONFIG_A = {**BASE,
            "num_hidden_layers": 4, "hybrid_layer_pattern": [0, 1, 1, 0], "moe_layer_freq": [0, 1, 1, 1],
            "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 24, "v_head_dim": 16,
            "swa_num_attention_heads": 8, "swa_num_key_value_heads": 2, "swa_head_dim": 24, "swa_v_head_dim": 16,
            "sliding_window": 4, "sliding_window_size": 4, "add_full_attention_sink_bias": False,
            "add_swa_attention_sink_bias": True, "attention_value_scale": 0.612,
            "partial_rotary_factor": 0.334, "rope_theta": 10000000, "swa_rope_theta": 10000,
            "rope_parameters": {"partial_rotary_factor": 0.334, "rope_theta": 10000000, "rope_type": "default",
                                "type": "default"},
            "n_routed_experts": 16, "num_experts_per_tok": 4, "n_group": 1, "topk_group": 1,
            "norm_topk_prob": True, "routed_scaling_factor": None, "attention_projection_layout": "fused_qkv"}
CONFIG_B = {**BASE,
            "num_hidden_layers": 3, "hybrid_layer_pattern": [1, 0, 1], "moe_layer_freq": [1, 1, 0],
            "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 32, "v_head_dim": 24,
            "swa_num_attention_heads": 4, "swa_num_key_value_heads": 2, "swa_head_dim": 16, "swa_v_head_dim": 16,
            "sliding_window": 3, "add_full_attention_sink_bias": True, "add_swa_attention_sink_bias": True,
            "attention_value_scale": None, "partial_rotary_factor": 0.25, "rope_theta": 50000,
            "swa_rope_theta": 5000,
            "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 50000, "rope_type": "default"},
            "n_routed_experts": 16, "num_experts_per_tok": 3, "n_group": 4, "topk_group": 2,
            "norm_topk_prob": False, "routed_scaling_factor": 2.5, "attention_projection_layout": "split"}
CASES = {"A": (CONFIG_A, 2), "B": (CONFIG_B, 1)}
IDS = np.array([[5, 17, 200, 3, 99, 42, 7, 255, 128, 1, 64, 33],
                [9, 9, 81, 150, 2, 240, 77, 18, 0, 111, 63, 5]], np.int64)


@pytest.fixture(scope="module")
def ref_pkg(tmp_path_factory):
    pkg = tmp_path_factory.mktemp("vendor") / "pkg"
    (pkg / "mimo_ref").mkdir(parents=True)
    for name, sha in PINS.items():
        data = (REFERENCE_DIR / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == sha, f"{name} changed upstream; re-read it before running"
        (pkg / "mimo_ref" / name).write_bytes(data)
    (pkg / "mimo_ref" / "__init__.py").write_text("")
    return pkg


def run_reference(pkg, work, cfg, sd):
    cfg_ref = {k: v for k, v in cfg.items() if k != "quantization_config"}
    (work / "ref_config.json").write_text(json.dumps(cfg_ref))
    np.savez(work / "state.npz", **sd)
    np.save(work / "ids.npy", IDS)
    out = work / "ref_out.npz"
    proc = subprocess.run([sys.executable, "-I", str(HERE / "mimo_v2_torch_runner.py"), "tiny", str(pkg),
                           str(work / "ref_config.json"), str(work / "state.npz"), str(work / "ids.npy"), str(out)],
                          capture_output=True, text=True, timeout=600, cwd=str(work))
    assert proc.returncode == 0, proc.stderr[-4000:]
    return dict(np.load(out))


@pytest.fixture(scope="module", params=sorted(CASES))
def case(request, ref_pkg, tmp_path_factory):
    cfg, tp = CASES[request.param]
    work = tmp_path_factory.mktemp(f"mimo_{request.param}")
    ckpt = work / "ckpt"
    sd = R.build_tiny_checkpoint(ckpt, copy.deepcopy(cfg), tp=tp, seed=7)
    ref = run_reference(ref_pkg, work, cfg, sd)
    return request.param, cfg, ckpt, sd, ref


def _family(ckpt, experts):
    from research.amoeba.sweep.mimo_v2 import MiMoV2Family
    return MiMoV2Family(ckpt, dtype="float32", experts=experts, fingerprint="tiny-synthetic")


def _sorted_routes(ids, wts):
    o = np.argsort(ids, axis=-1)
    return np.take_along_axis(ids, o, -1), np.take_along_axis(wts, o, -1)


def _sweep(fam):
    h = fam.embed(mx.array(IDS.astype(np.int32)))
    hs, pays = [], []
    for l in range(fam.L):
        h, p = fam.run_layer(l, h)
        mx.eval(h, *p.values())
        hs.append(np.array(h.astype(mx.float32)))
        pays.append({k: np.array(v.astype(mx.float32) if mx.issubdtype(v.dtype, mx.floating) else v)
                     for k, v in p.items()})
        pays[-1]["_dtypes"] = {k: v.dtype for k, v in p.items()}
        fam.release_layer(l)
    return hs, pays, np.array(fam.final_logits(h).astype(mx.float32))


@pytest.mark.parametrize("experts", ["routed", "all"])
def test_family_matches_vendor_reference(case, experts):
    name, cfg, ckpt, sd, ref = case
    fam = _family(ckpt, experts)
    assert fam.kinds == ["experts" if f else "dense" for f in cfg["moe_layer_freq"]]
    hs, pays, logits = _sweep(fam)
    report = {}
    B, T = IDS.shape
    k, d = cfg["num_experts_per_tok"], cfg["hidden_size"]
    for l in range(fam.L):          # sweep README payload contract
        want = ({"act": (B, T, fam.widths[l])} if fam.kinds[l] == "dense"
                else {"inds": (B, T, k), "w": (B, T, k), "rows": (B, T, k, d)})
        assert {n: v.shape for n, v in pays[l].items() if n != "_dtypes"} == want
        if fam.kinds[l] == "experts":
            assert pays[l]["_dtypes"]["w"] == mx.float32 and pays[l]["_dtypes"]["inds"] == mx.int32
    for l in range(fam.L):
        dh = np.abs(hs[l] - ref[f"h{l}"]).max()
        assert dh < 1e-4 * max(1.0, np.abs(ref[f"h{l}"]).max()), (name, l, dh)
        if fam.kinds[l] == "dense":
            np.testing.assert_allclose(pays[l]["act"], ref[f"act{l}"], rtol=1e-4, atol=1e-5)
            report[l] = ("dense", float(np.abs(pays[l]["act"] - ref[f"act{l}"]).max()))
            continue
        ours_i, ours_w = _sorted_routes(pays[l]["inds"].reshape(-1, k), pays[l]["w"].reshape(-1, k))
        ref_i, ref_w = _sorted_routes(ref[f"ids{l}"], ref[f"wts{l}"])
        np.testing.assert_array_equal(ours_i, ref_i)
        np.testing.assert_allclose(ours_w, ref_w, rtol=1e-5, atol=1e-6)
        assert pays[l]["w"].dtype == np.float32
        inds = pays[l]["inds"].reshape(-1, k)
        rows = pays[l]["rows"].reshape(B * T, k, -1)
        worst = 0.0
        for e in range(cfg["n_routed_experts"]):
            tok, slot = np.nonzero(inds == e)
            if tok.size == 0:
                assert f"y{l}_{e}" not in ref
                continue
            order = np.argsort(tok, kind="stable")          # torch.where order: ascending token
            ours = rows[tok[order], slot[order]]
            np.testing.assert_allclose(ours, ref[f"y{l}_{e}"], rtol=1e-4, atol=1e-5)
            worst = max(worst, float(np.abs(ours - ref[f"y{l}_{e}"]).max()))
        report[l] = ("experts", worst, float(np.abs(ours_w - ref_w).max()))
    d = np.abs(logits - ref["logits"])
    scale = np.abs(ref["logits"]).max()
    print(f"\nMiMo-V2 tiny {name}/{experts}: logits max|d| {d.max():.3g} (max|ref| {scale:.3g}); "
          f"per layer {report}")
    assert d.max() < 1e-4 * max(1.0, scale)
    assert (logits.argmax(-1) == ref["logits"].argmax(-1)).all()


def test_bfloat16_route_runs_close_to_reference(case):
    """The sweep's default dtype (bf16, gather_qmm experts): a loose sanity bound, not parity.

    bf16 rounding can flip a near-tied top-k choice (config B has 16 experts, unnormalised
    weights x2.5); a flipped route changes that token and, through attention, the later ones.
    Positions before a row's first flipped route are compared; flips must be near ties."""
    from research.amoeba.sweep.mimo_v2 import MiMoV2Family
    name, cfg, ckpt, sd, ref = case
    fam = MiMoV2Family(ckpt, dtype="bfloat16", experts="all", fingerprint="tiny-synthetic")
    _, pays, logits = _sweep(fam)
    B, T = IDS.shape
    k = cfg["num_experts_per_tok"]
    first = np.full(B, T)
    for l, p in enumerate(pays):
        t = p["_dtypes"]
        if fam.kinds[l] == "dense":
            assert t["act"] == mx.bfloat16
            continue
        assert (t["w"], t["rows"]) == (mx.float32, mx.bfloat16)
        flip = (np.sort(p["inds"].reshape(B, T, k), -1) != np.sort(ref[f"ids{l}"].reshape(B, T, k), -1)).any(-1)
        for b in range(B):
            if flip[b].any():
                first[b] = min(first[b], int(np.argmax(flip[b])))
    keep = np.arange(T)[None, :] < first[:, None]
    rel = np.abs(logits - ref["logits"])[keep].max() / np.abs(ref["logits"]).max()
    agree = (logits.argmax(-1) == ref["logits"].argmax(-1))[keep].mean()
    print(f"\nMiMo-V2 tiny {name} bf16: first route flip per row {first.tolist()} of T={T}; "
          f"before it: logits max rel err {rel:.3g}, top-1 agreement {agree:.3f}")
    assert keep.sum() >= T and rel < 0.05 and agree >= 0.8


def test_loader_dequant_matches_numpy_state_dict(case):
    """mlx2's loader (MLX from_fp8, per-rank qkv grids, MXFP4 view) vs the independent NumPy dequant."""
    from mlx2.runtime.models import mimo_v2 as M
    name, cfg, ckpt, sd, _ = case
    fam = _family(ckpt, "routed")
    for l in range(fam.L):
        w = fam.load_layer(l)
        p = f"model.layers.{l}."
        if cfg["attention_projection_layout"] == "fused_qkv":
            want = sd[p + "self_attn.qkv_proj.weight"]
            got = np.concatenate([np.array(w[f"self_attn.{n}_proj.weight"]) for n in "qkv"])
        else:
            want = np.concatenate([sd[p + f"self_attn.{n}_proj.weight"] for n in "qkv"])
            got = np.concatenate([np.array(w[f"self_attn.{n}_proj.weight"]) for n in "qkv"])
        np.testing.assert_array_equal(got, want)
        if fam.kinds[l] == "dense":
            for n in ("gate_proj", "up_proj", "down_proj"):
                np.testing.assert_array_equal(np.array(w[f"mlp.{n}.weight"]), sd[p + f"mlp.{n}.weight"])
        else:
            bank = fam._bank(l, mx.array(np.arange(cfg["n_routed_experts"])))
            for e in range(cfg["n_routed_experts"]):
                for j, n in enumerate(("gate_proj", "up_proj", "down_proj")):
                    got = np.array(M._dense_of(getattr(bank, ("gate", "up", "down")[j]), e, mx.float32))
                    np.testing.assert_array_equal(got, sd[f"{p}mlp.experts.{e}.{n}.weight"])
        fam.release_layer(l)


def test_tp_layout_is_not_the_global_reading(case):
    """A global (unsharded) reading of the rank-sharded scale grid is refused, not silently misread."""
    from mlx2.runtime.models.mimo_v2_weights import fused_qkv_shard_layout
    name, cfg, *_ = case
    if cfg["attention_projection_layout"] != "fused_qkv":
        pytest.skip("split projections")
    H, K, hd, vhd = 8, 2, 24, 16
    rows = H * hd + K * (hd + vhd)
    assert fused_qkv_shard_layout(rows, 2 * -(-(rows // 2) // 32), H, K, hd, vhd, 2, 32)[3] == 5
    with pytest.raises(ValueError):
        fused_qkv_shard_layout(rows, -(-rows // 32), H, K, hd, vhd, 2, 32)


def test_config_refuses_unsupported_variants():
    from mlx2.runtime.models.mimo_v2 import MiMoV2Config
    MiMoV2Config.from_dict(CONFIG_A)
    for patch in ({"scoring_func": "softmax"}, {"topk_method": "greedy"}, {"hidden_act": "gelu"},
                  {"attention_bias": True}, {"rope_parameters": {"rope_type": "yarn", "rope_theta": 1.0}},
                  {"rope_parameters": {**CONFIG_A["rope_parameters"], "partial_rotary_factor": 0.5}}):
        with pytest.raises(ValueError):
            MiMoV2Config.from_dict({**CONFIG_A, **patch})
