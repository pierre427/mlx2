"""CPU checks for the Raon-1-mini runtime, conversion and DFlash2 binding.

Tiny random configs only; no artifact and no GPU.  The full-size torch
reference comparison lives in scripts/raon1_mini_reference_oracle.py.
"""

import json

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_map


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def tiny_config(**overrides):
    config = {
        "model_type": "raon1_mini", "hidden_size": 64, "num_hidden_layers": 5, "num_attention_heads": 4,
        "head_dim": 16, "vocab_size": 97, "tie_word_embeddings": False, "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-5, "activation_situ_beta": 4.0, "activation_situ_linear_beta": 25.0,
        "intermediate_size": 96, "first_k_dense_replace": 1, "moe_layer_freq": 1, "num_experts": 8,
        "num_experts_per_token": 2, "moe_intermediate_size": 32, "num_shared_experts": 1,
        "moe_router_activation_func": "sigmoid", "moe_renormalize": True, "routed_scaling_factor": 1.0,
        "q_lora_rank": 24, "kv_lora_rank": 32, "qk_nope_head_dim": 16, "qk_rope_head_dim": 8,
        "v_head_dim": 16, "mla_use_nope": True, "mla_use_output_gate": True, "raon_mhc_streams": 4,
        "reasoning_effort_steering": {"alpha": {"none": 0, "low": 0.3, "medium": 0.1, "high": 0.05,
                                                "xhigh": 0.02, "max": 0},
                                      "default": "high", "layers": [2, 4]},
        "rda_config": {"num_heads": 4, "head_dim": 16, "conv_kernel_size": 4, "full_rank_gate": True,
                       "gate_lower_bound": -5.0, "mla_layers": [4, 5], "rda_layers": [1, 2, 3]},
        "rda_decay_rank": 8,
    }
    config.update(overrides)
    return config


def _model(seed=0, scale=0.25):
    from mlx2.runtime.models import raon1_mini as R

    model = R.Model(R.ModelArgs.from_dict(tiny_config()))
    mx.random.seed(seed)
    model.update(tree_map(lambda p: mx.random.normal(p.shape).astype(p.dtype) * scale, model.parameters()))
    return model


def test_layer_partition_is_one_based_and_steering_layers_carry_vectors():
    model = _model()
    kinds = [(layer.is_global, layer.is_moe, layer.steered) for layer in model.layers]
    # rda_layers [1,2,3] (1-based) -> 0-based 0..2 RDA; mla_layers [4,5] -> 3, 4.
    assert kinds == [(False, False, False), (False, True, False), (False, True, True),
                     (True, True, False), (True, True, True)]


def test_cached_decode_matches_full_forward():
    model = _model()
    tokens = mx.array([[3, 14, 15, 9, 26, 5, 35, 8, 9, 79]])
    full = model(tokens)
    cache = model.make_cache()
    rows = [model(tokens[:, :6], cache)]
    for i in range(6, tokens.shape[1]):
        rows.append(model(tokens[:, i:i + 1], cache))
    assert float(mx.abs(full - mx.concatenate(rows, 1)).max()) < 1e-5


def test_taps_are_full_stream_residuals():
    model = _model()
    tokens = mx.array([[1, 2, 3, 4]])
    logits, features = model.forward_with_taps(tokens, None, (1, 3))
    assert features.shape == (1, 4, 2 * 4 * 64)
    assert float(mx.abs(logits - model(tokens)).max()) == 0.0
    with pytest.raises(ValueError):
        model.forward_with_taps(tokens, None, (3, 1))


def test_mhc_chart_rows_sum_to_one_and_stay_nonnegative():
    from mlx2.runtime.models import raon1_mini as R

    q = R.helmert_basis(4)
    assert float(mx.abs(q.T @ q - mx.eye(3)).max()) < 1e-6
    assert float(mx.abs(q.T @ mx.ones((4,))).max()) < 1e-6
    model = _model(scale=1.0)
    stream = mx.random.normal((2, 3, 4, 64)).astype(mx.float32)
    _, post, res = model.layers[1].self_attn_raon_mhc.route(stream)
    assert float(mx.abs(res.sum(-1) - 1.0).max()) < 1e-5
    assert float(res.min()) >= -1e-6
    assert float(post.max()) <= 2.0


def test_effort_alpha_table():
    model = _model()
    assert model.effort_alpha(None) == 0.05
    assert model.effort_alpha("none") == 0.1      # config none 0 -> no-thinking 0.1
    assert model.effort_alpha("low") == 0.3
    assert model.effort_alpha("max") == 0.0
    with pytest.raises(ValueError):
        model.effort_alpha("extreme")
    assert model.set_reasoning_effort("none") == 0.1
    assert all(layer.effort_alpha == 0.1 for layer in model.layers)


def test_steering_changes_only_steered_layers():
    model = _model()
    tokens = mx.array([[5, 6, 7]])
    model.set_reasoning_effort("max")
    base = model(tokens)
    model.set_reasoning_effort("low")
    assert float(mx.abs(model(tokens) - base).max()) > 0
    for layer in model.layers:
        if layer.steered:
            layer.moe.reasoning_effort_steering = mx.zeros_like(layer.moe.reasoning_effort_steering)
    assert float(mx.abs(model(tokens) - base).max()) == 0.0


def test_sanitize_splits_vendor_merged_conv_and_hf_convs():
    model = _model()
    merged = mx.arange(3 * 64 * 4, dtype=mx.float32).reshape(3 * 64, 4, 1)
    out = model.sanitize({"model.layers.0.self_attn.qkv_conv1d.weight": merged,
                          "model.layers.1.self_attn.q_conv1d.weight": mx.ones((64, 4))})
    assert out["model.layers.0.self_attn.k_conv1d.conv.weight"].shape == (64, 4, 1)
    assert float(out["model.layers.0.self_attn.v_conv1d.conv.weight"][0, 0, 0]) == 2 * 64 * 4
    assert out["model.layers.1.self_attn.q_conv1d.conv.weight"].shape == (64, 4, 1)
    with pytest.raises(ValueError, match="conversion"):
        model.sanitize({"model.layers.1.moe.experts.0.gate_proj.weight": mx.zeros((2, 2))})


def test_mxfp8_repack_is_byte_exact():
    torch = pytest.importorskip("torch")
    from mlx2.conversion.raon1_mini import pack_codes

    torch.manual_seed(1)
    weight = (torch.randn(8, 64) * 3).to(torch.float8_e4m3fn)
    scale = torch.randint(118, 132, (8, 2), dtype=torch.uint8)
    reference = (weight.float() * torch.exp2(scale.float() - 127).repeat_interleave(32, 1)).numpy()
    codes = mx.array(weight.view(torch.uint8).numpy())
    dequant = mx.dequantize(pack_codes(codes), mx.array(scale.numpy()), group_size=32, bits=8, mode="mxfp8",
                             dtype=mx.float32)
    assert np.array_equal(np.array(dequant), reference)


def test_validate_config_refuses_unstaged_switches():
    from mlx2.adapters import raon1_mini as A

    with pytest.raises(ValueError, match="topology"):
        A.validate_config({"model_type": "raon1_mini"})
    with pytest.raises(ValueError):
        A.parse_policy({"mla_flash": "all"})
    with pytest.raises(ValueError):
        A.parse_policy({"reasoning_effort": "extreme"})
    assert A.parse_policy({})["mla"] == "auto"
    assert A.parse_policy({})["merged_projections"] is True


def test_registry_dispatches_raon():
    from mlx2.adapters.registry import _RESOLVERS

    assert "raon1_mini" in _RESOLVERS


def test_dflash2_config_reads_stream_count_and_sizes_fc():
    from mlx2.adapters.dflash2 import _expected_weight_shapes
    from mlx2.runtime.drafters.dflash2_config import DFlash2Config

    config = {
        "architectures": ["DFlash2DraftModel"], "model_type": "qwen3", "hidden_size": 64,
        "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 1, "head_dim": 16, "vocab_size": 97, "max_position_embeddings": 4096,
        "num_target_layers": 5, "rms_norm_eps": 1e-5, "hidden_act": "silu",
        "layer_types": ["sliding_attention"] * 2, "sliding_window": 32,
        "rope_parameters": {"rope_theta": 1e7, "rope_type": "default"}, "block_size": 8,
        "dflash_config": {"block_size": 8, "causal": False, "conv_group_size": 16, "conv_kernel_size": 2,
                          "mask_token_id": 96, "selector_rank": 8, "selector_top_k": 4,
                          "target_layer_ids": [1, 3], "target_streams_per_layer": 4,
                          "use_swa": True, "swa_window_size": 32},
    }
    args = DFlash2Config.from_dict(config)
    assert args.target_streams_per_layer == 4
    assert args.feature_width == 2 * 4 * 64
    assert _expected_weight_shapes(args)["fc.weight"] == [64, 2 * 4 * 64]
    single = DFlash2Config.from_dict({**config, "dflash_config": {
        k: v for k, v in config["dflash_config"].items() if k != "target_streams_per_layer"}})
    assert single.feature_width == 2 * 64
    from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel

    assert DFlash2DraftModel(args).fc.weight.shape == (64, 2 * 4 * 64)


def test_raon_dflash2_contract_refuses_single_stream_taps():
    from mlx2.adapters.raon1_mini_dflash2 import _check_declared_contract

    target = {"hidden_size": 2560, "vocab_size": 259840, "num_hidden_layers": 25}
    good = {"architectures": ["DFlash2DraftModel"], "hidden_size": 2560, "vocab_size": 259840,
            "num_target_layers": 25, "target_hidden_size": 10240,
            "dflash_config": {"target_layer_ids": [1, 6, 11, 16, 21], "target_streams_per_layer": 4,
                              "target_embedding_size": 2560, "target_feature_size": 51200}}
    _check_declared_contract(good, target)
    bad = json.loads(json.dumps(good))
    bad["dflash_config"]["target_streams_per_layer"] = 1
    with pytest.raises(ValueError, match="4 mHC streams"):
        _check_declared_contract(bad, target)


def test_merged_projections_match_parts():
    from mlx2.adapters import raon1_mini as A

    model = _model()
    tokens = mx.array([[5, 6, 7, 8]])
    model.configure(A.parse_policy({"merged_projections": False}))
    cache = model.make_cache()
    model(tokens, cache)
    plain = model(mx.array([[9]]), cache)
    model.configure(A.parse_policy({"merged_projections": True}))
    cache = model.make_cache()
    model(tokens, cache)
    merged = model(mx.array([[9]]), cache)
    assert model.layers[0].self_attn.calls["merged_projections"] > 0
    assert float(mx.abs(plain - merged).max()) < 1e-5


def test_rms_law_mlx_rounds_twice_and_fp32_matches_reference_rounding():
    # PR sweep 2026-11-25 (vllm-project/vllm#64672 numerics note): the
    # reference Raon1MiniRMSNorm multiplies the weight in fp32 and rounds once;
    # mx.fast.rms_norm rounds the normalized row before the weight product.
    from mlx2.runtime.models import raon1_mini as R

    mx.random.seed(3)
    x = (mx.random.normal((16, 256)) * 3).astype(mx.bfloat16)
    norm = R.RaonRMSNorm(256, eps=1e-5)
    norm.weight = (mx.random.normal((256,)) * 0.3 + 1).astype(mx.bfloat16)
    x32 = x.astype(mx.float32)
    normed = x32 * mx.rsqrt(mx.mean(x32 * x32, -1, keepdims=True) + 1e-5)
    twice = normed.astype(mx.bfloat16) * norm.weight
    once = (normed * norm.weight.astype(mx.float32)).astype(mx.bfloat16)
    assert norm.law == "mlx"
    assert mx.array_equal(norm(x), twice).item()
    norm.law = "fp32"
    out = norm(x)
    assert out.dtype == mx.bfloat16
    assert float(mx.abs(out.astype(mx.float32) - once.astype(mx.float32)).max()) == 0.0
    assert not mx.array_equal(once, twice).item()   # the laws really differ in bf16


def test_rms_law_policy_reaches_every_norm_and_defaults_to_mlx():
    from mlx2.adapters import raon1_mini as A
    from mlx2.runtime.models import raon1_mini as R

    assert A.parse_policy()["rms_law"] == "mlx"
    with pytest.raises(ValueError, match="rms_law"):
        A.parse_policy({"rms_law": "fast"})
    assert A.split_policy({"rms_law": "fp32", "draft_model": "x"}) == ({"rms_law": "fp32"}, {"draft_model": "x"})
    model = _model()
    norms = [m for _, m in model.named_modules() if isinstance(m, R.RaonRMSNorm)]
    # 2 per layer + final + 2 latent norms per MLA layer (2 MLA layers).
    assert len(norms) == 2 * 5 + 1 + 2 * 2
    tokens = mx.array([[3, 14, 15, 9, 26]])
    model.configure(A.parse_policy())
    base = model(tokens)
    model.configure(A.parse_policy({"rms_law": "fp32"}))
    assert all(m.law == "fp32" for m in norms)
    # In fp32 the laws coincide (CPU gather_mm is fp32-only, so the bf16
    # difference is pinned per module above, not end to end).
    assert float(mx.abs(model(tokens) - base).max()) < 1e-5
    model.configure(A.parse_policy())
    assert all(m.law == "mlx" for m in norms)
    assert mx.array_equal(model(tokens), base).item()


def test_steering_receipt_reports_thinking_off_reference_alpha():
    from mlx2.adapters import raon1_mini as A

    adapter = object.__new__(A.RaonAdapter)
    adapter.model = _model()
    adapter.policy = A.parse_policy()
    adapter.model.set_reasoning_effort(None)
    receipt = adapter.capability_receipts()["reasoning_effort_steering"]
    assert receipt["thinking_off"] == {"reference_alpha": 0.1, "served_alpha": 0.05, "matches_reference": False}
    adapter.policy = A.parse_policy({"reasoning_effort": "none"})
    adapter.model.set_reasoning_effort("none")
    assert adapter.capability_receipts()["reasoning_effort_steering"]["thinking_off"]["matches_reference"]


def test_mla_expanded_bound_moves_the_prefill_law_and_budget():
    from mlx2.adapters import raon1_mini as A
    from mlx2.adapters.raon1_mini_memory import RaonCacheBudget

    defaults = A.parse_policy({})
    assert A.split_policy({**defaults, "draft_model": "draft"}) == (defaults, {"draft_model": "draft"})
    assert defaults["mla_expanded_gib"] == 4
    with pytest.raises(ValueError):
        A.parse_policy({"mla_expanded_gib": 0})
    model = _model()
    model.configure(A.parse_policy({"mla_expanded_gib": 12}))
    mla = model.layers[3].self_attn
    assert mla.expanded_bytes == 12 << 30
    tokens = mx.array([list(range(1, 40))])
    model(tokens)
    assert mla.calls["auto_expanded"] > 0 and mla.calls["auto_latent"] == 0
    mla.expanded_bytes = 1  # below any prefix: chunks take the absorbed latent law
    model(tokens)
    assert mla.calls["auto_latent"] > 0
    base = RaonCacheBudget(7, 18, 512, 64, 40, 128, 128, 128, 2560, 259840, 3, 4, "auto")
    wide = RaonCacheBudget(7, 18, 512, 64, 40, 128, 128, 128, 2560, 259840, 3, 4, "auto", expanded_bytes=12 << 30)
    assert wide.prefill_transient_bytes(131072, 2048) > base.prefill_transient_bytes(131072, 2048)
    assert wide.prefill_transient_bytes(32768, 2048) == base.prefill_transient_bytes(32768, 2048)


def test_affine_install_and_merged_affine_projection_match_parts():
    from mlx2.adapters import raon1_mini as A
    from mlx2.runtime.models import raon1_mini as R

    model = _model()
    paths = []
    for i, layer in enumerate(model.layers):
        a = f"model.layers.{i}.self_attn."
        if layer.is_global:
            paths += [a + n for n in ("q_a_proj", "kv_a_proj_with_mqa", "g_proj", "o_proj")]
        else:
            paths += [a + n for n in ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_proj",
                                      "rda_w_proj", "o_proj")]
        m = f"model.layers.{i}." + ("moe.shared_experts." if layer.is_moe else "mlp.")
        paths += [m + n for n in ("gate_proj", "up_proj")]
    names = {p + ".scales" for p in paths}
    receipt = R.install_affine(model, {"quantization": {"group_size": 32, "bits": 4, "mode": "affine"}}, names)
    assert receipt["quantized_linears"] == len(paths)
    assert type(model.layers[0].self_attn.q_proj).__name__ == "QuantizedLinear"
    tokens = mx.array([[5, 6, 7, 8]])
    model.configure(A.parse_policy({"merged_projections": False}))
    cache = model.make_cache()
    model(tokens, cache)
    plain = model(mx.array([[9]]), cache)
    model.configure(A.parse_policy({"merged_projections": True}))
    cache = model.make_cache()
    model(tokens, cache)
    merged = model(mx.array([[9]]), cache)
    assert model.layers[0].self_attn.calls["merged_projections"] > 0
    assert float(mx.abs(plain - merged).max()) < 1e-4


def test_nax_levers_are_opt_in_and_decline_cleanly_on_cpu():
    from mlx2.adapters import raon1_mini as A

    policy = A.parse_policy({})
    assert policy["mla_flash"] == "off" and policy["moe_nax"] == "off"
    with pytest.raises(ValueError):
        A.parse_policy({"mla_flash": "all"})
    with pytest.raises(ValueError):
        A.parse_policy({"moe_nax": "nvfp4"})
    model = _model()
    tokens = mx.array([list(range(1, 40))])
    model.configure(A.parse_policy({}))
    base = model(tokens)
    model.configure(A.parse_policy({"mla_flash": "expanded"}))
    flashed = model(tokens)
    mla = model.layers[3].self_attn
    assert mla.calls["flash_declined"] > 0 and mla.calls.get("expanded_flash192", 0) == 0
    assert float(mx.abs(base - flashed).max()) == 0.0


def test_close_flushes_external_feedback_and_releases_drafter():
    from unittest.mock import Mock

    from mlx2.adapters.raon1_mini import RaonAdapter

    adapter = RaonAdapter.__new__(RaonAdapter)
    manager = adapter._external_feedback_manager = Mock()
    adapter.draft_model = object()
    adapter.close()
    manager.close.assert_called_once_with()
    assert adapter.draft_model is None


def test_flash_probe_cache_is_gpu_only(monkeypatch):
    from mlx2.runtime.models import raon1_mini as R

    cache = {}
    monkeypatch.setattr(R, "_EXPANDED_FLASH_OK", cache)
    assert not R.expanded_flash_available(40, mx.bfloat16)
    assert not cache
    cache[40, str(mx.bfloat16)] = True
    assert not R.expanded_flash_available(40, mx.bfloat16)


def test_nax_receipts_distinguish_selection_from_calls():
    from types import SimpleNamespace

    from mlx2.adapters.raon1_mini import RaonAdapter, parse_policy

    adapter = object.__new__(RaonAdapter)
    mla = {}
    adapter.model = SimpleNamespace(mechanism_status=lambda: dict(mla=mla, rda={}, effort_alpha=0),
                                    effort_alpha=lambda _: 0)
    adapter.policy = parse_policy()
    receipts = adapter.capability_receipts()
    assert all(receipts[key]["selected"] is False for key in ("mla_flash", "moe_nax_gather"))
    adapter.policy = parse_policy({"mla_flash": "expanded", "moe_nax": "gather"})
    receipts = adapter.capability_receipts()
    assert all(receipts[key]["selected"] is True for key in ("mla_flash", "moe_nax_gather"))
    assert receipts["mla_flash"]["observed_used"] is False
    mla["expanded_flash192_windows"] = 2
    flash = adapter.capability_receipts()["mla_flash"]
    assert flash["observed_used"] is True and flash["calls"] == 2


def test_decode_depth_near_tie_classifier_clauses():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("dd", root / "scripts/raon1_mini_decode_depth_ab.py")
    dd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dd)

    def arm(tokens, margins, tops):
        return {"tokens": tokens, "margins": margins, "top5": tops}

    # Exact tie in B between 7 and 9; A picked 9 at 0.125 -> exact-tie clause.
    a = arm([1, 9], [1.0, 0.125], [[[1, -0.1]], [[9, -0.5], [7, -0.625], [3, -4.0]]])
    b = arm([1, 7], [1.0, 0.0], [[[1, -0.1]], [[7, -0.6], [9, -0.6], [3, -4.0]]])
    assert dd.classify(a, b, 1)["verdict"] == "near_tie_candidate"
    # A large gap fails.
    b2 = arm([1, 7], [1.0, 2.0], [[[1, -0.1]], [[7, -0.1], [3, -2.1], [9, -3.5]]])
    a2 = arm([1, 9], [1.0, 2.0], [[[1, -0.1]], [[9, -0.1], [3, -2.1], [7, -3.5]]])
    assert dd.classify(a2, b2, 1)["verdict"] == "high_margin"


def test_hq_mixed_affine_recipe_end_to_end_on_tiny_model(monkeypatch):
    from mlx.utils import tree_flatten

    from mlx2.adapters import raon1_mini as A
    from mlx2.conversion import raon1_mini_hq as HQ
    from mlx2.runtime.models import raon1_mini as R

    # Tiny geometry fits group 32 only.
    monkeypatch.setattr(HQ, "FORMATS", {"a4g32": (4, 32, 5.0), "a8g32": (8, 32, 9.0)})
    model = _model(seed=3, scale=0.2)
    model.configure(A.parse_policy({"merged_projections": False, "mhc": "ops"}))
    mx.random.seed(7)
    seqs = [mx.random.randint(1, 96, (24,)).tolist() for _ in range(6)]
    hessians, rows, counts = HQ.calibrate(model, seqs, log=lambda *_: None)
    assert rows["1.moe_in"] == 6 * 24 and int(counts["1.moe_in"].sum()) == 6 * 24 * 2
    assert type(model.layers[1].moe.gate).__name__ == "Router"  # recorders removed
    settings = HQ.HQSettings(formats=("a4g32", "a8g32"), budget_bpw=7.0, dense_bits=8, dense_group=32)
    planes, quant, receipt = HQ.plan_and_quantize(model, hessians, counts, settings, log=lambda *_: None)
    assert receipt["moe_mean_bpw"] <= 7.0 and set(receipt["schedule"].values()) <= {"a4g32", "a8g32"}
    # Knapsack: an 8-bit layer is the most-distorted-at-4-bit one.
    if "a8g32" in receipt["schedule"].values():
        worst = max(receipt["distortion"], key=lambda k: receipt["distortion"][k]["a4g32"])
        assert receipt["schedule"][worst] == "a8g32"
    tensors = {**HQ._untouched(model, planes), **planes}
    reference = model(mx.array([seqs[0]]))
    fresh = R.Model(R.ModelArgs.from_dict(tiny_config()))
    config = {"quantization": {"group_size": 32, "bits": 8, "mode": "affine", **quant}}
    R.install_affine(fresh, config, tensors.keys())
    fresh.load_weights(list(fresh.sanitize(dict(tensors)).items()), strict=True)
    fresh.configure(A.parse_policy({"merged_projections": False, "mhc": "ops"}))
    got = fresh(mx.array([seqs[0]]))
    agree = float((mx.argmax(got, -1) == mx.argmax(reference, -1)).astype(mx.float32).mean())
    assert agree >= 0.9, agree
    assert not any(k.endswith(".conv.weight") for k in tensors)


def test_hq_knapsack_respects_budget_and_floors():
    from mlx2.conversion.raon1_mini_hq import knapsack

    d = {0: {"a4g64": 10.0, "a4g32": 6.0, "a8g64": 1.0}, 1: {"a4g64": 2.0, "a4g32": 1.9, "a8g64": 1.8},
         2: {"a4g64": 5.0, "a4g32": 3.0, "a8g64": 0.5}}
    # Budget 14.55 bits over 3 layers: the two best gains per bit (layer 0 at
    # 8/bit, layer 2 at 4/bit) fit; layer 1's 0.2/bit upgrade does not.
    choice = knapsack(d, budget_bpw=4.85)
    assert sum({"a4g64": 4.5, "a4g32": 5.0, "a8g64": 8.5}[f] for f in choice.values()) / 3 <= 4.85
    assert choice == {0: "a4g32", 1: "a4g64", 2: "a4g32"}
    with pytest.raises(ValueError):
        knapsack(d, budget_bpw=4.6, floors={0: "a8g64"})


def test_raon_auto_pairing_adds_no_policy_pins(monkeypatch):
    """2026-11-25: the paired drafter's revision was a pin, so the best-route
    seam merged ``draft_revision`` into the policy and every no-flag one-lane
    Raon server refused to load. The revision is identity only."""
    from mlx2 import best_route_defaults
    from mlx2.adapters import dflash2, drafter_pairing, raon1_mini, raon1_mini_dflash2
    from mlx2.adapters.raon1_mini import RaonAdapter, split_policy
    from types import SimpleNamespace

    monkeypatch.setattr(raon1_mini, "inspect_artifact",
                        lambda path: {"config": {}, "identity": {"fingerprint": "t" * 64}})
    monkeypatch.setattr(raon1_mini_dflash2, "inspect_drafter", lambda *a, **k: {"fingerprint": "w"})
    monkeypatch.setattr(dflash2, "content_revision", lambda record: "d" * 64)
    pairing = drafter_pairing.pair_drafter(
        {"adapter": "RaonAdapter", "path": "/target"},
        {"path": "/draft", "architectures": ["DFlash2DraftModel"], "name": "Raon-1-mini-dflash2"},
        target_revision=False)
    assert pairing["pins"] == {} and pairing["identity"] == "d" * 64
    assert pairing["execution_policy"] == {"draft_model": "/draft"}
    assert pairing["draft"] == "Raon-1-mini-dflash2@sha256:" + "d" * 64
    # The seam leaves the policy as the adapter accepts it.
    record = {"applied": True, "route": {}, "drafter": {"path": "/draft", "pins": pairing["pins"]}}
    args = SimpleNamespace(route_default_selection=record)
    policy = best_route_defaults.finish_best_route_default(args, {"draft_model": "/draft"})
    _, speculative = split_policy(policy)
    adapter = object.__new__(RaonAdapter)
    assert adapter._parse_external_policy(speculative, family="Raon-1-mini")
    assert adapter.external_policy == {"draft_model": "/draft"}


def test_mixed_affine_recipe_binds_storage_variant_and_declared_mode():
    from mlx2.adapters import raon1_mini as A
    from mlx2.runtime.models import raon1_mini as R

    config = dict(A._TOPOLOGY, rda_config=A._RDA,
                  reasoning_effort_steering={"layers": A.STEERING_LAYERS, "default": "high", "alpha": {"high": 0}},
                  mlx2_conversion={"recipe": A.MIX_RECIPE, "source_revision": A.SOURCE_REVISION},
                  quantization={"mode": "affine", "bits": 8, "group_size": 32})
    adapter = object.__new__(A.RaonAdapter)
    adapter.config = config
    adapter.artifact_layout = A.validate_config(config)
    assert adapter.artifact_variant == "mlx2-affine-mix-ordinary"
    receipt = R.install_for_layout(_model(), adapter.artifact_layout, config, ["lm_head.scales"])
    assert receipt["format"] == "mlx2-affine-mix"
    config["quantization"]["lm_head"] = {"mode": "mxfp8", "bits": 8, "group_size": 32}
    with pytest.raises(ValueError, match="mixed affine"):
        A.validate_config(config)


@pytest.mark.parametrize("name", ["alpha_post", "b_post", "alpha_z"])
def test_fused_mhc_rebuilds_replaced_parameters(name):
    from mlx2.runtime.models.raon1_mini_fused import site_params

    site = _model().layers[0].self_attn_raon_mhc
    old = site_params(site)
    assert site_params(site) is old
    owner = site.res if name == "alpha_z" else site
    setattr(owner, name, getattr(owner, name) + 1)
    updated = site_params(site)
    assert not mx.array_equal(old, updated).item()


def test_hq_partial_write_can_be_retried(tmp_path, monkeypatch):
    from mlx2.conversion import raon1_mini_hq as HQ

    dest = tmp_path / "converted"
    def fail(source, staging, *, settings, log):
        (staging / "model-00.safetensors").write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(HQ, "_write_hq", fail)
    with pytest.raises(OSError, match="disk full"):
        HQ.convert_hq(tmp_path / "source", dest)
    assert not dest.exists()
    def succeed(source, staging, *, settings, log):
        (staging / "model-00.safetensors").write_bytes(b"complete")
        return {"ok": True}

    monkeypatch.setattr(HQ, "_write_hq", succeed)
    assert HQ.convert_hq(tmp_path / "source", dest) == {"ok": True}
    assert (dest / "model-00.safetensors").read_bytes() == b"complete"
