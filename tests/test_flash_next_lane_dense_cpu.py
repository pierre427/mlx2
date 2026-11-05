"""Flash-Next ``lane_dense``: the lane matmul on dense projections only (CPU).

A tiny quantized qwen4_exp model stands in for the artifact.  The lane
kernels never run here (no GPU): ``installer.available`` is patched so the
install, its receipt and the adapter's checks run, and every forward used
below stays on stock arithmetic.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.adapters import flash_next_lane as fl
from mlx2.adapters.flash_next import FlashNextAdapter
from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.runtime import lane
from mlx2.runtime import verify_topology as vt
from mlx2.runtime.lane import installer
from mlx2.runtime.lane import policy as lane_policy
from mlx2.runtime.lane.installer import apc_lane_fingerprint


def tiny_flash_next():
    from mlx2.runtime.models.qwen4_exp import Model, ModelArgs

    text_config = dict(
        model_type="qwen4_exp_text", hidden_size=64, intermediate_size=0,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, vocab_size=128, linear_num_value_heads=4,
        linear_num_key_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, layer_types=["linear_attention", "full_attention"],
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=64,
        shared_expert_intermediate_size=64, hc_count=2, hc_lowrank=64,
        ple_layer_ids=[1], ple_embed_dim=64, ple_conv_kernel_size=4, ngram_size=3,
        heads_per_ngram=2, ngram_vocab_size_base=128,
        make_ngram_vocab_size_divisible_by=128, split_ngram_parts=1,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16,
        indexer_budget=8, indexer_compress_ratio=2, mtp_num_hidden_layers=1,
        rope_parameters={"type": "default", "rope_theta": 10000,
                         "partial_rotary_factor": 0.25},
    )
    mx.random.seed(5)
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=text_config))
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=64, bits=4,
                class_predicate=lambda _n, m: hasattr(m, "to_quantized")
                and m.weight.shape[-1] % 64 == 0)
    model.eval()
    mx.eval(model.parameters())
    return model


ON = FlashNextPolicy(lane_dense=True)


@pytest.fixture
def gpu_free_lane(monkeypatch):
    monkeypatch.setattr(installer, "available", lambda: True)
    from mlx2.runtime.models import qwen4_attn_rows

    monkeypatch.setattr(qwen4_attn_rows, "_ENABLED", False)


def _resolve(model, policy=ON, *, overrides=None, mode="auto"):
    return lane_policy.resolve(
        lane_policy.detect(model), family="flash-next",
        adapter=fl.lane_policy_defaults(policy), overrides=overrides, mode=mode)


def _serve(model, policy=ON, **kwargs):
    resolved = _resolve(model, policy, **kwargs)
    fl.validate_lane_policy(policy, model, resolved)
    receipt = lane.apply_policy(model, resolved)
    return receipt, fl.bind_lane_receipt(policy, model, receipt)


# ---------------------------------------------------------------- policy


def test_default_off_and_absent_from_receipts():
    policy = FlashNextPolicy()
    assert policy.lane_dense is False and policy.draft_loop_prior is None
    assert "lane_dense" not in policy.as_dict()
    assert "draft_loop_prior" not in policy.as_dict()
    assert fl.lane_policy_defaults(policy) is None
    assert ON.as_dict()["lane_dense"] is True


@pytest.mark.parametrize("other", [
    {"row_exact_verify": True}, {"invariant_prefill": True},
    {"tensorfold_qmv_rows": True}, {"tensorfold_prefill": True},
    {"fp32_head_logits": True},
])
def test_lane_dense_refuses_mechanisms_that_own_the_projections(other):
    with pytest.raises(ValueError, match="lane_dense cannot be combined"):
        FlashNextPolicy(lane_dense=True, **other)


def test_draft_loop_prior_is_named_bound_and_base_checked():
    with pytest.raises(ValueError, match="requires lane_dense"):
        FlashNextPolicy(draft_loop_prior="qwen38_27b", num_draft=3)
    with pytest.raises(ValueError, match="draft_loop_prior must be one of"):
        FlashNextPolicy(lane_dense=True, draft_loop_prior="flash_next", num_draft=3)
    # The bundled grid is for base depth 3; Flash-Next's default is 2.
    with pytest.raises(ValueError, match="set num_draft to 3"):
        FlashNextPolicy(lane_dense=True, draft_loop_prior="qwen38_27b")
    with pytest.raises(ValueError, match="lane_dense must be boolean"):
        FlashNextPolicy(lane_dense=1)
    policy = FlashNextPolicy(lane_dense=True, draft_loop_prior="qwen38_27b", num_draft=3)
    assert policy.as_dict()["draft_loop_prior"] == "qwen38_27b"
    assert fl.prior_resource(policy) == vt.PRIOR_RESOURCE


# ---------------------------------------------------------------- resolution


def test_auto_resolves_moe_off_without_and_crossover_with_the_opt_in():
    model = tiny_flash_next()
    assert lane_policy.detect(model)["moe"] is True
    assert lane_policy.resolve(lane_policy.detect(model), mode="auto")["mode"] == "off"
    resolved = _resolve(model)
    assert resolved["mode"] == "crossover"
    assert resolved["sources"]["mode"] == "adapter:flash-next.moe"
    assert resolved["skip"] == [pattern for pattern, _ in fl.EXCLUSIONS]
    # --lane-matmul off still wins.
    assert _resolve(model, mode="off")["mode"] == "off"


# ---------------------------------------------------------------- coverage


DENSE = {
    "language_model.lm_head",
    "language_model.model.layers.*.linear_attn.in_proj_qkv",
    "language_model.model.layers.*.linear_attn.in_proj_z",
    "language_model.model.layers.*.linear_attn.in_proj_b",
    "language_model.model.layers.*.linear_attn.in_proj_a",
    "language_model.model.layers.*.linear_attn.out_proj",
    "language_model.model.layers.*.self_attn.q_proj",
    "language_model.model.layers.*.self_attn.k_proj",
    "language_model.model.layers.*.self_attn.v_proj",
    "language_model.model.layers.*.self_attn.o_proj",
    "language_model.model.layers.*.self_attn.indexer.index_qk_proj",
    "mtp.fc_embedding",
    "mtp.fc_hidden",
    "mtp.layers.*.self_attn.q_proj",
    "mtp.layers.*.self_attn.k_proj",
    "mtp.layers.*.self_attn.v_proj",
    "mtp.layers.*.self_attn.o_proj",
    "mtp.layers.*.self_attn.indexer.index_qk_proj",
}


def test_coverage_is_the_dense_projections_and_moe_is_untouched(gpu_free_lane):
    model = tiny_flash_next()
    excluded_before = {
        name: (type(module), module.get("weight"), module.get("scales"))
        for name, module in model.named_modules()
        if fl._excluded(name) is not None and "weight" in module
    }
    x = (mx.random.normal((1, 9, 64), key=mx.random.key(1)) * 0.5).astype(mx.bfloat16)
    layer = model.language_model.model.layers[1]
    moe_before = layer.mlp(x)
    mx.eval(moe_before)

    receipt, report = _serve(model)
    assert set(report["covered"]) == DENSE
    assert sum(report["covered"].values()) == receipt["covered"]["affine-q4-g64"]
    for name, module in model.named_modules():
        assert installer.installed(module) == (
            fl._pattern(name) in DENSE and fl._excluded(name) is None), name
    # Every excluded projection kept its class and its very arrays.
    for name, (kind, weight, scales) in excluded_before.items():
        module = dict(model.named_modules())[name]
        assert type(module) is kind and module.get("weight") is weight
        assert module.get("scales") is scales
    assert any(".mlp.switch_mlp." in k for k in report["excluded"])
    assert {k for k in report["excluded"] if "hyper_connection" in k}
    assert {k for k in report["excluded"] if ".ple." in k}
    # The MoE block's arithmetic is bitwise unchanged.
    moe_after = layer.mlp(x)
    assert bool(mx.array_equal(moe_before, moe_after).item())
    lane.uninstall(model)


def test_numerics_law_is_recorded_and_namespaced(gpu_free_lane):
    dense = tiny_flash_next()
    receipt, report = _serve(dense)
    assert report["law_id"] == receipt["law_id"]
    assert "+skip[" in receipt["law_id"] and receipt["law_id"].endswith("+grouped")
    assert report["state"] == "implemented-unqualified"
    # The namespace also carries the installed arithmetic-class fingerprint
    # (device, MLX build, kernel digest) since 7af38665c.
    assert apc_lane_fingerprint("base", receipt) == (
        "base", "lane-matmul", receipt["law_id"], receipt["arithmetic_class"]["fingerprint"]
    )
    # An operator crossover over the whole model (no exclusions) is another law.
    whole = tiny_flash_next()
    other = lane.apply_policy(whole, lane_policy.resolve(
        lane_policy.detect(whole), mode="crossover"))
    assert other["law_id"] != receipt["law_id"]
    assert apc_lane_fingerprint("base", other) != apc_lane_fingerprint("base", receipt)
    lane.uninstall(dense)
    lane.uninstall(whole)


# ---------------------------------------------------------------- fail closed


@pytest.mark.parametrize("receipt", [
    None,
    {"law_id": "stock", "covered": {}, "policy": {"mode": "off"}},
    {"law_id": "stock", "covered": {}, "available": False,
     "refused": {"device_unsupported": 3}, "policy": {"mode": "crossover"}},
    {"law_id": "tensorfold-lane", "covered": {"q4": 3}},
])
def test_bind_refuses_when_the_lane_does_not_cover_the_model(receipt):
    model = tiny_flash_next()
    with pytest.raises(ValueError, match="lane_dense"):
        fl.bind_lane_receipt(ON, model, receipt)


def test_validate_refuses_lane_off_and_dropped_exclusions(gpu_free_lane):
    model = tiny_flash_next()
    with pytest.raises(ValueError, match="resolved to off"):
        fl.validate_lane_policy(ON, model, _resolve(model, mode="off"))
    # An operator skip list replaces the adapter's: the MoE block would be covered.
    with pytest.raises(ValueError, match="keeps on their kernels"):
        fl.validate_lane_policy(ON, model, _resolve(model, overrides={"skip": []}))


def test_exact_mode_collides_with_fused_attention_rows(gpu_free_lane, monkeypatch):
    from mlx2.runtime.models import qwen4_attn_rows

    model = tiny_flash_next()
    exact = _resolve(model, mode="exact")
    fl.validate_lane_policy(FlashNextPolicy(lane_dense=True, attn_fused_rows=False),
                            model, exact)
    with pytest.raises(ValueError, match="attn_fused_rows: .*q_proj"):
        fl.validate_lane_policy(ON, model, exact)  # attn_fused_rows on by default
    # Crossover (q4 from 8 rows) stays above the fused table's 7 rows.
    monkeypatch.setattr(qwen4_attn_rows, "_ENABLED", True)
    fl.validate_lane_policy(ON, model, _resolve(model))


def test_gdn_fused_input_table_collides_at_eight_rows(gpu_free_lane):
    model = tiny_flash_next()
    gdn = model.language_model.model.layers[0].linear_attn
    gdn.set_gdn_fused_inproj(True)
    with pytest.raises(ValueError, match="gdn_fused_inproj: .*in_proj_qkv"):
        fl.validate_lane_policy(ON, model, _resolve(model))
    # Also caught after install (the switch is live per layer).
    gdn.set_gdn_fused_inproj(False)
    resolved = _resolve(model)
    fl.validate_lane_policy(ON, model, resolved)
    receipt = lane.apply_policy(model, resolved)
    gdn.set_gdn_fused_inproj(True)
    with pytest.raises(ValueError, match="refused after install"):
        fl.bind_lane_receipt(ON, model, receipt)
    assert not any(installer.installed(m) for _n, m in model.named_modules())


def test_bind_refuses_an_installed_excluded_projection(gpu_free_lane):
    model = tiny_flash_next()
    resolved = _resolve(model)
    receipt = lane.apply_policy(
        model, {**resolved, "skip": [p for p in resolved["skip"] if "hyper" not in p]})
    with pytest.raises(ValueError, match="excluded projection installed: .*hyper_connection"):
        fl.bind_lane_receipt(ON, model, receipt)
    assert not any(installer.installed(m) for _n, m in model.named_modules())


# ---------------------------------------------------------------- adapter hooks


def _bare_adapter(policy, model=None):
    adapter = object.__new__(FlashNextAdapter)
    adapter.policy = policy
    adapter.model = model
    adapter._bind_draft_loop_prior()
    return adapter


def test_adapter_hooks_are_inert_without_the_opt_in_or_a_policy():
    model = tiny_flash_next()
    for adapter in (_bare_adapter(FlashNextPolicy(), model), _bare_adapter(None, model)):
        assert adapter.lane_policy_defaults() is None
        adapter.validate_serving_lane_policy({"mode": "off"})
        assert adapter.bind_serving_lane_matmul(None) is None
        assert adapter.draft_loop_prior_resource is None
    # Subclasses keep their own hooks and prior.
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter

    assert Qwen3827BAdapter.draft_loop_prior_resource == vt.PRIOR_RESOURCE
    assert Qwen3827BAdapter.bind_serving_lane_matmul is not FlashNextAdapter.bind_serving_lane_matmul
    assert FlashNextAdapter.draft_loop_prior_resource is None


def test_draft_loop_engages_under_the_opt_in(gpu_free_lane, monkeypatch):
    from mlx2.serving import ServingEngine

    policy = FlashNextPolicy(lane_dense=True, draft_loop_prior="qwen38_27b", num_draft=3)
    model = tiny_flash_next()
    adapter = _bare_adapter(policy, model)
    adapter.tokenizer = SimpleNamespace(encode=lambda text: [1, 2, 3])
    adapter.identity = {"fingerprint": "f"}
    adapter.environment = {}
    resolved = _resolve(model, policy)
    adapter.validate_serving_lane_policy(resolved)
    receipt = lane.apply_policy(model, resolved)
    report = adapter.bind_serving_lane_matmul(receipt)
    assert report["draft_loop_prior"]["borrowed"] is True
    assert report["draft_loop_prior"]["measured_on"] == "qwen3.8-27b"

    seen = {}

    def resolve(_model, **kwargs):
        seen.update(kwargs)
        return {"selected": {"by_width": {"1": [3, 7]}, "threshold": -0.4, "cohort": "any"}}

    monkeypatch.setattr(vt, "resolve_topology", resolve)
    engine = SimpleNamespace(
        draft_loop_mode="auto", draft_loop_threshold=-0.4, draft_loop_widths=[1, 2],
        mtp=True, max_lanes=1, verify_topology_receipt=None, lane_matmul_receipt=receipt)
    out = ServingEngine._select_draft_loop(engine, adapter, {"num_draft": 3})
    assert out["draft_loop"]["by_width"] == {"1": [3, 7]}
    assert seen["identity"]["lane_law"] == receipt["law_id"]
    assert seen["identity"]["draft_loop_prior_resource"] == vt.PRIOR_RESOURCE
    assert seen["vocab_size"] == 128 and seen["base"] == 3

    # Without the prior the opt-in still covers the model, but auto stays off.
    plain = _bare_adapter(ON, model)
    plain.tokenizer, plain.identity, plain.environment = adapter.tokenizer, {"fingerprint": "f"}, {}
    monkeypatch.setattr(vt, "resolve_topology", lambda *a, **k: pytest.fail("probed"))
    config = {"num_draft": 3}
    assert ServingEngine._select_draft_loop(engine, plain, config) is config
    assert "no source-bound" in engine.verify_topology_receipt["reason"]
    lane.uninstall(model)
