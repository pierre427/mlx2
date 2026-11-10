"""Flash-Next int8 prefill declaration and projection selection (CPU only).

No Metal kernel and no ``mx.eval`` of a kernel: a small qwen4_exp model is
quantized like the served checkpoints (routed experts 4-bit gs64, every
other projection 8-bit gs64) and the adapter's declared scopes, selector and
``checkpoint_census`` / ``auto_policy`` are checked against it.
"""

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.adapters.flash_next import FlashNextAdapter
from mlx2.runtime import int8_prefill as ip


def _model():
    from mlx2.runtime.models.qwen4_exp import Model, ModelArgs

    # Dimensions >= int8 prefill's MIN_DIM (256) so the trunk projections are
    # structurally eligible, as they are on the artifact.
    text_config = dict(
        model_type="qwen4_exp_text", hidden_size=256, intermediate_size=0,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        head_dim=64, vocab_size=256, linear_num_value_heads=4,
        linear_num_key_heads=4, linear_key_head_dim=64, linear_value_head_dim=64,
        linear_conv_kernel_dim=4, layer_types=["linear_attention", "full_attention"],
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=256,
        shared_expert_intermediate_size=256, hc_count=2, hc_lowrank=64,
        ple_layer_ids=[1], ple_embed_dim=256, ple_conv_kernel_size=4, ngram_size=3,
        heads_per_ngram=2, ngram_vocab_size_base=256,
        make_ngram_vocab_size_divisible_by=256, split_ngram_parts=1,
        indexer_n_heads=4, indexer_kv_heads=1, indexer_head_dim=64,
        indexer_budget=8, indexer_compress_ratio=2, mtp_num_hidden_layers=1,
        rope_parameters={"type": "default", "rope_theta": 10000,
                         "partial_rotary_factor": 0.25},
    )
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=text_config))
    model.set_dtype(mx.bfloat16)  # checkpoint scales/biases are bf16

    def bits(path, module):
        if not hasattr(module, "to_quantized") or module.weight.shape[-1] % 64:
            return False
        if "switch_mlp" in path.split("."):
            return {"group_size": 64, "bits": 4, "mode": "affine"}
        return {"group_size": 64, "bits": 8, "mode": "affine"}

    nn.quantize(model, group_size=64, bits=8, class_predicate=bits)
    return model


@pytest.fixture(scope="module")
def adapter():
    adapter = object.__new__(FlashNextAdapter)
    adapter.model = _model()
    return adapter


def _chosen(adapter, scope):
    select = adapter.int8_prefill_select(scope)
    return {path for path, _ in adapter.model.named_modules() if select(path, _)}


def _leaf(path):
    return ".".join(path.split(".")[-2:])


def test_declares_mlp_and_all(adapter):
    assert ip.adapter_scopes(adapter) == {"mlp", "all"}
    with pytest.raises(ValueError):
        FlashNextAdapter.int8_prefill_select("experts")


def test_mixed_prefill_decode_withdraws_the_declaration():
    """A mixed forward's row count includes its decode rows."""
    from mlx2.adapters.flash_next_policy import FlashNextPolicy

    mixed = object.__new__(FlashNextAdapter)
    mixed.policy = FlashNextPolicy(mixed_prefill_decode=True, prefill_plan_exact_slices=False)
    assert ip.adapter_scopes(mixed) == frozenset()
    with pytest.raises(ip.Int8PrefillError, match="does not declare"):
        ip.apply_for_adapter(mixed, ip.Int8PrefillPolicy(enabled=True, scope="mlp"))


def test_mlp_scope_takes_shared_expert_projections_only(adapter):
    chosen = _chosen(adapter, "mlp")
    assert {_leaf(p) for p in chosen} == {
        "shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert.down_proj"}
    assert len(chosen) == 3 * 2  # both trunk layers, not the MTP layer


def test_all_scope_adds_attention_and_gdn_projections(adapter):
    chosen = _chosen(adapter, "all")
    assert {_leaf(p) for p in chosen} == {
        "shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert.down_proj",
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
    }
    assert all(p.startswith("language_model.model.layers.") for p in chosen)


def test_hc_indexer_ple_router_experts_and_mtp_stay_stock(adapter):
    from mlx2.runtime.models.flash_tensorfold_qmv import hc_owned_projection_ids

    chosen = _chosen(adapter, "all")
    modules = dict(adapter.model.named_modules())
    hc_owned = hc_owned_projection_ids(adapter.model)
    assert hc_owned, "fixture lost its hyper-connection projections"
    assert not {id(modules[p]) for p in chosen} & hc_owned
    for path in chosen:
        segments = set(path.split("."))
        assert not segments & {
            "indexer", "ple", "switch_mlp", "gate", "shared_expert_gate",
            "in_proj_a", "in_proj_b", "mtp", "lm_head",
        }, path
        assert "hyper_connection" not in path, path


def test_census_is_q8_only_so_auto_picks_w8a8_not_w4a8(adapter):
    """The 4-bit weights are the routed experts, outside every scope."""
    for scope in ("mlp", "all"):
        census = ip.checkpoint_census(adapter, scope)
        assert census["q45"] == 0, census
        assert census["q8"] == len(_chosen(adapter, scope)), census
        assert census["other_quantized"] == census["linear"] == 0, census
    policy = ip.auto_policy(adapter, ip.checkpoint_census(adapter, "all"))
    assert policy.scope == "all"
    assert policy.q8_inplace and policy.q8_only and not policy.q45_inplace


def test_auto_stays_off_without_qualification_evidence(adapter, monkeypatch):
    monkeypatch.setattr(ip, "device_support", lambda: (True, "test M5"))
    policy, receipt = ip.resolve_auto(
        adapter, qualification_mode=False, qualified_settings=None,
        max_prefill_rows=8192 * 4)
    assert policy is None and receipt["resolved"] == "off", receipt
    policy, receipt = ip.resolve_auto(
        adapter, qualification_mode=True, qualified_settings=None,
        max_prefill_rows=8192 * 4)
    assert policy is not None and policy.q8_only and not policy.q45_inplace, receipt


@pytest.mark.parametrize("scope", ["mlp", "all"])
def test_installed_wrapper_keeps_the_shared_expert_decode_fold(scope, monkeypatch):
    """The default routed decode folds the shared expert by reading its arrays;
    the int8 class swap must not make that admission decline."""
    from mlx2.runtime.models import qwen3_next
    from mlx2.runtime.models import qwen4_routed_decode as routed

    adapter = object.__new__(FlashNextAdapter)
    adapter.model = _model()
    blocks = [layer.mlp for layer in adapter.model.language_model.model.layers]

    def admissions():
        out = []
        for block in blocks:
            shared = block.shared_expert
            out.append((
                routed.admit_shared_fold(shared, 256, 256).reason,
                [routed._linear_ok(shared[name], 8, *shared[name]["weight"].shape[:1],
                                   shared[name]["weight"].shape[1] * 4)
                 for name in ("gate_proj", "up_proj", "down_proj")],
                [qwen3_next._base_linear_ok(shared[name])
                 for name in ("gate_proj", "up_proj", "down_proj")],
            ))
        return out

    before = admissions()
    assert all(all(ok) for _, _, ok in before)
    monkeypatch.setattr(ip, "require_supported_device", lambda: "test M5")
    policy = ip.Int8PrefillPolicy(
        enabled=True, scope=scope, q8_inplace=True, act_scale="group64", q8_only=True)
    handle = ip.apply_for_adapter(adapter, policy)
    try:
        assert handle.q8_module_count() == len(_chosen(adapter, scope))
        swapped = blocks[0].shared_expert.gate_proj
        assert type(swapped) is not nn.QuantizedLinear
        assert type(swapped)._mlx2_int8_prefill_base is nn.QuantizedLinear
        assert admissions() == before
    finally:
        ip.remove(handle)
