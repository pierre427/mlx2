"""Synthetic Ling load/stream parity on CPU; no real checkpoint qualification."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from mlx2.adapters import ling3_flash
from mlx2.runtime.streamed_load import WeightStreamRequest


@pytest.fixture
def tiny_ling(tmp_path, monkeypatch, request):
    from transformers import AutoTokenizer

    from mlx2.adapters.ordinary_artifact import inspect_indexed_artifact
    from mlx2.runtime import tokenizer_integrity, tokenizer_utils
    from mlx2.runtime.models.bailing_hybrid import Model, ModelArgs

    config = dict(  # noqa: C408 -- configuration keyword names mirror ModelArgs
        model_type="bailing_hybrid", vocab_size=128, hidden_size=128,
        intermediate_size=128, moe_intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=2, num_experts=4,
        num_experts_per_tok=2, num_shared_experts=1, n_group=1, topk_group=1,
        first_k_dense_replace=1, layer_group_size=2, max_position_embeddings=64,
        rms_norm_eps=1e-6, rope_theta=10000, routed_scaling_factor=1.0,
        head_dim=64, kv_lora_rank=64, qk_rope_head_dim=32,
        qk_nope_head_dim=64, v_head_dim=64, no_kda_lora=True,
        gated_attention_proj_granularity_type="head_wise", eos_token_id=127,
        kda_safe_gate=True, kda_lower_bound=-5.0,
    )
    quant = {"group_size": 64, "bits": 4, "mode": "affine"}
    for layer in (1, 2):
        for projection, bits in (("up_proj", 6), ("down_proj", 8)):
            quant[f"model.layers.{layer}.mlp.switch_mlp.{projection}"] = {
                "group_size": 64, "bits": bits, "mode": "affine"
            }
    model = Model(ModelArgs.from_dict(config))
    nn.quantize(
        model, group_size=64, bits=4,
        class_predicate=lambda name, module: quant.get(name, hasattr(module, "to_quantized")),
    )
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    config["quantization"] = quant
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in weights}
    }))
    identity = inspect_indexed_artifact(tmp_path)["identity"]["fingerprint"]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True,
    ).strip()
    request.node.user_properties.extend([
        ("checkpoint", f"synthetic-ling-mixed-precision@sha256:{identity}"),
        ("source_revision", revision),
        ("working_tree_modified", "true"),
        ("device", "cpu"),
    ])
    monkeypatch.setattr(ling3_flash, "inspect_artifact", inspect_indexed_artifact)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(tokenizer_integrity, "repair_loaded_tokenizer", lambda *a, **k: {})
    monkeypatch.setattr(tokenizer_utils, "TokenizerWrapper", lambda *a, **k: SimpleNamespace())
    return tmp_path


def test_mixed_precision_stream_matches_resident_prefill_decode_and_state(tiny_ling):
    resident = ling3_flash.Ling3FlashAdapter(tiny_ling)
    streamed = None
    try:
        streamed = ling3_flash.Ling3FlashAdapter(tiny_ling, weight_streaming=WeightStreamRequest(
            mode="moe_experts", budget_bytes=128 << 10, read_workers=2,
        ))
        for layer in (1, 2):
            expert = resident.model.layers[layer].mlp.switch_mlp
            assert (expert.gate_proj.bits, expert.up_proj.bits, expert.down_proj.bits) == (4, 6, 8)
        caches = [adapter.model.make_cache() for adapter in (resident, streamed)]
        for tokens in ([[1, 2, 3]], [[4]], [[5]]):
            outputs = [adapter.model(mx.array(tokens), cache=cache)
                       for adapter, cache in zip((resident, streamed), caches)]
            mx.eval(outputs)
            assert mx.array_equal(*outputs).item()
            for left, right in zip(*caches):
                for (_, a), (_, b) in zip(tree_flatten(left.state), tree_flatten(right.state)):
                    assert mx.array_equal(a, b).item()
        assert streamed.weight_stream is not None
        assert streamed.weight_stream.plan.resident_remainder_bytes > 0
        assert streamed.weight_stream.plan.capacity_experts < 4
        assert streamed.weight_stream.receipt()["load_phase"] == "pre_materialization"
        assert streamed.weight_stream.stats.phase == "serve"
    finally:
        resident.close()
        if streamed is not None:
            streamed.close()
            assert streamed.weight_stream is None
            streamed.close()


def test_stream_is_closed_if_tokenizer_load_fails(tiny_ling, monkeypatch):
    from transformers import AutoTokenizer

    from mlx2.runtime import streamed_load

    managers = []
    load = streamed_load.load_streamed

    def capture(*args, **kwargs):
        result = load(*args, **kwargs)
        managers.append(result.manager)
        close = result.manager.close
        def closed():
            managers.append("closed")
            close()
        result.manager.close = closed
        return result

    def fail(*args, **kwargs):
        raise RuntimeError("tokenizer failed")

    monkeypatch.setattr(streamed_load, "load_streamed", capture)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", fail)
    with pytest.raises(RuntimeError, match="tokenizer failed"):
        ling3_flash.Ling3FlashAdapter(tiny_ling, weight_streaming=WeightStreamRequest(
            mode="moe_experts", budget_bytes=2 << 20, read_workers=2,
        ))
    assert len(managers) == 2 and managers[1] == "closed"
    from mlx2.adapters.process_globals import live_selections
    assert live_selections() == []


def test_unsupported_stream_mode_refused_before_artifact_load(monkeypatch):
    monkeypatch.setattr(ling3_flash, "inspect_artifact", lambda *a: pytest.fail("artifact opened"))
    with pytest.raises(ValueError, match="does not declare dense_mlp"):
        ling3_flash.Ling3FlashAdapter("/absent", weight_streaming=WeightStreamRequest(
            mode="dense_mlp", budget_bytes=1 << 20,
        ))


def test_bound_ubtree_selects_external_serving_backend():
    adapter = object.__new__(ling3_flash.Ling3FlashAdapter)
    adapter.draft_model = SimpleNamespace()
    adapter.external_policy = {"num_draft": 15, "cost_per_commit_handoff": False}
    config = adapter.execution_config(max_lanes=1, prefill_step=512)
    assert config["backend"] == "external_draft"
    assert config["num_draft"] == 15
    assert config["cost_per_commit_handoff"] is False


def test_unbound_ling_preserves_ordinary_serving_backend():
    adapter = object.__new__(ling3_flash.Ling3FlashAdapter)
    adapter.draft_model = None
    config = adapter.execution_config(max_lanes=1, prefill_step=512)
    assert config.get("backend", "ordinary") == "ordinary"
    assert config["num_draft"] == 0


@pytest.mark.parametrize("rows", [4, 8, 16, 32])
def test_explicit_tree_budget_reaches_batch_and_revisions_state(rows, monkeypatch):
    from mlx2.adapters.external_draft_policy import ExternalDraftAdapterMixin

    adapter = object.__new__(ling3_flash.Ling3FlashAdapter)
    adapter.external_policy = {"ast_tile_rows": rows, "ast_adaptive_rows": False}
    adapter.draft_model = SimpleNamespace(receipt_kind="external_ubtree")
    adapter.identity = {"fingerprint": "base"}
    adapter.layout = ling3_flash.CACHE_LAYOUT
    adapter.descriptor = ling3_flash.DESCRIPTOR
    adapter._bind_tree_controls()
    monkeypatch.setattr(ExternalDraftAdapterMixin, "create_external_batch", lambda self, **kwargs: kwargs)
    kwargs = adapter.create_external_batch()
    assert kwargs["tree_node_budget_by_lanes"][1] == rows - 1
    config = adapter.execution_config(max_lanes=4, prefill_step=512)
    assert config["tree_node_budget_by_lanes"] == kwargs["tree_node_budget_by_lanes"]
    assert config["dynamic_tree_overflow"] == "cohort"
    assert kwargs["ast_adaptive_rows"] is False
    assert kwargs["tensorfold_cohort_limit"] == 4
    assert adapter.draft_model.served_tree_budget == rows - 1
    assert adapter.descriptor.variant == f"oq4-ubtree-ast-b{rows - 1}"
    assert adapter.identity["fingerprint"] != "base"
    assert adapter.descriptor.cache_layout == adapter.layout


@pytest.mark.parametrize("policy", [
    {"ast_tile_rows": 3}, {"ast_tile_rows": 33}, {"ast_tile_rows": True},
    {"ast_target_executor": "qwen"},
    {"ast_adaptive_rows": "off"}, {"ast_adaptive_rows": 0},
])
def test_invalid_tree_controls_fail_closed(policy):
    with pytest.raises(ValueError):
        ling3_flash._tree_controls(policy)


def test_tensorfold_format_completion_covers_only_unsupported_projections(monkeypatch):
    from mlx2.runtime import lane

    adapter = object.__new__(ling3_flash.Ling3FlashAdapter)
    adapter.model = object()
    captured = {}

    def install(model, **kwargs):
        assert model is adapter.model
        captured.update(kwargs)
        return {"law_id": "owned-mpp-test", "covered": {"q8-g128": 3}}

    monkeypatch.setattr(lane, "install", install)
    receipt = {"law_id": "tensorfold-lane-qmm-v1+fused-groups"}
    adapter.bind_serving_lane_matmul(receipt)
    native = nn.QuantizedLinear(128, 128, group_size=64, bits=4)
    gap = nn.QuantizedLinear(128, 128, group_size=128, bits=8)
    native.set_dtype(mx.bfloat16)
    gap.set_dtype(mx.bfloat16)
    assert captured["skip"]("native", native)
    assert not captured["skip"]("gap", gap)
    assert receipt["format_completion"]["covered"] == {"q8-g128": 3}
    assert receipt["law_id"].endswith(":ling-format-completion:owned-mpp-test")


def test_packed_ast_defaults_to_exact_projection_law_and_refuses_uncovered_stock():
    from mlx2.runtime.lane.policy import resolve

    adapter = object.__new__(ling3_flash.Ling3FlashAdapter)
    adapter.draft_model = SimpleNamespace()
    adapter.external_policy = {}
    adapter.model = SimpleNamespace(named_modules=lambda: [("projection", nn.Linear(128, 128))])
    policy = resolve({"moe": True, "backend": "mpp"}, adapter=adapter.lane_policy_defaults())
    assert policy["mode"] == "exact"
    with pytest.raises(ValueError, match="complete row-invariant"):
        adapter.bind_serving_lane_matmul(None)
    adapter.external_policy = {"ast_target_executor": "serial"}
    assert adapter.lane_policy_defaults() == {}
    adapter.bind_serving_lane_matmul(None)


def test_ubtree_pld_graft_copies_specific_history_and_revisions_identity():
    class Base:
        receipt_kind = "external_ubtree"
        config = SimpleNamespace(block_size=16, target_layer_ids=(0,))
        @property
        def receipt_settings(self):
            return {"algorithm": "ubtree-pruned-v1"}

        def propose_tree(self, *args, **kwargs):
            return [([50, 51], [-1, 0])]

    adapter = object.__new__(ling3_flash.Ling3FlashAdapter)
    adapter.model = SimpleNamespace(
        speculative_args=SimpleNamespace(vocab_size=100, hidden_size=8), layers=[object()]
    )
    adapter.draft_model = base = Base()
    adapter.identity = {"fingerprint": "base-target-and-draft"}
    adapter.layout = ling3_flash.CACHE_LAYOUT
    adapter.descriptor = ling3_flash.DESCRIPTOR
    assert adapter._parse_external_policy({
        "draft_model": "/models/ubtree", "ast_graft_prompt_lookup": True,
    }, family="Ling")
    adapter.draft_model = base
    adapter._graft_prompt_lookup()
    tokens, _ = adapter.draft_model.propose_tree(
        [12], None, [], 15,
        processor_histories=[[10, 11, 12, 13, 14, 20, 10, 11]],
    )[0]
    assert tokens[:5] == [13, 14, 20, 10, 11]
    assert adapter.draft_model.stats["copy_first"] == 1
    assert adapter.identity["fingerprint"] != "base-target-and-draft"
    assert adapter.descriptor.cache_layout == adapter.layout
    assert adapter.execution_config(max_lanes=1, prefill_step=512)["backend"] == "external_draft"
    assert adapter.diagnostics()["prompt_lookup_graft"]["copy_first"] == 1


def test_checkpoint_bounded_kda_matches_independent_recurrence(tiny_ling, monkeypatch):
    from mlx2.runtime.models import bailing_hybrid
    from mlx2.runtime.models.cache import ArraysCache

    args = bailing_hybrid.ModelArgs.from_dict(json.loads((tiny_ling / "config.json").read_text()))
    layer = bailing_hybrid.KimiDeltaAttention(args)
    # Zero gate input makes the bound diagnostic: exp(-2.5), whereas the
    # former unbounded softplus gate produces 0.5.
    layer.A_log = mx.zeros((2,))
    layer.dt_bias = mx.zeros((128,))
    layer.f_proj.weight = mx.zeros_like(layer.f_proj.weight)
    layer.b_proj.weight = mx.zeros_like(layer.b_proj.weight)
    cache = ArraysCache(size=4)
    cache[3] = mx.ones((1, 2, 64, 64))
    update = bailing_hybrid.gated_delta_update
    checked = []

    def oracle(q, k, v, a, b, A_log, dt_bias, **kwargs):
        result, state = update(q, k, v, a, b, A_log, dt_bias, **kwargs)
        expected = np.array(kwargs["state"], copy=True)
        key, value = np.array(k), np.array(v)
        for t in range(key.shape[1]):
            expected *= np.float32(np.exp(-2.5))
            prediction = (expected * key[:, t, :, None, :]).sum(axis=-1)
            delta = (value[:, t] - prediction) * np.float32(0.5)
            expected += delta[..., None] * key[:, t, :, None, :]
        np.testing.assert_allclose(np.array(state), expected, rtol=2e-5, atol=2e-6)
        checked.append(True)
        return result, state

    monkeypatch.setattr(bailing_hybrid, "gated_delta_update", oracle)
    mx.eval(layer(mx.ones((1, 3, 128)), cache=cache))
    assert checked == [True]
    assert ling3_flash.CACHE_LAYOUT == "ling3-flash-hybrid-kda-bounded-mla-v3"
