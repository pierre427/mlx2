import json

import pytest

from mlx2.adapters import ling32_air, registry
from mlx2.contracts import Capability


def test_descriptor_declares_tools_and_card_sampling():
    assert Capability.TOOLS in ling32_air.DESCRIPTOR.capabilities
    assert ling32_air.DESCRIPTOR.family == "ling-3.2-air"
    assert ling32_air.DESCRIPTOR.metadata["qualified"] is False
    defaults = ling32_air.SAMPLING
    assert "1.0" in repr(defaults) or "temperature=1.0" in repr(defaults)


def test_recipe_detection_and_registry_dispatch(tmp_path, monkeypatch):
    config = {"model_type": "bailing_hybrid",
              "mlx2_conversion": {"recipe": ling32_air.RECIPE}}
    assert ling32_air.is_ling32_air(config)
    assert not ling32_air.is_ling32_air({"model_type": "bailing_hybrid"})
    seen = {}
    monkeypatch.setattr(ling32_air, "inspect_artifact",
                        lambda path: seen.setdefault("p", {"path": path, "has_mtp": True}))
    resolution = registry._ling3(tmp_path, config)
    assert resolution.adapter_type is ling32_air.Ling32AirAdapter
    assert seen["p"]["path"] == tmp_path
    assert Capability.MTP in resolution.descriptor.capabilities


def test_inspector_refuses_unpinned_conversion(tmp_path, monkeypatch):
    monkeypatch.setattr(ling32_air, "inspect_indexed_artifact", lambda _path: {
        "config": {"mlx2_conversion": {"recipe": ling32_air.RECIPE, "source_revision": "x"}},
        "weight_map": {}})
    with pytest.raises(ValueError, match="pinned"):
        ling32_air.inspect_artifact(tmp_path)


def test_template_messages_decode_string_arguments():
    adapter = object.__new__(ling32_air.Ling32AirAdapter)
    request = {"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "f", "arguments": json.dumps({"a": 1})}}]},
        {"role": "tool", "content": "ok"}]}
    messages = adapter._template_messages(request)
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == {"a": 1}
    # The request itself is not mutated.
    assert isinstance(request["messages"][1]["tool_calls"][0]["function"]["arguments"], str)
    bad = {"messages": [{"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "f", "arguments": "[1]"}}]}]}
    with pytest.raises(ValueError, match="JSON object"):
        adapter._template_messages(bad)


def test_page_in_reads_every_parameter_once():
    import mlx.core as mx
    from mlx import nn

    adapter = object.__new__(ling32_air.Ling32AirAdapter)
    adapter.model = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))
    receipt = adapter._page_in_weights()
    expected = sum(v.nbytes for v in (adapter.model.layers[0].weight, adapter.model.layers[0].bias,
                                      adapter.model.layers[1].weight, adapter.model.layers[1].bias))
    assert receipt["bytes"] == expected and receipt["seconds"] >= 0


def test_kernel_defaults_are_air_only():
    from mlx2.adapters import ling3_flash

    flash = ling3_flash.optimization_policy(None, ling3_flash.Ling3FlashAdapter.optimization_overrides)
    assert not any(flash.values())  # Ling 3 Flash: every optimization stays off
    air = ling3_flash.optimization_policy(None, ling32_air.Ling32AirAdapter.optimization_overrides)
    assert air["kda_fused"] is True and air["moe_nax"] is True
    assert air["routed_moe_decode"] is False
    assert sum(air.values()) == 2
    opt_out = ling3_flash.optimization_policy({"kda_fused": False},
                                              ling32_air.Ling32AirAdapter.optimization_overrides)
    assert opt_out["kda_fused"] is False
    with pytest.raises(ValueError, match="override"):
        ling3_flash.optimization_policy(None, {"nope": True})


def _pinned_config(**changes):
    config = {"mlx2_conversion": {"recipe": ling32_air.RECIPE,
                                  "source_revision": ling32_air.SOURCE_REVISION}}
    config.update({k: v for k, v in ling32_air.PINNED_LAWS.items() if v is not None})
    config.update(changes)
    return {key: value for key, value in config.items() if value is not ...}


@pytest.mark.parametrize("changes", [
    {"rope_scaling": {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144}},
    {"rope_parameters": {"rope_type": "yarn", "factor": 4.0}},
    {"rope_interleave": False},
    {"mrope_section": [16, 8, 8]},
    {"linear_silu": False},
    {"group_norm_size": 4},
    {"use_kda_lora": True},
    {"use_qk_norm": False},
    {"use_qk_norm": 1},
    {"num_kv_heads_for_linear_attn": 8},
])
def test_inspector_refuses_unreviewed_config_laws(tmp_path, monkeypatch, changes):
    # 2026-11-25 PR sweep (sglang #52593 reads a YaRN block from config.json):
    # a config law the runtime hard-codes must fail closed, not serve silently.
    monkeypatch.setattr(ling32_air, "inspect_indexed_artifact",
                        lambda _path: {"config": _pinned_config(**changes), "weight_map": {}})
    with pytest.raises(ValueError, match="config law is not the reviewed one"):
        ling32_air.inspect_artifact(tmp_path)


def test_inspector_accepts_released_laws_and_absent_interleave(tmp_path, monkeypatch):
    # The released config omits rope_interleave (runtime default True) and the
    # rope scaling blocks; the law check passes and the topology check runs next.
    config = _pinned_config(rope_interleave=...)
    assert "rope_interleave" not in config and "rope_scaling" not in config
    monkeypatch.setattr(ling32_air, "inspect_indexed_artifact",
                        lambda _path: {"config": config, "weight_map": {}})
    with pytest.raises(ValueError, match="topology"):
        ling32_air.inspect_artifact(tmp_path)


def test_video_placeholder_tokens_are_refused_in_every_mode():
    adapter = object.__new__(ling32_air.Ling32AirAdapter)
    adapter.config = {"image_token_id": 157157, "video_token_id": 156909}
    for vision in (False, True):
        adapter.vision_enabled = vision
        with pytest.raises(ValueError, match="video tokens"):
            adapter.validate_prefill_inputs({}, [1, 156909, 2], None)
        with pytest.raises(ValueError, match="video tokens"):
            adapter.validate_prefill_inputs(
                {"_mlx2_media_fingerprint": "f"}, [157157, 156909],
                {"pixel_values": 0, "image_grid_thw": 0, "_mlx2_rope_media_end": 1})
        assert adapter.validate_prefill_inputs({}, [1, 2, 3], None) is None


def _air_config():
    return {"num_hidden_layers": 42, "layer_group_size": 6, "kv_lora_rank": 512,
            "qk_rope_head_dim": 64, "num_attention_heads": 32, "head_dim": 128,
            "qk_nope_head_dim": 128, "v_head_dim": 128, "short_conv_kernel_size": 4,
            "hidden_size": 2560, "vocab_size": 157184, "num_nextn_predict_layers": 1}


def test_prefill_law_caps_rows_by_context_and_prices_bf16():
    """Week-models ladder 2026-11-25: 32K was refused at the autoscaled 2048 rows
    (27.6 GiB under the fp32 three-buffer law) while measured peaks rose ~1-2 GB."""
    from dataclasses import replace

    from mlx2.adapters.ling_memory import LingCacheBudget

    base = LingCacheBudget.from_config(_air_config(), mtp=True)
    air = replace(base, **ling32_air.PREFILL_LAW)
    rows = {ctx: air.recommended_prefill_rows(2048, context_tokens=ctx)
            for ctx in (4096, 16384, 33430, 65536, 131072, 262144)}
    assert rows == {4096: 2048, 16384: 2048, 33430: 512, 65536: 512, 131072: 256, 262144: 128}
    assert air.recommended_prefill_rows(256, context_tokens=4096) == 256  # never widens
    gib = 1 << 30
    assert air.prefill_transient_bytes(33430, 512) < base.prefill_transient_bytes(33430, 512) / 2
    assert air.prefill_transient_bytes(33430, rows[33430]) < 4 * gib
    assert air.prefill_transient_bytes(131072, rows[131072]) < 8 * gib
    # Ling 3 Flash (the shared budget's defaults) keeps its law and recommends nothing.
    assert base.recommended_prefill_rows(2048, context_tokens=131072) == 2048
    assert base.score_buffers == 3 and base.prefill_activation_bytes == 0


def test_adapter_binds_the_row_cap_on_the_model():
    from types import SimpleNamespace

    adapter = ling32_air.Ling32AirAdapter.__new__(ling32_air.Ling32AirAdapter)
    adapter.config, adapter.model = _air_config(), SimpleNamespace()
    adapter._bind_prefill_row_cap()
    assert adapter.model.recommended_prefill_rows(2048, context_tokens=33430) == 512
    assert adapter.cache_budget(mtp=True).prefill_score_budget_bytes == 4 << 30
