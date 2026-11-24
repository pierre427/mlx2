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
    assert air["kda_fused"] is True and air["routed_moe_decode"] is False
    assert sum(air.values()) == 1
    opt_out = ling3_flash.optimization_policy({"kda_fused": False},
                                              ling32_air.Ling32AirAdapter.optimization_overrides)
    assert opt_out["kda_fused"] is False
    with pytest.raises(ValueError, match="override"):
        ling3_flash.optimization_policy(None, {"nope": True})
