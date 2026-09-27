"""The three new APCv2 bridges are discoverable but need route evidence."""

import importlib.abc
import json
import sys

import pytest

from mlx2.adapters import registry
from mlx2.contracts import Capability, ModelDescriptor, StatePlane


class BlockMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise AssertionError("real MLX import during CPU-only registry test")
        return None


@pytest.mark.parametrize("model_type", ["lfm2_vl", "smolvlm", "qwen2_5_vl"])
def test_new_multimodal_routes_require_qualification_before_serving(
    tmp_path, monkeypatch, model_type,
):
    assert "mlx.core" not in sys.modules
    monkeypatch.setattr(sys, "meta_path", [BlockMLX(), *sys.meta_path])
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))

    class Candidate:
        default_route = "ordinary"

    descriptor = ModelDescriptor(
        model_type=model_type, family=model_type, variant="candidate",
        state_planes=frozenset({StatePlane.ATTENTION_KV}),
        capabilities=frozenset({Capability.TEXT, Capability.APC_V2}),
        cache_layout="test", metadata={"qualification": "pending"},
    )
    monkeypatch.setitem(
        registry._RESOLVERS, model_type,
        lambda path, config: registry.AdapterResolution(Candidate, descriptor, {}),
    )
    assert registry.inspect_model(tmp_path).adapter_type is Candidate
    with pytest.raises(ValueError, match="requires qualification mode"):
        registry.resolve_adapter(tmp_path)
    with pytest.raises(ValueError, match="native MTP"):
        registry.resolve_adapter(tmp_path, mtp=True, qualification_mode=True)
    assert registry.resolve_adapter(tmp_path, qualification_mode=True) is Candidate
    assert registry.resolve_adapter(tmp_path, qualification="receipt.json") is Candidate
    assert "mlx.core" not in sys.modules
