"""CPU-only LoRA identity gates before any slot materialization."""

import hashlib
import sys
import threading
from collections import Counter, OrderedDict
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from mlx2.api_resources import ResourceNotFound
from mlx2.runtime.multi_lora import MultiLoRAManager, RegisteredAdapter
from mlx2.serving import ServingEngine


def _manager():
    manager = MultiLoRAManager.__new__(MultiLoRAManager)
    manager.lock = threading.RLock()
    manager.registry = {"sql": SimpleNamespace(name="sql", fingerprint="revision-new")}
    manager.resident = OrderedDict({1: "sql"})
    manager.pins = Counter()
    manager.counts = Counter()
    return manager


def test_replaced_adapter_cannot_pin_a_slot_under_old_cache_fingerprint():
    manager = _manager()
    with pytest.raises(ValueError, match="changed"):
        manager.acquire("sql", expected_fingerprint="revision-old")
    assert manager.pins[1] == 0
    assert not manager.counts


def test_matching_adapter_revision_can_pin_existing_slot_without_tensor_work():
    manager = _manager()
    assert manager.acquire("sql", expected_fingerprint="revision-new") == (1, "hit")
    assert manager.pins[1] == 1


@pytest.mark.parametrize("model", [None, "fixture", "sql", "missing"])
def test_prepare_job_never_substitutes_base_for_a_removed_lora(model):
    engine = ServingEngine.__new__(ServingEngine)
    engine.model_path = "fixture"
    engine.qualification_mode = False
    engine.counts = Counter()
    engine.multi_lora = _manager()
    request = {"prompt": "hello"}
    if model is not None:
        request["model"] = model
    if model == "missing":
        with pytest.raises(ResourceNotFound, match="unknown model"):
            engine._prepare_job(request, tenant_id="tenant")
    else:
        job = engine._prepare_job(request, tenant_id="tenant")
        assert job.lora_name == ("sql" if model == "sql" else None)
        if model == "sql":
            assert job.request["_mlx2_lora_fingerprint"] == "revision-new"


def test_explicit_base_model_is_not_reinterpreted_as_an_adapter_name():
    engine = ServingEngine.__new__(ServingEngine)
    engine.model_path = "fixture"
    engine.qualification_mode = False
    engine.counts = Counter()
    engine.multi_lora = _manager()
    engine.multi_lora.registry["fixture"] = SimpleNamespace(name="fixture", fingerprint="other")
    job = engine._prepare_job({"prompt": "hello", "model": "fixture"}, tenant_id="tenant")
    assert job.lora_name is None
    assert "_mlx2_lora_fingerprint" not in job.request


@pytest.mark.parametrize("change_before_load", [True, False])
def test_lazy_materialization_loads_only_the_registered_bytes(tmp_path, monkeypatch, change_before_load):
    original = b"fixture weights"
    weights_path = tmp_path / "adapters.safetensors"
    weights_path.write_bytes(original)
    digest = hashlib.sha256(repr((1, 2.0, ("linear",))).encode() + original).hexdigest()
    adapter = RegisteredAdapter("sql", str(tmp_path), digest, 1, 2.0, ("linear",), {}, 8)
    manager = _manager()
    manager.wrapped = {"linear": SimpleNamespace(lora_a=SimpleNamespace(dtype=np.float32))}
    loaded = []

    def load(source, *, format=None):
        # Replace the path after validation: loading must use the exact bytes
        # that were fingerprinted, rather than reopening the mutable path.
        weights_path.write_bytes(b"replacement weights")
        assert format == "safetensors"
        loaded.append(source.read())
        return {"linear.lora_a": np.array([[1.0]]), "linear.lora_b": np.array([[3.0]])}

    mlx = ModuleType("mlx")
    core = ModuleType("mlx.core")
    core.load = load
    core.eval = lambda values: None
    core.float32 = np.float32
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    if change_before_load:
        weights_path.write_bytes(b"replacement weights")
        with pytest.raises(ValueError, match="changed"):
            manager._materialize(adapter)
        assert not loaded
        assert not adapter.tensors
    else:
        manager._materialize(adapter)
        assert loaded == [original]
        assert adapter.tensors["linear"][1].item() == 6.0
        assert manager.counts["materializations"] == 1
