from __future__ import annotations

from types import SimpleNamespace

import pytest

from mlx2.adapters.base import route_diagnostics_delta, route_diagnostics_snapshot
from mlx2.runtime.approximate_kv import (
    KVQuantizationDescriptor,
    declared_operations,
)
from mlx2.runtime.streamed_load import force_stock_expert_arithmetic


def test_approximate_operation_construction_is_adapter_factory_owned():
    descriptor = KVQuantizationDescriptor(8, 4, group_size=64)
    adapter = SimpleNamespace(
        approximate_kv_operations=lambda: {"private-kv": descriptor}
    )
    operations = declared_operations(adapter, adapter_fingerprint="artifact-a")
    assert operations["private-kv"].descriptor is descriptor
    assert operations["private-kv"].name == "private-kv"

    bad = SimpleNamespace(approximate_kv_operations=lambda: {"bad": object()})
    with pytest.raises(ValueError, match="adapter-owned factory"):
        declared_operations(bad, adapter_fingerprint="artifact-a")


def test_expert_streaming_uses_one_guard_protocol_without_setter_discovery():
    class Guard:
        def __init__(self):
            self.calls = 0

        def force_stock_for_expert_streaming(self):
            self.calls += 1
            return {"stock": 1}

    class SetterLookalike:
        def set_moe_routed_decode_mode(self, _mode):
            raise AssertionError("setter-name discovery must not run")

    guard = Guard()
    model = SimpleNamespace(
        named_modules=lambda: (("guard", guard), ("lookalike", SetterLookalike()))
    )
    assert force_stock_expert_arithmetic(model) == {"stock": 1}
    assert guard.calls == 1


def test_route_diagnostics_are_owned_by_snapshot_and_delta_provider():
    class Provider:
        def snapshot(self):
            return {"calls": 7}

        def delta(self, before, after):
            return {"calls": after["calls"] - before["calls"]}

    adapter = SimpleNamespace(
        route_diagnostics_provider=Provider(),
        diagnostics=lambda: (_ for _ in ()).throw(AssertionError("provider bypassed")),
    )
    assert route_diagnostics_snapshot(adapter) == {"calls": 7}
    assert route_diagnostics_delta(adapter, {"calls": 2}, {"calls": 7}) == {
        "calls": 5
    }


def test_production_adapter_diagnostics_use_bounded_default_provider():
    adapter = SimpleNamespace(diagnostics=lambda: {"calls": 9})
    assert route_diagnostics_snapshot(adapter) == {"calls": 9}
    before = {f"counter_{index}": index for index in range(256)}
    after = {key: value + 1 for key, value in before.items()}
    delta = route_diagnostics_delta(adapter, before, after)
    assert 0 < len(delta) <= 128
    assert set(delta.values()) == {1}


def test_segmented_mtp_qsa_alias_is_versioned_for_one_cycle():
    from mlx2.runtime.segmented_self_mtp import (
        note_segmented_self_mtp,
        segmented_self_mtp_stats,
    )

    segmented_self_mtp_stats(reset=True)
    note_segmented_self_mtp("async_qsa_shared_prefix_fused_layers", 2)
    stats = segmented_self_mtp_stats(reset=True)
    assert stats["schema_version"] == 2
    assert stats["async_shared_state_prefix_fused_layers"] == 2
    assert stats["async_qsa_shared_prefix_fused_layers"] == 2


def test_shared_runtime_has_no_model_or_process_global_dispatch():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "mlx2"
    approximate = (root / "runtime" / "approximate_kv.py").read_text()
    streamed = (root / "runtime" / "streamed_load.py").read_text()
    serving = (root / "serving.py").read_text()
    declared = approximate.split("def declared_operations", 1)[1]
    execution = serving.split("def _execution_diagnostics", 1)[1].split(
        "def validated_adapter_prefill_inputs", 1
    )[0]
    assert "isinstance(descriptor, Xing" not in declared
    assert "_STOCK_SETTERS" not in streamed
    assert "sys.modules" not in execution
