"""Synthetic CPU only; the diagnostic does not establish model performance."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest


@pytest.fixture
def probe(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts/research"))
    spec = importlib.util.spec_from_file_location(
        "ds4_layer_probe", root / "scripts/research/probe_deepseek_v4_layer_cost.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nested_accounting_has_no_double_count(probe):
    class MX:
        array = type("Array", (), {})

        @staticmethod
        def eval(value):
            pass

    ticks = iter([0.0, 1.0, 2.0, 3.0, 5.0, 8.0])
    profiler = probe.BoundaryProfiler(MX, lambda: next(ticks))
    profiler.call(
        "outer",
        "o",
        lambda: profiler.call("inner", "i", lambda: None, (), {}),
        (),
        {},
        layer=2,
    )
    summary = profiler.summary()
    assert summary["categories"]["inner"]["inclusive_seconds"] == 3
    assert summary["categories"]["outer"]["exclusive_seconds"] == 5
    assert summary["exclusive_total_seconds"] == 8
    assert summary["layers_exclusive_seconds"] == {"2": {"inner": 3.0, "outer": 5.0}}


def test_tiny_real_graph_restores_every_plane_and_hooks(probe):
    import mlx.core as mx
    from test_deepseek_v4_forward_cpu import config, SmallWeights
    from mlx2.runtime.models import deepseek_v4 as module

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        c = config()
        model = module.DeepSeekV4Model(
            c, SmallWeights(c), identity="synthetic-layer-cost"
        )
        cache = model.make_cache()
        output = model(mx.array([[1, 2, 3]], dtype=mx.uint32), cache=cache)
        mx.eval(output, [s.state for s in cache])
        original = module.Block.__call__
        result = probe.profile_frontier(model, cache, 4, mx)
        assert module.Block.__call__ is original
        assert all(s.offset == 3 for s in cache)
        assert result["logits"]["bitwise_equal"]
        assert result["state_arrays_bitwise_equal"] and result["state_metadata_equal"]
        assert set(result["categories"]) >= {
            "hc",
            "attention",
            "routed_moe",
            "shared_expert",
            "dense",
            "block_residual",
        }
        assert result["categories"]["block_residual"]["calls"] == 3
        assert set(result["layers_exclusive_seconds"]) == {"0", "1", "2"}
        assert result["exclusive_total_seconds"] <= result["profiled_seconds"]
        with pytest.raises(RuntimeError, match="injected"):
            with probe.instrument(model, probe.BoundaryProfiler(mx)):
                raise RuntimeError("injected")
        assert module.Block.__call__ is original
    finally:
        mx.set_default_device(previous)


def test_guard_rejects_sidecar_and_existing_output(probe, tmp_path):
    args = SimpleNamespace(
        run_gpu=True,
        ownership_window=tmp_path / "owner",
        output=tmp_path / "new",
        prefill_step=128,
        memory_limit_gib=108,
        execution_policy={},
    )
    probe.validate_args(args)
    args.execution_policy = {"draft_model": "foreign"}
    with pytest.raises(ValueError, match="ordinary"):
        probe.validate_args(args)
    args.execution_policy = {}
    args.output.mkdir()
    with pytest.raises(ValueError, match="existing"):
        probe.validate_args(args)
