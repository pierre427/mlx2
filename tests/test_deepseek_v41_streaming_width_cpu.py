"""Synthetic admission/ServingEngine seams; no real checkpoint or GPU load."""

import json
from typing import ClassVar

import pytest

from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter
from mlx2.runtime.streamed_load import MIN_LANE_HEADROOM_BYTES, WeightStreamRequest
from mlx2.serving import (
    ServingEngine,
    validate_weight_streaming_lanes,
    weight_streaming_lane_bound,
)


class DeclaredSource:
    weight_streaming_modes = frozenset({"moe_experts"})
    weight_streaming_max_lanes = staticmethod(
        DeepSeekV41SourceAdapter.weight_streaming_max_lanes
    )
    calls: ClassVar[list] = []

    def __init__(self, path, **options):
        type(self).calls.append(options)
        raise RuntimeError("synthetic stop after pre-materialization delivery")


class InheritedSource(DeclaredSource):
    # An inherited declaration is intentionally insufficient for widening.
    pass


def artifact(tmp_path):
    root = tmp_path / "synthetic-width-artifact"
    root.mkdir()
    name = "model.safetensors"
    (root / name).write_bytes(b"\0" * 8000)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"fixture": name}})
    )
    return root


def test_source_width_hook_is_policy_bound_explicit_and_not_inherited():
    for key, width in [("b1", 1), ("b2", 2), ("b4", 4)]:
        policy = {
            "deepseek_physical_batch": key,
            "moe_expert_streaming": {"enabled": True, "cache_gib": 1},
        }
        assert (
            weight_streaming_lane_bound(
                DeepSeekV41SourceAdapter, policy, mode="moe_experts"
            )
            == width
        )
        assert (
            weight_streaming_lane_bound(
                DeepSeekV41SourceAdapter, policy, mode="dense_mlp"
            )
            == 1
        )
    assert (
        weight_streaming_lane_bound(
            InheritedSource, {"deepseek_physical_batch": "b4"}, mode="moe_experts"
        )
        == 1
    )
    for bad in (
        {"deepseek_physical_batch": "b8"},
        {"deepseek_physical_batch": "b2", "draft_model": "embedded_dspark"},
        {"deepseek_physical_batch": "b2", "deepseek_ced": "dependency_exact"},
        {"deepseek_physical_batch": "b2", "deepseek_vision": "enabled"},
        {"deepseek_physical_batch": "b2", "deepseek_source_completion": "bounded_2"},
    ):
        with pytest.raises(ValueError):
            weight_streaming_lane_bound(
                DeepSeekV41SourceAdapter, bad, mode="moe_experts"
            )


def test_known_source_validate_arguments_accepts_widened_expert_route_without_loading():
    DeclaredSource.calls.clear()
    ServingEngine.validate_arguments(
        "synthetic",
        adapter_factory=DeclaredSource,
        max_lanes=4,
        max_inflight=4,
        execution_policy={
            "deepseek_physical_batch": "b4",
            "moe_expert_streaming": {"enabled": True, "cache_gib": 1},
        },
    )
    assert DeclaredSource.calls == []
    with pytest.raises(ValueError, match="max_lanes=1"):
        ServingEngine.validate_arguments(
            "synthetic",
            adapter_factory=DeclaredSource,
            max_lanes=4,
            max_inflight=4,
            execution_policy={
                "deepseek_physical_batch": "b4",
                "dense_weight_streaming": {"enabled": True, "staging_gib": 1},
            },
        )


@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("registry_resolved", [False, True])
def test_real_engine_explicit_and_auto_width_admission_reaches_factory(
    tmp_path, monkeypatch, automatic, registry_resolved
):
    from mlx2 import serving
    from mlx2.adapters import registry

    root = artifact(tmp_path)
    DeclaredSource.calls.clear()
    monkeypatch.setattr(
        serving, "preload_admission_limit_bytes", lambda: MIN_LANE_HEADROOM_BYTES + 7000
    )
    monkeypatch.setattr(registry, "resolve_adapter", lambda *a, **kw: DeclaredSource)
    policy = {"deepseek_physical_batch": "b4"}
    if not automatic:
        policy["moe_expert_streaming"] = {"enabled": True, "cache_gib": 1}
    engine = ServingEngine(
        str(root),
        adapter_factory=None if registry_resolved else DeclaredSource,
        max_lanes=4,
        max_inflight=4,
        execution_policy=policy,
    )
    try:
        engine.thread.join(10)
        assert not engine.thread.is_alive()
        assert "synthetic stop" in engine.error
        assert len(DeclaredSource.calls) == 1
        options = DeclaredSource.calls[0]
        request = options["weight_streaming"]
        assert (
            isinstance(request, WeightStreamRequest) and request.mode == "moe_experts"
        )
        assert request.budget_mode == ("auto" if automatic else "explicit")
        assert options["execution_policy"] == {"deepseek_physical_batch": "b4"}
        assert engine.weight_streaming_lane_bound == 4
        assert engine.weight_streaming_status()["maximum_lanes"] == 4
        assert engine.weight_streaming_route == "early"
    finally:
        engine.close()


@pytest.mark.parametrize("automatic", [False, True])
def test_source_default_width_and_overbound_fail_before_factory(
    tmp_path, monkeypatch, automatic
):
    from mlx2 import serving

    root = artifact(tmp_path)
    DeclaredSource.calls.clear()
    monkeypatch.setattr(
        serving, "preload_admission_limit_bytes", lambda: MIN_LANE_HEADROOM_BYTES + 7000
    )
    for source_policy, bound in [({}, 1), ({"deepseek_physical_batch": "b2"}, 2)]:
        policy = dict(source_policy)
        if not automatic:
            policy["moe_expert_streaming"] = {"enabled": True, "cache_gib": 1}
        with pytest.raises(ValueError, match=f"max_lanes={bound}"):
            ServingEngine(
                str(root),
                adapter_factory=DeclaredSource,
                max_lanes=4,
                max_inflight=4,
                execution_policy=policy,
            )
    assert DeclaredSource.calls == []


def test_invalid_generic_width_declaration_fails_closed():
    class Bad:
        weight_streaming_max_lanes = staticmethod(lambda policy, *, mode: True)

    with pytest.raises(ValueError, match="positive integer"):
        validate_weight_streaming_lanes(Bad, {}, mode="moe_experts", max_lanes=2)


@pytest.mark.parametrize("width", [2, 4])
def test_server_parser_policy_and_prebind_validation_use_resolved_source_class(
    tmp_path, width
):
    from mlx2.server import build_parser, serving_engine_kwargs

    policy = {
        "deepseek_physical_batch": f"b{width}",
        "moe_expert_streaming": {"enabled": True, "cache_gib": 1},
    }
    path = tmp_path / "execution.json"
    path.write_text(json.dumps(policy))
    args = build_parser().parse_args(
        [
            "--model",
            str(tmp_path),
            "--max-lanes",
            str(width),
            "--max-inflight",
            str(width),
            "--execution-policy",
            str(path),
        ]
    )
    loaded = json.loads(args.execution_policy.read_text())
    kwargs = serving_engine_kwargs(
        args, loaded, native_mtp=False, approximate_kv=None, max_request_bytes=1 << 20
    )
    ServingEngine.validate_arguments(
        args.model, adapter_factory=DeepSeekV41SourceAdapter, **kwargs
    )
    # The exact class supplied by the already-completed registry inspection is
    # required; validate-only does not perform filesystem/model resolution.
    with pytest.raises(ValueError, match="max_lanes=1"):
        ServingEngine.validate_arguments(args.model, **kwargs)


@pytest.mark.parametrize("width", [2, 4])
def test_real_serving_engine_concurrent_batch_requests_and_apc_replay(
    monkeypatch, tmp_path, width
):
    import mlx.core as mx
    from route_harness import collect, make_adapter, patch_host
    from test_deepseek_v41_serving_cpu import model

    from mlx2 import serving
    from mlx2.adapters.deepseek_v41 import DeepSeekV41Adapter
    from mlx2.adapters.deepseek_v41_batch import configure_physical_batch
    from mlx2.adapters.deepseek_v41_source import source_policy
    from mlx2.contracts import Capability
    from mlx2.runtime import generation_stream, weight_residency

    mx.set_default_device(mx.cpu)
    patch_host(monkeypatch)
    monkeypatch.setattr(
        weight_residency, "wire_serving_weights", lambda: {"fixture": "CPU"}
    )
    monkeypatch.setattr(
        serving.ServingEngine, "_start_residency_heartbeat", lambda self: None
    )
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    monkeypatch.setattr(
        generation_stream, "generation_stream", mx.new_thread_local_stream(mx.cpu)
    )
    harness = make_adapter(None, 19)

    class Adapter(DeepSeekV41Adapter):
        tokenizer = harness.tokenizer
        max_context = 512

        def __init__(self, path, execution_policy=None):
            self.model = model()
            self.weights = self.model.weights
            self.identity = {"fingerprint": "synthetic-serving-batch@cpu"}
            self.kernel_policy = source_policy(execution_policy)
            self.draft_model = None
            self.external_policy = {}
            self.engram_tables = {}
            self.preload_residency = None
            self.layout = self.descriptor.cache_layout
            self.config = vars(self.model.args)
            self.metadata = {}
            self.environment = {}
            configure_physical_batch(self, width=width)

        prompt_tokens = harness.prompt_tokens
        output_parser = harness.output_parser

        def diagnostics(self):
            return {
                "batch": dict(self.model.batch_counters),
                "qualification": "unqualified",
            }

        def close(self):
            pass

    engine = ServingEngine(
        str(tmp_path),
        adapter_factory=Adapter,
        mtp=False,
        max_lanes=width,
        max_inflight=width,
        prefill_step=2,
        max_context=512,
        cache_bytes=64 << 20,
        residency_keep_seconds=0,
        coalesce_window_ms=10,
        execution_policy={"deepseek_physical_batch": f"b{width}"},
    )
    try:
        assert engine.ready.wait(10), engine.error
        assert engine.error is None, engine.error
        prompts = [[1, 2, 3, 4, 5], [6, 7], [8, 9, 10], [11, 12, 13, 14]][:width]
        requests = [
            {
                "tokens": p,
                "max_tokens": 4,
                "temperature": 0,
                "batch_cohort": {"id": "synthetic-concurrent", "size": width},
            }
            for p in prompts
        ]
        jobs = [engine.submit(request) for request in requests]
        outputs = [collect(job, timeout=10) for job in jobs]
        assert all(result.get("error") is None for result in outputs), outputs
        assert all(len(result["tokens"]) == 4 for result in outputs)
        assert Capability.CONTINUOUS_BATCH in engine.adapter.descriptor.capabilities
        assert engine.adapter.model.batch_counters["maximum_live_requests"] == width
        assert (
            engine.adapter.model.batch_counters["maximum_joint_ffn_requests"] == width
        )
        replay = collect(
            engine.submit({"tokens": prompts[0], "max_tokens": 4, "temperature": 0}),
            timeout=10,
        )
        assert replay.get("error") is None, replay
        assert replay["tokens"] == outputs[0]["tokens"]
        assert replay["receipt"]["cached_tokens"] > 0
        assert replay["receipt"]["qualification"] == "unqualified"
    finally:
        engine.close()


def test_external_stream_contracts_are_own_declared_and_resident_draft_is_source_only():
    from mlx2.adapters.deepseek_v41_iq1 import DeepSeekV41IQ1Adapter
    from mlx2.adapters.deepseek_v41_mlx import DeepSeekV41MLXAdapter

    for adapter in (
        DeepSeekV41SourceAdapter,
        DeepSeekV41MLXAdapter,
        DeepSeekV41IQ1Adapter,
    ):
        hook = adapter.__dict__["weight_streaming_external_draft"]
        assert hook({"draft_model": "prompt_lookup"})
        assert not hook({"draft_model": "unknown"})
        assert not hook({"draft_model": "prompt_lookup", "draft_models": []})
        if adapter is DeepSeekV41SourceAdapter:
            assert hook({"draft_model": "embedded_dspark"})
        else:
            assert not hook({"draft_model": "embedded_dspark"})
        engine = object.__new__(ServingEngine)
        engine.adapter_factory = adapter
        engine.lane_matmul = "off"
        engine.sp_qmm_enabled = False
        from types import SimpleNamespace

        engine.int8_prefill_policy = SimpleNamespace(enabled=False)
        engine.verify_bitexact_policy = SimpleNamespace(enabled=False)
        engine.multi_lora_policy = None
        engine.lora_root = None
        engine._refuse_unevidenced_weight_streaming(
            {
                "draft_model": "prompt_lookup",
                "moe_expert_streaming": {"enabled": True, "cache_gib": 1},
            }
        )
        if adapter is DeepSeekV41SourceAdapter:
            engine._refuse_unevidenced_weight_streaming(
                {
                    "draft_model": "embedded_dspark",
                    "moe_expert_streaming": {"enabled": True, "cache_gib": 1},
                }
            )
        with pytest.raises(ValueError, match="external draft"):
            engine._refuse_unevidenced_weight_streaming({"draft_model": "unknown"})
    with pytest.raises(ValueError):
        DeepSeekV41SourceAdapter.weight_streaming_external_draft(
            {"draft_model": "embedded_dspark", "deepseek_physical_batch": "b2"}
        )
