from dataclasses import replace

"""Real serving/cache seams on tiny CPU weights; not checkpoint evidence."""

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from test_deepseek_v41_cpu import Weights, config

from mlx2.adapters.deepseek_v41 import DESCRIPTOR, DeepSeekV41Adapter, execution_policy
from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model


def model():
    c = config()
    w = Weights(c)
    w._dense = w.data
    for k in (
        "scheduled_expert_calls",
        "scheduled_grouped_calls",
        "scheduled_dense_calls",
    ):
        setattr(w, k, {})
    return DeepSeekV41Model(c, w, identity="synthetic-serving@cpu")


def test_native_model_and_apcv2_disk_branch_lifecycle(tmp_path):
    from mlx2.runtime.apc_v2 import inspect_apc_capabilities
    from mlx2.runtime.cow_cache import freeze_prompt_cache, restore_prompt_cache
    from mlx2.runtime.models.cache import load_prompt_cache, save_prompt_cache

    target = model()
    cache = target.make_cache()
    assert list(target.named_modules()) == [("", target)]
    assert target.eval() is target and inspect_apc_capabilities(cache) is not None
    target(mx.array([[1, 2, 3]], mx.int32), cache)
    expected = [c.snapshot() for c in cache]
    path = tmp_path / "v41.safetensors"
    save_prompt_cache(str(path), cache)
    restored = load_prompt_cache(str(path))
    assert [c.meta_state for c in restored] == [c.meta_state for c in cache]
    for row, snap in zip(restored, expected):
        for actual, wanted in zip(row.state, snap.arrays):
            if actual is not None:
                np.testing.assert_array_equal(np.array(actual), np.array(wanted))
    frozen, _ = freeze_prompt_cache(
        cache, key="v41-synthetic", tokens=(1, 2, 3), cache_type="deepseek_v41"
    )
    branch = restore_prompt_cache(frozen)
    try:
        target(mx.array([[4]], mx.int32), branch)
        assert all(c.offset == 4 for c in branch) and all(c.offset == 3 for c in cache)
        np.testing.assert_array_equal(
            np.array(cache[1].pooled), np.array(expected[1].arrays[3])
        )
    finally:
        branch.close()
        frozen.close()


def test_partial_load_cleanup_restores_wired_limit_after_reader_failure(monkeypatch):
    from types import SimpleNamespace

    events = []

    def broken_reader():
        events.append("reader")
        raise OSError("fixture read failure")

    adapter = object.__new__(DeepSeekV41Adapter)
    adapter.engram_tables = {1: SimpleNamespace(close=broken_reader)}
    adapter.weights = SimpleNamespace(close=lambda: events.append("weights"))
    adapter.preload_residency = {"wired": True, "previous_wired_limit_bytes": 123}
    monkeypatch.setattr(mx, "set_wired_limit", lambda value: events.append(value))
    with pytest.raises(OSError, match="fixture read failure"):
        adapter.close()
    assert events == ["reader", "weights", 123]
    assert adapter.weights is None and adapter.preload_residency is None
    adapter.close()
    assert events == ["reader", "weights", 123]


@pytest.mark.parametrize(
    "value",
    [
        {"draft_model": "v4.dspark"},
        {"deepseek_kv": "fp8_mixed"},
        {"ast_prompt_lookup": 1},
        {"num_draft": 3},
        {"ast_prompt_lookup": True, "num_draft": 6},
    ],
)
def test_foreign_or_incomplete_policies_refuse(value):
    with pytest.raises(ValueError):
        execution_policy(value)


def test_offload_budget_counts_sources_not_consumers():
    from mlx2.adapters.deepseek_v41_memory import DeepSeekV41CacheBudget

    budget = DeepSeekV41CacheBudget.from_config(config())
    assert budget.pooled_bytes(1024) == sum(
        budget._pool_capacity(1024, r) * (8 + 4) * 4 for r in (2, 1)
    )
    assert budget.project(1024) > budget.project_resident(1024)
    assert budget.prefill_transient_bytes(1024, 64) > budget.prefill_transient_bytes(
        1024, 1
    )


def test_copy_chain_rejected_suffix_restores_every_source():
    from mlx2.adapters.deepseek_v4_external import DeepSeekV4VerifyRows

    target = model()
    cache = target.make_cache()
    target(mx.array([[1, 2, 3]], mx.int32), cache)
    snapshots = [c.snapshot() for c in cache]
    owner = DeepSeekV4VerifyRows(target, [cache], (4,))
    transaction = owner.begin([3])
    logits, features = transaction.forward(mx.array([[4, 5, 6]], mx.int32))
    assert logits.shape[:2] == features.shape[:2] == (1, 3)
    transaction.commit([1])
    actual = target(mx.array([[7]], mx.int32), cache)
    for row, snap in zip(cache, snapshots):
        row.restore(snap)
    target(mx.array([[4]], mx.int32), cache)
    expected = target(mx.array([[7]], mx.int32), cache)
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


@pytest.mark.parametrize("copy_chain", [False, True])
def test_real_engine_cold_warm_cpu(monkeypatch, tmp_path, copy_chain):
    from route_harness import collect, make_adapter, patch_host

    from mlx2 import serving
    from mlx2.runtime import generation_stream, weight_residency

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
    harness = make_adapter(None, config().vocab_size)

    class Adapter(DeepSeekV41Adapter):
        tokenizer = harness.tokenizer
        max_context = 512

        def __init__(self, path, execution_policy=None):
            self.model = model()
            self.weights = self.model.weights
            self.identity = {"fingerprint": "v41-synthetic-engine"}
            self.kernel_policy = globals()["execution_policy"](execution_policy)
            self.draft_model = None
            if copy_chain:
                from mlx2.contracts import Capability
                from mlx2.runtime.drafters.prompt_lookup_draft import (
                    PromptLookupDraftModel,
                )

                self.draft_model = PromptLookupDraftModel(self.model, num_draft=5)
                self.descriptor = replace(
                    DESCRIPTOR,
                    capabilities=DESCRIPTOR.capabilities | {Capability.EXTERNAL_DRAFT},
                )
                self.default_route = "external_draft"
            self.external_policy = {}
            self.engram_tables = {}
            self.preload_residency = None
            self.layout = DESCRIPTOR.cache_layout
            self.config = vars(self.model.args)
            self.metadata = {}
            self.environment = {}

        prompt_tokens = harness.prompt_tokens
        output_parser = harness.output_parser

        def close(self):
            pass

    engine = serving.ServingEngine(
        str(tmp_path),
        adapter_factory=Adapter,
        execution_policy={"draft_model": "prompt_lookup", "num_draft": 5}
        if copy_chain
        else None,
        mtp=False,
        max_lanes=1,
        max_inflight=1,
        prefill_step=4,
        max_context=512,
        cache_bytes=64 << 20,
        residency_keep_seconds=0,
    )
    try:
        assert engine.ready.wait(10), engine.error
        assert engine.error is None, engine.error
        request = {
            "tokens": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "max_tokens": 2,
            "temperature": 0,
        }
        cold = collect(engine.submit(request), timeout=10)
        warm = collect(engine.submit(request), timeout=10)
        assert cold.get("error") is None, cold
        assert warm.get("error") is None, warm
        assert len(cold["tokens"]) == 2 and cold["tokens"] == warm["tokens"]
        assert (
            cold["receipt"]["cached_tokens"] == 0
            and warm["receipt"]["cached_tokens"] > 0
        )
    finally:
        engine.close()


def test_adaptive_copy_loop_matches_ordinary_target():
    from mlx2.adapters.deepseek_v41_external import create_batch
    from mlx2.runtime.drafters.prompt_lookup_draft import PromptLookupDraftModel

    target = model()
    prompt = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    cache = target.make_cache()
    logits = target(mx.array([prompt], mx.int32), cache)
    ordinary = []
    for _ in range(12):
        token = int(mx.argmax(logits[0, -1]).item())
        ordinary.append(token)
        logits = target(mx.array([[token]], mx.int32), cache)
    draft = PromptLookupDraftModel(target, num_draft=5)
    generator = create_batch(
        target,
        draft,
        binding=target.identity,
        num_draft=5,
        stop_tokens=set(),
        prefill_step_size=4,
        cost_per_commit_handoff=False,
    )
    generator.insert([prompt], max_tokens=[12])
    output = []
    for _ in range(30):
        _, rows = generator.next()
        output.extend(int(row.token) for row in rows)
        if not generator.lanes:
            break
    assert output == ordinary


def test_standard_external_route_selection_reaches_copy_chain(tmp_path, monkeypatch):
    import json

    from mlx2 import server
    from mlx2.adapters import deepseek_v41, registry

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek41_gguf"}))
    monkeypatch.setattr(
        deepseek_v41,
        "inspect_artifact",
        lambda path: {"identity": {"fingerprint": "f" * 64}},
    )
    resolution = registry.inspect_model(tmp_path)
    args = server.build_parser().parse_args(
        ["--model", str(tmp_path), "--external-draft", "--max-lanes", "1"]
    )
    policy = {"draft_model": "prompt_lookup", "num_draft": 5}
    server.validate_invocation_draft_adapter(args, resolution)
    selected = server.resolve_route_selection(args, policy, resolution)
    assert selected.route == "external_draft"
    resolved = server.resolve_execution_policy_defaults(
        policy, selected, resolution, max_lanes=1
    )
    assert execution_policy(resolved)["ast_prompt_lookup"] is True
    args.external_draft = False
    with pytest.raises(ValueError, match="requires --external-draft"):
        server.resolve_route_selection(args, policy, resolution)


def test_template_uses_v41_official_encoder_and_refuses_media():
    adapter = DeepSeekV41Adapter.__new__(DeepSeekV41Adapter)
    adapter.draft_model = None
    chat = adapter.render_prompt({"messages": [{"role": "user", "content": "hello"}]})
    thinking = adapter.render_prompt(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "enable_thinking": True,
            "reasoning_effort": 75,
        }
    )
    assert "<｜User｜>hello<｜Assistant｜>" in chat and chat.endswith("</think>")
    assert thinking.endswith("<think>") and "Reasoning Effort: 75" in thinking
    with pytest.raises(ValueError, match="text-only"):
        adapter.render_prompt(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": "no"}}],
                    }
                ]
            }
        )


def test_optimized_policies_are_explicit_and_foreign_v4_flags_refuse():
    from mlx2.adapters.deepseek_v41 import optimized_policy

    assert execution_policy(None)["deepseek_qat"] == "reference"
    assert execution_policy(None)["deepseek_expert_decode"] == "reference"
    assert execution_policy(None)["deepseek_expert_buffers"] == "stack"
    assert execution_policy(None)["deepseek_verify_commit"] == "replay"
    assert execution_policy(optimized_policy())["deepseek_expert_decode"] == "row_buffers"
    assert execution_policy(optimized_policy())["deepseek_qat"] == "fused_fp32"
    assert execution_policy(optimized_policy(copy_chain=True))["ast_prompt_lookup"]
    with pytest.raises(ValueError):
        execution_policy({"deepseek_index_qat": "fused_fp32"})
