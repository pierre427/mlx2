"""Adapter-owned serving hooks; fake weights only, no GPU/model-path claim."""

from types import SimpleNamespace

import pytest

from mlx2.adapters.deepseek_v4_gguf import (
    CACHE_LAYOUT,
    DESCRIPTOR,
    DeepSeekV4GGUFAdapter,
    split_execution_policy,
)
from mlx2.contracts import Capability, StatePlane


@pytest.mark.parametrize(
    "value",
    [
        {"num_draft": 5},
        {"draft_model": True},
        {"draft_model": "sidecar", "num_draft": 6},
        {"draft_model": "sidecar", "num_draft": True},
        {"draft_model": "sidecar", "deepseek_verifier": "tree"},
        {"draft_model": "sidecar", "ast_tree_tile_rows": 16},
    ],
)
def test_external_policy_fails_closed(value):
    with pytest.raises(ValueError):
        split_execution_policy(value)


def test_external_policy_preserves_kernel_choices_and_five_proposals():
    kernel, external = split_execution_policy(
        {
            "draft_model": "exact.gguf",
            "num_draft": 5,
            "deepseek_packed": "cooperative",
            "deepseek_expert_prefill": "segmented_nax_fp16",
        }
    )
    assert kernel["deepseek_packed"] == "cooperative"
    assert kernel["deepseek_expert_prefill"] == "segmented_nax_fp16"
    assert external == {
        "draft_model": "exact.gguf",
        "num_draft": 5,
        "deepseek_verifier": "sequential",
        "deepseek_draft_confidence": 0.6,
    }
    assert split_execution_policy(None)[1] == {}


@pytest.mark.parametrize("temperature", [1.0, None, True, -1.0, float("nan")])
def test_greedy_refusal_precedes_template_or_parser(temperature):
    adapter = DeepSeekV4GGUFAdapter.__new__(DeepSeekV4GGUFAdapter)
    adapter.draft_model = object()
    with pytest.raises(ValueError, match="temperature0"):
        adapter.prompt_tokens({"messages": [], "temperature": temperature})
    with pytest.raises(ValueError, match="temperature0"):
        adapter.output_parser({"messages": [], "temperature": temperature})


def test_external_config_and_generator_factory_owned(monkeypatch):
    from mlx2.adapters import deepseek_v4_external

    adapter = DeepSeekV4GGUFAdapter.__new__(DeepSeekV4GGUFAdapter)
    adapter.model = object()
    adapter.draft_model = object()
    adapter.external_policy = {"num_draft": 5}
    adapter.target_identity = {"fingerprint": "target"}
    seen = {}

    def factory(model, **kwargs):
        seen.update(model=model, **kwargs)
        return "owned"

    monkeypatch.setattr(
        deepseek_v4_external, "DeepSeekV4ExternalBatchGenerator", factory
    )
    assert adapter.create_external_batch(completion_batch_size=1) == "owned"
    assert seen["target_artifact_fingerprint"] == "target" and seen["num_draft"] == 5
    assert seen["cost_per_commit_handoff"] is False
    assert (
        adapter.execution_config(max_lanes=1, prefill_step=128)["backend"]
        == "external_draft"
    )
    with pytest.raises(ValueError, match="B1"):
        adapter.execution_config(max_lanes=2, prefill_step=128)


@pytest.mark.parametrize("mode", ["sequential", "causal_slab"])
def test_selected_external_descriptor_source_profile_and_resident_preload(
    monkeypatch, mode
):
    import mlx.core as mx

    from mlx2.adapters import deepseek_v4_dspark_artifact, deepseek_v4_external
    from mlx2.runtime.drafters import deepseek_v4_dspark
    from mlx2.runtime.gguf import weights

    events = []
    record = {"path": "sealed-sidecar", "stat": {"size": 42}, "sha256": "side-sha"}
    config = SimpleNamespace(validate_target=lambda args: events.append("validate"))
    monkeypatch.setattr(
        deepseek_v4_dspark_artifact, "file_stat", lambda path: {"size": 42}
    )
    monkeypatch.setattr(
        deepseek_v4_dspark.DeepSeekV4DSparkConfig, "from_index", lambda *a: config
    )

    class Weights:
        def __init__(self, *a, **kw):
            self.metadata = {}
            self.tensors = {
                n: SimpleNamespace(ggml_type=k)
                for n, k in [("norm", 0), ("hc", 1), ("expert", 16)]
            }
            self._dense = {}

        def array(self, name):
            events.append(name)

        def packed(self, name):
            events.append(name)

    monkeypatch.setattr(weights, "GGUFWeights", Weights)

    def backbone(*a, **kw):
        events.append("backbone")
        assert kw["source_snapshot"] == "f" * 64
        return SimpleNamespace(preload=lambda: events.append("support-preloaded"))

    monkeypatch.setattr(deepseek_v4_dspark, "DeepSeekV4DSparkBackbone", backbone)

    class Wrapper:
        def __init__(self, backbone, verification_mode, confidence_threshold):
            assert confidence_threshold == 0.6
            assert "support-preloaded" in events
            self.execution_binding = verification_mode + "-bound"
            self.receipt_settings = {"verification": verification_mode}

        def bind(self, model):
            return self

    monkeypatch.setattr(deepseek_v4_external, "DeepSeekV4DSparkDraftModel", Wrapper)
    monkeypatch.setattr(mx, "eval", lambda *a: None)
    adapter = DeepSeekV4GGUFAdapter.__new__(DeepSeekV4GGUFAdapter)
    adapter.external_record = record
    adapter.external_source = {"source_snapshot_sha256": "f" * 64}
    adapter.external_policy = {"num_draft": 5, "deepseek_verifier": mode}
    adapter.kernel_policy = split_execution_policy(None)[0]
    adapter.weights = object()
    adapter.model = SimpleNamespace(args=object())
    adapter.identity = {
        "fingerprint": "target",
        "weight_sha256": "target-sha",
        "path": "target-path",
    }
    adapter.layout = CACHE_LAYOUT
    adapter.descriptor = DESCRIPTOR
    adapter.metadata = {}
    adapter._load_external()
    assert events.index("expert") < events.index("backbone")
    assert events.index("norm") < events.index("backbone")
    assert Capability.EXTERNAL_DRAFT in adapter.descriptor.capabilities
    assert StatePlane.DRAFT in adapter.descriptor.state_planes
    assert adapter.descriptor.metadata["qualification"] == "unqualified"
    assert adapter.descriptor.metadata["packed_ast"] is False
    from mlx2.sampling_contract import GREEDY_REFUSAL_SAMPLING_CONTRACT
    assert adapter.descriptor.metadata["sampling_contract"]==GREEDY_REFUSAL_SAMPLING_CONTRACT
    assert "sampling_contract" not in DESCRIPTOR.metadata
    assert adapter.identity["target_fingerprint"] == "target"
    assert adapter.identity["fingerprint"] != "target"
    assert adapter.identity["fingerprint"] in adapter.layout
    assert adapter.default_route == "external_draft"
    assert adapter.profile_name(False).startswith(
        "deepseek-v4-vision-exp-dspark-greedy-" + mode + "-b1:")


def test_real_server_registry_route_and_policy_resolution(tmp_path, monkeypatch):
    import json

    from mlx2 import server
    from mlx2.adapters import deepseek_v4_gguf, registry
    from mlx2.serving import SERVER_OWNED_EXECUTION_POLICY_KEYS

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek4_gguf"}))
    monkeypatch.setattr(
        deepseek_v4_gguf,
        "inspect_artifact",
        lambda path: {"identity": {"fingerprint": "f" * 64}},
    )
    resolution = registry.inspect_model(tmp_path)
    assert resolution.adapter_type is DeepSeekV4GGUFAdapter
    assert resolution.default_route == "ordinary"
    assert Capability.EXTERNAL_DRAFT not in resolution.descriptor.capabilities
    args = server.build_parser().parse_args(
        ["--model", str(tmp_path), "--external-draft", "--max-lanes", "1"]
    )
    policy = {
        "draft_model": "exact-support.gguf",
        "num_draft": 5,
        "deepseek_verifier": "sequential",
        "deepseek_kv": "fp8_mixed",
    }
    server.validate_invocation_draft_adapter(args, resolution)
    selected = server.resolve_route_selection(args, policy, resolution)
    assert selected.route == "external_draft"
    resolved = server.resolve_execution_policy_defaults(
        policy, selected, resolution, max_lanes=1
    )
    assert resolved == policy
    handed = {
        key: value
        for key, value in resolved.items()
        if key not in SERVER_OWNED_EXECUTION_POLICY_KEYS
    }
    kernels, external = split_execution_policy(handed)
    assert external["num_draft"] == 5 and kernels["deepseek_kv"] == "fp8_mixed"
    args.external_draft = False
    with pytest.raises(ValueError, match="requires --external-draft"):
        server.resolve_route_selection(args, policy, resolution)


def test_external_target_contract_is_cpu_and_rejects_before_upload():
    from dataclasses import asdict

    from mlx2.runtime.drafters.deepseek_v4_dspark import (
        DeepSeekV4DSparkConfig,
        DSparkContractError,
    )
    from mlx2.runtime.models.deepseek_v4 import DeepSeekV4Config

    adapter = DeepSeekV4GGUFAdapter.__new__(DeepSeekV4GGUFAdapter)
    config = DeepSeekV4DSparkConfig(3, 5, 256, 128799, (40, 41, 42))
    adapter.external_record = {"config": asdict(config)}
    target = DeepSeekV4Config(compress_ratios=(0,) * 43)
    adapter._validate_external_target(target)
    target.rms_norm_eps = 2e-20
    with pytest.raises(DSparkContractError, match="rms_norm_eps"):
        adapter._validate_external_target(target)


def test_incompatible_target_gate_precedes_any_weight_provider(monkeypatch):
    from dataclasses import asdict

    from mlx2 import process_env
    from mlx2.adapters import deepseek_v4_dspark_artifact, deepseek_v4_gguf
    from mlx2.runtime.drafters.deepseek_v4_dspark import (
        DeepSeekV4DSparkConfig,
        DSparkContractError,
    )
    from mlx2.runtime.gguf import weights
    from mlx2.runtime.models.deepseek_v4 import DeepSeekV4Config

    bad_target = DeepSeekV4Config(compress_ratios=(0,) * 43, rms_norm_eps=2e-20)
    side = DeepSeekV4DSparkConfig(3, 5, 256, 128799, (40, 41, 42))
    monkeypatch.setattr(process_env, "require_process_numerics", lambda *a: None)
    monkeypatch.setattr(process_env, "clear_inherited_profile", lambda *a: None)
    monkeypatch.setattr(process_env, "assert_process_profile_applied", lambda *a: None)
    monkeypatch.setattr(process_env, "PROCESS_NUMERICS", {})
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(
        deepseek_v4_gguf,
        "inspect_artifact",
        lambda path: {"identity": {}, "metadata": {}, "config": {}},
    )
    monkeypatch.setattr(
        deepseek_v4_dspark_artifact,
        "seal_path",
        lambda path: SimpleNamespace(exists=lambda: True),
    )
    monkeypatch.setattr(
        deepseek_v4_dspark_artifact,
        "inspect_sidecar",
        lambda path: {"config": asdict(side)},
    )
    monkeypatch.setattr(
        deepseek_v4_dspark_artifact,
        "verified_source_identity",
        lambda: {"source_snapshot_sha256": "f" * 64},
    )
    monkeypatch.setattr(DeepSeekV4Config, "from_gguf", lambda metadata: bad_target)
    allocations = []

    def forbidden(*a, **kw):
        allocations.append(True)
        raise AssertionError("weight provider constructed too early")

    monkeypatch.setattr(weights, "GGUFWeights", forbidden)
    adapter = DeepSeekV4GGUFAdapter.__new__(DeepSeekV4GGUFAdapter)
    adapter.external_policy = {"draft_model": "sidecar"}
    adapter.kernel_policy = split_execution_policy(None)[0]
    with pytest.raises(DSparkContractError, match="rms_norm_eps"):
        adapter._load("unused")
    assert allocations == []


@pytest.mark.parametrize(
    "value", [True, None, "0.6", -0.01, 1.01, float("nan"), float("inf")]
)
def test_confidence_policy_rejects_invalid(value):
    with pytest.raises(ValueError, match="confidence"):
        split_execution_policy(
            {"draft_model": "exact.gguf", "deepseek_draft_confidence": value}
        )


@pytest.mark.parametrize("value", [0, 0.6, 1])
def test_confidence_policy_explicit_boundaries(value):
    _, external = split_execution_policy(
        {"draft_model": "exact.gguf", "deepseek_draft_confidence": value}
    )
    assert external["deepseek_draft_confidence"] == float(value)
