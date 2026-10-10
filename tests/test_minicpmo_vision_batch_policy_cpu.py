"""CPU-only MiniCPM-o policy identity and safe-default tests."""

from types import SimpleNamespace

import pytest

from mlx2.adapters import mlx_vlm
from mlx2.adapters.mlx_vlm import MiniCPMOAdapter
from mlx2.adapters.multimodal import MiniCPMOExecutionPolicy
from mlx2.media_qualification import evaluate_encoder_sequential_observation
from scripts.qualify_native_vlm_media import _observe_minicpmo_route_features


def test_artifact_batch_preference_does_not_enable_unqualified_batching():
    policy = MiniCPMOExecutionPolicy.from_config(
        {
            "batch_vision_input": True,
            "vision_batch_size": 16,
            "slice_mode": False,
            "chunk_input": False,
        }
    )

    assert policy.batch_vision_input is False
    assert policy.vision_batch_size == 1
    assert policy.receipt()["batch_vision_input"] is False
    assert policy.as_dict() == policy.receipt()


def test_batching_requires_a_scoped_explicit_option_and_is_identity_bound():
    Image = pytest.importorskip("PIL.Image")
    config = {
        "batch_vision_input": True,
        "vision_batch_size": 16,
        "slice_mode": False,
    }
    default = MiniCPMOExecutionPolicy.from_config(config)
    opt_in = MiniCPMOExecutionPolicy.from_config(
        config,
        media_options={"batch_vision_input": True, "vision_batch_size": 8},
    )

    assert default.receipt() != opt_in.receipt()
    assert opt_in.receipt()["batch_vision_input"] is True
    assert opt_in.receipt()["vision_batch_size"] == 8
    images = [Image.new("RGB", (4, 4)) for _ in range(17)]
    assert [len(batch) for batch in opt_in.vision_batches(images)] == [8, 8, 1]


@pytest.mark.parametrize(
    "media_options",
    [
        [],
        {"batch_vision_input": 1},
        {"batch_vision_input": True, "vision_batch_size": True},
        {"batch_vision_input": True, "vision_batch_size": 1},
        {"vision_batch_size": 2},
        {"unknown": True},
    ],
)
def test_malformed_or_ambiguous_opt_in_fails_closed(media_options):
    with pytest.raises(ValueError):
        MiniCPMOExecutionPolicy.from_config({}, media_options=media_options)


def test_adapter_routes_scoped_opt_in_into_runtime_policy_and_receipt(monkeypatch):
    installed = {}

    def fake_base_init(adapter, model_path, *, execution_policy=None):
        assert execution_policy is None
        adapter.identity = {"config": {"batch_vision_input": True}}
        adapter.model = SimpleNamespace()
        adapter.media_feature_cache = object()

    monkeypatch.setattr(mlx_vlm._MLXVLMAdapter, "__init__", fake_base_init)
    monkeypatch.setattr(
        mlx_vlm,
        "install_minicpmo_vision_batching",
        lambda model, policy: installed.update(policy=policy.receipt()),
    )
    monkeypatch.setattr(
        mlx_vlm,
        "install_media_feature_cache",
        lambda *args, **kwargs: None,
    )

    adapter = MiniCPMOAdapter(
        "unused",
        execution_policy={
            "minicpm_o_media": {
                "batch_vision_input": True,
                "vision_batch_size": 4,
            }
        },
    )

    assert adapter.policy.as_dict() == installed["policy"]
    assert installed["policy"]["batch_vision_input"] is True
    assert installed["policy"]["vision_batch_size"] == 4


def test_adapter_rejects_unscoped_controls_before_loading(monkeypatch):
    def should_not_load(*args, **kwargs):
        pytest.fail("invalid policy reached model loading")

    monkeypatch.setattr(mlx_vlm._MLXVLMAdapter, "__init__", should_not_load)
    with pytest.raises(ValueError, match="unsupported MiniCPM-o"):
        MiniCPMOAdapter("unused", execution_policy={"batch_vision_input": True})


def test_sequential_execution_evidence_requires_one_tower_call_per_image():
    policy = {
        **MiniCPMOExecutionPolicy(slice_mode=False).receipt(),
    }
    parity = {
        "passed": True,
        "shapes_match": True,
        "tensor_count": 1,
        "max_abs": 0.0,
    }
    evidence = {
        "schema": "mlx2.encoder-execution-observation.v1",
        "policy": policy,
        "input_count": 2,
        "tower_call_batch_sizes": [1, 1],
    }

    assert evaluate_encoder_sequential_observation(evidence, policy, parity)
    evidence["tower_call_batch_sizes"] = [2]
    assert not evaluate_encoder_sequential_observation(evidence, policy, parity)
    evidence["tower_call_batch_sizes"] = [1, 1]
    evidence["tower_call_batch_sizes"][1] = 2
    assert not evaluate_encoder_sequential_observation(evidence, policy, parity)


def test_route_feature_observation_excludes_reference_and_forward_tower_calls():
    calls = []

    def tower(pixels):
        calls.append("tower")
        return pixels

    source = SimpleNamespace(vision_tower=tower)
    pixels = SimpleNamespace(shape=(1, 3, 4, 4))
    source.vision_tower(pixels)  # ordinary source-reference feature call

    def route_features(batch):
        return (source.vision_tower(batch), source.vision_tower(batch))

    policy = MiniCPMOExecutionPolicy(slice_mode=False)
    result, observation = _observe_minicpmo_route_features(
        source, route_features, (pixels,), policy, input_count=2
    )
    source.vision_tower(pixels)  # ordinary/routed prefill forward call

    assert len(result) == 2
    assert calls == ["tower"] * 4
    assert observation["tower_call_batch_sizes"] == [1, 1]
    assert observation["input_count"] == 2
    assert source.vision_tower is tower
def test_strict_adapter_reader_selects_only_the_receipts_policy_gate(monkeypatch):
    from mlx2.adapters.mlx_vlm import MINICPMO
    from mlx2.media_qualification import (
        NATIVE_VLM_MEDIA_CHECKS,
        evaluate_encoder_batching_observation,
        resolve_conditional_qualification_checks,
    )
    from mlx2.qualification import (
        APPROVED_MEDIA_PRODUCERS,
        PINNED_MEDIA_SOURCE_REVISIONS,
        validate_adapter_qualification,
    )

    harness = {"name": "candidate-vlm", "sha256": "c" * 64}
    rules = MINICPMO.metadata["conditional_qualification_checks"]

    def evaluator(report, **kwargs):
        selected = resolve_conditional_qualification_checks(
            kwargs["expected_policy_checks"], report["settings"]
        )
        observed = {name: True for name in NATIVE_VLM_MEDIA_CHECKS["minicpmo"]}
        if "multimodal_encoder_sequential" in selected:
            parity = report["direct"]["image"]["parity"]
            feature_parity = parity["feature_parity"]
            observed["multimodal_encoder_sequential"] = (
                evaluate_encoder_sequential_observation(
                    feature_parity["encoder_execution"],
                    report["settings"]["adapter_policy"],
                    feature_parity,
                )
            )
            observed.pop("multimodal_encoder_batching")
        else:
            observed["multimodal_encoder_batching"] = (
                evaluate_encoder_batching_observation(
                    report.get("serving_encoder_batching")
                )
            )
            observed.pop("multimodal_encoder_sequential")
        return observed

    monkeypatch.setitem(
        APPROVED_MEDIA_PRODUCERS,
        "minicpmo",
        (harness, evaluator, NATIVE_VLM_MEDIA_CHECKS["minicpmo"], True),
    )
    runtime = {"source": "runtime"}
    artifact = "weights"
    settings = {
        "adapter_policy": MiniCPMOExecutionPolicy(slice_mode=False).receipt()
    }
    report = {
        "schema": "mlx2.media-serving-qualification.v1",
        "model_type": "minicpmo",
        "qualification_harness": harness,
        "source_revision": PINNED_MEDIA_SOURCE_REVISIONS["minicpmo"],
        "runtime": runtime,
        "artifact": artifact,
        "settings": settings,
        "conditional_qualification_checks": [
            {**rule, "setting": list(rule["setting"])} for rule in rules
        ],
        "direct": {
            "image": {
                "parity": {
                    "feature_parity": {
                        "passed": True,
                        "shapes_match": True,
                        "tensor_count": 1,
                        "max_abs": 0.0,
                        "encoder_execution": {
                            "schema": "mlx2.encoder-execution-observation.v1",
                            "policy": settings["adapter_policy"],
                            "input_count": 2,
                            "tower_call_batch_sizes": [1, 1],
                        },
                    },
                }
            }
        },
        "checks": {},
        "passed": True,
    }
    report["checks"] = {
        name: {"passed": value}
        for name, value in evaluator(
            report, expected_policy_checks=rules
        ).items()
    }
    assert "multimodal_encoder_sequential" in report["checks"]
    assert "multimodal_encoder_batching" not in report["checks"]
    assert all(
        validate_adapter_qualification(
            report,
            runtime=runtime,
            artifact=artifact,
            settings=settings,
            descriptor=MINICPMO,
        ).values()
    )

    opt_in_settings = {
        "adapter_policy": {
            **settings["adapter_policy"],
            "batch_vision_input": True,
            "vision_batch_size": 2,
        }
    }
    with pytest.raises(ValueError, match="does not match serving settings"):
        validate_adapter_qualification(
            report,
            runtime=runtime,
            artifact=artifact,
            settings=opt_in_settings,
            descriptor=MINICPMO,
        )
    report["settings"] = opt_in_settings
    report["checks"] = {
        name: {"passed": value}
        for name, value in evaluator(
            report, expected_policy_checks=rules
        ).items()
    }
    assert "multimodal_encoder_batching" in report["checks"]
    assert "multimodal_encoder_sequential" not in report["checks"]
    assert report["checks"]["multimodal_encoder_batching"]["passed"] is False
    report["passed"] = False
    with pytest.raises(ValueError, match="traces are missing or failed"):
        validate_adapter_qualification(
            report,
            runtime=runtime,
            artifact=artifact,
            settings=opt_in_settings,
            descriptor=MINICPMO,
        )
