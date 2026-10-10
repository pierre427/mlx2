from __future__ import annotations

import base64
import hashlib
import json
import queue
from types import SimpleNamespace

import pytest

from mlx2.media_qualification import (
    evaluate_encoder_batching_observation,
    evaluate_native_vlm_report,
)
from scripts.qualify_native_vlm_media import (
    FAMILIES,
    _batching_observation,
    _drain,
    _enable_batching_observer,
    _ordinary_bound_method,
    _parity_caches,
    evaluate_native_report,
    main,
)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _report(family):
    rev = "67599f2e8ec31bf35cbb7b02794114f20844f0bb"
    source = "a" * 64
    artifact = "b" * 64
    runtime = {"revision": rev, "source_sha256": source}
    settings = {"route": "ordinary"}
    direct = {
        "ordinary_text_parity": {
            "reference_revision": rev,
            "reference_source_sha256": source,
            "rows": [{"argmax_match": True, "max_abs": 0.0} for _ in range(5)],
        }
    }
    arms = {}
    for kind in FAMILIES[family]:
        raw = b"orig-" + kind.encode()
        changed = b"edit-" + kind.encode()
        direct[kind] = {
            "fixture": {
                "sha256": _sha(raw),
                "payload_base64": base64.b64encode(raw).decode(),
                "bytes": len(raw),
                "changed_sha256": _sha(changed),
                "changed_payload_base64": base64.b64encode(changed).decode(),
                "prompt_tokens": 30,
                "media_token_start": 4,
                "media_token_end": 15,
                "changed_prompt_tokens": 30,
                "changed_media_token_start": 4,
                "changed_media_token_end": 15,
                "media_fingerprint": "a",
                "changed_media_fingerprint": "b",
            },
            "parity": {
                "reference_revision": rev,
                "reference_source_sha256": source,
                "rows": [{"argmax_match": True, "max_abs": 0.0} for _ in range(5)],
            },
        }
        arms[kind] = {}
        for name, cached, text in (
            ("cold", 0, "same"),
            ("warm", 20, "same"),
            ("changed", 2, "edit"),
            ("return_original", 20, "same"),
        ):
            arms[kind][name] = {
                "finish_reason": "length",
                "text": text,
                "receipt": {
                    "route": "ordinary",
                    "qualification": "candidate",
                    "prompt_tokens": 30,
                    "cached_tokens": cached,
                },
            }
    batch_prompts = ("Two plus two is", "Three plus three is")

    def batch_row(prompt, request_id, text, *, cold):
        return {
            "prompt": prompt,
            "prompt_sha256": _sha(prompt.encode()),
            "request_id": request_id,
            "finish_reason": "length",
            "text": text,
            "receipt": {
                "request_id": request_id,
                "route": "ordinary",
                "qualification": "candidate",
                "cached_tokens": 0 if cold else 3,
            },
        }

    batch = {
        "peak_observed_width": 2,
        "reference_rows": [
            batch_row(batch_prompts[0], "ref-a", "4", cold=True),
            batch_row(batch_prompts[1], "ref-b", "6", cold=True),
        ],
        "rows": [
            batch_row(batch_prompts[0], "batch-a", "4", cold=False),
            batch_row(batch_prompts[1], "batch-b", "6", cold=False),
        ],
    }
    ordinary = {
        "finish_reason": "length",
        "text": "VLM_READY",
        "receipt": {
            "route": "ordinary",
            "qualification": "candidate",
            "request_controls": {
                "sampling_defaults": {"schema": "mlx2.sampling-defaults.v1"}
            },
        },
    }
    report = {
        "family": family,
        "producer_sha256": "c" * 64,
        "reference_revision": rev,
        "reference_source_sha256": source,
        "artifact_sha256": artifact,
        "source_runtime": runtime,
        "settings": settings,
        "ordinary_reference": ordinary,
        "batch": batch,
        "direct": direct,
        "arms": arms,
        "apc_stats_delta": {"hits": len(FAMILIES[family])},
        "runtime": {"serving": "identity"},
        "serving_runtime": {"serving": "identity"},
    }
    expected = {
        "producer_sha256": "c" * 64,
        "expected_source_revision": rev,
        "expected_source_sha256": source,
        "expected_artifact": artifact,
        "expected_runtime": runtime,
        "expected_settings": settings,
    }
    return report, expected


def _eval(r, e):
    return evaluate_native_report(r, **e)


def test_ordinary_method_binding_preserves_descriptors():
    class Parent:
        @staticmethod
        def static(value):
            return "static", value

        def instance(self, value):
            return self.label, value

        @classmethod
        def class_method(cls, value):
            return cls.__name__, value

    class Child(Parent):
        pass

    source = Child()
    source.label = "child instance"

    assert _ordinary_bound_method(source, "static")("x") == ("static", "x")
    assert _ordinary_bound_method(source, "instance")("x") == ("child instance", "x")
    assert _ordinary_bound_method(source, "class_method")("x") == ("Child", "x")
    assert _ordinary_bound_method(source, "missing") is None


@pytest.mark.parametrize(
    ("family", "policy_name", "policy", "expected_name", "expected_limit"),
    [
        (
            "gemma3n",
            "video_policy",
            SimpleNamespace(frame_batch_size=16),
            "_mlx2_gemma3n_vision_batching",
            16,
        ),
        (
            "minicpmo",
            "media_policy",
            SimpleNamespace(vision_batch_size=8),
            "_mlx2_minicpmo_vision_batching",
            8,
        ),
    ],
)
def test_batching_observer_reads_only_selected_family_policy(
    family, policy_name, policy, expected_name, expected_limit
):
    model = SimpleNamespace()
    adapter = SimpleNamespace(
        model=SimpleNamespace(_model=model), **{policy_name: policy}
    )

    observed = _enable_batching_observer(adapter, family)

    assert observed == {
        "schema": "mlx2.encoder-batching-observation.v1",
        "policy_limit": expected_limit,
        "enabled": True,
        "calls": [],
    }
    assert getattr(model, expected_name) is observed
    assert _batching_observation(adapter, family) == observed
    assert _batching_observation(adapter, family) is not observed


@pytest.mark.parametrize("family", ["gemma4", "unknown"])
def test_batching_observer_unsupported_family_does_not_mutate_adapter(family):
    model = SimpleNamespace(existing="preserved")
    adapter = SimpleNamespace(model=SimpleNamespace(_model=model))
    unsupported_adapter = SimpleNamespace()

    assert _enable_batching_observer(adapter, family) is None
    assert _enable_batching_observer(unsupported_adapter, family) is None
    assert vars(model) == {"existing": "preserved"}
    assert _batching_observation(adapter, family) is None
    assert _batching_observation(unsupported_adapter, family) is None


def test_parity_cache_creation_uses_runtime_factory_for_independent_models():
    source, routed = object(), object()
    calls = []

    def cache_factory(model):
        cache = object()
        calls.append((model, cache))
        return cache

    reference_cache, route_cache = _parity_caches(source, routed, cache_factory)

    assert [model for model, _ in calls] == [source, routed]
    assert reference_cache is calls[0][1]
    assert route_cache is calls[1][1]
    assert reference_cache is not route_cache


def test_native_producer_error_keeps_bounded_traceback_without_checks(
    tmp_path, monkeypatch, capsys
):
    model_path = tmp_path / "model"
    model_path.mkdir()

    monkeypatch.setattr(
        "scripts.qualify_native_vlm_media._prove_lease",
        lambda *args, **kwargs: "lease",
    )

    def fail_run(*args, **kwargs):
        raise RuntimeError("simulated parity failure")

    monkeypatch.setattr("scripts.qualify_native_vlm_media.run_family", fail_run)
    assert (
        main(
            [
                "gemma3n",
                str(model_path),
                "--generation",
                "1",
                "--cpg-owner-lock",
                str(tmp_path / "owner.lock"),
                "--cpg-task",
                "cpu-test",
            ]
        )
        == 1
    )

    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "producer_error"
    assert report["passed"] is False
    assert report["error"] == "RuntimeError: simulated parity failure"
    assert "fail_run" in report["traceback"]
    assert report["traceback"].count('File "') <= 8
    assert "checks" not in report


def test_native_stream_drain_collects_only_text_content_deltas():
    events = queue.Queue()
    for event in (
        {"delta": {"role": "assistant"}},
        {"delta": {"reasoning_content": "private reasoning"}},
        {"delta": {"content": None, "reasoning_content": "more reasoning"}},
        {"delta": {"content": "hel"}},
        {"delta": "lo"},
        {"finish_reason": "stop", "delta": {"content": "!"}, "receipt": {"ok": True}},
    ):
        events.put(event)

    result = _drain(SimpleNamespace(events=events, id="req-1"))

    assert result == {
        "finish_reason": "stop",
        "text": "hello!",
        "receipt": {"ok": True},
        "request_id": "req-1",
    }


@pytest.mark.parametrize(
    "event",
    [
        {"delta": {"content": {"text": "not text"}}, "finish_reason": "stop"},
        {"delta": ["not text"], "finish_reason": "stop"},
    ],
)
def test_native_stream_drain_rejects_invalid_content_types(event):
    events = queue.Queue()
    events.put(event)

    with pytest.raises(TypeError, match="serving delta"):
        _drain(SimpleNamespace(events=events, id="req-2"))


def test_native_stream_drain_preserves_finish_only_and_error_semantics():
    finish_events = queue.Queue()
    finish_events.put({"delta": None})
    finish_events.put({"finish_reason": "length"})
    assert _drain(SimpleNamespace(events=finish_events, id="req-3"))["text"] == ""

    error_events = queue.Queue()
    error_events.put({"error": "upstream failure"})
    with pytest.raises(RuntimeError, match="upstream failure"):
        _drain(SimpleNamespace(events=error_events, id="req-4"))


def test_native_report_evaluator_recomputes_all_supported_modalities():
    for family in FAMILIES:
        r, e = _report(family)
        checks = _eval(r, e)
        assert all(checks.values()), (family, checks)


def test_native_evaluator_rejects_identity_fixture_parity_and_media_reuse_tampering():
    r, e = _report("gemma3n")
    r["artifact_sha256"] = "d" * 64
    assert not any(_eval(r, e).values())
    r, e = _report("gemma3n")
    r["direct"]["image"]["fixture"]["payload_base64"] = base64.b64encode(
        b"tampered"
    ).decode()
    assert not _eval(r, e)["image"]
    r, e = _report("gemma3n")
    r["direct"]["image"]["parity"]["rows"][0]["argmax_match"] = False
    assert not _eval(r, e)["source_route_parity"]
    r, e = _report("gemma3n")
    r["arms"]["image"]["changed"]["receipt"]["cached_tokens"] = 9
    assert not _eval(r, e)["replay"]
    r, e = _report("gemma3n")
    del r["arms"]["image"]["return_original"]
    assert not _eval(r, e)["receipts"]
    r, e = _report("gemma3n")
    r["direct"]["image"]["fixture"]["changed_media_token_end"] = "15"
    assert not _eval(r, e)["image"]
    r, e = _report("gemma3n")
    r["apc_stats_delta"] = {"hits": True}
    assert not _eval(r, e)["prefix_reuse"]


def test_encoder_batching_observation_requires_real_bounded_overflow_chunks():
    assert evaluate_encoder_batching_observation(
        {
            "schema": "mlx2.encoder-batching-observation.v1",
            "enabled": True,
            "policy_limit": 16,
            "calls": [{"input_count": 17, "chunk_sizes": [16, 1]}],
        }
    )
    assert not evaluate_encoder_batching_observation(
        {
            "schema": "mlx2.encoder-batching-observation.v1",
            "enabled": True,
            "policy_limit": 16,
            "calls": [{"input_count": 17, "chunk_sizes": [17]}],
        }
    )
    assert not evaluate_encoder_batching_observation(
        {
            "schema": "mlx2.encoder-batching-observation.v1",
            "enabled": True,
            "policy_limit": 16,
            "calls": [{"input_count": 17, "chunk_sizes": [16]}],
        }
    )
    assert not evaluate_encoder_batching_observation(
        {
            "schema": "mlx2.encoder-batching-observation.v1",
            "policy_limit": 16,
            "calls": [{"input_count": 17, "chunk_sizes": [16, 1]}],
        }
    )
    assert not evaluate_encoder_batching_observation(
        {
            "schema": "mlx2.encoder-batching-observation.v1",
            "enabled": False,
            "policy_limit": 16,
            "calls": [{"input_count": 17, "chunk_sizes": [16, 1]}],
        }
    )
    assert not evaluate_encoder_batching_observation(
        {
            "schema": "mlx2.encoder-batching-observation.v1",
            "enabled": True,
            "policy_limit": 16,
            "calls": [{"input_count": 17, "chunk_sizes": [16, True]}],
        }
    )


def test_native_report_evaluator_refuses_unknown_family():
    with pytest.raises(ValueError, match="unsupported"):
        evaluate_native_vlm_report({}, expected_family="other", expected_harness={})


def _batch_probe(count=17, chunks=(16, 1)):
    return {
        "schema": "mlx2.encoder-batching-observation.v1",
        "enabled": True,
        "policy_limit": 16,
        "calls": [{"input_count": count, "chunk_sizes": list(chunks)}],
    }


def test_native_family_feature_gate_requires_direct_and_serving_encoder_evidence():
    report, _ = _report("gemma3n")
    revision = "67599f2e8ec31bf35cbb7b02794114f20844f0bb"
    runtime = {
        "schema": "mlx2.vlm-dependencies.v1",
        "family": "gemma3n",
        "source_sha256": "a" * 64,
        "dependency_files": 12,
        "reference_revision": revision,
    }
    harness = {"name": "native-producer", "sha256": "c" * 64}
    report.update(
        model_type="gemma3n",
        qualification_harness=harness,
        source_revision=revision,
        source_runtime=runtime,
        mlx_vlm_runtime=runtime,
        artifact=report["artifact_sha256"],
        runtime={"serving": "identity"},
        settings={"mlx_vlm": runtime},
        serving_encoder_batching=_batch_probe(),
    )
    target = report["direct"]["video"]
    target["encoder_batching"] = _batch_probe()
    target["parity"]["feature_parity"] = {
        "passed": True,
        "shapes_match": True,
        "tensor_count": 1,
        "max_abs": 0.0,
    }
    checks = evaluate_native_vlm_report(
        report, expected_family="gemma3n", expected_harness=harness
    )
    assert checks["multimodal_encoder_batching"] is True

    report["serving_encoder_batching"]["calls"][0]["chunk_sizes"] = [17]
    checks = evaluate_native_vlm_report(
        report, expected_family="gemma3n", expected_harness=harness
    )
    assert checks["multimodal_encoder_batching"] is False


def test_native_continuous_batch_is_a_real_required_gate():
    report, expected = _report("gemma4")
    assert _eval(report, expected)["continuous_batch"] is True
    report["batch"]["peak_observed_width"] = 1
    assert _eval(report, expected)["continuous_batch"] is False
    report, expected = _report("gemma4")
    report["batch"]["rows"][1] = dict(report["batch"]["rows"][0])
    assert _eval(report, expected)["continuous_batch"] is False
    report, expected = _report("gemma4")
    report["batch"]["rows"][0]["text"] = "6"
    assert _eval(report, expected)["continuous_batch"] is False
    report, expected = _report("gemma4")
    report["batch"]["rows"][0]["receipt"]["request_id"] = "other"
    assert _eval(report, expected)["continuous_batch"] is False
