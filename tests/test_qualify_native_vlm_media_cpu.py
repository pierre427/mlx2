from __future__ import annotations

import base64
import hashlib

import pytest

from mlx2.media_qualification import (
    evaluate_encoder_batching_observation,
    evaluate_native_vlm_report,
)
from scripts.qualify_native_vlm_media import FAMILIES, evaluate_native_report


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
