import json
from pathlib import Path

from scripts import smoke_tokenizers_v1_serving_ab as smoke


def _identity():
    return {
        "manifest": str(smoke.MANIFEST.resolve()),
        "manifest_expected_sha256": smoke.MANIFEST_SHA256,
        "manifest_actual_sha256": smoke.MANIFEST_SHA256,
        "artifact": str(smoke.ARTIFACT.resolve()),
        "files": {},
        "failures": [],
        "go": True,
    }


def _response(*, text="same", ids=None, prompt_tokens=1234):
    receipt = {
        "request_controls": {"skip_writing_prefix_cache": True},
    }
    if ids is not None:
        receipt["route_receipt"] = {"output_token_ids": ids}
    return {
        "choices": [{"text": text}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 2,
            "total_tokens": prompt_tokens + 2,
        },
        "mlx2": receipt,
    }


def _arm(*, candidate, response=None):
    v1 = {
        "implemented": True,
        "qualified": "cpu_exact_encode_candidate" if candidate else False,
        "selected": candidate,
        "observed_used": candidate,
        "serving_qualified": False,
    }
    if candidate:
        v1.update(
            manifest_path=str(smoke.MANIFEST.resolve()),
            counts={"successful_encodes": 1, "failures": 0},
        )
    return {
        "http_status": 200,
        "response": response or _response(ids=[7, 8]),
        "status": {
            "tokenizers_v1": v1,
            "incremental_tokenizer_cache": {"selected": False},
        },
    }


def test_live_retained_manifest_and_artifact_identity_are_exact():
    result = smoke.preflight()
    assert result["go"] is True, result["failures"]
    assert result["manifest_actual_sha256"] == smoke.MANIFEST_SHA256
    assert Path(result["artifact"]) == smoke.ARTIFACT.resolve()
    assert all(record["exists"] for record in result["files"].values())
    assert all(
        record["actual_sha256"] == record["expected_sha256"]
        for record in result["files"].values()
    )


def test_dry_run_has_exact_sequential_arms_and_candidate_off_by_default(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setattr(smoke, "preflight", _identity)
    out = tmp_path / "receipt.json"
    assert smoke.main(["--out", str(out), "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["will_execute"] is False
    assert plan["arms"] == [
        {"name": "ordinary", "worker_manifest": None},
        {"name": "candidate", "worker_manifest": str(smoke.MANIFEST.resolve())},
    ]
    command = plan["server_command"]
    assert command.count("mlx2.server") == 1
    assert command[command.index("--incremental-tokenizer-cache-entries") + 1] == "0"
    assert plan["apcv2_write_suppression"] is True
    assert plan["prompt_characters"] > 8192
    assert not out.exists()


def test_execution_refuses_without_owned_gpu_before_launch(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setattr(smoke, "preflight", _identity)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("service launch reached without ownership")

    monkeypatch.setattr(smoke.subprocess, "Popen", forbidden)
    code = smoke.main(["--out", str(tmp_path / "receipt.json")])
    assert code == 2
    assert "--i-own-gpu" in capsys.readouterr().err


def test_arm_environments_remove_inherited_selection_and_pin_candidate(monkeypatch):
    monkeypatch.setenv(smoke.WORKER_ENV, "/wrong/manifest.json")
    ordinary = smoke.server_environment(candidate=False)
    candidate = smoke.server_environment(candidate=True)
    assert smoke.WORKER_ENV not in ordinary
    assert candidate[smoke.WORKER_ENV] == str(smoke.MANIFEST.resolve())
    assert Path(candidate["PYTHONPATH"].split(smoke.os.pathsep)[0]) == (
        smoke.ROOT / "src"
    )


def test_evaluate_accepts_exact_text_ids_prompt_count_and_worker_counters():
    ordinary = _arm(candidate=False)
    candidate = _arm(candidate=True)
    assert smoke.evaluate(ordinary, candidate) == []


def test_evaluate_rejects_drift_wrong_status_and_missing_write_suppression():
    ordinary = _arm(candidate=False)
    candidate = _arm(
        candidate=True,
        response=_response(text="different", ids=[9], prompt_tokens=1235),
    )
    candidate["status"]["tokenizers_v1"]["observed_used"] = False
    candidate["status"]["tokenizers_v1"]["counts"] = {
        "successful_encodes": 0,
        "failures": 1,
    }
    candidate["status"]["incremental_tokenizer_cache"]["selected"] = True
    candidate["response"]["mlx2"]["request_controls"]["skip_writing_prefix_cache"] = (
        False
    )
    failures = smoke.evaluate(ordinary, candidate)
    assert any("response text differs" in failure for failure in failures)
    assert any("output token IDs differ" in failure for failure in failures)
    assert any("prompt-token count differs" in failure for failure in failures)
    assert any("successful_encodes" in failure for failure in failures)
    assert any("failures=1" in failure for failure in failures)
    assert any("incremental tokenizer cache" in failure for failure in failures)
    assert any("APCv2 write suppression" in failure for failure in failures)
