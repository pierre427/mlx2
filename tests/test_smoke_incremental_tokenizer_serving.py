import json
from pathlib import Path

from scripts import smoke_incremental_tokenizer_serving as smoke

REVISION = "a" * 64


def _receipt(action, *, revision=REVISION):
    return {
        "schema": "mlx2.incremental-tokenizer-cache.v1",
        "implemented": True,
        "qualified": False,
        "selected": True,
        "observed_used": action == "incremental_hit",
        "serving_qualified": False,
        "action": action,
        "exact": True,
        "tokenizer_revision": revision,
        "refusal": None,
    }


def _status():
    return {
        "incremental_tokenizer_cache": {
            "implemented": True,
            "qualified": False,
            "selected": True,
            "observed_used": True,
            "serving_qualified": False,
            "tokenizer_revision": REVISION,
            "refusal": None,
            "cold_validations": 1,
            "incremental_hits": 1,
        }
    }


def test_dry_run_prints_one_bounded_default_off_candidate_service(capsys, tmp_path):
    out = tmp_path / "receipt.json"
    assert (
        smoke.main(
            [
                "--model",
                "/nonexistent/flash-next",
                "--out",
                str(out),
                "--dry-run",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    command = plan["server_command"]
    assert plan["will_execute"] is False
    assert command.count("mlx2.server") == 1
    assert command[command.index("--incremental-tokenizer-cache-entries") + 1] == "4"
    assert "--ordinary" in command
    assert "--max-lanes" in command
    assert plan["requests"] == [
        {"name": "cold", "expected_action": "ordinary_full_validation"},
        {"name": "grown", "expected_action": "incremental_hit"},
    ]
    assert not out.exists()


def test_execution_refuses_without_gpu_ownership_before_launch(
    monkeypatch, capsys, tmp_path
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("subprocess launch reached without GPU ownership")

    monkeypatch.setattr(smoke.subprocess, "Popen", forbidden)
    code = smoke.main(
        [
            "--model",
            "/nonexistent/flash-next",
            "--out",
            str(tmp_path / "receipt.json"),
        ]
    )
    assert code == 2
    assert "--i-own-the-gpu" in capsys.readouterr().err


def test_evaluate_accepts_exact_current_cold_and_incremental_receipts():
    cold = {"mlx2": {"prompt_tokenization": _receipt("ordinary_full_validation")}}
    grown = {"mlx2": {"prompt_tokenization": _receipt("incremental_hit")}}
    assert smoke.evaluate(cold, grown, _status()) == []


def test_evaluate_rejects_stale_revision_wrong_action_and_missing_counters():
    cold = {
        "mlx2": {
            "prompt_tokenization": _receipt("host_prompt_cache_hit", revision="b" * 64)
        }
    }
    grown = {"mlx2": {"prompt_tokenization": _receipt("incremental_hit")}}
    status = _status()
    status["incremental_tokenizer_cache"]["incremental_hits"] = 0
    failures = smoke.evaluate(cold, grown, status)
    assert any("action" in failure for failure in failures)
    assert any("revision" in failure for failure in failures)
    assert "status: incremental_hits < 1" in failures


def test_server_environment_pins_this_checkout_first(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/other/src")
    first = smoke.server_environment()["PYTHONPATH"].split(smoke.os.pathsep)[0]
    assert Path(first) == smoke.ROOT / "src"


def test_stop_uses_process_group_term_and_records_clean_shutdown(monkeypatch):
    signals = []

    class Process:
        pid = 123
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, *, timeout):
            assert timeout == 7
            self.returncode = -15
            return self.returncode

    monkeypatch.setattr(
        smoke.os, "killpg", lambda pid, signal: signals.append((pid, signal))
    )
    result = smoke._stop(Process(), timeout=7)
    assert result == {
        "was_running": True,
        "terminated": True,
        "killed": False,
        "returncode": -15,
    }
    assert signals == [(123, smoke.signal.SIGTERM)]
