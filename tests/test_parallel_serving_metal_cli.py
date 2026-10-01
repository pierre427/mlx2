"""The HTTP Metal harness cannot start inference in a dry run or unowned run."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/validate_parallel_serving_metal.py"
)
spec = importlib.util.spec_from_file_location("parallel_serving_metal_cli", SCRIPT)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


def arguments(*extra):
    return harness.parser().parse_args(
        [
            "--model",
            "/missing",
            "--draft",
            "/missing",
            "--out",
            "/missing",
            "--dry-run",
            *extra,
        ]
    )


def test_http_validation_dry_run_cannot_import_tensor_or_spawn_server():
    code = """
import sys, runpy, subprocess
class Guard:
    def find_spec(self, fullname, *args, **kwargs):
        if fullname.startswith(('mlx', 'transformers', 'numpy')):
            raise RuntimeError('tensor import forbidden: ' + fullname)
sys.meta_path.insert(0, Guard())
def forbidden(*args, **kwargs):
    raise RuntimeError('subprocess launch forbidden')
subprocess.Popen = forbidden
sys.argv = [sys.argv[1], '--model', '/missing', '--draft', '/missing',
            '--out', '/missing', '--dry-run', '--target-verify-row-exact']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["will_execute"] is False
    assert plan["target_verify_row_exact"] is True


def test_http_validation_refuses_unowned_gpu_before_reading_artifacts():
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model",
            "/missing",
            "--draft",
            "/missing",
            "--out",
            "/missing",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "i-own-the-gpu" in result.stderr


def test_default_requests_remain_exact_and_default_timeouts_remain_240():
    plan = harness.preflight(arguments())
    assert plan["request_timeout_seconds"] == 240
    assert plan["request_budgets"] == {
        "mode": "legacy_default_payloads",
        "cold_warm_reference": 48,
        "mixed": [16, 32, 48, 64],
        "cancel_stream": 128,
        "performance_qualified": False,
    }
    assert harness.completion_body(
        "target", "hello", 48, route="ordinary_reference"
    ) == {
        "model": "target",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 48,
        "temperature": 0,
    }


@pytest.mark.parametrize("cap", [17, 32, 47, 64])
def test_bounded_override_preserves_matching_reference_and_varied_lane_budgets(cap):
    plan = harness.preflight(
        arguments("--max-tokens", str(cap), "--request-timeout-seconds", "900")
    )
    budgets = plan["request_budgets"]
    assert budgets["cold_warm_reference"] == budgets["cancel_stream"] == cap
    assert len(set(budgets["mixed"])) == 4
    assert max(budgets["mixed"]) <= cap
    assert min(budgets["mixed"]) > 0
    assert plan["request_timeout_seconds"] == 900
    ordinary = harness.completion_body(
        "target", "hello", cap, route="ordinary_reference"
    )
    candidate = harness.completion_body("target", "hello", cap, route="xpress_pool")
    assert {k: v for k, v in candidate.items() if k != "session_id"} == ordinary
    assert candidate["session_id"].startswith("pool-")
    assert not plan["performance_claim"]


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--max-tokens", "16"),
        ("--max-tokens", "65"),
        ("--request-timeout-seconds", "0"),
        ("--request-timeout-seconds", "901"),
    ],
)
def test_budget_and_request_timeout_outside_bounds_fail_before_server_launch(
    flag, value
):
    with pytest.raises(ValueError):
        harness.preflight(arguments(flag, value))


class SSE:
    def __init__(self, terminal=False):
        self.closed = False
        self.terminal = terminal

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def __iter__(self):
        for i in range(3):
            yield (
                "data: "
                + json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {"content": str(i)},
                                "finish_reason": "length"
                                if self.terminal and i == 1
                                else None,
                            }
                        ]
                    }
                )
                + "\n"
            ).encode()


def test_cancellation_closes_actual_active_response_with_override_timeout(monkeypatch):
    response = SSE()
    calls = []

    def urlopen(request, *, timeout):
        calls.append((json.loads(request.data), timeout))
        return response

    monkeypatch.setattr(harness.urllib.request, "urlopen", urlopen)
    body = harness.completion_body("target", "hello", 17, route="xpress_pool")
    receipt = harness.stream("http://example.invalid", body, cancel=True, timeout=900)
    assert calls == [({**body, "stream": True}, 900)]
    assert receipt["events"] == 2 and receipt["text"] == "01"
    assert receipt["client_closed_early"] and receipt["cancelled_while_active"]
    assert response.closed


def test_completed_stream_cannot_be_reported_as_active_cancellation(monkeypatch):
    response = SSE(terminal=True)
    monkeypatch.setattr(
        harness.urllib.request, "urlopen", lambda *_args, **_kwargs: response
    )
    with pytest.raises(AssertionError, match="finished before active"):
        harness.stream("http://example.invalid", {}, cancel=True)
    assert response.closed


def test_adaptive_policy_pinned_config_only_reaches_enabled_routes(tmp_path):
    import hashlib

    config = {
        "continuation_costs": {"1": list(range(1, 17)), "15": list(range(2, 18))},
        "draft_cost": 0.25,
        "mode": "per_request",
        "min_observations": 2,
    }
    path = tmp_path / "adaptive.json"
    raw = json.dumps(config).encode()
    path.write_bytes(raw)
    selected = arguments("--adaptive-policy", str(path))
    plan = harness.preflight(selected)
    record = plan["adaptive_policy"]
    assert (
        record["config"] == config
        and record["input_sha256"] == hashlib.sha256(raw).hexdigest()
    )
    assert record["input_bytes"] == len(raw) and record["validation_num_draft"] == 15
    assert record["cost_provenance"]["cost_freshness_verified"] is False
    assert record["cost_provenance"]["artifact_environment_binding_verified"] is False
    assert record["cost_provenance"]["context_applicability_verified"] is False
    assert harness.route_policy(selected, plan, "ordinary_reference") is None
    for route in ("xpress_full", "xpress_pool"):
        assert (
            harness.route_policy(selected, plan, route)["adaptive_verification"]
            == config
        )
    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "config.json").write_text(json.dumps({"num_hidden_layers": 3}))
    selected.draft = draft
    windowed = harness.route_policy(selected, plan, "xpress_windowed")
    assert windowed["adaptive_verification"] == config
    assert windowed["draft_attention_windows"] == [32, 32, 32]
    ordinary = harness.preflight(arguments())
    assert "adaptive_policy" not in ordinary
    base = {
        "draft_model": "/missing",
        "num_draft": 15,
        "xpress_num_passes": 6,
        "target_verify_row_exact": False,
    }
    assert harness.route_policy(arguments(), ordinary, "xpress_full") == base
    assert harness.route_policy(arguments(), ordinary, "xpress_pool") == {
        **base,
        "continuation_pool": {"limit": 15},
    }


@pytest.mark.parametrize(
    "raw,message",
    [
        (b"null", "dictionary"),
        (b"false", "dictionary"),
        (b"[]", "dictionary"),
        (b'{"verification_costs":[1,2]}', "cost model"),
        (b'{"draft_cost":1,"draft_cost":2}', "duplicate"),
        (b'{"draft_cost":NaN}', "nonfinite"),
        (b"\xff", "UTF-8 JSON"),
        (b"{", "UTF-8 JSON"),
    ],
)
def test_adaptive_policy_file_fails_closed_without_model_access(tmp_path, raw, message):
    path = tmp_path / "bad.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match=message):
        harness.preflight(arguments("--adaptive-policy", str(path)))


def test_adaptive_policy_read_is_bounded_before_cpu_library_import(tmp_path):
    path = tmp_path / "large.json"
    path.write_bytes(b" " * ((1 << 20) + 1))
    with pytest.raises(ValueError, match="1 MiB"):
        harness.preflight(arguments("--adaptive-policy", str(path)))


def test_optional_adaptive_dryrun_reads_only_explicit_policy_without_mlx(tmp_path):
    path = tmp_path / "adaptive.json"
    path.write_text(json.dumps({"verification_costs": list(range(1, 17))}))
    code = """
import sys,runpy,subprocess
class Guard:
 def find_spec(self,fullname,*args,**kwargs):
  if fullname=='mlx' or fullname.startswith(('mlx.','transformers','mlx2.runtime.models.')):
   raise RuntimeError('live model import forbidden: '+fullname)
sys.meta_path.insert(0,Guard())
def forbidden(*args,**kwargs):raise RuntimeError('server forbidden')
subprocess.Popen=forbidden
sys.argv=[sys.argv[1],'--model','/missing','--draft','/missing','--out','/missing','--dry-run','--adaptive-policy',sys.argv[2]]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(SCRIPT), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["adaptive_policy"]["config"]["verification_costs"] == list(range(1, 17))
    assert not plan["will_execute"] and not plan["qualified"]
