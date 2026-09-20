import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "agent_client_conformance.py"


def _run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, timeout=600, check=False,
    )


def test_gpu_mode_refuses_without_ownership_and_dry_run_prints_plan(tmp_path):
    refused = _run("--model", "/models/none", "--out", str(tmp_path))
    assert refused.returncode == 2
    assert "--i-own-the-gpu" in refused.stderr
    plan = _run("--model", "/models/none", "--dry-run", "--out", str(tmp_path))
    assert plan.returncode == 0
    parsed = json.loads(plan.stdout)
    server = parsed["server"]
    assert server[server.index("--agent-compat") + 1] == "auto"
    assert parsed["tasks"] == ["add_file", "fix_test"]
    assert "--qualification-mode" in server, "mlx2.server refuses to start without it"
    qualified = _run("--model", "/models/none", "--dry-run", "--out", str(tmp_path),
                     "--server-arg=--qualification", "--server-arg=/q.json")
    assert "--qualification-mode" not in json.loads(qualified.stdout)["server"]
    assert not any(tmp_path.iterdir()), "dry run must not start anything"


def test_harness_requires_exactly_one_mode():
    assert _run().returncode != 0


@pytest.mark.skipif(
    os.environ.get("MLX2_AGENT_CLIENTS") != "1"
    or not (shutil.which("codex") and shutil.which("claude")),
    reason="set MLX2_AGENT_CLIENTS=1 with codex and claude installed",
)
def test_real_clients_complete_tool_sessions_against_scripted_server(tmp_path):
    result = _run("--scripted", "--out", str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["results"]["codex"]["patch_applied"]
    assert summary["results"]["claude"]["tool_round_trip"]


def _harness():
    spec = importlib.util.spec_from_file_location("agent_client_conformance", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(client, task, repeat=0, passed=True, cached=(0, 5), rejected=0, non_200=0):
    return {"client": client, "task": task, "repeat": repeat, "passed": passed,
            "non_200": non_200, "cached_tokens": list(cached),
            "counter_delta": {"agent_compat_custom_tool_grammar_rejected": rejected} if rejected else {}}


GOOD_COUNTS = {"agent_compat_custom_tool_calls": 3, "agent_compat_detected_codex": 4,
               "agent_compat_detected_claude_code": 4}


def test_cached_tokens_read_from_responses_and_messages_streams():
    h = _harness()
    responses = {"response": 'data: {"usage": {"input_tokens_details": {"cached_tokens": 812}}}'}
    messages = {"response": 'data: {"message": {"usage": {"cache_read_input_tokens": 0}}}'}
    assert h.request_cached_tokens(responses) == 812
    assert h.request_cached_tokens(messages) == 0
    assert h.request_cached_tokens({"response": "{}"}) is None


def test_gpu_verdict_applies_preregistered_criterion():
    h = _harness()
    clients = ["codex", "claude"]
    rows = [_row(c, t, r) for c in clients for t in ("add_file", "fix_test") for r in (0, 1)]
    verdict = h.gpu_verdict(rows, GOOD_COUNTS, clients, "auto")
    assert verdict["go"], verdict["failures"]
    assert verdict["pass_rate"] == {"codex": "4/4", "claude": "4/4"}

    # 3/4 passes; a grammar rejection on the failing task does not count.
    rows[0] = _row("codex", "add_file", passed=False, rejected=1)
    assert h.gpu_verdict(rows, GOOD_COUNTS, clients, "auto")["go"]
    # ... but a rejection on a passing task does.
    rows[1] = _row("codex", "add_file", 1, rejected=1)
    assert not h.gpu_verdict(rows, GOOD_COUNTS, clients, "auto")["go"]
    rows[1] = _row("codex", "add_file", 1)
    # 2/4 is below the bar.
    rows[2] = _row("codex", "fix_test", passed=False)
    failures = h.gpu_verdict(rows, GOOD_COUNTS, clients, "auto")["failures"]
    assert any("codex pass rate 2/4" in f for f in failures)
    rows[2] = _row("codex", "fix_test")
    # An uncached later turn, a non-200, a missing detection each fail.
    rows[3] = _row("codex", "fix_test", 1, cached=(0, 9, 0))
    assert not h.gpu_verdict(rows, GOOD_COUNTS, clients, "auto")["go"]
    rows[3] = _row("codex", "fix_test", 1, non_200=1)
    assert not h.gpu_verdict(rows, GOOD_COUNTS, clients, "auto")["go"]
    rows[3] = _row("codex", "fix_test", 1)
    no_claude = {k: v for k, v in GOOD_COUNTS.items() if "claude" not in k}
    assert not h.gpu_verdict(rows, no_claude, clients, "auto")["go"]
    # Detection is not required for the forced-on comparison arm.
    assert h.gpu_verdict(rows, no_claude, clients, "on")["go"]
