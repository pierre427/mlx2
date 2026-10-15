"""Static and pure-control contracts for the owned Qwen checkpoint probe."""

import math
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest


def _probe(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return runpy.run_path(str(scripts / "probe_qwen38_checkpoint_pressure.py"))


def test_probe_uses_bounded_independent_arms_and_fresh_evidence(monkeypatch):
    probe = _probe(monkeypatch)
    source = Path(probe["__file__"]).read_text()
    assert probe["PHASES"] == ("greedy", "sampled", "corruption", "cleanup")
    assert "if target.exists():" in source
    assert 'parser.error("refusing to overwrite evidence")' in source
    assert "output.mkdir(parents=True)" in source
    assert '"--phase"' in source and "subprocess.run(" in source


def test_dynamic_step_bound_leaves_decode_and_finish_guard(monkeypatch):
    bound = _probe(monkeypatch)["executor_step_bound"]
    assert bound(32768, 128, 8) == 280
    assert bound(4096, 128, 4) == 52


def test_checkpoint_plan_is_nonterminal_and_deduplicated(monkeypatch):
    due = _probe(monkeypatch)["checkpoint_due"]
    live = [SimpleNamespace(finish_reason=None)]
    terminal = [SimpleNamespace(finish_reason="length")]
    assert due(live, 3, (1, 3, 5), set())
    assert not due([], 3, (1, 3, 5), set())
    assert not due(terminal, 3, (1, 3, 5), set())
    assert not due(live, 3, (1, 3, 5), {3})
    assert not due(live, 4, (1, 3, 5), set())


@pytest.mark.parametrize(
    ("startup", "phase"),
    [
        (True, 1200.0),
        (600.0, False),
        (math.nan, 1200.0),
        (600.0, math.inf),
        (0.0, 1200.0),
        (600.0, -1.0),
        (900.0001, 1200.0),
        (600.0, 1800.0001),
    ],
)
def test_probe_rejects_unbounded_or_nonfinite_timeouts(monkeypatch, startup, phase):
    validate = _probe(monkeypatch)["validate_timeouts"]
    with pytest.raises(ValueError, match="must be finite"):
        validate(startup, phase)


def test_probe_accepts_finite_timeout_ceiling(monkeypatch):
    probe = _probe(monkeypatch)
    probe["validate_timeouts"](
        probe["MAX_STARTUP_TIMEOUT_SECONDS"],
        probe["MAX_PHASE_TIMEOUT_SECONDS"],
    )


def test_probe_binds_source_at_both_phase_boundaries(monkeypatch):
    probe = _probe(monkeypatch)
    source = Path(probe["__file__"]).read_text()
    assert '"src/mlx2/runtime/sample_utils.py"' in source
    assert '"src/mlx2/adapters/qwen38_27b.py"' in source
    assert '"src/mlx2/serving.py"' in source
    assert "def greedy_sampler(values):" in source
    assert source.count("samplers=[greedy_sampler]") == 2
    assert 'raise RuntimeError("phase must start at nominal thermal state")' in source
    assert 'raise TimeoutError("checkpoint diagnostic phase deadline")' in source
    assert 'report["checkpoint_owner_cleaned"]' in source
    assert 'report["artifact_identity_unchanged"]' in source
    assert 'report["source_sha256_end"] = source_hashes()' in source
    assert 'report["source_unchanged"]' in source
    assert 'raise RuntimeError("checkpoint diagnostic source changed during phase")' in source
    assert 'raise RuntimeError(f"thermal watcher failed: {monitor_errors[0]}")' in source
    assert 'except subprocess.TimeoutExpired:' in source
