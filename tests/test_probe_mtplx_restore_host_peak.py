"""CPU-only source, sampler, and CLI checks; no model or service execution."""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from scripts.research import probe_mtplx_restore_host_peak as probe
from scripts.research.probe_mtplx_restore_host_peak import (
    Sampler,
    missing_restore_phases,
    parked_on_disk,
    phase_lines,
    summarize,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/research/probe_mtplx_restore_host_peak.py"


def test_parked_session_can_have_multiple_disk_checkpoints():
    state = {"park_pending": 0, "resident_entries": 0, "disk_entries": 2}
    assert parked_on_disk(state)
    assert not parked_on_disk({**state, "resident_entries": 1})
    assert not parked_on_disk({**state, "disk_entries": 0})


def test_phase_proof_accepts_optional_second_checkpoint_after_stamp():
    observed = {"materialize_block_file_1_before", "materialize_block_file_1_after",
                "payload_read_1_before", "payload_read_1_after",
                "payload_staging_1_before", "payload_staging_1_after",
                "load_prompt_cache_1_before", "load_prompt_cache_2_after",
                "mx_eval_1_before", "mx_eval_1_after",
                "freeze_prompt_cache_1_before", "freeze_prompt_cache_1_after",
                "publication_before", "publication_after"}
    assert missing_restore_phases(observed) == []
    assert missing_restore_phases(observed - {"publication_after"}) == ["publication_after"]


def test_live_owner_must_match_this_gpu_queue_session(tmp_path, monkeypatch):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    owner = {"lease_id": "session-cell", "session": "session", "pid": 123}
    for path in (left, right):
        path.write_text(json.dumps(owner))
    monkeypatch.setattr(probe, "LOCKS", (left, right))
    monkeypatch.setenv("GPUQ_SESSION", "other-session")
    with pytest.raises(RuntimeError, match="session does not match"):
        probe.gpu_owner()
    monkeypatch.setenv("GPUQ_SESSION", "session")
    assert probe.gpu_owner() == owner


def test_phase_map_brackets_restore_operations_and_publication():
    labels = [label for values in phase_lines().values() for label in values]
    for prefix in ("materialize_block_file_1", "payload_read_1", "payload_staging_1", "load_prompt_cache_1",
                   "mx_eval_1", "freeze_prompt_cache_1", "publication"):
        assert prefix + "_before" in labels
        assert prefix + "_after" in labels


def test_phase_trace_observes_actual_worker_read_and_model_thread_publication(tmp_path):
    from test_apcv2_sessions_persistence import _identity, _persistent, _state

    apc = _persistent(tmp_path)
    assert apc._cow_branching
    key, tokens, tag = _identity(), list(range(16)), ("tenant-a", "trace")
    seen = []
    hit = None
    try:
        apc.store(key, tokens, [_state(16)], session_tag=tag)
        apc.park_session(*tag, ttl_seconds=60)
        codes = probe.restore_trace_codes(type(apc))
        trace = probe.phase_tracer(codes, phase_lines(), lambda label:
            seen.append((label, threading.get_ident())))
        original_trace, original_thread_trace = sys.gettrace(), threading.gettrace()
        threading.settrace_all_threads(trace)
        try:
            hit = apc.lookup(key, tokens + [99])
        finally:
            threading.settrace_all_threads(original_thread_trace)
            sys.settrace(original_trace)
        assert hit.hit
        labels = {label for label, _thread in seen}
        assert missing_restore_phases(labels) == []
        model_thread = threading.get_ident()
        assert all(thread != model_thread for label, thread in seen
                   if label.startswith(("materialize_block_file", "payload_read")))
        assert all(thread == model_thread for label, thread in seen
                   if label.startswith(("load_prompt_cache", "mx_eval", "freeze_prompt_cache", "publication")))
        ordered = [label for label, _thread in seen]
        assert ordered.index("payload_read_1_after") < ordered.index("load_prompt_cache_1_before")
        assert ordered.index("load_prompt_cache_1_after") < ordered.index("publication_before")
    finally:
        if hit is not None and hit.cache is not None:
            hit.cache.close()
        apc.close()


def test_phase_map_refuses_when_staged_read_seam_changes(tmp_path):
    source = tmp_path / "apc.py"
    source.write_text(probe.APC_SOURCE.read_text().replace("handle.read(remaining + 1)", "handle.readall()"))
    with pytest.raises(RuntimeError, match="instrumentation refused"):
        phase_lines(source)


def test_sampler_has_interval_and_immediate_phase_samples():
    count = 0

    def read():
        nonlocal count
        count += 1
        return {"host_available_bytes": 100 - count,
                "physical_footprint_bytes": 100 + count,
                "mlx_active_bytes": count, "mlx_cache_bytes": count,
                "mlx_peak_bytes": count}

    sampler = Sampler(read, 10)
    sampler.start()
    time.sleep(0.035)
    sampler.sample("publication_before")
    sampler.stop()
    assert len(sampler.rows) >= 5
    assert sampler.rows[0]["phase"] == "pre_restore"
    assert sampler.rows[-1]["phase"] == "post_restore"
    assert any(row["phase"] == "publication_before" for row in sampler.rows)
    assert all(a["elapsed_ns"] <= b["elapsed_ns"] for a, b in zip(sampler.rows, sampler.rows[1:]))
    summary = summarize(sampler.rows)
    assert summary["host_available_bytes"]["delta_from_baseline"] < 0
    assert summary["physical_footprint_bytes"]["delta_from_baseline"] > 0


def test_dry_run_pins_inputs_without_loading_mlx_or_creating_output(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n")
    output = tmp_path / "run"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--model", str(model), "--output-dir", str(output),
         "--host-floor-gib", "8", "--allocator-cache-limit-gib", "8",
         "--sample-ms", "15", "--dry-run"],
        check=True, capture_output=True, text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["requires_gpu_owner"] is True
    assert plan["sample_ms"] == 15
    assert plan["execution_policy"]["host_memory_signals"]["minimum_host_available_gib"] == 8
    assert plan["session"] == "default/s1"
    assert not output.exists()


def test_live_mode_refuses_without_explicit_gpu_ownership(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n")
    output = tmp_path / "run"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--model", str(model), "--output-dir", str(output),
         "--host-floor-gib", "8", "--allocator-cache-limit-gib", "8"],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "--i-own-the-gpu" in result.stderr
    assert not output.exists()
