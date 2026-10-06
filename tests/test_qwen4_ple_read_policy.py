import os

import numpy as np
import pytest

from mlx2.runtime.models.qwen4_ple_nvme import (
    PREFILL_ID_THRESHOLD,
    FileBackedShardedEmbedding,
    ple_table_diagnostics,
    prefer_pooled_read,
)
from scripts.bench_qwen4_ple_adaptive_candidate import run_probe
from scripts.bench_qwen4_ple_read_policy import (
    PageResidency,
    benchmark,
    policy_outcomes,
    prefer_parallel,
    residency_label,
)


def test_pool_margin_matches_mlxserve_687_boundary():
    assert prefer_parallel(1000, 799)
    assert not prefer_parallel(1000, 800)
    assert not prefer_parallel(1000, 900)
    assert not prefer_parallel(0, 1)
    assert not prefer_parallel(1, 0)
    assert prefer_pooled_read(1000, 799)
    assert not prefer_pooled_read(1000, 800)
    assert not prefer_pooled_read(1000, 900)
    assert not prefer_pooled_read(0, 1)


def _table_file(tmp_path, *, rows=1024, dims=32):
    row_bytes = dims // 2 + 2 * (dims // 32) * 2
    path = tmp_path / "ple_rows.bin"
    values = np.arange(rows * row_bytes, dtype=np.uint8)
    path.write_bytes(values.tobytes())
    return path, row_bytes


def _adaptive_env(monkeypatch, *, warming=True):
    monkeypatch.setenv("MLX_QWEN4_PLE_NVME_READ_POLICY", "adaptive")
    monkeypatch.setenv(
        "MLX_QWEN4_PLE_NVME_ADAPTIVE_WARM", "1" if warming else "0"
    )
    monkeypatch.setenv("MLX_QWEN4_PLE_NVME_DECODE_WORKERS", "2")
    monkeypatch.setenv("MLX_QWEN4_PLE_NVME_PREFILL_WORKERS", "4")


def test_default_policy_preserves_receipt_shape(tmp_path, monkeypatch):
    monkeypatch.delenv("MLX_QWEN4_PLE_NVME_READ_POLICY", raising=False)
    path, _ = _table_file(tmp_path)
    table = FileBackedShardedEmbedding(str(path), 1024, 32, 2)
    try:
        table._pread_rows(np.arange(8, dtype=np.int64), 2)
        assert table.read_policy_status["configured"] == "pooled"
        assert table.read_policy_status["last_receipt"] is None
        assert not any(table.read_policy_status["counts"].values())
        assert "read_policy" not in ple_table_diagnostics(table)
    finally:
        table.close()


def test_adaptive_policy_refreshes_only_on_next_wide_foreground_lookup(
    tmp_path, monkeypatch
):
    _adaptive_env(monkeypatch, warming=True)
    timings = iter(((10_000, 1_000), (1_000, 5_000)))
    monkeypatch.setattr(
        FileBackedShardedEmbedding,
        "_measure_read_arms",
        lambda _self, _phase: next(timings),
    )
    path, _ = _table_file(tmp_path)
    table = FileBackedShardedEmbedding(str(path), 1024, 32, 2)
    try:
        assert table.read_policy_status["effective"] == "pooled"
        assert table.notify_adaptive_warm_complete()
        assert table._workers_for(PREFILL_ID_THRESHOLD - 1) == 2
        assert table.read_policy_status["warming"]["refresh_pending"]
        assert table._workers_for(PREFILL_ID_THRESHOLD) == 1
        status = table.read_policy_status
        assert status["effective"] == "serial"
        assert status["phase"] == "warmed"
        assert status["counts"]["load_calibrations"] == 1
        assert status["counts"]["warmed_calibrations"] == 1
        assert status["counts"]["warm_refreshes"] == 1
        assert not status["warming"]["refresh_pending"]

        rows = table._pread_rows(np.arange(8, dtype=np.int64), 1)
        assert rows.shape == (8, table.row_bytes)
        receipt = table.read_policy_status["last_receipt"]
        assert receipt == {
            "event": "foreground_read",
            "configured": "adaptive",
            "selected_arm": "serial",
            "actual_arm": "serial",
            "workers": 1,
            "rows": 8,
            "phase": "warmed",
            "warm_refresh_pending": False,
        }
        assert "read_policy" in ple_table_diagnostics(table)
    finally:
        table.close()


def test_adaptive_warm_disabled_never_rearms_after_demand_reads(tmp_path, monkeypatch):
    _adaptive_env(monkeypatch, warming=False)
    phases = []

    def measure(_self, phase):
        phases.append(phase)
        return (10_000, 1_000)

    monkeypatch.setattr(FileBackedShardedEmbedding, "_measure_read_arms", measure)
    path, _ = _table_file(tmp_path)
    table = FileBackedShardedEmbedding(str(path), 1024, 32, 2)
    try:
        assert not table.start_adaptive_warm()
        assert not table.notify_adaptive_warm_complete()
        table._pread_rows(np.arange(512, dtype=np.int64), table._workers_for(512))
        assert table._workers_for(512) == 4
        assert phases == ["load"]
        status = table.read_policy_status
        assert status["warming"] == {
            "enabled": False,
            "state": "disabled",
            "refresh_pending": False,
            "error": None,
        }
        assert status["counts"]["warmed_calibrations"] == 0
    finally:
        table.close()


def test_background_warmer_only_signals_and_foreground_consumes(tmp_path, monkeypatch):
    _adaptive_env(monkeypatch, warming=True)
    timings = iter(((10_000, 1_000), (1_000, 5_000)))
    monkeypatch.setattr(
        FileBackedShardedEmbedding,
        "_measure_read_arms",
        lambda _self, _phase: next(timings),
    )
    path, _ = _table_file(tmp_path)
    table = FileBackedShardedEmbedding(str(path), 1024, 32, 2)
    try:
        assert table.start_adaptive_warm()
        assert table.wait_for_adaptive_warm(5)
        before = table.read_policy_status
        assert before["effective"] == "pooled"
        assert before["warming"]["state"] == "completed"
        assert before["warming"]["refresh_pending"]
        assert before["counts"]["foreground_pooled_calls"] == 0
        assert table._workers_for(PREFILL_ID_THRESHOLD) == 1
        assert table.read_policy_status["effective"] == "serial"
    finally:
        table.close()


def test_warm_refresh_failure_refuses_current_and_future_reads(tmp_path, monkeypatch):
    _adaptive_env(monkeypatch, warming=True)
    phases = []

    def measure(_self, phase):
        phases.append(phase)
        if phase == "warmed":
            raise OSError("calibration failed")
        return (10_000, 1_000)

    monkeypatch.setattr(FileBackedShardedEmbedding, "_measure_read_arms", measure)
    path, _ = _table_file(tmp_path)
    table = FileBackedShardedEmbedding(str(path), 1024, 32, 2)
    try:
        assert table.notify_adaptive_warm_complete()
        with pytest.raises(OSError, match="calibration failed"):
            table._workers_for(PREFILL_ID_THRESHOLD)
        with pytest.raises(RuntimeError, match="warm refresh failed"):
            table._workers_for(1)
        status = table.read_policy_status
        assert status["warming"]["state"] == "refresh_failed"
        assert status["counts"]["warm_refresh_failures"] == 1
        assert status["last_receipt"]["event"] == "warm_refresh_failed"
    finally:
        table.close()


def test_adaptive_table_refuses_in_fork_child(tmp_path, monkeypatch):
    if not hasattr(os, "fork"):
        pytest.skip("requires fork")
    _adaptive_env(monkeypatch, warming=False)
    monkeypatch.setattr(
        FileBackedShardedEmbedding,
        "_measure_read_arms",
        lambda _self, _phase: (10_000, 1_000),
    )
    path, _ = _table_file(tmp_path)
    table = FileBackedShardedEmbedding(str(path), 1024, 32, 2)
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_fd)
        try:
            table._pread_rows(np.array([0], dtype=np.int64), 1)
        except RuntimeError as exc:
            os.write(write_fd, str(exc).encode())
        finally:
            os.close(write_fd)
            os._exit(0)
    os.close(write_fd)
    try:
        message = os.read(read_fd, 4096).decode()
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        assert "crossed fork" in message
        assert table._workers_for(1) == 2
    finally:
        os.close(read_fd)
        table.close()


@pytest.mark.parametrize(
    ("name", "value", "match"),
    (
        ("MLX_QWEN4_PLE_NVME_READ_POLICY", "auto", "must be one of"),
        ("MLX_QWEN4_PLE_NVME_ADAPTIVE_WARM", "yes", "must be 0 or 1"),
    ),
)
def test_adaptive_policy_configuration_fails_closed(
    tmp_path, monkeypatch, name, value, match
):
    _adaptive_env(monkeypatch, warming=True)
    monkeypatch.setenv(name, value)
    path, _ = _table_file(tmp_path)
    with pytest.raises(ValueError, match=match):
        FileBackedShardedEmbedding(str(path), 1024, 32, 2)


def test_residency_labels_do_not_call_mixed_or_unknown_cold():
    assert residency_label(None) == "unknown"
    assert residency_label(0) == "verified_nonresident"
    assert residency_label(0.5) == "mixed_residency"
    assert residency_label(1) == "verified_resident"


def _summary(serial, parallel):
    return {
        "serial": {"elapsed_ns": serial},
        "parallel": {"elapsed_ns": parallel},
        "prefer_parallel": prefer_parallel(serial, parallel),
    }


def test_dynamic_refresh_repairs_the_latch_and_warm_disabled_keeps_it():
    load = _summary(10_000, 1_000)
    warm = _summary(1_000, 5_000)
    enabled = policy_outcomes(load, warm, warming_enabled=True)
    assert enabled["latched_bug_manifested"]
    assert enabled["latched_bug_effective_after_warm"] == "parallel"
    assert enabled["dynamic_effective_after_warm"] == "serial"
    assert enabled["warm_disabled_effective_after_demand_warm"] == "parallel"
    assert enabled["latched_warm_regret_pct"] == 400
    assert enabled["dynamic_warm_regret_pct"] == 0

    disabled = policy_outcomes(load, warm, warming_enabled=False)
    assert not disabled["refresh_consumed"]
    assert disabled["dynamic_effective_after_warm"] == "parallel"
    assert disabled["warm_disabled_regret_pct"] == 400


def test_mincore_reports_pages_written_to_a_tiny_file_as_resident(tmp_path):
    path = tmp_path / "rows.bin"
    path.write_bytes(bytes(range(100)) * 100)
    with PageResidency(path) as residency:
        if not residency.supported:
            return
        pages = residency.pages_for_rows(
            np.array([0, 1, 50]), data_offset=0, row_bytes=100
        )
        assert pages
        assert residency.fraction(pages) == 1


def test_policy_benchmark_proves_only_controlled_resident_state(tmp_path):
    dims = 32
    row_bytes = dims // 2 + 2 * (dims // 32) * 2
    total_rows = 1024
    sidecar = tmp_path / "ple_rows.bin"
    sidecar.write_bytes(np.arange(total_rows * row_bytes, dtype=np.uint8).tobytes())
    manifest = {
        "total_rows": total_rows,
        "dims": dims,
        "num_shards": 2,
        "data_offset": 0,
    }
    report = benchmark(
        sidecar,
        manifest,
        rows=16,
        workers=2,
        reps=2,
        warming_enabled=True,
    )
    assert report["claims"]["cold"] is False
    assert report["claims"]["controlled_resident"]
    assert report["claims"]["performance_qualification"] is False
    assert len(report["load_runs"]) == len(report["controlled_resident_runs"]) == 2
    assert report["policies"]["refresh_consumed"]


def test_production_candidate_probe_is_exact_and_claim_bounded(tmp_path):
    sidecar, _ = _table_file(tmp_path)
    report = run_probe(
        sidecar,
        {"total_rows": 1024, "dims": 32, "num_shards": 2, "data_offset": 0},
        rows=512,
        seed=0x6871,
    )
    assert report["before_refresh"]["serial_exact"]
    assert report["after_refresh"]["serial_exact"]
    assert report["final_status"]["counts"]["warm_refreshes"] == 1
    assert report["claims"] == {
        "whole_table_warmed": False,
        "cold": False,
        "request_performance_qualification": False,
        "route_and_byte_parity_only": True,
    }
