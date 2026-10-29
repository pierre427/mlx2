"""Harness ownership only: synthetic sources, CPU, no model qualification."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models.deepseek_v41_engram import EngramRows, file_stamp
from test_deepseek_v41_expert_stream_cpu import fixture

mx.set_default_device(mx.cpu)


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/research"))
    return importlib.import_module("ablate_deepseek_v41")


@pytest.fixture
def registered(tmp_path):
    stream, _, _ = fixture(tmp_path)
    raw = np.zeros((8, 264), np.uint8)
    raw[:, :256] = 56 + np.arange(8)[:, None]
    raw[:, 256:] = 127
    path = tmp_path / "ple.bin"
    path.write_bytes(raw.tobytes())
    table = EngramRows(path, offset=0, rows=8, expected_stat=file_stamp(path.stat()),
                       cache_format="bf16_lut", cache_bytes=2048)
    tables = {0: table}
    stream.bind_ple_providers(tables)
    model = SimpleNamespace(engram_tables=tables)
    adapter = SimpleNamespace(model=model, weights=SimpleNamespace(expert_stream=stream), engram_tables=tables)
    try:
        yield adapter
    finally:
        table.close()
        stream.close()


def test_auxiliary_formats_preserve_exact_bytes_binding_and_budget(harness, registered):
    adapter = registered
    stream = adapter.weights.expert_stream
    original = adapter.engram_tables
    ids = np.array([0, 2, 1, 2], np.uint32)
    expected = original[0].read(ids)
    ple = harness.DiagnosticPLE(adapter, mx)
    try:
        for fmt in ("fp32", "bf16_lut", "fp32"):
            ple.select(fmt)
            assert adapter.model.engram_tables is ple.auxiliary[fmt]
            assert adapter.model.engram_tables[0].read(ids).tobytes() == expected.tobytes()
            assert adapter.engram_tables is original
            assert stream.ple_tables[0] is original[0]
        first = ple.auxiliary["fp32"][0]
        ple.select("fp32", reset=True)
        assert ple.auxiliary["fp32"][0] is first and not first._cache
        receipt = ple.receipt()
        assert receipt["runtime_binding_unchanged"]
        assert receipt["cache_budget_bytes"] + receipt["transient_bound_bytes"] <= 32 << 20
        with pytest.raises(ValueError, match="row identity/budget"):
            first.read(np.zeros(ple.max_rows + 1, np.uint32))
    finally:
        ple.close()
    assert adapter.model.engram_tables is original and not original[0]._closed
    assert all(t._closed for group in ple.auxiliary.values() for t in group.values())
    assert original[0].read(ids).tobytes() == expected.tobytes()


def test_refuses_unregistered_model_map(harness, registered):
    registered.model.engram_tables = dict(registered.engram_tables)
    with pytest.raises(ValueError, match="original registered"):
        harness.DiagnosticPLE(registered, mx)


@pytest.mark.parametrize("fence_fails,close_fails", [(True, False), (False, True), (True, True)])
def test_cleanup_attempts_all_independent_readers_and_preserves_first_error(
        harness, registered, monkeypatch, fence_fails, close_fails):
    fence_error, close_error = RuntimeError("fence failed"), RuntimeError("reader close failed")
    attempted = []
    def synchronize():
        attempted.append("fence")
        if fence_fails:
            raise fence_error
    ple = harness.DiagnosticPLE(registered, SimpleNamespace(synchronize=synchronize))
    for fmt in ("fp32", "bf16_lut"):
        ple.select(fmt)
        table = ple.auxiliary[fmt][0]
        table.read(np.array([0, 1], np.uint32))
        original_close = table.close
        def close(fmt=fmt, original_close=original_close):
            attempted.append(fmt)
            original_close()
            if fmt == "fp32" and close_fails:
                raise close_error
        monkeypatch.setattr(table, "close", close)
    with pytest.raises(RuntimeError) as caught:
        ple.close()
    assert caught.value is (fence_error if fence_fails else close_error)
    assert attempted == ["fence", "fp32", "bf16_lut"]
    assert all(table._closed for group in ple.auxiliary.values() for table in group.values())
    assert registered.model.engram_tables is registered.engram_tables
    assert not registered.engram_tables[0]._closed
    registered.engram_tables[0].read(np.array([0, 1], np.uint32))


@pytest.mark.parametrize("failure", [False, True])
def test_existing_coordinator_and_claims_survive(harness, registered, failure):
    stream = registered.weights.expert_stream
    stream.enable_elastic(registered.engram_tables)
    stream.bind_execution_synchronizer(mx.synchronize)
    original = stream.elastic
    original.set_claim("execution", 512)
    before = original.receipt()
    try:
        with harness.diagnostic_reclaim(stream, registered.engram_tables, mx) as receipt:
            assert stream.elastic is original
            assert receipt["claims"]["execution"] == 512
            assert registered.engram_tables[0].reserved_bytes() == 0
            if failure:
                raise RuntimeError("body failure")
    except RuntimeError as error:
        assert failure and str(error) == "body failure"
    assert stream.elastic is original
    assert original.receipt()["claims"] == before["claims"]
    assert original.receipt()["resources"] == before["resources"]
    assert stream.ple_tables[0] is registered.engram_tables[0]


@pytest.mark.parametrize("failure", [False, True])
def test_temporary_coordinator_restores_original_budgets(harness, registered, failure):
    stream = registered.weights.expert_stream
    table = registered.engram_tables[0]
    table.resize_cache_budget(1024)
    synchronize = stream._execution_synchronize
    try:
        with harness.diagnostic_reclaim(stream, registered.engram_tables, mx):
            assert table.reserved_bytes() == 0
            if failure:
                raise RuntimeError("body failure")
    except RuntimeError as error:
        assert failure and str(error) == "body failure"
    assert stream.elastic is None
    assert stream._execution_synchronize is synchronize
    assert table.reserved_bytes() == 1024
    assert stream.ple_tables[0] is table and not table._closed


def test_reclaim_refuses_auxiliary_provider_identity(harness, registered):
    ple = harness.DiagnosticPLE(registered, mx)
    try:
        ple.select("fp32")
        with pytest.raises(ValueError, match="original registered"):
            with harness.diagnostic_reclaim(registered.weights.expert_stream, registered.model.engram_tables, mx):
                pytest.fail("foreign PLE provider entered reclaim")
    finally:
        ple.close()


def test_failed_reclaim_keeps_poisoned_coordinator(harness, registered, monkeypatch):
    stream = registered.weights.expert_stream
    enable = stream.enable_elastic
    def fail(*args):
        raise RuntimeError("resize failed")
    def injected(tables):
        enable(tables)
        stream.elastic.resources["ple.0"].resize = fail
    monkeypatch.setattr(stream, "enable_elastic", injected)
    with pytest.raises(RuntimeError, match="poisoned coordinator"):
        with harness.diagnostic_reclaim(stream, registered.engram_tables, mx):
            pytest.fail("failed provider granted a claim")
    assert stream.elastic is not None and "resize failed" in stream.elastic.error
    assert not stream.elastic.claims


@pytest.mark.parametrize("entry", ["run", "compare_final"])
def test_entrypoint_failure_restores_registered_providers_and_policies(harness, registered, tmp_path, monkeypatch, entry):
    adapter = registered
    model, weights = adapter.model, adapter.weights
    names = ("expert_small_policy", "expert_reuse_policy", "expert_iq1_policy", "expert_down_policy",
             "expert_binding_policy", "expert_io_policy", "dense_epilogue_policy", "dense_pair_policy",
             "expert_groups_policy", "expert_simd_policy", "shared_policy")
    for name in names:
        setattr(weights, name, "candidate")
    for name in ("pointwise_policy", "attention_epilogue_policy", "hc_epilogue_policy", "verify_rows_policy"):
        setattr(model, name, "candidate")
    adapter.prompt_tokens = lambda request: list(range(10))
    model.make_cache = lambda: []
    def fail(*args):
        raise RuntimeError("forward failed")
    model.prefill_forward = fail
    created = []
    def reader(*args, **kwargs):
        table = EngramRows(*args, **kwargs)
        created.append(table)
        return table
    monkeypatch.setattr(harness, "EngramRows", reader)
    with pytest.raises(RuntimeError, match="forward failed"):
        getattr(harness, entry)(adapter, mx, tmp_path, lambda *args, **kwargs: None)
    assert model.engram_tables is adapter.engram_tables
    assert weights.expert_stream.ple_tables[0] is adapter.engram_tables[0]
    assert not adapter.engram_tables[0]._closed
    assert created and all(table._closed for table in created)
    assert all(getattr(weights, name) == "candidate" for name in names)
