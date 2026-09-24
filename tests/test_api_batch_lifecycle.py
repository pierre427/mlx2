"""Batch state transitions use only in-memory executors and file fixtures."""

import json
import threading
import time

import pytest

from mlx2.api_resources import BatchManager, FileStore, _tenant_name


def _create(manager):
    source = manager.file_store.create(
        "tenant", filename="input.jsonl", purpose="batch",
        content_type="application/jsonl",
        content=json.dumps({
            "custom_id": "row", "method": "POST", "url": "/v1/embeddings",
            "body": {"input": "hello"},
        }).encode(),
    )
    return manager.create("tenant", {
        "input_file_id": source["id"], "endpoint": "/v1/embeddings",
        "completion_window": "24h",
    })


def _terminal(manager, batch):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        record = manager.get("tenant", batch["id"])
        if record["status"] in {"completed", "failed", "cancelled"}:
            return record
        time.sleep(0.01)
    pytest.fail(f"batch did not terminate: {record['status']}")


@pytest.mark.parametrize("cancel", ["cancel", "drain"])
def test_cancellation_during_output_storage_is_observed(cancel):
    storing = threading.Event()
    release = threading.Event()

    class SlowStore(FileStore):
        def create(self, *args, **kwargs):
            if kwargs["filename"].endswith("-output.jsonl"):
                storing.set()
                assert release.wait(2)
            return super().create(*args, **kwargs)

    manager = BatchManager(SlowStore(), lambda *args: (200, {"ok": True}))
    batch = _create(manager)
    try:
        assert storing.wait(2)
        if cancel == "cancel":
            assert manager.cancel("tenant", batch["id"])["status"] == "cancelling"
        else:
            assert manager.abort_for_drain_timeout() == 1
    finally:
        release.set()
    result = _terminal(manager, batch)
    assert result["status"] == ("cancelled" if cancel == "cancel" else "failed")
    if cancel == "drain":
        assert result["errors"]["data"][0]["code"] == "drain_timeout"


def test_worker_persistence_failure_releases_active_batch_capacity():
    class BrokenPersistence(BatchManager):
        def _persist(self, record):
            if record["status"] != "validating":
                raise OSError("disk unavailable")

    manager = BrokenPersistence(FileStore(), lambda *args: (200, {}), max_batches=1)
    batch = _create(manager)
    result = _terminal(manager, batch)
    assert result["status"] == "failed"
    assert result["errors"]["data"][0]["code"] == "storage_error"
    assert manager.status()["states"].get("in_progress", 0) == 0


def test_failed_initial_persistence_does_not_leave_a_batch_without_a_worker():
    class BrokenPersistence(BatchManager):
        def _persist(self, record):
            raise OSError("disk unavailable")

    manager = BrokenPersistence(FileStore(), lambda *args: (200, {}), max_batches=1)
    with pytest.raises(OSError, match="disk unavailable"):
        _create(manager)
    assert manager.status()["batches"] == 0


@pytest.mark.parametrize("output_overflows", [False, True])
def test_transient_persistence_failure_does_not_count_a_row_twice(output_overflows):
    class FlakyPersistence(BatchManager):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = 0

        def _persist(self, record):
            self.calls += 1
            if self.calls == 3:
                raise OSError("transient disk failure after successful row")

    manager = FlakyPersistence(
        FileStore(max_file_bytes=512),
        lambda *args: (200, {"ok": "x" * 1024 if output_overflows else True}),
    )
    result = _terminal(manager, _create(manager))
    assert result["status"] == "failed"
    assert result["request_counts"] == {
        "total": 1, "completed": int(not output_overflows), "failed": int(output_overflows),
    }


def test_batch_restore_rejects_resource_identity_outside_tenant(tmp_path):
    tenant = tmp_path / _tenant_name("tenant")
    tenant.mkdir()
    source = tenant / "batch-corrupt.json"
    source.write_text(json.dumps({
        "id": "../escaped", "tenant_id": "tenant", "status": "completed",
    }))
    manager = BatchManager(FileStore(), lambda *args: None, root=tmp_path)
    assert not (tmp_path / "escaped.json").exists()
    assert not manager._batches
    assert source.exists()
    assert manager.status()["counts"]["restore_failures"] == 1


@pytest.mark.parametrize("bad_counts", [[], {"total": 1, "completed": 2, "failed": 0}])
def test_batch_restore_skips_invalid_counters(tmp_path, bad_counts):
    manager = BatchManager(FileStore(), lambda *args: (200, {}), root=tmp_path)
    _terminal(manager, _create(manager))
    source = next(tmp_path.glob("*/*.json"))
    record = json.loads(source.read_text())
    record["request_counts"] = bad_counts
    source.write_text(json.dumps(record))
    restored = BatchManager(FileStore(), lambda *args: None, root=tmp_path)
    assert restored.status()["batches"] == 0
    assert restored.status()["counts"]["restore_failures"] == 1
