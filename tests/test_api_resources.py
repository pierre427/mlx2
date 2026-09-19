import json
import threading
import time

import pytest

from mlx2.api_resources import BatchManager, FileStore, ResourceNotFound, ResponseStore
from mlx2.openai_compat import responses_input_items
from mlx2.serving import AdmissionClosed


def test_response_store_round_trips_across_restart_and_is_tenant_scoped(tmp_path):
    root = tmp_path / "responses"
    first = ResponseStore(root=root)
    payload = {"id": "resp_test", "object": "response", "status": "completed"}
    context = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "answer"},
    ]
    first.put("tenant-a", payload, context)
    stored_path = next(root.glob("*/*.json"))
    stored = json.loads(stored_path.read_text())
    assert set(stored) == {"tenant_id", "payload", "context_messages"}
    assert first.status()["bytes"] == len(stored_path.read_bytes())

    restored = ResponseStore(root=root)
    assert restored.get("tenant-a", "resp_test") == payload
    assert restored.context("tenant-a", "resp_test") == context
    assert responses_input_items(context, "resp_test") == [
        {
            "id": "in_test_0",
            "type": "message",
            "role": "user",
            "status": "completed",
            "content": [{"type": "input_text", "text": "hello"}],
        }
    ]
    with pytest.raises(ResourceNotFound):
        restored.get("tenant-b", "resp_test")
    assert restored.status()["durable"] is True


def test_response_input_items_fail_closed_without_derivable_context():
    with pytest.raises(RuntimeError, match="cannot be derived"):
        responses_input_items([], "resp_empty")


def test_file_store_durable_listing_filter_and_cursor(tmp_path):
    root = tmp_path / "files"
    store = FileStore(root=root)
    first = store.create(
        "tenant-a",
        filename="one.jsonl",
        purpose="batch",
        content_type="application/jsonl",
        content=b'{"one":1}\n',
    )
    second = store.create(
        "tenant-a",
        filename="two.txt",
        purpose="user_data",
        content_type="text/plain",
        content=b"two",
    )
    store.create(
        "tenant-b",
        filename="hidden.txt",
        purpose="user_data",
        content_type="text/plain",
        content=b"hidden",
    )

    restored = FileStore(root=root)
    page = restored.list("tenant-a", limit=1)
    assert page["object"] == "list"
    assert len(page["data"]) == 1
    assert page["has_more"] is True
    next_page = restored.list("tenant-a", limit=10, after=page["last_id"])
    assert {item["id"] for item in page["data"] + next_page["data"]} == {
        first["id"],
        second["id"],
    }
    assert restored.list("tenant-a", purpose="batch")["data"] == [first]
    assert restored.content("tenant-a", first["id"])[0] == b'{"one":1}\n'


def test_batch_restart_marks_interrupted_work_failed(tmp_path):
    files = FileStore(root=tmp_path / "files")
    source = files.create(
        "tenant-a",
        filename="requests.jsonl",
        purpose="batch",
        content_type="application/jsonl",
        content=(
            json.dumps(
                {
                    "custom_id": "request-1",
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {"input": "hello"},
                }
            ).encode()
            + b"\n"
        ),
    )
    release = threading.Event()

    def executor(endpoint, body, tenant_id):
        release.wait(5)
        return 200, {"object": "list", "data": []}

    root = tmp_path / "batches"
    manager = BatchManager(files, executor, root=root)
    batch = manager.create(
        "tenant-a",
        {
            "input_file_id": source["id"],
            "endpoint": "/v1/embeddings",
            "completion_window": "24h",
        },
    )

    restored = BatchManager(files, executor, root=root)
    interrupted = restored.get("tenant-a", batch["id"])
    assert interrupted["status"] == "failed"
    assert interrupted["errors"]["data"][0]["code"] == "server_restarted"
    assert restored.list("tenant-a")["data"][0]["id"] == batch["id"]
    release.set()


def test_batch_drain_timeout_stops_later_rows_and_reports_503(tmp_path):
    files = FileStore()
    lines = []
    for index in range(3):
        lines.append(
            json.dumps(
                {
                    "custom_id": f"request-{index}",
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {"input": str(index)},
                }
            )
        )
    source = files.create(
        "tenant",
        filename="requests.jsonl",
        purpose="batch",
        content_type="application/jsonl",
        content=("\n".join(lines) + "\n").encode(),
    )
    started = threading.Event()
    release = threading.Event()
    calls = []

    def executor(endpoint, body, tenant):
        calls.append(body["input"])
        started.set()
        release.wait(2)
        return 200, {"ok": True}

    manager = BatchManager(files, executor)
    batch = manager.create(
        "tenant",
        {
            "input_file_id": source["id"],
            "endpoint": "/v1/embeddings",
            "completion_window": "24h",
        },
    )
    assert started.wait(1)
    assert manager.abort_for_drain_timeout() == 1
    release.set()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        result = manager.get("tenant", batch["id"])
        if result["status"] == "failed":
            break
        time.sleep(0.01)
    assert result["status"] == "failed"
    assert result["errors"]["data"] == [
        {"code": "drain_timeout", "message": "drain timeout", "status_code": 503}
    ]
    assert result["request_counts"] == {"total": 3, "completed": 1, "failed": 2}
    assert calls == ["0"]


def test_batch_drain_rejection_is_a_503_response_not_invalid_request():
    files = FileStore()
    source = files.create(
        "tenant",
        filename="requests.jsonl",
        purpose="batch",
        content_type="application/jsonl",
        content=(
            json.dumps(
                {
                    "custom_id": "accepted-before-drain",
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {"input": "hello"},
                }
            ).encode()
            + b"\n"
        ),
    )

    def rejected_by_drain(*_args):
        raise AdmissionClosed("draining", "batch")

    manager = BatchManager(files, rejected_by_drain)
    batch = manager.create(
        "tenant",
        {
            "input_file_id": source["id"],
            "endpoint": "/v1/embeddings",
            "completion_window": "24h",
        },
    )
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        result = manager.get("tenant", batch["id"])
        if result["status"] == "completed":
            break
        time.sleep(0.01)
    assert result["status"] == "completed"
    assert result["request_counts"] == {"total": 1, "completed": 0, "failed": 1}
    assert result["error_file_id"] is None
    output, _content_type, _name = files.content(
        "tenant", result["output_file_id"]
    )
    row = json.loads(output)
    assert row["error"] is None
    assert row["response"]["status_code"] == 503
    assert row["response"]["body"]["error"]["code"] == "server_unavailable"
