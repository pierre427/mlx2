"""Merged kernel/weight-space HTTP contracts; CPU fakes, no model or GPU load."""

import json
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from urllib.request import urlopen

import pytest
from test_http_security import _request
from test_prometheus import FakeEngine, assert_valid_prometheus_text

from mlx2.http_security import policy_for_bind
from mlx2.kernel_reload_serving import KernelReloadMixin, KernelWeightSpaceConflict
from mlx2.server import handler_for

CHECKPOINT = "kernel-weight-space-http-fixture@sha256:" + "a" * 64
SOURCE_REVISION = "187ec85950f25ecdec8b6cee9bc9b9d367ae91ca"


class ConflictEngine(KernelReloadMixin, FakeEngine):
    """Use real compatibility guards and reload accounting, without a worker."""

    def __init__(self):
        super().__init__()
        self.operation_receipts = []
        self.reload_options = []
        self._kernel_reload_epoch = 0
        self._kernel_reload_phase = "idle"
        self._kernel_reload_request = None
        self._kernel_reload_identity = None
        self.weight_space_live = object()
        self.snapshot["model"] = CHECKPOINT
        self.snapshot["runtime"]["source_revision"] = SOURCE_REVISION

    def _reload_kernels(self, **options):
        self.reload_options.append(options)
        self._check_kernel_weight_space_compatibility()
        raise AssertionError("incompatible reload must refuse before source preparation")


class ConflictTools:
    def __init__(self, engine):
        self.engine = engine
        self.calls = []

    def capabilities(self):
        return {"enabled": True, "checkpoint": CHECKPOINT, "source_revision": SOURCE_REVISION}

    def call(self, body):
        self.calls.append(body)
        self.engine._check_weight_space_kernel_compatibility("weight_space_bind")
        raise AssertionError("incompatible binding must refuse before tool execution")


@contextmanager
def served(engine, tools):
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        handler_for(engine, http_security=policy_for_bind("127.0.0.1"),
                    admin_token="admin", weight_space_service=tools),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def post(port, path, body, *, token="admin"):
    return _request(port, "POST", path, body=json.dumps(body).encode(),
                    headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})


@pytest.mark.parametrize("path,body,operation,conflicts", [
    ("/v1/admin/kernels/reload", {"modules": ["private.fixture.kernel"]},
     "kernel_reload", ["live_weight_space"]),
    ("/v1/admin/weight-space/call", {"tool": "compile_analytical_repair", "arguments": {"capture_id": "registered"}},
     "weight_space_bind", ["kernel_reload_execution"]),
])
def test_both_admin_routes_preserve_compatibility_code_and_full_receipt(path, body, operation, conflicts):
    engine = ConflictEngine()
    if operation == "weight_space_bind":
        engine._kernel_reload_epoch = 1
        engine._kernel_reload_identity = "private-execution-identity"
    tools = ConflictTools(engine)
    with served(engine, tools) as port:
        status, _, response = post(port, path, body)
    assert status == KernelWeightSpaceConflict.status == 409
    assert response["error"]["code"] == "kernel_weight_space_incompatible"
    assert response["mlx2"] == engine.operation_receipts[-1] == {
        "schema": "mlx2.runtime-compatibility.v1", "operation": operation,
        "status": "refused", "reason": "kernel_weight_space_incompatible",
        "conflicts": conflicts, "qualified": False, "model_changed": False, "cache_changed": False,
    }
    assert len(engine.operation_receipts) == 1
    assert "private-execution-identity" not in json.dumps(response)
    assert engine.counts["kernel_reload_completed"] == 0
    assert engine.counts["kernel_reload_apc_entries_purged"] == 0


def test_reload_conflict_metric_and_weight_space_metrics_coexist_on_real_http_scrape():
    engine = ConflictEngine()
    tools = ConflictTools(engine)
    engine.counts["kernel_reload_rejected_private-unbounded-reason"] = 99
    engine.weight_space_metrics.record("program_create", "ok")
    with served(engine, tools) as port:
        status, _, response = post(port, "/v1/admin/kernels/reload", {"modules": ["private.fixture.kernel"]})
        assert status == 409 and response["error"]["code"] == "kernel_weight_space_incompatible"
        with urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10) as scrape:
            assert scrape.status == 200
            text = scrape.read().decode()
    assert_valid_prometheus_text(text)
    assert 'mlx2_kernel_reload_rejections_total{reason="weight_space_conflict"} 1' in text
    assert 'mlx2_kernel_reload_outcomes_total{outcome="rejected"} 1' in text
    assert "mlx2_kernel_reload_requests_total 1" in text
    assert 'mlx2_weight_space_operations_total{operation="program_create",outcome="ok"} 1' in text
    assert engine.counts["kernel_reload_rejected_weight_space_conflict"] == 1
    assert engine.counts["kernel_reload_rejected"] == 1
    assert engine.counts["kernel_reload_requests"] == 1
    assert "private-unbounded-reason" not in text
    assert "private.fixture.kernel" not in text


@pytest.mark.parametrize("path,body", [
    ("/v1/admin/kernels/reload", {"modules": ["private.fixture.kernel"]}),
    ("/v1/admin/weight-space/call", {"tool": "compile_analytical_repair", "arguments": {"capture_id": "registered"}}),
])
def test_unauthorized_calls_never_reach_compatibility_guards_or_receipts(path, body):
    engine = ConflictEngine()
    engine._kernel_reload_epoch = 1
    tools = ConflictTools(engine)
    with served(engine, tools) as port:
        status, _, response = post(port, path, body, token="wrong")
    assert status == 401
    assert "mlx2" not in response
    assert not engine.operation_receipts and not engine.reload_options and not tools.calls
    assert engine.counts["kernel_reload_requests"] == 0
