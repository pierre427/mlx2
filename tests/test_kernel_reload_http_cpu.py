"""Resident-kernel admin contract with a fake engine; no model/GPU is loaded."""

import json
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer

import pytest
from test_http_security import _request
from test_server_quiesce import AdminEngine

from mlx2.http_security import policy_for_bind
from mlx2.server import (
    build_parser,
    handler_for,
    serving_engine_kwargs,
    validate_kernel_dev_config,
    validate_kernel_reload_body,
)


class KernelEngine(AdminEngine):
    def kernel_reload_status(self):
        return {"enabled": True, "generation": 2}

    def reload_kernels(self, **options):
        self.calls.append(options)
        return {"status": "completed", "generation": 3, "apcv2_purged": True}


@contextmanager
def serving(engine, *, admin_token="admin"):
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        handler_for(engine, http_security=policy_for_bind("127.0.0.1"), admin_token=admin_token),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def request(port, method="POST", body=None, token="admin"):
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    return _request(
        port, method, "/v1/admin/kernels/reload" if method == "POST" else "/v1/admin/kernels",
        body=json.dumps(body).encode() if body is not None else None, headers=headers,
    )


def test_kernel_admin_requires_configured_key_and_authenticated_caller():
    engine = KernelEngine()
    with serving(engine, admin_token=None) as port:
        assert request(port, body={"modules": ["test.kernel"]})[0] == 403
        assert request(port, method="GET")[0] == 403
    with serving(engine) as port:
        assert request(port, body={"modules": ["test.kernel"]}, token=None)[0] == 401
        assert request(port, body={"modules": ["test.kernel"]}, token="wrong")[0] == 401
        assert request(port, method="GET", token="wrong")[0] == 401
    assert engine.calls == []


def test_kernel_reload_forwards_operations_and_returns_completed_receipt():
    engine = KernelEngine()
    options = {
        "modules": ["test.kernel"],
        "settings": [{"target": "test.kernel:MODE", "value": "candidate"}],
        "calls": [{"target": "test.kernel:set_rows", "args": [4], "kwargs": {}}],
        "drain_timeout_seconds": 2.5,
    }
    with serving(engine) as port:
        status, _, response = request(port, method="GET")
        assert status == 200 and response == {"enabled": True, "generation": 2}
        status, _, response = request(port, body=options)
    assert status == 200 and response["apcv2_purged"] is True
    assert engine.calls == [options]


def test_kernel_reload_rejects_invalid_requests_before_engine_operation():
    engine = KernelEngine()
    with serving(engine) as port:
        for bad in ({}, [], {"modules": "test.kernel"},
                    {"modules": ["test.kernel"], "keep_apc": True},
                    {"settings": [{"target": "x:y"}]},
                    {"calls": [{"target": "x:y", "args": "bad"}]},
                    {"modules": ["test.kernel"], "drain_timeout_seconds": True}):
            assert request(port, body=bad)[0] == 400
        status, _, _ = _request(
            port, "POST", "/v1/admin/kernels/reload", body=b"{broken",
            headers={"Authorization": "Bearer admin", "Content-Type": "application/json"},
        )
        assert status == 400
    assert engine.calls == []


def test_missing_engine_support_is_explicit():
    with serving(AdminEngine()) as port:
        assert request(port, method="GET")[2] == {"enabled": False}
        status, _, response = request(port, body={"modules": ["test.kernel"]})
        assert status == 409 and "disabled" in response["error"]["message"]


@pytest.mark.parametrize("failure,status", [
    (ValueError("unknown kernel"), 400),
    (SyntaxError("bad source"), 400),
    (ImportError("source outside root"), 400),
    (TimeoutError("drain expired"), 409),
    (RuntimeError("APCv2 purge failed; service closed"), 409),
    (FileNotFoundError("kernel source disappeared"), 409),
    (ZeroDivisionError("new kernel initialization failed"), 500),
])
def test_reload_failures_always_have_an_http_response(failure, status):
    class Failed(KernelEngine):
        def reload_kernels(self, **options):
            raise failure

    with serving(Failed()) as port:
        actual, _, response = request(port, body={"modules": ["test.kernel"]})
    assert actual == status and str(failure) in response["error"]["message"]
    if status == 500:
        assert response["error"]["code"] == "kernel_reload_failed"
        assert response["mlx2"]["kernel_reload"]["enabled"] is True


@pytest.mark.parametrize("failure,phase,status,code", [
    (ValueError("set_mode: unknown mode"), "failed", 500, "kernel_reload_failed"),
    (ImportError("edited import failed"), "failed", 500, "kernel_reload_failed"),
    (TimeoutError("could not drain"), "idle", 409, "kernel_reload_not_applied"),
])
def test_worker_recorded_failures_report_quarantine_and_receipt_not_exception_type(failure, phase, status, code):
    receipt = {"status": "failed" if phase == "failed" else "drain_timeout"}

    class Recorded(KernelEngine):
        def kernel_reload_status(self):
            return {"enabled": True, "phase": phase, "last_reload": receipt}

        def reload_kernels(self, **options):
            failure.kernel_reload_recorded = True
            raise failure

    with serving(Recorded()) as port:
        actual, _, response = request(port, body={"modules": ["test.kernel"]})
    assert actual == status and str(failure) in response["error"]["message"]
    assert response["error"]["code"] == code
    assert response["mlx2"]["kernel_reload"]["last_reload"] == receipt


def test_reload_envelope_rejects_nonfinite_drain_and_duplicate_modules():
    for timeout in (float("nan"), float("inf"), 0, 3601, True, "120"):
        with pytest.raises(ValueError):
            validate_kernel_reload_body({"modules": ["a"], "drain_timeout_seconds": timeout})
    with pytest.raises(ValueError, match="duplicates"):
        validate_kernel_reload_body({"modules": ["a", "a"]})


def test_server_flag_is_default_off_and_requires_key(tmp_path):
    default = build_parser().parse_args(["--model", "fixture"])
    assert default.kernel_dev_root is None
    enabled = build_parser().parse_args(["--model", "fixture", "--kernel-dev-root", str(tmp_path)])
    assert enabled.kernel_dev_root == tmp_path
    kw = {"native_mtp": False, "approximate_kv": None, "max_request_bytes": 65536}
    assert serving_engine_kwargs(default, None, **kw)["kernel_dev_root"] is None
    assert serving_engine_kwargs(enabled, None, **kw)["kernel_dev_root"] == tmp_path
    with pytest.raises(ValueError, match="admin-token-file"):
        validate_kernel_dev_config(tmp_path, None)
    with pytest.raises(ValueError, match="existing source directory"):
        validate_kernel_dev_config(tmp_path / "missing", "admin")
    validate_kernel_dev_config(tmp_path, "admin")
    validate_kernel_dev_config(None, None)


def test_stopped_engine_health_reports_stopped_with_its_error():
    class Stopped(KernelEngine):
        def status(self):
            return {**super().status(), "healthy": False, "error": "RuntimeError: device lost"}

    engine = Stopped()
    engine.state = "stopped"
    with serving(engine) as port:
        status, _, body = _request(port, "GET", "/health")
    assert status == 503
    assert body == {"status": "stopped", "error": "RuntimeError: device lost"}


def test_worker_stopped_reload_answers_503_with_its_rejection_receipt(monkeypatch):
    from test_kernel_reload_lifecycle_cpu import engine_fixture, start_reload

    import mlx2.kernel_reload as reload_module

    # The lifecycle fixture registers an owner; isolate the process registry
    # (as that module's autouse fixture does) so a rerun cannot conflict.
    monkeypatch.setattr(reload_module, "_OWNERS", {})
    monkeypatch.setattr(reload_module, "_CHANGED_OWNERS", set())
    stopped = engine_fixture()
    caller, result = start_reload(stopped, modules=["fixture.kernel"])
    stopped.stop_event.set()
    caller.join(timeout=2)
    error = result["error"]

    class Stopped(KernelEngine):
        def kernel_reload_status(self):
            return stopped.kernel_reload_status()

        def reload_kernels(self, **options):
            raise error

    with serving(Stopped()) as port:
        actual, _, response = request(port, body={"modules": ["test.kernel"]})
    assert actual == 503 and "stopped" in response["error"]["message"]
    assert response["error"]["code"] == "kernel_reload_not_applied"
    receipt = response["mlx2"]["kernel_reload"]["last_reload"]
    assert receipt["rejection"] == "worker_stopped" and receipt["code_executed"] is False
