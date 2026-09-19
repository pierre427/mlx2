import importlib.util
from io import BytesIO
import json
from pathlib import Path
import threading
from urllib.error import HTTPError

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "qwen36_http_qualification",
    ROOT / "scripts" / "qwen36_http_qualification.py",
)
probe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(probe)


def receipt(width=2, structured=None):
    value = {
        "ordinary_compute_width": width,
        "request_controls": {},
    }
    if structured:
        value["request_controls"]["structured_output"] = {
            "kind": structured,
            "enforced": True,
        }
    return value


class FakeClient:
    def __init__(self, *, fail_json=False, max_lanes=4):
        self.fail_json = fail_json
        self.status = {
            "healthy": True,
            "runtime": {"source_sha256": "runtime"},
            "artifact": {"fingerprint": "artifact"},
            "model": "qwen36",
            "settings": {"max_lanes": max_lanes, "max_inflight": 8},
            "max_lanes": max_lanes,
            "max_context": 262144,
            "qualification": "candidate",
            "capabilities": ["grammar"],
        }
        self.sequence = 0
        self.tenants = set()

    def get(self, path):
        if path == "/v1/status":
            return self.status
        assert path == "/v1/status/batching"
        rates = {tenant: 10.0 for tenant in self.tenants}
        completed = [
            {"request_id": f"r{index}", "tenant_id": tenant}
            for index, tenant in enumerate(sorted(self.tenants))
        ]
        events = [
            {"kind": kind, "request_id": row["request_id"]}
            for row in completed
            for kind in ("admitted", "dequeued", "lane_attached", "first_token", "terminal")
        ]
        count = len(completed)
        distribution = {"count": count, "p50": 1.0, "p95": 2.0, "p99": 3.0}
        return {
            "schema": "mlx2.batch-runtime.v1",
            "latency_ms": {"ttft": distribution, "itl": distribution},
            "fairness": {
                "jain_tenant_token_rate": 1.0 if rates else None,
                "tenant_token_rates": rates,
            },
            "completed_requests": completed,
            "events": events,
        }

    def post(self, body, *, tenant="qwen36-qualification"):
        self.sequence += 1
        self.tenants.add(tenant)
        if body.get("n", 1) > self.status["max_lanes"]:
            payload = BytesIO(json.dumps({"error": {"message": "capacity"}}).encode())
            raise HTTPError("http://fixture", 429, "full", {"Retry-After": "1"}, payload)
        if body.get("n") == 2:
            return {
                "choices": [
                    {"index": 0, "message": {"content": "first complete sentence"}},
                    {"index": 1, "message": {"content": "second complete sentence"}},
                ],
                "mlx2": {
                    "parallel_sampling": {
                        "schema": "mlx2.parallel-sampling-admission.v1", "samples": 2,
                    },
                    "samples": [receipt(2), receipt(2)],
                },
            }
        response_format = body.get("response_format", {})
        if response_format.get("type") == "json_object":
            return {
                "choices": [{"message": {"content": "not-json" if self.fail_json else '{"ready":true}'}}],
                "mlx2": receipt(structured="json_object"),
            }
        if response_format.get("type") == "json_schema":
            return {
                "choices": [{"message": {"content": '{"ready":true}'}}],
                "mlx2": receipt(structured="json_schema"),
            }
        if "grammar" in body:
            return {
                "choices": [{"message": {"content": "yes"}}],
                "mlx2": receipt(structured="grammar"),
            }
        return {
            "choices": [{"finish_reason": "length", "message": {"content": "x " * 64}}],
            "usage": {"completion_tokens": 64},
            "mlx2": {**receipt(4), "request_controls": {"min_tokens": body.get("min_tokens", 0)}},
        }


def test_coverage_definition_lists_every_overnight_http_gate():
    assert set(probe.COVERAGE_DEFINITION["gates"]) == {
        "response_format_json_object", "strict_json_schema", "explicit_grammar",
        "physical_n2", "overload_429_retry_after",
        "latency_ttft_itl_percentiles", "tenant_jain_fairness",
        "progress_events", "full_length_completions",
    }


def test_full_probe_checkpoints_complete_receipt(tmp_path):
    output = tmp_path / "receipt.json"
    report = probe.run_qualification(
        FakeClient(), output,
        source={"schema": probe.SOURCE_SCHEMA, "sha256": "source"},
        clock=iter(range(100)).__next__,
    )
    assert report["schema"] == probe.SCHEMA
    assert report["passed"]
    assert all(row["status"] == "passed" for row in report["gates"].values())
    assert json.loads(output.read_text()) == report


def test_fail_soft_continues_after_independent_gate_failure(tmp_path):
    report = probe.run_qualification(
        FakeClient(fail_json=True), tmp_path / "receipt.json",
        source={"schema": probe.SOURCE_SCHEMA, "sha256": "source"},
    )
    assert not report["passed"]
    assert report["gates"]["response_format_json_object"]["status"] == "failed"
    assert report["gates"]["strict_json_schema"]["status"] == "passed"
    assert report["gates"]["full_length_completions"]["status"] == "passed"


def test_full_length_gate_requires_length_finish_and_minimum_receipt():
    class UnattestedClient(FakeClient):
        def post(self, body, *, tenant="qwen36-qualification"):
            response = super().post(body, tenant=tenant)
            if body.get("min_tokens"):
                response["mlx2"]["request_controls"].pop("min_tokens")
            return response

    gates = probe.gate_batch_observability(UnattestedClient(), {})
    assert gates["full_length_completions"]["status"] == "failed"
    assert gates["latency_ttft_itl_percentiles"]["status"] == "passed"


def test_batch_evidence_is_collected_before_risky_structured_gate(tmp_path):
    calls = []

    class PoisoningClient(FakeClient):
        def get(self, path):
            calls.append(("get", path))
            return super().get(path)

        def post(self, body, *, tenant="qwen36-qualification"):
            kind = (body.get("response_format") or {}).get("type")
            calls.append(("post", kind or body.get("grammar") or body.get("n", 1)))
            if kind == "json_object":
                raise RuntimeError("worker poisoned")
            return super().post(body, tenant=tenant)

    report = probe.run_qualification(
        PoisoningClient(), tmp_path / "receipt.json",
        source={"schema": probe.SOURCE_SCHEMA, "sha256": "source"},
    )
    assert report["gates"]["response_format_json_object"]["status"] == "failed"
    assert report["gates"]["full_length_completions"]["status"] == "passed"
    json_object_index = calls.index(("post", "json_object"))
    assert all(calls.index(("post", value)) < json_object_index for value in (2, r"(?:yes|no)", "json_schema"))


def test_resume_keeps_passed_gates_and_retries_failed(tmp_path):
    output = tmp_path / "receipt.json"
    source = {"schema": probe.SOURCE_SCHEMA, "sha256": "source"}
    first = probe.run_qualification(FakeClient(fail_json=True), output, source=source)
    first_timestamp = first["gates"]["strict_json_schema"]["recorded_at"]
    second = probe.run_qualification(FakeClient(), output, source=source, resume=True)
    assert second["passed"]
    assert second["gates"]["strict_json_schema"]["recorded_at"] == first_timestamp
    assert second["gates"]["response_format_json_object"]["status"] == "passed"


def test_resume_rejects_changed_server_identity(tmp_path):
    output = tmp_path / "receipt.json"
    source = {"schema": probe.SOURCE_SCHEMA, "sha256": "source"}
    probe.run_qualification(FakeClient(), output, source=source)
    changed = FakeClient()
    changed.status["artifact"] = {"fingerprint": "other"}
    with pytest.raises(RuntimeError, match="identity changed"):
        probe.run_qualification(changed, output, source=source, resume=True)


def test_grammar_is_only_skipped_when_route_does_not_declare_it():
    client = FakeClient()
    client.status["capabilities"] = []
    result = probe.gate_explicit_grammar(client, client.status)
    assert result == {
        "status": "skipped", "applicable": False,
        "reason": "active route does not declare grammar capability",
    }


def test_overload_is_blocked_when_exact_bounded_method_cannot_run():
    client = FakeClient(max_lanes=20)
    client.status["settings"]["max_inflight"] = 200
    result = probe.gate_overload(client, client.status)
    assert result["status"] == "blocked"
    assert result["evidence"]["cohort_count"] > 16


def test_wide_b20_overload_uses_barrier_cohorts_and_quiesces():
    class WideClient(FakeClient):
        def __init__(self):
            super().__init__(max_lanes=20)
            self.status["settings"]["max_inflight"] = 40
            self.lock = threading.Lock()
            self.cohorts = 0

        def post(self, body, *, tenant="qwen36-qualification"):
            if body.get("n") == 8:
                with self.lock:
                    self.cohorts += 1
                    cohort = self.cohorts
                if cohort == 6:
                    payload = BytesIO(json.dumps({"error": {
                        "message": "parallel samples exceed available inflight capacity"
                    }}).encode())
                    raise HTTPError("http://fixture", 429, "full", {"Retry-After": "1"}, payload)
                return {"choices": [{"message": {"content": "done"}}] * 8}
            return super().post(body, tenant=tenant)

        def get(self, path):
            if path == "/v1/status/batching":
                return {"gauges": {
                    "inflight_requests": 0, "queue_depth": 0, "active_lanes": 0,
                }}
            return super().get(path)

    result = probe.gate_overload(WideClient(), WideClient().status)
    assert result["status"] == "passed"
    evidence = result["evidence"]
    assert evidence["cohort_width"] == 8
    assert evidence["cohort_count"] == 6
    assert evidence["reserved_capacity"] == 48
    assert any(row.get("http_status") == 429 for row in evidence["outcomes"])
    assert evidence["quiescence"]["passed"]
