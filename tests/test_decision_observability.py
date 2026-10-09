import hashlib
import json
import threading
from urllib.request import urlopen

import pytest

from mlx2.batch_metrics import HttpRuntimeMetrics
from mlx2.decisions.metrics import DecisionRuntimeMetrics, render_decision_metrics
from mlx2.decisions.qualification import (
    REQUIRED_CHECKS,
    install_qualification,
    qualification_basis,
)
from mlx2.decisions.server import (
    BoundedDecisionServer,
    DecisionApplication,
    make_handler,
)
from mlx2.prometheus import CONTENT_TYPE


class FakeDecisionEngine:
    family = "clef"
    variant = "clef-flash-9b"
    capabilities = ("text", "noul", "choice", "score")
    model_name = "clef-test"

    def __init__(self):
        self.artifact = {
            "identity": {
                "fingerprint": "a" * 64,
                "fingerprint_kind": "hub-blob-identity",
                "revision": "b" * 40,
            }
        }
        self._qualification = {"qualification": "unqualified", "qualified": False}
        self._counts = {
            "requests": 2,
            "failures": 0,
            "refusals": 1,
            "input_tokens": 42,
        }
        self.http_metrics = HttpRuntimeMetrics()
        self.decision_metrics = DecisionRuntimeMetrics()
        began = self.decision_metrics.loading_started()
        self.decision_metrics.loaded(began)
        began = self.decision_metrics.execution_started()
        self.decision_metrics.execution_finished(began, input_tokens=21)

    def route_receipt(self, *, observed_used):
        return {
            "route": "decision",
            "family": self.family,
            "variant": self.variant,
            "artifact_fingerprint": self.artifact["identity"]["fingerprint"],
            "artifact_revision": self.artifact["identity"]["revision"],
            "artifact_fingerprint_kind": "hub-blob-identity",
            "qualification": self._qualification["qualification"],
            "qualified": self._qualification["qualified"],
            "observed_used": observed_used,
        }

    def status(self):
        return {
            "route": self.route_receipt(observed_used=True),
            "counters": dict(self._counts),
        }

    def prometheus_metrics(self):
        return render_decision_metrics(self)


def test_decision_prometheus_is_parseable_and_bounded():
    parser = pytest.importorskip("prometheus_client.parser")
    engine = FakeDecisionEngine()
    started = engine.http_metrics.started()
    engine.http_metrics.completed("POST", "systemone", 200, started)
    rendered = engine.prometheus_metrics()
    families = {family.name for family in parser.text_string_to_metric_families(rendered)}
    assert "mlx2_decision_requests" in families
    assert "mlx2_decision_request_duration_seconds" in families
    assert "mlx2_http_requests" in families
    assert 'route="systemone"' in rendered
    assert "clef-test" not in rendered
    assert engine.artifact["identity"]["fingerprint"] not in rendered


def test_decision_metrics_endpoint_uses_prometheus_content_type():
    engine = FakeDecisionEngine()
    application = DecisionApplication(engine)
    server = BoundedDecisionServer(
        ("127.0.0.1", 0), make_handler(application), max_connections=1
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.server_port}/metrics") as response:
            payload = response.read().decode()
            assert response.status == 200
            assert response.headers["Content-Type"] == CONTENT_TYPE
        assert "mlx2_decision_qualified" in payload
        with urlopen(f"http://127.0.0.1:{server.server_port}/metrics") as response:
            second = response.read().decode()
        assert (
            'mlx2_http_requests_total{method="GET",route="metrics",status_class="2xx"} 1'
            in second
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_decision_qualification_is_exact_and_fail_closed(tmp_path):
    engine = FakeDecisionEngine()
    settings = {
        "served_model_name": engine.model_name,
        "capabilities": list(engine.capabilities),
        "max_connections": 8,
        "max_request_bytes": 4 << 20,
        "route": "decision",
    }
    receipt = qualification_basis(engine, settings)
    receipt.update(
        {
            "checks": {name: {"passed": True} for name in REQUIRED_CHECKS},
            "passed": True,
        }
    )
    evidence_path = tmp_path / "qualified-evidence.json"
    evidence_path.write_text("{}")
    receipt["evidence"] = {
        "path": evidence_path.name,
        "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
    }
    path = tmp_path / "qualified.json"
    path.write_text(json.dumps(receipt))
    state = install_qualification(path, engine=engine, settings=settings)
    assert state["qualified"] is True
    assert engine.route_receipt(observed_used=False)["qualification"] == "qualified"

    engine._qualification = {"qualification": "unqualified", "qualified": False}
    receipt["artifact"]["fingerprint"] = "c" * 64
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="artifact"):
        install_qualification(path, engine=engine, settings=settings)
    assert engine.route_receipt(observed_used=False)["qualified"] is False
