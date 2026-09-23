"""CPU checks for the warm HTTP throughput benchmark's route receipts."""

import importlib.util
from io import BytesIO
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_serving", ROOT / "scripts" / "benchmark_serving.py"
)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


def _receipt(route, width):
    """Per-request mlx2 receipts in the shape each serving route emits."""
    if route == "ordinary":
        return {"cached_tokens": 12, "mtp": None, "ordinary_compute_width": width}
    if route == "mtp":
        return {"cached_tokens": 12, "mtp": {"observed_compute_widths": [width]},
                "ordinary_compute_width": None}
    # Speculative lanes report their width in the speculation receipt;
    # serving.ordinary_compute_width() is None for them.
    return {"cached_tokens": 12, "mtp": None, "ordinary_compute_width": None,
            "speculation": {"execution": route, "target_width": width}}


def _serve(monkeypatch, route):
    """Answer the benchmark in process; returns the width receipts report."""
    status = {"healthy": True, "inflight": 0, "max_lanes": 4, "runtime": {"r": 1},
              "settings": {"speculation": route}, "artifact": "a", "profile": "p"}
    reported_width = [1]

    def urlopen(target, timeout=None):
        if isinstance(target, str):
            return BytesIO(json.dumps(status).encode())
        body = json.loads(target.data)
        payload = {
            "choices": [{"message": {"content": "tok " * body["max_tokens"]}}],
            "usage": {"completion_tokens": body["max_tokens"]},
            "mlx2": _receipt(route, reported_width[0]),
        }
        return BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(benchmark, "urlopen", urlopen)
    import mlx2.serving

    monkeypatch.setattr(mlx2.serving, "runtime_identity", lambda: status["runtime"])
    return reported_width


@pytest.mark.parametrize("route", ["ordinary", "mtp", "prompt_lookup", "external_draft_verify"])
def test_benchmark_reads_the_width_every_route_reports(tmp_path, monkeypatch, route):
    # Regression: the benchmark read ordinary_compute_width, which is None on
    # prompt-lookup and external-draft lanes, and died with "requested B1,
    # observed widths [None]" on every speculative route.
    _serve(monkeypatch, route)
    output = tmp_path / "bench.json"
    monkeypatch.setattr(sys, "argv", [
        "benchmark_serving.py", "--output", str(output), "--rounds", "1",
        "--widths", "1", "--max-tokens", "4",
    ])
    benchmark.main()
    report = json.loads(output.read_text())
    assert [row["observed_compute_widths"] for row in report["rows"]] == [[1]]


def test_benchmark_still_refuses_a_width_the_route_did_not_run(tmp_path, monkeypatch):
    _serve(monkeypatch, "prompt_lookup")  # every request reports B1
    monkeypatch.setattr(sys, "argv", [
        "benchmark_serving.py", "--output", str(tmp_path / "bench.json"),
        "--rounds", "1", "--widths", "2", "--max-tokens", "4",
    ])
    with pytest.raises(AssertionError, match="requested B2"):
        benchmark.main()
