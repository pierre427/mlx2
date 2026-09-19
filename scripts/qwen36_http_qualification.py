#!/usr/bin/env python3
"""Fail-soft live HTTP qualification for the Qwen3.6 overnight campaign.

The probe is deliberately separate from the broad serving qualifier.  Every
gate is checkpointed atomically, so an interrupted run can resume only when
the script and server identities are unchanged.  Live failures are recorded
and independent gates continue.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, Callable
from urllib.error import HTTPError
from urllib.request import Request, urlopen


SCHEMA = "mlx2.qwen36-http-qualification.v1"
SOURCE_SCHEMA = "mlx2.qwen36-http-qualification-source.v1"
COVERAGE_DEFINITION = {
    "schema": "mlx2.qwen36-http-coverage.v1",
    "gates": {
        "response_format_json_object": {"required": True},
        "strict_json_schema": {"required": True},
        "explicit_grammar": {
            "required": True,
            "applicable_when": "server capabilities contains grammar",
        },
        "physical_n2": {"required": True},
        "overload_429_retry_after": {
            "required": True,
            "method": "over-capacity n or barrier-started parallel cohorts",
        },
        "latency_ttft_itl_percentiles": {"required": True},
        "tenant_jain_fairness": {"required": True},
        "progress_events": {"required": True},
        "full_length_completions": {"required": True},
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity(path: Path | None = None) -> dict[str, Any]:
    source = Path(__file__) if path is None else path
    root = source.resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    return {
        "schema": SOURCE_SCHEMA,
        "path": "scripts/qwen36_http_qualification.py",
        "sha256": _sha256(source),
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
    }


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class HTTPClient:
    def __init__(self, base_url: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get(self, path: str) -> dict[str, Any]:
        with urlopen(self.base_url + path, timeout=min(self.timeout, 30)) as response:
            return json.load(response)

    def post(
        self, body: dict[str, Any], *, tenant: str = "qwen36-qualification"
    ) -> dict[str, Any]:
        request = Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "X-Tenant-ID": tenant},
        )
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)


def server_identity(status: dict[str, Any]) -> dict[str, Any]:
    """Bind resumable evidence to the loaded artifact and exact runtime."""
    return {
        "runtime": status.get("runtime"),
        "artifact": status.get("artifact"),
        "model": status.get("model"),
        "settings": status.get("settings"),
        "qualification": status.get("qualification"),
        "max_lanes": status.get("max_lanes", (status.get("settings") or {}).get("max_lanes")),
        "max_inflight": status.get("max_inflight", (status.get("settings") or {}).get("max_inflight")),
        "max_context": status.get("max_context"),
        "capabilities": sorted(status.get("capabilities") or []),
    }


def _prompt(text: str, *, max_tokens: int = 64, **extra: Any) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": text}],
        "temperature": 0,
        "max_tokens": max_tokens,
        **extra,
    }


def _content(response: dict[str, Any]) -> str:
    return response["choices"][0]["message"]["content"]


def _structured_receipt(response: dict[str, Any], kind: str) -> bool:
    return (
        response.get("mlx2", {})
        .get("request_controls", {})
        .get("structured_output")
        == {"kind": kind, "enforced": True}
    )


def _widths(receipt: dict[str, Any]) -> list[int]:
    mtp = receipt.get("mtp") or {}
    values = mtp.get("observed_compute_widths") or []
    if values:
        return [int(value) for value in values]
    speculation = receipt.get("speculation") or {}
    if speculation.get("target_width") is not None:
        return [int(speculation["target_width"])]
    width = receipt.get("ordinary_compute_width")
    return [] if width is None else [int(width)]


def gate_json_object(client: HTTPClient, _status: dict[str, Any]) -> dict[str, Any]:
    response = client.post(
        _prompt(
            'Return exactly one JSON object with the key "ready" set to true.',
            max_tokens=16,
            response_format={"type": "json_object"},
        )
    )
    value = json.loads(_content(response))
    passed = isinstance(value, dict)
    passed = passed and _structured_receipt(response, "json_object")
    return {"status": "passed" if passed else "failed", "evidence": response}


def gate_strict_schema(client: HTTPClient, _status: dict[str, Any]) -> dict[str, Any]:
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "ready",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"ready": {"const": True}},
                "required": ["ready"],
                "additionalProperties": False,
            },
        },
    }
    response = client.post(
        _prompt(
            'Return exactly {"ready":true}. Do not add any other keys or text.',
            max_tokens=16,
            response_format=response_format,
        )
    )
    passed = json.loads(_content(response)) == {"ready": True}
    passed = passed and _structured_receipt(response, "json_schema")
    return {"status": "passed" if passed else "failed", "evidence": response}


def gate_explicit_grammar(client: HTTPClient, status: dict[str, Any]) -> dict[str, Any]:
    if "grammar" not in (status.get("capabilities") or []):
        return {
            "status": "skipped",
            "applicable": False,
            "reason": "active route does not declare grammar capability",
        }
    response = client.post(
        _prompt("Reply with yes.", max_tokens=4, grammar=r"(?:yes|no)")
    )
    passed = _content(response) in {"yes", "no"}
    passed = passed and _structured_receipt(response, "grammar")
    return {"status": "passed" if passed else "failed", "evidence": response}


def gate_physical_n2(client: HTTPClient, _status: dict[str, Any]) -> dict[str, Any]:
    response = client.post(
        _prompt("Write a complete eight word sentence.", max_tokens=32, n=2)
    )
    choices = response.get("choices") or []
    receipts = response.get("mlx2", {}).get("samples") or []
    widths = [width for receipt in receipts for width in _widths(receipt)]
    admission = response.get("mlx2", {}).get("parallel_sampling") or {}
    passed = (
        len(choices) == len(receipts) == 2
        and [choice.get("index") for choice in choices] == [0, 1]
        and all(choice.get("message", {}).get("content", "").strip() for choice in choices)
        and admission.get("schema") == "mlx2.parallel-sampling-admission.v1"
        and admission.get("samples") == 2
        and max(widths, default=0) >= 2
    )
    return {
        "status": "passed" if passed else "failed",
        "evidence": {"response": response, "observed_compute_widths": widths},
    }


def gate_overload(client: HTTPClient, status: dict[str, Any]) -> dict[str, Any]:
    if status.get("qualification") != "candidate":
        return {
            "status": "blocked",
            "reason": "deterministic overload probe is restricted to qualification mode",
        }
    max_lanes = status.get("max_lanes")
    max_inflight = status.get(
        "max_inflight", (status.get("settings") or {}).get("max_inflight")
    )
    if type(max_lanes) is not int or type(max_inflight) is not int:
        return {
            "status": "blocked",
            "reason": "server does not publish integer max_lanes and max_inflight",
            "evidence": {"max_lanes": max_lanes, "max_inflight": max_inflight},
        }
    if not 1 <= max_lanes <= max_inflight:
        return {
            "status": "blocked",
            "reason": "published lane/inflight limits are invalid",
            "evidence": {"max_lanes": max_lanes, "max_inflight": max_inflight},
        }

    def error_evidence(error: HTTPError) -> dict[str, Any]:
        try:
            body = json.load(error)
        except Exception:
            body = None
        return {
            "http_status": error.code,
            "retry_after": error.headers.get("Retry-After"),
            "body": body,
        }

    # The smallest server can prove the boundary with one atomic parallel
    # sample reservation, without relying on scheduler timing.
    if max_lanes < 8:
        requested = max_lanes + 1
        try:
            client.post(_prompt("Reply with one word.", max_tokens=4, n=requested))
        except HTTPError as error:
            evidence = {
                **error_evidence(error),
                "method": "over_capacity_parallel_sample",
                "requested_samples": requested,
                "max_lanes": max_lanes,
                "max_inflight": max_inflight,
            }
            passed = error.code == 429 and evidence["retry_after"] is not None
            return {"status": "passed" if passed else "failed", "evidence": evidence}
        return {
            "status": "failed",
            "reason": "over-capacity parallel request was admitted",
            "evidence": {"requested_samples": requested, "max_lanes": max_lanes},
        }

    # Wide B20 servers cannot express n=max_lanes+1 because the protocol caps
    # n at eight. Barrier-start enough n=8 cohorts that their atomic slot
    # reservations exceed max_inflight. The single generation worker cannot
    # retire all earlier cohorts before the HTTP handlers attempt admission.
    cohort_width = min(8, max_lanes)
    cohort_count = max_inflight // cohort_width + 1
    if cohort_count > 16:
        return {
            "status": "blocked",
            "reason": "bounded overload probe would require more than 16 HTTP cohorts",
            "evidence": {
                "max_lanes": max_lanes, "max_inflight": max_inflight,
                "cohort_width": cohort_width, "cohort_count": cohort_count,
            },
        }
    barrier = threading.Barrier(cohort_count)

    def submit(index: int) -> dict[str, Any]:
        barrier.wait(timeout=10)
        try:
            response = client.post(
                _prompt(
                    "Write until the output limit about compiler verification.",
                    max_tokens=16,
                    n=cohort_width,
                ),
                tenant=f"qwen36-overload-{index}",
            )
            return {"kind": "success", "index": index, "response": response}
        except HTTPError as error:
            return {"kind": "http_error", "index": index, **error_evidence(error)}
        except Exception as error:
            return {
                "kind": "error", "index": index,
                "error": {"type": type(error).__name__, "message": str(error)},
            }

    with ThreadPoolExecutor(max_workers=cohort_count) as pool:
        outcomes = list(pool.map(submit, range(cohort_count)))
    successes = [row for row in outcomes if row["kind"] == "success"]
    overloads = [
        row for row in outcomes
        if row.get("http_status") == 429
        and row.get("retry_after")
        and "inflight" in str((row.get("body") or {}).get("error", {}).get("message", "")).lower()
    ]
    unexpected = [row for row in outcomes
                  if row["kind"] not in {"success", "http_error"}
                  or row["kind"] == "http_error" and row.get("http_status") != 429]

    deadline = time.monotonic() + 120
    quiescence = []
    while True:
        batching = client.get("/v1/status/batching")
        gauges = batching.get("gauges") or {}
        sample = {
            "at": time.time(),
            "inflight_requests": gauges.get("inflight_requests"),
            "queue_depth": gauges.get("queue_depth"),
            "active_lanes": gauges.get("active_lanes"),
        }
        quiescence.append(sample)
        if sample["inflight_requests"] == sample["queue_depth"] == 0:
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    quiesced = (
        quiescence[-1]["inflight_requests"] == 0
        and quiescence[-1]["queue_depth"] == 0
    )
    evidence = {
        "method": "barrier_parallel_cohort_reservations",
        "max_lanes": max_lanes,
        "max_inflight": max_inflight,
        "cohort_width": cohort_width,
        "cohort_count": cohort_count,
        "reserved_capacity": cohort_width * cohort_count,
        "outcomes": outcomes,
        "quiescence": {"passed": quiesced, "samples": quiescence},
    }
    passed = bool(successes) and bool(overloads) and not unexpected and quiesced
    return {"status": "passed" if passed else "failed", "evidence": evidence}


def gate_batch_observability(
    client: HTTPClient, _status: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    before = client.get("/v1/status/batching")
    marker = f"qwen36-http-{time.time_ns()}"
    requests = []
    for index in range(4):
        tenant = marker + ("-a" if index % 2 == 0 else "-b")
        body = _prompt(
            "Explain deterministic compiler testing.",
            max_tokens=64,
            min_tokens=64,
        )
        requests.append((body, tenant))
    responses: list[dict[str, Any]] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(client.post, body, tenant=tenant): tenant
                   for body, tenant in requests}
        for future in as_completed(futures):
            try:
                responses.append(future.result())
            except Exception as exc:  # continue independent live requests
                errors.append(f"{type(exc).__name__}: {exc}")
    after = client.get("/v1/status/batching")
    expected_tenants = {marker + "-a", marker + "-b"}
    full = (
        len(responses) == 4
        and not errors
        and all(
            response.get("usage", {}).get("completion_tokens") == 64
            and response.get("choices", [{}])[0].get("finish_reason") == "length"
            and response.get("mlx2", {}).get("request_controls", {}).get("min_tokens") == 64
            and _content(response).strip()
            for response in responses
        )
    )
    latency = after.get("latency_ms") or {}
    distributions = [latency.get(name) or {} for name in ("ttft", "itl")]
    latency_passed = all(
        distribution.get("count", 0) > 0
        and all(isinstance(distribution.get(name), (int, float))
                and math.isfinite(distribution[name])
                for name in ("p50", "p95", "p99"))
        for distribution in distributions
    )
    fairness = after.get("fairness") or {}
    rates = fairness.get("tenant_token_rates") or {}
    jain = fairness.get("jain_tenant_token_rate")
    fairness_passed = (
        expected_tenants <= set(rates)
        and all(isinstance(rates[name], (int, float)) and rates[name] > 0
                for name in expected_tenants)
        and isinstance(jain, (int, float))
        and math.isfinite(jain)
        and 0 < jain <= 1
    )
    request_ids = {
        row.get("request_id")
        for row in after.get("completed_requests", [])
        if row.get("tenant_id") in expected_tenants
    }
    events = [event for event in after.get("events", [])
              if event.get("request_id") in request_ids]
    event_kinds = {event.get("kind") for event in events}
    progress_passed = bool(request_ids) and {
        "admitted", "dequeued", "lane_attached", "first_token", "terminal"
    } <= event_kinds
    common = {
        "before": before,
        "after": after,
        "tenant_ids": sorted(expected_tenants),
        "request_ids": sorted(value for value in request_ids if value),
        "errors": errors,
        "responses": responses,
    }
    return {
        "latency_ttft_itl_percentiles": {
            "status": "passed" if latency_passed else "failed",
            "evidence": {**common, "latency_ms": latency},
        },
        "tenant_jain_fairness": {
            "status": "passed" if fairness_passed else "failed",
            "evidence": {**common, "fairness": fairness},
        },
        "progress_events": {
            "status": "passed" if progress_passed else "failed",
            "evidence": {**common, "event_kinds": sorted(event_kinds), "events": events},
        },
        "full_length_completions": {
            "status": "passed" if full else "failed",
            "evidence": common,
        },
    }


SINGLE_GATES: tuple[tuple[str, Callable[..., dict[str, Any]]], ...] = (
    # Run bounded/non-structured probes first.  Free-form json_object uses the
    # broadest grammar and therefore runs last, so a fail-closed worker error
    # cannot erase evidence from independent gates.
    ("physical_n2", gate_physical_n2),
    ("overload_429_retry_after", gate_overload),
    ("explicit_grammar", gate_explicit_grammar),
    ("strict_json_schema", gate_strict_schema),
    ("response_format_json_object", gate_json_object),
)


def run_qualification(
    client: HTTPClient,
    output: Path,
    *,
    resume: bool = False,
    source: dict[str, Any] | None = None,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    status = client.get("/v1/status")
    identity = server_identity(status)
    source = source or source_identity()
    report: dict[str, Any]
    if resume and output.exists():
        report = json.loads(output.read_text())
        if report.get("source") != source or report.get("server") != identity:
            raise RuntimeError("resume receipt source/server identity changed")
    else:
        report = {
            "schema": SCHEMA,
            "started_at": clock(),
            "updated_at": clock(),
            "source": source,
            "server": identity,
            "coverage": COVERAGE_DEFINITION,
            "gates": {},
            "passed": False,
        }
        atomic_json(output, report)

    def save(name: str, result: dict[str, Any]) -> None:
        report["gates"][name] = {"recorded_at": clock(), **result}
        report["updated_at"] = clock()
        required = COVERAGE_DEFINITION["gates"]
        report["passed"] = all(
            report["gates"].get(gate, {}).get("status") in {"passed", "skipped"}
            for gate in required
        )
        atomic_json(output, report)

    batch_names = {
        "latency_ttft_itl_percentiles", "tenant_jain_fairness",
        "progress_events", "full_length_completions",
    }
    # Capture batching evidence before structured-output probes.  A structured
    # processor is deliberately fail closed, and a runtime defect in one such
    # probe must not cascade into false failures for unrelated observability.
    if not all(report["gates"].get(name, {}).get("status") == "passed"
               for name in batch_names):
        try:
            results = gate_batch_observability(client, status)
        except Exception as exc:
            results = {
                name: {"status": "failed", "error": {
                    "type": type(exc).__name__, "message": str(exc)
                }} for name in batch_names
            }
        for name in sorted(results):
            save(name, results[name])

    for name, probe in SINGLE_GATES:
        if report["gates"].get(name, {}).get("status") in {"passed", "skipped"}:
            continue
        try:
            result = probe(client, status)
        except Exception as exc:
            result = {
                "status": "failed",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        save(name, result)

    report["completed_at"] = clock()
    report["passed"] = all(
        report["gates"].get(name, {}).get("status") in {"passed", "skipped"}
        for name in COVERAGE_DEFINITION["gates"]
    )
    atomic_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8296")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--print-coverage", action="store_true")
    args = parser.parse_args()
    if args.print_coverage:
        print(json.dumps(COVERAGE_DEFINITION, indent=2, sort_keys=True))
        return 0
    if args.output is None:
        parser.error("--output is required unless --print-coverage is used")
    report = run_qualification(
        HTTPClient(args.url, args.timeout), args.output, resume=args.resume
    )
    for name, result in report["gates"].items():
        print(f"{name}: {result['status']}", flush=True)
    print(f"receipt: {args.output}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
