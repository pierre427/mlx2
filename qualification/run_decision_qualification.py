"""Full functional promotion qualification for one standalone decision model."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import decision_reference as decision_reference_module
from prometheus_client.parser import text_string_to_metric_families

import mlx2.decisions.qualification as qualification_module
import mlx2.decisions.runtime as decision_runtime_module
from mlx2.decisions.qualification import (
    APPROVED_PRODUCER_SHA256,
    REQUIRED_CHECKS,
    qualification_basis,
    serving_settings,
)
from mlx2.decisions.runtime import load_decision_engine
from mlx2.decisions.schema import normalize_request

compare_with_upstream = decision_reference_module.compare_with_upstream


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def producer_hash() -> str:
    digest = hashlib.sha256()
    for relative, path in (
        ("qualification/decision_reference.py", Path(decision_reference_module.__file__)),
        ("qualification/run_decision_qualification.py", Path(__file__)),
    ):
        payload = path.resolve().read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative.encode())
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def json_request(port: int, method: str, path: str, payload=None):
    body = None
    headers = {"Host": f"127.0.0.1:{port}"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        headers.update(
            {"Content-Type": "application/json", "Content-Length": str(len(body))}
        )
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=180)
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        return response.status, json.loads(raw) if raw else None, dict(response.headers)
    finally:
        connection.close()


def text_request(port: int, path: str):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        connection.request("GET", path, headers={"Host": f"127.0.0.1:{port}"})
        response = connection.getresponse()
        raw = response.read().decode()
        return response.status, raw, dict(response.headers)
    finally:
        connection.close()


def finite_answer(answer: dict) -> bool:
    confidence = answer.get("confidence")
    if not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
        return False
    if not 0 <= confidence <= 1:
        return False
    if answer["type"] == "noul":
        probability = answer.get("probability")
        return (
            isinstance(answer.get("value"), bool)
            and isinstance(probability, (int, float))
            and math.isfinite(probability)
            and 0 <= probability <= 1
        )
    probabilities = answer.get("probabilities")
    return bool(probabilities) and all(
        isinstance(value, (int, float))
        and math.isfinite(value)
        and 0 <= value <= 1
        for value in probabilities.values()
    ) and abs(sum(probabilities.values()) - 1.0) <= 0.002


def questions(family: str) -> dict:
    levels = (
        ["none", "low", "moderate", "high", "urgent", "critical"]
        if family == "jev"
        else ["can wait", "today", "right now"]
    )
    return {
        "intent": {
            "type": "choice",
            "instructions": "What does the customer explicitly request?",
            "criteria": {
                "track": "the location or delivery status of an order",
                "refund": "money back for a charge or purchase",
            },
        },
        "angry": {"type": "noul", "instructions": "Is the customer angry?"},
        "urgency": {
            "type": "score",
            "instructions": "How urgent is the request?",
            "criteria": levels,
        },
    }


def payload(model: str, family: str, state, *, truncate=True) -> dict:
    return {
        "model": model,
        "state": state,
        "questions": questions(family),
        "truncate": truncate,
    }


def start_server(
    root: Path,
    model_path: Path,
    model_name: str,
    *,
    max_connections: int,
    max_request_bytes: int,
    qualification: Path | None = None,
):
    port = free_port()
    command = [
        sys.executable,
        "-m",
        "mlx2.decisions.server",
        "--model",
        str(model_path),
        "--served-model-name",
        model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-connections",
        str(max_connections),
        "--max-request-bytes",
        str(max_request_bytes),
    ]
    if qualification is not None:
        command.extend(["--qualification", str(qualification)])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    stdout_log = tempfile.TemporaryFile(  # noqa: SIM115 - closed by stop_server
        mode="w+", encoding="utf-8"
    )
    stderr_log = tempfile.TemporaryFile(  # noqa: SIM115 - closed by stop_server
        mode="w+", encoding="utf-8"
    )
    process = subprocess.Popen(
        command,
        cwd=root,
        env=environment,
        stdout=stdout_log,
        stderr=stderr_log,
        text=True,
    )
    process._decision_logs = (stdout_log, stderr_log)
    started = time.monotonic()
    deadline = started + 240
    try:
        while True:
            if process.poll() is not None:
                raise RuntimeError("decision server exited during startup")
            try:
                status, body, _headers = json_request(port, "GET", "/health")
                if status == 200 and body == {"status": "ok"}:
                    return process, port, time.monotonic() - started, command
            except (OSError, ValueError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("decision server did not become healthy")
            time.sleep(0.2)
    except BaseException as error:
        stopped = stop_server(process)
        if isinstance(error, RuntimeError):
            raise type(error)(
                f"{error}: {stopped['stderr_tail'][-4000:]}"
                f"{stopped['stdout_tail'][-1000:]}"
            ) from error
        raise


def stop_server(process) -> dict:
    if getattr(process, "_decision_logs_closed", False):
        return {
            "returncode": process.returncode,
            "stdout_tail": "",
            "stderr_tail": "",
        }
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=20)
    stdout_log, stderr_log = process._decision_logs
    stdout_log.seek(0)
    stderr_log.seek(0)
    stdout = stdout_log.read()
    stderr = stderr_log.read()
    stdout_log.close()
    stderr_log.close()
    process._decision_logs_closed = True
    return {
        "returncode": process.returncode,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-6000:],
    }


def sample_value(metrics: str, name: str, labels: dict | None = None) -> float:
    labels = labels or {}
    for family in text_string_to_metric_families(metrics):
        for sample in family.samples:
            if sample.name == name and all(sample.labels.get(k) == v for k, v in labels.items()):
                return float(sample.value)
    raise KeyError((name, labels))


def prepare_qualification(args, *, max_connections: int, max_request_bytes: int):
    """Bind the executing harness and collect direct/reference evidence."""

    expected_paths = {
        (args.root / "qualification" / "decision_reference.py").resolve(): Path(
            decision_reference_module.__file__
        ).resolve(),
        (args.root / "qualification" / "run_decision_qualification.py").resolve(): Path(
            __file__
        ).resolve(),
        (args.root / "src" / "mlx2" / "decisions" / "qualification.py").resolve(): Path(
            qualification_module.__file__
        ).resolve(),
        (args.root / "src" / "mlx2" / "decisions" / "runtime.py").resolve(): Path(
            decision_runtime_module.__file__
        ).resolve(),
    }
    mismatched = [
        (str(expected), str(actual))
        for expected, actual in expected_paths.items()
        if expected != actual
    ]
    if mismatched:
        raise RuntimeError(f"qualification imports do not come from --root: {mismatched}")
    actual_producer = producer_hash()
    if actual_producer != APPROVED_PRODUCER_SHA256:
        raise RuntimeError(
            f"qualification producer is {actual_producer}, approved is "
            f"{APPROVED_PRODUCER_SHA256}"
        )
    source_repositories = {
        "mlx-vlm": args.mlx_vlm.resolve(),
        "llama.cpp": args.llama_cpp.resolve(),
        "sglang": args.sglang.resolve(),
    }
    base_payload = payload(
        args.model_name,
        args.family,
        {
            "message": "I was charged twice. Refund the duplicate payment now.",
            "order_id": "A-123",
        },
    )
    engine = None
    try:
        engine = load_decision_engine(
            args.model_path, served_model_name=args.model_name
        )
        settings = serving_settings(
            engine,
            max_connections=max_connections,
            max_request_bytes=max_request_bytes,
        )
        basis = qualification_basis(engine, settings)
        direct_request = normalize_request(
            base_payload, default_model=args.model_name
        )
        reference = compare_with_upstream(
            engine, direct_request, source_repositories
        )
        direct = engine.predict(direct_request)
        max_context = engine.artifact["max_context"]
        return basis, base_payload, reference, direct, max_context
    finally:
        if engine is not None:
            engine.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--family",
        choices=("clef", "decision2", "pplx-decider", "jev"),
        required=True,
    )
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-fingerprint", required=True)
    parser.add_argument("--mlx-vlm", type=Path, required=True)
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--sglang", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.root = args.root.resolve()
    args.model_path = args.model_path.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging_path = args.output.with_name("." + args.output.name + ".staging")
    failed_path = args.output.with_name(args.output.stem + "-failed.json")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite qualification receipt {args.output}")
    if staging_path.exists():
        staging_path.unlink()
    max_connections = 2
    max_request_bytes = 4 << 20

    try:
        basis, base_payload, reference, direct, max_context = prepare_qualification(
            args,
            max_connections=max_connections,
            max_request_bytes=max_request_bytes,
        )
    except Exception as error:  # noqa: BLE001 - durable pre-HTTP failure record
        failure = {
            "schema": "mlx2.decision-qualification-failure.v1",
            "phase": "direct-reference",
            "family": args.family,
            "model": args.model_name,
            "error": f"{type(error).__name__}: {error}",
            "passed": False,
        }
        atomic_json(failed_path, failure)
        print(json.dumps(failure, sort_keys=True))
        return 1

    evidence = {
        "schema": "mlx2.decision-qualification-evidence.v1",
        "family": args.family,
        "model": args.model_name,
        "reference": reference,
        "direct": direct,
        "runs": [],
    }
    checks = {}

    process = None
    harness_error = None
    try:
        process, port, startup_seconds, command = start_server(
            args.root,
            args.model_path,
            args.model_name,
            max_connections=max_connections,
            max_request_bytes=max_request_bytes,
        )
        run = {"command": command, "startup_seconds": startup_seconds}
        evidence["runs"].append(run)
        status_code, initial, _ = json_request(port, "GET", "/v1/status")
        models_code, models, _ = json_request(port, "GET", "/v1/models")
        route = initial["route"]
        checks["artifact_identity"] = {
            "passed": (
                route["artifact_revision"] == args.expected_revision
                and route["artifact_fingerprint"] == args.expected_fingerprint
                and route["artifact_fingerprint_kind"] == "hub-blob-identity"
                and basis["artifact"]["revision"] == args.expected_revision
                and basis["artifact"]["fingerprint"] == args.expected_fingerprint
            ),
            "evidence": route,
        }
        integrity = initial.get("tokenizer_integrity", {})
        checks["tokenizer_integrity"] = {
            "passed": (
                integrity.get("file", {}).get("status") == "consistent"
                and integrity.get("declared", {}).get("status") == "repaired"
            ),
            "evidence": integrity,
        }

        first_code, first, _ = json_request(
            port, "POST", "/v1/systemone", base_payload
        )
        repeat_code, repeated, _ = json_request(
            port, "POST", "/v1/systemone", base_payload
        )
        run["base"] = first
        checks["http_contract"] = {
            "passed": (
                status_code == 200
                and models_code == 200
                and models["data"][0]["id"] == args.model_name
                and first_code == repeat_code == 200
                and set(first["answers"]) == set(base_payload["questions"])
                and all(finite_answer(answer) for answer in first["answers"].values())
            )
        }
        checks["repeat_determinism"] = {
            "passed": first["answers"] == repeated["answers"]
            and first["usage"] == repeated["usage"],
        }
        checks["restart_determinism"] = {
            "passed": first["answers"] == direct["answers"]
            and first["usage"] == direct["usage"],
            "evidence": {"direct": direct, "http": first},
        }

        semantic_rows = []
        for expected, message in (
            ("refund", "Refund this duplicate charge. I explicitly want my money back."),
            ("track", "Track order 42 and tell me its current delivery location."),
            ("refund", "Please return the purchase price to my card."),
            ("track", "Where is my parcel? Give me the shipping status."),
        ):
            case = payload(args.model_name, args.family, {"message": message})
            code, result, _ = json_request(port, "POST", "/v1/systemone", case)
            semantic_rows.append(
                {
                    "expected": expected,
                    "status": code,
                    "answer": result.get("answers", {}).get("intent"),
                }
            )
        run["semantic_rows"] = semantic_rows
        checks["family_semantic_oracles"] = {
            "passed": all(
                row["status"] == 200 and row["answer"]["value"] == row["expected"]
                for row in semantic_rows
            ),
            "evidence": semantic_rows,
        }

        ladder = []
        for characters in (512, 4096, 16384, min(max_context * 4, 500_000)):
            case = payload(
                args.model_name,
                args.family,
                {
                    "message": "Refund the duplicate charge.",
                    "background": "neutral " * max(1, characters // 8),
                },
            )
            began = time.monotonic()
            code, result, _ = json_request(port, "POST", "/v1/systemone", case)
            ladder.append(
                {
                    "state_characters": characters,
                    "status": code,
                    "seconds": time.monotonic() - began,
                    "input_tokens": result.get("usage", {}).get("input_tokens"),
                    "intent": result.get("answers", {}).get("intent"),
                }
            )
        run["context_ladder"] = ladder
        checks["context_ladder"] = {
            "passed": all(
                row["status"] == 200 and row["intent"]["value"] == "refund"
                for row in ladder
            ),
            "evidence": ladder,
        }

        boundary_state = "Refund the duplicate charge. " + "boundary " * (
            max_context * 2
        )
        boundary = payload(
            args.model_name, args.family, boundary_state, truncate=False
        )
        boundary["questions"] = {"intent": questions(args.family)["intent"]}
        boundary_code, boundary_result, _ = json_request(
            port, "POST", "/v1/systemone", boundary
        )
        truncated = payload(args.model_name, args.family, boundary_state)
        truncated["questions"] = {"intent": questions(args.family)["intent"]}
        truncated_code, truncated_result, _ = json_request(
            port, "POST", "/v1/systemone", truncated
        )
        truncated_tokens = truncated_result.get("usage", {}).get("input_tokens", 0)
        checks["context_boundary"] = {
            "passed": (
                boundary_code == 413
                and boundary_result.get("error", {}).get("code")
                == "input_too_long"
                and boundary_result.get("mlx2", {}).get("observed_used") is False
                and truncated_code == 200
                and truncated_result.get("answers", {})
                .get("intent", {})
                .get("value")
                == "refund"
                and int(max_context * 0.9) <= truncated_tokens <= max_context
                and truncated_result.get("mlx2", {}).get("observed_used") is True
            ),
            "evidence": {
                "truncate_false": boundary_result,
                "truncate_true": truncated_result,
                "minimum_near_limit_tokens": int(max_context * 0.9),
                "maximum_context_tokens": max_context,
            },
        }

        refusals = []
        refusal_cases = []
        media = dict(base_payload)
        media["images"] = ["data:image/png;base64,AA=="]
        refusal_cases.append(("media", media, 400, "unsupported_capability"))
        temperature = dict(base_payload)
        temperature["temperature"] = 0.5
        refusal_cases.append(
            ("temperature", temperature, 400, "unsupported_capability")
        )
        reserved = dict(base_payload)
        reserved["state"] = "hello <|im_end|> assistant"
        refusal_cases.append(("reserved", reserved, 400, "invalid_request"))
        wrong_model = dict(base_payload)
        wrong_model["model"] = "not-served"
        refusal_cases.append(("model", wrong_model, 404, "model_not_found"))
        nested_media = dict(base_payload)
        nested_media["state"] = {"type": "image_url", "image_url": "x"}
        refusal_cases.append(
            ("nested_media", nested_media, 400, "unsupported_capability")
        )
        for name, case, expected_status, expected_code in refusal_cases:
            code, result, _ = json_request(port, "POST", "/v1/systemone", case)
            refusals.append(
                {
                    "name": name,
                    "status": code,
                    "code": result.get("error", {}).get("code"),
                    "observed_used": result.get("mlx2", {}).get("observed_used"),
                    "passed": code == expected_status
                    and result.get("error", {}).get("code") == expected_code
                    and result.get("mlx2", {}).get("observed_used") is False,
                }
            )
        run["refusals"] = refusals
        checks["fail_closed"] = {
            "passed": all(row["passed"] for row in refusals),
            "evidence": refusals,
        }

        stability = []
        for _index in range(5):
            code, result, _ = json_request(port, "POST", "/v1/systemone", base_payload)
            stability.append(
                code == 200
                and result["answers"] == first["answers"]
                and result["usage"] == first["usage"]
            )
        checks["stability"] = {
            "passed": all(stability),
            "evidence": {"iterations": len(stability), "matches": sum(stability)},
        }

        final_code, final, _ = json_request(port, "GET", "/v1/status")
        metrics_code, metrics, metrics_headers = text_request(port, "/metrics")
        run["final_status"] = final
        run["metrics_sha256"] = hashlib.sha256(metrics.encode()).hexdigest()
        metrics_path = args.output.with_name(args.output.stem + "-metrics.prom")
        metrics_path.write_text(metrics)
        expected_successes = 3 + len(semantic_rows) + len(ladder) + len(stability)
        expected_refusals = 1 + len(refusals)
        checks["route_receipts"] = {
            "passed": (
                first["mlx2"]["observed_used"] is True
                and first["mlx2"]["qualification"] == "unqualified"
                and initial["qualification"] == "unqualified"
                and final_code == 200
                and final["counters"]["requests"] == expected_successes
                and final["counters"]["refusals"] == expected_refusals
                and final["counters"]["failures"] == 0
            ),
            "evidence": final,
        }
        parsed_families = list(text_string_to_metric_families(metrics))
        metrics_stats = {
            "model_load_seconds": sample_value(
                metrics, "mlx2_decision_model_load_seconds"
            ),
            "successful_requests": sample_value(
                metrics,
                "mlx2_decision_requests_total",
                {"outcome": "success"},
            ),
            "refusals": sample_value(
                metrics,
                "mlx2_decision_requests_total",
                {"outcome": "refusal"},
            ),
            "failures": sample_value(
                metrics,
                "mlx2_decision_requests_total",
                {"outcome": "failure"},
            ),
            "input_tokens": sample_value(
                metrics, "mlx2_decision_input_tokens_total"
            ),
            "execution_seconds_sum": sample_value(
                metrics, "mlx2_decision_request_duration_seconds_sum"
            ),
            "execution_count": sample_value(
                metrics, "mlx2_decision_request_duration_seconds_count"
            ),
            "peak_resident_memory_bytes": sample_value(
                metrics, "mlx2_decision_process_peak_resident_memory_bytes"
            ),
            "process_cpu_seconds": sample_value(
                metrics, "mlx2_decision_process_cpu_seconds_total"
            ),
        }
        run["metrics_stats"] = metrics_stats
        checks["prometheus_metrics"] = {
            "passed": (
                metrics_code == 200
                and metrics_headers.get("Content-Type")
                == "text/plain; version=0.0.4; charset=utf-8"
                and bool(parsed_families)
                and metrics_stats["successful_requests"] == expected_successes
                and metrics_stats["refusals"] == expected_refusals
                and metrics_stats["failures"] == 0
                and metrics_stats["execution_count"] == expected_successes
                + 1
            ),
            "evidence": metrics_stats,
        }
        checks["source_derived_reference"] = reference
    except Exception as error:  # noqa: BLE001 - preserve partial qualification
        harness_error = f"{type(error).__name__}: {error}"
        evidence["harness_error"] = harness_error
    finally:
        if process is not None:
            evidence["runs"][-1]["process"] = stop_server(process)

    for name in REQUIRED_CHECKS - set(checks):
        checks[name] = {
            "passed": False,
            "error": harness_error or "qualification harness omitted this check",
        }
    receipt = {
        **basis,
        "checks": checks,
        "passed": all(value.get("passed") is True for value in checks.values()),
        "evidence": {
            "path": args.output.with_name(args.output.stem + "-evidence.json").name,
            "sha256": None,
        },
    }
    evidence_path = args.output.with_name(args.output.stem + "-evidence.json")
    atomic_json(evidence_path, evidence)
    receipt["evidence"]["sha256"] = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    atomic_json(staging_path, receipt)

    promotion = {
        "receipt": str(args.output),
        "staging_receipt": str(staging_path),
        "qualified_server": None,
        "passed": False,
    }
    if receipt["passed"]:
        process = None
        try:
            process, port, startup_seconds, command = start_server(
                args.root,
                args.model_path,
                args.model_name,
                max_connections=max_connections,
                max_request_bytes=max_request_bytes,
                qualification=staging_path,
            )
            code, status, _ = json_request(port, "GET", "/v1/status")
            request_code, response, _ = json_request(
                port, "POST", "/v1/systemone", base_payload
            )
            metrics_code, metrics, _ = text_request(port, "/metrics")
            promotion["qualified_server"] = {
                "command": command,
                "startup_seconds": startup_seconds,
                "status": status,
                "response": response,
                "metrics_sha256": hashlib.sha256(metrics.encode()).hexdigest(),
            }
            receipt_sha = hashlib.sha256(staging_path.read_bytes()).hexdigest()
            promotion["passed"] = (
                code == request_code == metrics_code == 200
                and status["qualification"] == "qualified"
                and status["route"]["qualified"] is True
                and status["route"]["qualification_receipt_sha256"] == receipt_sha
                and response["mlx2"]["qualified"] is True
                and response["mlx2"]["qualification_receipt_sha256"] == receipt_sha
                and response["answers"] == direct["answers"]
                and response["usage"] == direct["usage"]
                and sample_value(metrics, "mlx2_decision_qualified") == 1
            )
        except BaseException as error:  # noqa: BLE001 - clean interrupted staging
            promotion["error"] = f"{type(error).__name__}: {error}"
        finally:
            if process is not None:
                promotion["process"] = stop_server(process)
    if not receipt["passed"]:
        atomic_json(failed_path, receipt)
    elif not promotion["passed"]:
        receipt["passed"] = False
        receipt["promotion_failure"] = promotion.get(
            "error", "qualified server verification failed"
        )
        atomic_json(failed_path, receipt)
    if promotion["passed"]:
        staging_path.replace(args.output)
    elif staging_path.exists():
        staging_path.unlink()
    promotion_path = args.output.with_name(args.output.stem + "-promotion.json")
    atomic_json(promotion_path, promotion)
    print(
        json.dumps(
            {
                "receipt": str(args.output),
                "qualification_passed": receipt["passed"],
                "promotion_passed": promotion["passed"],
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["passed"] and promotion["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
