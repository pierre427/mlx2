#!/usr/bin/env python3
"""Receipt-first smoke for adaptive Qwen4 PLE plus incremental tokenization.

This client never launches a server or imports MLX.  Its default ``--dry-run``
prints the exact isolated service plan.  Live execution is possible only
against an already-running loopback service and requires an explicit ownership
acknowledgement.  The result remains an unqualified smoke, not qualification or
a performance measurement.

The isolated server must use one lane and one inflight request.  Both requests
suppress APCv2 writes so the growing request cannot inherit device prefix state;
the incremental tokenizer cache remains independently eligible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

SCHEMA = "mlx2.qwen4-ple-incremental-serving-smoke.v1"
ROOT = Path(__file__).resolve().parents[1]
READ_POLICY_SCHEMA = "mlx2.qwen4-ple-read-policy.v1"
TOKENIZER_SCHEMA = "mlx2.incremental-tokenizer-cache.v1"
POLICY_PATH = ROOT / "scripts" / "fixtures" / "qwen4_ple_adaptive_no_warm_policy.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8398)
    parser.add_argument("--base-characters", type=int, default=8192)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--status-timeout", type=float, default=15)
    parser.add_argument("--live-url")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--i-own-request-route", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def server_environment() -> dict[str, str]:
    return {
        "PYTHONPATH": str(ROOT / "src"),
    }


def server_command(args) -> list[str]:
    return [
        args.python,
        "-u",
        "-m",
        "mlx2.server",
        "--model",
        str(args.model),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--ordinary",
        "--execution-policy",
        str(POLICY_PATH),
        "--max-lanes",
        "1",
        "--max-inflight",
        "1",
        "--incremental-tokenizer-cache-entries",
        "4",
        "--incremental-tokenizer-cache-characters",
        str(1 << 20),
        "--incremental-tokenizer-cache-tokens",
        str(1 << 18),
        "--host-prompt-cache-entries",
        "8",
        "--host-prompt-cache-tokens",
        str(1 << 18),
    ]


def _document(characters: int) -> str:
    unit = "English and CJK 你好; def f(x): return x + 1; punctuation []{}.\n"
    return (unit * (characters // len(unit) + 1))[:characters]


def request_bodies(model_id: str, characters: int) -> tuple[dict, dict]:
    document = _document(characters)
    cold_messages = [
        {"role": "system", "content": "Answer briefly and exactly."},
        {"role": "user", "content": document},
    ]
    grown_messages = [
        *cold_messages,
        {"role": "assistant", "content": "Acknowledged."},
        {"role": "user", "content": "Continue with one word."},
    ]
    common = {
        "model": model_id,
        "max_tokens": 1,
        "temperature": 0,
        "enable_thinking": False,
        # There is no read-bypass request control.  Suppressing the first
        # write guarantees the second request cannot read an APCv2 prefix.
        "skip_writing_prefix_cache": True,
    }
    return (
        {**common, "messages": cold_messages},
        {**common, "messages": grown_messages},
    )


def _find_mapping(value, key):
    if isinstance(value, dict):
        found = value.get(key)
        if isinstance(found, dict):
            return found
        for child in value.values():
            result = _find_mapping(child, key)
            if result is not None:
                return result
    elif isinstance(value, list):
        for child in value:
            result = _find_mapping(child, key)
            if result is not None:
                return result
    return None


def _adaptive_table(status: dict) -> tuple[dict | None, str | None]:
    tables = ((status.get("execution") or {}).get("ple_tables") or [])
    adaptive = [
        table
        for table in tables
        if (table.get("read_policy") or {}).get("configured") == "adaptive"
    ]
    if len(adaptive) != 1:
        return None, f"status: expected one adaptive PLE table, found {len(adaptive)}"
    return adaptive[0], None


def _count(policy: dict, name: str) -> int:
    value = (policy.get("counts") or {}).get(name, 0)
    return int(value) if type(value) is int else -1


def _validate_tokenizer_receipts(
    cold_payload: dict, grown_payload: dict, after: dict
) -> list[str]:
    failures = []
    current = after.get("incremental_tokenizer_cache") or {}
    cold = _find_mapping(cold_payload.get("mlx2"), "prompt_tokenization")
    grown = _find_mapping(grown_payload.get("mlx2"), "prompt_tokenization")
    revision = current.get("tokenizer_revision")
    for name, receipt, action, observed in (
        ("cold", cold, "ordinary_full_validation", False),
        ("grown", grown, "incremental_hit", True),
    ):
        if receipt is None:
            failures.append(f"{name}: missing prompt_tokenization receipt")
            continue
        expected = {
            "schema": TOKENIZER_SCHEMA,
            "selected": True,
            "qualified": False,
            "serving_qualified": False,
            "action": action,
            "exact": True,
            "tokenizer_revision": revision,
            "refusal": None,
            "observed_used": observed,
        }
        for key, value in expected.items():
            if receipt.get(key) != value:
                failures.append(
                    f"{name}: prompt_tokenization.{key}={receipt.get(key)!r}, "
                    f"expected {value!r}"
                )
    expected_status = {
        "schema": TOKENIZER_SCHEMA,
        "selected": True,
        "qualified": False,
        "serving_qualified": False,
        "observed_used": True,
        "default": "off",
        "refusal": None,
    }
    for key, value in expected_status.items():
        if current.get(key) != value:
            failures.append(
                f"status: incremental_tokenizer_cache.{key}="
                f"{current.get(key)!r}, expected {value!r}"
            )
    if not isinstance(revision, str) or len(revision) != 64:
        failures.append("status: tokenizer revision is not a SHA-256 identity")
    if int(current.get("cold_validations", 0)) < 1:
        failures.append("status: cold_validations < 1")
    if int(current.get("incremental_hits", 0)) < 1:
        failures.append("status: incremental_hits < 1")
    return failures


def evaluate(before: dict, cold_payload: dict, grown_payload: dict, after: dict) -> list[str]:
    """Validate status deltas from an otherwise idle single-request service."""

    failures = _validate_tokenizer_receipts(cold_payload, grown_payload, after)
    before_table, error = _adaptive_table(before)
    if error is not None:
        failures.append(f"before {error}")
        return failures
    after_table, error = _adaptive_table(after)
    if error is not None:
        failures.append(f"after {error}")
        return failures

    before_policy = before_table["read_policy"]
    after_policy = after_table["read_policy"]
    effective = before_policy.get("effective")
    if effective not in {"serial", "pooled"}:
        failures.append(f"before: invalid effective arm {effective!r}")
        effective = "serial"
    fixed = {
        "schema": READ_POLICY_SCHEMA,
        "configured": "adaptive",
        "selected": True,
        "qualified": False,
        "phase": "load",
        "failure": None,
    }
    for when, policy in (("before", before_policy), ("after", after_policy)):
        for key, value in fixed.items():
            if policy.get(key) != value:
                failures.append(
                    f"{when}: read_policy.{key}={policy.get(key)!r}, expected {value!r}"
                )
        if policy.get("effective") != effective:
            failures.append(f"{when}: effective arm changed with warming disabled")
        warming = policy.get("warming") or {}
        expected_warming = {
            "enabled": False,
            "state": "disabled",
            "refresh_pending": False,
            "error": None,
        }
        for key, value in expected_warming.items():
            if warming.get(key) != value:
                failures.append(
                    f"{when}: warming.{key}={warming.get(key)!r}, expected {value!r}"
                )
        calibration = policy.get("calibration") or {}
        if calibration.get("phase") != "load":
            failures.append(f"{when}: calibration phase is not load")
        if calibration.get("selected") != effective:
            failures.append(f"{when}: calibration selection does not match effective")
        if calibration.get("rows_per_arm") != 128:
            failures.append(f"{when}: calibration rows_per_arm is not 128")
        if int(calibration.get("serial_ns", 0)) <= 0 or int(
            calibration.get("pooled_ns", 0)
        ) <= 0:
            failures.append(f"{when}: calibration timers are not positive")
        counts = policy.get("counts") or {}
        for name in (
            "warmed_calibrations",
            "warm_signals",
            "warm_refreshes",
            "warm_refresh_failures",
        ):
            if counts.get(name) != 0:
                failures.append(f"{when}: {name} is not zero")
        if counts.get("load_calibrations") != 1:
            failures.append(f"{when}: load_calibrations is not one")

    for arm in ("serial", "pooled"):
        for unit in ("calls", "rows"):
            name = f"foreground_{arm}_{unit}"
            delta = _count(after_policy, name) - _count(before_policy, name)
            if arm == effective and delta <= 0:
                failures.append(f"PLE: selected {name} delta is not positive")
            if arm != effective and delta != 0:
                failures.append(f"PLE: unselected {name} delta is not zero")
    for name in ("lookups", "rows", "unique_rows", "bytes_read"):
        before_value = int(before_table.get(name, 0))
        after_value = int(after_table.get(name, 0))
        if after_value - before_value <= 0:
            failures.append(f"PLE: {name} delta is not positive")

    receipt = after_policy.get("last_receipt") or {}
    expected_receipt = {
        "event": "foreground_read",
        "configured": "adaptive",
        "selected_arm": effective,
        "actual_arm": effective,
        "phase": "load",
        "warm_refresh_pending": False,
    }
    for key, value in expected_receipt.items():
        if receipt.get(key) != value:
            failures.append(
                f"PLE last_receipt.{key}={receipt.get(key)!r}, expected {value!r}"
            )
    if int(receipt.get("rows", 0)) <= 0:
        failures.append("PLE last_receipt.rows is not positive")
    return failures


def _request(base: str, path: str, *, body=None, timeout: float, token: str | None):
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(base.rstrip("/") + path, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read())


def _status_advanced(before: dict, after: dict) -> bool:
    before_table, before_error = _adaptive_table(before)
    after_table, after_error = _adaptive_table(after)
    if before_error or after_error:
        return False
    before_counts = before_table["read_policy"].get("counts") or {}
    after_counts = after_table["read_policy"].get("counts") or {}
    return sum(
        int(after_counts.get(f"foreground_{arm}_calls", 0))
        - int(before_counts.get(f"foreground_{arm}_calls", 0))
        for arm in ("serial", "pooled")
    ) > 0 and int(
        (after.get("incremental_tokenizer_cache") or {}).get("incremental_hits", 0)
    ) >= 1


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be in 1..65535")
    if min(args.base_characters, args.request_timeout, args.status_timeout) <= 0:
        raise SystemExit("character and timeout values must be positive")
    plan = {
        "schema": SCHEMA,
        "status": "unqualified serving smoke; not executed by construction",
        "model": str(args.model),
        "server_environment": server_environment(),
        "execution_policy": json.loads(POLICY_PATH.read_text()),
        "server_command": server_command(args),
        "request_controls": {
            "count": 2,
            "max_tokens": 1,
            "skip_writing_prefix_cache": True,
            "base_characters": args.base_characters,
        },
        "safety": {
            "launches_server": False,
            "imports_mlx": False,
            "whole_table_warming": False,
            "single_lane_and_inflight_required": True,
            "converted_sidecar_load_full_hashes_and_evicts_32gb": True,
            "ple_receipt_scope": "process-wide status delta; isolated service required",
        },
        "will_send_requests": bool(
            args.live_url and args.i_own_request_route and not args.dry_run
        ),
    }
    if args.dry_run or args.live_url is None:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if not args.i_own_request_route:
        print(
            "refusing: live requests require --i-own-request-route on an isolated service",
            file=sys.stderr,
        )
        return 2
    if args.out is None:
        print("refusing: live requests require --out", file=sys.stderr)
        return 2
    token = None
    if args.api_key_file is not None:
        token = args.api_key_file.read_text().strip()
        if not token:
            print("refusing: API key file is empty", file=sys.stderr)
            return 2

    _, models = _request(
        args.live_url,
        "/v1/models",
        timeout=args.request_timeout,
        token=token,
    )
    model_id = models["data"][0]["id"]
    if model_id != args.model.name:
        print(
            f"refusing: live model {model_id!r} != requested artifact {args.model.name!r}",
            file=sys.stderr,
        )
        return 2
    _, before = _request(
        args.live_url, "/v1/status", timeout=args.request_timeout, token=token
    )
    before_table, before_error = _adaptive_table(before)
    if before_error is not None:
        print(f"refusing: {before_error}", file=sys.stderr)
        return 2
    counts = before_table["read_policy"].get("counts") or {}
    if any(int(counts.get(f"foreground_{arm}_calls", 0)) for arm in ("serial", "pooled")):
        print("refusing: PLE foreground counters are not pristine", file=sys.stderr)
        return 2

    cold_body, grown_body = request_bodies(model_id, args.base_characters)
    cold_code, cold = _request(
        args.live_url,
        "/v1/chat/completions",
        body=cold_body,
        timeout=args.request_timeout,
        token=token,
    )
    grown_code, grown = _request(
        args.live_url,
        "/v1/chat/completions",
        body=grown_body,
        timeout=args.request_timeout,
        token=token,
    )
    deadline = time.monotonic() + args.status_timeout
    after = before
    while time.monotonic() < deadline:
        _, after = _request(
            args.live_url, "/v1/status", timeout=args.request_timeout, token=token
        )
        if _status_advanced(before, after):
            break
        time.sleep(0.25)
    failures = []
    if cold_code != 200:
        failures.append(f"cold: HTTP {cold_code}")
    if grown_code != 200:
        failures.append(f"grown: HTTP {grown_code}")
    failures.extend(evaluate(before, cold, grown, after))
    result = {
        **plan,
        "status": "unqualified serving smoke executed",
        "will_send_requests": True,
        "model_id": model_id,
        "artifact_config_sha256": hashlib.sha256(
            (args.model / "config.json").read_bytes()
        ).hexdigest(),
        "failures": failures,
        "go": not failures,
        "before": before,
        "after": after,
        "receipts": {
            "cold_prompt_tokenization": _find_mapping(
                cold.get("mlx2"), "prompt_tokenization"
            ),
            "grown_prompt_tokenization": _find_mapping(
                grown.get("mlx2"), "prompt_tokenization"
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"go": result["go"], "failures": failures, "out": str(args.out)}, indent=2))
    return 0 if result["go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
