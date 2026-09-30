#!/usr/bin/env python3
"""Run a bounded live segmented/QSA rollback engagement gate.

The caller owns the CPG lease and holds both lab GPU flock files.  The receipt
fails closed unless one atomic B2 request engages segmented self-MTP and QSA
private-delta while recording no physical K/V rolls.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCKS = tuple(
    Path(value) for value in os.environ.get("MLX2_GPU_LOCK_PATHS", "").split(os.pathsep)
    if value
)
SCHEMA = "mlx2.qsa-rollback-live-gate.v1"
LONG_CONTEXT_WORDS = (
    "data the system value table record signal network window object format "
    "memory number simple paper river stone garden market winter summer letter "
    "music color train bridge forest island doctor teacher kitchen engine planet "
    "silver orange yellow mountain village morning evening history science future "
    "picture water light house story world money power place point group"
).split()


def _long_context_filler(tokens: int) -> str:
    """Deterministic one-token words shared with the serving qualifier."""
    state, words = 20260918, []
    for _ in range(tokens):
        state = (state * 1103515245 + 12345) % (1 << 31)
        words.append(LONG_CONTEXT_WORDS[(state >> 8) % len(LONG_CONTEXT_WORDS)])
    if words:
        words[0] = LONG_CONTEXT_WORDS[0]
    return " ".join(words)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_parent_locks() -> dict[str, str]:
    if len(LOCKS) != 2:
        raise SystemExit("set MLX2_GPU_LOCK_PATHS to the two host lock files")
    if os.environ.get("MLX2_GPU_DUAL_FLOCK_HELD") != "1":
        raise SystemExit("live gate requires MLX2_GPU_DUAL_FLOCK_HELD=1")
    result = {}
    for path in LOCKS:
        if not path.is_file():
            raise SystemExit(f"GPU lock is not a regular file: {path}")
        with path.open("a+") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                result[str(path)] = "contended_by_parent"
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                raise SystemExit(f"GPU lock is not held by parent: {path}")
    return result


def _request(base: str, path: str, body: dict | None = None, timeout: int = 300):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        base + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _wait_ready(base: str, server: subprocess.Popen, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"server exited during startup with {server.returncode}")
        try:
            last = _request(base, "/v1/status", timeout=2)
            if last.get("healthy"):
                return last
        except Exception:
            pass
        time.sleep(0.25)
    raise TimeoutError(f"server was not ready; last status={last!r}")


def _counter(status: dict, section: str, key: str) -> int:
    return int(((status.get("execution") or {}).get(section) or {}).get(key, 0))


def _delta(before: dict, after: dict, section: str, key: str) -> int:
    return _counter(after, section, key) - _counter(before, section, key)


def _validate(before: dict, after: dict, responses: list[dict]) -> dict:
    completion_tokens = [
        int(((response.get("usage") or {}).get("completion_tokens", 0)))
        for response in responses
    ]
    segmented_counter = max(
        _delta(before, after, "segmented_mtp", "transaction_branches"),
        _delta(before, after, "segmented_mtp", "true_batched_target_forwards"),
    )
    segmented_receipts = sum(
        bool(
            ((response.get("mlx2") or {}).get("mtp") or {}).get("route")
            == "segmented_self_mtp"
            and 2
            in (
                ((response.get("mlx2") or {}).get("mtp") or {}).get(
                    "observed_compute_widths"
                )
                or ()
            )
        )
        for response in responses
    )
    segmented = max(segmented_counter, segmented_receipts)
    private_delta = _delta(
        before, after, "segmented_mtp", "private_delta_attention_calls"
    )
    physical_rolls = _delta(
        before, after, "qsa_rollback", "physical_kv_roll_calls"
    )
    segmented_finalize = _delta(
        before, after, "qsa_rollback", "segmented_self_mtp_finalize_calls"
    )
    qsa_rollback_present = "qsa_rollback" in (after.get("execution") or {})
    failures = []
    if len(responses) != 2 or any("error" in response for response in responses):
        failures.append("atomic B2 request did not complete successfully")
    if len(completion_tokens) != 2 or any(tokens < 12 for tokens in completion_tokens):
        failures.append("atomic B2 request did not force 12 completion tokens per row")
    if segmented < 1:
        failures.append("segmented self-MTP did not engage")
    if private_delta < 1:
        failures.append("QSA private-delta did not engage")
    if not qsa_rollback_present:
        failures.append("QSA rollback telemetry is absent for this model topology")
    if physical_rolls != 0:
        failures.append(f"physical K/V rollback engaged {physical_rolls} time(s)")
    if private_delta > 0 and segmented_finalize < 1:
        failures.append("private-delta engaged without segmented finalize telemetry")
    return {
        "passed": not failures,
        "failures": failures,
        "deltas": {
            "segmented_engagement": segmented,
            "segmented_counter_engagement": segmented_counter,
            "segmented_receipt_engagement": segmented_receipts,
            "private_delta_attention_calls": private_delta,
            "segmented_self_mtp_finalize_calls": segmented_finalize,
            "physical_kv_roll_calls": physical_rolls,
            "completion_tokens": completion_tokens,
        },
        "qsa_rollback_telemetry_present": qsa_rollback_present,
        "zero_physical_rolls_is_nonvacuous": (
            qsa_rollback_present and private_delta > 0
        ),
    }


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--port", type=int, default=8396)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--cache-bytes", type=int, default=12 * 1024**3)
    parser.add_argument(
        "--shared-prefix-tokens",
        type=int,
        default=0,
        help="Prime this many one-token words into APCv2 before the atomic B2 gate",
    )
    parser.add_argument("--ownership-receipt", required=True)
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args(argv)
    if not args.i_own_the_gpu:
        parser.error("Metal execution requires --i-own-the-gpu")
    if args.shared_prefix_tokens < 0:
        parser.error("--shared-prefix-tokens must be non-negative")
    if args.shared_prefix_tokens and args.shared_prefix_tokens + 128 > args.max_context:
        parser.error("shared prefix needs at least 128 tokens of context headroom")
    locks = _require_parent_locks()
    model = args.model.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"refusing to overwrite existing receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            parser.error(f"port {args.port} is occupied")

    receipt = {
        "schema": SCHEMA,
        "status": "failed",
        "kind": "live fail-closed mechanism gate; not serving qualification",
        "started_at": datetime.now(UTC).isoformat(),
        "source": {
            "git_revision": _git("rev-parse", "HEAD"),
            "tracked_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "harness_sha256": _sha256(Path(__file__)),
        },
        "model_path": str(model),
        "ownership_receipt": args.ownership_receipt,
        "gpu_locks": locks,
        "request": {
            "max_context": args.max_context,
            "shared_prefix_tokens": args.shared_prefix_tokens,
            "batch_width": 2,
            "max_tokens": 16,
            "min_tokens": 12,
            "cache_bytes": args.cache_bytes,
        },
    }
    server = None
    log_path = output.with_suffix(".server.log")
    base = f"http://127.0.0.1:{args.port}"
    try:
        with tempfile.TemporaryDirectory(prefix="qsa-rollback-live-") as temporary:
            command = [
                args.python,
                "-m",
                "mlx2.server",
                "--model",
                str(model),
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
                "--native-mtp",
                "--qualification-mode",
                "--max-context",
                str(args.max_context),
                "--max-lanes",
                "2",
                "--max-inflight",
                "2",
                "--coalesce-window-ms",
                "50",
                "--batch-cohort-timeout-ms",
                "3000",
                "--cache-bytes",
                str(args.cache_bytes),
                "--cache-dir",
                str(Path(temporary) / "cache"),
            ]
            receipt["server_command"] = command
            env = {
                **os.environ,
                "PYTHONPATH": str(ROOT / "src"),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
            log = log_path.open("w")
            server = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            _wait_ready(base, server, args.timeout_seconds)
            if args.shared_prefix_tokens:
                content = (
                    _long_context_filler(args.shared_prefix_tokens)
                    + "\nReturn READY eight times separated by single spaces."
                )
            else:
                content = (
                    "Return READY eight times separated by single spaces. Context marker "
                    + ("alpha " * 96)
                )
            common = {
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 16,
                "min_tokens": 12,
                "temperature": 0,
                "think": False,
            }
            prime = None
            if args.shared_prefix_tokens:
                prime = _request(
                    base,
                    "/v1/chat/completions",
                    common,
                    timeout=args.timeout_seconds,
                )
            before = _request(base, "/v1/status")
            cohort = {"id": f"qsa-rollback-{uuid.uuid4().hex}", "size": 2}
            bodies = [
                {**common, "batch_cohort": cohort}
                for _ in range(2)
            ]
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(_request, base, "/v1/chat/completions", body)
                    for body in bodies
                ]
                responses = []
                for future in futures:
                    try:
                        responses.append(future.result(timeout=args.timeout_seconds))
                    except Exception as exc:
                        responses.append({"error": f"{type(exc).__name__}: {exc}"})
            after = _request(base, "/v1/status")
            validation = _validate(before, after, responses)
            receipt.update(
                {
                    "status_before": before,
                    "status_after": after,
                    "prime_response": prime,
                    "responses": responses,
                    "validation": validation,
                    "status": "passed" if validation["passed"] else "failed",
                }
            )
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
        receipt["finished_at"] = datetime.now(UTC).isoformat()
        receipt["server_log"] = str(log_path)
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt.get("validation", receipt.get("error")), indent=2))
    return 0 if receipt.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(run())
