"""Source-bound MTPLX #520 real-child research gate.

The default, CPU-only mode refuses stale qualification records before loading
MLX.  ``--candidate-probe --i-own-gpu`` is a separate experiment: it starts two
owned, private mlx2 children in qualification mode and measures their actual
HTTP stream and residency behavior. Candidate children are never selectable as
qualified supervisor routes.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "mlx2"


def source_sha256(source: Path = SOURCE) -> str:
    """Mirror ``serving.runtime_identity``'s Python-source digest, without MLX."""
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        digest.update(str(path.relative_to(source)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def interpreter_metadata(python: Path) -> dict[str, Any]:
    """Read selected interpreter package versions without importing MLX."""
    program = (
        "import importlib.metadata as m, json, platform; "
        "names=('mlx','transformers','numpy','tokenizers','jinja2','psutil','regex','safetensors'); "
        "print(json.dumps({'python':platform.python_version(),'macos':platform.mac_ver()[0],"
        "'packages':{n:m.version(n) for n in names}}))"
    )
    result = subprocess.run([str(python), "-c", program], text=True,
                            capture_output=True, timeout=20, check=True)
    return json.loads(result.stdout)


def qualification_preflight(model: Path, receipt: Path, source: Path = SOURCE,
                            runtime_meta: dict | None = None) -> dict[str, Any]:
    raw = receipt.read_bytes()
    record = json.loads(raw)
    observed_source = source_sha256(source)
    expected_source = (record.get("runtime") or {}).get("source_sha256")
    source_match = isinstance(expected_source, str) and expected_source == observed_source
    runtime_checks = {}
    if runtime_meta is not None:
        recorded = record.get("runtime") or {}
        for name in ("python", "macos", "mlx", "transformers"):
            actual = (runtime_meta.get("packages") or {}).get(name, runtime_meta.get(name))
            runtime_checks[name] = {"expected": recorded.get(name), "observed": actual,
                                    "match": recorded.get(name) == actual}
        for name, version in (recorded.get("dependencies") or {}).items():
            actual = (runtime_meta.get("packages") or {}).get(name)
            runtime_checks[name] = {"expected": version, "observed": actual,
                                    "match": version == actual}
    metadata_match = bool(runtime_checks) and all(row["match"] for row in runtime_checks.values())
    # A source match alone cannot qualify an artifact: native MLX, environment,
    # model fingerprint and *resolved* settings must also match at child load.
    return {
        "model_id": model.name,
        "model_path": str(model),
        "qualification_path": str(receipt),
        "qualification_sha256": hashlib.sha256(raw).hexdigest(),
        "record_passed": record.get("passed") is True,
        "record_source_sha256": expected_source,
        "candidate_source_sha256": observed_source,
        "source_match": source_match,
        "interpreter_checks": runtime_checks,
        "interpreter_metadata_match": metadata_match,
        "qualified_admission": (
            "refused_failed_record" if record.get("passed") is not True else
            "refused_stale_source" if not source_match else
            "refused_stale_interpreter" if runtime_meta is not None and not metadata_match else
            "needs_live_full_identity_check"
        ),
    }


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request(port: int, key: str, path: str, body: dict | None = None,
             timeout: float = 30) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    headers = {"Authorization": f"Bearer {key}"}
    payload = None
    if body is not None:
        payload = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    conn.request("POST" if body is not None else "GET", path, body=payload, headers=headers)
    return conn, conn.getresponse()


def _json(port: int, key: str, path: str) -> dict:
    conn, response = _request(port, key, path)
    try:
        if response.status != 200:
            raise RuntimeError(f"{path}: HTTP {response.status}")
        result = json.load(response)
        if not isinstance(result, dict):
            raise RuntimeError(f"{path}: expected object")
        return result
    finally:
        conn.close()


def _read_frame(response: http.client.HTTPResponse) -> tuple[bytes, Any]:
    frame = bytearray()
    while True:
        line = response.readline()
        if not line:
            raise RuntimeError("child stream ended before terminal receipt and [DONE]")
        frame.extend(line)
        if len(frame) > (2 << 20):
            raise RuntimeError("child SSE frame exceeded 2 MiB")
        if line in (b"\n", b"\r\n"):
            break
    payload = b"\n".join(line[5:].strip() for line in frame.splitlines() if line.startswith(b"data:"))
    if not payload:
        raise RuntimeError("child emitted an SSE frame without data")
    if payload == b"[DONE]":
        return bytes(frame), "[DONE]"
    event = json.loads(payload)
    if not isinstance(event, dict):
        raise RuntimeError("child emitted a non-object SSE event")
    return bytes(frame), event


def _stream(port: int, key: str, model_id: str, path: str, *, max_tokens: int = 8,
            min_tokens: int = 0, stop_after_first: bool = False) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model_id, "stream": True, "temperature": 0,
        "max_tokens": max_tokens, "enable_thinking": False,
    }
    if min_tokens:
        body["min_tokens"] = min_tokens
    if path == "/v1/chat/completions":
        body["messages"] = [{"role": "user", "content": "Reply with one short fact about apples."}]
    elif path == "/v1/completions":
        body["prompt"] = "One short fact about apples:"
    else:
        raise ValueError("unsupported endpoint")
    conn, response = _request(port, key, path, body, timeout=120)
    frames = 0
    terminal = None
    done = False
    try:
        if response.status != 200:
            raise RuntimeError(f"{path}: HTTP {response.status}: {response.read(4096)!r}")
        if response.getheader("Content-Type", "").split(";", 1)[0] != "text/event-stream":
            raise RuntimeError(f"{path}: non-SSE content type")
        while True:
            _raw, event = _read_frame(response)
            frames += 1
            if event == "[DONE]":
                if terminal is None:
                    raise RuntimeError("[DONE] without terminal mlx2 receipt")
                done = True
                break
            if event.get("model") != model_id:
                raise RuntimeError("child response model mismatch")
            if "mlx2" in event:
                receipt = event["mlx2"]
                if not isinstance(receipt, dict) or receipt.get("qualification") != "candidate" or receipt.get("route_receipt") != "candidate_validation":
                    raise RuntimeError("unexpected child terminal qualification/receipt")
                terminal = {
                    "route": receipt.get("route"),
                    "route_receipt": receipt.get("route_receipt"),
                    "qualification": receipt.get("qualification"),
                    "request_id": receipt.get("request_id"),
                }
            if stop_after_first:
                return {"frames": frames, "cancelled_client_after_first_frame": terminal is None,
                        "terminal_before_cancel": terminal is not None}
        return {"frames": frames, "terminal": terminal, "done": done}
    finally:
        conn.close()


def _check_status(port: int, key: str, model_id: str) -> dict[str, Any]:
    status = _json(port, key, "/v1/status")
    models = _json(port, key, "/v1/models")
    rows = models.get("data") or []
    if (status.get("state") != "ready" or status.get("healthy") is not True
            or status.get("model") != model_id or status.get("qualification") != "candidate"
            or status.get("route_receipt") != "candidate_validation"
            or not any(row.get("id") == model_id and row.get("loaded") is True for row in rows)):
        raise RuntimeError(f"child {model_id}: candidate status/identity mismatch")
    return {
        "model": model_id,
        "runtime": status.get("runtime"),
        "artifact": status.get("artifact"),
        "route": (status.get("settings") or {}).get("route"),
        "route_receipt": status.get("route_receipt"),
        "qualification": status.get("qualification"),
        "selected_capabilities": status.get("selected_capabilities"),
        "qualified_capabilities": status.get("qualified_capabilities"),
        "process_physical_footprint_bytes": status.get("process_physical_footprint_bytes"),
        "metal_active_bytes": status.get("metal_active_bytes"),
        "metal_peak_bytes": status.get("metal_peak_bytes"),
    }


def _launch(model: Path, port: int, key_file: Path, cache_dir: Path, log: Path,
            *, python: Path) -> tuple[subprocess.Popen, Any]:
    cache_dir.mkdir(parents=True)
    command = [
        str(python), "-u", "-m", "mlx2.server", "--model", str(model),
        "--host", "127.0.0.1", "--port", str(port), "--ordinary",
        "--qualification-mode", "--api-key-file", str(key_file),
        "--cache-dir", str(cache_dir), "--max-context", "2048",
        "--max-lanes", "1", "--max-inflight", "2",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    handle = log.open("wb")
    try:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    except BaseException:
        handle.close()
        raise
    return process, handle


def _wait_ready(process: subprocess.Popen, port: int, key: str, model_id: str,
                timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"child {model_id} exited during startup (rc {process.returncode})")
        try:
            status = _json(port, key, "/v1/status")
        except (OSError, TimeoutError, http.client.HTTPException, RuntimeError):
            # A listener can exist while its model is still loading. Only a
            # ready status is interpreted as an identity/qualification result.
            status = {}
        if status.get("state") == "ready":
            return _check_status(port, key, model_id)
        time.sleep(1)
    raise TimeoutError(f"child {model_id} did not reach candidate-ready state")


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def candidate_probe(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    if not args.i_own_gpu:
        raise ValueError("candidate probe requires --i-own-gpu under the shared GPU lease")
    if args.model_a.name == args.model_b.name:
        raise ValueError("two distinct exact model IDs required")
    for model in (args.model_a, args.model_b):
        if not model.is_dir():
            raise ValueError(f"model directory missing: {model}")
    out.mkdir(parents=True, exist_ok=True)
    rows: dict[str, Any] = {"tier": "research_candidate_only", "qualified_supervisor_admission": False,
                            "source_sha256": source_sha256(), "children": {}}
    processes: list[tuple[subprocess.Popen, Any]] = []
    with tempfile.TemporaryDirectory(prefix="mtplx520-key-") as temporary:
        key = secrets.token_urlsafe(32)
        key_file = Path(temporary) / "api-key"
        key_file.write_text(key + "\n")
        key_file.chmod(0o600)
        ports = [_port(), _port()]
        if ports[0] == ports[1] or 8282 in ports:
            raise RuntimeError("private port allocation collided")
        try:
            for label, model, port in zip(("a", "b"), (args.model_a, args.model_b), ports):
                process, handle = _launch(model, port, key_file, out / f"apc-{label}",
                                          out / f"child-{label}.log", python=args.python)
                processes.append((process, handle))
                rows["children"][label] = {"pid": process.pid, "port": port,
                                            "status": _wait_ready(process, port, key, model.name,
                                                                  args.startup_timeout)}
            # These are live simultaneous child footprints, not a projected
            # budget or a claim that shared Metal pages sum uniquely.
            snapshots = [_check_status(port, key, model.name)
                         for port, model in zip(ports, (args.model_a, args.model_b))]
            if any(type(row["process_physical_footprint_bytes"]) is not int
                   or row["process_physical_footprint_bytes"] <= 0
                   or type(row["metal_active_bytes"]) is not int
                   or row["metal_active_bytes"] <= 0 for row in snapshots):
                raise RuntimeError("real child residency counters were absent or zero")
            rows["simultaneous_residency"] = {
                "sampled_status": snapshots,
                "summed_process_physical_footprint_bytes": sum(
                    row["process_physical_footprint_bytes"] for row in snapshots),
                "summed_metal_active_bytes": sum(
                    row["metal_active_bytes"] for row in snapshots),
            }
            for label, model, port in zip(("a", "b"), (args.model_a, args.model_b), ports):
                rows["children"][label]["chat_sse"] = _stream(
                    port, key, model.name, "/v1/chat/completions")
                rows["children"][label]["completions_sse"] = _stream(
                    port, key, model.name, "/v1/completions")
            after_streams = [_check_status(port, key, model.name)
                             for port, model in zip(ports, (args.model_a, args.model_b))]
            rows["simultaneous_residency"]["after_streams"] = after_streams
            rows["simultaneous_residency"]["summed_footprint_after_streams_bytes"] = sum(
                int(row["process_physical_footprint_bytes"] or 0) for row in after_streams)
            rows["cancel"] = _stream(ports[0], key, args.model_a.name,
                                      "/v1/chat/completions", max_tokens=256,
                                      min_tokens=256, stop_after_first=True)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                status = _json(ports[0], key, "/v1/status")
                if status.get("accepted_lifecycles") == 0 and status.get("inflight") == 0:
                    rows["cancel"]["child_idle_after_disconnect"] = True
                    break
                time.sleep(0.5)
            else:
                rows["cancel"]["child_idle_after_disconnect"] = False
            if not rows["cancel"]["cancelled_client_after_first_frame"] or not rows["cancel"]["child_idle_after_disconnect"]:
                raise RuntimeError("cancellation was not observed before terminal or did not drain")
            # Kill only the child process this probe spawned. Its open stream
            # must not synthesize a terminal success; the peer must survive.
            conn, response = _request(ports[0], key, "/v1/chat/completions", {
                "model": args.model_a.name, "messages": [{"role": "user", "content": "Count upward."}],
                "stream": True, "temperature": 0, "max_tokens": 256, "min_tokens": 256,
                "enable_thinking": False,
            }, timeout=120)
            if response.status != 200:
                conn.close()
                raise RuntimeError(f"crash stream: HTTP {response.status}")
            _raw, first = _read_frame(response)
            if first == "[DONE]" or "mlx2" in first:
                conn.close()
                raise RuntimeError("crash stream ended before controlled child death")
            os.killpg(processes[0][0].pid, signal.SIGKILL)
            processes[0][0].wait(timeout=10)
            try:
                while True:
                    _raw, event = _read_frame(response)
                    if event == "[DONE]" or (isinstance(event, dict) and "mlx2" in event):
                        raise RuntimeError("dead child produced a terminal success")
            except (OSError, http.client.HTTPException, RuntimeError) as exc:
                if str(exc) == "dead child produced a terminal success":
                    raise
                rows["controlled_death"] = {"no_terminal_success": True,
                                             "stream_error": type(exc).__name__}
            finally:
                conn.close()
            rows["peer_after_death"] = _stream(
                ports[1], key, args.model_b.name, "/v1/chat/completions")
            rows["candidate_protocol_passed"] = True
        finally:
            for process, handle in reversed(processes):
                _stop(process)
                handle.close()
            (out / "candidate-partial.json").write_text(
                json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True)
    parser.add_argument("--qualification-a", type=Path, required=True)
    parser.add_argument("--qualification-b", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate-probe", action="store_true")
    parser.add_argument("--i-own-gpu", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--startup-timeout", type=float, default=600)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runtime_meta = interpreter_metadata(args.python)
    preflight = {
        "schema": "mlx2.mtplx520-real-child-gate.v1",
        "source_checkout": str(ROOT),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selected_interpreter": str(args.python),
        "qualified_preflight": [
            qualification_preflight(args.model_a, args.qualification_a,
                                    runtime_meta=runtime_meta),
            qualification_preflight(args.model_b, args.qualification_b,
                                    runtime_meta=runtime_meta),
        ],
    }
    preflight["qualified_children_admitted"] = False
    (args.out / "preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    if not args.candidate_probe:
        print(json.dumps({"preflight": str(args.out / "preflight.json"),
                          "qualified_children_admitted": False,
                          "candidate_probe_run": False}))
        # A static source match is insufficient to admit a route. Keep the
        # default command nonzero so it cannot accidentally be used as a
        # successful shell precondition for starting qualified children.
        return 2
    def interrupted(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"candidate probe received {signal.Signals(signum).name}")

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)
    try:
        probe = candidate_probe(args, args.out)
    except BaseException as exc:
        (args.out / "candidate-error.json").write_text(json.dumps({
            "schema": "mlx2.mtplx520-real-child-candidate-error.v1",
            "error_type": type(exc).__name__, "error": str(exc),
            "qualified_children_admitted": False,
        }, indent=2, sort_keys=True) + "\n")
        raise
    (args.out / "candidate-probe.json").write_text(json.dumps(probe, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"preflight": str(args.out / "preflight.json"),
                      "candidate_probe": str(args.out / "candidate-probe.json"),
                      "candidate_protocol_passed": probe["candidate_protocol_passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
