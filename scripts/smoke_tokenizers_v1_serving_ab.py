#!/usr/bin/env python3
"""Serving A/B smoke for the pinned HF Tokenizers v1 worker candidate.

The harness launches two ordinary-decode services sequentially: first without
the worker, then with the exact retained local manifest.  Both receive the
same long greedy completion request with APCv2 writes suppressed.  Response
text, output IDs when exposed, and prompt-token counts must agree exactly.

This is an unqualified serving smoke, not qualification or a performance run.
Native execution loads a real model and refuses without ``--i-own-gpu``.  That
flag acknowledges an already-owned lease; it does not acquire or claim either
project GPU lock.  ``--dry-run`` performs only local identity checks and does
not import MLX, launch a service, or start the v1 worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SCHEMA = "mlx2.tokenizers-v1-serving-ab-smoke.v1"
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "qualification/runs/tokenizers-v1-intake-20261004/worker-manifest.local.json"
)
MANIFEST_SHA256 = "b705dacd742d52bc82ce1cc66b8d50c08da20d9c9e16d646b82bdd94b4bcf2d0"
ARTIFACT = Path.home() / "mlx-models" / "Qwen3.8-Flash-Next-MLX-4bit-MTP"
WORKER_ENV = "MLX2_TOKENIZERS_V1_MANIFEST"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def preflight() -> dict:
    """Verify every manifest-bound file and the exact artifact relationship."""

    failures = []
    manifest = None
    actual_manifest_sha = None
    try:
        actual_manifest_sha = _sha256(MANIFEST)
        if actual_manifest_sha != MANIFEST_SHA256:
            failures.append("worker manifest SHA-256 differs from retained intake")
        manifest = json.loads(MANIFEST.read_text())
    except (OSError, ValueError) as error:
        failures.append(f"manifest unreadable: {type(error).__name__}: {error}")
    files = {}
    if manifest is not None:
        if manifest.get("schema") != "mlx2.tokenizers-v1-worker.v1":
            failures.append("manifest schema differs")
        if manifest.get("qualification") != "cpu_exact_encode_candidate":
            failures.append("manifest is not the retained CPU exact candidate")
        if manifest.get("minimum_chars") != 8192:
            failures.append("manifest minimum_chars differs from 8192")
        tokenizer_parent = Path(manifest["tokenizer"]["path"]).resolve().parent
        if tokenizer_parent != ARTIFACT.resolve():
            failures.append("manifest tokenizer is not bound to the required artifact")
        for field in (
            "python",
            "extension",
            "wheel",
            "tokenizer",
            "config",
            "chat_template",
            "cpu_receipt",
        ):
            entry = manifest.get(field) or {}
            path = Path(entry.get("path", ""))
            record = {
                "path": str(path),
                "expected_sha256": entry.get("sha256"),
                "exists": path.is_file(),
                "actual_sha256": None,
            }
            if path.is_file():
                record["actual_sha256"] = _sha256(path)
            if record["actual_sha256"] != record["expected_sha256"]:
                failures.append(f"manifest {field} identity mismatch")
            files[field] = record
    for required in ("config.json", "model.safetensors.index.json"):
        if not (ARTIFACT / required).is_file():
            failures.append(f"artifact is missing {required}")
    return {
        "manifest": str(MANIFEST.resolve()),
        "manifest_expected_sha256": MANIFEST_SHA256,
        "manifest_actual_sha256": actual_manifest_sha,
        "artifact": str(ARTIFACT.resolve()),
        "files": files,
        "failures": failures,
        "go": not failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8398)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--shutdown-timeout", type=float, default=120)
    parser.add_argument("--prompt-characters", type=int, default=9000)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--i-own-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def server_command(args) -> list[str]:
    return [
        args.python,
        "-u",
        "-m",
        "mlx2.server",
        "--model",
        str(ARTIFACT),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--ordinary",
        "--max-lanes",
        "1",
        "--max-inflight",
        "1",
        "--incremental-tokenizer-cache-entries",
        "0",
    ]


def server_environment(*, candidate: bool) -> dict[str, str]:
    source = str(ROOT / "src")
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        [source]
        + [item for item in existing.split(os.pathsep) if item and item != source]
    )
    environment.pop(WORKER_ENV, None)
    if candidate:
        environment[WORKER_ENV] = str(MANIFEST.resolve())
    return environment


def _request(base: str, path: str, body=None, *, timeout: float):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        base + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read())


def _find(value, key):
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = _find(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find(child, key)
            if found is not None:
                return found
    return None


def _output_ids(payload):
    found = _find(payload.get("mlx2"), "output_token_ids")
    if isinstance(found, list) and all(type(token) is int for token in found):
        return found
    return None


def _text(payload):
    choices = payload.get("choices") or []
    if not choices:
        return None
    return choices[0].get("text")


def summarize_response(payload: dict) -> dict:
    text = _text(payload)
    ids = _output_ids(payload)
    return {
        "text_sha256": (
            hashlib.sha256(text.encode()).hexdigest() if isinstance(text, str) else None
        ),
        "text_characters": len(text) if isinstance(text, str) else None,
        "output_token_ids_available": ids is not None,
        "output_token_ids": ids,
        "usage": payload.get("usage"),
        "skip_writing_prefix_cache": _find(
            payload.get("mlx2"), "skip_writing_prefix_cache"
        ),
    }


def evaluate(ordinary: dict, candidate: dict) -> list[str]:
    failures = []
    ordinary_response = ordinary.get("response") or {}
    candidate_response = candidate.get("response") or {}
    ordinary_status = ordinary.get("status") or {}
    candidate_status = candidate.get("status") or {}
    ordinary_v1 = ordinary_status.get("tokenizers_v1") or {}
    candidate_v1 = candidate_status.get("tokenizers_v1") or {}
    for name, arm in (("ordinary", ordinary), ("candidate", candidate)):
        if arm.get("http_status") != 200:
            failures.append(f"{name}: HTTP {arm.get('http_status')}")
        if (
            _find((arm.get("response") or {}).get("mlx2"), "skip_writing_prefix_cache")
            is not True
        ):
            failures.append(f"{name}: APCv2 write suppression absent from receipt")
        incremental = (arm.get("status") or {}).get("incremental_tokenizer_cache") or {}
        if incremental.get("selected") is not False:
            failures.append(f"{name}: incremental tokenizer cache is selected")
    if ordinary_v1.get("selected") is not False:
        failures.append("ordinary: tokenizers v1 is selected")
    if candidate_v1.get("selected") is not True:
        failures.append("candidate: tokenizers v1 is not selected")
    if candidate_v1.get("observed_used") is not True:
        failures.append("candidate: tokenizers v1 was not observed used")
    if candidate_v1.get("serving_qualified") is not False:
        failures.append("candidate: tokenizers v1 incorrectly serving-qualified")
    if Path(candidate_v1.get("manifest_path", "")).resolve() != MANIFEST.resolve():
        failures.append("candidate: selected manifest path differs")
    counts = candidate_v1.get("counts") or {}
    if int(counts.get("successful_encodes", 0)) <= 0:
        failures.append("candidate: successful_encodes <= 0")
    if int(counts.get("failures", -1)) != 0:
        failures.append(f"candidate: failures={counts.get('failures')!r}")
    if _text(ordinary_response) != _text(candidate_response):
        failures.append("response text differs")
    ordinary_ids = _output_ids(ordinary_response)
    candidate_ids = _output_ids(candidate_response)
    if (ordinary_ids is None) != (candidate_ids is None):
        failures.append("output token IDs are available in only one arm")
    elif ordinary_ids is not None and ordinary_ids != candidate_ids:
        failures.append("output token IDs differ")
    ordinary_prompt = (ordinary_response.get("usage") or {}).get("prompt_tokens")
    candidate_prompt = (candidate_response.get("usage") or {}).get("prompt_tokens")
    if ordinary_prompt != candidate_prompt or not isinstance(ordinary_prompt, int):
        failures.append("prompt-token count differs or is absent")
    return failures


def _stop(process: subprocess.Popen, *, timeout: float) -> dict:
    was_running = process.poll() is None
    result = {
        "was_running": was_running,
        "terminated": False,
        "killed": False,
        "returncode": None,
    }
    if was_running:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=timeout)
            result["terminated"] = True
        except ProcessLookupError:
            process.wait(timeout=min(timeout, 30))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
            result["killed"] = True
    result["returncode"] = process.returncode
    return result


def _prompt(characters: int) -> str:
    unit = "English and CJK 你好; def f(x): return x + 1; punctuation []{}.\n"
    body = (unit * (characters // len(unit) + 1))[:characters]
    return body + "\nAnswer with exactly one short sentence."


def _run_arm(args, *, name: str, candidate: bool, prompt: str) -> dict:
    command = server_command(args)
    log_path = args.out.with_suffix(args.out.suffix + f".{name}.server.log")
    log = log_path.open("w")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=server_environment(candidate=candidate),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    base = f"http://{args.host}:{args.port}"
    result = {
        "name": name,
        "candidate": candidate,
        "pid": process.pid,
        "log": str(log_path),
        "failures": [],
    }
    try:
        deadline = time.monotonic() + args.startup_timeout
        while True:
            if process.poll() is not None:
                raise RuntimeError(
                    f"server exited during startup: {process.returncode}"
                )
            try:
                health, _payload = _request(
                    base, "/health", timeout=min(args.request_timeout, 5)
                )
                if health == 200:
                    break
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError("server did not become healthy before timeout")
            time.sleep(2)
        _code, models = _request(base, "/v1/models", timeout=args.request_timeout)
        model_id = models["data"][0]["id"]
        if model_id != ARTIFACT.name:
            raise RuntimeError(
                f"launched model identity {model_id!r} != {ARTIFACT.name!r}"
            )
        body = {
            "model": model_id,
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "skip_writing_prefix_cache": True,
        }
        code, response = _request(
            base, "/v1/completions", body, timeout=args.request_timeout
        )
        _status_code, status = _request(
            base, "/v1/status", timeout=args.request_timeout
        )
        result.update(
            model_id=model_id,
            http_status=code,
            response=response,
            status=status,
        )
    except Exception as error:  # noqa: BLE001 - smoke records and cleans up
        result["failures"].append(f"{type(error).__name__}: {error}")
    finally:
        result["shutdown"] = _stop(process, timeout=args.shutdown_timeout)
        log.close()
        result["server_log_tail"] = log_path.read_text(errors="replace")[-4000:]
    return result


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be in 1..65535")
    if (
        min(
            args.startup_timeout,
            args.request_timeout,
            args.shutdown_timeout,
            args.max_tokens,
        )
        <= 0
    ):
        raise SystemExit("timeouts and --max-tokens must be positive")

    identity = preflight()
    minimum_chars = 8192
    if args.prompt_characters <= minimum_chars:
        raise SystemExit("--prompt-characters must be greater than 8192")
    plan = {
        "schema": SCHEMA,
        "status": "unqualified A/B smoke; not executed by construction",
        "identity": identity,
        "arms": [
            {"name": "ordinary", "worker_manifest": None},
            {"name": "candidate", "worker_manifest": str(MANIFEST.resolve())},
        ],
        "server_command": server_command(args),
        "prompt_characters": args.prompt_characters,
        "apcv2_write_suppression": True,
        "incremental_tokenizer_cache_selected": False,
        "will_execute": bool(args.i_own_gpu and not args.dry_run),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0 if identity["go"] else 1
    if not args.i_own_gpu:
        print(
            "refusing: real serving requires --i-own-gpu under an already-owned paired lease",
            file=sys.stderr,
        )
        return 2
    if not identity["go"]:
        print("refusing: retained manifest/artifact identity failed", file=sys.stderr)
        return 2
    with socket.socket() as probe:
        probe.settimeout(0.2)
        if probe.connect_ex((args.host, args.port)) == 0:
            print(
                f"refusing: {args.host}:{args.port} is already in use", file=sys.stderr
            )
            return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    prompt = _prompt(args.prompt_characters)
    prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
    raw_arms = []
    for name, candidate in (("ordinary", False), ("candidate", True)):
        arm = _run_arm(args, name=name, candidate=candidate, prompt=prompt)
        raw_arms.append(arm)
        # Each arm must fully release before the next binds the same port.
        if not arm["shutdown"]["terminated"] or arm["shutdown"]["killed"]:
            break

    failures = [
        f"{arm['name']}: {failure}" for arm in raw_arms for failure in arm["failures"]
    ]
    if len(raw_arms) == 2:
        failures.extend(evaluate(raw_arms[0], raw_arms[1]))
    else:
        failures.append(
            "candidate arm was not launched after unclean ordinary shutdown"
        )
    for arm in raw_arms:
        shutdown = arm["shutdown"]
        if not shutdown["was_running"] or not shutdown["terminated"]:
            failures.append(f"{arm['name']}: service did not terminate cleanly")
        if shutdown["killed"]:
            failures.append(f"{arm['name']}: service required SIGKILL")

    result = {
        **plan,
        "status": "unqualified A/B serving smoke executed",
        "will_execute": True,
        "prompt_sha256": prompt_sha256,
        "prompt_characters_actual": len(prompt),
        "arms": [
            {
                "name": arm["name"],
                "candidate": arm["candidate"],
                "model_id": arm.get("model_id"),
                "http_status": arm.get("http_status"),
                "response": summarize_response(arm.get("response") or {}),
                "tokenizers_v1": (arm.get("status") or {}).get("tokenizers_v1"),
                "incremental_tokenizer_cache": (arm.get("status") or {}).get(
                    "incremental_tokenizer_cache"
                ),
                "shutdown": arm["shutdown"],
                "log": arm["log"],
                "server_log_tail": arm["server_log_tail"],
            }
            for arm in raw_arms
        ],
        "failures": failures,
        "go": not failures,
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "go": result["go"],
                "failures": result["failures"],
                "out": str(args.out),
            },
            indent=2,
        )
    )
    return 0 if result["go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
