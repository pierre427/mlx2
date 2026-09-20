#!/usr/bin/env python3
"""Optional GPU smoke: a real server under tenant auth still serves its route.

Tenant auth is HTTP control-plane code and is fully CPU-verified by
tests/test_tenant_auth.py; this smoke only confirms the flags compose with a
real model load.  It loads a model, so it refuses to run without
--i-own-the-gpu (use the cpg_job lock wrapper).  --dry-run prints the plan.

Go criteria (all must hold, else exit 1):
  1. tenant-a key -> 200 with X-MLX2-Tenant-Auth: api_key
  2. no credential -> 401
  3. spoofed X-Tenant-ID with tenant-b key -> 403
  4. /v1/status tenant_auth.verified.api_key >= 1 (mechanism counter)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlx2.tenant_auth import generate_api_key, hash_api_key  # noqa: E402


def _call(base, method, path, body=None, headers=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    try:
        with urlopen(
            Request(base + path, method=method, data=data, headers=request_headers),
            timeout=timeout,
        ) as response:
            return response.status, dict(response.headers), response.read()
    except HTTPError as error:
        return error.code, dict(error.headers), error.read()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--qualification")
    parser.add_argument("--port", type=int, default=8391)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    work = Path(tempfile.mkdtemp(prefix="mlx2-tenant-auth-smoke-"))
    key_a, key_b = generate_api_key(), generate_api_key()
    keys_file = work / "keys.json"
    keys_file.write_text(
        json.dumps(
            {
                "version": 1,
                "keys": [
                    {"key_id": "a", "tenant": "tenant-a", "sha256": hash_api_key(key_a)},
                    {"key_id": "b", "tenant": "tenant-b", "sha256": hash_api_key(key_b)},
                ],
            }
        )
    )
    os.chmod(keys_file, 0o600)
    command = [
        args.python, "-m", "mlx2.server",
        "--model", args.model,
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--tenant-scoped-cache",
        "--tenant-auth-keys-file", str(keys_file),
        *(["--qualification", args.qualification] if args.qualification else ["--qualification-mode"]),
    ]
    plan = {
        "command": command,
        "checks": ["key_a 200 + receipt", "no credential 401", "spoof 403", "verified>=1"],
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.i_own_the_gpu:
        parser.error("refusing to load a model without --i-own-the-gpu")

    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    server = subprocess.Popen(command, cwd=ROOT, env=env)
    base = f"http://127.0.0.1:{args.port}"
    results = {}
    try:
        deadline = time.monotonic() + 900
        while True:
            try:
                if _call(base, "GET", "/health", timeout=5)[0] == 200:
                    break
            except (URLError, ConnectionError, OSError):
                pass
            if server.poll() is not None or time.monotonic() > deadline:
                raise SystemExit("server did not become healthy")
            time.sleep(2)
        status, _, raw = _call(base, "GET", "/v1/models", headers={"x-api-key": key_a})
        model = json.loads(raw)["data"][0]["id"]
        chat = {
            "model": model,
            "messages": [{"role": "user", "content": "Say ok."}],
            "max_tokens": 8,
        }
        status, headers, _ = _call(
            base, "POST", "/v1/chat/completions", chat,
            {"Authorization": f"Bearer {key_a}"},
        )
        results["authenticated"] = [status, headers.get("X-MLX2-Tenant-Auth")]
        results["unauthenticated"] = _call(base, "POST", "/v1/chat/completions", chat)[0]
        results["spoofed"] = _call(
            base, "POST", "/v1/chat/completions", chat,
            {"x-api-key": key_b, "X-Tenant-ID": "tenant-a"},
        )[0]
        _, _, raw = _call(base, "GET", "/v1/status", headers={"x-api-key": key_a})
        results["tenant_auth"] = json.loads(raw)["tenant_auth"]
    finally:
        server.terminate()
        server.wait(timeout=120)
    go = (
        results["authenticated"] == [200, "api_key"]
        and results["unauthenticated"] == 401
        and results["spoofed"] == 403
        and results["tenant_auth"]["verified"]["api_key"] >= 1
    )
    results["go"] = go
    text = json.dumps(results, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
