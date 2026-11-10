#!/usr/bin/env python3
"""Native Gemma 4 on the default (prompt-lookup) route vs native --ordinary.

Two servers with ``MLX2_GEMMA4_BACKEND=native``: ``--ordinary``, then the
no-flag default route (latched adaptive prompt lookup).  Greedy copy-heavy
and prose prompts; texts compared, and the PLD server's status is kept so
verify engagement (accepted drafts) is read from receipts, not assumed.

Usage (under the shared GPU queue):
  PYTHONPATH=$PWD/src python scripts/research/gemma4_native_pld_check.py \
      --model ~/mlx-models/<artifact> --out qualification/runs/<dir>
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_rows_attention_ab as h  # noqa: E402
import gemma4_native_ab as ab  # noqa: E402

COPY_SOURCE = (Path(__file__).resolve().parents[2] / "src/mlx2/runtime/prefill_checkpoint_rows.py")


def _prompts():
    source = COPY_SOURCE.read_text()[:5000]
    return {
        "copy": "Repeat the following Python source exactly, changing nothing:\n\n" + source,
        "edit": "Rewrite this code with every docstring removed and nothing else changed:\n\n"
        + source,
        "prose": ab.SHORT,
    }


def _serve(model, route, port, token_file, workdir):
    log = open(workdir / f"server-{port}.log", "w")
    cmd = [sys.executable, "-m", "mlx2.server", *route, "--model", str(model),
           "--host", "127.0.0.1", "--port", str(port), "--max-lanes", "2",
           "--max-context", "16384", "--admin-token-file", str(token_file),
           "--cache-dir", str(workdir / f"cache-{port}")]
    env = dict(os.environ, MLX2_GEMMA4_BACKEND="native")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 1200
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited rc={proc.returncode}; see {log.name}")
            try:
                if h._get(base, "/health"):
                    h.MODEL_ID[0] = h._get(base, "/v1/models")["data"][0]["id"]
                    return proc, base
            except OSError:
                time.sleep(2)
        raise RuntimeError("server did not become healthy")
    except BaseException:
        h._stop(proc)
        raise


def _chat(base, prompt, max_tokens):
    import urllib.request

    body = {"model": h.MODEL_ID[0], "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "enable_thinking": False}
    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     method="POST", headers={"Content-Type": "application/json"})
    sent = time.perf_counter()
    with urllib.request.urlopen(request, timeout=3600) as response:
        payload = json.loads(response.read())
    payload["_wall_s"] = round(time.perf_counter() - sent, 3)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--port", type=int, default=8491)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty evidence dir {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="gemma4-native-pld-"))
    token = secrets.token_hex(16)
    token_file = workdir / "admin.token"
    token_file.write_text(token + "\n")
    token_file.chmod(0o600)
    prompts = _prompts()
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=h.ROOT, capture_output=True,
                              text=True).stdout.strip()
    results = {"model": str(args.model), "source_revision": revision, "backend": "native",
               "arms": {}}
    try:
        for index, (name, route) in enumerate((("ordinary", ["--ordinary"]), ("default", []))):
            proc, base = _serve(args.model, route, args.port + index, token_file, workdir)
            arm = {"route_flags": route, "responses": {}}
            try:
                status = json.dumps(h._get(base, "/v1/status", token))
                arm["profile_receipt"] = sorted(
                    p for p in ("gemma4-native-apcv2-ordinary-pld", "gemma4-native-apcv2-ordinary")
                    if p in status)
                if "gemma4-native-apcv2-ordinary" not in status:
                    raise RuntimeError("server is not on the native backend")
                for key, prompt in prompts.items():
                    h._post(base, "/v1/admin/apc/clear", {"reason": "gemma4-native-pld"}, token)
                    arm["responses"][key] = _chat(base, prompt, args.max_tokens)
                arm["final_status"] = h._get(base, "/v1/status", token)
            finally:
                h._stop(proc)
            results["arms"][name] = arm
    finally:
        (args.out / "results.json").write_text(json.dumps(results, indent=1))
    summary = {"model": results["model"], "source_revision": revision, "prompts": {}}
    for key in prompts:
        a = results["arms"]["ordinary"]["responses"][key]
        b = results["arms"]["default"]["responses"][key]
        summary["prompts"][key] = {
            "texts_equal": a["choices"][0]["message"]["content"]
            == b["choices"][0]["message"]["content"],
            "completion_tokens": [(a.get("usage") or {}).get("completion_tokens"),
                                  (b.get("usage") or {}).get("completion_tokens")],
            "wall_s": [a["_wall_s"], b["_wall_s"]],
        }
    summary["profiles"] = {k: v["profile_receipt"] for k, v in results["arms"].items()}
    summary["pld_receipt"] = h._find(results["arms"]["default"]["final_status"], "prompt_lookup")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
