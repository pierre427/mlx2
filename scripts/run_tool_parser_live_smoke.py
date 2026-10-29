#!/usr/bin/env python3
"""One-artifact live smoke for the token-bound output parser (tool calls/thinking).

One bounded ownership unit, run under the GPU locks + CPG lease: ordinary
reference smoke (scripts/smoke_local_model.py) first, then one loopback server
on the production default route, three cells with POST /v1/admin/apc/clear
before each, raw responses + status + route receipts retained, and the
server's whole process group stopped in ``finally``.  Refuses a nonempty
output directory so retries stay distinguishable.

Cells
  a_tool_needed      one tool offered; the answer needs it -> exactly one
                     parsed call with valid JSON arguments for that tool
  b_prose_tag        tools offered; the user asks for Qwen tool-call syntax
                     with the literal tag written in prose -> zero tool calls,
                     the tag stays in content
  c_thinking_split   enable_thinking with a short reasoning question ->
                     non-empty reasoning_content, content without think tags

This is feature smoke evidence, not qualification.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_single_model_smoke_ladder import atomic_json, stop_server, wait_health  # noqa: E402

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}

CELLS = {
    "a_tool_needed": {
        "messages": [{"role": "user", "content":
                      "What is the weather in Montreal right now, in celsius? "
                      "Use the available tool to find out."}],
        "tools": [WEATHER_TOOL],
        "enable_thinking": False,
        "max_tokens": 256,
    },
    "b_prose_tag": {
        "messages": [{"role": "user", "content":
                      "Do not call any tool for this message. Explain in two short "
                      "paragraphs how Qwen models format a tool call in their output. "
                      "In your explanation, write the literal opening tag <tool_call> "
                      "and the closing tag </tool_call> directly in a normal sentence, "
                      "not inside a code block or backticks, and say what goes "
                      "between them."}],
        "tools": [WEATHER_TOOL],
        "enable_thinking": False,
        "max_tokens": 400,
    },
    "c_thinking_split": {
        "messages": [{"role": "user", "content":
                      "A train leaves at 14:35 and the trip takes 2 hours 50 minutes. "
                      "At what time does it arrive? Answer with the time only."}],
        "enable_thinking": True,
        "max_tokens": 2048,
    },
}


def http(url, body=None, *, token=None, timeout=900):
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"} if data is not None else {}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, dict(response.headers), json.loads(response.read())
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            value = json.loads(raw)
        except ValueError:
            value = {"raw": raw.decode(errors="replace")}
        return error.code, dict(error.headers), value


def judge(name, response):
    choice = (response.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    calls = msg.get("tool_calls") or []
    checks = []

    def check(label, ok, detail=None):
        checks.append({"check": label, "passed": bool(ok), **({"detail": detail} if detail is not None else {})})

    if name == "a_tool_needed":
        check("exactly_one_tool_call", len(calls) == 1, len(calls))
        if len(calls) == 1:
            fn = calls[0].get("function") or {}
            check("tool_name", fn.get("name") == "get_weather", fn.get("name"))
            try:
                args = json.loads(fn.get("arguments") or "")
                ok = isinstance(args, dict) and isinstance(args.get("city"), str) and "montr" in args["city"].lower()
                check("arguments_valid_json_with_city", ok, args)
                if "unit" in args:
                    check("unit_in_enum", args["unit"] in ("celsius", "fahrenheit"), args["unit"])
            except ValueError as error:
                check("arguments_valid_json_with_city", False, str(error))
        check("finish_reason_tool_calls", choice.get("finish_reason") == "tool_calls", choice.get("finish_reason"))
        check("no_raw_tag_in_content", "<tool_call>" not in content, content[:200])
    elif name == "b_prose_tag":
        check("zero_tool_calls", len(calls) == 0, len(calls))
        check("tag_stays_in_content", "<tool_call>" in content, content[:300])
        check("finish_reason_not_tool_calls", choice.get("finish_reason") != "tool_calls", choice.get("finish_reason"))
    elif name == "c_thinking_split":
        check("reasoning_nonempty", bool(reasoning.strip()), len(reasoning))
        check("content_nonempty", bool(content.strip()), content[:200])
        check("no_think_tags_in_content", "<think>" not in content and "</think>" not in content, content[:200])
        check("no_think_tags_in_reasoning", "<think>" not in reasoning and "</think>" not in reasoning)
        check("answer_17_25", "17:25" in content or "5:25" in content, content[:200])
        check("zero_tool_calls", len(calls) == 0, len(calls))
    return {"passed": all(c["passed"] for c in checks), "checks": checks,
            "content": content, "reasoning_chars": len(reasoning), "tool_calls": calls,
            "finish_reason": choice.get("finish_reason")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8461)
    ap.add_argument("--token-dir", required=True, help="private dir for the 0600 admin token (not evidence)")
    ap.add_argument("--lock-dir", default="/tmp/mlx2-gpu-a.lock")
    ap.add_argument("--server-arg", action="append", default=[])
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing nonempty evidence dir {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir()
    from mlx2.adapters.registry import inspect_model

    resolution = inspect_model(a.model)
    checkpoint = f"{Path(a.model).name}@sha256:{resolution.artifact_fingerprint}"
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src"], text=True).strip()
    owner = json.loads((Path(a.lock_dir) / "owner.json").read_text())
    lease = f"{owner.get('cpg_task')}:{owner.get('cpg_generation')}"
    report = {"schema": "mlx2.tool-parser-live-smoke.v1", "checkpoint": checkpoint,
              "mlx2_commit": commit, "src_dirty": bool(dirty), "model_path": a.model,
              "kind": "feature smoke; not qualification", "lock_owner": owner,
              "started_at": time.time(), "cells": {}, "status": "running"}
    listing = subprocess.check_output(["ps", "-axo", "pid,ppid,%cpu,%mem,command"], text=True).splitlines()
    # Model-relevant processes only (foreign model servers, other GPU jobs).
    report["process_inventory"] = "\n".join(
        listing[:1] + [line for line in listing[1:] if "python" in line.lower() or "mlx" in line.lower()])
    report["vm_stat_before"] = subprocess.check_output(["vm_stat"], text=True)
    report["pmset_therm_before"] = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
    save = lambda: atomic_json(out / "unit.json", report)  # noqa: E731
    save()
    token_dir = Path(a.token_dir)
    token_dir.mkdir(parents=True, exist_ok=True)
    token_file = token_dir / f"admin-token-{os.getpid()}"
    token = secrets.token_hex(24)
    token_file.write_text(token)
    os.chmod(token_file, 0o600)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    server = None
    base = f"http://127.0.0.1:{a.port}"
    try:
        ref = subprocess.run(
            [sys.executable, str(ROOT / "scripts/smoke_local_model.py"), "--model", a.model,
             "--out", str(out / "ordinary-reference.json"), "--max-tokens", "16",
             "--i-own-gpu", "--cpg-lease", lease],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
        (out / "ordinary-reference.log").write_text(ref.stdout + ref.stderr)
        report["ordinary_reference_rc"] = ref.returncode
        save()
        if ref.returncode != 0:
            raise RuntimeError("ordinary-reference smoke failed")
        command = [sys.executable, "-m", "mlx2.server", "--model", a.model, "--host", "127.0.0.1",
                   "--port", str(a.port), "--fabric", "off", "--cache-dir", str(out / "cache"),
                   "--admin-token-file", str(token_file), *a.server_arg]
        report["server_command"] = command
        save()
        with (out / "server.log").open("w") as log:
            server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
        report["server_pid"] = server.pid
        status = wait_health(base, server, 900)
        atomic_json(out / "startup-status.json", status)
        binding = (status.get("execution") or {}).get("artifact_binding") or {}
        target = binding.get("target_fingerprint") or binding.get("fingerprint")
        report["artifact_binding"] = binding
        report["served_route"] = (status.get("settings") or {}).get("route")
        if target != resolution.artifact_fingerprint:
            raise RuntimeError(f"target artifact mismatch {target}")
        for name in a.cells:
            body = {"model": Path(a.model).name, "temperature": 0, **CELLS[name]}
            code, headers, cleared = http(base + "/v1/admin/apc/clear", {"reason": f"cell-{name}"}, token=token)
            started = time.time()
            code_r, headers_r, response = http(base + "/v1/chat/completions", body)
            elapsed = time.time() - started
            raw = {"request": body, "apc_clear": {"status": code, "response": cleared},
                   "status": code_r, "headers": headers_r, "response": response, "elapsed_s": elapsed}
            atomic_json(out / "raw" / f"{name}.json", raw)
            verdict = judge(name, response) if code_r == 200 else {"passed": False, "http_status": code_r}
            verdict["apc_cleared"] = code == 200
            report["cells"][name] = verdict
            save()
            print(name, json.dumps({k: v for k, v in verdict.items() if k in ("passed", "finish_reason", "reasoning_chars")}), flush=True)
        atomic_json(out / "status-after.json", http(base + "/v1/status")[2])
        report["status"] = "passed" if all(c.get("passed") for c in report["cells"].values()) else "failed"
    except BaseException as error:  # noqa: BLE001 - one receipt, always
        report["status"] = "error"
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], flush=True)
    finally:
        stop_server(server)
        report["server_stopped"] = server is None or server.poll() is not None
        report["finished_at"] = time.time()
        report["vm_stat_after"] = subprocess.check_output(["vm_stat"], text=True)
        report["pmset_therm_after"] = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
        try:
            token_file.unlink()
        except OSError:
            pass
        save()
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
