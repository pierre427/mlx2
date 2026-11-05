#!/usr/bin/env python3
"""Agent-style Kimi tool-call smoke over one no-flag loopback server.

Declares tools the way Hermes does, checks the model calls the right
function with valid JSON arguments (non-streaming and streaming), returns a
tool result and checks the follow-up answer uses it, and checks
``tool_choice: "none"`` yields text. Caller owns the CPG lease and locks.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "search_files", "description": "Search the workspace for files matching a query.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpg-lock", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18294)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("refusing nonempty evidence directory")
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT/"scripts"))
    from qualification_gpu_ownership import require_qualification_lease
    owner = json.loads((args.cpg_lock/"owner.json").read_text())
    lease = require_qualification_lease(task_id=owner["cpg_task"], generation=owner["cpg_generation"],
                                        cpg_owner_lock=args.cpg_lock)
    from mlx2.adapters.kimi_linear import checkpoint_name
    receipt = {"checkpoint": checkpoint_name(args.model), "ownership": lease, "qualified": False,
               "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(),
               "kind": "agent-style tool-call smoke", "rows": []}
    root = f"http://127.0.0.1:{args.port}"

    def post(payload, stream=False):
        request = Request(root+"/v1/chat/completions", data=json.dumps({**payload, "stream": stream}).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=600) as response:
            if not stream:
                return json.load(response)
            calls, content = {}, ""
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                delta = json.loads(line[5:])["choices"][0].get("delta", {})
                content += delta.get("content") or ""
                for call in delta.get("tool_calls") or ():
                    entry = calls.setdefault(call["index"], {"name": "", "arguments": ""})
                    function = call.get("function", {})
                    entry["name"] += function.get("name") or ""
                    entry["arguments"] += function.get("arguments") or ""
            return {"content": content, "calls": [calls[i] for i in sorted(calls)]}

    env = {**os.environ, "PYTHONPATH": str(ROOT/"src")}
    server = subprocess.Popen([sys.executable, "-m", "mlx2.server", "--model", str(args.model), "--host", "127.0.0.1",
                               "--port", str(args.port), "--agent-compat", "on", "--cache-dir", str(args.output/"cache")],
                              stdout=(args.output/"server.log").open("w"), stderr=subprocess.STDOUT,
                              start_new_session=True, env=env, cwd=ROOT)
    try:
        deadline = time.monotonic()+300
        while True:
            try:
                with urlopen(root+"/v1/status", timeout=5) as r:
                    status = json.load(r)
                if status.get("ready") or status.get("state") == "ready":
                    break
            except OSError:
                pass
            if server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("server did not become ready")
            time.sleep(1)
        receipt["route"] = {k: status.get(k) for k in ("profile", "qualification", "route_receipt")}
        ask = {"model": args.model.name, "temperature": 0, "max_tokens": 256, "tools": TOOLS,
               "messages": [{"role": "user", "content": "What's the weather in Paris right now?"}]}
        first = post(ask)
        message = first["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        ok = (first["choices"][0].get("finish_reason") == "tool_calls" and len(calls) >= 1
              and calls[0]["function"]["name"] == "get_weather"
              and "paris" in json.loads(calls[0]["function"]["arguments"]).get("city", "").lower())
        receipt["rows"].append({"case": "call", "passed": ok, "message": message,
                                "finish_reason": first["choices"][0].get("finish_reason")})
        follow = {**ask, "messages": ask["messages"] + [
            {"role": "assistant", "content": message.get("content") or "", "tool_calls": calls},
            {"role": "tool", "tool_call_id": calls[0]["id"] if calls else "x",
             "content": json.dumps({"city": "Paris", "temp_c": 18, "sky": "sunny"})}]}
        second = post(follow)["choices"][0]["message"]
        receipt["rows"].append({"case": "use_result", "message": second,
                                "passed": "18" in (second.get("content") or "") and not second.get("tool_calls")})
        streamed = post({**ask, "messages": [{"role": "user", "content": "Find files about the KDA kernel in the workspace."}]},
                        stream=True)
        receipt["rows"].append({"case": "stream_call", "result": streamed, "passed": bool(streamed["calls"])
                                and streamed["calls"][0]["name"] == "search_files"
                                and isinstance(json.loads(streamed["calls"][0]["arguments"]), dict)
                                and "<|tool" not in streamed["content"]})
        plain = post({**ask, "tool_choice": "none"})["choices"][0]["message"]
        receipt["rows"].append({"case": "tool_choice_none", "message": plain,
                                "passed": not plain.get("tool_calls") and "<|tool" not in (plain.get("content") or "")})
        with urlopen(root+"/metrics", timeout=30) as r:
            (args.output/"metrics.prom").write_text(r.read().decode())
        receipt["passed"] = all(row["passed"] for row in receipt["rows"])
        for row in receipt["rows"]:
            print(json.dumps({"case": row["case"], "passed": row["passed"]}), flush=True)
    except BaseException as error:
        receipt["passed"] = False
        receipt["error"] = repr(error)
        raise
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
        (args.output/"receipt.json").write_text(json.dumps(receipt, indent=2, default=str)+"\n")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
