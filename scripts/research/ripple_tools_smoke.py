#!/usr/bin/env python3
"""Ripple live tools + reasoning smoke (one artifact, one loopback server).

Greedy requests against the no-flag route (or ``--server-arg`` overrides):
a tool-call turn non-streamed and streamed (parsed calls and content must
match), the follow-up turn after a ``tool`` result, a parallel-call prompt,
``tool_choice: "none"``, and a reasoning prompt (whether the model opens
``<think>`` and how the channels split). Records raw text via the token
trace. Requires the caller's CPG lease and paired locks; the server process
group is stopped in ``finally``. Sanity evidence, not qualification.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city"]}}}
TIME = {"type": "function", "function": {
    "name": "get_time", "description": "Current local time in a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpg-lock", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--port", type=int, default=18298)
    parser.add_argument("--server-arg", action="append", default=[])
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("refusing nonempty evidence directory")
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "scripts"))
    from qualification_gpu_ownership import require_qualification_lease
    from mlx2.adapters.registry import inspect_model

    owner = json.loads((args.cpg_lock / "owner.json").read_text())
    lease = require_qualification_lease(task_id=owner["cpg_task"], generation=owner["cpg_generation"],
                                        cpg_owner_lock=args.cpg_lock)
    resolution = inspect_model(args.model)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                         text=True, cwd=ROOT).strip())
    receipt = {"schema": "mlx2.ripple-tools-smoke.v1", "qualified": False,
               "checkpoint": f"{args.model.name}@sha256:{resolution.artifact_fingerprint}",
               "source_revision": {"head": head, "dirty_tracked": dirty}, "ownership": lease,
               "server_args": args.server_arg, "cases": {}}
    root = f"http://127.0.0.1:{args.port}"

    def post(path, payload, stream=False):
        data = json.dumps(payload).encode()
        request = Request(root + path, data=data, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=1800) as response:
            if not stream:
                return json.loads(response.read())
            content, reasoning, calls = "", "", {}
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    content += delta.get("content") or ""
                    reasoning += delta.get("reasoning_content") or delta.get("reasoning") or ""
                    for call in delta.get("tool_calls") or []:
                        slot = calls.setdefault(call.get("index", 0), {"name": "", "arguments": ""})
                        function = call.get("function") or {}
                        slot["name"] += function.get("name") or ""
                        slot["arguments"] += function.get("arguments") or ""
            return {"content": content, "reasoning": reasoning,
                    "tool_calls": [calls[k] for k in sorted(calls)]}

    def chat(messages, *, tools=None, stream=False, max_tokens=512, **extra):
        payload = {"model": args.model.name, "temperature": 0, "max_tokens": max_tokens,
                   "messages": messages, "stream": stream, **extra}
        if tools:
            payload["tools"] = tools
        if not stream:
            payload["return_token_trace"] = True
        result = post("/v1/chat/completions", payload, stream=stream)
        if stream:
            return result
        message = result["choices"][0]["message"]
        return {"content": message.get("content") or "",
                "reasoning": message.get("reasoning_content") or message.get("reasoning") or "",
                "tool_calls": [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                               for c in message.get("tool_calls") or []],
                "finish_reason": result["choices"][0].get("finish_reason"),
                "route": (result.get("mlx2") or {}).get("route"),
                "raw_text": ((result.get("mlx2") or {}).get("token_trace") or {}).get("text")}

    command = [args.python, "-u", "-m", "mlx2.server", "--model", str(args.model), "--host", "127.0.0.1",
               "--port", str(args.port), "--max-inflight", "1", "--max-lanes", "1", "--max-context", "8192",
               "--cache-bytes", str(1 << 30), "--cache-dir", str(args.output / "cache"), *args.server_arg]
    log = (args.output / "server.log").open("w")
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                              env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, cwd=ROOT)
    try:
        deadline = time.monotonic() + 900
        while True:
            if server.poll() is not None:
                raise RuntimeError("server exited before readiness")
            try:
                with urlopen(root + "/v1/status", timeout=10) as response:
                    if json.loads(response.read()).get("healthy"):
                        break
            except (URLError, TimeoutError, ConnectionError, OSError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("server readiness timed out")
            time.sleep(.5)
        ask = [{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}]
        cases = receipt["cases"]
        cases["tool_call"] = chat(ask, tools=[WEATHER])
        cases["tool_call_stream"] = chat(ask, tools=[WEATHER], stream=True)
        a, b = cases["tool_call"], cases["tool_call_stream"]
        cases["stream_matches"] = (a["content"] == b["content"] and a["reasoning"] == b["reasoning"]
                                   and [(c["name"], json.loads(c["arguments"])) for c in a["tool_calls"]]
                                   == [(c["name"], json.loads(c["arguments"])) for c in b["tool_calls"]])
        if a["tool_calls"]:
            call = a["tool_calls"][0]
            follow = ask + [{"role": "assistant", "content": a["content"] or None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": call["name"], "arguments": call["arguments"]}}]},
                {"role": "tool", "tool_call_id": "call_1",
                 "content": json.dumps({"city": "Paris", "temperature_c": 14, "conditions": "light rain"})}]
            cases["after_tool_result"] = chat(follow, tools=[WEATHER])
        cases["parallel"] = chat([{"role": "user", "content":
                                   "Get both the weather and the local time in Tokyo and in Lima. Call the tools."}],
                                  tools=[WEATHER, TIME])
        cases["tool_choice_none"] = chat(ask, tools=[WEATHER], tool_choice="none", max_tokens=128)
        reason = [{"role": "user", "content": "A train leaves at 9:40 and the trip takes 2 h 35 min. "
                                              "When does it arrive? Think it through, then answer."}]
        cases["reasoning"] = chat(reason, max_tokens=1024)
        cases["reasoning_stream"] = chat(reason, max_tokens=1024, stream=True)
        r, s = cases["reasoning"], cases["reasoning_stream"]
        cases["reasoning_stream_matches"] = r["content"] == s["content"] and r["reasoning"] == s["reasoning"]
        receipt["summary"] = {
            "tool_call_parsed": bool(a["tool_calls"]) and a["tool_calls"][0]["name"] == "get_weather",
            "stream_matches": cases["stream_matches"],
            "after_tool_result_mentions_rain": "rain" in (cases.get("after_tool_result") or {}).get(
                "content", "").lower(),
            "parallel_calls": len(cases["parallel"]["tool_calls"]),
            "tool_choice_none_no_calls": not cases["tool_choice_none"]["tool_calls"],
            "reasoning_channel_used": bool(r["reasoning"]),
            "reasoning_content_has_think_tag": "<think>" in r["content"] or "</think>" in r["content"],
            "reasoning_stream_matches": cases["reasoning_stream_matches"],
        }
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=10)
        log.close()
        receipt["server_stopped"] = server.poll() is not None
        tmp = args.output / "receipt.tmp"
        tmp.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
        os.replace(tmp, args.output / "receipt.json")
    print(json.dumps(receipt.get("summary")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
