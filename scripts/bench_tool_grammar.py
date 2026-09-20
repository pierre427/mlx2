"""Item 12 A/B probe: tool-call grammar during generation against a live server.

Run once against a server started without the item 12 policies (A) and once
with ``constrained_tool_grammar_auto`` / ``tool_grammar_streaming`` (B); compare
the JSON summaries.  Scenarios cover strict ``auto`` (a question that needs the
tool and one that does not), single-call ``auto``, tools plus a JSON answer,
``required``, and streamed Chat tool calls.  Every scenario records
HTTP status, whether the output honoured the contract, the engaged grammar
shape from the receipt, and wall/decode timing.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city", "unit"],
            "additionalProperties": False,
        },
    },
}
ANSWER = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
}
NEEDS_TOOL = "What is the weather in Toronto right now, in celsius? Use the tool."
NO_TOOL = "What is 17 + 25? Answer directly; no tool is needed."
MULTI = "Get the weather in Toronto and in Oslo, both in celsius."


def scenarios():
    base = {"tools": [WEATHER], "max_tokens": 512, "temperature": 0}
    yield "auto_needs_tool", {**base, "messages": [{"role": "user", "content": NEEDS_TOOL}]}, "call"
    yield "auto_no_tool", {**base, "messages": [{"role": "user", "content": NO_TOOL}]}, "text"
    yield "auto_single", {**base, "parallel_tool_calls": False, "messages": [{"role": "user", "content": MULTI}]}, "at_most_one"
    yield "auto_parallel", {**base, "messages": [{"role": "user", "content": MULTI}]}, "call"
    yield "required", {**base, "tool_choice": "required", "messages": [{"role": "user", "content": NO_TOOL}]}, "call"
    yield "answer_or_call_tool", {**base, "response_format": ANSWER, "messages": [{"role": "user", "content": NEEDS_TOOL}]}, "call_or_json"
    yield "answer_or_call_text", {**base, "response_format": ANSWER, "messages": [{"role": "user", "content": NO_TOOL}]}, "call_or_json"
    yield "stream_required", {**base, "stream": True, "tool_choice": "required", "messages": [{"role": "user", "content": NEEDS_TOOL}]}, "call"
    yield "stream_auto", {**base, "stream": True, "messages": [{"role": "user", "content": NEEDS_TOOL}]}, "call"


def _valid_call(call):
    try:
        arguments = json.loads(call["function"]["arguments"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return False
    return (
        call["function"].get("name") == "get_weather"
        and set(arguments) == {"city", "unit"}
        and isinstance(arguments["city"], str)
        and arguments["unit"] in {"celsius", "fahrenheit"}
    )


def _judge(expect, content, calls):
    if calls and not all(_valid_call(call) for call in calls):
        return False
    if expect == "call":
        return bool(calls)
    if expect == "text":
        return not calls and bool(content.strip()) and "<tool_call>" not in content
    if expect == "at_most_one":
        return len(calls) <= 1 and (calls or content.strip())
    if expect == "call_or_json":
        if calls:
            return True
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            return False
        return isinstance(value, dict) and set(value) == {"answer"}
    raise ValueError(expect)


def run(base_url, name, body, expect):
    started = time.perf_counter()
    request = Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    content, calls, receipt, first_call_at, chunks = "", [], {}, None, 0
    try:
        with urlopen(request, timeout=600) as response:
            status = response.status
            if body.get("stream"):
                for raw in response:
                    line = raw.decode().strip()
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    record = json.loads(line[6:])
                    if "error" in record:
                        return {"scenario": name, "status": status, "ok": False,
                                "error": record["error"]}
                    chunks += 1
                    choice = record["choices"][0]
                    delta = choice.get("delta", {})
                    content += delta.get("content") or ""
                    if delta.get("tool_calls") and first_call_at is None:
                        first_call_at = time.perf_counter() - started
                    calls.extend(delta.get("tool_calls") or ())
                    receipt = record.get("mlx2", receipt)
            else:
                data = json.load(response)
                message = data["choices"][0]["message"]
                content = message.get("content") or ""
                calls = message.get("tool_calls") or []
                receipt = data.get("mlx2", {})
    except HTTPError as error:
        return {"scenario": name, "status": error.code, "ok": False,
                "error": error.read().decode()[:400]}
    elapsed = time.perf_counter() - started
    controls = receipt.get("request_controls") or {}
    return {
        "scenario": name,
        "status": status,
        "ok": status == 200 and bool(_judge(expect, content, calls)),
        "calls": len(calls),
        "content_chars": len(content),
        "decode_grammar": (controls.get("tool_choice") or {}).get("decode_grammar"),
        "grammar": (controls.get("tool_choice") or {}).get("grammar"),
        "engine": (controls.get("structured_output") or {}).get("engine"),
        "completion_tokens": receipt.get("completion_tokens"),
        "elapsed_seconds": round(elapsed, 3),
        "first_call_seconds": None if first_call_at is None else round(first_call_at, 3),
        "stream_chunks": chunks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8285")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--label", default="run")
    parser.add_argument("--output")
    args = parser.parse_args()
    summary = [
        run(args.base, name, body, expect)
        for name, body, expect in scenarios()
        for _ in range(args.repeats)
    ]
    by_scenario = {}
    for row in summary:
        entry = by_scenario.setdefault(row["scenario"], {"ok": 0, "n": 0, "elapsed": []})
        entry["n"] += 1
        entry["ok"] += int(bool(row["ok"]))
        entry["elapsed"].append(row.get("elapsed_seconds") or 0.0)
    report = {
        "label": args.label,
        "scenarios": {
            name: {
                "ok": entry["ok"], "n": entry["n"],
                "median_elapsed_seconds": statistics.median(entry["elapsed"]),
            }
            for name, entry in by_scenario.items()
        },
        "rows": summary,
    }
    text = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
