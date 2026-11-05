#!/usr/bin/env python3
"""Ripple informal quality probe (one artifact, one loopback server).

Four groups of chat prompts with server-default sampling: world knowledge
(graded facts plus two fabrication traps), answer quality, scientific
reasoning/logic, and tool calling through a small agent loop whose tools
return canned results. Records every turn verbatim for a human read. Requires
the caller's CPG lease and paired locks; the server process group is stopped
in ``finally``. Informal evidence, not qualification.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]

KNOWLEDGE = [
    ("k_tungsten", "Which element has atomic number 74, and what physical property is it best known for?"),
    ("k_westphalia", "In what year was the Peace of Westphalia concluded, and which wars did it end?"),
    ("k_bulgakov", "Who wrote 'The Master and Margarita', and when was it first published?"),
    ("k_burkina", "What is the capital of Burkina Faso, and what was the country called before 1984?"),
    ("k_krebs", "Per molecule of acetyl-CoA, what does one turn of the citric acid cycle produce?"),
    ("k_fermat", "Who proved Fermat's Last Theorem, when, and what deeper result did the proof establish?"),
    ("k_chandrasekhar", "What is the Chandrasekhar limit, and roughly what is its value?"),
    ("k_hungarian", "Which language family does Hungarian belong to, and which living languages are its closest relatives?"),
    ("k_ramanujan", "Who brought Ramanujan to Cambridge, and why is 1729 famous?"),
    ("k_bach", "How many children did Johann Sebastian Bach have, and which of his sons became the best-known composers?"),
    ("k_cutoff", "What is the most recent year you have solid knowledge of? Name two notable events from that year."),
    ("k_trap_novel", "Summarize the plot of the 1987 novel 'The Glass Orchard' by Miriam Feld."),
    ("k_trap_einstein", "What did Einstein say about relativity in his 1921 Nobel lecture?"),
]

QUALITY = [
    ("q_crypto", "Explain to a smart 15-year-old how public-key cryptography works. Use one analogy and one tiny worked example with small numbers."),
    ("q_db", "A three-person startup is building a social app (profiles, posts, follows, feeds). Compare a relational database with a document store for this, then make a clear recommendation."),
    ("q_intervals", "Write a Python function that merges overlapping intervals. Include a few tests and state the time and space complexity."),
    ("q_fourday", "Give the strongest arguments for and against a four-day work week. Be balanced, then say what evidence would settle the question."),
]

SCIENCE = [
    ("s_pen", "A notebook and a pen cost $2.40 in total. The notebook costs $2.00 more than the pen. How much does the pen cost?"),
    ("s_knaves", "On an island, knights always tell the truth and knaves always lie. A says: 'B is a knave.' B says: 'A and C are the same type.' What is C? Show your reasoning."),
    ("s_tuesday", "I have two children. At least one of them is a boy who was born on a Tuesday. What is the probability that both children are boys? Assume sexes and weekdays are independent and uniform."),
    ("s_elevator", "A ball is released from rest 1.2 m above the floor of an elevator that is accelerating upward at 2.0 m/s^2. Taking g = 9.8 m/s^2, how long until it hits the elevator floor?"),
    ("s_propane", "Balance the combustion of propane and compute how many grams of CO2 are produced by burning 22 g of propane completely."),
    ("s_syllogism", "All bloops are razzies. Some razzies are lazzies. Does it follow that some bloops are lazzies? Explain."),
    ("s_heartbeats", "Estimate the number of heartbeats in an average human lifetime. Show your assumptions and arithmetic."),
    ("s_coffee", "My one houseplant grew faster after I started watering it with leftover coffee. Design a proper experiment to test whether coffee affects plant growth, and list the confounders it controls for."),
    ("s_sequence", "What comes next: 2, 6, 12, 20, 30, ? Explain the rule."),
    ("s_mislabeled", "Three boxes are labelled Apples, Oranges and Mixed, and every label is wrong. You may take one fruit from one box without looking inside. How do you relabel all three boxes correctly?"),
]


def fn(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required}}}


TOOLS = [
    fn("get_weather", "Current weather for a city.",
       {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["city"]),
    fn("convert_currency", "Convert an amount between ISO 4217 currencies at today's rate.",
       {"amount": {"type": "number"}, "from": {"type": "string"}, "to": {"type": "string"}}, ["amount", "from", "to"]),
    fn("calculator", "Evaluate an arithmetic expression.", {"expression": {"type": "string"}}, ["expression"]),
    fn("search_flights", "Search flights. Dates are YYYY-MM-DD.",
       {"origin": {"type": "string", "description": "IATA code"}, "destination": {"type": "string", "description": "IATA code"},
        "date": {"type": "string"}}, ["origin", "destination", "date"]),
    fn("create_event", "Create a calendar event.",
       {"title": {"type": "string"}, "start": {"type": "string", "description": "ISO 8601 local time"},
        "duration_minutes": {"type": "integer"},
        "attendees": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "email": {"type": "string"}}, "required": ["name", "email"]}}},
       ["title", "start", "duration_minutes", "attendees"]),
]

SYSTEM = "You are a helpful assistant with tools. Today is Thursday 2026-11-05; the user is in Montreal."

TOOL_CASES = [
    ("t_parallel", "What's the weather in Tokyo and in Osaka right now?"),
    ("t_chain", "I have 250 euros. How many Japanese yen is that, and how much per day if I spread it over 6 days?"),
    ("t_no_tool", "What is the capital of Italy?"),
    ("t_flight", "Find me flights from Montreal to Paris for next Friday."),
    ("t_error", "What's the weather in Atlantis?"),
    ("t_nested", "Schedule a 30-minute sync tomorrow at 10am called 'Q4 plan' with Ana (ana@example.com) and Bo (bo@example.org)."),
]


def run_tool(name, arguments):
    if name == "get_weather":
        city = str(arguments.get("city", "")).lower()
        table = {"tokyo": {"temp_c": 14, "conditions": "light rain"}, "osaka": {"temp_c": 17, "conditions": "cloudy"},
                 "paris": {"temp_c": 9, "conditions": "clear"}}
        for key, value in table.items():
            if key in city:
                return {"city": arguments["city"], **value}
        return {"error": f"unknown city: {arguments.get('city')}"}
    if name == "convert_currency":
        rates = {("EUR", "JPY"): 162.4, ("USD", "JPY"): 151.2, ("CAD", "EUR"): 0.66}
        key = (str(arguments.get("from", "")).upper(), str(arguments.get("to", "")).upper())
        if key not in rates:
            return {"error": f"no rate for {key[0]}->{key[1]}"}
        return {"amount": round(float(arguments["amount"]) * rates[key], 2), "currency": key[1], "rate": rates[key]}
    if name == "calculator":
        expression = str(arguments.get("expression", ""))
        if not set(expression) <= set("0123456789.+-*/() "):
            return {"error": "unsupported expression"}
        return {"result": eval(expression, {"__builtins__": {}})}  # noqa: S307 - digits/operators only
    if name == "search_flights":
        return {"flights": [
            {"flight": "AC870", "depart": f"{arguments.get('date')}T18:30", "arrive": "+1 07:35", "price_cad": 1142},
            {"flight": "AF347", "depart": f"{arguments.get('date')}T21:05", "arrive": "+1 10:10", "price_cad": 998}]}
    if name == "create_event":
        return {"ok": True, "event_id": "evt_7f3a"}
    return {"error": f"unknown tool {name}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpg-lock", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--port", type=int, default=18301)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--server-arg", action="append", default=[])
    parser.add_argument("--only", default="", help="comma-separated case ids (default: all)")
    parser.add_argument("--temperature", type=float, action="append", default=[],
                        help="repeat each case at these temperatures (default: server sampling defaults)")
    args = parser.parse_args()
    only = {item for item in args.only.split(",") if item}
    temperatures = args.temperature or [None]
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
    receipt = {"schema": "mlx2.ripple-quality-probe.v1", "qualified": False,
               "checkpoint": f"{args.model.name}@sha256:{resolution.artifact_fingerprint}",
               "source_revision": {"head": head, "dirty_tracked": dirty}, "ownership": lease,
               "server_args": args.server_arg, "max_tokens": args.max_tokens,
               "only": sorted(only), "temperatures": temperatures,
               "sampling": "server defaults, temperature overridden where listed",
               "cases": []}
    root = f"http://127.0.0.1:{args.port}"

    def save():
        tmp = args.output / "receipt.json.tmp"
        tmp.write_text(json.dumps(receipt, indent=2, default=str, ensure_ascii=False) + "\n")
        os.replace(tmp, args.output / "receipt.json")

    def chat(messages, tools=None, temperature=None):
        payload = {"model": args.model.name, "max_tokens": args.max_tokens, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = tools
        request = Request(root + "/v1/chat/completions", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        started = time.monotonic()
        try:
            with urlopen(request, timeout=3600) as response:
                result = json.loads(response.read())
        except HTTPError as error:
            return {"http_error": error.code, "body": error.read().decode(errors="replace")}, time.monotonic() - started
        return result, time.monotonic() - started

    def turn_record(result, seconds):
        if "http_error" in result:
            return {"error": result, "seconds": round(seconds, 1)}
        choice = result["choices"][0]
        message = choice["message"]
        receipt_ = result.get("mlx2") or {}
        return {"finish_reason": choice.get("finish_reason"), "seconds": round(seconds, 1),
                "completion_tokens": (result.get("usage") or {}).get("completion_tokens"),
                "sampling": receipt_.get("effective_sampling"), "sampling_defaults": receipt_.get("sampling_defaults"),
                "route": receipt_.get("route"),
                "reasoning": message.get("reasoning_content") or "", "content": message.get("content") or "",
                "tool_calls": [{"id": c.get("id"), "name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                               for c in message.get("tool_calls") or []]}

    command = [args.python, "-u", "-m", "mlx2.server", "--model", str(args.model), "--host", "127.0.0.1",
               "--port", str(args.port), "--max-inflight", "1", "--max-lanes", "1", "--max-context", "16384",
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
            except (URLError, OSError, ValueError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("server not ready")
            time.sleep(2)
        for group, cases in (("knowledge", KNOWLEDGE), ("quality", QUALITY), ("science", SCIENCE)):
            for case_id, prompt in cases:
                if only and case_id not in only:
                    continue
                for temperature in temperatures:
                    result, seconds = chat([{"role": "user", "content": prompt}], temperature=temperature)
                    receipt["cases"].append({"group": group, "id": case_id, "prompt": prompt,
                                             "temperature": temperature,
                                             "turns": [turn_record(result, seconds)]})
                    save()
        for case_id, prompt in TOOL_CASES:
            if only and case_id not in only:
                continue
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
            turns, executed = [], []
            for _ in range(5):
                result, seconds = chat(messages, TOOLS)
                record = turn_record(result, seconds)
                turns.append(record)
                if "error" in record or not record["tool_calls"]:
                    break
                messages.append({"role": "assistant", "content": record["content"] or None,
                                 "tool_calls": [{"id": c["id"], "type": "function",
                                                 "function": {"name": c["name"], "arguments": c["arguments"]}}
                                                for c in record["tool_calls"]]})
                for call in record["tool_calls"]:
                    try:
                        arguments = json.loads(call["arguments"])
                    except ValueError:
                        arguments = {}
                    output = run_tool(call["name"], arguments)
                    executed.append({"name": call["name"], "arguments": arguments, "output": output})
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(output)})
            receipt["cases"].append({"group": "tools", "id": case_id, "prompt": prompt, "turns": turns,
                                     "executed": executed})
            save()
    except BaseException as error:
        receipt["error"] = repr(error)
        raise
    finally:
        try:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        except ProcessLookupError:
            pass
        log.close()
        save()


if __name__ == "__main__":
    main()
