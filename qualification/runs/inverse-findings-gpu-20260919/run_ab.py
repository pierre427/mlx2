"""GPU A/B for the 2026-09-19 inverse-findings fixes (main 89817ef vs ffc6ceb).

One server at a time on a private port.  Each stage serves one model from one
tree (``new`` = the fixed tree, ``old`` = main before the fixes) and runs the
probes that exercise the changed code on that route:

* Flash-Next (Qwen XML via the shared OutputParser): named non-strict tool
  with a required argument, tool markup requested inside markdown code, and
  an ordinary auto tool call that must still parse.
* North (default-on thinking guard, PLD route so verify rows roll back):
  greedy guard parity old vs new, forced-trip budgets, a named non-strict
  tool, and a SIGTERM+SIGINT process-group stop during two live streams.
* Muse (ATEM): named non-strict tool with a required argument.

Usage: run_ab.py OLD_TREE NEW_TREE [stage ...]
"""
import json, os, signal, subprocess, sys, threading, time, urllib.request
from pathlib import Path

RUN = Path(__file__).resolve().parent
OLD, NEW = Path(sys.argv[1]), Path(sys.argv[2])
ONLY = set(sys.argv[3:])
PY = "~/Desktop/mlx2/.venv/bin/python"
MODELS = Path("~/mlx-models")
PORT = 8391
BASE = f"http://127.0.0.1:{PORT}"
ENV = {**os.environ, "PYTHONPATH": "src", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
LOCK = Path("/tmp/gpu.lock")
POLICIES = RUN / "policies"
POLICIES.mkdir(exist_ok=True)
FLASH_POLICY = {**json.loads((NEW / "qualification/policies/flash-next-mtp2.json").read_text()),
                "constrained_tool_grammar": True}
(POLICIES / "flash.json").write_text(json.dumps(FLASH_POLICY))
(POLICIES / "north-pld.json").write_text(json.dumps(
    {"prompt_lookup": {"num_draft": 8, "ngram_min": 3, "ngram_max": 6}, "constrained_tool_grammar": True}))
(POLICIES / "constrained.json").write_text(json.dumps({"constrained_tool_grammar": True}))

WEATHER = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city"]}}}]
NAMED = {"type": "function", "function": {"name": "get_weather"}}
NAMED_PROMPTS = ["Call the tool now.", "hello", "What's the weather like?",
                 "Use get_weather.", "What's the weather in Paris, in celsius?"]
FENCE_PROMPTS = [
    "Do NOT call any tool. For my documentation, reply with only a ```xml fenced code block "
    "that shows the exact raw <tool_call> markup you would emit to call get_weather for Paris.",
    "Do NOT call any tool. In one sentence, show the raw <tool_call> markup for get_weather "
    "with city Tokyo inside inline backticks, as documentation.",
    "Do not use tools. Explain your tool-call wire format: give a markdown code block containing a "
    "complete <tool_call>...</tool_call> example for get_weather(city=Oslo), then one sentence.",
]
PARITY_PROMPTS = [
    "What is the capital of Hungary? Answer in one word.",
    "Compute 17 * 23 and give only the number.",
    "Write a Python function that reverses a string. Code only.",
    "Is 221 prime? Answer yes or no with a one-line reason.",
    "List three primary colors, comma separated.",
    "A train leaves at 3pm and travels 2.5 hours. When does it arrive? Short answer.",
    "Name the largest planet in the solar system.",
    "What is the derivative of x^3 + 2x? Short answer.",
]


def post(path, body, timeout=900):
    request = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, {"error": error.read().decode(errors="replace")[:600]}


def chat(prompt, **extra):
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 512, **extra}
    status, data = post("/v1/chat/completions", body)
    if status != 200:
        return {"status": status, "error": data.get("error")}
    message = data["choices"][0]["message"]
    calls = [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
             for c in message.get("tool_calls") or []]
    receipt = data.get("mlx2") or {}
    return {"status": status, "content": message.get("content"), "reasoning": message.get("reasoning_content"),
            "tool_calls": calls, "finish": data["choices"][0]["finish_reason"],
            "completion_tokens": data["usage"]["completion_tokens"],
            "thinking_guard": (receipt.get("request_controls") or {}).get("thinking_guard")}


def named_required(**extra):
    rows = []
    for prompt in NAMED_PROMPTS:
        row = chat(prompt, tools=WEATHER, tool_choice=NAMED, **extra)
        args = {}
        if row.get("tool_calls"):
            try:
                args = json.loads(row["tool_calls"][0]["arguments"])
            except ValueError:
                args = {"<unparseable>": row["tool_calls"][0]["arguments"]}
        row["prompt"], row["has_required_city"] = prompt, bool(str(args.get("city", "")).strip())
        rows.append(row)
    return {"rows": rows, "with_city": sum(r["has_required_city"] for r in rows), "of": len(rows)}


def fence_probe(**extra):
    rows = []
    for prompt in FENCE_PROMPTS:
        row = chat(prompt, tools=WEATHER, tool_choice="auto", **extra)
        row["prompt"] = prompt
        row["markup_in_content"] = "<tool_call>" in (row.get("content") or "")
        rows.append(row)
    return {"rows": rows, "phantom_calls": sum(len(r.get("tool_calls") or []) for r in rows)}


def real_call(**extra):
    row = chat("What's the weather in Paris right now? Use the tool.", tools=WEATHER, tool_choice="auto", **extra)
    return {"row": row, "calls": len(row.get("tool_calls") or [])}


HARD_PROMPTS = [
    "How many positive integers below 1000 are divisible by 3 or 5 but not by 15? Show the count only.",
    "A bat and a ball cost $1.10 in total; the bat costs $1.00 more than the ball. Then a second bat costs twice the "
    "first ball plus 3 cents. What is the total cost of both bats and the ball in cents? Number only.",
    "Find the smallest positive integer n such that n^2 + n + 41 is composite. Number only.",
    "What is the 20th Fibonacci number if F1 = F2 = 1? Number only.",
]


def trip():
    rows = []
    for budget in (24, 64):
        for prompt in HARD_PROMPTS:
            rows.append(dict(chat(prompt, max_tokens=384, thinking_budget=budget), prompt=prompt, budget=budget))
    return {"rows": rows}


def parity():
    rows = [dict(chat(p, max_tokens=1024), prompt=p) for p in PARITY_PROMPTS]
    forced = [dict(chat(p, max_tokens=512, thinking_budget=48), prompt=p, budget=48)
              for p in PARITY_PROMPTS[:4]]
    return {"default_guard": rows, "budget_48": forced}


def signal_pair(server):
    """Two live streams, then SIGTERM and SIGINT to the process group together."""
    results = [{} for _ in range(2)]
    started = [threading.Event() for _ in range(2)]

    def stream(index):
        body = {"messages": [{"role": "user", "content": f"Count slowly from 1 to 60, one number per line. ({index})"}],
                "temperature": 0, "max_tokens": 400, "stream": True, "enable_thinking": False}
        request = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        chunks, done, finish, error = 0, False, None, None
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                for raw in response:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        done = True
                        break
                    event = json.loads(payload)
                    for choice in event.get("choices", []):
                        finish = choice.get("finish_reason") or finish
                    chunks += 1
                    if chunks == 3:
                        started[index].set()
        except Exception as exc:  # noqa: BLE001 - a cut stream is the thing we measure
            error = f"{type(exc).__name__}: {exc}"
        started[index].set()
        results[index] = {"chunks": chunks, "done": done, "finish_reason": finish, "error": error}

    threads = [threading.Thread(target=stream, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for event in started:
        event.wait(300)
    sent = time.time()
    os.killpg(server.pid, signal.SIGTERM)
    time.sleep(0.02)
    os.killpg(server.pid, signal.SIGINT)
    for thread in threads:
        thread.join(300)
    try:
        server.wait(120)
    except subprocess.TimeoutExpired:
        pass
    return {"streams": results, "server_exit_code": server.poll(),
            "server_exit_seconds": round(time.time() - sent, 2),
            "all_streams_completed": all(r.get("done") for r in results)}


def healthy():
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=3) as response:
            return json.loads(response.read()).get("status") == "ok"
    except Exception:
        return False


def stop(server):
    if server.poll() is None:
        try:
            os.killpg(server.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        server.wait(60)
    time.sleep(3)


def serve(name, tree, model, args):
    log = open(RUN / f"{name}-server.log", "w")
    command = [PY, "-u", "-m", "mlx2.server", "--model", str(MODELS / model), "--host", "127.0.0.1",
               "--port", str(PORT), "--cache-dir", str(RUN / "cache" / name), "--max-context", "32768",
               "--max-lanes", "4", "--max-inflight", "8", "--cache-bytes", str(8 << 30),
               "--qualification-mode", *args]
    server = subprocess.Popen(command, cwd=tree, env=ENV, stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=True)
    deadline = time.time() + 1800
    while time.time() < deadline and server.poll() is None and not healthy():
        time.sleep(3)
    return server, " ".join(command[3:])


FLASH = "Qwen3.8-Flash-Next-MLX-4bit-MTP"
NORTH = "North-Mini-Code-1.0-mlx-4bit"
MUSE = "Muse-Glimmer-30B-mlx-4bit"
STAGES = [
    ("flash-new", NEW, FLASH, ["--execution-policy", str(POLICIES / "flash.json")], ["named", "fence", "real"]),
    ("flash-old", OLD, FLASH, ["--execution-policy", str(POLICIES / "flash.json")], ["named", "fence", "real"]),
    ("north-new", NEW, NORTH, ["--prompt-lookup", "--execution-policy", str(POLICIES / "north-pld.json"),
                               "--drain-on-sigterm", "60"], ["parity", "named", "signal"]),
    ("north-old", OLD, NORTH, ["--prompt-lookup", "--execution-policy", str(POLICIES / "north-pld.json"),
                               "--drain-on-sigterm", "60"], ["parity", "named", "signal"]),
    ("muse-new", NEW, MUSE, ["--ordinary", "--execution-policy", str(POLICIES / "constrained.json")], ["named"]),
    ("muse-old", OLD, MUSE, ["--ordinary", "--execution-policy", str(POLICIES / "constrained.json")], ["named"]),
    ("north-new-a", NEW, NORTH, ["--prompt-lookup", "--execution-policy", str(POLICIES / "north-pld.json")],
     ["parity", "trip"]),
    ("north-new-b", NEW, NORTH, ["--prompt-lookup", "--execution-policy", str(POLICIES / "north-pld.json")],
     ["parity", "trip"]),
    ("north-old-a", OLD, NORTH, ["--prompt-lookup", "--execution-policy", str(POLICIES / "north-pld.json")],
     ["parity", "trip"]),
]


def main():
    if LOCK.exists():
        sys.exit(f"GPU lock held: {LOCK.read_text().strip()}")
    LOCK.write_text(f"claude mlx2 inverse-findings A/B pid {os.getpid()}\n")
    try:
        for name, tree, model, args, probes in STAGES:
            if ONLY and name not in ONLY:
                continue
            out = RUN / f"{name}.json"
            record = {"stage": name, "tree": str(tree), "started": time.strftime("%F %T")}
            record["head"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=tree,
                                            capture_output=True, text=True).stdout.strip()
            server, record["command"] = serve(name, tree, model, args)
            try:
                if not healthy():
                    record["failed"] = "server did not become healthy"
                    record["log_tail"] = (RUN / f"{name}-server.log").read_text()[-2000:]
                    continue
                extra = {"enable_thinking": False} if model != NORTH else {}
                for probe in probes:
                    print(name, probe, flush=True)
                    if probe == "named":
                        record["named"] = named_required(**extra)
                    elif probe == "fence":
                        record["fence"] = fence_probe(**extra)
                    elif probe == "real":
                        record["real"] = real_call(**extra)
                    elif probe == "trip":
                        record["trip"] = trip()
                    elif probe == "parity":
                        record["parity"] = parity()
                    elif probe == "signal":
                        record["signal"] = signal_pair(server)
                    out.write_text(json.dumps(record, indent=2))
            finally:
                stop(server)
                record["finished"] = time.strftime("%F %T")
                out.write_text(json.dumps(record, indent=2))
    finally:
        if LOCK.exists() and str(os.getpid()) in LOCK.read_text():
            LOCK.unlink()


if __name__ == "__main__":
    main()
