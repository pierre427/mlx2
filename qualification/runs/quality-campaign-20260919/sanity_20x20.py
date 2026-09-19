"""20 rounds x 20 simultaneous requests with graded, varied tasks.

Viability/correctness, not performance: every response is graded for the task
and screened for degeneration, leaked control text and protocol errors.
"""
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict

BASE, OUT = sys.argv[1], sys.argv[2]
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
WIDTH = 20
DEFAULT_THINKING_ALLOWANCE_TOKENS = 2048
SERVER_MAX_TOKENS = 2_097_152


def _server_thinking_policy():
    try:
        with urllib.request.urlopen(BASE + "/v1/status", timeout=30) as response:
            status = json.loads(response.read())
        return (
            bool(status.get("thinking_default")),
            max(
                DEFAULT_THINKING_ALLOWANCE_TOKENS,
                int(status.get("thinking_allowance_tokens") or 0),
            ),
        )
    except Exception:  # noqa: BLE001 - an old server has no such field
        return False, DEFAULT_THINKING_ALLOWANCE_TOKENS

# A model that thinks by default (`/v1/status.thinking_default`, e.g. Xing) is
# tested the way it is served: reasoning left on, with a reasoning allowance on
# every budget.  SANITY_THINK=0/1 overrides the detection.
_think_override = {"0": False, "1": True}.get(
    os.environ.get("SANITY_THINK", ""), None
)
if _think_override is None:
    THINK, _thinking_allowance = _server_thinking_policy()
else:
    THINK = _think_override
    _thinking_allowance = DEFAULT_THINKING_ALLOWANCE_TOKENS
THINKING_BUDGET_TOKENS = _thinking_allowance if THINK else 0
NO_THINK = {} if THINK else {"enable_thinking": False, "reasoning_effort": "none"}
CAPITALS = [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Egypt", "Cairo"), ("Canada", "Ottawa"),
            ("Spain", "Madrid"), ("Germany", "Berlin"), ("Kenya", "Nairobi"), ("Peru", "Lima"), ("Norway", "Oslo"),
            ("Greece", "Athens"), ("Portugal", "Lisbon"), ("Austria", "Vienna"), ("Ireland", "Dublin"), ("Cuba", "Havana"),
            ("Poland", "Warsaw"), ("Sweden", "Stockholm"), ("Finland", "Helsinki"), ("Hungary", "Budapest"), ("Chile", "Santiago")]
WORDS = [("cat", "chat"), ("dog", "chien"), ("house", "maison"), ("water", "eau"), ("book", "livre")]
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
SCHEMA = {"type": "json_schema", "json_schema": {"name": "city", "strict": True, "schema": {
    "type": "object", "properties": {"city": {"type": "string"}, "population": {"type": "integer"}, "coastal": {"type": "boolean"}},
    "required": ["city", "population", "coastal"], "additionalProperties": False}}}


def with_thinking_budget(body):
    if not THINK or "max_tokens" not in body:
        return body
    budget = int(body["max_tokens"])
    return {
        **body,
        "max_tokens": max(
            budget,
            min(budget + THINKING_BUDGET_TOKENS, SERVER_MAX_TOKENS),
        ),
    }


def user(text, **extra):
    return with_thinking_budget({
        "messages": [{"role": "user", "content": text}],
        "temperature": 0,
        "max_tokens": 96,
        **NO_THINK,
        **extra,
    })


def has(*needles):
    return lambda r: all(n.lower() in r["content"].lower() for n in needles)


def parsed(r):
    try:
        return json.loads(r["content"])
    except (TypeError, ValueError):
        return None


def sampled_prose_grade(record):
    """Grade the requested six sentences, not an undocumented word quota."""
    text = record["content"]
    sentences = [sentence for sentence in re.split(r"[.!?]+", text) if sentence.strip()]
    return len(text.split()) >= 24 and len(sentences) >= 6


def tasks(rnd):
    a, b = 17 + rnd * 3, 25 + rnd * 7
    country, capital = CAPITALS[rnd % 20]
    country2, capital2 = CAPITALS[(rnd + 7) % 20]
    en, fr = WORDS[rnd % 5]
    code = f"ORCHID-{5000 + rnd * 37}"
    ledger = "".join(f"Ledger line {i}: account {(i * 13 + rnd) % 500} moved {(i * 17 + rnd) % 900} credits.\n" for i in range(90))
    needle = f"Run {rnd}.\n" + ledger[: len(ledger) // 2] + f"Important: the launch password is {code}.\n" + ledger[len(ledger) // 2:] + "What is the launch password? Reply with the password only."
    start = 3 + rnd
    return [
        ("arithmetic", user(f"What is {a} + {b}? Reply with the number only."), has(str(a + b))),
        ("arithmetic_mul", user(f"What is {rnd + 3} times 12? Reply with the number only."), has(str((rnd + 3) * 12))),
        ("capital", user(f"What is the capital of {country}? One word."), has(capital)),
        ("capital_sentence", user(f"Write four sentences about the capital city of {country2}, naming the city.", max_tokens=160), has(capital2)),
        ("sequence", user(f"Continue the sequence with the next number only: {start}, {start + 4}, {start + 8}, {start + 12},"), has(str(start + 16))),
        ("translate", user(f"Translate the English word '{en}' to French, then use the French word in two short French sentences.", max_tokens=120), has(fr)),
        ("code", user(f"Write a Python function named add_{rnd} that returns the sum of its two arguments. Include a docstring and two example calls in comments.", max_tokens=200),
         lambda r, rnd=rnd: f"def add_{rnd}" in r["content"] and "return" in r["content"]),
        ("list", user("List three primary colors and describe each one in a sentence.", max_tokens=160), lambda r: sum(c in r["content"].lower() for c in ("red", "blue", "yellow", "green")) >= 3),
        ("json_object", user(f"Return a JSON object with keys \"country\" and \"capital\" for {country}.", response_format={"type": "json_object"}),
         lambda r, capital=capital: isinstance(parsed(r), dict) and capital.lower() in json.dumps(parsed(r)).lower()),
        ("json_schema", user(f"Describe the city {capital2} as JSON.", response_format=SCHEMA),
         lambda r: isinstance(parsed(r), dict) and isinstance(parsed(r).get("population"), int) and isinstance(parsed(r).get("coastal"), bool)),
        ("stop_string", user("Count from 1 to 20, one number per line.", stop=["10"]),
         lambda r: r["finish"] == "stop" and "11" not in r["content"] and "9" in r["content"]),
        ("stream", user(f"What is {a} minus {rnd}? Reply with the number only.", stream=True), has(str(a - rnd))),
        ("tool_call", user(f"What is the weather in {capital} right now? Use the tool.", tools=TOOLS, max_tokens=128),
         lambda r, capital=capital: any(c.get("function", {}).get("name") == "get_weather" and capital.lower() in c.get("function", {}).get("arguments", "").lower() for c in r["tool_calls"])),
        ("system_multi_turn", with_thinking_budget({"messages": [{"role": "system", "content": "You are a terse assistant. Answer in as few words as possible."},
                                                                  {"role": "user", "content": f"Remember the number {a}."}, {"role": "assistant", "content": "Noted."},
                                                                  {"role": "user", "content": "What number did I ask you to remember?"}], "temperature": 0, "max_tokens": 48, **NO_THINK}), has(str(a))),
        ("needle", user(needle, max_tokens=32), has(code)),
        ("sampled_prose", user(f"Write six sentences about a lighthouse keeper named number {rnd}.", temperature=0.8, top_p=0.95, seed=rnd, max_tokens=200),
         # Yesterday's Muse prompt-lookup miss was a complete six-sentence
         # answer rejected only because it used fewer than 30 words.  Retain
         # the substantive instruction: six nonempty sentences and enough
         # prose to expose degeneration, without grading verbosity as quality.
         sampled_prose_grade),
        ("sampled_topk", user(f"Give three fun facts about the number {a}, one sentence each.", temperature=0.7, top_k=40, seed=rnd + 100, max_tokens=160), lambda r: len(r["content"].split()) >= 15),
        ("logprobs", user(f"Name one fruit that is {['red', 'yellow', 'green', 'orange'][rnd % 4]}. One word.", logprobs=True, top_logprobs=2, max_tokens=16),
         lambda r: bool(r["logprobs"]) and len(r["content"]) > 0),
        ("two_samples", user(f"Suggest a name for a boat, round {rnd}. Name only.", n=2, temperature=0.9, seed=rnd, max_tokens=24), lambda r: r["choices"] == 2),
        ("explain", user(f"Explain in five sentences what compiler optimization number {rnd + 1} of a typical -O2 pipeline might do.", max_tokens=200),
         lambda r: len(r["content"].split()) >= 30),
    ]


LEAKS = ("<|", "|>", "<think>", "</think>", "<tool_call>", "[PAD]", "<unk>", "�")


def screen(record, task):
    issues = []
    text = record["content"]
    if record["code"] != 200:
        return [f"http_{record['code']}"]
    if not text.strip() and not record["tool_calls"]:
        issues.append("empty")
    for marker in LEAKS:
        if marker in text and not (task == "tool_call" and marker == "<tool_call>"):
            issues.append("leak:" + marker)
    words = text.split()
    if len(words) >= 24:
        grams = Counter(tuple(words[i:i + 4]) for i in range(len(words) - 3))
        if grams and grams.most_common(1)[0][1] / max(len(words) - 3, 1) > 0.25:
            issues.append("repetition")
    if re.search(r"(.)\1{24,}", text):
        # Padding inside a complete, valid JSON document is cosmetic (North
        # pads values with the whitespace the grammar allows); anywhere else a
        # long run is degeneration.
        try:
            json.loads(text)
            issues.append("json_padding")
        except Exception:  # noqa: BLE001
            issues.append("char_run")
    if record["finish"] not in ("stop", "length", "tool_calls"):
        issues.append(f"finish:{record['finish']}")
    return issues


def call(body):
    """A 429 is the server's retryable back-pressure signal; behave like a client."""
    for attempt in range(4):
        record = call_once(body)
        record["retries_429"] = attempt
        if record["code"] != 429:
            break
        time.sleep(3)
    return record


def call_once(body):
    request = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    started = time.time()
    record = {"code": None, "content": "", "reasoning": "", "tool_calls": [], "finish": None, "logprobs": None, "choices": 0, "usage": {}, "mlx2": {}}
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            record["code"] = response.status
            if body.get("stream"):
                done = False
                for raw in response:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        done = True
                        break
                    event = json.loads(payload)
                    if event.get("error"):
                        record["code"] = 500
                        record["content"] = json.dumps(event["error"])
                        break
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {})
                        record["content"] += delta.get("content") or ""
                        record["reasoning"] += delta.get("reasoning_content") or ""
                        record["finish"] = choice.get("finish_reason") or record["finish"]
                    record["usage"] = event.get("usage") or record["usage"]
                    record["mlx2"] = event.get("mlx2") or record["mlx2"]
                if not done:
                    record["finish"] = "no_done_event"
                record["choices"] = 1
            else:
                data = json.loads(response.read())
                choices = data.get("choices", [])
                record["choices"] = len(choices)
                message = choices[0].get("message", {}) if choices else {}
                record["content"] = message.get("content") or ""
                record["reasoning"] = message.get("reasoning_content") or ""
                record["tool_calls"] = message.get("tool_calls") or []
                record["finish"] = choices[0].get("finish_reason") if choices else None
                record["logprobs"] = choices[0].get("logprobs") if choices else None
                record["usage"] = data.get("usage", {})
                record["mlx2"] = {k: data.get("mlx2", {}).get(k) for k in ("ordinary_compute_width", "cached_tokens", "ttft_seconds", "elapsed_seconds")}
                record["mlx2"]["speculation"] = bool(data.get("mlx2", {}).get("mtp") or data.get("mlx2", {}).get("speculation"))
    except urllib.error.HTTPError as error:
        record["code"] = error.code
        record["content"] = error.read().decode(errors="replace")[:400]
    except Exception as error:  # noqa: BLE001
        record["code"] = -1
        record["content"] = repr(error)[:400]
    record["seconds"] = round(time.time() - started, 2)
    return record


def status():
    with urllib.request.urlopen(BASE + "/v1/status", timeout=30) as response:
        return json.loads(response.read())


def main():
    before = status()
    records, rounds = [], []
    for rnd in range(ROUNDS):
        batch = tasks(rnd)
        assert len(batch) == WIDTH
        results = [None] * WIDTH
        def invoke(i, *, current_batch=batch, current_results=results):
            current_results[i] = call(current_batch[i][1])

        threads = [threading.Thread(target=invoke, args=(i,)) for i in range(WIDTH)]
        started = time.time()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        elapsed = time.time() - started
        tokens = 0
        for (name, _body, grade), record in zip(batch, results):
            try:
                correct = bool(record["code"] == 200 and grade(record))
            except Exception:  # noqa: BLE001
                correct = False
            issues = screen(record, name)
            tokens += int(record["usage"].get("completion_tokens") or 0)
            records.append({"round": rnd, "task": name, "correct": correct, "issues": issues, **record,
                            "content": record["content"][:600], "reasoning": record["reasoning"][:200],
                            "logprobs": bool(record["logprobs"])})
        rounds.append({"round": rnd, "seconds": round(elapsed, 2), "completion_tokens": tokens, "tokens_per_second": round(tokens / elapsed, 1),
                       "errors": sum(r["code"] != 200 for r in results)})
        print(f"round {rnd:02d} {elapsed:6.1f}s {tokens:5d} tok {tokens / elapsed:7.1f} tok/s errors={rounds[-1]['errors']}", flush=True)
    after = status()
    by_task = defaultdict(lambda: {"n": 0, "correct": 0, "issues": Counter()})
    for record in records:
        entry = by_task[record["task"]]
        entry["n"] += 1
        entry["correct"] += record["correct"]
        entry["issues"].update(record["issues"])
    summary = {task: {"n": e["n"], "correct": e["correct"], "issues": dict(e["issues"])} for task, e in by_task.items()}
    widths = Counter(r["mlx2"].get("ordinary_compute_width") for r in records if r["mlx2"])
    counts = {k: after["counts"].get(k, 0) - before["counts"].get(k, 0) for k in after["counts"] if k != "peak_observed_width"}
    report = {
        "schema": "mlx2.sanity-20x20.v1", "base": BASE, "thinking": THINK, "thinking_budget_tokens": THINKING_BUDGET_TOKENS if THINK else 0, "rounds": ROUNDS, "width": WIDTH,
        "settings": {k: after["settings"].get(k) for k in ("profile", "mtp", "speculation", "max_lanes", "max_context")},
        "runtime_source": after.get("runtime", {}).get("source_sha256"),
        "requests": len(records), "http_errors": sum(r["code"] != 200 for r in records),
        "graded_correct": sum(r["correct"] for r in records), "with_issues": sum(bool(r["issues"]) for r in records),
        "issue_totals": dict(Counter(i for r in records for i in r["issues"])),
        "retried_429": sum(r.get("retries_429", 0) for r in records),
        "by_task": summary, "rounds_detail": rounds, "observed_widths": {str(k): v for k, v in widths.items()},
        "counts_delta": {k: v for k, v in counts.items() if v}, "peak_observed_width": after["counts"].get("peak_observed_width"),
        "scheduler": after.get("scheduler"), "healthy": after["healthy"], "inflight": after["inflight"], "error": after.get("error"),
        "active_leases": after.get("apcv2", {}).get("cow", {}).get("active_leases"),
        "records": records,
    }
    with open(OUT, "w") as output:
        json.dump(report, output, indent=1, ensure_ascii=False)
    hard = report["http_errors"] + sum(v for k, v in report["issue_totals"].items() if k.startswith(("http", "empty", "leak", "char_run", "finish")))
    print(f"SUMMARY requests={len(records)} correct={report['graded_correct']} hard_failures={hard} issues={report['issue_totals']} healthy={after['healthy']}", flush=True)
    for task, entry in summary.items():
        print(f"  {task:20s} {entry['correct']:2d}/{entry['n']} {entry['issues'] or ''}", flush=True)
    sys.exit(0 if hard == 0 and after["healthy"] and after["inflight"] == 0 and report["graded_correct"] >= 0.85 * len(records) else 1)


if __name__ == "__main__":
    main()
