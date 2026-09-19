"""Is Xing4.0's reasoning long-but-healthy or pathological?

usage: thinking_length_study.py BASE_URL OUT.json

Thinking on, 8192-token budget, no thinking guard.  Per request: reasoning
tokens until ``</think>`` (id 10), whether it closed, the answer length and
finish reason, and repetition inside the reasoning: the share of 6-grams that
recur, the most repeated 6-gram, and the longest span that occurs twice.
"""
import json
import sys
import time
import urllib.request
from collections import Counter

BASE, OUT = sys.argv[1], sys.argv[2]
THINK_END = 10
PROMPTS = [
    ("qualifier:mixed_warm_compiler", "Explain how a compiler works, in numbered sections."),
    ("qualifier:mixed_warm_database", "Explain how a database transaction works, in numbered sections."),
    ("qualifier:sampling_controls", "Name three fruits."),
    ("qualifier:tools_plain", "Reply with exactly HERMES_READY"),
    ("arith", "What is 17 * 23? Answer briefly."),
    ("reasoning", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"),
    ("code", "Write a Python function that returns the n-th Fibonacci number iteratively."),
    ("open", "Give three tips for writing clear technical documentation."),
]
MODES = {
    "greedy": {"temperature": 0, "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0},
    "vendor": {"seed": 7},  # adapter defaults: temperature 1.0, top_p 0.95, repetition 1.05
}


def run(prompt, extra):
    body = {"messages": [{"role": "user", "content": prompt}], "enable_thinking": True,
            "max_tokens": 8192, "logprobs": True, **extra}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        data = json.load(r)
    return data, time.time() - start


def repetition(ids, n=6):
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    if not grams:
        return {"recurring_6gram_share": 0.0, "top_6gram_count": 0, "longest_repeat": 0}
    counts = Counter(grams)
    recurring = sum(1 for g in grams if counts[g] > 1) / len(grams)
    longest, seen = 0, {}
    for length in (8, 16, 32, 64, 128, 256):
        found = False
        seen = set()
        for i in range(len(ids) - length + 1):
            key = tuple(ids[i:i + length])
            if key in seen:
                found = True
                break
            seen.add(key)
        if found:
            longest = length
        else:
            break
    return {"recurring_6gram_share": round(recurring, 3), "top_6gram_count": counts.most_common(1)[0][1],
            "longest_repeat_at_least": longest}


results = []
for name, prompt in PROMPTS:
    for mode, extra in MODES.items():
        data, seconds = run(prompt, extra)
        choice = data["choices"][0]
        ids = [entry["id"] for entry in choice["logprobs"]["content"]]
        closed = THINK_END in ids
        cut = ids.index(THINK_END) if closed else len(ids)
        reasoning, answer = ids[:cut], ids[cut + 1:]
        row = {
            "prompt": name, "mode": mode, "seconds": round(seconds, 1),
            "completion_tokens": len(ids), "reasoning_tokens": len(reasoning), "closed": closed,
            "answer_tokens": len(answer), "finish": choice["finish_reason"],
            "answer_head": (choice["message"].get("content") or "")[:100],
            "reasoning_tail": (choice["message"].get("reasoning_content") or "")[-160:],
            **repetition(reasoning),
        }
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
json.dump(results, open(OUT, "w"), indent=2, ensure_ascii=False)
