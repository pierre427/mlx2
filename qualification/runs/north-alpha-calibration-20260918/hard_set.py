"""Harder multi-step problems: does always-on alpha steering cost accuracy?"""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE, OUT = sys.argv[1], sys.argv[2]
PROBLEMS = [
    ("A store sells pencils at 3 for 1.20 dollars. How much do 25 pencils cost in dollars? Reply with the number only.", "10"),
    ("The sum of three consecutive integers is 72. What is the largest of them? Reply with the number only.", "25"),
    ("A tank fills in 6 hours by pipe A and in 3 hours by pipe B. How many hours to fill it with both open? Reply with the number only.", "2"),
    ("What is the remainder when 2 to the power of 100 is divided by 7? Reply with the number only.", "2"),
    ("How many positive divisors does 360 have? Reply with the number only.", "24"),
    ("A rectangle has perimeter 50 and length 4 times its width. What is its area? Reply with the number only.", "100"),
    ("How many ways can 5 people be arranged in a row if two particular people must stand together? Reply with the number only.", "48"),
    ("What is the 10th term of the arithmetic sequence 7, 11, 15, ...? Reply with the number only.", "43"),
    ("If 3x + 7 = 2x + 19, what is x squared? Reply with the number only.", "144"),
    ("How many trailing zeros does 50 factorial have? Reply with the number only.", "12"),
    ("A clock shows 3:15. What is the smaller angle between the hands, in degrees? Reply with the number only.", "7.5"),
    ("What is the sum of all two-digit multiples of 7? Reply with the number only.", "728"),
    ("In how many years will 1000 dollars double at 10 percent simple interest? Reply with the number only.", "10"),
    ("What is the smallest positive integer divisible by 1 through 10? Reply with the number only.", "2520"),
    ("Write a Python function named is_palindrome that returns True when its string argument reads the same backwards, ignoring case. Code only.", "def is_palindrome"),
    ("How many squares of any size are on a standard 8 by 8 chessboard? Reply with the number only.", "204"),
]


def ask(prompt, extra):
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 4000, **extra}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        data = json.loads(r.read())
    guard = (data["mlx2"]["request_controls"] or {}).get("thinking_guard") or {}
    return {"content": (data["choices"][0]["message"].get("content") or "").strip(), "tokens": data["usage"]["completion_tokens"],
            "think": guard.get("think_tokens"), "tripped": guard.get("tripped"), "steered": (guard.get("steering") or {}).get("steered_steps")}

results = {}
for arm, extra in (("off", {"thinking_steer_alpha": 0, "thinking_budget": 0}), ("guard_only", {"thinking_steer_alpha": 0}), ("alpha_0.2", {}), ("alpha_0.4", {"thinking_steer_alpha": 0.4})):
    started = time.time()
    with ThreadPoolExecutor(max_workers=len(PROBLEMS)) as pool:
        rows = list(pool.map(lambda p: ask(p[0], extra), PROBLEMS))
    for (prompt, expected), row in zip(PROBLEMS, rows):
        row["correct"] = expected.replace(" ", "").lower() in row["content"].replace(" ", "").replace(",", "").lower()
    results[arm] = rows
    print(f"== {arm}: correct {sum(r['correct'] for r in rows)}/{len(rows)} tokens {sum(r['tokens'] for r in rows)} answered {sum(bool(r['content']) for r in rows)} wall {time.time()-started:.0f}s", flush=True)
    for (prompt, expected), row in zip(PROBLEMS, rows):
        print(f"   {'OK ' if row['correct'] else 'BAD'} tok={row['tokens']:5d} think={row['think']} trip={row['tripped']} steered={row['steered']} want={expected} | {row['content'][:40]!r}", flush=True)
json.dump(results, open(OUT, "w"), indent=1)
