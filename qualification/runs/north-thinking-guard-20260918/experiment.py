"""North run-on reasoning: guard off vs thinking budgets (tau/alpha soft release + JUICE budget)."""
import json, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

BASE, OUT = sys.argv[1], sys.argv[2]
CASES = [  # (name, prompt, grader, kind)
    ("explain_9", "Explain in five sentences what compiler optimization number 9 of a typical -O2 pipeline might do.", lambda t: len(t.split()) >= 30, "run_on"),
    ("explain_13", "Explain in five sentences what compiler optimization number 13 of a typical -O2 pipeline might do.", lambda t: len(t.split()) >= 30, "run_on"),
    ("explain_15", "Explain in five sentences what compiler optimization number 15 of a typical -O2 pipeline might do.", lambda t: len(t.split()) >= 30, "run_on"),
    ("explain_18", "Explain in five sentences what compiler optimization number 18 of a typical -O2 pipeline might do.", lambda t: len(t.split()) >= 30, "run_on"),
    ("translate_house", "Translate the English word 'house' to French, then use the French word in two short French sentences.", lambda t: "maison" in t.lower(), "run_on"),
    ("capital_hungary", "What is the capital of Hungary? One word.", lambda t: "budapest" in t.lower(), "run_on"),
    ("add_62_130", "What is 62 + 130? Reply with the number only.", lambda t: "192" in t, "control"),
    ("add_44_88", "What is 44 + 88? Reply with the number only.", lambda t: "132" in t, "control"),
    ("paris_coastal", "Is Paris a coastal city? Answer yes or no.", lambda t: t.strip().lower().startswith("no"), "control"),
    ("primes_below_60", "How many prime numbers are there below 60? Reply with the number only.", lambda t: "17" in t, "hard"),
    ("digit_sum", "What is the sum of the digits of 2 to the power of 20? Reply with the number only.", lambda t: "31" in t, "hard"),   # 1048576 -> 31
    ("train_meet", "Two trains start 300 km apart and head toward each other at 70 km/h and 80 km/h. After how many hours do they meet? Reply with the number only.", lambda t: "2" in t, "hard"),
    ("sort_words", "Sort these words alphabetically and reply with them comma-separated: pear, apple, mango, fig, banana, cherry.", lambda t: t.lower().replace(" ", "").startswith("apple,banana,cherry,fig,mango,pear"), "hard"),
]
ARMS = [("off", None), ("budget_1024", 1024), ("budget_512", 512), ("budget_256", 256)]


def ask(prompt, budget):
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 2400}
    if budget:
        body["thinking_budget"] = budget
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()[:300]}
    message = data["choices"][0]["message"]
    return {"content": (message.get("content") or "").strip(), "reasoning_chars": len(message.get("reasoning_content") or ""),
            "reasoning_tail": (message.get("reasoning_content") or "")[-160:], "finish": data["choices"][0]["finish_reason"],
            "completion_tokens": data["usage"]["completion_tokens"], "seconds": round(time.time() - started, 1),
            "guard": (data.get("mlx2", {}).get("request_controls") or {}).get("thinking_guard")}

results = {}
for arm, budget in ARMS:
    with ThreadPoolExecutor(max_workers=len(CASES)) as pool:
        rows = list(pool.map(lambda case: ask(case[1], budget), CASES))
    for (name, _prompt, grade, kind), row in zip(CASES, rows):
        row["kind"] = kind
        row["answered"] = bool(row.get("content"))
        row["correct"] = bool(row.get("content")) and bool(grade(row["content"]))
        results.setdefault(arm, {})[name] = row
    done = results[arm]
    print(f"== {arm}: answered {sum(r['answered'] for r in done.values())}/{len(CASES)} correct {sum(r['correct'] for r in done.values())}/{len(CASES)} "
          f"tokens {sum(r.get('completion_tokens', 0) for r in done.values())}", flush=True)
    for name, r in done.items():
        g = r.get("guard") or {}
        print(f"   {name:16s} {r['kind']:8s} {'OK ' if r['correct'] else ('ans' if r['answered'] else '---')} tok={r.get('completion_tokens')} think={g.get('think_tokens')} "
              f"trip={g.get('tripped')}@{g.get('tripped_at')} forced={g.get('forced_close')} | {r.get('content', r.get('error', ''))[:70]!r}", flush=True)
json.dump(results, open(OUT, "w"), indent=1, ensure_ascii=False)
