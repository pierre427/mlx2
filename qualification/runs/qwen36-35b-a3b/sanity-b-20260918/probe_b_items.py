"""Live sanity probes for the CPU-only ("B") items against a running mlx2 server."""
import json
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = sys.argv[1]
OUT = sys.argv[2]
results = {}


def post(body, timeout=1800):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), time.time() - t
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), time.time() - t


def status():
    with urllib.request.urlopen(BASE + "/v1/status", timeout=10) as r:
        return json.loads(r.read())


def chat(content, **extra):
    return {"messages": [{"role": "user", "content": content}], "enable_thinking": False, **extra}


def record(name, ok, detail):
    results[name] = {"ok": bool(ok), **detail}
    print(f"{'PASS' if ok else 'FAIL'} {name}: {json.dumps(detail, default=str)[:400]}", flush=True)


def seg_stats():
    ex = status().get("execution", {}) or {}
    for key in ("segmented_self_mtp", "segmented", "segmented_mtp"):
        if isinstance(ex.get(key), dict):
            return ex[key]
    return {k: v for k, v in ex.items() if "segment" in str(k)}


s0 = status()
counts0 = dict(s0["counts"])
seg0 = seg_stats()

# --- 1. Promotion-join: long B2 cohort, then a short (<=16 token) arrival mid-decode.
long_prompt = "Write a very long, detailed essay about the history of the Roman Empire, at least 800 words."
outcomes = {}


def run(name, body, timeout=1800):
    outcomes[name] = post(body, timeout)


threads = [threading.Thread(target=run, args=(f"long{i}", chat(long_prompt, max_tokens=400, temperature=0.7, seed=100 + i)))
           for i in range(2)]
for th in threads:
    th.start()
time.sleep(6)  # cohort has decoded for a while -> promoted physical batch
run("short", chat("Say exactly: OK", max_tokens=8, temperature=0))
for th in threads:
    th.join()
seg1 = seg_stats()
reconciled = (seg1.get("physical_join_flag_reconciled", 0) or 0) - (seg0.get("physical_join_flag_reconciled", 0) or 0)
codes = {k: v[0] for k, v in outcomes.items()}
record("promotion_join_short_arrival", all(c == 200 for c in codes.values()) and status()["healthy"],
       {"codes": codes, "short_elapsed_s": round(outcomes["short"][2], 1),
        "short_content": outcomes["short"][1].get("choices", [{}])[0].get("message", {}).get("content", "")[:40],
        "physical_join_flag_reconciled_delta": reconciled,
        "widths": [outcomes[k][1].get("mlx2", {}).get("ordinary_compute_width") for k in outcomes],
        "mtp": [(outcomes[k][1].get("mlx2", {}).get("mtp") or {}).get("accepted_tokens") for k in outcomes]})

# --- 2. Scheduler-waiting watchdog: one request decoding > 60 s, late arrival must not get the memory 429.
outcomes = {}
t_long = threading.Thread(target=run, args=("very_long", chat("Write an extremely long story about a lighthouse keeper. Keep going for as long as possible.", max_tokens=3000, temperature=0.7, seed=7)))
t_long.start()
time.sleep(8)
run("late", chat("Reply with the single word: ready", max_tokens=6, temperature=0))
t_long.join()
late_code, late_body, late_elapsed = outcomes["late"]
record("watchdog_late_arrival_not_429", late_code == 200 and outcomes["very_long"][0] == 200,
       {"late_code": late_code, "late_elapsed_s": round(late_elapsed, 1), "late_error": late_body.get("error", {}).get("message"),
        "very_long_code": outcomes["very_long"][0], "very_long_tokens": outcomes["very_long"][1].get("usage", {}).get("completion_tokens"),
        "very_long_elapsed_s": round(outcomes["very_long"][2], 1),
        "memory_admission_timeouts_delta": status()["counts"].get("memory_admission_timeouts", 0) - counts0.get("memory_admission_timeouts", 0)})

# --- 3. Cache pressure: many distinct ~3K-token prompts against a 1 GiB resident cap; then warm hits.
prompts = []
for i in range(14):
    filler = (f"Document {i}: " + "The quick brown fox jumps over the lazy dog number %d. " % i) * 120
    prompts.append(filler + " Summarize in one sentence.")
apc0 = status()["apcv2"]
for i, p in enumerate(prompts):
    code, body, _ = post(chat(p, max_tokens=6, temperature=0))
    assert code == 200, (i, code, body)
apc1 = status()["apcv2"]
# warm repeats of the first three (likely spilled) and the last two (resident)
warm = []
for p in prompts[:3] + prompts[-2:]:
    code, body, el = post(chat(p, max_tokens=6, temperature=0))
    warm.append((code, body.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0), round(el, 2)))
apc2 = status()["apcv2"]
disk = apc2.get("idle_disk", {})
record("cache_pressure_spill_restore", all(c == 200 for c, _, _ in warm) and all(ct > 0 for _, ct, _ in warm) and status()["healthy"],
       {"warm": warm, "resident_bytes_gib": round(apc2.get("nbytes", 0) / 2**30, 3), "entries": apc2.get("entries") or apc2.get("size"),
        "idle_disk": {k: disk.get(k) for k in ("idle_spills", "pressure_spills", "restores", "restore_failures", "restore_budget_deferrals", "disk_entries", "disk_evictions")},
        "memory_pressure_evictions_delta": status()["counts"].get("memory_pressure_evictions", 0) - counts0.get("memory_pressure_evictions", 0),
        "apcv2_store_failures": status()["counts"].get("apcv2_store_failures", 0),
        "persistent_blocks": {k: v for k, v in apc2.items() if "block" in str(k)}})

# --- 4. Host prompt cache: repeated identical request -> hits; oversize/disabled counters present.
hp0 = status()["host_prompt_cache"]
for _ in range(3):
    post(chat("What is 2+2?", max_tokens=4, temperature=0))
hp1 = status()["host_prompt_cache"]
record("host_prompt_cache", hp1.get("hits", 0) - hp0.get("hits", 0) >= 2 and "oversize_skips" in hp1,
       {"before": hp0, "after": hp1})

# --- 5. Fairness counters under mixed load (long prefill + short decodes).
big = ("Context paragraph. " * 1400) + " Answer with one word: done."
outcomes = {}
ths = [threading.Thread(target=run, args=(f"prefill{i}", chat(big + f" ({i})", max_tokens=8, temperature=0))) for i in range(2)]
ths += [threading.Thread(target=run, args=(f"decode{i}", chat("Count from one to thirty in words.", max_tokens=120, temperature=0.7, seed=i))) for i in range(2)]
for th in ths:
    th.start()
for th in ths:
    th.join()
sched = status().get("scheduler", {})
fair = {k: v for k, v in sched.items() if "fair" in str(k) or "debt" in str(k) or "contended" in str(k) or "prefill_cap" in str(k)}
record("decode_time_fairness_mixed_load", all(v[0] == 200 for v in outcomes.values()) and bool(fair),
       {"codes": {k: v[0] for k, v in outcomes.items()}, "fairness_counters": fair,
        "elapsed": {k: round(v[2], 1) for k, v in outcomes.items()}})

final = status()
record("server_alive_final", final["healthy"] and final["inflight"] == 0,
       {"inflight": final["inflight"], "counts_delta": {k: final["counts"].get(k, 0) - counts0.get(k, 0) for k in
        ("completed", "cancelled", "memory_admission_timeouts", "memory_pressure_evictions", "apcv2_store_failures", "structured_output_failures", "mtp_sidecar_missing_misses")}})
json.dump(results, open(OUT, "w"), indent=2, default=str)
print("\nSUMMARY:", sum(r["ok"] for r in results.values()), "/", len(results), "passed")
