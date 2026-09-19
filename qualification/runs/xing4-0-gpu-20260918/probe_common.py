import json, sys, time, threading, urllib.error, urllib.request

BASE = sys.argv[1]
OUT = sys.argv[2]
results = {}


def post(body, timeout=1800):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), time.time() - started
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), time.time() - started


def status():
    with urllib.request.urlopen(BASE + "/v1/status", timeout=30) as r:
        return json.loads(r.read())


def chat(content, **extra):
    return {"messages": [{"role": "user", "content": content}], "temperature": 0,
            "enable_thinking": False, "reasoning_effort": "none", "max_tokens": 48, **extra}


def text(body):
    return (body.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()


def record(name, ok, detail):
    results[name] = {"ok": bool(ok), **detail}
    print(("PASS " if ok else "FAIL ") + name + " " + json.dumps(detail, ensure_ascii=False, default=str)[:500], flush=True)


def parallel(bodies):
    out = [None] * len(bodies)

    def run(i):
        out[i] = post(bodies[i])

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(bodies))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def finish():
    st = status()
    record("server_alive_final", st["healthy"] and st["inflight"] == 0,
           {"inflight": st["inflight"], "error": st.get("error")})
    json.dump(results, open(OUT, "w"), indent=2, ensure_ascii=False, default=str)
    passed = sum(r["ok"] for r in results.values())
    print(f"SUMMARY {passed}/{len(results)}", flush=True)
    sys.exit(0 if passed == len(results) else 1)
