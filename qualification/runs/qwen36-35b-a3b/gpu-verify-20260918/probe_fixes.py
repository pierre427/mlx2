"""Drive the live mlx2 server through each hardening fix and record receipts."""
import json
import socket
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8297"
OUT = sys.argv[2] if len(sys.argv) > 2 else "probe-results.json"
results = {}


def post(body, path="/v1/chat/completions", timeout=900):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def health():
    with urllib.request.urlopen(BASE + "/health", timeout=10) as r:
        return json.loads(r.read())


def status():
    with urllib.request.urlopen(BASE + "/v1/status", timeout=10) as r:
        return json.loads(r.read())


def chat(content, **extra):
    return {"messages": [{"role": "user", "content": content}], "temperature": 0, **extra}


def record(name, ok, detail):
    results[name] = {"ok": bool(ok), **detail}
    print(f"{'PASS' if ok else 'FAIL'} {name}: {json.dumps(detail)[:300]}", flush=True)


assert health()["status"] == "ok"
counts0 = status()["counts"]

# 1. n=3 fanout with max_tokens >= 256: siblings must not overflow; all complete.
t = time.time()
code, body = post(chat("Write a numbered list of 40 short facts about the ocean.", n=3, max_tokens=320, temperature=0.7, seed=11))
elapsed = time.time() - t
choices = body.get("choices", [])
usages = [s["completion_tokens"] for s in body.get("mlx2", {}).get("samples", [])]
counts = status()["counts"]
record("fanout_n3", code == 200 and len(choices) == 3 and all(c["finish_reason"] in ("length", "stop") for c in choices)
       and counts.get("cancelled", 0) == counts0.get("cancelled", 0),
       {"code": code, "choices": len(choices), "completion_tokens": usages, "elapsed_s": round(elapsed, 1),
        "fanout_boundaries": counts.get("apcv2_fanout_boundaries"), "cohort_attach_failures": counts.get("batch_cohort_attachment_failures", 0),
        "widths": [s.get("ordinary_compute_width") for s in body.get("mlx2", {}).get("samples", [])]})

# 2. structured output json_object (thinking off) completes and parses.
t = time.time()
code, body = post(chat("Return a JSON object with keys name (string) and year (integer) describing the Eiffel Tower.",
                       response_format={"type": "json_object"}, max_tokens=120, enable_thinking=False))
content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
try:
    parsed = json.loads(content)
except Exception:
    parsed = None
record("json_object", code == 200 and isinstance(parsed, dict), {"code": code, "content": content[:200], "elapsed_s": round(time.time() - t, 1),
                                                                 "receipt_structured": body.get("mlx2", {}).get("request_controls", {}).get("structured_output")})

# 3. strict json_schema with optional properties (the new chain construction).
schema = {"type": "json_schema", "json_schema": {"name": "x", "strict": True, "schema": {
    "type": "object", "properties": {"city": {"type": "string"}, "country": {"type": "string"}, "population": {"type": "integer"}},
    "required": ["city"], "additionalProperties": False}}}
t = time.time()
code, body = post(chat("Describe Paris as JSON.", response_format=schema, max_tokens=80, enable_thinking=False))
content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
try:
    parsed = json.loads(content)
except Exception:
    parsed = None
record("json_schema_optional_chain", code == 200 and isinstance(parsed, dict) and "city" in parsed,
       {"code": code, "content": content[:200], "elapsed_s": round(time.time() - t, 1)})

# 4. thinking + structured -> 400, not silent misplacement.
code, body = post(chat("Return JSON.", response_format={"type": "json_object"}, reasoning_effort="high", max_tokens=50))
record("thinking_plus_structured_400", code == 400, {"code": code, "error": body.get("error", {}).get("message")})

# 5. grammar dead end: pieces admitted then no continuation -> 502 for this request; server survives.
code, body = post(chat("Say yes.", grammar="yesno", max_tokens=8, enable_thinking=False))
alive = health()["status"] == "ok"
record("grammar_dead_end_502_alive", code in (502, 200) and alive, {"code": code, "error": body.get("error", {}).get("message"), "alive": alive,
                                                                     "structured_failures": status()["counts"].get("structured_output_failures", 0)})

# 6. ReDoS-shaped grammar: must fail closed (502) or complete, never kill the worker or exceed budget by much.
t = time.time()
code, body = post(chat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", grammar="(a|aa)+b", max_tokens=48, enable_thinking=False, logit_bias={}))
elapsed = time.time() - t
alive = health()["status"] == "ok"
record("grammar_redos_alive", alive and code in (502, 200), {"code": code, "elapsed_s": round(elapsed, 1), "alive": alive,
                                                            "error": body.get("error", {}).get("message"),
                                                            "content": body.get("choices", [{}])[0].get("message", {}).get("content", "")[:80]})

# 7. top_k == vocab_size -> 400, server alive.
vocab = None
code, body = post(chat("hi", top_k=248320, temperature=0.7, max_tokens=4))
alive = health()["status"] == "ok"
record("top_k_eq_vocab_400", code == 400 and alive, {"code": code, "error": body.get("error", {}).get("message"), "alive": alive})

# 8. sub-float32 temperature -> 400; tiny finite temperature accepted.
code, body = post(chat("hi", temperature=1e-40, max_tokens=4))
code2, body2 = post(chat("Say hello.", temperature=1e-5, max_tokens=8))
record("temperature_floor", code == 400 and code2 == 200, {"reject": code, "accept": code2, "content": body2.get("choices", [{}])[0].get("message", {}).get("content", "")[:60]})

# 9. min_tokens + structured -> 400.
code, body = post(chat("Return JSON.", response_format={"type": "json_object"}, min_tokens=4, max_tokens=50, enable_thinking=False))
record("min_tokens_structured_400", code == 400, {"code": code})

# 10. stop string inside a tool call -> stop, not 502.
tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
code, body = post(chat("Call get_weather for Paris.", tools=tools, stop=["<parameter"], max_tokens=64, enable_thinking=False))
record("stop_inside_tool_call", code == 200, {"code": code, "finish": body.get("choices", [{}])[0].get("finish_reason"), "error": body.get("error", {}).get("message")})

# 11. slowloris: trickled body gets 408 within the deadline while others are served.
def trickle():
    s = socket.create_connection(("127.0.0.1", int(BASE.rsplit(":", 1)[1])))
    s.settimeout(60)
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: 4000\r\n\r\n{")
    t0 = time.time()
    data = b""
    try:
        while b"\r\n\r\n" not in data:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
    except socket.timeout:
        pass
    s.close()
    return data.split(b"\r\n", 1)[0].decode(errors="replace"), round(time.time() - t0, 1)

with ThreadPoolExecutor(max_workers=2) as pool:
    fut = pool.submit(trickle)
    time.sleep(1)
    code, body = post(chat("Say hi.", max_tokens=8, enable_thinking=False))
    line, waited = fut.result()
record("slowloris_408", "408" in line and code == 200 and waited < 35, {"status_line": line, "waited_s": waited, "concurrent_request": code})

# 12. warm APC hit still works (LRU change) and cached_tokens reported.
prompt = "Explain prefix caching in two sentences. " * 4
code1, b1 = post(chat(prompt, max_tokens=24, enable_thinking=False))
code2, b2 = post(chat(prompt, max_tokens=24, enable_thinking=False))
record("apc_warm_hit", code2 == 200 and b2.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0) > 0,
       {"cold_cached": b1.get("usage", {}).get("prompt_tokens_details"), "warm_cached": b2.get("usage", {}).get("prompt_tokens_details"),
        "mtp": (b2.get("mlx2", {}).get("mtp") or {}).get("accepted_tokens") if isinstance(b2.get("mlx2", {}).get("mtp"), dict) else b2.get("mlx2", {}).get("mtp")})

# 13. server health/counters at the end.
final = status()
record("server_alive_final", final["healthy"] and final["inflight"] == 0, {"inflight": final["inflight"], "counts": {k: v for k, v in final["counts"].items() if k in (
    "completed", "cancelled", "structured_output_failures", "mtp_sidecar_missing_misses", "apcv2_fanout_boundaries", "apcv2_store_failures", "memory_pressure_evictions")}})

json.dump(results, open(OUT, "w"), indent=2)
print("\nSUMMARY:", sum(r["ok"] for r in results.values()), "/", len(results), "passed")
