"""Xing4.0 serving performance and mechanism evidence for one route.

Greedy, thinking off, fixed prompts; explicit neutral penalties so every
route decodes the same tokens.  Records TTFT for a long prompt, single-stream
and 4-way decode throughput, and the route's mechanism counters.
"""
from probe_common import *

NEUTRAL = {"repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}
essay = "Write a detailed technical essay about how compilers optimize loops, with numbered sections."
long_prompt = ("The quarterly logistics ledger lists shipments, depots, carriers and dates. " * 700) + \
    "\nSummarize the ledger in one sentence."


def stream_ttft(body):
    body = {**body, "stream": True}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if line.startswith(b"data: ") and b"content" in line:
                ttft = time.time() - started
                for _ in r:
                    pass
                return ttft
    return None


# warm-up (compile, allocator)
post(chat("Say hello.", max_tokens=16, **NEUTRAL))
ttfts = [stream_ttft(chat(long_prompt + f" Variant {i}.", max_tokens=8, **NEUTRAL)) for i in range(2)]
record("long_prompt_ttft", all(t is not None for t in ttfts),
       {"prompt_words": len(long_prompt.split()), "ttft_s": [round(t, 3) for t in ttfts]})

code, body, elapsed = post(chat(essay, max_tokens=512, **NEUTRAL))
tokens = body.get("usage", {}).get("completion_tokens", 0)
single = tokens / elapsed if elapsed else 0
record("single_stream", code == 200 and tokens > 100,
       {"tokens": tokens, "seconds": round(elapsed, 2), "tok_s": round(single, 1), "head": text(body)[:120]})

started = time.time()
many = parallel([chat(essay + f" Focus area {i}.", max_tokens=384, **NEUTRAL) for i in range(4)])
wall = time.time() - started
total = sum(b.get("usage", {}).get("completion_tokens", 0) for _c, b, _e in many)
record("four_way", all(c == 200 for c, _b, _e in many),
       {"tokens": total, "wall_s": round(wall, 2), "aggregate_tok_s": round(total / wall, 1)})

think_code, think_body, _ = post({**chat("What is 17 * 23?", max_tokens=600, **NEUTRAL),
                                  "enable_thinking": True, "reasoning_effort": "high"})
message = think_body.get("choices", [{}])[0].get("message", {})
record("thinking_channel", think_code == 200 and "391" in (message.get("content") or "")
       and bool(message.get("reasoning_content")),
       {"content": (message.get("content") or "")[:80], "reasoning_chars": len(message.get("reasoning_content") or "")})

st = status()
results["_mechanisms"] = {
    "ok": True,
    "profile": st.get("profile") or st.get("settings", {}).get("profile"),
    "int8_prefill": st.get("int8_prefill"),
    "scheduler": {k: v for k, v in st.get("scheduler", {}).items()
                  if any(s in k for s in ("mtp", "pld", "accept", "draft", "segmented"))},
    "diagnostics": st.get("diagnostics") or st.get("adapter"),
}
finish()
