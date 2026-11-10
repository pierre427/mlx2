#!/usr/bin/env python3
"""Drive the 72-prompt routing capture (9 categories x 8, last 2 held out) against the tapped server.

Writes replies.jsonl (prompt, reply, request time window in ns) next to the capture; resumable by id.
usage: routing_capture_driver.py PORT OUT_DIR [MAX_TOKENS]   (env SPLIT=train|heldout limits the prompts)
"""
import json, os, sys, time, urllib.request
sys.path.insert(0, "/Volumes/T7/kimi-k3.1-pro/expert-draft-linear-20261107")
from ed_prompts import CATS  # the fixed 72-prompt set from the Kimi expert-drafting study

port, out = int(sys.argv[1]), sys.argv[2]
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 320
base = f"http://127.0.0.1:{port}"
path = f"{out}/replies.jsonl"
done = {json.loads(l)["id"] for l in open(path)} if os.path.exists(path) else set()
model = json.loads(urllib.request.urlopen(base + "/v1/models").read())["data"][0]["id"]
for cat, prompts in CATS.items():
    for i, text in enumerate(prompts):
        pid = f"{cat}-{i}"
        if pid in done or (os.environ.get("SPLIT") and os.environ["SPLIT"] != ("heldout" if i >= 6 else "train")):
            continue
        body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "temperature": 0}
        t0 = time.time_ns()
        req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        resp = json.loads(urllib.request.urlopen(req, timeout=7200).read())
        t1 = time.time_ns()
        msg = resp["choices"][0]["message"]
        row = {"id": pid, "cat": cat, "split": "heldout" if i >= 6 else "train", "text": text,
               "reply": msg.get("content") or "", "reasoning": msg.get("reasoning_content"),
               "usage": resp.get("usage"), "finish": resp["choices"][0].get("finish_reason"), "t0_ns": t0, "t1_ns": t1,
               "mlx2": resp.get("mlx2")}
        with open(path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        u = resp.get("usage") or {}
        print(time.strftime("%H:%M:%S"), pid, u.get("prompt_tokens"), u.get("completion_tokens"), f"{(t1 - t0) / 1e9:.0f}s", flush=True)
print("CAPTURE DRIVER DONE", flush=True)
