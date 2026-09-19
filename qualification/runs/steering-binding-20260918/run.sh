#!/bin/zsh
# Private port, own-PID cleanup, never touches another session's GPU lock.
cd /private/tmp/mlx2-claude-calib
PY=~/Desktop/mlx2/.venv/bin/python
D=qualification/runs/steering-binding-20260918
M=~/mlx-models
PORT=8341
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
until [ ! -e /tmp/gpu.lock ]; do sleep 20; done
sleep 3; [ -e /tmp/gpu.lock ] && exec $0
lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && { echo "port busy" > $D/done.txt; exit 1; }
echo "claude steering-binding pid $$ $(date)" > /tmp/gpu.lock
rm -rf $D/cache-*; : > $D/results.jsonl
probe() {  # name, expect_start(yes|no), server args...
  local name=$1 expect=$2; shift 2
  local started=$(date +%s)
  $PY -u -m mlx2.server --host 127.0.0.1 --port $PORT --max-context 32768 --max-lanes 4 --max-inflight 8 --cache-bytes 4294967296 --qualification-mode "$@" > $D/$name-server.log 2>&1 &
  local SERVER=$!
  local up=no
  for i in $(seq 1 400); do
    if curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q '"ok"'; then up=yes; break; fi
    kill -0 $SERVER 2>/dev/null || break
    curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q '"error": "' && break
    sleep 3
  done
  local ready=$(( $(date +%s) - started ))
  NAME=$name EXPECT=$expect UP=$up READY=$ready PORT=$PORT LOG=$D/$name-server.log $PY - >> $D/results.jsonl <<'PYEOF'
import json, os, urllib.request
row = {"stage": os.environ["NAME"], "expected_start": os.environ["EXPECT"], "started": os.environ["UP"], "seconds_to_ready": int(os.environ["READY"])}
base = f"http://127.0.0.1:{os.environ['PORT']}"
if os.environ["UP"] == "yes":
    settings = json.loads(urllib.request.urlopen(base + "/v1/status", timeout=30).read())["settings"]
    row["thinking_steer"] = settings.get("thinking_steer"); row["thinking_budget"] = settings.get("thinking_budget")
    body = {"messages": [{"role": "user", "content": "What is the capital of Hungary? One word."}], "temperature": 0, "max_tokens": 2400}
    data = json.loads(urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=900).read())
    guard = (data["mlx2"]["request_controls"] or {}).get("thinking_guard") or {}
    row["answer"] = data["choices"][0]["message"].get("content"); row["tokens"] = data["usage"]["completion_tokens"]
    row["steered_steps"] = (guard.get("steering") or {}).get("steered_steps"); row["steer_source"] = (guard.get("steering") or {}).get("source")
else:
    try:
        row["health"] = json.loads(urllib.request.urlopen(base + "/health", timeout=5).read())
    except Exception as error:  # noqa: BLE001
        row["health"] = repr(error)[:200]
    tail = open(os.environ["LOG"]).read().strip().splitlines()[-3:]
    row["log_tail"] = [line[:240] for line in tail]
print(json.dumps(row))
PYEOF
  kill -TERM $SERVER 2>/dev/null; for i in $(seq 1 30); do kill -0 $SERVER 2>/dev/null || break; sleep 1; done; kill -9 $SERVER 2>/dev/null; sleep 2
}
probe north4-shipped yes --model $M/North-Mini-Code-1.0-mlx-4bit --ordinary --cache-dir $D/cache-a
probe north8-autocalibrate yes --model $M/North-Mini-Code-1.0-mlx-8bit --ordinary --cache-dir $D/cache-b
probe north8-stored yes --model $M/North-Mini-Code-1.0-mlx-8bit --ordinary --cache-dir $D/cache-b
probe north8-explicit-no-auto no --model $M/North-Mini-Code-1.0-mlx-8bit --ordinary --cache-dir $D/cache-c --no-thinking-auto-calibration --thinking-steer-alpha 0.2
probe qwen36-explicit-unsupported no --model $M/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-oQ4e-mtp --ordinary --cache-dir $D/cache-d --thinking-steer-alpha 0.2
cp $D/cache-b/commit-directions/*.json $D/north8-auto-calibration.json 2>/dev/null
rm -f /tmp/gpu.lock; echo finished > $D/done.txt
