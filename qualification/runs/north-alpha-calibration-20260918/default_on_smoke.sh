#!/bin/zsh
# Default-on smoke: North with NO thinking flags must guard and steer; an explicit 0 must turn it off.
cd /private/tmp/mlx2-claude-northdefault
PY=~/Desktop/mlx2/.venv/bin/python
D=qualification/runs/north-alpha-calibration-20260918
PORT=8341
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
until [ ! -e /tmp/gpu.lock ]; do sleep 20; done
sleep 3; [ -e /tmp/gpu.lock ] && exec $0
lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && { echo "port busy" > $D/default-on.done; exit 1; }
echo "claude north default-on smoke pid $$ $(date)" > /tmp/gpu.lock
$PY scripts/qualify_serving.py --preflight-only --output $D/default-on-preflight.json > /dev/null 2>&1
$PY -u -m mlx2.server --model ~/mlx-models/North-Mini-Code-1.0-mlx-4bit --host 127.0.0.1 --port $PORT --max-context 32768 --max-lanes 4 --max-inflight 8 --cache-bytes 8589934592 --cache-dir $D/cache --qualification-mode --ordinary > $D/default-on-server.log 2>&1 &
SERVER=$!
until curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q '"ok"'; do sleep 3; kill -0 $SERVER 2>/dev/null || break; done
$PY - > $D/default-on-smoke.json <<PYEOF
import json, urllib.request
def post(extra):
    body={"messages":[{"role":"user","content":"What is the capital of Hungary? One word."}],"temperature":0,"max_tokens":2400,**extra}
    r=urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:$PORT/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=900)
    d=json.loads(r.read()); g=(d["mlx2"]["request_controls"] or {}).get("thinking_guard")
    return {"content":d["choices"][0]["message"].get("content"),"tokens":d["usage"]["completion_tokens"],"guard":g}
s=json.loads(urllib.request.urlopen("http://127.0.0.1:$PORT/v1/status").read())["settings"]
print(json.dumps({"settings":{k:s.get(k) for k in ("thinking_budget","thinking_steer","thinking_defaults_source")},
  "default":post({}),"request_off":post({"thinking_budget":0,"thinking_steer_alpha":0})},indent=1))
PYEOF
$PY scripts/qualify_serving.py --url http://127.0.0.1:$PORT --output $D/default-on-qualification.json --preflight-receipt $D/default-on-preflight.json --timeout 1800 > $D/default-on-qualifier.log 2>&1
echo "qualifier rc=$?" > $D/default-on.done
kill -TERM $SERVER 2>/dev/null; for i in $(seq 1 30); do kill -0 $SERVER 2>/dev/null || break; sleep 1; done; kill -9 $SERVER 2>/dev/null
rm -f /tmp/gpu.lock; echo finished >> $D/default-on.done
