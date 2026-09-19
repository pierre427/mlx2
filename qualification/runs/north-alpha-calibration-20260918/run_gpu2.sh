#!/bin/zsh
# Private port, own-PID cleanup only, and never overwrite someone else's GPU lock.
cd /private/tmp/mlx2-claude-alpha
PY=~/Desktop/mlx2/.venv/bin/python
D=qualification/runs/north-alpha-calibration-20260918
M=~/mlx-models/North-Mini-Code-1.0-mlx-4bit
PORT=8341
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
rm -f $D/done2.txt
if [ -e /tmp/gpu.lock ]; then echo "gpu lock held: $(cat /tmp/gpu.lock)" > $D/done2.txt; echo finished >> $D/done2.txt; exit 1; fi
if lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then echo "port $PORT busy" > $D/done2.txt; echo finished >> $D/done2.txt; exit 1; fi
echo "claude north alpha pid $$ $(date)" > /tmp/gpu.lock
$PY -u -m mlx2.server --model $M --host 127.0.0.1 --port $PORT --max-context 32768 --max-lanes 20 --max-inflight 40 --cache-bytes 8589934592 --cache-dir $D/cache --qualification-mode --thinking-budget 512 --thinking-steer-alpha 0.2 --ordinary > $D/server-ordinary.log 2>&1 &
SERVER=$!
until curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q '"ok"'; do sleep 3; kill -0 $SERVER 2>/dev/null || break; done
if kill -0 $SERVER 2>/dev/null; then
  $PY $D/hard_set.py http://127.0.0.1:$PORT $D/hard-set.json > $D/hard-set.log 2>&1
  $PY qualification/runs/sanity-20x20-20260918/sanity_20x20.py http://127.0.0.1:$PORT $D/sanity-ordinary-alpha.json > $D/sanity-ordinary-alpha.log 2>&1
  curl -s http://127.0.0.1:$PORT/v1/status > $D/status-ordinary.json
fi
kill -TERM $SERVER 2>/dev/null; for i in $(seq 1 30); do kill -0 $SERVER 2>/dev/null || break; sleep 1; done; kill -9 $SERVER 2>/dev/null
pkill -9 -P $SERVER 2>/dev/null
rm -f /tmp/gpu.lock; echo finished >> $D/done2.txt
