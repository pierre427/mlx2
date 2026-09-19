#!/bin/zsh
cd /private/tmp/mlx2-claude-guard
PY=~/Desktop/mlx2/.venv/bin/python
D=qualification/runs/north-thinking-guard-20260918
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
rm -f $D/done2.txt; echo "claude north guard $(date)" > /tmp/gpu.lock
$PY -u -m mlx2.server --model ~/mlx-models/North-Mini-Code-1.0-mlx-4bit --host 127.0.0.1 --port 8297 --max-context 32768 --max-lanes 20 --max-inflight 40 --cache-bytes 8589934592 --cache-dir $D/cache --ordinary --qualification-mode --thinking-budget 512 > $D/server.log 2>&1 &
SERVER=$!
until curl -s -m 2 http://127.0.0.1:8297/health | grep -q '"ok"'; do sleep 3; kill -0 $SERVER 2>/dev/null || break; done
$PY qualification/runs/sanity-20x20-20260918/sanity_20x20.py http://127.0.0.1:8297 $D/north-sanity-guarded.json > $D/north-sanity-guarded.log 2>&1
kill -TERM $SERVER; sleep 15; pkill -9 -f "mlx2.server.*--port 8297"
rm -f /tmp/gpu.lock; echo finished > $D/done2.txt
