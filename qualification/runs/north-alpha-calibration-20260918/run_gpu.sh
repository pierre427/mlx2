#!/bin/zsh
cd /private/tmp/mlx2-claude-alpha
PY=~/Desktop/mlx2/.venv/bin/python
D=qualification/runs/north-alpha-calibration-20260918
M=~/mlx-models/North-Mini-Code-1.0-mlx-4bit
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
rm -f $D/done.txt; echo "claude north alpha $(date)" > /tmp/gpu.lock
serve() { $PY -u -m mlx2.server --model $M --host 127.0.0.1 --port 8297 --max-context 32768 --max-lanes 20 --max-inflight 40 --cache-bytes 8589934592 --cache-dir $D/cache --qualification-mode --thinking-budget 512 --thinking-steer-alpha 0.2 "$@" > $D/server-$NAME.log 2>&1 & SERVER=$!; until curl -s -m 2 http://127.0.0.1:8297/health | grep -q '"ok"'; do sleep 3; kill -0 $SERVER 2>/dev/null || break; done }
stop() { kill -TERM $SERVER; sleep 12; pkill -9 -f "mlx2.server.*--port 8297"; sleep 2 }
NAME=ordinary; serve --ordinary
$PY $D/hard_set.py http://127.0.0.1:8297 $D/hard-set.json > $D/hard-set.log 2>&1
$PY qualification/runs/sanity-20x20-20260918/sanity_20x20.py http://127.0.0.1:8297 $D/sanity-ordinary-alpha.json > $D/sanity-ordinary-alpha.log 2>&1
curl -s http://127.0.0.1:8297/v1/status > $D/status-ordinary.json
stop
NAME=pld; serve --prompt-lookup --execution-policy /private/tmp/mlx2-claude-alpha/qualification/runs/integration-gpu-20260918/policies/pld.json
$PY qualification/runs/sanity-20x20-20260918/sanity_20x20.py http://127.0.0.1:8297 $D/sanity-pld-alpha.json > $D/sanity-pld-alpha.log 2>&1
stop
rm -f /tmp/gpu.lock; echo finished > $D/done.txt
