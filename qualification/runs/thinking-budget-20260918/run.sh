#!/bin/zsh
cd /private/tmp/mlx2-claude-budget
PY=~/Desktop/mlx2/.venv/bin/python
D=qualification/runs/thinking-budget-20260918
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "claude thinking-budget $(date)" > /tmp/gpu.lock
$PY scripts/qualify_serving.py --preflight-only --output $D/preflight.json > $D/preflight.log 2>&1
$PY -u -m mlx2.server --model ~/mlx-models/North-Mini-Code-1.0-mlx-4bit --host 127.0.0.1 --port 8297 --max-context 32768 --max-lanes 20 --max-inflight 40 --cache-bytes 8589934592 --cache-dir $D/cache --ordinary --qualification-mode > $D/server.log 2>&1 &
SERVER=$!
until curl -s -m 2 http://127.0.0.1:8297/health | grep -q '"ok"'; do sleep 3; kill -0 $SERVER 2>/dev/null || break; done
$PY qualification/runs/sanity-20x20-20260918/sanity_20x20.py http://127.0.0.1:8297 $D/north-sanity.json > $D/north-sanity.log 2>&1
echo "sanity rc=$?" >> $D/done.txt
$PY scripts/qualify_serving.py --url http://127.0.0.1:8297 --output $D/north-qualification.json --preflight-receipt $D/preflight.json --timeout 1800 > $D/north-qualifier.log 2>&1
echo "qualifier rc=$?" >> $D/done.txt
kill -TERM $SERVER; sleep 15; pkill -9 -f "mlx2.server.*--port 8297"
rm -f /tmp/gpu.lock
echo finished >> $D/done.txt
