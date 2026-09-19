#!/bin/bash
# 35B self-MTP A/B: main (base) vs claude/ecosystem-probes-20260919 (fix).
# Same driver (scripts/benchmark_serving.py from the fix tree), greedy, 160 tokens.
set -u
PY=~/Desktop/mlx2/.venv/bin/python
MODEL=~/mlx-models/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-oQ4e-mtp
OUT=${OUT:-.}
DRIVER=/private/tmp/mlx2-probes/scripts/benchmark_serving.py
PORT=8285
rc_all=0
for arm in base fix; do
  if [ $arm = base ]; then TREE=/private/tmp/mlx2-probes-base; else TREE=/private/tmp/mlx2-probes; fi
  echo "=== arm=$arm tree=$TREE $(git -C $TREE log --oneline -1)"
  PYTHONPATH=$TREE/src $PY -m mlx2.server --model "$MODEL" --port $PORT \
    --max-context 32768 --max-lanes 20 --max-inflight 24 --qualification-mode \
    > $OUT/server-35b-$arm.log 2>&1 &
  SPID=$!
  ready=0
  for i in $(seq 1 180); do
    if curl -sf http://127.0.0.1:$PORT/v1/status 2>/dev/null | $PY -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('state')=='ready' else 1)"; then ready=1; break; fi
    if ! kill -0 $SPID 2>/dev/null; then break; fi
    sleep 2
  done
  if [ $ready = 1 ]; then
    curl -s http://127.0.0.1:$PORT/v1/status > $OUT/status-35b-$arm.json
    PYTHONPATH=$TREE/src $PY $DRIVER --url http://127.0.0.1:$PORT --output $OUT/bench-35b-$arm.json \
      --rounds 3 --widths 1 2 4 --max-tokens 160 || rc_all=1
  else
    echo "server for $arm never became ready"; tail -30 $OUT/server-35b-$arm.log; rc_all=1
  fi
  kill -TERM $SPID 2>/dev/null; wait $SPID 2>/dev/null
  sleep 3
done
exit $rc_all
