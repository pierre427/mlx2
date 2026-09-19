#!/bin/bash
set -u
S=/private/tmp/claude-501/-Users-user-Desktop-mlx2/dcc2c62c-d339-4686-a019-625b816d27d9/scratchpad
PY=~/Desktop/mlx2/.venv/bin/python
OV=$S/tile8-overlay
TREE=/private/tmp/mlx2-probes-base
rc=0
(cd /private/tmp/mlx-4529/python/tests && MLX_ENABLE_TF32=0 PYTHONPATH=$OV $PY -m unittest test_quantized test_reduce 2>&1 | tail -3)
run_arm () { # model tag extra lanes widths
  PYTHONPATH=$3$TREE/src $PY -m mlx2.server --model $1 --port 8285 --max-context 32768 --max-lanes $4 \
    --max-inflight 12 --qualification-mode > $S/qmvm/server-$2.log 2>&1 &
  SPID=$!
  for i in $(seq 1 240); do
    curl -sf http://127.0.0.1:8285/v1/status 2>/dev/null | grep -q '"state": *"ready"' && break
    kill -0 $SPID 2>/dev/null || break; sleep 2
  done
  PYTHONPATH=$3$TREE/src $PY $TREE/scripts/benchmark_serving.py --output $S/qmvm/bench-$2.json \
    --rounds 3 --widths $5 --max-tokens 160 || rc=1
  kill -TERM $SPID; wait $SPID; sleep 3
}
M27=~/mlx-models/Qwen3.8-27B-oQ4e-mtp
M35=~/mlx-models/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-oQ4e-mtp
for rep in 1 2 3; do
  for arm in main new; do
    if [ $arm = new ]; then X=$OV:; else X=; fi
    echo "=== 27B $arm rep$rep"; run_arm $M27 nax-27b-$arm-$rep "$X" 8 "1 2 4"
  done
done
for rep in 1 2; do
  for arm in main new; do
    if [ $arm = new ]; then X=$OV:; else X=; fi
    echo "=== 35B $arm rep$rep"; run_arm $M35 nax-35b-$arm-$rep "$X" 8 "1 2 4"
  done
done
exit $rc
