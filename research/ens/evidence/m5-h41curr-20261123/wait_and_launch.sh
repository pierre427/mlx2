#!/bin/zsh
# Launch the slot once no other cpg_job is alive (the first attempt was refused: the lease was held by another job).
cd ~/Desktop/mlx2-wt-ens-a0
EV=$PWD/research/ens/evidence/m5-h41curr-20261123
until ! pgrep -f "results/cpg_job.py run" >/dev/null; do sleep 20; done
echo "launch $(date)" >> $EV/slot.log
CPG=~/Desktop/CPG-Radio/.venv/bin/python; REV=$(git rev-parse --short=9 HEAD)
$CPG ~/Desktop/mlx-uag/results/cpg_job.py run --label ens-h41-curr-$REV --out $EV/cpg.log --lease --lock --lock-dir /tmp/ens-h41-curr-$REV-cpg.lock --require-radio --heartbeat-s 60 -- \
  $CPG scripts/run_with_gpu_locks.py --session ens-m5-20261122 --label ens-h41-curr-$REV --cpg-owner-lock /tmp/ens-h41-curr-$REV-cpg.lock --cpg-task beff4e634ddb46978a84a1aed315f785 --receipt $EV/gpu-receipt.json -- \
  env PYTHONPATH=$PWD/src:$PWD HF_HUB_OFFLINE=1 zsh $EV/slot.sh > $EV/run.log 2>&1 < /dev/null
