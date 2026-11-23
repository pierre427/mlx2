#!/bin/zsh
# One M5 slot (coordinator 2026-11-22): tile-trainer retest, then seed-4 C6 and C7. Self-capped at 145 min.
cd "$(dirname "$0")/../../../.."
E=research/ens/evidence/m5-seed4-20261122
( sleep 8700; echo "watchdog: 145 min cap reached, stopping" >> $E/slot.log; pkill -TERM -f "m5-seed4-20261122/train_seed4.sh"; pkill -TERM -f "research.ens.train_core --arm c[67] --seed 4" ) &
WD=$!
PY=~/Desktop/mlx2/.venv/bin/python
echo "retest start $(date)" >> $E/slot.log
$PY research/ens/evidence/m5-tiles-20261122/timing.py > research/ens/evidence/m5-tiles-20261122/run.log 2>&1 || echo "retest failed rc=$?" >> $E/slot.log
echo "retest done $(date)" >> $E/slot.log
zsh $E/train_seed4.sh
echo "seed4 done $(date)" >> $E/slot.log
kill $WD 2>/dev/null; true
