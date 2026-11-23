#!/bin/zsh
# One M5 slot (coordinator 2026-11-23, CPG task beff4e63...): H41 v2b recall core with a facts curriculum.
# d128/L4 (better than d256 at this budget, m5-h41recall-20261123), 96k steps, 1 fact for the first 25%, ramp
# to 5 facts by 60%, p_recall 0.5; recall/ordinary ratios logged every 2000 steps. Hard cap 40 min (coordinator).
cd "$(dirname "$0")/../../../.."
E=research/ens/evidence/m5-h41curr-20261123
( sleep 2280; echo "watchdog: 38 min cap reached, stopping $(date)" >> $E/slot.log; pkill -TERM -f "research.ens.recall_core .*m5-h41curr-20261123" ) &
WD=$!
PY=~/Desktop/mlx2/.venv/bin/python
git rev-parse HEAD > $E/commit.txt; git rev-parse origin/main > $E/origin_main.txt
echo "start $(date)" >> $E/slot.log
$PY -m research.ens.recall_core --d 128 --heads 4 --layers 4 --steps 96000 --log-every 2000 --eval-every 1 \
    --curriculum 0.6 --curriculum-hold 0.25 --p-recall 0.5 --out $E/core-recall-curr > $E/core-recall-curr.log 2>&1 \
    && echo "done $(date)" >> $E/slot.log || echo "stopped rc=$? $(date)" >> $E/slot.log
$PY -m research.ens.core001 $E/core-recall-curr > $E/core001.log 2>&1 || true
kill $WD 2>/dev/null; touch $E/DONE
