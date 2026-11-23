#!/bin/zsh
# One M5 slot (coordinator 2026-11-23, CPG task beff4e63...): H41 v2 recall cores back to back, d256/L6 then
# d128/L4, 48k steps each, one lease. Hard cap 75 min for both (coordinator): the watchdog stops whatever is
# still running and keeps what finished. M5-trained (receipts carry host/device).
cd "$(dirname "$0")/../../../.."
E=research/ens/evidence/m5-h41recall-20261123
( sleep 4500; echo "watchdog: 75 min cap reached, stopping $(date)" >> $E/slot.log; pkill -TERM -f "research.ens.recall_core .*m5-h41recall-20261123" ) &
WD=$!
PY=~/Desktop/mlx2/.venv/bin/python
git rev-parse HEAD > $E/commit.txt; git rev-parse origin/main > $E/origin_main.txt
for spec in "d256 256 4 6" "d128 128 4 4"; do
  set -- ${=spec}
  echo "$1 start $(date)" >> $E/slot.log
  $PY -m research.ens.recall_core --d $2 --heads $3 --layers $4 --steps 48000 --log-every 1000 --out $E/core-recall-$1 \
      > $E/core-recall-$1.log 2>&1 && echo "$1 done $(date)" >> $E/slot.log || echo "$1 stopped rc=$? $(date)" >> $E/slot.log
  $PY -m research.ens.core001 $E/core-recall-$1 > $E/core001-$1.log 2>&1 || true
done
kill $WD 2>/dev/null; touch $E/DONE
