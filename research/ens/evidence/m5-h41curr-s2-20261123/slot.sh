#!/bin/zsh
# chunk ens-h41-curr-s2 (m5_queue.sh): hard cap 20 min (watchdog 19.5 min), partials kept
cd ~/Desktop/mlx2-wt-ens-a0
now=$(date +%s); start_cut=$(date -j -f "%H:%M" "22:35" +%s); end_cut=$(date -j -f "%H:%M" "22:55" +%s)
if (( now > start_cut )); then echo "refused: past 22:35 at $(date)" >> ~/Desktop/mlx2-wt-ens-a0/research/ens/evidence/m5-h41curr-s2-20261123/slot.log; echo "rc=refused-late" > ~/Desktop/mlx2-wt-ens-a0/research/ens/evidence/m5-h41curr-s2-20261123/STOPPED; exit 0; fi
cap=$(( end_cut - now < 1170 ? end_cut - now : 1170 ))
PAT="python -m research\.ens\..* .*research/ens/evidence/m5-h41curr-s2-20261123"                 # the training process only, never the cpg_job wrapper
( sleep $cap; echo "watchdog: cap $cap s reached, stopping $(date)" >> ~/Desktop/mlx2-wt-ens-a0/research/ens/evidence/m5-h41curr-s2-20261123/slot.log; pkill -TERM -f "$PAT"; sleep 30; pkill -KILL -f "$PAT" ) &
WD=$!
git rev-parse HEAD > ~/Desktop/mlx2-wt-ens-a0/research/ens/evidence/m5-h41curr-s2-20261123/commit.txt; git rev-parse origin/main > ~/Desktop/mlx2-wt-ens-a0/research/ens/evidence/m5-h41curr-s2-20261123/origin_main.txt
echo "chunk start $(date)" >> ~/Desktop/mlx2-wt-ens-a0/research/ens/evidence/m5-h41curr-s2-20261123/slot.log
export PYTHONPATH=~/Desktop/mlx2-wt-ens-a0/src:~/Desktop/mlx2-wt-ens-a0
{ ~/Desktop/mlx2/.venv/bin/python -m research.ens.recall_core --d 128 --heads 4 --layers 4 --steps 96000 --log-every 2000 --eval-every 1 --curriculum 0.6 --curriculum-hold 0.25 --p-recall 0.5 --seed 2 --out research/ens/evidence/m5-h41curr-s2-20261123/core-recall-curr && ~/Desktop/mlx2/.venv/bin/python -m research.ens.core001 research/ens/evidence/m5-h41curr-s2-20261123/core-recall-curr ; } > ~/Desktop/mlx2-wt-ens-a0/research/ens/evidence/m5-h41curr-s2-20261123/chunk.log 2>&1
rc=$?
kill $WD 2>/dev/null
if (( rc == 0 )); then echo "chunk done $(date)" >> ~/Desktop/mlx2-wt-ens-a0/research/ens/evidence/m5-h41curr-s2-20261123/slot.log; touch ~/Desktop/mlx2-wt-ens-a0/research/ens/evidence/m5-h41curr-s2-20261123/DONE
else echo "chunk stopped rc=$rc $(date)" >> ~/Desktop/mlx2-wt-ens-a0/research/ens/evidence/m5-h41curr-s2-20261123/slot.log; echo "rc=$rc" > ~/Desktop/mlx2-wt-ens-a0/research/ens/evidence/m5-h41curr-s2-20261123/STOPPED; fi
