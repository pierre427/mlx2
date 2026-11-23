#!/bin/zsh
# Batched-vs-sequential tile trainer parity on HELD accuracy (queue item 4). CPU only (ENS_DEVICE=cpu, M5 CPU).
# Same core, same families, same seed/steps/lr: contrib.py once per family vs contrib_batched.py for all four.
set -e
cd ~/Desktop/mlx2-wt-ens-a0
export PYTHONPATH="$PWD/src:$PWD" ENS_DEVICE=cpu
PY=~/Desktop/mlx2/.venv/bin/python
E=research/ens/evidence/tile-parity-20261122
CORE=research/ens/evidence/m5-seed4-20261122/core-c6
FAMS=(program.ADD.h1 graph.6-2.h1 fst.3-1.h1 graph.4-2.h1)
git rev-parse HEAD > $E/commit.txt
T=$(mktemp -d /private/tmp/claude-501/-Users-user--claude-spares-spare-D4A4tx/124a866a-8d7f-4e98-b436-d74179e7bdaf/scratchpad/parity.XXXX)
echo $T > $E/workdir.txt
t0=$(date +%s)
for f in $FAMS; do $PY -m research.ens.contrib --core $CORE --family $f --kind mlp --seed 0 --steps 2000 --lr 1e-3 --out $T/seq-$f > $T/seq-$f.log 2>&1; done
t1=$(date +%s)
$PY -m research.ens.contrib_batched --core $CORE --families $FAMS --kind mlp --seed 0 --steps 2000 --lr 1e-3 --out-root $T --prefix bat- > $T/bat.log 2>&1
t2=$(date +%s)
for f in $FAMS; do for a in seq bat; do $PY -m research.ens.evaluate --core $CORE --module $T/$a-$f --out $T/eval-$a-$f > $T/eval-$a-$f.log 2>&1; mkdir -p $E/eval-$a-$f; cp $T/eval-$a-$f/*.json $E/eval-$a-$f/; done; done
echo "{\"sequential_s\": $((t1-t0)), \"batched_s\": $((t2-t1))}" > $E/timing.json
touch $E/DONE
