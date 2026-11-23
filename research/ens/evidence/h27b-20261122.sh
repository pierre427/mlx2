#!/bin/zsh
# H27b: a stronger LoRA pull - rank 32, 24k steps, reader init 0.01, on the frozen pilot-2 C7 at early / tau 0.5
set -e
cd ~/ens-lab
export PYTHONPATH=$PWD
PY=~/mlx2-0324-venv/bin/python
BASE=evidence/confirm-20261122/seed1/core-c7   # strongest C7 (learned 3 two-op families); was pilot-2
E=evidence/h27b-20261122
mkdir -p $E
t=c6l-early-t0.5-r32-24k
$PY -m research.ens.train_core --arm c6l --lora-rank 32 --reader-init 0.01 --seed 0 --steps 24000 --log-every 1000 --site early --tau 0.5 --lr 1e-3 \
    --from-core $BASE --out $E/core-$t > $E/core-$t.log 2>&1
$PY -m research.ens.core001 $E/core-$t > $E/core001.log 2>&1
$PY evidence/h27-20261122/ordinary_with_tile.py $E/core-$t
for fam in program.ADD.h1 graph.6-2.h1 fst.3-1.h1 graph.4-2.h1; do
  $PY -m research.ens.evaluate --core $E/core-$t --family $fam --out $E/oracle-$t-$fam > $E/oracle-$t-$fam.log 2>&1
  m=$t-mlp-$fam
  $PY -m research.ens.contrib --core $E/core-$t --family $fam --kind mlp --seed 0 --steps 6000 --lr 1e-3 --out $E/mod-$m > $E/mod-$m.log 2>&1
  $PY -m research.ens.evaluate --core $E/core-$t --module $E/mod-$m --out $E/eval-$m > $E/eval-$m.log 2>&1
done
touch $E/DONE
