#!/bin/zsh
# C6A dev check (dev laws only): can a small adapter reader retrofit the frozen pilot-2 C7 at early / tau 0.5?
set -e
cd ~/ens-lab
export PYTHONPATH=$PWD
PY=~/mlx2-0324-venv/bin/python
P2=evidence/pilot2-20261121
E=evidence/c6adev-20261122
mkdir -p $E
t=c6a-early-t0.5
$PY -m research.ens.train_core --arm c6a --seed 0 --steps 6000 --log-every 500 --site early --tau 0.5 --lr 1e-3 \
    --from-core $P2/core-c7 --out $E/core-$t > $E/core-$t.log 2>&1
$PY -m research.ens.core001 $E/core-$t > $E/core001.log 2>&1
for fam in program.ADD.h1 graph.6-2.h1 fst.3-1.h1 graph.4-2.h1; do
  $PY -m research.ens.evaluate --core $E/core-$t --family $fam --out $E/oracle-$t-$fam > $E/oracle-$t-$fam.log 2>&1
  m=$t-mlp-$fam
  $PY -m research.ens.contrib --core $E/core-$t --family $fam --kind mlp --seed 0 --steps 6000 --lr 1e-3 --out $E/mod-$m > $E/mod-$m.log 2>&1
  $PY -m research.ens.evaluate --core $E/core-$t --module $E/mod-$m --out $E/eval-$m > $E/eval-$m.log 2>&1
done
touch $E/DONE
