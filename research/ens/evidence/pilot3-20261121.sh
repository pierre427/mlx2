#!/bin/zsh
# A0 pilot 3: re-scoped extension families on the pilot-2 cores, plus a diagnostic c6r at tau 1.5.
set -e
cd ~/ens-lab
export PYTHONPATH=$PWD
PY=~/mlx2-0324-venv/bin/python
P2=evidence/pilot2-20261121
E=evidence/pilot3-20261121
mkdir -p $E
$PY -m research.ens.train_core --arm c6r --seed 0 --steps 6000 --log-every 500 --tau 1.5 --from-core $P2/core-c7 --out $E/core-c6r-t1.5 > $E/core-c6r-t1.5.log 2>&1
typeset -A CORE
CORE=(c6 $P2/core-c6 c7 $P2/core-c7 c6r-t1.5 $E/core-c6r-t1.5)
for fam in program.ADD.h1 graph.6-2.h1 fst.3-1.h1 graph.4-2.h1; do
  for arm in c6 c7 c6r-t1.5; do
    for kind in mlp oracle-table; do
      t=$arm-$kind-$fam
      $PY -m research.ens.contrib --core $CORE[$arm] --family $fam --kind $kind --seed 0 --out $E/mod-$t > $E/mod-$t.log 2>&1
      $PY -m research.ens.evaluate --core $CORE[$arm] --module $E/mod-$t --out $E/eval-$t > $E/eval-$t.log 2>&1
    done
  done
done
touch $E/DONE
