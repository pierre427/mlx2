#!/bin/zsh
# H7 chain probe on c6 (early) and the best screened c6r (ingress); contributors on that c6r and the c6
# oracle-table under the current family definitions (recipe: mlp 6000 steps lr 1e-3).
set -e
cd ~/ens-lab
export PYTHONPATH=$PWD
PY=~/mlx2-0324-venv/bin/python
P2=evidence/pilot2-20261121
S=evidence/screens-20261121
E=evidence/h7-c6r-20261121
R=$S/core-c6r-ingress-t1.5-lr1e-3
mkdir -p $E
$PY -m research.ens.chain --core $P2/core-c6 --out $E/chain-c6 > $E/chain-c6.log 2>&1
$PY -m research.ens.chain --core $R --out $E/chain-c6r > $E/chain-c6r.log 2>&1
for fam in program.ADD.h1 graph.6-2.h1 fst.3-1.h1 graph.4-2.h1; do
  for spec in "c6r $R mlp 6000" "c6r $R oracle-table 2000" "c6 $P2/core-c6 oracle-table 2000"; do
    set -- ${=spec}; arm=$1; core=$2; kind=$3; st=$4; t=$arm-$kind-$fam
    $PY -m research.ens.contrib --core $core --family $fam --kind $kind --seed 0 --steps $st --lr 1e-3 --out $E/mod-$t > $E/mod-$t.log 2>&1
    $PY -m research.ens.evaluate --core $core --module $E/mod-$t --out $E/eval-$t > $E/eval-$t.log 2>&1
  done
done
touch $E/DONE
