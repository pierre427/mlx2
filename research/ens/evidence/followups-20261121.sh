#!/bin/zsh
# H11 follow-up: key-table tiles trained against the 1-byte codebook form. H8 follow-up: routers reading the key position.
set -e
cd ~/ens-lab
export PYTHONPATH=$PWD
PY=~/mlx2-0324-venv/bin/python
C6=evidence/pilot2-20261121/core-c6
S=evidence/screens-20261121
H=evidence/h8-h11-20261121
E=evidence/followups-20261121
mkdir -p $E
for fam in program.ADD.h1 graph.6-2.h1 fst.3-1.h1 graph.4-2.h1; do
  t=c6-keytable-snap-lr1e-2-$fam
  $PY -m research.ens.contrib --core $C6 --family $fam --kind key-table --snap-codebook --seed 0 --steps 6000 --lr 1e-2 --out $E/mod-$t > $E/mod-$t.log 2>&1
  for qz in none codebook; do
    $PY -m research.ens.evaluate --core $C6 --module $E/mod-$t --quant $qz --out $E/eval-$t-$qz > $E/eval-$t-$qz.log 2>&1
  done
done
$PY -m research.ens.route --core $C6 --module-a $S/mod-c6-mlp-s6000-lr1e-3-graph.6-2.h1 --module-b $H/mod-c6-mlp-graph.6-2.h2 --out $E/route-graph62 > $E/route-graph62.log 2>&1
$PY -m research.ens.route --core $C6 --module-a $S/mod-c6-mlp-s6000-lr1e-3-fst.3-1.h1 --module-b $H/mod-c6-mlp-fst.3-1.h2 --out $E/route-fst31 > $E/route-fst31.log 2>&1
touch $E/DONE
