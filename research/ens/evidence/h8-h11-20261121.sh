#!/bin/zsh
# H8 routing (three pairs) and H11 key-table tiles with quantized storage, on the pilot-2 c6 core.
set -e
cd ~/ens-lab
export PYTHONPATH=$PWD
PY=~/mlx2-0324-venv/bin/python
C6=evidence/pilot2-20261121/core-c6
S=evidence/screens-20261121
E=evidence/h8-h11-20261121
mkdir -p $E
# second laws on dev structures (route_dev), same contributor recipe as the screen winner
for fam in graph.6-2.h2 fst.3-1.h2; do
  $PY -m research.ens.contrib --core $C6 --family $fam --kind mlp --seed 0 --steps 6000 --lr 1e-3 --out $E/mod-c6-mlp-$fam > $E/mod-c6-mlp-$fam.log 2>&1
done
M() { echo $S/mod-c6-mlp-s6000-lr1e-3-$1; }
$PY -m research.ens.route --core $C6 --module-a $(M graph.4-2.h1) --module-b $(M fst.3-1.h1) --out $E/route-cross > $E/route-cross.log 2>&1
$PY -m research.ens.route --core $C6 --module-a $(M graph.6-2.h1) --module-b $E/mod-c6-mlp-graph.6-2.h2 --out $E/route-graph62 > $E/route-graph62.log 2>&1
$PY -m research.ens.route --core $C6 --module-a $(M fst.3-1.h1) --module-b $E/mod-c6-mlp-fst.3-1.h2 --out $E/route-fst31 > $E/route-fst31.log 2>&1
# H11: key-table tiles, two learning rates (a 16 x 64 table may want a larger one), four storage formats
for fam in program.ADD.h1 graph.6-2.h1 fst.3-1.h1 graph.4-2.h1; do
  for lr in 1e-3 1e-2; do
    t=c6-keytable-lr$lr-$fam
    $PY -m research.ens.contrib --core $C6 --family $fam --kind key-table --seed 0 --steps 6000 --lr $lr --out $E/mod-$t > $E/mod-$t.log 2>&1
    for qz in none int8 int4 codebook; do
      $PY -m research.ens.evaluate --core $C6 --module $E/mod-$t --quant $qz --out $E/eval-$t-$qz > $E/eval-$t-$qz.log 2>&1
    done
  done
done
touch $E/DONE
