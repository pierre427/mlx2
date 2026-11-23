#!/bin/zsh
# H17 self-claiming tiles on the pilot-2 c6: three trios (a pair, then a third tile added later).
set -e
cd ~/ens-lab
export PYTHONPATH=$PWD
PY=~/mlx2-0324-venv/bin/python
C6=evidence/pilot2-20261121/core-c6
S=evidence/screens-20261121
H=evidence/h8-h11-20261121
E=evidence/h17-20261121
mkdir -p $E
M() { echo $S/mod-c6-mlp-s6000-lr1e-3-$1; }
$PY -m research.ens.claim --core $C6 --modules $(M graph.6-2.h1) $H/mod-c6-mlp-graph.6-2.h2 $(M fst.3-1.h1) --out $E/claim-graph62-plus-fst31 > $E/claim-graph62-plus-fst31.log 2>&1
$PY -m research.ens.claim --core $C6 --modules $(M fst.3-1.h1) $H/mod-c6-mlp-fst.3-1.h2 $(M graph.4-2.h1) --out $E/claim-fst31-plus-graph42 > $E/claim-fst31-plus-graph42.log 2>&1
$PY -m research.ens.claim --core $C6 --modules $(M program.ADD.h1) $(M graph.4-2.h1) $(M fst.3-1.h1) --out $E/claim-cross > $E/claim-cross.log 2>&1
# H11 follow-up: key-choice tiles (the 1-byte form is the parameter)
for fam in program.ADD.h1 graph.6-2.h1 fst.3-1.h1 graph.4-2.h1; do
  t=c6-keychoice-lr1e-1-$fam
  $PY -m research.ens.contrib --core $C6 --family $fam --kind key-choice --seed 0 --steps 6000 --lr 1e-1 --out $E/mod-$t > $E/mod-$t.log 2>&1
  $PY -m research.ens.evaluate --core $C6 --module $E/mod-$t --out $E/eval-$t > $E/eval-$t.log 2>&1
done
touch $E/DONE
