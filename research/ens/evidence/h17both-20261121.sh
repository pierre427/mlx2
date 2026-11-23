#!/bin/zsh
# H17 follow-up: claim heads reading SLOT + key together
set -e
cd ~/ens-lab
export PYTHONPATH=$PWD
PY=~/mlx2-0324-venv/bin/python
C6=evidence/pilot2-20261121/core-c6
S=evidence/screens-20261121
H=evidence/h8-h11-20261121
E=evidence/h17both-20261121
mkdir -p $E
M() { echo $S/mod-c6-mlp-s6000-lr1e-3-$1; }
$PY -m research.ens.claim --core $C6 --reads both --modules $(M graph.6-2.h1) $H/mod-c6-mlp-graph.6-2.h2 $(M fst.3-1.h1) --out $E/claim-graph62-plus-fst31 > $E/claim-graph62-plus-fst31.log 2>&1
$PY -m research.ens.claim --core $C6 --reads both --modules $(M fst.3-1.h1) $H/mod-c6-mlp-fst.3-1.h2 $(M graph.4-2.h1) --out $E/claim-fst31-plus-graph42 > $E/claim-fst31-plus-graph42.log 2>&1
$PY -m research.ens.claim --core $C6 --reads both --modules $(M program.ADD.h1) $(M graph.4-2.h1) $(M fst.3-1.h1) --out $E/claim-cross > $E/claim-cross.log 2>&1
touch $E/DONE
