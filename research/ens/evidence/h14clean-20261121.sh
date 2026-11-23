#!/bin/zsh
# H14 clean test: same structure, same keys, two laws (an irreducible conflict content cannot separate)
set -e
cd ~/ens-lab
export PYTHONPATH=$PWD
PY=~/mlx2-0324-venv/bin/python
C6=evidence/pilot2-20261121/core-c6
E=evidence/h14clean-20261121
mkdir -p $E
$PY -m research.ens.split --core $C6 --family-a graph.6-2.h1 --family-b graph.6-2.h2 --out $E/split-graph62-h1h2 > $E/split-graph62-h1h2.log 2>&1
$PY -m research.ens.split --core $C6 --family-a fst.3-1.h1 --family-b fst.3-1.h2 --out $E/split-fst31-h1h2 > $E/split-fst31-h1h2.log 2>&1
touch $E/DONE
