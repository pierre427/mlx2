#!/bin/zsh
# H14: emergent tile splitting on two conflicted mixed pools
set -e
cd ~/ens-lab
export PYTHONPATH=$PWD
PY=~/mlx2-0324-venv/bin/python
C6=evidence/pilot2-20261121/core-c6
E=evidence/h14-20261121
mkdir -p $E
$PY -m research.ens.split --core $C6 --family-a graph.4-2.h1 --family-b fst.3-1.h1 --out $E/split-graph42-fst31 > $E/split-graph42-fst31.log 2>&1
$PY -m research.ens.split --core $C6 --family-a program.ADD.h1 --family-b graph.6-2.h1 --out $E/split-progadd-graph62 > $E/split-progadd-graph62.log 2>&1
touch $E/DONE
