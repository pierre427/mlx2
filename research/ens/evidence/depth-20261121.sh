#!/bin/zsh
# H19b / H21a: depth inside one core call (attention and triplet tiles, with tile-alone and shuffled-prompt controls)
set -e
cd ~/ens-lab
export PYTHONPATH=$PWD
PY=~/mlx2-0324-venv/bin/python
E=evidence/depth-20261121
mkdir -p $E
$PY -m research.ens.depth --core evidence/pilot2-20261121/core-c6 --out $E/depth-c6 > $E/depth-c6.log 2>&1
touch $E/DONE
