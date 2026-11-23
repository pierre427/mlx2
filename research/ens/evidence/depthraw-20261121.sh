#!/bin/zsh
# H19b / H21a follow-up: tiles read the core d-wide state (not the frozen tap), 6000 steps
set -e
cd ~/ens-lab
export PYTHONPATH=$PWD
PY=~/mlx2-0324-venv/bin/python
E=evidence/depthraw-20261121
mkdir -p $E
$PY -m research.ens.depth --core evidence/pilot2-20261121/core-c6 --steps 6000 --cells FST:3:2 GRAPH:8:2 \
    --arms raw-r0 raw-attn2 raw-attn4 raw-tri1 raw-alone-attn2 raw-alone-tri1 raw-shuf-attn2 --out $E/depth-c6 > $E/depth-c6.log 2>&1
touch $E/DONE
