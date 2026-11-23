#!/bin/zsh
# Diagnose why depth tiles do not learn (D1 architecture/optimiser, D2 multi-step pointer lookup, D3 budget)
set -e
cd ~/ens-lab
export PYTHONPATH=$PWD
PY=~/mlx2-0324-venv/bin/python
C6=evidence/pilot2-20261121/core-c6
E=evidence/depthdiag-20261121
mkdir -p $E
# D1: one-step lookup, tile alone and through the core
$PY -m research.ens.depth --core $C6 --steps 6000 --cells FST:3:1 GRAPH:8:1 --arms raw-alone-attn1 raw-attn1 alone-attn1 --out $E/d1 > $E/d1.log 2>&1
# D2/D3: two-step, 4x steps, higher lr, tile alone
$PY -m research.ens.depth --core $C6 --steps 24000 --lr 3e-3 --cells FST:3:2 --arms raw-alone-attn2 raw-alone-attn4 --out $E/d23 > $E/d23.log 2>&1
touch $E/DONE
