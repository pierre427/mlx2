#!/bin/zsh
# H29a: does bus-training delay or prevent two-op composition? C6 and C7, seed 1, confirmation settings, 48k steps,
# per-family ordinary/native curves every 2000 steps.
set -e
cd ~/ens-lab
export PYTHONPATH=$PWD
PY=~/mlx2-0324-venv/bin/python
E=evidence/h29a-20261122
mkdir -p $E
for arm in c7 c6; do
  $PY -m research.ens.train_core --arm $arm --seed 1 --steps 48000 --log-every 400 --family-eval --site early --tau 0.5 --bus-width 64 \
      --out $E/core-$arm-48k > $E/core-$arm-48k.log 2>&1
done
touch $E/DONE
