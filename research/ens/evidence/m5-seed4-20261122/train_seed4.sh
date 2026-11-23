#!/bin/zsh
# Seed-4 C6 and C7 for the A0 confirmation, trained on the M5 with the confirmation protocol exactly as
# confirm.sh runs it (48k steps, early / tau 0.5 / width 64 / 6 layers, --family-eval). The M3 confirmation
# reuses these cores (its reuse check verifies site, tau, width, layers, steps and seed).
set -e
cd "$(dirname "$0")/../../../.."
PY=~/Desktop/mlx2/.venv/bin/python
E=research/ens/evidence/m5-seed4-20261122
for arm in c6 c7; do
  $PY -m research.ens.train_core --arm $arm --seed 4 --steps 48000 --log-every 500 --family-eval --site early --tau 0.5 \
      --bus-width 64 --layers 6 --out $E/core-$arm > $E/core-$arm.log 2>&1
  $PY -m research.ens.core001 $E/core-$arm > $E/core001-$arm.log 2>&1 || true
done
git rev-parse HEAD > $E/commit.txt; git rev-parse origin/main > $E/origin_main.txt
touch $E/DONE
