#!/bin/zsh
# H30 (Pierre: the core is too flat): 12-layer cores, bus still after block 1. C7-L12 and C6-L12 (seed 1, 24k, per-family
# curves), then the LoRA adapter C6L on C7-L12; CORE-001; contributors on the 4 dev laws for C6-L12 and C6L-L12.
set -e
cd ~/ens-lab
export PYTHONPATH=$PWD
PY=~/mlx2-0324-venv/bin/python
E=evidence/h30-20261122
mkdir -p $E
COMMON=(--seed 1 --site early --tau 0.5 --bus-width 64 --layers 12)
$PY -m research.ens.train_core --arm c7 $COMMON --steps 24000 --log-every 400 --family-eval --out $E/core-c7-L12 > $E/core-c7-L12.log 2>&1
$PY -m research.ens.train_core --arm c6 $COMMON --steps 24000 --log-every 400 --family-eval --out $E/core-c6-L12 > $E/core-c6-L12.log 2>&1
$PY -m research.ens.train_core --arm c6l $COMMON --steps 24000 --log-every 1000 --lora-rank 32 --reader-init 0.01 --lr 1e-3 \
    --from-core $E/core-c7-L12 --out $E/core-c6l-L12 > $E/core-c6l-L12.log 2>&1
for arm in c7 c6 c6l; do $PY -m research.ens.core001 $E/core-$arm-L12 > $E/core001-$arm.log 2>&1; done
$PY evidence/h27-20261122/ordinary_with_tile.py $E/core-c6l-L12
for arm in c6 c6l; do
  for fam in program.ADD.h1 graph.6-2.h1 fst.3-1.h1 graph.4-2.h1; do
    $PY -m research.ens.evaluate --core $E/core-$arm-L12 --family $fam --out $E/oracle-$arm-$fam > $E/oracle-$arm-$fam.log 2>&1
    m=$arm-mlp-$fam
    $PY -m research.ens.contrib --core $E/core-$arm-L12 --family $fam --kind mlp --seed 0 --steps 6000 --lr 1e-3 --out $E/mod-$m > $E/mod-$m.log 2>&1
    $PY -m research.ens.evaluate --core $E/core-$arm-L12 --module $E/mod-$m --out $E/eval-$m > $E/eval-$m.log 2>&1
  done
done
touch $E/DONE
