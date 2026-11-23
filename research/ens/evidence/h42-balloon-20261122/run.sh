#!/bin/zsh
cd ~/Desktop/mlx2-wt-ens-grow
export PYTHONPATH="$PWD/src:$PWD" ENS_DEVICE=cpu
PY=~/Desktop/mlx2/.venv/bin/python
S=$1
git rev-parse HEAD > $S/commit.txt; git diff --stat > $S/dirty.txt
C=(--d 32 --heads 2 --bus-width 16 --batch 64 --warmup 100 --window 150 --eval-every 4)
for s in 0 1 2; do
  $PY -m research.ens.grow_train $C --mode balloon --layers 2 --steps 3000 --seed $s --out $S/balloon-s$s > $S/balloon-s$s.log 2>&1 &
  $PY -m research.ens.grow_train $C --mode balloon --prune-frac 0 --block-thresh 0 --layers 2 --steps 3000 --seed $s --out $S/noprune-s$s > $S/noprune-s$s.log 2>&1 &
done; wait
for s in 0 1 2; do
  read F L FF <<< $(python3 -c "import json;r=json.load(open('$S/balloon-s$s/receipt.json'));print(r['train_flops'],r['config']['layers'],r['config']['ffn'])")
  for spec in "big 3 192 81124" "small 2 128 46244"; do
    set -- ${=spec}; st=$(python3 -c "print(round($F/(6*$4*64*32)))")
    $PY -m research.ens.grow_train $C --no-grow --layers $2 --ffn $3 --steps $st --seed $s --out $S/$1eq-s$s > $S/$1eq-s$s.log 2>&1 &
  done
  P=$(PYTHONPATH="$PWD/src:$PWD" $PY -c "
from research.ens.model import Core, CoreConfig
from mlx.utils import tree_flatten
c=Core(CoreConfig(d=32,heads=2,layers=$L,ffn=$FF,bus_width=16,tau=0.5)); print(sum(v.size for _,v in tree_flatten(c.parameters())))")
  st=$(python3 -c "print(round($F/(6*$P*64*32)))")
  echo "seed $s flops $F final L$L f$FF params $P final_eq_steps $st" >> $S/budget.txt
  $PY -m research.ens.grow_train $C --no-grow --layers $L --ffn $FF --steps $st --seed $s --out $S/finaleq-s$s > $S/finaleq-s$s.log 2>&1 &
done; wait
touch $S/DONE
