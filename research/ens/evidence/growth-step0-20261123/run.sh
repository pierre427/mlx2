#!/bin/zsh
cd ~/Desktop/mlx2-wt-ens-grow
export PYTHONPATH="$PWD/src:$PWD" ENS_DEVICE=cpu
PY=~/Desktop/mlx2/.venv/bin/python
S=$1
git rev-parse HEAD > $S/commit.txt; git diff > $S/diff.patch
C=(--d 32 --heads 2 --bus-width 16 --batch 64 --warmup 800 --window 1200 --eval-every 2 --steps 24000 --no-grow)
for s in 0 1 2; do
  $PY -m research.ens.grow_train $C --layers 2 --start-op insert_block --seed $s --out $S/grown0-L3-s$s > $S/grown0-L3-s$s.log 2>&1 &
  $PY -m research.ens.grow_train $C --layers 3 --seed $s --out $S/fresh-L3-s$s > $S/fresh-L3-s$s.log 2>&1 &
  $PY -m research.ens.grow_train $C --layers 2 --start-op widen_ffn --seed $s --out $S/grown0-f192-s$s > $S/grown0-f192-s$s.log 2>&1 &
  $PY -m research.ens.grow_train $C --layers 2 --ffn 192 --seed $s --out $S/fresh-f192-s$s > $S/fresh-f192-s$s.log 2>&1 &
done; wait; touch $S/DONE
