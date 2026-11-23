#!/bin/zsh
cd ~/Desktop/mlx2-wt-ens-grow
export PYTHONPATH="$PWD/src:$PWD" ENS_DEVICE=cpu
PY=~/Desktop/mlx2/.venv/bin/python
S=$1
C=(--d 32 --heads 2 --bus-width 16 --batch 64 --warmup 100 --window 150 --eval-every 4 --stall 0.03 --trial-steps 40 --trial-budget 0.2 --margin 0.002 --accept per-flop)
for s in 0 1 2; do $PY -m research.ens.grow_train $C --layers 2 --steps 3000 --seed $s --out $S/grow-s$s > $S/grow-s$s.log 2>&1 & done; wait
for s in 0 1 2; do
  F=$(python3 -c "import json;print(json.load(open('$S/grow-s$s/receipt.json'))['train_flops'])")
  big=$(python3 -c "print(round($F/(6*81124*64*32)))"); small=$(python3 -c "print(round($F/(6*46244*64*32)))")
  echo "seed $s flops $F big_steps $big small_steps $small" >> $S/budget.txt
  $PY -m research.ens.grow_train $C --layers 3 --ffn 192 --no-grow --steps $big --seed $s --out $S/bigeq-s$s > $S/bigeq-s$s.log 2>&1 &
  $PY -m research.ens.grow_train $C --layers 2 --no-grow --steps $small --seed $s --out $S/smalleq-s$s > $S/smalleq-s$s.log 2>&1 &
done; wait
touch $S/DONE
