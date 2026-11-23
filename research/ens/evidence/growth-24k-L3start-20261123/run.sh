#!/bin/zsh
# 3-layer start (a 2-layer start puts the bus site after the last block; step-0 diagnostic 2026-11-23).
# Long-schedule growth sweep (Pierre 2026-11-23: don't stop on early results). 24k steps = 8x the 3k toy, with
# windows and trials scaled 8x too (window 1200, trial 320, warmup 800). Every mode vs fixed arms at equal FLOPs.
cd ~/Desktop/mlx2-wt-ens-grow
export PYTHONPATH="$PWD/src:$PWD" ENS_DEVICE=cpu
PY=~/Desktop/mlx2/.venv/bin/python
S=$1
git rev-parse HEAD > $S/commit.txt
C=(--d 32 --heads 2 --bus-width 16 --batch 64 --warmup 800 --window 1200 --eval-every 2)
G=(--stall 0.03 --trial-steps 320 --trial-budget 0.2 --margin 0.002)
for s in 0 1 2; do
  $PY -m research.ens.grow_train $C --mode balloon --layers 3 --steps 24000 --seed $s --out $S/balloon-s$s > $S/balloon-s$s.log 2>&1 &
  $PY -m research.ens.grow_train $C $G --layers 3 --steps 24000 --seed $s --out $S/earnloss-s$s > $S/earnloss-s$s.log 2>&1 &
  $PY -m research.ens.grow_train $C $G --accept per-flop --layers 3 --steps 24000 --seed $s --out $S/earnflop-s$s > $S/earnflop-s$s.log 2>&1 &
done; wait
params() { PYTHONPATH="$PWD/src:$PWD" $PY -c "
from research.ens.model import Core, CoreConfig
from mlx.utils import tree_flatten
c=Core(CoreConfig(d=$1,heads=$2,layers=$3,ffn=$4,bus_width=16,tau=0.5)); print(sum(v.size for _,v in tree_flatten(c.parameters())))"; }
for arm in balloon earnloss earnflop; do for s in 0 1 2; do
  read F L FF D H <<< $(python3 -c "import json;r=json.load(open('$S/$arm-s$s/receipt.json'));c=r['config'];print(r['train_flops'],c['layers'],c['ffn'],c['d'],c['heads'])")
  st=$(python3 -c "print(round($F/(6*103716*64*32)))")
  echo "$arm seed $s flops $F final d$D h$H L$L f$FF big_steps $st" >> $S/budget.txt
  $PY -m research.ens.grow_train $C --no-grow --layers 4 --ffn 192 --steps $st --seed $s --out $S/big-for-$arm-s$s > $S/big-for-$arm-s$s.log 2>&1 &
  st3=$(python3 -c "print(round($F/(6*62692*64*32)))")
  $PY -m research.ens.grow_train $C --no-grow --layers 3 --steps $st3 --seed $s --out $S/start-for-$arm-s$s > $S/start-for-$arm-s$s.log 2>&1 &
  P=$(params $D $H $L $FF); st2=$(python3 -c "print(round($F/(6*$P*64*32)))")
  $PY -m research.ens.grow_train $C --no-grow --d $D --heads $H --layers $L --ffn $FF --steps $st2 --seed $s --out $S/final-for-$arm-s$s > $S/final-for-$arm-s$s.log 2>&1 &
done; wait; done
touch $S/DONE
