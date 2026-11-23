#!/bin/zsh
# Queue: H26 Matryoshka bus (seed 0), then H25 capacity done properly (d 320, 24k steps, C6 and C7)
set -e
cd ~/ens-lab
export PYTHONPATH=$PWD
PY=~/mlx2-0324-venv/bin/python
E=evidence/h26-20261122
mkdir -p $E
t=c6-early-t0.5-w64-mat0.5
$PY -m research.ens.train_core --arm c6 --seed 0 --steps 12000 --log-every 500 --site early --tau 0.5 --bus-width 64 --matryoshka 0.5 --out $E/core-$t > $E/core-$t.log 2>&1
$PY evidence/ordshare-20261122/ordsuite.py $E/core-$t && mv $E/ordinary_suite.json $E/ordinary_suite-$t.json
for fam in program.ADD.h1 graph.6-2.h1 fst.3-1.h1 graph.4-2.h1; do
  for half in full low; do
    $PY -m research.ens.evaluate --core $E/core-$t --family $fam --half $half --out $E/oracle-$t-$half-$fam > $E/oracle-$t-$half-$fam.log 2>&1
  done
  m=$t-mlp-$fam
  $PY -m research.ens.contrib --core $E/core-$t --family $fam --kind mlp --seed 0 --steps 6000 --lr 1e-3 --out $E/mod-$m > $E/mod-$m.log 2>&1
  for half in full low; do
    $PY -m research.ens.evaluate --core $E/core-$t --module $E/mod-$m --half $half --out $E/eval-$m-$half > $E/eval-$m-$half.log 2>&1
  done
done
touch $E/DONE
E=evidence/capacity24k-20261122
mkdir -p $E
for arm in c6 c7; do
  $PY -m research.ens.train_core --arm $arm --seed 0 --steps 24000 --log-every 1000 --site early --tau 0.5 --bus-width 64 --d 320 --heads 4 --out $E/core-$arm-d320 > $E/core-$arm-d320.log 2>&1
  $PY evidence/ordshare-20261122/ordsuite.py $E/core-$arm-d320 && mv $E/ordinary_suite.json $E/ordinary_suite-$arm-d320.json
done
touch $E/DONE
