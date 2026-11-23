#!/bin/zsh
# H25 capacity probe: does a wider core (d 320) shrink the C7 - C6 ordinary gap? one arm per run (daytime-sized)
set -e
cd ~/ens-lab
export PYTHONPATH=$PWD
PY=~/mlx2-0324-venv/bin/python
E=evidence/capacity-20261122
ARM=$1
mkdir -p $E
$PY -m research.ens.train_core --arm $ARM --seed 0 --steps 12000 --log-every 500 --site early --tau 0.5 --bus-width 64 --d 320 --heads 4 \
    --out $E/core-$ARM-d320 > $E/core-$ARM-d320.log 2>&1
$PY evidence/ordshare-20261122/ordsuite.py $E/core-$ARM-d320
mv $E/ordinary_suite.json $E/ordinary_suite-$ARM-d320.json
touch $E/DONE-$ARM
