#!/bin/zsh
# H24: does the C6 ordinary-task regression come from its 25% ordinary-row share? seed 0, early/0.5/64, p_ordinary 0.5
set -e
cd ~/ens-lab
export PYTHONPATH=$PWD
PY=~/mlx2-0324-venv/bin/python
E=evidence/ordshare-20261122
mkdir -p $E
$PY -m research.ens.train_core --arm c6 --seed 0 --steps 12000 --log-every 500 --site early --tau 0.5 --bus-width 64 --p-ordinary 0.5 --out $E/core-c6-pord0.5 > $E/core-c6-pord0.5.log 2>&1
$PY -c "
import json; from research.ens import common as C; from research.ens.evaluate import ordinary_suite
core,ck=C.load_core(/core-c6-pord0.5); core.freeze()
C.write_json(__import__(pathlib).Path(/ordinary_suite.json), {checkpoint: ck, ordinary_suite: ordinary_suite(core), **C.source_revision()})
"
$PY -m research.ens.contrib --core $E/core-c6-pord0.5 --family graph.6-2.h1 --kind mlp --seed 0 --steps 6000 --lr 1e-3 --out $E/mod-graph.6-2.h1 > $E/mod.log 2>&1
$PY -m research.ens.evaluate --core $E/core-c6-pord0.5 --module $E/mod-graph.6-2.h1 --out $E/eval-graph.6-2.h1 > $E/eval.log 2>&1
touch $E/DONE
