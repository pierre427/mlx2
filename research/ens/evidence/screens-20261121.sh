#!/bin/zsh
# A0 recipe screens on dev families only (2026-11-21):
#  A. c6r recipe: frozen pilot-2 c7, ingress, reader-init 0.01, tau x lr; oracle ceilings on the 4 dev families.
#  B. contributor recipe: MLP on the pilot-2 c6 core, steps x lr, on the 4 dev families.
set -e
cd ~/ens-lab
export PYTHONPATH=$PWD
PY=~/mlx2-0324-venv/bin/python
P2=evidence/pilot2-20261121
E=evidence/screens-20261121
mkdir -p $E
DEV=(program.ADD.h1 graph.6-2.h1 fst.3-1.h1 graph.4-2.h1)
for cfg in "1.5 1e-3" "1.5 3e-3" "1.0 3e-3"; do
  set -- ${=cfg}; tau=$1; lr=$2; t=c6r-ingress-t$tau-lr$lr
  $PY -m research.ens.train_core --arm c6r --seed 0 --steps 6000 --log-every 500 --site ingress --tau $tau --lr $lr \
      --reader-init 0.01 --from-core $P2/core-c7 --out $E/core-$t > $E/core-$t.log 2>&1
  for fam in $DEV; do
    $PY -m research.ens.evaluate --core $E/core-$t --family $fam --out $E/oracle-$t-$fam > $E/oracle-$t-$fam.log 2>&1
  done
done
for cfg in "2000 1e-3" "2000 3e-3" "6000 1e-3" "6000 3e-3"; do
  set -- ${=cfg}; st=$1; lr=$2
  for fam in $DEV; do
    t=c6-mlp-s$st-lr$lr-$fam
    $PY -m research.ens.contrib --core $P2/core-c6 --family $fam --kind mlp --seed 0 --steps $st --lr $lr --out $E/mod-$t > $E/mod-$t.log 2>&1
    $PY -m research.ens.evaluate --core $P2/core-c6 --module $E/mod-$t --out $E/eval-$t > $E/eval-$t.log 2>&1
  done
done
touch $E/DONE
