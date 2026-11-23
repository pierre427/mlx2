#!/bin/zsh
# Can C6R learn at the early site with the small reader init? (needed if the sweep picks early for C6)
set -e
cd ~/ens-lab
export PYTHONPATH=$PWD
PY=~/mlx2-0324-venv/bin/python
P2=evidence/pilot2-20261121
E=evidence/c6rearly-20261122
mkdir -p $E
for tau in 0.5 1.5; do
  t=c6r-early-t$tau
  $PY -m research.ens.train_core --arm c6r --seed 0 --steps 6000 --log-every 500 --site early --tau $tau --lr 1e-3 \
      --reader-init 0.01 --from-core $P2/core-c7 --out $E/core-$t > $E/core-$t.log 2>&1
  for fam in program.ADD.h1 graph.6-2.h1 fst.3-1.h1 graph.4-2.h1; do
    $PY -m research.ens.evaluate --core $E/core-$t --family $fam --out $E/oracle-$t-$fam > $E/oracle-$t-$fam.log 2>&1
    m=$t-mlp-$fam
    $PY -m research.ens.contrib --core $E/core-$t --family $fam --kind mlp --seed 0 --steps 6000 --lr 1e-3 --out $E/mod-$m > $E/mod-$m.log 2>&1
    $PY -m research.ens.evaluate --core $E/core-$t --module $E/mod-$m --out $E/eval-$m > $E/eval-$m.log 2>&1
  done
done
touch $E/DONE
