#!/bin/zsh
set -e
cd ~/ens-lab
export PYTHONPATH=$PWD
PY=~/mlx2-0324-venv/bin/python
E=evidence/pilot2-20261121
mkdir -p $E
for arm in c6 c7; do
  $PY -m research.ens.train_core --arm $arm --seed 0 --steps 12000 --log-every 500 --out $E/core-$arm > $E/core-$arm.log 2>&1
done
$PY -m research.ens.train_core --arm c6r --seed 0 --steps 6000 --log-every 500 --from-core $E/core-c7 --out $E/core-c6r > $E/core-c6r.log 2>&1
for fam in perm.8 fst.3-2; do
  for arm in c6 c7 c6r; do
    for kind in mlp oracle-table; do
      $PY -m research.ens.contrib --core $E/core-$arm --family $fam --kind $kind --seed 0 --out $E/mod-$arm-$kind-$fam > $E/mod-$arm-$kind-$fam.log 2>&1
      $PY -m research.ens.evaluate --core $E/core-$arm --module $E/mod-$arm-$kind-$fam --out $E/eval-$arm-$kind-$fam > $E/eval-$arm-$kind-$fam.log 2>&1
    done
  done
done
echo DONE > $E/DONE
