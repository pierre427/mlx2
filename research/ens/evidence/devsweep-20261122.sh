#!/bin/zsh
# A0 dev sweep (night window, about 4 h on the M3, seed 0): site, tau and bus width for c6, plus a tau-matched c6r.
# Each core: oracle ceilings, MLP and oracle-table contributors on the 4 dev laws, and the H7 chain.
# Contributor recipe from screens-20261121: mlp 6000 steps lr 1e-3.
set -e
cd ~/ens-lab
export PYTHONPATH=$PWD
PY=~/mlx2-0324-venv/bin/python
E=evidence/devsweep-20261122
P2=evidence/pilot2-20261121
mkdir -p $E
DEV=(program.ADD.h1 graph.6-2.h1 fst.3-1.h1 graph.4-2.h1)
typeset -A CORE
# c6 configurations: site tau width
for cfg in "ingress 0.5 64" "ingress 1.5 64" "early 1.5 64" "early 0.5 32" "early 0.5 16"; do
  set -- ${=cfg}; site=$1; tau=$2; w=$3; t=c6-$site-t$tau-w$w
  $PY -m research.ens.train_core --arm c6 --seed 0 --steps 12000 --log-every 500 --site $site --tau $tau --bus-width $w \
      --out $E/core-$t > $E/core-$t.log 2>&1
  CORE[$t]=$E/core-$t
done
$PY -m research.ens.train_core --arm c6r --seed 0 --steps 6000 --log-every 500 --site ingress --tau 0.5 --lr 1e-3 \
    --reader-init 0.01 --from-core $P2/core-c7 --out $E/core-c6r-ingress-t0.5-w64 > $E/core-c6r-ingress-t0.5-w64.log 2>&1
CORE[c6r-ingress-t0.5-w64]=$E/core-c6r-ingress-t0.5-w64
CORE[c6-early-t0.5-w64]=$P2/core-c6
for t in ${(k)CORE}; do
  c=$CORE[$t]
  $PY -m research.ens.chain --core $c --out $E/chain-$t > $E/chain-$t.log 2>&1
  for fam in $DEV; do
    $PY -m research.ens.evaluate --core $c --family $fam --out $E/oracle-$t-$fam > $E/oracle-$t-$fam.log 2>&1
    for spec in "mlp 6000" "oracle-table 2000"; do
      set -- ${=spec}; kind=$1; st=$2; m=$t-$kind-$fam
      $PY -m research.ens.contrib --core $c --family $fam --kind $kind --seed 0 --steps $st --lr 1e-3 --out $E/mod-$m > $E/mod-$m.log 2>&1
      $PY -m research.ens.evaluate --core $c --module $E/mod-$m --out $E/eval-$m > $E/eval-$m.log 2>&1
    done
  done
done
touch $E/DONE
