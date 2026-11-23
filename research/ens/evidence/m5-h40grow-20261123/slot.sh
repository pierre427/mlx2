#!/bin/zsh
# One M5 slot: H40 x growth (Pierre 2026-11-23). GDN-hybrid seed GGGA grows to GGGAGGGA at 25% of 24k steps;
# then fixed GGGA and fixed GGGAGGGA, each given exactly the growing run's training FLOPs (6*N*T). Same seed (0) in
# every arm. FLOPs-matching arithmetic -> matching.json. Arms not finished by the cap are recorded as skipped.
cd "$(dirname "$0")/../../../.."
E=research/ens/evidence/m5-h40grow-20261123
CAP=${CAP_S:-3540}                                   # coordinator: 60 min for all three arms
( sleep $CAP; echo "watchdog: cap reached, stopping $(date)" >> $E/slot.log; pkill -TERM -f "research.ens.hybrid_grow .*m5-h40grow-20261123" ) &
WD=$!
PY=~/Desktop/mlx2/.venv/bin/python
git rev-parse HEAD > $E/commit.txt; git rev-parse origin/main > $E/origin_main.txt
echo "grow start $(date)" >> $E/slot.log
$PY -m research.ens.hybrid_grow --pattern GGGA --grow-at 0.25 --steps 24000 --out $E/grow > $E/grow.log 2>&1 || echo "grow stopped rc=$?" >> $E/slot.log
F=$($PY -c "import json;print(json.load(open('$E/grow/receipt.json'))['train_flops'])")
for pat in GGGA GGGAGGGA; do
  P=$(PYTHONPATH=$PWD/src:$PWD $PY -c "
from research.ens.hybrid import HybridConfig, HybridCore
from mlx.utils import tree_flatten
c=HybridCore(HybridConfig(d=64,heads=4,ffn=256,seq_len=32,pattern='$pat',bus_width=64,gdn_chunk=16)); print(sum(v.size for _,v in tree_flatten(c.parameters())))")
  ST=$(python3 -c "print(round($F/(6*$P*64*32)))")
  echo "fixed $pat params $P steps $ST start $(date)" >> $E/slot.log
  python3 -c "import json,os;p='$E/matching.json';d=json.load(open(p)) if os.path.exists(p) else {'rule':'steps = grow_train_flops / (6 * params * batch 64 * seq 32)','grow_train_flops':$F,'seed':0};d['fixed-$pat']={'params':$P,'steps':$ST,'flops':6*$P*64*32*$ST};json.dump(d,open(p,'w'),indent=1)"
  $PY -m research.ens.hybrid_grow --pattern $pat --steps $ST --out $E/fixed-$pat > $E/fixed-$pat.log 2>&1 || echo "fixed $pat stopped rc=$?" >> $E/slot.log
done
for a in grow fixed-GGGA fixed-GGGAGGGA; do [ -f $E/$a/receipt.json ] || echo "$a: skipped or unfinished (cap)" >> $E/skipped.txt; done
kill $WD 2>/dev/null; touch $E/DONE
