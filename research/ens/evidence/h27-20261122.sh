#!/bin/zsh
# H27: can a tile that ships a LoRA for the blocks after the site pull the frozen, never-bus-trained C7? (dev laws only)
set -e
cd ~/ens-lab
export PYTHONPATH=$PWD
PY=~/mlx2-0324-venv/bin/python
P2=evidence/pilot2-20261121
E=evidence/h27-20261122
mkdir -p $E
t=c6l-early-t0.5-r8
$PY -m research.ens.train_core --arm c6l --lora-rank 8 --reader-init 0.01 --seed 0 --steps 6000 --log-every 500 --site early --tau 0.5 --lr 1e-3 \
    --from-core $P2/core-c7 --out $E/core-$t > $E/core-$t.log 2>&1
$PY -m research.ens.core001 $E/core-$t > $E/core001.log 2>&1
cat > $E/ordinary_with_tile.py <<'PYEOF'
"""Ordinary accuracy with the tile admitted (LoRA on, zero message) vs bypass, on fresh train-family rows."""
import sys, random, pathlib, mlx.core as mx
from research.ens import common as C, families as F
from research.ens.model import BusCall, value_logits
d = sys.argv[1]; core, ck = C.load_core(d); core.freeze()
rng = random.Random(7); fams = F.split("train")
rows = [F.sample_row(fams[rng.randrange(len(fams))], rng) for _ in range(2048)]
tok, q, f, a, ans, _ = C.batch(rows, "ordinary")
zero = mx.zeros((len(rows), core.cfg.bus_width))
acc = lambda lg: float(mx.mean(mx.argmax(value_logits(lg, a), -1) == ans).item())
C.write_json(pathlib.Path(d).parent / "ordinary_with_tile.json", {"checkpoint": ck, "bypass": acc(core(tok)),
             "tile_admitted_zero_message": acc(core(tok, BusCall("early", q, f, zero))), **C.source_revision()})
PYEOF
$PY $E/ordinary_with_tile.py $E/core-$t
for fam in program.ADD.h1 graph.6-2.h1 fst.3-1.h1 graph.4-2.h1; do
  $PY -m research.ens.evaluate --core $E/core-$t --family $fam --out $E/oracle-$t-$fam > $E/oracle-$t-$fam.log 2>&1
  m=$t-mlp-$fam
  $PY -m research.ens.contrib --core $E/core-$t --family $fam --kind mlp --seed 0 --steps 6000 --lr 1e-3 --out $E/mod-$m > $E/mod-$m.log 2>&1
  $PY -m research.ens.evaluate --core $E/core-$t --module $E/mod-$m --out $E/eval-$m > $E/eval-$m.log 2>&1
done
touch $E/DONE
