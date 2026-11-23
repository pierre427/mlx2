#!/bin/zsh
set -e
cd ~/ens-lab
export PYTHONPATH=$PWD
PY=~/mlx2-0324-venv/bin/python
E=evidence/ordshare-20261122
cat > $E/ordsuite.py <<'PYEOF'
import sys, pathlib
from research.ens import common as C
from research.ens.evaluate import ordinary_suite
d = sys.argv[1]
core, ck = C.load_core(d); core.freeze()
C.write_json(pathlib.Path(d).parent / "ordinary_suite.json", {"checkpoint": ck, "ordinary_suite": ordinary_suite(core), **C.source_revision()})
PYEOF
$PY $E/ordsuite.py $E/core-c6-pord0.5
$PY -m research.ens.contrib --core $E/core-c6-pord0.5 --family graph.6-2.h1 --kind mlp --seed 0 --steps 6000 --lr 1e-3 --out $E/mod-graph.6-2.h1 > $E/mod.log 2>&1
$PY -m research.ens.evaluate --core $E/core-c6-pord0.5 --module $E/mod-graph.6-2.h1 --out $E/eval-graph.6-2.h1 > $E/eval.log 2>&1
touch $E/DONE
