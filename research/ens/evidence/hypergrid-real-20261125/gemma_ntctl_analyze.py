"""No-think ablation control (gemma_probe.py --nothink-control): accuracy per set, held-out odd vs even."""
import json, sys
import numpy as np
R = json.load(open(sys.argv[1] + ".json")); RO = {int(k): v for k, v in R["readouts"].items()}
sets = [k for k in RO[0] if k.startswith("ntabl_")]
for lab, ids in (("held-out odd", [i for i in RO if i % 2]), ("even (sets chosen on their thinking)", [i for i in RO if not i % 2])):
    nb = np.array([RO[i]["nothink"]["correct"] for i in ids])
    print(f"--- {lab}, n={len(ids)}: no-think unablated {nb.mean():.1%}")
    for k in sets:
        v = np.array([RO[i][k]["correct"] for i in ids])
        print(f"  {k[6:]:9s} {v.mean():6.1%}  right->wrong {int((nb & ~v).sum()):2d}  wrong->right {int((~nb & v).sum()):2d}  entropy {np.mean([RO[i][k]['entropy'] for i in ids]):.3f}")
