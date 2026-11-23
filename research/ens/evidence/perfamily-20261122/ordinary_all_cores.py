"""Read-only: per-family ordinary accuracy on every finished c6/c7 core (CPU)."""
import json, random, glob, pathlib, mlx.core as mx
from research.ens import common as C, families as F
from research.ens.model import value_logits
fams = F.split("train")
dirs = sorted(set(str(pathlib.Path(p).parent) for p in glob.glob("evidence/*/core-*/receipt.json") + glob.glob("evidence/*/*/core-*/receipt.json")))
short = lambda fid: fid.replace("program.", "p.").replace("graph.", "g.")
print("core".ljust(52), "arm site tau  w  d   steps pord |", " ".join(short(f.fid)[:7].rjust(7) for f in fams))
for d in dirs:
    r = json.loads((pathlib.Path(d) / "receipt.json").read_text())
    if r.get("arm") not in ("c6", "c7"): continue
    a = r["args"]
    try:
        core, ck = C.load_core(d); core.freeze()
    except Exception as e:
        print(d, "load failed", e); continue
    acc = []
    for fam in fams:
        rng = random.Random(1234); rows = [F.sample_row(fam, rng) for _ in range(300)]
        tok, q, f, ai, ans, zs = C.batch(rows, "ordinary")
        acc.append(float(mx.mean(mx.argmax(value_logits(core(tok), ai), -1) == ans).item()))
    print(d.replace("evidence/", "")[:52].ljust(52), r["arm"], str(a.get("site"))[:5].ljust(5), a.get("tau"), a.get("bus_width"), a.get("d"), str(a.get("steps")).rjust(5),
          a.get("p_ordinary"), "s%d |" % a.get("seed"), " ".join(f"{x:7.2f}" for x in acc), flush=True)
