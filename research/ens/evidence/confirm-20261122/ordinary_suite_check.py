"""Full ordinary suite (2048 fresh train-family rows, z in the prompt, no bus) on the seed 0-2 confirmation cores."""
import json
from research.ens import common as C
from research.ens.evaluate import ordinary_suite
out = {}
for seed in (0, 1, 2):
    for arm in ("c6", "c7"):
        core, ck = C.load_core(f"evidence/confirm-20261122/seed{seed}/core-{arm}")
        core.freeze()
        out[f"seed{seed}/{arm}"] = {"checkpoint": ck, "ordinary_suite": ordinary_suite(core)}
out["source"] = C.source_revision(); out["host"] = C.host_record()
C.write_json(__import__("pathlib").Path("evidence/confirm-20261122/ordinary_suite_s012.json"), out)
print(json.dumps({k: round(v["ordinary_suite"], 4) for k, v in out.items() if k.startswith("seed")}))
