import sys, pathlib
from research.ens import common as C
from research.ens.evaluate import ordinary_suite
d = sys.argv[1]
core, ck = C.load_core(d); core.freeze()
C.write_json(pathlib.Path(d).parent / "ordinary_suite.json", {"checkpoint": ck, "ordinary_suite": ordinary_suite(core), **C.source_revision()})
