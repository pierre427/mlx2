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
