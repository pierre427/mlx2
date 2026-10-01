"""Build an isolated inference-only online-softmax materialization candidate."""

import argparse
import hashlib
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--period", type=int, default=32)
    args = p.parse_args()
    if args.output.exists() or not 1 <= args.period <= 4096:
        p.error("fresh output and period1..4096 required")
    source = args.source.read_text()
    loop = "for k, v, kp in _tiles(\n            blocks, key_tile, minimum=minimum, maximum=maximum_position\n        ):"
    update = "            maximum = new_max\n"
    if source.count(loop) != 1 or source.count(update) != 1:
        raise ValueError("reference scan structure differs; review candidate transformation")
    candidate = source.replace(loop, "for tile_index, (k, v, kp) in enumerate(_tiles(\n            blocks, key_tile, minimum=minimum, maximum=maximum_position\n        )):")
    candidate = candidate.replace(update, update + f"            if (tile_index + 1) % {args.period} == 0:\n                mx.eval(accum, denom, maximum)\n")
    compile(candidate, str(args.output), "exec")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(candidate)
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
