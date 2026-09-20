#!/usr/bin/env python3
"""Offline counterfactual for the collect-only MoE expert atlas.

Replays a recorded expert access trace and reports the hit rate an
atlas-pinned resident set *would* have achieved, against the plain LRU that
actually ran.  Nothing here runs a model, and nothing in the serving path
consults the atlas: this analysis is the gate for ever building pinning.

    python scripts/analyze_expert_atlas.py \
        --trace runs/expert-trace.bin --atlas <model_dir>/weight_atlas.json \
        --capacity 64 --out runs/atlas-counterfactual.json

Any wall-clock or throughput figure from the streamed run that produced the
trace is NOT a benchmark and must not be recorded as one.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlx2.runtime.expert_atlas import (  # noqa: E402
    DEFAULT_MIN_SAMPLES,
    load_atlas,
    replay_counterfactual,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, help="expert access trace (.bin)")
    parser.add_argument("--atlas", default=None, help="weight_atlas.json to replay")
    parser.add_argument("--capacity", type=int, required=True,
                        help="per-layer resident expert capacity to model")
    parser.add_argument("--pin-fractions", default="0,0.1,0.2,0.33,0.5")
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    atlas = load_atlas(args.atlas) if args.atlas else None
    if args.atlas and atlas is None:
        print(
            f"atlas {args.atlas} is missing, stale, torn or geometry-mismatched; "
            "replaying LRU only",
            file=sys.stderr,
        )
    fractions = [float(part) for part in args.pin_fractions.split(",") if part.strip()]
    report = replay_counterfactual(
        args.trace,
        capacity=args.capacity,
        atlas=atlas,
        pin_fractions=fractions,
        min_samples=args.min_samples,
    )
    report["trace"] = str(args.trace)
    report["atlas"] = str(args.atlas) if args.atlas else None

    width = 78
    print("=" * width)
    print("MoE expert atlas counterfactual (collect-only atlas; LRU actually ran)")
    print("=" * width)
    print(
        f"{'pin_fraction':>13} {'pinned/layer':>13} {'hit_rate':>10} "
        f"{'page_ins':>10} {'vs LRU':>10}"
    )
    for row in report["rows"]:
        print(
            f"{row['pin_fraction']:>13.2f} {row['pinned_per_layer']:>13d} "
            f"{row['hit_rate']:>10.4f} {row['page_ins']:>10d} "
            f"{row['page_ins_vs_lru']:>+10d}"
        )
    print()
    print(report["verdict"])

    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
