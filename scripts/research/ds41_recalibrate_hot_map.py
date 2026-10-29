# SPDX-License-Identifier: MIT
"""Rebuild a DS4.1 Source hot map from the accumulated expert atlas, offline.

CPU only; reads the atlas JSON and writes a new map (never overwrites). The
result is a *recalibrated* map: the selected-defaults guard accepts it when it
binds the same Source fingerprint and hot budget, receipts report it as
``recalibrated`` (distinct from the pinned measured map), and its content
digest enters the effective artifact identity. Provenance:
provenance/deepseek-v41-source-expert-residency.json.

    python scripts/research/ds41_recalibrate_hot_map.py ATLAS OUTPUT \
        [--hot-bytes 60000000000] [--parent CURRENT_HOT_MAP]
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))

from mlx2.adapters.deepseek_v41_source_hotmap import build_recalibrated  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('atlas', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--hot-bytes', type=int, default=60_000_000_000)
    parser.add_argument('--parent', type=Path, default=None, help='hot map being replaced (digest recorded)')
    args = parser.parse_args(argv)
    result = build_recalibrated(args.atlas.resolve(), args.output.resolve(), hot_bytes=args.hot_bytes,
                                parent_hot_map=args.parent.resolve() if args.parent else None)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
