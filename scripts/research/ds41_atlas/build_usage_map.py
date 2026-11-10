"""Write a source-bound per-layer expert usage map from the routing-capture trace.

usage: build_usage_map.py TRACE.npz OUT.json [--split train|heldout|all] [--decode-only]

Counts every (layer, expert) selection of the chosen split (default: train,
tok_split == 0, prefill and decode rows) and writes the
``mlx2.ds41-source-expert-usage.v1`` file that ``deepseek_source_experts.tiers.usage_map``
reads (deepseek_v41_source_tiers.load_usage). CPU only; numpy only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

SOURCE = 'c30fd25e3cc79ba7044686fa3f9a7a1c0f514f77ee6293d2f5f62bc409e02ddb'


def usage_counts(ids, split, decode, *, which='train', decode_only=False, experts=384):
    mask = np.ones(len(ids), bool)
    if which != 'all':
        mask &= split == (0 if which == 'train' else 1)
    if decode_only:
        mask &= decode == 1
    selected = ids[mask].astype(np.int64)            # tokens x layers x top-k
    layers = selected.shape[1]
    counts = np.zeros((layers, experts), np.int64)
    for layer in range(layers):
        np.add.at(counts[layer], selected[:, layer, :].ravel(), 1)
    return counts, int(mask.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trace'); parser.add_argument('out')
    parser.add_argument('--split', choices=('train', 'heldout', 'all'), default='train')
    parser.add_argument('--decode-only', action='store_true')
    args = parser.parse_args()
    z = np.load(args.trace, allow_pickle=False)
    counts, tokens = usage_counts(z['ids'], z['tok_split'], z['tok_decode'], which=args.split,
                                  decode_only=args.decode_only)
    trace_sha = hashlib.sha256(Path(args.trace).read_bytes()).hexdigest()
    value = {'schema': 'mlx2.ds41-source-expert-usage.v1', 'source': SOURCE,
             'derived_from': {'trace': str(Path(args.trace).resolve()), 'trace_sha256': trace_sha,
                              'split': args.split, 'decode_only': args.decode_only, 'tokens': tokens},
             'counts': counts.tolist()}
    out = Path(args.out); tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(value) + '\n'); os.replace(tmp, out)
    print(json.dumps({'out': str(out), 'tokens': tokens, 'sha256': hashlib.sha256(out.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
