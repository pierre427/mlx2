"""CPU sim: per-layer vs global LRU for the reduced (Q1) pool on the routing trace.

usage: q1_pool_partition.py TRACE.npz [--q4-gb 60] [--q2-gb 0] [--pool-gb 15,7.5,30]

Map from TRAIN-split usage (the same resolver bands as the server: per layer,
top q4 then q2 by usage, the rest q1). Replay HELD-OUT decode tokens; every
selection of a q1-mapped expert is a Q1 pool access (originals ignored: this
isolates the reduced pool, assuming resident q4 originals never shadow q1
experts). Reports Q1 bank reads per decode token for one global LRU of
``pool`` bytes vs 40 per-layer LRUs of pool/40 bytes each.
"""
import sys as _tree_sys
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[4] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
import argparse
from collections import OrderedDict
import json

import numpy as np

from mlx2.adapters import deepseek_v41_source_tiers as tiers


def lru_misses(stream, capacity):
    cache, misses = OrderedDict(), 0
    for key in stream:
        if key in cache:
            cache.move_to_end(key)
            continue
        misses += 1
        if capacity <= 0:
            continue
        cache[key] = None
        if len(cache) > capacity:
            cache.popitem(last=False)
    return misses


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('trace')
    parser.add_argument('--q4-gb', type=float, default=60.0); parser.add_argument('--q2-gb', type=float, default=0.0)
    parser.add_argument('--pool-gb', default='15,7.5,30')
    args = parser.parse_args()
    z = np.load(args.trace, allow_pickle=False)
    ids, split, dec = z['ids'].astype(np.int64), z['tok_split'], z['tok_decode']
    layers = ids.shape[1]
    train = ids[split == 0]
    usage = np.zeros((layers, 384), np.int64)
    for layer in range(layers):
        np.add.at(usage[layer], train[:, layer, :].ravel(), 1)
    held = ids[(split == 1) & (dec == 1)]
    results = []
    for pool_gb in (float(v) for v in args.pool_gb.split(',')):
        q4, q2, q1 = int(args.q4_gb * 1e9), int(args.q2_gb * 1e9), int(pool_gb * 1e9)
        configured = ('q4', 'q2', 'q1') if q2 else ('q4', 'q1')
        totals = {'q4': q4, 'q1': q1, **({'q2': q2} if q2 else {})}
        layout = tiers.resolve(budget_bytes=q4 + q2 + q1, totals=totals, usage=usage.tolist(), tiers=configured)
        q1_set = [{e for e, c in enumerate(row) if c == '1'} for row in layout['map']]
        stream = [(l, int(e)) for tok in held for l in range(layers) for e in tok[l] if int(e) in q1_set[l]]
        per_layer_streams = [[k for k in stream if k[0] == l] for l in range(layers)]
        objects = q1 // tiers.Q1_BYTES
        glob = lru_misses(stream, objects)
        part = sum(lru_misses(s, layout['budgets'][l]['q1'] // tiers.Q1_BYTES) for l, s in enumerate(per_layer_streams))
        results.append(dict(pool_gb=pool_gb, q4_gb=args.q4_gb, q2_gb=args.q2_gb, decode_tokens=len(held),
                            q1_accesses_per_token=len(stream) / len(held),
                            global_lru_misses_per_token=glob / len(held),
                            per_layer_lru_misses_per_token=part / len(held)))
        print(json.dumps(results[-1]), flush=True)


if __name__ == '__main__':
    main()
