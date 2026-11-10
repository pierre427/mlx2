#!/usr/bin/env python3
"""Routed (layer, expert) set from a routing_capture_server capture.bin (all prefill and decode rows).

usage: capture_experts.py CAPTURE.bin OUT.json [--merge EXISTING.json ...]
Writes {"<layer>": [experts]} and prints per-layer counts and the total.
"""
import json, sys
from collections import defaultdict
import numpy as np

cap, out = sys.argv[1], sys.argv[2]
merge = sys.argv[sys.argv.index("--merge") + 1:] if "--merge" in sys.argv else []
used = defaultdict(set)
with open(cap, "rb") as f:
    while True:
        try:
            head = np.load(f, allow_pickle=False)
        except (EOFError, ValueError):
            break
        ids = np.load(f, allow_pickle=False); np.load(f, allow_pickle=False); np.load(f, allow_pickle=False)
        used[int(head[0])].update(int(e) for e in ids.reshape(-1))
for m in merge:
    for k, v in json.load(open(m)).items():
        used[int(k)].update(v)
json.dump({str(l): sorted(used[l]) for l in sorted(used)}, open(out, "w"))
print({l: len(used[l]) for l in sorted(used)}, "total", sum(map(len, used.values())))
