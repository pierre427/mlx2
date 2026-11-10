#!/usr/bin/env python3
"""mlx2.server with a read-only routing tap on the DS4.1 Source MoE entry (research capture only).

Every SourceV41Weights.moe call appends one record to CAPTURE (env MLX2_DS41_ROUTING_CAPTURE):
  np.save(int64 [layer, rows, time_ns]); np.save(ids int16 [rows, 6]); np.save(scores f16 [rows, 6]);
  np.save(x f16 [rows, hidden])   -- x is the router input (the overlap/lookahead route_hidden).
The tap copies arrays after the call returns; it never changes inputs, routes, or outputs.
Run exactly like mlx2.server; refuses an existing capture file.
"""
import os, queue, sys, threading, time
import numpy as np

path = os.environ["MLX2_DS41_ROUTING_CAPTURE"]
if os.path.exists(path):
    sys.exit("refusing existing capture " + path)
out = open(path, "xb")
q = queue.Queue(maxsize=4096)


def writer():
    while True:
        item = q.get()
        if item is None:
            break
        for a in item:
            np.save(out, a, allow_pickle=False)
        out.flush()


threading.Thread(target=writer, daemon=True).start()

import mlx.core as mx  # noqa: E402
from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights  # noqa: E402

_moe = SourceV41Weights.moe


def moe(self, x, ids, scores, prefix, **kw):
    result = _moe(self, x, ids, scores, prefix, **kw)
    layer = int(str(prefix).rstrip(".").split(".")[-1])
    xs = mx.array(x).astype(mx.float16); i = mx.array(ids).astype(mx.int16); s = mx.array(scores).astype(mx.float16)
    mx.eval(xs, i, s)
    rows = int(i.shape[0])
    q.put((np.array([layer, rows, time.time_ns()], np.int64), np.array(i).reshape(rows, -1),
           np.array(s).reshape(rows, -1), np.array(xs).reshape(rows, -1)))
    return result


SourceV41Weights.moe = moe

from mlx2.server import main  # noqa: E402

if __name__ == "__main__":
    try:
        main()
    finally:
        q.put(None); time.sleep(2); out.close()
