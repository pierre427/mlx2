"""Mirror of deepseek_v41.Attention.select's packed scoring loop at real DS4.1 indexer geometry.

Usage: PYTHONPATH=src .venv/bin/python scripts/bench_ds41_packed_index_decode_once.py   (under the GPU lock)
"""
import sys, time, json
import mlx.core as mx
from mlx2.runtime.models.deepseek_v41_packed_qat import produce, decode
from mlx2.runtime.models import deepseek_v41 as v41
mx.set_default_device(mx.gpu)
H, D = 32, 128
def run(count, length, once):
    mx.random.seed(0)
    keys = mx.random.normal((1, count, D))
    packed = produce(keys, "index_fp4")
    query = mx.random.normal((1, H, length, D))
    weight = mx.random.normal((1, length, H))
    mx.eval(packed.codes, packed.scales, query, weight)
    tile = max(1, min(128, v41.INDEX_SCORE_SCRATCH_BYTES // max(1, H * count * 4)))
    def body():
        cache = {} if once and length > tile else None
        outs = []
        dec = 0
        for first in range(0, length, tile):
            last = min(length, first + tile)
            parts = []
            for kf in range(0, count, 4096):
                kl = min(count, kf + 4096)
                k = None if cache is None else cache.get(kf)
                if k is None:
                    k = decode(packed.sliced((slice(None), slice(kf, kl)))); dec += 1
                    if cache is not None: cache[kf] = k
                part = mx.einsum("bhsd,btd->bsht", query[:, :, first:last], k)
                parts.append(mx.sum(mx.maximum(part, 0) * weight[:, first:last, :, None], 2))
            outs.append(mx.concatenate(parts, -1))
        out = mx.concatenate(outs, 1); mx.eval(out); return out, dec
    ref, dec = body()
    ts = []
    for _ in range(5):
        t = time.perf_counter(); body(); ts.append(time.perf_counter() - t)
    ts.sort()
    return ref, dec, ts[len(ts)//2], tile
res = []
for count in (4096, 16384, 32768, 65536):
    for length in (1, 16, 512, 2048):
        a, da, ta, tile = run(count, length, False)
        b, db, tb, _ = run(count, length, True)
        same = mx.array_equal(a, b).item()
        res.append(dict(count=count, length=length, tile=tile, decodes_off=da, decodes_on=db,
                        ms_off=round(ta*1e3, 2), ms_on=round(tb*1e3, 2), speedup=round(ta/tb, 2), bit_identical=same))
        print(json.dumps(res[-1]), flush=True)
