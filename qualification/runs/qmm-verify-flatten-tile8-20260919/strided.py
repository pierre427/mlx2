"""Cache-cold time of the real verify shape: strided (lanes, L, K) view of a (lanes, L+1, K) buffer."""
import json, statistics, sys, time
import mlx.core as mx
assert "--i-own-the-gpu" in sys.argv
SHAPES = {"27b_qkv": (8192, 5120), "27b_up": (17408, 5120), "27b_down": (5120, 17408), "fn_attn": (8192, 4096)}
for bits in (4, 8):
    for name, (N, K) in SHAPES.items():
        per = N * K * bits // 8
        copies = max(4, (1 << 30) // per)
        mats = []
        for _ in range(copies):
            mats.append(mx.quantize((mx.random.normal((N, K)) * 0.02).astype(mx.float16), group_size=64, bits=bits))
            mx.eval(mats[-1])
        for lanes, L in ((1, 3), (1, 4), (2, 3)):
            full = mx.random.normal((lanes, L + 1, K)).astype(mx.float16)
            x = full[:, :L]
            mx.eval(full)
            run = lambda: [mx.quantized_matmul(x, *q, transpose=True, group_size=64, bits=bits) for q in mats]
            for _ in range(2):
                mx.eval(run())
            s = []
            for _ in range(6):
                t0 = time.perf_counter(); mx.eval(run()); s.append((time.perf_counter() - t0) / copies)
            print(json.dumps({"bits": bits, "shape": name, "lanes": lanes, "L": L, "us": round(statistics.median(s) * 1e6, 1)}), flush=True)
        del mats
        mx.clear_cache()
