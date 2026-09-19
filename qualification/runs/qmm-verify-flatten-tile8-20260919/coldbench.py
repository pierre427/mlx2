"""Cache-cold qmm: rotate through distinct weight copies so every op streams from DRAM."""
import json, statistics, sys, time
import mlx.core as mx
assert "--i-own-the-gpu" in sys.argv
SHAPES = {"dense27b_qkv": (8192, 5120), "dense27b_up": (17408, 5120), "dense27b_down": (5120, 17408),
          "a3b_attn": (4096, 2048), "a3b_shared_up": (1024, 2048), "flashnext_attn": (8192, 4096)}
TARGET_BYTES = 1 << 30
for bits in (4, 8):
    for name, (N, K) in SHAPES.items():
        per = N * K * bits // 8
        copies = max(4, TARGET_BYTES // per)
        mats = []
        for c in range(copies):
            w = (mx.random.normal((N, K)) * 0.02).astype(mx.float16)
            mats.append(mx.quantize(w, group_size=64, bits=bits))
            mx.eval(mats[-1])
        base = None
        for m in (1, 2, 3, 4, 5, 6, 8, 12, 16):
            x = mx.random.normal((m, K)).astype(mx.float16)
            mx.eval(x)
            def run():
                return [mx.quantized_matmul(x, *q, transpose=True, group_size=64, bits=bits) for q in mats]
            for _ in range(2):
                mx.eval(run())
            s = []
            for _ in range(8):
                t0 = time.perf_counter(); mx.eval(run()); s.append((time.perf_counter() - t0) / copies)
            t = statistics.median(s); base = base or t
            gbps = (per + 2 * 2 * N * K // 64) / t / 1e9
            print(json.dumps({"bits": bits, "shape": name, "m": m, "us": round(t * 1e6, 1),
                              "ratio": round(t / base, 2), "GBps": round(gbps)}), flush=True)
        del mats
        mx.clear_cache()
