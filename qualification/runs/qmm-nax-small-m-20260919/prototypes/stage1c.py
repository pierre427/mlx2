"""Stage 1b: NAX compute ceiling. Cache-resident W, 64 ops per eval."""
import json, statistics, sys, time
import mlx.core as mx
sys.argv.append("--i-own-the-gpu")
from stage1 import nax_mm
for (Nn, K) in ((8192, 1024), (17408, 512)):
    w = (mx.random.normal((Nn, K)) * 0.02).astype(mx.float16)
    mx.eval(w)
    for M in (4, 8, 12, 16):
        x = mx.random.normal((M, K)).astype(mx.float16)
        mx.eval(x)
        row = {"N": Nn, "K": K, "M": M}
        for name, fn in [("mx_matmul", lambda: x @ w.T)] + [
            (f"tm{tm}_tn{tn}_sg{sg}", (lambda tm=tm, tn=tn, sg=sg: nax_mm(x, w, tm, tn, sg)))
            for tm, tn, sg in ((8, 32, 4), (8, 64, 4), (16, 32, 4), (16, 64, 4), (16, 128, 4), (32, 64, 4))
            if M <= tm
        ]:
            run = lambda: [fn() for _ in range(64)]
            for _ in range(3):
                mx.eval(run())
            s = []
            for _ in range(8):
                t0 = time.perf_counter(); mx.eval(run()); s.append((time.perf_counter() - t0) / 64)
            t = statistics.median(s)
            tm = int(name.split("_")[0][2:]) if name.startswith("tm") else M
            row[name] = {"us": round(t * 1e6, 2), "tflops": round(2 * M * Nn * K / t / 1e12, 1),
                         "tflops_padded": round(2 * tm * Nn * K / t / 1e12, 1)}
        print(json.dumps(row), flush=True)
