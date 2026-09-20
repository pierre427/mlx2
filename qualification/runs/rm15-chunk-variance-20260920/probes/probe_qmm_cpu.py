"""rm15 step 0b: quantized matmul row-chunking on CPU.

If mx.quantized_matmul is row-independent, splitting M must be bit-exact.
A non-zero delta here means the kernel's accumulation/strategy is keyed on M,
which is exactly what a prefill chunk-size change does to every linear layer.
"""
import json
import mlx.core as mx

mx.set_default_device(mx.cpu)
mx.random.seed(0)


def maxdiff(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


res = []
for bits, group in ((4, 64), (4, 32), (8, 64)):
    for dt in (mx.float32, mx.bfloat16, mx.float16):
        K, N = 1024, 1024
        W = mx.random.normal((N, K)).astype(dt)
        wq, scales, biases = mx.quantize(W, group_size=group, bits=bits)
        M = 512
        x = mx.random.normal((M, K)).astype(dt)
        full = mx.quantized_matmul(x, wq, scales, biases, transpose=True,
                                   group_size=group, bits=bits)
        mx.eval(full)
        entry = {"bits": bits, "group": group, "dtype": str(dt), "M": M,
                 "deltas": {}}
        for c in (1, 16, 64, 128, 256):
            y = mx.concatenate([
                mx.quantized_matmul(x[i:i + c], wq, scales, biases, transpose=True,
                                    group_size=group, bits=bits)
                for i in range(0, M, c)], axis=0)
            mx.eval(y)
            entry["deltas"][str(c)] = maxdiff(full, y)
        res.append(entry)

print(json.dumps({"device": "cpu", "probes": res}, indent=2))
