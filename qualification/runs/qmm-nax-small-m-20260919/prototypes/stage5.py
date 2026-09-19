"""Stage 5: NAX reads raw 4-bit (uint4b_format) / 8-bit (uint8_t) weights directly.

Per quantization group g (K chunk of GS): tmp = x[:, g] @ q[n, g]^T on NAX (mode
multiply), then acc += s[n,g] * tmp + b[n,g] * xsum[m,g] on the cooperative
tensor. xsum (M x K/GS) is precomputed. No dequantization, no threadgroup tiles.
"""
import json
import statistics
import sys
import time

import mlx.core as mx

HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
using namespace metal;
"""

SRC = """
    constexpr int TM = {TM};
    constexpr int TN = {TN};
    constexpr int GS = {GS};
    constexpr int SG = {SG};
    constexpr int S = {S};
    const int M = m_dim[0];
    const int N = n_dim[0];
    const int K = k_dim[0];
    const int in_g = K / GS;
    const int n0 = int(threadgroup_position_in_grid.x) * TN;
    const int split = int(threadgroup_position_in_grid.y);
    const int per_split = (in_g + S - 1) / S;
    const int g0 = split * per_split;
    const int g1 = min(in_g, g0 + per_split);

    constexpr auto desc = matmul2d_descriptor(TM, TN, GS, false, true, true,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto A = tensor<device T, dextents<int32_t, 2>, tensor_inline>((device T*)x, dextents<int32_t, 2>(K, M));
    auto B = tensor<device {WT}, dextents<int32_t, 2>, tensor_inline>((device {WP}*)w, dextents<int32_t, 2>(K, N));
    auto tA0 = A.slice(0, 0);
    auto tB0 = B.slice(0, n0);
    auto tmp = op.get_destination_cooperative_tensor<decltype(tA0), decltype(tB0), float>();
    auto acc = op.get_destination_cooperative_tensor<decltype(tA0), decltype(tB0), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {{ if (acc.is_valid_element(i)) acc[i] = 0; }}
    for (int g = g0; g < g1; g++) {{
        auto tA = A.slice(g * GS, 0);
        auto tB = B.slice(g * GS, n0);
        op.run(tA, tB, tmp);
        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {{
            if (acc.is_valid_element(i)) {{
                auto idx = tmp.get_multidimensional_index(i);
                int n = min(n0 + idx[0], N - 1);
                int m = min(int(idx[1]), M - 1);
                acc[i] += float(scales[n * in_g + g]) * tmp[i] + float(biases[n * in_g + g]) * xsum[m * in_g + g];
            }}
        }}
    }}
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {{
        if (acc.is_valid_element(i)) {{
            auto idx = acc.get_multidimensional_index(i);
            int n = n0 + idx[0], m = idx[1];
            if (m < M && n < N) out[(size_t(split) * M + m) * N + n] = acc[i];
        }}
    }}
"""

_K = {}


def raw_qmm(x, wq, s, b, bits=4, gs=64, tm=16, tn=32, sg=4, splits=1):
    M, K = x.shape
    N = wq.shape[0]
    wt = "uint4b_format" if bits == 4 else "uint8_t"
    key = (bits, gs, tm, tn, sg, splits)
    if key not in _K:
        _K[key] = mx.fast.metal_kernel(
            name="nax_raw_" + "_".join(str(v) for v in key),
            input_names=["x", "w", "scales", "biases", "xsum", "m_dim", "n_dim", "k_dim"], output_names=["out"],
            header=HEADER, source=SRC.format(TM=tm, TN=tn, GS=gs, SG=sg, S=splits, WT=wt, WP="uchar" if bits == 4 else "uint8_t"))
    xsum = x.astype(mx.float32).reshape(M, K // gs, gs).sum(-1)
    (y,) = _K[key](inputs=[x, wq, s, b, xsum, mx.array([M], dtype=mx.int32), mx.array([N], dtype=mx.int32),
                           mx.array([K], dtype=mx.int32)],
                   template=[("T", x.dtype)], grid=(((N + tn - 1) // tn) * 32 * sg, splits, 1),
                   threadgroup=(32 * sg, 1, 1), output_shapes=[(splits, M, N)], output_dtypes=[mx.float32])
    y = y[0] if splits == 1 else y.sum(axis=0)
    return y.astype(x.dtype)


SHAPES = {"27b_qkv": (8192, 5120), "27b_up": (17408, 5120), "27b_down": (5120, 17408), "fn_attn": (8192, 4096)}
CONFIGS = {
    "tn32_sg4_s1": dict(tn=32, sg=4, splits=1), "tn32_sg4_s2": dict(tn=32, sg=4, splits=2),
    "tn64_sg4_s1": dict(tn=64, sg=4, splits=1), "tn64_sg4_s2": dict(tn=64, sg=4, splits=2),
    "tn32_sg2_s2": dict(tn=32, sg=2, splits=2), "tn64_sg8_s2": dict(tn=64, sg=8, splits=2),
}


def cold(fn, mats):
    run = lambda: [fn(q) for q in mats]
    for _ in range(2):
        mx.eval(run())
    s = []
    for _ in range(6):
        t0 = time.perf_counter(); mx.eval(run()); s.append((time.perf_counter() - t0) / len(mats))
    return statistics.median(s)


def main():
    assert "--i-own-the-gpu" in sys.argv
    if "--check" in sys.argv:
        for bits in (4, 8):
            q = mx.quantize((mx.random.normal((96, 1024)) * 0.02).astype(mx.float16), group_size=64, bits=bits)
            for M in (6, 16):
                x = mx.random.normal((M, 1024)).astype(mx.float16)
                r = mx.quantized_matmul(x, *q, transpose=True, group_size=64, bits=bits).astype(mx.float32)
                try:
                    y = raw_qmm(x, *q, bits=bits).astype(mx.float32)
                    print("check", bits, M, "relerr %.2e" % (mx.max(mx.abs(y - r)) / (mx.max(mx.abs(r)) + 1e-6)).item())
                except Exception as e:  # noqa: BLE001
                    print("check", bits, M, "ERR", [l[:300] for l in str(e).splitlines() if "error" in l][:3])
        return
    bits_list = [int(a.split("=")[1]) for a in sys.argv if a.startswith("--bits=")] or [4]
    for bits in bits_list:
        for name, (N, K) in SHAPES.items():
            copies = max(4, (1 << 30) // (N * K * bits // 8))
            mats = []
            for _ in range(copies):
                mats.append(mx.quantize((mx.random.normal((N, K)) * 0.02).astype(mx.float16), group_size=64, bits=bits))
                mx.eval(mats[-1])
            x1 = mx.random.normal((1, K)).astype(mx.float16)
            t1 = cold(lambda q: mx.quantized_matmul(x1, *q, transpose=True, group_size=64, bits=bits), mats)
            for M in (6, 8, 12, 16):
                x = mx.random.normal((M, K)).astype(mx.float16)
                mx.eval(x)
                tr = cold(lambda q: mx.quantized_matmul(x, *q, transpose=True, group_size=64, bits=bits), mats)
                ref = mx.quantized_matmul(x, *mats[0], transpose=True, group_size=64, bits=bits).astype(mx.float32)
                row = {"bits": bits, "shape": name, "M": M, "shipped_x": round(tr / t1, 2)}
                for tag, cfg in CONFIGS.items():
                    try:
                        got = raw_qmm(x, *mats[0], bits=bits, **cfg).astype(mx.float32)
                        err = float((mx.max(mx.abs(ref - got)) / (mx.max(mx.abs(ref)) + 1e-6)).item())
                        t = cold(lambda q: raw_qmm(x, *q, bits=bits, **cfg), mats)
                        row[tag] = {"x": round(t / t1, 2), "err": round(err, 4)}
                    except Exception as e:  # noqa: BLE001
                        row[tag] = {"error": [l[:300] for l in str(e).splitlines() if "error" in l][:3]}
                print(json.dumps(row), flush=True)
            del mats
            mx.clear_cache()


if __name__ == "__main__":
    main()
