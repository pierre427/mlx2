"""Stage 3: fused 4-bit affine dequant -> threadgroup fp16 -> NAX matmul2d, small M.

y[M, N] = x[M, K] @ dequant(W)^T, group_size 64. A threadgroup owns TN output
rows; per K chunk of KC it loads the packed weights coalesced, dequantizes into
threadgroup memory, then runs matmul2d (TM-row tile) accumulating in a
cooperative tensor. Optional split-K (S) writes fp32 partials that are summed.
Compared cache-cold against mx.quantized_matmul on the installed build.
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
    constexpr int KC = {KC};
    constexpr int NT = {NT};
    constexpr int GS = 64;
    constexpr int S = {S};
    const int M = m_dim[0];
    const int N = n_dim[0];
    const int K = k_dim[0];
    const int n0 = int(threadgroup_position_in_grid.x) * TN;
    const int split = int(threadgroup_position_in_grid.y);
    const uint tix = thread_index_in_threadgroup;

    threadgroup half Ws[TN * KC];

    constexpr auto desc = matmul2d_descriptor(
        TM, TN, KC, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NT / 32>> op;

    auto A = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        (device half*)x, dextents<int32_t, 2>(K, M));
    auto B = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        Ws, dextents<int32_t, 2>(KC, TN));
    auto tA0 = A.slice(0, 0);
    auto cT = op.get_destination_cooperative_tensor<decltype(tA0), decltype(B), float>();
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {{
        if (cT.is_valid_element(i)) cT[i] = 0;
    }}

    const int in_w = K / 2;          // bytes per packed row (4-bit)
    const int in_g = K / GS;
    const int chunks = K / KC;
    const int per_split = (chunks + S - 1) / S;
    const int c0 = split * per_split;
    const int c1 = min(chunks, c0 + per_split);
    // Each thread dequantizes one 64-value group per pass.
    constexpr int GROUPS = TN * KC / GS;

    for (int c = c0; c < c1; c++) {{
        const int k0 = c * KC;
        for (int gi = tix; gi < GROUPS; gi += NT) {{
            const int r = gi / (KC / GS);
            const int g = gi % (KC / GS);
            const int row = min(n0 + r, N - 1);
            const device uint4* src = (const device uint4*)((const device uint8_t*)w + row * in_w + (k0 + g * GS) / 2);
            const float s = float(scales[row * in_g + k0 / GS + g]);
            const float b = float(biases[row * in_g + k0 / GS + g]);
            threadgroup half4* dst = (threadgroup half4*)(Ws + r * KC + g * GS);
            #pragma unroll
            for (int h = 0; h < 2; h++) {{
                uint4 q = src[h];
                uint words[4] = {{q.x, q.y, q.z, q.w}};
                #pragma unroll
                for (int j = 0; j < 4; j++) {{
                    uint u = words[j];
                    dst[h * 8 + j * 2] = half4(s * float(u & 0xf) + b, s * float((u >> 4) & 0xf) + b,
                                               s * float((u >> 8) & 0xf) + b, s * float((u >> 12) & 0xf) + b);
                    dst[h * 8 + j * 2 + 1] = half4(s * float((u >> 16) & 0xf) + b, s * float((u >> 20) & 0xf) + b,
                                                   s * float((u >> 24) & 0xf) + b, s * float(u >> 28) + b);
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto tA = A.slice(k0, 0);
        op.run(tA, B, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {{
        if (cT.is_valid_element(i)) {{
            auto idx = cT.get_multidimensional_index(i);
            int n = n0 + idx[0];
            int m = idx[1];
            if (m < M && n < N) out[(size_t(split) * M + m) * N + n] = cT[i];
        }}
    }}
"""

_K = {}


def fused(x, wq, s, b, tm=16, tn=32, kc=256, nt=128, splits=1):
    M, K = x.shape
    N = wq.shape[0]
    key = (tm, tn, kc, nt, splits)
    if key not in _K:
        _K[key] = mx.fast.metal_kernel(
            name=f"nax_q4_tm{tm}_tn{tn}_kc{kc}_nt{nt}_s{splits}",
            input_names=["x", "w", "scales", "biases", "m_dim", "n_dim", "k_dim"], output_names=["out"],
            header=HEADER, source=SRC.format(TM=tm, TN=tn, KC=kc, NT=nt, S=splits))
    (y,) = _K[key](inputs=[x, wq, s, b, mx.array([M], dtype=mx.int32), mx.array([N], dtype=mx.int32),
                           mx.array([K], dtype=mx.int32)],
                   grid=(N // tn * nt, splits, 1), threadgroup=(nt, 1, 1),
                   output_shapes=[(splits, M, N)], output_dtypes=[mx.float32])
    y = y[0] if splits == 1 else y.sum(axis=0)
    return y.astype(x.dtype)


SHAPES = {"27b_qkv": (8192, 5120), "27b_up": (17408, 5120), "27b_down": (5120, 17408), "fn_attn": (8192, 4096)}
CONFIGS = [  # (tm, tn, kc, nt, splits)
    (16, 32, 256, 128, 1), (16, 32, 256, 128, 2), (16, 32, 256, 128, 4),
    (16, 64, 128, 128, 1), (16, 64, 128, 128, 2), (16, 32, 128, 64, 2),
]


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
    for name, (N, K) in SHAPES.items():
        copies = max(4, (1 << 30) // (N * K // 2))
        mats = []
        for _ in range(copies):
            mats.append(mx.quantize((mx.random.normal((N, K)) * 0.02).astype(mx.float16), group_size=64, bits=4))
            mx.eval(mats[-1])
        x1 = mx.random.normal((1, K)).astype(mx.float16)
        t1 = cold(lambda q: mx.quantized_matmul(x1, *q, transpose=True, group_size=64, bits=4), mats)
        for M in (4, 6, 8, 12, 16):
            x = mx.random.normal((M, K)).astype(mx.float16)
            mx.eval(x)
            tr = cold(lambda q: mx.quantized_matmul(x, *q, transpose=True, group_size=64, bits=4), mats)
            ref = mx.quantized_matmul(x, *mats[0], transpose=True, group_size=64, bits=4).astype(mx.float32)
            row = {"shape": name, "M": M, "m1_us": round(t1 * 1e6, 1), "shipped_x": round(tr / t1, 2)}
            for cfg in CONFIGS:
                tm, tn, kc, nt, sp = cfg
                if (K // kc) < sp:
                    continue
                tag = f"tm{tm}tn{tn}kc{kc}nt{nt}s{sp}"
                try:
                    got = fused(x, *mats[0], *cfg).astype(mx.float32)
                    err = float((mx.max(mx.abs(ref - got)) / (mx.max(mx.abs(ref)) + 1e-6)).item())
                    t = cold(lambda q: fused(x, *q, *cfg), mats)
                    row[tag] = {"x": round(t / t1, 2), "err": round(err, 4)}
                except Exception as e:  # noqa: BLE001
                    row[tag] = {"error": [l for l in str(e).splitlines() if 'error' in l][:2]}
            print(json.dumps(row), flush=True)
        del mats
        mx.clear_cache()


if __name__ == "__main__":
    main()
