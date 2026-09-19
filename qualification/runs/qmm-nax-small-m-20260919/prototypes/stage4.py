"""Stage 4: fused affine dequant -> NAX matmul2d, small M, with overlap.

MODE "single": all simdgroups dequantize a chunk, barrier, all run NAX (stage 3).
MODE "ws":     warp-specialized double buffer. CSG consumer simdgroups run NAX on
               buffer c % 2 while PSG producer simdgroups dequantize chunk c + 1
               into the other buffer; one barrier per chunk.
Weight/activation tiles use the activation dtype T (half or bfloat). Affine
bits 4 or 8, group size 32/64/128. K must be a multiple of KC.
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

template <typename T>
METAL_FUNC vec<T, 4> v4(float a, float b, float c, float d) {
  return vec<T, 4>(T(a), T(b), T(c), T(d));
}

template <typename T, int BITS, int GS>
METAL_FUNC void dq_group(const device uint8_t* src, float s, float b, threadgroup T* dst) {
  // one quantization group: GS values
  if (BITS == 4) {
    const device uint4* q4 = (const device uint4*)src;   // 16 bytes = 32 values
    for (int h = 0; h < GS / 32; h++) {
      uint4 q = q4[h];
      uint words[4] = {q.x, q.y, q.z, q.w};
      for (int j = 0; j < 4; j++) {
        uint u = words[j];
        threadgroup vec<T, 4>* d = (threadgroup vec<T, 4>*)(dst + h * 32 + j * 8);
        d[0] = v4<T>(s * float(u & 0xf) + b, s * float((u >> 4) & 0xf) + b,
                         s * float((u >> 8) & 0xf) + b, s * float((u >> 12) & 0xf) + b);
        d[1] = v4<T>(s * float((u >> 16) & 0xf) + b, s * float((u >> 20) & 0xf) + b,
                         s * float((u >> 24) & 0xf) + b, s * float(u >> 28) + b);
      }
    }
  } else {
    const device uint4* q8 = (const device uint4*)src;   // 16 bytes = 16 values
    for (int h = 0; h < GS / 16; h++) {
      uint4 q = q8[h];
      uint words[4] = {q.x, q.y, q.z, q.w};
      for (int j = 0; j < 4; j++) {
        uint u = words[j];
        threadgroup vec<T, 4>* d = (threadgroup vec<T, 4>*)(dst + h * 16 + j * 4);
        d[0] = v4<T>(s * float(u & 0xff) + b, s * float((u >> 8) & 0xff) + b,
                         s * float((u >> 16) & 0xff) + b, s * float(u >> 24) + b);
      }
    }
  }
}
"""

COMMON = """
    constexpr int TM = {TM};
    constexpr int TN = {TN};
    constexpr int KC = {KC};
    constexpr int GS = {GS};
    constexpr int BITS = {BITS};
    constexpr int S = {S};
    const int M = m_dim[0];
    const int N = n_dim[0];
    const int K = k_dim[0];
    const int n0 = int(threadgroup_position_in_grid.x) * TN;
    const int split = int(threadgroup_position_in_grid.y);
    const uint tix = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int in_w = K * BITS / 8;
    const int in_g = K / GS;
    const int chunks = K / KC;
    const int per_split = (chunks + S - 1) / S;
    const int c0 = split * per_split;
    const int c1 = min(chunks, c0 + per_split);
    constexpr int GROUPS = TN * KC / GS;
    constexpr int GPR = KC / GS;           // groups per row per chunk
"""

SINGLE = COMMON + """
    constexpr int NT = {NT};
    threadgroup T Ws[TN * KC];
    constexpr auto desc = matmul2d_descriptor(TM, TN, KC, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NT / 32>> op;
    auto A = tensor<device T, dextents<int32_t, 2>, tensor_inline>((device T*)x, dextents<int32_t, 2>(K, M));
    auto B = tensor<threadgroup T, dextents<int32_t, 2>, tensor_inline>(Ws, dextents<int32_t, 2>(KC, TN));
    auto tA0 = A.slice(0, 0);
    auto cT = op.get_destination_cooperative_tensor<decltype(tA0), decltype(B), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {{ if (cT.is_valid_element(i)) cT[i] = 0; }}
    for (int c = c0; c < c1; c++) {{
        const int k0 = c * KC;
        for (int gi = tix; gi < GROUPS; gi += NT) {{
            const int r = gi / GPR, g = gi % GPR;
            const int row = min(n0 + r, N - 1);
            dq_group<T, BITS, GS>((const device uint8_t*)w + row * in_w + (k0 + g * GS) * BITS / 8,
                float(scales[row * in_g + k0 / GS + g]), float(biases[row * in_g + k0 / GS + g]),
                Ws + r * KC + g * GS);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto tA = A.slice(k0, 0);
        op.run(tA, B, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {{
        if (cT.is_valid_element(i)) {{
            auto idx = cT.get_multidimensional_index(i);
            int n = n0 + idx[0], m = idx[1];
            if (m < M && n < N) out[(size_t(split) * M + m) * N + n] = cT[i];
        }}
    }}
"""

WS = COMMON + """
    constexpr int CSG = {CSG};
    constexpr int PSG = {PSG};
    constexpr int NP = PSG * 32;
    threadgroup T Ws[2][TN * KC];
    constexpr auto desc = matmul2d_descriptor(TM, TN, KC, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<CSG>> op;
    auto A = tensor<device T, dextents<int32_t, 2>, tensor_inline>((device T*)x, dextents<int32_t, 2>(K, M));
    auto B0 = tensor<threadgroup T, dextents<int32_t, 2>, tensor_inline>(Ws[0], dextents<int32_t, 2>(KC, TN));
    auto B1 = tensor<threadgroup T, dextents<int32_t, 2>, tensor_inline>(Ws[1], dextents<int32_t, 2>(KC, TN));
    auto tA0 = A.slice(0, 0);
    auto cT = op.get_destination_cooperative_tensor<decltype(tA0), decltype(B0), float>();
    const bool consumer = sg < CSG;
    const uint ptix = tix - CSG * 32;
    if (consumer) {{
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) {{ if (cT.is_valid_element(i)) cT[i] = 0; }}
    }}
    // prologue: everyone fills buffer 0 with chunk c0
    if (c0 < c1) {{
        for (int gi = tix; gi < GROUPS; gi += (CSG + PSG) * 32) {{
            const int r = gi / GPR, g = gi % GPR;
            const int row = min(n0 + r, N - 1);
            const int k0 = c0 * KC;
            dq_group<T, BITS, GS>((const device uint8_t*)w + row * in_w + (k0 + g * GS) * BITS / 8,
                float(scales[row * in_g + k0 / GS + g]), float(biases[row * in_g + k0 / GS + g]),
                Ws[0] + r * KC + g * GS);
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int c = c0; c < c1; c++) {{
        const int buf = (c - c0) & 1;
        if (consumer) {{
            auto tA = A.slice(c * KC, 0);
            if (buf == 0) op.run(tA, B0, cT); else op.run(tA, B1, cT);
        }} else if (c + 1 < c1) {{
            const int k0 = (c + 1) * KC;
            threadgroup T* dst = Ws[buf ^ 1];
            for (int gi = ptix; gi < GROUPS; gi += NP) {{
                const int r = gi / GPR, g = gi % GPR;
                const int row = min(n0 + r, N - 1);
                dq_group<T, BITS, GS>((const device uint8_t*)w + row * in_w + (k0 + g * GS) * BITS / 8,
                    float(scales[row * in_g + k0 / GS + g]), float(biases[row * in_g + k0 / GS + g]),
                    dst + r * KC + g * GS);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if (consumer) {{
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) {{
            if (cT.is_valid_element(i)) {{
                auto idx = cT.get_multidimensional_index(i);
                int n = n0 + idx[0], m = idx[1];
                if (m < M && n < N) out[(size_t(split) * M + m) * N + n] = cT[i];
            }}
        }}
    }}
"""

_K = {}


def fused(x, wq, s, b, bits, gs, mode="ws", tm=16, tn=32, kc=128, nt=64, csg=2, psg=2, splits=1):
    M, K = x.shape
    N = wq.shape[0]
    key = (mode, bits, gs, tm, tn, kc, nt, csg, psg, splits)
    if key not in _K:
        src = (SINGLE if mode == "single" else WS).format(
            TM=tm, TN=tn, KC=kc, GS=gs, BITS=bits, S=splits, NT=nt, CSG=csg, PSG=psg)
        _K[key] = mx.fast.metal_kernel(
            name="nax_" + "_".join(str(v) for v in key), input_names=["x", "w", "scales", "biases", "m_dim", "n_dim", "k_dim"],
            output_names=["out"], header=HEADER, source=src)
    threads = nt if mode == "single" else (csg + psg) * 32
    (y,) = _K[key](inputs=[x, wq, s, b, mx.array([M], dtype=mx.int32), mx.array([N], dtype=mx.int32),
                           mx.array([K], dtype=mx.int32)],
                   template=[("T", x.dtype)],
                   grid=(((N + tn - 1) // tn) * threads, splits, 1), threadgroup=(threads, 1, 1),
                   output_shapes=[(splits, M, N)], output_dtypes=[mx.float32])
    y = y[0] if splits == 1 else y.sum(axis=0)
    return y.astype(x.dtype)


SHAPES = {"27b_qkv": (8192, 5120), "27b_up": (17408, 5120), "27b_down": (5120, 17408), "fn_attn": (8192, 4096)}
CONFIGS = {
    "single_tn32_kc128_nt64_s2": dict(mode="single", tn=32, kc=128, nt=64, splits=2),
    "single_tn64_kc128_nt128_s2": dict(mode="single", tn=64, kc=128, nt=128, splits=2),
    "ws_c1p1_tn32_kc128_s2": dict(mode="ws", tn=32, kc=128, csg=1, psg=1, splits=2),
    "ws_c2p2_tn32_kc128_s2": dict(mode="ws", tn=32, kc=128, csg=2, psg=2, splits=2),
    "ws_c1p3_tn32_kc128_s2": dict(mode="ws", tn=32, kc=128, csg=1, psg=3, splits=2),
    "ws_c2p2_tn64_kc128_s1": dict(mode="ws", tn=64, kc=128, csg=2, psg=2, splits=1),
    "ws_c2p2_tn64_kc128_s2": dict(mode="ws", tn=64, kc=128, csg=2, psg=2, splits=2),
    "ws_c1p1_tn32_kc256_s2": dict(mode="ws", tn=32, kc=256, csg=1, psg=1, splits=2),
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
    only = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--configs=")]
    configs = {k: v for k, v in CONFIGS.items() if not only or k in only[0].split(",")}
    bits_list = [int(a.split("=")[1]) for a in sys.argv if a.startswith("--bits=")] or [4]
    ms = [int(v) for a in sys.argv if a.startswith("--ms=") for v in a.split("=")[1].split(",")] or [6, 8, 12, 16]
    for bits in bits_list:
        for name, (N, K) in SHAPES.items():
            copies = max(4, (1 << 30) // (N * K * bits // 8))
            mats = []
            for _ in range(copies):
                mats.append(mx.quantize((mx.random.normal((N, K)) * 0.02).astype(mx.float16), group_size=64, bits=bits))
                mx.eval(mats[-1])
            x1 = mx.random.normal((1, K)).astype(mx.float16)
            t1 = cold(lambda q: mx.quantized_matmul(x1, *q, transpose=True, group_size=64, bits=bits), mats)
            for M in ms:
                x = mx.random.normal((M, K)).astype(mx.float16)
                mx.eval(x)
                tr = cold(lambda q: mx.quantized_matmul(x, *q, transpose=True, group_size=64, bits=bits), mats)
                ref = mx.quantized_matmul(x, *mats[0], transpose=True, group_size=64, bits=bits).astype(mx.float32)
                row = {"bits": bits, "shape": name, "M": M, "m1_us": round(t1 * 1e6, 1), "shipped_x": round(tr / t1, 2)}
                for tag, cfg in configs.items():
                    try:
                        got = fused(x, *mats[0], bits, 64, **cfg).astype(mx.float32)
                        err = float((mx.max(mx.abs(ref - got)) / (mx.max(mx.abs(ref)) + 1e-6)).item())
                        t = cold(lambda q: fused(x, *q, bits, 64, **cfg), mats)
                        row[tag] = {"x": round(t / t1, 2), "err": round(err, 4)}
                    except Exception as e:  # noqa: BLE001
                        row[tag] = {"error": [l[:300] for l in str(e).splitlines() if "error" in l][:3]}
                print(json.dumps(row), flush=True)
            del mats
            mx.clear_cache()


if __name__ == "__main__":
    main()
