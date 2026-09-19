"""Stage 2: NAX consumption rate with the weight operand in threadgroup memory.

Each threadgroup fills a TN x KC fp16 weight tile in threadgroup memory once, then
runs matmul2d on it R times (accumulating), with x from device (tiny, cached).
Reports weights consumed per second by NAX = grid_tgs * R * TN * KC / t.
The bar: 4-bit weights streamed at DRAM bandwidth need ~1.0e12 weights/s.
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
    constexpr int R = {R};
    const int M = m_dim[0];
    uint tg = threadgroup_position_in_grid.x;
    uint tix = thread_index_in_threadgroup;

    threadgroup half Ws[TN * KC];
    for (int i = tix; i < TN * KC; i += {NT}) {{
        Ws[i] = half(0.001f * float((i * 7 + int(tg)) % 97));
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr auto desc = matmul2d_descriptor(
        TM, TN, KC, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<{SG}>> op;

    auto A = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        (device half*)x, dextents<int32_t, 2>(KC, M));
    auto B = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        Ws, dextents<int32_t, 2>(KC, TN));
    auto cT = op.get_destination_cooperative_tensor<decltype(A), decltype(B), float>();
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {{
        if (cT.is_valid_element(i)) cT[i] = 0;
    }}
    for (int r = 0; r < R; r++) {{
        op.run(A, B, cT);
    }}
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {{
        if (cT.is_valid_element(i)) {{
            auto idx = cT.get_multidimensional_index(i);
            int n = int(tg) * TN + idx[0];
            int m = idx[1];
            if (m < M) out[size_t(m) * (TN * {TGS}) + n] = cT[i];
        }}
    }}
"""


def run(tm, tn, kc, sg, tgs, M, R):
    nt = 32 * sg
    k = mx.fast.metal_kernel(
        name=f"nax_s2_tm{tm}_tn{tn}_kc{kc}_sg{sg}_r{R}_g{tgs}",
        input_names=["x", "m_dim"], output_names=["out"], header=HEADER,
        source=SRC.format(TM=tm, TN=tn, KC=kc, R=R, SG=sg, NT=nt, TGS=tgs))
    x = mx.random.normal((M, kc)).astype(mx.float16)
    mx.eval(x)

    def f():
        return k(inputs=[x, mx.array([M], dtype=mx.int32)], grid=(tgs * nt, 1, 1), threadgroup=(nt, 1, 1),
                 output_shapes=[(M, tn * tgs)], output_dtypes=[mx.float32])[0]

    for _ in range(3):
        mx.eval([f() for _ in range(8)])
    s = []
    for _ in range(8):
        t0 = time.perf_counter(); mx.eval([f() for _ in range(8)]); s.append((time.perf_counter() - t0) / 8)
    t = statistics.median(s)
    return t, tgs * R * tn * kc / t


def main():
    assert "--i-own-the-gpu" in sys.argv
    tgs = 40 * 16  # plenty of threadgroups per core
    for M in (8, 16):
        for tm, tn, kc, sg in ((8, 32, 256, 4), (16, 32, 256, 4), (16, 64, 128, 4), (16, 32, 256, 2),
                               (16, 64, 256, 4), (32, 32, 256, 4)):
            if M > tm or tn * kc * 2 > 32768:
                continue
            for R in (1, 16, 64):
                try:
                    t, wps = run(tm, tn, kc, sg, tgs, M, R)
                    print(json.dumps({"M": M, "tm": tm, "tn": tn, "kc": kc, "sg": sg, "R": R,
                                      "us": round(t * 1e6, 1), "weights_per_s": f"{wps:.3e}",
                                      "useful_tflops": round(2 * M * wps / 1e12, 1)}), flush=True)
                except Exception as e:  # noqa: BLE001
                    print(json.dumps({"M": M, "tm": tm, "tn": tn, "kc": kc, "sg": sg, "R": R, "error": str(e)[:400]}), flush=True)


if __name__ == "__main__":
    main()
