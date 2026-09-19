"""Stage 1: NAX (MPP matmul2d) throughput at small M, fp16 operands.

y[M, N] = x[M, K] @ W[N, K]^T with one threadgroup per TN output columns and a
TM-row (padded) M tile. Measures warm (compute-bound, small W reused) and cold
(DRAM-bound, rotating W copies) against mx.matmul and the bandwidth floor.
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
    const int M = m_dim[0];
    const int N = n_dim[0];
    const int K = k_dim[0];
    uint2 tgid = threadgroup_position_in_grid.xy;

    constexpr auto desc = matmul2d_descriptor(
        TM, TN, static_cast<int>(dynamic_extent),
        /*transpose_left=*/false, /*transpose_right=*/true,
        /*relaxed_precision=*/{RELAX},
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<{SG}>> op;

    auto A = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        (device half*)x, dextents<int32_t, 2>(K, M));
    auto B = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        (device half*)w, dextents<int32_t, 2>(K, N));
    auto tA = A.slice(0, 0);
    auto tB = B.slice(0, int(tgid.x) * TN);
    auto cT = op.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {{
        if (cT.is_valid_element(i)) cT[i] = 0;
    }}
    op.run(tA, tB, cT);
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {{
        if (cT.is_valid_element(i)) {{
            auto idx = cT.get_multidimensional_index(i);
            int n = int(tgid.x) * TN + idx[0];
            int m = idx[1];
            if (m < M && n < N) out[size_t(m) * N + n] = half(cT[i]);
        }}
    }}
"""

_K = {}


def nax_mm(x, w, tm, tn, sg=4, relax=True):
    M, K = x.shape
    N = w.shape[0]
    key = (tm, tn, sg, relax)
    if key not in _K:
        _K[key] = mx.fast.metal_kernel(
            name=f"nax_s1_tm{tm}_tn{tn}_sg{sg}_r{int(relax)}",
            input_names=["x", "w", "m_dim", "n_dim", "k_dim"], output_names=["out"],
            header=HEADER, source=SRC.format(TM=tm, TN=tn, SG=sg, RELAX="true" if relax else "false"))
    (y,) = _K[key](inputs=[x, w, mx.array([M], dtype=mx.int32), mx.array([N], dtype=mx.int32),
                           mx.array([K], dtype=mx.int32)],
                   grid=(N // tn * 32 * sg, 1, 1), threadgroup=(32 * sg, 1, 1),
                   output_shapes=[(M, N)], output_dtypes=[mx.float16])
    return y


def bench(fn, mats, reps=6):
    run = lambda: [fn(q) for q in mats]
    for _ in range(2):
        mx.eval(run())
    s = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(run()); s.append((time.perf_counter() - t0) / len(mats))
    return statistics.median(s)


def main():
    assert "--i-own-the-gpu" in sys.argv
    configs = [(8, 32), (16, 32), (16, 64), (32, 32), (32, 64)]
    for regime, (N, K), copies in (("warm", (4096, 4096), 1), ("cold", (8192, 5120), 12), ("cold", (17408, 5120), 6)):
        mats = [(mx.random.normal((N, K)) * 0.02).astype(mx.float16) for _ in range(copies)]
        mx.eval(mats)
        for M in (1, 4, 8, 16):
            x = mx.random.normal((M, K)).astype(mx.float16)
            mx.eval(x)
            t_ref = bench(lambda w: x @ w.T, mats)
            row = {"regime": regime, "N": N, "K": K, "M": M, "mx_matmul_us": round(t_ref * 1e6, 1),
                   "bw_floor_us": round(N * K * 2 / 520e9 * 1e6, 1)}
            ref = (x @ mats[0].T).astype(mx.float32)
            for tm, tn in configs:
                if M > tm:
                    continue
                try:
                    got = nax_mm(x, mats[0], tm, tn).astype(mx.float32)
                    err = float((mx.max(mx.abs(ref - got)) / (mx.max(mx.abs(ref)) + 1e-6)).item())
                    t = bench(lambda w: nax_mm(x, w, tm, tn), mats)
                    tflops = 2 * M * N * K / t / 1e12
                    tflops_padded = 2 * tm * N * K / t / 1e12
                    row[f"tm{tm}_tn{tn}"] = {"us": round(t * 1e6, 1), "err": round(err, 5),
                                             "tflops": round(tflops, 2), "tflops_padded": round(tflops_padded, 2)}
                except Exception as e:  # noqa: BLE001
                    row[f"tm{tm}_tn{tn}"] = {"error": str(e)[:300]}
            print(json.dumps(row), flush=True)
        del mats
        mx.clear_cache()


if __name__ == "__main__":
    main()
