"""Synthetic MPP matmul2d tile probe: where is the FP16 tile pipeline floor?

No GGUF decode, no routing: an FP16 GEMM y[M,N] = x[M,K] . w[N,K]^T shaped
like one GLM MoE layer's gate projection at a 512-row prefill chunk, through a
generated kernel family. Each variant fixes:

* TM x TN threadgroup tile, one execution_simdgroups<SG> op, K step BK;
* A (activations) and B (weights) each read in place as a device tensor or
  staged through threadgroup memory (our K-block weights must be staged);
* staged B single- or double-buffered (one barrier per K step when double).

Accumulation is multiply_accumulate into a persistent destination
cooperative tensor (FP32). mx.matmul on the same FP16 operands is the
"achievable" reference. Timings are in-graph (several calls per evaluation).
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""


def source(tm, tn, bk, sg, a_dev, b_dev, double):
    threads = 32 * sg
    lines = [f"""
constexpr int TM = {tm}, TN = {tn}, BK = {bk};
const int tid = int(thread_index_in_threadgroup);
const int row0 = int(threadgroup_position_in_grid.y) * TM;
const int col0 = int(threadgroup_position_in_grid.x) * TN;
constexpr auto desc = matmul2d_descriptor(TM, TN, BK, false, true, false,
    matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc, execution_simdgroups<{sg}>> op;
tensor<device half, dextents<int32_t, 2>, tensor_inline> tA((device half*)x + int64_t(row0) * K,
    dextents<int32_t, 2>(K, M - row0));
tensor<device half, dextents<int32_t, 2>, tensor_inline> tB((device half*)w + int64_t(col0) * K,
    dextents<int32_t, 2>(K, N - col0));
"""]
    nbuf = 2 if double else 1
    if not a_dev:
        lines.append(f"threadgroup half sa[{nbuf}][TM * BK];\n")
    if not b_dev:
        lines.append(f"threadgroup half sb[{nbuf}][TN * BK];\n")
    a0 = "tA.slice(0, 0)" if a_dev else "tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[0], dextents<int32_t, 2>(BK, TM))"
    b0 = "tB.slice(0, 0)" if b_dev else "tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb[0], dextents<int32_t, 2>(BK, TN))"
    lines.append(f"""
auto a00 = {a0};
auto b00 = {b0};
auto c = op.template get_destination_cooperative_tensor<decltype(a00), decltype(b00), float>();
for (uint16_t j = 0; j < c.get_capacity(); ++j) if (c.is_valid_element(j)) c[j] = 0.0f;
""")
    stage = []
    if not a_dev:
        stage.append(f"""
    for (int i = tid; i < TM * BK; i += {threads}) {{
        const int r = row0 + i / BK; sa[SLOT][i] = r < M ? x[ulong(r) * K + KB + i % BK] : half(0.0f);
    }}""")
    if not b_dev:
        stage.append(f"""
    for (int i = tid; i < TN * BK; i += {threads}) {{
        const int r = col0 + i / BK; sb[SLOT][i] = r < N ? w[ulong(r) * K + KB + i % BK] : half(0.0f);
    }}""")
    stage_code = "".join(stage)
    a_k = "tA.slice(kb, 0)" if a_dev else "tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[s], dextents<int32_t, 2>(BK, TM))"
    b_k = "tB.slice(kb, 0)" if b_dev else "tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb[s], dextents<int32_t, 2>(BK, TN))"
    if stage_code and double:
        lines.append(stage_code.replace("SLOT", "0").replace("KB", "0"))
        lines.append("\nthreadgroup_barrier(mem_flags::mem_threadgroup);\n")
        lines.append(f"""
for (int kb = 0, s = 0; kb < K; kb += BK, s ^= 1) {{
    if (kb + BK < K) {{ {stage_code.replace("SLOT", "s ^ 1").replace("KB", "(kb + BK)")} }}
    auto a = {a_k};
    auto b = {b_k};
    op.run(a, b, c);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}}
""")
    elif stage_code:
        lines.append(f"""
for (int kb = 0, s = 0; kb < K; kb += BK) {{
    {stage_code.replace("SLOT", "0").replace("KB", "kb")}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto a = {a_k};
    auto b = {b_k};
    op.run(a, b, c);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}}
""")
    else:
        lines.append(f"""
for (int kb = 0, s = 0; kb < K; kb += BK) {{
    auto a = {a_k};
    auto b = {b_k};
    op.run(a, b, c);
}}
""")
    lines.append("""
for (uint16_t j = 0; j < c.get_capacity(); ++j) {
    if (c.is_valid_element(j)) {
        const auto index = c.get_multidimensional_index(j);
        const int col = col0 + index[0], row = row0 + index[1];
        if (col < N && row < M) y[ulong(row) * N + col] = c[j];
    }
}
""")
    return "".join(lines), threads


def register_source(tmr, sgs):
    """B fed from registers (lane matmul's right-input fragment map), A in place.

    One execution_simdgroup op per simdgroup: 16*TMR rows x 32 columns x 64 K;
    each of SGS simdgroups owns its own 32-column block, so a threadgroup
    covers 32*SGS columns with no threadgroup memory and no barriers.
    """
    return f"""
constexpr int TMR = {tmr}, NT = 32, GS = 64;
const ushort lane = thread_index_in_simdgroup;
const ushort sg = simdgroup_index_in_threadgroup;
const int rb = int(threadgroup_position_in_grid.y) * 16 * TMR;
const int n0 = (int(threadgroup_position_in_grid.x) * {sgs} + int(sg)) * NT;
constexpr auto desc = matmul2d_descriptor(16 * TMR, NT, GS, false, true, false,
    matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc, execution_simdgroup> op;
tensor<device half, dextents<int32_t, 2>, tensor_inline> tA((device half*)x + int64_t(rb) * K,
    dextents<int32_t, 2>(K, M - rb));
const int bqi = (lane & 1) | (((lane >> 3) & 1) << 1);
const int bnb = ((lane >> 1) & 3) | (((lane >> 4) & 1) << 2);
auto a0 = tA.slice(0, 0);
auto cb0 = op.template get_right_input_cooperative_tensor<half, half, float>();
auto c = op.template get_destination_cooperative_tensor<decltype(a0), metal::remove_addrspace_t<decltype(cb0)>, float>();
for (uint16_t j = 0; j < c.get_capacity(); ++j) if (c.is_valid_element(j)) c[j] = 0.0f;
for (int g = 0; g < K / GS; ++g) {{
    auto cb = op.template get_right_input_cooperative_tensor<half, half, float>();
    for (int j = 0; j < 4; ++j) {{
        const int col = n0 + bnb + 8 * j;
        const device half* wr = w + ulong(col < N ? col : 0) * K + g * GS + 4 * bqi;
        if (VEC) {{
            for (int kb = 0; kb < 4; ++kb) {{
                const half4 v = col < N ? *((const device half4*)(wr + kb * 16)) : half4(0.0h);
                cb[kb * 16 + j * 4 + 0] = v.x; cb[kb * 16 + j * 4 + 1] = v.y;
                cb[kb * 16 + j * 4 + 2] = v.z; cb[kb * 16 + j * 4 + 3] = v.w;
            }}
        }} else
        for (int kb = 0; kb < 4; ++kb)
            for (int kk = 0; kk < 4; ++kk)
                cb[kb * 16 + j * 4 + kk] = col < N ? wr[kb * 16 + kk] : half(0.0f);
    }}
    auto a = tA.slice(g * GS, 0);
    op.run(a, cb, c);
}}
for (uint16_t j = 0; j < c.get_capacity(); ++j) {{
    if (c.is_valid_element(j)) {{
        const auto index = c.get_multidimensional_index(j);
        const int col = n0 + index[0], row = rb + index[1];
        if (col < N && row < M) y[ulong(row) * N + col] = c[j];
    }}
}}
""", 32 * sgs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--source-revision", default="unrecorded")
    parser.add_argument("--register-only", action="store_true",
                        help="only mx.matmul and the register-fed-B family")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    import mlx.core as mx

    info = mx.device_info()
    if info.get("max_recommended_working_set_size"):
        mx.set_wired_limit(info["max_recommended_working_set_size"])
    m, n, k = args.m, args.n, args.k
    x = mx.random.normal((m, k)).astype(mx.float16)
    w = mx.random.normal((n, k)).astype(mx.float16)
    mx.eval(x, w)
    flops = 2.0 * m * n * k
    ref = (x.astype(mx.float32) @ w.astype(mx.float32).T)

    def timed(fn):
        mx.eval(fn())
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            mx.eval([fn() for _ in range(args.reps)])
            best = min(best, (time.perf_counter() - t0) / args.reps)
        return best

    rows = []
    t = timed(lambda: x @ w.T)
    rows.append({"variant": "mx.matmul fp16 (fp16 out)", "ms": t * 1e3, "tflop_s": flops / t / 1e12})
    print(f"{'mx.matmul fp16':52s} {t*1e3:7.3f} ms {flops/t/1e12:6.2f} TF/s", flush=True)
    configs = []
    if args.register_only:
        configs = None
    for (tm, tn), sg, bk, a_dev, b_dev, double in ([] if configs is None else itertools.product(
            [(32, 64), (64, 64), (64, 128), (128, 128)], (4, 8), (64, 128), (True, False), (True, False), (True, False))):
        if a_dev and b_dev and double:
            continue  # nothing staged: buffering is meaningless
        staged = (0 if a_dev else tm * bk) + (0 if b_dev else tn * bk)
        if staged * 2 * (2 if double else 1) > 32768:
            continue  # threadgroup memory limit
        if (tm, tn) == (128, 128) and sg == 4:
            continue
        configs.append((tm, tn, bk, sg, a_dev, b_dev, double))
    for tm, tn, bk, sg, a_dev, b_dev, double in (configs or []):
        label = (f"TM{tm} TN{tn} BK{bk} SG{sg} A:{'dev' if a_dev else 'tg'} B:{'dev' if b_dev else 'tg'}"
                 f"{' x2buf' if double and not (a_dev and b_dev) else ''}")
        src, threads = source(tm, tn, bk, sg, a_dev, b_dev, double)
        try:
            kernel = mx.fast.metal_kernel(
                name="mpp_probe_" + label.replace(" ", "_").replace(":", ""),
                input_names=["x", "w"], output_names=["y"], header=HEADER, source=src,
                ensure_row_contiguous=True)
            call = lambda kernel=kernel, tm=tm, tn=tn, threads=threads: kernel(
                inputs=[x, w], template=[("M", m), ("N", n), ("K", k)],
                grid=(((n + tn - 1) // tn) * threads, (m + tm - 1) // tm, 1), threadgroup=(threads, 1, 1),
                output_shapes=[(m, n)], output_dtypes=[mx.float32])[0]
            out = call()
            err = float(mx.max(mx.abs(out - ref)).item())
            t = timed(call)
            rows.append({"variant": label, "tm": tm, "tn": tn, "bk": bk, "sg": sg, "a_device": a_dev,
                         "b_device": b_dev, "double_buffer": double, "ms": t * 1e3,
                         "tflop_s": flops / t / 1e12, "max_abs_err_vs_fp32": err})
            print(f"{label:52s} {t*1e3:7.3f} ms {flops/t/1e12:6.2f} TF/s  err {err:.2e}", flush=True)
        except Exception as error:  # noqa: BLE001 -- a refused variant is a result too
            rows.append({"variant": label, "error": str(error)[:300]})
            print(f"{label:52s} FAILED {str(error)[:120]}", flush=True)
    for tmr, sgs, vec in itertools.product((2, 4), (4, 8), (0, 1)):
        label = f"REG-B TM{16 * tmr} TN32/sg SG{sgs} A:dev B:regs{' half4' if vec else ''}"
        src, threads = register_source(tmr, sgs)
        try:
            kernel = mx.fast.metal_kernel(
                name="mpp_probe_regb_" + str(tmr) + "_" + str(sgs) + "_" + str(vec), input_names=["x", "w"],
                output_names=["y"],
                header=HEADER, source=src, ensure_row_contiguous=True)
            call = lambda kernel=kernel, tmr=tmr, sgs=sgs, threads=threads, vec=vec: kernel(
                inputs=[x, w], template=[("M", m), ("N", n), ("K", k), ("VEC", vec)],
                grid=(((n + 32 * sgs - 1) // (32 * sgs)) * threads, (m + 16 * tmr - 1) // (16 * tmr), 1),
                threadgroup=(threads, 1, 1), output_shapes=[(m, n)], output_dtypes=[mx.float32])[0]
            out = call()
            err = float(mx.max(mx.abs(out - ref)).item())
            t = timed(call)
            rows.append({"variant": label, "tmr": tmr, "sg": sgs, "register_b": True, "ms": t * 1e3,
                         "tflop_s": flops / t / 1e12, "max_abs_err_vs_fp32": err})
            print(f"{label:52s} {t*1e3:7.3f} ms {flops/t/1e12:6.2f} TF/s  err {err:.2e}", flush=True)
        except Exception as error:  # noqa: BLE001 -- a refused variant is a result too
            rows.append({"variant": label, "error": str(error)[:300]})
            print(f"{label:52s} FAILED {str(error)[:160]}", flush=True)
    (args.out / "probe.json").write_text(json.dumps(
        {"schema": "mlx2.mpp-tile-probe.v1", "source_revision": args.source_revision, "device": info,
         "shape": [m, n, k], "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
