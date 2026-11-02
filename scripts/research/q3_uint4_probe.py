"""Synthetic probe: Q3 codes read in place as MPP uint4b, folded per scale group.

Question: can prefill feed Q3_K-like weights to the M5 tensor units without
decoding them through threadgroup memory? Q3_K values are ``d*sc*(q-4)`` with
codes ``q in 0..7`` and one scale per 16 K. Stored as nibbles (a lossless
container), the op reads codes in place (``uint4b_format``, as
``prefill_metal_q4native``) and returns the unscaled fp32 product per group
``P[m, n] = x[m, group] . code[n, group]``; the kernel folds

    y[m, n] = sum over g of  s[n, g] * P[m, n, g] + b[n, g] * xs[m, g]

with ``s = d*sc`` and ``b = -4*d*sc``. ``xs`` (fp32 group input sums) is a
precomputed input shared by every output tile. Each variant fixes:

* GS: group size = MMA K step (64 is q4native, 16 is Q3_K's scale group);
* KB: K covered per barrier pair (scales for KB/GS groups staged at once);
* TN: columns per threadgroup (TM = 32 rows, four simdgroups).

``nofold`` is the same in-place uint4 read with multiply_accumulate and no
scales: the MMA floor this pipeline can reach. Codes are synthetic and row
major in K (two per byte, low nibble first); the GGUF Q3N container would need
this order. Reference is fp64; ``fp16 dequant mx.matmul`` gives the stock
error for the fp64-RMS ratio (exactness class evidence).
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
using namespace metal;
"""


def source(gs, kb, tn, xdt, fold, dyn=False, signed=False, unroll=1):
    wfmt = "int4b_format" if signed else "uint4b_format"
    # Dynamic K: sub-byte static extents must be multiples of 32, so the
    # GS-wide view is built from a pointer with runtime extent GS and row stride K.
    # Unroll U groups: U independent MMAs into U destinations, then one fold
    # pass, so the tensor unit is not idle while each result is read back.
    runs, terms = [], []
    for u in range(unroll):
        if dyn:
            runs.append(f"""const int kk{u} = k0 + (g + {u}) * GS;
            tensor<device {xdt}, dextents<int32_t, 2>, tensor_inline> a{u}(
                (device {xdt}*)X + int64_t(row0) * K + kk{u}, dextents<int32_t, 2>(GS, M - row0),
                array<int32_t, 2>{{{{1, K}}}});
            tensor<device {wfmt}, dextents<int32_t, 2>, tensor_inline> b{u}(
                (device uchar*)W + int64_t(col0) * (K / 2) + kk{u} / 2, dextents<int32_t, 2>(GS, N - col0),
                array<int32_t, 2>{{{{1, K}}}});""")
        else:
            runs.append(f"auto a{u} = tA.slice(k0 + (g + {u}) * GS, 0);\n            auto b{u} = tB.slice(k0 + (g + {u}) * GS, 0);")
        runs.append(f"auto P{u} = op.get_destination_cooperative_tensor<decltype(a{u}), decltype(b{u}), float>();\n"
                    f"            op.run(a{u}, b{u}, P{u});")
        terms.append(f"acc = fma(sb[g + {u}][n].x, P{u}[j], acc);" if signed else
                     f"acc = fma(sb[g + {u}][n].x, P{u}[j], fma(sb[g + {u}][n].y, xsum[g + {u}][m], acc));")
    view = "\n            ".join(runs)
    fold_terms = "\n                    ".join(terms)
    kdesc = "static_cast<int>(dynamic_extent)" if dyn else "GS"
    sl = "slice"
    if not fold:
        return f"""
    constexpr int TM = 32, TN = {tn}, BK = {gs};
    const int row0 = int(threadgroup_position_in_grid.x) * TM;
    const int col0 = int(threadgroup_position_in_grid.y) * TN;
    constexpr auto desc = matmul2d_descriptor(TM, TN, BK, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    tensor<device {xdt}, dextents<int32_t, 2>, tensor_inline> tA(
        (device {xdt}*)X + int64_t(row0) * K, dextents<int32_t, 2>(K, M - row0));
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tB(
        (device uchar*)W + int64_t(col0) * (K / 2), dextents<int32_t, 2>(K, N - col0));
    auto a0 = tA.slice(0, 0);
    auto b0 = tB.slice(0, 0);
    auto c = op.get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    for (uint16_t j = 0; j < c.get_capacity(); ++j)
        if (c.is_valid_element(j)) c[j] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += BK) {{
        auto a = tA.slice(k0, 0);
        auto b = tB.slice(k0, 0);
        op.run(a, b, c);
    }}
    for (uint16_t j = 0; j < c.get_capacity(); ++j) {{
        if (c.is_valid_element(j)) {{
            const auto index = c.get_multidimensional_index(j);
            const int col = col0 + index[0], row = row0 + index[1];
            if (col < N && row < M) Y[size_t(row) * N + col] = c[j];
        }}
    }}
    """
    return f"""
    constexpr int TM = 32, TN = {tn}, GS = {gs}, KB = {kb}, NG = KB / GS, KG = K / GS, U = {unroll};
    static_assert(K % KB == 0 && KB % GS == 0 && NG % U == 0, "K must tile");
    const int tid = int(thread_index_in_threadgroup);
    const int row0 = int(threadgroup_position_in_grid.x) * TM;
    const int col0 = int(threadgroup_position_in_grid.y) * TN;
    constexpr auto desc = matmul2d_descriptor(TM, TN, {kdesc}, false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<4>> op;
    tensor<device {xdt}, dextents<int32_t, 2>, tensor_inline> tA(
        (device {xdt}*)X + int64_t(row0) * K, dextents<int32_t, 2>(K, M - row0));
    tensor<device {wfmt}, dextents<int32_t, 2>, tensor_inline> tB(
        (device uchar*)W + int64_t(col0) * (K / 2), dextents<int32_t, 2>(K, N - col0));
    auto a0 = tA.template {sl}(0, 0);
    auto b0 = tB.template {sl}(0, 0);
    auto c = op.get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    for (uint16_t j = 0; j < c.get_capacity(); ++j)
        if (c.is_valid_element(j)) c[j] = 0.0f;
    threadgroup float xsum[NG][TM];
    threadgroup float2 sb[NG][TN];
    for (int k0 = 0; k0 < K; k0 += KB) {{
        const int g0 = k0 / GS;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = tid; e < NG * TM; e += 128) {{
            const int g = e / TM, r = e % TM;
            xsum[g][r] = row0 + r < M ? XS[size_t(row0 + r) * KG + g0 + g] : 0.0f;
        }}
        for (int e = tid; e < NG * TN; e += 128) {{
            const int g = e / TN, n = e % TN, col = col0 + n;
            sb[g][n] = col < N ? float2(S[size_t(col) * KG + g0 + g], B[size_t(col) * KG + g0 + g])
                               : float2(0.0f);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int g = 0; g < NG; g += U) {{
            {view}
            for (uint16_t j = 0; j < c.get_capacity(); ++j) {{
                if (c.is_valid_element(j)) {{
                    const auto index = c.get_multidimensional_index(j);
                    const int n = index[0], m = index[1];
                    (void)m;
                    float acc = c[j];
                    {fold_terms}
                    c[j] = acc;
                }}
            }}
        }}
    }}
    for (uint16_t j = 0; j < c.get_capacity(); ++j) {{
        if (c.is_valid_element(j)) {{
            const auto index = c.get_multidimensional_index(j);
            const int col = col0 + index[0], row = row0 + index[1];
            if (col < N && row < M) Y[size_t(row) * N + col] = c[j];
        }}
    }}
    """


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shape", action="append", default=None,
                        help="MxNxK; repeatable (default 1024x4096x4096 and 64x1536x4096)")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--gs", type=int, nargs="+", default=[64, 32, 16])
    parser.add_argument("--kb", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--tn", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--unroll", type=int, nargs="+", default=[1])
    parser.add_argument("--kinds", nargs="+", default=None)
    parser.add_argument("--no-nofold", action="store_true")
    parser.add_argument("--source-revision", default="unrecorded")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    import mlx.core as mx
    import numpy as np

    info = mx.device_info()
    if info.get("max_recommended_working_set_size"):
        mx.set_wired_limit(info["max_recommended_working_set_size"])

    def timed(fn, reps):
        mx.eval(fn())
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            mx.eval([fn() for _ in range(reps)])
            best = min(best, (time.perf_counter() - t0) / reps)
        return best

    kernels = {}
    rows = []
    for shape in args.shape or ["1024x4096x4096", "64x1536x4096"]:
        m, n, k = (int(v) for v in shape.split("x"))
        rng = np.random.default_rng(0)
        x16 = rng.standard_normal((m, k)).astype(np.float16)
        codes = rng.integers(0, 8, (n, k), dtype=np.uint8)
        packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
        signed_codes = (codes.astype(np.int16) - 4) & 15  # two's complement int4 of q - 4
        packed_s = (signed_codes[:, 0::2] | (signed_codes[:, 1::2] << 4)).astype(np.uint8)
        flops = 2.0 * m * n * k
        print(f"shape M{m} N{n} K{k}", flush=True)
        for gs in args.gs:
            kg = k // gs
            s = (rng.uniform(0.5, 1.5, (n, kg)) * 1e-2).astype(np.float32)
            b = (-4.0 * s).astype(np.float32)
            wfull = (np.repeat(s, gs, 1).astype(np.float64) * (codes.astype(np.float64) - 4.0))
            ref = x16.astype(np.float64) @ wfull.T
            stock = np.array((mx.array(x16) @ mx.array(wfull.astype(np.float16)).T).astype(mx.float32))
            stock_rms = float(np.sqrt(np.mean((stock - ref) ** 2)))
            xs = x16.astype(np.float32).reshape(m, kg, gs).sum(-1, dtype=np.float32)
            inputs = {"X": mx.array(x16), "W": mx.array(packed.reshape(-1)), "WS": mx.array(packed_s.reshape(-1)),
                      "S": mx.array(s.reshape(-1)), "B": mx.array(b.reshape(-1)), "XS": mx.array(xs.reshape(-1))}
            mx.eval(list(inputs.values()))
            kinds = args.kinds or (("fold", "sfold") if gs >= 32 else ("dfold", "dsfold"))
            variants = [(kind, gs, kb, tn, u) for kind, kb, tn, u in itertools.product(
                [kd for kd in kinds if gs >= 32 or kd.startswith("d")], args.kb, args.tn, args.unroll)]
            if gs == 64 and not args.no_nofold:
                variants = [("nofold", 64, 64, tn, 1) for tn in args.tn] + variants
            for kind, g2, kb, tn, u in variants:
                if kb < g2 * u or (kb // g2) % u:
                    continue
                fold, dyn, signed = kind != "nofold", kind.startswith("d"), "s" in kind[:3]
                w = inputs["WS"] if signed else inputs["W"]
                label = f"{kind} GS{g2} KB{kb} TN{tn} U{u} half"
                key = (kind, g2, kb, tn, u)
                try:
                    if key not in kernels:
                        kernels[key] = mx.fast.metal_kernel(
                            name=f"q3u4_{kind}_{g2}_{kb}_{tn}_{u}",
                            input_names=["X", "W", "S", "B", "XS"] if fold else ["X", "W"],
                            output_names=["Y"], header=HEADER, source=source(g2, kb, tn, "half", fold, dyn, signed, u),
                            ensure_row_contiguous=True)
                    kernel = kernels[key]
                    names = ["X", "W", "S", "B", "XS"] if fold else ["X", "W"]
                    call = lambda kernel=kernel, names=names, tn=tn, inputs=inputs, m=m, n=n, k=k, w=w: kernel(
                        inputs=[w if v == "W" else inputs[v] for v in names], template=[("M", m), ("N", n), ("K", k)],
                        grid=(((m + 31) // 32) * 128, (n + tn - 1) // tn, 1), threadgroup=(128, 1, 1),
                        output_shapes=[(m, n)], output_dtypes=[mx.float32])[0]
                    out = np.array(call()).astype(np.float64)
                    if fold:
                        rms = float(np.sqrt(np.mean((out - ref) ** 2)))
                        err = {"fp64_rms": rms, "stock_fp16_rms": stock_rms, "ratio": rms / stock_rms}
                    else:
                        ref0 = x16.astype(np.float64) @ codes.astype(np.float64).T
                        err = {"max_abs_vs_fp64": float(np.max(np.abs(out - ref0)))}
                    t = timed(call, args.reps)
                    row = {"shape": [m, n, k], "variant": label, "ms": t * 1e3, "tflop_s": flops / t / 1e12, **err}
                    rows.append(row)
                    tail = (f"fp64-rms ratio {err['ratio']:.4f}" if fold
                            else f"max_abs {err['max_abs_vs_fp64']:.2e}")
                    print(f"  {label:34s} {t*1e3:7.3f} ms {flops/t/1e12:6.2f} TF/s  {tail}", flush=True)
                except Exception as error:  # noqa: BLE001 -- a refused variant is a result too
                    rows.append({"shape": [m, n, k], "variant": label, "error": str(error)[:400]})
                    print(f"  {label:34s} FAILED {str(error)[:200]}", flush=True)
        xh = mx.array(x16)
        wh = mx.array(rng.standard_normal((n, k)).astype(np.float16))
        mx.eval(xh, wh)
        t = timed(lambda xh=xh, wh=wh: xh @ wh.T, args.reps)
        rows.append({"shape": [m, n, k], "variant": "mx.matmul fp16", "ms": t * 1e3, "tflop_s": flops / t / 1e12})
        print(f"  {'mx.matmul fp16 (dense ref)':34s} {t*1e3:7.3f} ms {flops/t/1e12:6.2f} TF/s", flush=True)
    (args.out / "probe.json").write_text(json.dumps(
        {"schema": "mlx2.q3-uint4-probe.v1", "source_revision": args.source_revision, "device": info,
         "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
