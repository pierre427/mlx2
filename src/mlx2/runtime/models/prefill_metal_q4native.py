# SPDX-License-Identifier: MIT
"""Native-q4 M5 prefill tiles; original mlx2 implementation.

The opt-in ``metal_q4native`` TensorFold prefill backend.  It keeps the
``metal`` backend's tile (32 rows x 64 columns per threadgroup, four
simdgroups, one 64-wide quantization group per K step) and launch grid, but
changes how the weights reach the M5 tensor units and therefore the
numerical law:

* ``metal`` (``prefill_metal.py``): every 64x64 weight block is dequantized
  to ``bfloat(q * s + b)`` in threadgroup memory and multiplied by the staged
  bf16 activations, accumulating in fp32 inside the tensor op.
* ``metal_q4native`` (this file): MPP reads MLX's packed nibbles in place as
  ``uint4b_format`` (no threadgroup staging of weights or activations).  For
  group ``g`` the op returns the unscaled fp32 product
  ``P[m, n] = x[m, g-block] . q[n, g-block]`` and the kernel applies the
  affine law in registers, groups in order::

      y[m, n] = sum over g of  s[n, g] * P[m, n, g] + b[n, g] * xs[m, g]

  where ``xs[m, g]`` is the fp32 sum of the group's 64 inputs (four quarter
  sums in index order, combined as ``(q0 + q1) + (q2 + q3)``).  This is the
  lane matmul's affine law (``runtime/lane/matmul.py``) at prefill tile
  shape; the weights are never rounded to bf16.

Why device-tensor reads and not a register-decoded right input: MPP accepts
input cooperative tensors only with ``execution_simdgroup`` scope (SDK 27
``__get_right_input_cooperative_tensor`` static_assert), while this tile runs
the op over ``execution_simdgroups<4>``.  The in-place ``uint4b_format`` read
is the WWDC26 session 330 "read quantized weights natively" path and keeps
the established 4-simdgroup tile.

``relaxed_precision`` is false here (the ``metal`` tile sets it true), so the
law does not depend on what the implementation chooses to relax.

The ``metal`` generator and its sources are untouched; this module shares no
code with it so that its file hash (part of the Nanbeige kernel contract)
does not move.
"""

HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
using namespace metal;
"""
TILE_M, TILE_N, TILE_K, THREADS = 32, 64, 64, 128
GROUP_SIZE = TILE_K
# Numerical-law identity carried in the install receipt's "kernel" field and
# so in the prefill execution identity / APCv2 prefill fingerprint.
KERNEL_IDENTITY = "q4g64_prefill_m32n64k64_nax_uint4b_affine_v1"


def _body(width, indices, biases, output, tile_offset=0):
    declarations, initializes, stages, runs, accumulates = [], [], [], [], []
    for p, i in enumerate(indices):
        declarations.append(f"""
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tB{p}(
        (device uchar*)W{i} + int64_t(col0) * (K / 2), dextents<int32_t, 2>(K, N - col0));
    auto b{p}0 = tB{p}.slice(0, 0);
    auto c{p} = op.get_destination_cooperative_tensor<decltype(a0), decltype(b{p}0), float>();
    threadgroup float sb{p}[2][TN][2];
        """)
        initializes.append(f"""
    for (uint16_t j = 0; j < c{p}.get_capacity(); ++j)
        if (c{p}.is_valid_element(j)) c{p}[j] = 0.0f;
        """)
        stages.append(f"""
        if (tid < TN) {{
            const int col = col0 + tid;
            sb{p}[g & 1][tid][0] = col < N ? float(S{i}[col * KG + g]) : 0.0f;
            sb{p}[g & 1][tid][1] = col < N ? float(B{i}[col * KG + g]) : 0.0f;
        }}
        """)
        runs.append(f"""
        auto b{p} = tB{p}.slice(g * GS, 0);
        auto P{p} = op.get_destination_cooperative_tensor<decltype(a), decltype(b{p}), float>();
        op.run(a, b{p}, P{p});
        """)
        accumulates.append(
            f"c{p}[j] = fma(sb{p}[g & 1][n][0], P{p}[j], "
            f"fma(sb{p}[g & 1][n][1], xs, c{p}[j]));\n"
        )
    epilogue = "bfloat value = bfloat(c0[j]);\n"
    if biases[0]:
        epilogue += f"value = bfloat(float(value) + float(BIAS{indices[0]}[col]));\n"
    if len(indices) == 2:
        epilogue += "bfloat up = bfloat(c1[j]);\n"
        if biases[1]:
            epilogue += f"up = bfloat(float(up) + float(BIAS{indices[1]}[col]));\n"
        epilogue += """
            const float g = float(value);
            const bfloat e = bfloat(1.0f / (1.0f + precise::exp(abs(g))));
            const bfloat sig = g < 0.0f ? e : bfloat(1.0f - float(e));
            value = bfloat(float(bfloat(g * float(sig))) * float(up));
        """
    epilogue += f"{output}[size_t(row) * N + col] = value;"
    return f"""
    constexpr int N = {width};
    constexpr int TM = {TILE_M}, TN = {TILE_N}, GS = {GROUP_SIZE}, NTH = {THREADS};
    constexpr int KG = K / GS;
    static_assert(K % GS == 0, "K must be a multiple of the group size");
    static_assert(NTH == 4 * TM, "four threads sum each row's group inputs");
    const int tid = int(thread_index_in_threadgroup);
    const int row0 = int(threadgroup_position_in_grid.x) * TM;
    const int col0 = (int(threadgroup_position_in_grid.y) - {tile_offset}) * TN;
    constexpr auto desc = matmul2d_descriptor(TM, TN, GS, false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<{THREADS // 32}>> op;
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA(
        (device bfloat*)X + int64_t(row0) * K, dextents<int32_t, 2>(K, M - row0));
    auto a0 = tA.slice(0, 0);
    {"".join(declarations)}
    {"".join(initializes)}
    threadgroup float xsum[2][TM];
    const int xr = tid >> 2, xq = tid & 3;
    const bool xrow = row0 + xr < M;
    const device bfloat* xp = X + (xrow ? size_t(row0 + xr) * K : size_t(0)) + xq * (GS / 4);
    for (int g = 0; g < KG; ++g) {{
        // Group input sums: four quarters in index order, (q0 + q1) + (q2 + q3).
        float xq_sum = 0.0f;
        if (xrow)
            for (int i = 0; i < GS / 4; ++i) xq_sum += float(xp[g * GS + i]);
        xq_sum += simd_shuffle_xor(xq_sum, ushort(1));
        xq_sum += simd_shuffle_xor(xq_sum, ushort(2));
        if (xq == 0) xsum[g & 1][xr] = xq_sum;
        {"".join(stages)}
        auto a = tA.slice(g * GS, 0);
        {"".join(runs)}
        // One barrier per group: the double-buffered xsum/sb slot g & 1 is
        // rewritten only after every thread has passed the next barrier.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint16_t j = 0; j < c0.get_capacity(); ++j) {{
            if (c0.is_valid_element(j)) {{
                const auto index = c0.get_multidimensional_index(j);
                const int n = index[0];
                const float xs = xsum[g & 1][index[1]];
                {"".join(accumulates)}
            }}
        }}
    }}
    for (uint16_t j = 0; j < c0.get_capacity(); ++j) {{
        if (c0.is_valid_element(j)) {{
            const auto index = c0.get_multidimensional_index(j);
            const int col = col0 + index[0], row = row0 + index[1];
            if (col < N && row < M) {{ {epilogue} }}
        }}
    }}
    """


def source_for(widths, biases):
    """Grouped projections; same inputs, outputs and grid as the metal tile."""
    inputs, blocks, offset = ["X", "M"], [], 0
    for i, (n, biased) in enumerate(zip(widths, biases, strict=True)):
        inputs.extend([f"W{i}", f"S{i}", f"B{i}"])
        if biased:
            inputs.append(f"BIAS{i}")
        end = offset + (n + TILE_N - 1) // TILE_N
        blocks.append(
            f"if (int(threadgroup_position_in_grid.y) >= {offset} && "
            f"int(threadgroup_position_in_grid.y) < {end}) {{\n"
            + _body(n, (i,), (biased,), f"Y{i}", offset)
            + "\n}"
        )
        offset = end
    return "\n".join(blocks), inputs


def swiglu_source(width, biases):
    """Fused gate/up/SwiGLU; same inputs, output and grid as the metal tile."""
    inputs = ["X", "M", "W0", "S0", "B0", "W1", "S1", "B1"]
    inputs += [f"BIAS{i}" for i, biased in enumerate(biases) if biased]
    return _body(width, (0, 1), biases, "OUT"), inputs
