# SPDX-License-Identifier: MIT
"""Source generator for M5 prefill tiles; original mlx2 implementation."""

HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
using namespace metal;
"""
TILE_M, TILE_N, TILE_K, THREADS = 32, 64, 64, 128


def _body(width, indices, biases, output, tile_offset=0):
    declarations, loads, runs, initializes = [], [], [], []
    for p, i in enumerate(indices):
        declarations.append(f"""
        threadgroup bfloat ws{p}[TN * TK];
        auto B{p}t = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>(
            ws{p}, dextents<int32_t, 2>(TK, TN));
        auto c{p} = op.get_destination_cooperative_tensor<decltype(At), decltype(B{p}t), float>();
        """)
        initializes.append(f"""
        for (uint16_t j = 0; j < c{p}.get_capacity(); ++j)
            if (c{p}.is_valid_element(j)) c{p}[j] = 0.0f;
        """)
        loads.append(f"""
        for (int j = tid; j < TN * TK; j += NTH) {{
            const int col = col0 + j / TK, k = base + j % TK;
            bfloat value = bfloat(0);
            if (col < N) {{
                const uint q = (W{i}[size_t(col) * (K / 8) + k / 8] >> ((k % 8) * 4)) & 15u;
                const int group = col * (K / 64) + k / 64;
                value = bfloat(float(q) * float(S{i}[group]) + float(B{i}[group]));
            }}
            ws{p}[j] = value;
        }}
        """)
        runs.append(f"op.run(At, B{p}t, c{p});")
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
    constexpr int TM = {TILE_M}, TN = {TILE_N}, TK = {TILE_K}, NTH = {THREADS};
    const int tid = int(thread_index_in_threadgroup);
    const int row0 = int(threadgroup_position_in_grid.x) * TM;
    const int col0 = (int(threadgroup_position_in_grid.y) - {tile_offset}) * TN;
    threadgroup bfloat xs[TM * TK];
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<{THREADS // 32}>> op;
    auto At = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>(
        xs, dextents<int32_t, 2>(TK, TM));
    {"".join(declarations)}
    {"".join(initializes)}
    for (int base = 0; base < K; base += TK) {{
        for (int j = tid; j < TM * TK; j += NTH) {{
            const int row = row0 + j / TK;
            xs[j] = row < M ? X[size_t(row) * K + base + j % TK] : bfloat(0);
        }}
        {"".join(loads)}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {"".join(runs)}
        threadgroup_barrier(mem_flags::mem_threadgroup);
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
    inputs = ["X", "M", "W0", "S0", "B0", "W1", "S1", "B1"]
    inputs += [f"BIAS{i}" for i, biased in enumerate(biases) if biased]
    return _body(width, (0, 1), biases, "OUT"), inputs
