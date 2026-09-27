# SPDX-License-Identifier: Apache-2.0
# Research-only extraction of jundot/omlx PR #3958 at 3e9703518984c294c2ed989cf968727d0e0ffa56.
# Its split-K morphology was ported from MTPLX by Youssof Altoukhi (Apache-2.0).
# See provenance/q4-omlx3958-ksplit.json. No runtime route is modified.
"""Optional upstream split-K geometry arm for the isolated q4 comparator."""

from __future__ import annotations

_KERNEL_CACHE = {}


def eligible(m: int, n: int, k: int, bits: int = 4, group_size: int = 64,
             dtype: str = "bfloat16") -> bool:
    return (m in (4, 6) and n >= 16384 and n % 4 == 0 and k % 64 == 0
            and bits == 4 and group_size == 64 and dtype == "bfloat16")


def _pack_block(m: int, bits: int, sfx: str) -> str:
    p = f"pack{sfx}"
    lines = [f"int k_base{sfx} = {p} * 8;", f"int gi{sfx} = k_base{sfx} / GS;"]
    for r in range(m):
        lines.append(f"Vec8 v{sfx}_{r} = xv[({r} * K + k_base{sfx}) / 8];")
    if bits == 4:
        for j in range(4):
            lines.append(f"uint32_t p{sfx}_{j} = w_q[(n0 + {j}) * K_by_p + {p}];")
        for j in range(4):
            lines.append(
                f"float s{sfx}_{j} = float(scales[(n0 + {j}) * K_by_gs + gi{sfx}]);"
                f" float b{sfx}_{j} = float(biases[(n0 + {j}) * K_by_gs + gi{sfx}]);"
            )
        for j in range(4):
            block = [
                "{",
                f"    uint32_t packed = p{sfx}_{j};",
                f"    float s = s{sfx}_{j};",
                f"    float b = b{sfx}_{j};",
                "    for (int ki = 0; ki < 8; ++ki) {",
                "        float wv = float((packed >> (ki * 4)) & 0xFu) * s + b;",
            ]
            for r in range(m):
                block.append(
                    f"        acc[{j} * {m} + {r}] += float(v{sfx}_{r}[ki]) * wv;"
                )
            block.extend(["    }", "}"])
            lines.extend(block)
    else:
        for j in range(4):
            lines.append(
                f"uint32_t pa{sfx}_{j} = w_q[(n0 + {j}) * K_by_w + {p} * 2];"
                f" uint32_t pb{sfx}_{j} = w_q[(n0 + {j}) * K_by_w + {p} * 2 + 1];"
            )
        for j in range(4):
            lines.append(
                f"float s{sfx}_{j} = float(scales[(n0 + {j}) * K_by_gs + gi{sfx}]);"
                f" float b{sfx}_{j} = float(biases[(n0 + {j}) * K_by_gs + gi{sfx}]);"
            )
        for j in range(4):
            block = [
                "{",
                f"    uint32_t pa = pa{sfx}_{j};",
                f"    uint32_t pb = pb{sfx}_{j};",
                f"    float s = s{sfx}_{j};",
                f"    float b = b{sfx}_{j};",
                "    for (int ki = 0; ki < 4; ++ki) {",
                "        float wa = float((pa >> (ki * 8)) & 0xFFu) * s + b;",
                "        float wb = float((pb >> (ki * 8)) & 0xFFu) * s + b;",
            ]
            for r in range(m):
                block.append(
                    f"        acc[{j} * {m} + {r}] += float(v{sfx}_{r}[ki]) * wa;"
                )
                block.append(
                    f"        acc[{j} * {m} + {r}] += float(v{sfx}_{r}[ki + 4]) * wb;"
                )
            block.extend(["    }", "}"])
            lines.extend(block)
    return "\n            ".join(lines)


def _build_ksplit_kernel(m: int, bits: int, group_size: int, dtype, *, k_parts: int):
    import mlx.core as mx

    key = ("ksplit", m, bits, group_size, dtype, k_parts)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    n_acc = 4 * m
    loop = f"""
        for (int packA = p_start + int(lane); packA < p_end; packA += 32) {{
            {_pack_block(m, bits, "A")}
        }}
    """

    source = f"""
        using namespace metal;
        constexpr int GS = {group_size};
        constexpr int K_PARTS = {k_parts};

        uint part = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint tg_n = threadgroup_position_in_grid.y;

        int K = int(K_size);
        int K_by_p = K / 8;
        int K_by_w = K / 4;
        int K_by_gs = K / GS;
        int per_part = K_by_p / K_PARTS;
        int N = int(N_size);
        int n0 = int(tg_n) * 4;
        int p_start = int(part) * per_part;
        int p_end = (int(part) == K_PARTS - 1) ? K_by_p : p_start + per_part;

        float acc[{n_acc}];
        _Pragma("unroll")
        for (int i = 0; i < {n_acc}; ++i) {{
            acc[i] = 0.0f;
        }}

        using Vec8 = vec<T, 8>;
        const device Vec8 *xv = (const device Vec8*)x;

        {loop}

        _Pragma("unroll")
        for (int i = 0; i < {n_acc}; ++i) {{
            acc[i] = simd_sum(acc[i]);
        }}

        threadgroup float partials[K_PARTS * {n_acc}];
        if (lane == 0) {{
            _Pragma("unroll")
            for (int i = 0; i < {n_acc}; ++i) {{
                partials[int(part) * {n_acc} + i] = acc[i];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (part == 0 && lane < {n_acc}) {{
            float total = 0.0f;
            _Pragma("unroll")
            for (int p = 0; p < K_PARTS; ++p) {{
                total += partials[p * {n_acc} + int(lane)];
            }}
            int j = int(lane) / {m};
            int row = int(lane) - j * {m};
            y[row * N + n0 + j] = T(total);
        }}
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"mlx2_research_omlx_vk_ks_m{m}_q{bits}_kp{k_parts}_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "K_size", "N_size"],
        output_names=["y"],
        source=source,
    )
    _KERNEL_CACHE[key] = kernel
    return kernel

def qmm(mx, x, q):
    w, scales, biases = q
    m, k = x.shape
    n = w.shape[0]
    if not eligible(m, n, k, dtype=str(x.dtype).removeprefix("mlx.core.")):
        raise ValueError("ineligible #3958 split-K geometry")
    if w.shape != (n, k // 8) or scales.shape != (n, k // 64) or biases.shape != scales.shape:
        raise ValueError("packed affine q4 layout mismatch")
    kernel = _build_ksplit_kernel(m, 4, 64, x.dtype, k_parts=2)
    return kernel(inputs=[x, w, scales, biases, k, n],
                  template=[("T", x.dtype)], grid=(64, n // 4, 1),
                  threadgroup=(64, 1, 1), output_shapes=[(m, n)],
                  output_dtypes=[x.dtype])[0]
