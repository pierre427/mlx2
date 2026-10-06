"""Source gates for the full Qwen dense target path's typed arithmetic."""

from pathlib import Path

KERNELS = (
    Path(__file__).resolve().parents[1]
    / "src/tensorfold/kernels/qwen/dense/v1"
)


def test_dense_norms_follow_mlx_rms_thread_and_storage_laws():
    required = (
        "constexpr int E = 4;",
        "constexpr int TPG = K > 4096 ? 1024 : 32 * ((K + 127) / 128);",
        "threadgroup float red[32];",
        "for (int r = 0; r < K; r += TPG * E)",
        "ss = simd_sum(red[thread_index_in_simdgroup]);",
        "metal::precise::rsqrt(ss / float(K) + eps[0]);",
        "const bfloat normalized = bfloat(float(h) * inv);",
        "bfloat(Wt[r + int(t) * E + i] * normalized)",
    )
    for name, marker in (("lane_glue.py", "_NORM_XS"), ("row_glue.py", "_NORM")):
        source = (KERNELS / name).read_text()
        assert marker in source
        for spelling in required:
            assert spelling in source
        assert "const float inv = metal::rsqrt(total / float(K)" not in source
        assert "ss = fma(hv[i], hv[i], ss);" not in source


def test_dense_mlp_swiglu_keeps_bf16_boundaries_and_stable_sigmoid():
    required = (
        "const bfloat mlp_sigmoid_exp = bfloat(metal::exp(metal::abs(gf)));",
        "const bfloat mlp_sigmoid_low = bfloat(bfloat(1.0f) / bfloat(bfloat(1.0f) + mlp_sigmoid_exp));",
        "const bfloat mlp_sigmoid = gf < bfloat(0.0f)",
        "? mlp_sigmoid_low : bfloat(bfloat(1.0f) - mlp_sigmoid_low);",
        "const bfloat activated = bfloat(gf * mlp_sigmoid);",
    )
    writes = {
        "lane_glue.py": "const bfloat h = bfloat(activated * UP[e]);",
        "row_glue.py": (
            "HOUT[m * N + i] = bfloat(activated * GU[m * 2 * N + N + i]);"
        ),
    }
    for name, write in writes.items():
        source = (KERNELS / name).read_text()
        for spelling in required:
            assert spelling in source
        assert write in source
        assert "metal::exp(-gf)" not in source

    from tensorfold.kernels.qwen.dense.v1 import lane_fuse

    fused = lane_fuse.sources()["mlp_act"]
    for spelling in required:
        assert spelling in fused
    assert "GATE[e + int(m) * N]" in fused
    assert "UP[e + int(m) * N + N]" in fused
    assert "GATE[e]" not in fused
    assert "UP[e]" not in fused


def test_norm_launch_geometry_matches_embedded_mlx_geometry():
    lane = (KERNELS / "lane_glue.py").read_text()
    row = (KERNELS / "row_glue.py").read_text()
    spelling = "1024 if K > 4096 else 32 * ((K + 127) // 128)"
    assert lane.count(spelling) == 1
    assert row.count(spelling) == 2
    assert "threadgroup=(K // 16, 1, 1)" not in row


def test_tree_attention_uses_the_actual_ordinary_sdpa_law():
    stream = (KERNELS / "stream_attention.py").read_text()
    multi = (KERNELS / "lane_multi.py").read_text()
    family = (
        Path(__file__).resolve().parents[1]
        / "src/tensorfold/families/qwen3_5/__init__.py"
    ).read_text()

    assert "def ordinary_tree_sdpa(" in stream
    assert "mx.fast.scaled_dot_product_attention(" in stream
    assert "node_keys, node_values = keys[:, :, :stop], values[:, :, :stop]" in stream
    assert "mx.take(keys, indices, axis=2)" in stream
    assert "stream_attention.ordinary_tree_sdpa(" in multi
    assert "kv.append(cache.update_and_fetch(k_s, v_s))" in multi
    assert "def _ordinary_qkv_rows(" in multi
    assert "attn.q_norm(queries[:, row:row + 1])" in multi
    assert "attn.k_norm(keys[:, row:row + 1]" in multi
    assert "attn.rope(q, offset=int(position))" in multi
    assert "offset=pos" not in multi
    assert "lane_attention.install()" not in family
