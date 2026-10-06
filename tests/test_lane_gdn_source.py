"""Static contracts for the Qwen lane GDN normalization kernels."""

import ast
from math import isclose, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "src/tensorfold/kernels/qwen/dense/v1"


def test_lane_gdn_kernels_scale_l2_epsilon_with_head_width():
    """FLA L2 epsilon becomes ``epsilon / DK`` in RMSNorm form."""

    for filename in ("lane_glue.py", "stream_gdn.py"):
        source = (KERNELS / filename).read_text()
        assert source.count("const float norm_eps = 1e-6f / float(DK);") == 1
        assert source.count(
            "const float inv = metal::precise::rsqrt(ss / float(DK) + norm_eps);"
        ) == 1
        assert source.count(
            ": float(bfloat(metal::precise::rsqrt(float(DK))));"
        ) == 1
        assert "metal::rsqrt(ss / float(DK) + 1e-6f)" not in source
        assert "bfloat(metal::rsqrt(float(DK)))" not in source


def test_scaled_rms_epsilon_is_the_fla_l2_epsilon():
    for head_width in (16, 128):
        for sum_squares in (0.0, 1e-8, 1e-4, 1.0):
            rms_then_k_scale = (1.0 / sqrt(head_width)) / sqrt(
                sum_squares / head_width + 1e-6 / head_width
            )
            l2_normalizer = 1.0 / sqrt(sum_squares + 1e-6)
            assert isclose(rms_then_k_scale, l2_normalizer, rel_tol=1e-15)

    old_normalizer = (1.0 / sqrt(128)) / sqrt(1e-6)
    assert old_normalizer < (1.0 / sqrt(1e-6)) / 10


def _output_dtypes(filename: str, function: str) -> list[str]:
    tree = ast.parse((KERNELS / filename).read_text())
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function
    )
    return [
        ast.unparse(keyword.value)
        for node in ast.walk(definition)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "output_dtypes"
    ]


def test_lane_gdn_kernels_keep_beta_in_fp32():
    inputs = {
        "lane_glue.py": "float(Bin[w * NV + hv])",
        "stream_gdn.py": "float(Bin[w * ZS + BO + hv])",
    }
    for filename, beta_input in inputs.items():
        source = (KERNELS / filename).read_text()
        assert source.count(f"const float beta_x = {beta_input};") == 1
        assert source.count(
            "const float beta_y = 1.0f / "
            "(1.0f + metal::precise::exp(metal::abs(beta_x)));"
        ) == 1
        assert source.count(
            "BETA[w * NV + hv] = beta_x < 0.0f ? beta_y : 1.0f - beta_y;"
        ) == 1
        assert "BETA[w * NV + hv] = bfloat(" not in source
        assert "metal::exp(-float(Bin[" not in source

    assert _output_dtypes("lane_glue.py", "gdn_pre") == [
        "[qkv.dtype, qkv.dtype, qkv.dtype, mx.float32, mx.float32]"
    ]
    assert _output_dtypes("stream_gdn.py", "gdn_pre") == [
        "[qkv.dtype, qkv.dtype, qkv.dtype, mx.float32, mx.float32]"
    ]
    assert _output_dtypes("row_glue.py", "gdn_pre") == [
        "[y.dtype, y.dtype, y.dtype, mx.float32, mx.float32, y.dtype]"
    ]
    assert _output_dtypes("row_streams.py", "recur") == [
        "[y.dtype, y.dtype, y.dtype, mx.float32, mx.float32, y.dtype]",
        "[q.dtype] + [mx.float32] * streams",
    ]


def test_gdn_pre_kernels_keep_the_served_conv_silu_rounding_law():
    required = (
        "const bfloat conv = bfloat(acc);",
        "const bfloat sigmoid_exp = bfloat(metal::exp(metal::abs(conv)));",
        (
            "const bfloat sigmoid_low = bfloat(bfloat(1.0f) / "
            "bfloat(bfloat(1.0f) + sigmoid_exp));"
        ),
        "const bfloat sigmoid = conv < bfloat(0.0f)",
        "? sigmoid_low : bfloat(bfloat(1.0f) - sigmoid_low);",
        "vals[j] = float(bfloat(conv * sigmoid));",
    )
    for filename in ("lane_glue.py", "stream_gdn.py"):
        source = (KERNELS / filename).read_text()
        for spelling in required:
            assert source.count(spelling) == 1
        assert "metal::exp(-conv)" not in source
        assert "1.0f / (1.0f + metal::exp" not in source


def test_gdn_decay_uses_compensated_bf16_softplus_and_precise_fp32_exps():
    required = (
        "const bfloat av = bfloat(",
        "const bfloat hi = metal::max(av, zero);",
        "const bfloat lo = metal::min(av, zero);",
        "const bfloat softplus_arg = bfloat(metal::exp(lo - hi));",
        "const float arg_plus_one = 1.0f + arg;",
        "arg * (metal::log(arg_plus_one) / (arg_plus_one - 1.0f));",
        "const bfloat sp = metal::isnan(av)",
        "? metal::numeric_limits<bfloat>::quiet_NaN()",
        "? hi : bfloat(hi + bfloat(log1p)));",
        "G[w * NV + hv] = metal::precise::exp(",
        "-metal::precise::exp(float(ALOG[hv])) * float(sp));",
    )
    for filename in ("lane_glue.py", "stream_gdn.py"):
        source = (KERNELS / filename).read_text()
        for spelling in required:
            assert source.count(spelling) == 1
        assert "metal::log(1.0f + metal::exp(-metal::abs" not in source
        assert "G[w * NV + hv] = metal::exp(" not in source


def test_gdn_post_kernels_keep_precise_rms_and_bf16_norm_boundaries():
    required = (
        "const float inv = metal::precise::rsqrt(ss / float(DV) + eps[0]);",
        "const bfloat normalized = bfloat(yv[j] * inv);",
        "const bfloat normed = bfloat(NW[d] * normalized);",
        "const float gate_exp = metal::exp(metal::abs(zf));",
        "const float gate_low = 1.0f / (1.0f + gate_exp);",
        "const float gate_sigmoid = zf < 0.0f ? gate_low : 1.0f - gate_low;",
        "const float gate = zf * gate_sigmoid;",
    )
    sources = {
        "lane_glue.py": "const bfloat o = bfloat(float(normed) * gate);",
        "row_glue.py": (
            "OUT[m * NV * DV + hv * DV + d] = bfloat(float(normed) * gate);"
        ),
    }
    for filename, write in sources.items():
        source = (KERNELS / filename).read_text()
        for spelling in required:
            assert source.count(spelling) == 1
        assert source.count(write) == 1
        assert "metal::rsqrt(ss / float(DV)" not in source
        assert "metal::exp(-zf)" not in source
        assert "float(NW[d]) * (yv[j] * inv)" not in source


def test_all_derived_gdn_sources_flow_from_the_corrected_canonical_bodies():
    from tensorfold.kernels.qwen.dense.v1 import lane_fuse, row_glue, row_streams

    pre_markers = (
        "const float inv = metal::precise::rsqrt(ss / float(DK) + norm_eps);",
        "const bfloat sigmoid_exp = bfloat(metal::exp(metal::abs(conv)));",
        "const bfloat softplus_arg = bfloat(metal::exp(lo - hi));",
        "G[w * NV + hv] = metal::precise::exp(",
    )
    for source in (row_glue.sources()["gdn_pre"], row_streams._pre_source(2)):
        for marker in pre_markers:
            assert marker in source

    post_markers = (
        "const float inv = metal::precise::rsqrt(ss / float(DV) + eps[0]);",
        "const bfloat normalized = bfloat(yv[j] * inv);",
        "const float gate_exp = metal::exp(metal::abs(zf));",
        "bfloat(float(normed) * gate)",
    )
    for source in (row_glue.sources()["gdn_post"], lane_fuse.sources()["gdn_post"]):
        for marker in post_markers:
            assert marker in source
