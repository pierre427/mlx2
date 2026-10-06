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
            "const float inv = metal::rsqrt(ss / float(DK) + norm_eps);"
        ) == 1
        assert "metal::rsqrt(ss / float(DK) + 1e-6f)" not in source


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
