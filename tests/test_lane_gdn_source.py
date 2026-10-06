"""Static contracts for the Qwen lane GDN normalization kernels."""

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
