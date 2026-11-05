"""Ripple metal_glue option: plumbing and exact CPU fallback (kernels are Metal-only)."""
import mlx.core as mx

mx.set_default_device(mx.cpu)

from mlx2.runtime.models import ricursive, ricursive_glue
from test_ripple_memcopy_cpu import assert_bitwise, build
from test_ripple_metal_rmsnorm_cpu import run


def test_cpu_fallback_is_the_eager_law_bitwise():
    eager = build(mx.bfloat16, False)
    glue = build(mx.bfloat16, False)
    glue.set_metal_glue(True)
    assert all(m.metal_glue for _, m in glue.named_modules() if isinstance(m, (ricursive.Attention, ricursive.MLP)))
    before = ricursive_glue.STATS["fallbacks"]
    for a, b in zip(run(glue), run(eager), strict=True):
        assert_bitwise(a, b)
    assert ricursive_glue.STATS["fallbacks"] > before
    assert ricursive_glue.STATS["qk_norm_rope"] == 0


def test_option_typed_and_affine_default_on():
    from mlx2.adapters.ripple import OPTIONS, resolved_options

    assert OPTIONS["metal_glue"] is bool
    assert resolved_options("affine", {})["metal_glue"] is True
    assert resolved_options("affine", {"metal_glue": False})["metal_glue"] is False
    assert "metal_glue" not in resolved_options("nvfp4", {})
