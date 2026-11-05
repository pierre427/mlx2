"""Ripple Metal RMSNorm option: plumbing and CPU fallback.

The kernel itself runs only on Metal (scripts/research/ripple_rmsnorm_gate.py
checks it there); on CPU every call must fall back to the eager law exactly.
"""
import mlx.core as mx

mx.set_default_device(mx.cpu)

from mlx2.runtime.models import ricursive, ricursive_rmsnorm
from test_ripple_memcopy_cpu import assert_bitwise, build


def run(model):
    cache = model.make_cache()
    outs = [model(mx.array([[1, 2, 3, 4, 5, 6, 7]]), cache=cache)]
    for _ in range(4):
        outs.append(model(mx.argmax(outs[-1][:, -1:], axis=-1), cache=cache))
    mx.eval(outs)
    return outs


def test_cpu_fallback_is_the_eager_law_bitwise():
    eager = build(mx.bfloat16, False)
    metal = build(mx.bfloat16, False)
    metal.set_metal_rmsnorm(True)
    assert all(m.metal for _, m in metal.named_modules() if isinstance(m, ricursive.RMSNorm))
    before = dict(ricursive_rmsnorm.STATS)
    for a, b in zip(run(metal), run(eager), strict=True):
        assert_bitwise(a, b)
    assert ricursive_rmsnorm.STATS["fallbacks"] > before["fallbacks"]
    assert ricursive_rmsnorm.STATS["calls"] == before["calls"]


def test_supported_refuses_cpu_and_mismatched_inputs():
    x, w = mx.ones((2, 8), dtype=mx.bfloat16), mx.ones((8,), dtype=mx.bfloat16)
    assert not ricursive_rmsnorm.supported(x, w)  # CPU device
    assert ricursive_rmsnorm.threads(128) == 32 and ricursive_rmsnorm.threads(5120) == 256


def test_option_is_typed_and_default_off():
    from mlx2.adapters.ripple import OPTIONS, split_options

    assert OPTIONS["metal_rmsnorm"] is bool
    assert split_options({"metal_rmsnorm": True})[1] == {"metal_rmsnorm": True}
    model = build(mx.bfloat16, False)
    assert not any(m.metal for _, m in model.named_modules() if isinstance(m, ricursive.RMSNorm))


def test_affine_defaults_on_others_opt_in_and_explicit_false_wins():
    from mlx2.adapters.ripple import resolved_options

    assert resolved_options("affine", {}) == {"metal_rmsnorm": True, "gqa_folded_attention": True, "metal_glue": True}
    assert resolved_options("affine", {"metal_rmsnorm": False})["metal_rmsnorm"] is False
    assert resolved_options("nvfp4", {}) == {}
    assert resolved_options("bf16", {"metal_rmsnorm": True}) == {"metal_rmsnorm": True}
