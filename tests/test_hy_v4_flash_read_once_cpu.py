"""Hy4-flash opt-in read-once expert kernels on CPU: policy, admission, fallback.

The kernels are Metal-only (tests/test_switch_read_once_metal.py proves
their numerics).  On CPU this proves the contract around them: the policy key
parses and fails closed, it gets its own APCv2 namespace and refuses a
deterministic tile set fitted without it, the admission names a reason for
every call it declines, a declined call runs the stock SwitchGLU bit for bit,
and the route receipt separates ``selected`` from ``observed_used``.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.models import hy_v4_flash as H  # noqa: E402
from mlx2.runtime.models import invariant_prefill as INV  # noqa: E402
from mlx2.runtime.models import switch_layers as SL  # noqa: E402
from mlx2.runtime.models import switch_read_once as RO  # noqa: E402
from test_hy_v4_flash_cpu import LONG  # noqa: E402
from test_hy_v4_flash_fused_moe_cpu import pack  # noqa: E402,F401  (module fixture)


def _glu(D=512, F=768, E=16, bits=4, group=64, activation=None, dtype=mx.bfloat16):
    mx.random.seed(3)
    glu = SL.SwitchGLU(D, F, E, activation=H.ClampedSwiGLU(7.0) if activation is None else activation)
    for name in ("gate_proj", "up_proj", "down_proj"):
        bank = getattr(glu, name).to_quantized(group, bits)
        bank.scales, bank.biases = bank.scales.astype(dtype), bank.biases.astype(dtype)
        setattr(glu, name, bank)
    return glu


def _route(rows, k=4, E=16, seed=0):
    rng = np.random.default_rng(seed)
    return mx.array(np.stack([rng.choice(E, k, replace=False) for _ in range(rows)]).astype(np.uint32))


def test_admission_reasons_structural_before_device():
    glu = _glu()
    x = mx.zeros((4, 512), dtype=mx.bfloat16)
    ind = _route(4)
    # Every structural check passes: on CPU the only reason is the device.
    assert RO.refusal(glu, x, ind) == "cpu_device"
    assert RO.refusal(glu, x[:1], ind[:1]) == "below_min_rows"
    assert RO.refusal(glu, x.astype(mx.float32), ind) == "activation_dtype"
    big = _route(RO.MAX_ASSIGNMENTS // 4 + 1)
    assert RO.refusal(glu, mx.zeros((big.shape[0], 512), mx.bfloat16), big) == "above_max_assignments"
    assert RO.refusal(_glu(activation=nn.GELU()), x, ind) == "activation_not_swiglu"
    assert RO.refusal(_glu(group=32), x, ind) == "table_not_affine_q4_g64"
    assert RO.refusal(_glu(bits=8), x, ind) == "table_not_affine_q4_g64"
    assert RO.refusal(_glu(D=320), mx.zeros((4, 320), mx.bfloat16), ind) == "dims_not_multiple_of_lane_step"
    assert RO.refusal(_glu(dtype=mx.float16), x, ind) == "scale_dtype_mismatch"
    assert RO.refusal(_glu(dtype=mx.float16), x.astype(mx.float16), ind) == "cpu_device"

    class OtherClamp(H.ClampedSwiGLU):  # a subclass may change the formula
        pass

    assert RO.refusal(_glu(activation=OtherClamp(7.0)), x, ind) == "activation_not_swiglu"
    dense = SL.SwitchGLU(512, 768, 16, activation=H.ClampedSwiGLU(7.0))
    assert RO.refusal(dense, x, ind) == "bank_SwitchLinear"
    with INV.scope(True):
        assert RO.refusal(glu, x, ind) == "invariant_lane"
    # The plain SwiGLU and a fused [gate|up] bank are admissible shapes.
    assert RO.refusal(_glu(activation=SL.SwiGLU()), x, ind) == "cpu_device"
    fused = SL.GateUpFusedSwitchGLU.from_switch_glu(_glu())
    assert RO._banks(fused)[3:] == (2 * 768, 768, 768)


def test_switch_glu_declines_and_counts_on_cpu():
    RO.status(reset=True)
    glu = _glu()
    assert RO.switch_glu(glu, mx.zeros((4, 512), mx.bfloat16), _route(4)) is None
    st = RO.status(reset=True)
    assert st["calls"] == 0 and st["fallbacks"] == 1 and st["reasons"] == {"cpu_device": 1}
    assert st["observed_used"] is False and st["qualified"] is False


def _moe_args():
    from test_hy_v4_flash_cpu import tiny_config

    return H.ModelArgs.from_hf_config(tiny_config(
        hidden_size=64, moe_intermediate_size=256, routed_expert_hidden_size=256,
        n_routed_experts=16, num_experts_per_tok=4, swiglu_limit=1.5))


def test_moe_fallback_is_the_stock_route_bit_for_bit():
    mx.random.seed(5)
    moe = H.MoE(_moe_args())
    moe.set_dtype(mx.bfloat16)
    nn.quantize(moe, group_size=64, bits=4,
                class_predicate=lambda p, m: hasattr(m, "to_quantized") and not p.endswith("gate"))
    mx.eval(moe.parameters())
    x = (mx.random.normal((1, 6, 64)) * 0.5).astype(mx.bfloat16)
    ref = moe(x)
    moe.expert_read_once = True
    y = moe(x)
    assert np.array_equal(np.array(y.astype(mx.float32)), np.array(ref.astype(mx.float32)))
    assert moe.read_once_calls == 0 and moe.read_once_declines == {"cpu_device": 1}
    moe(x[:, :1])
    assert moe.read_once_declines == {"cpu_device": 1, "below_min_rows": 1}


def test_policy_key_parses_namespaces_and_fails_closed():
    from mlx2.adapters import hy_v4_flash as A

    model, external, _ = A._parse_policy({"moe_expert_read_once": True})
    assert model == {"moe_expert_read_once": True} and external == {}
    with pytest.raises(ValueError, match="moe_expert_read_once must be a boolean"):
        A._parse_policy({"moe_expert_read_once": 1})
    # Route-neutral: allowed with the embedded drafter (the verify route it is for).
    assert A._parse_policy({"moe_expert_read_once": True, "draft_model": "embedded_dfly"})[2] is not None
    base = A.CACHE_LAYOUT
    assert A.numerics_layout(base, moe_expert_read_once=True) == base + "+moe-read-once-v1"
    assert A.numerics_layout(base, moe_expert_read_once=False) == base
    # A deterministic tile set fitted on the stock experts reads the key's default.
    assert "moe_expert_read_once" in A.DET_TILE_NUMERICS_KEYS
    assert A.DET_TILE_NUMERICS_DEFAULTS["moe_expert_read_once"] is False


def test_feature_grid_declares_the_key():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "build_model_feature_grid",
        Path(__file__).resolve().parents[1] / "scripts" / "build_model_feature_grid.py")
    grid = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grid)
    assert "moe_expert_read_once" in grid.FEATURES


def test_adapter_receipt_selected_vs_observed(pack):  # noqa: F811
    from mlx2.adapters import hy_v4_flash as A

    root, _ = pack
    default = A.HYV4FlashAdapter(str(root))
    try:
        r = default.diagnostics()["moe_expert_read_once"]
        assert r["selected"] is False and r["observed_used"] is False and r["qualified"] is False
        assert "+moe-read-once" not in default.layout
        ref = default.model(mx.array([LONG[:12]]))
    finally:
        default.close()
    adapter = A.HYV4FlashAdapter(str(root), execution_policy={"moe_expert_read_once": True})
    try:
        assert "+moe-read-once-v1" in adapter.layout
        assert adapter.descriptor.cache_layout == adapter.layout
        y = adapter.model(mx.array([LONG[:12]]))
        r = adapter.diagnostics()["moe_expert_read_once"]
        # The tiny pack (fp32 activations, 64-wide latent) is outside the
        # kernel's admission: every call declines with a structural reason and
        # runs the stock route, bit for bit.
        assert r["selected"] and not r["observed_used"] and r["layer_calls"] == 0
        declined = r["declined_layer_calls"]
        assert declined and set(declined) <= {"activation_dtype", "dims_not_multiple_of_lane_step"}
        assert np.array_equal(np.array(y.astype(mx.float32)), np.array(ref.astype(mx.float32)))
    finally:
        adapter.close()
