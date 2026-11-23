"""``inkling2_moe_decode_kernel`` on Metal (GPU; run under the GPU lock).

The kernels against the shipped ``moe_fast.fused_call`` for one token at two
geometries: the CPU tests' tiny NVFP4 model (hidden 128, inter 64) and one
random NVFP4 SparseMoE at the Mini's widths (hidden 2304, inter 1152, 32
experts, top 8).  Every ``QDOT_CONTRACT`` x ``ROUTE_SUM`` variant is run and
recorded; the module defaults must be bitwise, and in the tiny model the
option must engage on every one-token NVFP4 visit with decode logits bitwise.
Set ``INKLING2_MOE_DECODE_METAL_RECEIPT`` to a path for a JSON summary.
"""

import json
import os

import mlx.core as mx
import pytest

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")
RESULTS = {}


@pytest.fixture(autouse=True)
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        yield
    finally:
        mx.clear_cache()
        mx.set_default_device(previous)


def teardown_module(_module):
    path = os.environ.get("INKLING2_MOE_DECODE_METAL_RECEIPT")
    if path:
        with open(path, "w") as handle:
            json.dump(RESULTS, handle, indent=1, sort_keys=True)


def _wide_moe(seed=0):
    from test_inkling2_cpu import TINY

    from mlx2.runtime.models.inkling2 import TextConfig
    from mlx2.runtime.models.inkling2.language import SparseMoE
    from mlx2.runtime.nvfp4_linear import ScaledNVFP4SwitchLinear

    hidden, inter, experts = 2304, 1152, 32
    config = TextConfig.from_dict({**TINY, "hidden_size": hidden, "intermediate_size": inter,
                                   "n_routed_experts": experts, "num_experts_per_tok": 8})
    moe = SparseMoE(config)
    mx.random.seed(seed)
    moe.gate_weight = (mx.random.normal((experts, hidden)) * 0.02).astype(mx.bfloat16)
    moe.e_score_correction_bias = mx.random.normal((experts,)) * 0.01
    sw = moe.switch_mlp
    scale13 = mx.random.uniform(0.5, 2.0, (experts,)) * 1e-3
    for name, shape, scale in (("gate_proj", (experts, inter, hidden), scale13),
                               ("up_proj", (experts, inter, hidden), scale13),
                               ("down_proj", (experts, hidden, inter),
                                mx.random.uniform(0.5, 2.0, (experts,)) * 1e-3)):
        module = ScaledNVFP4SwitchLinear(shape[2], shape[1], experts)
        w = mx.random.normal(shape) * 0.05 / scale[:, None, None]
        q, s = mx.quantize(w, group_size=16, bits=4, mode="nvfp4")
        module.weight, module.scales = q, s
        module.weight_scale_2 = scale.astype(mx.float32)
        module.freeze()
        setattr(sw, name, module)
    mx.eval(moe.parameters())
    moe.set_fast_path(True, True)
    return moe


def _tiny_moe(layer=3):
    from test_inkling2_nonbitwise_cpu import _quantized_model

    model = _quantized_model()
    model.configure_kernels(moe_fused=True)
    return model.model.layers[layer].mlp


@pytest.mark.parametrize("geometry", ["tiny", "wide"])
def test_variants_against_shipped(geometry):
    from mlx2.runtime.models.inkling2 import moe_decode_kernel as mdk
    from mlx2.runtime.models.inkling2 import moe_fast
    from mlx2.runtime.models.inkling2.language import _fp32

    moe = _tiny_moe() if geometry == "tiny" else _wide_moe()
    experts, refusal = moe._fast_experts()
    assert refusal is None
    hidden = int(moe["gate_weight"].shape[1])
    results = {}
    for seed in range(8):
        x = (mx.random.normal((1, 1, hidden), key=mx.random.key(seed)) * 2).astype(mx.bfloat16)
        assert mdk.decline_reason(moe, experts, x) is None
        ref = moe_fast.fused_call(moe, experts, x)
        order, scores = mdk.route_compiled(x, _fp32(moe, "gate_weight"),
                                           moe.e_score_correction_bias, moe.top_k)
        for contract in mdk._CONTRACTS:
            for route_sum in mdk._ROUTE_SUMS:
                y = mdk.launch(moe, experts, x, order, scores, contract=contract,
                               route_sum=route_sum)
                key = f"{contract}/{route_sum}"
                row = results.setdefault(key, {"bitwise": 0, "max_diff": 0.0, "calls": 0})
                row["calls"] += 1
                row["bitwise"] += int(bool(mx.array_equal(y, ref)))
                diff = mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max().item()
                row["max_diff"] = max(row["max_diff"], diff)
    RESULTS[geometry] = results
    default = results[f"{mdk.QDOT_CONTRACT}/{mdk.ROUTE_SUM}"]
    assert any(r["bitwise"] == r["calls"] for r in results.values()), results
    assert default["bitwise"] == default["calls"], results


def test_tiny_model_decode_engages_and_is_bitwise():
    from test_inkling2_cpu import _run
    from test_inkling2_nonbitwise_cpu import _quantized_model

    model = _quantized_model()
    tokens = mx.random.randint(0, 500, (1, 30), key=mx.random.key(7))
    plan = [(0, 20)] + [(i, i + 1) for i in range(20, 30)]
    model.configure_kernels(moe_fused=True)
    base = _run(model, tokens, plan)
    model.configure_kernels(moe_fused=True, moe_decode_kernel=True)
    got = _run(model, tokens, plan)
    receipt = model.kernel_receipt()
    steps = len(plan) - 1
    RESULTS["tiny_model"] = {"calls": receipt["moe_decode_kernel_calls"],
                             "declines": receipt["moe_decode_kernel_declines"],
                             "bitwise": bool(mx.array_equal(got, base)),
                             "max_diff": mx.abs(got - base).max().item()}
    assert receipt["moe_decode_kernel_calls"] == 44 * steps
    assert receipt["moe_decode_kernel_declines"] == {"expert_format": 4 * steps, "rows": 48}
    assert mx.array_equal(got, base)
