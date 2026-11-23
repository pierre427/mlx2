"""``inkling2_fuse_*`` on Metal (GPU; run under the GPU lock).

Random tiny decoders at two geometries: the CPU test's (hidden 128, head 32)
and a wider one closer to the Mini's kernel shapes (hidden 512, head 128 so
SDPA takes its vector kernel, controller rank 64).  The four exact options
are checked bitwise against the reference on the GPU (prefill, chunked
prefill, decode with the ring wrapped, B1 and B2, FP32 and BF16).  The two
kernel options are near-exact: the tests bound the gap and record whether it
is zero.  Set ``INKLING2_FUSE_METAL_RECEIPT`` to a path to get a JSON summary.
"""

import json
import os

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")

EXACT = ("fuse_controllers", "fuse_residual", "fuse_sconv", "fuse_attention")
ALL = {name: True for name in EXACT}
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
    path = os.environ.get("INKLING2_FUSE_METAL_RECEIPT")
    if path:
        with open(path, "w") as handle:
            json.dump(RESULTS, handle, indent=1, sort_keys=True)


def _model(geometry, bf16, seed=0):
    from mlx2.runtime.models.inkling2 import Model, TextConfig
    from test_inkling2_cpu import TINY

    config = dict(TINY)
    if geometry == "wide":
        config.update(hidden_size=512, head_dim=128, swa_head_dim=128, d_rel=16,
                      intermediate_size=128,
                      loop_config=dict(core_start=6, core_end=18, num_passes=3, gate_rank=64))
    mx.random.seed(seed)
    model = Model(TextConfig.from_dict(config))
    weights = []
    for key, value in tree_flatten(model.parameters()):
        noise = mx.random.normal(value.shape) * (0.05 if value.ndim > 1 else 0.3)
        value = noise + (1.0 if key.endswith("norm.weight") else 0.0)
        weights.append((key, value.astype(mx.bfloat16) if bf16 else value))
    model.load_weights(weights, strict=True)
    mx.eval(model.parameters())
    return model


def _run(model, tokens, plan):
    cache = model.make_cache()
    outs = []
    for a, b in plan:
        outs.append(model(tokens[:, a:b], cache=cache))
        mx.eval(outs[-1])
    return mx.concatenate(outs, axis=1)


PLAN = [(0, 20), (20, 27)] + [(i, i + 1) for i in range(27, 40)]


@pytest.mark.parametrize("geometry", ["tiny", "wide"])
@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("option", [*EXACT, "all"])
def test_exact_options_bitwise_on_gpu(option, rows, bf16, geometry):
    model = _model(geometry, bf16)
    model.configure_kernels(moe_fused=True, moe_weighted_sum=True)
    tokens = mx.random.randint(0, 500, (rows, 40), key=mx.random.key(rows))
    base = _run(model, tokens, PLAN)
    model.configure_kernels(moe_fused=True, moe_weighted_sum=True,
                            **(ALL if option == "all" else {option: True}))
    got = _run(model, tokens, PLAN)
    diff = mx.abs(got - base).reshape(rows, 40, -1).max(axis=(0, 2))
    bad = [i for i, v in enumerate(diff.tolist()) if v > 0]
    RESULTS[f"exact/{option}/{geometry}/{'bf16' if bf16 else 'fp32'}/B{rows}"] = {
        "bitwise": not bad, "first_bad_position": bad[0] if bad else None,
        "max_abs": float(diff.max().item())}
    assert not bad, f"first differing position {bad[0]}, max {diff.max().item()}"


VERIFY_PLAN = [(0, 16), (16, 24), (24, 32), (32, 40), (40, 41), (41, 49), (49, 50)]


@pytest.mark.parametrize("geometry", ["tiny", "wide"])
@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("rows", [1, 2])
def test_controllers_batched_rows_bitwise(rows, bf16, geometry):
    """DSpark-verify-shaped calls (8-row blocks, B1 and B2) and short prefill
    chunks keep the reference controllers (the batched form is one-row only,
    see ``fuse.batched_rows_ok``) and match the reference bitwise."""
    model = _model(geometry, bf16)
    model.configure_kernels(moe_fused=True, moe_weighted_sum=True)
    tokens = mx.random.randint(0, 500, (rows, 50), key=mx.random.key(7 + rows))
    base = _run(model, tokens, VERIFY_PLAN)
    model.configure_kernels(moe_fused=True, moe_weighted_sum=True, fuse_controllers=True)
    got = _run(model, tokens, VERIFY_PLAN)
    counts = model.kernel_receipt()["fuse_counts"]
    diff = mx.abs(got - base).reshape(rows, 50, -1).max(axis=(0, 2))
    bad = [i for i, v in enumerate(diff.tolist()) if v > 0]
    RESULTS[f"controllers_rows/{geometry}/{'bf16' if bf16 else 'fp32'}/B{rows}"] = {
        "bitwise": not bad, "first_bad_position": bad[0] if bad else None,
        "max_abs": float(diff.max().item()), "counts": {
            k: counts.get(k, 0) for k in ("controllers_batched", "controllers_batched_rows",
                                          "controllers_unbatched")}}
    assert counts.get("controllers_batched_rows", 0) == 0
    assert counts.get("controllers_unbatched", 0) > 0
    assert not bad, f"first differing position {bad[0]}, max {diff.max().item()}"


@pytest.mark.parametrize("rows", [1])
@pytest.mark.parametrize("bf16", [False, True])
def test_qkv_controllers_checkpoint_geometry_bitwise(bf16, rows):
    """``qkv_controllers`` at the Mini's width and rank (D 2304, gate rank 64)
    against the three reference controllers, one row (the only row count
    the gate admits).  The multi-row form tried at fc9ac7ed8 was not bitwise
    at 2..32 rows (qualification/runs/inkling2-gpu-20261122/16eed37f5-fusemetal)."""
    from types import SimpleNamespace

    from mlx2.runtime.models.inkling2 import TextConfig
    from mlx2.runtime.models.inkling2 import fuse as F
    from mlx2.runtime.models.inkling2.language import MessageController
    from test_inkling2_cpu import TINY

    config = TextConfig.from_dict({**TINY, "hidden_size": 2304, "loop_config": dict(
        core_start=6, core_end=18, num_passes=3, gate_rank=64)})
    mx.random.seed(rows)
    ctrls = {}
    for name in ("attn", "k", "v", "mlp"):
        ctrl = MessageController(config)
        for w in ("down_weight", "forget_weight", "gain_weight"):
            value = mx.random.normal(ctrl[w].shape) * 0.05
            ctrl[w] = value.astype(mx.bfloat16) if bf16 else value
        ctrls[name] = ctrl
    layer = SimpleNamespace(controllers=ctrls)
    shapes = {1: (1, 1), 2: (2, 1), 3: (1, 3), 8: (1, 8), 16: (2, 8), 24: (3, 8), 32: (4, 8)}
    b, length = shapes[rows]
    for nq in (1, 3):
        mixed = mx.random.normal((b, length, nq, 2304))
        messages = mx.random.normal((b, length, 4, 2304))
        norms = mx.fast.rms_norm(messages, None, ctrls["attn"].eps)
        got = F.qkv_controllers(layer, mixed, messages, norms)
        ref = mx.stack([ctrls[n](mixed[..., min(i, nq - 1), :], messages[..., i, :])
                        for i, n in enumerate(("attn", "k", "v"))], axis=-2)
        mx.eval(got, ref)
        same = bool(mx.array_equal(got, ref))
        RESULTS[f"controllers_ckpt/{'bf16' if bf16 else 'fp32'}/rows{rows}/nq{nq}"] = {
            "bitwise": same, "max_abs": float(mx.abs(got - ref).max().item())}
        assert same, f"rows {rows} nq {nq}: max {mx.abs(got - ref).max().item()}"


@pytest.mark.parametrize("bf16", [False, True])
def test_sconv_kernel_near_exact(bf16):
    """The one-launch conv against the exact path, module level, every tail."""
    from mlx2.runtime.models.inkling.cache import InklingConvolutionCache

    model = _model("wide", bf16)
    model.configure_kernels(fuse_sconv=True)
    conv = model.model.layers[0].attn_sconv
    dtype = mx.bfloat16 if bf16 else mx.float32
    worst, bitwise = 0.0, True
    for rows in (1, 2):
        for length in (1, 8):
            x = mx.random.normal((rows, length, 512)).astype(dtype)
            xr = mx.random.normal((rows, length, 512)).astype(dtype)
            res = mx.random.normal((rows, length, 512))
            state = mx.random.normal((rows, 3, 512))
            for tail, extra in (("plain", ()), ("attn_boundary", ()), ("attn_residual", (xr,)),
                                ("mlp_residual", (res, xr))):
                outs = []
                for kernel in (False, True):
                    names = {"fuse_sconv"} | ({"fuse_sconv_kernel"} if kernel else set())
                    object.__setattr__(conv, "_fuse", frozenset(names))
                    cache = InklingConvolutionCache()
                    cache[2] = state
                    out = conv(x, cache, None, tail=tail, extra=extra)
                    mx.eval(out, cache[2])
                    outs.append((out, cache[2]))
                (a, sa), (b, sb) = outs
                assert a.dtype == b.dtype and a.shape == b.shape
                assert mx.array_equal(sa, sb)
                d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()
                bitwise &= d == 0
                worst = max(worst, d / max(1e-6, mx.abs(a.astype(mx.float32)).max().item()))
    assert conv.fuse_counts.get("sconv_kernel", 0) > 0
    RESULTS[f"sconv_kernel/{'bf16' if bf16 else 'fp32'}"] = {"bitwise": bitwise,
                                                             "max_rel": worst}
    assert worst <= (1e-2 if bf16 else 1e-5)


def test_residual_kernel_near_exact():
    from mlx2.runtime.models.inkling2.language import _attention_residual

    model = _model("wide", True)
    model.configure_kernels(fuse_residual_kernel=True)
    module = model.model.layers[7].attention_input
    worst = 0.0
    for n in (1, 4, 9):
        blocks = [mx.random.normal((1, 1, 512)).astype(mx.bfloat16) for _ in range(n)]
        partial = mx.random.normal((1, 1, 512))
        ref = _attention_residual(mx.stack([*blocks, partial], axis=-2), module.query, True)
        got = module(blocks, partial)
        assert got.shape == ref.shape and got.dtype == ref.dtype
        worst = max(worst, mx.abs(got - ref).max().item())
    assert module.fuse_counts["residual_kernel"] == 3
    RESULTS["residual_kernel/module"] = {"max_abs": worst}
    assert worst < 1e-5


@pytest.mark.parametrize("bf16", [False, True])
def test_kernel_options_model_level(bf16):
    """All options, kernels included, on one model: logit gap and greedy
    agreement against the reference (near-exact, so recorded, not bitwise)."""
    model = _model("wide", bf16)
    model.configure_kernels(moe_fused=True, moe_weighted_sum=True)
    tokens = mx.random.randint(0, 500, (1, 40), key=mx.random.key(7))
    base = _run(model, tokens, PLAN)
    model.configure_kernels(moe_fused=True, moe_weighted_sum=True, **ALL,
                            fuse_sconv_kernel=True, fuse_residual_kernel=True)
    got = _run(model, tokens, PLAN)
    counts = model.kernel_receipt()["fuse_counts"]
    gap = mx.abs(got - base).max().item()
    agree = (mx.argmax(got, -1) == mx.argmax(base, -1)).mean().item()
    RESULTS[f"kernels_model/{'bf16' if bf16 else 'fp32'}"] = {
        "max_abs_logit": gap, "argmax_agreement": agree,
        "sconv_kernel": counts.get("sconv_kernel", 0),
        "residual_kernel": counts.get("residual_kernel", 0),
        "message_kernel": counts.get("message_kernel", 0)}
    assert counts.get("sconv_kernel", 0) > 0 and counts.get("residual_kernel", 0) > 0
    assert counts.get("message_kernel", 0) > 0
    assert gap < (0.5 if bf16 else 1e-3)


def test_gpu_graph_launch_counts():
    """GPU-stream dot-graph kernel counts of one B1 decode step (BF16),
    reference against each option (recorded)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "research"))
    import inkling2_launch_count as lc

    def kernels(**options):
        model = _model("wide", True)
        model.configure_kernels(moe_fused=True, moe_weighted_sum=True, **options)
        out, cache = lc.decode_step(model, device="gpu", prefill_gpu=True)
        return lc.count_graph(out, *lc.cache_arrays(cache))["kernels"]

    counts = {"reference": kernels()}
    for name in EXACT:
        counts[name] = kernels(**{name: True})
    counts["all_exact"] = kernels(**ALL)
    counts["all_with_kernels"] = kernels(**ALL, fuse_sconv_kernel=True,
                                         fuse_residual_kernel=True)
    RESULTS["gpu_graph_kernels_wide_bf16"] = counts
    assert counts["all_with_kernels"] < counts["all_exact"] < counts["reference"]


# ----- 2026-11-22 options from omlx#5538 / Rapid-MLX (reimplemented) -----

DEFAULTS = {**ALL, "fuse_sconv_kernel": True}  # the shipped fusion defaults


def _gap(got, base):
    diff = mx.abs(got.astype(mx.float32) - base.astype(mx.float32))
    return {"bitwise": bool(mx.array_equal(got, base)), "max_abs": float(diff.max().item()),
            "argmax_agreement": float((mx.argmax(got, -1) == mx.argmax(base, -1)).mean().item())}


@pytest.mark.parametrize("geometry", ["tiny", "wide"])
@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("option", ["fuse_norm_fp32", "fuse_shared_tau", "head_softcap_compiled"])
def test_exact_options_2_bitwise_on_gpu(option, rows, bf16, geometry):
    """Intended bit-identical options on top of the shipped defaults."""
    model = _model(geometry, bf16)
    model.configure_kernels(moe_fused=True, moe_weighted_sum=True, **DEFAULTS)
    tokens = mx.random.randint(0, 500, (rows, 40), key=mx.random.key(rows))
    base = _run(model, tokens, PLAN)
    model.configure_kernels(moe_fused=True, moe_weighted_sum=True, **DEFAULTS, **{option: True})
    got = _run(model, tokens, PLAN)
    receipt = model.kernel_receipt()
    result = _gap(got, base)
    result["counts"] = {k: v for k, v in receipt["fuse_counts"].items()
                        if k in ("norm_fp32", "tau_shared_hits", "tau_skipped")}
    result["head_softcap_compiled_calls"] = model.__dict__.get("head_softcap_compiled_calls", 0)
    RESULTS[f"exact2/{option}/{geometry}/{'bf16' if bf16 else 'fp32'}/B{rows}"] = result
    assert result["bitwise"], result


@pytest.mark.parametrize("kind", ["affine4", "mxfp8", "nvfp4"])
def test_gather_lhs_bitwise_on_gpu(kind):
    from mlx2.runtime.models import gather_lhs
    from mlx2.runtime.models.switch_layers import QuantizedSwitchLinear
    from mlx2.runtime.nvfp4_linear import ScaledNVFP4SwitchLinear

    mx.random.seed(5)
    w = mx.random.normal((8, 128, 256)) * 0.1
    if kind == "nvfp4":
        switch = ScaledNVFP4SwitchLinear(256, 128, 8)
        q = mx.quantize(w, group_size=16, bits=4, mode="nvfp4")
        switch.weight, switch.scales = q[0], q[1]
        switch.weight_scale_2 = mx.random.uniform(0.5, 2.0, (8,))
    else:
        mode, bits, group = ("affine", 4, 64) if kind == "affine4" else ("mxfp8", 8, 32)
        switch = QuantizedSwitchLinear(256, 128, 8, bias=False, group_size=group, bits=bits,
                                       mode=mode)
        q = mx.quantize(w, group_size=group, bits=bits, mode=mode)
        switch.weight, switch.scales = q[0], q[1]
        if len(q) > 2 and q[2] is not None:
            switch.biases = q[2]
    same = True
    for shape in ((1, 1, 8), (2, 1, 8), (1, 8, 8)):
        x = mx.random.normal((*shape[:-1], 1, 1, 256)).astype(mx.bfloat16)
        idx = mx.random.randint(0, 8, shape).astype(mx.uint32)
        gather_lhs.set_enabled(switch, False)
        ref = switch(x, idx)
        gather_lhs.set_enabled(switch, True)
        got = switch(x, idx)
        mx.eval(ref, got)
        same &= bool(mx.array_equal(got, ref))
    RESULTS[f"gather_lhs/{kind}"] = {"bitwise": same}
    assert same


@pytest.mark.parametrize("fma", [0, 1])
@pytest.mark.parametrize("bf16", [False, True])
def test_dconv_kernel_vs_compiled_reference(bf16, fma):
    """The one-launch dynamic conv against ``fuse._dynamic_conv`` (module
    level, 1 and 8 rows, with and without state), recording which FMA
    setting is bitwise; the model-level gap is in the next test."""
    from mlx2.runtime.models.inkling.cache import InklingConvolutionCache
    from mlx2.runtime.models.inkling2 import fuse_kernels as K

    model = _model("wide", bf16)
    model.configure_kernels(fuse_sconv=True)
    layer = next(l for l in model.model.layers if not l.is_sliding)
    conv = layer.self_attn.k_sconv
    dtype = mx.bfloat16 if bf16 else mx.float32
    channels = conv.static_weight.shape[1]
    K.DCONV_FMA = fma
    worst, bitwise = 0.0, True
    try:
        for rows in (1, 2):
            for length in (1, 8):
                x = mx.random.normal((rows, length, channels)).astype(dtype)
                dyn_in = mx.random.normal((rows, length, model.config.hidden_size)).astype(dtype)
                for with_state in (False, True):
                    outs = []
                    for kernel in (False, True):
                        names = {"fuse_sconv"} | ({"fuse_dconv_kernel"} if kernel else set())
                        object.__setattr__(conv, "_fuse", frozenset(names))
                        cache = InklingConvolutionCache()
                        if with_state:
                            cache[conv.conv_idx] = mx.random.normal((rows, 3, channels),
                                                                    key=mx.random.key(1))
                        out = conv(x, dyn_in, cache, None)
                        mx.eval(out, cache[conv.conv_idx])
                        outs.append((out, cache[conv.conv_idx]))
                    (a, sa), (b, sb) = outs
                    assert mx.array_equal(sa, sb)
                    d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()
                    bitwise &= d == 0
                    worst = max(worst, d)
    finally:
        K.DCONV_FMA = 0
    assert conv.fuse_counts.get("dconv_kernel", 0) > 0
    RESULTS[f"dconv_kernel/{'bf16' if bf16 else 'fp32'}/fma{fma}"] = {"bitwise": bitwise,
                                                                      "max_abs": worst}
    assert worst <= (0.05 if bf16 else 1e-4)


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("option", ["fuse_dconv_kernel", "fuse_message_kernel",
                                    "compile_decode_blocks", "fuse_gather_lhs"])
def test_new_kernel_options_model_level(option, bf16):
    """Each new near-exact option (and the gather cache) on the shipped
    defaults: logit gap and greedy agreement, recorded."""
    model = _model("wide", bf16)
    model.configure_kernels(moe_fused=True, moe_weighted_sum=True, **DEFAULTS)
    tokens = mx.random.randint(0, 500, (1, 40), key=mx.random.key(7))
    base = _run(model, tokens, PLAN)
    model.configure_kernels(moe_fused=True, moe_weighted_sum=True, **DEFAULTS, **{option: True})
    got = _run(model, tokens, PLAN)
    receipt = model.kernel_receipt()
    result = _gap(got, base)
    result["counts"] = {k: v for k, v in receipt["fuse_counts"].items()
                        if k in ("dconv_kernel", "message_mix_kernel")}
    result["compiled_moe_calls"] = receipt["compiled_moe_calls"]
    RESULTS[f"new_model/{option}/{'bf16' if bf16 else 'fp32'}"] = result
    assert result["max_abs"] < (1.0 if bf16 else 1e-3), result


@pytest.mark.parametrize("bf16", [False, True])
def test_safe_math_residual_kernel_gap(bf16):
    """``fuse_kernels.SAFE_MATH`` (fp math_mode safe, contract off) against
    the default build for the residual kernel's model-level gap."""
    from mlx2.runtime.models.inkling2 import fuse_kernels as K

    out = {}
    for safe in (False, True):
        K.SAFE_MATH = safe
        try:
            model = _model("wide", bf16)
            model.configure_kernels(moe_fused=True, moe_weighted_sum=True, **DEFAULTS)
            tokens = mx.random.randint(0, 500, (1, 40), key=mx.random.key(7))
            base = _run(model, tokens, PLAN)
            model.configure_kernels(moe_fused=True, moe_weighted_sum=True, **DEFAULTS,
                                    fuse_residual_kernel=True)
            out["safe" if safe else "default"] = _gap(_run(model, tokens, PLAN), base)
        finally:
            K.SAFE_MATH = False
    RESULTS[f"safe_math_residual/{'bf16' if bf16 else 'fp32'}"] = out


def test_gpu_graph_launch_counts_new_options():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "research"))
    import inkling2_launch_count as lc

    def kernels(**options):
        model = _model("wide", True)
        model.configure_kernels(moe_fused=True, moe_weighted_sum=True, **DEFAULTS, **options)
        out, cache = lc.decode_step(model, device="gpu", prefill_gpu=True)
        return lc.count_graph(out, *lc.cache_arrays(cache))["kernels"]

    counts = {"defaults": kernels()}
    for name in ("fuse_norm_fp32", "fuse_shared_tau", "fuse_dconv_kernel",
                 "fuse_message_kernel", "head_softcap_compiled", "fuse_gather_lhs"):
        counts[name] = kernels(**{name: True})
    counts["all_new"] = kernels(fuse_norm_fp32=True, fuse_shared_tau=True,
                                fuse_dconv_kernel=True, fuse_message_kernel=True,
                                head_softcap_compiled=True, fuse_gather_lhs=True)
    RESULTS["gpu_graph_kernels_new_options"] = counts
