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
