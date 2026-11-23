"""``inkling2_fuse_*`` launch-count fusions, CPU only.

The four exact options must be bit-identical to the reference across
prefill, chunked prefill, decode, B1 and B2, the ring wrap (window 8), left
padding, log scaling below and above its floor, FP32 and BF16 weights, a
speculative trim, and DSpark verify.  The two near-exact kernel options are
GPU only: on the CPU they decline (counted) and the exact path runs.
"""

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx2.runtime.models.inkling2 import Model, TextConfig
from test_inkling2_cpu import TINY, _run, _tiny_model

EXACT = ("fuse_controllers", "fuse_residual", "fuse_sconv", "fuse_attention")
ALL = {name: True for name in EXACT}


@pytest.fixture(autouse=True)
def _cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def _bf16(model):
    # CPU GatherMM is FP32-only: routed experts stay FP32.
    model.load_weights([(k, v if ".switch_mlp." in k else v.astype(mx.bfloat16))
                        for k, v in tree_flatten(model.parameters())])
    mx.eval(model.parameters())
    return model


def _model(floor=None, bf16=False, seed=0):
    if floor is None:
        model = _tiny_model(seed)
    else:
        mx.random.seed(seed)
        model = Model(TextConfig.from_dict({**TINY, "log_scaling_n_floor": floor}))
        ref = _tiny_model(seed)
        model.load_weights(tree_flatten(ref.parameters()), strict=True)
        mx.eval(model.parameters())
    return _bf16(model) if bf16 else model


PLAN = [(0, 20), (20, 27)] + [(i, i + 1) for i in range(27, 40)]


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("option", [*EXACT, "all"])
def test_exact_options_are_bitwise_reference(option, rows, bf16):
    """Chunked prefill then 13 decode steps; window 8 wraps the ring; log
    floor 4 keeps tau above 1 on every global call."""
    model = _model(bf16=bf16)
    tokens = mx.random.randint(0, 500, (rows, 40))
    base = _run(model, tokens, PLAN)
    model.configure_kernels(**(ALL if option == "all" else {option: True}))
    got = _run(model, tokens, PLAN)
    counts = model.kernel_receipt()["fuse_counts"]
    model.configure_kernels()
    assert mx.array_equal(got, base)
    assert counts, "the option never engaged"
    assert mx.array_equal(_run(model, tokens, PLAN), base)


def test_engagement_counters():
    model = _model()
    tokens = mx.random.randint(0, 500, (1, 40))
    model.configure_kernels(**ALL)
    _run(model, tokens, PLAN)
    counts = model.kernel_receipt()["fuse_counts"]
    for key in ("controllers_batched", "input_norms_batched", "controllers_unbatched",
                "residual_mix", "sconv_plain", "sconv_attn_residual", "sconv_attn_boundary",
                "sconv_mlp_residual", "dynamic_sconv", "mask_index_built", "mask_index_reused",
                "ring_sink_concat", "gate_finish"):
        assert counts.get(key, 0) > 0, key
    # One index build per distinct key per forward: 2 per decode step
    # (local and global), far fewer than the 48 visits.
    assert counts["mask_index_reused"] > 10 * counts["mask_index_built"] / 2


@pytest.mark.parametrize("bf16", [False, True])
def test_log_scaling_skip_below_floor_and_compute_above(bf16):
    """Floor 24: tau is exactly 1 for the first 24 positions (skipped) and
    computed after (compiled chain)."""
    model = _model(floor=24, bf16=bf16)
    tokens = mx.random.randint(0, 500, (1, 40))
    plan = [(0, 12)] + [(i, i + 1) for i in range(12, 30)] + [(30, 40)]
    base = _run(model, tokens, plan)
    model.configure_kernels(fuse_attention=True)
    got = _run(model, tokens, plan)
    counts = model.kernel_receipt()["fuse_counts"]
    model.configure_kernels()
    assert counts.get("tau_skipped", 0) > 0
    assert mx.array_equal(got, base)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_mask_with_left_padding_is_bitwise_reference(dtype):
    """``fuse.banded_mask`` against the v1 reference mask, padded and not,
    local and global, prefill rows and one decode row; the second call with
    the same key reuses the memoized index."""
    from mlx2.runtime.models.inkling.language import banded_additive_mask
    from mlx2.runtime.models.inkling2 import fuse

    attn = _model().model.layers[0].self_attn
    holder = {"memo": {}}
    object.__setattr__(attn, "_fuse_holder", holder)
    proj = (mx.random.normal((16, 8)) * 0.3).astype(dtype)
    pad = mx.array([3, -1])
    for lq, size, sliding, extent in ((5, 12, 8, 8), (1, 12, 8, 8), (5, 30, 0, 8), (1, 30, 0, 8)):
        rel = mx.random.normal((2, lq, 4, 16)).astype(dtype)
        for lp in (None, pad):
            for _ in range(2):
                ref = banded_additive_mask(rel, proj, size - lq, size, sliding, extent,
                                           left_padding=lp)
                got = fuse.banded_mask(attn, rel, proj, size - lq, size, sliding, extent,
                                       left_padding=lp)
                assert got.dtype == ref.dtype and mx.array_equal(got, ref)
    assert attn.fuse_counts["mask_index_reused"] > 0


def test_masked_short_convolution_is_bitwise_reference():
    """A padding mask keeps ``_next_conv_state`` (no context view)."""
    from mlx2.runtime.models.inkling.cache import InklingConvolutionCache

    model = _model()
    conv = model.model.layers[0].attn_sconv
    x = mx.random.normal((2, 5, 128))
    mask = mx.array([[False, False, True, True, True], [True] * 5])
    outs = {}
    for fused in (False, True):
        object.__setattr__(conv, "_fuse", frozenset({"fuse_sconv"} if fused else ()))
        cache = InklingConvolutionCache()
        cache[2] = mx.random.normal((2, 3, 128), key=mx.random.key(1))
        out = conv(x, cache, mask)
        outs[fused] = (out, cache[2])
    object.__setattr__(conv, "_fuse", frozenset())
    assert mx.array_equal(outs[True][0], outs[False][0])
    assert mx.array_equal(outs[True][1], outs[False][1])


def test_with_other_options_and_trim():
    """Together with fused projections and the sink slot, across a
    speculative-style KV trim (the fused conv state is a context view)."""
    from mlx2.runtime.models.cache import KVCache

    model = _model()
    tokens = mx.random.randint(0, 500, (1, 30))

    def run():
        cache = model.make_cache()
        outs = [model(tokens[:, :20], cache=cache)]
        outs.append(model(tokens[:, 20:25], cache=cache))
        for plane in cache:
            if type(plane) is KVCache:
                plane.trim(3)
        outs.append(model(tokens[:, 22:23], cache=cache))
        outs += [model(tokens[:, i : i + 1], cache=cache) for i in range(23, 30)]
        return mx.concatenate(outs, axis=1)

    model.configure_kernels(fused_projections=True, sink_slot=True)
    base = run()
    model.configure_kernels(fused_projections=True, sink_slot=True, **ALL)
    got = run()
    model.configure_kernels()
    assert mx.array_equal(got, base)


def test_kernel_options_decline_on_cpu_and_stay_exact():
    model = _model()
    tokens = mx.random.randint(0, 500, (1, 30))
    plan = [(0, 20)] + [(i, i + 1) for i in range(20, 30)]
    base = _run(model, tokens, plan)
    model.configure_kernels(**ALL, fuse_sconv_kernel=True, fuse_residual_kernel=True)
    got = _run(model, tokens, plan)
    counts = model.kernel_receipt()["fuse_counts"]
    model.configure_kernels()
    assert mx.array_equal(got, base)
    assert counts["kernel_declined:not_gpu"] > 0
    assert "sconv_kernel" not in counts and "residual_kernel" not in counts


def test_policy_names_and_dependency():
    from mlx2.adapters.inkling2 import KERNEL_OPTIONS, REFERENCE_POLICY, validate_policy

    # The four exact options and the short-conv kernel are default on since
    # 2026-11-22 (Pierre); the residual kernel stays off.
    for name in EXACT + ("fuse_sconv_kernel",):
        assert KERNEL_OPTIONS["inkling2_" + name] == name
        assert validate_policy(None)[0][name] is True
    assert KERNEL_OPTIONS["inkling2_fuse_residual_kernel"] == "fuse_residual_kernel"
    assert validate_policy(None)[0]["fuse_residual_kernel"] is False
    # Unset, the short-conv kernel follows fuse_sconv; set, it must agree.
    assert validate_policy({"inkling2_fuse_sconv": False})[0]["fuse_sconv_kernel"] is False
    assert validate_policy({"inkling2_fuse_sconv_kernel": False})[0]["fuse_sconv_kernel"] is False
    assert REFERENCE_POLICY["inkling2_fuse_sconv_kernel"] is False
    with pytest.raises(ValueError, match="fuse_sconv_kernel requires"):
        validate_policy({"inkling2_fuse_sconv": False, "inkling2_fuse_sconv_kernel": True})
    with pytest.raises(ValueError, match="requires fuse_sconv"):
        _tiny_model().configure_kernels(fuse_sconv_kernel=True)


def test_launch_count_drops():
    """Dot-graph kernels of one B1 decode step (the CPU graph; see
    scripts/research/inkling2_launch_count.py for the GPU-stream count)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "research"))
    import inkling2_launch_count as lc

    def kernels(**options):
        model = _model(bf16=True)
        model.configure_kernels(**options)
        out, cache = lc.decode_step(model)
        return lc.count_graph(out, *lc.cache_arrays(cache))["kernels"]

    base = kernels()
    each = {name: kernels(**{name: True}) for name in EXACT}
    assert all(value < base for value in each.values()), each
    assert kernels(**ALL) < 0.75 * base


# ----- 2026-11-22 options from omlx#5538 / Rapid-MLX (reimplemented) -----

EXACT_2 = ("fuse_norm_fp32", "fuse_shared_tau")


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("option", [*EXACT_2, "all2"])
def test_exact_options_2_are_bitwise_defaults(option, rows, bf16):
    """FP32 norm weights and one shared log-tau on top of the four exact
    options: bit-identical to the four alone (log floor 4: tau computed on
    every global call, so the shared tau engages)."""
    model = _model(bf16=bf16)
    tokens = mx.random.randint(0, 500, (rows, 40))
    model.configure_kernels(**ALL)
    base = _run(model, tokens, PLAN)
    extra = {name: True for name in EXACT_2} if option == "all2" else {option: True}
    model.configure_kernels(**ALL, **extra)
    got = _run(model, tokens, PLAN)
    counts = model.kernel_receipt()["fuse_counts"]
    model.configure_kernels()
    assert mx.array_equal(got, base)
    if option in ("fuse_norm_fp32", "all2") and bf16:
        assert counts.get("norm_fp32", 0) > 0
    if option in ("fuse_shared_tau", "all2"):
        assert counts.get("tau_shared_hits", 0) > 0


@pytest.mark.parametrize("bits,mode", [(4, "affine"), (8, "affine")])
def test_gather_lhs_cache_is_bitwise(bits, mode):
    """Cached lhs indices (``fuse_gather_lhs``) equal gather_qmm's default."""
    from mlx2.runtime.models import gather_lhs
    from mlx2.runtime.models.switch_layers import QuantizedSwitchLinear

    mx.random.seed(3)
    switch = QuantizedSwitchLinear(64, 32, 8, bias=False, group_size=32, bits=bits, mode=mode)
    w = mx.random.normal((8, 32, 64))
    q = mx.quantize(w, group_size=32, bits=bits, mode=mode)
    switch.weight, switch.scales = q[0], q[1]
    if len(q) > 2:
        switch.biases = q[2]
    for shape in ((1, 1, 2), (2, 1, 2), (1, 3, 2), (4, 2)):
        x = mx.random.normal((*shape[:-1], 1, 1, 64))
        idx = mx.random.randint(0, 8, shape).astype(mx.uint32)
        gather_lhs.set_enabled(switch, False)
        ref = switch(x, idx)
        gather_lhs.set_enabled(switch, True)
        got = switch(x, idx)
        assert mx.array_equal(got, ref), shape
    assert switch.__dict__["_lhs_cache_counts"]["hits"] == 4
    gather_lhs.set_enabled(switch, True)
    sorted_out = switch(mx.random.normal((4, 1, 64)), mx.array([0, 1, 1, 5], mx.uint32),
                        sorted_indices=True)
    assert sorted_out.shape[0] == 4
    assert switch.__dict__["_lhs_cache_counts"]["hits"] == 4  # sorted calls never use it


def test_new_gpu_options_decline_or_stay_off_on_cpu():
    """The near-exact kernels, the compiled MoE block and the compiled head
    softcap are GPU only: on the CPU the exact path runs, bit-identically."""
    model = _model()
    tokens = mx.random.randint(0, 500, (1, 40))
    model.configure_kernels(**ALL)
    base = _run(model, tokens, PLAN)
    model.configure_kernels(**ALL, fuse_dconv_kernel=True, fuse_message_kernel=True,
                            head_softcap_compiled=True)
    got = _run(model, tokens, PLAN)
    counts = model.kernel_receipt()["fuse_counts"]
    model.configure_kernels()
    assert mx.array_equal(got, base)
    assert counts.get("kernel_declined:not_gpu", 0) > 0
    assert counts.get("kernel_declined:message_not_gpu", 0) > 0
    assert "dconv_kernel" not in counts and "message_mix_kernel" not in counts
    with pytest.raises(ValueError, match="fuse_dconv_kernel requires"):
        model.configure_kernels(fuse_dconv_kernel=True)
    with pytest.raises(ValueError, match="fuse_message_kernel requires"):
        model.configure_kernels(fuse_message_kernel=True)


def test_compiled_moe_uses_live_router_weights_and_counts_calls():
    from mlx2.runtime.models.inkling2 import moe_fast

    model = _model()
    model.configure_kernels(moe_fused=True)
    moe = model.model.layers[0].mlp
    experts, refusal = moe._fast_experts()
    assert refusal is None
    row = mx.random.normal((1, 1, model.config.hidden_size), key=mx.random.key(13))
    for _ in range(2):
        got = moe._compiled_moe(experts, False, row)
        expected = moe_fast.fused_call(moe, experts, row)
        assert mx.allclose(got, expected)
        moe.gate_weight = -moe.gate_weight
    receipt = model.kernel_receipt()
    assert receipt["compiled_moe_calls"] == 2
    assert receipt["moe_fused_calls"] == 4
    assert receipt["moe_fast_declines"] == {}


def test_shared_tau_engages_with_attention_fusion_off():
    model = _model(floor=4)
    tokens = mx.random.randint(0, 500, (1, 12), key=mx.random.key(14))
    baseline = model(tokens)
    model.configure_kernels(fuse_shared_tau=True)
    assert mx.array_equal(model(tokens), baseline)
    assert model.kernel_receipt()["fuse_counts"]["tau_shared_hits"] > 0
