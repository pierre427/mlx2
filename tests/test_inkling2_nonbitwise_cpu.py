"""Inkling 2 opt-in prefill/head/compile options, CPU only.

Each option is default off. Off, the model is the reference bitwise; on, it
is within tolerance of the reference on the tiny model (window 8) over
chunked prefill + decode, B1/B2 and left padding, and its counters show it
engaged. GPU-only kernels are exercised here through their CPU emulations
(the Metal sources run in tests/test_inkling2_nonbitwise_metal.py).
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from test_inkling2_cpu import _modelopt_nvfp4, _run, _tiny_model

from mlx2.runtime.models.cache import BatchKVCache, BatchRotatingKVCache, KVCache

PLAN = [(0, 20), (20, 27)] + [(i, i + 1) for i in range(27, 36)]


@pytest.fixture(autouse=True)
def _cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def _bf16(model):
    # CPU GatherMM is FP32-only: dense routed experts stay FP32.
    model.load_weights([(k, v if ".switch_mlp." in k else v.astype(mx.bfloat16))
                        for k, v in tree_flatten(model.parameters())])
    mx.eval(model.parameters())
    return model


def _quantized_model():
    """The checkpoint's expert formats on the tiny geometry: NVFP4 (scaled,
    per-expert global scale) on layers 0-19, MXFP8 on 20-23, BF16 elsewhere."""
    from mlx2.adapters.inkling2 import MXFP8, NVFP4, install_experts

    model = _tiny_model()
    num, hidden, inter = 8, 128, 64
    layers, kinds = {}, {}
    for i in range(24):
        mx.random.seed(100 + i)
        w13 = mx.random.normal((num, 2 * inter, hidden)) * 0.05
        w2 = mx.random.normal((num, hidden, inter)) * 0.05
        if i < 20:
            a, b = _modelopt_nvfp4(w13, num), _modelopt_nvfp4(w2, num)
            layers[i], kinds[i] = {
                "w13_weight": a["weight"], "w13_weight_scale": a["scale"],
                "w13_weight_amax": a["amax"],
                "w13_weight_outer_scale_denominator": a["denominator"],
                "w2_weight": b["weight"], "w2_weight_scale": b["scale"],
                "w2_weight_amax": b["amax"],
                "w2_weight_outer_scale_denominator": b["denominator"]}, NVFP4
        else:
            q13, s13 = mx.quantize(w13, group_size=32, bits=8, mode="mxfp8")
            q2, s2 = mx.quantize(w2, group_size=32, bits=8, mode="mxfp8")
            layers[i], kinds[i] = {"w13_weight": q13.view(mx.uint8), "w13_weight_scale": s13,
                                   "w2_weight": q2.view(mx.uint8),
                                   "w2_weight_scale": s2}, MXFP8
    install_experts(model, layers, kinds)
    model.load_weights([(k, v.astype(mx.bfloat16) if v.dtype == mx.float32
                         and ".switch_mlp." not in k else v)
                        for k, v in tree_flatten(model.parameters())])
    mx.eval(model.parameters())
    return model


def _padded_run(model, tokens, plan, pads=(0, 3)):
    """B2 with left padding: global visits on BatchKVCache, sliding visits on
    BatchRotatingKVCache (window 8)."""
    cache = model.make_cache()
    for i in range(0, len(cache), 2):
        cache[i] = (BatchKVCache(list(pads)) if type(cache[i]) is KVCache
                    else BatchRotatingKVCache(8, list(pads)))
    return mx.concatenate([model(tokens[:, a:b], cache=cache) for a, b in plan], axis=1)


# ----- inkling2_compile_decode_blocks (item 3) -----


def _allow_compiled_moe_on_cpu(monkeypatch):
    from mlx2.runtime.models.inkling2 import language

    monkeypatch.setattr(language, "_compiled_moe_device", lambda: True)


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("compiled_first", [False, True])
def test_compiled_decode_moe_traces_and_is_bitwise_eager(monkeypatch, quantized, compiled_first):
    """The compiled one-token MoE block traces (no uncaptured inputs) and
    equals the eager fused path bitwise through prefill + decode. Quantized:
    NVFP4 + MXFP8 experts with BF16 router weights (the FP32 router copy is
    an explicit argument); dense: FP32 router weights (the same array is
    also module state). ``compiled_first``: the trace builds the gather_lhs
    constants before any eager call does."""
    model = _quantized_model() if quantized else _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 36), key=mx.random.key(5))
    base_opts = dict(moe_fused=True, fuse_gather_lhs=True, fuse_attention=True,
                     fuse_controllers=True, fuse_residual=True, fuse_sconv=True,
                     sink_slot=True)
    _allow_compiled_moe_on_cpu(monkeypatch)

    def eager():
        model.configure_kernels(**base_opts)
        return _run(model, tokens, PLAN)

    def compiled():
        model.configure_kernels(**base_opts, compile_decode_blocks=True)
        out = _run(model, tokens, PLAN)
        return out, model.kernel_receipt()

    if compiled_first:
        got, receipt = compiled()
        base = eager()
    else:
        base = eager()
        got, receipt = compiled()
    decode_steps = sum(1 for a, b in PLAN if b - a == 1)
    assert receipt["compiled_moe_calls"] == 48 * decode_steps
    assert mx.array_equal(got, base)
    # Off again: the reference path, bitwise.
    model.configure_kernels()
    ref = _run(model, tokens, PLAN)
    model.configure_kernels(**base_opts)
    assert mx.array_equal(_run(model, tokens, PLAN), base)
    assert mx.abs(ref - base).max().item() < 1e-3


def test_compiled_decode_moe_declines_multi_row_calls(monkeypatch):
    """B2 decode (and prefill) rows are not one-token blocks: the eager fused
    path runs, counted, so B2 with left padding is unchanged bitwise."""
    model = _quantized_model()
    tokens = mx.random.randint(0, 500, (2, 30), key=mx.random.key(6))
    plan = [(0, 12), (12, 20)] + [(i, i + 1) for i in range(20, 30)]
    model.configure_kernels(moe_fused=True)
    base = _padded_run(model, tokens, plan)
    _allow_compiled_moe_on_cpu(monkeypatch)
    model.configure_kernels(moe_fused=True, compile_decode_blocks=True)
    got = _padded_run(model, tokens, plan)
    receipt = model.kernel_receipt()
    assert receipt["compiled_moe_calls"] == 0 and receipt["moe_fused_calls"] > 0
    assert mx.array_equal(got, base)


def test_compiled_decode_moe_traces_once_per_layer_and_keeps_the_fused_build(monkeypatch):
    """A trace puts the original arrays back into the captured modules as new
    Python objects. The identity-keyed caches are re-stamped, so the stacked
    gate/up build is not redone and the block does not retrace per token;
    the FP32 router copy (BF16 router weights) is cast once, outside the
    trace, and reused."""
    from mlx2.runtime.models.inkling2 import moe_fast

    model = _quantized_model()
    _allow_compiled_moe_on_cpu(monkeypatch)
    model.configure_kernels(moe_fused=True, compile_decode_blocks=True)
    builds = []
    original = moe_fast.build
    monkeypatch.setattr(moe_fast, "build", lambda sw: builds.append(1) or original(sw))
    moe = model.model.layers[0].mlp
    assert moe.gate_weight.dtype == mx.bfloat16
    cache = model.make_cache()
    tokens = mx.random.randint(0, 500, (1, 12), key=mx.random.key(7))
    model(tokens[:, :6], cache=cache)
    router = moe.__dict__["_fp32_gate_weight"][1]
    fns = []
    for i in range(6, 12):
        mx.eval(model(tokens[:, i:i + 1], cache=cache))
        fns.append(moe.__dict__["_compiled_moe_fn"])
        assert moe.__dict__["_fp32_gate_weight"][1] is router
        assert moe.__dict__["_fp32_gate_weight"][0] is moe.gate_weight
    receipt = model.kernel_receipt()
    assert builds == []
    assert all(f is fns[0] for f in fns)
    assert receipt["compiled_moe_traces"] == 24  # one per physical layer
    assert receipt["compiled_moe_calls"] == 48 * 6
    # A real replacement is still seen: the router copy follows it.
    moe.gate_weight = -moe.gate_weight
    mx.eval(model(tokens[:, 11:12], cache=cache))
    assert moe.__dict__["_fp32_gate_weight"][1] is not router


# ----- inkling2_prefill_mask_bf16_budget (item 1a) -----


def _gap(got, base):
    return {"max_abs": float(mx.abs(got - base).max().item()),
            "argmax_equal": float((mx.argmax(got, -1) == mx.argmax(base, -1)).mean().item()),
            "bitwise": bool(mx.array_equal(got, base))}


GAPS = {}


def _record(key, gap):
    GAPS[key] = gap
    print("gap", key, gap)  # visible with -s: the measured CPU gap


def test_policy_defaults_off():
    from mlx2.adapters.inkling2 import KERNEL_OPTIONS, validate_policy

    names = ["prefill_mask_bf16_budget", "prefill_mask_with_sink", "head_kernel_wide",
             "prefill_tiled_attention"]
    kernels = validate_policy(None)[0]
    for name in names:
        assert KERNEL_OPTIONS["inkling2_" + name] == name
        assert kernels[name] is False
        extra = {"inkling2_head_kernel": True} if name == "head_kernel_wide" else {}
        assert validate_policy({"inkling2_" + name: True, **extra})[0][name] is True
        with pytest.raises(TypeError):
            validate_policy({"inkling2_" + name: 1})


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("bf16", [False, True])
def test_mask_bf16_budget_doubles_bf16_slices(monkeypatch, bf16, padded):
    """A small budget forces several mask slices; BF16 slices get twice the
    rows with the option on, FP32 is untouched (bitwise). Prefill + decode,
    B1 or B2 with left padding."""
    from mlx2.runtime.models.inkling2 import language

    model = _bf16(_tiny_model()) if bf16 else _tiny_model()
    rows = 2 if padded else 1
    tokens = mx.random.randint(0, 500, (rows, 36), key=mx.random.key(8))
    run = (lambda: _padded_run(model, tokens, PLAN)) if padded else (
        lambda: _run(model, tokens, PLAN))
    monkeypatch.setattr(language, "_MASK_BUDGET_BYTES", 4096)
    attn = model.model.layers[5].self_attn  # global layer
    base = run()
    # The prefill chunk's slice count: rows per slice at the first chunk.
    model(tokens[:, :20], cache=model.make_cache()) if not padded else None
    rows_off = attn.mask_rows_last
    model.configure_kernels(prefill_mask_bf16_budget=True)
    got = run()
    receipt = model.kernel_receipt()
    model(tokens[:, :20], cache=model.make_cache()) if not padded else None
    rows_on = attn.mask_rows_last
    model.configure_kernels()
    assert mx.array_equal(run(), base)  # off again: the reference
    if bf16:
        assert receipt["prefill_mask_bf16_budget_calls"] > 0
        if not padded:
            assert rows_on == 2 * rows_off
        gap = _gap(got, base)
        _record(f"mask_bf16_budget/rows{rows}", gap)
        assert gap["max_abs"] < 0.05 and gap["argmax_equal"] == 1.0, gap
    else:
        assert receipt["prefill_mask_bf16_budget_calls"] == 0
        assert rows_on == rows_off
        assert mx.array_equal(got, base)


# ----- inkling2_prefill_mask_with_sink (item 1b) -----


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("padded", [False, True])
def test_mask_with_sink_is_bitwise(monkeypatch, padded, bf16, fused):
    """Bit-identical to the concatenated sink column: prefill (several
    slices under a small budget) + decode (sink SDPA off, so one-token calls
    take the masked path too), sliding and global layers, B1 or B2 with left
    padding, with and without the attention fusion's memoized indices."""
    from mlx2.runtime.models.inkling2 import language

    model = _bf16(_tiny_model()) if bf16 else _tiny_model()
    tokens = mx.random.randint(0, 500, (2 if padded else 1, 36), key=mx.random.key(9))
    run = (lambda: _padded_run(model, tokens, PLAN)) if padded else (
        lambda: _run(model, tokens, PLAN))
    monkeypatch.setattr(language, "_MASK_BUDGET_BYTES", 4096)
    base_opts = {"fuse_attention": fused}
    model.configure_kernels(**base_opts)
    base = run()
    model.configure_kernels(**base_opts, prefill_mask_with_sink=True)
    got = run()
    receipt = model.kernel_receipt()
    model.configure_kernels(**base_opts)
    assert mx.array_equal(run(), base)
    assert receipt["prefill_mask_with_sink_calls"] > 48 * len(PLAN)  # sliced
    assert mx.array_equal(got, base)


def test_mask_with_sink_values_match_concatenation():
    """The mask itself, element for element, at an offset past the extent."""
    from mlx2.runtime.models.inkling.language import banded_additive_mask
    from mlx2.runtime.models.inkling2.prefill_attention import banded_mask_with_sink

    mx.random.seed(4)
    rel = mx.random.normal((2, 5, 3, 4)).astype(mx.bfloat16)
    proj = mx.random.normal((4, 8)).astype(mx.bfloat16)
    sinks = mx.random.normal((2, 3, 5, 1)).astype(mx.bfloat16)
    pad = mx.array([0, 7])
    for window in (0, 8):
        ref = banded_additive_mask(rel, proj, 12, 17, window, 8, left_padding=pad)
        ref = mx.concatenate([ref, sinks.astype(ref.dtype)], axis=-1)
        got = banded_mask_with_sink(SimpleNamespace(), rel, proj, sinks, 12, 17, window, 8,
                                    left_padding=pad)
        assert got.dtype == ref.dtype and mx.array_equal(got, ref)


# ----- inkling2_head_kernel_wide (item 2) -----


@pytest.mark.parametrize("k", [128, 256, 640, 2304, 4096 + 128])
def test_head_wide_loads_keep_every_lanes_groups_and_order(k):
    """The wide kernel's load-and-shuffle schedule hands each lane exactly the
    groups the narrow kernel's ``for k = lane; k < K/4; k += 32`` loop sums,
    in the same order (so every simd_sum input is the same value)."""
    from mlx2.runtime.models.inkling2.language import head_wide_lane_groups

    for lane in range(32):
        assert head_wide_lane_groups(k, lane) == list(range(lane, k // 4, 32))


def test_head_kernel_wide_plumbing_and_cpu_reference():
    from mlx2.adapters.inkling2 import validate_policy

    # head_kernel is default on (Pierre 2026-11-25), so wide alone is valid;
    # wide with head_kernel turned off still fails closed.
    assert validate_policy({"inkling2_head_kernel_wide": True})[0]["head_kernel_wide"] is True
    with pytest.raises(ValueError, match="requires inkling2_head_kernel"):
        validate_policy({"inkling2_head_kernel_wide": True, "inkling2_head_kernel": False})
    model = _bf16(_tiny_model())
    with pytest.raises(ValueError, match="head_kernel_wide requires head_kernel"):
        model.configure_kernels(head_kernel_wide=True)
    tokens = mx.random.randint(0, 500, (2, 10), key=mx.random.key(10))
    base = _run(model, tokens, [(0, 6)] + [(i, i + 1) for i in range(6, 10)])
    model.configure_kernels(head_kernel=True, head_kernel_wide=True)
    assert model.kernel_policy["head_kernel_wide"] is True
    got = _run(model, tokens, [(0, 6)] + [(i, i + 1) for i in range(6, 10)])
    receipt = model.kernel_receipt()
    # GPU only: on the CPU the reference FP32 head runs, bitwise.
    assert receipt["head_kernel_calls"] == receipt["head_kernel_wide_calls"] == 0
    assert mx.array_equal(got, base)


# ----- inkling2_prefill_tiled_attention (item 1c) -----


def _attention_case(seed, *, b=2, hq=6, hkv=2, length=37, size=90, dim=32, extent=16,
                    window=0, pads=(0, 11), dtype=mx.bfloat16):
    mx.random.seed(seed)

    def unit(shape):
        x = mx.random.normal(shape)
        return x / mx.linalg.norm(x, axis=-1, keepdims=True) * 4.0

    q = unit((b, hq, length, dim)).astype(dtype)
    k = unit((b, hkv, size, dim)).astype(dtype)
    v = mx.random.normal((b, hkv, size, dim)).astype(dtype)
    rel = (mx.random.normal((b, length, hq, extent)) * 0.5).astype(dtype)
    sinks = mx.random.normal((b, length, hq)).astype(dtype)
    pad = None if pads is None else mx.array(list(pads))
    kw = dict(offset=size - length, left_padding=pad, window=window, scale=dim ** -0.5)
    return (q, k, v, rel, sinks), kw


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("case", [
    dict(), dict(window=8), dict(pads=None, b=1), dict(hq=4, hkv=4), dict(hq=8, hkv=1),
    dict(length=2, size=2, pads=None, b=1), dict(length=64, size=64, extent=1024),
    dict(length=40, size=700, extent=64),
])
def test_tiled_kernel_law_matches_the_composed_reference(case, dtype):
    """The kernel's algorithm (tiled_reference: key tiles, online softmax
    from the sink, FP16 staging for BF16) against the composed law (dense
    mask + zero sink column + SDPA). Records the gap."""
    from mlx2.runtime.models.inkling2 import splitk
    from mlx2.runtime.models.inkling2.prefill_attention import tiled_reference

    args, kw = _attention_case(1, dtype=dtype, **case)
    ref = splitk.reference_attention(*args, **kw).astype(mx.float32)
    got = tiled_reference(*args, **kw).astype(mx.float32)
    gap = float(mx.abs(got - ref).max().item())
    _record(f"tiled_law/{dtype}/{sorted(case.items())}", {"max_abs": gap})
    assert gap < (2e-5 if dtype == mx.float32 else 3e-2), gap


def _engage_tiled(monkeypatch, law):
    from mlx2.runtime.models.inkling2 import prefill_attention

    seen = []

    def fake(q, k, v, rel, sinks, **kw):
        seen.append((q.shape[2], kw["window"], kw["left_padding"] is not None, kw["offset"]))
        return law(q, k, v, rel, sinks, **kw)

    monkeypatch.setattr(prefill_attention, "decline_reason", lambda *a, **kw: None)
    monkeypatch.setattr(prefill_attention, "tiled_prefill_attention", fake)
    return seen


@pytest.mark.parametrize("padded", [False, True])
def test_tiled_wiring_reproduces_the_composed_path_bitwise(monkeypatch, padded):
    """Engaged on the CPU with the composed law in place of the kernel: the
    arguments (log-tau-scaled q and bias rows, sinks, offset, padding) give
    the reference bitwise, so only the kernel's arithmetic differs."""
    from mlx2.runtime.models.inkling2 import splitk

    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (2 if padded else 1, 36), key=mx.random.key(11))
    run = (lambda: _padded_run(model, tokens, PLAN)) if padded else (
        lambda: _run(model, tokens, PLAN))
    base = run()
    seen = _engage_tiled(monkeypatch, splitk.reference_attention)
    model.configure_kernels(prefill_tiled_attention=True)
    got = run()
    receipt = model.kernel_receipt()
    model.configure_kernels()
    multi = sum(1 for a, b in PLAN if b - a > 1)
    assert receipt["prefill_tiled_calls"] == 8 * multi  # global visits only
    assert receipt["prefill_tiled_declines"] == {}
    assert {w for _, w, _, _ in seen} == {0}
    assert all(padded == p for _, _, p, _ in seen)
    assert mx.array_equal(got, base)


def _fp32_exact_law(q, k, v, rel, sinks, **kw):
    """The composed law evaluated in FP32 and rounded once: a more accurate
    attention than the BF16 reference, used as the chaos control."""
    from mlx2.runtime.models.inkling2 import splitk

    f = [a.astype(mx.float32) for a in (q, k, v, rel, sinks)]
    return splitk.reference_attention(*f, **kw).astype(q.dtype)


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("padded", [False, True])
def test_tiled_emulated_kernel_is_near_reference(monkeypatch, padded, bf16):
    """The kernel's algorithm engaged in the model (prefill chunks + decode,
    B1 or B2 with left padding) against the reference; off again is the
    reference bitwise. FP32: within 1e-4, argmax equal. BF16: the random
    tiny decoder is chaotic over 48 visits (a more accurate attention, the
    FP32 law rounded once, already moves logits by ~0.8 and flips argmaxes),
    so the gap is bounded relative to that control."""
    from mlx2.runtime.models.inkling2.prefill_attention import tiled_reference

    model = _bf16(_tiny_model()) if bf16 else _tiny_model()
    tokens = mx.random.randint(0, 500, (2 if padded else 1, 36), key=mx.random.key(12))
    run = (lambda: _padded_run(model, tokens, PLAN)) if padded else (
        lambda: _run(model, tokens, PLAN))
    base = run()

    def engaged(law):
        _engage_tiled(monkeypatch, law)
        model.configure_kernels(prefill_tiled_attention=True)
        out = run()
        calls = model.kernel_receipt()["prefill_tiled_calls"]
        model.configure_kernels()
        return out, calls

    got, calls = engaged(tiled_reference)
    assert mx.array_equal(run(), base)
    assert calls > 0
    gap = _gap(got, base)
    if not bf16:
        _record(f"tiled_model/fp32/rows{2 if padded else 1}", gap)
        assert gap["max_abs"] < 1e-4 and gap["argmax_equal"] == 1.0, gap
        return
    control = _gap(engaged(_fp32_exact_law)[0], base)
    _record(f"tiled_model/bf16/rows{2 if padded else 1}", {**gap, "control": control})
    assert gap["max_abs"] <= 3 * control["max_abs"] + 0.05, (gap, control)
    assert gap["argmax_equal"] >= control["argmax_equal"] - 0.15, (gap, control)


def test_tiled_declines_on_cpu_and_is_the_reference():
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 36), key=mx.random.key(13))
    base = _run(model, tokens, PLAN)
    model.configure_kernels(prefill_tiled_attention=True)
    got = _run(model, tokens, PLAN)
    receipt = model.kernel_receipt()
    assert receipt["prefill_tiled_calls"] == 0
    assert receipt["prefill_tiled_declines"] == {"declined:cpu": 8 * 2}
    assert mx.array_equal(got, base)


def test_tiled_and_flash_prefill_are_alternatives():
    from mlx2.adapters.inkling2 import validate_policy

    with pytest.raises(ValueError, match="alternatives"):
        validate_policy({"inkling2_prefill_tiled_attention": True,
                         "inkling2_flash_prefill": True})
    with pytest.raises(ValueError, match="alternatives"):
        _tiny_model().configure_kernels(prefill_tiled_attention=True, flash_prefill=True)


def test_tiled_decline_reasons():
    from mlx2.runtime.models.inkling2.prefill_attention import decline_reason

    (q, k, v, _, _), _ = _attention_case(2)
    kw = dict(n_heads=6, n_kv=2, head_dim=32)
    assert decline_reason(q, k, v, **kw) == "cpu"
    assert decline_reason(q, k, v, device=mx.gpu, **kw) is None
    assert decline_reason(q[:, :, :1], k, v, device=mx.gpu, **kw) == "rows"
    assert decline_reason(q.astype(mx.float16), k, v, device=mx.gpu, **kw) == "dtype"
    assert decline_reason(q, (k, k), v, device=mx.gpu, **kw) == "cache_format"
    assert decline_reason(q, k, v, device=mx.gpu, n_heads=6, n_kv=4, head_dim=32) == "geometry"
    assert decline_reason(q, k, v, device=mx.gpu, n_heads=6, n_kv=2, head_dim=256) == "geometry"


# ----- the GPU A/B script's prefill comparison (scripts/research/inkling2_fuse_ab.py) -----


def _ab_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/research/inkling2_fuse_ab.py"
    spec = importlib.util.spec_from_file_location("inkling2_fuse_ab", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ab_script_prefill_rows_and_first_divergence(monkeypatch):
    """Per-row top-5 of every prefill row, argmax agreement and the first
    differing row (with margins) between arms; option names with the policy
    prefix; engagement counter deltas."""
    from mlx2.runtime.models.inkling2.prefill_attention import tiled_reference

    ab = _ab_script()
    assert ab.option_name("inkling2_prefill_tiled_attention") == "prefill_tiled_attention"
    assert ab.option_name("head_kernel_wide") == "head_kernel_wide"
    with pytest.raises(SystemExit):
        ab.option_name("no_such_option")
    monkeypatch.setattr(ab, "CHUNK", 16)
    model = _bf16(_tiny_model())
    model.configure_kernels()
    toggle = ab.Toggle(model, ["prefill_tiled_attention"])
    tokens = mx.random.randint(0, 500, (40,), key=mx.random.key(14)).tolist()
    _engage_tiled(monkeypatch, tiled_reference)
    c0 = ab.counters(model)
    _, last_a, _, top_a = ab.prefill(model, toggle, tokens, False)
    c1 = ab.counters(model)
    _, last_b, _, top_b = ab.prefill(model, toggle, tokens, True)
    c2 = ab.counters(model)
    assert top_a[0].shape == (40, 5) and top_a[1].shape == (40, 5)
    assert bool(mx.all(top_a[1][:, :-1] >= top_a[1][:, 1:]).item())
    agreement, first = ab.prefill_divergence(top_a, top_b)
    same, _ = ab.prefill_divergence(top_a, top_a)
    assert same == "40/40"
    assert agreement.endswith("/40")
    if first is not None:
        assert first["off_top5"][0][0] != first["on_top5"][0][0]
        assert first["off_margin"] >= 0 and first["on_margin"] >= 0
    assert ab.counter_delta(c0, c1) == {}
    assert ab.counter_delta(c1, c2)["prefill_tiled_calls"] == 8 * 3
