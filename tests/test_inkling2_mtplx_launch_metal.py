"""MTPLX launch-count ingest on Metal (GPU; run under the GPU lease only).

Every launch here is meant to be bitwise to the composed path it replaces,
by construction (MLX's own product code compiled from the installed headers,
MLX's tiles for the composed shape, the composed path's rounding points).
These tests check that claim directly, outside the first-use proof: the
launch's raw outputs against the composed path's, bit for bit, at the
Mini's real shapes (hidden 2304, 24/8 heads x 128, o_proj 3072 -> 2304),
for B1 and B2, one-token decode and 1..8-row verify blocks, BF16 and FP32
(FP32 rows >= 2 run MLX's steel GEMM, which is not twinned: the tests check
those decline). The pass-message read is a choice of MLX ops whose
equality depends on MLX's kernel choice, so its test records the verdict and
checks that what ``rows()`` returns is always the per-visit read. A model-
level check runs the wide random decoder with every switch on against off.

Set ``INKLING2_MTPLX_LAUNCH_METAL_RECEIPT`` to a path for a JSON summary
(verdicts and engagement counters per case).
"""

import json
import os

import mlx.core as mx
import pytest
from mlx import nn
from mlx.utils import tree_flatten

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")

RESULTS = {}
MINI = dict(hidden_size=2304, num_attention_heads=24, num_key_value_heads=8, head_dim=128,
            swa_num_attention_heads=24, swa_num_key_value_heads=8, swa_head_dim=128,
            sliding_window_size=1024, rel_extent=1024, d_rel=16, intermediate_size=64,
            loop_config=dict(core_start=6, core_end=18, num_passes=3, gate_rank=64))
# (B, L): one-token decode B1/B2 and verify blocks of 1..8 rows (B * L <= 8).
SHAPES = [(1, 1), (2, 1)] + [(1, n) for n in range(2, 9)] + [(2, 2), (2, 4)]
DTYPES = [mx.bfloat16, mx.float32]


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
    path = os.environ.get("INKLING2_MTPLX_LAUNCH_METAL_RECEIPT")
    if path:
        with open(path, "w") as handle:
            json.dump(RESULTS, handle, indent=1, sort_keys=True, default=str)


def _config(**extra):
    from mlx2.runtime.models.inkling2 import TextConfig
    from test_inkling2_cpu import TINY

    config = dict(TINY)
    config.update(MINI)
    config.update(extra)
    return TextConfig.from_dict(config)


def _randomize(module, dtype, seed):
    mx.random.seed(seed)
    weights = []
    for key, value in tree_flatten(module.parameters()):
        noise = mx.random.normal(value.shape) * (0.02 if value.ndim > 1 else 0.3)
        weights.append((key, noise.astype(dtype)))
    module.load_weights(weights, strict=True)
    mx.eval(module.parameters())
    return module


_ATTN = {}


def _attention(layer_idx, dtype):
    from mlx2.runtime.models.inkling2.language import Attention

    key = (layer_idx, dtype)
    if key not in _ATTN:
        _ATTN[key] = _randomize(Attention(_config(), layer_idx), dtype, seed=layer_idx)
    return _ATTN[key]


def _expected_decline(dtype, rows):
    """The reason MLX's steel GEMM (not twinned) is expected here, or None."""
    from mlx2.runtime.models.inkling2 import library_kernels as lib

    if rows < 2:
        return None
    if dtype == mx.float32:
        return "rows_dtype"
    return "gpu_generation" if lib.gpu_generation() < 15 else None


def _same(a, b):
    from mlx2.runtime.models.inkling2 import library_kernels as lib

    mx.eval(a, b)
    return a.shape == b.shape and a.dtype == b.dtype and bool(
        mx.array_equal(lib.bits(a), lib.bits(b)).item())


# ----- stage A: the six input projections ---------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("b,length", SHAPES)
@pytest.mark.parametrize("layer_idx", [0, 5])  # sliding, global
@pytest.mark.parametrize("inputs", ["shared", "distinct"])  # first pass, loop pass
@pytest.mark.parametrize("stacked", [False, True])  # inkling2_fused_projections
def test_stage_a_raw_launch_is_bitwise(dtype, b, length, layer_idx, inputs, stacked):
    from mlx2.runtime.models.inkling2 import stage_a

    attn = _attention(layer_idx, dtype)
    if stacked:
        attn.fuse_projections()
    else:
        object.__setattr__(attn, "_fused_bounds", None)
    mx.random.seed(100 + b * 10 + length)
    x = mx.random.normal((b, length, 2304)).astype(dtype)
    kx = vx = x
    if inputs == "distinct":
        kx = mx.random.normal((b, length, 2304)).astype(dtype)
        vx = mx.random.normal((b, length, 2304)).astype(dtype)
    planned = stage_a.plan(attn, x, kx, vx)
    case = f"stage_a/{dtype}/b{b}l{length}/layer{layer_idx}/{inputs}/stacked{int(stacked)}"
    expected = _expected_decline(dtype, b * length)
    if expected:
        assert planned == expected  # MLX's steel GEMM here: declined
        RESULTS[case] = "declined:" + expected
        return
    assert not isinstance(planned, str), planned
    fused, ref = planned.fused(), planned.reference()
    same = [_same(f, r) for f, r in zip(fused, ref)]
    RESULTS[case] = same
    assert all(same), dict(zip(stage_a._NAMES, same))


def test_stage_a_project_proves_then_engages():
    from mlx2.runtime.models.inkling2 import stage_a

    attn = _attention(5, mx.bfloat16)
    object.__setattr__(attn, "_fused_bounds", None)
    stage_a.PROOFS.reset()
    attn.__dict__.pop("stage_a_counts", None)
    x = mx.random.normal((1, 1, 2304)).astype(mx.bfloat16)
    first = stage_a.project(attn, x, x, x)
    second = stage_a.project(attn, x, x, x)
    ref = attn._project_inputs_composed(x, x, x)
    assert all(_same(a, r) and _same(c, r) for a, c, r in zip(first, second, ref))
    assert attn.stage_a_counts == {"proved": 1, "calls": 1}, attn.stage_a_counts


# ----- gate in the split-K merge -------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("b,length", SHAPES)
@pytest.mark.parametrize("keys", [300, 4096])  # part kernel splits / direct kernel at L=1
@pytest.mark.parametrize("fuse_attention", [True, False])
def test_gate_merge_is_bitwise(dtype, b, length, keys, fuse_attention):
    from mlx2.runtime.models.inkling2 import fuse as _fuse

    attn = _attention(5, dtype)
    object.__setattr__(attn, "_fuse", frozenset({"fuse_attention"}) if fuse_attention
                       else frozenset())
    try:
        _gate_merge_case(attn, dtype, b, length, keys, fuse_attention)
    finally:
        object.__setattr__(attn, "_fuse", _fuse.NONE)


def _gate_merge_case(attn, dtype, b, length, keys, fuse_attention):
    from mlx2.runtime.models.inkling2 import splitk

    hq, hkv, d = 24, 8, 128
    mx.random.seed(keys + 7 * length + b)
    q = (mx.random.normal((b, hq, length, d)) * 0.1).astype(dtype)
    k = (mx.random.normal((b, hkv, keys, d)) * 0.1).astype(dtype)
    v = mx.random.normal((b, hkv, keys, d)).astype(dtype)
    rel = mx.random.normal((b, length, hq, 1024)).astype(dtype)
    sinks = mx.random.normal((b, length, hq)).astype(dtype)
    gate = (mx.random.normal((b, length, hq * d)) * 3).astype(dtype)
    parts = splitk.partials(q, k, v, rel, sinks, offset=keys - length, left_padding=None,
                            window=0, scale=d ** -0.5)
    case = f"gate_merge/{dtype}/b{b}l{length}/keys{keys}/fa{int(fuse_attention)}"
    if parts[0] == "out":
        RESULTS[case] = "single_split"
        pytest.skip("one split: the partials kernel writes the output (gate not fused)")
    fused = splitk.merge_gated(q, *parts[1:], gate)
    ref = attn._gate(splitk.merge(q, *parts[1:]), gate, b, length, dtype)
    RESULTS[case] = _same(fused, ref)
    assert RESULTS[case]


# ----- o_proj + attention short conv + residual -----------------------------


class _Layer(nn.Module):
    def __init__(self, boundary, dtype):
        from mlx2.runtime.models.inkling2.language import ShortConvolution

        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.o_proj = nn.Linear(3072, 2304, bias=False)
        self.attn_sconv = ShortConvolution(2304, 4, 2)
        self.is_block_boundary = boundary
        _randomize(self, dtype, seed=int(boundary))
        self.attn_sconv.conv.weight = self.attn_sconv.conv.weight.astype(dtype)
        object.__setattr__(self.attn_sconv, "_fuse",
                           frozenset({"fuse_sconv", "fuse_sconv_kernel"}))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("b,length", SHAPES)
@pytest.mark.parametrize("boundary", [False, True])
@pytest.mark.parametrize("has_state", [True, False])
def test_attn_output_raw_launch_is_bitwise(dtype, b, length, boundary, has_state):
    from mlx2.runtime.models.inkling2 import attn_output

    layer = _Layer(boundary, dtype)
    mx.random.seed(b * 31 + length)
    gated = mx.random.normal((b, length, 3072)).astype(dtype)
    x = mx.random.normal((b, length, 2304)).astype(dtype)
    state = mx.random.normal((b, 3, 2304)) if has_state else None
    conv = [None, None, state, None]
    planned = attn_output.plan_residual(layer, gated, conv, None, x)
    case = f"attn_output/{dtype}/b{b}l{length}/boundary{int(boundary)}/state{int(has_state)}"
    expected = _expected_decline(dtype, b * length)
    if expected:
        assert planned == expected
        RESULTS[case] = "declined:" + expected
        return
    assert not isinstance(planned, str), planned
    out_y, res, nstate = planned.fused()
    ref = planned.reference()  # writes conv[2]
    same = [_same(res, ref), _same(nstate, conv[2]), _same(out_y, planned.held["attn_out"])]
    RESULTS[case] = same
    assert all(same), same


# ----- embedding + norm -----------------------------------------------------


class _Embed(nn.Module):
    def __init__(self, dtype, width=2304):
        super().__init__()
        self.embed_tokens = nn.Embedding(4096, width)
        self.embed_norm = nn.RMSNorm(width, eps=1e-6)
        _randomize(self, dtype, seed=3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("b,length", SHAPES + [(1, 512)])
def test_embed_raw_launch_is_bitwise(dtype, b, length):
    from mlx2.runtime.models.inkling2 import frame_kernels

    model = _Embed(dtype)
    ids = mx.random.randint(0, 4096, (b, length)).astype(mx.int32)
    planned = frame_kernels.plan(model, ids)
    assert not isinstance(planned, str), planned
    case = f"embed/{dtype}/b{b}l{length}"
    RESULTS[case] = _same(planned.fused(), planned.reference())
    assert RESULTS[case]


# ----- pass-wide message reads ----------------------------------------------


@pytest.mark.parametrize("b,length", SHAPES)
@pytest.mark.parametrize("states", [2, 3])
@pytest.mark.parametrize("table_dtype", [mx.bfloat16, mx.float32])
def test_pass_messages_verdict_and_output(b, length, states, table_dtype):
    from mlx2.runtime.models.inkling2 import pass_messages

    pass_messages.PROOFS.reset()
    owner = nn.Module()
    mx.random.seed(states * 13 + length)
    bank = mx.fast.rms_norm(mx.random.normal((b, length, states, 2304)), None, 1e-6)
    table = (mx.random.normal((12, 4, 2304)) * 0.02).astype(table_dtype)
    per_visit = [pass_messages.per_visit(bank, table[i], 2304) for i in range(12)]
    batched = pass_messages.batched(bank, table, 2304)
    raw = all(_same(a, r) for a, r in zip(batched, per_visit))
    first = pass_messages.rows(owner, bank, table, 2304)
    second = pass_messages.rows(owner, bank, table, 2304)
    assert all(_same(a, r) for a, r in zip(first, per_visit))
    assert all(_same(a, r) for a, r in zip(second, per_visit))
    counts = owner.pass_messages_counts
    assert counts == ({"proved": 1, "calls": 1} if raw else {"refused_first": 1, "refused": 1})
    RESULTS[f"pass_messages/b{b}l{length}/states{states}/{table_dtype}"] = raw


# ----- model level: every switch on against off -------------------------


NEW = ("pass_messages", "stage_a_kernel", "gate_merge", "attn_output_kernel", "embed_kernel")
PLAN = [(0, 40), (40, 41), (41, 42)] + [(42 + sum(range(n)), 42 + sum(range(n + 1)))
                                        for n in range(1, 9)]


def _wide_model(dtype):
    from mlx2.runtime.models.inkling2 import Model, TextConfig
    from test_inkling2_cpu import TINY

    config = dict(TINY)
    config.update(hidden_size=512, head_dim=128, swa_head_dim=128, d_rel=16,
                  intermediate_size=128,
                  loop_config=dict(core_start=6, core_end=18, num_passes=3, gate_rank=64))
    mx.random.seed(0)
    model = Model(TextConfig.from_dict(config))
    weights = []
    for key, value in tree_flatten(model.parameters()):
        noise = mx.random.normal(value.shape) * (0.05 if value.ndim > 1 else 0.3)
        value = noise + (1.0 if key.endswith("norm.weight") else 0.0)
        weights.append((key, value.astype(dtype)))
    model.load_weights(weights, strict=True)
    mx.eval(model.parameters())
    return model


def _defaults(model):
    import inspect

    from mlx2.adapters.inkling2 import KERNEL_DEFAULTS

    params = inspect.signature(model.configure_kernels).parameters
    return {k: (True if v is None else v) for k, v in KERNEL_DEFAULTS.items()
            if k in params and not k.startswith("moe_")}


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows", [1, 2])
def test_model_all_switches_on_bitwise_and_engaged(dtype, rows):
    mx.random.seed(5)
    tokens = mx.random.randint(0, 500, (rows, PLAN[-1][1]))
    outs = {}
    receipt = None
    for on in (False, True):
        model = _wide_model(dtype)
        model.configure_kernels(**{**_defaults(model), **{n: on for n in NEW}})
        cache = model.make_cache()
        steps = []
        for a, b in PLAN:
            steps.append(model(tokens[:, a:b], cache=cache))
            mx.eval(steps[-1])
        outs[on] = mx.concatenate(steps, axis=1)
        if on:
            receipt = model.kernel_receipt()
    case = f"model/{dtype}/b{rows}"
    RESULTS[case] = {"bitwise": _same(outs[True], outs[False]),
                     **{k: receipt[k] for k in ("pass_messages", "stage_a", "gate_merge",
                                                "attn_output", "embed_kernel")}}
    assert RESULTS[case]["bitwise"]
    # Engagement: every family proved at least one shape and then ran it,
    # except the products at FP32 B2 (every call has >= 2 rows: steel GEMM).
    for name in ("stage_a", "attn_output", "embed_kernel"):
        counts = receipt[name]
        expected = _expected_decline(dtype, rows)
        if expected and name != "embed_kernel":
            assert counts.get("declined:" + expected, 0) > 0 and not counts.get("calls"), counts
            continue
        assert counts.get("proved", 0) > 0 and counts.get("calls", 0) > 0, (name, counts)
    assert receipt["gate_merge"].get("proved", 0) + receipt["gate_merge"].get(
        "declined:single_split", 0) > 0, receipt["gate_merge"]


# ----- the 2026-11-21 BF16 gap of the near-exact kernels, localised ---------


@pytest.mark.parametrize("kernel", ["fuse_sconv_kernel", "fuse_residual_kernel", "both"])
def test_bf16_gap_localisation(kernel):
    """Diagnostic (records; asserts only that it ran). The 2026-11-21 tiny-
    model run had argmax agreement 0.775 with the sconv and residual kernels
    both on in BF16. Here each kernel alone against the same base, B1
    decode, next to the chunking-only BF16 gap of the same model (the
    yardstick of how far BF16 rounding alone moves this random model)."""
    from mlx2.runtime.models.inkling2 import Model, TextConfig
    from test_inkling2_cpu import TINY

    def build():
        mx.random.seed(0)
        model = Model(TextConfig.from_dict(dict(TINY)))
        weights = []
        for key, value in tree_flatten(model.parameters()):
            noise = mx.random.normal(value.shape) * (0.05 if value.ndim > 1 else 0.3)
            value = noise + (1.0 if key.endswith("norm.weight") else 0.0)
            weights.append((key, value.astype(mx.bfloat16)))
        model.load_weights(weights, strict=True)
        return model

    base = {"fuse_controllers": True, "fuse_residual": True, "fuse_sconv": True,
            "fuse_attention": True}
    kernels = ({"fuse_sconv_kernel": True, "fuse_residual_kernel": True} if kernel == "both"
               else {kernel: True})
    arms = {"base": {}, "chunked": {}, kernel: kernels}
    mx.random.seed(9)
    tokens = mx.random.randint(0, 500, (1, 60))
    logits = {}
    for name, extra in arms.items():
        model = build()
        model.configure_kernels(**base, **extra)
        cache = model.make_cache()
        plan = [(0, 9), (9, 20)] if name == "chunked" else [(0, 20)]
        steps = [model(tokens[:, a:b], cache=cache)[:, -1:] for a, b in plan][-1:]
        for i in range(20, 60):
            steps.append(model(tokens[:, i:i + 1], cache=cache))
            mx.eval(steps[-1])
        logits[name] = mx.concatenate(steps, axis=1).astype(mx.float32)
    summary = {}
    for name in ("chunked", kernel):
        diff = mx.abs(logits[name] - logits["base"])
        agree = mx.mean(mx.argmax(logits[name], -1) == mx.argmax(logits["base"], -1))
        summary[name] = {"bitwise": _same(logits[name], logits["base"]),
                         "max_abs": float(mx.max(diff).item()),
                         "argmax_agreement": float(agree.item())}
    RESULTS[f"bf16_gap/{kernel}"] = summary
