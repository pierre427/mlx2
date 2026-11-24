"""CPU checks for the opt-in device-routed Ling MoE decode wrapper.

The Metal kernels cannot run here; their bit equality with ``gather_qmv`` is a
GPU claim (``ling_routed_decode.self_check`` under the GPU lease). These tests
pin the wrapper's glue (route order, dtypes, activation handoff, shapes) with
CPU stand-ins for the two kernels, and every decline reason.
"""
import mlx.core as mx
import pytest
from mlx import nn

from mlx2.runtime.models import deepseek_v41_routed_decode as v41
from mlx2.runtime.models import ling_routed_decode as routed
from mlx2.runtime.models.switch_layers import SwiGLU, SwitchGLU

pytestmark = pytest.mark.usefixtures("cpu_device")

HIDDEN, INTER, EXPERTS, TOP = 512, 192, 16, 8


@pytest.fixture
def cpu_device():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


class PostClamp(nn.Module):
    """Ling's post-activation law ``min(silu(g), L) * clip(u, -L, L)``."""

    def __init__(self, limit):
        super().__init__()
        self.limit = limit
        self.calls = 0

    def __call__(self, x, gate):
        self.calls += 1
        return mx.minimum(nn.silu(gate), self.limit) * mx.clip(x, -self.limit, self.limit)


def _bank(activation=None, hidden=HIDDEN, inter=INTER, bits=4):
    mx.random.seed(7)
    glu = SwitchGLU(hidden, inter, EXPERTS, activation=activation if activation is not None else SwiGLU())
    nn.quantize(glu, group_size=64, bits=bits)
    glu.set_dtype(mx.bfloat16)
    mx.eval(glu.parameters())
    return glu


def _route(rows=1, seed=3):
    mx.random.seed(seed)
    x = (mx.random.normal((1, rows, HIDDEN)) * 2).astype(mx.bfloat16)
    scores = mx.random.normal((1, rows, EXPERTS))
    ids = mx.argpartition(-scores, kth=TOP - 1, axis=-1)[..., :TOP].astype(mx.uint32)
    return x, ids


def _cpu_gate_up(x, ids, gate, up, spec):
    """Per-slot ``quantized_matmul`` stand-in for the gate_up kernel.

    It runs in the activation dtype like CPU ``gather_qmm`` and returns F32
    holding those values, so the wrapper's cast back is exact and the glue
    can be compared bitwise on CPU.
    """
    rows = []
    for which in (gate, up):
        out = []
        for e in ids.tolist():
            w, s, b = (a[e] for a in which)
            out.append(mx.quantized_matmul(x[None], w, s, b, transpose=True,
                                           group_size=spec["group_size"], bits=spec["bits"])[0])
        rows.append(mx.stack(out).astype(mx.float32))
    return tuple(rows)


def _cpu_down(h, ids, weights, spec):
    out = []
    for slot, e in enumerate(ids.tolist()):
        w, s, b = (a[e] for a in weights)
        out.append(mx.quantized_matmul(h[slot:slot + 1], w, s, b, transpose=True,
                                       group_size=spec["group_size"], bits=spec["bits"])[0])
    return mx.stack(out).astype(mx.float32)


@pytest.fixture
def cpu_kernels(monkeypatch):
    monkeypatch.setattr(routed, "_metal_ready", lambda: True)
    monkeypatch.setattr(v41, "gate_up", _cpu_gate_up)
    monkeypatch.setattr(v41, "down", _cpu_down)
    routed.status(reset=True)


def test_ling_shapes_pass_the_v41_kernel_gate():
    # Ling-3.2-air: hidden 2560 (qmv_fast, 5 x 512), moe_intermediate 768
    # (down takes the tail-safe qmv), top-8, 4-bit g64 affine banks.
    specs = {k: {"bits": 4, "group_size": 64} for k in ("gate", "up", "down")}
    assert v41.supports(2560, 768, 8, specs)
    assert not v41.supports(2560, 1024, 8, specs)  # down K % 512 == 0 is not the qmv branch


@pytest.mark.parametrize("activation", [None, "clamped"])
def test_glue_matches_switch_glu(cpu_kernels, activation):
    act = PostClamp(0.5) if activation else None
    glu = _bank(act)
    x, ids = _route()
    out = routed.switch_glu(glu, x, ids)
    assert out is not None and routed.status()["engaged"] == 1
    reference = glu(x, ids)
    assert out.shape == reference.shape == (1, 1, TOP, HIDDEN)
    assert out.dtype == reference.dtype == mx.bfloat16
    # Glue only (route order, slot alignment, activation handoff, casts): the
    # Metal kernels' equality with gather_qmv is checked on the GPU.
    assert mx.array_equal(out.view(mx.uint16), reference.view(mx.uint16)).item()
    if act is not None:
        assert act.calls == 2  # once on the routed path, once in the reference


def test_clamp_law_is_the_banks_own_activation(cpu_kernels):
    clamped, plain = _bank(PostClamp(0.05)), _bank()
    x, ids = _route()
    a, b = routed.switch_glu(clamped, x, ids), routed.switch_glu(plain, x, ids)
    assert not mx.array_equal(a, b).item()  # the post-activation clamp is in effect


def test_slot_order_follows_route_order(cpu_kernels):
    glu = _bank()
    x, ids = _route()
    out = routed.switch_glu(glu, x, ids)
    flipped = routed.switch_glu(glu, x, ids[..., ::-1])
    assert mx.array_equal(out[..., ::-1, :], flipped).item()


def test_declines(monkeypatch, cpu_kernels):
    glu = _bank()
    x, ids = _route(rows=2)
    assert routed.switch_glu(glu, x, ids) is None
    x, ids = _route()
    monkeypatch.setattr(routed, "_metal_ready", lambda: False)
    assert routed.switch_glu(glu, x, ids) is None
    monkeypatch.setattr(routed, "_metal_ready", lambda: True)
    small = SwitchGLU(128, 64, EXPERTS)
    nn.quantize(small, group_size=64, bits=4)
    small.set_dtype(mx.bfloat16)
    assert routed.refusal(small, mx.zeros((1, 1, 128), mx.bfloat16), ids) == "kernel_shape"
    dense = SwitchGLU(HIDDEN, INTER, EXPERTS)
    assert routed.refusal(dense, x, ids) == "not_affine_bank"
    biased = SwitchGLU(HIDDEN, INTER, EXPERTS, bias=True)
    nn.quantize(biased, group_size=64, bits=4)
    assert routed.refusal(biased, x, ids) == "not_affine_bank"
    counts = routed.status()
    assert counts["decline:not_one_row"] == 1 and counts["decline:no_metal"] == 1
    assert "engaged" not in counts


def test_declines_every_case_that_would_change_the_reference(monkeypatch, cpu_kernels):
    from mlx2.runtime.models import invariant_prefill, switch_grouped_experts

    glu = _bank()
    x, ids = _route()
    assert routed.refusal(glu, x, ids) is None
    # gather_qmm promotes to (x, scales): unconverted f32 metadata, or f16 on
    # down only, would make SwitchGLU return another dtype.
    unconverted = SwitchGLU(HIDDEN, INTER, EXPERTS)
    nn.quantize(unconverted, group_size=64, bits=4)
    assert routed.refusal(unconverted, x, ids) == "scale_dtype"
    mixed = _bank()
    mixed.down_proj.scales = mixed.down_proj.scales.astype(mx.float16)
    assert routed.refusal(mixed, x, ids) == "scale_dtype"
    # top >= 20 sorts under auto, and a sorted reference may take gather_qmm_rhs.
    wide = mx.arange(20, dtype=mx.uint32).reshape(1, 1, 20) % EXPERTS
    assert routed.refusal(glu, x, wide) == "sorted_reference"
    glu.set_sort_policy("unsorted_decode")
    assert routed.refusal(glu, x, wide) != "sorted_reference"
    glu.set_sort_policy("auto")
    assert routed.refusal(glu, x, mx.concatenate([ids, ids], axis=0)) == "not_one_row"
    fused = nn.Module()
    fused.activation = SwiGLU()
    assert routed.refusal(fused, x, ids) == "not_split_bank"
    assert routed.refusal(glu, x.astype(mx.int32), ids) == "activation_dtype"
    assert routed.refusal(glu, x, ids.astype(mx.float32)) == "index_dtype"
    monkeypatch.setattr(switch_grouped_experts, "MODE", "q4")
    assert routed.refusal(glu, x, ids) == "grouped_experts_active"
    monkeypatch.setattr(switch_grouped_experts, "MODE", "off")
    monkeypatch.setattr(invariant_prefill, "active", lambda: True)
    assert routed.refusal(glu, x, ids) == "invariant_lane"


def test_self_check_leaves_counters_alone(cpu_kernels):
    glu = _bank()
    x, ids = _route()
    routed.status(reset=True)
    assert routed.self_check(glu, x, ids) == {"engaged": True, "exact": True, "max_abs": 0.0}
    assert routed.status() == {}


def test_default_cpu_device_declines_without_stand_ins():
    glu = _bank()
    x, ids = _route()
    routed.status(reset=True)
    assert routed.switch_glu(glu, x, ids) is None
    assert routed.status() == {"decline:no_metal": 1}


def test_implementation_identity_is_stable():
    assert routed.implementation_identity() == routed.implementation_identity()
    assert len(routed.implementation_identity()) == 64


_ARRAY_SYNCS = ("item", "tolist", "__bool__", "__int__", "__float__", "__index__", "__array__")


def _host_sync_calls(monkeypatch, fn):
    """Host transfers made while ``fn`` builds its graph.

    Wraps every mx.array host conversion (explicit and implicit: ``bool()``,
    ``int()``, ``float()``, indexing, numpy ``__array__``), ``mx.eval`` /
    ``mx.async_eval`` and ``np.asarray`` / ``np.array`` for the duration.
    A buffer-protocol read from C (``memoryview(arr)``) is not seen.
    """
    import numpy as np

    seen = []

    def wrap(owner, name):
        original = getattr(owner, name, None)
        if original is None:
            return

        def spy(*args, **kwargs):
            seen.append(name)
            return original(*args, **kwargs)
        monkeypatch.setattr(owner, name, spy)

    for name in _ARRAY_SYNCS:
        wrap(mx.array, name)
    for name in ("eval", "async_eval"):
        wrap(mx, name)
    for name in ("asarray", "array"):
        wrap(np, name)
    try:
        return seen, fn()
    finally:
        monkeypatch.undo()


def test_probe_sees_host_syncs(monkeypatch):
    import numpy as np

    def control():
        y = mx.ones((2,))
        y.sum().item()
        y.tolist()
        bool(y[0])
        int(y[0])
        float(y[1])
        np.asarray(y)
        mx.eval(y)
    seen, _ = _host_sync_calls(monkeypatch, control)
    for name in ("item", "tolist", "__bool__", "__int__", "__float__", "asarray", "eval"):
        assert name in seen


@pytest.mark.parametrize("laws", [False, True])
def test_ling_decode_forward_reads_no_route_ids_to_host(monkeypatch, laws):
    """The premise check: resident Ling MoE decode has no route-ID readback.

    DS4.1's 7.3 -> 22.1 tok/s came from removing ``np.asarray(ids)`` in its
    streamed-expert provider. A one-token bailing_hybrid forward (KDA + MLA +
    SparseMoeBlock with top-8 of 16 experts) builds its whole graph without a
    single host transfer, so the routed kernels can only win on launches.
    """
    from mlx2.runtime.models import bailing_hybrid as bh

    config = dict(  # noqa: C408 -- keyword names mirror ModelArgs
        model_type="bailing_hybrid", vocab_size=128, hidden_size=128,
        intermediate_size=128, moe_intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=2, num_experts=16,
        num_experts_per_tok=8, num_shared_experts=1, n_group=1, topk_group=1,
        first_k_dense_replace=1, layer_group_size=2, max_position_embeddings=64,
        rms_norm_eps=1e-6, rope_theta=10000, routed_scaling_factor=1.0,
        head_dim=64, kv_lora_rank=64, qk_rope_head_dim=32, qk_nope_head_dim=64,
        v_head_dim=64, no_kda_lora=True, gated_attention_proj_granularity_type="head_wise",
        eos_token_id=127, kda_safe_gate=True, kda_lower_bound=-5.0,
    )
    if laws:  # Ling-3.2-air: fp32 router and route weights, post-activation clamps
        config.update(vendor_v32_laws=True, router_dtype="fp32",
                      expert_swiglu_limit_list=[0, 0, 7.0, 7.0],
                      share_expert_swiglu_limit_list=[0, 0, 7.0, 7.0])
    model = bh.Model(bh.ModelArgs.from_dict(config))
    nn.quantize(model, group_size=64, bits=4,
                class_predicate=lambda name, m: hasattr(m, "to_quantized")
                and not (laws and name.endswith("mlp.gate.gate_proj")))
    mx.eval(model.parameters())
    assert sum(isinstance(layer.mlp, bh.SparseMoeBlock) for layer in model.model.layers) == 3
    cache = model.make_cache()
    mx.eval(model(mx.array([[1, 2, 3, 4, 5, 6]]), cache=cache))
    seen, out = _host_sync_calls(monkeypatch, lambda: model(mx.array([[7]]), cache=cache))
    mx.eval(out)
    assert seen == []
    if laws:
        assert any(b.mlp.expert_swiglu_limit for b in model.model.layers
                   if isinstance(b.mlp, bh.SparseMoeBlock))
