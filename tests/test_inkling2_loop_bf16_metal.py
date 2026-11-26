"""``inkling2_fuse_loop_bf16`` on Metal (GPU; run under the GPU lock).

The BF16-weight loop controllers (``fuse_kernels.loop_controllers``) at the
Mini's controller geometry (width 2304, rank 64): near-exact against the
reference ``MessageController`` math, bitwise with ``message_mix`` on the
FP32 copies for the gates, and row-independent (a row's output does not
depend on how many rows share the call).  Then a wide random decoder with
the option on: engagement and logit gap.  Set
``INKLING2_LOOP_BF16_METAL_RECEIPT`` to a path for a JSON summary."""

import json
import os

import mlx.core as mx
import pytest

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")

RESULTS = {}
W, R = 2304, 64


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
    path = os.environ.get("INKLING2_LOOP_BF16_METAL_RECEIPT")
    if path:
        with open(path, "w") as handle:
            json.dump(RESULTS, handle, indent=1, sort_keys=True)


class _Owner:
    pass


def _weights(c, seed):
    key = mx.random.key(seed)
    k1, k2, k3 = mx.random.split(key, 3)
    down = (mx.random.normal((c, R, 2 * W), key=k1) * 0.02).astype(mx.bfloat16)
    forget = (mx.random.normal((c, W, R), key=k2) * 0.05).astype(mx.bfloat16)
    gain = (mx.random.normal((c, W, R), key=k3) * 0.05).astype(mx.bfloat16)
    return down, forget, gain


def _inputs(n, c, cx, seed):
    k1, k2 = mx.random.split(mx.random.key(seed + 100))
    x = mx.random.normal((n, cx, W), key=k1)
    m = mx.random.normal((n, c, W), key=k2)
    return x, m


def _reference(x, m, down, forget, gain, eps=1e-6):
    """``MessageController.__call__`` per controller over all rows."""
    c = m.shape[1]
    outs = []
    for i in range(c):
        xi = x[:, 0 if x.shape[1] == 1 else i]
        mi = m[:, i]
        pair = mx.concatenate([mx.fast.rms_norm(v, None, eps) for v in (xi, mi)], axis=-1)
        hidden = (pair.astype(mx.bfloat16) @ down[i].T).astype(mx.float32)
        f = mx.tanh(hidden @ forget[i].astype(mx.float32).T)
        g = mx.tanh(hidden @ gain[i].astype(mx.float32).T)
        outs.append((1 - f) * xi + g * mi)
    return mx.stack(outs, axis=1)


def _kernel(x, m, down, forget, gain, eps=1e-6):
    from mlx2.runtime.models.inkling2 import fuse_kernels as K

    return K.loop_controllers(_Owner(), x, m, mx.fast.rms_norm(x, None, eps),
                              mx.fast.rms_norm(m, None, eps), down, forget, gain)


@pytest.mark.parametrize("c,cx", [(3, 3), (3, 1), (1, 1)])
@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16, 32])
def test_loop_bf16_near_exact_and_row_independent(rows, c, cx):
    down, forget, gain = _weights(c, rows)
    x, m = _inputs(rows, c, cx, rows)
    got = _kernel(x, m, down, forget, gain)
    ref = _reference(x, m, down, forget, gain)
    singles = mx.concatenate([_kernel(x[i:i + 1], m[i:i + 1], down, forget, gain)
                              for i in range(rows)], axis=0)
    mx.eval(got, ref, singles)
    gap = float(mx.abs(got - ref).max())
    RESULTS[f"kernel/rows{rows}/c{c}/cx{cx}"] = {
        "max_abs_vs_reference": gap, "bitwise_vs_reference": bool(mx.array_equal(got, ref)),
        "row_independent": bool(mx.array_equal(got, singles))}
    assert mx.array_equal(got, singles)
    assert gap < 2e-2


def test_loop_bf16_gates_bitwise_with_message_mix():
    """Given the same ``hidden``, the BF16 gates equal ``message_mix`` over
    the FP32 copies bit for bit (widening BF16 is exact)."""
    from mlx2.runtime.models.inkling2 import fuse_kernels as K

    c = 3
    _down, forget, gain = _weights(c, 5)
    x, m = _inputs(1, c, c, 5)
    hidden = mx.random.normal((c, R), key=mx.random.key(9)).astype(mx.bfloat16).astype(mx.float32)
    mix = K.message_mix(_Owner(), hidden, forget.astype(mx.float32), gain.astype(mx.float32),
                        x.reshape(c, W), m.reshape(c, W))
    (ours,) = K._loop_kernel("loop_gates_bf16", K._LOOP_GATES_SOURCE,
                             ["hidden", "forget", "gain", "x", "message"], ["out"])(
        inputs=[hidden, forget, gain, x, m],
        template=[("W", W), ("R", R), ("C", c), ("CX", c)],
        grid=(W, c, 1), threadgroup=(256, 1, 1),
        output_shapes=[(1, c, W)], output_dtypes=[mx.float32])
    mx.eval(mix, ours)
    RESULTS["gates_bitwise_with_message_mix"] = bool(mx.array_equal(mix, ours.reshape(c, W)))
    assert mx.array_equal(mix, ours.reshape(c, W))


def test_loop_bf16_wide_call_row_independent():
    """A 64-row call (four 16-row verify lanes) gives each row its one-row values."""
    down, forget, gain = _weights(3, 3)
    x, m = _inputs(64, 3, 3, 3)
    got = _kernel(x, m, down, forget, gain)
    singles = mx.concatenate([_kernel(x[i:i + 1], m[i:i + 1], down, forget, gain)
                              for i in range(0, 64, 7)], axis=0)
    mx.eval(got, singles)
    assert got is not None
    assert mx.array_equal(got[::7], singles)


@pytest.mark.parametrize("plan_name", ["decode", "verify"])
def test_loop_bf16_model_level(plan_name):
    """Wide BF16 random decoder on the shipped fusion defaults with
    ``fuse_loop_bf16``: engaged on every looped visit, logit gap bounded by
    the BF16 chunking control."""
    from test_inkling2_fuse_metal import DEFAULTS, PLAN, _bf16_bound, _gap, _model, _run

    plan = PLAN if plan_name == "decode" else [(0, 20), (20, 27), (27, 35), (35, 40)]
    model = _model("wide", True)
    model.configure_kernels(moe_fused=True, moe_weighted_sum=True, **DEFAULTS)
    tokens = mx.random.randint(0, 500, (1, 40), key=mx.random.key(7))
    base = _run(model, tokens, plan)
    bound, control = _bf16_bound(model, tokens, base, 1.0)
    model.configure_kernels(moe_fused=True, moe_weighted_sum=True, **DEFAULTS,
                            fuse_loop_bf16=True)
    got = _run(model, tokens, plan)
    counts = model.kernel_receipt()["fuse_counts"]
    result = _gap(got, base)
    result.update(bound=bound, bf16_chaos_control=control,
                  loop_bf16_kernel=counts.get("loop_bf16_kernel", 0),
                  declines={k: v for k, v in counts.items() if "loop_bf16" in k and "declined" in k})
    RESULTS[f"model/{plan_name}"] = result
    assert result["loop_bf16_kernel"] > 0
    assert result["max_abs"] < bound, result
