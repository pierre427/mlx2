"""Exact parity for the mlx2-owned Qwen3.8 tree/cohort GDN backend.

Every operation is compared with the served eager ops the ordinary decode
path runs, not with the vendored TensorFold kernels: those copied MLX's
``metal::exp`` sigmoid, which MLX 0.32.4 no longer serves, and a comparison
against them passed only because random inputs never reach -6.84375, the one
bf16 conv output where the two spellings round apart.  The pre kernel is
checked on a value sweep (an identity last conv tap makes each conv output
exactly its qkv value, covering every bf16 value with magnitude in
[2**-6, 32]) and on random conv weights through each row's tree window; the
tree, replay and conv-tail kernels are checked against the served one-token
``gated_delta_kernel`` stepped along each node's path.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx2.runtime.models import qwen38_tree_gdn as owned
from mlx2.runtime.models.gated_delta import compute_g, gated_delta_kernel, normalize_gdn_qk
from mlx2.runtime.models.precise_ops import gate_sigmoid
from mlx2.runtime.models.served_exp import covering_rows

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")

NK, NV, DK, DV, TAPS = 16, 48, 128, 128, 4
C = 2 * NK * DK + NV * DV
PARENTS = ([-1, 0, 0, 1, 1], [-1, 0, 1, 1, 2])


@pytest.fixture(autouse=True)
def gpu():
    # The served ops are the GPU ones (tests/conftest.py defaults to the CPU).
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _equal(have, want, what):
    mx.eval(have, want)
    assert have.shape == want.shape and have.dtype == want.dtype, what
    width = mx.uint16 if have.dtype.size == 2 else mx.uint32
    bad = int((have.view(width) != want.view(width)).sum().item())
    assert bad == 0, f"{what}: {bad} of {have.size} elements differ from the served eager op"


def _draw(shape, seed, dtype=mx.bfloat16, scale=0.05):
    return (mx.random.normal(shape, key=mx.random.key(seed)) * scale).astype(dtype)


def _sweep(rows, seed):
    encodings = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    widened = (encodings.astype(np.uint32) << 16).view(np.float32)
    keep = np.isfinite(widened) & (np.abs(widened) >= 2.0 ** -6) & (np.abs(widened) <= 32.0)
    grid = mx.array(encodings[keep]).view(mx.bfloat16)
    return covering_rows(grid, rows, C, seed)[None]


def _eager_pre(qkv, states, weight, zba, a_log, dt_bias):
    """The ordinary prework of every row, through its own tree window."""
    rows = []
    offset = 0
    for stream, parents in enumerate(PARENTS):
        _, paths = owned.tree_paths(parents)
        for path in paths:
            window = mx.concatenate(
                [states[stream], qkv[:, [offset + row for row in path]]], axis=1
            )[:, -TAPS:]
            rows.append(nn.silu(mx.conv1d(window, weight, groups=C)))
        offset += len(parents)
    conv = mx.concatenate(rows, axis=1)
    count = int(conv.shape[1])
    q, k, v = mx.split(conv, [NK * DK, 2 * NK * DK], -1)
    q, k = normalize_gdn_qk(q.reshape(1, count, NK, DK), k.reshape(1, count, NK, DK))
    b = zba[..., NV * DV:NV * DV + NV]
    a = zba[..., NV * DV + NV:]
    g = compute_g(a_log, a, dt_bias).astype(mx.float32)
    beta = gate_sigmoid(b.astype(mx.float32))
    return q, k, v.reshape(1, count, NV, DV), g, beta


def _inputs(qkv):
    rows = int(qkv.shape[1])
    zba = mx.concatenate([
        _draw((1, rows, NV * DV), 2),
        _draw((1, rows, 2 * NV), 4, scale=3.0),
    ], axis=-1)
    states = [_draw((1, TAPS - 1, C), 10 + lane) for lane in range(len(PARENTS))]
    a_log = mx.linspace(-2.0, 1.0, NV).astype(mx.bfloat16)
    dt_bias = _draw((NV,), 31, scale=0.5)
    return zba, states, a_log, dt_bias


def _run_pre(qkv, weight, zba, states, a_log, dt_bias):
    plan = owned.ConvPlan(PARENTS, TAPS - 1)
    return owned.gdn_pre(qkv, states, weight, plan, zba, a_log, dt_bias, nk=NK, nv=NV, dk=DK, dv=DV)


def test_tree_pre_matches_served_eager_prework_on_a_silu_value_sweep():
    rows = sum(len(p) for p in PARENTS)
    qkv = _sweep(rows, 9600)
    weight = mx.zeros((C, TAPS, 1), dtype=mx.bfloat16)
    weight[:, TAPS - 1, 0] = 1
    zba, states, a_log, dt_bias = _inputs(qkv)
    have = _run_pre(qkv, weight, zba, states, a_log, dt_bias)
    want = _eager_pre(qkv, states, weight, zba, a_log, dt_bias)
    assert bool(mx.any(qkv == mx.array(-6.84375, dtype=mx.bfloat16)).item())
    for name, left, right in zip("qkvgb", have, want):
        _equal(left, right, f"gdn_pre {name} (sweep)")


@pytest.mark.parametrize("packed", [False, True])
def test_mixed_f32_norm_tree_post_matches_ordinary(packed):
    from mlx2.runtime import qwen38_tensorfold
    from mlx2.runtime.models.qwen3_next import Qwen3NextRMSNormGated
    from mlx2.runtime.tensorfold_qwen38 import lane_fuse, lane_glue

    qwen38_tensorfold._install_owned_gdn()
    rows = 4
    y = _draw((1, rows, NV, DV), 9161, scale=2.0)
    z = _draw((1, rows, NV * DV), 9162, scale=3.0)
    norm = Qwen3NextRMSNormGated(DV)
    norm.weight = mx.linspace(0.897123, 1.271236, DV)
    weight_bits = norm.weight.view(mx.uint32)
    expected = norm(y, z.reshape(y.shape)).reshape(1, rows, NV * DV)
    if packed:
        z = mx.concatenate([z, mx.zeros((1, rows, 2 * NV), dtype=mx.bfloat16)], -1)
    actual = (lane_fuse if packed else lane_glue).gdn_post(y, z, norm.weight, norm.eps)
    _equal(actual, expected, "mixed GDN post")
    _equal(norm.weight.view(mx.uint32), weight_bits, "preserved F32 norm")


@pytest.mark.parametrize("mixed_f32", [False, True])
def test_owned_tree_backend_matches_served_eager_ops_all_operations(mixed_f32):
    rows = sum(len(p) for p in PARENTS)
    qkv = _draw((1, rows, C), 1, scale=1.0)
    weight = _draw((C, TAPS, 1), 3, scale=0.5)
    zba, states, a_log, dt_bias = _inputs(qkv)
    if mixed_f32:
        a_log = mx.linspace(-1.912345, 0.987654, NV)
        dt_bias = mx.linspace(-0.712345, 0.934567, NV)
    recurrent = [_draw((1, NV, DV, DK), 20 + lane, mx.float32) for lane in range(len(PARENTS))]

    pre = _run_pre(qkv, weight, zba, states, a_log, dt_bias)
    eager = _eager_pre(qkv, states, weight, zba, a_log, dt_bias)
    for name, left, right in zip("qkvgb", pre, eager):
        _equal(left, right, f"gdn_pre {name} (tree windows)")

    # The split entry point (installed as TensorFold's lane_glue.gdn_pre) on stream 0.
    first = len(PARENTS[0])
    plan = owned.ConvPlan(PARENTS, TAPS - 1)
    split = owned.gdn_pre_split(
        qkv[:, :first], states[0], weight, plan.windows[: first * TAPS],
        zba[:, :first, -NV:], zba[:, :first, NV * DV:NV * DV + NV], a_log, dt_bias,
        nk=NK, nv=NV, dk=DK, dv=DV,
    )
    for name, left, right in zip("qkvgb", split, eager):
        _equal(left, right[:, :first], f"gdn_pre_split {name}")

    # Tree and replay against the served one-token kernel stepped along each path.
    q, k, v, g, beta = eager
    tree_y = owned.tree(*pre[:5], recurrent, owned.TreePlan(PARENTS))
    paths = ([0, 1, 3], [0, 1, 2, 4])
    firsts = (0, first)
    offset = 0
    want_y = []
    want_states = []
    for stream, parents in enumerate(PARENTS):
        _, node_paths = owned.tree_paths(parents)
        for path in node_paths:
            state = recurrent[stream]
            for row in path:
                at = slice(offset + row, offset + row + 1)
                y, state = gated_delta_kernel(q[:, at], k[:, at], v[:, at], g[:, at], beta[:, at], state)
            want_y.append(y)
        state = recurrent[stream]
        for row in paths[stream]:
            at = slice(offset + row, offset + row + 1)
            _, state = gated_delta_kernel(q[:, at], k[:, at], v[:, at], g[:, at], beta[:, at], state)
        want_states.append(state)
        offset += len(parents)
    _equal(tree_y, mx.concatenate(want_y, axis=1), "tree")

    commit = owned.CommitPlan(paths, firsts, TAPS - 1)
    for stream, (left, right) in enumerate(zip(owned.replay(*pre[:5], recurrent, commit), want_states)):
        _equal(left, right, f"replay stream {stream}")

    tails = owned.conv_tails(states, qkv, commit)
    for stream, tail in enumerate(tails):
        rows_ = [firsts[stream] + row for row in paths[stream]]
        want = mx.concatenate([states[stream], qkv[:, rows_]], axis=1)[:, -(TAPS - 1):]
        _equal(tail, want, f"conv tail stream {stream}")

    stats = owned.stats()
    assert stats["backend"] == "mlx2_qwen38_unified_tree_gdn_v1"
    assert stats["pre_parity"] == "exact"
    assert stats["pre_calls"] >= 1
    assert stats["tree_calls"] >= 1
    assert stats["replay_calls"] >= 1
    assert stats["tail_calls"] >= 1
