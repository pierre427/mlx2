"""The Qwen row-exact leaves on the shared ``mlx2.runtime.row_exact`` package.

Every migrated form must give each verify row the bits of its own one-row
call, whatever form the package's parity canary admits on this backend, and a
failed canary must fall back to exactly that per-row call.  On the CPU the
row-exact qmv kernel declines (Metal-only), so these tests pin the plumbing;
Metal bits are checked by scripts/gpu_check_row_exact_dense.py and
scripts/check_mtp_row_exact.py.
"""

from functools import partial

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from mlx2.runtime import row_exact_verify as REV
from mlx2.runtime.models import qwen3_next as QN
from mlx2.runtime.models import row_exact_components as REC
from mlx2.runtime.row_exact import canary, ops

DENSE_ROUTES = ({"matvec_rows": 1}, {"per_row_canary": 1}, {"per_row_biased": 1})


@pytest.fixture(autouse=True)
def _cpu_and_fresh_canaries():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    canary.CANARIES.reset()
    yield
    canary.CANARIES.reset()
    mx.set_default_device(previous)


@pytest.fixture
def failing_canaries(monkeypatch):
    """Every canary fails: each op must take its per-row form."""
    monkeypatch.setattr(canary.CANARIES, "exact", lambda key, probe, **_: False)
    monkeypatch.setattr(canary.CANARIES, "exact_private", lambda key, probe, **_: False)


def _bits(value):
    mx.eval(value)
    view = {mx.bfloat16: mx.uint16, mx.float16: mx.uint16, mx.float32: mx.uint32}
    return np.array(value.view(view[value.dtype]))


def _same(a, b):
    return np.array_equal(_bits(a), _bits(b))


def _x(shape, seed):
    return (mx.random.normal(shape, key=mx.random.key(seed)) * 0.5).astype(mx.bfloat16)


def _quantized(k, n, *, bits=4, seed=0):
    mx.random.seed(seed)
    base = nn.Linear(k, n, bias=False)
    base.set_dtype(mx.bfloat16)
    return nn.QuantizedLinear.from_linear(base, group_size=64, bits=bits)


def _dense(k, n, *, bias=False, seed=0):
    mx.random.seed(seed)
    layer = nn.Linear(k, n, bias=bias)
    layer.set_dtype(mx.bfloat16)
    return layer


def _swap(layer, mixin):
    layer.__class__ = REC.row_exact_subclass(mixin, type(layer))
    return layer


def _one_row_calls(call, x):
    """``call`` on every row of ``x`` ([..., K]) as a [1, 1, K] input."""
    flat = x.reshape(-1, 1, 1, x.shape[-1])
    out = mx.concatenate([call(flat[r]) for r in range(flat.shape[0])], axis=0)
    return out.reshape(*x.shape[:-1], out.shape[-1])


# ----------------------------------------------------------- projections


@pytest.mark.parametrize("shape", [(1, 5, 128), (2, 3, 128)])
def test_quantized_mixin_strip_matches_one_row_calls(shape):
    layer = _swap(_quantized(128, 48), REC.RowExactQuantizedLinearMixin)
    x = _x(shape, 1)
    record = REV.Window(int(np.prod(shape[:-1])))
    with REV.window(record):
        y = layer(x)
    want = _one_row_calls(lambda row: nn.QuantizedLinear.__call__(layer, row), x)
    assert y.shape == want.shape and _same(y, want)
    # The canary's probe ran inside this window's first call but outside the
    # window: only the op's own route is noted (CPU: the kernel declines).
    assert record.exact and record.stages == {"projections": {"per_row": 1}}
    # Weight-bound projections keep per-module (private) verdicts.
    assert canary.CANARIES.snapshot()["private"]["checked"] == 1


@pytest.mark.parametrize("shape", [(1, 5, 64), (3, 2, 64)])
def test_dense_mixin_strip_matches_one_row_calls(shape):
    layer = _swap(_dense(64, 24), REC.RowExactLinearMixin)
    x = _x(shape, 2)
    record = REV.Window(int(np.prod(shape[:-1])))
    with REV.window(record):
        y = layer(x)
    want = _one_row_calls(lambda row: nn.Linear.__call__(layer, row), x)
    assert y.shape == want.shape and _same(y, want)
    assert record.exact and record.stages["projections"] in DENSE_ROUTES


def test_dense_mixin_with_bias_still_matches_its_module_call():
    layer = _swap(_dense(64, 24, bias=True), REC.RowExactLinearMixin)
    x = _x((1, 4, 64), 3)
    record = REV.Window(4)
    with REV.window(record):
        y = layer(x)
    assert _same(y, _one_row_calls(lambda row: nn.Linear.__call__(layer, row), x))
    assert record.stages["projections"] in DENSE_ROUTES


def test_failed_canaries_run_the_one_row_calls(failing_canaries):
    quantized = _swap(_quantized(128, 32), REC.RowExactQuantizedLinearMixin)
    dense = _swap(_dense(64, 16), REC.RowExactLinearMixin)
    xq, xd = _x((1, 4, 128), 4), _x((1, 4, 64), 5)
    record = REV.Window(4)
    with REV.window(record):
        yq, yd = quantized(xq), dense(xd)
    assert _same(yq, _one_row_calls(lambda r: nn.QuantizedLinear.__call__(quantized, r), xq))
    assert _same(yd, _one_row_calls(lambda r: nn.Linear.__call__(dense, r), xd))
    assert record.exact and record.stages == {"projections": {"per_row_canary": 2}}


def test_dense_linear_accepts_the_route_class_swap_only():
    swapped = _swap(_dense(32, 8), REC.RowExactLinearMixin)

    class Other(nn.Linear):
        pass

    other = Other(32, 8, bias=False)
    other.set_dtype(mx.bfloat16)
    x = _x((1, 3, 32), 6)
    record = REV.Window(3)
    with REV.window(record):
        ops.dense_linear(swapped, x, stage="swapped")
        ops.dense_linear(other, x, stage="other")
    assert record.stages["swapped"] in DENSE_ROUTES
    assert record.stages["other"] == {"per_row_unsupported": 1}


def test_grouped_projections_match_one_row_calls():
    group = tuple(
        _swap(_quantized(128, n, seed=n), REC.RowExactQuantizedLinearMixin)
        for n in (64, 32, 8, 8)
    )
    x = _x((2, 3, 128), 7)
    record = REV.Window(6)
    with REV.window(record):
        outputs = REC.grouped_quantized_linears(record, group, x)
    for layer, y in zip(group, outputs, strict=True):
        assert _same(y, _one_row_calls(partial(nn.QuantizedLinear.__call__, layer), x))
    # CPU: the grouped kernel declines, each projection runs per row.
    assert record.exact and record.stages == {"projections": {"per_row": 4}}


def test_grouped_projections_with_a_failed_canary(failing_canaries):
    group = tuple(_quantized(128, n, seed=n) for n in (16, 8))
    x = _x((1, 5, 128), 8)
    record = REV.Window(5)
    with REV.window(record):
        outputs = ops.quantized_linears(group, x, stage="projections")
    for layer, y in zip(group, outputs, strict=True):
        assert _same(y, _one_row_calls(layer, x))
    assert record.stages == {"projections": {"per_row_canary": 2}}


def test_canary_probe_runs_outside_the_verify_window():
    seen = []

    def probe(rows):
        seen.append(REV.current())
        x = _x((1, rows, 8), rows)
        return x, x

    record = REV.Window(3)
    with REV.window(record):
        assert canary.CANARIES.exact(("outside-window-probe",), probe)
        assert REV.current() is record
    assert seen and all(window is None for window in seen)
    assert record.stages == {} and record.exact


# ----------------------------------------------------------- MoE blocks


def test_fused_router_block_runs_once_per_row():
    class Block:
        moe_router_mode = "fused"

        def __call__(self, x):
            # Mixes rows: only a per-row call gives each row its own bits.
            return x * x.sum(axis=-2, keepdims=True)

    block = Block()
    block.__class__ = REC.row_exact_subclass(REC.RowExactMoEMixin, Block)
    x = _x((2, 3, 16), 9)
    record = REV.Window(6)
    with REV.window(record):
        y = block(x)
    assert _same(y, _one_row_calls(Block.__call__.__get__(block), x))
    assert record.stages == {"moe": {"per_row_block": 6}}


def test_moe_window_projections_match_one_row_calls():
    gate, shared_gate = _quantized(128, 16, bits=8, seed=1), _quantized(128, 1, bits=8, seed=2)
    flat = _x((5, 128), 10)
    record = REV.Window(5)
    with REV.window(record):
        logits, gate_logit = QN._row_projections(record, (gate, shared_gate), flat)
        (alone,) = QN._row_projections(record, (gate,), flat)
    one = lambda layer: _one_row_calls(layer, flat[:, None])[:, 0]
    assert _same(logits, one(gate)) and _same(alone, one(gate))
    assert _same(gate_logit, one(shared_gate))
    assert record.stages == {"projections": {"per_row": 3}}


@pytest.mark.parametrize("canaries", ["admitted", "failed"])
def test_moe_window_dense_shared_gate_matches_one_row_calls(monkeypatch, canaries):
    if canaries == "failed":
        monkeypatch.setattr(canary.CANARIES, "exact", lambda key, probe, **_: False)
        monkeypatch.setattr(canary.CANARIES, "exact_private", lambda key, probe, **_: False)
    shared_gate = _dense(128, 1, seed=3)
    assert QN._dense_gate_ok(shared_gate)
    flat = _x((4, 128), 11)
    record = REV.Window(4)
    with REV.window(record):
        y = QN._row_gate_logits(record, shared_gate, flat)
    want = _one_row_calls(shared_gate, flat[:, None])[:, 0]
    assert y.shape == (4, 1) and _same(y, want)
    # A one-output gate always runs one one-row call per row, whatever the
    # canary would say (``ops._single_output``).
    assert record.stages == {"projections": {"per_row_single_output": 1}}
    # Outside a row-exact window (the batch_decode consumer) nothing is noted.
    assert _same(QN._row_gate_logits(None, shared_gate, flat), want)


# ----------------------------------------------------------- status


def test_route_status_reports_package_canaries():
    from test_qwen4_row_exact_verify import _quantized_tiny_model

    from mlx2.runtime.models.qwen4_row_exact import install

    handle = install(_quantized_tiny_model())
    try:
        status = handle.status()
        # The rows actually probed so far (none before any verify window),
        # shared and per-module (private) verdicts, inconclusive refusals.
        canaries = status["canaries"]
        assert canaries["probe_rows"] == [] and canaries["failed"] == []
        assert set(canaries["private"]) == {"checked", "inconclusive", "failed"}
        assert canaries["inconclusive"] == 0
    finally:
        handle.remove()


# ----------------------------------------------------------- hot-path cost


def test_verdicts_are_cached_on_the_module_per_generation(monkeypatch):
    """The verify path calls these ops hundreds of times per window: after
    the first call per (module, dtype, width) the registry is not consulted;
    a canary reset invalidates the module's cached verdicts."""
    calls = []
    exact = canary.CANARIES.exact_private

    def counting(key, probe, **options):
        calls.append(key[0])
        return exact(key, probe, **options)

    monkeypatch.setattr(canary.CANARIES, "exact_private", counting)
    layer = _quantized(128, 16)
    dense = _dense(64, 8)
    group = (_quantized(128, 8, seed=1), _quantized(128, 8, seed=2))
    xq, xd = _x((1, 3, 128), 12), _x((1, 3, 64), 13)
    for _ in range(3):
        ops.quantized_linear(layer, xq, stage="q")
        ops.dense_linear(dense, xd, stage="d")
        ops.quantized_linears(group, xq, stage="g")
    assert calls == ["quantized_rows", "dense_linear", "quantized_group"]
    ops.quantized_linear(layer, xq.astype(mx.float16), stage="q")  # new dtype: new verdict
    assert calls[-1] == "quantized_rows" and len(calls) == 4
    canary.CANARIES.reset()
    ops.quantized_linear(layer, xq, stage="q")
    assert len(calls) == 5
