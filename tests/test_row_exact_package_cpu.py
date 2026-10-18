"""The shared exact multi-row package: ops, canaries, recipes, planner."""
import json

import mlx.core as mx
import mlx.nn as nn
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime import row_exact_verify as REV
from mlx2.runtime.row_exact import canary, ops, planner, recipes


@pytest.fixture(autouse=True)
def _fresh_canaries():
    canary.CANARIES.reset()
    yield
    canary.CANARIES.reset()


def _strip(rows, width, seed=0):
    return (mx.random.normal((1, rows, width), key=mx.random.key(seed)) * 0.5).astype(
        mx.bfloat16
    )


def _same_bits(a, b):
    mx.eval(a, b)
    return canary.bit_mismatches(a, b) == 0


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("rows", [2, 3, 5])
def test_dense_linear_matches_one_row_calls(rows, bias):
    linear = nn.Linear(64, 48, bias=bias)
    linear.set_dtype(mx.bfloat16)
    x = _strip(rows, 64)
    record = REV.Window(rows)
    with REV.window(record):
        y = ops.dense_linear(linear, x, stage="probe")
    # Whatever form the canary admits on this backend, the bits are the
    # module's one-row bits (a biased Linear's fused addmm falls back).
    assert _same_bits(y, ops.per_row(linear, x))
    assert set(record.stages["probe"]) <= {"matvec_rows", "per_row_canary", "per_row_biased"}


def test_quantized_linear_matches_one_row_calls():
    base = nn.Linear(128, 64, bias=False)
    base.set_dtype(mx.bfloat16)
    linear = nn.QuantizedLinear.from_linear(base, group_size=64, bits=8)
    x = _strip(4, 128, seed=1)
    assert _same_bits(ops.quantized_linear(linear, x), ops.per_row(linear, x))


def test_row_dependent_op_fails_its_canary_and_runs_per_row():
    # Mixing rows (a mean over the row axis) is exactly what a canary must catch.
    # (A full mean-subtraction makes every one-row reference zero, which the
    # canary rightly calls inconclusive; half of it still mixes rows.)
    leaky = lambda x: x - 0.5 * x.mean(axis=1, keepdims=True)
    x = _strip(3, 32, seed=2)
    record = REV.Window(3)
    with REV.window(record):
        y = ops.row_independent(leaky, x, stage="leaky", key=("leaky",))
    assert record.stages["leaky"] == {"per_row_canary": 1}
    assert _same_bits(y, ops.per_row(leaky, x))
    failed = canary.CANARIES.snapshot()["failed"]
    assert len(failed) == 1 and failed[0]["mismatches"] > 0


def test_canary_runs_once_per_key_and_build(monkeypatch):
    calls = {"n": 0}

    def probe(rows):
        calls["n"] += 1
        x = _strip(rows, 8)
        return x, x

    assert canary.CANARIES.exact(("k",), probe)
    assert canary.CANARIES.exact(("k",), probe)
    assert calls["n"] == len(canary.PROBE_ROWS)
    monkeypatch.setattr(canary, "build_identity", lambda: ("other-build", "gpu"))
    assert canary.CANARIES.exact(("k",), probe)
    assert calls["n"] == 2 * len(canary.PROBE_ROWS)


def test_router_select_matches_per_row_selection():
    weight = (mx.random.normal((16, 32), key=mx.random.key(5)) * 0.2).astype(mx.bfloat16)

    def select(logits):
        scores = mx.sigmoid(logits.astype(mx.float32))
        inds = mx.argpartition(-scores, kth=3, axis=-1)[..., :4]
        return inds, mx.take_along_axis(scores, inds, axis=-1)

    x = _strip(5, 32, seed=6)
    inds, scores = ops.router_select(weight, x, select, key=("test_router",))
    rows = [select(x[:, r : r + 1] @ weight.T) for r in range(5)]
    assert _same_bits(inds, mx.concatenate([r[0] for r in rows], axis=1))
    assert _same_bits(scores, mx.concatenate([r[1] for r in rows], axis=1))


def test_routed_experts_and_combine_match_per_row():
    table = (mx.random.normal((8, 16, 24), key=mx.random.key(7)) * 0.3).astype(mx.bfloat16)

    def experts(x, indices):
        # [1, R, top_k, 16]: each chosen expert's matrix applied to its row.
        return (table[indices] @ x[:, :, None, :, None])[..., 0]

    x = _strip(4, 24, seed=8)
    indices = mx.array([[[0, 3], [1, 2], [7, 4], [5, 6]]], dtype=mx.uint32)
    routed = ops.routed_experts(experts, x, indices, num_experts=8, key=("toy",))
    reference = mx.concatenate(
        [experts(x[:, r : r + 1], indices[:, r : r + 1]) for r in range(4)], axis=1
    )
    assert _same_bits(routed, reference)
    scores = mx.random.uniform(shape=(1, 4, 2), key=mx.random.key(9))
    combined = ops.combine_experts(routed, scores)
    assert combined.shape == (1, 4, 16)


def test_run_layerwise_notes_recipes_and_taps():
    class Block:
        def __init__(self, scale):
            self.scale = scale

        def __call__(self, x, mask=None, cache=None):
            return x * self.scale

    layers = [Block(2.0), Block(3.0), Block(0.5)]
    pick = lambda layer: recipes.SEGMENT if layer.scale == 3.0 else recipes.PER_ROW
    x = _strip(3, 8, seed=10).astype(mx.float32)
    record = REV.Window(3)
    with REV.window(record):
        hidden, taps = recipes.run_layerwise(layers, pick, x, [None] * 3, capture_layers=(1,))
    assert record.stages["blocks"] == {"per_row": 2, "segment": 1}
    assert _same_bits(hidden, x * 3.0)
    assert len(taps) == 1 and _same_bits(taps[0], x * 6.0)


def test_planner_heuristic_and_measured_table(tmp_path, monkeypatch):
    kwargs = dict(bits=8, n=4096, k=8192, limit=4, tiles=512, narrow_tiles=2048)
    device, build = "Test GPU", "9.9.9"
    # The heuristic is checked with no table at all: a measured table shipped
    # for this host (tables/<device>__<build>.json) must not answer for it.
    monkeypatch.setattr(planner, "TABLE_DIR", tmp_path)
    monkeypatch.setattr(planner, "device_name", lambda: device)
    monkeypatch.setattr("mlx2.runtime.mlx_build.installed_mlx_build", lambda: build)
    planner.reset()
    assert planner.rows_per_group(rows=2, **kwargs) == (1, "heuristic")
    assert planner.rows_per_group(rows=5, **kwargs) == (3, "heuristic")
    assert planner.rows_per_group(rows=7, **kwargs) == (4, "heuristic")
    planner.reset()
    (tmp_path / planner.table_name(device, build)).write_text(
        json.dumps(
            {
                "device": device,
                "mlx_build": build,
                "rows_per_group": {
                    planner.table_key(8, 4096, 8192, 5): 4,
                    planner.table_key(8, 4096, 8192, 6): 9,  # invalid: above limit
                },
            }
        )
    )
    try:
        assert planner.rows_per_group(rows=5, **kwargs) == (4, "table")
        assert planner.rows_per_group(rows=6, **kwargs) == (3, "heuristic")
    finally:
        planner.reset()


def test_nemotron_routed_layerwise_strip_matches_per_row_blocks(monkeypatch):
    from mlx2.runtime.models.nemotron35_super import Model as SuperModel
    from mlx2.runtime.models.nemotron_h import Model as LanguageModel
    from tests.test_nemotron_external_taps_cpu import tiny_model

    import copy

    mx.random.seed(21)
    args = copy.copy(tiny_model().args)
    args.hybrid_override_pattern = ["E"]
    args.num_hidden_layers = 1
    args.moe_latent_size = 4
    args.moe_intermediate_size = 6
    args.moe_shared_expert_intermediate_size = 7
    args.n_group = 1
    args.n_routed_experts = 4
    args.n_shared_experts = 1
    args.topk_group = 1
    args.num_experts_per_tok = 2
    args.norm_topk_prob = True
    args.routed_scaling_factor = 1.0
    model = SuperModel.__new__(SuperModel)
    LanguageModel.__init__(model, args)
    model.eval()
    moe = [layer for layer in model.layers if layer.block_type == "E"]
    x = _strip(3, args.hidden_size, seed=22)
    block = moe[0]
    # The production gate admits only the 120B geometry; the toy shape runs the
    # same code path.
    monkeypatch.setattr(type(block.mixer), "_routed_batch_geometry_refusal", lambda self, rows: None)
    normalized = block.norm(x)
    batched = block.mixer._routed_rows_exact(normalized)
    reference = ops.per_row(block.mixer, normalized)
    assert _same_bits(batched, reference)
    model._short_segment_policy = "layerwise_short_exact_routed_batch"
    recipe = model._row_exact_recipe(block)
    assert recipe.name == "moe_exact"
    record = REV.Window(3)
    with REV.window(record):
        out = recipe.run(block, x, None)
    model._row_exact_close(record)
    assert _same_bits(out, ops.per_row(lambda row: block(row, mask=None, cache=None), x))
    status = model.row_exact_status()
    assert status["windows"] == 1 and status["windows_exact"] == 1
    assert status["stages"]["moe_router"] == {"batched_selection": 1}


# --- codex review 2026-10-17 (findings 1-4) -------------------------------


def test_verdict_is_per_requested_row_count():
    # Exact through eight rows but not at six: the six-row verdict must fail.
    def probe(rows):
        x = _strip(rows, 8)
        return (x + 1 if rows == 6 else x), x

    assert canary.CANARIES.exact(("rows-key",), probe, rows=8)
    assert not canary.CANARIES.exact(("rows-key",), probe, rows=6)
    assert canary.CANARIES.exact(("rows-key",), probe, rows=8)


def test_all_zero_reference_is_inconclusive_and_not_shared():
    calls = {"n": 0}

    def probe(rows):
        calls["n"] += 1
        zero = mx.zeros((1, rows, 8), dtype=mx.bfloat16)
        return zero, zero

    assert not canary.CANARIES.exact(("zero-key",), probe, rows=3)
    assert not canary.CANARIES.exact(("zero-key",), probe, rows=3)
    assert calls["n"] == 2  # not cached: the next module probes for itself
    assert canary.CANARIES.snapshot()["checked"] == 0


def test_weight_swap_reprobes_the_module_verdict(monkeypatch):
    linear = nn.Linear(32, 16, bias=False)
    linear.set_dtype(mx.bfloat16)
    x = _strip(3, 32, seed=3)
    seen = []
    real = canary.CANARIES.exact_private

    def spy(key, probe, *, rows):
        seen.append((key[0], rows))
        return real(key, probe, rows=rows)

    monkeypatch.setattr(canary.CANARIES, "exact_private", spy)
    ops.dense_linear(linear, x)
    ops.dense_linear(linear, x)
    assert len(seen) == 1  # cached on the module
    linear.update({"weight": (linear.weight * 2).astype(mx.bfloat16)})
    ops.dense_linear(linear, x)
    assert len(seen) == 2  # new arrays: probed again
    ops.dense_linear(linear, _strip(5, 32, seed=4))
    assert seen[-1] == ("dense_linear", 5)  # a new row count: probed at it


def test_biased_dense_layer_always_runs_its_own_one_row_call():
    linear = nn.Linear(32, 16, bias=True)
    linear.set_dtype(mx.bfloat16)
    linear.update({"bias": mx.zeros((16,), dtype=mx.bfloat16)})  # would fool a canary
    x = _strip(4, 32, seed=5)
    record = REV.Window(4)
    with REV.window(record):
        y = ops.dense_linear(linear, x, stage="biased")
    assert record.stages["biased"] == {"per_row_biased": 1}
    assert _same_bits(y, ops.per_row(linear, x))


def test_reset_during_a_probe_fails_closed_and_keeps_nothing():
    def probe(rows):
        canary.CANARIES.reset()  # a reset lands mid-probe
        x = _strip(rows, 8)
        return x, x

    assert not canary.CANARIES.exact(("reset-key",), probe, rows=2)
    assert canary.CANARIES.snapshot()["checked"] == 0


def test_same_shaped_modules_never_share_a_verdict():
    # Module B must probe its own arrays even though A has the same geometry.
    calls = []
    real = canary.CANARIES.exact_private

    def spy(key, probe, *, rows):
        calls.append(key)
        return real(key, probe, rows=rows)

    a, b = (nn.Linear(32, 16, bias=False) for _ in range(2))
    for layer in (a, b):
        layer.set_dtype(mx.bfloat16)
    x = _strip(3, 32, seed=8)
    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as patch:
        patch.setattr(canary.CANARIES, "exact_private", spy)
        ops.dense_linear(a, x)
        ops.dense_linear(b, x)
    assert len(calls) == 2


def test_a_failing_concurrent_probe_wins():
    def failing(rows):
        x = _strip(rows, 8)
        return x + 1, x

    def passing(rows):
        # A concurrent first use of the same key fails while this one runs.
        assert not canary.CANARIES.exact(("race-key",), failing, rows=rows)
        x = _strip(rows, 8)
        return x, x

    assert not canary.CANARIES.exact(("race-key",), passing, rows=2)
    assert not canary.CANARIES.exact(("race-key",), passing, rows=2)


def test_group_records_each_members_own_route(monkeypatch):
    from mlx2.runtime.models import row_exact_qmv as REQ

    base = [nn.Linear(64, 32, bias=False) for _ in range(2)]
    for b in base:
        b.set_dtype(mx.bfloat16)
    group = [nn.QuantizedLinear.from_linear(b, group_size=64, bits=8) for b in base]
    x = _strip(3, 64, seed=6)
    monkeypatch.setattr(ops, "_module_verdict", lambda *a, **k: True)

    def fake(linears, strip, *, per_member=False):
        outputs = tuple(ops.per_row(m, strip) for m in linears)
        return outputs, (["kernel", "per_row"] if per_member else "mixed")

    monkeypatch.setattr(REQ, "quantized_linears", fake)
    record = REV.Window(3)
    with REV.window(record):
        ops.quantized_linears(group, x, stage="grp")
    assert record.stages["grp"] == {"kernel": 1, "per_row": 1}


def test_ungrouped_launch_reports_mixed_not_the_last_route(monkeypatch):
    from mlx2.runtime.models import row_exact_qmv as REQ

    base = [nn.Linear(64, 32, bias=False) for _ in range(2)]
    for b in base:
        b.set_dtype(mx.bfloat16)
    group = [nn.QuantizedLinear.from_linear(b, group_size=64, bits=8) for b in base]
    routes = iter(["kernel", "per_row"])
    monkeypatch.setattr(REQ, "_group_plan", lambda *a, **k: None)
    monkeypatch.setattr(
        REQ, "quantized_linear", lambda linear, x, call=None: (linear(x), next(routes))
    )
    _outputs, route = REQ.quantized_linears(group, _strip(3, 64, seed=7))
    assert route == "mixed"


def test_router_select_with_a_cast_probes_once_and_holds_one_entry(monkeypatch):
    # 2026-10-17: router_select casts its weight per call; keyed on the cast
    # copy, every call re-probed (verify 37 -> 66 ms on Nemotron) and stored
    # another weight copy.  The verdict is held against the stored weight.
    weight = (mx.random.normal((16, 32), key=mx.random.key(5)) * 0.2).astype(mx.bfloat16)

    def select(logits):
        inds = mx.argpartition(-logits, kth=3, axis=-1)[..., :4]
        return inds, mx.take_along_axis(logits, inds, axis=-1)

    calls = []
    real = canary.CANARIES.exact_private

    def spy(key, probe, *, rows):
        calls.append(key)
        return real(key, probe, rows=rows)

    monkeypatch.setattr(canary.CANARIES, "exact_private", spy)
    ops._LOGITS_VERDICTS.clear()
    x = _strip(3, 32, seed=6)
    for _ in range(5):
        ops.router_select(weight, x, select, compute_dtype=mx.float32, key=("cast",))
    assert len(calls) == 1
    assert len(ops._LOGITS_VERDICTS) == 1


def test_single_output_dense_layer_always_runs_per_row(monkeypatch):
    """A ``[1, K]`` dense layer (a bf16 MoE shared gate) runs one one-row call
    per row and never consults a canary: on MLX 0.32.4 no multi-row form
    reproduced its M=N=1 one-row product on ~1 in 8192 rows, far below what
    a probe-row canary can detect (Flash-Next, 2026-10-17)."""
    def no_canary(*args, **kwargs):
        raise AssertionError("a one-output dense layer consulted a canary")

    monkeypatch.setattr(ops.CANARIES, "exact", no_canary)
    linear = nn.Linear(64, 1, bias=False)
    linear.set_dtype(mx.bfloat16)
    calls = []
    plain = nn.Linear.__call__

    def spy(self, v):
        calls.append(tuple(v.shape))
        return plain(self, v)

    monkeypatch.setattr(nn.Linear, "__call__", spy)
    x = (mx.random.normal((1, 5, 64), key=mx.random.key(7))).astype(mx.bfloat16)
    record = REV.Window(5)
    with REV.window(record):
        y = ops.dense_linear(linear, x, stage="gate")
        z = ops.dense_logits(linear.weight, x, stage="logit")
    assert calls == [(1, 1, 64)] * 5
    want = mx.concatenate([plain(linear, x[:, r : r + 1]) for r in range(5)], axis=1)
    assert mx.array_equal(y.view(mx.uint16), want.view(mx.uint16)).item()
    assert mx.array_equal(z.view(mx.uint16), want.view(mx.uint16)).item()
    assert record.stages == {"gate": {"per_row_single_output": 1}, "logit": {"per_row_single_output": 1}}
