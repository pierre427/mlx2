"""Batched fused GDN verify (B lanes x S rows): admission, per-lane
bookkeeping, rollback routing, counters and policy on the CPU.

Metal bit-exactness lives in ``scripts/check_qwen4_gdn_batch_verify.py``.  On
the CPU an admitted batched verify records "Metal runtime unavailable" in the
batch-verify counters and runs the stock chain; any other reason is a
geometry refusal the GPU would also make.  The per-lane rollback bookkeeping
is exercised with stand-in kernels whose outputs encode the lane and the
replayed count, through both the merged ``ArraysCache`` (``per_row_fn``) and
the served ``SegmentedBatchArraysCache`` (per-row slices of ``fn(m)``).
"""

import mlx.core as mx
import pytest

from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.runtime.models import qwen4_exp
from mlx2.runtime.models import qwen4_fused_gdn_verify as V
from mlx2.runtime.models.cache import ArraysCache
from mlx2.runtime.models.qwen4_fused_gdn import CONV_DIM, NUM_VALUE_HEADS, VALUE_DIM
from mlx2.runtime.segmented_batch_cache import SegmentedBatchArraysCache
from test_gdn_mask_geometry import production_gdn_qwen4  # noqa: F401 - fixture

ADMITTED_ON_CPU = "Metal runtime unavailable"


def _operands(rows, steps, *, state_rows=None, dtype=mx.bfloat16):
    state_rows = rows if state_rows is None else state_rows
    return dict(
        qkv=mx.zeros((rows, steps, CONV_DIM), dtype),
        z=mx.zeros((rows, steps, VALUE_DIM), dtype),
        b=mx.zeros((rows, steps, NUM_VALUE_HEADS), dtype),
        a=mx.zeros((rows, steps, NUM_VALUE_HEADS), dtype),
        conv_state=mx.zeros((state_rows, 3, CONV_DIM), dtype),
        recurrent_state=mx.zeros((state_rows, NUM_VALUE_HEADS, 128, 128), mx.float32),
        conv_weight=mx.zeros((CONV_DIM, 4, 1), dtype),
        A_log=mx.zeros((NUM_VALUE_HEADS,), mx.float32),
        dt_bias=mx.zeros((NUM_VALUE_HEADS,), dtype),
        norm_weight=mx.zeros((128,), dtype),
    )


def _admit(rows, steps, **overrides):
    kwargs = dict(
        _operands(rows, steps), mask=None, spans=(), speculating=True, training=False,
        sharded=False, num_key_heads=16, num_value_heads=48, key_head_dim=128,
        value_head_dim=128, conv_kernel=4, gate_activation="sigmoid",
    )
    kwargs.update(overrides)
    return V.admit_qwen4_fused_gdn_batch_verify(**kwargs)


@pytest.fixture
def wide_verify():
    previous = V.set_verify_max_steps(17)
    try:
        yield
    finally:
        V.set_verify_max_steps(previous)


@pytest.mark.parametrize("rows", [2, 3, 4, 8, 16])
@pytest.mark.parametrize("steps", [2, 3, 9, 17])
def test_admission_accepts_every_lane_count_and_width(wide_verify, rows, steps):
    assert _admit(rows, steps).accepted
    ragged = [steps] + [1 + (r % steps) for r in range(1, rows)]
    mask = mx.arange(steps)[None, :] < mx.array(ragged)[:, None]
    assert _admit(rows, steps, spans=ragged, mask=mask).accepted
    assert V.batch_verify_row_steps(ragged, mask, rows, steps) == tuple(ragged)
    assert V.batch_verify_row_steps((), None, rows, steps) == (steps,) * rows


def test_admission_refusals_are_named(wide_verify):
    mask = mx.ones((2, 3), mx.bool_)
    assert _admit(1, 3).reason == "single row (B=1 kernel)"
    assert _admit(17, 3).reason == "batch of 17 rows > 16"
    assert _admit(2, 1).reason == "verify width 1 below 2"
    assert _admit(2, 18).reason == "verify width 18 above 17"
    assert _admit(2, 3, speculating=False).reason == "not a speculative verify"
    assert _admit(2, 3, training=True).reason == "training"
    assert _admit(2, 3, sharded=True).reason == "distributed sharding"
    assert _admit(2, 3, spans=None).reason == "rollback geometry not describable"
    assert _admit(2, 3, mask=mask).reason == "masked batched verify"
    assert _admit(3, 3, spans=[3, 3]).reason == "rollback spans for 2 rows, batch of 3"
    assert _admit(2, 3, spans=[3, 0], mask=mask).reason == "idle lane in batched verify"
    assert _admit(2, 3, spans=[3, 2]).reason == "padded lane without a mask"
    assert _admit(2, 3, spans=[3, 4], mask=mask).reason == "lane span exceeds the verify width"
    stale = _admit(3, 3, **_operands(3, 3, state_rows=2))
    assert not stale.accepted and stale.reason.startswith("conv_state shape")
    assert _admit(2, 3, gate_activation="swish").reason == "output gate 'swish'"
    wrong = _operands(2, 3)
    wrong["recurrent_state"] = wrong["recurrent_state"].astype(mx.bfloat16)
    assert _admit(2, 3, **wrong).reason == "recurrent_state must be float32 or float16"


def test_admission_follows_the_selected_verify_width_bound():
    previous = V.set_verify_max_steps(8)
    try:
        assert _admit(2, 8).accepted
        assert _admit(2, 9).reason == "verify width 9 above 8"
    finally:
        V.set_verify_max_steps(previous)


def test_lane_sources_are_checked_derivations_of_the_b1_bodies():
    for source in (V._BATCH_SOURCE, V._BATCH_SOURCE_ST16, V._BATCH_REPLAY_SOURCE,
                   V._BATCH_REPLAY_SOURCE_ST16):
        assert "uint row = (uint)S + tap" not in source
        assert "for (uint t = 0; t < (uint)S;" not in source
        assert "for (uint t = 0; t < RS; ++t)" in source
        assert "const uint SNAPS = RS - 1u;" in source
        assert "const uint RS = (uint)row_steps[batch_row];" in source
        # Padded rows are written, never left uninitialized.
        assert "for (uint t = RS; t < (uint)S; ++t)" in source
    assert "if (t + 1u == RS)" in V._BATCH_REPLAY_SOURCE
    assert "device half* so" in V._BATCH_SOURCE_ST16
    # Everything after the lane prefix is the B=1 body with the width swapped.
    body = V._BATCH_SOURCE.split("  {\n", 1)[1]
    expected = (
        V._SOURCE.replace("threadgroup_position_in_grid.z", "head_in_row")
        .replace("constexpr uint SNAPS = (uint)S - 1u;", "const uint SNAPS = RS - 1u;")
        .replace("uint row = (uint)S + tap;", "uint row = RS + tap;")
        .replace("for (uint t = 0; t < (uint)S; ++t) {", "for (uint t = 0; t < RS; ++t) {")
    )
    assert expected in body


def test_launch_wrapper_rejects_row_steps_that_do_not_fit():
    ops = _operands(2, 3)
    args = (ops["qkv"], ops["z"], ops["b"], ops["a"], ops["conv_state"], ops["conv_weight"],
            ops["A_log"], ops["dt_bias"], ops["recurrent_state"], ops["norm_weight"], 1e-6)
    for bad in ((3,), (3, 0), (3, 4)):
        with pytest.raises(ValueError, match="row steps"):
            V.qwen4_fused_gdn_batch_verify(*args, bad, threadgroup_y=32)
    with pytest.raises(ValueError, match="threadgroup_y"):
        V.qwen4_fused_gdn_batch_replay_verify(*args, (3, 2), threadgroup_y=3)


# ---------------------------------------------------------------- policy ----

def test_policy_field_defaults_row_exact_and_off_is_recorded():
    # Default "row_exact" since 2026-10-01; receipts record it only when it
    # differs from the default.
    default = FlashNextPolicy()
    assert default.fused_gdn_batch_verify == "row_exact"
    assert "fused_gdn_batch_verify" not in default.as_dict()
    assert default.environment()["MLX_QWEN4_FUSED_GDN_BATCH_VERIFY"] == "row_exact"
    off = FlashNextPolicy.from_mapping({"fused_gdn_batch_verify": "off"})
    assert off.as_dict()["fused_gdn_batch_verify"] == "off"
    assert "MLX_QWEN4_FUSED_GDN_BATCH_VERIFY" not in off.environment()
    with pytest.raises(ValueError, match="fused_gdn_batch_verify"):
        FlashNextPolicy.from_mapping({"fused_gdn_batch_verify": "on"})


@pytest.mark.parametrize("value", ["off", "row_exact"])
def test_policy_field_round_trips_through_receipts(value):
    policy = FlashNextPolicy.from_mapping({"fused_gdn_batch_verify": value})
    assert FlashNextPolicy.from_mapping(policy.as_dict()) == policy
    default = FlashNextPolicy.__dataclass_fields__["fused_gdn_batch_verify"].default
    assert ("fused_gdn_batch_verify" in policy.as_dict()) == (value != default)


# --------------------------------------------------------------- routing ----

def _gdn_layers(model):
    return [m for _, m in model.named_modules() if isinstance(m, qwen4_exp.GatedDeltaNet)]


def _ragged_cache(rows_spans, *, seed=0, kind="merged"):
    """A speculating cache with per-lane states and ``prepare``d lengths."""
    rows = len(rows_spans)
    mx.random.seed(seed)
    conv = (mx.random.normal((rows, 3, CONV_DIM)) * 0.5).astype(mx.bfloat16)
    state = mx.random.normal((rows, NUM_VALUE_HEADS, 128, 128)) * 0.05
    if kind == "merged":
        cache = ArraysCache(2)
        cache[0], cache[1] = conv, state
        cache.start_speculation()
    else:
        lanes = []
        for r in range(rows):
            lane = ArraysCache(2)
            lane[0], lane[1] = conv[r:r + 1], state[r:r + 1]
            lane.start_speculation()
            lanes.append(lane)
        cache = SegmentedBatchArraysCache(lanes)
    cache.prepare(lengths=list(rows_spans))
    return cache, conv, state


def test_default_off_keeps_the_existing_b1_refusal(production_gdn_qwen4):
    layer = _gdn_layers(production_gdn_qwen4)[0]
    assert layer.fused_gdn_batch_verify_mode == "off"
    layer.set_fused_gdn_verify_mode("fused")
    layer.set_fused_gdn_replay_rollback_mode("compact")
    qwen4_exp.qwen4_fused_gdn_stats(production_gdn_qwen4, reset=True)
    cache, _, _ = _ragged_cache([3, 3, 3])
    x = mx.zeros((3, 3, layer.hidden_size), mx.bfloat16)
    mx.eval(layer(x, mask=cache.make_mask(3), cache=cache))
    stats = qwen4_exp.qwen4_fused_gdn_stats(production_gdn_qwen4)
    assert "batch_verify" not in stats
    assert stats["replay_fallback_reasons"].get("batch of 3 rows", 0) == 1, stats


@pytest.mark.parametrize("spans", [[4, 4, 4], [4, 1, 3, 2]])
def test_row_exact_reaches_the_launch_gate_and_matches_stock_on_cpu(
    production_gdn_qwen4, spans
):
    """On the CPU the admitted batched verify falls back, so its outputs and
    rollback must be the stock chain's exactly; the counters record it."""
    layer = _gdn_layers(production_gdn_qwen4)[0]
    mx.random.seed(5)
    width = max(spans)
    x = (mx.random.normal((len(spans), width, layer.hidden_size)) * 0.5).astype(mx.bfloat16)
    outs = {}
    for mode in ("off", "row_exact"):
        layer.set_fused_gdn_batch_verify_mode(mode)
        qwen4_exp.qwen4_fused_gdn_stats(production_gdn_qwen4, reset=True)
        cache, _, _ = _ragged_cache(spans, seed=1)
        out = layer(x, mask=cache.make_mask(width), cache=cache)
        cache.finalize()
        cache.trim_ragged([max(0, s - 1) for s in spans])
        outs[mode] = (out, cache[0], cache[1])
        mx.eval(outs[mode])
    stats = qwen4_exp.qwen4_fused_gdn_stats(production_gdn_qwen4)
    layer.set_fused_gdn_batch_verify_mode("off")
    for p, q in zip(outs["off"], outs["row_exact"]):
        assert mx.array_equal(p, q).item()
    batch = stats["batch_verify"]
    assert batch["fallback_reasons"] == {ADMITTED_ON_CPU: 1}, batch
    assert batch["modes"] == ["row_exact"] and batch["calls"] == 0
    # A B>1 block no longer books the B=1 "batch of N rows" refusal.
    assert not any("rows" in r for r in stats["replay_fallback_reasons"])


def test_mode_setter_rejects_unknown_modes(production_gdn_qwen4):
    layer = _gdn_layers(production_gdn_qwen4)[0]
    with pytest.raises(ValueError, match="batch verify mode"):
        layer.set_fused_gdn_batch_verify_mode("multi_row")


# ---------------------------------------------- per-lane bookkeeping (fake) --

class _FakeKernels:
    """Stand-ins whose outputs encode lane and count, so every per-lane
    restore can be checked exactly on the CPU.

    The verify returns ``state + 100`` as the post-verify state and a tape;
    the reconstruct returns ``checkpoint + accepted[r]`` for lane ``r``.
    """

    def __init__(self):
        self.launches = []
        self.reconstructs = []

    def verify(self, qkv, z, b, a, conv_state, conv_weight, A_log, dt_bias,
               recurrent_state, norm_weight, norm_eps, row_steps, *, threadgroup_y):
        rows, steps = int(qkv.shape[0]), int(qkv.shape[1])
        self.launches.append((rows, steps, tuple(row_steps), threadgroup_y))
        ends = mx.array(list(row_steps), dtype=mx.int32)
        combined = mx.concatenate([conv_state, qkv], axis=1)
        conv_out = qwen4_exp._row_tail(combined, ends, 3)
        out = mx.zeros(z.shape, z.dtype)
        keys = mx.zeros((rows, steps - 1, 16, 128), qkv.dtype)
        corrections = mx.zeros((rows, steps - 1, NUM_VALUE_HEADS, 128), mx.float32)
        decay = mx.zeros((rows, steps - 1, NUM_VALUE_HEADS), mx.float32)
        return (out, conv_out, recurrent_state + 100, keys, corrections, decay)

    def reconstruct(self, state, keys, corrections, decay, accepted, *, threadgroup_y):
        assert isinstance(accepted, mx.array), "batched rollback must be per lane"
        counts = accepted.reshape(-1).astype(mx.int32)
        self.reconstructs.append(counts.tolist())
        return state + counts.astype(state.dtype).reshape(-1, 1, 1, 1)


@pytest.fixture
def fake_kernels(monkeypatch):
    fake = _FakeKernels()
    monkeypatch.setattr(qwen4_exp, "fused_gdn_runtime_supported", lambda: True)
    monkeypatch.setattr(qwen4_exp, "served_silu_refusal", lambda: None)
    monkeypatch.setattr(
        qwen4_exp, "probe_qwen4_fused_gdn_batch_verify", lambda *a, **k: 32
    )
    monkeypatch.setattr(qwen4_exp, "qwen4_fused_gdn_batch_replay_verify", fake.verify)
    monkeypatch.setattr(qwen4_exp, "qwen4_fused_gdn_reconstruct", fake.reconstruct)
    return fake


@pytest.mark.parametrize("kind", ["merged", "segmented"])
def test_partial_accept_restores_every_lane_from_its_own_record(
    production_gdn_qwen4, fake_kernels, kind
):
    layer = _gdn_layers(production_gdn_qwen4)[0]
    layer.set_fused_gdn_batch_verify_mode("row_exact")
    layer.set_fused_gdn_replay_rollback_mode("compact")
    spans = [5, 2, 4, 1]
    accepted = [3, 2, 1, 1]  # lane 1 and lane 3 keep everything
    try:
        qwen4_exp.qwen4_fused_gdn_stats(production_gdn_qwen4, reset=True)
        cache, conv, state = _ragged_cache(spans, seed=2, kind=kind)
        mx.random.seed(9)
        x = (mx.random.normal((4, 5, layer.hidden_size)) * 0.5).astype(mx.bfloat16)
        qkv = layer._input_projections(x)[0]
        out = layer(x, mask=cache.make_mask(5), cache=cache)
        mx.eval(out)
        assert out.shape == (4, 5, layer.hidden_size)
        assert fake_kernels.launches == [(4, 5, tuple(spans), 32)]
        cache.finalize()
        drops = [s - m for s, m in zip(spans, accepted)]
        cache.trim_ragged(drops)
        restored_conv, restored_state = cache[0], cache[1]
        mx.eval(restored_conv, restored_state)
        combined = mx.concatenate([conv, qkv], axis=1)
        for r, (span, m) in enumerate(zip(spans, accepted)):
            if m == span:  # untouched lane: the verify's own final state
                want_state = state[r] + 100
            else:
                want_state = state[r] + m
            assert mx.array_equal(restored_state[r], want_state).item(), r
            assert mx.array_equal(restored_conv[r], combined[r, m:m + 3]).item(), r
        stats = qwen4_exp.qwen4_fused_gdn_stats(production_gdn_qwen4)["batch_verify"]
        assert stats["calls"] == 1 and stats["rows"] == 4 and stats["tokens"] == 12
        assert stats["ragged_calls"] == 1 and stats["compact_calls"] == 1
        assert stats["fallbacks"] == 0 and stats["rollback_calls"] >= 1
        if kind == "merged":
            # One dispatch for every lane through per_row_fn.
            assert len(fake_kernels.reconstructs) == 1
            assert [fake_kernels.reconstructs[0][r] for r in (0, 2)] == [3, 1]
        else:
            # The segmented view slices one uniform-count rebuild per length.
            assert sorted({tuple(c) for c in fake_kernels.reconstructs}) == [
                (1, 1, 1, 1), (3, 3, 3, 3)
            ]
    finally:
        layer.set_fused_gdn_batch_verify_mode("off")


def test_lane_churn_reruns_admission_and_rebuilds_per_membership(
    production_gdn_qwen4, fake_kernels
):
    """Lanes leaving and joining between rounds: each round's segmented view
    is its own batch; the kernel sees each membership's own spans."""
    layer = _gdn_layers(production_gdn_qwen4)[0]
    layer.set_fused_gdn_batch_verify_mode("row_exact")
    layer.set_fused_gdn_replay_rollback_mode("compact")
    try:
        lanes = []
        for r in range(4):
            lane = ArraysCache(2)
            lane[0] = mx.full((1, 3, CONV_DIM), r, mx.bfloat16)
            lane[1] = mx.full((1, NUM_VALUE_HEADS, 128, 128), float(r))
            lane.start_speculation()
            lanes.append(lane)
        for members, spans in (([0, 1, 2], [3, 3, 2]), ([0, 2, 3], [2, 3, 3]), ([0, 3], [3, 1])):
            view = SegmentedBatchArraysCache([lanes[i] for i in members])
            view.prepare(lengths=spans)
            x = mx.zeros((len(members), max(spans), layer.hidden_size), mx.bfloat16)
            mx.eval(layer(x, mask=view.make_mask(max(spans)), cache=view))
            view.finalize()
            view.trim_ragged([s - 1 for s in spans])
        assert [l[:3] for l in fake_kernels.launches] == [
            (3, 3, (3, 3, 2)), (3, 3, (2, 3, 3)), (2, 3, (3, 1)),
        ]
        # Lane 1 left after round 1 and lane 3 joined in round 2: each lane's
        # state carries only its own rounds (+100 per verify, then +1 restore).
        assert lanes[1][1][0, 0, 0, 0].item() == 1 + 1
        assert lanes[3][1][0, 0, 0, 0].item() == 3 + 1 + 100
        assert lanes[0][1][0, 0, 0, 0].item() == 0 + 1 + 1 + 1
    finally:
        layer.set_fused_gdn_batch_verify_mode("off")


def test_snapshot_mode_restores_each_lane_from_its_snapshot(
    production_gdn_qwen4, monkeypatch
):
    layer = _gdn_layers(production_gdn_qwen4)[0]

    def fake(qkv, z, b, a, conv_state, conv_weight, A_log, dt_bias, state,
             norm_weight, eps, row_steps, *, threadgroup_y):
        rows, steps = int(qkv.shape[0]), int(qkv.shape[1])
        snaps = mx.stack([state + 10 * (p + 1) for p in range(steps - 1)], axis=1)
        conv_snaps = mx.stack(
            [conv_state + (p + 1) for p in range(steps - 1)], axis=1)
        return (mx.zeros(z.shape, z.dtype), conv_state + 50, state + 100, snaps, conv_snaps)

    monkeypatch.setattr(qwen4_exp, "fused_gdn_runtime_supported", lambda: True)
    monkeypatch.setattr(qwen4_exp, "served_silu_refusal", lambda: None)
    monkeypatch.setattr(qwen4_exp, "probe_qwen4_fused_gdn_batch_verify", lambda *a, **k: 32)
    monkeypatch.setattr(qwen4_exp, "qwen4_fused_gdn_batch_verify", fake)
    layer.set_fused_gdn_batch_verify_mode("row_exact")
    layer.set_fused_gdn_replay_rollback_mode("snapshots")
    try:
        spans, accepted = [4, 3, 4], [2, 3, 1]
        cache, conv, state = _ragged_cache(spans, seed=3)
        x = mx.zeros((3, 4, layer.hidden_size), mx.bfloat16)
        mx.eval(layer(x, mask=cache.make_mask(4), cache=cache))
        cache.finalize()
        cache.trim_ragged([s - m for s, m in zip(spans, accepted)])
        for r, (span, m) in enumerate(zip(spans, accepted)):
            want = state[r] + (100 if m == span else 10 * m)
            assert mx.array_equal(cache[1][r], want).item(), r
        assert layer.fused_gdn_batch_verify_compact_calls == 0
    finally:
        layer.set_fused_gdn_batch_verify_mode("off")
        layer.set_fused_gdn_replay_rollback_mode("compact")


def test_kill_switch_returns_b_gt_1_blocks_to_the_stock_chain(
    production_gdn_qwen4, fake_kernels
):
    layer = _gdn_layers(production_gdn_qwen4)[0]
    layer.set_fused_gdn_batch_verify_mode("off")
    cache, _, _ = _ragged_cache([3, 2], seed=4)
    x = mx.zeros((2, 3, layer.hidden_size), mx.bfloat16)
    mx.eval(layer(x, mask=cache.make_mask(3), cache=cache))
    assert fake_kernels.launches == []


def test_refusal_leaves_the_cache_untouched(production_gdn_qwen4, fake_kernels):
    """A counted refusal (here a leading pad) runs the stock chain, which
    alone advances the cache: the batched route never half-commits."""
    layer = _gdn_layers(production_gdn_qwen4)[0]
    layer.set_fused_gdn_batch_verify_mode("row_exact")
    try:
        cache = ArraysCache(2, left_padding=[1, 0])
        cache[0] = mx.zeros((2, 3, CONV_DIM), mx.bfloat16)
        cache[1] = mx.zeros((2, NUM_VALUE_HEADS, 128, 128))
        cache.start_speculation()
        x = mx.zeros((2, 3, layer.hidden_size), mx.bfloat16)
        mx.eval(layer(x, mask=cache.make_mask(3), cache=cache))
        assert fake_kernels.launches == []
        assert layer.fused_gdn_batch_verify_fallback_reasons == {
            "rollback geometry not describable": 1
        }
        assert cache.left_padding.tolist() == [-2, -3]
    finally:
        layer.set_fused_gdn_batch_verify_mode("off")


# ------------------------------------------------------------- row exact ----

def test_row_exact_route_counts_a_batched_verify_as_gdn_engaged(production_gdn_qwen4):
    from mlx2.runtime.models import qwen4_row_exact

    handle = qwen4_row_exact.install(production_gdn_qwen4)
    try:
        layers = [layer.linear_attn for layer in handle._gdn]
        assert layers
        before = handle._gdn_engaged()
        for layer in layers:
            layer.fused_gdn_batch_verify_calls += 1
        assert handle._gdn_engaged() - before == len(layers)
    finally:
        for layer in [layer.linear_attn for layer in handle._gdn]:
            layer.fused_gdn_batch_verify_calls = 0
        handle.remove()
        object.__delattr__(production_gdn_qwen4, "_mlx2_row_exact_verify")
