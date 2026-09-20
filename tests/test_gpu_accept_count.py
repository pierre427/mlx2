"""Device-count (GPU-resident accepted prefix) rollback for native MTP/GDN.

Metal kernels cannot run on CPU, so the dynamic reconstruct kernel is checked
here through its NumPy mirror and its dispatch contract; kernel-vs-template
bit equality runs in ``scripts/bench_gpu_accept_count.py --device gpu``.
"""
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime import verify_sync
from mlx2.runtime.models import qwen3_5, qwen4_exp
from mlx2.runtime.models import qwen4_fused_gdn_verify as fused
from mlx2.runtime.models.cache import ArraysCache
from scripts.bench_gpu_accept_count import (
    check_reference,
    ragged_case,
    reconstruct_dynamic_reference,
)
from scripts.bench_qwen4_gdn_replay_cpu import reconstruct_state

# --- dynamic reconstruct kernel: source + reference -------------------------


def test_dynamic_source_reads_device_count_and_keeps_template_arithmetic():
    source = fused._RECONSTRUCT_DYNAMIC_SOURCE
    assert "(uint)M" not in source
    assert "accepted[row]" in source
    assert "for (uint t = 0; t < steps; ++t)" in source
    # The per-step recurrence is textually the template kernel's.
    for line in ("st = st * decay;", "st = st + key * correction;"):
        assert source.count(line) == 1
        assert fused._RECONSTRUCT_SOURCE.count(line) == 1
    # Clamped to the tape, and zero keeps the checkpoint.
    assert "requested <= 0" in source and "(uint)SNAPS" in source


@pytest.mark.parametrize("width", [2, 3, 5, 8, 17])
def test_dynamic_reference_matches_template_for_every_prefix(width):
    initial, keys, corrections, decay = ragged_case(width, 2)
    for accepted in range(1, width):
        got = reconstruct_dynamic_reference(
            initial, keys, corrections, decay, np.array([accepted], np.int32)
        )
        for row in range(2):
            want = reconstruct_state(
                initial[row], keys[row], corrections[row], decay[row], accepted
            )
            assert np.array_equal(got[row], want)


@pytest.mark.parametrize("width", [3, 8, 17])
def test_dynamic_reference_ragged_rows_in_one_call(width):
    rows = 5
    initial, keys, corrections, decay = ragged_case(width, rows)
    accepted = np.array([0, 1, width - 1, (width - 1) // 2 + 1, width + 3], np.int32)
    got = reconstruct_dynamic_reference(initial, keys, corrections, decay, accepted)
    assert np.array_equal(got[0], initial[0])  # zero -> checkpoint
    for row in range(1, rows):
        steps = min(int(accepted[row]), width - 1)  # kernel clamps to the tape
        want = reconstruct_state(
            initial[row], keys[row], corrections[row], decay[row], steps
        )
        assert np.array_equal(got[row], want)


def test_cpu_reference_report_passes():
    report = check_reference((2, 3, 17))
    assert all(row["all_prefixes_bit_exact"] for row in report["results"])


class _FakeKernel:
    """Stand-in for the Metal kernel: records the launch, runs the mirror."""

    def __init__(self):
        self.calls = []

    def __call__(
        self, *, inputs, template, grid, threadgroup, output_shapes, output_dtypes
    ):
        self.calls.append(
            {"template": dict(template), "grid": grid, "inputs": inputs}
        )
        (state, keys, corrections, decay, accepted) = inputs
        out = reconstruct_dynamic_reference(
            np.asarray(state),
            np.asarray(keys.astype(mx.float32)),
            np.asarray(corrections),
            np.asarray(decay),
            np.asarray(accepted),
        )
        assert tuple(out.shape) == tuple(output_shapes[0])
        return [mx.array(out)]


def _tape(rows, width):
    (hk, hv, dk, dv) = (
        fused.NUM_KEY_HEADS,
        fused.NUM_VALUE_HEADS,
        fused.KEY_HEAD_DIM,
        fused.VALUE_HEAD_DIM,
    )
    initial, keys, corrections, decay = ragged_case(
        width, rows, hk=hk, hv=hv, dk=dk, dv=dv
    )
    return (
        mx.array(initial),
        mx.array(keys).astype(mx.bfloat16),
        mx.array(corrections),
        mx.array(decay),
    )


def test_array_count_dispatches_one_dynamic_launch_per_call():
    kernel = _FakeKernel()
    (state, keys, corrections, decay) = _tape(3, 5)
    with patch.object(fused, "_reconstruct_dynamic_kernel", return_value=kernel):
        out = fused.qwen4_fused_gdn_reconstruct(
            state, keys, corrections, decay, mx.array([0, 2, 4]), threadgroup_y=4
        )
    assert len(kernel.calls) == 1
    call = kernel.calls[0]
    assert call["template"]["SNAPS"] == 4 and "M" not in call["template"]
    assert call["grid"] == (32, 4, 3 * fused.NUM_VALUE_HEADS)
    assert call["inputs"][-1].dtype == mx.int32
    assert np.array_equal(np.asarray(out[0]), np.asarray(state[0]))
    # A scalar count broadcasts to every row.
    with patch.object(fused, "_reconstruct_dynamic_kernel", return_value=kernel):
        fused.qwen4_fused_gdn_reconstruct(
            state, keys, corrections, decay, mx.array(2, mx.uint8), threadgroup_y=4
        )
    assert kernel.calls[-1]["inputs"][-1].tolist() == [2, 2, 2]


@pytest.mark.parametrize(
    "accepted",
    [mx.array([1, 2]), mx.array([[1, 1, 1]]), mx.array([1.0, 1.0, 1.0])],
)
def test_array_count_shape_and_dtype_fail_closed(accepted):
    (state, keys, corrections, decay) = _tape(3, 4)
    with pytest.raises(ValueError):
        fused.qwen4_fused_gdn_reconstruct(
            state, keys, corrections, decay, accepted, threadgroup_y=4
        )


def test_int_count_keeps_template_path_and_validation():
    (state, keys, corrections, decay) = _tape(1, 4)
    with (
        patch.object(fused, "_reconstruct_dynamic_kernel") as dynamic,
        pytest.raises(ValueError),
    ):
        fused.qwen4_fused_gdn_reconstruct(
            state, keys, corrections, decay, 0, threadgroup_y=4
        )
    dynamic.assert_not_called()


def test_dynamic_probe_is_cached_separately_and_off_by_default():
    assert fused._PROBED_DYNAMIC_REPLAY_STEPS is not fused._PROBED_REPLAY_STEPS
    assert qwen4_exp._FUSED_GDN_DYNAMIC_ACCEPT is False
    assert qwen3_5._GDN_ARRAY_ACCEPT is False


# --- Qwen4 compact-replay closures ------------------------------------------


def _dynamic_closures(width, rows=1):
    kernel = _FakeKernel()
    holder = SimpleNamespace(
        conv_kernel_size=4,
        fused_gdn_replay_rollback_calls=0,
        fused_gdn_replay_dynamic_rollback_calls=0,
        fused_gdn_replay_rollback_tokens=0,
    )
    (state, keys, corrections, decay) = _tape(rows, width)
    rng = np.random.default_rng(width)
    conv = mx.array(rng.normal(size=(rows, 3, 24)).astype(np.float32))
    qkv = mx.array(rng.normal(size=(rows, width, 24)).astype(np.float32))
    (fn, per_row) = qwen4_exp.GatedDeltaNet._dynamic_replay_rollback(
        holder, conv, state, qkv, keys, corrections, decay, 4
    )
    return holder, kernel, fn, per_row, (conv, qkv, state, keys, corrections, decay)


@pytest.mark.parametrize("width", [2, 3, 8])
def test_qwen4_dynamic_fn_int_and_array_agree_with_template_reference(width):
    (holder, kernel, fn, per_row, tape) = _dynamic_closures(width)
    (conv, qkv, state, keys, corrections, decay) = tape
    combined = mx.concatenate([conv, qkv], axis=1)
    with patch.object(fused, "_reconstruct_dynamic_kernel", return_value=kernel):
        for m in range(1, width):
            want_state = reconstruct_state(
                np.asarray(state[0]),
                np.asarray(keys[0].astype(mx.float32)),
                np.asarray(corrections[0]),
                np.asarray(decay[0]),
                m,
            )
            for got in (fn(m), fn(mx.array(m)), per_row([m]), per_row(mx.array([m]))):
                assert np.array_equal(
                    np.asarray(got[0]), np.asarray(combined[:, m : m + 3])
                )
                assert np.array_equal(np.asarray(got[1][0]), want_state)
        with pytest.raises(ValueError):
            fn(width)  # a full acceptance never reconstructs
    assert holder.fused_gdn_replay_dynamic_rollback_calls == 4 * (width - 1)
    # Array counts are never read back, so they do not add host token counts.
    assert holder.fused_gdn_replay_rollback_tokens == 2 * sum(range(1, width))


def test_qwen4_dynamic_rows_are_ragged_in_one_launch():
    width, rows = 5, 3
    (_, kernel, _, per_row, tape) = _dynamic_closures(width, rows)
    (conv, qkv, *_rest) = tape
    combined = np.asarray(mx.concatenate([conv, qkv], axis=1))
    with patch.object(fused, "_reconstruct_dynamic_kernel", return_value=kernel):
        got = per_row([0, 2, 4])
    assert len(kernel.calls) == 1
    for row, m in enumerate([0, 2, 4]):
        assert np.array_equal(np.asarray(got[0][row]), combined[row, m : m + 3])


def test_qwen4_setter_validates_and_stats_stay_default_shaped():
    holder = SimpleNamespace(fused_gdn_dynamic_accept=False)
    qwen4_exp.GatedDeltaNet.set_fused_gdn_dynamic_accept(holder, True)
    assert holder.fused_gdn_dynamic_accept is True
    with pytest.raises(ValueError):
        qwen4_exp.GatedDeltaNet.set_fused_gdn_dynamic_accept(holder, 1)
    model = _tiny_qwen4()
    assert "replay_dynamic_rollback_calls" not in qwen4_exp.qwen4_fused_gdn_stats(model)
    for module in model.modules():
        if isinstance(module, qwen4_exp.GatedDeltaNet):
            module.set_fused_gdn_dynamic_accept(True)
    stats = qwen4_exp.qwen4_fused_gdn_stats(model)
    assert stats["replay_dynamic_rollback_calls"] == 0


# --- generic masked replay ---------------------------------------------------


def _tiny_qwen35_layer(dtype=mx.float32):
    args = qwen3_5.TextModelArgs(
        hidden_size=32,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
    )
    layer = qwen3_5.GatedDeltaNet(args)
    layer.set_dtype(dtype)
    mx.eval(layer.parameters())
    return layer


def _record(layer, rows, width, *, warm=True, seed=3):
    mx.random.seed(seed)
    dtype = layer.in_proj_qkv.weight.dtype
    cache = ArraysCache(2)
    if warm:
        layer(mx.random.normal((rows, 5, 32)).astype(dtype), cache=cache)
    cache.start_speculation()
    layer(mx.random.normal((rows, width, 32)).astype(dtype), cache=cache)
    return cache, cache._rollbacks[-1]


def _assert_entries_equal(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert np.array_equal(
            np.asarray(a.astype(mx.float32)), np.asarray(b.astype(mx.float32))
        )


def test_generic_default_stages_no_per_row_replay():
    layer = _tiny_qwen35_layer()
    (_, record) = _record(layer, 2, 4)
    assert record.per_row_fn is None


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("warm", [True, False])
def test_generic_masked_replay_equals_sliced_replay(monkeypatch, dtype, warm):
    monkeypatch.setattr(qwen3_5, "_GDN_ARRAY_ACCEPT", True)
    layer = _tiny_qwen35_layer(dtype)
    width, rows = 6, 3
    (_, record) = _record(layer, rows, width, warm=warm)
    assert record.per_row_fn is not None
    for m in range(1, width):
        sliced = record.fn(m)
        _assert_entries_equal(record.fn(mx.array(m)), sliced)
        _assert_entries_equal(record.per_row_fn([m] * rows), sliced)
        _assert_entries_equal(record.per_row_fn(mx.array([m] * rows)), sliced)
    # Zero accepted tokens is the pre-forward snapshot.
    zero = record.per_row_fn([0] * rows)
    _assert_entries_equal(zero[:1], record.snapshot[:1])
    if record.snapshot[1] is not None:
        _assert_entries_equal(zero[1:], record.snapshot[1:])
    # Ragged rows: every row matches its own sliced replay.
    lengths = [1, 4, 2]
    ragged = record.per_row_fn(mx.array(lengths))
    for row, m in enumerate(lengths):
        want = record.fn(m)
        _assert_entries_equal(
            [t[row : row + 1] for t in ragged], [t[row : row + 1] for t in want]
        )


def test_generic_trim_ragged_one_graph_matches_default(monkeypatch):
    drops = [3, 1, 4]
    finals = {}
    for flag in (False, True):
        monkeypatch.setattr(qwen3_5, "_GDN_ARRAY_ACCEPT", flag)
        mx.random.seed(17)
        layer = _tiny_qwen35_layer()
        (cache, _) = _record(layer, 3, 5)
        with patch.object(
            qwen3_5.GatedDeltaNet,
            "_masked_rollback",
            autospec=True,
            side_effect=qwen3_5.GatedDeltaNet._masked_rollback,
        ) as masked:
            cache.trim_ragged(drops)
        finals[flag] = list(cache.cache)
        assert masked.call_count == (1 if flag else 0)
    _assert_entries_equal(finals[True], finals[False])


# --- end-to-end self-MTP rounds ----------------------------------------------


def _tiny_qwen4():
    from test_batched_mtp import _tiny_qwen4_model

    mx.random.seed(41)
    return _tiny_qwen4_model()


def _snapshot(caches):
    out = []
    for cache in caches:
        state = cache.state
        stack = [state]
        while stack:
            item = stack.pop()
            if isinstance(item, mx.array):
                out.append(np.asarray(item.astype(mx.float32)))
            elif isinstance(item, (list, tuple)):
                stack.extend(reversed(item))
    return out


def test_qwen4_forced_ragged_round_matches_default(monkeypatch):
    from test_batched_mtp import _forced_cycle

    prompts = ([1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12])
    results = {}
    for flag in (False, True):
        monkeypatch.setattr(qwen3_5, "_GDN_ARRAY_ACCEPT", flag)
        model = _tiny_qwen4()
        detached = _forced_cycle(model, prompts, (0, 2))
        results[flag] = [_snapshot(lane.caches.target) for lane in detached]
    for got, want in zip(results[True], results[False]):
        assert len(got) == len(want)
        for a, b in zip(got, want):
            assert np.array_equal(a, b)


def test_greedy_round_has_one_hybrid_sync_with_array_accept(monkeypatch):
    from mlx2.runtime.hybrid_speculative import (
        attach_self_mtp_lanes,
        prepare_self_mtp_lane,
        propose_batched_self_mtp,
    )
    from mlx2.runtime.sample_utils import LaneRNG

    monkeypatch.setattr(qwen3_5, "_GDN_ARRAY_ACCEPT", True)
    monkeypatch.setenv("MLX_LM_SYNC_TRACE", "1")
    model = _tiny_qwen4()
    lanes = [
        prepare_self_mtp_lane(
            mx.array(prompt, mx.uint32),
            model,
            uid=uid,
            max_tokens=8,
            prompt_cache=None,
            mtp_state=None,
            lane_rng=LaneRNG(900 + uid),
            num_draft=2,
            sampling_temp=0.0,
            sampling_top_p=1.0,
            sampling_top_k=0,
            sampling_min_p=0.0,
            accept_rule="residual",
            logits_processors=[],
            prefill_step_size=8,
            share_qsa_indices=False,
        )[0]
        for uid, prompt in enumerate(([1, 7, 3, 9], [4, 4, 2, 8, 6]))
    ]
    batch = attach_self_mtp_lanes(model, None, lanes)
    state = verify_sync._state()
    before = len(state["rounds"])
    with patch.object(
        qwen3_5.GatedDeltaNet,
        "_masked_rollback",
        autospec=True,
        side_effect=qwen3_5.GatedDeltaNet._masked_rollback,
    ) as masked:
        proposal = propose_batched_self_mtp(model, batch)
    rounds = state["rounds"][before:]
    assert len(rounds) == 1
    hybrid = {
        site: count
        for site, count in rounds[0]["sites"].items()
        if site.startswith("hybrid.")
    }
    assert hybrid == {"hybrid.greedy.accept_boundary": 1}
    # The rollback used the one-graph replay once per GDN layer and added no
    # host read of its own.
    # This seed's random tiny model rejects drafts, so a rollback does run.
    assert any(drop > 0 for drop in proposal.target_drops)
    gdn_layers = sum(
        isinstance(m, qwen3_5.GatedDeltaNet) for m in model.modules()
    )
    assert masked.call_count == gdn_layers


# --- policy, metrics and qualification surfaces -----------------------------


def test_flash_next_policy_default_is_byte_identical_and_opt_in_sets_env():
    from mlx2.adapters.flash_next_policy import FlashNextPolicy

    default = FlashNextPolicy()
    assert "fused_gdn_dynamic_accept" not in default.as_dict()
    assert "MLX_QWEN4_FUSED_GDN_DYNAMIC_ACCEPT" not in default.environment()
    enabled = FlashNextPolicy.from_mapping({"fused_gdn_dynamic_accept": True})
    assert enabled.as_dict()["fused_gdn_dynamic_accept"] is True
    assert enabled.environment()["MLX_QWEN4_FUSED_GDN_DYNAMIC_ACCEPT"] == "1"
    with pytest.raises(ValueError):
        FlashNextPolicy.from_mapping({"fused_gdn_dynamic_accept": 1})


def test_dynamic_rollback_counter_is_exported_only_when_reported():
    from mlx2.prometheus import PrometheusBuilder, _add_execution

    rendered = {}
    for name, gdn in (
        ("off", {"replay_rollback_calls": 2}),
        ("on", {"replay_rollback_calls": 2, "replay_dynamic_rollback_calls": 2}),
    ):
        builder = PrometheusBuilder()
        _add_execution(builder, {"fused_gdn": gdn})
        rendered[name] = builder.render()
    assert "replay_dynamic_rollback_calls" not in rendered["off"]
    assert (
        'mlx2_fused_gdn_events_total{event="replay_dynamic_rollback_calls"} 2'
        in rendered["on"]
    )


def test_feature_check_only_when_dynamic_accept_selected():
    from mlx2.qualification import _route_feature_checks

    env = {
        "MLX_QWEN4_FUSED_GDN_VERIFY": "1",
        "MLX_QWEN4_FUSED_GDN_REPLAY_ROLLBACK": "1",
    }
    base = {"mtp": True, "environment": env}
    assert "feature_fused_gdn_dynamic_accept" not in _route_feature_checks(base)
    selected = {
        "mtp": True,
        "environment": {**env, "MLX_QWEN4_FUSED_GDN_DYNAMIC_ACCEPT": "1"},
    }
    assert "feature_fused_gdn_dynamic_accept" in _route_feature_checks(selected)
