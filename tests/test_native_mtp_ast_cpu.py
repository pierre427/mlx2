"""CPU evidence for AST over a target's own native MTP head (Flash-Next family).

The tiny qwen4_exp model carries an MTP head and a per-layer n-gram embedding
(PLE) layer, so these tests cover the pieces the real Flash-Next route needs:

- the progressive target-row executor on the row-exact verify route gives
  every verify row (logits and MTP seed feature) the bits of serial one-token
  decode at that position, PLE included, and commits the exact prefix state;
- a PLE contribution misaligned on verify rows is detected (falsifier);
- the native-MTP drafter's chain equals a hand-run head chain, and its
  complete paths reuse that chain's logits;
- end to end, the external engine (continuation pool + exact progressive
  verification + confidence-tree alternatives) emits ordinary greedy output.

The fused GDN verify kernel is Metal-only; as in test_qwen4_row_exact_verify
its engagement is injected so a CPU window can attest.  Several stages the
route relies on are row-invariant on Metal but not on the CPU backend, so the
bitwise checks run the trunk one token at a time inside the window
(``_SHAPE_STABLE_SHORT_FORWARD``): they establish the executor's plumbing
(rows, MTP seed features, PLE history, commit/abort state), not Metal
arithmetic.  The Metal bits are checked by the GPU smoke (B1 output equals
ordinary decode, every launch attested row-exact) and by the route's own
Metal gates (scripts/check_mtp_row_exact.py).
"""

from __future__ import annotations

import copy

import mlx.core as mx
import numpy as np
import pytest
from test_apc_hits_hybrid_gdn_self_mtp import tiny_qwen4_mtp
from test_qwen4_row_exact_verify import _fake_gdn_engaged
from test_standard_xpress_serving_cpu import drain, reference

from mlx2.adapters.proposal_path_sources import build_continuation_drafter
from mlx2.runtime.apc_v2 import APCv2
from mlx2.runtime.drafters.native_mtp import (
    NativeMTPDraftModel,
    inspect_native_mtp_policy,
)
from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows
from mlx2.runtime.models import qwen4_exp as Q
from mlx2.runtime.models.qwen4_row_exact import install
from mlx2.runtime.progressive_external_verify import (
    PROGRESSIVE_FIDELITY_EXACT,
    PROGRESSIVE_TARGET_PROTOCOL_VERSION,
    execute_progressive_target_rows,
)

PIN = "a" * 64


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _route(model):
    handle = install(model)
    handle.enable(True)
    _fake_gdn_engaged(handle)
    model.select_external_feature_source("mtp_seed")
    model.configure_progressive_external_row_exact(True)
    return handle


def _layers(model):
    return (len(model.language_model.model.layers) - 1,)


def _prefilled(model, prompt):
    cache = model.make_cache()
    mx.eval(model(mx.array([prompt]), cache=cache))
    return cache


def _serial(model, cache, tokens):
    """One-token decode rows: (logits, MTP seed) per position."""
    cache = copy.deepcopy(cache)
    logits, seeds = [], []
    for token in tokens:
        mixed, hyper = model.mtp_backbone(mx.array([[token]]), cache=cache)
        logits.append(model.logits(mixed))
        seeds.append(hyper)
    out = mx.concatenate(logits, axis=1), mx.concatenate(seeds, axis=1)
    mx.eval(out)
    return out, cache


def _verify(model, cache, tokens):
    owner = HybridVerifyRows([copy.deepcopy(cache)])
    transaction = owner.begin(lengths=[len(tokens)])
    logits, features, detail = execute_progressive_target_rows(
        model,
        mx.array([tokens]),
        transaction.caches,
        _layers(model),
        protocol=PROGRESSIVE_TARGET_PROTOCOL_VERSION,
        fidelity=PROGRESSIVE_FIDELITY_EXACT,
    )
    mx.eval(logits, features)
    return logits, features, detail, transaction


def test_feature_source_and_protocol_are_explicit_and_fail_closed():
    model, _ = tiny_qwen4_mtp()
    assert model.external_feature_source is None
    assert model.progressive_external_verify_protocol is None
    with pytest.raises(ValueError, match="row-exact verify route"):
        model.configure_progressive_external_row_exact(True)
    handle = install(model)
    handle.enable(True)
    with pytest.raises(ValueError, match="native MTP head feature source"):
        model.configure_progressive_external_row_exact(True)
    model.select_external_feature_source("mtp_seed")
    model.configure_progressive_external_row_exact(True)
    assert model.progressive_external_verify_protocol == PROGRESSIVE_TARGET_PROTOCOL_VERSION
    with pytest.raises(ValueError, match="last layer"):
        model.forward_with_taps(mx.array([[1]]), model.make_cache(), (0,))
    model.configure_progressive_external_row_exact(False)
    with pytest.raises(RuntimeError, match="not selected"):
        model.progressive_external_target_rows(mx.array([[1, 2]]), model.make_cache(), _layers(model))


def test_dflash2_feature_source_uses_completed_layer_hc_mean():
    model, _ = tiny_qwen4_mtp()
    trunk = model.language_model.model
    width = trunk.args.hidden_size
    streams = trunk.args.hc_count
    hidden = mx.arange(streams * width, dtype=mx.float32).reshape(1, 1, -1)
    model.select_external_feature_source("completed_layer_hc_mean")
    actual = trunk._external_capture(hidden)
    expected = hidden.reshape(1, 1, streams, width).mean(axis=-2)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert model.external_feature_width == width
    assert model.external_feature_convention == "completed_layer_hc_mean"
    assert model.external_execution_receipt["feature_convention"] == (
        "completed_layer_hc_mean"
    )


def test_native_mtp_feature_source_receipt_names_its_seed_law():
    model, _ = tiny_qwen4_mtp()
    model.select_external_feature_source("mtp_seed")
    assert model.external_feature_convention == "native_mtp_scheme_a_hc_seed"
    assert model.external_execution_receipt["feature_convention"] == (
        "native_mtp_scheme_a_hc_seed"
    )


@pytest.fixture
def per_token_trunk(monkeypatch):
    monkeypatch.setattr(Q, "_SHAPE_STABLE_SHORT_FORWARD", True)


@pytest.mark.parametrize("width", [2, 3, 4])
def test_verify_rows_and_seeds_are_serial_decode_bits_with_ple(width, per_token_trunk):
    model, _ = tiny_qwen4_mtp()
    handle = _route(model)
    prompt = [1, 7, 3, 9, 2, 8, 4, 6, 5, 11]
    cache = _prefilled(model, prompt)
    tokens = [13, 2, 40, 7][:width]
    (expected_logits, expected_seeds), _ = _serial(model, cache, tokens)
    logits, features, detail, transaction = _verify(model, cache, tokens)
    assert detail["row_exact"] is True
    assert detail["execution"] == "adapter_owned"
    np.testing.assert_array_equal(np.asarray(logits), np.asarray(expected_logits))
    np.testing.assert_array_equal(np.asarray(features), np.asarray(expected_seeds))
    assert handle.status()["windows_row_exact"] >= 1
    # The PLE layer sits on the GDN cache that carries its n-gram history.
    assert any(layer.ple is not None for layer in model.language_model.model.layers)
    transaction.abort()


@pytest.mark.parametrize("kept", [1, 2, 3])
def test_partial_commit_state_equals_that_many_decode_steps(kept, per_token_trunk):
    model, _ = tiny_qwen4_mtp()
    _route(model)
    prompt = [1, 7, 3, 9, 2, 8, 4, 6, 5, 11]
    cache = _prefilled(model, prompt)
    tokens = [13, 2, 40, 7]
    _, _, _, transaction = _verify(model, cache, tokens)
    (committed,) = transaction.commit(accepted_lengths=[kept])
    (_, _), serial_cache = _serial(model, cache, tokens[:kept])
    probe = [21]
    (left, left_seed), _ = _serial(model, committed, probe)
    (right, right_seed), _ = _serial(model, serial_cache, probe)
    np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    np.testing.assert_array_equal(np.asarray(left_seed), np.asarray(right_seed))


def test_skipped_ple_on_verify_rows_is_detected(monkeypatch, per_token_trunk):
    """Falsifier: a PLE contribution dropped on verify rows (inside the
    row-exact window) must make the executor's rows differ from decode."""
    from mlx2.runtime import row_exact_verify as REV

    model, _ = tiny_qwen4_mtp()
    _route(model)
    prompt = [1, 7, 3, 9, 2, 8, 4, 6, 5, 11]
    cache = _prefilled(model, prompt)
    tokens = [13, 2, 40, 7]
    (expected_logits, _), _ = _serial(model, cache, tokens)
    logits, _, _, transaction = _verify(model, cache, tokens)
    transaction.abort()
    np.testing.assert_array_equal(np.asarray(logits), np.asarray(expected_logits))
    original = Q.PLELayer.__call__

    def skipped(self, x, *args, **kwargs):
        out = original(self, x, *args, **kwargs)
        return mx.zeros_like(out) if REV.current() is not None else out

    monkeypatch.setattr(Q.PLELayer, "__call__", skipped)
    logits, _, _, transaction = _verify(model, cache, tokens)
    transaction.abort()
    assert not np.array_equal(np.asarray(logits), np.asarray(expected_logits))


def test_shifted_ple_tokens_on_verify_rows_are_detected(monkeypatch):
    """Falsifier on the wide (one-launch) path: PLE reading the wrong token
    for verify rows changes their logits beyond CPU rounding noise."""
    model, _ = tiny_qwen4_mtp()
    _route(model)
    prompt = [1, 7, 3, 9, 2, 8, 4, 6, 5, 11]
    cache = _prefilled(model, prompt)
    tokens = [13, 2, 40, 7]
    (expected_logits, _), _ = _serial(model, cache, tokens)
    logits, _, _, transaction = _verify(model, cache, tokens)
    transaction.abort()
    np.testing.assert_allclose(np.asarray(logits), np.asarray(expected_logits), atol=1e-5)
    original = Q.PLELayer.__call__

    def shifted(self, x, input_ids, *args, **kwargs):
        if input_ids is not None and input_ids.shape[1] > 1:
            input_ids = mx.concatenate([input_ids[:, :1], input_ids[:, :-1]], axis=1)
        return original(self, x, input_ids, *args, **kwargs)

    monkeypatch.setattr(Q.PLELayer, "__call__", shifted)
    logits, _, _, transaction = _verify(model, cache, tokens)
    transaction.abort()
    assert np.abs(np.asarray(logits) - np.asarray(expected_logits)).max() > 1e-3


def test_batched_one_token_launch_is_not_attested():
    model, _ = tiny_qwen4_mtp()
    _route(model)
    rows = [_prefilled(model, [1, 2, 3, 4]), _prefilled(model, [5, 6, 7, 8])]
    owner = HybridVerifyRows([copy.deepcopy(row) for row in rows])
    transaction = owner.begin(lengths=[1, 1])
    with pytest.raises(RuntimeError, match="fell off its exact path"):
        execute_progressive_target_rows(
            model,
            mx.array([[9], [10]]),
            transaction.caches,
            _layers(model),
            protocol=PROGRESSIVE_TARGET_PROTOCOL_VERSION,
            fidelity=PROGRESSIVE_FIDELITY_EXACT,
        )
    transaction.abort()
    assert model.progressive_external_lane_exact_min_width == 2


def _drafter(model, num_draft=3):
    return NativeMTPDraftModel(model, num_draft=num_draft, revision=PIN)


def test_drafter_chain_equals_a_hand_run_head_chain():
    model, _ = tiny_qwen4_mtp()
    model.select_external_feature_source("mtp_seed")
    draft = _drafter(model)
    prompt = [1, 7, 3, 9, 2, 8]
    target = model.make_cache()
    seeds = model.prefill_body(mx.array([prompt]), target, _layers(model))
    anchor = 5
    pairing = [prompt[1:] + [anchor]]
    tokens, laws = draft.draft_distributions(
        [anchor], seeds, draft.make_cache(), 3, [None], [0.0],
        context_tokens=pairing, processor_histories=[prompt],
    )
    head = model.make_mtp_cache()
    logits, seed = model.mtp_step(seeds, mx.array(pairing, mx.uint32), head)
    expected = []
    top, seed = logits[:, -1], seed[:, -1:]
    for position in range(3):
        if position:
            top, seed = model.mtp_step(seed, mx.array([[expected[-1]]], mx.uint32), head)
            top = top[:, -1]
        expected.append(int(mx.argmax(top[0]).item()))
    assert tokens == [expected]
    assert [law.token for law in laws[0]] == expected
    from mlx2.runtime.proposal_providers import ContinuationContext

    paths = draft.propose_complete_paths(
        ContinuationContext(tuple(prompt), anchor, 3, None, None), 15
    )
    assert paths[0].tokens == tuple(expected)
    assert all(1 <= len(path.tokens) <= 3 for path in paths)
    assert len({path.tokens for path in paths}) == len(paths)
    assert draft.stats["complete_path_stash_misses"] == 0
    assert draft.propose_complete_paths(
        ContinuationContext(tuple(prompt), anchor + 1, 3, None, None), 15
    ) == []


def test_policy_names_the_head_by_the_target_artifact(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    record = inspect_native_mtp_policy(
        {"draft_model": str(target), "num_draft": 4}, target, "f" * 64
    )
    assert record["args"].block_size == 5 and record["args"].native_mtp_head
    with pytest.raises(ValueError, match="native MTP head"):
        inspect_native_mtp_policy({"draft_model": str(tmp_path)}, target, "f" * 64)


def _ast_generator(
    model,
    wrapper,
    *,
    lanes=1,
    num_draft=4,
    tile=2,
    tile_policy=None,
    stage_row_telemetry=False,
    ast_memo=None,
):
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    extra = (
        {
            "progressive_cross_session_pack_rows": 4,
            "progressive_cross_session_pack_mode": "ordered_live_frontier",
            "multilane_draft_cap": 4,
        }
        if lanes > 1
        else {}
    )
    return ExternalDraftBatchGenerator(
        model,
        draft_model=wrapper,
        binding="native-mtp-ast-cpu",
        completion_batch_size=lanes,
        num_draft=num_draft,
        prefill_step_size=64,
        ready_drain="all",
        continuation_pool=wrapper.policy,
        continuation_pool_ast_bridge=True,
        progressive_verification_tile=tile,
        progressive_verification_tile_policy=tile_policy,
        progressive_terminal_input_widths=(
            model.progressive_external_exact_terminal_input_widths
            if tile_policy == "auto"
            else None
        ),
        progressive_alternative_branches=4,
        progressive_alternative_strategy="confidence_tree",
        progressive_alternative_depth_policy="confidence_tail",
        ast_stage_row_telemetry=stage_row_telemetry,
        ast_memo=ast_memo,
        **extra,
    )


def _wrapper(model, draft, *, planner=False):
    return build_continuation_drafter(
        model,
        draft,
        {"sources": ["external"], "limit": 15, "ast_planner": planner},
        target_revision=PIN,
        draft_revision="b" * 64,
        tokenizer_revision=PIN,
        session_revision="c" * 64,
    )


def test_ast_over_native_mtp_matches_ordinary_greedy_b1():
    model, _ = tiny_qwen4_mtp()
    handle = _route(model)
    draft = _drafter(model, num_draft=4)
    wrapper = _wrapper(model, draft)
    batch = _ast_generator(model, wrapper)
    prompt = [1, 7, 3, 9, 2, 8, 4, 6, 5, 11, 13, 2]
    batch.insert([prompt], max_tokens=[16], sampling_configs=[{"sampling_temp": 0}])
    output, final = drain(batch)
    assert output[0] == reference(model, prompt, 16)
    receipt = final[0].speculative_receipt
    assert receipt["external_rounds"] > 0 and receipt["proposed"] > 0
    progressive = receipt["progressive_verification"]
    assert progressive["observed_used"] and progressive["fidelity"] == "exact"
    executor = progressive["target_row_executor"]
    assert executor["row_exact_launches"] > 0 and executor["nonexact_launches"] == 0
    assert sum(executor["adapter_route_hist"].values()) == executor["launches"]
    layout = progressive["last_ragged_layout"]
    capability = layout["execution_capability"]
    # This direct tiny fixture has no device/build execution profile; semantic
    # discovery must attach its evidence but refuse to advertise availability.
    assert capability["available"] is False
    assert capability["discovery"] == "unknown_fail_closed"
    assert capability["supported_layout_classes"] == ["padded_rectangular"]
    memo = layout["memo_identity_material"]
    assert memo["capability_manifest_sha256"] == capability["manifest_sha256"]
    assert memo["cohort_geometry_sha256"] == layout["geometry_sha256"]
    assert len(memo["owner_semantic_contract_sha256"]) == 64
    assert memo["authoritative_cache"] == "APCv2"
    assert memo["memo_implemented"] is True
    assert memo["memo_enabled"] is False
    assert memo["memo_selected_for_execution"] is False
    assert memo["memo_observed_used"] is False
    assert memo["memo_outcome"] is None
    assert "stage_rows" not in executor["last"]
    geometry = progressive["ordered_alternatives"]["geometry_totals"]
    assert progressive["ordered_alternatives"]["winner_contract"] == (
        "exact_max_visited_committed_prefix_then_full_rank"
    )
    assert geometry["verified_branches"] > 0
    assert geometry["unique_target_rows"] > 0
    assert (
        geometry["shared_prefix_target_rows"]
        + geometry["branch_suffix_target_rows"]
        == geometry["unique_target_rows"]
    )
    assert sum(geometry["active_branch_hist"].values()) == progressive[
        "ordered_alternatives"
    ]["rounds"]
    assert sum(geometry["shared_prefix_token_hist"].values()) == progressive[
        "ordered_alternatives"
    ]["rounds"]
    assert geometry["counterfactual_partial_uplift_rounds"] >= 0
    assert geometry["counterfactual_partial_uplift_tokens"] >= 0
    assert sum(geometry["provider_rank_verified_hist"].values()) == geometry[
        "verified_branches"
    ]
    assert set(geometry["provider_rank_verified_hist"]) == set(
        geometry["provider_rank_committed_inputs"]
    )
    assert set(geometry["provider_rank_verified_hist"]) == set(
        geometry["provider_rank_full_accept_hist"]
    )
    assert sum(geometry["alternative_divergence_position_hist"].values()) == (
        geometry["verified_branches"]
        - progressive["ordered_alternatives"]["rounds"]
    )
    stats = batch.scheduler_stats
    assert stats.get("external_progressive_alternative_rounds", 0) > 0
    assert stats["external_progressive_alternative_unique_target_rows"] > 0
    assert draft.stats["complete_path_calls"] > 0
    assert draft.stats["complete_path_stash_misses"] == 0
    assert handle.status()["windows_not_exact"] == 0


def test_ast_memo_reuses_static_geometry_without_selecting_execution():
    model, _ = tiny_qwen4_mtp()
    _route(model)
    draft = _drafter(model, num_draft=4)
    wrapper = _wrapper(model, draft)
    apc = APCv2(
        layout_name="test-layout",
        max_size=2,
        max_bytes=1 << 20,
        ast_memo_enabled=True,
        ast_memo_debug=True,
    )
    batch = _ast_generator(model, wrapper, ast_memo=apc.ast_memo_handle())
    prompt = [1, 7, 3, 9, 2, 8, 4, 6, 5, 11, 13, 2]
    try:
        batch.insert(
            [prompt],
            max_tokens=[16],
            sampling_configs=[{"sampling_temp": 0}],
            ast_memo_layout_revisions=["d" * 64],
        )
        output, final = drain(batch)
        assert output[0] == reference(model, prompt, 16)
        memo = final[0].speculative_receipt["progressive_verification"][
            "last_ragged_layout"
        ]["memo_identity_material"]
        assert memo["memo_enabled"] is True
        assert memo["memo_observed_used"] is True
        assert memo["memo_outcome"] in {"miss", "hit"}
        assert memo["memo_selected_for_execution"] is False
        snapshot = apc.ast_memo_receipt()
        assert snapshot["executor_entries"] >= 1
        assert snapshot["stats"]["stores"] >= 1
        assert snapshot["stats"]["hits"] >= 1
    finally:
        batch.close()
        apc.close(release_memory=False)


@pytest.mark.parametrize("memo_enabled", [False, True])
def test_capability_discovery_is_cached_per_owner_contract(monkeypatch, memo_enabled):
    from mlx2.runtime import ast_execution_capability

    model, _ = tiny_qwen4_mtp()
    _route(model)
    draft = _drafter(model, num_draft=4)
    wrapper = _wrapper(model, draft)
    apc = (
        APCv2(
            layout_name="test-layout",
            max_size=2,
            max_bytes=1 << 20,
            ast_memo_enabled=True,
        )
        if memo_enabled
        else None
    )
    batch = _ast_generator(
        model,
        wrapper,
        ast_memo=None if apc is None else apc.ast_memo_handle(),
    )
    original = ast_execution_capability.query_ast_execution_capability
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        ast_execution_capability, "query_ast_execution_capability", counted
    )
    prompt = [1, 7, 3, 9, 2, 8, 4, 6, 5, 11, 13, 2]
    try:
        batch.insert(
            [prompt],
            max_tokens=[16],
            sampling_configs=[{"sampling_temp": 0}],
            **(
                {"ast_memo_layout_revisions": ["d" * 64]}
                if memo_enabled
                else {}
            ),
        )
        output, final = drain(batch)
        assert output[0] == reference(model, prompt, 16)
        assert final[0].speculative_receipt["external_rounds"] > 1
        assert len(calls) == 1
    finally:
        batch.close()
        if apc is not None:
            apc.close(release_memory=False)


@pytest.mark.parametrize("named", [False, True])
def test_failed_cohort_insert_does_not_leak_ast_memo_scope(named):
    model, _ = tiny_qwen4_mtp()
    _route(model)
    draft = _drafter(model, num_draft=4)
    wrapper = _wrapper(model, draft)
    apc = APCv2(
        layout_name="test-layout",
        max_size=2,
        max_bytes=1 << 20,
        ast_memo_enabled=True,
    )
    batch = _ast_generator(
        model, wrapper, lanes=2, ast_memo=apc.ast_memo_handle()
    )
    try:
        with pytest.raises(ValueError, match="prompt tokens"):
            batch.insert(
                [[1, 2, 3], []],
                max_tokens=[4, 4],
                sampling_configs=[{"sampling_temp": 0}, {"sampling_temp": 0}],
                ast_memo_layout_revisions=["d" * 64, "d" * 64],
                **(
                    {"session_keys": ["e" * 64, "f" * 64]}
                    if named
                    else {}
                ),
            )
        assert apc.ast_memo_receipt()["scopes"] == 0
    finally:
        batch.close()
        apc.close(release_memory=False)


def test_ast_stage_row_telemetry_is_opt_in_and_excluded_from_feedback(monkeypatch):
    model, _ = tiny_qwen4_mtp()
    _route(model)
    draft = _drafter(model, num_draft=4)
    wrapper = _wrapper(model, draft, planner=True)
    batch = _ast_generator(
        model, wrapper, stage_row_telemetry=True
    )

    def forbidden_feedback(*_args, **_kwargs):
        pytest.fail("diagnostic AST telemetry must not update planner feedback")

    monkeypatch.setattr(
        batch.ast_execution_planner, "observe_launch", forbidden_feedback
    )
    prompt = [1, 7, 3, 9, 2, 8, 4, 6, 5, 11, 13, 2]
    batch.insert([prompt], max_tokens=[8], sampling_configs=[{"sampling_temp": 0}])
    output, final = drain(batch)
    assert output[0] == reference(model, prompt, 8)
    executor = final[0].speculative_receipt["progressive_verification"][
        "target_row_executor"
    ]
    target_rows = executor["last"]["stage_rows"]["stages"]["target.forward"]
    assert target_rows["logical_rows"] <= target_rows["physical_rows"]
    assert target_rows["padding_rows"] == (
        target_rows["physical_rows"] - target_rows["logical_rows"]
    )
    assert batch.ast_execution_planner.receipt()["cost_samples"] == 0


def test_auto_tile_runtime_receipt_names_policy_resolution_and_cohort():
    model, _ = tiny_qwen4_mtp()
    _route(model)
    draft = _drafter(model, num_draft=4)
    wrapper = _wrapper(model, draft)
    batch = _ast_generator(
        model, wrapper, num_draft=4, tile=3, tile_policy="auto"
    )
    prompt = [1, 7, 3, 9, 2, 8, 4, 6]
    batch.insert([prompt], max_tokens=[8], sampling_configs=[{"sampling_temp": 0}])
    output, final = drain(batch)
    assert output[0] == reference(model, prompt, 8)
    receipt = final[0].speculative_receipt["progressive_verification"]
    assert receipt["verification_tile"] == 3
    assert receipt["verification_tile_policy"] == "auto"
    assert receipt["verification_tile_resolution_max_lanes"] == 1


def test_ast_over_native_mtp_two_lanes_each_match_ordinary():
    model, _ = tiny_qwen4_mtp()
    _route(model)
    draft = _drafter(model, num_draft=4)
    wrapper = _wrapper(model, draft)
    batch = _ast_generator(model, wrapper, lanes=2)
    prompts = [[1, 7, 3, 9, 2, 8, 4, 6], [5, 11, 13, 2, 30, 31, 4]]
    batch.insert(prompts, max_tokens=[12, 12],
                 sampling_configs=[{"sampling_temp": 0}, {"sampling_temp": 0}])
    output, final = drain(batch)
    for uid, prompt in enumerate(prompts):
        assert output[uid] == reference(model, prompt, 12)
        assert final[uid].speculative_receipt["external_rounds"] > 0


def test_flash_next_policy_split_routes_external_keys_to_the_native_head():
    from mlx2.adapters.flash_next import split_native_mtp_external_policy

    assert split_native_mtp_external_policy(None) == (None, {})
    native = {"num_draft": 2, "row_exact_verify": True}
    assert split_native_mtp_external_policy(native) == (native, {})
    flash, external = split_native_mtp_external_policy(
        {"draft_model": "/m", "num_draft": 4, "row_exact_verify": True,
         "progressive_verification_tile": 2,
         "cost_per_commit_handoff": {"enabled": True}}
    )
    assert flash == {"row_exact_verify": True}
    assert external == {"draft_model": "/m", "num_draft": 4,
                        "progressive_verification_tile": 2,
                        "cost_per_commit_handoff": {"enabled": True}}
    flash, external = split_native_mtp_external_policy(
        {"draft_model": "/d", "draft_revision": "d" * 64,
         "target_revision": "t" * 64, "row_exact_verify": True}
    )
    assert flash == {"row_exact_verify": True}
    assert external["draft_revision"] == "d" * 64
    assert external["target_revision"] == "t" * 64
    flash, external = split_native_mtp_external_policy(
        {
            "draft_model": "/d",
            "row_exact_verify": True,
            "progressive_ast_rootless_fallback": False,
        }
    )
    assert flash == {"row_exact_verify": True}
    assert external["progressive_ast_rootless_fallback"] is False
    with pytest.raises(ValueError, match="require draft_model"):
        split_native_mtp_external_policy({"progressive_verification_tile": 2})


def _bound_adapter(tmp_path, model, **policy):
    from mlx2.adapters.flash_next import FlashNextAdapter
    from mlx2.adapters.flash_next_policy import FlashNextPolicy

    adapter = object.__new__(FlashNextAdapter)
    adapter.model = model
    adapter.policy = FlashNextPolicy(row_exact_verify=policy.pop("row_exact", True))
    adapter.identity = {"fingerprint": "f" * 64}
    adapter.layout = "qwen4-exp-layer-segments-v1"
    adapter.external_policy = {}
    adapter.draft_model = None
    value = {
        "draft_model": str(tmp_path),
        "num_draft": 4,
        "continuation_pool": {"sources": ["external"], "limit": 15},
        "progressive_verification_tile": 2,
        "progressive_alternative_branches": 4,
        "progressive_alternative_strategy": "confidence_tree",
        "progressive_alternative_depth_policy": "confidence_tail",
        **policy,
    }
    adapter._bind_native_mtp_external(tmp_path, value)
    return adapter


def test_flash_next_adapter_binds_native_mtp_ast_and_serves_ordinary_greedy(tmp_path):
    from mlx2.contracts import Capability

    model, _ = tiny_qwen4_mtp()
    handle = install(model)
    handle.enable(True)
    _fake_gdn_engaged(handle)
    adapter = _bound_adapter(tmp_path, model)
    assert Capability.EXTERNAL_DRAFT in adapter.descriptor.capabilities
    assert adapter.profile_name(False) == "flash-next-apcv2-native-mtp-ast"
    metadata = adapter.descriptor.metadata["progressive_verification"]
    assert metadata["target_protocol"] == PROGRESSIVE_TARGET_PROTOCOL_VERSION
    assert metadata["qualified"] is False
    config = adapter.execution_config(max_lanes=1, prefill_step=64)
    assert config["backend"] == "external_draft"
    assert config["progressive_verification_tile"] == 2
    batch = adapter.create_external_batch(
        completion_batch_size=1, prefill_step_size=64, ready_drain="all"
    )
    prompt = [1, 7, 3, 9, 2, 8, 4, 6, 5, 11, 13, 2]
    batch.insert([prompt], max_tokens=[14], sampling_configs=[{"sampling_temp": 0}])
    output, final = drain(batch)
    assert output[0] == reference(model, prompt, 14)
    receipt = final[0].speculative_receipt
    assert receipt["kind"] == "external_native_mtp"
    assert receipt["progressive_verification"]["target_row_executor"]["row_exact_launches"] > 0


def test_flash_next_native_mtp_route_refuses_unproven_shapes(tmp_path):
    model, _ = tiny_qwen4_mtp()
    handle = install(model)
    handle.enable(True)
    with pytest.raises(ValueError, match="row_exact_verify"):
        _bound_adapter(tmp_path, model, row_exact=False)
    with pytest.raises(ValueError, match="exact only"):
        _bound_adapter(tmp_path, model, progressive_verification_fidelity="near_tie_bounded")
    with pytest.raises(ValueError, match="target artifact itself"):
        _bound_adapter(tmp_path, model, draft_model=str(tmp_path / "other"))


def test_flash_next_refuses_too_narrow_exact_multilane_tile_before_serving(tmp_path):
    model, _ = tiny_qwen4_mtp()
    handle = install(model)
    handle.enable(True)
    _fake_gdn_engaged(handle)
    adapter = _bound_adapter(
        tmp_path,
        model,
        num_draft=3,
        progressive_verification_tile=1,
        progressive_multilane_draft_cap=3,
    )

    assert adapter.execution_config(
        max_lanes=1, prefill_step=64
    )["progressive_verification_tile"] == 1
    with pytest.raises(
        ValueError,
        match="tile is narrower than the target's declared multi-lane exact width",
    ):
        adapter.execution_config(max_lanes=4, prefill_step=64)


def test_flash_next_auto_tile_refuses_k2_at_multilane_minimum_width(tmp_path):
    model, _ = tiny_qwen4_mtp()
    handle = install(model)
    handle.enable(True)
    _fake_gdn_engaged(handle)
    adapter = _bound_adapter(
        tmp_path,
        model,
        num_draft=2,
        progressive_verification_tile="auto",
        progressive_multilane_draft_cap=2,
    )

    config = adapter.execution_config(max_lanes=1, prefill_step=64)
    assert config["progressive_verification_tile"] == 1
    assert config["progressive_verification_tile_policy"] == "auto"
    with pytest.raises(
        ValueError,
        match="has no target-declared exact tile for the proposal/cohort geometry",
    ):
        adapter.execution_config(max_lanes=4, prefill_step=64)


def test_engine_memory_reservation_reads_the_head_geometry():
    """The serving engine passes memory_headroom, which makes admission read
    the drafter's KV geometry (a GPU smoke found it missing)."""
    model, _ = tiny_qwen4_mtp()
    _route(model)
    draft = _drafter(model, num_draft=4)
    wrapper = _wrapper(model, draft)
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    batch = ExternalDraftBatchGenerator(
        model, draft_model=wrapper, binding="native-mtp-ast-headroom",
        completion_batch_size=1, num_draft=4, prefill_step_size=64,
        ready_drain="all", memory_headroom=lambda: 1 << 40,
        continuation_pool=wrapper.policy, continuation_pool_ast_bridge=True,
        progressive_verification_tile=2, progressive_alternative_branches=4,
        progressive_alternative_strategy="confidence_tree",
        progressive_alternative_depth_policy="confidence_tail",
    )
    prompt = [1, 7, 3, 9, 2, 8, 4, 6]
    batch.insert([prompt], max_tokens=[8], sampling_configs=[{"sampling_temp": 0}])
    output, _ = drain(batch)
    assert output[0] == reference(model, prompt, 8)
    assert draft.config.num_key_value_heads > 0 and draft.config.head_dim > 0


def test_serving_engine_drives_the_native_mtp_ast_route(tmp_path, monkeypatch):
    """The real ServingEngine on the external route with the bound adapter:
    greedy output equals ordinary, the warm repeat hits APCv2 and still
    equals it, and the receipt names the native-MTP drafter."""
    from test_apc_hits_hybrid_gdn_self_mtp import Parser, make_adapter, run

    from mlx2 import memory, serving
    from mlx2.runtime import os_memory
    from mlx2.serving import ServingEngine

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "src"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    model, vocab = tiny_qwen4_mtp()
    handle = install(model)
    handle.enable(True)
    _fake_gdn_engaged(handle)
    adapter = _bound_adapter(tmp_path, model)
    # The real class guards APCv2 headroom with the file-backed PLE profile's
    # memory model, which a tiny CPU model does not have.
    adapter.__class__ = type(
        "TinyFlashNext", (type(adapter),), {"apc_cache_headroom_guard": False}
    )
    base = make_adapter(model, vocab)
    adapter.tokenizer = base.tokenizer
    adapter.max_context = 512
    adapter.environment = {}
    adapter.prompt_tokens = lambda request: list(request["tokens"])
    adapter.output_parser = lambda _request: Parser()
    adapter.diagnostics = dict
    adapter.close = lambda: None
    from mlx2.adapters.flash_next import qsa_indexer_geometry

    text_config = dict(model.args.text_config)
    adapter._attention_topology_config = text_config
    adapter.qsa_indexer = qsa_indexer_geometry(text_config)
    engine = ServingEngine(
        "tiny", adapter_factory=lambda _path: adapter, qualification_mode=True,
        mtp=False, max_lanes=1, prefill_step=16,
    )
    try:
        assert engine.ready.wait(60), engine.error
        prompt = [(7 * i + 3) % (vocab - 2) + 1 for i in range(40)]
        cold, receipt, _ = run(engine, prompt, max_tokens=10)
        warm, _, job = run(engine, prompt, max_tokens=10)
    finally:
        engine.close()
    assert cold == reference(model, prompt, 10)
    assert warm == cold
    assert int(job.cached_tokens or 0) > 0
    speculation = receipt.get("speculation") or {}
    assert speculation.get("kind") == "external_native_mtp"


def test_standard_verify_fallback_runs_on_the_row_exact_window(per_token_trunk):
    """GPU smoke 2026-10-18: requests whose processors are not branch-safe
    keep AST on its primary chain, and the engine then verifies through
    forward_with_taps.  On the exact route that forward must be a row-exact
    window (it was the stock wide forward, and greedy output diverged)."""
    model, _ = tiny_qwen4_mtp()
    handle = _route(model)
    prompt = [1, 7, 3, 9, 2, 8, 4, 6, 5, 11]
    cache = _prefilled(model, prompt)
    tokens = [13, 2, 40]
    (expected, expected_seed), _ = _serial(model, cache, tokens)
    before = handle.status()["windows_row_exact"]
    owner = HybridVerifyRows([copy.deepcopy(cache)])
    transaction = owner.begin(lengths=[len(tokens)])
    logits, seeds = model.forward_with_taps(mx.array([tokens]), transaction.caches, _layers(model))
    transaction.abort()
    assert handle.status()["windows_row_exact"] == before + 1
    np.testing.assert_array_equal(np.asarray(logits), np.asarray(expected))
    np.testing.assert_array_equal(np.asarray(seeds), np.asarray(expected_seed))


def test_unsafe_processor_keeps_ast_on_primary_and_output_exact():
    model, _ = tiny_qwen4_mtp()
    handle = _route(model)
    draft = _drafter(model, num_draft=4)
    wrapper = _wrapper(model, draft)
    batch = _ast_generator(model, wrapper)

    def identity(tokens, logits):  # no branch-safe contract declared
        return logits

    prompt = [1, 7, 3, 9, 2, 8, 4, 6, 5, 11, 13, 2]
    batch.insert([prompt], max_tokens=[14], sampling_configs=[{"sampling_temp": 0}],
                 logits_processors=[[identity]])
    output, _ = drain(batch)
    assert output[0] == reference(model, prompt, 14)
    stats = batch.scheduler_stats
    assert stats.get("external_progressive_ast_primary_fallback_rounds", 0) > 0
    assert any(key.startswith("external_progressive_unsupported_processor:") for key in stats)
    status = handle.status()
    assert status["windows_row_exact"] > 0 and status["windows_not_exact"] == 0


def _reference_with(model, prompt, length, processor):
    cache, tokens, output = model.make_cache(), list(prompt), []
    for step in range(length):
        logits = model(mx.array([tokens if step == 0 else [tokens[-1]]]), cache=cache)
        row = processor(mx.array(tokens), logits[:, -1].astype(mx.float32))
        token = int(mx.argmax(row[0]).item())
        output.append(token)
        tokens.append(token)
    return output


def test_presence_penalty_is_branch_safe_and_ast_engages_with_it():
    """Qwen instruct profiles default presence_penalty to 1.5; with it AST
    must still branch, and stay equal to penalized ordinary greedy."""
    from mlx2.runtime.sample_utils import make_presence_penalty
    from mlx2.runtime.speculative_topology import processors_branch_safe

    model, _ = tiny_qwen4_mtp()
    _route(model)
    prompt = [1, 7, 3, 9, 2, 8, 4, 6, 5, 11, 13, 2]
    penalty = make_presence_penalty(1.5, 0, generation_start=len(prompt))
    assert processors_branch_safe([penalty])
    assert penalty.forbidden_token_ids_at_length(len(prompt) + 3) == ()
    draft = _drafter(model, num_draft=4)
    wrapper = _wrapper(model, draft)
    batch = _ast_generator(model, wrapper)
    batch.insert([prompt], max_tokens=[16], sampling_configs=[{"sampling_temp": 0}],
                 logits_processors=[[penalty]])
    output, final = drain(batch)
    assert output[0] == _reference_with(model, prompt, 16, penalty)
    stats = batch.scheduler_stats
    assert stats.get("external_progressive_alternative_rounds", 0) > 0
    assert not any(key.startswith("external_progressive_unsupported_processor:") for key in stats)
    executor = final[0].speculative_receipt["progressive_verification"]["target_row_executor"]
    assert executor["row_exact_launches"] > 0 and executor["nonexact_launches"] == 0
