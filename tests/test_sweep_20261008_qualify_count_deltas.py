"""c8-qual-ops-2 / qualification#1 (sweep 2026-10-08): feature gates that
read lifetime serving counters (verify_bitexact, apc_inflight_prefix_wait,
memory_preemption and the older APCv2 checkpoint, SRPT, tool-grammar,
weight-streaming, approximate-KV, int8-prefill, SPOMIN and Flash-Next dynamic
accept observations) passed a qualifier run against a server whose earlier
traffic had engaged the mechanism, although this run never engaged it.  Like
the external-draft family (rfix-kad 2026-10-07), each is now final - initial;
without an initial snapshot the final value still counts."""

from __future__ import annotations

import copy

import pytest

from mlx2.qualification import required_feature_checks
from scripts.qualify_serving import feature_observations

SETTINGS = {
    "mtp": False,
    "verify_bitexact": {"enabled": True},
    "apc_inflight_prefix_wait": {
        "enabled": True, "min_shared_tokens": 1024, "max_wait_ms": 300_000},
    "memory_preemption": {"enabled": True, "stall_seconds": 60.0, "on_pressure": True},
}
# Every counter one of the gates below reads, as an earlier run left it.
HISTORY = {
    "settings": SETTINGS,
    "counts": {
        "apc_inflight_checkpoints_published": 7,
        "apc_inflight_prefix_hits": 3,
        "memory_preemptions": 2,
        "preempted_replays": 2,
        "apc_interior_checkpoints_captured": 4,
        "apc_interior_checkpoints_published": 4,
        "apc_rolling_checkpoints_published": 3,
        "apc_rolling_checkpoints_cancel_published": 1,
        "apc_junction_checkpoints_published": 2,
        "constrained_tool_grammar_auto_engagements": 5,
        "constrained_tool_grammar_streams": 6,
        "stream_page_ins_total": 11,
        "dense_stream_page_ins_total": 12,
    },
    "apcv2": {"lifetime": {"rolling_hits": 3, "junction_hits": 2}},
    "scheduler": {
        "prefill_scheduling_bypasses": 2,
        "prefill_scheduling_bypass_forced": 1,
    },
    "verify_bitexact": {"active": True, "dispatches": 9},
    "approximate_kv": {"applied": 3, "mtp_lanes": 2},
    "int8_prefill": {"counts": {"engaged_calls": 40}},
    "spomin_live_surgery": {"counts": {"applied": 2}},
    "execution": {"fused_gdn": {"replay_dynamic_rollback_calls": 8}},
}
GATES = {
    "verify_bitexact": 9,
    "apc_inflight_prefix_wait": 3,
    "memory_preemption": 2,
    "apc_interior_checkpoints": 4,
    "apc_rolling_checkpoints": 3,
    "apc_junction_checkpoints": 2,
    "prefill_scheduling": 3,
    "tool_grammar_auto": 5,
    "tool_grammar_streaming": 6,
    "moe_expert_streaming": 11,
    "dense_weight_streaming": 12,
    "approximate_kv": 3,
    "approximate_kv_mtp": 2,
    "int8_prefill": 40,
    "spomin_surgery": 2,
    "fused_gdn_dynamic_accept": 8,
}


def _grow(status, *path, by=1):
    node = status
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] += by


def test_the_named_gates_are_required_for_these_settings():
    required = required_feature_checks(SETTINGS)
    assert {"feature_verify_bitexact", "feature_apc_inflight_prefix_wait",
            "feature_memory_preemption"} <= required


def test_identical_snapshots_observe_nothing():
    # A server that engaged every mechanism before the run, and nothing
    # during it: no gate may pass on that history.
    observed = feature_observations(copy.deepcopy(HISTORY), initial=HISTORY)
    assert {name: observed[name] for name in GATES} == dict.fromkeys(GATES, 0)


def test_run_growth_is_observed_as_a_delta():
    final = copy.deepcopy(HISTORY)
    _grow(final, "verify_bitexact", "dispatches", by=5)
    for key in final["counts"]:
        _grow(final, "counts", key)
    _grow(final, "apcv2", "lifetime", "rolling_hits")
    _grow(final, "apcv2", "lifetime", "junction_hits")
    _grow(final, "scheduler", "prefill_scheduling_bypasses")
    _grow(final, "approximate_kv", "applied", by=2)
    _grow(final, "approximate_kv", "mtp_lanes")
    _grow(final, "int8_prefill", "counts", "engaged_calls", by=6)
    _grow(final, "spomin_live_surgery", "counts", "applied")
    _grow(final, "execution", "fused_gdn", "replay_dynamic_rollback_calls", by=3)
    observed = feature_observations(final, initial=HISTORY)
    assert {name: observed[name] for name in GATES} == {
        "verify_bitexact": 5,
        "apc_inflight_prefix_wait": 1,
        "memory_preemption": 1,
        "apc_interior_checkpoints": 1,
        "apc_rolling_checkpoints": 1,
        "apc_junction_checkpoints": 1,
        "prefill_scheduling": 1,
        "tool_grammar_auto": 1,
        "tool_grammar_streaming": 1,
        "moe_expert_streaming": 1,
        "dense_weight_streaming": 1,
        "approximate_kv": 2,
        "approximate_kv_mtp": 1,
        "int8_prefill": 6,
        "spomin_surgery": 1,
        "fused_gdn_dynamic_accept": 3,
    }


@pytest.mark.parametrize("grown, gate", [
    # A publication during the run whose follower hit was earlier traffic.
    (("counts", "apc_inflight_checkpoints_published"), "apc_inflight_prefix_wait"),
    # A follower hit during the run against an earlier publication.
    (("counts", "apc_inflight_prefix_hits"), "apc_inflight_prefix_wait"),
    # A preemption during the run whose replay happened earlier, and back.
    (("counts", "memory_preemptions"), "memory_preemption"),
    (("counts", "preempted_replays"), "memory_preemption"),
    (("counts", "apc_interior_checkpoints_captured"), "apc_interior_checkpoints"),
    (("counts", "apc_rolling_checkpoints_published"), "apc_rolling_checkpoints"),
    (("apcv2", "lifetime", "rolling_hits"), "apc_rolling_checkpoints"),
    (("counts", "apc_junction_checkpoints_published"), "apc_junction_checkpoints"),
    (("apcv2", "lifetime", "junction_hits"), "apc_junction_checkpoints"),
])
def test_half_a_mechanism_during_the_run_is_not_engagement(grown, gate):
    final = copy.deepcopy(HISTORY)
    _grow(final, *grown)
    assert feature_observations(final, initial=HISTORY)[gate] == 0


def test_a_cancelled_rolling_publication_during_the_run_still_counts():
    # The KV route's only rolling publication is a cancelled prefill's
    # partial cache; that one counter is its evidence.
    final = copy.deepcopy(HISTORY)
    _grow(final, "counts", "apc_rolling_checkpoints_cancel_published")
    assert feature_observations(final, initial=HISTORY)["apc_rolling_checkpoints"] == 1


def test_inactive_bitexact_mode_never_counts():
    final = copy.deepcopy(HISTORY)
    final["verify_bitexact"] = {"active": False, "dispatches": 50}
    assert feature_observations(final, initial=HISTORY)["verify_bitexact"] == 0


def test_without_initial_the_final_value_counts():
    observed = feature_observations(copy.deepcopy(HISTORY))
    assert {name: observed[name] for name in GATES} == GATES


def test_restart_bound_persistence_and_the_host_gauge_stay_absolute():
    # apc_persistence is deliberately restart-bound (one process writes, the
    # next rescans and restores) and host_memory_signals is a gauge; neither
    # is a run delta.
    status = {
        "apcv2": {"idle_disk": {"persisted_writes": 1, "restores": 1},
                  "persistence": {"rescan": {"registered": 1}}},
        "host_memory_available_bytes": 8 << 30,
    }
    observed = feature_observations(copy.deepcopy(status), initial=status)
    assert observed["apc_persistence"] == 1
    assert observed["host_memory_signals"] == 1


# --- Review round 1: every selectable feature gate ---------------------------
#
# The first pass left the adapter-diagnostic gates (segmented / indexed /
# pooled / scatter QSA, PLE, Flash-Next fused GDN decode, fused MoE, APC
# sessions and the indexed merge latches) reading lifetime state.  Every gate
# feature_observations reports is now classified and exercised: identical
# initial and final snapshots of a server that engaged everything before the
# run must pass nothing but the by-design exceptions below.

# Status observations that are deliberately not run engagement.
NOT_RUN_LOCAL = {
    # One process writes the snapshot; the next rescans and restores it.
    "apc_persistence",
    # A gauge: the host reading is present or not.
    "host_memory_signals",
    # A load-time weight transform with no per-call counter.
    "fp32_head_logits",
}
# Evidence from offline reports passed beside the snapshots, never /v1/status.
EXTERNAL_EVIDENCE = {"adaptive_mtp_depth", "mtp_ordinary_handoff", "approximate_kv_fidelity"}
# Latched booleans with no run counter: (initial False -> final True) is the
# only attributable engagement.
LATCHES = {
    "indexed_fused_merge": ("execution", "indexed_qsa", "fused_merge", "engaged"),
    "indexed_output_gate": ("execution", "indexed_qsa", "fused_merge", "gate_engaged"),
}


def _merged(base, extra):
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merged(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


# HISTORY plus every other counter or latch a status-derived gate reads, as an
# earlier run left it.  Every value a gate's growth below touches is nonzero
# (or True) here, so a gate that read it as a lifetime value would pass.
ENGAGED = _merged(HISTORY, {
    "settings": {
        "sp_qmm": "measured",
        "qsdpa_verify_kernel": {"enabled": True, "min_context": 1024},
        "execution_policy": {
            "progressive_verification_tile": 4,
            "progressive_multilane_draft_cap": 2,
        },
    },
    "counts": {"ingress_cohort_target_reached": 2},
    "scheduler": {
        "external_rounds": 9, "draft_fallbacks": 0, "proposed_tokens": 30,
        "paired_cache_resumes": 4, "segmented_transactions": 9,
        "segmented_rollbacks": 3, "pld_retrieval_cycles": 5, "pld_proposed": 12,
        "pld_rollbacks": 2, "pld_batched_rounds": 3, "pld_rotating_replay_rounds": 2,
        "self_mtp_copy_rounds": 4, "fly_relaxed_accepts": 3,
        "external_batched_prefill_rounds": 2, "external_batched_prefill_lanes": 4,
        "external_pairwise_selection_groups": 3,
        "external_tree_rounds": 5, "external_tensorfold_target_rounds": 5,
        "external_progressive_verify_tile": 4,
        "external_progressive_verify_rounds": 3,
        "external_progressive_verify_launches": 7,
        "external_progressive_verify_target_rows": 20,
        "external_progressive_verify_full_tiles": 3,
        "external_multilane_draft_cap": 2,
        "external_multilane_draft_cap_rounds": 3,
        "external_multilane_draft_cap_lanes": 6,
        "decode_fairness_prefill_chunks": 4,
        "decode_first_published_rounds": 6,
        "decode_fairness_slice_floor_lifts": 2,
    },
    "recent_receipts": [{"mtp": {"verification": "fly", "relaxed_accepts": 2}}],
    "sp_qmm": {"enabled": True, "modules": 4, "routed_calls": 10},
    "qsdpa_verify": {"counts": {"verify_kernel_calls": 5}},
    "lane_matmul": {"counts": {"lane_calls": 40}},
    "apcv2": {
        "idle_disk": {
            "parks": 3, "resumes": 2, "prefetch_restores_ok": 1, "prefetch_hits": 1,
            "persisted_writes": 2, "restores": 1,
        },
        "persistence": {"rescan": {"registered": 1}},
    },
    "host_memory_available_bytes": 8 << 30,
    "execution": {
        "indexed_qsa": {
            "counts": {"engaged": 5},
            "fused_merge": {"engaged": True, "gate_engaged": True},
        },
        "segmented_mtp": {
            "shared_qsa_batched_selections": 4,
            "async_qsa_promotion_engaged": 2,
            "private_delta_attention_calls": 3,
        },
        "round_levers": {
            "ple_tail_prefetch_tables": 6, "qsa_pooled_key_cache_hits": 8,
            "qsa_scatter_chosen_calls": 9, "eager_async_evals": 5,
        },
        "ple_tables": [{"lookups": 7}, {"lookups": 2}],
        "ple_compile": {
            "enabled": True,
            "counts": {"builds": 1, "hits": 9, "fallbacks": 0, "skips": 0},
        },
        "fused_gdn": {
            # Flash-Next decode / verify / replay and default-on forms.
            "fused_calls": 11, "decode_fallback_reasons": {"batch of 2 rows": 1},
            "verify_calls": 4, "replay_rollback_calls": 2,
            "prefill_calls": 6, "batch_decode": {"calls": 3},
            "batch_verify": {"calls": 2},
            # Qwen3.8-27B decode switch and prefill switch.
            "enabled": True, "prefill_enabled": True, "decode_calls": 7,
            "batch_decode_calls": 2, "tree_calls": 3, "prefill_chunk_calls": 2,
        },
        "moe": {
            "fused_gate_up_layers": 2, "dispatches": {"scalar": 12, "tile4": 3},
            "weighted_sum": {"calls": 4},
            "moe_window": {"topk_calls": {"launch": 3, "fold": 2}},
            "routed_decode": {"calls": 5},
        },
        "tensorfold_prefill": {"counters": {"projection_calls": 4}},
        "gdn_prefill_scan": {"counters": {"calls": 3}},
        "varlen_dense_mlp": {
            "counters": {"mlp_compaction_calls": 2, "padding_token_rows": 9}},
        "varlen_sparse_moe": {
            "counters": {"moe_compaction_calls": 2, "padding_token_rows": 9}},
        "hc_decode": {"enabled": True, "calls": 3},
        "attn_fused_rows": {"enabled": True, "counts": {
            "projection_grouped": 3, "prep_rows": 3, "sdpa_1pass_rows": 2,
            "sdpa_2pass_rows": 1, "mask_rows": 2, "index_q_rows": 2}},
        "tensorfold_longctx": {
            "qsa_fused_scores": {"enabled": True, "counts": {"engaged": 4}}},
        "moe_nax_gather": {"mode": "gather", "calls": {"gather": 3}},
        "fused_gdn_decode": {"fused_calls": 5},
        "decode_wins": {
            "gdn": {"batch_decode": {"calls": 2}, "verify": {"calls": 2},
                    "batch_verify": {"calls": 2}},
            "moe": {"window_calls": 3, "routed_gate_up_calls": 4,
                    "topk_launch_calls": 2},
        },
        "moe_pad": {"choices": {"adaptive_pad": 3}},
        "invariant_prefill": {"counts": {"forwards": 4}},
        "gdn_state": {"state_dtype": "float16",
                      "counters": {"kernel_launches": 3, "ops_calls": 1}},
        "gdn_core": {"enabled": True, "calls": 6},
        "fp32_head_logits": {"enabled": True, "extra_resident_bytes": 4096},
        "qsa_nax_prefill": {"engagements": 3},
    },
})

_X = ("execution",)
_S = ("scheduler",)
# The run-local growth that engages each counter-backed gate: {path: amount}.
GROWTH = {
    **{gate: {path: 1} for gate, path in {
        "verify_bitexact": ("verify_bitexact", "dispatches"),
        "approximate_kv": ("approximate_kv", "applied"),
        "approximate_kv_mtp": ("approximate_kv", "mtp_lanes"),
        "int8_prefill": ("int8_prefill", "counts", "engaged_calls"),
        "spomin_surgery": ("spomin_live_surgery", "counts", "applied"),
        "tool_grammar_auto": ("counts", "constrained_tool_grammar_auto_engagements"),
        "tool_grammar_streaming": ("counts", "constrained_tool_grammar_streams"),
        "moe_expert_streaming": ("counts", "stream_page_ins_total"),
        "dense_weight_streaming": ("counts", "dense_stream_page_ins_total"),
        "apc_rolling_checkpoints": ("counts", "apc_rolling_checkpoints_cancel_published"),
        "prefill_scheduling": _S + ("prefill_scheduling_bypasses",),
        "ingress_cohort": ("counts", "ingress_cohort_target_reached"),
        "external_draft": _S + ("external_rounds",),
        "proposal_distribution": _S + ("proposed_tokens",),
        "paired_draft_cache": _S + ("paired_cache_resumes",),
        "segmented_transaction": _S + ("segmented_transactions",),
        "segmented_rollback": _S + ("segmented_rollbacks",),
        "prompt_lookup": _S + ("pld_retrieval_cycles",),
        "prompt_lookup_proposals": _S + ("pld_proposed",),
        "prompt_lookup_rollback": _S + ("pld_rollbacks",),
        "prompt_lookup_batched_verify": _S + ("pld_batched_rounds",),
        "prompt_lookup_rotating_replay": _S + ("pld_rotating_replay_rounds",),
        "self_mtp_copy_draft": _S + ("self_mtp_copy_rounds",),
        "fly_verification": _S + ("fly_relaxed_accepts",),
        "external_pairwise_selection": _S + ("external_pairwise_selection_groups",),
        "decode_fairness": _S + ("decode_fairness_prefill_chunks",),
        "decode_first": _S + ("decode_first_published_rounds",),
        "decode_fairness_slice_floor": _S + ("decode_fairness_slice_floor_lifts",),
        "sp_qmm": ("sp_qmm", "routed_calls"),
        "qsdpa_verify_kernel": ("qsdpa_verify", "counts", "verify_kernel_calls"),
        "lane_matmul": ("lane_matmul", "counts", "lane_calls"),
        "shared_qsa": _X + ("segmented_mtp", "shared_qsa_batched_selections"),
        "async_promotion": _X + ("segmented_mtp", "async_qsa_promotion_engaged"),
        "private_delta": _X + ("segmented_mtp", "private_delta_attention_calls"),
        "indexed_qsa": _X + ("indexed_qsa", "counts", "engaged"),
        "known_tail_prefetch": _X + ("round_levers", "ple_tail_prefetch_tables"),
        "pooled_qsa": _X + ("round_levers", "qsa_pooled_key_cache_hits"),
        "scatter_qsa": _X + ("round_levers", "qsa_scatter_chosen_calls"),
        "eager_dispatch": _X + ("round_levers", "eager_async_evals"),
        "file_backed_ple": _X + ("ple_tables", 0, "lookups"),
        "compiled_ple": _X + ("ple_compile", "counts", "hits"),
        "fused_gdn_decode": _X + ("fused_gdn", "fused_calls"),
        "fused_gdn_verify": _X + ("fused_gdn", "verify_calls"),
        "fused_gdn_replay_rollback": _X + ("fused_gdn", "replay_rollback_calls"),
        "fused_gdn_dynamic_accept": _X + ("fused_gdn", "replay_dynamic_rollback_calls"),
        "fused_gdn_prefill": _X + ("fused_gdn", "prefill_calls"),
        "fused_gdn_batch_decode": _X + ("fused_gdn", "batch_decode", "calls"),
        "fused_gdn_batch_verify": _X + ("fused_gdn", "batch_verify", "calls"),
        "qwen38_fused_gdn": _X + ("fused_gdn", "decode_calls"),
        "qwen38_fused_gdn_prefill": _X + ("fused_gdn", "prefill_chunk_calls"),
        "fused_moe": _X + ("moe", "dispatches", "scalar"),
        "moe_weighted_sum": _X + ("moe", "weighted_sum", "calls"),
        "moe_topk_fold": _X + ("moe", "moe_window", "topk_calls", "launch"),
        "moe_routed_decode": _X + ("moe", "routed_decode", "calls"),
        "prefill_scan": _X + ("gdn_prefill_scan", "counters", "calls"),
        "prefill_projection": _X + ("tensorfold_prefill", "counters", "projection_calls"),
        "hc_decode": _X + ("hc_decode", "calls"),
        "qsa_fused_scores": _X + ("tensorfold_longctx", "qsa_fused_scores", "counts", "engaged"),
        "moe_nax_gather": _X + ("moe_nax_gather", "calls", "gather"),
        "qwen36_fused_gdn_decode": _X + ("fused_gdn_decode", "fused_calls"),
        "qwen36_fused_gdn_batch_decode": _X + ("decode_wins", "gdn", "batch_decode", "calls"),
        "qwen36_fused_gdn_verify": _X + ("decode_wins", "gdn", "verify", "calls"),
        "qwen36_fused_gdn_batch_verify": _X + ("decode_wins", "gdn", "batch_verify", "calls"),
        "qwen36_moe_window": _X + ("decode_wins", "moe", "window_calls"),
        "moe_rhs_pad": _X + ("moe_pad", "choices", "adaptive_pad"),
        "invariant_prefill": _X + ("invariant_prefill", "counts", "forwards"),
        "gdn_state_fp16": _X + ("gdn_state", "counters", "kernel_launches"),
        "gdn_core": _X + ("gdn_core", "calls"),
        "qsa_nax_prefill": _X + ("qsa_nax_prefill", "engagements"),
    }.items()},
    # Gates that need two or more counters to move together.
    "apc_inflight_prefix_wait": {
        ("counts", "apc_inflight_checkpoints_published"): 1,
        ("counts", "apc_inflight_prefix_hits"): 1},
    "memory_preemption": {
        ("counts", "memory_preemptions"): 1, ("counts", "preempted_replays"): 1},
    "apc_interior_checkpoints": {
        ("counts", "apc_interior_checkpoints_captured"): 1,
        ("counts", "apc_interior_checkpoints_published"): 1},
    "apc_junction_checkpoints": {
        ("counts", "apc_junction_checkpoints_published"): 1,
        ("apcv2", "lifetime", "junction_hits"): 1},
    "apc_sessions": {
        ("apcv2", "idle_disk", key): 1
        for key in ("parks", "resumes", "prefetch_restores_ok", "prefetch_hits")},
    "external_tree": {
        _S + ("external_tree_rounds",): 1,
        _S + ("external_tensorfold_target_rounds",): 1},
    "external_varlen_prefill": {
        _S + ("external_batched_prefill_rounds",): 1,
        _S + ("external_batched_prefill_lanes",): 1},
    "progressive_verification": {
        _S + ("external_progressive_verify_rounds",): 1,
        _S + ("external_progressive_verify_full_tiles",): 1,
        _S + ("external_progressive_verify_launches",): 2,
        _S + ("external_progressive_verify_target_rows",): 4},
    "progressive_multilane_draft_cap": {
        _S + ("external_multilane_draft_cap_rounds",): 1,
        _S + ("external_multilane_draft_cap_lanes",): 2},
    "varlen_dense_mlp": {
        _X + ("varlen_dense_mlp", "counters", "mlp_compaction_calls"): 1,
        _X + ("varlen_dense_mlp", "counters", "padding_token_rows"): 1},
    "varlen_sparse_moe": {
        _X + ("varlen_sparse_moe", "counters", "moe_compaction_calls"): 1,
        _X + ("varlen_sparse_moe", "counters", "padding_token_rows"): 1},
    **{gate: {
        _X + ("attn_fused_rows", "counts", "projection_grouped"): 1,
        _X + ("attn_fused_rows", "counts", "prep_rows"): 1,
        _X + ("attn_fused_rows", "counts", "sdpa_1pass_rows"): 1,
        **extra,
    } for gate, extra in {
        "attn_fused_rows": {},
        "attn_fused_rows_qsa_mask": {_X + ("attn_fused_rows", "counts", "mask_rows"): 1},
        "attn_fused_rows_index_q": {_X + ("attn_fused_rows", "counts", "index_q_rows"): 1},
    }.items()},
}


def _read(status, path):
    node = status
    for key in path:
        node = node[key]
    return node


def test_every_feature_gate_is_classified():
    observable = set(feature_observations({}))
    classes = [set(GROWTH), set(LATCHES), NOT_RUN_LOCAL, EXTERNAL_EVIDENCE]
    assert sum(len(names) for names in classes) == len(set().union(*classes))
    assert set().union(*classes) == observable


def test_the_history_engaged_every_gate_it_grows():
    # Without this, identical snapshots would observe nothing vacuously.
    for gate, growth in GROWTH.items():
        for path in growth:
            assert _read(ENGAGED, path) > 0, (gate, path)
    for gate, path in LATCHES.items():
        assert _read(ENGAGED, path) is True, gate


def test_identical_snapshots_of_an_engaged_server_pass_no_gate():
    observed = feature_observations(copy.deepcopy(ENGAGED), initial=ENGAGED)
    passing = {name for name, value in observed.items() if value > 0}
    assert passing == NOT_RUN_LOCAL


@pytest.mark.parametrize("gate", sorted(GROWTH))
def test_each_gate_passes_on_its_own_run_growth(gate):
    final = copy.deepcopy(ENGAGED)
    for path, amount in GROWTH[gate].items():
        _grow(final, *path, by=amount)
    assert feature_observations(final, initial=ENGAGED)[gate] > 0


@pytest.mark.parametrize("gate", sorted(LATCHES))
def test_a_latch_set_during_the_run_is_engagement(gate):
    initial = copy.deepcopy(ENGAGED)
    node = _read(initial, LATCHES[gate][:-1])
    node[LATCHES[gate][-1]] = False
    assert feature_observations(copy.deepcopy(ENGAGED), initial=initial)[gate] == 1
    # Without an initial snapshot the final latch still counts.
    assert feature_observations(copy.deepcopy(ENGAGED))[gate] == 1


def test_a_latch_already_set_before_the_run_fails_closed_with_a_reason():
    import scripts.qualify_serving as qualify

    reasons = qualify.latched_before_run(ENGAGED)
    assert set(reasons) == set(LATCHES)
    for gate, reason in reasons.items():
        assert ".".join(LATCHES[gate][1:]) in reason
        assert "initial" in reason
    assert qualify.latched_before_run(None) == {}
    clear = copy.deepcopy(ENGAGED)
    clear["execution"]["indexed_qsa"]["fused_merge"] = {
        "engaged": False, "gate_engaged": False}
    assert qualify.latched_before_run(clear) == {}


def test_fused_gdn_decode_compares_call_and_refusal_deltas():
    # A geometry refusal during the run voids the run's fused decode calls.
    refused = copy.deepcopy(ENGAGED)
    _grow(refused, "execution", "fused_gdn", "fused_calls", by=100)
    refused["execution"]["fused_gdn"]["decode_fallback_reasons"][
        "rollback geometry not describable"] = 1
    assert feature_observations(refused, initial=ENGAGED)["fused_gdn_decode"] == 0
    # One from before the run does not, and only the run's calls count.
    before = copy.deepcopy(ENGAGED)
    before["execution"]["fused_gdn"]["decode_fallback_reasons"][
        "masked decode"] = 7
    after = copy.deepcopy(before)
    _grow(after, "execution", "fused_gdn", "fused_calls", by=4)
    assert feature_observations(after, initial=before)["fused_gdn_decode"] == 4


def test_compiled_ple_keeps_its_lifetime_health_guard():
    # A demoted signature stays eager for the process, so a fallback at any
    # time voids the run's hits.
    initial = copy.deepcopy(ENGAGED)
    initial["execution"]["ple_compile"]["counts"]["fallbacks"] = 1
    final = copy.deepcopy(initial)
    _grow(final, "execution", "ple_compile", "counts", "hits", by=5)
    assert feature_observations(final, initial=initial)["compiled_ple"] == 0
    final["execution"]["ple_compile"]["counts"]["fallbacks"] = 0
    initial["execution"]["ple_compile"]["counts"]["fallbacks"] = 0
    assert feature_observations(final, initial=initial)["compiled_ple"] == 5


def test_fused_moe_counts_run_dispatches_only_with_fused_layers():
    final = copy.deepcopy(ENGAGED)
    _grow(final, "execution", "moe", "dispatches", "tile4", by=6)
    assert feature_observations(final, initial=ENGAGED)["fused_moe"] == 6
    final["execution"]["moe"]["fused_gate_up_layers"] = 0
    assert feature_observations(final, initial=ENGAGED)["fused_moe"] == 0
