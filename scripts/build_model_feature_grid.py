"""Build a review grid from mlx2's registered models and serving switches.

This is a static source inventory. It never imports mlx2 or loads a model.
Unrecorded evidence means unreviewed, not unsupported or safe to enable.

Dimensions: model family, ``checkpoint`` and feature/flag, plus the mlx2
commit each decision was established on (``source_revision``).  A
``checkpoint`` of ``*`` is a family-level decision (what the adapter
declares); evidence from running a model names the exact checkpoint it ran,
``<artifact name>@<fingerprint>`` (Pierre, 2026-10-17: checkpoint and commit
are dimensions of every test and qualification record).  A row that claims
``qualified=yes`` or ``observed_used=yes`` without both is refused.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import re
import subprocess
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "research" / "model-feature-grid.tsv"
DECISIONS = ROOT / "docs" / "research" / "model-feature-decisions.tsv"
REGISTRY = ROOT / "src" / "mlx2" / "adapters" / "registry.py"
SERVER = ROOT / "src" / "mlx2" / "server.py"

# Registry model_type values with distinct adapter or topology decisions.
VARIANTS = {
    "glm53_gguf": ("reap50-q3-k-m-packed-ordinary",),
    "qwen3_5": ("4b", "9b", "27b"),
    "qwen3_5_moe": ("35b_a3b", "122b_a10b"),
    "gemma4": ("a4b", "31b"),
    "laguna": ("xs21", "s21"),
    "nemotron_h": ("super", "lightning"),
    "qwen4_exp": ("base", "flash-next"),
    "flint": ("40b-a2.5b",),
    "llama": ("base", "33b"),
}

# Features that are selected through execution policy or adapter configuration
# rather than a dedicated command-line switch. This is a review vocabulary,
# not a statement that every registered model implements each feature.
FEATURES = (
    "oq4e_conversion",
    # Morning follow-up probes (2026-11-06): opt-in research options.
    "batch_rows_attention", "prefill_tail_row", "prefill_tail_skip",
    # Nemotron 3.5 Super VL NVFP4 export (2026-11-10): checkpoint defaults.
    "vendor_fp8_dense", "nvfp4_expert_scale_fold",
    "row_exact_self_batched_scaled_projections", "mamba_fused_step_ordinary",
    "fp_row_exact_verify", "fp_row_exact_fused_kernel",
    # Residual ablation (Heretic concept, 2026-11-05): a baked derived checkpoint.
    "residual_ablation_bake",
    # Ripple perf pass (2026-11-04).
    "fused_elementwise", "hot_path_copy_cleanup", "continuous_batch_lanes4",
    "prefill_step_2048", "sliding_window_blocked_attention",
    "metal_rmsnorm", "gqa_folded_attention",
    # Ripple glue/tree pass (2026-11-05).
    "metal_glue", "ast_dflash2_tree_tiles", "external_draft_num_draft", "gsm8k_spec_acceptance",
    "sampling_defaults", "anthropic_thinking_default",
    "source_engram_carry_tuple", "source_expert_direct_final_reads",
    "source_expert_decayed_retention", "native_fp_tensor_verify", "apc_inpass_checkpoints",
    "ling_optimizations", "ast_apcv2_current_source",
    "ling_kda_fusion_components", "ling_split_mla_components",
    "inkling_fused_ops", "inkling_q2_down_combine", "inkling_inline_relative_attention",
    "mtp_tree_pipeline",
    "integer_exact_microscaling", "exl3_upstream_weight_format", "exl3_execution_rows", "exl3_execution_fused", "exl3_execution_fused_nax", "exl3_execution_fused_w8a8", "exl3_execution_nax", "exl3_execution_nax_fused", "exl3_resident_dense",
    "resident_kernel_reload", "kimi_linear_supervised_lora_research",
    "kimi_router_score_profile_one_domain_prompt",
    "kimi_moe_behavior_research", "kimi_moe_loop_research",
    "kimi_moe_terminal_route_counters", "kimi_moe_segment_bounds_research",
    "kimi_moe_protected_unit_lifecycle",
    "hysparse2_nlsa_selection",
    # Existing reviewed GLM rows also need inventory cells for regeneration.
    'glm53_mla_prompt_expanded_f16_v1', 'glm53_prompt_decoded_decoded_f16_v1', 'glm53_prompt_decoded_decoded_f32_v1', 'glm53_route_defaults_kblock_v6', 'glm53_sparse_prefill_gathered_f16_from_n', 'glm53_sparse_prefill_gathered_v2_below_threshold_reference',
    'kda_vector_kernel', 'kimi_mla_latent_tiled', 'kimi_mla_fused_decode', 'kimi_mla_flash_all', 'kimi_mla_flash_law_v2', 'kimi_kda_norm_rounding_ab', 'kimi_session_memory_budget', 'kimi_kda_mla_prometheus', 'native_bf16_nax_prefill', 'context_default_1m', 'kimi_grouped_kda_projections', 'kimi_fused_kda_norm', 'kimi_chunk32_prefill', 'kimi_mla_auto', 'kimi_mla_latent_single', 'kda_chunk_wy', 'router_combine_unbiased', 'kda_fused_projections_norm', 'kimi_ast_packed_tree', 'kimi_ast_serial_tree', 'tensorfold_lane_raw',
    "gguf_native_packed_read_view", "gguf_native_k_blocks",
    "glm53_shared_layer_ast", "draft_bf16_activation_boundary", "mla_prefill_causal_key_trim",
    "glm53_indexer_pools_incremental_v1", "glm53_indexer_select_topk_v1",
    "glm53_indexer_select_topk_v1_kernel", "glm53_sparse_attention_gathered_v1",
    "glm53_sparse_attention_fused_decode_v1", "glm53_sparse_prefill_gathered_v1",
    "glm53_tree_rows_attention_batched_v1", "glm53_tree_frontier_tails_v1",
    "glm53_router_fused_v1", "glm53_kda_small_fused_v1",
    "glm53_compile_regions_v1", "glm53_kda_fused_decode_v1", "glm53_kv_append_rotate_v1",
    "glm53_mla_fused_decode_v1", "glm53_hc_fused_v1", "glm53_router_fused_v2",
    "glm53_f32_weights_bf16_v1", "glm53_kblock_element_fma_v1", "glm53_command_buffers_ops100_v1_prompt_eval_layer_v1",
    "ordinary_prefill_plan_exact", "lane_dense", "native_mtp_num_draft", "mtp_batched_prefill",
    "qsa_batch_decode_sparse", "varlen_sparse_moe", "fp32_head_logits", "gdn_state_dtype_float16", "qsa_lazy_causal_mask", "qsa_stage1_direct_selector", "ple_read_policy_adaptive", "mtp_draft_vocab", "compact_moe_prefill", "qsa_prefetch_partials", "tensorfold_qmv_rows", "tensorfold_prefill", "gdn_core",
    "prefill_plan_exact_slices", "async_qsa_promotion", "apc_eviction_queued_protection",
    "ordinary_reference", "apc_v2", "apc_interior_checkpoints", "apc_junction_checkpoints",
    "apc_rolling_checkpoints", "apc_session_persistence", "tools",
    "vision_input", "video_input", "audio_input", "native_mtp",
    "mtp_ordinary_handoff", "adaptive_mtp_depth", "self_mtp_copy_draft", "speculation_request_opt_out",
    "external_draft", "external_draft_dspark", "external_draft_dspine", "flat_segmented_ast", "ast_speculative_batching",
    "ast_bf16_assistant", "ast_bf16_dspark", "ast_bf16_dflash2",
    "packed_prefill_serial_equivalence", "prompt_lookup", "prompt_lookup_target_s1",
    "decode_first", "prefill_scheduling", "mixed_prefill_decode",
    "varlen_paged_attention", "row_exact_verify", "tensorfold_tree15",
    "ast_ubtree_b15", "gemma3n_kv_shared_row_pruning", "multi_lora_fp32_delta",
    "fused_gdn", "fused_gdn_batch_verify", "moe_rhs_pad",
    "pla_chunk32_prefill", "hc_prefill_fusion", "tensorfold_gdn_decode_block",
    "lfm_moe_decode", "ple_windowed_gather",
    "moe_nax_gather", "moe_routed_decode", "moe_topk_fold", "moe_grouped_rows",
    "paired_shared_expert_q8", "gguf_nax_prefill", "deepseek_index_qat",
    "engram_nvme_offload", "shared_csa2",
    "context_ladder_64k",
    "moe_prefill_row_buffers", "verified_prefix_commit",
    "ten_token_time_attribution", "context_ladder_8k_row_buffers", "engram_lookup_attribution",
    "moe_short_row_exact", "expert_stream_predictive_atlas", "engram_compact_bf16_cache",
    "verified_layerwise_experts", "native_dense_epilogue_fusion", "expert_ple_elastic_residency",
    "ten_token_final_efficiency",
    "dense_tensor_paging", "model_park_activate", "dense_expert_ple_residency",
    "model_residency_fifo", "dense_page_time_attribution",
    "dense_direct_positional_read", "dense_working_set_retention", "peer_omlx_exact_image",
    "single_1k_three_repeat_perf", "native_affine_layer_ab", "atlas_prefetch_miss_ranking",
    "standalone_native_affine",
    # Codex IronMLX/FlashInfer candidates, first GPU checks (2026-11-05).
    "lane_arithmetic_class_apc_identity", "cost_per_commit_time_credit",
    "ironmlx_fused_prefill_swiglu", "expert_stream_scan_resistant_admission",
    # omlx-layout V4.1 route (2026-11-04).
    "source_hc_mega", "omlx_expert_bank_gather",
    "omlx_layer_eval_async", "source_woa_mxfp8_storage",
    "omlx_expert_bank_routed_decode", "omlx_prefill_scan_resistant",
    "compact_iq1_experts",
    "splash_iq1_decoder",
    "tiled_iq1_expert_arena", "gpu_expert_slot_routing",
    "expert_async_completion_epochs", "tiled_iq1_column_f32",
    "tiled_runtime_capacity", "tiled_iq1_1k_three_repeat_perf",
    "source_shard_read_batch", "source_expert_activation_reuse",
    "source_q1_g64_component", "source_expert_lfu90", "source_frequency_atlas",
    "source_expert_lfu75", "source_q1_g64_mixed",
    "source_selected_kernel_bundle_original_lfu", "source_expert_io_overlap_v1",
    "source_expert_retention_cycle_aware_v1", "source_expert_lookahead_heads_d12_priority_v1",
    "source_expert_loader_priority_v1_lookahead_d2", "source_router_bf16_coeff_gemv_v1",
    "source_shared_middle_fused_middle_v1", "source_bf16_dense_resident_f32_v1",
    "source_engram_io_tiered_v1",
    "source_route_drop_min_v1",
    "source_expert_tiers_q2_reablate", "source_route_bias_resident", "source_expert_exl3_bank",
    "source_expert_math_hoisted_fused", "source_resident_math_probe",
    "source_expert_packed_buffers", "source_expert_mapped_input",
    "source_expert_middle_fusion", "source_expert_gate_up_middle_batch",
    "source_expert_gate_up_middle_wide", "source_expert_down_combine_fusion",
    "source_prefill_direct_part_combine",
    "source_routed_resident_ablation",
    "source_prefill_gate_up_worklist",
    "source_prefill_down_worklist",
    "source_prefill_mapped_assembly",
    "source_prefill_bf16_p1",
    "source_prefill_splitk_nax",
    "source_original_binding_checks",
    "source_prefill_wide_legacy_tiles",
    "source_prefill_wide_16x64",
    "source_prefill_wide_fused_middle",
    "source_prefill_wide_fused_middle_16x64",
    "source_hc_coefficients",
    'source_dense_fp8_qat',
    'source_shared_gate_up_qat_reuse',
    'source_woa_bf16_direct_f32',
    'source_finite_all_retained_index',
    'source_selected_kv_decode_reuse',
    'source_all_retained_attention_broadcast',
    'source_qakv_qat_reuse',
    'source_fp8_projection_tiles',
    'source_peer_dwarfstar_attention',
    'source_peer_omlx_attention',
    'source_peer_dwarfstar_fused_staging',
    'source_fp8_native_bf16_pp',
    'source_woa_native_bf16_pp',
    'source_small_m_paired_nax',
    'source_prefill_q1_m1_owner_store',
    'source_ffn_input_qat_reuse',
    'source_native_mx_expert_tiles',
    'source_native_mx_expert_stage',
    'source_native_mx_runtime_dynamic',
    'source_native_mx_original_pp_half',
    'source_native_mx_immutable_layout_cache',
    'source_prefill_frame_reuse',
    'source_prefill_q1_tiny_batch',
    'source_ordered_expert_combine',
    'source_original_paired_dispatch',
    'source_q1_paired_dispatch',
    'peer_omlx_source_offload_baseline',
    'source_prefill_nax_ordered_middle',
    'source_prefill_nax_k16_profile',
    'source_hc_boundary',
    'source_q_b_prefill_tiles',
    'source_output_chain_fusion',
    'source_head_group_attention',
    'source_model_epilogue_fastpaths',
    'source_bounded_completion_epochs',
    'source_phase_resident_diagnostic',
    'source_woa_bf16_legacy_mma',
    'source_head_bf16_direct_f32',
    'source_dense_nax_candidate',
    "ced_encoder_only_prefill", "csa2_packed_kv", "csa2_candidate_only_scoring",
    "ced_dependency_complete_prefill", "ced_state_only_chunks", "ordinary_physical_batch", "numeric_reasoning_strength",
    "effective_weight_identity", "explicit_long_context_admission", "router_precision_admission",
    "moe_window_row_exact", "attention_window_row_exact",
    "dspark_verify_schedule_survival_miss_v1", "verify_moe_rows_m1_rows_v1", "dspark_verify_expert_ledger",
    "progressive_ast_rootless_fallback",
    "qsa_nax_prefill", "qsa_nax_batched", "qsa_nax_law_v2", "qsa_nax_decode", "qsa_b1_selected",
    "invariant_prefill", "quantized_kv", "approximate_kv",
    "host_memory_signals", "host_available_floor", "weight_streaming",
    "apc_pressure_drop_superseded",
    "constrained_tool_grammar", "strict_json_schema", "thinking_guard",
    "exact_prefill_projection_law", "native_mtp_packed_paths",
    "shared_root_tensorfold_tree_target", "ast_native_mtp_packed_3x5",
    "ast_native_mtp_packed_5x3", "best_route_default",
    "external_draft_dspark_ast", "external_mtp_head", "external_mtp_head_ast",
    "external_draft_assistant_sampled_law", "external_draft_lilicorr_native_nvfp4",
    "gguf_text_conversion_adapter", "ordinary_direct_smoke", "prefill_step_256",
    "native_mtp_context_ladder_128k_before_residual_repair",
    "residual_compute_dtype_normalization", "native_mtp_context_ladder_128k",
    "ast_native_mtp_mixed_f32_gdn",
    "ast_ling_packed32_pld_cohort4_apcv2", "ast_ling_packed_raw_tensorfold",
    "ast_ling_packed_tensorfold_completion", "ast_ling_packed_tensorfold_http",
    "ast_ubtree_adaptive_b15", "ast_ubtree_fixed_b3",
    "ast_prompt_lookup", "prompt_lookup_default_policy", "prompt_lookup_span2", "q4_fused_v1",
    "verification_mode", "spawn_subagent", "agentic_chat_profile_256k",
    "q8_packed_residual_rms", "artifact_quantization",
    "decode_component_profile", "q8_projection_policy", "packed_attention_ast",
    "lean_greedy_scores", "w8a8_prefill", "q8_swiglu_epilogue",
    "fused_mlp_stage_a", "row_exact_qmv_explicit_geometry",
    "uzu_attention_prefix_research", "uzu_matrix_gdn_research", "moe_down_geometry_research",
    "q4_gate_three_rows",
    "compiled_rope_append", "compiled_decoder_blocks", "bounded_kv_reservation",
    "implicit_tree_mask", "gqa7_attention", "context_load_prefill",
    "match_aware_pld", "external_dflash2", "fixed_ast_geometry",
    "draft_precision_kernel_joint", "guarded_draft_vocab", "gqa7_shared_kv_rows",
    "segmented_lane_kv", "four_row_q8_weight_reuse", "physical_rope_kv_append",
    "packed_ast_cohort_tuning", "external_dflash2_q8_kv",
    "desktop_tool_discovery_readable_roundtrip",
    "desktop_tool_snapshot_readable_roundtrip",
    "desktop_media_tool_schema_declaration",
    "desktop_autostart_health_loading",
    "desktop_round_budget_and_result_context_controls",
    "desktop_chat_multiple_images",
    "desktop_chat_context_meter",
    "desktop_prompt_presets_and_stable_apcv2_requests",
    "desktop_same_model_delegation_and_health",
    "desktop_agent_history_code_and_scroll",
    "source_ast_stage_observer_setup_failure",
    "source_ast_synchronized_stage_attribution",
    "source_ast_paired_projection_actual_raw_components",
    "source_ast_paired_projection_whole_tree_abba",
    "source_ast_stage_profile_focused_cpu",
    "source_ast_paired_projection_focused_cpu",
    "source_ast_context4k_paced_raw_gate_partial",
    "source_ast_context4k_resume_thermal_admission_stop",
    "source_ast_context4k_private_prefix_resume",
    "source_ast_fully_batched_exact_context_4k",
    "source_ast_selected_kv_reuse_abba_4k",
    "source_ast_continuation_focused_cpu_final",
    "source_ast_index_actual_1k_capture",
    "source_ast_index_actual_1k_replay",
    "source_ast_fully_batched_exact_context_1k",
    "source_ast_selected_kv_reuse_first_pair",
    "source_ast_selected_kv_reuse_abba_1k",
    "source_ast_prepared_numeric_initial_gate",
    "source_ast_load_bound_prepared_numeric",
    "source_ast_prepared_generic_execution",
    "source_ast_mapped_kv_components",
    "source_ast_mapped_index_components",
    "source_ast_cold_http_trace_proxy_failure",
    "source_ast_fully_batched_cold_http",
    "source_ast_context4k_whole_setup_refusal",
    "source_ast_context4k_chunked_thermal_stop",
    "source_ast_actual_stride_m64_attention",
    "source_ast_fully_batched_m1_private_exact",
    "source_ast_batched_focused_cpu_regression",
    "source_ast_batched_m1_full_gate_r2",
    "source_ast_batched_m1_full_gate_r3",
    "source_ast_batched_m1_full_gate_r4",
    "source_ast_role_aware_attention_components",
    "source_ast_batched_actual_chain_stage_diagnostic",
    "source_ast_actual_local_pipeline_replay",
    "source_ast_native_batched_gpu_timeline",
    "source_ast_descendant_distribution_schedule",
    "source_ast_apcv2_private_segment_profiles",
    "source_ast_native_m16_dense_components",
    "source_ast_resident_m16_expert_kernel",
    "source_ast_fully_batched_native_private",
    "source_ast_batched_m1_dense_components",
    "source_ast_batched_m1_initial_full_gate",
    "source_ast_down_group_packing",
    "source_ast_pinned_float_matmul_negative",
    "source_ast_head_axis_attention_components",

    # Source implementation and bounded evidence states; vocabulary only.
    'source_adaptive_demand_admission',
    'source_adaptive_sequential_experts',
    'source_adaptive_shared_total_capacity',
    'source_apcv2_canonical_http_cache_and_disk',
    'source_apcv2_checkpoint_resource_hooks_cpu',
    'source_apcv2_expert_hints_actual_process_restart',
    'source_apcv2_expert_hints_cancel_and_recovery',
    'source_apcv2_expert_hints_optional_pressure',
    'source_apcv2_expert_resource_metadata_cpu',
    'source_apcv2_http_harness_contract_failures',
    'source_apcv2_pressure_original_hid_boundary',
    'source_ast_generic_six_plane_handoff',
    'source_ast_hybrid_actual_both_paths_and_continuous',
    'source_ast_hybrid_actual_miss_scope_failure',
    'source_ast_hybrid_manager_focused_cpu',
    'source_ast_resident_1k_attempt_r1',
    'source_ast_resident_1k_attempt_r2',
    'source_ast_resident_1k_attempt_r3',
    'source_ast_resident_1k_emitted_ab_r4',
    'source_ast_resident_1k_emitted_ab_r5',
    'source_ast_resident_4k_emitted_ab_r4',
    'source_ast_resident_4k_progress_callback_setup_failure',
    'source_ast_resident_4k_setup_r1',
    'source_ast_target_m1_row_invariant_16',
    'source_bounded_ced_private_256',
    'source_dense_coefficient_views_v1',
    'source_dspark_ast_http_3x5',
    'source_dspark_embedded_head_3x5',
    'source_head_bf16_direct_control',
    'source_private_resident_decode_context_16k',
    'source_private_resident_decode_context_1k',
    'source_private_resident_decode_context_4k',
    'source_private_resident_decode_context_8k',
    'source_resident_bank_cache_descriptor_cow_v1',
    'source_resident_bank_gpu_routes_v1',
    'source_resident_bank_route_fused_v1',
    'source_resident_bank_same_source_control',
    'source_resident_prefill_expert_major_wide_v1',
    'source_resident_prefill_task_striding_8_2_v1',
    'source_resident_submission_group2_group4_whole_forward_v1',
    'source_row_packet_v1',
    'source_short_decode_joint_f32_v1',
    'source_submission_shared_first_v1',
    # 2026-10-28 GPU campaign rows (integration branch of 2026-10-28).
    'apcv2_prefill_slice_schedule',
    'mx_lane_serving_route',
    'source_prefill_residency_protect_decode_v1',
    'source_prefill_readahead_layer_ahead_v1',
)


def registered_types() -> list[str]:
    tree = ast.parse(REGISTRY.read_text())
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign):
            continue
        if not isinstance(node.target, ast.Name) or node.target.id != "_RESOLVERS":
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        return sorted(ast.literal_eval(key) for key in node.value.keys)
    raise RuntimeError("_RESOLVERS dictionary was not found")


def server_flags() -> set[str]:
    tree = ast.parse(SERVER.read_text())
    flags = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        for arg in node.args:
            if (
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and arg.value.startswith("--")
            ):
                flags.add(arg.value)
    return flags


def environment_flags() -> set[str]:
    pattern = re.compile(r"\b(?:MLX2|MLX_LM|MLX_QWEN\w*|MLX_GDN|MLX_ENABLE)_+[A-Z][A-Z0-9_]*\b")
    flags = set()
    for path in (ROOT / "src" / "mlx2").rglob("*.py"):
        flags.update(pattern.findall(path.read_text()))
    return flags


FAMILY = "*"
UNRECORDED = {"", "not_recorded", "unreviewed"}


def strong_claim(value):
    return str(value).lower().startswith(("yes", "qualified"))


def row_digest(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@cache
def _commit_exists_and_reachable(revision):
    result = subprocess.run(["git", "rev-parse", "--verify", revision + "^{commit}"], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    if result.returncode:
        return False
    refs = subprocess.check_output(["git", "for-each-ref", "--contains", result.stdout.strip(),
                                    "--format=%(refname)"], cwd=ROOT, text=True)
    return bool(refs.strip())


def strong_claim_problems(row):
    problems = []
    if re.fullmatch(r".+@(sha256:)?[0-9a-f]{8,64}", row["checkpoint"]) is None:
        problems.append("checkpoint must name an artifact fingerprint")
    revision = row.get("source_revision", "")
    revisions = re.findall(r"\b[0-9a-f]{7,40}\b", revision)
    if not revisions or any(not _commit_exists_and_reachable(value) for value in revisions):
        problems.append("source revision is absent, invalid or unreachable from a ref")
    evidence = row.get("evidence", "")
    paths = []
    for item in evidence.split(";"):
        item = item.strip()
        # Historical branch:path and file:symbol notation are read-only citations.
        if ":qualification/" in item:
            item = item.split(":", 1)[1]
        if item.startswith(("qualification/", "docs/", "scripts/", "src/", "tests/", "provenance/", "output/", "/")):
            paths.append(item.split(":", 1)[0])
    if not paths or any(not (ROOT / item).exists() for item in paths):
        problems.append("evidence path is absent or not recorded")
    return problems


def validate_decisions(rows) -> dict:
    """Index reviewed rows by (model, checkpoint, scope, feature, source); refuse
    duplicates and strong claims without their checkpoint and commit."""
    overrides = {}
    for row in rows:
        checkpoint = (row.get("checkpoint") or "").strip()
        if not checkpoint:
            raise ValueError(f"decision without a checkpoint (use {FAMILY!r} for family level): {row['model']} {row['feature_or_flag']}")
        key = (row["model"], checkpoint, row["scope"], row["feature_or_flag"],
               (row.get("source_revision") or "").strip())
        if key in overrides:
            raise ValueError(f"duplicate reviewed decision: {key}")
        strong = strong_claim(row["qualified"]) or strong_claim(row["observed_used"])
        if strong and checkpoint == FAMILY:
            raise ValueError(f"qualified/observed decision must name its checkpoint: {key}")
        if strong and row.get("source_revision", "").strip() in UNRECORDED:
            raise ValueError(f"qualified/observed decision must name its mlx2 commit: {key}")
        if strong:
            problems = strong_claim_problems(row)
            if problems:
                legacy_path = ROOT / "docs/research/model-feature-legacy-exceptions.json"
                legacy = json.loads(legacy_path.read_text()) if legacy_path.is_file() else {}
                if row_digest(row) not in legacy.get("rows", {}):
                    raise ValueError(f"invalid qualified/observed decision {key}: {'; '.join(problems)}")
        overrides[key] = row
    return overrides


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)
    models = [
        f"{model_type}:{variant}"
        for model_type in registered_types()
        for variant in VARIANTS.get(model_type, ("base",))
    ]
    flags = (
        [("feature", flag) for flag in FEATURES]
        + [("server_cli", flag) for flag in sorted(server_flags())]
        + [("environment", flag) for flag in sorted(environment_flags())]
    )
    with DECISIONS.open(newline="") as handle:
        decisions = list(csv.DictReader(handle, delimiter="\t"))
    overrides = validate_decisions(decisions)
    revision_keys = {}
    for key in overrides:
        revision_keys.setdefault(key[:4], []).append(key)
    columns = (
        "model", "checkpoint", "scope", "feature_or_flag", "implemented", "qualified",
        "selected", "observed_used", "decision", "reason", "evidence",
        "source_revision", "artifact_profile",
    )
    buffer = io.StringIO(newline="")
    with buffer as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for model in models:
            # Family-level cells for every switch, then each reviewed exact
            # checkpoint of this family with its own rows.
            checkpoints = [FAMILY] + sorted({
                key[1] for key in overrides if key[0] == model and key[1] != FAMILY
            })
            for checkpoint in checkpoints:
                for scope, flag in flags:
                    key = (model, checkpoint, scope, flag)
                    matches = sorted(revision_keys.get(key, ()))
                    if checkpoint != FAMILY and not matches:
                        continue
                    # Historical outcomes remain separate rows. Never select the
                    # latest source or merge its state with an earlier result.
                    for revision_key in matches or [None]:
                        row = dict.fromkeys(columns, "not_recorded")
                        row.update({"model": model, "checkpoint": checkpoint, "scope": scope,
                                    "feature_or_flag": flag,
                                    "implemented": "unreviewed", "qualified": "unreviewed",
                                    "selected": "unreviewed", "observed_used": "unreviewed",
                                    "decision": "review_needed"})
                        if revision_key is not None:
                            row.update(overrides.pop(revision_key))
                        writer.writerow(row)
        rendered = handle.getvalue()
    if overrides:
        raise ValueError(f"reviewed decisions absent from inventory: {sorted(overrides)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import os
    import tempfile
    descriptor, temporary = tempfile.mkstemp(dir=args.output.parent, prefix=".feature-grid-")
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(rendered)
        os.replace(temporary, args.output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f"{args.output}: {len(models)} model variants x {len(flags)} switches = {len(models) * len(flags)} review cells; {len(decisions)} reviewed")


if __name__ == "__main__":
    main()
