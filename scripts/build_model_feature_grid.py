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
    "qwen3_5": ("4b", "9b", "27b"),
    "qwen3_5_moe": ("35b_a3b", "122b_a10b"),
    "gemma4": ("a4b", "31b"),
    "laguna": ("xs21", "s21"),
    "nemotron_h": ("super", "lightning"),
    "qwen4_exp": ("base", "flash-next"),
    "flint": ("40b-a2.5b",),
}

# Features that are selected through execution policy or adapter configuration
# rather than a dedicated command-line switch. This is a review vocabulary,
# not a statement that every registered model implements each feature.
FEATURES = (
    "ordinary_reference", "apc_v2", "apc_interior_checkpoints", "apc_junction_checkpoints",
    "apc_rolling_checkpoints", "apc_session_persistence", "tools",
    "vision_input", "video_input", "audio_input", "native_mtp",
    "mtp_ordinary_handoff", "adaptive_mtp_depth", "self_mtp_copy_draft",
    "external_draft", "external_draft_dspark", "flat_segmented_ast", "ast_speculative_batching",
    "packed_prefill_serial_equivalence", "prompt_lookup", "prompt_lookup_target_s1",
    "decode_first", "prefill_scheduling", "mixed_prefill_decode",
    "varlen_paged_attention", "row_exact_verify", "tensorfold_tree15",
    "ast_ubtree_b15",
    "fused_gdn", "fused_gdn_batch_verify", "moe_rhs_pad",
    "moe_nax_gather", "moe_routed_decode", "moe_topk_fold", "moe_grouped_rows",
    "moe_window_row_exact", "attention_window_row_exact",
    "progressive_ast_rootless_fallback",
    "qsa_nax_prefill", "qsa_nax_batched", "qsa_nax_decode",
    "invariant_prefill", "quantized_kv", "approximate_kv",
    "host_memory_signals", "host_available_floor", "weight_streaming",
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
    """Index reviewed rows by (model, checkpoint, scope, feature); refuse
    duplicates and strong claims without their checkpoint and commit."""
    overrides = {}
    for row in rows:
        checkpoint = (row.get("checkpoint") or "").strip()
        if not checkpoint:
            raise ValueError(f"decision without a checkpoint (use {FAMILY!r} for family level): {row['model']} {row['feature_or_flag']}")
        key = (row["model"], checkpoint, row["scope"], row["feature_or_flag"])
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
                    if checkpoint != FAMILY and key not in overrides:
                        continue
                    row = dict.fromkeys(columns, "not_recorded")
                    row.update({"model": model, "checkpoint": checkpoint, "scope": scope,
                                "feature_or_flag": flag,
                                "implemented": "unreviewed", "qualified": "unreviewed",
                                "selected": "unreviewed", "observed_used": "unreviewed",
                                "decision": "review_needed"})
                    row.update(overrides.pop(key, {}))
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
