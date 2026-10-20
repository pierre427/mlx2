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

import ast
import csv
import re
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
}

# Features that are selected through execution policy or adapter configuration
# rather than a dedicated command-line switch. This is a review vocabulary,
# not a statement that every registered model implements each feature.
FEATURES = (
    "apc_v2", "apc_interior_checkpoints", "apc_junction_checkpoints",
    "apc_rolling_checkpoints", "apc_session_persistence", "tools",
    "vision_input", "video_input", "audio_input", "native_mtp",
    "mtp_ordinary_handoff", "adaptive_mtp_depth", "self_mtp_copy_draft",
    "external_draft", "flat_segmented_ast", "ast_speculative_batching",
    "packed_prefill_serial_equivalence", "prompt_lookup", "prompt_lookup_target_s1",
    "decode_first", "prefill_scheduling", "mixed_prefill_decode",
    "varlen_paged_attention", "row_exact_verify", "tensorfold_tree15",
    "fused_gdn", "fused_gdn_batch_verify", "moe_rhs_pad",
    "moe_nax_gather", "moe_routed_decode", "moe_topk_fold",
    "moe_window_row_exact", "attention_window_row_exact",
    "progressive_ast_rootless_fallback",
    "qsa_nax_prefill", "qsa_nax_batched", "qsa_nax_decode",
    "invariant_prefill", "quantized_kv", "approximate_kv",
    "host_memory_signals", "host_available_floor", "weight_streaming",
    "constrained_tool_grammar", "strict_json_schema", "thinking_guard",
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
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value.startswith("--"):
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
        strong = row["qualified"] == "yes" or row["observed_used"] == "yes"
        if strong and checkpoint == FAMILY:
            raise ValueError(f"qualified/observed decision must name its checkpoint: {key}")
        if strong and row.get("source_revision", "").strip() in UNRECORDED:
            raise ValueError(f"qualified/observed decision must name its mlx2 commit: {key}")
        overrides[key] = row
    return overrides


def main() -> None:
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
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
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
    if overrides:
        raise ValueError(f"reviewed decisions absent from inventory: {sorted(overrides)}")
    print(f"{OUT}: {len(models)} model variants x {len(flags)} switches = {len(models) * len(flags)} review cells; {len(decisions)} reviewed")


if __name__ == "__main__":
    main()
