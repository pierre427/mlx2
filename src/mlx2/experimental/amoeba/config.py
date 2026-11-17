"""Fixed data defaults (mapping-workflow section 4) and compute profiles.

The data rules never change with the profile. They change only when a box misses its target, and then
only as a better-data fix, recorded with the reason (``--data-override KEY=VALUE --override-reason``).
A profile changes compute only (steps, sequence counts, prompts per box, probe subsampling); every
receipt names its profile, and a ``quick`` result is a pipeline check, never a box-quality claim.
"""
from __future__ import annotations

import math

DATA_DEFAULTS = {
    # rule 1: windows from seeded random positions across the whole document, several per long document
    "window_tokens": 128,            # rule 4
    "windows_per_doc": 2,
    "window_seed": 0,
    # rule 2: drop an opening shared by >= 2 rows and > 20% of a topic's rows
    "strip_prefix": 0.2,
    # rule 3: code documents start at their first line of code (domains of these taxonomy branches are "code")
    "code_cut": True,
    "code_topic_prefixes": ["prog."],
    # rule 5: on-policy continuations of a 96-token mid-document passage, 64 new tokens, 1 greedy + 3 at T=0.7,
    # recorded seeds; the carve scores the continuation plus the last 32 prompt tokens
    "gen_prompt_tokens": 96,
    "gen_new_tokens": 64,
    "gen_samples": 3,
    "gen_temperature": 0.7,
    "gen_seed0": 7000,
    "keep_prompt": 32,
    # rule 6: held text is never model-generated and never chooses anything (enforced in the box set and carve)
    # rule 7: boxes only from the taxonomy and from autodetected clusters (no hand-picked task boxes)
}

VERDICT = {
    "held_target_nats": 0.05,         # pass: core + box within +0.05 nats of the full model on held text
    "pooling": "pooled over held rows: probe windows + 1 greedy + 3 T=0.7 per prompt (1:3 by row count)",
    "val_target_nats": 0.03,          # +0.05 minus the measured val->held gap; +0.03 until the gap is measured
    "random_control_ratio": 0.5,      # a box must beat a same-size random box clearly: held dCE <= 0.5 x random's
    "random_margin_floor_nats": 0.01, # ... and the random box worse by at least this (a whole-pool box cannot pass)
    "core_covers_nats": 0.05,         # diagnostic only: core alone within this on val is reported, never skips a box
}

# Section 7: the gradient direction check guards the code. It refuses only on real faults: correlation below 0.9,
# a size ratio outside 0.67-1.5, or no clean test directions. It uses bf16-safe steps (2^-4 to 2^-1) and never
# gates on the relative error (bf16 noise). These are the values the FN runs use (coordinator 2026-11-16).
GRAD_GATE = ["--dir-grad-eps", "0.0625,0.125,0.25,0.5", "--dir-grad-min-corr", "0.9", "--dir-grad-tol", "1e9",
             "--dir-grad-norm-ratio", "0.67,1.5", "--dir-grad-k", "8",
             "--dir-grad-note", "mapping-workflow s7: catastrophic-only; the exact masked evaluation is the final word"]

PARTITION_RULES = {
    "core_or_3plus": "core", "exactly_2": "cross-box tile for that pair", "exactly_1": "that box",
    "none": "remainder tile, grouped by cluster",
}

# core_targets "" : no bisect on held (held only checks); the export cut is chosen on the carve's spare split
PROFILES = {
    "full": {
        "map_bs": 8, "map_probes_per_node": 0, "ve_per_layer": 64,
        "core_steps": 150, "core_bs": 2, "core_lr": 0.1, "core_n_mean": 64, "core_n_fit": 480, "core_n_held": 24,
        "core_targets": "", "core_export_target": 0.18, "core_gen_free_per_domain": 72,
        "core_gen_held_per_domain": 8,
        "box_steps": 60, "box_bs": 2, "box_lr": 0.1, "box_n_mean": 12, "box_n_fit": 144, "box_bisect_iters": 12,
        "box_bisect_tol": 0.005, "box_min_fit": 32, "box_carve_prompts": 16, "box_val_prompts": 8,
        "box_held_prompts": 16, "box_min_carve": 52, "box_min_val": 16, "box_min_held": 16,
        "cluster_box_docs": 64, "cluster_box_held": 24,
        "gate_tokens_per_window": 32, "gate_windows_per_box": 64, "gate_negatives": 512,
        "val_target_nats": None,
    },
    "quick": {
        "map_bs": 8, "map_probes_per_node": 6, "ve_per_layer": 16,
        "core_steps": 12, "core_bs": 2, "core_lr": 0.1, "core_n_mean": 12, "core_n_fit": 48, "core_n_held": 12,
        "core_targets": "", "core_export_target": 0.18, "core_gen_free_per_domain": 6,
        "core_gen_held_per_domain": 3,
        "box_steps": 8, "box_bs": 2, "box_lr": 0.1, "box_n_mean": 4, "box_n_fit": 24, "box_bisect_iters": 8,
        "box_bisect_tol": 0.01, "box_min_fit": 16, "box_carve_prompts": 3, "box_val_prompts": 2,
        "box_held_prompts": 3, "box_min_carve": 16, "box_min_val": 6, "box_min_held": 6,
        "cluster_box_docs": 24, "cluster_box_held": 8,
        "gate_tokens_per_window": 16, "gate_windows_per_box": 24, "gate_negatives": 96,
        "val_target_nats": None,
    },
}


def cluster_k(total_units: int) -> int:
    """Section 5 step 1: k ~ 50-130 by model size (50 at ~0.1M units, 130 at >= ~0.6M)."""
    return int(min(130, max(50, round(math.sqrt(max(1, total_units)) / 6))))


def parse_overrides(items: list[str]) -> dict:
    out = {}
    for it in items or []:
        k, sep, v = it.partition("=")
        if not sep:
            raise SystemExit(f"override {it!r}: expected KEY=VALUE")
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = {"true": True, "false": False}.get(v.lower(), v)
    return out


def resolve(profile: str, compute_overrides: dict, data_overrides: dict, reason: str) -> dict:
    if profile not in PROFILES:
        raise SystemExit(f"unknown profile {profile!r} ({', '.join(PROFILES)})")
    comp = dict(PROFILES[profile])
    bad = set(compute_overrides) - set(comp)
    if bad:
        raise SystemExit(f"unknown compute settings {sorted(bad)}")
    comp.update(compute_overrides)
    data = dict(DATA_DEFAULTS)
    bad = set(data_overrides) - set(data)
    if bad:
        raise SystemExit(f"unknown data rules {sorted(bad)}")
    if data_overrides and not reason:
        raise SystemExit("a data-rule change needs --override-reason (mapping-workflow s4: only as a better-data fix "
                         "for a box that missed its target)")
    data.update(data_overrides)
    verdict = dict(VERDICT)
    if comp.get("val_target_nats") is not None:       # the measured val->held gap is known (next-round.json): +0.05 minus it
        verdict["val_target_nats"] = float(comp["val_target_nats"])
    return {"profile": profile, "compute": comp, "data": data, "verdict": verdict, "grad_gate": list(GRAD_GATE),
            "partition_rules": dict(PARTITION_RULES),
            "data_overrides": {"values": data_overrides, "reason": reason} if data_overrides else None}
