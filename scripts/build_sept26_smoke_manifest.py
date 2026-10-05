#!/usr/bin/env python3
"""Rebuild the bounded Sept 26 smoke queue from the pinned local inventory."""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "docs/research/local-model-adapter-inventory-2026-09-26.md"
OUTPUT = ROOT / "docs/research/sept26-smoke-manifest.json"
ROOTS = {
    "M": Path.home() / "mlx-models",
    "T": Path("/Volumes/T7/models"),
    "U": Path.home() / "mlx-models",
    "H": Path.home() / ".cache/huggingface/hub",
    "A": Path.home() / "Library/Application Support/mlxuag-thinkingcap/models",
}
WHOLE_FAMILIES = {
    "agnes", "deepseek_v4", "gpt_oss", "gpt_oss_puzzle", "granitemoe_swa",
    "hy_v3", "llada", "llama", "muse_glimmer", "olmo_hils", "qwen2",
    "qwen3", "qwen3_moe", "diffusion_gemma", "nemotron3_diarization",
    "phi4mm", "qwen2_5_vl", "smolvlm",
}
DIRECT_RUNNERS = {
    "muse_glimmer": "mlx2.adapters.muse_glimmer_vision_candidate:MuseGlimmerVisionCandidate.generate_response",
    "llada": "mlx2.adapters.llada:LLaDADenoisingAdapter.generate",
    "diffusion_gemma": "mlx2.adapters.diffusion_gemma:DiffusionGemmaAdapter.generate_text",
    "nemotron3_diarization": "mlx2-diarize offline",
    "phi4mm": "mlx2.adapters.phi4mm_candidate:Phi4MMCandidate.generate",
    "qwen2_5_vl": "multimodal-owner:image-conditioned Qwen2.5-VL smoke",
    "smolvlm": "multimodal-owner:image-conditioned SmolVLM2 smoke",
}


def rows(text: str, start: str, end: str):
    chunk = text.split(start, 1)[1].split(end, 1)[0]
    for line in chunk.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 7 or not parts[1].startswith("`"):
            continue
        family, size, status, root, relative = parts[1:6]
        yield family.strip("`"), float(size), status, root.strip("`"), relative.strip("`")


def include(row) -> bool:
    family, _, _, _, relative = row
    return (
        family in WHOLE_FAMILIES
        or family == "laguna" and "Laguna-S-" in relative
        or family == "qwen3_5" and "Qwen3.5-4B" in relative
        or family == "qwen3_5_moe" and "122B-A10B" in relative
        or family == "qwen4_exp" and (
            "MTP-VLM" in relative
            or relative == "Qwen3.8-Flash-Next-Uncensored-MLX-Serve-4bit"
        )
    )


def config_hash(path: Path) -> str | None:
    config = path / "config.json"
    return hashlib.sha256(config.read_bytes()).hexdigest() if config.is_file() else None


def original_record(row, source):
    family, size, status, root, relative = row
    path = ROOTS[root] / relative
    if size > 128:
        category, runner = "over_capacity", None
        reason = "Apparent checkpoint size exceeds M5 128 GB; no bounded full-load attempt."
    elif family == "qwen4_exp":
        category, runner = "incomplete", None
        reason = "Current Flash-Next registry requires ple_rows.bin; this layout lacks it."
    elif family in DIRECT_RUNNERS:
        category, runner = "direct_media_run", DIRECT_RUNNERS[family]
        reason = "Use bounded modality-specific output; ordinary text probe does not cover this path."
    else:
        category, runner = "ordinary_run", "scripts/smoke_local_model.py"
        if family == "olmo_hils":
            runner += " --adapter-class mlx2.adapters.olmo_hils:OlmoHiLSAdapter"
            reason = "HiLS direct candidate has a custom landmark cache outside normal registry."
        elif size > 64:
            reason = "Check live-service memory headroom before the ordinary B1 probe."
        else:
            reason = "Run one bounded 12-token ordinary B1 probe."
    priority = (
        4 if category in {"over_capacity", "incomplete", "unsupported_lifecycle"}
        else 1 if size <= 4 and category == "ordinary_run"
        else 1 if size <= 8 and category == "direct_media_run"
        else 2 if size <= 32 else 3
    )
    return {
        "id": relative.replace("/", "__"), "family": family, "path": str(path),
        "apparent_checkpoint_gb": size, "inventory_status": status,
        "inventory_section": source, "exists": path.is_dir(),
        "config_sha256": config_hash(path), "category": category,
        "priority": priority, "expected_runner": runner, "reason": reason,
        "smoke_state": "not_run", "qualification_state": "unchanged",
    }


EXTRAS = (
    ("lfm2_vl", "LFM2.5-VL-3B", 6.25, "direct_media_run", 1,
     "multimodal-owner:image-conditioned LFM2.5-VL smoke", "New target outside original inventory."),
    ("lfm2_vl_dspark", "LFM2.5-VL-3B-DSpark", 0.56, "direct_media_run", 2,
     "mlx2.adapters.lfm25_vl.generate_candidate_dspark", "Auxiliary draft: target-bound offline verification only."),
    ("qwen4_exp_converted", "Qwen3.8-Flash-Next-Uncensored-MLX-Serve-4bit-MLX2", 100.0,
     "ordinary_run", 3, "scripts/smoke_local_model.py", "Converted target; check streamed PLE and live memory headroom."),
)
MEDIA_EXTRAS = (
    ("qwen_image21_source", "Qwen-Image-2.1", 31.0, "direct_media_run", 3,
     "scripts/qualify_qwen_image21.py", "Direct image source layout; short image output only."),
    ("qwen_image21_8bit", "Qwen-Image-2.1-MLX-8bit", 17.0, "direct_media_run", 2,
     "scripts/qualify_qwen_image21.py", "Prior 512px/20-step output receipt exists; recheck only if needed."),
    ("qwen_image21_uncensored", "Qwen-Image-2.1-Uncensored-MLX", 13.0, "direct_media_run", 2,
     "scripts/qualify_qwen_image21.py", "Direct image pipeline; short image output only."),
    ("qwen_image21_uncensored_4bit", "Qwen-Image-2.1-Uncensored-MLX-4bit", 3.7,
     "direct_media_run", 1, "scripts/qualify_qwen_image21.py", "Small direct image pipeline; short image output only."),
    ("ltx25_source", "LTX-2.5", 66.0, "unsupported_lifecycle", 4, None,
     "Source pipeline alone is not the receipted MLX bridge artifact."),
    ("ltx25_mlx", "LTX-2.5-MLX", 36.0, "direct_media_run", 3,
     "scripts/qualify_ltx25.py", "Receipted MLX bridge candidate; bounded video/audio output."),
)


def main():
    raw = INVENTORY.read_bytes()
    text = raw.decode()
    targets = list(rows(text, "### Target checkpoints (83)", "### Auxiliary"))
    media = list(rows(text, "### Media and speech checkpoints (8)", "## Weight-bearing"))
    assert len(targets) == 83 and len(media) == 8
    selected = [(row, "original_target") for row in targets if include(row)]
    selected += [(row, "original_media") for row in media if include(row)]
    assert len(selected) == 49, collections.Counter(row[0] for row, _ in selected)
    records = [original_record(row, source) for row, source in selected]
    for family, relative, size, category, priority, runner, reason in EXTRAS:
        path = ROOTS["M"] / relative
        records.append({
            "id": family, "family": family, "path": str(path),
            "apparent_checkpoint_gb": size, "inventory_status": "later_arrival",
            "inventory_section": "additional_candidate", "exists": path.is_dir(),
            "config_sha256": config_hash(path), "category": category,
            "priority": priority, "expected_runner": runner, "reason": reason,
            "smoke_state": "not_run", "qualification_state": "unchanged",
        })
    for family, relative, size, category, priority, runner, reason in MEDIA_EXTRAS:
        path = ROOTS["U"] / relative
        records.append({
            "id": family, "family": family, "path": str(path),
            "apparent_checkpoint_gb": size, "inventory_status": "later_arrival",
            "inventory_section": "additional_candidate", "exists": path.is_dir(),
            "config_sha256": config_hash(path), "category": category,
            "priority": priority, "expected_runner": runner, "reason": reason,
            "smoke_state": "not_run", "qualification_state": "unchanged",
        })
    assert len({record["path"] for record in records}) == len(records)
    assert all(record["exists"] for record in records), [
        record["path"] for record in records if not record["exists"]
    ]
    representative = {}
    for record in records:
        if record["category"] not in {"ordinary_run", "direct_media_run"}:
            continue
        key = record["family"]
        incumbent = representative.get(key)
        if incumbent is None or record["apparent_checkpoint_gb"] < incumbent["apparent_checkpoint_gb"]:
            representative[key] = record
    for record in records:
        record["representative_first"] = representative.get(record["family"]) is record
    manifest = {
        "schema": "mlx2.sept26-smoke-manifest.v1",
        "source_inventory": str(INVENTORY.relative_to(ROOT)),
        "source_inventory_sha256": hashlib.sha256(raw).hexdigest(),
        "scope": "49 original triage rows plus 9 later-arrival and direct-media candidates; no load result implied",
        "original_triage_count": 49, "additional_count": 9,
        "category_labels": {
            "ordinary_run": "ordinary run", "direct_media_run": "direct-media run",
            "over_capacity": "over-capacity", "incomplete": "incomplete",
            "unsupported_lifecycle": "unsupported lifecycle",
        },
        "counts": dict(sorted(collections.Counter(r["category"] for r in records).items())),
        "records": records,
    }
    OUTPUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{OUTPUT}: {len(records)} rows, {manifest['counts']}")


if __name__ == "__main__":
    main()
