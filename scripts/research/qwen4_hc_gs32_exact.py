#!/usr/bin/env python3
"""Static intake helpers for an exact Qwen4 HC group-size-32 candidate.

This module intentionally does not import MLX.  It provides three gates that
must pass before the deferred Metal probe is useful:

* inspect an artifact's config and safetensors index without loading tensors;
* pin the current mlx2 HC kernel strings before rewriting group pointers; and
* distinguish the exact composed-MLX law in oMLX #4245 from the faster,
  text-changing FP32-epilogue law in oMLX #4248.

The rewrite is process-local research machinery.  It is not a production
route and it does not change mlx2's group-size-64 default.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

OMLX_4245_HEAD = "6cf0312ffd8fc6bb5b9a7c1113e4595ca5a88065"
OMLX_4245_MERGE = "ce44a29a472c435da45b336d8cc1ab8da7d5e646"
OMLX_4248_HEAD = "b9b8bb381868489509c158cfdad022be36c4469d"
OMLX_4248_MERGE = "84e3b4370598e67da5b3085130f46b5dff3c25ee"

# origin/main 7a4800824367e32109a83f988fa254b1dd436d36.
SOURCE_HASHES = {
    "HEADER": "ea6144e27273e381c2aa7b62b32b4c84c3c2e81f04f6e03ff3f99c49385f8238",
    "NORM_DOWN_SOURCE": "8df40f3f8ac4ab5c47c585f07ddfa0f7e94a6587e67c5640e654b60600cec20d",
    "UP_MIX_SOURCE": "3dd95389ddeb12a54a5dab5146bedac18eebc3b7b5471ca469d89d72faeeb3c6",
}

HC_PARTS = (
    "input_mix_weight_down",
    "input_mix_weight_up",
    "block_inject_weight",
)
_HC_TENSOR = re.compile(
    r"^(?P<module>.+(?:attn_hyper_connection|mlp_hyper_connection|"
    r"hyper_connection_mixer))\.(?P<projection>input_mix_weight_down|"
    r"input_mix_weight_up|block_inject_weight)\."
    r"(?P<part>weight|scales|biases)$"
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def source_constants(path: Path) -> dict[str, str]:
    """Read literal kernel strings without importing the MLX module."""

    wanted = set(SOURCE_HASHES)
    values: dict[str, str] = {}
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            value = ast.literal_eval(node.value)
            if not isinstance(value, str):
                raise TypeError(f"{target.id} is no longer a string literal")
            values[target.id] = value
    missing = wanted - values.keys()
    if missing:
        raise ValueError(f"missing HC source literals: {sorted(missing)}")
    return values


def verify_source_pin(values: dict[str, str]) -> dict[str, str]:
    observed = {name: _sha(values[name]) for name in SOURCE_HASHES}
    drift = {
        name: {"expected": SOURCE_HASHES[name], "observed": observed[name]}
        for name in SOURCE_HASHES
        if observed[name] != SOURCE_HASHES[name]
    }
    if drift:
        raise ValueError(f"HC source drift; re-audit before rewriting: {drift}")
    return observed


def exact_gs32_sources(values: dict[str, str]) -> dict[str, str]:
    """Parameterize only group addressing, retaining mlx2's operation order."""

    verify_source_pin(values)
    header = values["HEADER"]
    norm = values["NORM_DOWN_SOURCE"]
    up = values["UP_MIX_SOURCE"]

    old = "template <typename T, int BITS, typename P>\ninline float hcd_wide_row("
    new = "template <typename T, int BITS, int GS, typename P>\ninline float hcd_wide_row("
    if header.count(old) != 1:
        raise ValueError("unexpected hcd_wide_row declaration count")
    header = header.replace(old, new)
    if header.count("g * 64 + sc * 8") != 1 or header.count("sc < 8") != 1:
        raise ValueError("unexpected qmv_wide group traversal")
    header = header.replace("g * 64 + sc * 8", "g * GS + sc * 8")
    header = header.replace("sc < 8", "sc < GS / 8")

    def projection(source: str, tags: tuple[str, ...]) -> str:
        for tag in tags:
            needle = f"hcd_wide_row<T, {tag}>("
            if source.count(needle) != 1:
                raise ValueError(f"unexpected {tag} wide-row call count")
            source = source.replace(needle, f"hcd_wide_row<T, {tag}, GS>(")
        source = source.replace(" / 64", " / GS").replace("64 / ", "GS / ")
        if " / 64" in source or "64 / " in source:
            raise ValueError("unparameterized group addressing remains")
        return source

    return {
        "HEADER": header,
        "NORM_DOWN_SOURCE": projection(norm, ("DB", "IB")),
        "UP_MIX_SOURCE": projection(up, ("UB",)),
    }


def pointer_geometry(*, width: int, bits: int, group_size: int, fast: bool) -> dict:
    """CPU-only proof that the candidate's scale pointer covers each value.

    ``fast`` models qmv_fast's per-lane vector (16/8 values for q4/q8);
    false models plain qmv (8/4). qmv_wide directly enumerates whole groups,
    so it is covered by the reported ``wide_subchunks_per_group`` invariant.
    """

    if bits not in {4, 8}:
        raise ValueError("mlx2 HC candidate supports q4 and q8 only")
    values_per_lane = (32 // bits) * (2 if fast else 1)
    block = 32 * values_per_lane
    if width % group_size or group_size % values_per_lane:
        raise ValueError("geometry does not have whole quantization groups")
    mismatches = []
    cells = 0
    for base in range(0, width, block):
        for lane in range(32):
            observed_group = base // group_size + lane // (group_size // values_per_lane)
            for offset in range(values_per_lane):
                k = base + lane * values_per_lane + offset
                if k >= width:  # plain qmv's guarded final block
                    continue
                expected_group = k // group_size
                cells += 1
                if observed_group != expected_group:
                    mismatches.append(
                        {"k": k, "observed": observed_group, "expected": expected_group}
                    )
    return {
        "width": width,
        "bits": bits,
        "group_size": group_size,
        "law": "qmv_fast" if fast else "qmv",
        "values_per_lane": values_per_lane,
        "cells": cells,
        "mismatches": mismatches,
        "wide_subchunks_per_group": group_size // 8,
    }


def _quantization(config: dict) -> dict:
    value = config.get("quantization", {})
    return value if isinstance(value, dict) else {}


def _projection_spec(config: dict, path: str) -> dict:
    quant = _quantization(config)
    candidates = (
        path,
        path.removeprefix("model."),
        path.removeprefix("language_model."),
        path.replace("model.language_model.", "language_model.", 1),
    )
    override = next(
        (quant[key] for key in candidates if isinstance(quant.get(key), dict)),
        {},
    )
    return {
        "bits": override.get("bits", quant.get("bits")),
        "group_size": override.get("group_size", quant.get("group_size")),
        "mode": override.get("mode", quant.get("mode", "affine")),
    }


def inspect_mapping(config: dict, weight_map: dict[str, str]) -> dict:
    """Describe complete HC modules and the exact-gs32 metadata gate."""

    tensors: dict[str, dict[str, set[str]]] = {}
    for name in sorted(weight_map):
        match = _HC_TENSOR.match(name)
        if match:
            tensors.setdefault(match.group("module"), {}).setdefault(
                match.group("projection"), set()
            ).add(match.group("part"))

    modules = []
    for module_name, projections in sorted(tensors.items()):
        described = {}
        reasons = []
        for projection in HC_PARTS:
            parts = projections.get(projection)
            if parts is None:
                if projection != "block_inject_weight":
                    reasons.append(f"missing {projection}")
                continue
            path = f"{module_name}.{projection}"
            quantized = bool(parts & {"scales", "biases"})
            complete = parts == ({"weight", "scales", "biases"} if quantized else {"weight"})
            spec = _projection_spec(config, path) if quantized else {
                "bits": None,
                "group_size": None,
                "mode": "dense",
            }
            described[projection] = {
                "parts": sorted(parts),
                "quantized": quantized,
                "complete": complete,
                **spec,
            }
            if not complete:
                reasons.append(f"partial {projection}")
            if projection in {"input_mix_weight_down", "input_mix_weight_up"} and not quantized:
                reasons.append(f"dense {projection}")
            if quantized and spec["group_size"] != 32:
                reasons.append(f"{projection} group_size={spec['group_size']!r}")
            if quantized and spec["bits"] not in {4, 8}:
                reasons.append(f"{projection} bits={spec['bits']!r}")
            if quantized and spec["mode"] != "affine":
                reasons.append(f"{projection} mode={spec['mode']!r}")
        modules.append(
            {
                "path": module_name,
                "projections": described,
                "metadata_eligible": not reasons,
                "decline_reasons": reasons,
                "native_checks_remaining": [
                    "tensor shapes and uint32 packed weights",
                    "bf16 affine scales/biases",
                    "hc_norm bf16/fp32 layout and exact norm convention",
                    "MLX kernel-law selection and composed-bit equality",
                ],
            }
        )

    eligible = [item for item in modules if item["metadata_eligible"]]
    return {
        "model_type": config.get("model_type"),
        "modules": modules,
        "module_count": len(modules),
        "eligible_module_count": len(eligible),
        "all_modules_metadata_eligible": bool(modules) and len(eligible) == len(modules),
        "artifact_gate": "pass" if modules and len(eligible) == len(modules) else "blocked",
    }


def inspect_artifact(path: Path) -> dict:
    root = path.expanduser().resolve()
    config = json.loads((root / "config.json").read_text())
    index = json.loads((root / "model.safetensors.index.json").read_text())
    return {"artifact": str(root), **inspect_mapping(config, index.get("weight_map", {}))}


def describe(source: Path, artifacts: list[Path]) -> dict:
    values = source_constants(source)
    hashes = verify_source_pin(values)
    rewritten = exact_gs32_sources(values)
    return {
        "schema": "mlx2.qwen4-hc-gs32-exact-static.v1",
        "mlx2_source": str(source.resolve()),
        "source_hashes": hashes,
        "candidate_hashes": {name: _sha(value) for name, value in rewritten.items()},
        "cpu_geometry": [
            pointer_geometry(width=width, bits=bits, group_size=32, fast=fast)
            for width, fast in ((10240, True), (320, False))
            for bits in (4, 8)
        ],
        "upstream": {
            "exact_law": {"pr": 4245, "head": OMLX_4245_HEAD, "merge": OMLX_4245_MERGE},
            "non_reference_law": {"pr": 4248, "head": OMLX_4248_HEAD, "merge": OMLX_4248_MERGE},
            "related_open_issue": {
                "issue": 4209,
                "finding": (
                    "gs32 canonical execution changed greedy output between MTP on/off; "
                    "#4248 made them agree by moving both onto its text-changing fused law, "
                    "but the canonical-path cause was not isolated"
                ),
            },
        },
        "law": {
            "selected": "mlx2 composed MLX operation order; group addressing only is parameterized",
            "excluded": "oMLX #4248 FP32 epilogues and norm-storage policy; its PR reports changed text",
        },
        "route": {
            "production_changed": False,
            "default_enabled": False,
            "artifact_gate_required": True,
            "qualified": False,
            "selected": False,
            "observed_used": False,
        },
        "artifacts": [inspect_artifact(path) for path in artifacts],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("src/mlx2/runtime/models/qwen4_hc_decode.py"),
    )
    parser.add_argument("artifacts", nargs="*", type=Path)
    args = parser.parse_args()
    print(json.dumps(describe(args.source, args.artifacts), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
