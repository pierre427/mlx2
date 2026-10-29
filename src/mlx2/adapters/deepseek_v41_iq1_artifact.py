# SPDX-License-Identifier: MIT
"""Standalone compact IQ1_S artifact admission without either source checkpoint."""

import math

from .deepseek_v41_mlx_artifact import (
    _inspect_artifact,
    _int,
    _sha,
    _shape,
    _storage_inventory,
    asset_path,
)

MODEL_TYPE = "deepseek41_iq1"
FORMAT = "mlx2-v41-iq1s-v1"
SCHEMA = "mlx2.deepseek-v41-iq1.v1"
MANIFEST_FILE = "mlx2-v41-iq1-manifest.json"
NUMERIC_PROFILE = "native_iq1_s_packed"


def compact_storage_inventory(tensors, files):
    """Admit flat, independently owned IQ1_S planes and unchanged dense/PLE."""
    if any(not isinstance(row.get("storage"), dict) for row in tensors.values()):
        raise ValueError("invalid compact tensor storage record")
    compact = {
        name: row
        for name, row in tensors.items()
        if isinstance(row.get("storage"), dict)
        and row["storage"].get("kind") == "expert-iq1_s"
    }
    if not compact:
        raise ValueError("compact V4.1 artifact has no IQ1_S experts")
    expert_files, used, groups = set(), {}, {}
    for name, row in compact.items():
        shape, logical = _shape(row.get("shape")), _shape(row.get("numpy_shape"))
        if (
            len(shape) != 3
            or logical != shape[::-1]
            or row.get("ggml_type") != 19
            or shape[0] % 256
            or shape[1] % 256
            or not _int(row.get("offset"))
            or not _int(row.get("nbytes"), 1)
            or row.get("nbytes") != math.prod(shape) // 256 * 50
        ):
            raise ValueError("invalid compact IQ1_S source geometry")
        prefix, suffix = name.split(".ffn_", 1) if ".ffn_" in name else ("", "")
        plane = suffix.removesuffix("_exps.weight")
        storage = row["storage"]
        names, keys = storage.get("expert_files"), storage.get("keys")
        if (
            not prefix
            or plane not in ("gate", "up", "down")
            or not isinstance(names, list)
            or len(names) != logical[0]
            or not all(isinstance(path, str) for path in names)
            or len(set(names)) != len(names)
            or keys != {"data": plane + ".data"}
        ):
            raise ValueError("invalid compact per-expert file inventory")
        groups.setdefault(prefix, {})[plane] = names
        count = row["nbytes"] // logical[0]
        for path in names:
            record = files.get(path)
            tensor = (
                record.get("tensors", {}).get(keys["data"])
                if isinstance(record, dict)
                else None
            )
            if (
                not isinstance(tensor, dict)
                or tensor.get("dtype") != "U8"
                or tensor.get("shape") != [count]
            ):
                raise ValueError(
                    "compact IQ1_S plane is not the exact flat byte extent"
                )
            if keys["data"] in used.setdefault(path, set()):
                raise ValueError("compact expert plane is multiply owned")
            used[path].add(keys["data"])
            expert_files.add(path)
    for group in groups.values():
        if (
            set(group) != {"gate", "up", "down"}
            or not group["gate"] == group["up"] == group["down"]
        ):
            raise ValueError("compact expert projections disagree on file ownership")
    for path in expert_files:
        if set(files[path].get("tensors", {})) != {"gate.data", "up.data", "down.data"}:
            raise ValueError("compact expert file must own exactly three IQ1_S planes")
    remaining = {name: row for name, row in tensors.items() if name not in compact}
    if any(
        row.get("storage", {}).get("kind") == "expert-affine"
        for row in remaining.values()
    ):
        raise ValueError("compact artifact cannot mix affine expert storage")
    kinds = _storage_inventory(
        remaining,
        {name: row for name, row in files.items() if name not in expert_files},
    )
    return {**kinds, **dict.fromkeys(expert_files, "expert")}


def inspect_artifact(model_path, *, verify_hashes=False):
    result = _inspect_artifact(
        model_path,
        manifest_file=MANIFEST_FILE,
        schema=SCHEMA,
        model_type=MODEL_TYPE,
        weight_format=FORMAT,
        numeric_profile=NUMERIC_PROFILE,
        storage_inventory=compact_storage_inventory,
        verify_hashes=verify_hashes,
    )
    parent = result["manifest"].get("source_affine")
    if (
        not isinstance(parent, dict)
        or not _sha(parent.get("derived_identity_sha256"))
        or not _sha(parent.get("fingerprint"))
    ):
        raise ValueError("compact artifact lacks its source affine manifest identity")
    # COW clones have independent inodes. Hardlinks would invalidate the
    # reference artifact's ctime seal and are not a supported sharing scheme.
    for path, kind in result["file_kinds"].items():
        if (
            kind in ("dense", "engram")
            and asset_path(result["root"], path).stat().st_nlink != 1
        ):
            raise ValueError(
                "compact dense/PLE assets must be independent COW clones, not hardlinks"
            )
    result["source_affine_fingerprint"] = parent["fingerprint"]
    result["source_affine_derived_identity_sha256"] = parent["derived_identity_sha256"]
    result["source_affine_required"] = False
    return result
