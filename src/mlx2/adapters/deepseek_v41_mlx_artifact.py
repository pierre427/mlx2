# SPDX-License-Identifier: MIT
"""GPU-free admission of a complete, standalone V4.1 affine checkpoint.

The original GGUF is provenance only. Every serving byte is addressed inside
this directory. See provenance/deepseek-v41-mlx-native.json.
"""

import hashlib
import itertools
import json
import math
import stat
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

from ..runtime.immutable_weight_pages import same_stamp

from .deepseek_v41_artifact import (
    REFERENCE_FILE,
    TOKENIZER_SHA256,
    WEIGHT_BYTES,
    WEIGHT_SHA256,
    validate_header,
)

MODEL_TYPE = "deepseek41_mlx"
FORMAT = "mlx2-v41-affine-v1"
SCHEMA = "mlx2.deepseek-v41-native.v1"
MANIFEST_FILE = "mlx2-v41-native-manifest.json"
NUMERIC_PROFILE = "native_f32_qmm"
_DTYPES = {"U32": 4, "F32": 4, "F16": 2, "BF16": 2, "I8": 1, "U8": 1}


def file_stamp(st):
    return {
        "dev": st.st_dev,
        "ino": st.st_ino,
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "ctime_ns": st.st_ctime_ns,
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate native manifest key: " + key)
        result[key] = value
    return result


def _json(raw):
    def invalid(value):
        raise ValueError("nonfinite native manifest JSON: " + value)

    return json.loads(raw, object_pairs_hook=_unique_object, parse_constant=invalid)


def _sha(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _int(value, minimum=0):
    return type(value) is int and value >= minimum


def _shape(value):
    if (
        not isinstance(value, (list, tuple))
        or not value
        or not all(_int(n, 1) for n in value)
    ):
        raise ValueError("invalid native tensor shape")
    return tuple(value)


def _sealed_stat(row):
    first, second = row.get("stat"), row.get("source_stat")
    if first is not None and second is not None and first != second:
        raise ValueError("conflicting native file stat seals")
    seal = first if first is not None else second
    if (
        not isinstance(seal, dict)
        or set(seal) != {"dev", "ino", "size", "mtime_ns", "ctime_ns"}
        or not all(_int(v) for v in seal.values())
    ):
        raise ValueError("invalid native file stat seal")
    return seal


def asset_path(root, relative):
    """Refuse traversal and symlink components before opening an asset."""
    if not isinstance(relative, str) or "\\" in relative or not relative:
        raise ValueError("invalid native artifact asset path")
    parts = PurePosixPath(relative)
    if (
        parts.is_absolute()
        or any(p in (".", "..") for p in parts.parts)
        or str(parts) != relative
    ):
        raise ValueError("native artifact asset escapes its root")
    path = Path(root)
    for part in parts.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("native artifact asset is a symlink: " + relative)
    if not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
        raise ValueError("native artifact asset is not regular: " + relative)
    return path


def _digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_layout(row):
    tensors = row.get("tensors", {})
    if not isinstance(tensors, dict):
        raise ValueError("invalid native file tensor inventory")  # noqa: TRY004 - malformed artifact admission
    if not tensors:
        return
    header = row.get("header_bytes")
    if not _int(header, 8) or header > row["size"]:
        raise ValueError("invalid native safetensors header extent")
    spans = []
    for key, tensor in tensors.items():
        if not isinstance(key, str) or not isinstance(tensor, dict):
            raise ValueError("invalid native safetensors tensor record")  # noqa: TRY004 - malformed artifact admission
        shape = _shape(tensor.get("shape"))
        dtype, offsets = tensor.get("dtype"), tensor.get("data_offsets")
        if (
            dtype not in _DTYPES
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(_int(n) for n in offsets)
            or offsets[1] <= offsets[0]
            or offsets[1] - offsets[0] != math.prod(shape) * _DTYPES[dtype]
            or header + offsets[1] > row["size"]
        ):
            raise ValueError("invalid native safetensors tensor extent")
        spans.append(tuple(offsets))
    spans.sort()
    if (
        spans[0][0] != 0
        or any(left[1] != right[0] for left, right in itertools.pairwise(spans))
        or header + spans[-1][1] != row["size"]
    ):
        raise ValueError("native safetensors payload has gaps/overlap/unbound bytes")


def _storage_inventory(tensors, files):
    """Check every source tensor has the complete native storage law."""
    used = {}
    kinds, expert_groups = {}, {}

    def claim(file, key, shape, dtypes):
        if file not in files or not isinstance(key, str):
            raise ValueError("native tensor references an absent file/key")
        row = files[file].get("tensors", {}).get(key)
        if (
            not isinstance(row, dict)
            or tuple(row.get("shape", ())) != tuple(shape)
            or row.get("dtype") not in dtypes
        ):
            raise ValueError("native tensor storage geometry changed: " + key)
        if key in used.setdefault(file, set()):
            raise ValueError("native tensor storage is multiply owned: " + key)
        used[file].add(key)

    for name, tensor in tensors.items():
        shape = _shape(tensor.get("shape"))
        logical = _shape(tensor.get("numpy_shape"))
        if (
            logical != shape[::-1]
            or not _int(tensor.get("nbytes"), 1)
            or not _int(tensor.get("offset"))
        ):
            raise ValueError("native source tensor layout changed: " + name)
        storage, ggml = tensor.get("storage", {}), tensor.get("ggml_type")
        if not isinstance(storage, dict):
            raise ValueError("invalid native tensor storage record")  # noqa: TRY004 - malformed artifact admission
        source_formats = {0: (1, 4), 1: (1, 2), 8: (32, 34), 19: (256, 50), 24: (1, 1)}
        block = source_formats.get(ggml)
        if (
            block is None
            or math.prod(shape) % block[0]
            or tensor["nbytes"] != math.prod(shape) // block[0] * block[1]
        ):
            raise ValueError("native source tensor byte extent changed: " + name)
        kind = storage.get("kind")
        if kind == "engram":
            file = storage.get("file")
            if (
                ggml != 24
                or len(shape) != 2
                or shape[0] != 264
                or storage.get("row_bytes") != 264
                or storage.get("rows") != logical[0]
                or file not in files
                or files[file]["size"] != logical[0] * 264
                or files[file].get("tensors")
                or file in kinds
            ):
                raise ValueError("invalid standalone native Engram table")
            kinds[file] = kind
        elif kind == "raw":
            if ggml not in (0, 1):
                raise ValueError("unsupported native raw source tensor")
            file = storage.get("file")
            claim(
                file,
                storage.get("key"),
                logical,
                {"F32"} if ggml == 0 else {"F16", "F32"},
            )
            if kinds.setdefault(file, "dense") != "dense":
                raise ValueError("native dense file ownership overlap")
        elif kind in ("affine", "expert-affine"):
            expert = kind == "expert-affine"
            bits = 2 if expert else 8
            if (
                ggml != (19 if expert else 8)
                or len(shape) != (3 if expert else 2)
                or storage.get("bits") != bits
                or storage.get("group_size") != 32
                or storage.get("mode") != "affine"
                or logical[-1] % 32
            ):
                raise ValueError("unsupported native affine recipe: " + name)
            names = storage.get("expert_files") if expert else [storage.get("file")]
            if (
                not isinstance(names, list)
                or not all(isinstance(n, str) for n in names)
                or len(names) != (logical[0] if expert else 1)
                or len(set(names)) != len(names)
            ):
                raise ValueError("native expert inventory is incomplete")
            keys = storage.get("keys")
            if not isinstance(keys, dict) or set(keys) != {
                "qweight",
                "scales",
                "biases",
            }:
                raise ValueError("native affine tensor keys changed")
            if expert:
                prefix, suffix = name.split(".ffn_", 1) if ".ffn_" in name else ("", "")
                plane = suffix.removesuffix("_exps.weight")
                if not prefix or plane not in ("gate", "up", "down"):
                    raise ValueError("native expert source name changed")
                expert_groups.setdefault(prefix, {})[plane] = names
            matrix = logical[1:] if expert else logical
            for file in names:
                claim(
                    file,
                    keys["qweight"],
                    (*matrix[:-1], matrix[-1] // (32 // bits)),
                    {"U32"},
                )
                for key in ("scales", "biases"):
                    claim(file, keys[key], (*matrix[:-1], matrix[-1] // 32), {"F32"})
                previous = kinds.setdefault(file, "expert" if expert else "dense")
                if previous != ("expert" if expert else "dense"):
                    raise ValueError("native expert and dense file ownership overlap")
        else:
            raise ValueError("unsupported native tensor storage: " + name)
    for name, row in files.items():
        if set(row.get("tensors", {})) != used.get(name, set()):
            raise ValueError("unbound native file tensors: " + name)
    for group in expert_groups.values():
        if (
            set(group) != {"gate", "up", "down"}
            or not group["gate"] == group["up"] == group["down"]
        ):
            raise ValueError("native per-expert projection files disagree")
    return kinds


def inspect_artifact(model_path, *, verify_hashes=False):
    return _inspect_artifact(
        model_path,
        manifest_file=MANIFEST_FILE,
        schema=SCHEMA,
        model_type=MODEL_TYPE,
        weight_format=FORMAT,
        numeric_profile=NUMERIC_PROFILE,
        storage_inventory=_storage_inventory,
        verify_hashes=verify_hashes,
    )


def _inspect_artifact(
    model_path,
    *,
    manifest_file,
    schema,
    model_type,
    weight_format,
    numeric_profile,
    storage_inventory,
    verify_hashes=False,
):
    """Stat-sealed startup inspection; full payload hashing is explicit."""
    if type(verify_hashes) is not bool:
        raise ValueError("native hash verification must be boolean")
    root = Path(model_path).expanduser().resolve()
    manifest_path = asset_path(root, manifest_file)
    before = file_stamp(manifest_path.stat())
    if before["size"] > 128 << 20:
        raise ValueError("native manifest exceeds bounded metadata size")
    raw = manifest_path.read_bytes()
    manifest = _json(raw)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != schema
        or manifest.get("model_type") != model_type
        or manifest.get("format") != weight_format
        or manifest.get("numeric_profile") != numeric_profile
        or manifest.get("finished") is not True
    ):
        raise ValueError("native V4.1 conversion is unfinished or incompatible")
    source = manifest.get("source", {})
    if (
        not isinstance(source, dict)
        or source.get("gguf_sha256") != WEIGHT_SHA256
        or source.get("size") != WEIGHT_BYTES
        or not _sha(source.get("fingerprint"))
    ):
        raise ValueError("native V4.1 source identity changed")
    recipe = manifest.get("recipe")
    recipe_hash = hashlib.sha256(
        json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if not isinstance(recipe, dict) or manifest.get("recipe_sha256") != recipe_hash:
        raise ValueError("native V4.1 recipe identity changed")
    tensors, files, metadata = (
        manifest.get("tensors"),
        manifest.get("files"),
        manifest.get("metadata"),
    )
    if (
        not isinstance(tensors, dict)
        or not isinstance(files, dict)
        or not isinstance(metadata, dict)
    ):
        raise ValueError("native V4.1 inventory is absent")  # noqa: TRY004 - malformed artifact admission
    if any(
        not isinstance(name, str) or not isinstance(row, dict)
        for name, row in tensors.items()
    ):
        raise ValueError("invalid native source tensor descriptions")
    descriptions = {
        name: SimpleNamespace(
            shape=_shape(row.get("shape")), ggml_type=row.get("ggml_type")
        )
        for name, row in tensors.items()
    }
    args = validate_header(metadata, descriptions)
    for name, row in files.items():
        if (
            name == manifest_file
            or not isinstance(row, dict)
            or not _int(row.get("size"), 1)
            or not _sha(row.get("sha256"))
        ):
            raise ValueError("invalid native artifact file record")
        path = asset_path(root, name)
        stamp = file_stamp(path.stat(follow_symlinks=False))
        if not same_stamp(stamp, _sealed_stat(row)) or stamp["size"] != row["size"]:
            raise ValueError("native converted file changed after sealing: " + name)
        _file_layout(row)
        if (verify_hashes or (not row.get("tensors") and path.suffix == ".json")) and (
            _digest(path) != row["sha256"] or not same_stamp(file_stamp(path.stat()), stamp)
        ):
            raise ValueError("native converted file hash changed: " + name)
    kinds = storage_inventory(tensors, files)
    if (
        REFERENCE_FILE not in files
        or files[REFERENCE_FILE]["sha256"] != TOKENIZER_SHA256
        or "config.json" not in files
    ):
        raise ValueError("native artifact lacks its pinned tokenizer/config")
    config = _json(asset_path(root, "config.json").read_bytes())
    if (
        config.get("model_type") != model_type
        or config.get("mlx2_weight_format") != weight_format
    ):
        raise ValueError("native V4.1 routing config changed")
    if file_stamp(manifest_path.stat()) != before:
        raise ValueError("native manifest changed during inspection")
    fingerprint = hashlib.sha256(raw).hexdigest()
    totals = {
        kind: sum(files[name]["size"] for name, value in kinds.items() if value == kind)
        for kind in ("dense", "expert", "engram")
    }
    return {
        "root": root,
        "manifest": manifest,
        "metadata": metadata,
        "config": config,
        "args": args,
        "tensors": tensors,
        "files": files,
        "file_kinds": kinds,
        "identity": {
            "path": str(root),
            "fingerprint": fingerprint,
            "weight_sha256": fingerprint,
            "source_weight_sha256": WEIGHT_SHA256,
            "files": [
                (name, row["size"], _sealed_stat(row)["mtime_ns"])
                for name, row in sorted(files.items())
            ],
        },
        "inventory": {
            "tensor_count": len(tensors),
            "file_count": len(files),
            "storage_bytes": totals,
        },
        "weight_stat": before,
        "qualification": "unqualified",
        "weight_format": weight_format,
        "numeric_profile": numeric_profile,
        "original_gguf_required": False,
    }
