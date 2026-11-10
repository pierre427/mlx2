# SPDX-License-Identifier: MIT
"""CPU-only admission of one pinned omlx-layout V4.1 MLX checkpoint.

The checkpoint keeps the official V4.1 tensor names under ``language_model.``
with MLX affine weights (per-module bits from ``omlx_deepseek_v41``), stacked
routed experts and three-plane affine Engram tables. Every byte is pinned by
SHA-256 in deepseek_v41_omlx_pins.json and sealed by stat after one hash pass.
See provenance/deepseek-v41-omlx.json.
"""
import hashlib
import itertools
import json
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from pathlib import Path

import numpy as np

from ..runtime.immutable_weight_pages import file_stamp, read_page, same_stamp
from ..runtime.models.deepseek_v41_config import DeepSeekV41Config
from .deepseek_v41_source_artifact import (
    aliases,
    checked_file,
    digest,
    validate_router_precision,
)

MODEL_TYPE = "deepseek_v41"
FORMAT = "omlx-deepseek-v41-affine-v1"
SEAL = "mlx2-omlx-integrity.json"
SEAL_SCHEMA = "mlx2.v41.omlx-seal.v1"
PINS = json.loads(Path(__file__).with_name("deepseek_v41_omlx_pins.json").read_text())
PREFIX = "language_model."
DTYPES = {"F32": "<f4", "BF16": "<u2", "U32": "<u4", "U8": "u1"}
# Supported module formats; the MTP head's MXFP8/MXFP4 is inventoried, not served.
AFFINE_BITS = (2, 4, 6, 8)
EXPERT_PROJECTIONS = {"w1": "gate", "w3": "up", "w2": "down"}


def is_omlx_config(config):
    meta = config.get("omlx_deepseek_v41") if isinstance(config, dict) else None
    return isinstance(meta, dict) and meta.get("version") == 1


def prepare_artifact(model_path, *, progress=None):
    """Hash every pinned file once, then publish an immutable stat-bound seal."""
    root = Path(model_path).resolve()
    target = root / SEAL
    if target.exists():
        return inspect_artifact(root)

    def verify(item):
        name, record = item
        path = checked_file(root, name)
        before = file_stamp(path.stat())
        if before["size"] != record["size"]:
            raise ValueError("omlx V4.1 size mismatch: " + name)
        h = hashlib.sha256()
        with path.open("rb", buffering=0) as stream:
            if os.uname().sysname == "Darwin":
                import fcntl
                fcntl.fcntl(stream.fileno(), 48, 1)
            while block := stream.read(16 << 20):
                h.update(block)
        checked_file(root, name, before)
        if h.hexdigest() != record["sha256"]:
            raise ValueError("omlx V4.1 SHA256 mismatch: " + name)
        if progress:
            progress(name, record["size"])
        return name, before

    with ThreadPoolExecutor(max_workers=4) as pool:
        stamps = dict(pool.map(verify, PINS["files"].items()))
    for name, stamp in stamps.items():
        checked_file(root, name, stamp)
    seal = {"schema": SEAL_SCHEMA, "pins_sha256": digest(PINS), "files": stamps}
    # A unique temporary per run: a fixed name opened with "x" left a crashed
    # run's file behind and refused every later prepare (sweep 2026-11-06 DS41-3).
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        os.fchmod(fd, 0o644)  # mkstemp creates 0600; keep the seal readable as before
        with os.fdopen(fd, "w") as out:
            json.dump(seal, out, indent=2)
            out.write("\n")
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return inspect_artifact(root)


def read_headers(root, seals):
    tensors = {}
    for name in sorted(n for n in seals if n.endswith(".safetensors")):
        path = checked_file(root, name, seals[name])
        with path.open("rb") as stream:
            size = int.from_bytes(stream.read(8), "little")
            if not 2 <= size <= 16 << 20:
                raise ValueError("unbounded safetensors header")
            header = json.loads(stream.read(size))
        header.pop("__metadata__", None)
        extents = []
        for key, row in header.items():
            first, last = row["data_offsets"]
            dtype, shape = row["dtype"], row["shape"]
            if (dtype not in DTYPES or not shape or any(type(n) is not int or n <= 0 for n in shape)
                    or not 0 <= first < last <= seals[name]["size"] - size - 8
                    or math.prod(shape) * np.dtype(DTYPES[dtype]).itemsize != last-first
                    or key in tensors):
                raise ValueError("invalid omlx V4.1 tensor extent: " + key)
            tensors[key] = {"file": name, "offset": size+8+first, "size": last-first, "dtype": dtype, "shape": tuple(shape)}
            extents.append((first, last))
        extents.sort()
        if (extents[0][0] or extents[-1][1] != seals[name]["size"]-size-8
                or any(a[1] != b[0] for a, b in itertools.pairwise(extents))):
            raise ValueError("omlx V4.1 shard has overlapping or uncovered payload")
    return tensors


def read_tensors(artifact, names):
    """Exact sealed extents, one descriptor per shard, page cache bypassed."""
    names = tuple(names)
    shards = {}
    for name in names:
        row = artifact["tensors"][name]
        shards.setdefault(row["file"], []).append((name, row))
    result = {}
    for shard, rows in shards.items():
        seal = artifact["seals"][shard]
        path = checked_file(artifact["root"], shard, seal)
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            if not same_stamp(file_stamp(os.fstat(fd)), seal):
                raise ValueError("omlx V4.1 shard changed before bounded read")
            if os.uname().sysname == "Darwin":
                import fcntl
                fcntl.fcntl(fd, 48, 1)
                fcntl.fcntl(fd, 45, 0)
            for name, row in rows:
                raw = read_page(fd, row["offset"], row["size"], 16 << 20)
                result[name] = raw.view(np.dtype(DTYPES[row["dtype"]])).reshape(row["shape"])
            if not same_stamp(file_stamp(os.fstat(fd)), seal):
                raise ValueError("omlx V4.1 shard changed during bounded read")
            checked_file(artifact["root"], shard, seal)
        finally:
            os.close(fd)
    return {name: result[name] for name in names}


def _module_spec(row, name):
    if not isinstance(row, dict):
        raise ValueError("invalid omlx quantized module record: " + name)  # noqa: TRY004 - malformed artifact admission
    if row.get("mode") == "affine":
        if (set(row) != {"bits", "group_size", "mode", "quantize_input"} or row["bits"] not in AFFINE_BITS
                or row["group_size"] not in (32, 64, 128) or type(row["quantize_input"]) is not bool):
            raise ValueError("unsupported omlx affine module: " + name)
    elif row.get("mode") in ("mxfp8", "mxfp4"):
        if set(row) != {"bits", "mode"} or row["bits"] != (8 if row["mode"] == "mxfp8" else 4):
            raise ValueError("unsupported omlx MX module: " + name)
    else:
        raise ValueError("unsupported omlx module mode: " + name)
    return dict(row)


def _affine_geometry(tensors, module, spec, *, stacked=False):
    """Logical (out, in) from packed U32 codes and BF16 scales/biases."""
    weight, scales, biases = (tensors.get(module+suffix) for suffix in (".weight", ".scales", ".biases"))
    if weight is None or scales is None or biases is None:
        raise ValueError("omlx affine module lacks weight/scales/biases: " + module)
    rank = 3 if stacked else 2
    if (weight["dtype"] != "U32" or scales["dtype"] != "BF16" or biases["dtype"] != "BF16"
            or len(weight["shape"]) != rank or scales["shape"] != biases["shape"]
            or weight["shape"][:-1] != scales["shape"][:-1]):
        raise ValueError("omlx affine module precision/shape mismatch: " + module)
    inputs = scales["shape"][-1]*spec["group_size"]
    if weight["shape"][-1]*32 != inputs*spec["bits"]:
        raise ValueError("omlx affine packed width disagrees with its bits: " + module)
    return (*weight["shape"][:-1], inputs)


def _config(config):
    text = dict(config["text_config"])
    text["compress_ratios"] = text["compress_ratios"][:text["num_hidden_layers"]]
    text["rope_scaling"] = {**text["rope_scaling"], "type": "yarn"}
    allowed = {f.name for f in fields(DeepSeekV41Config)} - {"model_type"}
    return DeepSeekV41Config(**{k: v for k, v in text.items() if k in allowed})


def inspect_artifact(model_path):
    root = Path(model_path).expanduser().resolve()
    seal = json.loads(checked_file(root, SEAL).read_text())
    if (seal.get("schema") != SEAL_SCHEMA or seal.get("pins_sha256") != digest(PINS)
            or set(seal["files"]) != set(PINS["files"])):
        raise ValueError("omlx V4.1 seal does not match the pinned release")
    for name, stamp in seal["files"].items():
        checked_file(root, name, stamp)
    config = json.loads((root / "config.json").read_text())
    if not is_omlx_config(config):
        raise ValueError("config.json is not an omlx V4.1 layout")
    meta = config["omlx_deepseek_v41"]
    if meta.get("engram_in_index") is not True:
        raise ValueError("omlx V4.1 Engram tables must be indexed tensors")
    args = _config(config)
    tensors = read_headers(root, seal["files"])
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    if set(index) != set(tensors) or any(tensors[k]["file"] != v for k, v in index.items()):
        raise ValueError("omlx V4.1 tensor index and shards disagree")
    modules = {name: _module_spec(row, name) for name, row in meta["quantized_modules"].items()}

    # Backbone: official names (Source aliases) under the language_model prefix.
    mapping = {alias: PREFIX+name for alias, name in aliases(args).items()}
    dense, quant = {}, {}
    for alias, name in mapping.items():
        module = name.removesuffix(".weight")
        if module in modules:
            spec = modules[module]
            if spec["mode"] != "affine":
                raise ValueError("omlx backbone module is not affine: " + module)
            quant[alias] = dict(spec, module=module, logical=_affine_geometry(tensors, module, spec))
        elif name in tensors:
            if tensors[name]["dtype"] not in ("BF16", "F32"):
                raise ValueError("omlx raw backbone tensor must be BF16/F32: " + name)
            dense[alias] = name
        else:
            raise ValueError("omlx V4.1 lacks required backbone tensor: " + name)
    router_precision = validate_router_precision(tensors, args, prefix=PREFIX+"layers")

    experts = {}
    for layer in range(args.num_hidden_layers):
        layer_specs = {}
        for projection, kind in EXPERT_PROJECTIONS.items():
            module = f"{PREFIX}layers.{layer}.ffn.experts.{projection}"
            spec = modules.get(module)
            if spec is None or spec["mode"] != "affine":
                raise ValueError("omlx routed experts must be affine: " + module)
            logical = _affine_geometry(tensors, module, spec, stacked=True)
            expected = ((args.n_routed_experts, args.hidden_size, args.moe_intermediate_size) if kind == "down"
                        else (args.n_routed_experts, args.moe_intermediate_size, args.hidden_size))
            if logical != expected:
                raise ValueError("omlx routed expert geometry differs from config: " + module)
            layer_specs[kind] = dict(spec, module=module, logical=logical)
        # Every expert path prepares one input from the gate's law and feeds it
        # to gate and up (the fused routed-decode kernel takes a single input),
        # so up must share the gate's input quantization (sweep 2026-11-06 DS41-4).
        if layer_specs["gate"]["quantize_input"] != layer_specs["up"]["quantize_input"]:
            raise ValueError(f"omlx routed experts need one input law for gate and up: layer {layer}")
        experts[layer] = layer_specs

    engram = {}
    tables = meta["engram_tables"]
    for layer, rows in zip(args.engram_layer_ids, args.engram_num_embeddings):
        module = f"{PREFIX}layers.{layer}.engram.embed"
        table = tables.get(module)
        if (not isinstance(table, dict) or table.get("mode") != "affine" or table.get("bits") != 4
                or table.get("group_size") != 32):
            raise ValueError("omlx Engram table must be affine 4-bit group 32: " + module)
        planes = {field: tensors.get(table.get(field+"_key")) for field in ("weight", "scale", "bias")}
        weight, scale, bias = planes["weight"], planes["scale"], planes["bias"]
        if (weight is None or scale is None or bias is None
                or weight["dtype"] != "U32" or weight["shape"] != (rows, 32)
                or scale["dtype"] != "BF16" or scale["shape"] != (rows, 8)
                or bias["dtype"] != "BF16" or bias["shape"] != (rows, 8)
                or len({weight["file"], scale["file"], bias["file"]}) != 1):
            raise ValueError("omlx Engram planes differ from 4-bit group-32 layout: " + module)
        engram[layer] = {"file": weight["file"], "rows": rows, "offset": weight["offset"],
                         "scale_offset": scale["offset"], "bias_offset": bias["offset"]}

    used = {row["module"] for row in quant.values()} | {
        row["module"] for layer in experts.values() for row in layer.values()}
    unused = {name for name in modules if name not in used and not name.startswith(PREFIX+"mtp.")}
    if unused:
        raise ValueError("omlx V4.1 declares quantized modules this route would not serve: "
                         + ", ".join(sorted(unused)[:4]))

    def total(predicate):
        return sum(v["size"] for k, v in tensors.items() if predicate(k))
    expert_bytes = total(lambda k: ".ffn.experts." in k and k.startswith(PREFIX+"layers."))
    engram_bytes = total(lambda k: ".engram.embed." in k)
    storage = {"dense": total(lambda k: k.startswith(PREFIX) and not k.startswith(PREFIX+"mtp.")
                              and ".ffn.experts." not in k and ".engram.embed." not in k),
               "expert": expert_bytes, "engram": engram_bytes,
               "mtp": total(lambda k: k.startswith(PREFIX+"mtp.")),
               "vision": total(lambda k: not k.startswith(PREFIX))}
    fingerprint = digest(PINS)
    return {"root": root, "config": config, "args": args, "tensors": tensors, "aliases": mapping,
            "dense": dense, "quant": quant, "experts": experts, "engram": engram, "seals": seal["files"],
            "identity": {"path": str(root), "fingerprint": fingerprint, "weight_sha256": fingerprint,
                         "repository": PINS["repository"], "revision": PINS["revision"],
                         "files": [(n, v["size"], v["mtime_ns"]) for n, v in seal["files"].items()]},
            "inventory": {"storage_bytes": storage,
                          "expert_item_bytes": expert_bytes // (args.num_hidden_layers*args.n_routed_experts)},
            "qualification": "unqualified", "weight_format": FORMAT, "router_precision": router_precision}
