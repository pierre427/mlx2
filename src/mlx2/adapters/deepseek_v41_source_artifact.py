# SPDX-License-Identifier: MIT
"""CPU-only pinned official V4.1 safetensors admission and bounded tensor reads."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import stat

import numpy as np

from ..runtime.immutable_weight_pages import file_stamp, read_page, same_stamp
from ..runtime.models.deepseek_v41_config import DeepSeekV41Config

MODEL_TYPE = "deepseek_v41"
SEAL = "mlx2-source-integrity.json"
PINS = json.loads(Path(__file__).with_name("deepseek_v41_source_pins.json").read_text())
DTYPES = {"F32": "<f4", "BF16": "<u2", "F8_E4M3": "u1", "F8_E8M0": "u1", "I8": "u1"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


TIERS = "mlx2-source-tiers.json"
_TIER_CACHE = {}


def tier_directories(root):
    """Directories that may hold a shard symlinked from ``root`` (storage tiering).

    Declared by a regular, non-symlink ``mlx2-source-tiers.json`` in the root:
    ``{"schema": "mlx2.v41.source-tiers.v1", "directories": ["/abs/dir", ...]}``.
    Absent file: no symlinks are accepted (the original rule)."""
    path = Path(root) / TIERS
    try:
        st = path.lstat()
    except FileNotFoundError:
        return ()
    key = (str(root), st.st_ino, st.st_mtime_ns, st.st_size)
    if key not in _TIER_CACHE:
        if not stat.S_ISREG(st.st_mode) or st.st_size > 1 << 16:
            raise ValueError("source tier declaration must be a small regular file")
        value = json.loads(path.read_text())
        if type(value) is not dict:
            raise ValueError("source tier declaration is malformed")
        dirs = value.get("directories")
        if (value.get("schema") != "mlx2.v41.source-tiers.v1" or type(dirs) is not list
                or not all(type(d) is str and os.path.isabs(d) for d in dirs)):
            raise ValueError("source tier declaration is malformed")
        _TIER_CACHE[key] = tuple(Path(d).resolve() for d in dirs)
    return _TIER_CACHE[key]


def checked_file(root, name, expected=None):
    """Return the regular file holding ``name``; stamp-checked when ``expected``.

    A shard may be a symlink only into a declared tier directory, with the same
    file name; the returned path (and the stamp) is then the target's, so every
    open stays O_NOFOLLOW on a regular file."""
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("source shard must be a direct regular file")
    path = root / name
    st = path.lstat()
    if stat.S_ISLNK(st.st_mode):
        target = path.resolve(strict=True)
        if target.name != name or target.parent not in tier_directories(root):
            raise ValueError("source shard symlink outside declared tiers: " + name)
        path, st = target, target.lstat()
    if not stat.S_ISREG(st.st_mode) or (expected is not None and not same_stamp(file_stamp(st), expected)):
        raise ValueError("official source file changed: " + name)
    return path


def prepare_artifact(model_path, *, progress=None):
    """Hash all pinned bytes once, then publish an immutable stat-bound seal."""
    root = Path(model_path).resolve()
    target = root / SEAL
    if target.exists():
        return inspect_artifact(root)

    def verify(item):
        name, record = item
        path = checked_file(root, name)
        before = file_stamp(path.stat())
        if before["size"] != record["size"]:
            raise ValueError("official source size mismatch: " + name)
        h = hashlib.sha256()
        with path.open("rb", buffering=0) as stream:
            if os.uname().sysname == "Darwin":
                import fcntl
                fcntl.fcntl(stream.fileno(), 48, 1)
            while block := stream.read(16 << 20):
                h.update(block)
        checked_file(root, name, before)
        if h.hexdigest() != record["sha256"]:
            raise ValueError("official source SHA256 mismatch: " + name)
        if progress:
            progress(name, record["size"])
        return name, before

    with ThreadPoolExecutor(max_workers=2) as pool:
        stamps = dict(pool.map(verify, PINS["files"].items()))
    for name, stamp in stamps.items():
        checked_file(root, name, stamp)
    seal = {"schema": "mlx2.v41.official-source-seal.v1", "pins_sha256": digest(PINS), "files": stamps}
    temporary = target.with_suffix(".tmp")
    with temporary.open("x") as out:
        json.dump(seal, out, indent=2)
        out.write("\n")
    os.replace(temporary, target)
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
            if (dtype not in DTYPES or any(type(n) is not int or n <= 0 for n in shape)
                    or not 0 <= first < last <= seals[name]["size"] - size - 8
                    or math.prod(shape) * np.dtype(DTYPES[dtype]).itemsize != last-first
                    or key in tensors):
                raise ValueError("invalid official tensor extent: " + key)
            tensors[key] = dict(file=name, offset=size+8+first, size=last-first, dtype=dtype, shape=tuple(shape))
            extents.append((first, last))
        extents.sort()
        if (extents[0][0] or extents[-1][1] != seals[name]["size"]-size-8
                or any(a[1] != b[0] for a, b in zip(extents, extents[1:]))):
            raise ValueError("source shard has overlapping or uncovered payload")
    return tensors


def aliases(config):
    result = {"token_embd.weight": "embed.weight", "output.weight": "head.weight", "output_norm.weight": "norm.weight"}
    names = {"attn_q_a": "attn.wq_a", "attn_q_b": "attn.wq_b", "attn_kv": "attn.wkv",
             "attn_output_a": "attn.wo_a", "attn_output_b": "attn.wo_b",
             "attn_q_a_norm": "attn.q_norm", "attn_kv_a_norm": "attn.kv_norm",
             "attn_norm": "attn_norm", "ffn_norm": "ffn_norm", "ffn_gate_inp": "ffn.gate"}
    for layer in range(config.num_hidden_layers):
        p, s = f"blk.{layer}.", f"layers.{layer}."
        result.update({p+k+".weight": s+v+".weight" for k, v in names.items()})
        result[p+"attn_sinks.weight"] = s+"attn.attn_sink"
        result[p+"exp_probs_b.bias"] = s+"ffn.gate.bias"
        result[p+"exp_probs_b_vl.bias"] = s+"ffn.gate.bias_vl"
        for kind in ("attn", "ffn"):
            for part in ("fn", "base", "scale"):
                result[p+f"hc_{kind}_{part}.weight"] = s+f"hc_{kind}_{part}"
        for kind, projection in (("gate", "w1"), ("up", "w3"), ("down", "w2")):
            result[p+f"ffn_{kind}_shexp.weight"] = s+f"ffn.shared_experts.{projection}.weight"
        if layer in config.kv_source_layer_ids:
            for key, name in {"attn_compressor_kv": "attn.compressor.wkv", "attn_compressor_norm": "attn.compressor.norm",
                              "indexer.attn_k": "attn.indexer.wk", "indexer.k_norm": "attn.indexer.k_norm"}.items():
                result[p+key+".weight"] = s+name+".weight"
            if config.compress_ratios[layer] > 1:
                result[p+"attn_compressor_gate.weight"] = s+"attn.compressor.wgate.weight"
        if layer in config.index_source_layer_ids:
            result[p+"indexer.attn_q_b.weight"] = s+"attn.indexer.wq_b.weight"
            result[p+"indexer.proj.weight"] = s+"attn.indexer.weights_proj.weight"
        if layer in config.engram_layer_ids:
            result[p+"engram_q_norm.weight"] = s+"engram.q_weight"
            result[p+"engram_k_norm.weight"] = s+"engram.k_weight"
            result[p+"engram_kv.weight"] = s+"engram.wkv.weight"
    return result


def validate_router_precision(tensors, args, *, prefix="layers", layers=None, experts=None):
    """Router precision is checkpoint math, independent of expert storage."""
    layers = args.num_hidden_layers if layers is None else layers
    experts = args.n_routed_experts if experts is None else experts
    for layer in range(layers):
        root = f"{prefix}.{layer}.ffn.gate."
        for field, dtype, shape in (
            ("weight", "BF16", (experts, args.hidden_size)),
            ("bias", "F32", (experts,)),
            ("bias_vl", "F32", (experts,)),
        ):
            value = tensors.get(root + field)
            if value is None or value["dtype"] != dtype or tuple(value["shape"]) != shape:
                raise ValueError("official router precision/shape mismatch: " + root + field)
        if root + "scale" in tensors or root + "weight_scale_inv" in tensors:
            raise ValueError("official BF16 router must not have quantization scales")
    return {"weight": "BF16", "correction_bias": "F32", "image_bias": "F32",
            "layers": layers, "experts": experts, "requantized": False}


def inspect_artifact(model_path):
    root = Path(model_path).expanduser().resolve()
    seal = json.loads(checked_file(root, SEAL).read_text())
    if (seal.get("schema") != "mlx2.v41.official-source-seal.v1" or seal.get("pins_sha256") != digest(PINS)
            or set(seal["files"]) != set(PINS["files"])):
        raise ValueError("official source seal does not match pinned release")
    for name, stamp in seal["files"].items():
        checked_file(root, name, stamp)
    config = json.loads((root / "config.json").read_text())
    text = dict(config["text_config"])
    text["compress_ratios"] = text["compress_ratios"][:text["num_hidden_layers"]]
    text["rope_scaling"] = {**text["rope_scaling"], "type": "yarn"}
    allowed = {f.name for f in fields(DeepSeekV41Config)} - {"model_type"}
    args = DeepSeekV41Config(**{k: v for k, v in text.items() if k in allowed})
    tensors = read_headers(root, seal["files"])
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    if set(index) != set(tensors) or any(tensors[k]["file"] != v for k, v in index.items()):
        raise ValueError("source tensor index and shards disagree")
    router_precision = validate_router_precision(tensors, args)
    mapping = aliases(args)
    if set(mapping.values()) - tensors.keys():
        raise ValueError("official source lacks required backbone tensors")
    dense = sum(tensors[n]["size"] + tensors.get(n.removesuffix("weight")+"scale", {}).get("size", 0) for n in mapping.values())
    expert = sum(v["size"] for k, v in tensors.items() if k.startswith("layers.") and ".experts." in k)
    engram = sum(v["size"] for k, v in tensors.items() if k.startswith("layers.") and ".engram.embed." in k)
    fingerprint = digest(PINS)
    return {"root": root, "config": config, "args": args, "tensors": tensors, "aliases": mapping,
            "seals": seal["files"], "identity": {"path": str(root), "fingerprint": fingerprint,
            "weight_sha256": fingerprint, "files": [(n, v["size"], v["mtime_ns"]) for n, v in seal["files"].items()]},
            "inventory": {"storage_bytes": {"dense": dense, "expert": expert, "engram": engram}},
            "qualification": "unqualified", "weight_format": "official-mxfp4-fp8-bf16",
            "router_precision": router_precision}


def read_tensors(artifact, names):
    """Read exact extents with one sealed descriptor per participating shard.

    A routed expert has six tensors, normally in one shard. Sharing the open
    and integrity checks avoids repeating filesystem metadata work for each
    tensor without reading gaps, retaining descriptors, or changing payloads.
    """
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
                raise ValueError("source changed before bounded read")
            if os.uname().sysname == "Darwin":
                import fcntl
                fcntl.fcntl(fd, 48, 1)
                fcntl.fcntl(fd, 45, 0)
            for name, row in rows:
                raw = read_page(fd, row["offset"], row["size"], 16 << 20)
                result[name] = raw.view(np.dtype(DTYPES[row["dtype"]])).reshape(row["shape"])
            # Check both the descriptor and path: an unlinked/replaced shard
            # must not make bytes from an obsolete inode appear current.
            if not same_stamp(file_stamp(os.fstat(fd)), seal):
                raise ValueError("source changed during bounded read")
            checked_file(artifact["root"], shard, seal)
        finally:
            os.close(fd)
    return {name: result[name] for name in names}
