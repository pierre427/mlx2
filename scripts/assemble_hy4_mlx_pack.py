#!/usr/bin/env python3
"""Assemble an mlx2 Hy4-flash artifact: an MLX pack plus the original's DFly2 and vision tower.

Original mlx2 script.  CPU and file I/O only; never imports MLX.

Third-party MLX packs of tencent/Hy4-flash (e.g. nace-ai/Hy4-flash-MLX-4.6bit)
ship the text trunk only: no ``model.dfly_layers.*`` (the embedded DFly2
drafter) and no ``vit.*`` (the vision tower).  This writes a new artifact
directory that

* hardlinks every pack data file (weights, GGUF Engram tables, tokenizer,
  templates, LICENSE/README) so nothing is copied and the pack stays
  untouched; the pack's Python files are not carried;
* copies the original checkpoint's ``model.dfly_layers.*`` and ``vit.*``
  tensors byte for byte (bf16, unquantized, checkpoint names) into
  ``model-dfly2.safetensors`` and ``model-vision.safetensors``, checking a
  sha256 of every tensor's bytes against the source;
* writes ``model.safetensors.index.json`` (pack map + the new shards) and the
  pack's ``config.json`` with an ``mlx2_assembly`` provenance block.

The adapter then finds both through its existing paths (``has_dfly``,
``has_vision``).  Refuses a nonempty output directory or a source whose
config disagrees with the pack on the DFly2/vision geometry.

Usage::

    assemble_hy4_mlx_pack.py PACK ORIGINAL OUT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path

PREFIXES = {"model-dfly2.safetensors": "model.dfly_layers.", "model-vision.safetensors": "vit."}
SKIP = {"config.json", "model.safetensors.index.json"}
SKIP_SUFFIXES = (".py",)
SKIP_DIRS = {".cache", "hy4_2bit"}
# The vision tower comes from the original, so its image-processor settings
# must too when the pack (a text-trunk pack) does not ship them.
PROCESSOR_FILES = ("processor_config.json", "preprocessor_config.json")
WIDTH = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "U8": 1, "U32": 4, "F8_E4M3": 1}


def header(path: Path):
    with open(path, "rb") as handle:
        (size,) = struct.unpack("<Q", handle.read(8))
        if not 2 <= size <= 1 << 28:
            raise ValueError(f"implausible safetensors header in {path}")
        return json.loads(handle.read(size)), 8 + size


def copy_tensors(original: Path, names: list[str], weight_map: dict, out: Path) -> dict:
    """Byte-copy ``names`` from the original's shards into one safetensors file."""
    entries, sources, offset = {}, [], 0
    headers = {}
    for name in sorted(names):
        shard = weight_map[name]
        if shard not in headers:
            headers[shard] = header(original / shard)
        meta, data0 = headers[shard]
        entry = meta[name]
        begin, end = entry["data_offsets"]
        count = 1
        for dim in entry["shape"]:
            count *= dim
        if entry["dtype"] not in WIDTH or end - begin != count * WIDTH[entry["dtype"]]:
            raise ValueError(f"{name}: byte range disagrees with dtype/shape")
        entries[name] = {"dtype": entry["dtype"], "shape": entry["shape"],
                         "data_offsets": [offset, offset + end - begin]}
        sources.append((name, original / shard, data0 + begin, end - begin))
        offset += end - begin
    blob = json.dumps({**entries, "__metadata__": {"format": "pt"}}, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    digests = {}
    tmp = out.with_suffix(".partial")
    with open(tmp, "wb") as sink:
        sink.write(struct.pack("<Q", len(blob)))
        sink.write(blob)
        for name, path, start, size in sources:
            with open(path, "rb") as source:
                source.seek(start)
                data = source.read(size)
            if len(data) != size:
                raise OSError(f"short read of {name}")
            digests[name] = hashlib.sha256(data).hexdigest()
            sink.write(data)
    # Read back and compare every tensor before publishing the shard.
    meta, data0 = header(tmp)
    with open(tmp, "rb") as check:
        for name, entry in meta.items():
            if name == "__metadata__":
                continue
            begin, end = entry["data_offsets"]
            check.seek(data0 + begin)
            if hashlib.sha256(check.read(end - begin)).hexdigest() != digests[name]:
                raise ValueError(f"{name}: copied bytes differ from the source")
    os.replace(tmp, out)
    return digests


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hub_revision(path: Path) -> str | None:
    """The commit a ``huggingface_hub`` local-dir download recorded, else None."""
    meta = path / ".cache" / "huggingface" / "download" / "config.json.metadata"
    if meta.is_file():
        first = meta.read_text().splitlines()[:1]
        if first and re.fullmatch(r"[0-9a-f]{40}", first[0].strip()):
            return first[0].strip()
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pack", type=Path)
    parser.add_argument("original", type=Path)
    parser.add_argument("out", type=Path)
    a = parser.parse_args(argv)
    pack, original, out = (p.expanduser().resolve() for p in (a.pack, a.original, a.out))
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    pack_cfg = json.loads((pack / "config.json").read_text())
    orig_cfg = json.loads((original / "config.json").read_text())
    if pack_cfg.get("model_type") != "hy_v4_flash" or orig_cfg.get("model_type") != "hy_v4_flash":
        raise SystemExit("both artifacts must be hy_v4_flash")
    if pack_cfg.get("vision_config") != orig_cfg.get("vision_config"):
        raise SystemExit("pack and original disagree on vision_config")
    for key in [k for k in orig_cfg if k.startswith("dfly_")] + ["speculative_config"]:
        if pack_cfg.get(key) != orig_cfg.get(key):
            raise SystemExit(f"pack and original disagree on {key}")
    # Hardlinks need one volume; check before writing anything (against the
    # nearest existing ancestor, so a refusal creates no directories).
    parent = out
    while not parent.exists():
        parent = parent.parent
    if os.stat(parent).st_dev != os.stat(pack).st_dev:
        raise SystemExit(f"{out} must be on the pack's volume (the pack's files are hardlinked)")
    pack_map = json.loads((pack / "model.safetensors.index.json").read_text())["weight_map"]
    orig_map = json.loads((original / "model.safetensors.index.json").read_text())["weight_map"]
    if any(k.startswith(tuple(PREFIXES.values())) for k in pack_map):
        raise SystemExit("pack already carries DFly2 or vision tensors")
    for name in PROCESSOR_FILES:
        if ((pack / name).is_file() and (original / name).is_file()
                and json.loads((pack / name).read_text()) != json.loads((original / name).read_text())):
            raise SystemExit(f"pack and original disagree on {name}")
    created = not out.exists()
    out.mkdir(parents=True, exist_ok=True)
    try:
        return _assemble(pack, original, out, pack_cfg, pack_map, orig_map)
    except BaseException:
        # Leave no half-built artifact behind (it would block a rerun and
        # could be mistaken for a complete one); only what this run made.
        if created:
            shutil.rmtree(out, ignore_errors=True)
        else:
            for child in out.iterdir():
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
        raise


def _assemble(pack, original, out, pack_cfg, pack_map, orig_map) -> int:
    linked = []
    for path in sorted(pack.rglob("*")):
        rel = path.relative_to(pack)
        if (path.is_dir() or rel.parts[0] in SKIP_DIRS or rel.name in SKIP
                or rel.suffix in SKIP_SUFFIXES or rel.name.startswith(".")):
            continue
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, target)
        linked.append(str(rel))
    processor_from_original = []
    for name in PROCESSOR_FILES:
        if not (pack / name).is_file() and (original / name).is_file():
            shutil.copy2(original / name, out / name)
            processor_from_original.append(name)
    weight_map = dict(pack_map)
    added = {}
    for shard, prefix in PREFIXES.items():
        names = [k for k in orig_map if k.startswith(prefix)]
        if not names:
            raise SystemExit(f"original carries no {prefix}* tensors")
        digests = copy_tensors(original, names, orig_map, out / shard)
        weight_map.update({name: shard for name in names})
        added[shard] = {"tensors": len(names), "prefix": prefix,
                        "sha256": sha256_file(out / shard),
                        "tensor_sha256": hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()}
        print(f"wrote {shard}: {len(names)} tensors", flush=True)
    source_rev = hub_revision(original)
    config = dict(pack_cfg, mlx2_assembly={
        "schema": "mlx2.hy4-assembly.v1",
        "script": "scripts/assemble_hy4_mlx_pack.py",
        "pack": {"path": str(pack), "config_sha256": sha256_file(pack / "config.json"),
                 "index_sha256": sha256_file(pack / "model.safetensors.index.json")},
        "original": {"path": str(original), "config_sha256": sha256_file(original / "config.json"),
                     "index_sha256": sha256_file(original / "model.safetensors.index.json"),
                     "revision": source_rev or "unrecorded (no huggingface_hub download metadata)"},
        "added_shards": added,
        "hardlinked_from_pack": linked,
        "processor_config_from_original": processor_from_original,
        "note": "DFly2 and vision tower are the original's bf16 tensors, unquantized; the text trunk is the pack's",
    })
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"format": "mlx"}, "weight_map": weight_map}, indent=1) + "\n")
    print(json.dumps({"out": str(out), "added": {k: v["tensors"] for k, v in added.items()},
                      "linked_files": len(linked)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
