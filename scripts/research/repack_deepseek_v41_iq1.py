#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only lossless affine-to-IQ1_S repacking; no original GGUF is needed.

Original project inverse of the pinned local deepseek_v41_native_layer.py
conversion, using the existing MIT-noticed IQ1_GRID codebook. No new external
implementation is mined. Each bounded inverse chunk must reproduce all three
affine arrays bitwise. Nonunique half-scale/factor encodings are canonicalized;
this proves represented weights, not reconstruction of unavailable source bytes.

The artifact writer uses APFS clonefile for unchanged dense/PLE/assets, with no
physical-copy fallback and no hardlinks. It never modifies/deletes its source.
"""
from __future__ import annotations

import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src

import argparse
from copy import deepcopy
import ctypes
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import time

import numpy as np

from deepseek_v41_native_layer import MAX_CHUNK_ELEMENTS, _iq1_affine_grid, affine_spec, convert_affine
from convert_deepseek_v41_mlx import (atomic_json, canonical, check_space,
                                     conversion_lock, file_hash, read_exact, safe_path,
                                     stamp, write_at)

MANIFEST = "mlx2-v41-iq1-manifest.json"
SCHEMA = "mlx2.deepseek-v41-iq1.v1"
MODEL_TYPE = "deepseek41_iq1"
FORMAT = "mlx2-v41-iq1s-v1"
NUMERIC_PROFILE = "native_iq1_s_packed"
MAX_EXPERT_SOURCE_BYTES = 64 << 20


@lru_cache(maxsize=1)
def inverse_grid():
    words = _iq1_affine_grid()
    if len(np.unique(words)) != len(words):
        raise ValueError("IQ1 codebook inversion is not unique")
    inverse = np.full(65536, -1, np.int16)
    inverse[words] = np.arange(len(words), dtype=np.int16)
    inverse.flags.writeable = False
    return inverse


def repack_iq1(qweight, scales, biases, shape, *, max_elements=MAX_CHUNK_ELEMENTS):
    """Return contiguous IQ1_S bytes and an exact forward-roundtrip receipt.

    Integer GCD recovers the maximal shared scale. At most eight odd divisors
    are tested, and only on blocks whose observed factors permit each divisor.
    Every operation is vectorized over bounded blocks; no scalar weight loop or
    floating full-weight expansion occurs. Choose the first/largest-magnitude
    valid base scale, then the smallest factors. Zero-scale blocks use factor1
    and positive delta; their lost original factor/delta bits are nonunique.
    """
    spec = affine_spec(19, shape)
    if type(max_elements) is not int or max_elements < 1 or spec["elements"] > max_elements:
        raise ValueError("IQ1 inverse exceeds explicit positive element budget")
    for value, key, dtype in ((qweight, "packed_shape", np.uint32), (scales, "scales_shape", np.float32),
                              (biases, "biases_shape", np.float32)):
        if (not isinstance(value, np.ndarray) or value.dtype != dtype or value.shape != spec[key]
                or not value.flags.c_contiguous):
            raise ValueError("affine inverse requires declared contiguous native shape/dtype")
    if not np.isfinite(scales).all() or not np.isfinite(biases).all():
        raise ValueError("IQ1 inverse requires finite affine metadata")
    groups = scales.reshape(-1, 8)
    group_biases = biases.reshape(-1, 8)
    blocks = len(groups)
    zero = np.all(groups == 0, axis=1)
    plus_bias = groups * np.float32(-.875)
    minus_bias = groups * np.float32(-1.125)
    plus = group_biases.view(np.uint32) == plus_bias.view(np.uint32)
    minus = group_biases.view(np.uint32) == minus_bias.view(np.uint32)
    if not np.all(plus | minus):
        raise ValueError("affine biases do not encode IQ1 quarter-delta law")
    # Prefer positive delta when the zero bias cannot identify its source bit.
    negative_delta = ~plus
    selected_base = np.zeros(blocks, np.float32)
    factors = np.ones((blocks, 8), np.uint16)
    candidates = np.zeros(blocks, np.uint8)
    nonzero_indices = np.flatnonzero(~zero)
    if len(nonzero_indices):
        values = groups[nonzero_indices]
        bits = values.view(np.uint32)
        exponents = ((bits >> 23) & 255).astype(np.int16)
        minimum_exp = exponents.min(axis=1)
        shifts = exponents - minimum_exp[:, None]
        if (np.any(values == 0) or np.any((bits >> 31) != (bits[:, :1] >> 31))
                or np.any(minimum_exp == 0) or np.any(shifts > 4)):
            raise ValueError("affine group scales have no common finite-half IQ1 base")
        significands = ((bits & 0x7fffff) | 0x800000) << shifts.astype(np.uint32)
        gcd = np.gcd.reduce(significands, axis=1)
        reduced_factors = significands // gcd[:, None]
        max_factors = reduced_factors.max(axis=1)
        if np.any(max_factors > 15) or np.any((reduced_factors & 1) == 0):
            raise ValueError("affine group scales have no common finite-half IQ1 base")
        common_base = np.ldexp(gcd.astype(np.float32), minimum_exp - 150)
        common_base = np.copysign(common_base, values[:, 0])
        with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
            for divisor in (1, 3, 5, 7, 9, 11, 13, 15):
                eligible = np.flatnonzero(max_factors * divisor <= 15)
                if not len(eligible):
                    continue
                base = (common_base[eligible] / np.float32(divisor)).astype(np.float16).astype(np.float32)
                ratio = reduced_factors[eligible] * divisor
                valid = (np.isfinite(base) & (base != 0)
                         & np.all(base[:, None] * ratio == values[eligible], axis=1))
                indices = nonzero_indices[eligible[valid]]
                first = candidates[indices] == 0
                selected_base[indices[first]] = base[valid][first]
                factors[indices[first]] = ratio[valid][first].astype(np.uint16)
                candidates[indices] += 1
    if np.any((candidates == 0) & ~zero):
        raise ValueError("affine group scales have no common finite-half IQ1 base")
    if np.any(zero):
        # A valid source block has one common sign, including signed-zero scales.
        signs = groups[zero].view(np.uint32)
        if np.any(signs != signs[:, :1]):
            raise ValueError("inconsistent signed-zero group scales")
        selected_base[zero] = groups[zero, 0]
    words = qweight.reshape(-1, 8, 2)
    halves = np.empty((blocks, 8, 4), np.uint16)
    halves[..., ::2] = words & np.uint32(65535)
    halves[..., 1::2] = words >> np.uint32(16)
    indices = inverse_grid()[halves]
    if np.any(indices < 0):
        raise ValueError("affine codes do not belong to the IQ1 codebook")
    indices = indices.astype(np.uint16)
    result = np.empty((blocks, 50), np.uint8)
    result[:, :2] = selected_base.astype("<f2").reshape(-1, 1).view(np.uint8)
    result[:, 2:34] = (indices & 255).astype(np.uint8).reshape(-1, 32)
    shifts = np.arange(4, dtype=np.uint16) * 3
    high = np.bitwise_or.reduce((indices >> 8) << shifts, axis=-1)
    high |= ((factors - 1) // 2) << 12
    high |= negative_delta.astype(np.uint16) << 15
    result[:, 34:] = high.astype("<u2").view(np.uint8)
    raw = result.reshape(-1)
    restored = convert_affine(raw, 19, tuple(shape), max_elements=max_elements)
    if any(not np.array_equal(a.view(np.uint32), b.view(np.uint32))
           for a, b in zip(restored, (qweight, scales, biases))):
        raise ValueError("IQ1 forward conversion changed affine source bits")
    ambiguous = (candidates > 1) | zero
    return raw, dict(blocks=blocks, elements=spec["elements"], forward_affine_bitwise_exact=True,
                     unique_original_encoding_blocks=int(np.count_nonzero(~ambiguous)),
                     ambiguous_encoding_blocks=int(np.count_nonzero(ambiguous)),
                     zero_scale_blocks=int(np.count_nonzero(zero)),
                     half_d_min=float(selected_base.min()), half_d_max=float(selected_base.max()),
                     half_d_nonzero_subnormal_blocks=int(np.count_nonzero((np.abs(selected_base) > 0)
                                                                         & (np.abs(selected_base) < 2**-14))))


def clone_file(source, destination):
    """Require APFS COW cloning; no overwrite, hardlink or copy fallback."""
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or destination.exists() or destination.is_symlink():
        raise ValueError("COW clone requires a regular source and absent destination")
    before = stamp(source)
    try:
        function = ctypes.CDLL(None, use_errno=True).clonefile
    except AttributeError as error:
        raise OSError("clonefile is unavailable; physical-copy fallback is forbidden") from error
    function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    function.restype = ctypes.c_int
    if function(os.fsencode(source), os.fsencode(destination), 0):
        code = ctypes.get_errno()
        raise OSError(code, "APFS clonefile failed; no copy fallback: " + os.strerror(code))
    after = stamp(destination)
    if stamp(source) != before or after["ino"] == before["ino"] or after["size"] != before["size"]:
        raise ValueError("COW clone changed source seal or created a shared inode")
    return after


def compact_layout(shapes):
    layout, position = {}, 0
    for kind in ("gate", "up", "down"):
        spec = affine_spec(19, tuple(shapes[kind]))
        size = spec["source_bytes"]
        layout[kind + ".data"] = dict(dtype="U8", shape=[size], data_offsets=[position, position + size])
        position += size
    header = canonical({**layout, "__metadata__": {"format": "mlx", "packing": "GGML_IQ1_S/256/50"}})
    header += b" " * (-len(header) % 8)
    return layout, struct.pack("<Q", len(header)) + header, position


def verify_file(root, relative, record):
    path = safe_path(root, relative)
    before = stamp(path)
    if before["size"] != record["size"] or file_hash(path) != record["sha256"] or stamp(path) != before:
        raise ValueError("compact file hash/extent changed: " + relative)
    if record.get("header_bytes"):
        with path.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            if size + 8 != record["header_bytes"] or size > 16 << 20:
                raise ValueError("compact safetensors header extent invalid")
            parsed = json.loads(stream.read(size))
        parsed.pop("__metadata__", None)
        if parsed != record["tensors"]:
            raise ValueError("compact safetensors header differs from manifest")
        end = 0
        for item in sorted(parsed.values(), key=lambda x: x["data_offsets"][0]):
            first, last = item["data_offsets"]
            if (first != end or last - first != math.prod(item["shape"])
                    * {"U8": 1, "F16": 2, "F32": 4, "U32": 4}[item["dtype"]]):
                raise ValueError("compact safetensors layout has gaps/overlap")
            end = last
        if end + record["header_bytes"] != record["size"]:
            raise ValueError("compact safetensors payload extent mismatch")
    return before


def write_expert(source_root, relative, source_record, shapes, destination, *, rows_per_chunk=64):
    from mlx2.adapters.deepseek_v41_mlx_weights import read_native_file
    if source_record["size"] > MAX_EXPERT_SOURCE_BYTES:
        raise ValueError("one affine expert exceeds bounded source read")
    arrays = read_native_file(Path(source_root), relative, source_record)
    layout, header, payload = compact_layout(shapes)
    fd = os.open(destination, os.O_CREAT | os.O_TRUNC | os.O_RDWR, 0o600)
    proof = dict(blocks=0, elements=0, unique_original_encoding_blocks=0,
                 ambiguous_encoding_blocks=0, zero_scale_blocks=0,
                 half_d_min=None, half_d_max=None, half_d_nonzero_subnormal_blocks=0,
                 forward_affine_bitwise_exact=True, chunk_count=0)
    try:
        os.ftruncate(fd, len(header) + payload)
        write_at(fd, header, 0)
        for kind in ("gate", "up", "down"):
            rows, width = shapes[kind]
            step = min(rows_per_chunk, MAX_CHUNK_ELEMENTS // width)
            if type(rows_per_chunk) is not int or rows_per_chunk < 1 or step < 1:
                raise ValueError("invalid bounded repacking row chunk")
            for first in range(0, rows, step):
                last = min(rows, first + step)
                part = [arrays[kind + "." + name][first:last] for name in ("qweight", "scales", "biases")]
                raw, checked = repack_iq1(*part, (last - first, width))
                row_bytes = width // 256 * 50
                offset = len(header) + layout[kind + ".data"]["data_offsets"][0] + first * row_bytes
                write_at(fd, raw, offset)
                # Every stored chunk is read back, then repromoted to source bits.
                stored = read_exact(fd, len(raw), offset)
                promoted = convert_affine(stored, 19, (last - first, width))
                if any(not np.array_equal(a.view(np.uint32), b.view(np.uint32)) for a, b in zip(promoted, part)):
                    raise ValueError("stored compact bytes changed affine values")
                for key in proof:
                    if key == "half_d_min":
                        proof[key] = checked[key] if proof[key] is None else min(proof[key], checked[key])
                    elif key == "half_d_max":
                        proof[key] = checked[key] if proof[key] is None else max(proof[key], checked[key])
                    elif key in checked and key != "forward_affine_bitwise_exact":
                        proof[key] += checked[key]
                proof["chunk_count"] += 1
        os.fsync(fd)
    finally:
        os.close(fd)
    if stamp(safe_path(source_root, relative)) != source_record["stat"]:
        raise ValueError("affine source changed during expert repacking")
    return dict(size=len(header) + payload, header_bytes=len(header), tensors=layout,
                source_affine_sha256=source_record["sha256"], proof=proof)


def build_plan(source_manifest):
    tensors = deepcopy(source_manifest["tensors"])
    expert_shapes = {}
    for name, tensor in tensors.items():
        storage = tensor["storage"]
        if storage["kind"] != "expert-affine":
            continue
        plane = name.split(".ffn_", 1)[1].removesuffix("_exps.weight")
        shape = tuple(tensor["numpy_shape"][1:])
        for path in storage["expert_files"]:
            expert_shapes.setdefault(path, {})[plane] = shape
        tensor["storage"] = dict(kind="expert-iq1_s", expert_files=storage["expert_files"],
                                 keys={"data": plane + ".data"}, representation="ggml_iq1_s",
                                 block_elements=256, block_bytes=50)
    sizes = {}
    for path, shapes in expert_shapes.items():
        if set(shapes) != {"gate", "up", "down"}:
            raise ValueError("compact expert projection inventory incomplete")
        _, header, payload = compact_layout(shapes)
        sizes[path] = len(header) + payload
    cloned = {p: r for p, r in source_manifest["files"].items() if p not in expert_shapes and p != "config.json"}
    return tensors, expert_shapes, sizes, cloned


def recipe():
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    return dict(version=1, canonicalization="largest-magnitude valid finite-half base; positive zero delta; original codebook indices",
                validation="bitwise forward affine comparison of every in-memory and stored chunk",
                block_elements=256, block_bytes=50, scalar_expansion=False,
                source_sha256={str(p.relative_to(root)): file_hash(p) for p in
                               (Path(__file__), directory / "deepseek_v41_native_layer.py",
                                root / "src/mlx2/runtime/gguf/iq1_codebook.py")})


def repack(source, output, *, reserve_bytes=100 << 30, max_files=None, plan_only=False, phase=print):
    from mlx2.adapters.deepseek_v41_mlx_artifact import inspect_artifact
    source = Path(source).resolve()
    output = Path(output)
    if output.is_symlink():
        raise ValueError("compact output root is a symlink")
    output = output.resolve()
    if source == output or source in output.parents:
        raise ValueError("compact artifact must have its own sibling directory")
    admitted = inspect_artifact(source)
    original = admitted["manifest"]
    tensors, experts, sizes, cloned = build_plan(original)
    summary = dict(expert_files=len(experts), new_expert_bytes=sum(sizes.values()),
                   clone_logical_bytes=sum(r["size"] for r in cloned.values()),
                   additional_physical_bytes_estimate=sum(sizes.values()) + (128 << 20),
                   physical_estimate_scope="new compact experts plus128MiB metadata; unchanged files require APFS COW",
                   minimum_free_reserve_bytes=reserve_bytes, clone_copy_fallback=False)
    if plan_only:
        return summary
    output.mkdir(parents=True, exist_ok=True)
    with conversion_lock(output):
        packing_recipe = recipe()
        recipe_hash = hashlib.sha256(canonical(packing_recipe)).hexdigest()
        binding = dict(derived_identity_sha256=original["derived_identity_sha256"],
                       fingerprint=admitted["identity"]["fingerprint"],
                       manifest_sha256=admitted["identity"]["fingerprint"], path=str(source))
        path = output / MANIFEST
        if path.exists():
            manifest = json.loads(path.read_text())
            if manifest.get("source_affine") != binding or manifest.get("recipe_sha256") != recipe_hash:
                raise ValueError("compact resume source/recipe binding differs")
            if manifest.get("finished"):
                raise ValueError("compact artifact already complete; use its standalone inspector")
        else:
            if any(p.name != ".conversion.lock" for p in output.iterdir()):
                raise ValueError("refusing unrelated compact output directory")
            manifest = dict(schema=SCHEMA, model_type=MODEL_TYPE, format=FORMAT, numeric_profile=NUMERIC_PROFILE,
                            finished=False, qualification="unqualified", source=original["source"], source_affine=binding,
                            metadata=original["metadata"], tensors=tensors, files={}, assets=[], recipe=packing_recipe,
                            recipe_sha256=recipe_hash, tokenizer_binding=original.get("tokenizer_binding", {}),
                            source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                            writer_sha256=file_hash(__file__), plan=summary, started_unix=time.time())
            atomic_json(path, manifest)
        def source_check():
            if stamp(source / "mlx2-v41-native-manifest.json") != admitted["weight_stat"]:
                raise ValueError("affine source manifest changed during repacking")
        records, completed = {}, 0
        for relative in sorted(experts):
            source_check()
            target = safe_path(output, relative)
            receipt_path = safe_path(output, "conversion-state/" + hashlib.sha256(relative.encode()).hexdigest() + ".json")
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if receipt.get("path") != relative or receipt.get("recipe_sha256") != recipe_hash:
                    raise ValueError("compact resume receipt binding changed")
                record = receipt["record"]
                if verify_file(output, relative, record) != record["stat"]:
                    raise ValueError("compact resume stat seal changed")
            else:
                if max_files is not None and completed >= max_files:
                    break
                check_space(output, sizes[relative], reserve_bytes)
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = safe_path(output, relative + ".partial")
                record = write_expert(source, relative, original["files"][relative], experts[relative], partial)
                source_check()
                record["sha256"] = file_hash(partial)
                os.replace(partial, target)
                record["stat"] = stamp(target)
                verify_file(output, relative, record)
                atomic_json(receipt_path, dict(path=relative, recipe_sha256=recipe_hash, record=record))
                completed += 1
            records[relative] = record
            if len(records) % 32 == 0 or completed == 1:
                progress = dict(completed_experts=len(records), total_experts=len(experts),
                                bytes=sum(r["size"] for r in records.values()), last_file=relative, updated_unix=time.time())
                atomic_json(output / "conversion-progress.json", progress)
                phase(json.dumps(progress))
        if len(records) != len(experts):
            return dict(finished=False, newly_written=completed, completed_experts=len(records), total_experts=len(experts))
        for relative, source_record in sorted(cloned.items()):
            source_check()
            original_path = safe_path(source, relative)
            if stamp(original_path) != source_record["stat"]:
                raise ValueError("affine clone source changed after admission")
            check_space(output, 0, reserve_bytes)
            target = safe_path(output, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = safe_path(output, relative + ".partial")
            if partial.exists():
                partial.unlink()
            clone_file(original_path, partial)
            os.replace(partial, target)
            record = deepcopy(source_record)
            record["stat"] = stamp(target)
            record["copy_method"] = "apfs_clonefile_cow"
            verify_file(output, relative, record)
            if stamp(original_path) != source_record["stat"]:
                raise ValueError("cloning changed affine source seal")
            records[relative] = record
        config = dict(model_type=MODEL_TYPE, mlx2_weight_format=FORMAT, manifest_file=MANIFEST)
        atomic_json(output / "config.json", config)
        records["config.json"] = dict(size=(output / "config.json").stat().st_size,
                                      sha256=file_hash(output / "config.json"), stat=stamp(output / "config.json"),
                                      tensors={}, header_bytes=0)
        manifest["files"] = records
        manifest["assets"] = list(original["assets"])
        phase(json.dumps(dict(phase="verify_all_compact_and_cloned_hashes", files=len(records), updated_unix=time.time())))
        for relative, record in records.items():
            if verify_file(output, relative, record) != record["stat"]:
                raise ValueError("compact output changed before final publication")
        source_check()
        # Revalidate every original affine stat without rehashing485GB: source
        # expert bytes were hash-checked onread and cloned bytes against their seals.
        source_after = inspect_artifact(source)
        if source_after["identity"]["fingerprint"] != binding["fingerprint"]:
            raise ValueError("affine source changed while producing compact sibling")
        totals = {key: sum(r.get("proof", {}).get(key, 0) for p, r in records.items() if p in experts)
                  for key in ("blocks", "elements", "unique_original_encoding_blocks", "ambiguous_encoding_blocks", "zero_scale_blocks", "chunk_count", "half_d_nonzero_subnormal_blocks")}
        totals["half_d_min"] = min(records[p]["proof"]["half_d_min"] for p in experts)
        totals["half_d_max"] = max(records[p]["proof"]["half_d_max"] for p in experts)
        manifest["verification"] = dict(passed=True, all_file_hashes_checked=True, source_affine_unchanged=True,
                                        forward_affine_bitwise_exact=True, source_tensor_count=len(tensors),
                                        file_count=len(records), converted_bytes=sum(r["size"] for r in records.values()), **totals)
        manifest["finished"] = True
        manifest["finished_unix"] = time.time()
        manifest["derived_identity_sha256"] = hashlib.sha256(canonical(dict(source=binding, recipe=recipe_hash,
                                                        files={p: r["sha256"] for p, r in records.items()}))).hexdigest()
        atomic_json(path, manifest)
        return manifest["verification"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--reserve-gib", type=int, default=100)
    args = parser.parse_args()
    if args.reserve_gib < 0 or (args.max_files is not None and args.max_files < 0):
        parser.error("reserve/max-files must be nonnegative")
    print(json.dumps(repack(args.source, args.output, reserve_bytes=args.reserve_gib << 30,
                            max_files=args.max_files, plan_only=args.plan), sort_keys=True))


if __name__ == "__main__":
    main()
