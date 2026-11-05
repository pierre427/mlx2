# SPDX-License-Identifier: MIT
"""CPU-only, streaming conversion into ordinary SC-MXFP4 artifacts.

Only the explicit offline operation requantizes weights. Source tensors are
read by row through safetensors byte mappings; unselected tensor bytes are
preserved. No MLX import, model construction or GPU operation occurs here.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import time
from typing import Iterator

import numpy as np

from .format import SC_FORMAT, pack_numpy, to_mxfp4_scales_numpy

SCHEMA = "mlx2.sc-mxfp4-conversion.v1"
_UPSTREAM = "71ab38d3ad7e94279368accd0e880df90a63ae1c"
_WIDTHS = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
           "U16": 2, "I16": 2, "BF16": 2, "F16": 2, "U32": 4, "I32": 4,
           "F32": 4, "U64": 8, "I64": 8, "F64": 8}
_DTYPES = {"U8": "u1", "I8": "i1", "U16": "<u2", "I16": "<i2", "BF16": "<u2",
           "F16": "<f2", "U32": "<u4", "I32": "<i4", "F32": "<f4", "F64": "<f8"}
_EXPERT = re.compile(r"^(.*)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
_QUANT = {"group_size": 32, "bits": 4, "mode": "mxfp4"}


class ConversionError(ValueError):
    """The explicit offline artifact conversion cannot be completed safely."""


def _json(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ConversionError(f"duplicate JSON key {key!r}: {path}")
            out[key] = value
        return out
    with Path(path).open() as stream:
        return json.load(stream, object_pairs_hook=unique)


def _atomic(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


@dataclass(frozen=True)
class Tensor:
    path: Path
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int

    def raw(self):
        return np.memmap(self.path, mode="r", dtype=np.uint8, offset=self.offset,
                         shape=(self.nbytes,))

    def rows(self, start, end):
        if self.dtype not in _DTYPES or len(self.shape) < 2:
            raise ConversionError(f"unsupported row source {self.name}: {self.dtype}/{self.shape}")
        data = np.memmap(self.path, mode="r", dtype=_DTYPES[self.dtype],
                         offset=self.offset, shape=self.shape)
        row = np.array(data.reshape(-1, self.shape[-1])[start:end], copy=True)
        if self.dtype == "BF16":
            return (row.astype(np.uint32) << 16).view(np.float32)
        return row


class Source:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.config = _json(self.path / "config.json")
        if not isinstance(self.config, dict):
            raise ConversionError("source config must be a JSON object")
        self.index = None
        index_path = self.path / "model.safetensors.index.json"
        if index_path.is_file():
            self.index = _json(index_path).get("weight_map")
            if (not isinstance(self.index, dict) or not self.index
                    or any(not isinstance(k, str) or not isinstance(v, str) for k, v in self.index.items())):
                raise ConversionError("source has an invalid or empty tensor index")
            names = sorted(set(self.index.values()))
        else:
            names = ["model.safetensors"]
        self.tensors = {}
        self.shards = []
        for name in names:
            if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
                raise ConversionError("source shard paths must be safetensors basenames")
            file = self.path / name
            stat = file.stat()
            with file.open("rb") as stream:
                prefix = stream.read(8)
                if len(prefix) != 8:
                    raise ConversionError("truncated safetensors prefix")
                length = struct.unpack("<Q", prefix)[0]
                if length > min(128 << 20, stat.st_size - 8):
                    raise ConversionError("invalid safetensors header length")
                header_bytes = stream.read(length)
            try:
                def unique(pairs):
                    result = {}
                    for key, value in pairs:
                        if key in result:
                            raise ConversionError(f"duplicate safetensors header key {key!r}")
                        result[key] = value
                    return result
                header = json.loads(header_bytes, object_pairs_hook=unique)
            except (ValueError, UnicodeError) as error:
                raise ConversionError("invalid safetensors header JSON") from error
            if not isinstance(header, dict):
                raise ConversionError("safetensors header must be an object")
            intervals = []
            for key, spec in header.items():
                if key == "__metadata__":
                    continue
                if key in self.tensors:
                    raise ConversionError(f"duplicate tensor {key}")
                if self.index is not None and self.index.get(key) != name:
                    raise ConversionError(f"index/header disagreement: {key}")
                if not isinstance(key, str) or not isinstance(spec, dict):
                    raise ConversionError("safetensors tensor entry must be an object")
                dtype, shape, offsets = spec.get("dtype"), spec.get("shape"), spec.get("data_offsets")
                if dtype not in _WIDTHS or not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
                    raise ConversionError(f"unsupported tensor header: {key}")
                size = math.prod(shape) * _WIDTHS[dtype]
                if not isinstance(offsets, list) or len(offsets) != 2 or any(type(i) is not int for i in offsets):
                    raise ConversionError(f"invalid tensor offsets: {key}")
                start, end = offsets
                if start < 0 or end - start != size or 8 + length + end > stat.st_size:
                    raise ConversionError(f"tensor bytes lie outside shard: {key}")
                intervals.append((start, end))
                self.tensors[key] = Tensor(file, key, dtype, tuple(shape), 8 + length + start, size)
            ordered = sorted(intervals)
            if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
                raise ConversionError("overlapping source tensor regions")
            self.shards.append({"name": name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                "header_sha256": hashlib.sha256(header_bytes).hexdigest()})
        if self.index is not None and set(self.index) != set(self.tensors):
            raise ConversionError("source index is not closed over shard tensors")
        metadata = {name: _sha(self.path / name) for name in
                    ("config.json", "model.safetensors.index.json", "tokenizer.json",
                     "tokenizer_config.json", "chat_template.jinja", "generation_config.json")
                    if (self.path / name).is_file()}
        binding = {"scheme": "metadata-shard-header-stat-v1", "metadata": metadata, "shards": self.shards}
        self.fingerprint = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
        self.identity = {"path": str(self.path), "fingerprint": self.fingerprint, **binding}

    def quantization(self, module):
        global_quant = self.config.get("quantization", self.config.get("quantization_config")) or {}
        value = dict(global_quant)
        override = global_quant.get(module)
        if isinstance(override, dict):
            value.update(override)
        return value


def selected(module: str, scope: str, *, grouped=False) -> bool:
    """Structural scope shared with installer; routers/head/modality stay out."""
    if scope not in {"experts", "mlp", "all"}:
        raise ConversionError("scope must be experts, mlp or all")
    from ..int8_prefill import EXCLUDED_SEGMENTS, MLP_SEGMENTS
    segments = module.split(".")
    excluded = EXCLUDED_SEGMENTS - {"switch_mlp", "experts"}
    if not segments or any(part in excluded for part in segments):
        return False
    if segments[-1] in {"gate", "router", "shared_expert_gate"}:
        return False
    if scope == "experts":
        return grouped
    return scope == "all" or any(part in MLP_SEGMENTS for part in segments)


@dataclass(frozen=True)
class Plan:
    name: str
    members: tuple[Tensor, ...]
    shape: tuple[int, ...]
    grouped: bool
    convert: bool


def plan_tensors(source: Source, scope="mlp", *, normalize_only=False, protected_tensors=()):
    """Group numbered HF expert projections without inspecting model names."""
    protected_tensors = frozenset(protected_tensors)
    groups = {}
    consumed = set()
    for name, tensor in source.tensors.items():
        match = _EXPERT.match(name)
        if match:
            prefix, index, projection = match.groups()
            target = f"{prefix}.switch_mlp.{projection}.weight"
            groups.setdefault(target, {})[int(index)] = tensor
    plans = []
    for target, experts in groups.items():
        if target in source.tensors:
            raise ConversionError(f"native/grouped tensor collision: {target}")
        if sorted(experts) != list(range(len(experts))):
            raise ConversionError(f"expert indices must be contiguous from zero: {target}")
        members = tuple(experts[i] for i in range(len(experts)))
        first = members[0]
        if len(first.shape) != 2 or any(t.shape != first.shape or t.dtype != first.dtype for t in members):
            raise ConversionError(f"expert projection shapes/dtypes disagree: {target}")
        # Numbered quantized experts need all their metadata regrouped too;
        # refuse rather than silently orphan source scales/biases.
        if any(t.name.removesuffix("weight") + "scales" in source.tensors for t in members):
            raise ConversionError("numbered quantized experts require a native grouped source artifact")
        consumed.update(t.name for t in members)
        plans.append(Plan(target, members, (len(members), *first.shape), True,
                          not normalize_only and selected(target.removesuffix(".weight"), scope, grouped=True)))
    quant_metadata = set()
    for name, tensor in source.tensors.items():
        if name in consumed:
            continue
        grouped = len(tensor.shape) == 3 and ".switch_mlp." in name
        convert = (not normalize_only and name not in protected_tensors and name.endswith(".weight") and len(tensor.shape) in (2, 3)
                   and selected(name.removesuffix(".weight"), scope, grouped=grouped))
        if convert:
            quant_metadata.update(name.removesuffix("weight") + suffix for suffix in ("scales", "biases"))
        plans.append(Plan(name, (tensor,), tensor.shape, grouped, convert))
    return sorted((p for p in plans if p.name not in quant_metadata), key=lambda p: p.name)


class ShardWriter:
    def __init__(self, directory, number):
        self.directory = directory
        self.name = f"model-{number:05d}.safetensors"
        self.data_path = directory / f".data-{number:05d}.tmp"
        self.data = self.data_path.open("w+b")
        self.entries = {}
        self.size = 0

    def reserve(self, name, dtype, shape):
        if name in self.entries:
            raise ConversionError(f"duplicate output tensor {name}")
        size = math.prod(shape) * _WIDTHS[dtype]
        start = self.size
        self.size += size
        self.data.truncate(self.size)
        self.entries[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [start, self.size]}
        return start

    def write(self, offset, value):
        self.data.seek(offset)
        self.data.write(value)

    def finish(self):
        self.data.flush()
        os.fsync(self.data.fileno())
        header = json.dumps({"__metadata__": {"format": "mlx"}, **self.entries}, separators=(",", ":")).encode()
        header += b" " * ((-len(header)) % 64)
        tmp = self.directory / (self.name + ".tmp")
        with tmp.open("wb") as target:
            target.write(struct.pack("<Q", len(header)))
            target.write(header)
            self.data.seek(0)
            shutil.copyfileobj(self.data, target, length=8 << 20)
            target.flush()
            os.fsync(target.fileno())
        self.data.close()
        os.replace(tmp, self.directory / self.name)
        self.data_path.unlink()


class Shards:
    def __init__(self, path, max_bytes):
        self.path, self.max_bytes = path, max_bytes
        self.writer = None
        self.number = 0
        self.index = {}
        self.total_size = 0

    def reserve(self, specs):
        total = sum(math.prod(shape) * _WIDTHS[dtype] for _, dtype, shape in specs)
        if self.writer is None or self.writer.size and self.writer.size + total > self.max_bytes:
            if self.writer is not None:
                self.writer.finish()
            self.number += 1
            self.writer = ShardWriter(self.path, self.number)
        writer = self.writer
        offsets = {name: writer.reserve(name, dtype, shape) for name, dtype, shape in specs}
        for name in offsets:
            self.index[name] = writer.name
        self.total_size += total
        return writer, offsets

    def finish(self):
        if self.writer is not None:
            self.writer.finish()
            self.writer = None


def _quantized_geometry(source, tensor):
    cfg = source.quantization(tensor.name.removesuffix(".weight"))
    mode, bits, gs = cfg.get("mode", "affine"), cfg.get("bits"), cfg.get("group_size")
    if tensor.dtype != "U32" or type(bits) is not int or type(gs) is not int or bits not in (4, 8) or gs <= 0:
        raise ConversionError(f"unsupported quantized source descriptor: {tensor.name}")
    if mode not in {"affine", "mxfp4", "mxfp8"} or (mode == "mxfp4" and (bits, gs) != (4, 32)) or (mode == "mxfp8" and (bits, gs) != (8, 32)):
        raise ConversionError(f"unsupported source quantization {mode}/{bits}/{gs}")
    k = tensor.shape[-1] * 32 // bits
    expected = (*tensor.shape[:-1], k // gs)
    scale = source.tensors.get(tensor.name.removesuffix("weight") + "scales")
    allowed_scales = {"F16", "BF16", "F32", "F64"} if mode == "affine" else {"U8"}
    if k % gs or scale is None or scale.shape != expected or scale.dtype not in allowed_scales:
        raise ConversionError("quantized source scale geometry/dtype differs from weights")
    if mode == "affine":
        bias = source.tensors.get(tensor.name.removesuffix("weight") + "biases")
        if bias is None or bias.shape != expected or bias.dtype not in allowed_scales:
            raise ConversionError("quantized affine source bias geometry/dtype differs")
    return cfg, k


def _source_rows(source, tensor, start, end, allow_requantization):
    raw = tensor.rows(start, end)
    if tensor.dtype in {"BF16", "F16", "F32", "F64"}:
        # BF16/FP16 are exact in F32. A declared F64 source must retain its
        # precision until the quantizer chooses its grid and rounding; an
        # intermediate F32 cast can move values across E2M1 midpoints.
        return raw if tensor.dtype == "F64" else raw.astype(np.float32, copy=False)
    module = tensor.name.removesuffix(".weight")
    cfg = source.quantization(module)
    if tensor.dtype != "U32" or not allow_requantization:
        raise ConversionError(f"source {tensor.name} is already quantized or unsupported; explicit --allow-requantization is required")
    mode, bits, gs = cfg.get("mode", "affine"), cfg.get("bits"), cfg.get("group_size")
    scale_name = module + ".scales"
    if scale_name not in source.tensors:
        raise ConversionError(f"quantized source lacks {scale_name}")
    scales = source.tensors[scale_name].rows(start, end)
    if mode == "affine" and bits in (4, 8) and type(gs) is int:
        bias_name = module + ".biases"
        if bias_name not in source.tensors:
            raise ConversionError("affine source lacks biases")
        biases = source.tensors[bias_name].rows(start, end)
        codes = np.stack([(raw >> (bits * i)) & ((1 << bits) - 1) for i in range(32 // bits)], -1).reshape(len(raw), -1)
        return (codes.astype(np.float32) * np.repeat(scales.astype(np.float32), gs, -1)
                + np.repeat(biases.astype(np.float32), gs, -1))
    if mode == "mxfp8" and bits == 8 and gs == 32:
        codes = raw.view(np.uint8)
        exp, mant = (codes & 127) >> 3, codes & 7
        if np.any((exp == 15) & (mant == 7)):
            raise ConversionError("nonfinite MXFP8 source")
        mag = np.where(exp == 0, mant.astype(np.float32) * 2.**-9,
                       (1 + mant.astype(np.float32) / 8) * np.exp2(exp.astype(np.int16) - 7))
        value = np.where(codes & 128, -mag, mag)
        return value * np.repeat(np.exp2(scales.astype(np.int16) - 127), 32, -1)
    if mode == "mxfp4" and bits == 4 and gs == 32:
        # This is the explicitly authorized offline requantization path.
        # Decode ordinary MXFP4 block by block: the lossless runtime folding
        # helper correctly refuses wide exponent spreads, but those are the
        # rows an offline conversion must be able to constrain.
        if np.any(scales == 255):
            raise ConversionError("nonfinite MXFP4 source scale")
        codes = np.stack([(raw >> (4 * i)) & 15 for i in range(8)], -1).reshape(len(raw), -1)
        magnitude = np.asarray(SC_FORMAT.code_mag, np.float64)[codes & 7] / 2
        value = np.where(codes & 8, -magnitude, magnitude)
        return value * np.repeat(np.exp2(scales.astype(np.int16) - 127), 32, -1)
    raise ConversionError(f"unsupported source quantization {mode}/{bits}/{gs}")


def _chunks(plan, rows_per_chunk):
    for tensor in plan.members:
        count = math.prod(tensor.shape[:-1])
        for start in range(0, count, rows_per_chunk):
            yield tensor, start, min(start + rows_per_chunk, count)


def _mapped_bounded(executor, function, items, pending):
    if executor is None:
        yield from map(function, items)
        return
    queue = deque()
    iterator = iter(items)
    for _ in range(pending):
        try: queue.append(executor.submit(function, next(iterator)))
        except StopIteration: break
    while queue:
        yield queue.popleft().result()
        try: queue.append(executor.submit(function, next(iterator)))
        except StopIteration: pass


def _prepare_bound_sidecars(source):
    """Dispatch supported artifact schemas; never copy stale revision bindings."""
    from ...adapters.flash_next_conversion import MANIFEST_NAME, SIDECAR_NAME, prepare_ple_rebind
    for item in source.path.glob("*.json"):
        if item.name in {"config.json", "model.safetensors.index.json", MANIFEST_NAME, "mlx2-sc-mxfp4-conversion.json"}:
            continue
        if item.name == "mtp_draft_vocab.json":
            raise ConversionError("artifact-bound MTP draft vocabulary must be rebuilt with scripts/build_mtp_draft_vocab_manifest.py after conversion; automatic rebinding is unsupported")
        value = _json(item)
        if isinstance(value, dict) and {"config_sha256", "index_sha256", "source_index_sha256", "artifact_fingerprint", "target_fingerprint"} & value.keys():
            raise ConversionError(f"unsupported revision-bound sidecar {item.name}; rebuild it against the converted artifact")
    if (source.path / MANIFEST_NAME).exists() or (source.path / SIDECAR_NAME).exists():
        return prepare_ple_rebind(source)
    return None


def _output_identity(path):
    try:
        from ...adapters.standard_decoder import inspect_artifact
        artifact = inspect_artifact(path)
        return {"scheme": "mlx2-standard-decoder", **artifact["identity"]}
    except (ValueError, KeyError):
        value = Source(path)
        return value.identity


def convert_artifact(source, output, *, scope="mlp", rows_per_chunk=128, workers=1,
                     allow_requantization=False, normalize_only=False, shard_bytes=2 << 30,
                     progress=None):
    """Create a new artifact; source and existing nonempty outputs are untouched."""
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    if scope not in {"experts", "mlp", "all"}:
        raise ConversionError("scope must be experts, mlp or all")
    if source == output or source in output.parents or output in source.parents:
        raise ConversionError("source and output directories must be separate and non-nested")
    if type(rows_per_chunk) is not int or rows_per_chunk < 1 or type(workers) is not int or not 1 <= workers <= 32:
        raise ConversionError("positive row chunk and 1..32 CPU workers required")
    if type(shard_bytes) is not int or shard_bytes < 1024:
        raise ConversionError("shard byte limit must be at least 1024")
    if type(allow_requantization) is not bool or type(normalize_only) is not bool:
        raise ConversionError("conversion mode flags must be booleans")
    if output.exists() and any(output.iterdir()):
        raise ConversionError("refusing to overwrite a nonempty output directory")
    src = Source(source)
    sidecar_binding = _prepare_bound_sidecars(src)
    protected = sidecar_binding["protected_tensors"] if sidecar_binding is not None else ()
    plans = plan_tensors(src, scope, normalize_only=normalize_only, protected_tensors=protected)
    converted = [p for p in plans if p.convert]
    if not normalize_only and not converted:
        raise ConversionError("scope selects no convertible projection tensors")
    # Preflight every selected source before creating output or doing work.
    for plan in converted:
        member = plan.members[0]
        quantized = member.dtype not in {"BF16", "F16", "F32", "F64"}
        cfg, k = _quantized_geometry(src, member) if quantized else (None, member.shape[-1])
        if k % 128 or k > SC_FORMAT.max_reduction or plan.shape[-2] % 64:
            raise ConversionError(f"unsupported SC-MXFP4 projection geometry: {plan.name}, N={plan.shape[-2]}, K={k}")
        if quantized and not allow_requantization:
            raise ConversionError(f"already-quantized source {member.name} requires --allow-requantization")
    output.mkdir(parents=True, exist_ok=True)
    (output / ".incomplete").write_text(SCHEMA + "\n")
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[4], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        revision = "unavailable"
    receipt = {"schema": SCHEMA, "status": "converting", "started_at": time.time(),
               "source": src.identity, "source_checkpoint": source.name + "@sha256:" + src.fingerprint,
               "source_revision": revision, "format": SC_FORMAT.revision, "scope": scope,
               "normalize_only": normalize_only, "allow_requantization": allow_requantization,
               "source_code": {"repository": "https://github.com/shinyoungkang1/mxfp4-int8-m5", "revision": _UPSTREAM, "license": "MIT"},
               "conversion_driver_sha256": _sha(__file__), "workers": workers, "rows_per_chunk": rows_per_chunk,
               "projection_stats": [], "qualified": False, "selected": False, "observed_used": False}
    shards = Shards(output, shard_bytes)
    executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        config = json.loads(json.dumps(src.config))
        quant = dict(config.get("quantization", config.get("quantization_config")) or _QUANT)
        for ordinal, plan in enumerate(plans):
            if not plan.convert:
                dtype = plan.members[0].dtype
                writer, offsets = shards.reserve([(plan.name, dtype, plan.shape)])
                cursor = offsets[plan.name]
                for tensor in plan.members:
                    raw = tensor.raw()
                    for start in range(0, tensor.nbytes, 8 << 20):
                        chunk = raw[start:start + (8 << 20)]
                        writer.write(cursor, chunk)
                        cursor += chunk.nbytes
                continue
            member = plan.members[0]
            old, k = _quantized_geometry(src, member) if member.dtype == "U32" else (None, member.shape[-1])
            shape = (*plan.shape[:-1], k)
            scale_name = plan.name.removesuffix("weight") + "scales"
            code_shape, scale_shape = (*shape[:-1], k // 8), (*shape[:-1], k // 32)
            writer, offsets = shards.reserve([(plan.name, "U32", code_shape), (scale_name, "U8", scale_shape)])
            code_cursor, scale_cursor = offsets[plan.name], offsets[scale_name]
            stats = {"tensor": plan.name, "shape": list(shape), "source_dtype": member.dtype,
                     "source_tensors": [t.name for t in plan.members], "rows": 0, "constrained_rows": 0,
                     "constrained_blocks": 0, "source_requantized": old is not None}
            def convert_chunk(item):
                tensor, start, end = item
                weights = _source_rows(src, tensor, start, end, allow_requantization)
                packed = pack_numpy(weights, allow_requantization=True)
                return packed.codes, to_mxfp4_scales_numpy(packed), dict(packed.stats)
            for codes, scales, item_stats in _mapped_bounded(executor, convert_chunk,
                    _chunks(plan, rows_per_chunk), max(1, workers * 2)):
                writer.write(code_cursor, codes.tobytes(order="C")); code_cursor += codes.nbytes
                writer.write(scale_cursor, scales.tobytes(order="C")); scale_cursor += scales.nbytes
                for key in ("rows", "constrained_rows", "constrained_blocks"):
                    stats[key] += item_stats.get(key, 0)
            quant[plan.name.removesuffix(".weight")] = dict(_QUANT)
            receipt["projection_stats"].append(stats)
            if progress:
                progress({"completed": ordinal + 1, "total": len(plans), **stats})
        shards.finish()
        if not normalize_only:
            config["quantization"] = quant
            config["quantization_config"] = json.loads(json.dumps(quant))
        excluded = {"config.json", "model.safetensors.index.json", "mlx2-sc-mxfp4-conversion.json"}
        for item in source.iterdir():
            if item.is_file() and item.name not in excluded and item.suffix != ".safetensors" and not item.name.startswith("."):
                if item.suffix in {".json", ".txt", ".jinja", ".model", ".tiktoken", ".bin", ".ids", ".LICENSE"} or item.name.startswith(("LICENSE", "NOTICE", "README")):
                    shutil.copy2(item, output / item.name)
        _atomic(output / "config.json", config)
        _atomic(output / "model.safetensors.index.json", {"metadata": {"total_size": shards.total_size}, "weight_map": shards.index})
        if sidecar_binding is not None:
            from ...adapters.flash_next_conversion import finish_ple_rebind, rebound_ple_artifact_identity
            receipt["ple_sidecar_rebind"] = finish_ple_rebind(src, Source(output), sidecar_binding, conversion_schema=SCHEMA)
            identity = rebound_ple_artifact_identity(output)
        else:
            identity = _output_identity(output)
        receipt.update(status="complete", finished_at=time.time(), output=identity,
                       checkpoint=output.name + "@sha256:" + identity["fingerprint"],
                       converted_modules=len(converted), normalized_groups=sum(len(p.members) > 1 for p in plans),
                       source_unchanged=(Source(source).identity == src.identity))
        if not receipt["source_unchanged"]:
            raise ConversionError("source identity changed during conversion")
        _atomic(output / "mlx2-sc-mxfp4-conversion.json", receipt)
        (output / ".incomplete").unlink()
        return receipt
    except BaseException as error:
        receipt.update(status="failed", finished_at=time.time(), error=repr(error))
        _atomic(output / "mlx2-sc-mxfp4-conversion.json", receipt)
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if shards.writer is not None and not shards.writer.data.closed:
            shards.writer.data.close()
