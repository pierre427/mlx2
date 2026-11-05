"""Declarative owner assets for bounded, CPU-only experiment tools.

JSON/NPZ configuration cannot enable GPU work, inject Python callbacks, or grant
semantic-memory authority. Expert file paths are operator configuration only;
tools receive registered IDs and bounded pumps, never arbitrary read access.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from .asset_arrays import bounded_npz_arrays
from .excision import ExcisionCapture, LayerExcisionPlan
from .experts import ExpertKey, ExpertMaskPlan, ExpertRecord, ExpertResidencyController
from .memory import CalibrationSplit, CapturedActivations, MemorySource
from .programs import _identity
from .tool_service import MAX_FILE_BYTES, MAX_JSON_BYTES, MAX_RESOURCES, _private_file

SCHEMA = "mlx2.weight-experiment-assets.v1"


def _fields(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= set(value) or set(value) - set(required) - set(optional):
        raise ValueError("experiment asset fields differ from the declared schema")
    return value


def _arrays(root, relative):
    return bounded_npz_arrays(_private_file(root, relative), maximum_bytes=MAX_FILE_BYTES)


def _stat_identity(info):
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise ValueError("expert source must be a regular operator-owned file")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


class BoundExpertFiles:
    """Stat-bound random reads, with content verified by each ExpertRecord.

    No whole-file reads or persistent descriptors. Stat identity catches normal
    source changes; it is not claimed as a cryptographic source-file hash.
    Each controller checks the exact per-slice SHA-256 before publishing bytes.
    """

    def __init__(self, files, records, source_fingerprint):
        if not isinstance(files, dict) or not 1 <= len(files) <= 128:
            raise ValueError("expert source file index must contain 1..128 files")
        records = tuple(records)
        if not records or len(records) > 65536 or any(not isinstance(record, ExpertRecord) for record in records):
            raise ValueError("bounded expert records required")
        self.files, self.identities = {}, {}
        self.source_fingerprint = source_fingerprint
        for name, value in files.items():
            if not isinstance(name, str) or not name or not isinstance(value, str):
                raise ValueError("expert source requires explicit file names and absolute paths")
            path = Path(value)
            if not path.is_absolute():
                raise ValueError("expert source paths must be absolute operator bindings")
            self.files[name] = path
            self.identities[name] = _stat_identity(path.lstat())
        self.records = {record.key: record for record in records}
        for record in records:
            if (record.source_id not in self.files
                    or record.offset + record.nbytes > self.identities[record.source_id][2]):
                raise ValueError("expert record exceeds its registered file extent")

    def validate(self):
        for name, path in self.files.items():
            if _stat_identity(path.lstat()) != self.identities[name]:
                raise ValueError("expert source file revision changed")
        return self.source_fingerprint

    def read(self, record, relative_offset, length):
        if self.records.get(record.key) != record:
            raise ValueError("expert record is not registered")
        if (type(relative_offset) is not int or type(length) is not int or relative_offset < 0
                or not 1 <= length <= 64 * 1024**2 or relative_offset + length > record.nbytes):
            raise ValueError("expert read must be one bounded indexed slice")
        self.validate()
        descriptor = os.open(self.files[record.source_id], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            if _stat_identity(os.fstat(descriptor)) != self.identities[record.source_id]:
                raise ValueError("expert source changed while opening")
            value = os.pread(descriptor, length, record.offset + relative_offset)
            if (len(value) != length or _stat_identity(os.fstat(descriptor)) != self.identities[record.source_id]):
                raise ValueError("expert source truncated or changed while reading")
            self.validate()
            return value
        finally:
            os.close(descriptor)


def configure_experiment_assets(service, root, raw_experiments, *, checkpoint, source_revision):
    """Attach only host-registered inputs. A serving manager is attached elsewhere."""
    _identity(checkpoint, source_revision)
    if "assets" not in raw_experiments:
        return
    root = Path(root)
    payload = json.loads(_private_file(root, raw_experiments["assets"], maximum=MAX_JSON_BYTES))
    groups = {"expert_sources", "excision_inventories", "excision_captures", "memory_calibrations", "quality_captures"}
    _fields(payload, {"schema"}, groups)
    if payload["schema"] != SCHEMA:
        raise ValueError("unknown experiment assets schema")
    for group in groups:
        values = payload.get(group, [])
        if not isinstance(values, list) or len(values) > MAX_RESOURCES:
            raise ValueError("experiment asset registry exceeds bounds")
    if sum(len(payload.get(group, [])) for group in groups) > MAX_RESOURCES:
        raise ValueError("aggregate experiment asset registry exceeds bounds")

    loaded_bytes = 0

    def load_arrays(relative):
        nonlocal loaded_bytes
        arrays = _arrays(root, relative)
        loaded_bytes += sum(value.nbytes for value in arrays.values())
        if loaded_bytes > MAX_FILE_BYTES:
            raise ValueError("aggregate experiment captures exceed the loaded tensor byte bound")
        return arrays

    inventories = {}
    for item in payload.get("excision_inventories", []):
        _fields(item, {"inventory_id", "layer_count", "hidden_size"})
        if item["inventory_id"] in inventories:
            raise ValueError("duplicate excision inventory")
        inventories[item["inventory_id"]] = item
        service.register_excision_inventory(**item, checkpoint=checkpoint, source_revision=source_revision)
    for item in payload.get("excision_captures", []):
        _fields(item, {"capture_id", "inventory_id", "removed_layers", "arrays", "excised_outputs",
                       "reference_outputs", "sample_ids", "input_digest", "calibration_ids", "heldout_ids"})
        inventory = inventories[item["inventory_id"]]
        plan = LayerExcisionPlan(checkpoint, source_revision, inventory["layer_count"],
                                 tuple(item["removed_layers"]), inventory["hidden_size"])
        arrays = load_arrays(item["arrays"])
        capture = ExcisionCapture(plan.fingerprint, tuple(item["sample_ids"]), arrays[item["excised_outputs"]],
                                  arrays[item["reference_outputs"]], item["input_digest"])
        service.register_excision_capture(item["capture_id"], capture=capture,
            split=CalibrationSplit(tuple(item["calibration_ids"]), tuple(item["heldout_ids"])))

    for item in payload.get("expert_sources", []):
        _fields(item, {"source_id", "plan", "arrays", "importance"}, {"records", "files", "residency"})
        plan = ExpertMaskPlan.from_dict(item["plan"])
        if (plan.checkpoint, plan.source_revision) != (checkpoint, source_revision):
            raise ValueError("expert plan checkpoint/source differs from configured engine")
        arrays = load_arrays(item["arrays"])
        importance = {int(layer): arrays[name] for layer, name in item["importance"].items()}
        if len(importance) != len(item["importance"]):
            raise ValueError("expert importance repeats numeric layer identity")
        raw_records = item.get("records", [])
        if not isinstance(raw_records, list) or len(raw_records) > 65536:
            raise ValueError("expert record index exceeds bounds")
        records = tuple(ExpertRecord(**{**row, "key": ExpertKey(**row["key"])}) for row in raw_records)
        controller = None
        if "residency" in item:
            settings = _fields(item["residency"], {"budget_bytes", "chunk_bytes"},
                               {"overlay_bytes", "bytes_per_second", "maximum_pending"})
            bound = BoundExpertFiles(item.get("files"), records, plan.source_fingerprint)
            controller = ExpertResidencyController(plan=plan, records=records, reader=bound.read,
                source_validator=bound.validate, **settings)
        elif "files" in item:
            raise ValueError("expert files require explicit bounded residency settings")
        service.register_expert_source(item["source_id"], plan=plan, importance=importance,
                                       records=records, controller=controller)

    for item in payload.get("memory_calibrations", []):
        _fields(item, {"capture_id", "source", "arrays", "sample_ids", "targets", "calibration_ids", "heldout_ids"})
        source = MemorySource.from_dict(item["source"])
        if source.checkpoint != checkpoint:
            raise ValueError("memory source differs from configured checkpoint")
        arrays = load_arrays(item["arrays"])
        if not isinstance(item["targets"], dict) or not 1 <= len(item["targets"]) <= 32:
            raise ValueError("memory captures require bounded explicit targets")
        captures = []
        for target, refs in item["targets"].items():
            _fields(refs, {"inputs", "target_deltas"})
            captures.append(CapturedActivations(target, tuple(item["sample_ids"]),
                                               arrays[refs["inputs"]], arrays[refs["target_deltas"]]))
        service.register_memory_calibration(item["capture_id"], captures=tuple(captures),
            split=CalibrationSplit(tuple(item["calibration_ids"]), tuple(item["heldout_ids"])),
            source=source, source_revision=source_revision)

    for item in payload.get("quality_captures", []):
        _fields(item, {"capture_id", "arrays", "reference_logits", "candidate_logits", "next_token_ids",
                       "tokenizer_fingerprint", "input_digest"}, {"mask"})
        arrays = load_arrays(item["arrays"])
        service.register_quality_capture(item["capture_id"], reference_logits=arrays[item["reference_logits"]],
            candidate_logits=arrays[item["candidate_logits"]], next_token_ids=arrays[item["next_token_ids"]],
            tokenizer_fingerprint=item["tokenizer_fingerprint"], input_digest=item["input_digest"],
            mask=arrays[item["mask"]] if "mask" in item else None)
