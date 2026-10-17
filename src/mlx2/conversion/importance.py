"""Source-bound activation moments and explicit routed-expert coverage.

Original artifact/merge contract. Statistics are approximate calibration
evidence, never exact target/APCv2 state. No MLX or accelerator is imported.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
import zipfile
from pathlib import Path

import numpy as np

from ..adapters.nemotron35_super_config import (
    SOURCE_CONFIG_SHA256,
    SOURCE_INDEX_SHA256,
    SOURCE_REVISION,
)
from .nvfp4 import sha256

SCHEMA = "mlx2.activation-imatrix.v1"
HEX = re.compile(r"[0-9a-f]{64}")


def statistics_specs(plan):
    """Only low-bit expert banks need activation-weighted refinement."""
    specs = {}
    for name, quant in plan["modules"].items():
        if quant["bits"] >= 8:
            continue
        shape = plan["schema"][name + ".scales"]["shape"]
        if len(shape) != 3:
            raise ValueError("Expected an explicit routed-expert scale bank")
        specs[name] = (shape[0], shape[-1] * quant["group_size"])
    return specs


def validate_binding(binding):
    expected = {"source_revision": SOURCE_REVISION, "source_config_sha256": SOURCE_CONFIG_SHA256,
                "source_index_sha256": SOURCE_INDEX_SHA256}
    if not isinstance(binding, dict) or any(binding.get(k) != v for k, v in expected.items()):
        raise ValueError("Activation statistics source binding differs")
    for key in ("proxy_artifact", "corpus_sha256", "collector_sha256"):
        if not isinstance(binding.get(key), str) or HEX.fullmatch(binding[key]) is None:
            raise ValueError(f"Activation statistics lack a content-bound {key}")
    if not isinstance(binding.get("processor_revision"), str) or not binding["processor_revision"]:
        raise ValueError("Activation statistics lack a processor revision")


class ActivationMoments:
    def __init__(self, specs, binding):
        validate_binding(binding)
        if not isinstance(specs, dict) or not specs:
            raise ValueError("Activation statistics require an explicit module schema")
        self.binding = dict(binding)
        self.specs = {}
        for name, shape in sorted(specs.items()):
            if (not isinstance(name, str) or not name or not isinstance(shape, (list, tuple))
                    or len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape)):
                raise ValueError("Invalid routed activation statistic geometry")
            self.specs[name] = tuple(shape)
        self.sums = {name: np.zeros(shape, np.float64) for name, shape in self.specs.items()}
        self.counts = {name: np.zeros(shape[0], np.uint64) for name, shape in self.specs.items()}
        self.samples = {}

    def add_reduced(self, name, sums, counts):
        if name not in self.specs:
            raise ValueError("Unplanned activation statistics module")
        sums, counts = np.asarray(sums), np.asarray(counts)
        if (sums.shape != self.specs[name] or counts.shape != (self.specs[name][0],)
                or not np.issubdtype(sums.dtype, np.floating)
                or not np.issubdtype(counts.dtype, np.integer)
                or not np.isfinite(sums).all() or np.any(sums < 0) or np.any(counts < 0)
                or np.any(sums[counts == 0] != 0)):
            raise ValueError("Invalid reduced activation sums/counts")
        counts = counts.astype(np.uint64)
        if np.any(self.counts[name] > np.iinfo(np.uint64).max - counts):
            raise ValueError("Activation counts overflow")
        self.sums[name] += sums
        self.counts[name] += counts
        if not np.isfinite(self.sums[name]).all():
            raise ValueError("Activation sums overflow")

    def record_sample(self, sample_id, *, modality, input_sha256):
        if (not isinstance(sample_id, str) or not sample_id or sample_id in self.samples
                or modality not in {"image_text", "text"}
                or not isinstance(input_sha256, str) or HEX.fullmatch(input_sha256) is None):
            raise ValueError("Invalid or duplicate calibration sample identity")
        self.samples[sample_id] = {"modality": modality, "input_sha256": input_sha256}

    def coverage(self, *, minimum_count=32):
        if type(minimum_count) is not int or minimum_count < 1:
            raise ValueError("Expert coverage minimum must be positive")
        details = {}
        for name, counts in self.counts.items():
            details[name] = {"experts": len(counts), "minimum_count": int(counts.min()),
                             "zero_experts": int(np.count_nonzero(counts == 0)),
                             "undercovered_experts": int(np.count_nonzero(counts < minimum_count)),
                             "assignments": int(counts.sum()),
                             "zero_energy_experts": int(np.count_nonzero((counts > 0) & ~self.sums[name].any(axis=1)))}
        return {"complete": all(v["undercovered_experts"] == 0 for v in details.values()),
                "minimum_required": minimum_count, "modules": details,
                "image_text_samples": sum(v["modality"] == "image_text" for v in self.samples.values()),
                "text_samples": sum(v["modality"] == "text" for v in self.samples.values())}

    def require_coverage(self, *, minimum_count=32, minimum_image_samples=1024):
        coverage = self.coverage(minimum_count=minimum_count)
        if (not coverage["complete"] or coverage["image_text_samples"] < minimum_image_samples):
            raise ValueError("Activation calibration lacks image-text samples or routed-expert coverage")
        return coverage

    def require_image_samples(self, *, minimum_image_samples=1024):
        coverage = self.coverage()
        if coverage["image_text_samples"] < minimum_image_samples:
            raise ValueError("Activation calibration lacks the required image-text samples")
        return coverage

    def expert_policy(self, *, minimum_count=32):
        if type(minimum_count) is not int or minimum_count < 1:
            raise ValueError("Expert coverage minimum must be positive")
        modules = {}
        for name, counts in self.counts.items():
            energetic = self.sums[name].any(axis=1)
            weighted = (counts >= minimum_count) & energetic
            reasons = [None if use_weighted else ("undercovered" if int(count) < minimum_count else "zero_energy")
                       for count, use_weighted in zip(counts, weighted, strict=True)]
            modules[name] = {
                "observation_counts": [int(value) for value in counts],
                "activation_weighted_mask": [bool(value) for value in weighted],
                "fallback_reasons": reasons,
                "activation_weighted_experts": int(np.count_nonzero(weighted)),
                "fallback_experts": int(np.count_nonzero(~weighted)),
                "fallback_undercovered_experts": int(np.count_nonzero(counts < minimum_count)),
                "fallback_zero_energy_experts": int(np.count_nonzero((counts >= minimum_count) & ~energetic)),
            }
        return {"minimum_observations": minimum_count, "modules": modules,
                "activation_weighted_experts": sum(v["activation_weighted_experts"] for v in modules.values()),
                "fallback_experts": sum(v["fallback_experts"] for v in modules.values())}

    def importance_for(self, name, expert, *, minimum_count=None):
        count = int(self.counts[name][expert])
        if minimum_count is not None:
            if type(minimum_count) is not int or minimum_count < 1:
                raise ValueError("Expert coverage minimum must be positive")
            if count < minimum_count:
                return None
        if count == 0:
            raise ValueError("Cannot fabricate importance for an unseen expert")
        values = (self.sums[name][expert] / count).astype(np.float32)
        # A visited but all-zero ReLU input has no measured sensitivity. Keep
        # the ordinary weight-MSE fit and report zero-energy experts explicitly.
        return values if np.any(values > 0) else None

    def merge(self, other):
        if self.binding != other.binding or self.specs != other.specs:
            raise ValueError("Cannot merge calibration with another source/proxy/corpus/collector")
        if self.samples.keys() & other.samples.keys():
            raise ValueError("Cannot double-count duplicate calibration samples")
        for name in self.specs:
            self.add_reduced(name, other.sums[name], other.counts[name])
        self.samples.update(other.samples)
        return self

    def save(self, destination):
        destination = Path(destination).absolute()
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.with_name(destination.name + ".incomplete-" + uuid.uuid4().hex)
        staging.mkdir()
        arrays, entries = {}, {}
        for index, name in enumerate(self.specs):
            sums, counts = f"s{index}", f"n{index}"
            arrays[sums], arrays[counts] = self.sums[name], self.counts[name]
            entries[name] = {"shape": list(self.specs[name]), "sums": sums, "counts": counts}
        np.savez(staging / "moments.npz", **arrays)
        manifest = {"schema": SCHEMA, "binding": self.binding, "entries": entries,
                    "samples": self.samples, "coverage": self.coverage(),
                    "moments_sha256": sha256(staging / "moments.npz"),
                    "qualified": False, "selected": False}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        for name in ("moments.npz", "manifest.json"):
            with (staging / name).open("rb") as handle:
                os.fsync(handle.fileno())
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        os.rename(staging, destination)
        return manifest

    @classmethod
    def load(cls, path, *, specs):
        path = Path(path).resolve()
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest.get("schema") != SCHEMA:
            raise ValueError("Unsupported activation statistics schema")
        expected = {name: list(shape) for name, shape in specs.items()}
        if {name: item["shape"] for name, item in manifest["entries"].items()} != expected:
            raise ValueError("Activation statistics module geometry differs")
        data = path / "moments.npz"
        if data.is_symlink() or sha256(data) != manifest["moments_sha256"]:
            raise ValueError("Activation statistics tensor bytes differ")
        keys, maximum = set(), 0
        for entry in manifest["entries"].values():
            keys.update((entry["sums"], entry["counts"]))
            maximum += (np.prod(entry["shape"]) + entry["shape"][0]) * 8 + 8192
        if len(keys) != 2 * len(manifest["entries"]):
            raise ValueError("Activation statistics arrays alias multiple modules")
        with zipfile.ZipFile(data) as archive:
            names = archive.namelist()
            if (len(names) != len(keys) or set(names) != {name + ".npy" for name in keys}
                    or sum(item.file_size for item in archive.infolist()) > maximum):
                raise ValueError("Activation statistics archive closure or size differs")
        result = cls(specs, manifest["binding"])
        with np.load(data, allow_pickle=False) as arrays:
            for name, entry in manifest["entries"].items():
                sums, counts = arrays[entry["sums"]], arrays[entry["counts"]]
                if sums.dtype != np.float64 or counts.dtype != np.uint64:
                    raise ValueError("Activation statistics dtype differs")
                result.add_reduced(name, sums, counts)
        for sample_id, sample in manifest["samples"].items():
            result.record_sample(sample_id, **sample)
        if result.coverage() != manifest["coverage"]:
            raise ValueError("Activation statistics coverage receipt differs")
        return result


def collector_digest(paths):
    digest = hashlib.sha256()
    for path in sorted(Path(p) for p in paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
