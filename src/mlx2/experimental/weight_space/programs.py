"""Content-bound, reversible low-rank weight programs, using NumPy only.

Factors follow mlx2's serving convention: ``x @ A @ B``. Analytical repair
uses a published directional-edit mechanism; see provenance/weight-space.md.
Export does not qualify a patch, merge it into a checkpoint, or select it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mlx2.runtime.weight_space_metrics import (
    record_weight_space_bytes,
    set_weight_space_gauge,
    track_weight_space,
)

SCHEMA_VERSION = 1
MAX_RANK = 1024
MAX_TARGETS = 128
MAX_ELEMENTS = 16 * 1024 * 1024
MAX_TOTAL_ELEMENTS = 32 * 1024 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_ARTIFACT_BYTES = MAX_TOTAL_ELEMENTS * 4 + 1024 * 1024
NUMERICAL_POLICY = "unmerged_float32_factors"
_TARGET = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*\Z")
_STATUS = {"implemented": True, "qualified": False, "selected": False, "observed_used": False}


def _json(value):
    try:
        result = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must be finite JSON") from exc
    if len(result.encode()) > MAX_JSON_BYTES:
        raise ValueError("metadata exceeds JSON size bound")
    return result


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _identity(checkpoint, source_revision):
    if (
        not isinstance(checkpoint, str)
        or len(checkpoint) > 1024
        or checkpoint.count("@") != 1
        or any(not part or "*" in part for part in checkpoint.split("@"))
        or checkpoint.split("@")[-1] in {"main", "master", "latest", "HEAD"}
        or any(c.isspace() for c in checkpoint)
    ):
        raise ValueError("checkpoint must be exact artifact@fingerprint, without wildcards")
    if (
        not isinstance(source_revision, str)
        or not source_revision.strip()
        or len(source_revision) > 256
        or "*" in source_revision
        or source_revision in {"main", "master", "latest", "HEAD"}
        or any(c.isspace() for c in source_revision)
    ):
        raise ValueError("source_revision must identify the source revision")


def _array(value, *, ndim=2):
    array = np.asarray(value)
    if array.ndim != ndim or any(d <= 0 for d in array.shape) or array.size > MAX_ELEMENTS:
        raise ValueError("tensor shape is empty, invalid, or exceeds the element bound")
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError("tensors must contain finite real numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.ascontiguousarray(array, dtype="<f4")
    if not np.isfinite(array).all():
        raise ValueError("tensor values overflow float32")
    # A bytes-backed array cannot have its writable flag re-enabled by callers.
    return np.frombuffer(array.tobytes(), dtype="<f4").reshape(array.shape)


def _tensor_spec(name, array):
    return {"name": name, "shape": list(array.shape), "sha256": _sha(array.tobytes())}


def _session_restricted(value):
    if isinstance(value, dict):
        if value.get("contains_private_memory") is True or value.get("promotion_scope") == "session" or value.get("data_scope") == "session":
            return True
        return any(_session_restricted(item) for item in value.values())
    if isinstance(value, list):
        return any(_session_restricted(item) for item in value)
    return False


def _derived_provenance(payload, parents):
    payload["parent_provenance"] = [
        {"program_id": p.program_id, "construction": p.construction, "provenance": p.provenance}
        for p in parents
    ]
    if _session_restricted(payload):
        payload.update(data_scope="session", contains_private_memory=True, promotion_scope="session")
    return payload


@dataclass(frozen=True, eq=False)
class Factor:
    target: str
    a: np.ndarray
    b: np.ndarray


@dataclass(frozen=True, eq=False)
class WeightProgram:
    """Immutable factors and canonical JSON metadata; use ``create_program``."""

    checkpoint: str
    source_revision: str
    construction: str
    provenance_json: str
    approximate: bool
    numerical_policy: str
    factors: tuple[Factor, ...]

    @property
    def provenance(self):
        return json.loads(self.provenance_json)

    @property
    def rank(self):
        return self.factors[0].a.shape[1]

    @property
    def factor_map(self):
        return {factor.target: (factor.a, factor.b) for factor in self.factors}

    def _content(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "weight_program",
            "checkpoint": self.checkpoint,
            "source_revision": self.source_revision,
            "construction": self.construction,
            "provenance": self.provenance,
            "numerical_policy": self.numerical_policy,
            "approximate": self.approximate,
            "rank": self.rank,
            "targets": [
                {
                    "target": factor.target,
                    "a": _tensor_spec(f"{factor.target}.lora_a", factor.a),
                    "b": _tensor_spec(f"{factor.target}.lora_b", factor.b),
                }
                for factor in self.factors
            ],
            "status": dict(_STATUS),
        }

    @property
    def program_id(self):
        return "sha256:" + _sha(_json(self._content()).encode())

    def manifest(self):
        content = self._content()
        return {**content, "program_id": "sha256:" + _sha(_json(content).encode())}

    def to_dict(self):
        """Bounded JSON representation for existing capsule storage."""
        payload = {
            "manifest": self.manifest(),
            "factors": {
                f.target: {"a": f.a.tolist(), "b": f.b.tolist()} for f in self.factors
            },
        }
        _json(payload)
        return payload

    @classmethod
    def from_dict(cls, payload):
        _json(payload)
        if not isinstance(payload, dict) or set(payload) != {"manifest", "factors"}:
            raise ValueError("invalid embedded weight program")
        factors = payload["factors"]
        if not isinstance(factors, dict) or any(
            not isinstance(v, dict) or set(v) != {"a", "b"} for v in factors.values()
        ):
            raise ValueError("invalid embedded factors")
        return _from_manifest(payload["manifest"], {k: (v["a"], v["b"]) for k, v in factors.items()})


@track_weight_space("program_create")
def create_program(
    factors: Mapping,
    *,
    checkpoint,
    source_revision,
    construction,
    provenance,
    approximate=False,
    numerical_policy=NUMERICAL_POLICY,
):
    _identity(checkpoint, source_revision)
    if not isinstance(construction, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,95}", construction):
        raise ValueError("construction must be a bounded descriptive identifier")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("nonempty provenance is required")
    if not isinstance(approximate, bool) or numerical_policy != NUMERICAL_POLICY:
        raise ValueError("unsupported approximation or numerical policy")
    provenance_json = _json(provenance)
    if not isinstance(factors, Mapping) or not 1 <= len(factors) <= MAX_TARGETS:
        raise ValueError("a bounded nonempty target mapping is required")
    if any(not isinstance(target, str) for target in factors):
        raise ValueError("target must be an explicit module path")
    result = []
    total_elements = 0
    for target, pair in sorted(factors.items()):
        if not isinstance(target, str) or len(target) > 512 or not _TARGET.fullmatch(target):
            raise ValueError("target must be an explicit module path")
        if target.endswith((".weight", ".lora_a", ".lora_b")):
            raise ValueError("target identifies a module, not a tensor leaf")
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("each target requires an (A, B) pair")
        a, b = (_array(value) for value in pair)
        total_elements += a.size + b.size
        if total_elements > MAX_TOTAL_ELEMENTS:
            raise ValueError("program exceeds total factor element bound")
        if a.shape[1] != b.shape[0] or not 1 <= a.shape[1] <= MAX_RANK:
            raise ValueError("factor rank mismatch or rank outside 1..1024")
        result.append(Factor(target, a, b))
    rank = max(f.a.shape[1] for f in result)
    padded_elements = sum((f.a.shape[0] + f.b.shape[1]) * rank for f in result)
    if padded_elements > MAX_TOTAL_ELEMENTS:
        raise ValueError("program exceeds total factor element bound")
    padded = []
    for f in result:
        if f.a.shape[1] != rank:
            a = np.pad(f.a, ((0, 0), (0, rank - f.a.shape[1])))
            b = np.pad(f.b, ((0, rank - f.b.shape[0]), (0, 0)))
            f = Factor(f.target, _array(a), _array(b))
        padded.append(f)
    return WeightProgram(
        checkpoint, source_revision, construction, provenance_json,
        approximate, numerical_policy, tuple(padded),
    )


def _from_manifest(manifest, factors):
    if not isinstance(manifest, dict):
        raise ValueError("weight program manifest must be an object")  # noqa: TRY004 - artifact validation API
    try:
        program = create_program(
            factors,
            checkpoint=manifest["checkpoint"],
            source_revision=manifest["source_revision"],
            construction=manifest["construction"],
            provenance=manifest["provenance"],
            approximate=manifest["approximate"],
            numerical_policy=manifest["numerical_policy"],
        )
    except KeyError as exc:
        raise ValueError("weight program manifest is incomplete") from exc
    if manifest != program.manifest():
        raise ValueError("weight program manifest/content identity mismatch")
    return program


@track_weight_space("directional_repair")
def analytical_directional_repair(
    weights, directions, selected_columns, *, strength, checkpoint, source_revision, provenance,
):
    """Compile W' = W - strength*u*(mask*(u.T@W)) for output×input W.

    Directions must already be unit vectors. No calibration/model loading is
    performed. This float32 factorization does not reproduce BF16 fused rounding.
    """
    if not isinstance(strength, (int, float)) or isinstance(strength, bool) or not math.isfinite(strength):
        raise ValueError("strength must be finite")
    if not isinstance(weights, Mapping) or set(weights) != set(directions) or set(weights) != set(selected_columns):
        raise ValueError("weights, directions and selected_columns must cover identical targets")
    factors, detail = {}, {}
    for target, value in weights.items():
        w = _array(value)
        u = _array(directions[target], ndim=1)
        if len(u) != w.shape[0] or not np.isclose(np.linalg.norm(u.astype(np.float64)), 1, rtol=1e-5, atol=1e-6):
            raise ValueError("direction must be a unit vector matching output width")
        selected = selected_columns[target]
        if not isinstance(selected, (list, tuple)) or not selected or any(
            isinstance(j, bool) or not isinstance(j, (int, np.integer)) or not 0 <= j < w.shape[1]
            for j in selected
        ) or len(set(selected)) != len(selected):
            raise ValueError("selected columns must be unique valid integer indices")
        selected = sorted(int(j) for j in selected)
        a = np.zeros((w.shape[1], 1), dtype=np.float32)
        a[selected, 0] = u.astype(np.float64) @ w[:, selected].astype(np.float64)
        b = (-float(strength) * u.astype(np.float64))[None, :]
        factors[target] = a, b
        detail[target] = {
            "weight_sha256": _sha(w.tobytes()),
            "direction_sha256": _sha(u.tobytes()),
            "selected_columns": selected,
            "base_weight_frobenius_norm": float(np.linalg.norm(w.astype(np.float64))),
        }
    return create_program(
        factors, checkpoint=checkpoint, source_revision=source_revision,
        construction="analytical_directional_repair",
        provenance={"source": provenance, "strength": float(strength), "targets": detail},
    )


@track_weight_space("compose")
def compose_programs(programs, scales=None, *, provenance=None):
    """Concatenate factors to represent a fixed sum; no SVD or weight merging."""
    programs = tuple(programs)
    if not programs or len(programs) > MAX_TARGETS:
        raise ValueError("composition needs a bounded nonempty program sequence")
    if any(not isinstance(p, WeightProgram) for p in programs):
        raise ValueError("composition accepts WeightProgram inputs")
    first = programs[0]
    if any((p.checkpoint, p.source_revision) != (first.checkpoint, first.source_revision) for p in programs):
        raise ValueError("composed programs must bind the same checkpoint and source revision")
    scales = [1.0] * len(programs) if scales is None else list(scales)
    if len(scales) != len(programs) or any(
        isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s) for s in scales
    ):
        raise ValueError("one finite scale is required per program")
    rank = sum(p.rank for p in programs)
    if rank > MAX_RANK:
        raise ValueError("composed rank exceeds 1024; truncation must be explicitly requested")
    targets = sorted({f.target for p in programs for f in p.factors})
    factors = {}
    target_dims = {}
    for target in targets:
        present = [p.factor_map[target] for p in programs if target in p.factor_map]
        dims = {(a.shape[0], b.shape[1]) for a, b in present}
        if len(dims) != 1:
            raise ValueError("target dimensions differ across programs")
        n_in, n_out = dims.pop()
        target_dims[target] = n_in, n_out
    if sum((i + o) * rank for i, o in target_dims.values()) > MAX_TOTAL_ELEMENTS:
        raise ValueError("composition exceeds total factor element bound")
    for target in targets:
        n_in, n_out = target_dims[target]
        aa, bb = [], []
        for p, scale in zip(programs, scales):
            a, b = p.factor_map.get(target, (
                np.zeros((n_in, p.rank), dtype=np.float32),
                np.zeros((p.rank, n_out), dtype=np.float32),
            ))
            aa.append(a)
            bb.append(b.astype(np.float64) * scale)
        factors[target] = np.concatenate(aa, axis=1), np.concatenate(bb, axis=0)
    return create_program(
        factors, checkpoint=first.checkpoint, source_revision=first.source_revision,
        construction="fixed_composition", approximate=any(p.approximate for p in programs),
        provenance=_derived_provenance(
            {"source": provenance or {}, "parents": [p.program_id for p in programs], "scales": scales}, programs,
        ),
    )


@track_weight_space("truncate")
def truncate_program(program, rank, *, provenance=None):
    """Explicit approximate SVD compression, without materializing dense deltas."""
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank < program.rank:
        raise ValueError("truncation rank must be positive and below the input rank")
    factors, errors = {}, {}
    for f in program.factors:
        qa, ra = np.linalg.qr(f.a.astype(np.float64), mode="reduced")
        qb, rb = np.linalg.qr(f.b.T.astype(np.float64), mode="reduced")
        u, s, vh = np.linalg.svd(ra @ rb.T, full_matrices=False)
        retained = min(rank, len(s))
        roots = np.sqrt(s[:retained])
        factors[f.target] = (qa @ u[:, :retained]) * roots, roots[:, None] * (vh[:retained] @ qb.T)
        errors[f.target] = {
            "discarded_frobenius_norm": float(np.linalg.norm(s[retained:])),
            "original_frobenius_norm": float(np.linalg.norm(s)),
            "retained_rank": retained,
        }
    return create_program(
        factors, checkpoint=program.checkpoint, source_revision=program.source_revision,
        construction="svd_truncation", approximate=True,
        provenance=_derived_provenance(
            {"source": provenance or {}, "parent": program.program_id, "requested_rank": rank, "errors": errors}, [program],
        ),
    )


@track_weight_space("program_inspect")
def inspect_program(program):
    result = program.manifest()
    result["factor_statistics"] = []
    for f in program.factors:
        a, b = f.a.astype(np.float64), f.b.astype(np.float64)
        squared_norm = float(np.sum((a.T @ a) * (b @ b.T)))
        result["factor_statistics"].append({
            "target": f.target,
            "a_frobenius_norm": float(np.linalg.norm(a)),
            "b_frobenius_norm": float(np.linalg.norm(b)),
            "delta_frobenius_norm": math.sqrt(max(0, squared_norm)),
            "stored_bytes": f.a.nbytes + f.b.nbytes,
        })
    return result


def _config(program):
    return {
        "fine_tune_type": "lora",
        "lora_parameters": {"rank": program.rank, "scale": 1.0, "dropout": 0.0, "keys": [f.target for f in program.factors]},
        "weight_program": program.manifest(),
    }


@track_weight_space("program_export")
def export_program(program, path):
    """Atomically publish a complete LoRA directory; refuse nonempty outputs."""
    from safetensors.numpy import save_file

    # Reconstruct to validate even a caller-created dataclass instance.
    program = _from_manifest(program.manifest(), program.factor_map)
    path = Path(path).expanduser().absolute()
    if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
        raise ValueError("output must be absent or an empty directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.tmp-", dir=path.parent))
    try:
        arrays = {name: value for f in program.factors for name, value in (
            (f"{f.target}.lora_a", f.a), (f"{f.target}.lora_b", f.b),
        )}
        save_file(arrays, str(temporary / "adapters.safetensors"))
        (temporary / "weight_program.json").write_text(_json(program.manifest()) + "\n")
        (temporary / "adapter_config.json").write_text(_json(_config(program)) + "\n")
        for file in temporary.iterdir():
            with file.open("rb") as stream:
                os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    stored_bytes = sum(f.a.nbytes + f.b.nbytes for f in program.factors)
    record_weight_space_bytes("program_export", stored_bytes)
    set_weight_space_gauge("compiled_program_bytes", stored_bytes)
    return path


def _read_json(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("missing, symlinked or oversized JSON artifact")
    try:
        data = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid JSON artifact") from exc
    _json(data)
    return data


def _read_tensors(path, *, weights_payload=None):
    from safetensors.numpy import load, load_file

    if weights_payload is None:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_ARTIFACT_BYTES:
            raise ValueError("missing, symlinked or oversized tensor artifact")
    elif not isinstance(weights_payload, bytes) or len(weights_payload) > MAX_ARTIFACT_BYTES:
        raise ValueError("tensor payload must be bounded bytes")
    try:
        tensors = load_file(str(path)) if weights_payload is None else load(weights_payload)
    except Exception as exc:
        raise ValueError("invalid safetensors artifact") from exc
    if len(tensors) > MAX_TARGETS * 2 or sum(v.size for v in tensors.values()) > MAX_TOTAL_ELEMENTS:
        raise ValueError("tensor artifact exceeds element bound")
    return tensors


@track_weight_space("program_load")
def load_program(path, *, weights_payload=None):
    path = Path(path).expanduser()
    if path.is_symlink() or not path.is_dir():
        raise ValueError("program path must be a real directory")
    manifest = _read_json(path / "weight_program.json")
    tensors = _read_tensors(path / "adapters.safetensors", weights_payload=weights_payload)
    try:
        if not isinstance(manifest["targets"], list) or not 1 <= len(manifest["targets"]) <= MAX_TARGETS:
            raise ValueError("manifest target count exceeds bound")
        targets = [item["target"] for item in manifest["targets"]]
        expected = {f"{target}.{leaf}" for target in targets for leaf in ("lora_a", "lora_b")}
        if set(tensors) != expected or any(t.dtype != np.dtype("float32") for t in tensors.values()):
            raise ValueError("program tensor coverage or dtype mismatch")
        factors = {target: (tensors[f"{target}.lora_a"], tensors[f"{target}.lora_b"]) for target in targets}
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid manifest target metadata") from exc
    program = _from_manifest(manifest, factors)
    if _read_json(path / "adapter_config.json") != _config(program):
        raise ValueError("adapter config does not match bound weight program")
    return program


def validate_program_artifact(
    path, *, expected_checkpoint=None, expected_source_revision=None, weights_payload=None,
):
    program = load_program(path, weights_payload=weights_payload)
    if expected_checkpoint is not None and program.checkpoint != expected_checkpoint:
        raise ValueError("weight program checkpoint mismatch")
    if expected_source_revision is not None and program.source_revision != expected_source_revision:
        raise ValueError("weight program source revision mismatch")
    return program.manifest()


@track_weight_space("lora_import")
def import_lora(path, *, checkpoint, source_revision, provenance):
    """Import only existing mlx2 LoRA factor layout; consume scale exactly once."""
    path = Path(path).expanduser()
    if path.is_symlink() or not path.is_dir():
        raise ValueError("LoRA path must be a real directory")
    config = _read_json(path / "adapter_config.json")
    if not isinstance(config, dict) or config.get("fine_tune_type", "lora") != "lora":
        raise ValueError("only mlx2 LoRA configuration is supported")
    # Match read_lora_config(): a leftover manifest (even with the config
    # marker stripped) must go through load_program, never plain-LoRA import.
    manifest = path / "weight_program.json"
    if "weight_program" in config or manifest.exists() or manifest.is_symlink():
        program = load_program(path)
        if (program.checkpoint, program.source_revision) != (checkpoint, source_revision):
            raise ValueError("existing weight program binding mismatch")
        return program
    parameters = config.get("lora_parameters")
    if not isinstance(parameters, dict) or set(parameters) - {"rank", "scale", "dropout", "keys"}:
        raise ValueError("invalid mlx2 LoRA parameters")
    rank, scale, keys = parameters.get("rank", 8), parameters.get("scale", 20.0), parameters.get("keys")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= MAX_RANK:
        raise ValueError("invalid LoRA rank")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale):
        raise ValueError("invalid LoRA scale")
    if parameters.get("dropout", 0) != 0 or not isinstance(keys, list) or not keys or any(not isinstance(k, str) for k in keys) or len(set(keys)) != len(keys):
        raise ValueError("LoRA requires zero dropout and unique explicit keys")
    tensors = _read_tensors(path / "adapters.safetensors")
    if set(tensors) != {f"{key}.{leaf}" for key in keys for leaf in ("lora_a", "lora_b")}:
        raise ValueError("LoRA tensor coverage mismatch")
    factors = {}
    for key in keys:
        a, b = _array(tensors[f"{key}.lora_a"]), _array(tensors[f"{key}.lora_b"])
        if a.shape[1] != rank or b.shape[0] != rank:
            raise ValueError("LoRA tensor rank mismatch")
        factors[key] = a, b.astype(np.float64) * scale
    return create_program(
        factors, checkpoint=checkpoint, source_revision=source_revision,
        construction="imported_mlx2_lora",
        provenance={"source": provenance, "config_sha256": _sha(_json(config).encode()), "imported_scale": scale},
    )
