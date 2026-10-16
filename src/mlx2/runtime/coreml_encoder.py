"""Pinned MiniLM sentence encoder bundle executed by Core ML with ``CPU_AND_NE``.

The bundle (``mlx2.coreml-encoder.v1``) binds compiled fixed-shape models to
the pinned MiniLM checkpoint, its exact tokenizer bytes and representation
contract, and a passing build-time parity record against the ordinary MLX
reference. ``CPU_AND_NE`` excludes the GPU; it does not prove that the Neural
Engine executed anything, and ``MLComputePlan`` is anticipated placement only.

Importing this module must not import Core ML Tools, NumPy or MLX.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import multiprocessing
import os
import re
import stat
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ..adapters import minilm
from .ane_projection import _copy_compiled_tree, compiled_model_digest

MANIFEST_NAME = "mlx2-coreml-encoder.json"
SCHEMA = "mlx2.coreml-encoder.v1"
MAX_MANIFEST_BYTES = 1 << 20
MAX_BUCKETS = 16
PARITY_MIN_COSINE_FLOOR = 0.999
NORM_TOLERANCE = 1e-2
VOCAB_SIZE = 30522
PLAN_NOTE = "anticipated placement, not observed execution"
FINGERPRINT_SCHEMA = "mlx2.coreml-encoder.fingerprint.v1"
# Source, parity and placement fields are recorded by the builder; the
# runtime checks them for consistency but cannot re-derive them from a
# compiled graph.
SOURCE_RELATIONSHIP = "builder_attested_unverified"
BUILDER_TOOL = "mlx2.runtime.coreml_encoder_build"
BUILDER_OPTIONAL = frozenset({"graph_layout", "layer_norm"})
GRAPH_LAYOUTS = frozenset({"sequence_major", "channels_first"})
_HEX64 = re.compile(r"[0-9a-f]{64}")
_KEYS = {
    "schema", "family", "source", "tokenizer", "representation", "precision",
    "compute_units", "buckets", "builder", "build_parity", "compute_plan",
}
_BUCKET_KEYS = {
    "sequence_length", "batch", "path", "tree_sha256", "inputs", "output", "output_shape",
}
_INPUTS = {"input_ids": "int32", "attention_mask": "int32"}


def bundle_fingerprint(manifest) -> str:
    """Route identity: SHA-256 of a canonical JSON record of the served bytes.

    The record is ``{"schema": FINGERPRINT_SCHEMA, "manifest_sha256":
    sha256(canonical manifest), "tokenizer_sha256": ..., "bucket_tree_sha256":
    {"S": tree_sha256, ...}}``; canonical JSON is ``sort_keys=True``,
    ``separators=(",", ":")``, ``ensure_ascii=True``. Tree and tokenizer bytes
    are verified against these digests before use, so the fingerprint binds
    the compiled graph actually executed rather than the builder's claim
    about which checkpoint it was compiled from.
    """
    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()

    return hashlib.sha256(canonical({
        "schema": FINGERPRINT_SCHEMA,
        "manifest_sha256": hashlib.sha256(canonical(manifest)).hexdigest(),
        "tokenizer_sha256": manifest["tokenizer"]["sha256"],
        "bucket_tree_sha256": {
            str(bucket["sequence_length"]): bucket["tree_sha256"] for bucket in manifest["buckets"]
        },
    })).hexdigest()


def source_fingerprint() -> str:
    """The pinned MiniLM artifact fingerprint, as ``minilm.inspect_artifact`` derives it."""
    return hashlib.sha256(json.dumps(minilm.ARTIFACT_FILES, sort_keys=True).encode()).hexdigest()


def _coreml_runtime():
    # Deliberately inside the owned execution path; tests substitute a fake.
    try:
        import coremltools
    except ImportError as exc:
        raise RuntimeError("Core ML encoder execution requires coremltools in a macOS runtime") from exc
    return coremltools


def _fail(message):
    raise ValueError(f"Core ML encoder manifest: {message}")


def _exact(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        _fail(f"{name} must be an object with exactly {sorted(keys)}")
    return value


def _text(value, name, limit=256):
    if not isinstance(value, str) or not value or len(value) > limit:
        _fail(f"{name} must be nonempty bounded text")
    return value


def _digest(value, name):
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        _fail(f"{name} must be a lowercase SHA-256 digest")
    return value


def _count(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        _fail(f"{name} must be an integer >= {minimum}")
    return value


def _real(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(f"{name} must be a finite number")
    return float(value)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(name):
    _fail(f"nonstandard JSON constant {name}")


def _confined(root: Path, relative, *, directory: bool) -> Path:
    """Resolve a manifest path under ``root`` without symlinks or traversal."""
    if (
        not isinstance(relative, str) or not relative or len(relative) > 256
        or "\\" in relative or "\x00" in relative
    ):
        _fail("bundle paths must be bounded relative POSIX text")
    parts = relative.split("/")
    if PurePosixPath(relative).is_absolute() or any(part in {"", ".", ".."} for part in parts):
        _fail(f"bundle path escapes or is not normalized: {relative!r}")
    current = root
    for part in parts:
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            _fail(f"bundle path is missing: {relative!r}")
        if stat.S_ISLNK(info.st_mode):
            _fail(f"bundle path contains a symlink: {relative!r}")
    if directory and not stat.S_ISDIR(info.st_mode):
        _fail(f"bundle path must be a directory: {relative!r}")
    if not directory and not stat.S_ISREG(info.st_mode):
        _fail(f"bundle path must be a regular file: {relative!r}")
    return current


def _read_regular(path: Path, limit: int) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        _fail(f"cannot open {path.name} as a regular non-symlink file ({error.strerror})")
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            _fail(f"{path.name} must be a regular file")
        data = stream.read(limit + 1)
    if len(data) > limit:
        _fail(f"{path.name} exceeds its byte limit")
    return data


def _validate(manifest, root: Path):
    _exact(manifest, _KEYS, "manifest")
    if manifest["schema"] != SCHEMA:
        _fail(f"schema must be {SCHEMA}")
    if manifest["family"] != "minilm":
        _fail("family must be minilm")
    contract = minilm.representation_contract()
    source = _exact(manifest["source"], {
        "model", "revision", "artifact_fingerprint", "model_safetensors_sha256",
    }, "source")
    if source["model"] != contract["model"] or source["revision"] != minilm.MODEL_REVISION:
        _fail("source model/revision differs from the pinned MiniLM checkpoint")
    if source["artifact_fingerprint"] != source_fingerprint():
        _fail("source artifact fingerprint differs from the pinned MiniLM artifact")
    if source["model_safetensors_sha256"] != minilm.ARTIFACT_FILES["model.safetensors"][1]:
        _fail("source weights differ from the pinned MiniLM checkpoint")
    tokenizer = _exact(manifest["tokenizer"], {"path", "bytes", "sha256"}, "tokenizer")
    size, digest = minilm.ARTIFACT_FILES["tokenizer.json"]
    if tokenizer["path"] != "tokenizer.json" or tokenizer["bytes"] != size or tokenizer["sha256"] != digest:
        _fail("tokenizer must be the pinned MiniLM tokenizer.json")
    if type(tokenizer["bytes"]) is not int:
        _fail("tokenizer byte size must be an integer")
    _confined(root, tokenizer["path"], directory=False)
    if manifest["representation"] != contract:
        _fail("representation differs from the MiniLM representation contract")
    if manifest["precision"] != "float16":
        _fail("precision must be float16")
    if manifest["compute_units"] != "cpu_and_ne":
        _fail("compute_units must be cpu_and_ne")
    buckets = manifest["buckets"]
    if not isinstance(buckets, list) or not 1 <= len(buckets) <= MAX_BUCKETS:
        _fail(f"buckets must be a list of 1 to {MAX_BUCKETS} entries")
    lengths, paths = [], set()
    for bucket in buckets:
        _exact(bucket, _BUCKET_KEYS, "bucket")
        length = bucket["sequence_length"]
        if type(length) is not int or not 2 <= length <= minilm.MAX_SEQUENCE_LENGTH:
            _fail("bucket sequence_length must be an integer in [2, 256]")
        if lengths and length <= lengths[-1]:
            _fail("bucket sequence lengths must be strictly ascending")
        lengths.append(length)
        if type(bucket["batch"]) is not int or bucket["batch"] != 1:
            _fail("bucket batch must be 1")
        if not isinstance(bucket["path"], str) or not bucket["path"].endswith(".mlmodelc"):
            _fail("bucket path must name a compiled .mlmodelc directory")
        if bucket["path"] in paths:
            _fail("bucket paths must be unique")
        paths.add(bucket["path"])
        _confined(root, bucket["path"], directory=True)
        _digest(bucket["tree_sha256"], "bucket tree_sha256")
        if bucket["inputs"] != _INPUTS:
            _fail("bucket inputs must be int32 input_ids and attention_mask")
        if bucket["output"] != "embedding" or bucket["output_shape"] != [1, minilm.DIMENSIONS] or any(
            type(value) is not int for value in bucket["output_shape"]
        ):
            _fail("bucket output must be embedding [1, 384]")
    if lengths[-1] != minilm.MAX_SEQUENCE_LENGTH:
        _fail("the largest bucket must cover the 256-token MiniLM recipe")
    names = {str(length) for length in lengths}
    builder = manifest["builder"]
    required = {"tool", "source_sha256", "coremltools", "macos", "minimum_deployment_target"}
    if not isinstance(builder, dict) or not required <= set(builder) <= required | BUILDER_OPTIONAL:
        _fail(f"builder must contain {sorted(required)} and optionally {sorted(BUILDER_OPTIONAL)}")
    if "graph_layout" in builder and builder["graph_layout"] not in GRAPH_LAYOUTS:
        _fail(f"builder graph_layout must be one of {sorted(GRAPH_LAYOUTS)}")
    if "layer_norm" in builder:
        # fp16 LayerNorm cannot represent the source 1e-12 epsilon; the
        # builder records the substituted fp16-safe value.
        norm = _exact(builder["layer_norm"], {"precision", "epsilon", "source_epsilon"}, "builder layer_norm")
        if norm["precision"] != "float16" or _real(norm["source_epsilon"], "source_epsilon") != 1e-12:
            _fail("builder layer_norm must describe float16 LayerNorm of the 1e-12 source epsilon")
        if not 0 < _real(norm["epsilon"], "layer_norm epsilon") <= 1e-3:
            _fail("builder layer_norm epsilon must be in (0, 1e-3]")
    if builder["tool"] != BUILDER_TOOL:
        _fail(f"builder tool must be {BUILDER_TOOL}")
    _digest(builder["source_sha256"], "builder source_sha256")
    for key in ("coremltools", "macos", "minimum_deployment_target"):
        _text(builder[key], f"builder {key}")
    parity = _exact(manifest["build_parity"], {
        "reference", "threshold_min_cosine", "compute_units", "cases", "per_bucket",
        "min_cosine", "max_abs_error", "passed",
    }, "build_parity")
    if parity["reference"] != "mlx_bert_cpu_float32":
        _fail("build parity reference must be mlx_bert_cpu_float32")
    threshold = _real(parity["threshold_min_cosine"], "parity threshold")
    if not PARITY_MIN_COSINE_FLOOR <= threshold <= 1:
        _fail(f"parity threshold must be in [{PARITY_MIN_COSINE_FLOOR}, 1]")
    if parity["compute_units"] != ["cpu_only", "cpu_and_ne"]:
        _fail("build parity must cover cpu_only and cpu_and_ne")
    _count(parity["cases"], "parity cases", 1)
    if parity["passed"] is not True:
        _fail("build parity did not pass")
    per_bucket = parity["per_bucket"]
    if not isinstance(per_bucket, dict) or set(per_bucket) != names:
        _fail("build parity must report every bucket")
    for name, row in per_bucket.items():
        _exact(row, {"cases", "min_cosine", "max_abs_error"}, f"parity bucket {name}")
        _count(row["cases"], f"parity bucket {name} cases", 1)
        if _real(row["min_cosine"], "parity min_cosine") < threshold:
            _fail(f"bucket {name} parity is below the threshold")
        if _real(row["max_abs_error"], "parity max_abs_error") < 0:
            _fail("parity max_abs_error must be nonnegative")
    if _real(parity["min_cosine"], "parity min_cosine") < threshold:
        _fail("build parity min_cosine is below the threshold")
    if _real(parity["max_abs_error"], "parity max_abs_error") < 0:
        _fail("parity max_abs_error must be nonnegative")
    plan = _exact(manifest["compute_plan"], {"available", "note", "per_bucket", "placement_gate"}, "compute_plan")
    if type(plan["available"]) is not bool or plan["note"] != PLAN_NOTE:
        _fail("compute_plan must be an anticipated-placement record")
    rows = plan["per_bucket"]
    if not isinstance(rows, dict) or set(rows) != (names if plan["available"] else set()):
        _fail("compute_plan per_bucket must cover every bucket exactly when available, else be empty")
    for name, row in rows.items():
        _exact(row, {"neural_engine", "cpu", "unknown", "neural_engine_compute_fraction"}, f"compute_plan bucket {name}")
        for key in ("neural_engine", "cpu", "unknown"):
            _count(row[key], f"compute_plan {name} {key}")
        if not 0 <= _real(row["neural_engine_compute_fraction"], "neural_engine_compute_fraction") <= 1:
            _fail("neural_engine_compute_fraction must be in [0, 1]")
    gate = plan["placement_gate"]
    if not plan["available"]:
        if gate is not None:
            _fail("placement_gate must be null when the compute plan is unavailable")
        return
    _exact(gate, {"min_neural_engine_compute_fraction", "passed_buckets", "cpu_planned_buckets"}, "placement_gate")
    minimum = _real(gate["min_neural_engine_compute_fraction"], "placement gate minimum")
    if not 0 <= minimum <= 1:
        _fail("placement gate minimum must be in [0, 1]")
    # The recorded gate must be exactly the partition its own fractions and
    # threshold imply (the builder's coreml_encoder_build.placement_gate rule).
    if gate != expected_placement_gate(rows, minimum):
        _fail("placement_gate differs from the partition implied by the per-bucket fractions")


def expected_placement_gate(per_bucket, minimum) -> dict:
    """Complete disjoint partition: fraction >= minimum passes, every other bucket is CPU-planned."""
    passed = sorted(int(name) for name, row in per_bucket.items()
                    if row["neural_engine_compute_fraction"] >= minimum)
    return {
        "min_neural_engine_compute_fraction": minimum,
        "passed_buckets": passed,
        "cpu_planned_buckets": sorted(int(name) for name in per_bucket if int(name) not in passed),
    }


@dataclass(frozen=True)
class EncoderBundle:
    """A validated bundle; ``manifest`` is a private copy of the reviewed JSON."""

    path: Path
    manifest: dict
    manifest_sha256: str

    @property
    def buckets(self) -> tuple[int, ...]:
        return tuple(bucket["sequence_length"] for bucket in self.manifest["buckets"])

    @property
    def fingerprint(self) -> str:
        """Identity of exactly what this route serves (see ``bundle_fingerprint``)."""
        return bundle_fingerprint(self.manifest)

    def source_attestation(self) -> dict:
        """The builder's claim about its source checkpoint; not verifiable from the bundle."""
        source = self.manifest["source"]
        return {
            "model": source["model"],
            "revision": source["revision"],
            "source_fingerprint": source["artifact_fingerprint"],
            "model_safetensors_sha256": source["model_safetensors_sha256"],
            "source_relationship": SOURCE_RELATIONSHIP,
        }

    def read_tokenizer(self) -> bytes:
        """Return exactly the pinned tokenizer bytes, read once without following links."""
        size, digest = minilm.ARTIFACT_FILES["tokenizer.json"]
        data = _read_regular(_confined(self.path, "tokenizer.json", directory=False), size)
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Core ML encoder bundle tokenizer.json differs from the pinned MiniLM tokenizer")
        return data


def inspect_bundle(bundle_dir) -> EncoderBundle:
    """Validate the manifest and bundle layout. Compiled trees are hashed by the executor."""
    root = Path(bundle_dir).expanduser().resolve()
    if not root.is_dir():
        _fail("bundle must be a directory")
    raw = _read_regular(root / MANIFEST_NAME, MAX_MANIFEST_BYTES)
    try:
        manifest = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"invalid JSON ({error})")
    _validate(manifest, root)
    return EncoderBundle(root, manifest, hashlib.sha256(raw).hexdigest())


def load_manifest(bundle_dir) -> dict:
    """Return the strictly validated manifest of a bundle directory."""
    return copy.deepcopy(inspect_bundle(bundle_dir).manifest)




def anticipated_placement(manifest) -> dict:
    """Per-bucket build-time placement label; never a claim of observed execution."""
    plan = manifest["compute_plan"]
    if not plan["available"]:
        return {str(bucket["sequence_length"]): "not_inspected" for bucket in manifest["buckets"]}
    passed = set(plan["placement_gate"]["passed_buckets"])
    return {
        str(bucket["sequence_length"]): (
            "neural_engine_planned" if bucket["sequence_length"] in passed else "cpu_planned"
        )
        for bucket in manifest["buckets"]
    }


def _plan_counts(runtime, paths) -> dict | None:
    """Anticipated per-bucket operation placement under CPU_AND_NE, or None."""
    planner = getattr(runtime.models, "MLComputePlan", None)
    if planner is None:
        return None
    names = {"MLCPUComputeDevice": "cpu", "MLNeuralEngineComputeDevice": "neural_engine"}
    result = {}
    for length, path in paths.items():
        plan = planner.load_from_path(str(path), compute_units=runtime.ComputeUnit.CPU_AND_NE)
        program = plan.model_structure.program
        if program is None:
            raise ValueError("Core ML encoder compute plan is not an ML Program")
        counts = {"neural_engine": 0, "cpu": 0, "unknown": 0}

        def visit(block, depth=0, plan=plan, counts=counts):
            if depth > 16:
                raise ValueError("Core ML encoder compute plan exceeds the nesting limit")
            for operation in block.operations:
                if sum(counts.values()) >= 65536:
                    raise ValueError("Core ML encoder compute plan exceeds the operation limit")
                usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
                device = type(None if usage is None else usage.preferred_compute_device).__name__
                if device == "MLGPUComputeDevice":
                    raise ValueError("CPU_AND_NE compute plan unexpectedly prefers GPU")
                counts[names.get(device, "unknown")] += 1
                for child in operation.blocks:
                    visit(child, depth + 1)

        for function in program.functions.values():
            visit(function.block)
        result[str(length)] = counts
    return result


def _load_models(runtime, paths) -> dict:
    units = runtime.ComputeUnit.CPU_AND_NE
    return {
        length: runtime.models.CompiledMLModel(str(path), compute_units=units)
        for length, path in paths.items()
    }


def _worker_main(commands, results, paths, runtime_module):
    """Persistent spawned Core ML owner; never imports MLX.

    Core ML's Python predict can hold the GIL for a whole native call, which
    would serialize every other thread of the serving process. The parent only
    waits on a pipe, which releases the GIL.
    """
    try:
        import importlib

        import numpy as np

        runtime = _coreml_runtime() if runtime_module is None else importlib.import_module(runtime_module)
        paths = {int(length): Path(path) for length, path in paths.items()}
        models = _load_models(runtime, paths)
        results.send({
            "ready": True, "pid": os.getpid(),
            "coremltools_version": str(getattr(runtime, "__version__", "unknown")),
        })
    except BaseException as error:  # noqa: BLE001 - report every load failure to the owner
        results.send({"ready": False, "error": f"{type(error).__name__}: {error}"})
        return
    while True:
        try:
            command = commands.recv()
        except (EOFError, OSError):
            return
        if not isinstance(command, tuple) or not command or command[0] == "stop":
            return
        ticket = command[1] if len(command) > 1 else None
        try:
            if command[0] == "predict":
                _, _, length, ids, mask = command
                outputs = models[length].predict({"input_ids": ids, "attention_mask": mask})
                payload = {"ticket": ticket, "ok": True, "outputs": (
                    {str(key): np.asarray(value) for key, value in outputs.items()}
                    if isinstance(outputs, dict) else None
                )}
            elif command[0] == "plan":
                payload = {"ticket": ticket, "ok": True, "plan": _plan_counts(runtime, paths)}
            else:
                raise ValueError(f"unknown command {command[0]!r}")
        except Exception as error:  # noqa: BLE001 - return native failures to the owner
            payload = {"ticket": ticket, "ok": False, "error": f"{type(error).__name__}: {error}"}
        results.send(payload)


class _ProcessBackend:
    kind = "process"

    def __init__(self, paths, *, runtime_module, load_timeout):
        context = multiprocessing.get_context("spawn")
        child_commands, self._commands = context.Pipe(duplex=False)
        self._results, child_results = context.Pipe(duplex=False)
        self._process = context.Process(
            target=_worker_main,
            args=(child_commands, child_results, {length: str(path) for length, path in paths.items()},
                  runtime_module),
            name="mlx2-coreml-encoder", daemon=True,
        )
        self._tickets = itertools.count(1)
        try:
            self._process.start()
            child_commands.close()
            child_results.close()
            ready = self._receive(load_timeout)
            if ready.get("ready") is not True:
                raise RuntimeError(f"Core ML encoder worker failed to load: {ready.get('error')}")
        except BaseException:
            self.abort()
            raise
        self.pid = ready.get("pid")
        self.version = str(ready.get("coremltools_version"))

    def _receive(self, timeout):
        if not self._results.poll(timeout):
            raise TimeoutError(f"Core ML encoder worker exceeded its {timeout:g} s deadline")
        try:
            reply = self._results.recv()
        except (EOFError, OSError) as error:
            raise RuntimeError("Core ML encoder worker exited") from error
        if not isinstance(reply, dict):
            raise RuntimeError("Core ML encoder worker returned a malformed reply")  # noqa: TRY004 - a worker fault, not a caller type error
        return reply

    def _call(self, command, timeout):
        if not self._process.is_alive():
            raise RuntimeError("Core ML encoder worker is not alive")
        ticket = next(self._tickets)
        try:
            self._commands.send((command[0], ticket, *command[1:]))
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError("Core ML encoder worker exited") from error
        reply = self._receive(timeout)
        if reply.get("ticket") != ticket:
            raise RuntimeError("Core ML encoder worker reply does not match the request")
        if reply.get("ok") is not True:
            raise RuntimeError(f"Core ML encoder worker failed: {reply.get('error')}")
        return reply

    def predict(self, length, ids, mask, timeout):
        return self._call(("predict", length, ids, mask), timeout).get("outputs")

    def plan(self, timeout):
        return self._call(("plan",), timeout).get("plan")

    def abort(self):
        # Reap whenever a child was started, including one that already
        # exited: an unjoined dead child stays a zombie.
        if self._process.pid is not None:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(5)
        self._commands.close()
        self._results.close()

    def close(self):
        if self._process.pid is not None and self._process.is_alive():
            try:
                self._commands.send(("stop",))
            except (BrokenPipeError, OSError):
                pass
            self._process.join(10)
        self.abort()


class _InProcessBackend:
    """Test and diagnostic backend: same runtime, no process isolation or deadline."""

    kind = "in_process"
    pid = None

    def __init__(self, paths):
        self._runtime = _coreml_runtime()
        self._paths = dict(paths)
        self._models = _load_models(self._runtime, self._paths)
        self.version = str(getattr(self._runtime, "__version__", "unknown"))

    def predict(self, length, ids, mask, timeout):
        return self._models[length].predict({"input_ids": ids, "attention_mask": mask})

    def plan(self, timeout):
        return _plan_counts(self._runtime, self._paths)

    def abort(self):
        self._models = {}

    close = abort


def _validated_output(np, outputs):
    if not isinstance(outputs, dict) or set(outputs) != {"embedding"}:
        raise RuntimeError("Core ML encoder returned unexpected output features")
    output = np.asarray(outputs["embedding"])
    # Core ML may widen an fp16 feature to float32 for Python.
    if output.dtype not in (np.float16, np.float32) or output.shape != (1, minilm.DIMENSIONS):
        raise RuntimeError("Core ML encoder output must be float16/float32 [1, 384]")
    values = output.astype(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("Core ML encoder output must be finite")
    if abs(float(np.linalg.norm(values)) - 1.0) > NORM_TOLERANCE:
        raise RuntimeError("Core ML encoder output is not L2-normalized")
    return [float(value) for value in values[0]]


def _positive_seconds(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 3600:
        raise ValueError(f"{name} must be a number of seconds in (0, 3600]")
    return float(value)


class CoreMLEncoderExecutor:
    """Width-one fixed-bucket encoder; GPU excluded, never falls back.

    Every bucket tree is hashed, copied into private storage and re-hashed
    before Core ML loads the copy with ``CPU_AND_NE``. By default a persistent
    spawned worker owns Core ML; each call has a deadline. A dead, late or
    malformed worker reply, or any native failure, raises and marks the
    executor broken until it is closed. Prediction and close share one lock,
    so close waits for an in-flight call (bounded by its deadline).
    """

    def __init__(self, bundle, *, backend="process", timeout=30.0, load_timeout=600.0,
                 runtime_module=None):
        if backend not in {"process", "in_process"}:
            raise ValueError("Core ML encoder backend must be process or in_process")
        if runtime_module is not None and (
            backend != "process" or not isinstance(runtime_module, str)
            or not all(part.isidentifier() for part in runtime_module.split("."))
        ):
            raise ValueError("runtime_module must name an importable module for the process backend")
        self.timeout = _positive_seconds(timeout, "timeout")
        load_timeout = _positive_seconds(load_timeout, "load_timeout")
        info = bundle if isinstance(bundle, EncoderBundle) else inspect_bundle(bundle)
        self.bundle = info
        self._lock = threading.RLock()
        self._closed = False
        self._broken = None
        self._backend = None
        self._paths = {}
        self._digests = {}
        self._prediction_starts = 0
        self._predictions = {length: 0 for length in info.buckets}
        self._storage = tempfile.TemporaryDirectory(prefix="mlx2-coreml-encoder-")
        try:
            for bucket in info.manifest["buckets"]:
                length, declared = bucket["sequence_length"], bucket["tree_sha256"]
                source = _confined(info.path, bucket["path"], directory=True)
                if compiled_model_digest(source) != declared:
                    raise ValueError(f"Core ML encoder bucket {length} bytes differ from the manifest")
                target = Path(self._storage.name) / f"seq{length}.mlmodelc"
                _copy_compiled_tree(source, target)
                if compiled_model_digest(target) != declared:
                    raise ValueError(f"Core ML encoder bucket {length} private copy differs from the manifest")
                self._paths[length], self._digests[length] = target, declared
            self._backend = (
                _ProcessBackend(self._paths, runtime_module=runtime_module, load_timeout=load_timeout)
                if backend == "process" else _InProcessBackend(self._paths)
            )
        except BaseException:
            self._storage.cleanup()
            raise

    @property
    def buckets(self) -> tuple[int, ...]:
        return self.bundle.buckets

    def _require_usable(self):
        if self._closed:
            raise RuntimeError("Core ML encoder executor is closed")
        if self._broken is not None:
            raise RuntimeError(f"Core ML encoder executor is broken ({self._broken}); restart the route")

    def _break(self, error):
        self._broken = f"{type(error).__name__}: {error}"
        self._backend.abort()

    def predict(self, input_ids, attention_mask):
        """Return one validated 384-float embedding for an exact-bucket input."""
        import numpy as np

        with self._lock:
            self._require_usable()
            arrays = []
            for name, value in (("input_ids", input_ids), ("attention_mask", attention_mask)):
                if not isinstance(value, np.ndarray) or value.dtype != np.int32 or value.ndim != 2 or value.shape[0] != 1:
                    raise ValueError(f"Core ML encoder {name} must be an int32 [1, S] array")
                arrays.append(np.array(value, dtype=np.int32, order="C", copy=True))
            ids, mask = arrays
            length = ids.shape[1]
            if mask.shape != ids.shape or length not in self._paths:
                raise ValueError(f"Core ML encoder input length {length} is not a declared bucket {list(self.buckets)}")
            tokens = int(mask.sum())
            if (
                not np.isin(mask, (0, 1)).all() or tokens < 1
                or not (mask[0, :tokens] == 1).all()
                or (ids < 0).any() or (ids >= VOCAB_SIZE).any() or (ids[0, tokens:] != 0).any()
            ):
                raise ValueError("Core ML encoder inputs must be right-padded valid ids with a prefix mask")
            self._prediction_starts += 1
            try:
                vector = _validated_output(np, self._backend.predict(length, ids, mask, self.timeout))
            except BaseException as error:
                self._break(error)
                raise
            self._predictions[length] += 1
            return vector

    def status(self) -> dict:
        """Host-only facts; never waits on or calls the native runtime."""
        backend = self._backend
        return {
            "schema": SCHEMA,
            "backend": "coreml",
            "isolation": None if backend is None else backend.kind,
            "worker_pid": None if backend is None else backend.pid,
            "deadline_seconds": self.timeout,
            "compute_units": "cpu_and_ne",
            "gpu": "excluded",
            "precision": "float16",
            "buckets": list(self.buckets),
            "bundle_fingerprint": self.bundle.fingerprint,
            "bundle_manifest_sha256": self.bundle.manifest_sha256,
            "source": self.bundle.source_attestation(),
            "tree_sha256": {str(length): digest for length, digest in self._digests.items()},
            "coremltools_version": None if backend is None else backend.version,
            "prediction_starts": self._prediction_starts,
            "predictions_completed": sum(self._predictions.values()),
            "per_bucket_predictions": {str(length): count for length, count in self._predictions.items()},
            "neural_engine_observed": False,
            "broken": self._broken,
            "closed": self._closed,
        }

    def compute_plan(self) -> dict:
        """Anticipated per-bucket placement of the retained copies; never observed use."""
        with self._lock:
            self._require_usable()
            # A missing MLComputePlan API is reported as unavailable (None);
            # every raised failure, including a GPU-preferring plan, is a
            # contract violation of the loaded graph and breaks the executor.
            try:
                rows = self._backend.plan(self.timeout)
            except BaseException as error:
                self._break(error)
                raise
            if rows is not None and (not isinstance(rows, dict) or set(rows) != {str(b) for b in self.buckets} or any(
                not isinstance(row, dict) or set(row) != {"neural_engine", "cpu", "unknown"}
                or any(type(value) is not int or value < 0 for value in row.values())
                for row in rows.values()
            )):
                error = RuntimeError("Core ML encoder worker returned a malformed compute plan")
                self._break(error)
                raise error
            return {
                "available": rows is not None, "note": PLAN_NOTE, "per_bucket": rows or {},
                "observed_used": False, "neural_engine_observed": False,
            }

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._backend is not None:
                self._backend.close()
            self._storage.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
