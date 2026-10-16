"""Core ML encoder bundle contract and executor, with a fake Core ML runtime.

Nothing here executes Core ML, MLX on GPU, or a real compiled model.
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS

import coreml_fake_runtime as fake
import numpy as np
import pytest

from mlx2.runtime import ane_projection as ap
from mlx2.runtime import coreml_encoder as ce


@pytest.fixture
def tokenizer_bytes(monkeypatch):
    # Contract tests never depend on the developer's Hugging Face cache.
    return fake.pin_tokenizer(monkeypatch, fake.tiny_tokenizer_bytes())


@pytest.fixture
def bundle(tmp_path, tokenizer_bytes):
    return fake.write_bundle(tmp_path / "bundle", tokenizer_bytes)


@pytest.fixture
def runtime(monkeypatch):
    """In-process recording fake; asserts the exact CPU_AND_NE policy object."""
    policy = object()
    state = NS(loads=[], inputs=[], output=None, enter=None, release=None, policy=policy, operations=None)

    class Model:
        def __init__(self, path, *, compute_units):
            assert compute_units is policy
            state.loads.append(Path(path))
            self.path = Path(path)

        def predict(self, features):
            state.inputs.append((self.path.name, features))
            if state.enter is not None:
                state.enter.set()
                assert state.release.wait(5)
            if state.output is not None:
                return state.output
            return {"embedding": fake.fake_vector(features["input_ids"], features["attention_mask"])}

    cpu = type("MLCPUComputeDevice", (), {})()
    ne = type("MLNeuralEngineComputeDevice", (), {})()
    state.operations = [NS(operator_name="cast", blocks=[NS(operations=[NS(operator_name="linear", blocks=[], device=ne)])], device=cpu)]

    class Plan:
        @classmethod
        def load_from_path(cls, path, *, compute_units):
            assert compute_units is policy
            return NS(
                model_structure=NS(program=NS(functions={"main": NS(block=NS(operations=state.operations))})),
                get_compute_device_usage_for_mlprogram_operation=lambda op: NS(preferred_compute_device=op.device),
            )

    fake_runtime = NS(models=NS(CompiledMLModel=Model, MLComputePlan=Plan),
                      ComputeUnit=NS(CPU_AND_NE=policy), __version__="fake-in-process")
    monkeypatch.setattr(ce, "_coreml_runtime", lambda: fake_runtime)
    return state


def edit(path, change):
    manifest = json.loads((path / ce.MANIFEST_NAME).read_text())
    change(manifest)
    (path / ce.MANIFEST_NAME).write_text(json.dumps(manifest))


def padded(ids, length):
    values = np.zeros((1, length), dtype=np.int32)
    mask = np.zeros((1, length), dtype=np.int32)
    values[0, :len(ids)] = ids
    mask[0, :len(ids)] = 1
    return values, mask


def test_valid_bundle_loads_private_copies_with_cpu_and_ne_only(bundle, runtime):
    manifest = ce.load_manifest(bundle)
    assert [row["sequence_length"] for row in manifest["buckets"]] == [32, 64, 128, 256]
    with ce.CoreMLEncoderExecutor(bundle, backend="in_process") as executor:
        assert len(runtime.loads) == 4
        for row, retained in zip(manifest["buckets"], runtime.loads):
            source = bundle / row["path"]
            assert retained != source and ap.compiled_model_digest(retained) == row["tree_sha256"]
        (bundle / "models/seq32.mlmodelc/model.mil").write_bytes(b"changed after admission")
        assert ap.compiled_model_digest(runtime.loads[0]) == manifest["buckets"][0]["tree_sha256"]
        ids, mask = padded([101, 7592, 102], 32)
        vector = executor.predict(ids, mask)
        assert type(vector) is list and len(vector) == 384 and all(type(v) is float for v in vector)
        np.testing.assert_allclose(vector, fake.fake_vector(ids, mask)[0], rtol=0, atol=0)
        name, features = runtime.inputs[-1]
        assert name == "seq32.mlmodelc" and features["input_ids"] is not ids
        status = executor.status()
        assert status["per_bucket_predictions"]["32"] == 1 and status["predictions_completed"] == 1
        assert status["compute_units"] == "cpu_and_ne" and status["gpu"] == "excluded"
        assert status["neural_engine_observed"] is False and status["isolation"] == "in_process"
        plan = executor.compute_plan()
        assert plan["per_bucket"]["32"] == {"neural_engine": 1, "cpu": 1, "unknown": 0}
        assert plan["observed_used"] is plan["neural_engine_observed"] is False
    assert not any(path.exists() for path in runtime.loads)
    with pytest.raises(RuntimeError, match="closed"):
        executor.predict(ids, mask)


def test_gpu_plan_preference_is_a_contract_failure(bundle, runtime):
    gpu = type("MLGPUComputeDevice", (), {})()
    runtime.operations = [NS(operator_name="matmul", blocks=[], device=gpu)]
    with ce.CoreMLEncoderExecutor(bundle, backend="in_process") as executor:
        with pytest.raises(ValueError, match="prefers GPU"):
            executor.compute_plan()
        assert "prefers GPU" in executor.status()["broken"]
        with pytest.raises(RuntimeError, match="broken"):
            executor.predict(*padded([101, 102], 32))
        with pytest.raises(RuntimeError, match="broken"):
            executor.compute_plan()


def test_malformed_plan_breaks_the_executor(bundle, runtime):
    runtime.operations = None  # the native plan object is unusable
    with ce.CoreMLEncoderExecutor(bundle, backend="in_process") as executor:
        with pytest.raises(TypeError):
            executor.compute_plan()
        assert executor.status()["broken"]


def _set(path, value):
    def change(manifest):
        target = manifest
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return change


def _drop(path):
    def change(manifest):
        target = manifest
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]
    return change


@pytest.mark.parametrize("change", [
    _set(("schema",), "mlx2.coreml-encoder.v2"),
    _set(("family",), "bert"),
    _set(("extra",), 1),
    _drop(("builder",)),
    _set(("source", "revision"), "0" * 40),
    _set(("source", "artifact_fingerprint"), "0" * 64),
    _set(("source", "model_safetensors_sha256"), "0" * 64),
    _set(("source", "model"), "sentence-transformers/all-MiniLM-L12-v2"),
    _set(("tokenizer", "sha256"), "0" * 64),
    _set(("tokenizer", "bytes"), 1),
    _set(("tokenizer", "path"), "vocab.txt"),
    _set(("representation", "dimensions"), 256),
    _set(("representation", "qualified"), True),
    _set(("precision",), "float32"),
    _set(("compute_units",), "all"),
    _set(("compute_units",), "cpu_and_gpu"),
    _set(("buckets",), []),
    _set(("buckets", 0, "sequence_length"), 64),
    _set(("buckets", 3, "sequence_length"), 512),
    _set(("buckets", 0, "sequence_length"), True),
    _set(("buckets", 0, "batch"), 2),
    _set(("buckets", 0, "inputs"), {"input_ids": "int64", "attention_mask": "int32"}),
    _set(("buckets", 0, "output"), "last_hidden_state"),
    _set(("buckets", 0, "output_shape"), [1, 256]),
    _set(("buckets", 0, "tree_sha256"), "ABC"),
    _set(("buckets", 0, "extra"), 1),
    _set(("builder", "tool"), "other"),
    _drop(("builder", "macos")),
    _set(("build_parity", "passed"), False),
    _set(("build_parity", "min_cosine"), 0.99),
    _set(("build_parity", "threshold_min_cosine"), 0.9),
    _set(("build_parity", "reference"), "torch"),
    _set(("build_parity", "compute_units"), ["cpu_and_ne"]),
    _set(("build_parity", "per_bucket", "32", "min_cosine"), 0.5),
    _drop(("build_parity", "per_bucket", "64")),
    _set(("build_parity", "per_bucket", "64", "cases"), 0),
    _set(("builder", "notes"), "free text"),
    _set(("builder", "graph_layout"), "nchw"),
    _set(("builder", "layer_norm", "precision"), "float32"),
    _set(("builder", "layer_norm", "source_epsilon"), 1e-5),
    _set(("builder", "layer_norm", "epsilon"), 0),
    _set(("builder", "layer_norm", "extra"), 1),
    _set(("compute_plan", "placement_gate"), None),
    _set(("compute_plan", "note"), "observed on the Neural Engine"),
    _drop(("compute_plan", "per_bucket", "32")),
    _set(("compute_plan", "per_bucket", "32", "gpu"), 1),
    _set(("compute_plan", "per_bucket", "32", "neural_engine_compute_fraction"), 1.5),
    _drop(("compute_plan", "per_bucket", "32", "neural_engine_compute_fraction")),
    _drop(("compute_plan", "placement_gate")),
    _set(("compute_plan", "placement_gate", "extra"), 1),
    _set(("compute_plan", "placement_gate", "min_neural_engine_compute_fraction"), 1.5),
    # Contradictory receipts: every recorded gate must equal the partition
    # implied by its own threshold and the per-bucket fractions.
    _set(("compute_plan", "placement_gate", "passed_buckets"), [256, 128, 64, 32]),
    _set(("compute_plan", "placement_gate", "passed_buckets"), [32, 64, 128]),
    _set(("compute_plan", "placement_gate", "cpu_planned_buckets"), [32]),
    _set(("compute_plan", "placement_gate", "passed_buckets"), [32, 48, 64, 128, 256]),
    _set(("compute_plan", "placement_gate", "min_neural_engine_compute_fraction"), 0.99),
    _set(("compute_plan", "per_bucket", "64", "neural_engine_compute_fraction"), 0.2),
])
def test_manifest_fields_fail_closed(bundle, change):
    edit(bundle, change)
    with pytest.raises(ValueError, match="Core ML encoder manifest"):
        ce.load_manifest(bundle)


def test_placement_gate_is_a_recomputed_partition_and_cpu_buckets_are_reported(tmp_path, tokenizer_bytes):
    bundle = fake.write_bundle(tmp_path / "mixed", tokenizer_bytes, fractions={32: 0.4, 256: 0.0})
    edit(bundle, _drop(("builder", "graph_layout")))
    manifest = ce.load_manifest(bundle)
    assert manifest["compute_plan"]["placement_gate"] == {
        "min_neural_engine_compute_fraction": 0.9, "passed_buckets": [64, 128], "cpu_planned_buckets": [32, 256],
    }
    assert ce.anticipated_placement(manifest) == {
        "32": "cpu_planned", "64": "neural_engine_planned", "128": "neural_engine_planned", "256": "cpu_planned",
    }
    unavailable = fake.write_bundle(tmp_path / "unplanned", tokenizer_bytes, plan_available=False)
    assert set(ce.anticipated_placement(ce.load_manifest(unavailable)).values()) == {"not_inspected"}
    edit(unavailable, _set(("compute_plan", "placement_gate"), {
        "min_neural_engine_compute_fraction": 0.9, "passed_buckets": [], "cpu_planned_buckets": []}))
    with pytest.raises(ValueError, match="must be null"):
        ce.load_manifest(unavailable)


@pytest.mark.parametrize("fractions", [{}, {32: 0.0}, {32: 0.9, 64: 0.8999, 128: 1.0, 256: 0.0}])
def test_gate_rule_matches_the_builder(fractions):
    build = pytest.importorskip("mlx2.runtime.coreml_encoder_build")
    rows = {str(n): {"neural_engine_compute_fraction": fractions.get(n, 0.95)} for n in (32, 64, 128, 256)}
    assert ce.expected_placement_gate(rows, build.MIN_NE_COMPUTE_FRACTION) == build.placement_gate(rows)


def test_bundle_fingerprint_binds_served_bytes_not_the_source_claim(tmp_path, tokenizer_bytes):
    first = ce.inspect_bundle(fake.write_bundle(tmp_path / "a", tokenizer_bytes))
    manifest = first.manifest
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    expected = ce.hashlib.sha256(canonical({
        "schema": "mlx2.coreml-encoder.fingerprint.v1",
        "manifest_sha256": ce.hashlib.sha256(canonical(manifest)).hexdigest(),
        "tokenizer_sha256": manifest["tokenizer"]["sha256"],
        "bucket_tree_sha256": {str(row["sequence_length"]): row["tree_sha256"] for row in manifest["buckets"]},
    })).hexdigest()
    assert first.fingerprint == expected != ce.source_fingerprint()
    assert first.source_attestation()["source_relationship"] == "builder_attested_unverified"
    assert first.source_attestation()["source_fingerprint"] == ce.source_fingerprint()
    # Same claimed source, different compiled bytes: a different identity.
    second = ce.inspect_bundle(fake.write_bundle(tmp_path / "b", tokenizer_bytes, buckets=(64, 256)))
    assert second.fingerprint != first.fingerprint
    assert second.source_attestation() == first.source_attestation()


@pytest.mark.parametrize("raw", [
    b'{"schema": "a", "schema": "b"}', b'{"schema": NaN}', b"[]", b"\xff",
])
def test_manifest_json_is_strict(bundle, raw):
    (bundle / ce.MANIFEST_NAME).write_bytes(raw)
    with pytest.raises(ValueError, match="Core ML encoder manifest"):
        ce.load_manifest(bundle)


@pytest.mark.parametrize("path", ["../outside.mlmodelc", "/tmp/x.mlmodelc", "models/./seq32.mlmodelc",
                                  "models//seq32.mlmodelc", "models\\seq32.mlmodelc", "models/missing.mlmodelc"])
def test_bucket_paths_are_confined(bundle, path):
    edit(bundle, _set(("buckets", 0, "path"), path))
    with pytest.raises(ValueError, match="Core ML encoder manifest"):
        ce.load_manifest(bundle)


def test_symlinks_are_rejected_everywhere(bundle, tmp_path, runtime):
    elsewhere = tmp_path / "elsewhere.mlmodelc"
    (bundle / "models/seq32.mlmodelc").rename(elsewhere)
    (bundle / "models/seq32.mlmodelc").symlink_to(elsewhere)
    with pytest.raises(ValueError, match="symlink"):
        ce.load_manifest(bundle)
    (bundle / "models/seq32.mlmodelc").unlink()
    elsewhere.rename(bundle / "models/seq32.mlmodelc")
    tokenizer = tmp_path / "tokenizer.json"
    (bundle / "tokenizer.json").rename(tokenizer)
    (bundle / "tokenizer.json").symlink_to(tokenizer)
    with pytest.raises(ValueError, match="symlink"):
        ce.load_manifest(bundle)
    (bundle / "tokenizer.json").unlink()
    tokenizer.rename(bundle / "tokenizer.json")
    manifest = tmp_path / "manifest.json"
    (bundle / ce.MANIFEST_NAME).rename(manifest)
    (bundle / ce.MANIFEST_NAME).symlink_to(manifest)
    with pytest.raises(ValueError, match="regular non-symlink"):
        ce.load_manifest(bundle)
    (bundle / ce.MANIFEST_NAME).unlink()
    manifest.rename(bundle / ce.MANIFEST_NAME)
    (bundle / "models/seq64.mlmodelc/link").symlink_to(bundle / "tokenizer.json")
    with pytest.raises(ValueError, match="regular files"):
        ce.CoreMLEncoderExecutor(bundle, backend="in_process")
    assert not runtime.loads


def test_tokenizer_bytes_are_verified(bundle):
    data = (bundle / "tokenizer.json").read_bytes()
    (bundle / "tokenizer.json").write_bytes(data[:-1] + b" ")
    info = ce.inspect_bundle(bundle)
    with pytest.raises(ValueError, match="differs from the pinned MiniLM tokenizer"):
        info.read_tokenizer()


def test_tree_digest_mismatch_and_changed_private_copy_fail_before_load(bundle, runtime, monkeypatch):
    (bundle / "models/seq128.mlmodelc/weights/weight.bin").write_bytes(b"drift")
    with pytest.raises(ValueError, match="bucket 128 bytes differ"):
        ce.CoreMLEncoderExecutor(bundle, backend="in_process")
    bundle = fake.write_bundle(bundle.parent / "second", bundle.joinpath("tokenizer.json").read_bytes())
    created = []
    original = ce._copy_compiled_tree

    def corrupting(source, destination):
        original(source, destination)
        created.append(destination)
        target = destination / "model.mil"
        target.chmod(0o600)
        target.write_bytes(b"swapped after copy")

    monkeypatch.setattr(ce, "_copy_compiled_tree", corrupting)
    with pytest.raises(ValueError, match="private copy differs"):
        ce.CoreMLEncoderExecutor(bundle, backend="in_process")
    assert created and not created[0].parent.exists()
    assert not runtime.loads


@pytest.mark.parametrize("make", [
    lambda: padded([101, 102], 40),
    lambda: (np.zeros((1, 32), np.int64), np.ones((1, 32), np.int32)),
    lambda: (np.zeros((2, 32), np.int32), np.ones((2, 32), np.int32)),
    lambda: ([[101, 102]], [[1, 1]]),
    lambda: (padded([101, 102], 32)[0], padded([101, 102], 64)[1]),
    lambda: (padded([101, 102], 32)[0], np.zeros((1, 32), np.int32)),
    lambda: (padded([101, 102], 32)[0], np.roll(padded([101, 102], 32)[1], 1)),
    lambda: (padded([101, 102], 32)[0] + 1, padded([101, 102], 32)[1]),
    lambda: (padded([101, 40000], 32)[0], padded([101, 40000], 32)[1]),
    lambda: (padded([101, 102], 32)[0], padded([101, 102], 32)[1] * 2),
])
def test_invalid_inputs_fail_before_native_predict(bundle, runtime, make):
    with ce.CoreMLEncoderExecutor(bundle, backend="in_process") as executor:
        with pytest.raises(ValueError):
            executor.predict(*make())
        assert not runtime.inputs and executor.status()["prediction_starts"] == 0
        assert executor.status()["broken"] is None


@pytest.mark.parametrize("output", [
    {"embedding": np.ones((1, 383), np.float32) / 383 ** 0.5},
    {"embedding": np.ones((384,), np.float32) / 384 ** 0.5},
    {"embedding": np.full((1, 384), np.nan, np.float32)},
    {"embedding": np.full((1, 384), np.inf, np.float16)},
    {"embedding": np.ones((1, 384), np.float32)},
    {"embedding": np.ones((1, 384), np.float64) / 384 ** 0.5},
    {"embedding": np.ones((1, 384), np.int32)},
    {"embedding": np.ones((1, 384), np.float32) / 384 ** 0.5, "extra": np.zeros(1)},
    {"hidden": np.ones((1, 384), np.float32) / 384 ** 0.5},
    [np.ones((1, 384), np.float32) / 384 ** 0.5],
])
def test_invalid_native_output_breaks_the_executor(bundle, runtime, output):
    runtime.output = output
    with ce.CoreMLEncoderExecutor(bundle, backend="in_process") as executor:
        with pytest.raises(RuntimeError, match="Core ML encoder"):
            executor.predict(*padded([101, 102], 32))
        status = executor.status()
        assert status["prediction_starts"] == 1 and status["predictions_completed"] == 0
        assert status["broken"]
        runtime.output = None
        with pytest.raises(RuntimeError, match="broken"):
            executor.predict(*padded([101, 102], 32))


def test_float16_output_within_norm_tolerance_is_accepted(bundle, runtime):
    vector = np.ones((1, 384), np.float16) / np.float16(384 ** 0.5)
    runtime.output = {"embedding": vector * np.float16(1.005)}
    with ce.CoreMLEncoderExecutor(bundle, backend="in_process") as executor:
        assert len(executor.predict(*padded([101, 102], 32))) == 384


def test_close_waits_for_inflight_prediction(bundle, runtime):
    runtime.enter, runtime.release = threading.Event(), threading.Event()
    executor = ce.CoreMLEncoderExecutor(bundle, backend="in_process")
    retained = runtime.loads[0]
    results = []
    predictor = threading.Thread(target=lambda: results.append(executor.predict(*padded([101, 102], 32))))
    closer = threading.Thread(target=executor.close)
    try:
        predictor.start()
        assert runtime.enter.wait(5)
        closer.start()
        closer.join(0.05)
        assert closer.is_alive() and retained.exists()
    finally:
        runtime.release.set()
        predictor.join(5)
        if closer.ident is not None:
            closer.join(5)
        executor.close()
    assert len(results) == 1 and not retained.exists()
    assert executor.status()["closed"] is True


@pytest.mark.parametrize("kwargs", [
    {"backend": "gpu"}, {"timeout": 0}, {"timeout": float("nan")}, {"load_timeout": True},
    {"backend": "in_process", "runtime_module": "coreml_fake_runtime"}, {"runtime_module": "a b"},
])
def test_executor_options_are_validated(bundle, kwargs):
    with pytest.raises(ValueError):
        ce.CoreMLEncoderExecutor(bundle, **kwargs)


@pytest.fixture
def control(tmp_path, monkeypatch):
    path = tmp_path / "fake-control"
    monkeypatch.setenv("MLX2_FAKE_COREML_CONTROL", str(path))
    return path


def test_process_backend_runs_buckets_in_a_spawned_worker_and_closes(bundle, control):
    executor = ce.CoreMLEncoderExecutor(bundle, runtime_module="coreml_fake_runtime", timeout=10, load_timeout=60)
    try:
        status = executor.status()
        assert status["isolation"] == "process" and status["worker_pid"] not in (None, os.getpid())
        assert status["coremltools_version"] == fake.__version__
        worker = executor._backend._process
        for length, count in ((32, 3), (64, 40), (256, 256)):
            ids, mask = padded([101, *[2000] * (count - 2), 102], length)
            np.testing.assert_allclose(executor.predict(ids, mask), fake.fake_vector(ids, mask)[0], rtol=0, atol=0)
        assert executor.status()["per_bucket_predictions"] == {"32": 1, "64": 1, "128": 0, "256": 1}
        plan = executor.compute_plan()
        assert plan["available"] is False and plan["per_bucket"] == {}
        retained = executor._paths[32]
    finally:
        executor.close()
    assert not worker.is_alive() and not retained.exists()


@pytest.mark.parametrize("mode, error", [
    ("sleep", TimeoutError), ("exit", RuntimeError), ("nan", RuntimeError), ("raise", RuntimeError),
])
def test_process_worker_failure_breaks_and_never_falls_back(bundle, control, mode, error):
    executor = ce.CoreMLEncoderExecutor(bundle, runtime_module="coreml_fake_runtime", timeout=1, load_timeout=60)
    try:
        worker = executor._backend._process
        assert len(executor.predict(*padded([101, 102], 32))) == 384
        control.write_text(mode)
        started = time.monotonic()
        with pytest.raises(error):
            executor.predict(*padded([101, 102], 32))
        assert time.monotonic() - started < 10
        assert executor.status()["broken"]
        # The executor itself reaped the child (no zombie), dead or killed.
        with pytest.raises(ChildProcessError):
            os.waitpid(worker.pid, os.WNOHANG)
        assert worker.exitcode is not None
        control.write_text("")
        with pytest.raises(RuntimeError, match="broken"):
            executor.predict(*padded([101, 102], 32))
    finally:
        executor.close()


def test_process_load_failure_cleans_up(bundle, control):
    control.write_text("fail_load")
    before = set(Path(ce.tempfile.gettempdir()).glob("mlx2-coreml-encoder-*"))
    with pytest.raises(RuntimeError, match="injected load failure"):
        ce.CoreMLEncoderExecutor(bundle, runtime_module="coreml_fake_runtime", load_timeout=60)
    assert set(Path(ce.tempfile.gettempdir()).glob("mlx2-coreml-encoder-*")) == before


def test_import_is_accelerator_runtime_free():
    code = (
        "import sys; import mlx2.runtime.coreml_encoder; "
        "bad = [m for m in sys.modules if m.split('.')[0] in {'coremltools', 'numpy', 'mlx', 'tokenizers'}]; "
        "assert not bad, bad"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
