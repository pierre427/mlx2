"""MiniLM Core ML embedding adapter, registry routing and serving receipts.

A fake Core ML runtime stands in for coremltools and a generated WordPiece
tokenizer stands in for the pinned one, so these tests never depend on the
developer's Hugging Face cache. Nothing executes Core ML, and the adapter
route itself must never import MLX.
"""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import coreml_fake_runtime as fake
import numpy as np
import pytest

from mlx2.adapters import minilm, minilm_coreml
from mlx2.adapters.minilm_coreml import MiniLMCoreMLAdapter
from mlx2.adapters.registry import inspect_model, resolve_adapter
from mlx2.contracts import Capability
from mlx2.runtime import coreml_encoder as ce

TESTS = Path(__file__).resolve().parent


@pytest.fixture
def tokenizer_bytes(monkeypatch):
    return fake.pin_tokenizer(monkeypatch, fake.tiny_tokenizer_bytes())


@pytest.fixture
def bundle(tmp_path, tokenizer_bytes):
    return fake.write_bundle(tmp_path / "bundle", tokenizer_bytes)


class Recorder:
    """In-process runtime double; the real executor validation stays in place."""

    def __init__(self, monkeypatch):
        self.calls = []
        self.entered, self.release = threading.Event(), None
        recorder = self

        class Model(fake.CompiledMLModel):
            def __init__(self, path, *, compute_units):
                # In process, conftest's CPU MLX is loaded; the worker-only
                # MLX check belongs to the process tests.
                assert compute_units == fake.ComputeUnit.CPU_AND_NE
                self.length = int(Path(path).name.removeprefix("seq").removesuffix(".mlmodelc"))

            def predict(self, features):
                recorder.calls.append((self.length, features["input_ids"].copy(), features["attention_mask"].copy()))
                recorder.entered.set()
                if recorder.release is not None:
                    assert recorder.release.wait(5)
                return super().predict(features)

        module = type(fake)("recording_coreml")
        module.__version__, module.ComputeUnit = "recording", fake.ComputeUnit
        module.models = type(fake.models)(CompiledMLModel=Model)
        monkeypatch.setattr(ce, "_coreml_runtime", lambda: module)
        monkeypatch.setattr(
            minilm_coreml, "_make_executor", lambda bundle: ce.CoreMLEncoderExecutor(bundle, backend="in_process")
        )


@pytest.fixture
def recorder(monkeypatch):
    return Recorder(monkeypatch)


@pytest.fixture
def adapter(bundle, recorder):
    result = MiniLMCoreMLAdapter(bundle)
    yield result
    result.close()


def reference_tokenizer(data):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_str(data.decode())
    tokenizer.enable_truncation(256, direction="right")
    tokenizer.no_padding()
    return tokenizer


def varied_words(count):
    return " ".join(fake.TINY_WORDS[index % 7] for index in range(count))


def test_registry_routes_a_bundle_directory_with_a_bundle_identity(bundle):
    resolved = inspect_model(bundle)
    assert resolved.adapter_type is MiniLMCoreMLAdapter
    assert resolve_adapter(bundle) is MiniLMCoreMLAdapter
    assert resolved.descriptor.capabilities == {Capability.EMBEDDINGS}
    assert resolved.descriptor.variant == "all-minilm-l6-v2-coreml-cpu-and-ne"
    assert resolved.descriptor.metadata["qualification"] == "unqualified"
    assert resolved.default_route == "embeddings"
    info = ce.inspect_bundle(bundle)
    # Identity is the served bundle, not the builder-attested MLX source.
    assert resolved.artifact_fingerprint == info.fingerprint != ce.source_fingerprint()
    identity = resolved.artifact["identity"]
    assert identity["compiled_payload_verified"] is False
    assert identity["source"] == {
        "model": "sentence-transformers/all-MiniLM-L6-v2", "revision": minilm.MODEL_REVISION,
        "source_fingerprint": ce.source_fingerprint(),
        "model_safetensors_sha256": minilm.ARTIFACT_FILES["model.safetensors"][1],
        "source_relationship": "builder_attested_unverified",
    }
    representation = resolved.descriptor.metadata["representation"]
    assert {key: representation[key] for key in minilm.representation_contract()} == minilm.representation_contract()
    assert representation["execution"] == {
        "backend": "coreml", "compute_units": "cpu_and_ne", "gpu": "excluded", "precision": "float16",
        "buckets": [32, 64, 128, 256], "neural_engine_observed": False,
    }


def test_registry_inspection_imports_no_tensor_tokenizer_or_coreml_runtime(bundle):
    code = """
import importlib.abc, json, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mlx', 'torch', 'transformers', 'tokenizers', 'coremltools', 'numpy'}:
            raise AssertionError('runtime import during inspection: ' + fullname)
sys.meta_path.insert(0, Block())
from mlx2.adapters import minilm
minilm.ARTIFACT_FILES['tokenizer.json'] = tuple(json.loads(sys.argv[2]))
from mlx2.adapters.registry import inspect_model
result = inspect_model(sys.argv[1])
assert result.adapter_type.__name__ == 'MiniLMCoreMLAdapter' and result.default_route == 'embeddings'
"""
    pin = json.dumps(minilm.ARTIFACT_FILES["tokenizer.json"])
    result = subprocess.run([sys.executable, "-c", code, str(bundle), pin], capture_output=True, text=True,
                            timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not (fake.TOKENIZER_SOURCE.parent / "model.safetensors").is_file(),
                    reason="optional: pinned MiniLM snapshot is absent")
def test_plain_minilm_directory_still_resolves_to_the_mlx_adapter():
    assert inspect_model(fake.TOKENIZER_SOURCE.parent).adapter_type is minilm.MiniLMAdapter


def test_invalid_manifest_fails_closed_and_never_falls_back_to_mlx_minilm(bundle):
    (bundle / "config.json").write_text(json.dumps({"model_type": "bert"}))
    manifest = json.loads((bundle / ce.MANIFEST_NAME).read_text())
    manifest["compute_units"] = "all"
    (bundle / ce.MANIFEST_NAME).write_text(json.dumps(manifest))
    for resolve in (inspect_model, resolve_adapter):
        with pytest.raises(ValueError, match="Core ML encoder manifest: compute_units"):
            resolve(bundle)
    (bundle / ce.MANIFEST_NAME).write_text("{")
    with pytest.raises(ValueError, match="Core ML encoder manifest"):
        inspect_model(bundle)


def test_embed_pads_to_smallest_bucket_and_truncates_at_256(adapter, recorder, tokenizer_bytes):
    texts = ["hello world", varied_words(40), varied_words(300)]
    reference = [reference_tokenizer(tokenizer_bytes).encode(text).ids for text in texts]
    vectors, tokens = adapter.embed(texts)
    assert [len(ids) for ids in reference] == [4, 42, 256]
    assert tokens == 4 + 42 + 256
    assert [call[0] for call in recorder.calls] == [32, 64, 256]
    for (length, ids, mask), expected, vector in zip(recorder.calls, reference, vectors):
        count = len(expected)
        assert ids.shape == mask.shape == (1, length) and ids.dtype == mask.dtype == np.int32
        assert ids[0, :count].tolist() == expected and not ids[0, count:].any()
        assert mask[0, :count].all() and not mask[0, count:].any()
        assert expected[0] == 101 and expected[-1] == 102
        np.testing.assert_allclose(vector, fake.fake_vector(ids, mask)[0], rtol=0, atol=0)
    assert adapter.embed(texts[:1])[0][0] == vectors[0]


@pytest.mark.parametrize("dimensions", [1, 128, 383, 385, True, 384.0])
def test_trained_dimensions_do_not_silently_truncate(adapter, recorder, dimensions):
    with pytest.raises(ValueError, match="exactly 384"):
        adapter.embed(["hello"], dimensions=dimensions)
    assert not recorder.calls


def test_input_bounds_are_enforced_before_execution(adapter, recorder):
    for texts, message in [
        ([], "1 to 2048"), ("hello", "1 to 2048"), ([""], "nonempty"), ([None], "nonempty"),
        (["x"] * 2049, "1 to 2048"), (["x" * ((1 << 20) + 1)], "1 MiB"), (["é" * (1 << 19) + "x"], "1 MiB"),
    ]:
        with pytest.raises(ValueError, match=message):
            adapter.embed(texts)
    assert not recorder.calls and adapter.representation_execution_count() == 0
    assert len(adapter.embed(["hello"], dimensions=384)[0][0]) == 384


def test_contract_identity_diagnostics_and_close(tmp_path, tokenizer_bytes, recorder):
    bundle = fake.write_bundle(tmp_path / "mixed", tokenizer_bytes, fractions={256: 0.0})
    with pytest.raises(ValueError, match="generation execution policies"):
        MiniLMCoreMLAdapter(bundle, execution_policy={"prefill_step_size": 1})
    adapter = MiniLMCoreMLAdapter(bundle)
    try:
        before = adapter.representation_contract()
        assert {key: before[key] for key in ("implemented", "qualified", "selected", "observed_used")} == {
            "implemented": True, "qualified": False, "selected": True, "observed_used": False,
        }
        execution = before["execution"]
        assert execution["neural_engine_observed"] is False and execution["gpu"] == "excluded"
        assert execution["anticipated_placement"] == {
            "32": "neural_engine_planned", "64": "neural_engine_planned", "128": "neural_engine_planned",
            "256": "cpu_planned",
        }
        assert execution["build_parity"]["passed"] is True
        assert execution["build_parity"]["attestation"] == "builder_attested_unverified"
        assert execution["source"]["source_relationship"] == "builder_attested_unverified"
        info = ce.inspect_bundle(bundle)
        assert adapter.identity["fingerprint"] == execution["bundle_fingerprint"] == info.fingerprint
        assert adapter.identity["fingerprint_kind"] == "coreml_encoder_bundle"
        assert adapter.identity["source"]["source_fingerprint"] == ce.source_fingerprint()
        assert adapter.identity["compiled_payload_verified"] is True
        assert adapter.identity["bundle_manifest_sha256"] == info.manifest_sha256
        assert adapter.descriptor.metadata["execution_kind"] == "embeddings"
        adapter.embed([varied_words(300)])
        assert adapter.representation_contract()["observed_used"] is True
        diagnostics = adapter.diagnostics()
        assert diagnostics["coreml"]["per_bucket_predictions"]["256"] == 1
        assert diagnostics["coreml"]["neural_engine_observed"] is False
        assert diagnostics["coreml"]["source"]["source_relationship"] == "builder_attested_unverified"
        assert diagnostics["coreml"]["bundle_fingerprint"] == info.fingerprint
        assert diagnostics["apcv2"] == diagnostics["kv_cache"] == "not_applicable"
        executor = adapter.executor
    finally:
        adapter.close()
    assert executor.status()["closed"] is True
    with pytest.raises(RuntimeError, match="closed"):
        adapter.embed(["hello"])
    adapter.close()


def test_close_never_splits_a_multi_text_request(adapter, recorder):
    recorder.release = threading.Event()
    results, errors = [], []

    def run():
        try:
            results.append(adapter.embed(["hello world", "goodbye sea", "the lamps of the ship"]))
        except BaseException as error:  # noqa: BLE001 - surfaced by the assertions
            errors.append(error)

    worker = threading.Thread(target=run)
    closer = threading.Thread(target=adapter.close)
    try:
        worker.start()
        assert recorder.entered.wait(5)
        closer.start()
        closer.join(0.1)
        # Close waits for the whole request, not merely the in-flight text.
        assert closer.is_alive() and adapter.executor is not None
    finally:
        recorder.release.set()
        worker.join(5)
        if closer.ident is not None:
            closer.join(5)
    assert not errors and len(results[0][0]) == 3 and len(recorder.calls) == 3
    assert adapter.executor is None
    count = len(recorder.calls)
    with pytest.raises(RuntimeError, match="closed"):
        adapter.embed(["hello", "world"])
    assert len(recorder.calls) == count


def test_serving_engine_reports_unqualified_coreml_route(bundle, recorder):
    from mlx2.api_resources import CapabilityUnavailable
    from mlx2.server import embeddings_payload
    from mlx2.serving import ServingEngine

    fingerprint = ce.inspect_bundle(bundle).fingerprint
    engine = ServingEngine(str(bundle), mtp=False)
    try:
        assert engine.ready.wait(15), engine.error
        assert engine.error is None
        before = engine.status()
        assert before["settings"]["route"] == "embeddings" and before["capabilities"] == ["embeddings"]
        assert before["qualification"] == "unqualified" and before["qualified_capabilities"] == []
        assert before["artifact"] == fingerprint
        for field in ("representation", "route_receipt"):
            assert {key: before[field][key] for key in ("implemented", "qualified", "selected", "observed_used")} == {
                "implemented": True, "qualified": False, "selected": True, "observed_used": False,
            }
        assert before["representation"]["execution"]["backend"] == "coreml"
        assert before["representation"]["execution"]["source"]["source_relationship"] == "builder_attested_unverified"
        assert before["execution"]["coreml"]["isolation"] == "in_process"
        result = embeddings_payload(engine, {"input": ["hello world", "goodbye"], "dimensions": 384})
        assert len(result["data"]) == 2 and result["usage"]["prompt_tokens"] == 7
        assert result["mlx2"]["observed_used"] is True
        assert result["mlx2"]["representation"]["execution"]["neural_engine_observed"] is False
        receipt = engine.operation_receipts[-1]
        assert receipt["observed_used"] is True and receipt["qualified"] is False and receipt["selected"] is True
        assert receipt["artifact"] == fingerprint
        with pytest.raises(ValueError, match="384"):
            engine.embed(["hello"], dimensions=32)
        assert engine.operation_receipts[-1]["observed_used"] is False
        with pytest.raises(CapabilityUnavailable, match="generation"):
            engine.submit({"prompt": "hello"})
    finally:
        engine.close()
    assert not engine.thread.is_alive()


def test_adapter_route_never_imports_mlx_and_runs_core_ml_in_a_worker(bundle):
    code = """
import importlib.abc, json, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mlx', 'coremltools'}:
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[2])
from mlx2.adapters import minilm
minilm.ARTIFACT_FILES['tokenizer.json'] = tuple(json.loads(sys.argv[3]))
from mlx2.adapters import minilm_coreml
from mlx2.adapters.registry import resolve_adapter
from mlx2.runtime.coreml_encoder import CoreMLEncoderExecutor
minilm_coreml._make_executor = lambda bundle: CoreMLEncoderExecutor(
    bundle, runtime_module='coreml_fake_runtime', load_timeout=60)
if __name__ == '__main__':
    adapter = resolve_adapter(sys.argv[1])(sys.argv[1])
    vectors, tokens = adapter.embed(['hello world', 'goodbye'])
    assert len(vectors) == 2 and all(len(v) == 384 for v in vectors) and tokens == 7
    status = adapter.diagnostics()['coreml']
    assert status['isolation'] == 'process' and status['predictions_completed'] == 2
    adapter.close()
    leaked = [name for name in sys.modules if name.split('.')[0] in {'mlx', 'coremltools'}]
    assert not leaked, leaked
    print('ok')
"""
    env = {key: value for key, value in os.environ.items() if key != "MLX2_FAKE_COREML_CONTROL"}
    pin = json.dumps(minilm.ARTIFACT_FILES["tokenizer.json"])
    result = subprocess.run([sys.executable, "-c", code, str(bundle), str(TESTS), pin], capture_output=True,
                            text=True, timeout=120, check=False, env=env)
    assert result.returncode == 0 and result.stdout.strip() == "ok", result.stdout + result.stderr


@pytest.mark.skipif(not fake.TOKENIZER_SOURCE.is_file(), reason="optional: pinned MiniLM tokenizer.json is absent")
def test_real_pinned_tokenizer_recipe(tmp_path, recorder):
    data = fake.TOKENIZER_SOURCE.read_bytes()
    assert fake.pin_entry(data) == minilm.ARTIFACT_FILES["tokenizer.json"]
    adapter = MiniLMCoreMLAdapter(fake.write_bundle(tmp_path / "real", data))
    try:
        texts = ["This is an example sentence", "CAFÉ naïve résumé!", "中文 mixed text"]
        _, tokens = adapter.embed(texts)
        expected = [reference_tokenizer(data).encode(text).ids for text in texts]
        assert [call[1][0, :len(ids)].tolist() for call, ids in zip(recorder.calls, expected)] == expected
        assert tokens == sum(len(ids) for ids in expected)
        assert adapter.embed(["hello world", "goodbye"])[1] == 7
    finally:
        adapter.close()
