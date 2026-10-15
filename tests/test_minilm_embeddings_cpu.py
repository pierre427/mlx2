"""BERT math, exact artifact gate and the real embedding-only HTTP serving path."""

import io
import json
import os
import struct
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.minilm import (
    ARTIFACT_FILES,
    DESCRIPTOR,
    MODEL_REVISION,
    MiniLMAdapter,
    expected_weight_schema,
    inspect_artifact,
    validate_weight_header,
)
from mlx2.adapters.registry import AdapterResolution, inspect_model
from mlx2.api_resources import CapabilityUnavailable
from mlx2.contracts import Capability
from mlx2.runtime.models.bert import (
    BertArgs,
    BertModel,
    encoder_weights,
    normalized_masked_mean,
)
from mlx2.server import handler_for, resolve_route_selection
from mlx2.serving import ServingEngine


@pytest.fixture(scope="module")
def artifact_path():
    path = Path(os.environ.get(
        "MLX2_MINILM_REFERENCE_PATH",
        str(Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
            / "snapshots" / MODEL_REVISION),
    ))
    if not (path / "model.safetensors").is_file():
        pytest.skip("pinned MiniLM local artifact is absent; never download in tests")
    return path


@pytest.fixture(scope="module")
def adapter(artifact_path):
    assert mx.default_device() == mx.cpu
    result = MiniLMAdapter(artifact_path)
    yield result
    result.close()


@pytest.fixture
def torch_cpu():
    torch = pytest.importorskip("torch")
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    try:
        yield torch
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)


def _header(schema):
    offset = 0
    result = {}
    for name, (dtype, shape) in schema.items():
        size = int(np.prod(shape)) * (8 if dtype == "I64" else 4)
        result[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    return result, offset


@pytest.mark.parametrize("mutation", ["missing", "extra", "shape", "dtype", "overlap", "hole", "trailing"])
def test_weight_schema_refuses_every_silent_loading_mismatch(mutation):
    header, payload_bytes = _header(expected_weight_schema())
    if mutation == "missing":
        header.pop("pooler.dense.bias")
    elif mutation == "extra":
        header["unused.weight"] = dict(header["pooler.dense.bias"])
    elif mutation == "shape":
        header["pooler.dense.bias"]["shape"] = [383]
    elif mutation == "dtype":
        header["pooler.dense.bias"]["dtype"] = "F16"
    elif mutation in {"overlap", "hole"}:
        header["pooler.dense.bias"]["data_offsets"] = [0, 384 * 4]
        if mutation == "hole":
            header["pooler.dense.bias"]["data_offsets"] = [payload_bytes, payload_bytes + 384 * 4]
    else:
        payload_bytes += 4
    raw = json.dumps(header).encode()
    with pytest.raises(ValueError, match="MiniLM"):
        validate_weight_header(io.BytesIO(struct.pack("<Q", len(raw)) + raw), len(raw) + 8 + payload_bytes)


def test_weight_schema_accepts_exact_header_and_rejects_duplicate_keys():
    header, size = _header(expected_weight_schema())
    raw = json.dumps(header).encode()
    validate_weight_header(io.BytesIO(struct.pack("<Q", len(raw)) + raw), size + len(raw) + 8)
    raw = b'{"same": {}, "same": {}}'
    with pytest.raises(ValueError, match="duplicate"):
        validate_weight_header(io.BytesIO(struct.pack("<Q", len(raw)) + raw), len(raw) + 8)


def test_masked_pooling_ignores_padding_includes_special_rows_and_normalizes():
    hidden = mx.array([[[1, 0], [0, 2], [3, 0], [100, 100]]], dtype=mx.float32)
    actual = normalized_masked_mean(hidden, mx.array([[1, 1, 1, 0]]))
    np.testing.assert_allclose(np.array(actual), [[2 / 5**0.5, 1 / 5**0.5]], atol=1e-7)
    for values, mask in [
        (mx.zeros((1, 2, 3)), mx.array([[1, 1]])),
        (mx.ones((1, 2, 3)), mx.array([[0, 0]])),
        (mx.full((1, 2, 3), float("nan")), mx.array([[1, 1]])),
        (mx.full((1, 2, 3), float("inf")), mx.array([[1, 1]])),
    ]:
        with pytest.raises(ValueError, match="nonfinite, zero"):
            normalized_masked_mean(values, mask)


def test_small_bert_matches_independent_transformers_hidden_states_and_padding(torch_cpu):
    from transformers import BertConfig
    from transformers import BertModel as Reference

    torch = torch_cpu
    torch.manual_seed(71)
    args = BertArgs(vocab_size=19, hidden_size=12, intermediate_size=29,
                    num_hidden_layers=2, num_attention_heads=3,
                    max_position_embeddings=16)
    config = BertConfig(**vars(args), hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    config._attn_implementation = "eager"
    reference = Reference(config, add_pooling_layer=False).cpu().eval()
    model = BertModel(args)
    model.load_weights(encoder_weights({
        name: mx.array(value.detach().cpu().numpy())
        for name, value in reference.state_dict().items()
    }), strict=True)
    ids = [[1, 2, 3, 4], [5, 6, 0, 0]]
    mask = [[1, 1, 1, 1], [1, 1, 0, 0]]
    types = [[0, 0, 1, 1], [0, 1, 0, 0]]
    with torch.inference_mode():
        expected = reference(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask),
                             token_type_ids=torch.tensor(types)).last_hidden_state.numpy()
    actual = model(mx.array(ids), mx.array(mask), mx.array(types))
    np.testing.assert_allclose(np.array(actual), expected, atol=1e-6, rtol=1e-5)
    short = model(mx.array([[5, 6]]), mx.array([[1, 1]]), mx.array([[0, 1]]))
    np.testing.assert_allclose(np.array(actual)[1, :2], np.array(short)[0], atol=1e-6, rtol=1e-5)
    # A future-token change affects the first token: this is bidirectional,
    # not an accidental reuse of the decoder's causal attention helper.
    changed = model(mx.array([[1, 2, 3, 9]]), mx.ones((1, 4), dtype=mx.bool_))
    original = model(mx.array([[1, 2, 3, 4]]), mx.ones((1, 4), dtype=mx.bool_))
    assert np.max(np.abs(np.array(changed[0, 0] - original[0, 0]))) > 1e-5


def test_resolver_declares_only_trained_embeddings_and_rejects_generation_flags(artifact_path):
    resolved = inspect_model(artifact_path)
    assert resolved.adapter_type is MiniLMAdapter
    assert resolved.descriptor.capabilities == {Capability.EMBEDDINGS}
    assert resolved.default_route == "embeddings"
    assert resolved.artifact["weight_payload_verified"] is False
    assert not resolved.descriptor.state_planes
    args = SimpleNamespace(ordinary=False, external_draft=False, native_mtp=False, prompt_lookup=False)
    assert resolve_route_selection(args, None, resolved).route == "embeddings"
    for flag in vars(args):
        with pytest.raises(ValueError, match="embedding-only"):
            resolve_route_selection(SimpleNamespace(**{**vars(args), flag: True}), None, resolved)
    with pytest.raises(ValueError, match="embedding-only"):
        resolve_route_selection(args, {"draft_model": "bad"}, resolved)


def test_artifact_resolution_does_not_import_tensor_or_reference_runtimes(artifact_path):
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mlx', 'torch', 'transformers', 'tokenizers'}:
            raise AssertionError('tensor/tokenizer import during inspection: ' + fullname)
sys.meta_path.insert(0, Block())
from mlx2.adapters.registry import inspect_model
result = inspect_model(sys.argv[1])
assert result.default_route == 'embeddings'
print(result.artifact_fingerprint)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(artifact_path)], capture_output=True,
        text=True, timeout=10, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(result.stdout.strip()) == 64


def test_generic_bert_cannot_be_relabelled_as_trained_minilm(tmp_path, artifact_path):
    for name in ARTIFACT_FILES:
        target = tmp_path / name
        target.parent.mkdir(exist_ok=True)
        target.symlink_to(artifact_path / name)
    (tmp_path / "config.json").unlink()
    config = json.loads((artifact_path / "config.json").read_text())
    config["hidden_act"] = "gelu_new"
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="pinned artifact content mismatch"):
        inspect_artifact(tmp_path)


def test_real_checkpoint_wordpiece_recipe_matches_pinned_transformers(adapter, artifact_path, torch_cpu, record_property):
    from transformers import AutoModel, AutoTokenizer

    torch = torch_cpu
    texts = [
        "This is an example sentence", "Each sentence is converted", "CAFÉ naïve résumé!",
        "中文 mixed text", "hello " * 300, "   ", "a",
    ]
    reference_tokenizer = AutoTokenizer.from_pretrained(artifact_path, local_files_only=True)
    encoded = reference_tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    for text, ids, mask in zip(texts, encoded["input_ids"], encoded["attention_mask"]):
        actual = adapter.tokenizer.encode(text)
        assert actual.ids == ids[mask.bool()].tolist()
        assert len(actual.ids) <= 256
        assert actual.ids[0] == 101 and actual.ids[-1] == 102
    reference = AutoModel.from_pretrained(
        artifact_path, local_files_only=True, attn_implementation="eager", dtype=torch.float32
    ).cpu().eval()
    with torch.inference_mode():
        hidden = reference(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        expected = torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9), p=2, dim=1).numpy()
    vectors, tokens = adapter.embed(texts)
    actual = np.asarray(vectors)
    record_property("reference_max_abs_error", float(np.max(np.abs(actual - expected))))
    record_property("reference_min_cosine", float(np.min(np.sum(actual * expected, axis=1))))
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-4)
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1, atol=2e-7)
    assert tokens == int(encoded["attention_mask"].sum())
    assert adapter.identity["weight_payload_verified"] is True
    np.testing.assert_allclose(adapter.embed([texts[0]])[0][0], actual[0], atol=1e-7)


@pytest.mark.parametrize("dimensions", [1, 128, 383, 385, True, 384.0])
def test_trained_dimensions_do_not_silently_truncate(adapter, dimensions):
    with pytest.raises(ValueError, match="exactly 384"):
        adapter.embed(["hello"], dimensions=dimensions)


def test_adapter_input_bounds_are_enforced_before_tensor_execution(adapter):
    for texts in [[], [""], [None], ["x"] * 2049, ["x" * ((1 << 20) + 1)]]:
        with pytest.raises(ValueError):
            adapter.embed(texts)


def test_real_http_embedding_route_is_trained_unqualified_and_generation_fails_closed(artifact_path):
    from test_tenant_auth import _call

    engine = ServingEngine(str(artifact_path), mtp=False)
    server = thread = None
    try:
        assert engine.ready.wait(15), engine.error
        assert engine.error is None
        before = engine.status()
        assert before["settings"]["route"] == "embeddings"
        assert before["capabilities"] == ["embeddings"]
        assert before["qualification"] == "unqualified"
        assert before["qualified_capabilities"] == []
        for field in ("representation", "route_receipt"):
            assert {key: before[field][key] for key in (
                "implemented", "qualified", "selected", "observed_used"
            )} == {"implemented": True, "qualified": False, "selected": True, "observed_used": False}
        assert before["representation"]["kind"] == "task_trained_sentence_embedding"
        assert engine.apc is None
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        code, _, raw = _call(base, "POST", "/v1/embeddings", {"input": ["hello world", "goodbye"], "dimensions": 384})
        assert code == 200, raw
        result = json.loads(raw)
        assert result["mlx2"]["observed_used"] is True
        assert result["mlx2"]["representation"]["kind"] == "task_trained_sentence_embedding"
        assert len(result["data"]) == 2
        assert all(len(row["embedding"]) == 384 for row in result["data"])
        assert result["usage"]["prompt_tokens"] == 7
        for field in ("representation", "route_receipt"):
            assert engine.status()[field]["observed_used"] is True
        assert engine.operation_receipts[-1]["representation"]["revision"] == MODEL_REVISION
        assert engine.operation_receipts[-1]["observed_used"] is True
        assert engine.operation_receipts[-1]["selected"] is True
        assert engine.operation_receipts[-1]["qualified"] is False
        for endpoint, body in [
            ("/v1/completions", {"prompt": "hello", "max_tokens": 1}),
            ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hello"}]}),
            ("/v1/rerank", {"query": "hello", "documents": ["hello"]}),
            ("/v1/responses", {"input": "hello"}),
            ("/tokenize", {"prompt": "hello"}),
        ]:
            code, _, raw = _call(base, "POST", endpoint, body)
            assert code == 501, (endpoint, code, raw)
        with pytest.raises(CapabilityUnavailable, match="generation"):
            engine.submit({"prompt": "hello"})
        assert not engine.jobs and engine.apc is None
        engine.quiesce(drain_timeout_seconds=1, suspend=False)
        assert engine.wait_for_quiesce(2)
        assert engine.service_state()["state"] == "quiesced"
        engine.resume()
        assert len(engine.embed(["hello"])[0][0]) == 384
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(2)
            assert not thread.is_alive()
        engine.close()
        assert not engine.thread.is_alive()


def test_payload_hash_is_verified_before_tensor_loading(tmp_path, artifact_path, monkeypatch):
    for name in ARTIFACT_FILES:
        target = tmp_path / name
        target.parent.mkdir(exist_ok=True)
        if name == "model.safetensors":
            import shutil

            shutil.copyfile(artifact_path / name, target)
            with target.open("r+b") as stream:
                stream.seek(-1, 2)
                value = stream.read(1)[0]
                stream.seek(-1, 2)
                stream.write(bytes([value ^ 1]))
        else:
            target.symlink_to(artifact_path / name)
    # Schema/size still pass: only exact-byte verification catches this drift.
    inspect_artifact(tmp_path)
    monkeypatch.setattr(mx, "load", lambda *_args, **_kwargs: pytest.fail("unverified tensor load"))
    with pytest.raises(ValueError, match="pinned artifact content mismatch: model.safetensors"):
        MiniLMAdapter(tmp_path)


def test_shutdown_waits_for_exclusive_embedding_owner(artifact_path, monkeypatch):
    engine = ServingEngine(str(artifact_path), mtp=False)
    entered, release = threading.Event(), threading.Event()
    errors = []
    worker = closer = None
    try:
        assert engine.ready.wait(15), engine.error
        adapter = engine.adapter
        original = adapter.embed

        def blocked(*args, **kwargs):
            entered.set()
            assert release.wait(5)
            return original(*args, **kwargs)

        monkeypatch.setattr(adapter, "embed", blocked)

        def execute():
            try:
                engine.embed(["hello"])
            except BaseException as error:  # noqa: BLE001 - preserve worker failures for the assertion
                errors.append(error)

        worker = threading.Thread(target=execute)
        worker.start()
        assert entered.wait(2)
        closer = threading.Thread(target=engine.close)
        closer.start()
        assert engine.stop_event.wait(2)
        time.sleep(0.05)
        assert engine.thread.is_alive() and closer.is_alive()
        assert adapter.model is not None
        release.set()
        worker.join(5)
        closer.join(5)
        assert not worker.is_alive() and not closer.is_alive()
        assert not errors
        assert adapter.model is None
        with pytest.raises(RuntimeError, match="not ready"):
            engine.embed(["too late"])
    finally:
        release.set()
        if worker is not None:
            worker.join(5)
        if closer is not None:
            closer.join(5)
        engine.close()


@pytest.mark.parametrize("options", [
    {"prompt_lookup": True}, {"lane_matmul": "exact"}, {"cache_dir": "unused-cache"},
    {"max_context": 128},
])
def test_embedding_worker_refuses_unsupported_options(artifact_path, options):
    engine = ServingEngine(str(artifact_path), mtp=False, **options)
    try:
        engine.thread.join(10)
        assert not engine.thread.is_alive()
        assert not engine.ready.is_set()
        assert "embedding" in engine.error
        assert engine.adapter is None and engine.apc is None
    finally:
        engine.close()


def test_embedding_default_cannot_be_declared_for_generation_descriptor():
    from dataclasses import replace

    descriptor = replace(DESCRIPTOR, capabilities=frozenset({Capability.TEXT, Capability.EMBEDDINGS}))
    with pytest.raises(ValueError, match="embedding-only"):
        _ = AdapterResolution(MiniLMAdapter, descriptor, {}).default_route


def test_embedding_receipts_distinguish_refusal_inflight_and_failed_forward(artifact_path, monkeypatch):
    engine = ServingEngine(str(artifact_path), mtp=False)
    entered, release = threading.Event(), threading.Event()
    errors, worker = [], None
    try:
        assert engine.ready.wait(15), engine.error
        assert engine.error is None
        with pytest.raises(ValueError, match="384"):
            engine.embed(["hello"], dimensions=32)
        assert engine.operation_receipts[-1]["observed_used"] is False
        assert engine.status()["route_receipt"]["observed_used"] is False

        def fail_forward(*args):
            entered.set()
            assert release.wait(5)
            raise RuntimeError("injected encoder failure")

        monkeypatch.setattr(engine.adapter, "model", fail_forward)

        def execute():
            try:
                engine.embed(["hello"])
            except RuntimeError as error:
                errors.append(str(error))

        worker = threading.Thread(target=execute)
        worker.start()
        assert entered.wait(2)
        during = engine.status()
        assert during["route_receipt"]["observed_used"] is True
        assert during["representation"]["observed_used"] is True
        release.set()
        worker.join(5)
        assert not worker.is_alive()
        assert errors == ["injected encoder failure"]
        receipt = engine.operation_receipts[-1]
        assert receipt["status"] == "failed" and receipt["observed_used"] is True
        assert receipt["representation"]["observed_used"] is True
        assert engine.status()["counts"].get("embeddings_completed", 0) == 0
        with pytest.raises(ValueError, match="384"):
            engine.embed(["hello"], dimensions=32)
        assert engine.operation_receipts[-1]["observed_used"] is False
        assert engine.status()["route_receipt"]["observed_used"] is True
    finally:
        release.set()
        if worker is not None:
            worker.join(5)
        engine.close()
        assert not engine.thread.is_alive()
    assert engine.status()["route_receipt"]["observed_used"] is True
    assert engine.status()["representation"]["observed_used"] is True
