"""Pinned relevance classifier, pair recipe and real HTTP rerank contract."""

import io
import json
import os
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.minilm import validate_weight_header
from mlx2.adapters.minilm_reranker import (
    ARTIFACT_FILES,
    MODEL_REVISION,
    MiniLMRerankerAdapter,
    inspect_artifact,
    weight_schema,
)
from mlx2.adapters.registry import inspect_model
from mlx2.contracts import Capability
from mlx2.server import handler_for, resolve_route_selection
from mlx2.serving import ServingEngine


@pytest.fixture(scope="module")
def artifact_path():
    path = Path(os.environ.get("MLX2_RERANKER_REFERENCE_PATH", str(
        Path.home() / ".cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L6-v2"
        / "snapshots" / MODEL_REVISION
    )))
    if not (path / "model.safetensors").is_file():
        pytest.skip("pinned reranker artifact absent; never download in tests")
    return path


@pytest.fixture(scope="module")
def adapter(artifact_path):
    assert mx.default_device() == mx.cpu
    result = MiniLMRerankerAdapter(artifact_path)
    yield result
    result.close()


def test_reranker_resolver_and_cli_only_select_rerank(artifact_path):
    resolution = inspect_model(artifact_path)
    assert resolution.default_route == "rerank"
    assert resolution.descriptor.capabilities == {Capability.RERANK}
    assert resolution.artifact["weight_payload_verified"] is False
    args = SimpleNamespace(ordinary=False, native_mtp=False, external_draft=False, prompt_lookup=False)
    assert resolve_route_selection(args, {}, resolution).route == "rerank"
    args.ordinary = True
    with pytest.raises(ValueError, match="generation"):
        resolve_route_selection(args, {}, resolution)


def test_wrong_trained_recipe_refused_before_tensor_load(tmp_path, artifact_path):
    for name in ARTIFACT_FILES:
        target = tmp_path / name
        if name == "config.json":
            value = json.loads((artifact_path / name).read_text())
            value["sbert_ce_default_activation_function"] = "torch.nn.Sigmoid"
            target.write_text(json.dumps(value))
        else:
            target.symlink_to(artifact_path / name)
    with pytest.raises(ValueError, match="pinned artifact"):
        inspect_artifact(tmp_path)


@pytest.mark.parametrize("mutation", ["missing_classifier", "head_shape", "head_dtype", "extra"])
def test_reranker_header_requires_actual_trained_head(mutation):
    import struct

    from test_minilm_embeddings_cpu import _header

    header, payload_bytes = _header(weight_schema())
    if mutation == "missing_classifier":
        header.pop("classifier.weight")
    elif mutation == "head_shape":
        header["classifier.weight"]["shape"] = [384, 1]
    elif mutation == "head_dtype":
        header["classifier.weight"]["dtype"] = "F16"
    else:
        header["embedding_only.weight"] = header["classifier.weight"]
    data = json.dumps(header).encode()
    with pytest.raises(ValueError):
        validate_weight_header(io.BytesIO(struct.pack("<Q", len(data)) + data),
                               8 + len(data) + payload_bytes, schema=weight_schema())


def test_real_pair_logits_match_pinned_transformers(adapter, artifact_path, record_property):
    torch = pytest.importorskip("torch")
    from transformers import AutoTokenizer, BertForSequenceClassification

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        tokenizer = AutoTokenizer.from_pretrained(artifact_path, local_files_only=True, trust_remote_code=False)
        reference = BertForSequenceClassification.from_pretrained(
            artifact_path, local_files_only=True, attn_implementation="eager"
        ).cpu().eval()
        queries = ["Which planet is the Red Planet?", "café 中文", "hello " * 600]
        documents = [
            ["Mars is called the Red Planet.", "Saturn has rings.", "A red apple is a fruit."],
            ["Un café au lait.", "中文科学", "   "],
            ["world " * 600, "short"],
        ]
        errors = []
        for query, docs in zip(queries, documents):
            scores, tokens = adapter.rerank(query, docs)
            expected_tokens = 0
            for document, actual in zip(docs, scores):
                encoded = tokenizer(query, document, truncation=True, max_length=512, return_tensors="pt")
                pair = adapter.tokenizer.encode(query, document)
                assert pair.ids == encoded["input_ids"][0].tolist()
                assert pair.type_ids == encoded["token_type_ids"][0].tolist()
                expected_tokens += int(encoded["attention_mask"].sum())
                with torch.inference_mode():
                    expected = float(reference(**encoded).logits[0, 0])
                errors.append(abs(actual - expected))
            assert tokens == expected_tokens
        record_property("maximum_absolute_relevance_logit_error", max(errors))
        assert max(errors) < 3e-5
        scores, _ = adapter.rerank(queries[0], documents[0])
        assert scores[0] > scores[1] and scores[0] > scores[2]
        assert min(scores) < 0 and max(scores) > 1  # Raw trained logits, no hidden sigmoid.
        np.testing.assert_array_equal(scores, adapter.rerank(queries[0], documents[0])[0])
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("query,documents", [
    ("", ["x"]), (None, ["x"]), ("x", []), ("x", [""]),
    ("x", [None]), ("x", ["a"] * 1001), ("x" * ((1 << 20) + 1), ["x"]),
])
def test_reranker_bounds_precede_execution(adapter, query, documents):
    before = adapter.representation_execution_count()
    with pytest.raises(ValueError):
        adapter.rerank(query, documents)
    assert adapter.representation_execution_count() == before


def test_real_http_reranking_and_unsupported_routes(artifact_path):
    from test_tenant_auth import _call

    engine = ServingEngine(str(artifact_path), mtp=False)
    server = thread = None
    try:
        assert engine.ready.wait(15), engine.error
        assert engine.error is None
        before = engine.status()
        assert before["route_receipt"]["route"] == "rerank"
        assert before["route_receipt"]["observed_used"] is False
        assert before["qualification"] == "unqualified"
        assert before["capabilities"] == ["rerank"] and engine.apc is None
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        body = {"query": "Which planet is red?", "documents": ["Saturn has rings.", "Mars is the red planet."],
                "top_n": 1, "return_documents": True}
        code, _, raw = _call(base, "POST", "/v1/rerank", body)
        assert code == 200, raw
        result = json.loads(raw)
        assert result["mlx2"]["representation"]["score_semantics"] == "raw_relevance_logit_higher_is_better"
        assert result["mlx2"]["observed_used"] is True
        assert result["results"][0]["index"] == 1
        assert result["results"][0]["document"]["text"] == body["documents"][1]
        assert result["usage"]["prompt_tokens"] > 0
        assert engine.status()["route_receipt"]["observed_used"] is True
        receipt = engine.operation_receipts[-1]
        assert receipt["observed_used"] is True and receipt["qualified"] is False
        assert receipt["representation"]["score_activation"] == "identity"
        for endpoint, payload in [
            ("/v1/embeddings", {"input": "hello"}),
            ("/v1/completions", {"prompt": "hello", "max_tokens": 1}),
            ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hello"}]}),
            ("/v1/responses", {"input": "hello"}),
            ("/tokenize", {"prompt": "hello"}),
        ]:
            code, _, raw = _call(base, "POST", endpoint, payload)
            assert code == 501, (endpoint, raw)
        engine.quiesce(drain_timeout_seconds=1, suspend=False)
        assert engine.wait_for_quiesce(2)
        engine.resume()
        assert len(engine.rerank("planet", ["Mars"])) == 1
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(2)
        engine.close()
        assert not engine.thread.is_alive()
