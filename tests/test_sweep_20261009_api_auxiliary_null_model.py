"""``/v1/embeddings`` and ``/v1/rerank`` read ``"model": null`` as the loaded model.

Review round 2 of the null-model fix (da66ee917): both auxiliary payload
builders compared ``body.get("model", model) != model``, so an explicit JSON
null (vLLM-style clients serialise an unset ``Optional[str]`` model as null)
was a 404 "unknown model" while omitting the field selected the loaded model.
A recognised null is an absent field everywhere else on the API.  A wrong
name or a wrong type is still refused; these endpoints serve the base model
only (no LoRA selection).
"""

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from test_serving_contract import FakeEngine

from mlx2.api_resources import ResourceNotFound
from mlx2.server import embeddings_payload, handler_for, rerank_payload


class AuxiliaryEngine(FakeEngine):
    def embed(self, inputs, *, dimensions=None):
        return [[1.0, 2.0] for _ in inputs], 3

    def rerank(self, query, documents):
        return [0.5] * len(documents)


EMBED = {"input": "hello"}
RERANK = {"query": "q", "documents": ["a", "b"]}


@pytest.fixture
def endpoint():
    engine = AuxiliaryEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def _post(base, path, body):
    request = Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        return exc.code, json.load(exc)


@pytest.mark.parametrize(
    "path, body", [("/v1/embeddings", EMBED), ("/v1/rerank", RERANK)]
)
def test_null_model_means_the_loaded_model(endpoint, path, body):
    assert _post(endpoint, path, body)[0] == 200
    status, payload = _post(endpoint, path, {**body, "model": None})
    assert status == 200, payload
    assert payload["model"] == "fixture"
    assert _post(endpoint, path, {**body, "model": "fixture"})[0] == 200
    for wrong in ("other", 5, True, ["fixture"]):
        status, payload = _post(endpoint, path, {**body, "model": wrong})
        assert status == 404, (wrong, payload)


def test_payload_builders_resolve_a_null_model():
    engine = AuxiliaryEngine()
    assert embeddings_payload(engine, {**EMBED, "model": None})["model"] == "fixture"
    assert rerank_payload(engine, {**RERANK, "model": None})["model"] == "fixture"
    for builder, body in ((embeddings_payload, EMBED), (rerank_payload, RERANK)):
        with pytest.raises(ResourceNotFound):
            builder(engine, {**body, "model": "other"})
