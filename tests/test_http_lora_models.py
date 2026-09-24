"""Advertised LoRA model ids must reach generation without loading tensors."""

import json
from http.client import HTTPConnection

import pytest
from test_http_request_lifecycle import _server
from test_server_quiesce import AdminEngine

from mlx2.server import handler_for
from mlx2.serving import Job


class LoRAEngine(AdminEngine):
    def __init__(self):
        super().__init__()
        self.requests = []

    def status(self):
        return {**super().status(), "multi_lora": {"enabled": True, "registered": ["sql"]}}

    def submit(self, request, *, tenant_id="default"):
        self.requests.append(request)
        job = Job(request, tenant_id=tenant_id)
        job.prompt_tokens, job.completion_tokens = 3, 1
        job.events.put({"text": "hello"})
        job.events.put({"finish_reason": "stop", "receipt": {"cache": "apcv2"}})
        return job

    def render_prompt(self, request):
        self.requests.append(request)
        return [1, 2, 3]

    def count_tokens(self, request):
        return len(self.render_prompt(request))

    def apply_template(self, request):
        self.requests.append(request)
        return "rendered"


ROUTES = [
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
    ("/v1/completions", {"prompt": "hi"}),
    ("/v1/responses", {"input": "hi", "store": False}),
    ("/v1/messages", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}),
    ("/v1/messages/count_tokens", {"messages": [{"role": "user", "content": "hi"}]}),
    ("/tokenize", {"prompt": "hi"}),
    ("/apply-template", {"messages": [{"role": "user", "content": "hi"}]}),
]


@pytest.mark.parametrize("path,body", ROUTES)
@pytest.mark.parametrize("model,expected", [("sql", 200), ("missing", 404)])
def test_registered_generation_models_and_unknown_names(path, body, model, expected):
    engine = LoRAEngine()
    with _server(engine) as (_, port):
        connection = HTTPConnection("127.0.0.1", port, timeout=2)
        try:
            connection.request("POST", path, body=json.dumps({**body, "model": model}))
            response = connection.getresponse()
            payload = json.loads(response.read())
            assert response.status == expected, payload
        finally:
            connection.close()
    if expected == 200:
        assert engine.requests[-1]["model"] == model
        if "model" in payload:
            assert payload["model"] == model
    else:
        assert not engine.requests


@pytest.mark.parametrize("path,body", ROUTES[:3])
def test_batch_generation_accepts_registered_lora_models(path, body):
    engine = LoRAEngine()
    handler_for(engine)
    status, payload = engine.api_resources["batches"].executor(
        path, {**body, "model": "sql"}, "tenant"
    )
    assert status == 200
    assert payload["model"] == "sql"
    assert engine.requests[-1]["model"] == "sql"
