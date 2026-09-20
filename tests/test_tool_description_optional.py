"""A tool definition without ``description`` is legal and must render.

``description`` is optional in both tool schemas we accept: the Responses and
Chat validators only require ``name`` (``parameters`` is already defaulted to
``{}``), and ``anthropic_compat._chat_tool`` copies ``description`` only when
the caller sent one.  A chat template that renders the field through
``tojson`` used to meet the Jinja ``Undefined`` a missing key leaves behind and
raise ``TypeError: Object of type Undefined is not JSON serializable``, which
reached the caller as an unhandled 500.
"""

import json
import threading
from collections import Counter
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from mlx2.anthropic_compat import anthropic_request_to_chat
from mlx2.server import handler_for, validate_request
from mlx2.serving import (
    Job,
    PromptTemplateError,
    PromptTemplateFailure,
    render_prompt_tokens,
)

from test_agent_conformance import (  # noqa: F401 - shared fixtures/helpers
    _claude,
    _codex,
    _post,
    serve,
)

# The shape every Qwen-family template uses for tools: the whole definition is
# serialized, so a missing optional key is an Undefined inside ``tojson``.
TEMPLATE = (
    "{% for tool in tools %}{{ tool.function.name }}: "
    "{{ tool.function.description | tojson }}\n{% endfor %}"
    "{% for message in messages %}{{ message.role }}\n{% endfor %}"
)


def render(request):
    """Render one validated request through a real transformers Jinja env."""
    from transformers.utils.chat_template_utils import _compile_jinja_template

    template = _compile_jinja_template(TEMPLATE)
    return template.render(
        messages=request["messages"], tools=request.get("tools")
    )


def test_native_chat_tool_without_description_renders():
    request = validate_request(
        {
            "model": "fixture",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "weather", "parameters": {"type": "object"}},
                }
            ],
        }
    )
    # Renders instead of raising ``TypeError: Object of type Undefined is
    # not JSON serializable`` out of the template's ``tojson``.
    assert render(request).startswith('weather: ""')
    assert request["tools"][0]["function"]["description"] == ""


def test_supplied_description_is_preserved():
    request = validate_request(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "weather", "description": "forecast"},
                }
            ],
        }
    )
    assert request["tools"][0]["function"]["description"] == "forecast"
    with pytest.raises(ValueError, match="tool description must be text"):
        validate_request(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "weather", "description": 7},
                    }
                ],
            }
        )


def test_anthropic_translation_defaults_description_for_count_tokens():
    """``/v1/messages/count_tokens`` renders without ``validate_request``."""
    chat = anthropic_request_to_chat(
        {
            "model": "fixture",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "weather", "input_schema": {"type": "object"}}],
        },
        count_tokens=True,
        model="fixture",
    )
    render(chat)
    assert chat["tools"][0]["function"]["description"] == ""


def test_translated_claude_code_payload_without_description_renders(serve):
    """A real captured Claude Code turn with one tool's description removed."""
    engine, base = serve()
    body = _claude(0, engine)
    assert "description" in body["tools"][0]
    body["tools"][0].pop("description")
    status, _ = _post(base, body, path="/v1/messages")
    assert status == 200
    request = engine.requests[-1]
    render(request)
    assert request["tools"][0]["function"]["description"] == ""


def test_translated_codex_payload_without_description_renders(serve):
    """A real captured Codex turn with one function tool's description gone."""
    engine, base = serve()
    body = _codex(0, engine)
    function_tool = next(
        tool for tool in body["tools"] if tool.get("type") == "function"
    )
    assert "description" in function_tool
    function_tool.pop("description")
    status, _ = _post(base, body, path="/v1/responses")
    assert status == 200
    request = engine.requests[-1]
    render(request)
    rendered = {
        tool["function"]["name"]: tool["function"].get("description")
        for tool in request["tools"]
    }
    assert rendered[function_tool["name"]] == ""


class Boom:
    def __init__(self, error):
        self.error = error

    def prompt_tokens(self, request):
        raise self.error


def test_render_boundary_shapes_template_failures():
    import jinja2

    request = {
        "messages": [],
        "tools": [{"type": "function", "function": {"name": "weather"}}],
    }
    tojson = Boom(TypeError("Object of type Undefined is not JSON serializable"))
    with pytest.raises(PromptTemplateError) as failure:
        render_prompt_tokens(tojson, request)
    assert "chat template" in str(failure.value)

    undefined = Boom(jinja2.exceptions.UndefinedError("'dict object' has no attribute 'description'"))
    with pytest.raises(PromptTemplateError) as named:
        render_prompt_tokens(undefined, request)
    assert "tools[0].function.description" in str(named.value)

    # A failure the request does not explain stays a server fault, but it is
    # reported as a rendering failure instead of an unhandled TypeError.
    with pytest.raises(PromptTemplateFailure, match="chat template rendering failed"):
        render_prompt_tokens(Boom(TypeError("model is broken")), request)
    # Existing 4xx shapes pass through untouched.
    with pytest.raises(ValueError, match="tool call arguments"):
        render_prompt_tokens(Boom(ValueError("tool call arguments")), request)


class TemplateFailureEngine:
    """Fails every job the way a broken chat template does."""

    model_path = "fixture"
    max_context = 16384

    def __init__(self, *, synchronous):
        self.synchronous = synchronous
        self.counts = Counter()
        self.lock = threading.Lock()
        self.job = None

    def status(self):
        return {"healthy": True, "error": None, "model": "fixture", "http": {}}

    def batching_status(self):
        return {"schema": "mlx2.batch-runtime.v1", "gauges": {"queue_depth": 0}}

    def submit(self, request, *, tenant_id="default"):
        if self.synchronous:
            raise PromptTemplateFailure("chat template rendering failed: boom")
        self.job = Job(request)
        self.job.tenant_id = tenant_id
        self.job.prompt_tokens, self.job.completion_tokens = 0, 0
        if request.get("stream"):
            # A hosted-tool continuation renders a second prompt after the
            # stream is open, so the failure can only be an SSE error event.
            self.job.events.put({"delta": {"content": "thinking"}})
        self.job.events.put(
            {
                "error": "this model's chat template requires a field the "
                "request does not provide (tools[0].function.description is "
                "missing)",
                "status": 400,
            }
        )
        return self.job


@pytest.fixture(params=[True, False], ids=["synchronous", "engine_event"])
def failing_server(request):
    engine = TemplateFailureEngine(synchronous=request.param)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield engine, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


SURFACES = {
    "chat": (
        "/v1/chat/completions",
        {"model": "fixture", "messages": [{"role": "user", "content": "hi"}]},
    ),
    "responses": ("/v1/responses", {"model": "fixture", "input": "hi"}),
    "anthropic": (
        "/v1/messages",
        {
            "model": "fixture",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ),
}


def call(base, path, body):
    return urlopen(
        Request(
            base + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
    )


@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_template_failure_is_never_an_unhandled_500(failing_server, surface):
    engine, base = failing_server
    path, body = SURFACES[surface]
    with pytest.raises(HTTPError) as failure:
        call(base, path, body)
    payload = json.loads(failure.value.read())
    message = json.dumps(payload)
    if engine.synchronous:
        assert failure.value.code == 500
        assert "chat template rendering failed" in message
    else:
        assert failure.value.code == 400
        assert "tools[0].function.description" in message
    assert "internal server error" not in message


@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_template_failure_streams_an_error_event(failing_server, surface):
    engine, base = failing_server
    path, body = SURFACES[surface]
    if engine.synchronous:
        # The failure precedes the first byte, so it is still a status code.
        with pytest.raises(HTTPError) as failure:
            call(base, path, {**body, "stream": True})
        assert failure.value.code == 500
        return
    with call(base, path, {**body, "stream": True}) as response:
        assert response.status == 200
        stream = response.read().decode()
    assert "tools[0].function.description" in stream
    assert "internal server error" not in stream
