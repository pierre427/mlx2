"""Request and tenant contracts: synthetic token IDs, no checkpoint or Metal."""
from types import SimpleNamespace

import pytest

from mlx2.adapters.ordinary_text import _template_history
from mlx2.openai_compat import ToolContractError, enforce_tool_contract
from mlx2 import server
from mlx2.server import validate_request


def test_verification_accepts_openai_text_parts_in_initial_system_message():
    result = validate_request({"messages": [
        {"role": "system", "content": [{"type": "text", "text": "Verify carefully"}]},
        {"role": "user", "content": "Hello"}], "verification": True})
    assert result["enable_thinking"] is True
    assert result["messages"][0]["content"] == "Verify carefully"


@pytest.mark.parametrize("arguments", ["[]", "null", "broken", [], None, 12])
def test_malformed_tool_history_is_a_request_error_before_template_render(arguments):
    messages = [{"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "example", "arguments": arguments}}]}]
    with pytest.raises(ValueError, match="tool call history arguments must be a JSON object"):
        validate_request({"messages": messages})
    with pytest.raises(ValueError, match=r"messages\[0\].*JSON object"):
        _template_history(messages)


def test_undeclared_tool_diagnostic_does_not_echo_unbounded_names():
    with pytest.raises(ToolContractError) as caught:
        enforce_tool_contract({"tools": []}, [{"function": {"name": "x" * 10000, "arguments": "{}"}}])
    assert str(caught.value).endswith("[name exceeds diagnostic limit]")
    assert len(str(caught.value)) < 200


class Ledger:
    def __init__(self, *, refuse=False):
        self.calls = []
        self.refuse = refuse
    def admit(self, tenant, *, requested_output):
        self.calls.append((tenant, requested_output))
        if self.refuse:
            raise ValueError("quota refused")
        return SimpleNamespace(output_cap=2)
    def record(self, tenant, **record):
        self.calls.append((tenant, record))


def test_owned_generation_reserves_caps_and_records_tenant_usage():
    ledger = Ledger()
    seen = []
    def generate(prompt, budget, **options):
        seen.append((prompt, budget))
        return {"token_ids": [3, 4]}
    result = server.metered_owned_generate(SimpleNamespace(generate=generate), [1, 2, 3], 10,
                                    ledger=ledger, tenant_id="tenant-a")
    assert result == {"token_ids": [3, 4]}
    assert seen == [([1, 2, 3], 2)]
    assert ledger.calls[0] == ("tenant-a", 10)
    tenant, record = ledger.calls[1]
    assert tenant == "tenant-a"
    assert record["status"] == "completed"
    assert record["prompt_tokens"] == 3
    assert record["completion_tokens"] == 2


def test_owned_quota_refusal_never_calls_worker():
    def generate(*args, **options):
        pytest.fail("worker started after admission refusal")
    with pytest.raises(ValueError, match="quota refused"):
        server.metered_owned_generate(SimpleNamespace(generate=generate), [1], 10, ledger=Ledger(refuse=True))


def test_owned_worker_failure_records_terminal_and_releases_reservation():
    ledger = Ledger()
    def generate(*args, **options):
        raise ConnectionAbortedError("client left")
    with pytest.raises(ConnectionAbortedError):
        server.metered_owned_generate(SimpleNamespace(generate=generate), [1], 10, ledger=ledger)
    assert ledger.calls[1][1]["status"] == "cancelled"
    assert ledger.calls[1][1]["reservation"].output_cap == 2


@pytest.mark.parametrize("path,headers,body", [
    ("/v1/completions", {"X-MLX2-TensorFold-Owned": "1"}, {"prompt": "hello", "max_tokens": 4}),
    ("/v1/experimental/tensorfold-owned/generate", {}, {"token_ids": [10], "max_new_tokens": 4}),
])
def test_owned_http_endpoints_apply_tenant_ledger(path, headers, body):
    import json
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.request import Request, urlopen
    from test_tensorfold_owned_http_cpu import FakeEngine
    ledger = Ledger()
    calls = []
    def generate(prompt, budget, **options):
        calls.append(budget)
        return {"token_ids": [11, 12], "route_receipt": {}, "finish_reason": "length"}
    handler = server.handler_for(FakeEngine(), usage_ledger=ledger,
                                 tensorfold_owned_router=SimpleNamespace(generate=generate))
    http = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(f"http://127.0.0.1:{http.server_port}{path}",
                          data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json", "X-Tenant-ID": "tenant-http", **headers})
        with urlopen(request, timeout=3) as response:
            assert response.status == 200
        assert calls == [2]
        assert ledger.calls[0] == ("tenant-http", 4)
        assert ledger.calls[1][0] == "tenant-http"
        assert ledger.calls[1][1]["completion_tokens"] == 2
    finally:
        http.shutdown(); http.server_close(); thread.join()


@pytest.mark.parametrize("budget", [None, "4", 4.5, True, 0, 193])
def test_owned_budget_is_validated_before_reservation_cap(budget):
    ledger = Ledger()
    with pytest.raises(ValueError, match="max_new_tokens 1..192"):
        server.metered_owned_generate(SimpleNamespace(generate=lambda *a, **kw: pytest.fail("invalid budget ran")),
                                      [1], budget, ledger=ledger)
    assert ledger.calls == []


def test_deeply_nested_embedded_tool_arguments_are_a_request_error():
    arguments = '{"a":' + '[' * 12000 + '0' + ']' * 12000 + '}'
    messages = [{"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "example", "arguments": arguments}}]}]
    with pytest.raises(ValueError, match="JSON object"):
        validate_request({"messages": messages})
    with pytest.raises(ValueError, match="JSON object"):
        _template_history(messages)
