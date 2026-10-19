"""Host-only guards for the bounded AST diagnostic harness."""

import importlib.util
import io
import json
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / (
    "qualification/runs/ast-smokes-20261018/ast_smoke.py"
)
SPEC = importlib.util.spec_from_file_location("ast_smoke_20261018", PATH)
AST_SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AST_SMOKE)


def test_skip_b4_removes_both_cold_and_warm_phases():
    phases = AST_SMOKE.ast_phase_plan(
        {"short_explain": "prompt"}, skip_b4=True, skip_sampled=True
    )
    assert phases == {"b1": {"short_explain": "prompt"}}


def test_b4_plan_names_both_real_batch_phases():
    phases = AST_SMOKE.ast_phase_plan(
        {"a": "one", "b": "two", "c": "three", "d": "four"}
    )
    assert phases["b4_cold"] is True
    assert phases["b4_warm"] is True
    assert phases["sampled"] is True


@pytest.mark.parametrize("done", [False, True])
def test_stream_records_done_independently_of_eof(monkeypatch, done):
    event = {"choices": [{"delta": {"content": "ok"}}],
             "usage": {"prompt_tokens": 2, "completion_tokens": 1},
             "mlx2": {"route": "ordinary", "token_trace": {"token_ids": [42]}}}
    response = io.BytesIO(("data: " + json.dumps(event) + "\n\n" +
                           ("data: [DONE]\n\n" if done else "")).encode())
    response.status = 200
    class Connection:
        closed = False
        def request(self, *args, **kwargs):
            pass
        def getresponse(self):
            return response
        def close(self):
            self.closed = True
    connection = Connection()
    monkeypatch.setattr(AST_SMOKE.http.client, "HTTPConnection", lambda *a, **k: connection)
    row = AST_SMOKE.stream(1, "model", "prompt", 1)
    assert row["done"] is done
    assert row["token_ids"] == [42]
    assert connection.closed
