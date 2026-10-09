"""Custom-tool grammar validation is time-bounded.

A client declares the grammar and, on the default ``auto`` agent-compat
policy, can opt into ``validate`` by header, so a backtracking-heavy regex
must not pin an HTTP handler thread (and one of the server's connection
slots) for hours.  A validation that cannot finish within its budget fails
closed like a mismatch.
"""

import json
import threading
from collections import Counter

from mlx2.agent_compat import (
    AgentCompatPolicy,
    rewrite_responses_output,
    translate_responses_tools,
)
from mlx2.openai_compat import ToolContractError

REDOS = r"(.*?,){11}P"  # ~2.5x more backtracking per extra "1," repeat


def _validate(text, definition=REDOS):
    compat = AgentCompatPolicy("auto", "off").resolve(
        {"X-MLX2-Agent-Compat": "on", "X-MLX2-Custom-Tool-Grammar": "validate"},
        "default",
        "/v1/responses",
    )
    tools = [{
        "type": "custom",
        "name": "t",
        "format": {"type": "grammar", "syntax": "regex", "definition": definition},
    }]
    _, tool_map = translate_responses_tools(tools, compat, Counter())
    payload = {
        "output": [{
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "t",
            "arguments": json.dumps({"input": text}),
        }],
        "mlx2": {},
    }
    counts = Counter()
    outcome = {}

    def run():
        try:
            rewrite_responses_output(payload, tool_map, compat, counts)
            outcome["result"] = payload["output"][0]
        except Exception as error:  # noqa: BLE001 - recorded for the assert
            outcome["result"] = error

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(5.0)
    return worker, outcome, counts


def test_backtracking_custom_tool_grammar_is_time_bounded():
    # 72 characters: about 1000 s of backtracking without a match timeout.
    worker, outcome, counts = _validate("1," * 36)
    assert not worker.is_alive(), "custom-tool grammar validation ran unbounded"
    assert isinstance(outcome["result"], ToolContractError)
    assert "time budget" in str(outcome["result"])
    assert counts["agent_compat_custom_tool_grammar_rejected"] == 1


def test_bounded_validation_still_accepts_a_matching_input():
    worker, outcome, counts = _validate("1," * 11 + "P")
    assert not worker.is_alive()
    assert outcome["result"]["type"] == "custom_tool_call"
    assert outcome["result"]["input"] == "1," * 11 + "P"
    assert counts["agent_compat_custom_tool_grammar_validated"] == 1
