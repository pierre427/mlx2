"""Synthetic host conservation checks; checkpoint not applicable."""
import copy

import pytest

from mlx2.generation_trace import GeneratedTokenTrace, reconcile_completion_tokens


def response(tokens):
    trace = GeneratedTokenTrace()
    for index, token in enumerate(tokens):
        trace.append(token, 1 + index % 3, None)
    return {"usage": {"completion_tokens": len(tokens)},
            "mlx2": {"completion_tokens": len(tokens), "token_trace": trace.receipt()}}


@pytest.mark.parametrize("tokens,maximum", [([], 1), ([0], 1), ([7, 7, 0], 8),
                                          ([1, 2, 3, 4], 4)])
def test_emitted_events_include_repetitions_stop_and_final_clamp(tokens, maximum):
    row = reconcile_completion_tokens(response(tokens), maximum=maximum)
    assert row["trace_tokens"] == row["usage_tokens"] == len(tokens)
    assert row["passed"]


@pytest.mark.parametrize("path,value", [
    (("usage", "completion_tokens"), 4),
    (("usage", "completion_tokens"), True),
    (("mlx2", "completion_tokens"), 4),
    (("mlx2", "completion_tokens"), "3"),
    (("mlx2", "token_trace", "completion_tokens"), 4),
    (("mlx2", "token_trace", "truncated"), True),
    (("mlx2", "token_trace", "token_ids"), [7, 7]),
])
def test_disagreement_refuses_a_performance_number(path, value):
    result = copy.deepcopy(response([7, 7, 0]))
    owner = result
    for key in path[:-1]:
        owner = owner[key]
    owner[path[-1]] = value
    with pytest.raises(ValueError):
        reconcile_completion_tokens(result, maximum=4)


def test_missing_evidence_and_output_past_maximum_fail_closed():
    for result in ({}, {"mlx2": {}}, {"usage": {}}):
        with pytest.raises(ValueError):
            reconcile_completion_tokens(result, maximum=4)
    with pytest.raises(ValueError):
        reconcile_completion_tokens(response([1, 2]), maximum=1)
    for invalid in (False, 0, -1, 4097, 2.0):
        with pytest.raises(ValueError):
            reconcile_completion_tokens(response([]), maximum=invalid)
