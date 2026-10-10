"""Authoritative generation evidence without model/device execution."""

from types import SimpleNamespace as NS

import pytest

from mlx2.generation_trace import (
    GeneratedTokenTrace, MAX_TOKEN_TRACE_TOKENS, read_token_trace,
)
from mlx2.server import validate_request
from mlx2.serving import HostPromptCache, Job, record_generation_output
from test_structured_deferral import _collect, scripted_engine  # noqa: F401


def test_trace_is_opt_in_and_preserves_producing_width_after_join_and_shrink():
    for enabled in (False, True):
        job = Job({"return_token_trace": enabled})
        job.observed_width = 4  # historical maximum must not replace producing width
        for token, width in ((7, 2), (8, 4), (9, 1)):
            response = NS(token=token, execution_width=width, mtp_receipt=None)
            record_generation_output(job, response, mtp=False, external_draft=False, prompt_lookup=False)
        assert job.ordinary_compute_widths == {1, 2, 4}
        if not enabled:
            assert job.token_trace is None
            continue
        receipt = {"token_trace": job.token_trace.receipt()}
        trace = read_token_trace(receipt, expected_tokens=3)
        assert trace["token_ids"] == [7, 8, 9]
        assert trace["execution_widths"] == [2, 4, 1]
        assert trace["ordinary_execution_widths"] == [2, 4, 1]
        trace["token_ids"].clear()
        assert job.token_trace.token_ids == [7, 8, 9]  # detached receipt


def test_speculative_width_is_not_reported_as_ordinary():
    job = Job({"return_token_trace": True})
    for receipt, width in (({"route": "mtp"}, 4), (None, 2)):
        record_generation_output(job, NS(token=7, execution_width=width, mtp_receipt=receipt),
                                 mtp=True, external_draft=False, prompt_lookup=False)
    assert job.ordinary_compute_widths == {2}
    assert job.token_trace.ordinary_execution_widths == [None, 2]


def test_trace_bound_is_explicit_and_incomplete_evidence_fails_closed():
    trace = GeneratedTokenTrace()
    for token in range(MAX_TOKEN_TRACE_TOKENS + 1):
        trace.append(token, 2, 2)
    receipt = trace.receipt()
    assert len(receipt["token_ids"]) == MAX_TOKEN_TRACE_TOKENS
    assert receipt["completion_tokens"] == MAX_TOKEN_TRACE_TOKENS + 1
    assert receipt["truncated"] is True
    with pytest.raises(ValueError, match="complete authoritative"):
        read_token_trace({"token_trace": receipt})
    for invalid in ({}, {"token_trace": {"token_ids": [1]}}):
        with pytest.raises(ValueError, match="authoritative"):
            read_token_trace(invalid)


def test_trace_request_validation_and_prompt_cache_exclusion():
    for enabled in (False, True):
        request = validate_request({"prompt": "x", "return_token_trace": enabled}, chat=False)
        assert request["return_token_trace"] is enabled
    for invalid in (0, 1, "true"):
        with pytest.raises(ValueError, match="return_token_trace must be boolean"):
            validate_request({"prompt": "x", "return_token_trace": invalid}, chat=False)
    assert "return_token_trace" not in validate_request(
        {"prompt": "x", "return_token_trace": None}, chat=False
    )
    assert "return_token_trace" in HostPromptCache._NON_PROMPT_FIELDS


def test_actual_worker_receipt_keeps_tokens_hidden_by_text_parser(scripted_engine):
    build, state = scripted_engine
    engine = build(declare_marker=True)
    # The thinking-close marker and EOS count as generated IDs but are absent
    # from rendered answer text. No tokenizer round trip can recover them.
    state["script"] = [2, 5, 12, 0]
    job = engine.submit({"messages": [{"role": "user", "content": "x"}],
                         "enable_thinking": True, "max_tokens": 4,
                         "return_token_trace": True})
    reasoning, content, event = _collect(job)
    assert "error" not in event
    receipt = event["receipt"]
    trace = read_token_trace(receipt, expected_tokens=4)
    assert trace["token_ids"] == state["script"]
    assert trace["execution_widths"] == [1] * 4
    assert receipt["ordinary_compute_widths"] == [1]
    assert receipt["request_controls"]["return_token_trace"] is True
    assert "</think>" not in reasoning + content
    assert "<eos>" not in reasoning + content
