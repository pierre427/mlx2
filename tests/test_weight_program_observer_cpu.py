"""Host-only tool-call observation contracts; no model or tensor execution."""

import copy
import queue
import threading
from types import SimpleNamespace

import pytest

from mlx2.runtime.weight_program_observer import observe_weight_program_job

PROGRAM = {
    "program_id": "sha256:program", "checkpoint": "fixture@checkpoint",
    "source_revision": "source", "numerical_policy": "unmerged_float32_factors",
    "runtime_arithmetic": "fp32_delta_round_once_v1", "observed_used": True,
}


def job(*, args='{"x": 2}', finish="tool_calls", compiled=True):
    receipt = {"checkpoint": "fixture@checkpoint", "cached_tokens": 0,
               "completion_tokens": 12}
    if compiled:
        receipt.update(weight_program=copy.deepcopy(PROGRAM), lora={
            "name": "candidate", "fingerprint": "fingerprint", "slot": 1,
            "adapter_row_forwards": 4,
            "adapter_delta_applications": 4,
        })
    events = queue.Queue()
    events.put({"delta": {"tool_calls": [{"index": 0, "id": "call_id",
                "type": "function", "function": {"name": "sum", "arguments": args[:4]}}]}})
    events.put({"delta": {"tool_calls": [{"index": 0,
                "function": {"arguments": args[4:]}}]}})
    events.put({"finish_reason": finish, "receipt": receipt})
    return SimpleNamespace(
        events=events, cancelled=threading.Event(), base_checkpoint="fixture@checkpoint",
        weight_program=copy.deepcopy(PROGRAM) if compiled else None,
        lora_name="candidate" if compiled else None,
        lora_fingerprint="fingerprint" if compiled else None,
        request={"_mlx2_weight_program_replay": "trusted-nonce", "skip_writing_prefix_cache": True}
                if compiled else {},
    )


def mutate_receipt(value, change):
    events = [value.events.get_nowait() for _ in range(value.events.qsize())]
    change(events[-1]["receipt"])
    for event in events:
        value.events.put(event)


def test_real_delta_fragments_and_runtime_identity_form_observation():
    observed = observe_weight_program_job(job())
    assert observed == {"tool_name": "sum", "arguments": {"x": 2}, "protocol_valid": True,
                        "program_id": "sha256:program", "fingerprint": "fingerprint",
                        "checkpoint": "fixture@checkpoint", "replayed": True, "engaged": True}


def test_base_observation_is_not_candidate_engagement():
    observed = observe_weight_program_job(job(compiled=False))
    assert observed["protocol_valid"] is True
    assert observed["checkpoint"] == "fixture@checkpoint"
    assert observed["program_id"] is None
    assert observed["engaged"] is False
    assert observed["replayed"] is False


@pytest.mark.parametrize("args", ['{"x": 1, "x": 2}', '{"x": NaN}', '{"x": 1e999}', '[2]', '{broken'])
def test_invalid_arguments_never_count_as_protocol_success(args):
    assert observe_weight_program_job(job(args=args))["protocol_valid"] is False


@pytest.mark.parametrize("finish", ["length", "stop", "cancelled"])
def test_truncated_or_non_tool_terminal_never_passes(finish):
    assert observe_weight_program_job(job(finish=finish))["protocol_valid"] is False


def test_second_call_and_model_authored_scores_do_not_pass():
    value = job()
    first = value.events.get_nowait()
    first["delta"]["tool_calls"].append({"index": 1, "id": "other", "function": {
        "name": "sum", "arguments": '{"x": 2}'}})
    remaining = [value.events.get_nowait() for _ in range(value.events.qsize())]
    value.events.put({"delta": {"content": '{"passed":true,"engaged":true}'}})
    value.events.put(first)
    for event in remaining:
        value.events.put(event)
    assert observe_weight_program_job(value)["protocol_valid"] is False


@pytest.mark.parametrize("field,value", [("fingerprint", "other"), ("name", "other")])
def test_runtime_adapter_identity_mismatch_fails_closed(field, value):
    item = job()
    mutate_receipt(item, lambda receipt: receipt["lora"].update({field: value}))
    with pytest.raises(ValueError, match="identity mismatch"):
        observe_weight_program_job(item)


def test_counter_and_fresh_replay_are_independent_requirements():
    item = job()
    mutate_receipt(item, lambda receipt: receipt["lora"].update(adapter_row_forwards=0))
    assert observe_weight_program_job(item)["engaged"] is False
    item = job()
    mutate_receipt(item, lambda receipt: receipt["lora"].update(adapter_delta_applications=0))
    assert observe_weight_program_job(item)["engaged"] is False
    item = job()
    mutate_receipt(item, lambda receipt: receipt.update(cached_tokens=10))
    observed = observe_weight_program_job(item)
    assert observed["engaged"] is True and observed["replayed"] is False


def test_observer_cancels_on_bound_or_stream_error():
    item = job()
    with pytest.raises(ValueError, match="byte bound"):
        observe_weight_program_job(item, max_argument_bytes=1)
    assert item.cancelled.is_set()
    item = job()
    item.events = queue.Queue()
    item.events.put({"error": "failure"})
    with pytest.raises(RuntimeError, match="serving failed"):
        observe_weight_program_job(item)
    assert item.cancelled.is_set()
