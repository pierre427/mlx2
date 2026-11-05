"""Bounded host observation of a serving Job for the opt-in repair controller.

Only emitted tool-call deltas and the engine's terminal receipt are consumed.
Model prose, proposed grades and caller-provided pass flags are never scores.
This module performs no inference and never invokes the emitted tool.
"""

from __future__ import annotations

import json
import math
import queue
import time


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON argument key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("nonfinite JSON argument")


def observe_weight_program_job(job, *, timeout_seconds=60.0, max_events=100_000,
                               max_argument_bytes=1 << 20):
    """Consume one Job, returning observations for a host-owned tool oracle.

    The initial repair operation accepts exactly one complete function call.
    Malformed/multiple/truncated calls produce ``protocol_valid=False``; error,
    timeout, event overflow or mismatched runtime identity fail the trial.
    """
    if (type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 3600
            or type(max_events) is not int or not 1 <= max_events <= 1_000_000
            or type(max_argument_bytes) is not int
            or not 1 <= max_argument_bytes <= 16 << 20):
        raise ValueError("invalid observation bounds")
    deadline = time.monotonic() + timeout_seconds
    calls, valid, byte_count = {}, True, 0
    terminal = None
    try:
        for _ in range(max_events):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("weight program observation timed out")
            try:
                event = job.events.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError("weight program observation timed out") from exc
            if not isinstance(event, dict):
                raise ValueError("malformed serving event")  # noqa: TRY004 - one transport-validation failure API
            if "error" in event:
                raise RuntimeError("weight program serving failed: " + str(event["error"])[:1024])
            delta = event.get("delta", {})
            if not isinstance(delta, dict):
                raise ValueError("malformed serving delta")  # noqa: TRY004 - one transport-validation failure API
            byte_count += len(json.dumps(delta, ensure_ascii=False).encode())
            if byte_count > max_argument_bytes:
                raise ValueError("weight program output exceeds observation byte bound")
            chunks = delta.get("tool_calls", [])
            if not isinstance(chunks, list):
                valid = False
                chunks = []
            for chunk in chunks:
                if not isinstance(chunk, dict) or type(chunk.get("index")) is not int:
                    valid = False
                    continue
                index = chunk["index"]
                if index != 0 or chunk.get("type", "function") != "function":
                    valid = False
                function = chunk.get("function", {})
                if not isinstance(function, dict):
                    valid = False
                    continue
                call = calls.setdefault(index, {"name": "", "arguments": "", "id": None})
                if "id" in chunk:
                    if (not isinstance(chunk["id"], str) or not chunk["id"]
                            or call["id"] not in (None, chunk["id"])):
                        valid = False
                    call["id"] = chunk["id"]
                for key in ("name", "arguments"):
                    if key in function:
                        if not isinstance(function[key], str):
                            valid = False
                        else:
                            call[key] += function[key]
            if "finish_reason" in event:
                terminal = event
                break
        if terminal is None:
            raise ValueError("weight program output exceeds observation event bound")
    except BaseException:
        cancelled = getattr(job, "cancelled", None)
        if cancelled is not None:
            cancelled.set()
        raise

    receipt = terminal.get("receipt")
    if not isinstance(receipt, dict):
        raise ValueError("weight program observation requires the engine receipt")  # noqa: TRY004 - one receipt-validation API
    checkpoint = getattr(job, "base_checkpoint", None)
    if not isinstance(checkpoint, str) or receipt.get("checkpoint") != checkpoint:
        raise ValueError("observed base checkpoint identity mismatch")
    valid = bool(valid and terminal["finish_reason"] == "tool_calls"
                 and len(calls) == 1 and 0 in calls
                 and not receipt.get("truncated_tool_call")
                 and not receipt.get("stop_truncated_tool_call"))
    call = calls.get(0, {})
    name, arguments = call.get("name"), None
    if not isinstance(name, str) or not name or not call.get("id"):
        valid = False
    try:
        arguments = json.loads(call.get("arguments", ""), object_pairs_hook=_object,
                               parse_constant=_invalid_constant)
        json.dumps(arguments, allow_nan=False)
        if not isinstance(arguments, dict):
            valid = False
    except (ValueError, TypeError):
        valid = False
    program = receipt.get("weight_program")
    expected_program = getattr(job, "weight_program", None)
    lora = receipt.get("lora") or {}
    fingerprint, program_id, engaged = None, None, False
    if expected_program is not None:
        if (not isinstance(program, dict)
                or any(program.get(key) != expected_program.get(key)
                       for key in ("program_id", "checkpoint", "source_revision", "numerical_policy", "runtime_arithmetic"))
                or program.get("checkpoint") != checkpoint
                or lora.get("fingerprint") != getattr(job, "lora_fingerprint", None)
                or lora.get("name") != getattr(job, "lora_name", None)):
            raise ValueError("observed weight program identity mismatch")
        fingerprint = lora["fingerprint"]
        program_id = program["program_id"]
        engaged = bool(program.get("observed_used") is True
                       and type(lora.get("slot")) is int and lora["slot"] > 0
                       and type(lora.get("adapter_row_forwards")) is int
                       and lora["adapter_row_forwards"] > 0
                       and type(lora.get("adapter_delta_applications")) is int
                       and lora["adapter_delta_applications"] > 0
                       and type(receipt.get("completion_tokens")) is int
                       and receipt["completion_tokens"] > 0)
    elif program is not None:
        raise ValueError("unexpected weight program in base observation")
    replayed = bool(expected_program is not None
                    and job.request.get("_mlx2_weight_program_replay")
                    and job.request.get("skip_writing_prefix_cache") is True
                    and receipt.get("cached_tokens") == 0)
    return {
        "tool_name": name, "arguments": arguments, "protocol_valid": valid,
        "program_id": program_id, "fingerprint": fingerprint,
        "checkpoint": checkpoint, "replayed": replayed, "engaged": engaged,
    }
