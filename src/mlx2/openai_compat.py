"""Bounded OpenAI wire compatibility over mlx2's existing serving contract.

This module intentionally translates only semantics the local runtime can
enforce. Multimodal and hosted-tool inputs cross explicit adapter/executor
boundaries; routes without those capabilities fail before job admission.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from copy import deepcopy

from . import agent_compat as _agent
from .api_resources import CapabilityUnavailable


RESPONSES_FIELDS = frozenset(
    {
        "input",
        "instructions",
        "max_output_tokens",
        "metadata",
        "model",
        "parallel_tool_calls",
        "previous_response_id",
        "sampling_profile",
        "store",
        "stream",
        "temperature",
        "thinking_budget",
        "text",
        "tool_choice",
        "tools",
        "top_p",
        "session_id",
        "reasoning",
        "include",
        "user",
        "prompt_cache_key",
        "truncation",
        "service_tier",
        "stream_options",
        "top_logprobs",
        "return_progress",
        "client_metadata",
    }
)


class ToolContractError(ValueError):
    """Generated tool output violated a validated request contract."""


class ResponsesInputItemsUnavailable(RuntimeError):
    """A stored response lacks enough existing context for an input-item view."""


RESPONSES_INPUT_ITEM_INCLUDES = frozenset(
    {"reasoning.encrypted_content", "message.input_image.image_url"}
)


def _responses_content(content, *, file_resolver=None):
    if isinstance(content, str):
        if not content:
            raise ValueError("Responses input text must not be empty")
        return content
    if not isinstance(content, list) or not content:
        raise ValueError("Responses message content must be text or a nonempty list")
    pieces = []
    has_media = False
    for part in content:
        if not isinstance(part, Mapping):
            raise ValueError("Responses content parts must be objects")
        if part.get("type") in {"input_text", "output_text"} and isinstance(
            part.get("text"), str
        ):
            pieces.append({"type": "text", "text": part["text"]})
        elif part.get("type") == "input_file":
            if file_resolver is None:
                raise ValueError("input_file requires the local Files API")
            pieces.append({"type": "text", "text": file_resolver(part)})
        elif part.get("type") == "input_image":
            unknown = set(part) - {"type", "image_url", "file_id", "detail"}
            if unknown or not any(part.get(key) for key in ("image_url", "file_id")):
                raise ValueError("input_image requires image_url or file_id")
            pieces.append(dict(part))
            has_media = True
        elif part.get("type") == "input_audio":
            unknown = set(part) - {"type", "input_audio", "audio_url", "file_id"}
            if unknown or not any(
                part.get(key) for key in ("input_audio", "audio_url", "file_id")
            ):
                raise ValueError("input_audio requires data, audio_url, or file_id")
            pieces.append(dict(part))
            has_media = True
        elif part.get("type") in {"input_video", "video_url"}:
            unknown = set(part) - {
                "type", "video_url", "file_id", "fps", "max_frames"
            }
            if unknown or not any(part.get(key) for key in ("video_url", "file_id")):
                raise ValueError("input_video requires video_url or file_id")
            pieces.append({**part, "type": "input_video"})
            has_media = True
        else:
            raise ValueError("unsupported Responses content part")
    text = "".join(part["text"] for part in pieces if part["type"] == "text")
    if not text and not has_media:
        raise ValueError("Responses input text must not be empty")
    return pieces if has_media else text


def _responses_messages(
    value,
    *,
    file_resolver=None,
    signer=None,
    model=None,
    tenant_id="default",
    rejection_counter=None,
    compat=None,
    counts=None,
) -> list[dict]:
    compat_on = compat is not None and compat.enabled
    if isinstance(value, str):
        if not value:
            raise ValueError("input must not be empty")
        return [{"role": "user", "content": value}]
    if not isinstance(value, list) or not value:
        raise ValueError("input must be text or a nonempty list of text messages")
    messages = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("Responses input items must be objects")
        item_type = item.get("type", "message")
        if not compat_on and (
            item_type in {"custom_tool_call", "custom_tool_call_output"}
            or (item_type == "function_call" and "namespace" in item)
            or (item_type == "message" and "phase" in item)
        ):
            _agent.require_compat(
                compat,
                f"{item_type} items"
                + (" with namespace" if item_type == "function_call" else "")
                + (" with phase" if item_type == "message" else ""),
            )
        if item_type == "reasoning":
            # ``content`` (null or reasoning_text parts) is accepted because
            # this server's own reasoning output items carry it, so a plain
            # ``input += response.output`` loop echoes it back.  It is never
            # trusted: replay still requires the signed ``encrypted_content``.
            unknown = set(item) - {
                "type", "id", "status", "summary", "encrypted_content", "content"
            }
            if unknown:
                raise ValueError("unsupported reasoning item fields")
            summary = item.get("summary", [])
            if not isinstance(summary, list) or any(
                not isinstance(part, Mapping)
                or part.get("type") != "summary_text"
                or not isinstance(part.get("text"), str)
                for part in summary
            ):
                raise ValueError("reasoning summary must contain summary_text parts")
            summary_text = "".join(part["text"] for part in summary)
            replay = None
            if signer is not None and "encrypted_content" in item:
                replay = signer.verify_responses(
                    item["encrypted_content"],
                    model=model,
                    tenant=tenant_id,
                    expected_text=summary_text if summary else None,
                )
            if replay is None:
                if rejection_counter is not None:
                    rejection_counter[0] += 1
            elif messages and messages[-1].get("role") == "assistant":
                messages[-1]["reasoning_content"] = (
                    messages[-1].get("reasoning_content", "") + replay
                )
            else:
                messages.append(
                    {"role": "assistant", "content": "", "reasoning_content": replay}
                )
            continue
        if item_type == "function_call_output" or (
            compat_on and item_type == "custom_tool_call_output"
        ):
            if set(item) - {"type", "call_id", "output", "id", "status"}:
                raise ValueError(f"unsupported {item_type} fields")
            call_id, output = item.get("call_id"), item.get("output")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError(f"{item_type} requires call_id")
            if compat_on:
                output = _agent.tool_output_text(output, kind=item_type)
            if not isinstance(output, str):
                raise ValueError("function_call_output requires text output")
            messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": output}
            )
            continue
        if compat_on and item_type == "custom_tool_call":
            if set(item) - {"type", "call_id", "name", "input", "id", "status"}:
                raise ValueError("unsupported custom_tool_call fields")
            if not all(
                isinstance(item.get(key), str) and item[key]
                for key in ("call_id", "name")
            ) or not isinstance(item.get("input"), str):
                raise ValueError("custom_tool_call requires call_id, name, and input")
            _append_call(
                messages,
                item["call_id"],
                item["name"],
                _agent.shim_arguments(item["input"]),
            )
            _agent.count(counts, "agent_compat_custom_tool_replays")
            continue
        if item_type == "function_call":
            allowed = {"type", "call_id", "name", "arguments", "id", "status"}
            if compat_on:
                allowed = allowed | {"namespace"}
            if set(item) - allowed:
                raise ValueError("unsupported function_call fields")
            if not all(
                isinstance(item.get(key), str) and item[key]
                for key in ("call_id", "name", "arguments")
            ):
                raise ValueError("function_call requires call_id, name, and arguments")
            _append_call(
                messages,
                item["call_id"],
                _agent.qualified_call_name(item) if compat_on else item["name"],
                item["arguments"],
            )
            continue
        if item_type != "message":
            raise ValueError("unsupported Responses input item type")
        role = item.get("role")
        if role not in {"user", "assistant", "system", "developer"}:
            raise ValueError("unsupported Responses message role")
        if compat_on:
            unknown = set(item) - {"type", "role", "content", "id", "status", "phase"}
            if unknown:
                raise ValueError(
                    "unsupported Responses message fields: " + ", ".join(sorted(unknown))
                )
            phase = item.get("phase")
            if phase is not None:
                if phase not in _agent.MESSAGE_PHASES:
                    raise ValueError("message phase must be commentary or final_answer")
                _agent.count(counts, "agent_compat_phase_inputs")
        content = _responses_content(item.get("content"), file_resolver=file_resolver)
        previous = messages[-1] if messages else None
        if (
            role == "assistant"
            and isinstance(content, str)
            and previous is not None
            and previous.get("role") == "assistant"
            and not previous.get("content")
            and not previous.get("tool_calls")
        ):
            # A replayed reasoning item precedes its message: one turn.  This
            # and the call merge in ``_append_call`` hold in every mode, so a
            # stateless ``input += response.output`` replay renders the same
            # history as ``previous_response_id`` does.
            previous["content"] = content
            continue
        messages.append(
            {
                "role": "system" if role == "developer" else role,
                "content": content,
            }
        )
    return messages


def _append_call(messages, call_id, name, arguments):
    call = {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }
    previous = messages[-1] if messages else None
    if previous is not None and previous.get("role") == "assistant":
        # Reasoning, commentary text and (parallel) calls of one model turn
        # render as the single assistant message the model produced.
        previous.setdefault("tool_calls", []).append(call)
        return
    messages.append({"role": "assistant", "content": "", "tool_calls": [call]})


def _responses_tools(value, *, tool_backend=None):
    if not isinstance(value, list) or not 1 <= len(value) <= 64:
        raise ValueError("tools must contain 1 to 64 function definitions")
    if tool_backend is not None:
        return tool_backend.prepare(value)
    tools = []
    for tool in value:
        if not isinstance(tool, Mapping):
            raise ValueError("Responses tools must be objects")
        if tool.get("type") != "function":
            raise CapabilityUnavailable(
                "the loaded route does not implement hosted or MCP tools"
            )
        unknown = set(tool) - {"type", "name", "description", "parameters", "strict"}
        if unknown:
            raise ValueError(
                "unsupported Responses function fields: " + ", ".join(sorted(unknown))
            )
        function = {key: tool[key] for key in tool if key != "type"}
        tools.append({"type": "function", "function": function})
    return tools, {}


def _responses_text_format(value):
    if value is None:
        return None
    if (
        not isinstance(value, Mapping)
        or not value
        or set(value) - {"format", "verbosity"}
    ):
        raise ValueError("text only supports format and verbosity")
    if value.get("verbosity", "medium") not in {"low", "medium", "high"}:
        raise ValueError("text.verbosity must be low, medium, or high")
    format_value = value.get("format", {"type": "text"})
    if not isinstance(format_value, Mapping):
        raise ValueError("text.format must be an object")
    kind = format_value.get("type")
    if kind == "text" and set(format_value) == {"type"}:
        return None
    if kind == "json_object" and set(format_value) == {"type"}:
        return {"type": "json_object"}
    if kind == "json_schema" and not set(format_value) - {
        "type",
        "name",
        "description",
        "schema",
        "strict",
    }:
        return {
            "type": "json_schema",
            "json_schema": {
                key: format_value[key]
                for key in ("name", "description", "schema", "strict")
                if key in format_value
            },
        }
    raise ValueError("text.format supports text, json_object, or strict json_schema")


def _metadata(value) -> dict:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or len(value) > 16:
        raise ValueError("metadata must be an object with at most 16 entries")
    result = {}
    for key, item in value.items():
        if not isinstance(key, str) or not 1 <= len(key) <= 64:
            raise ValueError("metadata keys must contain 1 to 64 characters")
        if not isinstance(item, str):
            raise ValueError("metadata values must be strings")
        if len(item) > 512:
            raise ValueError("metadata values must contain at most 512 characters")
        result[key] = item
    return result


def responses_to_chat_request(
    body,
    *,
    file_resolver=None,
    tool_backend=None,
    signer=None,
    tenant_id="default",
    model=None,
    agent_compat=None,
    counts=None,
) -> tuple[dict, dict]:
    """Translate one bounded Responses request to chat serving."""
    compat_on = agent_compat is not None and agent_compat.enabled
    if not isinstance(body, Mapping):
        raise ValueError("request must be a JSON object")
    unknown = set(body) - RESPONSES_FIELDS
    if unknown:
        raise ValueError(f"unsupported Responses fields: {', '.join(sorted(unknown))}")
    if "input" not in body:
        raise ValueError("input is required")
    if "store" in body and not isinstance(body["store"], bool):
        raise ValueError("store must be boolean")
    if "user" in body and not isinstance(body["user"], str):
        raise ValueError("user must be text")
    if "client_metadata" in body:
        # Client telemetry (Codex sends turn/session ids).  Validated and
        # ignored like ``user``: it never changes local execution.
        client_metadata = body["client_metadata"]
        if (
            not isinstance(client_metadata, Mapping)
            or len(client_metadata) > 32
            or any(
                not isinstance(key, str)
                or not isinstance(value, str)
                or len(key) > 256
                or len(value) > 8192
                for key, value in client_metadata.items()
            )
        ):
            raise ValueError(
                "client_metadata must map at most 32 text keys to text values"
            )
    if "prompt_cache_key" in body and (
        not isinstance(body["prompt_cache_key"], str)
        or not body["prompt_cache_key"]
    ):
        raise ValueError("prompt_cache_key must be nonempty text")
    if body.get("truncation", "disabled") != "disabled":
        raise ValueError("truncation only supports disabled")
    service_tier = body.get("service_tier", "auto")
    if not isinstance(service_tier, str) or service_tier not in {"auto", "default"}:
        raise ValueError("service_tier must be auto or default")
    include = body.get("include", [])
    if not isinstance(include, list) or any(
        not isinstance(item, str) or item not in {
            "reasoning.encrypted_content",
            "message.output_text.logprobs",
        }
        for item in include
    ):
        raise ValueError(
            "include only supports reasoning.encrypted_content and "
            "message.output_text.logprobs"
        )
    if body.get("top_logprobs", 0) and "message.output_text.logprobs" not in include:
        raise ValueError(
            "top_logprobs requires include: message.output_text.logprobs"
        )
    if "stream_options" in body:
        options = body["stream_options"]
        if (
            not isinstance(options, Mapping)
            or set(options) - {"include_usage", "include_obfuscation"}
            or any(not isinstance(value, bool) for value in options.values())
        ):
            raise ValueError("stream_options values must be boolean")
    previous_response_id = body.get("previous_response_id")
    if previous_response_id is not None and (
        not isinstance(previous_response_id, str) or not previous_response_id
    ):
        raise ValueError("previous_response_id must be nonempty text")

    rejection_counter = [0]
    context_messages = _responses_messages(
        body["input"],
        file_resolver=file_resolver,
        signer=signer,
        model=model or body.get("model"),
        tenant_id=tenant_id,
        rejection_counter=rejection_counter,
        compat=agent_compat,
        counts=counts,
    )
    messages = list(context_messages)
    instructions = body.get("instructions")
    if instructions is not None:
        if not isinstance(instructions, str) or not instructions:
            raise ValueError("instructions must be nonempty text")
        messages.insert(0, {"role": "system", "content": instructions})

    request = {"messages": messages}
    aliases = {
        "model": "model",
        "max_output_tokens": "max_tokens",
        "parallel_tool_calls": "parallel_tool_calls",
        "sampling_profile": "sampling_profile",
        "stream": "stream",
        "temperature": "temperature",
        "thinking_budget": "thinking_budget",
        "tool_choice": "tool_choice",
        "top_p": "top_p",
        "top_logprobs": "top_logprobs",
        "return_progress": "return_progress",
    }
    for source, target in aliases.items():
        if source in body:
            request[target] = body[source]
    if "session_id" in body:
        request["session_id"] = body["session_id"]
    if "message.output_text.logprobs" in include:
        request["logprobs"] = True
    if "reasoning" in body:
        reasoning = body["reasoning"]
        if not isinstance(reasoning, Mapping) or set(reasoning) - {
            "effort", "summary"
        }:
            raise ValueError("reasoning only supports effort and summary")
        if "effort" in reasoning:
            request["reasoning_effort"] = reasoning["effort"]
        if reasoning.get("summary", "auto") not in {
            "auto", "concise", "detailed"
        }:
            raise ValueError("reasoning.summary must be auto, concise, or detailed")
    choice = request.get("tool_choice")
    if isinstance(choice, Mapping) and set(choice) == {"type", "name"}:
        allowed_choice = {"function", "custom"} if compat_on else {"function"}
        if choice.get("type") not in allowed_choice:
            raise ValueError("mlx2 Responses supports function tool_choice only")
        request["tool_choice"] = {
            "type": "function",
            "function": {"name": choice.get("name")},
        }
    tool_executors = {}
    tool_map = None
    if "tools" in body and compat_on and tool_backend is None:
        tools, tool_map = _agent.translate_responses_tools(
            body["tools"], agent_compat, counts
        )
        if tools:
            request["tools"] = tools
        else:
            # Every declared tool was hosted and dropped: plain generation.
            request.pop("parallel_tool_calls", None)
            if request.get("tool_choice") not in {None, "auto", "none"}:
                raise ValueError("tool_choice names a tool this route cannot run")
            request.pop("tool_choice", None)
    elif "tools" in body:
        request["tools"], tool_executors = _responses_tools(
            body["tools"], tool_backend=tool_backend
        )
    response_format = _responses_text_format(body.get("text"))
    if response_format is not None:
        request["response_format"] = response_format
    options = {
        "metadata": _metadata(body.get("metadata")),
        "store": body.get("store", True),
        "previous_response_id": previous_response_id,
        "context_messages": context_messages,
        "tool_executors": tool_executors,
    }
    if "tool_choice" in body:
        # The response object echoes the caller's value, not the internal
        # chat shape it was translated to.
        options["tool_choice"] = deepcopy(body["tool_choice"])
    if include:
        options["include"] = include
    if rejection_counter[0]:
        options["reasoning_signature_rejections"] = rejection_counter[0]
    if compat_on:
        options["agent_compat"] = tool_map or {
            "custom": {}, "namespaces": {}, "dropped": []
        }
        _agent.count(counts, "agent_compat_requests")
    return request, options


def normalize_tool_choice(body: dict) -> dict:
    """Validate tool controls while preserving their enforceable semantics."""
    result = dict(body)
    choice = result.get("tool_choice", "auto")
    tools = result.get("tools")
    names = {
        tool["function"]["name"]
        for tool in tools or ()
        if isinstance(tool, Mapping) and isinstance(tool.get("function"), Mapping)
    }
    if isinstance(choice, str):
        if choice not in {"auto", "none", "required"}:
            raise ValueError("tool_choice supports auto, none, required, or one function")
        if choice == "required" and not names:
            raise ValueError("tool_choice required needs at least one function tool")
    elif isinstance(choice, Mapping):
        if set(choice) != {"type", "function"} or choice.get("type") != "function":
            raise ValueError("named tool_choice must select one function")
        function = choice.get("function")
        if not isinstance(function, Mapping) or set(function) != {"name"}:
            raise ValueError("named tool_choice requires exactly function.name")
        name = function.get("name")
        if not isinstance(name, str) or name not in names:
            raise ValueError("named tool_choice must name a declared function")
        # Limit both prompt rendering and parsing to the function the caller
        # selected.  Terminal validation below remains the fail-closed guard.
        result["tools"] = [
            tool for tool in tools if tool["function"]["name"] == name
        ]
    else:
        raise ValueError("invalid tool_choice")
    parallel = result.get("parallel_tool_calls", True)
    if not isinstance(parallel, bool):
        raise ValueError("parallel_tool_calls must be boolean")
    if not names and "parallel_tool_calls" in result:
        raise ValueError("parallel_tool_calls requires function tools")
    return result


def _validate_schema_value(schema, value, path="arguments"):
    if "enum" in schema:
        if value not in schema["enum"]:
            raise ValueError(f"{path} is not one of the strict tool enum values")
        return
    if "const" in schema:
        if value != schema["const"]:
            raise ValueError(f"{path} does not match the strict tool const value")
        return
    kind = schema.get("type")
    if isinstance(kind, list):
        errors = []
        for item in kind:
            try:
                _validate_schema_value({**schema, "type": item}, value, path)
                return
            except ValueError as error:
                errors.append(error)
        raise ValueError(f"{path} does not match any strict tool type") from errors[-1]
    matches = {
        "string": lambda: isinstance(value, str),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": lambda: isinstance(value, bool),
        "null": lambda: value is None,
        "array": lambda: isinstance(value, list),
        "object": lambda: isinstance(value, dict),
    }
    if kind not in matches or not matches[kind]():
        raise ValueError(f"{path} does not match strict tool type {kind!r}")
    if kind == "number" and isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be a finite strict tool number")
    if kind == "array":
        for index, item in enumerate(value):
            _validate_schema_value(schema["items"], item, f"{path}[{index}]")
    elif kind == "object":
        properties = schema.get("properties", {})
        missing = set(schema.get("required", ())) - set(value)
        if missing:
            raise ValueError(f"{path} is missing required keys: {', '.join(sorted(missing))}")
        if schema.get("additionalProperties", False) is False:
            extra = set(value) - set(properties)
            if extra:
                raise ValueError(f"{path} has undeclared keys: {', '.join(sorted(extra))}")
        for name, item in value.items():
            if name in properties:
                _validate_schema_value(properties[name], item, f"{path}.{name}")


def enforce_tool_contract(
    body: Mapping,
    calls: list[dict],
    *,
    finish_reason: str | None = None,
) -> None:
    """Fail closed if generated calls violate requested choice/schema controls."""
    choice = body.get("tool_choice", "auto")
    exhausted = finish_reason == "length" and not calls
    if choice == "none" and calls:
        raise ToolContractError("model emitted a tool call while tool_choice was none")
    if choice == "required" and not calls and not exhausted:
        raise ToolContractError("model did not emit a required tool call")
    if isinstance(choice, Mapping):
        required_name = choice["function"]["name"]
        if (not calls and not exhausted) or any(
            call.get("function", {}).get("name") != required_name for call in calls
        ):
            raise ToolContractError(
                f"model did not exclusively call required function {required_name!r}"
            )
    if body.get("parallel_tool_calls", True) is False and len(calls) > 1:
        raise ToolContractError(
            "model emitted parallel calls while parallel_tool_calls was false"
        )

    definitions = {
        tool["function"]["name"]: tool["function"] for tool in body.get("tools", ())
    }
    for call in calls:
        function = call.get("function", {})
        definition = definitions.get(function.get("name"))
        if definition is None:
            raise ToolContractError("model called an undeclared function")
        if definition.get("strict") is not True:
            continue
        try:
            arguments = json.loads(
                function.get("arguments", ""),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ToolContractError(f"invalid JSON constant {value}")
                ),
            )
        except (TypeError, json.JSONDecodeError) as error:
            raise ToolContractError("strict tool arguments are not valid JSON") from error
        try:
            _validate_schema_value(definition["parameters"], arguments)
        except ValueError as error:
            raise ToolContractError(str(error)) from error


def response_id(job) -> str:
    """Return a Responses-shaped identifier without changing the engine job id."""
    return job.id if str(job.id).startswith("resp_") else f"resp_{job.id}"


def _stored_input_content(content, *, include) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if not isinstance(content, list) or not content:
        raise ResponsesInputItemsUnavailable(
            "response input items cannot be derived from stored message content"
        )
    rendered = []
    for part in content:
        if not isinstance(part, Mapping):
            raise ResponsesInputItemsUnavailable(
                "response input items cannot be derived from stored message content"
            )
        kind = part.get("type")
        if kind == "text" and isinstance(part.get("text"), str):
            rendered.append({"type": "input_text", "text": part["text"]})
        elif kind == "input_image":
            image = {
                "type": "input_image",
                "detail": part.get("detail", "auto"),
            }
            if isinstance(part.get("file_id"), str):
                image["file_id"] = part["file_id"]
            if (
                "message.input_image.image_url" in include
                and isinstance(part.get("image_url"), str)
            ):
                image["image_url"] = part["image_url"]
            if "file_id" not in image and "image_url" not in image:
                rendered.append(
                    {
                        "type": "input_text",
                        "text": "[input image omitted from stored item view]",
                    }
                )
            else:
                rendered.append(image)
        elif kind in {"input_audio", "input_video"}:
            rendered.append(
                {
                    "type": "input_text",
                    "text": f"[{kind.removeprefix('input_')} omitted from stored item view]",
                }
            )
        else:
            raise ResponsesInputItemsUnavailable(
                "response input items cannot be derived from stored message content"
            )
    return rendered


def responses_input_items(
    context_messages,
    response_identifier: str,
    *,
    include=(),
    signer=None,
    model=None,
    tenant_id="default",
) -> list[dict]:
    """Derive public input items solely from main's stored chat context.

    The final context entry is the response output appended by ``store_response``;
    preceding entries are the input context.  Reconstruction deliberately emits
    reduced items when the chat contract discarded Responses-only metadata.
    """
    if (
        not isinstance(context_messages, list)
        or len(context_messages) < 2
        or not isinstance(context_messages[-1], Mapping)
        or context_messages[-1].get("role") != "assistant"
    ):
        raise ResponsesInputItemsUnavailable(
            "response input items cannot be derived from stored context"
        )
    include = frozenset(include)
    unknown_include = include - RESPONSES_INPUT_ITEM_INCLUDES
    if unknown_include:
        raise ValueError(
            "unsupported input_items include values: "
            + ", ".join(sorted(unknown_include))
        )
    suffix = str(response_identifier).removeprefix("resp_")
    items = []
    for index, message in enumerate(context_messages[:-1]):
        if not isinstance(message, Mapping):
            raise ResponsesInputItemsUnavailable(
                "response input items cannot be derived from stored context"
            )
        role = message.get("role")
        if role in {"user", "system"}:
            items.append(
                {
                    "id": f"in_{suffix}_{index}",
                    "type": "message",
                    "status": "completed",
                    "role": role,
                    "content": _stored_input_content(
                        message.get("content"), include=include
                    ),
                }
            )
            continue
        if role == "tool":
            call_id = message.get("tool_call_id")
            output = message.get("content")
            if not isinstance(call_id, str) or not isinstance(output, str):
                raise ResponsesInputItemsUnavailable(
                    "response input items cannot be derived from stored tool context"
                )
            items.append(
                {
                    "id": f"fco_{suffix}_{index}",
                    "type": "function_call_output",
                    "status": "completed",
                    "call_id": call_id,
                    "output": output,
                }
            )
            continue
        if role != "assistant":
            raise ResponsesInputItemsUnavailable(
                "response input items cannot be derived from stored message role"
            )
        reasoning = message.get("reasoning_content")
        if reasoning:
            if not isinstance(reasoning, str):
                raise ResponsesInputItemsUnavailable(
                    "response reasoning input cannot be derived from stored context"
                )
            reasoning_item = {
                "id": f"rs_{suffix}_{index}",
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": reasoning}],
            }
            if "reasoning.encrypted_content" in include:
                if signer is None or model is None:
                    raise ResponsesInputItemsUnavailable(
                        "reasoning encrypted_content cannot be derived for this response"
                    )
                reasoning_item["encrypted_content"] = signer.sign_responses(
                    model=model, tenant=tenant_id, text=reasoning
                )
            items.append(reasoning_item)
        content = message.get("content", "")
        if content:
            if not isinstance(content, str):
                raise ResponsesInputItemsUnavailable(
                    "assistant input cannot be derived from stored context"
                )
            items.append(
                {
                    "id": f"msg_{suffix}_{index}",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": content,
                            "annotations": [],
                        }
                    ],
                }
            )
        calls = message.get("tool_calls", [])
        if not isinstance(calls, list):
            raise ResponsesInputItemsUnavailable(
                "function-call inputs cannot be derived from stored context"
            )
        for call_index, call in enumerate(calls):
            function = call.get("function") if isinstance(call, Mapping) else None
            call_id = call.get("id") if isinstance(call, Mapping) else None
            if (
                not isinstance(call_id, str)
                or not isinstance(function, Mapping)
                or not isinstance(function.get("name"), str)
                or not isinstance(function.get("arguments"), str)
            ):
                raise ResponsesInputItemsUnavailable(
                    "function-call inputs cannot be derived from stored context"
                )
            items.append(
                {
                    "id": call_id or f"fc_{suffix}_{index}_{call_index}",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": call_id,
                    "name": function["name"],
                    "arguments": function["arguments"],
                }
            )
    if not items:
        raise ResponsesInputItemsUnavailable(
            "response input items cannot be derived from stored context"
        )
    return items


def responses_input_items_page(items, *, limit=20, after=None, order="desc"):
    """Apply the public cursor controls to reconstructed input items."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer from 1 to 100")
    if order not in {"asc", "desc"}:
        raise ValueError("order must be asc or desc")
    records = list(items if order == "asc" else reversed(items))
    if after is not None:
        positions = [
            index for index, item in enumerate(records) if item["id"] == after
        ]
        if not positions:
            raise ValueError("after cursor does not exist")
        records = records[positions[0] + 1 :]
    data = records[:limit]
    return {
        "object": "list",
        "data": deepcopy(data),
        "first_id": data[0]["id"] if data else None,
        "last_id": data[-1]["id"] if data else None,
        "has_more": len(records) > limit,
    }


def responses_logprob(value: Mapping) -> dict:
    """Render one internal token probability on the Responses wire."""
    token = str(value.get("token", ""))
    return {
        "token": token,
        "logprob": value.get("logprob"),
        "bytes": list(
            value.get("bytes")
            if value.get("bytes") is not None
            else token.encode("utf-8")
        ),
        "top_logprobs": value.get("top_logprobs", []),
    }


class ResponsesTextLogprobs:
    """Attribute per-token logprobs to the Responses output text.

    The engine emits a token's ``{"logprob"}`` event before the deltas that
    token produced, and a token may produce none while the parser holds text
    back.  A logprob therefore belongs to the item of the next delta, and
    logprobs with no later delta (the stop token) to the item of the last one.
    Only tokens that produced output text belong to ``output_text.logprobs``:
    reasoning and tool-call items have no logprob slot on the wire, and
    attributing their tokens to the message would open it out of order.
    """

    def __init__(self):
        self.text = []
        self._pending = []
        self._last_was_text = False

    def logprob(self, value) -> None:
        self._pending.append(value)

    def delta(self, delta: Mapping) -> list:
        """Return the pending logprobs that ride on this delta's text."""
        if not any(
            delta.get(key) for key in ("content", "reasoning_content", "tool_calls")
        ):
            return []
        attached, self._pending = self._pending, []
        self._last_was_text = bool(delta.get("content"))
        if not self._last_was_text:
            return []
        self.text.extend(attached)
        return attached

    def finish(self) -> list:
        """Return trailing logprobs that belong to the output text."""
        trailing, self._pending = self._pending, []
        if not self._last_was_text:
            return []
        self.text.extend(trailing)
        return trailing


def responses_payload(
    *,
    job,
    model,
    choice,
    usage,
    receipt,
    metadata,
    store=True,
    previous_response_id=None,
    signer=None,
    tenant_id="default",
    include=(),
    agent_compat=None,
    compat_tool_map=None,
    counts=None,
    output_order=None,
    tool_choice=None,
):
    """Render a completed chat choice as a Responses API object.

    ``tool_choice`` is the caller's original value; without it the chat
    request's value is echoed.
    """
    message = choice["message"]
    output_by_kind = {}
    response_identifier = response_id(job)
    reasoning = message.get("reasoning_content", "")
    if reasoning:
        reasoning_item = {
            "id": "rs_" + response_identifier.removeprefix("resp_"),
            "type": "reasoning",
            "status": "completed",
            "summary": [{"type": "summary_text", "text": reasoning}],
            "content": [{"type": "reasoning_text", "text": reasoning}],
        }
        if "reasoning.encrypted_content" in include and signer is not None:
            reasoning_item["encrypted_content"] = signer.sign_responses(
                model=model, tenant=tenant_id, text=reasoning
            )
        output_by_kind["reasoning"] = reasoning_item
    if message.get("content") or not message.get("tool_calls"):
        output_by_kind["message"] = {
                "id": f"msg_{job.id}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": message["content"],
                        "annotations": [],
                        **(
                            {
                                "logprobs": [
                                    responses_logprob(value)
                                    for value in choice.get("logprobs", {}).get(
                                        "content", []
                                    )
                                ]
                            }
                            if "message.output_text.logprobs" in include
                            else {}
                        ),
                    }
                ],
            }
    order = list(output_order or ("reasoning", "message"))
    output = [output_by_kind.pop(kind) for kind in order if kind in output_by_kind]
    output.extend(output_by_kind.values())
    for index, call in enumerate(message.get("tool_calls", ())):
        function = call["function"]
        output.append(
            {
                "id": call.get("id", f"fc_{job.id}_{index}"),
                "type": "function_call",
                "status": "completed",
                "call_id": call.get("id", f"call_{job.id}_{index}"),
                "name": function["name"],
                "arguments": function["arguments"],
            }
        )
    payload = {
        "id": response_identifier,
        "object": "response",
        "created_at": int(job.created),
        "status": "completed",
        "model": model,
        "output": output,
        "parallel_tool_calls": bool(job.request.get("parallel_tool_calls", True)),
        "tool_choice": tool_choice
        if tool_choice is not None
        else job.request.get("tool_choice", "auto"),
        "metadata": metadata,
        "previous_response_id": previous_response_id,
        "store": bool(store),
        "usage": {
            "input_tokens": usage["prompt_tokens"],
            "output_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "input_tokens_details": {
                "cached_tokens": getattr(job, "cached_tokens", 0)
            },
            "output_tokens_details": {
                "reasoning_tokens": int(getattr(job, "reasoning_tokens", 0))
            },
        },
        "mlx2": receipt,
    }
    if compat_tool_map is not None and agent_compat is not None:
        payload["mlx2"] = dict(receipt) if isinstance(receipt, Mapping) else receipt
        _agent.rewrite_responses_output(
            payload, compat_tool_map, agent_compat, counts
        )
    return payload
