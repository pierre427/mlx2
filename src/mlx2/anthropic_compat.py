"""Anthropic Messages compatibility over the mlx2 chat serving contract."""

from __future__ import annotations

import json
import uuid

from . import agent_compat as _agent

class ModelOutputError(RuntimeError):
    """The model emitted a wire value that cannot be represented safely."""


def _object(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _only(value, allowed, name):
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"unsupported {name} fields: {', '.join(sorted(unknown))}")


def _text(value, name, *, empty=True):
    if not isinstance(value, str) or (not empty and not value):
        raise ValueError(f"{name} must be {'nonempty ' if not empty else ''}text")
    return value


def _cache_control(value):
    """Validate Anthropic's prompt-cache marker; APCv2 itself is automatic."""
    value = _object(value, "cache_control")
    _only(value, {"type", "ttl"}, "cache_control")
    if value.get("type") != "ephemeral":
        raise ValueError("cache_control type must be ephemeral")
    if "ttl" in value and value["ttl"] not in {"5m", "1h"}:
        raise ValueError("cache_control ttl must be 5m or 1h")


def _chat_tool(tool, *, anthropic=False, default_strict=None):
    tool = _object(tool, "tool")
    if anthropic:
        _only(tool, {"name", "description", "input_schema", "cache_control"}, "tool")
        if "cache_control" in tool:
            _cache_control(tool["cache_control"])
        name = _text(tool.get("name"), "tool name", empty=False)
        schema = _object(tool.get("input_schema"), "tool input_schema")
        # ``description`` is optional here; the empty default keeps a chat
        # template that renders it (``| tojson``) from meeting a Jinja
        # Undefined.  ``/v1/messages/count_tokens`` renders without passing
        # through ``validate_request``, so the default has to be set here too.
        function = {"name": name, "description": "", "parameters": schema}
        if "description" in tool:
            function["description"] = _text(tool["description"], "tool description")
        return {"type": "function", "function": function}

    _only(
        tool,
        {"type", "name", "description", "parameters", "strict"},
        "Responses tool",
    )
    if tool.get("type") != "function":
        raise ValueError("only function tools are supported")
    function = {
        "name": _text(tool.get("name"), "tool name", empty=False),
        "description": "",
        "parameters": _object(tool.get("parameters", {}), "tool parameters"),
    }
    if "description" in tool:
        function["description"] = _text(tool["description"], "tool description")
    if "strict" in tool:
        if not isinstance(tool["strict"], bool):
            raise ValueError("tool strict must be boolean")
        function["strict"] = tool["strict"]
    elif default_strict is not None:
        function["strict"] = bool(default_strict)
    return {"type": "function", "function": function}


def _anthropic_system(value):
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise ValueError("system must be text or a list of text blocks")
    parts = []
    for block in value:
        block = _object(block, "system block")
        _only(block, {"type", "text", "cache_control"}, "system block")
        if "cache_control" in block:
            _cache_control(block["cache_control"])
        if block.get("type") != "text":
            raise ValueError("only text system blocks are supported")
        parts.append(_text(block.get("text"), "system block text"))
    return "".join(parts)


def _anthropic_message(
    message, *, signer=None, model=None, tenant_id="default", compat=False,
    counts=None, resolved=None,
):
    message = _object(message, "message")
    _only(message, {"role", "content"}, "message")
    role = message.get("role")
    if compat and role == "system":
        # Claude Code's mid-conversation-system beta.  Folded into a user turn
        # by ``fold_system_messages`` once the whole transcript is known.
        return [{"role": "system", "content": _anthropic_system(
            message.get("content")
        )}], 0
    if role not in {"user", "assistant"}:
        raise ValueError("Anthropic message role must be user or assistant")
    content = message.get("content")
    if isinstance(content, str):
        return [{"role": role, "content": content}], 0
    if not isinstance(content, list) or not content:
        raise ValueError("message content must be text or a nonempty block list")
    if role == "assistant":
        text, reasoning, calls = [], [], []
        rejections = 0
        for block in content:
            block = _object(block, "assistant content block")
            kind = block.get("type")
            if kind == "text":
                _only(
                    block,
                    {"type", "text", "cache_control", "citations"},
                    "text block",
                )
                if block.get("citations") is not None:
                    raise ValueError("assistant text citations are unsupported")
                if "cache_control" in block:
                    _cache_control(block["cache_control"])
                text.append(_text(block.get("text"), "text block text"))
            elif kind == "tool_use":
                _only(
                    block,
                    {
                        "type",
                        "id",
                        "name",
                        "input",
                        "cache_control",
                        "caller",
                    },
                    "tool_use block",
                )
                caller = block.get("caller")
                if caller is not None:
                    caller = _object(caller, "tool_use caller")
                    if caller.get("type") != "direct":
                        raise ValueError("non-direct tool_use caller is unsupported")
                    _only(caller, {"type"}, "tool_use caller")
                if "cache_control" in block:
                    _cache_control(block["cache_control"])
                calls.append(
                    {
                        "id": _text(block.get("id"), "tool_use id", empty=False),
                        "type": "function",
                        "function": {
                            "name": _text(
                                block.get("name"), "tool_use name", empty=False
                            ),
                            "arguments": json.dumps(
                                _object(block.get("input"), "tool_use input"),
                                allow_nan=False,
                                separators=(",", ":"),
                            ),
                        },
                    }
                )
            elif kind == "thinking":
                _only(
                    block,
                    {"type", "thinking", "signature", "cache_control"},
                    "thinking block",
                )
                if "cache_control" in block:
                    _cache_control(block["cache_control"])
                thinking = _text(block.get("thinking"), "thinking text")
                signature = block.get("signature")
                if (
                    not compat
                    and isinstance(signature, str)
                    and signature.startswith("mlx2.thinkingc.")
                ):
                    _agent.require_compat(resolved, "omitted-display thinking signatures")
                restored = (
                    signer.verify_anthropic_carrying(
                        signature, model=model, tenant=tenant_id
                    )
                    if compat
                    and not thinking
                    and signer is not None
                    and isinstance(signature, str)
                    else None
                )
                if restored is not None:
                    reasoning.append(restored)
                    _agent.count(counts, "agent_compat_thinking_restored")
                elif (
                    signer is not None
                    and isinstance(signature, str)
                    and signer.verify_anthropic(
                        signature, model=model, tenant=tenant_id, text=thinking
                    )
                ):
                    reasoning.append(thinking)
                else:
                    rejections += 1
            elif kind == "redacted_thinking":
                _only(
                    block,
                    {"type", "data", "cache_control"},
                    "redacted_thinking block",
                )
                if "cache_control" in block:
                    _cache_control(block["cache_control"])
                _text(block.get("data"), "redacted thinking data")
                rejections += 1
            elif kind in {"image", "document"}:
                raise ValueError("image and document content blocks are unsupported")
            else:
                raise ValueError(f"unsupported assistant content block: {kind!r}")
        translated = {"role": "assistant", "content": "".join(text)}
        if reasoning:
            translated["reasoning_content"] = "".join(reasoning)
        if calls:
            translated["tool_calls"] = calls
        return [translated], rejections

    translated = []
    pending = []

    def flush_text():
        if pending:
            translated.append({"role": "user", "content": "".join(pending)})
            pending.clear()

    for block in content:
        block = _object(block, "user content block")
        kind = block.get("type")
        if kind == "text":
            _only(block, {"type", "text", "cache_control"}, "text block")
            if "cache_control" in block:
                _cache_control(block["cache_control"])
            pending.append(_text(block.get("text"), "text block text"))
        elif kind == "tool_result":
            _only(
                block,
                {"type", "tool_use_id", "content", "is_error", "cache_control"},
                "tool_result block",
            )
            if "cache_control" in block:
                _cache_control(block["cache_control"])
            flush_text()
            result = block.get("content", "")
            if isinstance(result, list):
                pieces = []
                for part in result:
                    part = _object(part, "tool_result content block")
                    _only(
                        part,
                        {"type", "text", "cache_control"},
                        "tool_result content block",
                    )
                    if "cache_control" in part:
                        _cache_control(part["cache_control"])
                    if part.get("type") != "text":
                        raise ValueError("tool_result only supports text blocks")
                    pieces.append(_text(part.get("text"), "tool_result text"))
                result = "".join(pieces)
            result = _text(result, "tool_result content")
            if "is_error" in block and not isinstance(block["is_error"], bool):
                raise ValueError("tool_result is_error must be boolean")
            translated.append(
                {
                    "role": "tool",
                    "tool_call_id": _text(
                        block.get("tool_use_id"), "tool_result tool_use_id", empty=False
                    ),
                    # The chat contract has no typed error bit; the tool's
                    # textual result is preserved exactly for the next turn.
                    "content": result,
                }
            )
        elif kind in {"image", "document"}:
            raise ValueError("image and document content blocks are unsupported")
        else:
            raise ValueError(f"unsupported user content block: {kind!r}")
    flush_text()
    return translated, 0


def _anthropic_tool_choice(value):
    value = _object(value, "tool_choice")
    kind = value.get("type")
    if kind in {"auto", "any", "none"}:
        _only(value, {"type", "disable_parallel_tool_use"}, "tool_choice")
        if "disable_parallel_tool_use" in value and not isinstance(
            value["disable_parallel_tool_use"], bool
        ):
            raise ValueError("disable_parallel_tool_use must be boolean")
        return {"auto": "auto", "any": "required", "none": "none"}[kind]
    if kind == "tool":
        _only(value, {"type", "name", "disable_parallel_tool_use"}, "tool_choice")
        if "disable_parallel_tool_use" in value and not isinstance(
            value["disable_parallel_tool_use"], bool
        ):
            raise ValueError("disable_parallel_tool_use must be boolean")
        return {
            "type": "function",
            "function": {
                "name": _text(value.get("name"), "tool_choice name", empty=False)
            },
        }
    raise ValueError("tool_choice type must be auto, any, tool, or none")


def anthropic_request_to_chat(
    body,
    *,
    count_tokens=False,
    signer=None,
    tenant_id="default",
    model=None,
    translation_metadata=None,
    agent_compat=None,
    counts=None,
):
    """Translate Anthropic Messages/count_tokens input to a chat request."""
    body = _object(body, "request")
    compat = agent_compat is not None and agent_compat.enabled
    _only(
        body,
        {
            *(("context_management", "output_config") if compat else ()),
            "model",
            "system",
            "messages",
            "max_tokens",
            "stop_sequences",
            "temperature",
            "top_p",
            "top_k",
            "tools",
            "tool_choice",
            "thinking",
            "metadata",
            "stream",
            "service_tier",
            "container",
            "mcp_servers",
            "session_id",
            "return_progress",
        },
        "Anthropic request",
    )
    if not count_tokens and "max_tokens" not in body:
        raise ValueError("max_tokens is required")
    if "container" in body:
        raise ValueError("container is unsupported")
    if "mcp_servers" in body:
        raise ValueError("mcp_servers is unsupported")
    if body.get("service_tier", "auto") not in {"auto", "standard_only"}:
        raise ValueError("service_tier must be auto or standard_only")
    messages = []
    if "system" in body:
        messages.append(
            {"role": "system", "content": _anthropic_system(body["system"])}
        )
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("messages must be a nonempty list")
    if any(not isinstance(message, dict) for message in raw_messages):
        raise ValueError("messages must contain objects")
    if raw_messages[-1].get("role") == "assistant":
        raise ValueError("assistant prefill is unsupported")
    rejections = 0
    for message in raw_messages:
        translated, dropped = _anthropic_message(
            message,
            signer=signer,
            model=model or body.get("model"),
            tenant_id=tenant_id,
            compat=compat,
            counts=counts,
            resolved=agent_compat,
        )
        messages.extend(translated)
        rejections += dropped
    if compat:
        if "context_management" in body:
            messages = _context_management(body["context_management"], messages, counts)
        messages = _agent.fold_system_messages(messages, counts)
    # Anthropic extended thinking is opt-in.  Keep this explicit so adapters
    # whose native default is thinking-on do not override Messages semantics.
    result = {"messages": messages, "enable_thinking": False}
    if translation_metadata is not None:
        translation_metadata["reasoning_signature_rejections"] = rejections
    if "session_id" in body:
        result["session_id"] = body["session_id"]
    if "model" in body:
        result["model"] = _text(body["model"], "model", empty=False)
    for source, target in (
        ("max_tokens", "max_tokens"),
        ("stop_sequences", "stop"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("stream", "stream"),
        ("return_progress", "return_progress"),
    ):
        if source in body:
            result[target] = body[source]
    if count_tokens and result.get("stream"):
        raise ValueError("stream is unsupported for count_tokens")
    if "tools" in body:
        if not isinstance(body["tools"], list):
            raise ValueError("tools must be a list")
        result["tools"] = [_chat_tool(tool, anthropic=True) for tool in body["tools"]]
    if "tool_choice" in body:
        result["tool_choice"] = _anthropic_tool_choice(body["tool_choice"])
        if body["tool_choice"].get("disable_parallel_tool_use") is True:
            result["parallel_tool_calls"] = False
    if "thinking" in body:
        thinking = _object(body["thinking"], "thinking")
        kind = thinking.get("type")
        if kind == "disabled":
            _only(thinking, {"type"}, "thinking")
            result["enable_thinking"] = False
        elif kind == "enabled":
            _only(thinking, {"type", "budget_tokens"}, "thinking")
            budget = thinking.get("budget_tokens")
            if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
                raise ValueError(
                    "thinking budget_tokens must be a non-negative integer"
                )
            result.update(
                enable_thinking=True,
                thinking_budget=budget,
                thinking_budget_mode="history",
            )
            if (
                type(body.get("max_tokens")) is int
                and budget >= body["max_tokens"]
            ):
                raise ValueError("thinking budget_tokens must be less than max_tokens")
        elif compat and kind == "adaptive":
            _only(thinking, {"type", "display"}, "thinking")
            if thinking.get("display", "summarized") not in {"summarized", "omitted"}:
                raise ValueError("thinking display must be summarized or omitted")
            # Effort-scaled default budget; never the history-pure close mode.
            result["enable_thinking"] = True
            _agent.count(counts, "agent_compat_adaptive_thinking")
            if thinking.get("display") == "omitted":
                _agent.count(counts, "agent_compat_thinking_omitted")
        else:
            raise ValueError("thinking type must be enabled or disabled")
    if compat and "output_config" in body:
        config = _object(body["output_config"], "output_config")
        _only(config, {"effort"}, "output_config")
        if "effort" in config:
            if config["effort"] not in {"low", "medium", "high", "xhigh", "max"}:
                raise ValueError("output_config effort must be low, medium, high, xhigh, or max")
            result["reasoning_effort"] = config["effort"]
            _agent.count(counts, "agent_compat_output_effort")
    if "metadata" in body:
        _object(body["metadata"], "metadata")
    return result


anthropic_to_chat = anthropic_request_to_chat


def _context_management(value, messages, counts=None):
    """Apply the context edits mlx2 can reproduce exactly; reject the rest."""
    value = _object(value, "context_management")
    _only(value, {"edits"}, "context_management")
    edits = value.get("edits", [])
    if not isinstance(edits, list):
        raise ValueError("context_management edits must be a list")
    for edit in edits:
        edit = _object(edit, "context_management edit")
        if edit.get("type") != "clear_thinking_20251015":
            raise ValueError(
                f"unsupported context_management edit: {edit.get('type')!r}"
            )
        _only(edit, {"type", "keep"}, "clear_thinking edit")
        keep = edit.get("keep", {"type": "thinking_turns", "value": 1})
        if keep == "all":
            continue
        keep = _object(keep, "clear_thinking keep")
        _only(keep, {"type", "value"}, "clear_thinking keep")
        turns = keep.get("value")
        if (
            keep.get("type") != "thinking_turns"
            or isinstance(turns, bool)
            or not isinstance(turns, int)
            or turns < 1
        ):
            raise ValueError("clear_thinking keep must be all or thinking_turns >= 1")
        thinking = [
            index
            for index, message in enumerate(messages)
            if message.get("role") == "assistant" and message.get("reasoning_content")
        ]
        cleared = thinking[: max(0, len(thinking) - turns)]
        if cleared:
            messages = [dict(message) for message in messages]
            for index in cleared:
                messages[index].pop("reasoning_content", None)
            _agent.count(counts, "agent_compat_thinking_cleared", len(cleared))
    return messages


def thinking_omitted(request) -> bool:
    """Whether the (agent-compat) request asked for omitted thinking display."""
    thinking = (request or {}).get("thinking")
    return (
        isinstance(thinking, dict)
        and thinking.get("type") == "adaptive"
        and thinking.get("display") == "omitted"
    )


def _anthropic_stop(finish, request, receipt):
    if finish == "length":
        return "max_tokens", None
    if finish == "tool_calls":
        return "tool_use", None
    stopped = (receipt or {}).get("stop_sequence")
    if stopped is not None:
        return "stop_sequence", stopped
    return "end_turn", None


def _anthropic_content(
    message, *, signer=None, model=None, tenant_id="default", omitted=False
):
    content = []
    reasoning = message.get("reasoning_content", "")
    if reasoning:
        if omitted and signer is not None:
            content.append(
                {
                    "type": "thinking",
                    "thinking": "",
                    "signature": signer.sign_anthropic_carrying(
                        model=model, tenant=tenant_id, text=reasoning
                    ),
                }
            )
            reasoning = ""
    if reasoning:
        signature = (
            signer.sign_anthropic(model=model, tenant=tenant_id, text=reasoning)
            if signer is not None
            else ""
        )
        content.append(
            {"type": "thinking", "thinking": reasoning, "signature": signature}
        )
    text = message.get("content", "")
    if text:
        content.append({"type": "text", "text": text})
    for call in message.get("tool_calls", []):
        arguments = call["function"].get("arguments", "{}")
        try:
            parsed = json.loads(arguments)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ModelOutputError("model produced invalid tool input JSON") from exc
        if not isinstance(parsed, dict):
            raise ModelOutputError("model tool input must be a JSON object")
        content.append(
            {
                "type": "tool_use",
                "id": call.get("id") or "toolu_" + uuid.uuid4().hex[:24],
                "name": call["function"]["name"],
                "input": parsed,
            }
        )
    return content


def chat_result_to_anthropic(
    result, request=None, *, signer=None, tenant_id="default"
):
    """Translate one nonstreaming chat result to an Anthropic message."""
    request = request or {}
    choice = result["choices"][0]
    usage = result.get("usage", {})
    receipt = result.get("mlx2") or {}
    cached_tokens = usage.get("prompt_tokens_details", {}).get(
        "cached_tokens", receipt.get("cached_tokens", 0)
    )
    stop_reason, stop_sequence = _anthropic_stop(
        choice.get("finish_reason"), request, receipt
    )
    value = {
        "id": (
            str(result["id"])
            if str(result["id"]).startswith("msg_")
            else "msg_" + str(result["id"]).removeprefix("chatcmpl-")
        ),
        "type": "message",
        "role": "assistant",
        "model": result.get("model"),
        "content": _anthropic_content(
            choice.get("message", {}),
            signer=signer,
            model=result.get("model"),
            tenant_id=tenant_id,
            omitted=thinking_omitted(request),
        ),
        "stop_reason": stop_reason,
        "stop_sequence": stop_sequence,
        "usage": {
            "input_tokens": max(0, usage.get("prompt_tokens", 0) - cached_tokens),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_read_input_tokens": cached_tokens,
            "cache_creation_input_tokens": 0,
        },
        "mlx2": result.get("mlx2"),
    }
    return value


chat_to_anthropic = chat_result_to_anthropic


def api_error(status, message, *, anthropic=False):
    """Return the endpoint-specific error object for an HTTP status."""
    if not anthropic:
        error = {
                "message": str(message),
                "type": "invalid_request_error"
                if status in {400, 404, 413}
                else "server_error",
        }
        if status == 404:
            error["code"] = "not_found"
        return {"error": error}
    error_type = (
        "not_found_error"
        if status == 404
        else "invalid_request_error"
        if status in {400, 408, 413}
        else "overloaded_error"
        if status == 429
        else "api_error"
    )
    return {"type": "error", "error": {"type": error_type, "message": str(message)}}


class AnthropicStreamTranslator:
    """Translate mlx2's internal chat deltas into Anthropic SSE events."""

    def __init__(
        self,
        *,
        message_id,
        model,
        request=None,
        input_tokens=0,
        cache_read_input_tokens=0,
        signer=None,
        tenant_id="default",
    ):
        self.message_id = (
            str(message_id)
            if str(message_id).startswith("msg_")
            else "msg_" + str(message_id).removeprefix("chatcmpl-")
        )
        self.model = model
        self.request = request or {}
        self.input_tokens = int(input_tokens)
        self.cache_read_input_tokens = int(cache_read_input_tokens)
        self.signer = signer
        self.tenant_id = tenant_id
        self.blocks = []
        self.calls = []
        self.active = None
        self.omitted = thinking_omitted(self.request) and signer is not None

    @staticmethod
    def _event(kind, **values):
        return {"type": kind, **values}

    def start(self):
        return [
            self._event(
                "message_start",
                message={
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": max(
                            0, self.input_tokens - self.cache_read_input_tokens
                        ),
                        "output_tokens": 0,
                        "cache_read_input_tokens": self.cache_read_input_tokens,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )
        ]

    def _close_active(self):
        if self.active is None:
            return []
        index = self.active
        block = self.blocks[index]
        events = []
        if block["key"] == "reasoning":
            text = "".join(block["parts"])
            signature = (
                self.signer.sign_anthropic_carrying(
                    model=self.model, tenant=self.tenant_id, text=text
                )
                if self.omitted
                else self.signer.sign_anthropic(
                    model=self.model, tenant=self.tenant_id, text=text
                )
                if self.signer is not None
                else ""
            )
            events.append(
                self._event(
                    "content_block_delta",
                    index=index,
                    delta={"type": "signature_delta", "signature": signature},
                )
            )
        events.append(self._event("content_block_stop", index=index))
        self.active = None
        return events

    def _open(self, key, content_block):
        if self.active is not None and self.blocks[self.active]["key"] == key:
            return []
        events = self._close_active()
        self.blocks.append({"key": key, "parts": []})
        self.active = len(self.blocks) - 1
        events.append(
            self._event(
                "content_block_start",
                index=self.active,
                content_block=content_block,
            )
        )
        return events

    def delta(self, delta):
        events = []
        reasoning = delta.get("reasoning_content", "")
        if reasoning:
            events.extend(
                self._open(
                    "reasoning", {"type": "thinking", "thinking": "", "signature": ""}
                )
            )
            self.blocks[self.active]["parts"].append(reasoning)
            if not self.omitted:
                events.append(
                    self._event(
                        "content_block_delta",
                        index=self.active,
                        delta={"type": "thinking_delta", "thinking": reasoning},
                    )
                )
        text = delta.get("content", "")
        if text:
            events.extend(self._open("text", {"type": "text", "text": ""}))
            self.blocks[self.active]["parts"].append(text)
            events.append(
                self._event(
                    "content_block_delta",
                    index=self.active,
                    delta={"type": "text_delta", "text": text},
                )
            )
        for call in delta.get("tool_calls", []):
            function = call["function"]
            index = len(self.calls)
            stored = {
                "id": call.get("id") or "toolu_" + uuid.uuid4().hex[:24],
                "name": function["name"],
                "arguments": function.get("arguments", ""),
            }
            self.calls.append(stored)
            key = f"call:{index}"
            events.extend(
                self._open(
                    key,
                    {
                        "type": "tool_use",
                        "id": stored["id"],
                        "name": stored["name"],
                        "input": {},
                    },
                )
            )
            events.append(
                self._event(
                    "content_block_delta",
                    index=self.active,
                    delta={
                        "type": "input_json_delta",
                        "partial_json": stored["arguments"],
                    },
                )
            )
        return events

    def finish(self, finish_reason, usage, receipt):
        for call in self.calls:
            try:
                value = json.loads(call["arguments"])
            except (TypeError, json.JSONDecodeError) as error:
                raise ModelOutputError("model produced invalid tool input JSON") from error
            if not isinstance(value, dict):
                raise ModelOutputError("model tool input must be a JSON object")
        events = self._close_active()
        stop_reason, stop_sequence = _anthropic_stop(
            finish_reason, self.request, receipt
        )
        events.append(
            self._event(
                "message_delta",
                delta={"stop_reason": stop_reason, "stop_sequence": stop_sequence},
                usage={"output_tokens": usage.get("completion_tokens", 0)},
                mlx2=receipt,
            )
        )
        events.append(self._event("message_stop"))
        return events

    def failure(self, message, status=503):
        return [api_error(status, message, anthropic=True)]



__all__ = [
    "AnthropicStreamTranslator",
    "ModelOutputError",
    "anthropic_request_to_chat",
    "chat_result_to_anthropic",
    "api_error",
]
