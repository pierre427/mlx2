"""Local OpenAI-compatible HTTP serving over the mlx2 execution lifecycle."""

from __future__ import annotations

import argparse
import base64
import collections
import contextlib
from copy import deepcopy
from dataclasses import dataclass
from email import policy as email_policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
import os
import hmac
import hashlib
import ipaddress
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit
import queue
import re
import select
import socket
import threading
import time
import struct
import uuid

from .serving import (
    APCReuseDisabled,
    AdmissionClosed,
    Overloaded,
    PowerGovernorUnavailable,
    PromptTemplateFailure,
    ServingEngine,
    SuspendUnavailable,
    take_prompt_progress,
)
from .power_governor import PowerSignalUnavailable
from .batch_metrics import http_metric_route
# The --cache-bytes default and its post-load clamp live in cache_sizing: pure
# arithmetic with no MLX import, so the parser stays GPU-free.
from .cache_sizing import (  # noqa: F401 - re-exported for callers and tests
    MAX_DEFAULT_CACHE_BYTES,
    MIN_DEFAULT_CACHE_BYTES,
    default_cache_bytes,
    physical_memory_bytes,
)
from .logprobs import MAX_TOP_LOGPROBS, wants_logprobs

# Labeled-choice classifier bound; independent of the logprob serialization cap.
MAX_CLASSIFIER_LABELS = 11
from .request_limits import (
    DEFAULT_OUTPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    validate_default_max_tokens,
)
from .openai_compat import (
    ToolContractError,
    ResponsesInputItemsUnavailable,
    ResponsesTextLogprobs,
    RESPONSES_INPUT_ITEM_INCLUDES,
    enforce_tool_contract,
    normalize_tool_choice,
    refuse_video_sampling_options,
    response_id,
    responses_input_items,
    responses_input_items_page,
    responses_logprob,
    responses_payload,
    responses_to_chat_request,
)
from .api_resources import (
    BatchManager,
    CapabilityUnavailable,
    FileStore,
    ResourceNotFound,
    ResponseStore,
)
from .tool_backend import HostedToolError
from .agent_compat import (
    AgentCompatError,
    AgentCompatPolicy,
    load_tenant_policy,
    fold_system_messages,
)
from .http_security import (
    GateRejection,
    authorize as authorize_http_request,
    load_api_key,
    load_secret_file,
    policy_for_bind,
)
from .anthropic_compat import (
    MAX_STOP_SEQUENCES as ANTHROPIC_MAX_STOP_SEQUENCES,
    AnthropicStreamTranslator,
    ModelOutputError,
    anthropic_request_to_chat,
    api_error as translated_api_error,
    chat_result_to_anthropic,
)


MIN_REQUEST_BODY_BYTES = 2 << 20
MAX_REQUEST_BODY_BYTES = 32 << 20
REQUEST_BODY_BYTES_PER_CONTEXT_TOKEN = 32
REQUEST_BODY_JSON_OVERHEAD_BYTES = 64 << 10
# 1/temperature must stay finite in float32; below this the sampler's scaled
# logprobs become NaN.
MIN_POSITIVE_TEMPERATURE = 1.0 / 3.4028234663852886e38
SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
ADMIN_PREFETCH_LIMIT = 32
# Uploaded file types a browser cannot run.  Files accepts any declared type,
# but /content serves only these as declared; anything else (text/html,
# image/svg+xml, ...) would execute on the API origin when opened.
INERT_FILE_CONTENT_TYPES = frozenset(
    {
        "text/plain",
        "text/markdown",
        "application/json",
        "application/jsonl",
        "application/x-ndjson",
    }
)


def served_file_content_type(content_type):
    """Return the Content-Type /v1/files/{id}/content may send for a file."""
    base = str(content_type or "").split(";", 1)[0].strip().lower()
    return base if base in INERT_FILE_CONTENT_TYPES else "application/octet-stream"


def load_admin_token(path):
    """Load an optional admin bearer token from an owner-only regular file."""
    return load_secret_file(path, flag="--admin-token-file")


# CONTRACT-STUB(owner=item 11)
def load_api_key_file(path):
    """Load an API key from an owner-only 0600 regular file."""
    try:
        return load_admin_token(path)
    except ValueError as error:
        raise ValueError(
            str(error).replace("--admin-token-file", "--api-key-file")
        ) from error


def resolve_reasoning_signing_key(args):
    """Persist the reasoning signing key beside durable API state.

    Stored Responses outlive the process under ``--api-state-dir``; a random
    per-process key would make every signature issued before a restart fail
    verification. Configured sources and the explicit ephemeral opt-out win.
    """
    if (
        args.api_state_dir is None
        or args.reasoning_signing_key_file
        or args.reasoning_signing_key_env
        or args.reasoning_signing_ephemeral
    ):
        return None
    from .reasoning_signatures import PERSISTENT_KEY_NAME, ensure_persistent_key

    path = ensure_persistent_key(
        Path(args.api_state_dir).expanduser().resolve() / PERSISTENT_KEY_NAME
    )
    args.reasoning_signing_key_file = str(path)
    return path


def is_loopback_address(client_address):
    text = str(client_address)
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    try:
        address = ipaddress.ip_address(text.split("%", 1)[0])
    except ValueError:
        return False
    return (
        isinstance(address, ipaddress.IPv4Address)
        and address in ipaddress.ip_network("127.0.0.0/8")
    ) or address == ipaddress.ip_address("::1")


# With tenant auth on, every path is guarded except this allowlist (vLLM
# #56269: enumerate exemptions, never the guarded prefixes).
TENANT_AUTH_OPEN_PATHS = frozenset({"/health"})
TENANT_AUTH_LOOPBACK_OPEN_PATHS = frozenset({"/metrics"})
TENANT_AUTH_ADAPTER_PATHS = frozenset(
    {"/v1/load_lora_adapter", "/v1/unload_lora_adapter"}
)
# The gate API key and tenant credentials are both read from Authorization
# and x-api-key, and tenant auth refuses the two headers when they differ, so
# together they refuse every request that does not present one value that is
# both the gate key and a tenant key.  Tenant auth already guards every route.
TENANT_AUTH_GATE_KEY_CONFLICT = (
    "--api-key-file/--api-key-env cannot be combined with tenant auth: both "
    "read Authorization and x-api-key, so every request would be refused; "
    "tenant credentials already guard every route except /health and "
    "loopback /metrics"
)


def authorize_admin(client_address, authorization, token=None):
    """Return an HTTP status/message pair for the fail-closed admin surface."""
    if not is_loopback_address(client_address):
        return 403, "admin endpoints require a loopback client"
    if token is None:
        return None
    prefix = "Bearer "
    presented = (
        authorization[len(prefix) :]
        if isinstance(authorization, str) and authorization.startswith(prefix)
        else ""
    )
    if not hmac.compare_digest(
        presented.encode("utf-8"), token.encode("utf-8")
    ):
        return 401, "invalid admin bearer token"
    return None


def validate_quiesce_body(body):
    if not isinstance(body, dict):
        raise ValueError("quiesce request must be a JSON object")
    unknown = set(body) - {"drain_timeout_seconds", "suspend"}
    if unknown:
        raise ValueError("unsupported quiesce fields: " + ", ".join(sorted(unknown)))
    timeout = body.get("drain_timeout_seconds", 600)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("drain_timeout_seconds must be numeric")
    if not 0.1 <= timeout <= 3600:
        raise ValueError("drain_timeout_seconds must be between 0.1 and 3600")
    timeout = float(timeout)
    suspend = body.get("suspend", True)
    if not isinstance(suspend, bool):
        raise ValueError("suspend must be boolean")
    return {"drain_timeout_seconds": timeout, "suspend": suspend}


def validate_power_body(body):
    """``POST /v1/admin/power``: ``{"mode": ..., "watts": ...}``, both optional."""
    from .power_governor import MODES

    if not isinstance(body, dict):
        raise ValueError("power request must be a JSON object")
    unknown = set(body) - {"mode", "watts"}
    if unknown:
        raise ValueError("unsupported power fields: " + ", ".join(sorted(unknown)))
    if not body:
        raise ValueError("power request needs mode and/or watts")
    mode = body.get("mode")
    if mode is not None and mode not in MODES:
        raise ValueError("mode must be one of " + ", ".join(MODES))
    watts = body.get("watts")
    if watts is not None and (
        isinstance(watts, bool)
        or not isinstance(watts, (int, float))
        or not math.isfinite(watts)
        or not 1 <= watts <= 1000
    ):
        raise ValueError("watts must be a number from 1 to 1000")
    return {"mode": mode, "watts": None if watts is None else float(watts)}


def validate_resume_body(body):
    if not isinstance(body, dict):
        raise ValueError("resume request must be a JSON object")
    if set(body) - {"prefetch_sessions"}:
        raise ValueError("resume only accepts prefetch_sessions")
    sessions = body.get("prefetch_sessions", [])
    if not isinstance(sessions, list) or len(sessions) > ADMIN_PREFETCH_LIMIT:
        raise ValueError(
            f"prefetch_sessions must be a list of at most {ADMIN_PREFETCH_LIMIT} entries"
        )
    normalized = []
    for item in sessions:
        if not isinstance(item, dict) or set(item) != {"tenant", "session_id"}:
            raise ValueError("each prefetch session requires only tenant and session_id")
        tenant = item["tenant"]
        if not isinstance(tenant, str) or not 1 <= len(tenant) <= 128:
            raise ValueError("prefetch tenant must contain 1 to 128 characters")
        normalized.append((tenant, validate_session_id(item["session_id"])))
    return tuple(normalized)


def default_max_tokens_arg(value):
    """Argparse adapter for the exact ServingEngine output-default bound."""
    try:
        if not isinstance(value, str):
            return validate_default_max_tokens(value)
        value = int(value)
        return validate_default_max_tokens(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def prefill_step_arg(value):
    try:
        step = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("prefill step must be a positive integer") from error
    if step <= 0:
        raise argparse.ArgumentTypeError("prefill step must be a positive integer")
    return step


def validate_session_id(value):
    """Validate the bounded APCv2 session identifier on every API surface."""
    if not isinstance(value, str) or SESSION_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "session_id must be 1 to 128 ASCII letters, digits, dot, underscore, "
            "colon, or hyphen and start with a letter or digit"
        )
    return value


def resolve_request_session(body, header_value):
    if not isinstance(body, dict):
        return body
    value = body.get("session_id")
    if value is not None:
        value = validate_session_id(value)
    if header_value is not None:
        header_value = validate_session_id(header_value)
        if value is not None and value != header_value:
            raise ValueError("session_id conflicts with X-mlx2-Session-ID")
        value = header_value
    return body if value is None else {**body, "session_id": value}

_AUDIO_RESPONSE_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "aac": "audio/aac",
    "opus": "audio/opus",
    "pcm": "audio/pcm",
}


def validate_speech_request(body):
    """Validate the OpenAI speech request without selecting an audio backend."""
    if not isinstance(body, dict):
        raise ValueError("speech request must be a JSON object")
    unknown = set(body) - {
        "model",
        "input",
        "voice",
        "instructions",
        "response_format",
        "speed",
        "stream_format",
    }
    if unknown:
        raise ValueError("unsupported speech fields: " + ", ".join(sorted(unknown)))
    input_text = body.get("input")
    if not isinstance(input_text, str) or not input_text or len(input_text) > 4096:
        raise ValueError("speech input must contain 1 to 4096 characters")
    voice = body.get("voice", "default")
    if isinstance(voice, dict):
        if set(voice) != {"id"} or not isinstance(voice["id"], str):
            raise ValueError("custom speech voice must contain only a string id")
        voice = {"id": voice["id"]}
        voice_length = len(voice["id"])
    else:
        voice_length = len(voice) if isinstance(voice, str) else 0
    if not voice_length or voice_length > 128:
        raise ValueError("speech voice must be a nonempty name or custom voice id")
    instructions = body.get("instructions")
    if instructions is not None and (
        not isinstance(instructions, str) or len(instructions) > 4096
    ):
        raise ValueError("speech instructions must be a string up to 4096 characters")
    response_format = body.get("response_format", "mp3")
    if not isinstance(response_format, str) or response_format not in _AUDIO_RESPONSE_TYPES:
        raise ValueError(
            "speech response_format must be one of "
            + ", ".join(sorted(_AUDIO_RESPONSE_TYPES))
        )
    speed = body.get("speed", 1.0)
    if isinstance(speed, bool) or not isinstance(speed, (int, float)):
        raise ValueError("speech speed must be numeric")
    if not 0.25 <= speed <= 4.0:
        raise ValueError("speech speed must be finite and between 0.25 and 4.0")
    speed = float(speed)
    stream_format = body.get("stream_format", "audio")
    if stream_format not in ("audio", "sse"):
        raise ValueError("speech stream_format must be audio or sse")
    return {
        "model": body.get("model"),
        "input": input_text,
        "voice": voice,
        "instructions": instructions,
        "response_format": response_format,
        "speed": speed,
        "stream_format": stream_format,
    }


def request_body_limit(max_context: int, configured: int | None = None) -> int:
    """Return a bounded HTTP envelope large enough for the configured context."""
    if isinstance(max_context, bool) or not isinstance(max_context, int) or max_context <= 0:
        raise ValueError("max_context must be a positive integer")
    if configured is not None:
        if isinstance(configured, bool) or not isinstance(configured, int) or configured <= 0:
            raise ValueError("max_request_bytes must be a positive integer")
        if configured > MAX_REQUEST_BODY_BYTES:
            raise ValueError(
                f"max_request_bytes must not exceed {MAX_REQUEST_BODY_BYTES}"
            )
        return configured
    derived = (
        max_context * REQUEST_BODY_BYTES_PER_CONTEXT_TOKEN
        + REQUEST_BODY_JSON_OVERHEAD_BYTES
    )
    return min(MAX_REQUEST_BODY_BYTES, max(MIN_REQUEST_BODY_BYTES, derived))


DEFAULT_MAX_CONNECTIONS = 32


def _connection_limit_response(head):
    """The 503 a connection gets when every connection slot is taken."""
    target = head.split(b"\r\n", 1)[0].split(b" ")
    path = target[1].decode("latin-1") if len(target) > 1 else ""
    payload = json.dumps(
        translated_api_error(
            503,
            "server connection limit reached; retry",
            anthropic=path.startswith("/v1/messages"),
        )
    ).encode()
    return (
        b"HTTP/1.1 503 Service Unavailable\r\n"
        b"Content-Type: application/json\r\n"
        b"Retry-After: 1\r\n"
        b"Connection: close\r\n"
        + f"Content-Length: {len(payload)}\r\n\r\n".encode()
        + payload
    )


def _read_ready(sock, limit=1 << 16):
    """Read what has already arrived on a non-blocking socket, up to ``limit``.

    Returns ``(data, closed)``; ``closed`` says the peer has shut its side.
    """
    data = b""
    with contextlib.suppress(BlockingIOError):
        while len(data) < limit:
            chunk = sock.recv(min(1 << 16, limit - len(data)))
            if not chunk:
                return data, True
            data += chunk
    return data, False


class BoundedHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    # A connection arriving with every slot taken (idle keep-alive sockets
    # hold theirs for the idle timeout) is told to retry rather than reset.
    # At most this many refusals run at once, each for at most
    # ``refusal_seconds``; past that the accept thread answers at once,
    # without waiting for the request, and up to ``max_lingering`` of those
    # sockets stay open (drained, never waited on) until the client closes
    # or ``refusal_seconds`` passes.
    max_refusals = 8
    refusal_seconds = 1.0
    max_lingering = 64

    def __init__(
        self, server_address, *args, max_connections=DEFAULT_MAX_CONNECTIONS, **kwargs
    ):
        self.connections = threading.BoundedSemaphore(max_connections)
        self.refusals = threading.BoundedSemaphore(self.max_refusals)
        # (deadline, socket) answered by _refuse_now; accept thread only.
        self._lingering = collections.deque()
        # ``--host ::1``, ``[::1]`` and ``::`` are documented binds; the
        # inherited AF_INET socket cannot resolve them.  Names (``localhost``)
        # stay IPv4 so the default bind does not move to ``::1``.
        host, *rest = server_address
        literal = str(host).strip()
        if literal.startswith("[") and literal.endswith("]"):
            literal = literal[1:-1]
        try:
            ipv6 = ipaddress.ip_address(literal.split("%", 1)[0]).version == 6
        except ValueError:
            ipv6 = False
        if ipv6:
            self.address_family = socket.AF_INET6
            server_address = (literal, *rest)
        super().__init__(server_address, *args, **kwargs)

    def server_bind(self):
        if self.address_family == socket.AF_INET6 and self.server_address[0] == "::":
            # The IPv6 wildcard also serves IPv4, as 0.0.0.0 does.
            with contextlib.suppress(AttributeError, OSError):
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()

    def process_request(self, request, address):
        if not self.connections.acquire(blocking=False):
            self._refuse(request)
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self.connections.release()
            raise

    def _refuse(self, request):
        if not self.refusals.acquire(blocking=False):
            self._refuse_now(request)
            return
        try:
            threading.Thread(
                target=self._send_refusal,
                args=(request,),
                name="mlx2-http-refusal",
                daemon=True,
            ).start()
        except BaseException:
            self.refusals.release()
            self.shutdown_request(request)
            raise

    def _refuse_now(self, request):
        # Runs on the accept thread, so nothing here waits: the reply goes
        # out on a non-blocking socket and the socket lingers, drained
        # between accepts, so request bytes arriving later do not reach a
        # closed socket, whose RST would cut the reply.
        try:
            request.setblocking(False)
            request.send(_connection_limit_response(_read_ready(request)[0]))
            request.shutdown(socket.SHUT_WR)
        except OSError:
            self.shutdown_request(request)
            return
        self._lingering.append((time.monotonic() + self.refusal_seconds, request))
        while len(self._lingering) > self.max_lingering:
            self.shutdown_request(self._lingering.popleft()[1])

    def service_actions(self):
        super().service_actions()
        now = time.monotonic()
        for _ in range(len(self._lingering)):
            deadline, request = self._lingering.popleft()
            try:
                finished = _read_ready(request)[1] or now >= deadline
            except OSError:
                finished = True
            if finished:
                self.shutdown_request(request)
            else:
                self._lingering.append((deadline, request))

    def server_close(self):
        super().server_close()
        while self._lingering:
            self.shutdown_request(self._lingering.popleft()[1])

    def _send_refusal(self, request):
        deadline = time.monotonic() + self.refusal_seconds

        def receive(size):
            budget = deadline - time.monotonic()
            if budget <= 0:
                return b""
            request.settimeout(budget)
            return request.recv(size)

        try:
            # Closing with unread request bytes sends a RST that can discard
            # the reply, so read the head first, then shut the write side and
            # drain any body until the client closes or the budget ends.
            head = b""
            while b"\r\n\r\n" not in head and len(head) < 65536:
                chunk = receive(4096)
                if not chunk:
                    break
                head += chunk
            request.settimeout(self.refusal_seconds)
            request.sendall(_connection_limit_response(head))
            request.shutdown(socket.SHUT_WR)
            drained = 0
            while drained < MAX_REQUEST_BODY_BYTES:
                chunk = receive(1 << 16)
                if not chunk:
                    break
                drained += len(chunk)
        except OSError:
            pass
        finally:
            self.shutdown_request(request)
            self.refusals.release()

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.connections.release()


def _canonical_json_order(value):
    """Copy JSON data with object members in one canonical order.

    Members are sorted by name; arrays keep their order.  A ``properties`` map
    lists its sibling ``required`` names first, then the optional ones (each
    group sorted), because the constrained-output compiler only accepts
    required properties ahead of optional ones.
    """
    if isinstance(value, dict):
        required = value.get("required")
        ordered = {}
        for key in sorted(value, key=str):
            child = value[key]
            if key == "properties" and isinstance(child, dict) and isinstance(required, list):
                names = sorted(child, key=str)
                names = [n for n in names if n in required] + [
                    n for n in names if n not in required
                ]
                ordered[key] = {name: _canonical_json_order(child[name]) for name in names}
            else:
                ordered[key] = _canonical_json_order(child)
        return ordered
    if isinstance(value, list):
        return [_canonical_json_order(item) for item in value]
    return value


def canonical_tool_definition(tool):
    """Copy one function tool with a canonical member order (omlx #4138).

    Agent clients resend equivalent tool schemas with object members in a
    different order each turn; templates render tools through ``tojson``, which
    keeps that order, so the prompt diverged inside the tool block and the
    prefix cache missed the whole conversation.  The wrappers keep the
    conventional ``type, function`` / ``name, description, parameters`` lead;
    everything else follows ``_canonical_json_order``.  Values, array order
    and the caller's objects are unchanged.
    """
    def lead(mapping, first):
        keys = [key for key in first if key in mapping]
        keys += sorted((key for key in mapping if key not in first), key=str)
        return {key: mapping[key] for key in keys}

    function = lead(
        {key: _canonical_json_order(child) for key, child in tool["function"].items()},
        ("name", "description", "parameters"),
    )
    rest = {key: _canonical_json_order(child) for key, child in tool.items() if key != "function"}
    return lead({**rest, "function": function}, ("type", "function"))


def validate_request(
    body,
    chat=True,
    structured_thinking=False,
    allow_buffered_tool_stream=False,
    constrained_tool_grammar=False,
    max_tools=64,
    max_stops=4,
):
    """Validate one request body.

    ``structured_thinking`` says the serving adapter declares a thinking-close
    marker, so structured output may be combined with thinking (the grammar is
    deferred past the marker).  Without it the combination stays rejected.
    """
    if not isinstance(body, dict):
        raise ValueError("request must be a JSON object")
    body = normalize_client_options(body)
    supported = {
        "model",
        "messages",
        "prompt",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "max_tokens",
        "max_completion_tokens",
        "min_tokens",
        "thinking_budget",
        "thinking_budget_mode",
        "thinking_steer_alpha",
        "n",
        "stream",
        "enable_thinking",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "stop",
        "stream_options",
        "reasoning_effort",
        "context_limit",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "sampling_profile",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "response_format",
        "grammar",
        "batch_cohort",
        "mlx_fault",
        "skip_writing_prefix_cache",
        "paged_native_qwen3",
        "paged_native_qwen3_b2",
        "paged_native_hybrid_b2",
        "paged_native_hybrid_packed_prefill",
        "paged_native_packed_n20_research",
        "native_research_input_id",
        "native_research_inputs_sha256",
        "session_id",
        "return_progress",
        "verify_bitexact",
    }
    unknown = set(body) - supported
    if unknown:
        raise ValueError(f"unsupported request fields: {', '.join(sorted(unknown))}")
    if "max_completion_tokens" in body:
        if "max_tokens" in body and body["max_tokens"] != body["max_completion_tokens"]:
            raise ValueError("max_tokens and max_completion_tokens disagree")
        body = {**body, "max_tokens": body["max_completion_tokens"]}
    if "stream_options" in body:
        options = body["stream_options"]
        if (
            not isinstance(options, dict)
            or set(options) - {"include_usage"}
            or not isinstance(options.get("include_usage", False), bool)
        ):
            raise ValueError("stream_options only supports boolean include_usage")
    if chat:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in (
                "system",
                "user",
                "assistant",
                "tool",
            ):
                raise ValueError("invalid text message role")
            content = message.get("content")
            if isinstance(content, list):
                if message["role"] != "user" or not content:
                    raise ValueError(
                        "multimodal content requires a nonempty user message"
                    )
                allowed_parts = {
                    "text", "image_url", "input_image", "input_audio", "input_video"
                }
                if any(
                    not isinstance(part, dict)
                    or not isinstance(part.get("type"), str)
                    or part.get("type") not in allowed_parts
                    for part in content
                ):
                    raise ValueError("invalid multimodal content part")
                for part in content:
                    if part["type"] == "input_video":
                        refuse_video_sampling_options(part)
            elif not isinstance(content, str) and not (
                message["role"] == "assistant" and message.get("tool_calls")
            ):
                raise ValueError("message content must be text or multimodal parts")
            if "tool_calls" in message:
                calls = message["tool_calls"]
                if not isinstance(calls, list) or any(
                    not isinstance(c, dict)
                    or not isinstance(c.get("function"), dict)
                    or not isinstance(c["function"].get("name"), str)
                    for c in calls
                ):
                    raise ValueError("invalid tool call history")
        if any(
            isinstance(message.get("content"), list)
            and any(
                part["type"] == "input_video" and {"fps", "max_frames"} & set(part)
                for part in message["content"]
            )
            for message in messages
        ):
            # Only null sampling options are left (requested ones were refused
            # above); they ask for nothing, so the engine never sees them.
            messages = [
                {
                    **message,
                    "content": [
                        refuse_video_sampling_options(part)
                        if part["type"] == "input_video"
                        else part
                        for part in message["content"]
                    ],
                }
                if isinstance(message.get("content"), list)
                else message
                for message in messages
            ]
            body = {**body, "messages": messages}
        if any(
            message["role"] == "assistant" and message.get("content") is None
            for message in messages
        ):
            # A tool-call turn's null or omitted content means "no content", but chat
            # templates read it as a value: Granite's concatenates it and
            # fails (transformers #45422 drops the key before rendering).  An
            # empty string is the one spelling every served template accepts:
            # Granite's indexes the key and fails on an absent one too, and
            # Qwen, Flash-Next, North, Nemotron, Laguna, LFM and Agnes render
            # "" exactly like a null or absent content.
            body = {
                **body,
                "messages": [
                    {**message, "content": ""}
                    if message["role"] == "assistant" and message.get("content") is None
                    else message
                    for message in messages
                ],
            }
    elif not isinstance(body.get("prompt"), str) or not body["prompt"]:
        raise ValueError("prompt must be a non-empty string")
    # OpenAI clients serialize an unset constraint as an explicit null.  Treat
    # it exactly as an absent field so the engine never sees a constraint key
    # that carries no constraint.
    body = {
        key: value
        for key, value in body.items()
        if key not in {"response_format", "grammar"} or value is not None
    }
    if "response_format" in body or "grammar" in body:
        if "response_format" in body and "grammar" in body:
            raise ValueError("response_format and grammar are mutually exclusive")
        from .structured_output import compile_constraint

        compile_constraint(body.get("response_format"), body.get("grammar"))
        if body.get("response_format") == {"type": "text"}:
            body = {k: v for k, v in body.items() if k != "response_format"}
        elif chat and body.get("enable_thinking") and not structured_thinking:
            # Only chat requests open a reasoning channel; raw completions
            # never do.  The engine repeats this check with the adapter's own
            # thinking semantics and marker declaration; rejecting here avoids
            # reserving a slot and lease.
            raise ValueError("structured output requires thinking to be disabled")
        if body.get("min_tokens", 0):
            # min_tokens masks EOS while a completed grammar admits only EOS;
            # the intersection is an empty support, which is neither
            # constraint honoured.  Grammar completion ends the output.
            raise ValueError("min_tokens cannot be combined with structured output")
    if "logprobs" in body and not isinstance(body["logprobs"], bool):
        # Legacy /v1/completions puts the alternative count in logprobs.
        legacy_count = body["logprobs"]
        if (
            chat
            or not isinstance(legacy_count, int)
            or not 0 <= legacy_count <= MAX_TOP_LOGPROBS
        ):
            raise ValueError(
                f"logprobs must be boolean or a completions integer from 0 to {MAX_TOP_LOGPROBS}"
            )
        if "top_logprobs" in body and body["top_logprobs"] != legacy_count:
            raise ValueError("logprobs and top_logprobs disagree")
        body = {**body, "logprobs": True, "top_logprobs": legacy_count}
    if "verify_bitexact" in body and not isinstance(body["verify_bitexact"], bool):
        raise ValueError("verify_bitexact must be boolean")
    top_logprobs = body.get("top_logprobs", 0)
    if isinstance(top_logprobs, bool) or not isinstance(top_logprobs, int) or not 0 <= top_logprobs <= MAX_TOP_LOGPROBS:
        raise ValueError(f"top_logprobs must be an integer from 0 to {MAX_TOP_LOGPROBS}")
    if "tools" in body:
        tools = body["tools"]
        if not chat or not isinstance(tools, list) or not 1 <= len(tools) <= max_tools:
            raise ValueError(f"tools must contain 1 to {max_tools} function definitions")
        names = set()
        normalized_tools = []
        for tool in tools:
            if (
                not isinstance(tool, dict)
                or tool.get("type") != "function"
                or not isinstance(tool.get("function"), dict)
            ):
                raise ValueError("only function tools are supported")
            function = tool["function"]
            name = function.get("name")
            if (
                not isinstance(name, str)
                or not name
                or name in names
                or len(name) > 128
            ):
                raise ValueError("tool names must be nonempty and unique")
            names.add(name)
            strict = function.get("strict", False)
            if not isinstance(strict, bool) or not isinstance(
                function.get("parameters", {}), dict
            ):
                raise ValueError("tool strict must be boolean and parameters an object")
            if not isinstance(function.get("description", ""), str):
                raise ValueError("tool description must be text")
            declared_parameters = function.get("parameters", {})
            parameters = declared_parameters
            if strict:
                from .runtime.tool_parsers._schema import resolve_local_refs
                from .structured_output import compile_constraint

                # Compile the member order the strict grammar will be built
                # from at admission: the canonical copy below.
                parameters = _canonical_json_order(
                    resolve_local_refs(declared_parameters)
                )
                compile_constraint(
                    {
                        "type": "json_schema",
                        "json_schema": {
                            "name": name,
                            "strict": True,
                            "schema": parameters,
                        },
                    }
                )
            # ``description`` is optional in the OpenAI and Anthropic tool
            # schemas we accept, but a chat template that renders it through
            # ``tojson`` raises on the Jinja Undefined a missing key leaves
            # behind.  Carry the absent optional field as the empty string --
            # the same defaulting ``parameters`` already gets -- so a request
            # that is legal by the schema renders instead of failing.
            # Equivalent schemas sent with members in another order must
            # render the same prompt prefix (omlx #4138).
            normalized_tools.append(
                canonical_tool_definition(
                    {
                        **tool,
                        "function": {
                            "description": "",
                            **function,
                            "parameters": parameters,
                        },
                    }
                )
            )
        body = {**body, "tools": normalized_tools}
    body = normalize_tool_choice(body)
    constrained_tools = body.get("tool_choice") == "required" or isinstance(
        body.get("tool_choice"), dict
    )
    if (
        constrained_tool_grammar
        and constrained_tools
        and "response_format" not in body
        and "grammar" not in body
        and not body.get("min_tokens", 0)
    ):
        def reject_uncheckable_schema(schema):
            if not isinstance(schema, dict):
                return
            if "anyOf" in schema or "oneOf" in schema:
                raise ValueError(
                    "strict tool schema unions are unsupported by the post-generation validator"
                )
            for child in (schema.get("properties") or {}).values():
                reject_uncheckable_schema(child)
            if isinstance(schema.get("items"), dict):
                reject_uncheckable_schema(schema["items"])

        for tool in body.get("tools", ()):
            if tool["function"].get("strict") is True:
                reject_uncheckable_schema(tool["function"].get("parameters", {}))
    if "stop" in body:
        stops = [body["stop"]] if isinstance(body["stop"], str) else body["stop"]
        if (
            not isinstance(stops, list)
            or not 1 <= len(stops) <= max_stops
            or any(not isinstance(s, str) or not 1 <= len(s) <= 256 for s in stops)
        ):
            raise ValueError(
                f"stop must contain 1 to {max_stops} nonempty strings of at most "
                "256 characters"
            )
    samples = body.get("n", 1)
    if isinstance(samples, bool) or not isinstance(samples, int) or not 1 <= samples <= 8:
        raise ValueError("n must be an integer from 1 to 8")
    if samples > 1 and body.get("stream", False):
        raise ValueError("streaming parallel samples are not supported")
    cohort = body.get("batch_cohort")
    if cohort is not None:
        if (
            not isinstance(cohort, dict)
            or set(cohort) != {"id", "size"}
            or not isinstance(cohort.get("id"), str)
            or not 1 <= len(cohort["id"]) <= 128
            or isinstance(cohort.get("size"), bool)
            or not isinstance(cohort.get("size"), int)
            or not 1 <= cohort["size"] <= 64
        ):
            raise ValueError(
                "batch_cohort requires a nonempty id and integer size from 1 to 64"
            )
        if samples != 1:
            raise ValueError("batch_cohort cannot be combined with n greater than 1")
    for key, lower, upper in (
        ("max_tokens", 1, MAX_OUTPUT_TOKENS),
        ("min_tokens", 0, MAX_OUTPUT_TOKENS),
        ("thinking_budget", 0, MAX_OUTPUT_TOKENS),
    ):
        if key not in body:
            continue
        value = body[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not lower <= value <= upper
        ):
            raise ValueError(f"{key} must be an integer between {lower} and {upper}")
    if (
        "max_tokens" in body
        and body.get("min_tokens", 0) > body["max_tokens"]
    ):
        raise ValueError("min_tokens must not exceed max_tokens")
    value = body.get("top_k", 20)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= 2**31 - 1
    ):
        raise ValueError(f"top_k must be an integer between 0 and {2**31 - 1}")
    steer = body.get("thinking_steer_alpha", 0)
    if isinstance(steer, bool) or not isinstance(steer, (int, float)) or not 0 <= steer <= 1:
        raise ValueError("thinking_steer_alpha must be a number between 0 and 1")
    temperature = body.get("temperature", 0.7)
    if (
        isinstance(temperature, (int, float))
        and not isinstance(temperature, bool)
        and 0 < temperature < MIN_POSITIVE_TEMPERATURE
    ):
        raise ValueError(
            f"temperature must be 0 or at least {MIN_POSITIVE_TEMPERATURE:.3g}"
        )
    for key, default, lower, upper in (
        ("temperature", 0.7, 0, 2),
        ("top_p", 0.8, 0, 1),
        ("min_p", 0.0, 0, 1),
        ("repetition_penalty", 1.0, 0.01, 10),
        ("presence_penalty", 0.0, -2, 2),
        ("frequency_penalty", 0.0, -2, 2),
    ):
        value = body.get(key, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not lower <= value <= upper
            or not math.isfinite(value)
        ):
            raise ValueError(f"{key} must be finite and between {lower} and {upper}")
    if "sampling_profile" in body:
        # mlx2 extension: select one of the model's vendor sampling profiles
        # (e.g. "coding"); the engine rejects names the adapter lacks.
        from .sampling_defaults import validate_sampling_profile

        validate_sampling_profile(body["sampling_profile"])
    if "logit_bias" in body:
        biases = body["logit_bias"]
        if not isinstance(biases, dict) or len(biases) > 4096:
            raise ValueError("logit_bias must be an object with at most 4096 tokens")
        for token, value in biases.items():
            if (not isinstance(token, str) or not token.isdecimal()
                or not 0 <= int(token) < 2**31 or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not -100 <= value <= 100 or not math.isfinite(value)):
                raise ValueError("invalid logit_bias token or value")
    for key in (
        "stream", "enable_thinking", "skip_writing_prefix_cache", "return_progress",
        "paged_native_qwen3",
        "paged_native_qwen3_b2",
        "paged_native_hybrid_b2",
        "paged_native_hybrid_packed_prefill",
    ):
        if key in body and not isinstance(body[key], bool):
            raise ValueError(f"{key} must be boolean")
    if body.get("return_progress") and not body.get("stream"):
        # Progress is a pre-output stream event; a non-streaming response has
        # nowhere to put it.  Reject rather than silently ignore.
        raise ValueError("return_progress requires stream")
    if body.get("paged_native_qwen3") and body.get("skip_writing_prefix_cache") is not True:
        raise ValueError("native paged Qwen3 requires skip_writing_prefix_cache=true")
    if body.get("paged_native_qwen3_b2"):
        if body.get("paged_native_hybrid_b2") is True:
            raise ValueError("native B2 capabilities are mutually exclusive")
        if (body.get("paged_native_qwen3") is True or
                body.get("skip_writing_prefix_cache") is not True or
                not isinstance(body.get("batch_cohort"), dict) or
                body["batch_cohort"].get("size") != 2 or
                body.get("max_tokens") not in (2, 32) or body.get("temperature") != 0):
            raise ValueError("native B2 requires a separate cold two-member cohort, a pinned greedy token count and no APCv2 write")
    from .runtime.paged_packed_prefill_serving_profile import validate_request as validate_packed_request
    validate_packed_request(body)
    from .runtime.paged_n20_request import validate_request as validate_n20_request
    validate_n20_request(body)
    if body.get("paged_native_hybrid_b2"):
        maximum = body.get("max_tokens")
        if (body.get("paged_native_qwen3") is True or body.get("paged_native_qwen3_b2") is True or
                body.get("skip_writing_prefix_cache") is not True or
                not isinstance(body.get("batch_cohort"), dict) or
                body["batch_cohort"].get("size") != 2 or
                type(maximum) is not int or not 1 <= maximum <= 256 or body.get("temperature") != 0):
            raise ValueError("hybrid native B2 requires a separate cold two-member greedy cohort and no APCv2 write")
    if "session_id" in body:
        validate_session_id(body["session_id"])
    if body.get("thinking_budget_mode", "state_aware") not in (
        "state_aware",
        "history",
    ):
        raise ValueError("thinking_budget_mode must be state_aware or history")
    if "seed" in body and (
        isinstance(body["seed"], bool)
        or not isinstance(body["seed"], int)
        or not 0 <= body["seed"] < 2**32
    ):
        raise ValueError("seed must be an unsigned 32-bit integer")
    if (
        body.get("stream", False)
        and body.get("tools")
        and not allow_buffered_tool_stream
    ):
        strict = any(
            tool["function"].get("strict") is True for tool in body["tools"]
        )
        if (
            strict
            or body.get("tool_choice") == "required"
            or isinstance(body.get("tool_choice"), dict)
            or body.get("parallel_tool_calls", True) is False
        ):
            raise ValueError(
                "streaming strict, required, named, or single-call tool contracts "
                "are not implemented"
            )
    return body


def normalize_client_options(body):
    """Translate common local-client controls without silently dropping them.

    Qwen's template exposes a thinking toggle, not calibrated effort budgets.
    Named positive effort levels therefore select thinking; explicit toggles
    take precedence over the effort hint. num_ctx is a request ceiling, bounded
    again by the qualified server ceiling in ServingEngine.
    """
    result = dict(body)
    options = result.pop("options", {})
    if not isinstance(options, dict):
        raise ValueError("options must be an object")
    aliases = {
        "num_predict": "max_tokens",
        "num_ctx": "context_limit",
        "thinking_budget": "thinking_budget",
    }
    allowed = {"temperature", "top_p", "top_k", "min_p", "seed", "stop", *aliases}
    if set(options) - allowed:
        raise ValueError("unsupported options: " + ", ".join(sorted(set(options) - allowed)))
    for name, value in options.items():
        target = aliases.get(name, name)
        if target in result and result[target] != value:
            raise ValueError(f"options.{name} and {target} disagree")
        result[target] = value
    if "context_limit" in result:
        limit = result["context_limit"]
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("options.num_ctx must be a positive integer")
    template = result.pop("chat_template_kwargs", {})
    if not isinstance(template, dict) or set(template) - {"enable_thinking"}:
        raise ValueError("chat_template_kwargs only supports enable_thinking")
    toggles = []
    for name, value in (
        ("enable_thinking", result.get("enable_thinking")),
        ("think", result.pop("think", None)),
        ("chat_template_kwargs.enable_thinking", template.get("enable_thinking")),
    ):
        if value is not None:
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean")
            toggles.append(value)
    effort = result.get("reasoning_effort")
    if effort is not None:
        if not isinstance(effort, str) or effort not in {
            "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"
        }:
            raise ValueError("invalid reasoning_effort")
    if toggles:
        if any(value != toggles[0] for value in toggles):
            raise ValueError("thinking toggles disagree")
        result["enable_thinking"] = toggles[0]
    elif effort is not None:
        result["enable_thinking"] = effort != "none"
    return result


# A streamed response silent this long gets a keepalive: Node's fetch (undici)
# drops a connection after 300 s without bytes (FreeToken #572, Splash #208),
# and a long prefill or a buffered tool call sends none.
DEFAULT_SSE_KEEPALIVE_SECONDS = 15.0
SSE_KEEPALIVE = b": keep-alive\n\n"


class ClientGone(RuntimeError):
    """The HTTP client closed its connection while a response was pending."""


def client_disconnected(connection) -> bool:
    """Whether ``connection`` has been closed by the peer (EOF), without blocking.

    A non-streaming request blocks on the engine's event queue and never
    touches the socket until the response is ready, so a client that gave up
    would otherwise keep its lane decoding to ``max_tokens``.
    """
    if connection is None:
        return False
    try:
        readable, _, _ = select.select([connection], [], [], 0)
        if not readable:
            return False
        return connection.recv(1, socket.MSG_PEEK) == b""
    except (OSError, ValueError):
        return True


def wait_event(
    events,
    *,
    connection=None,
    deadline_seconds=1800.0,
    poll_seconds=0.5,
    idle_seconds=None,
    on_idle=None,
    idle_since=None,
):
    """Block for the next engine event, watching the client socket meanwhile.

    ``on_idle`` (a stream's keepalive) runs whenever ``idle_seconds`` have
    passed since ``idle_since()``, the last byte the response wrote -- not
    since the last engine event, which a buffered stream consumes without
    writing.  The overall deadline is unchanged.
    """
    deadline = time.monotonic() + deadline_seconds
    watching = on_idle is not None and bool(idle_seconds) and idle_since is not None
    # ``on_idle`` may decline to write (a request not yet on a lane); the
    # next attempt then waits a full interval instead of retrying at once,
    # which spun a core polling the socket for every queued stream.
    next_attempt = None
    while True:
        now = time.monotonic()
        remaining = deadline - now
        if remaining <= 0:
            raise TimeoutError("generation timed out")
        timeout = min(poll_seconds, remaining)
        if watching:
            due = idle_since() + idle_seconds
            if next_attempt is not None:
                due = max(due, next_attempt)
            if now >= due:
                on_idle()
                next_attempt = now + idle_seconds
                due = max(idle_since() + idle_seconds, next_attempt)
            timeout = max(0.0, min(timeout, due - now))
        try:
            return events.get(timeout=timeout)
        except queue.Empty:
            if client_disconnected(connection):
                raise ClientGone("client disconnected before the response was ready")


class SampleFailed(RuntimeError):
    """A parallel sample ended with an error event; carries its HTTP status."""

    def __init__(self, message, status, mlx2=None, code=None):
        super().__init__(message)
        self.status = status
        self.mlx2 = mlx2
        self.code = code


@contextlib.contextmanager
def wrongly_typed_request_is_invalid():
    """Report a wrongly typed field in a parsed JSON body as a client error.

    Request validation and translation compare client values against sets
    and read nested values as mappings, so a list or object where a string
    was expected raises TypeError or AttributeError instead of ValueError.
    Wrap only validation and translation in this: the same exception types
    raised while generating are server faults, not malformed requests.
    """
    try:
        yield
    except (TypeError, AttributeError) as error:
        raise ValueError(f"request field has the wrong JSON type: {error}") from error


def parse_multipart_form(content_type, raw):
    """Parse one bounded multipart request without the deprecated cgi module."""
    if not isinstance(content_type, str) or not content_type.lower().startswith(
        "multipart/form-data;"
    ):
        raise ValueError("Files API requires multipart/form-data")
    message = BytesParser(policy=email_policy.default).parsebytes(
        b"Content-Type: " + content_type.encode("ascii") + b"\r\n\r\n" + raw
    )
    if not message.is_multipart():
        raise ValueError("invalid multipart form")
    fields = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name or name in fields:
            raise ValueError("multipart fields must have unique names")
        fields[name] = {
            "filename": part.get_filename(),
            "content_type": part.get_content_type(),
            "content": part.get_payload(decode=True) or b"",
        }
    return fields


def _request_json(raw):
    try:
        value = json.loads(raw)
    except RecursionError as error:
        raise ValueError("request JSON is nested too deeply") from error
    if not isinstance(value, dict):
        raise ValueError("request must be a JSON object")
    # A ``\ud800`` escape (or surrogate bytes, which ``json.loads`` decodes
    # with ``surrogatepass``) parses to a lone surrogate no tokenizer can
    # encode; it surfaced as a 500 template failure (omlx #4253).  Only bodies
    # that can hold one pay for the check.
    if isinstance(raw, str):
        raw = raw.encode("utf-8", "surrogatepass")
    elif (encoding := json.detect_encoding(raw)) not in ("utf-8", "utf-8-sig"):
        # ``json.loads`` also accepts UTF-16 and UTF-32, where neither form
        # below is spelled with these bytes: scan the body's UTF-8 form.
        raw = raw.decode(encoding, "surrogatepass").encode("utf-8", "surrogatepass")
    if b"\\ud" in raw or b"\\uD" in raw or b"\xed" in raw:
        try:
            json.dumps(value, ensure_ascii=False).encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError(
                "request text contains an unpaired UTF-16 surrogate"
            ) from error
    return value


def embeddings_payload(
    engine, body, *, admission_class="embeddings", admitted=False
):
    """Validate and execute an adapter-owned embedding capability."""
    if not isinstance(body, dict):
        raise ValueError("embedding request must be an object")
    unknown = set(body) - {"input", "model", "dimensions", "encoding_format", "user"}
    if unknown:
        raise ValueError("unsupported embedding fields: " + ", ".join(sorted(unknown)))
    status = engine.status()
    model = status.get("model")
    if body.get("model", model) != model:
        raise ResourceNotFound("unknown model")
    raw_inputs = body.get("input")
    if isinstance(raw_inputs, str):
        inputs = [raw_inputs]
    elif isinstance(raw_inputs, list) and raw_inputs and all(
        isinstance(item, str) for item in raw_inputs
    ):
        inputs = list(raw_inputs)
    else:
        raise ValueError("embedding input must be text or a nonempty list of text")
    if len(inputs) > 2048 or any(not item for item in inputs):
        raise ValueError("embedding inputs must be nonempty and contain at most 2048 items")
    dimensions = body.get("dimensions")
    if dimensions is not None and (
        isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1
    ):
        raise ValueError("embedding dimensions must be a positive integer")
    encoding = body.get("encoding_format", "float")
    if not isinstance(encoding, str) or encoding not in {"float", "base64"}:
        raise ValueError("encoding_format must be float or base64")
    provider = getattr(engine, "embed", None)
    if not callable(provider):
        raise CapabilityUnavailable(
            "the loaded adapter does not implement an embedding representation"
        )
    provider_options = {"dimensions": dimensions}
    if admitted or admission_class != "embeddings":
        provider_options.update(
            admission_class=admission_class,
            admitted=admitted,
        )
    result = provider(inputs, **provider_options)
    vectors, prompt_tokens = (
        result if isinstance(result, tuple) and len(result) == 2 else (result, 0)
    )
    if not isinstance(vectors, (list, tuple)) or len(vectors) != len(inputs):
        raise RuntimeError("embedding adapter returned the wrong number of vectors")
    data = []
    for index, vector in enumerate(vectors):
        if not isinstance(vector, (list, tuple)) or not vector:
            raise RuntimeError("embedding adapter returned an invalid vector")
        values = [float(value) for value in vector]
        if any(not math.isfinite(value) for value in values):
            raise RuntimeError("embedding adapter returned a non-finite vector")
        if dimensions is not None and len(values) != dimensions:
            raise RuntimeError("embedding adapter did not honor dimensions")
        embedding = (
            values
            if encoding == "float"
            else base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode()
        )
        data.append({"object": "embedding", "embedding": embedding, "index": index})
    prompt_tokens = int(prompt_tokens)
    if prompt_tokens < 0:
        raise RuntimeError("embedding adapter returned invalid usage")
    return {
        "object": "list",
        "data": data,
        "model": model,
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    }


def rerank_payload(engine, body):
    """Validate and execute the common Cohere/Jina-shaped rerank contract."""
    if not isinstance(body, dict):
        raise ValueError("rerank request must be an object")
    unknown = set(body) - {"model", "query", "documents", "top_n", "return_documents"}
    if unknown:
        raise ValueError("unsupported rerank fields: " + ", ".join(sorted(unknown)))
    status = engine.status()
    model = status.get("model")
    if body.get("model", model) != model:
        raise ResourceNotFound("unknown model")
    query, documents = body.get("query"), body.get("documents")
    if not isinstance(query, str) or not query:
        raise ValueError("rerank query must be nonempty text")
    if not isinstance(documents, list) or not 1 <= len(documents) <= 1000:
        raise ValueError("rerank documents must contain 1 to 1000 text items")
    if any(not isinstance(item, str) or not item for item in documents):
        raise ValueError("rerank documents must be nonempty text")
    top_n = body.get("top_n", len(documents))
    if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= len(documents):
        raise ValueError("top_n must fit the document count")
    if not isinstance(body.get("return_documents", False), bool):
        raise ValueError("return_documents must be boolean")
    provider = getattr(engine, "rerank", None)
    if not callable(provider):
        raise CapabilityUnavailable(
            "the loaded adapter does not implement a reranking score"
        )
    scores = provider(query, documents)
    if not isinstance(scores, (list, tuple)) or len(scores) != len(documents):
        raise RuntimeError("rerank adapter returned the wrong number of scores")
    ranked = []
    for index, score in enumerate(scores):
        score = float(score)
        if not math.isfinite(score):
            raise RuntimeError("rerank adapter returned a non-finite score")
        item = {"index": index, "relevance_score": score}
        if body.get("return_documents", False):
            item["document"] = {"text": documents[index]}
        ranked.append(item)
    ranked.sort(key=lambda item: (-item["relevance_score"], item["index"]))
    return {
        "id": "rerank_" + uuid.uuid4().hex,
        "model": model,
        "results": ranked[:top_n],
    }


def _generation_model(status, body):
    """Resolve the base or an advertised concurrent LoRA generation model."""
    base = status.get("model")
    selected = body.get("model", base)
    if selected == base:
        return base
    registered = (status.get("multi_lora") or {}).get("registered") or ()
    if isinstance(selected, str) and selected in registered:
        return selected
    raise ResourceNotFound("unknown model")


def prompt_render_payload(engine, body, path):
    """Answer ``POST /tokenize`` or ``POST /apply-template`` without generating.

    The body is a Chat (``messages``) or Completions (``prompt``) request and is
    validated exactly like one, then rendered through the same adapter call
    admission uses.  Media parts are rejected: their prompt tokens depend on
    encoder preparation that these read-only endpoints do not run.
    """
    if not isinstance(body, dict):
        raise ValueError("request must be a JSON object")
    status = engine.status()
    _generation_model(status, body)
    with wrongly_typed_request_is_invalid():
        request = validate_request(
            body,
            "messages" in body,
            structured_thinking=bool(
                (status.get("structured_output") or {}).get("thinking_deferral")
            ),
            allow_buffered_tool_stream=True,
            constrained_tool_grammar=bool(
                (status.get("settings") or {}).get("constrained_tool_grammar")
            ),
        )
    if any(
        isinstance(message.get("content"), list)
        for message in request.get("messages", ())
    ):
        raise ValueError(f"{path} does not render image, video, or audio content")
    if path == "/tokenize":
        tokens = engine.render_prompt(request)
        return {
            "tokens": tokens,
            "count": len(tokens),
            "max_model_len": getattr(engine, "max_context", None),
        }
    prompt = engine.apply_template(request)
    if prompt is None:
        raise CapabilityUnavailable(
            "the loaded adapter does not expose a text prompt renderer"
        )
    return {"prompt": prompt}


def tensorfold_owned_client_disconnected(connection):
    """Observe an HTTP peer FIN without consuming a pipelined request byte."""

    try:
        if not select.select([connection], [], [], 0)[0]:
            return False
        return connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b""
    except BlockingIOError:
        return False
    except (OSError, ValueError):
        return True


def tensorfold_owned_completion_payload(engine, router, body, path, *, route="auto",
                                        disconnect_probe=None):
    """Explicit greedy text bridge into a cache-isolated native worker."""

    if path not in {"/v1/completions", "/v1/chat/completions"}:
        raise ValueError("TensorFold-owned header requires a completion endpoint")
    if route not in {"auto", "serial"}:
        raise ValueError("TensorFold-owned completion route must be auto or serial")
    if not isinstance(body, dict):
        raise ValueError("request must be a JSON object")
    chat = path == "/v1/chat/completions"
    allowed = {"model", "messages" if chat else "prompt", "temperature",
               "top_p", "top_k",
               "max_tokens", "max_completion_tokens", "n", "stream"}
    if chat:
        allowed.add("enable_thinking")
    unknown = set(body) - allowed
    if unknown:
        raise ValueError("TensorFold-owned greedy route does not support: "
                         + ", ".join(sorted(unknown)))
    if body.get("temperature", 0) != 0 or body.get("n", 1) != 1 or body.get("stream", False):
        raise ValueError("TensorFold-owned completion requires temperature=0, n=1, stream=false")
    if body.get("top_p", 1) != 1 or body.get("top_k", 0) != 0:
        raise ValueError("TensorFold-owned greedy route only accepts top_p=1 and top_k=0")
    if chat and body.get("enable_thinking", False) is not False:
        raise ValueError("TensorFold-owned chat requires enable_thinking=false")
    status = engine.status()
    model = _generation_model(status, body)
    if model != status.get("model"):
        raise ValueError("TensorFold-owned completion requires the base model")
    request = validate_request({**body, "enable_thinking": False} if chat else body, chat)
    budget = request.get("max_tokens", 64)
    if not 1 <= budget <= 192:
        raise ValueError("TensorFold-owned completion supports max_tokens 1..192")
    prompt_ids = engine.render_prompt(request)
    with engine.prompt_lock:
        tokenizer = engine.adapter.tokenizer
        eos_ids = sorted(int(token) for token in tokenizer.eos_token_ids)
    generated = router.generate(prompt_ids, budget, route=route, eos_ids=eos_ids,
                                timeout=840, allow_long=True,
                                disconnect_probe=disconnect_probe)
    token_ids = generated["token_ids"]
    with engine.prompt_lock:
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
    receipt = {**generated["route_receipt"], "selected_by": "explicit_standard_header",
               "output_token_ids_sha256": hashlib.sha256(json.dumps(
                   token_ids, separators=(",", ":")).encode()).hexdigest()}
    choice = {"index": 0, "finish_reason": generated["finish_reason"]}
    if chat:
        choice["message"] = {"role": "assistant", "content": text}
    else:
        choice["text"] = text
    return {
        "id": ("chatcmpl_" if chat else "cmpl_") + uuid.uuid4().hex,
        "object": "chat.completion" if chat else "text_completion",
        "created": int(time.time()), "model": model, "choices": [choice],
        "usage": {"prompt_tokens": len(prompt_ids),
                  "completion_tokens": len(token_ids),
                  "total_tokens": len(prompt_ids) + len(token_ids)},
        "mlx2": receipt,
    }


def lora_control_payload(engine, body, *, load):
    """Execute vLLM-compatible dynamic LoRA control through an engine hook."""
    if not isinstance(body, dict):
        raise ValueError("LoRA request must be an object")
    allowed = {"lora_name", "lora_path", "base_model_name"} if load else {
        "lora_name",
        "lora_int_id",
    }
    unknown = set(body) - allowed
    if unknown:
        raise ValueError("unsupported LoRA fields: " + ", ".join(sorted(unknown)))
    name = body.get("lora_name")
    if not isinstance(name, str) or not name or len(name) > 128:
        raise ValueError("lora_name must contain 1 to 128 characters")
    method = getattr(
        engine, "load_lora_adapter" if load else "unload_lora_adapter", None
    )
    if not callable(method):
        raise CapabilityUnavailable(
            "the loaded engine does not implement revision-bound LoRA lifecycle"
        )
    if load:
        path = body.get("lora_path")
        if not isinstance(path, str) or not path:
            raise ValueError("lora_path must be nonempty text")
        if body.get("base_model_name") is not None and not isinstance(
            body["base_model_name"], str
        ):
            raise ValueError("base_model_name must be text")
        result = method(
            name,
            path,
            base_model_name=body.get("base_model_name"),
        )
    else:
        if body.get("lora_int_id") is not None and (
            isinstance(body["lora_int_id"], bool)
            or not isinstance(body["lora_int_id"], int)
        ):
            raise ValueError("lora_int_id must be an integer")
        result = method(name, lora_int_id=body.get("lora_int_id"))
    if not isinstance(result, dict):
        raise RuntimeError("LoRA lifecycle hook must return a receipt object")
    return {"id": name, "object": "lora.adapter", **result}


def collect_nonstream_job(job, body, *, chat, responses=False):
    """Collect one already-submitted sample without writing an HTTP response.

    ``responses`` keeps only the output-text tokens' logprobs, the ones a
    Responses message carries (see ``ResponsesTextLogprobs``).
    """
    parts, reasoning, calls, probabilities = [], [], [], []
    text_logprobs = ResponsesTextLogprobs()
    while True:
        try:
            event = job.events.get(timeout=1800)
        except queue.Empty as exc:
            raise TimeoutError("generation timed out") from exc
        if "error" in event:
            raise SampleFailed(
                event["error"],
                event.get("status", 503),
                event.get("mlx2"),
                event.get("code"),
            )
        if "logprob" in event:
            probabilities.append(event["logprob"])
            text_logprobs.logprob(event["logprob"])
        if "text" in event or "delta" in event:
            delta = event.get("delta", {"content": event.get("text", "")})
            text_logprobs.delta(delta)
            parts.append(delta.get("content", ""))
            reasoning.append(delta.get("reasoning_content", ""))
            calls.extend(delta.get("tool_calls", []))
        if "finish_reason" not in event:
            continue
        text_logprobs.finish()
        choice = {"finish_reason": event["finish_reason"]}
        if wants_logprobs(body):
            choice["logprobs"] = {
                "content": text_logprobs.text if responses else probabilities
            }
        message = {"role": "assistant", "content": "".join(parts)}
        if any(reasoning):
            message["reasoning_content"] = "".join(reasoning)
        if calls:
            message["tool_calls"] = [
                {key: value for key, value in call.items() if key != "index"}
                for call in calls
            ]
        enforce_tool_contract(
            body,
            message.get("tool_calls", []),
            finish_reason=event["finish_reason"],
        )
        choice.update({"message": message} if chat else {"text": "".join(parts)})
        return choice, {
            "prompt_tokens": job.prompt_tokens,
            "completion_tokens": job.completion_tokens,
            "total_tokens": job.prompt_tokens + job.completion_tokens,
            "completion_tokens_details": {
                "reasoning_tokens": int(getattr(job, "reasoning_tokens", 0) or 0)
            },
        }, event["receipt"]


def score_choice_tokens_via_engine(engine, prompt, token_ids, *, tenant_id):
    """Score a bounded choice set through the ordinary serving lifecycle."""
    if not isinstance(token_ids, dict) or not 2 <= len(token_ids) <= MAX_CLASSIFIER_LABELS:
        raise ValueError(f"classifier requires 2..{MAX_CLASSIFIER_LABELS} labeled token ids")
    request = {
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "enable_thinking": False,
        "logprobs": True,
        "top_logprobs": len(token_ids),
        "logit_bias": {str(token): 100 for token in token_ids.values()},
        "skip_writing_prefix_cache": True,
    }
    job = engine.submit(request, tenant_id=tenant_id)
    try:
        choice, _usage, _receipt = collect_nonstream_job(job, request, chat=False)
    finally:
        job.cancelled.set()
    content = choice.get("logprobs", {}).get("content", [])
    if not content:
        raise RuntimeError("classifier serving job returned no target logprobs")
    row = content[0]
    candidates = [row, *row.get("top_logprobs", ())]
    by_id = {int(item["id"]): float(item["logprob"]) for item in candidates}
    missing = set(token_ids.values()) - set(by_id)
    if missing:
        raise RuntimeError(f"classifier logprobs omitted choice tokens: {sorted(missing)}")
    return {label: by_id[token] for label, token in token_ids.items()}


def collect_parallel_samples(jobs, body, *, chat, connection=None):
    """Drain every sample's event queue concurrently.

    Samples decode in lockstep once the siblings join the leader's batch, and
    each ``Job.events`` queue is bounded.  Draining them one after another let
    every sibling fill its queue while the leader was being read, at which
    point the engine cancels it as a slow consumer.  One collector thread per
    sample keeps every queue drained; the first failure wins and the caller
    cancels the rest.
    """
    results = [None] * len(jobs)
    errors = [None] * len(jobs)

    def collect(index, job):
        try:
            results[index] = collect_nonstream_job(job, body, chat=chat)
        except BaseException as exc:  # propagated to the HTTP thread below
            errors[index] = exc
            # The response is already lost; stop paying for the other samples.
            for other in jobs:
                other.cancelled.set()

    threads = [
        threading.Thread(
            target=collect, args=(index, job), name=f"mlx2-sample-{index}", daemon=True
        )
        for index, job in enumerate(jobs)
    ]
    for thread in threads:
        thread.start()
    while any(thread.is_alive() for thread in threads):
        for thread in threads:
            thread.join(timeout=0.5)
        if client_disconnected(connection):
            for job in jobs:
                job.cancelled.set()
            for thread in threads:
                thread.join(timeout=5)
            raise ClientGone("client disconnected before the response was ready")
    failures = [error for error in errors if error is not None]
    if failures:
        # Report the cause, not a consequence: the leader's own failure or a
        # sibling's, ahead of the cancellations and fanout aborts it induced.
        induced = ("cancelled", "parallel prefill leader ended before APCv2 fanout")
        causes = [error for error in failures if str(error) not in induced]
        raise (causes or failures)[0]
    return results


def tenant_batching_status(engine, tenant_id):
    """``/v1/status/batching`` for one authenticated tenant.

    The snapshot lists requests, their tenants and events, so a tenant sees
    only its own; the aggregates stay.  An engine that cannot scope its
    snapshot shows none of the per-request rows.
    """
    import inspect

    method = engine.batching_status
    try:
        scoped = "tenant_id" in inspect.signature(method).parameters
    except (TypeError, ValueError):
        scoped = False
    if scoped:
        return method(tenant_id=tenant_id)
    status = dict(method())
    for key in ("active_requests", "completed_requests", "events"):
        if key in status:
            status[key] = []
    fairness = status.get("fairness")
    if isinstance(fairness, dict) and "tenant_token_rates" in fairness:
        status["fairness"] = {**fairness, "tenant_token_rates": {}}
    return status


class _HeaderPhaseReader:
    """``rfile`` wrapper that bounds the whole request-line and header phase.

    ``BufferedReader.readline`` restarts the socket timeout after every
    received byte, so a client trickling one header byte per timeout holds a
    handler thread (and one of the connection slots) indefinitely.  ``peek``
    performs at most one raw receive, which lets an absolute deadline be
    checked between receives.  The deadline starts at the first byte of a
    request so idle keep-alive time is not charged to it.
    """

    def __init__(self, rfile, connection, idle_timeout, deadline_seconds):
        self._rfile = rfile
        self._connection = connection
        self._idle_timeout = idle_timeout
        self._deadline_seconds = deadline_seconds
        self.deadline = None
        self.expired = False

    def begin_request(self):
        self.deadline = None
        self.expired = False

    def readline(self, limit=-1):
        if limit is not None and limit == 0:
            return b""
        chunks = []
        size = 0
        while True:
            if self.deadline is None:
                self._connection.settimeout(self._idle_timeout)
            else:
                budget = self.deadline - time.monotonic()
                if budget <= 0:
                    self.expired = True
                    raise TimeoutError("request headers were not received in time")
                self._connection.settimeout(min(self._idle_timeout, budget))
            try:
                buffered = self._rfile.peek(1)
            except TimeoutError:
                if self.deadline is not None:
                    self.expired = True
                raise
            if not buffered:
                break
            if self.deadline is None:
                self.deadline = time.monotonic() + float(self._deadline_seconds())
            end = buffered.find(b"\n")
            take = len(buffered) if end < 0 else end + 1
            if limit is not None and limit >= 0:
                take = min(take, limit - size)
            # Served from the buffer ``peek`` filled: no further receive.
            chunk = self._rfile.read(take)
            chunks.append(chunk)
            size += len(chunk)
            if chunk.endswith(b"\n") or (limit is not None and 0 <= limit <= size):
                break
        return b"".join(chunks)

    def __getattr__(self, name):
        return getattr(self._rfile, name)


def handler_for(
    engine,
    *,
    max_request_bytes: int | None = None,
    request_tracer=None,
    response_store=None,
    file_store=None,
    batch_manager=None,
    api_state_dir=None,
    tool_backend=None,
    admin_token=None,
    http_security=None,
    tenant_authenticator=None,
    agent_compat=None,
    semantic_middleware=None,
    tensorfold_owned_router=None,
    sse_keepalive_seconds: float | None = DEFAULT_SSE_KEEPALIVE_SECONDS,
):
    if (
        tenant_authenticator is not None
        and getattr(http_security, "api_key", None) is not None
    ):
        raise ValueError(TENANT_AUTH_GATE_KEY_CONFLICT)
    compat_policy = AgentCompatPolicy.coerce(agent_compat)
    engine.agent_compat = compat_policy
    if tenant_authenticator is not None:
        engine.tenant_authenticator = tenant_authenticator
    if request_tracer is not None:
        engine.request_tracer = request_tracer
    # Preserve the standalone/test embedding contract: callers that do not
    # configure a transport limit retain the historical 2 MiB bound. The CLI
    # always resolves and passes its context-derived value explicitly.
    body_limit = request_body_limit(
        1,
        MIN_REQUEST_BODY_BYTES
        if max_request_bytes is None
        else max_request_bytes,
    )
    state_root = (
        Path(api_state_dir).expanduser().resolve()
        if api_state_dir is not None
        else None
    )
    response_store = response_store or ResponseStore(
        root=state_root / "responses" if state_root is not None else None
    )
    file_store = file_store or FileStore(
        root=state_root / "files" if state_root is not None else None
    )
    engine.media_file_loader = lambda tenant_id, file_id: file_store.content(
        tenant_id, file_id
    )
    if semantic_middleware is not None:
        token_resolver = getattr(engine.adapter, "classifier_token_ids", None)
        if callable(token_resolver):
            semantic_middleware.configure_classifier(
                token_resolver(("store", "defer", "reject"))
            )

    def ensure_semantic_classifier():
        if (
            semantic_middleware is None
            or semantic_middleware.classifier_token_ids is not None
        ):
            return
        token_resolver = getattr(engine.adapter, "classifier_token_ids", None)
        if callable(token_resolver):
            semantic_middleware.configure_classifier(
                token_resolver(("store", "defer", "reject"))
            )

    def tenant_file_text(tenant_id, part):
        unknown = set(part) - {"type", "file_id", "file_data", "filename"}
        if unknown:
            raise ValueError(
                "unsupported input_file fields: " + ", ".join(sorted(unknown))
            )
        if "file_id" in part and "file_data" in part:
            raise ValueError("input_file accepts file_id or file_data, not both")
        if "file_id" in part:
            content, content_type, filename = file_store.content(
                tenant_id, part["file_id"]
            )
        elif "file_data" in part:
            encoded = part["file_data"]
            if not isinstance(encoded, str) or not encoded:
                raise ValueError("input_file file_data must be nonempty base64 text")
            if encoded.startswith("data:"):
                header, separator, encoded = encoded.partition(",")
                if not separator or ";base64" not in header:
                    raise ValueError("input_file data URL must use base64")
                content_type = header[5:].split(";", 1)[0]
            else:
                content_type = "text/plain"
            try:
                content = base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError) as error:
                raise ValueError("input_file file_data is not valid base64") from error
            filename = part.get("filename", "inline.txt")
        else:
            raise ValueError("input_file requires file_id or file_data")
        content_type = content_type.split(";", 1)[0].lower()
        if content_type not in INERT_FILE_CONTENT_TYPES:
            raise ValueError("mlx2 input_file supports UTF-8 text files only")
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("input_file content must be valid UTF-8") from error
        if not text:
            raise ValueError("input_file content must not be empty")
        return f"\n<file name={json.dumps(filename)}>\n{text}\n</file>\n"

    def check_compat_chain(tenant_id, previous_id, agent_compat):
        """A conversation must keep one agent-compat mode across its chain."""
        stored_enabled = response_store.agent_compat_enabled(tenant_id, previous_id)
        if stored_enabled is not None and stored_enabled != agent_compat.enabled:
            engine_counts = getattr(engine, "counts", None)
            if engine_counts is not None:
                engine_counts["agent_compat_mode_conflicts"] += 1
            raise AgentCompatError(
                f"previous_response_id {previous_id!r} was created with "
                f"agent-compat {'on' if stored_enabled else 'off'}, but this "
                f"request resolved {agent_compat.describe()}; a conversation "
                "must keep one agent-compat mode (set X-MLX2-Agent-Compat)"
            )

    def prepare_responses_request(raw_body, tenant_id, agent_compat):
        request, options = responses_to_chat_request(
            raw_body,
            file_resolver=lambda part: tenant_file_text(tenant_id, part),
            tool_backend=tool_backend,
            signer=getattr(engine, "reasoning_signer", None),
            tenant_id=tenant_id,
            model=raw_body.get("model") or engine.status().get("model"),
            agent_compat=agent_compat,
            counts=getattr(engine, "counts", None),
        )
        rejections = options.get("reasoning_signature_rejections", 0)
        if rejections:
            engine_counts = getattr(engine, "counts", None)
            if engine_counts is not None:
                engine_counts["reasoning_signature_rejections"] += rejections
        previous_context = []
        if options["previous_response_id"] is not None:
            previous_context = response_store.context(
                tenant_id, options["previous_response_id"]
            )
            # responses_to_chat_request prepends a system message only for
            # non-null instructions; an explicit null adds none to split off.
            instruction = (
                request["messages"][:1]
                if raw_body.get("instructions") is not None
                else []
            )
            current = request["messages"][len(instruction) :]
            request["messages"] = instruction + previous_context + current
        if options["previous_response_id"] is not None:
            check_compat_chain(
                tenant_id, options["previous_response_id"], agent_compat
            )
        # Every Responses request renders one leading system message:
        # ``instructions`` first, then the leading developer/system items in
        # order.  Templates such as Qwen3.5/3.6 reject a system message
        # anywhere but first, so a later developer/system item folds into the
        # adjacent user turn.  Only compat-mode folds feed the compat counter.
        request["messages"] = fold_system_messages(
            request["messages"],
            getattr(engine, "counts", None) if agent_compat.enabled else None,
        )
        options["previous_context"] = previous_context
        return request, options

    def store_response(tenant_id, options, payload, message):
        if not options["store"]:
            return
        assistant = {
            key: deepcopy(message[key])
            for key in (
                "role",
                "content",
                "tool_calls",
                *(
                    ("reasoning_content",)
                    if "reasoning.encrypted_content" in options.get("include", ())
                    else ()
                ),
            )
            if key in message
        }
        context = (
            options["previous_context"]
            + options["context_messages"]
            + options.get("hosted_messages", [])
            + [assistant]
        )
        response_store.put(tenant_id, payload, context)

    def _batch_execute(endpoint, raw_body, tenant_id):
        if not isinstance(raw_body, dict):
            raise ValueError("batch row body must be an object")
        if raw_body.get("stream"):
            raise ValueError("batch rows cannot stream")
        if endpoint == "/v1/embeddings":
            return 200, embeddings_payload(
                engine,
                raw_body,
                admission_class="batch",
                admitted=True,
            )
        responses_api = endpoint == "/v1/responses"
        options = None
        body = raw_body
        batch_compat = compat_policy.resolve(
            {}, tenant_id, endpoint, getattr(engine, "counts", None)
        )
        if responses_api:
            body, options = prepare_responses_request(raw_body, tenant_id, batch_compat)
            if options.get("tool_executors"):
                # The hosted tool loop lives in the interactive handler; a
                # batch row would return the internal call unexecuted and
                # count as completed, so refuse the row instead.
                raise CapabilityUnavailable(
                    "batch rows cannot run hosted (MCP) tools; send this "
                    "request to /v1/responses directly"
                )
        chat = endpoint != "/v1/completions"
        status = engine.status()
        body = validate_request(
            body,
            chat,
            structured_thinking=bool(
                (status.get("structured_output") or {}).get("thinking_deferral")
            ),
            constrained_tool_grammar=bool(
                (status.get("settings") or {}).get("constrained_tool_grammar")
            ),
            max_tools=128 if responses_api and batch_compat.enabled else 64,
        )
        model = _generation_model(status, body)
        if body.get("n", 1) != 1:
            raise ValueError("batch rows currently require n=1")
        submit_kwargs = {"tenant_id": tenant_id}
        if hasattr(engine, "_ensure_admission"):
            submit_kwargs.update(admission_class="batch", admitted=True)
        job = engine.submit(body, **submit_kwargs)
        try:
            choice, usage, receipt = collect_nonstream_job(
                job, body, chat=chat, responses=responses_api
            )
        finally:
            job.cancelled.set()
        if responses_api and batch_compat.notable:
            receipt = {**(receipt or {}), "agent_compat": batch_compat.receipt()}
        choice["index"] = 0
        if responses_api:
            payload = responses_payload(
                job=job,
                model=model,
                choice=choice,
                usage=usage,
                receipt=receipt,
                metadata=options["metadata"],
                store=options["store"],
                previous_response_id=options["previous_response_id"],
                signer=getattr(engine, "reasoning_signer", None),
                tenant_id=tenant_id,
                include=options.get("include", ()),
                tool_choice=options.get("tool_choice"),
                agent_compat=batch_compat,
                compat_tool_map=options.get("agent_compat"),
                counts=getattr(engine, "counts", None),
            )
            store_response(tenant_id, options, payload, choice["message"])
            return 200, payload
        return 200, {
            "id": job.id,
            "object": "chat.completion" if chat else "text_completion",
            "created": int(job.created),
            "model": model,
            "choices": [choice],
            "usage": usage,
            "mlx2": receipt,
        }

    batches = batch_manager or BatchManager(
        file_store,
        _batch_execute,
        root=state_root / "batches" if state_root is not None else None,
        overload_errors=(Overloaded,),
    )
    engine.api_resources = {
        "responses": response_store,
        "files": file_store,
        "batches": batches,
    }
    if tool_backend is not None:
        engine.api_resources["mcp_tools"] = tool_backend

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        # A per-recv socket timeout does not bound a body that trickles one
        # byte every few seconds; each such connection would hold a handler
        # thread and, at 32 of them, the whole connection semaphore.  Bound
        # the entire body read instead.
        REQUEST_BODY_DEADLINE_SECONDS = 30.0
        # The request line and headers get the same absolute bound; the
        # per-receive timeout alone let a header trickle hold a slot forever.
        REQUEST_HEADER_DEADLINE_SECONDS = 30.0

        def setup(self):
            super().setup()
            self.connection.settimeout(30)
            self.rfile = _HeaderPhaseReader(
                self.rfile,
                self.connection,
                30,
                lambda: self.REQUEST_HEADER_DEADLINE_SECONDS,
            )
            self._http_started_at = None
            self._http_recorded = False
            self._request_trace = None
            self._tenant_id = "default"
            self._tenant_auth_method = None
            self._body_consumed = False
            self._connection_header_sent = False
            self._interim_response = False
            self._request_body_remaining = 0
            self._request_parsed = False

        def handle_one_request(self):
            self.rfile.begin_request()
            self._request_parsed = False
            super().handle_one_request()
            if self.rfile.expired:
                # The stdlib swallowed the timeout; tell the client why the
                # connection is going away.
                if not self._request_parsed:
                    # The request line itself missed the deadline, so
                    # parse_request never ran: nothing of this request is
                    # known, and a keep-alive connection still holds the
                    # previous request's path, metric and auth receipt.
                    self._begin_request_state()
                    self.requestline = ""
                    self.command = None
                    self.path = ""
                    self.request_version = self.protocol_version
                    self.headers = None
                self.close_connection = True
                try:
                    self.api_error(
                        408,
                        "request headers were not received in time",
                        headers={"Connection": "close"},
                    )
                except OSError:
                    pass

        def _begin_request_state(self):
            # BaseHTTPRequestHandler reuses this instance for every request on
            # a keep-alive connection. Start after the request line arrives so
            # idle time between requests is not charged to the next request.
            metrics = getattr(engine, "http_metrics", None)
            self._http_started_at = (
                metrics.started() if metrics is not None else time.monotonic()
            )
            self._http_recorded = False
            self._request_trace = None
            self._tenant_id = "default"
            self._tenant_auth_method = None
            self._body_consumed = False
            self._connection_header_sent = False
            self._request_body_remaining = 0

        def parse_request(self):
            self._request_parsed = True
            self._begin_request_state()
            if not super().parse_request():
                return False
            try:
                if self.headers.get_all("Transfer-Encoding"):
                    raise ValueError("Transfer-Encoding is unsupported")
                self._request_body_remaining = self._content_length()
            except ValueError as error:
                self.close_connection = True
                self.api_error(
                    400,
                    str(error),
                    anthropic=self.path.split("?", 1)[0].startswith("/v1/messages"),
                    headers={"Connection": "close"},
                )
                return False
            # The header phase may have left a short remaining budget on the
            # socket; body reads set their own.
            self.connection.settimeout(30)
            return True

        def send_header(self, keyword, value):
            if keyword.lower() == "connection":
                self._connection_header_sent = True
            super().send_header(keyword, value)

        def handle_expect_100(self):
            # An interim 100 Continue invites the body; it is not a reply
            # sent in place of reading it.
            self._interim_response = True
            try:
                return super().handle_expect_100()
            finally:
                self._interim_response = False

        def _request_body_unread(self):
            if self._body_consumed:
                return False
            headers = getattr(self, "headers", None)
            if headers is None:
                return False
            if headers.get("Transfer-Encoding") is not None:
                return True
            return any(
                value.strip() != "0" for value in headers.get_all("Content-Length", ())
            )

        def _content_length(self):
            """The declared body size, parsed as strictly as HTTP defines it.

            ``int()`` also accepts a sign, underscores, surrounding spaces
            and non-ASCII digits, and a proxy may frame the same bytes
            differently; refuse anything but one decimal byte count.
            """
            values = self.headers.get_all("Content-Length", ("0",))
            value = values[0].strip(" \t") if len(values) == 1 else ""
            if not (value.isascii() and value.isdigit()):
                raise ValueError("Content-Length must be one decimal byte count")
            return int(value)

        def end_headers(self):
            # Tenant-auth receipt on every response, streams included.  The
            # method only: never the tenant, key id or credential.
            if self._tenant_auth_method is not None:
                self.send_header("X-MLX2-Tenant-Auth", self._tenant_auth_method)
            if not self._interim_response and self._request_body_unread():
                # Answered before the declared body was read: its bytes would
                # be parsed as the next request on this connection, which a
                # connection-reusing proxy turns into request smuggling.
                if not self._connection_header_sent:
                    self.send_header("Connection", "close")
                self.close_connection = True
            super().end_headers()

        def _authenticate_tenant(self, path):
            """Bind ``self._tenant_id`` for this request; False = refused."""
            self._tenant_auth_method = None
            if tenant_authenticator is None:
                self._tenant_id = self.headers.get("X-Tenant-ID", "default")
                return True
            self._tenant_id = None
            if path in TENANT_AUTH_OPEN_PATHS or (
                path in TENANT_AUTH_LOOPBACK_OPEN_PATHS
                and is_loopback_address(self.client_address[0])
            ):
                return True
            from .tenant_auth import TenantAuthError

            try:
                principal = tenant_authenticator.authenticate(
                    self.headers,
                    required_scope="adapters"
                    if path in TENANT_AUTH_ADAPTER_PATHS
                    else "inference",
                )
            except TenantAuthError as failure:
                logging.getLogger("mlx2.tenant_auth").warning(
                    "tenant auth refused %s %s from %s: %s",
                    self.command,
                    self._metric_route(),
                    self.client_address[0],
                    failure.reason,
                )
                # The body (if any) is unread; do not reuse the connection.
                self.close_connection = True
                self.api_error(
                    failure.status,
                    failure.public_message,
                    anthropic=path.startswith("/v1/messages"),
                    headers={"WWW-Authenticate": "Bearer"}
                    if failure.status == 401
                    else None,
                )
                return False
            self._tenant_id = principal.tenant
            self._tenant_auth_method = principal.method
            return True

        def _ensure_trace(self):
            if self._request_trace is not None or request_tracer is None:
                return
            route = self._metric_route()
            self._request_trace = request_tracer.start(
                f"{self.command} {route}",
                dict(self.headers.items()),
                {
                    "http.request.method": self.command,
                    "http.route": route,
                },
            )

        def _metric_route(self):
            return http_metric_route(self.path)

        def _session_route(self):
            parsed = urlsplit(self.path)
            prefix = "/v1/apc/sessions"
            if parsed.path == prefix:
                return parsed, None, None
            if not parsed.path.startswith(prefix + "/"):
                return None
            parts = parsed.path[len(prefix) + 1 :].split("/")
            if len(parts) not in {1, 2}:
                return None
            session_id = validate_session_id(unquote(parts[0]))
            return parsed, session_id, parts[1] if len(parts) == 2 else None

        def _sessions_available(self):
            return bool(
                getattr(engine, "apc_sessions_enabled", False)
                and hasattr(engine, "apc_session_state")
            )

        def _authorize_request(self):
            """Host/Origin/API-key gate; ``http_security=None`` disables it."""
            if http_security is None:
                return True
            path = self.path.split("?", 1)[0]
            try:
                authorize_http_request(
                    http_security,
                    self.command,
                    # Raw target: only the exact ``/health`` route is exempt.
                    self.path,
                    self.headers,
                    local_address=self.connection.getsockname()[0],
                    admin_token_configured=admin_token is not None,
                )
            except GateRejection as rejection:
                # The body (if any) is unread; close rather than parse it as
                # the next request on this connection.
                self.send_json(
                    rejection.status,
                    rejection.payload(anthropic=path.startswith("/v1/messages")),
                    headers={"Connection": "close", **(rejection.headers or {})},
                )
                return False
            return True

        def _authorize_admin(self):
            failure = authorize_admin(
                self.client_address[0],
                self.headers.get("Authorization"),
                admin_token,
            )
            if failure is None:
                return True
            status, message = failure
            headers = {"WWW-Authenticate": "Bearer"} if status == 401 else None
            self.send_json(
                status,
                {"error": {"message": message, "type": "server_error"}},
                headers=headers,
            )
            return False

        def _read_json_body(self, *, allow_empty=False, max_bytes=None):
            size = self._content_length()
            if size == 0 and allow_empty:
                return {}
            limit = body_limit if max_bytes is None else min(body_limit, max_bytes)
            if not 0 < size <= limit:
                raise ValueError(f"body must contain at most {limit} bytes")
            return _request_json(self.read_body(size))

        def _session_failure(self, error):
            if isinstance(error, APCReuseDisabled):
                self.error(409, str(error))
                return
            from .runtime.apc_v2 import (
                APCSessionCapacityError,
                APCSessionNotFound,
                APCSessionUnavailable,
            )

            if isinstance(error, (APCSessionNotFound, LookupError)):
                self.error(404, "unknown APCv2 session")
            elif isinstance(error, AdmissionClosed):
                self.api_error(503, str(error), retry_after=True)
            elif isinstance(error, APCSessionUnavailable):
                self.error(503, str(error))
            elif isinstance(error, APCSessionCapacityError):
                self.error(429 if "queue" in str(error) else 409, str(error))
            elif isinstance(error, (ValueError, json.JSONDecodeError)):
                self.error(400, str(error))
            else:
                logging.getLogger("mlx2.server").exception(
                    "APCv2 session control failed",
                    exc_info=(type(error), error, error.__traceback__),
                )
                self.error(500, "internal server error")

        def _read_session_body(self, *, allow_empty=False):
            return self._read_json_body(allow_empty=allow_empty)

        def _record_http(self, status):
            if self._http_recorded:
                return
            self._http_recorded = True
            metrics = getattr(engine, "http_metrics", None)
            if metrics is not None:
                metrics.completed(
                    self.command,
                    self._metric_route(),
                    int(status),
                    self._http_started_at,
                )
            if self._request_trace is not None:
                self._request_trace.finish(int(status))

        def read_body(self, size):
            deadline = time.monotonic() + self.REQUEST_BODY_DEADLINE_SECONDS
            chunks = []
            remaining = size
            while remaining > 0:
                budget = deadline - time.monotonic()
                if budget <= 0:
                    raise TimeoutError("request body was not received in time")
                self.connection.settimeout(min(30, budget))
                # read() may loop over many socket receives internally,
                # restarting the socket timeout after each trickled byte.
                # read1() returns after one receive so the absolute deadline
                # is checked even while the client keeps sending data.
                chunk = self.rfile.read1(min(remaining, 1 << 20))
                if not chunk:
                    raise ValueError("request body ended early")
                chunks.append(chunk)
                remaining -= len(chunk)
                self._request_body_remaining -= len(chunk)
            self.connection.settimeout(30)
            self._body_consumed = True
            return b"".join(chunks)

        def log_message(self, fmt, *args):
            logging.getLogger("mlx2.http").info(fmt, *args)

        def send_json(self, status, value, *, headers=None):
            payload = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            for name, value in (headers or {}).items():
                self.send_header(name, str(value))
            self.end_headers()
            try:
                self.wfile.write(payload)
            finally:
                self._record_http(status)

        def send_text(self, status, value, *, content_type="text/plain; charset=utf-8"):
            payload = value.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            finally:
                self._record_http(status)

        def error(self, status, message, *, mlx2=None, code=None):
            payload = {
                "error": {
                    "message": message,
                    "type": "invalid_request_error"
                    if status in {400, 413}
                    else "server_error",
                }
            }
            if code is not None:
                payload["error"]["code"] = code
            if mlx2 is not None:
                payload["mlx2"] = mlx2
            self.send_json(
                status,
                payload,
                headers={"Retry-After": "1"} if status == 429 else None,
            )

        def api_error(
            self,
            status,
            message,
            *,
            anthropic=False,
            retry_after=False,
            mlx2=None,
            code=None,
            headers=None,
        ):
            payload = translated_api_error(status, message, anthropic=anthropic)
            if code is not None and not anthropic:
                payload["error"]["code"] = code
            if mlx2 is not None:
                payload["mlx2"] = mlx2
            extra = dict(headers or {})
            if status == 429 or retry_after:
                extra["Retry-After"] = "1"
            self.send_json(status, payload, headers=extra or None)

        def do_GET(self):
            self._ensure_trace()
            if not self._authorize_request():
                return
            path = self.path.split("?", 1)[0]
            if path == "/v1/admin/state":
                if not self._authorize_admin():
                    return
                self.send_json(200, engine.service_state())
                return
            if path == "/v1/admin/power":
                if not self._authorize_admin():
                    return
                state = getattr(engine, "power_governor_state", lambda: None)()
                if state is None:
                    self.api_error(409, "power governor is not enabled")
                else:
                    self.send_json(200, state)
                return
            if not self._authenticate_tenant(path):
                return
            try:
                session_route = self._session_route()
            except ValueError as error:
                self.error(400, str(error))
                return
            if session_route is not None:
                parsed, session_id, action = session_route
                if not self._sessions_available() or action is not None:
                    self.error(404, "unknown endpoint")
                    return
                tenant = self._tenant_id
                try:
                    if session_id is not None:
                        value = engine.apc_session_state(tenant, session_id)
                    else:
                        query = parse_qs(parsed.query, keep_blank_values=True)
                        if set(query) - {"limit", "cursor"} or any(
                            len(values) != 1 for values in query.values()
                        ):
                            raise ValueError(
                                "session list supports only one limit and cursor"
                            )
                        value = engine.apc_sessions(
                            tenant,
                            limit=int(query.get("limit", ["50"])[0]),
                            cursor=int(query.get("cursor", ["0"])[0]),
                        )
                    self.send_json(200, value)
                except Exception as error:
                    self._session_failure(error)
                return
            tenant_id = self._tenant_id
            parsed = urlsplit(self.path)
            path = parsed.path
            query = parse_qs(parsed.query, keep_blank_values=True)
            if path == "/v1/files":
                try:
                    limit = int(query.get("limit", ["20"])[0])
                    self.send_json(
                        200,
                        file_store.list(
                            tenant_id,
                            limit=limit,
                            after=query.get("after", [None])[0],
                            purpose=query.get("purpose", [None])[0],
                        ),
                    )
                except ValueError as error:
                    self.error(400, str(error))
                return
            if path == "/v1/batches":
                try:
                    limit = int(query.get("limit", ["20"])[0])
                    self.send_json(
                        200,
                        batches.list(
                            tenant_id,
                            limit=limit,
                            after=query.get("after", [None])[0],
                        ),
                    )
                except ValueError as error:
                    self.error(400, str(error))
                return
            if path.startswith("/v1/responses/") and path.endswith("/input_items"):
                response_id_ = path.removeprefix("/v1/responses/").removesuffix(
                    "/input_items"
                )
                try:
                    if not response_id_ or set(query) - {
                        "after",
                        "include",
                        "include[]",
                        "limit",
                        "order",
                    }:
                        raise ValueError("invalid response input-items query")
                    if any(
                        len(query.get(name, [])) > 1
                        for name in ("after", "limit", "order")
                    ):
                        raise ValueError(
                            "response input-items cursors and controls must be singular"
                        )
                    include = [
                        *query.get("include", []),
                        *query.get("include[]", []),
                    ]
                    unknown_include = set(include) - RESPONSES_INPUT_ITEM_INCLUDES
                    if unknown_include:
                        raise ValueError(
                            "unsupported input_items include values: "
                            + ", ".join(sorted(unknown_include))
                        )
                    record = response_store.input_record(tenant_id, response_id_)
                    items = responses_input_items(
                        record.get("context_messages"),
                        response_id_,
                        include=include,
                        signer=getattr(engine, "reasoning_signer", None),
                        model=record["payload"].get("model"),
                        tenant_id=tenant_id,
                    )
                    self.send_json(
                        200,
                        responses_input_items_page(
                            items,
                            limit=int(query.get("limit", ["20"])[0]),
                            after=query.get("after", [None])[0],
                            order=query.get("order", ["desc"])[0],
                        ),
                    )
                except ResourceNotFound:
                    self.error(404, "response not found")
                except ResponsesInputItemsUnavailable as error:
                    self.api_error(409, str(error))
                except ValueError as error:
                    self.error(400, str(error))
                return
            if path.startswith("/v1/responses/"):
                response_id_ = path.removeprefix("/v1/responses/")
                try:
                    self.send_json(200, response_store.get(tenant_id, response_id_))
                except ResourceNotFound:
                    self.error(404, "response not found")
                return
            if path.startswith("/v1/files/"):
                suffix = path.removeprefix("/v1/files/")
                content_request = suffix.endswith("/content")
                file_id = suffix.removesuffix("/content")
                try:
                    if content_request:
                        content, content_type, _ = file_store.content(tenant_id, file_id)
                        self.send_bytes(
                            200,
                            content,
                            content_type=served_file_content_type(content_type),
                            headers={
                                "X-Content-Type-Options": "nosniff",
                                "Content-Disposition": "attachment",
                            },
                        )
                    else:
                        self.send_json(200, file_store.get(tenant_id, file_id))
                except ResourceNotFound:
                    self.error(404, "file not found")
                return
            if path.startswith("/v1/batches/"):
                batch_id = path.removeprefix("/v1/batches/")
                try:
                    self.send_json(200, batches.get(tenant_id, batch_id))
                except ResourceNotFound:
                    self.error(404, "batch not found")
                return
            if self.path == "/metrics":
                from .prometheus import CONTENT_TYPE, render_engine_metrics

                try:
                    payload = (
                        engine.prometheus_metrics()
                        if hasattr(engine, "prometheus_metrics")
                        else render_engine_metrics(engine)
                    )
                except Exception:  # exporter failure must not take down serving
                    logging.getLogger("mlx2.metrics").exception(
                        "Prometheus exposition failed"
                    )
                    counts = getattr(engine, "counts", None)
                    if counts is not None:
                        lock = getattr(engine, "lock", None)
                        if lock is None:
                            counts["telemetry_export_errors"] += 1
                            failures = counts["telemetry_export_errors"]
                        else:
                            with lock:
                                counts["telemetry_export_errors"] += 1
                                failures = counts["telemetry_export_errors"]
                    else:
                        failures = 1
                    payload = (
                        "# HELP mlx2_telemetry_source_available Whether the host-side telemetry snapshot was rendered successfully.\n"
                        "# TYPE mlx2_telemetry_source_available gauge\n"
                        "mlx2_telemetry_source_available 0\n"
                        "# HELP mlx2_telemetry_export_errors_total Prometheus exposition failures.\n"
                        "# TYPE mlx2_telemetry_export_errors_total counter\n"
                        f"mlx2_telemetry_export_errors_total {failures}\n"
                    )
                    self.send_text(503, payload, content_type=CONTENT_TYPE)
                    return
                if tenant_authenticator is not None:
                    payload += tenant_authenticator.prometheus()
                self.send_text(200, payload, content_type=CONTENT_TYPE)
                return
            status = engine.status()
            if self.path == "/health":
                service_state = (status.get("quiesce") or {}).get(
                    "state", "serving"
                )
                self.send_json(
                    200 if status["healthy"] and service_state == "serving" else 503,
                    (
                        {"status": service_state}
                        if service_state != "serving"
                        else {
                            "status": "ok" if status["healthy"] else "unavailable",
                            "error": status["error"],
                        }
                    ),
                )
            elif self.path == "/v1/models":
                self.send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": status.get("model", Path(engine.model_path).name),
                                "object": "model",
                                "owned_by": "mlx2",
                                "loaded": status["healthy"],
                                "capabilities": status.get("capabilities", []),
                                "qualification": status.get(
                                    "qualification", "unavailable"
                                ),
                            },
                            # vLLM shape: each concurrent LoRA adapter is a
                            # selectable model whose parent is the base.
                            *(
                                {
                                    "id": name,
                                    "object": "model",
                                    "owned_by": "mlx2",
                                    "parent": status.get("model", Path(engine.model_path).name),
                                    "loaded": status["healthy"],
                                    # Served by the base route.
                                    "capabilities": status.get("capabilities", []),
                                }
                                for name in (status.get("multi_lora") or {}).get(
                                    "registered", ()
                                )
                            ),
                        ],
                    },
                )
            elif self.path == "/v1/status":
                if tensorfold_owned_router is not None:
                    status = {**status, "tensorfold_owned": tensorfold_owned_router.status()}
                if tenant_authenticator is not None and "recent_receipts" in status:
                    # Receipts carry request and session ids, seeds and
                    # logit_bias: show a tenant only its own.  An engine that
                    # cannot scope them shows none.
                    scoped = getattr(engine, "recent_receipts", None)
                    status = {
                        **status,
                        "recent_receipts": scoped(tenant_id=self._tenant_id)
                        if callable(scoped)
                        else [],
                    }
                self.send_json(
                    200,
                    {
                        **status,
                        "api_resources": {
                            name: resource.status()
                            for name, resource in engine.api_resources.items()
                        },
                        "tenant_auth": tenant_authenticator.status()
                        if tenant_authenticator is not None
                        else {"enabled": False},
                        "agent_compat": compat_policy.status(),
                        "semantic_memory": semantic_middleware.status()
                        if semantic_middleware is not None
                        else {"enabled": False},
                    },
                )
            elif self.path == "/v1/status/batching":
                self.send_json(
                    200,
                    engine.batching_status()
                    if tenant_authenticator is None
                    else tenant_batching_status(engine, self._tenant_id),
                )
            else:
                self.error(404, "unknown endpoint")

        def send_bytes(
            self, status, value, *, content_type="application/octet-stream", headers=None
        ):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(value)))
            for name, header in (headers or {}).items():
                self.send_header(name, str(header))
            self.end_headers()
            try:
                self.wfile.write(value)
            finally:
                self._record_http(status)

        def do_DELETE(self):
            self._ensure_trace()
            if not self._authorize_request():
                return
            if not self._authenticate_tenant(self.path.split("?", 1)[0]):
                return
            try:
                session_route = self._session_route()
            except ValueError as error:
                self.error(400, str(error))
                return
            if session_route is not None:
                _parsed, session_id, action = session_route
                if (
                    not self._sessions_available()
                    or session_id is None
                    or action is not None
                ):
                    self.error(404, "unknown endpoint")
                    return
                try:
                    value = engine.apc_session_delete(
                        self._tenant_id, session_id
                    )
                    if semantic_middleware is not None:
                        value = {
                            **value,
                            "semantic_memory_deleted": semantic_middleware.delete_session(
                                self._tenant_id, session_id
                            ),
                        }
                    self.send_json(200, value)
                except Exception as error:
                    self._session_failure(error)
                return
            tenant_id = self._tenant_id
            path = self.path.split("?", 1)[0]
            try:
                if path.startswith("/v1/responses/"):
                    self.send_json(
                        200,
                        response_store.delete(
                            tenant_id, path.removeprefix("/v1/responses/")
                        ),
                    )
                elif path.startswith("/v1/files/"):
                    self.send_json(
                        200,
                        file_store.delete(tenant_id, path.removeprefix("/v1/files/")),
                    )
                else:
                    self.error(404, "unknown endpoint")
            except ResourceNotFound:
                self.error(404, "resource not found")

        def do_POST(self):
            self._pending_responses_prologue = None
            self._ensure_trace()
            if not self._authorize_request():
                return
            path = self.path.split("?", 1)[0]
            if path in {"/v1/admin/quiesce", "/v1/admin/resume"}:
                if not self._authorize_admin():
                    return
                try:
                    body = self._read_json_body(allow_empty=True, max_bytes=65536)
                    if path.endswith("/quiesce"):
                        options = validate_quiesce_body(body)
                        value = engine.quiesce(**options)
                    else:
                        sessions = validate_resume_body(body)
                        value = engine.resume(prefetch_sessions=sessions)
                    self.send_json(202, value)
                except (APCReuseDisabled, SuspendUnavailable) as error:
                    self.api_error(409, str(error))
                except (ValueError, json.JSONDecodeError) as error:
                    self.api_error(400, str(error))
                return
            if path == "/v1/admin/power":
                if not self._authorize_admin():
                    return
                try:
                    options = validate_power_body(
                        self._read_json_body(max_bytes=4096)
                    )
                    setter = getattr(engine, "set_power_governor", None)
                    if setter is None:
                        raise PowerGovernorUnavailable("power governor is not enabled")
                    self.send_json(200, setter(**options))
                except (PowerGovernorUnavailable, PowerSignalUnavailable) as error:
                    self.api_error(409, str(error))
                except (ValueError, json.JSONDecodeError) as error:
                    self.api_error(400, str(error))
                return
            if not self._authenticate_tenant(path):
                return
            owned_header = self.headers.get("X-MLX2-TensorFold-Owned")
            if (self.headers.get("X-MLX2-TensorFold-Route") is not None
                    and owned_header is None):
                self.api_error(400, "TensorFold route control requires the owned profile")
                return
            if owned_header is not None:
                if tensorfold_owned_router is None or owned_header != "1":
                    self.api_error(400, "TensorFold-owned completion route is unavailable")
                    return
                try:
                    body = self._read_json_body(max_bytes=1 << 20)
                    route = self.headers.get("X-MLX2-TensorFold-Route", "auto")
                    result = tensorfold_owned_completion_payload(
                        engine, tensorfold_owned_router, body, path, route=route,
                        disconnect_probe=lambda: tensorfold_owned_client_disconnected(
                            self.connection))
                    self.send_json(200, result)
                except ConnectionAbortedError:
                    return
                except (ValueError, json.JSONDecodeError) as error:
                    self.api_error(400, str(error))
                except TimeoutError:
                    self.api_error(504, "TensorFold-owned worker timed out")
                except RuntimeError as error:
                    self.api_error(503, str(error))
                return
            if path == "/v1/experimental/tensorfold-owned/generate":
                if tensorfold_owned_router is None:
                    self.error(404, "unknown endpoint")
                    return
                try:
                    body = self._read_json_body(max_bytes=1 << 20)
                    if (not isinstance(body, dict)
                            or not {"token_ids", "max_new_tokens"} <= set(body)
                            or set(body) - {"token_ids", "max_new_tokens", "route"}):
                        raise ValueError("TensorFold-owned request requires token_ids and max_new_tokens; route is optional")
                    result = tensorfold_owned_router.generate(
                        body["token_ids"], body["max_new_tokens"],
                        route=body.get("route", "auto"),
                        disconnect_probe=lambda: tensorfold_owned_client_disconnected(
                            self.connection))
                    self.send_json(200, result)
                except ConnectionAbortedError:
                    return
                except (ValueError, json.JSONDecodeError) as error:
                    self.api_error(400, str(error))
                except TimeoutError:
                    self.api_error(504, "TensorFold-owned worker timed out")
                except RuntimeError as error:
                    self.api_error(503, str(error))
                return
            try:
                session_route = self._session_route()
            except ValueError as error:
                self.error(400, str(error))
                return
            if session_route is not None:
                _parsed, session_id, action = session_route
                if (
                    not self._sessions_available()
                    or session_id is None
                    or action not in {"park", "resume"}
                ):
                    self.error(404, "unknown endpoint")
                    return
                try:
                    body = self._read_session_body(allow_empty=action == "resume")
                    if set(body) - {"ttl_seconds"}:
                        raise ValueError(
                            "session controls only accept ttl_seconds"
                        )
                    tenant = self._tenant_id
                    if action == "park":
                        if "ttl_seconds" not in body:
                            raise ValueError("park requires ttl_seconds")
                        value = engine.apc_session_park(
                            tenant, session_id, ttl_seconds=body["ttl_seconds"]
                        )
                        self.send_json(200, value)
                    else:
                        value = engine.apc_session_resume(
                            tenant,
                            session_id,
                            ttl_seconds=body.get("ttl_seconds"),
                        )
                        self.send_json(202, value)
                except Exception as error:
                    self._session_failure(error)
                return
            if path not in {
                "/v1/chat/completions",
                "/v1/completions",
                "/v1/responses",
                "/v1/files",
                "/v1/batches",
                "/v1/embeddings",
                "/v1/rerank",
                "/v1/audio/speech",
                "/v1/load_lora_adapter",
                "/v1/unload_lora_adapter",
                "/v1/messages",
                "/v1/messages/count_tokens",
                "/tokenize",
                "/apply-template",
            } and not (path.startswith("/v1/batches/") and path.endswith("/cancel")):
                self.error(404, "unknown endpoint")
                return
            job = None
            jobs = []
            admission_lease = None
            streaming = False
            buffered_tool_stream = False
            buffered_hosted_stream = False
            responses_api = path == "/v1/responses"
            anthropic = path in {"/v1/messages", "/v1/messages/count_tokens"}
            anthropic_count_tokens = path == "/v1/messages/count_tokens"
            anthropic_request = None
            anthropic_translator = None
            response_metadata = {}
            response_options = None
            tenant_id = self._tenant_id
            response_message_started = False
            response_message_output_index = None
            response_reasoning_started = False
            response_reasoning_output_index = None
            response_output_order = []
            agent_compat = None
            semantic_state = None
            semantic_score_tokens = lambda prompt, token_ids: score_choice_tokens_via_engine(
                engine, prompt, token_ids, tenant_id=tenant_id
            )
            # Item 12: tool calls streamed as they complete because the decode
            # grammar is engaged (tool_grammar_streaming); Responses ids and
            # output indexes already sent for function_call items.
            grammar_tool_stream = False
            grammar_stream_decided = False

            def stream_failure(status, message, *, code=None):
                """Report a failure on a response already committed as SSE.

                A keepalive can commit the stream before a later step (a
                hosted continuation's submit) fails; a second HTTP status
                line written into the event stream is not a response.
                """
                if anthropic and anthropic_translator is not None:
                    for failure in anthropic_translator.failure(message, status):
                        self._anthropic_sse(failure)
                elif responses_api:
                    self._responses_failure(
                        job,
                        message,
                        translated_api_error(status, message)["error"]["type"],
                        code=code,
                    )
                else:
                    error = {"message": message}
                    if code is not None:
                        error["code"] = code
                    self._sse({"error": error})
                    self._sse("[DONE]")
                self._record_http(200)

            prologue_sent = False
            grammar_stream_ready = False
            grammar_message_index = None
            streamed_calls = {}
            self._responses_sequence = 0
            self._chat_role_pending = True
            try:
                size = self._content_length()
                if not 0 < size <= body_limit:
                    self.close_connection = True
                    self.api_error(
                        413,
                        f"body must contain at most {body_limit} bytes",
                        anthropic=anthropic,
                    )
                    return
                try:
                    raw = self.read_body(size)
                except (TimeoutError, socket.timeout):
                    self.close_connection = True
                    self.api_error(
                        408,
                        "request body was not received in time",
                        anthropic=anthropic,
                    )
                    return
                if path == "/v1/files":
                    fields = parse_multipart_form(
                        self.headers.get("Content-Type", ""), raw
                    )
                    if set(fields) != {"purpose", "file"}:
                        raise ValueError("Files API requires purpose and file fields")
                    purpose = fields["purpose"]["content"].decode("utf-8")
                    upload = fields["file"]
                    self.send_json(
                        200,
                        file_store.create(
                            tenant_id,
                            filename=upload["filename"] or "upload",
                            purpose=purpose,
                            content_type=upload["content_type"],
                            content=upload["content"],
                        ),
                    )
                    return
                body = _request_json(raw)
                if path in {
                    "/v1/chat/completions",
                    "/v1/completions",
                    "/v1/responses",
                    "/v1/messages",
                }:
                    with wrongly_typed_request_is_invalid():
                        body = resolve_request_session(
                            body, self.headers.get("X-mlx2-Session-ID")
                        )
                if responses_api or anthropic:
                    agent_compat = compat_policy.resolve(
                        self.headers,
                        tenant_id,
                        self.path,
                        getattr(engine, "counts", None),
                    )
                if path == "/v1/audio/speech":
                    with wrongly_typed_request_is_invalid():
                        speech = validate_speech_request(body)
                    if speech["stream_format"] == "sse":
                        raise CapabilityUnavailable(
                            "speech SSE requires a qualified streaming audio adapter"
                        )
                    model = engine.status().get("model")
                    if speech["model"] is not None and speech["model"] != model:
                        raise ResourceNotFound("unknown model")
                    output = engine.synthesize_speech(
                        speech["input"],
                        voice=speech["voice"],
                        instructions=speech["instructions"],
                        response_format=speech["response_format"],
                        speed=speech["speed"],
                    )
                    expected = _AUDIO_RESPONSE_TYPES[speech["response_format"]]
                    if output.media_type != expected:
                        raise RuntimeError(
                            "audio adapter returned a MIME type inconsistent with response_format"
                        )
                    self.send_bytes(200, output.data, content_type=output.media_type)
                    return
                if path == "/v1/batches":

                    def create_batch():
                        with wrongly_typed_request_is_invalid():
                            return batches.create(tenant_id, body)

                    admit_batch = getattr(engine, "admit_batch_submission", None)
                    value = (
                        admit_batch(create_batch)
                        if callable(admit_batch)
                        else create_batch()
                    )
                    self.send_json(200, value)
                    return
                if path.startswith("/v1/batches/") and path.endswith("/cancel"):
                    batch_id = path.removeprefix("/v1/batches/").removesuffix(
                        "/cancel"
                    )
                    self.send_json(200, batches.cancel(tenant_id, batch_id))
                    return
                if path == "/v1/embeddings":
                    self.send_json(200, embeddings_payload(engine, body))
                    return
                if path == "/v1/rerank":
                    self.send_json(200, rerank_payload(engine, body))
                    return
                if path == "/v1/load_lora_adapter":
                    self.send_json(200, lora_control_payload(engine, body, load=True))
                    return
                if path in {"/tokenize", "/apply-template"}:
                    self.send_json(200, prompt_render_payload(engine, body, path))
                    return
                if path == "/v1/unload_lora_adapter":
                    self.send_json(200, lora_control_payload(engine, body, load=False))
                    return
                if anthropic:
                    anthropic_request = body
                    status = engine.status()
                    model = _generation_model(status, body)
                    translation_metadata = {}
                    with wrongly_typed_request_is_invalid():
                        body = anthropic_request_to_chat(
                            body,
                            count_tokens=anthropic_count_tokens,
                            signer=getattr(engine, "reasoning_signer", None),
                            tenant_id=tenant_id,
                            model=body.get("model") or engine.status().get("model"),
                            translation_metadata=translation_metadata,
                            agent_compat=agent_compat,
                            counts=getattr(engine, "counts", None),
                        )
                    rejections = translation_metadata.get(
                        "reasoning_signature_rejections", 0
                    )
                    if rejections:
                        engine_counts = getattr(engine, "counts", None)
                        if engine_counts is not None:
                            engine_counts["reasoning_signature_rejections"] += rejections
                    if anthropic_count_tokens:
                        self.send_json(
                            200, {"input_tokens": int(engine.count_tokens(body))}
                        )
                        return
                if responses_api:
                    with wrongly_typed_request_is_invalid():
                        body, response_options = prepare_responses_request(
                            body, tenant_id, agent_compat
                        )
                    response_metadata = response_options["metadata"]
                chat = path != "/v1/completions"
                status = engine.status()
                with wrongly_typed_request_is_invalid():
                    body = validate_request(
                        body,
                        chat,
                        structured_thinking=bool(
                            (status.get("structured_output") or {}).get(
                                "thinking_deferral"
                            )
                        ),
                        allow_buffered_tool_stream=True,
                        constrained_tool_grammar=bool(
                            (status.get("settings") or {}).get(
                                "constrained_tool_grammar"
                            )
                        ),
                        max_tools=128
                        if responses_api and agent_compat.enabled
                        else 64,
                        max_stops=ANTHROPIC_MAX_STOP_SEQUENCES if anthropic else 4,
                    )
                if semantic_middleware is not None and chat:
                    ensure_semantic_classifier()
                    body, semantic_state = semantic_middleware.prepare(
                        body,
                        tenant_id=tenant_id,
                        authenticated_tenant=tenant_authenticator is not None,
                    )
                if any(
                    isinstance(message.get("content"), list)
                    for message in body.get("messages", ())
                ) and not (
                    callable(getattr(engine, "supports_multimodal", None))
                    and engine.supports_multimodal()
                ):
                    counts = getattr(engine, "counts", None)
                    if counts is not None:
                        counts["multimodal_rejected"] += 1
                    raise CapabilityUnavailable(
                        "the loaded adapter has no qualified image, video, or audio encoder"
                    )
                buffered_tool_stream = bool(
                    body.get("stream")
                    and not anthropic
                    and not responses_api
                    and body.get("tools")
                    and (
                        any(
                            tool["function"].get("strict") is True
                            for tool in body["tools"]
                        )
                        or body.get("tool_choice") == "required"
                        or isinstance(body.get("tool_choice"), dict)
                        or body.get("parallel_tool_calls", True) is False
                    )
                )
                buffered_hosted_stream = bool(
                    responses_api
                    and body.get("stream")
                    and response_options.get("tool_executors")
                )
                # Agent-compat rewrites custom and namespace shim calls into
                # other item types and names only in the final payload, so an
                # item streamed early would not be the item completed.
                compat_tool_map = (response_options or {}).get("agent_compat") or {}
                grammar_stream_ready = bool(
                    body.get("stream")
                    and body.get("tools")
                    and not buffered_hosted_stream
                    and not compat_tool_map.get("custom")
                    and not compat_tool_map.get("namespaces")
                    and (status.get("settings") or {}).get("tool_grammar_streaming")
                )
                model = _generation_model(status, body)
                sample_count = body.get("n", 1)
                acquire_admission = getattr(engine, "acquire_admission", None)
                if callable(acquire_admission):
                    admission_lease = acquire_admission("generation")
                if sample_count > 1:
                    admission = engine.admit_parallel_samples(sample_count)
                    base_seed = body.get("seed")
                    samples = []
                    for index in range(sample_count):
                        sample = {**body, "n": 1}
                        if base_seed is not None:
                            sample["seed"] = (base_seed + index) % (2**32)
                        samples.append(sample)
                    submit_many_kwargs = {
                        "tenant_id": self._tenant_id
                    }
                    if admission_lease is not None:
                        submit_many_kwargs["admitted"] = True
                    jobs.extend(engine.submit_many(samples, **submit_many_kwargs))
                    results = collect_parallel_samples(
                        jobs, body, chat=chat, connection=self.connection
                    )
                    choices, usages, receipts = zip(*results)
                    for index, choice in enumerate(choices):
                        choice["index"] = index
                    # Every sample shares one prompt: count it once, as the
                    # single-sample path and OpenAI's n > 1 usage do, and
                    # report the leader's prefix-cache hit for it.
                    usage = {
                        "prompt_tokens": usages[0]["prompt_tokens"],
                        "completion_tokens": sum(item["completion_tokens"] for item in usages),
                        "completion_tokens_details": {
                            "reasoning_tokens": sum(
                                item["completion_tokens_details"]["reasoning_tokens"]
                                for item in usages
                            )
                        },
                    }
                    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                    usage["prompt_tokens_details"] = {
                        "cached_tokens": int(getattr(jobs[0], "cached_tokens", 0) or 0)
                    }
                    self.send_json(
                        200,
                        {
                            "id": jobs[0].id,
                            "object": "chat.completion" if chat else "text_completion",
                            "created": int(jobs[0].created),
                            "model": model,
                            "choices": list(choices),
                            "usage": usage,
                            "mlx2": {"parallel_sampling": admission, "samples": list(receipts)},
                        },
                    )
                    return
                submit_kwargs = {"tenant_id": tenant_id}
                if admission_lease is not None:
                    submit_kwargs["admitted"] = True
                job = engine.submit(body, **submit_kwargs)
                parts, reasoning, calls, probabilities = [], [], [], []
                text_logprobs = ResponsesTextLogprobs()
                hosted_rounds = 0
                hosted_usage = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cached_tokens": 0,
                    "reasoning_tokens": 0,
                }
                hosted_receipts = []
                def commit_stream():
                    """Commit the SSE response headers (once)."""
                    nonlocal streaming
                    if streaming:
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    streaming = True

                def open_stream():
                    """Commit the SSE response and send its protocol prologue."""
                    nonlocal anthropic_translator, prologue_sent
                    commit_stream()
                    prologue_sent = True
                    if anthropic:
                        anthropic_translator = AnthropicStreamTranslator(
                            message_id=job.id,
                            model=model,
                            request=anthropic_request,
                            input_tokens=job.prompt_tokens,
                            cache_read_input_tokens=job.cached_tokens,
                            signer=getattr(engine, "reasoning_signer", None),
                            tenant_id=tenant_id,
                        )
                        for initial in anthropic_translator.start():
                            self._anthropic_sse(initial)
                    elif responses_api:
                        self._responses_sse(
                            {
                                "type": "response.created",
                                "response": {
                                    "id": response_id(job),
                                    "object": "response",
                                    "created_at": int(job.created),
                                    "status": "in_progress",
                                    "model": model,
                                    "output": [],
                                },
                            }
                        )

                def ensure_prologue():
                    """Send ``response.created`` if a keepalive opened the stream without it."""
                    self._pending_responses_prologue = None
                    if streaming and not prologue_sent:
                        open_stream()

                def keepalive():
                    """Send bytes on a stream that has been silent too long.

                    A long prefill sends nothing until the first token, and a
                    buffered tool stream nothing until its terminal contract
                    has passed.  Once the job is admitted (its prompt counted,
                    which the Anthropic prologue reports) the stream is opened
                    early; a buffered stream then reports a terminal failure as
                    an SSE error event instead of an HTTP status, and still
                    sends no call before the contract passes.  Before
                    admission nothing is committed, so admission errors keep
                    their status.
                    """
                    if not streaming:
                        if job.uid is None:
                            return  # not attached to a lane yet
                        if buffered_hosted_stream:
                            # Its response is named after the last hosted
                            # round's job; the prologue waits for that, or
                            # for a failure, which must follow it.
                            commit_stream()
                            self._pending_responses_prologue = ensure_prologue
                        else:
                            open_stream()
                    self.wfile.write(SSE_KEEPALIVE)
                    self.wfile.flush()
                    self._last_stream_write = time.monotonic()

                def with_keepalive(call):
                    """Run a blocking ``call`` (a hosted tool) keeping the stream alive."""
                    if not (body.get("stream") and sse_keepalive_seconds):
                        return call()
                    outcome = {}

                    def run():
                        try:
                            outcome["value"] = call()
                        except BaseException as error:  # noqa: BLE001 - re-raised below
                            outcome["error"] = error

                    worker = threading.Thread(target=run, daemon=True)
                    worker.start()
                    next_attempt = 0.0
                    while worker.is_alive():
                        due = max(
                            self._last_stream_write + sse_keepalive_seconds, next_attempt
                        )
                        worker.join(max(0.0, min(0.5, due - time.monotonic())))
                        if worker.is_alive() and time.monotonic() >= due:
                            keepalive()
                            next_attempt = time.monotonic() + sse_keepalive_seconds
                    if "error" in outcome:
                        raise outcome["error"]
                    return outcome["value"]

                self._last_stream_write = time.monotonic()

                while True:
                    # A stream can go quiet during a long prefill after its
                    # first progress event. Keep watching the socket so a
                    # disconnected client releases its lane and the partial
                    # cache can be published for APCv2 rolling recovery.
                    event = wait_event(
                        job.events,
                        connection=self.connection,
                        idle_seconds=sse_keepalive_seconds,
                        on_idle=keepalive if body.get("stream") else None,
                        idle_since=lambda: self._last_stream_write,
                    )
                    if "error" in event:
                        mlx2 = event.get("mlx2")
                        if streaming:
                            if anthropic:
                                for failure in anthropic_translator.failure(
                                    event["error"], event.get("status", 503)
                                ):
                                    if mlx2 is not None:
                                        failure["mlx2"] = mlx2
                                    self._anthropic_sse(failure)
                            elif responses_api:
                                response = {
                                    "id": response_id(job),
                                    "object": "response",
                                    "status": "failed",
                                    "error": {
                                        "message": event["error"],
                                        "type": "invalid_request_error"
                                        if event.get("status") == 400
                                        else "server_error",
                                    },
                                }
                                if event.get("code") is not None:
                                    response["error"]["code"] = event["code"]
                                if mlx2 is not None:
                                    response["mlx2"] = mlx2
                                ensure_prologue()
                                self._responses_sse(
                                    {
                                        "type": "response.failed",
                                        "response": response,
                                    }
                                )
                            else:
                                failure = {"error": {"message": event["error"]}}
                                if event.get("code") is not None:
                                    failure["error"]["code"] = event["code"]
                                if mlx2 is not None:
                                    failure["mlx2"] = mlx2
                                self._sse(failure)
                                self._sse("[DONE]")
                            self._record_http(200)
                        elif anthropic:
                            self.api_error(
                                event.get("status", 503),
                                event["error"],
                                anthropic=True,
                                mlx2=mlx2,
                            )
                        else:
                            self.error(
                                event.get("status", 503),
                                event["error"],
                                mlx2=mlx2,
                                code=event.get("code"),
                            )
                        return
                    prompt_progress = None
                    if "prompt_progress" in event:
                        prompt_progress = take_prompt_progress(job, event)
                        if (
                            not body.get("stream")
                            or buffered_tool_stream
                            or buffered_hosted_stream
                        ):
                            # Buffered streams commit no bytes before terminal
                            # validation, so there is no stream to report into.
                            continue
                    # Decided once, at the first event past prompt progress
                    # (a keepalive may already have opened the stream, but no
                    # delta has been sent).
                    grammar_stream_decision, grammar_stream_decided = (
                        not grammar_stream_decided, True
                    )
                    if (
                        grammar_stream_ready
                        and grammar_stream_decision
                        and getattr(job, "tool_grammar_status", None) == "engaged"
                    ):
                        # Admission engaged the tool grammar before the first
                        # token, so calls are well-formed as they complete:
                        # stream them; the terminal contract still runs at the
                        # end and a failure becomes an SSE error event.
                        grammar_tool_stream = True
                        buffered_tool_stream = False
                        engine_counts = getattr(engine, "counts", None)
                        if engine_counts is not None:
                            with getattr(engine, "lock", None) or contextlib.nullcontext():
                                engine_counts["constrained_tool_grammar_streams"] += 1
                    if (
                        body.get("stream")
                        and not streaming
                        and not buffered_tool_stream
                        and not buffered_hosted_stream
                    ):
                        open_stream()
                    if prompt_progress is not None:
                        if anthropic:
                            self._anthropic_sse(
                                {"type": "ping", "prompt_progress": prompt_progress}
                            )
                        elif responses_api:
                            self._responses_sse(
                                {
                                    "type": "response.in_progress",
                                    "response": {
                                        "id": response_id(job),
                                        "object": "response",
                                        "created_at": int(job.created),
                                        "status": "in_progress",
                                        "model": model,
                                        "output": [],
                                    },
                                    "prompt_progress": prompt_progress,
                                }
                            )
                        else:
                            choice = {"index": 0, "finish_reason": None}
                            choice.update({"delta": {}} if chat else {"text": ""})
                            self._sse(
                                {
                                    "id": job.id,
                                    "object": "chat.completion.chunk"
                                    if chat
                                    else "text_completion",
                                    "created": int(job.created),
                                    "model": model,
                                    "choices": [choice],
                                    "prompt_progress": prompt_progress,
                                }
                            )
                        continue
                    if "logprob" in event:
                        probabilities.append(event["logprob"])
                        if responses_api:
                            # Held until the token's delta names its item.
                            text_logprobs.logprob(event["logprob"])
                        elif streaming and not buffered_tool_stream:
                            choice = {"index": 0, "finish_reason": None,
                                      "logprobs": {"content": [event["logprob"]]}}
                            choice.update({"delta": {}} if chat else {"text": ""})
                            self._sse({"id": job.id,
                                       "object": "chat.completion.chunk" if chat else "text_completion",
                                       "created": int(job.created), "model": model,
                                       "choices": [choice]})
                    if "text" in event or "delta" in event:
                        delta = event.get("delta", {"content": event.get("text", "")})
                        text_logprobs_delta = (
                            text_logprobs.delta(delta) if responses_api else []
                        )
                        parts.append(delta.get("content", ""))
                        reasoning.append(delta.get("reasoning_content", ""))
                        calls.extend(delta.get("tool_calls", []))
                        # A keepalive may have opened a buffered stream early;
                        # it still sends nothing before its terminal contract.
                        if (
                            streaming
                            and not buffered_tool_stream
                            and not buffered_hosted_stream
                        ):
                            if anthropic:
                                for translated in anthropic_translator.delta(delta):
                                    self._anthropic_sse(translated)
                            elif responses_api:
                                reasoning_delta = delta.get("reasoning_content", "")
                                if reasoning_delta:
                                    if response_reasoning_output_index is None:
                                        response_reasoning_output_index = len(
                                            response_output_order
                                        ) + len(streamed_calls)
                                        response_output_order.append("reasoning")
                                    if not response_reasoning_started:
                                        self._responses_reasoning_start(
                                            job, response_reasoning_output_index
                                        )
                                        response_reasoning_started = True
                                    for event_type, index_name in (
                                        (
                                            "response.reasoning_summary_text.delta",
                                            "summary_index",
                                        ),
                                        (
                                            "response.reasoning_text.delta",
                                            "content_index",
                                        ),
                                    ):
                                        self._responses_sse(
                                            {
                                                "type": event_type,
                                                "item_id": f"rs_{response_id(job).removeprefix('resp_')}",
                                                "output_index": response_reasoning_output_index,
                                                index_name: 0,
                                                "delta": reasoning_delta,
                                            }
                                        )
                                content = delta.get("content", "")
                                if content:
                                    if response_message_output_index is None:
                                        response_message_output_index = len(
                                            response_output_order
                                        ) + len(streamed_calls)
                                        response_output_order.append("message")
                                    response_message_index = response_message_output_index
                                    if grammar_tool_stream:
                                        # Text after streamed calls follows them.
                                        if grammar_message_index is None:
                                            grammar_message_index = response_message_output_index
                                        response_message_index = grammar_message_index
                                    if not response_message_started:
                                        self._responses_message_start(
                                            job, response_message_index
                                        )
                                        response_message_started = True
                                    self._responses_sse(
                                        {
                                            "type": "response.output_text.delta",
                                            "item_id": f"msg_{job.id}",
                                            "output_index": response_message_index,
                                            "content_index": 0,
                                            "delta": content,
                                            **(
                                                {
                                                    "logprobs": [
                                                        responses_logprob(value)
                                                        for value in text_logprobs_delta
                                                    ]
                                                }
                                                if wants_logprobs(body)
                                                else {}
                                            ),
                                        }
                                    )
                                if grammar_tool_stream:
                                    for call in delta.get("tool_calls", ()):
                                        self._responses_call_stream(
                                            job,
                                            call,
                                            streamed_calls,
                                            offset=int(
                                                len(response_output_order)
                                            ),
                                        )
                            else:
                                choice = {"index": 0, "finish_reason": None}
                                choice.update(
                                    {"delta": self._chat_delta(delta)}
                                    if chat
                                    else {"text": delta.get("content", "")}
                                )
                                self._sse(
                                    {
                                        "id": job.id,
                                        "object": "chat.completion.chunk"
                                        if chat
                                        else "text_completion",
                                        "created": int(job.created),
                                        "model": model,
                                        "choices": [choice],
                                    }
                                )
                    if "finish_reason" in event:
                        receipt = event["receipt"]
                        text_logprobs_trailing = text_logprobs.finish()
                        semantic_receipt = (
                            semantic_middleware.receipt(semantic_state)
                            if semantic_middleware is not None
                            else None
                        )
                        if semantic_receipt is not None:
                            receipt = {**(receipt or {}), "semantic_memory": semantic_receipt}
                        if agent_compat is not None and agent_compat.notable:
                            receipt = {
                                **(receipt or {}),
                                "agent_compat": agent_compat.receipt(),
                            }
                        usage = {
                            "prompt_tokens": job.prompt_tokens,
                            "completion_tokens": job.completion_tokens,
                            "total_tokens": job.prompt_tokens + job.completion_tokens,
                            "prompt_tokens_details": {
                                "cached_tokens": job.cached_tokens
                            },
                            # The engine counts every token decoded in the
                            # reasoning channel, a truncated think block too.
                            "completion_tokens_details": {
                                "reasoning_tokens": int(
                                    getattr(job, "reasoning_tokens", 0) or 0
                                )
                            },
                        }
                        hosted_usage["prompt_tokens"] += usage["prompt_tokens"]
                        hosted_usage["completion_tokens"] += usage["completion_tokens"]
                        hosted_usage["cached_tokens"] += int(job.cached_tokens or 0)
                        hosted_usage["reasoning_tokens"] += int(
                            getattr(job, "reasoning_tokens", 0) or 0
                        )
                        hosted_receipts.append(receipt)
                        executors = (
                            response_options.get("tool_executors", {})
                            if responses_api
                            else {}
                        )
                        executable = calls and all(
                            call.get("function", {}).get("name") in executors
                            for call in calls
                        )
                        if not executable and any(
                            call.get("function", {}).get("name") in executors
                            for call in calls
                        ):
                            # Running only the hosted subset would leave a
                            # turn no transcript can replay: the client never
                            # sees the internal call or its output.
                            raise ToolContractError(
                                "the model mixed server-executed (hosted) and "
                                "client tool calls in one turn"
                            )
                        if executable:
                            if hosted_rounds >= 8:
                                raise ToolContractError(
                                    "hosted tool execution exceeded 8 model/tool rounds"
                                )
                            message = {
                                "role": "assistant",
                                "content": "".join(parts),
                                "tool_calls": [
                                    {key: value for key, value in call.items() if key != "index"}
                                    for call in calls
                                ],
                            }
                            enforce_tool_contract(body, message["tool_calls"])
                            outputs = []
                            for call in message["tool_calls"]:
                                function = call["function"]
                                try:
                                    arguments = json.loads(function.get("arguments", "{}"))
                                except json.JSONDecodeError as error:
                                    raise ToolContractError(
                                        "hosted tool arguments are not valid JSON"
                                    ) from error
                                if not isinstance(arguments, dict):
                                    raise ToolContractError(
                                        "hosted tool arguments must be a JSON object"
                                    )
                                result = with_keepalive(
                                    lambda: tool_backend.execute(
                                        executors[function["name"]], arguments
                                    )
                                )
                                outputs.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": call.get("id"),
                                        "content": json.dumps(
                                            result, allow_nan=False, separators=(",", ":")
                                        ),
                                    }
                                )
                            body = {
                                **body,
                                "messages": [*body["messages"], message, *outputs],
                            }
                            if body.get("tool_choice") not in (None, "auto", "none"):
                                # ``required`` and a named choice bind the
                                # first round, whose executed calls met them
                                # above; a continuation must be free to answer.
                                body["tool_choice"] = "auto"
                            response_options.setdefault("hosted_messages", []).extend(
                                (message, *outputs)
                            )
                            hosted_rounds += 1
                            continuation_kwargs = {"tenant_id": tenant_id}
                            if admission_lease is not None:
                                continuation_kwargs["admitted"] = True
                            job = engine.submit(body, **continuation_kwargs)
                            parts, reasoning, calls, probabilities = [], [], [], []
                            text_logprobs = ResponsesTextLogprobs()
                            continue
                        if hosted_rounds:
                            # Every round's tokens, including its prefix-cache
                            # hits and reasoning, count toward the response.
                            usage = {
                                **usage,
                                "prompt_tokens": hosted_usage["prompt_tokens"],
                                "completion_tokens": hosted_usage["completion_tokens"],
                                "total_tokens": hosted_usage["prompt_tokens"]
                                + hosted_usage["completion_tokens"],
                                "prompt_tokens_details": {
                                    "cached_tokens": hosted_usage["cached_tokens"]
                                },
                                "completion_tokens_details": {
                                    "reasoning_tokens": hosted_usage["reasoning_tokens"]
                                },
                            }
                            receipt = {
                                **receipt,
                                "hosted_tools": {
                                    "rounds": hosted_rounds,
                                    "receipts": hosted_receipts,
                                },
                            }
                        if buffered_hosted_stream and not prologue_sent:
                            open_stream()
                            # The buffered message is opened, with its whole
                            # text as one delta, where the final payload
                            # places it: a reasoning item may precede it.
                        choice = {"index": 0, "finish_reason": event["finish_reason"]}
                        if buffered_tool_stream:
                            message = {"role": "assistant", "content": "".join(parts)}
                            if any(reasoning):
                                message["reasoning_content"] = "".join(reasoning)
                            if calls:
                                message["tool_calls"] = [
                                    {key: value for key, value in call.items() if key != "index"}
                                    for call in calls
                                ]
                            # The defining property of this path: validate the
                            # terminal contract before sending any call (and
                            # before committing HTTP bytes, unless a keepalive
                            # already opened the stream).
                            enforce_tool_contract(
                                body,
                                message.get("tool_calls", []),
                                finish_reason=event["finish_reason"],
                            )
                            if not streaming:
                                open_stream()
                            delta = {"role": "assistant"}
                            if message.get("content"):
                                delta["content"] = message["content"]
                            if message.get("reasoning_content"):
                                delta["reasoning_content"] = message["reasoning_content"]
                            if message.get("tool_calls"):
                                delta["tool_calls"] = [
                                    {**call, "index": index}
                                    for index, call in enumerate(message["tool_calls"])
                                ]
                            self._sse(
                                {
                                    "id": job.id,
                                    "object": "chat.completion.chunk",
                                    "created": int(job.created),
                                    "model": model,
                                    "choices": [
                                        {"index": 0, "finish_reason": None, "delta": delta}
                                    ],
                                }
                            )
                            self._sse(
                                {
                                    "id": job.id,
                                    "object": "chat.completion.chunk",
                                    "created": int(job.created),
                                    "model": model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "finish_reason": event["finish_reason"],
                                            "delta": {},
                                        }
                                    ],
                                    "usage": usage,
                                    "mlx2": receipt,
                                }
                            )
                            self._sse_usage_chunk(
                                job, body, model=model, usage=usage, chat=True
                            )
                            self._sse("[DONE]")
                            self._record_http(200)
                            if semantic_middleware is not None:
                                semantic_middleware.complete(
                                    semantic_state,
                                    message.get("content", ""),
                                    score_tokens=semantic_score_tokens,
                                )
                            return
                        if streaming:
                            if anthropic:
                                message = {
                                    "role": "assistant",
                                    "content": "".join(parts),
                                }
                                if any(reasoning):
                                    message["reasoning_content"] = "".join(reasoning)
                                if calls:
                                    message["tool_calls"] = [
                                        {key: value for key, value in call.items() if key != "index"}
                                        for call in calls
                                    ]
                                enforce_tool_contract(
                                    body,
                                    message.get("tool_calls", []),
                                    finish_reason=event["finish_reason"],
                                )
                                for final in anthropic_translator.finish(
                                    event["finish_reason"], usage, receipt
                                ):
                                    self._anthropic_sse(final)
                            elif responses_api:
                                message = {
                                    "role": "assistant",
                                    "content": "".join(parts),
                                }
                                if any(reasoning):
                                    message["reasoning_content"] = "".join(reasoning)
                                if calls:
                                    message["tool_calls"] = [
                                        {key: value for key, value in call.items() if key != "index"}
                                        for call in calls
                                    ]
                                enforce_tool_contract(
                                    body,
                                    message.get("tool_calls", []),
                                    finish_reason=event["finish_reason"],
                                )
                                choice["message"] = message
                                if wants_logprobs(body):
                                    choice["logprobs"] = {
                                        "content": text_logprobs.text
                                    }
                                    if (
                                        text_logprobs_trailing
                                        and response_message_started
                                    ):
                                        # The stop token's logprob follows
                                        # the text it ends.
                                        self._responses_sse(
                                            {
                                                "type": "response.output_text.delta",
                                                "item_id": f"msg_{job.id}",
                                                "output_index": response_message_output_index,
                                                "content_index": 0,
                                                "delta": "",
                                                "logprobs": [
                                                    responses_logprob(value)
                                                    for value in text_logprobs_trailing
                                                ],
                                            }
                                        )
                                payload = responses_payload(
                                    job=job,
                                    model=model,
                                    choice=choice,
                                    usage=usage,
                                    receipt=receipt,
                                    metadata=response_metadata,
                                    store=response_options["store"],
                                    previous_response_id=response_options[
                                        "previous_response_id"
                                    ],
                                    signer=getattr(engine, "reasoning_signer", None),
                                    tenant_id=tenant_id,
                                    include=response_options.get("include", ()),
                                    tool_choice=response_options.get("tool_choice"),
                                    agent_compat=agent_compat,
                                    compat_tool_map=response_options.get(
                                        "agent_compat"
                                    ),
                                    counts=getattr(engine, "counts", None),
                                    output_order=response_output_order,
                                )
                                if response_output_order or streamed_calls:
                                    streamed_output_indexes = dict(streamed_calls)
                                    if response_reasoning_output_index is not None:
                                        streamed_output_indexes[
                                            f"rs_{response_id(job).removeprefix('resp_')}"
                                        ] = response_reasoning_output_index
                                    if response_message_output_index is not None:
                                        streamed_output_indexes[
                                            f"msg_{job.id}"
                                        ] = response_message_output_index
                                    payload["output"].sort(
                                        key=lambda item: streamed_output_indexes.get(
                                            item["id"], float("inf")
                                        )
                                    )
                                store_response(
                                    tenant_id, response_options, payload, message
                                )
                                for output_index, item in enumerate(payload["output"]):
                                    if item["id"] in streamed_calls:
                                        pass  # opened on arrival; closed below
                                    elif item["type"] == "message":
                                        if not response_message_started:
                                            self._responses_message_start(
                                                job, output_index
                                            )
                                            response_message_started = True
                                            if message["content"]:
                                                # The one delta carries the whole
                                                # text, so it carries the whole
                                                # logprob list the item reports.
                                                self._responses_sse(
                                                    {
                                                        "type": "response.output_text.delta",
                                                        "item_id": item["id"],
                                                        "output_index": output_index,
                                                        "content_index": 0,
                                                        "delta": message["content"],
                                                        **(
                                                            {
                                                                "logprobs": item["content"][0][
                                                                    "logprobs"
                                                                ]
                                                            }
                                                            if "logprobs" in item["content"][0]
                                                            else {}
                                                        ),
                                                    }
                                                )
                                        self._responses_sse(
                                            {
                                                "type": "response.output_text.done",
                                                "item_id": item["id"],
                                                "output_index": output_index,
                                                "content_index": 0,
                                                "text": message["content"],
                                            }
                                        )
                                        self._responses_sse(
                                            {
                                                "type": "response.content_part.done",
                                                "item_id": item["id"],
                                                "output_index": output_index,
                                                "content_index": 0,
                                                "part": item["content"][0],
                                            }
                                        )
                                    elif item["type"] == "reasoning":
                                        reasoning_text = item["content"][0]["text"]
                                        if not response_reasoning_started:
                                            self._responses_reasoning_start(
                                                job, output_index
                                            )
                                            response_reasoning_started = True
                                            for event_type, index_name in (
                                                (
                                                    "response.reasoning_summary_text.delta",
                                                    "summary_index",
                                                ),
                                                (
                                                    "response.reasoning_text.delta",
                                                    "content_index",
                                                ),
                                            ):
                                                self._responses_sse(
                                                    {
                                                        "type": event_type,
                                                        "item_id": item["id"],
                                                        "output_index": output_index,
                                                        index_name: 0,
                                                        "delta": reasoning_text,
                                                    }
                                                )
                                        self._responses_sse(
                                            {
                                                "type": "response.reasoning_summary_text.done",
                                                "item_id": item["id"],
                                                "output_index": output_index,
                                                "summary_index": 0,
                                                "text": reasoning_text,
                                            }
                                        )
                                        self._responses_sse(
                                            {
                                                "type": "response.reasoning_text.done",
                                                "item_id": item["id"],
                                                "output_index": output_index,
                                                "content_index": 0,
                                                "text": reasoning_text,
                                            }
                                        )
                                    elif item["type"] == "custom_tool_call":
                                        self._responses_sse(
                                            {
                                                "type": "response.output_item.added",
                                                "output_index": output_index,
                                                "item": {
                                                    **item,
                                                    "status": "in_progress",
                                                    "input": "",
                                                },
                                            }
                                        )
                                        self._responses_sse(
                                            {
                                                "type": "response.custom_tool_call_input.delta",
                                                "item_id": item["id"],
                                                "output_index": output_index,
                                                "delta": item["input"],
                                            }
                                        )
                                        self._responses_sse(
                                            {
                                                "type": "response.custom_tool_call_input.done",
                                                "item_id": item["id"],
                                                "output_index": output_index,
                                                "input": item["input"],
                                            }
                                        )
                                    elif item["type"] == "function_call":
                                        pending = {
                                            **item,
                                            "status": "in_progress",
                                            "arguments": "",
                                        }
                                        self._responses_sse(
                                            {
                                                "type": "response.output_item.added",
                                                "output_index": output_index,
                                                "item": pending,
                                            }
                                        )
                                        self._responses_sse(
                                            {
                                                "type": "response.function_call_arguments.delta",
                                                "item_id": item["id"],
                                                "output_index": output_index,
                                                "delta": item["arguments"],
                                            }
                                        )
                                        self._responses_sse(
                                            {
                                                "type": "response.function_call_arguments.done",
                                                "item_id": item["id"],
                                                "output_index": output_index,
                                                "arguments": item["arguments"],
                                            }
                                        )
                                    else:
                                        self._responses_sse(
                                            {
                                                "type": "response.output_item.added",
                                                "output_index": output_index,
                                                "item": {
                                                    **item,
                                                    "status": "in_progress",
                                                    "summary": [],
                                                    **(
                                                        {"encrypted_content": None}
                                                        if "encrypted_content" in item
                                                        else {}
                                                    ),
                                                },
                                            }
                                        )
                                    self._responses_sse(
                                        {
                                            "type": "response.output_item.done",
                                            "output_index": output_index,
                                            "item": item,
                                        }
                                    )
                                self._responses_sse(
                                    {"type": "response.completed", "response": payload}
                                )
                            else:
                                if grammar_tool_stream:
                                    # Terminal contract before the final chunk;
                                    # a violation becomes an SSE error event.
                                    enforce_tool_contract(
                                        body,
                                        [
                                            {k: v for k, v in c.items() if k != "index"}
                                            for c in calls
                                        ],
                                        finish_reason=event["finish_reason"],
                                    )
                                choice.update(
                                    {"delta": self._chat_delta({})}
                                    if chat
                                    else {"text": ""}
                                )
                                self._sse(
                                    {
                                        "id": job.id,
                                        "object": "chat.completion.chunk"
                                        if chat
                                        else "text_completion",
                                        "created": int(job.created),
                                        "model": model,
                                        "choices": [choice],
                                        "usage": usage,
                                        "mlx2": receipt,
                                    }
                                )
                                self._sse_usage_chunk(
                                    job, body, model=model, usage=usage, chat=chat
                                )
                                self._sse("[DONE]")
                            self._record_http(200)
                        else:
                            if wants_logprobs(body):
                                choice["logprobs"] = {
                                    "content": text_logprobs.text
                                    if responses_api
                                    else probabilities
                                }
                            message = {"role": "assistant", "content": "".join(parts)}
                            if any(reasoning):
                                message["reasoning_content"] = "".join(reasoning)
                            if calls:
                                message["tool_calls"] = [
                                    {k: v for k, v in c.items() if k != "index"}
                                    for c in calls
                                ]
                            enforce_tool_contract(
                                body,
                                message.get("tool_calls", []),
                                finish_reason=event["finish_reason"],
                            )
                            choice.update(
                                {"message": message}
                                if chat
                                else {"text": "".join(parts)}
                            )
                            if anthropic:
                                result = {
                                    "id": job.id,
                                    "created": int(job.created),
                                    "model": model,
                                    "choices": [choice],
                                    "usage": usage,
                                    "mlx2": receipt,
                                }
                                self.send_json(
                                    200,
                                    chat_result_to_anthropic(
                                        result,
                                        anthropic_request,
                                        signer=getattr(
                                            engine, "reasoning_signer", None
                                        ),
                                        tenant_id=tenant_id,
                                    ),
                                )
                            elif responses_api:
                                payload = responses_payload(
                                    job=job,
                                    model=model,
                                    choice=choice,
                                    usage=usage,
                                    receipt=receipt,
                                    metadata=response_metadata,
                                    store=response_options["store"],
                                    previous_response_id=response_options[
                                        "previous_response_id"
                                    ],
                                    signer=getattr(engine, "reasoning_signer", None),
                                    tenant_id=tenant_id,
                                    include=response_options.get("include", ()),
                                    tool_choice=response_options.get("tool_choice"),
                                    agent_compat=agent_compat,
                                    compat_tool_map=response_options.get(
                                        "agent_compat"
                                    ),
                                    counts=getattr(engine, "counts", None),
                                )
                                store_response(
                                    tenant_id, response_options, payload, message
                                )
                                self.send_json(
                                    200,
                                    payload,
                                )
                            else:
                                self.send_json(
                                    200,
                                    {
                                        "id": job.id,
                                        "object": "chat.completion"
                                        if chat
                                        else "text_completion",
                                        "created": int(job.created),
                                        "model": model,
                                        "choices": [choice],
                                        "usage": usage,
                                        "mlx2": receipt,
                                    },
                                )
                        if semantic_middleware is not None:
                            semantic_middleware.complete(
                                semantic_state,
                                "".join(parts),
                                score_tokens=semantic_score_tokens,
                            )
                        return
            except ToolContractError as exc:
                if streaming and responses_api:
                    self._responses_failure(job, str(exc), "model_contract_error")
                    self._record_http(200)
                elif streaming and anthropic and anthropic_translator is not None:
                    for failure in anthropic_translator.failure(str(exc), 502):
                        self._anthropic_sse(failure)
                    self._record_http(200)
                elif streaming:
                    self._sse({"error": {"message": str(exc)}})
                    self._sse("[DONE]")
                    self._record_http(200)
                else:
                    self.api_error(502, str(exc), anthropic=anthropic)
            except ModelOutputError as exc:
                if streaming:
                    stream_failure(502, str(exc))
                else:
                    self.api_error(502, str(exc), anthropic=anthropic)
            except HostedToolError as exc:
                # The MCP server failed, not the request: a bad gateway.
                if streaming:
                    stream_failure(502, str(exc))
                else:
                    self.api_error(502, str(exc), anthropic=anthropic)
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                error_code = getattr(exc, "code", None)
                if getattr(exc, "schema_reference_error", False):
                    engine_counts = getattr(engine, "counts", None)
                    if engine_counts is not None:
                        engine_lock = getattr(engine, "lock", None)
                        if engine_lock is None:
                            engine_counts["schema_ref_failures"] += 1
                        else:
                            with engine_lock:
                                engine_counts["schema_ref_failures"] += 1
                if streaming and anthropic and anthropic_translator is not None:
                    for failure in anthropic_translator.failure(str(exc), 400):
                        self._anthropic_sse(failure)
                    self._record_http(200)
                elif streaming and responses_api:
                    self._responses_failure(
                        job, str(exc), "invalid_response", code=error_code
                    )
                    self._record_http(200)
                elif streaming:
                    error = {"message": str(exc)}
                    if error_code is not None:
                        error["code"] = error_code
                    self._sse({"error": error})
                    self._sse("[DONE]")
                    self._record_http(200)
                else:
                    self.api_error(
                        400,
                        str(exc),
                        anthropic=anthropic,
                        code=error_code,
                    )
            except PromptTemplateFailure as exc:
                # A template that fails for a reason the request does not
                # explain is a server fault, but the caller is told which
                # stage failed rather than reading an unhandled TypeError.
                logging.getLogger("mlx2.server").exception(
                    "chat template rendering failed",
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
                if streaming and anthropic and anthropic_translator is not None:
                    for failure in anthropic_translator.failure(str(exc), 500):
                        self._anthropic_sse(failure)
                    self._record_http(200)
                elif streaming and responses_api:
                    self._responses_failure(job, str(exc), "server_error")
                    self._record_http(200)
                elif streaming:
                    self._sse({"error": {"message": str(exc)}})
                    self._sse("[DONE]")
                    self._record_http(200)
                else:
                    self.api_error(500, str(exc), anthropic=anthropic)
            except ResourceNotFound as exc:
                if streaming:
                    stream_failure(404, str(exc).strip("'"))
                else:
                    self.api_error(404, str(exc).strip("'"), anthropic=anthropic)
            except CapabilityUnavailable as exc:
                if streaming:
                    stream_failure(501, str(exc))
                else:
                    self.api_error(501, str(exc), anthropic=anthropic)
            except Overloaded as exc:
                if streaming:
                    stream_failure(429, str(exc))
                else:
                    self.api_error(429, str(exc), anthropic=anthropic)
            except AdmissionClosed as exc:
                if streaming:
                    stream_failure(503, str(exc))
                else:
                    self.api_error(
                        503,
                        str(exc),
                        anthropic=anthropic,
                        retry_after=True,
                    )
            except SampleFailed as exc:
                # Parallel samples report the engine's own status, exactly as
                # the single-sample path does for its terminal error event.
                if streaming:
                    stream_failure(exc.status, str(exc), code=exc.code)
                else:
                    self.api_error(
                        exc.status,
                        str(exc),
                        anthropic=anthropic,
                        mlx2=exc.mlx2,
                        code=exc.code,
                    )
            except ClientGone:
                engine_counts = getattr(engine, "counts", None)
                if engine_counts is not None:
                    engine_counts["client_disconnects"] += 1
                self.close_connection = True
                self._record_http(499)
            except (BrokenPipeError, ConnectionResetError):
                # A streaming client that goes away mid-stream is the same
                # disconnect as ClientGone and is counted like it.
                engine_counts = getattr(engine, "counts", None)
                if engine_counts is not None:
                    engine_counts["client_disconnects"] += 1
                self._record_http(499)
            except (RuntimeError, TimeoutError) as exc:
                if streaming:
                    stream_failure(503, str(exc))
                else:
                    self.api_error(503, str(exc), anthropic=anthropic)
            except Exception as exc:
                # A fault the handler does not classify must still answer and
                # be counted; an escaping exception drops the connection with
                # no response and no HTTP metric.  The detail stays in the log.
                logging.getLogger("mlx2.server").exception(
                    "request failed unexpectedly",
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
                message = "internal server error"
                if streaming and anthropic and anthropic_translator is not None:
                    for failure in anthropic_translator.failure(message, 500):
                        self._anthropic_sse(failure)
                    self._record_http(200)
                elif streaming and responses_api:
                    self._responses_failure(job, message, "server_error")
                    self._record_http(200)
                elif streaming:
                    self._sse({"error": {"message": message}})
                    self._sse("[DONE]")
                    self._record_http(200)
                else:
                    self.api_error(500, message, anthropic=anthropic)
            finally:
                if job is not None:
                    job.cancelled.set()
                for sample_job in jobs:
                    sample_job.cancelled.set()
                release_admission = getattr(engine, "release_admission", None)
                if callable(release_admission):
                    release_admission(admission_lease)

        def _sse(self, value):
            data = (
                value if isinstance(value, str) else json.dumps(value, allow_nan=False)
            )
            self.wfile.write(f"data: {data}\n\n".encode())
            self.wfile.flush()
            self._last_stream_write = time.monotonic()

        def _chat_delta(self, delta):
            """Name the assistant role on the first content (or final) delta.

            Progress and logprob-only chunks keep their empty delta.
            """
            if not self._chat_role_pending:
                return delta
            self._chat_role_pending = False
            return {"role": "assistant", **delta}

        def _sse_usage_chunk(self, job, body, *, model, usage, chat):
            """OpenAI ``stream_options.include_usage``: a choice-less usage chunk.

            Usage also stays on the finish chunk, where earlier mlx2 clients
            read it; this chunk is the standard place for OpenAI clients.
            """
            if not (body.get("stream_options") or {}).get("include_usage"):
                return
            self._sse(
                {
                    "id": job.id,
                    "object": "chat.completion.chunk" if chat else "text_completion",
                    "created": int(job.created),
                    "model": model,
                    "choices": [],
                    "usage": usage,
                }
            )

        def _responses_call_stream(self, job, call, streamed, *, offset):
            """Item 12: open one grammar-engaged function_call item early.

            ``output_item.done`` is withheld until the terminal tool contract
            has passed; ``streamed`` maps item ids to the index sent.
            """
            index = len(streamed)
            function = call["function"]
            item = {
                "id": call.get("id", f"fc_{job.id}_{index}"),
                "type": "function_call",
                "status": "in_progress",
                "call_id": call.get("id", f"call_{job.id}_{index}"),
                "name": function["name"],
                "arguments": "",
            }
            output_index = offset + index
            streamed[item["id"]] = output_index
            self._responses_sse(
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": item,
                }
            )
            self._responses_sse(
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": item["id"],
                    "output_index": output_index,
                    "delta": function["arguments"],
                }
            )
            self._responses_sse(
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": item["id"],
                    "output_index": output_index,
                    "arguments": function["arguments"],
                }
            )

        def _responses_reasoning_start(self, job, output_index):
            self._responses_sse(
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": {
                        "id": f"rs_{response_id(job).removeprefix('resp_')}",
                        "type": "reasoning",
                        "status": "in_progress",
                        "summary": [],
                        "content": [],
                    },
                }
            )

        def _responses_message_start(self, job, output_index):
            self._responses_sse(
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": {
                        "id": f"msg_{job.id}",
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    },
                }
            )
            self._responses_sse(
                {
                    "type": "response.content_part.added",
                    "item_id": f"msg_{job.id}",
                    "output_index": output_index,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "text": "",
                        "annotations": [],
                    },
                }
            )

        def _responses_failure(self, job, message, error_type, *, code=None):
            # A keepalive may have opened a hosted-tool stream before its
            # response.created; clients expect that event first.
            pending = getattr(self, "_pending_responses_prologue", None)
            if pending is not None:
                pending()
            error = {"message": message, "type": error_type}
            if code is not None:
                error["code"] = code
            self._responses_sse(
                {
                    "type": "response.failed",
                    "response": {
                        "id": response_id(job) if job is not None else None,
                        "object": "response",
                        "status": "failed",
                        "error": error,
                    },
                }
            )

        def _responses_sse(self, value):
            value = {**value, "sequence_number": self._responses_sequence}
            self._responses_sequence += 1
            data = json.dumps(value, allow_nan=False)
            self.wfile.write(f"event: {value['type']}\ndata: {data}\n\n".encode())
            self.wfile.flush()
            self._last_stream_write = time.monotonic()

        def _anthropic_sse(self, value):
            data = json.dumps(value, allow_nan=False)
            self.wfile.write(f"event: {value['type']}\ndata: {data}\n\n".encode())
            self.wfile.flush()
            self._last_stream_write = time.monotonic()

    return Handler


def build_tenant_authenticator(args):
    """Return the configured TenantAuthenticator, or None when auth is off."""
    from .tenant_auth import configured

    enabled = bool(
        args.tenant_auth_keys_file
        or args.tenant_auth_token_secret_file
        or args.tenant_auth_token_secret_env
    )
    if not enabled:
        if args.tenant_auth_allow_shared_cache:
            raise ValueError("--tenant-auth-allow-shared-cache requires tenant auth")
        if not is_loopback_address(args.host) and args.host != "localhost":
            logging.getLogger("mlx2.tenant_auth").warning(
                "serving on non-loopback host %s without tenant auth: "
                "X-Tenant-ID is unauthenticated",
                args.host,
            )
        return None
    if not args.tenant_scoped_cache and not args.tenant_auth_allow_shared_cache:
        raise ValueError(
            "tenant auth requires --tenant-scoped-cache (or explicitly "
            "--tenant-auth-allow-shared-cache)"
        )
    if args.api_key_file or args.api_key_env:
        raise ValueError(TENANT_AUTH_GATE_KEY_CONFLICT)
    return configured(
        keys_file=args.tenant_auth_keys_file,
        token_secret_file=args.tenant_auth_token_secret_file,
        token_secret_env=args.tenant_auth_token_secret_env,
        header_policy=args.tenant_header_policy,
        token_max_ttl=args.tenant_auth_token_max_ttl,
        cache_isolation="tenant" if args.tenant_scoped_cache else "shared",
    )


DEFAULT_MAX_LANES = 16
DEFAULT_MAX_INFLIGHT = 32


class _ExplicitCacheBytes(argparse.Action):
    """Record that --cache-bytes was given, so the post-load clamp skips it."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        namespace.cache_bytes_source = "explicit"


def build_parser():
    parser = argparse.ArgumentParser(description="mlx2 APCv2 Flash-Next server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8285)
    parser.add_argument(
        "--admin-token-file",
        help="owner-only 0600 file containing the loopback admin bearer token",
    )
    tenant_auth = parser.add_argument_group(
        "tenant authentication",
        "opt-in: derive the tenant from a verified credential instead of "
        "X-Tenant-ID; unauthenticated requests fail closed with 401/403",
    )
    tenant_auth.add_argument(
        "--tenant-auth-keys-file",
        help="JSON keys file mapping SHA-256 API-key digests to tenant ids",
    )
    tenant_secret = tenant_auth.add_mutually_exclusive_group()
    tenant_secret.add_argument(
        "--tenant-auth-token-secret-file",
        help="owner-only 0600 file with the HMAC secret for mlx2t1 tenant tokens",
    )
    tenant_secret.add_argument(
        "--tenant-auth-token-secret-env",
        help="environment variable holding the HMAC secret for tenant tokens",
    )
    tenant_auth.add_argument(
        "--tenant-header-policy",
        choices=("must-match", "ignore"),
        default="must-match",
        help="authenticated mode: a present X-Tenant-ID must match (403 "
        "otherwise) or is ignored",
    )
    tenant_auth.add_argument(
        "--tenant-auth-token-max-ttl",
        type=int,
        default=7 * 24 * 3600,
        metavar="SECONDS",
        help="reject tenant tokens whose exp - iat exceeds this",
    )
    tenant_auth.add_argument(
        "--tenant-auth-allow-shared-cache",
        action="store_true",
        help="permit tenant auth without --tenant-scoped-cache (a shared "
        "prefix cache leaks prompts across tenants through TTFT/cached_tokens)",
    )
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "extra Host header name accepted (any port; repeatable). Loopback "
            "binds always accept localhost, 127.0.0.1 and [::1]; a non-loopback "
            "bind checks Host only when at least one --allowed-host is given"
        ),
    )
    api_key = parser.add_mutually_exclusive_group()
    api_key.add_argument(
        "--api-key-file",
        help=(
            "owner-only 0600 file with the API key required on every route "
            "except GET /health (Authorization: Bearer or x-api-key)"
        ),
    )
    api_key.add_argument(
        "--api-key-env",
        metavar="NAME",
        help="environment variable holding the API key (e.g. MLX2_API_KEY)",
    )
    parser.add_argument(
        "--drain-on-sigterm",
        type=float,
        metavar="SECONDS",
        help="drain accepted work before SIGTERM/SIGINT shutdown (default: immediate)",
    )
    parser.add_argument(
        "--sse-keepalive-seconds",
        type=float,
        default=DEFAULT_SSE_KEEPALIVE_SECONDS,
        metavar="SECONDS",
        help=(
            "send an SSE keepalive on a stream silent this long, opening it "
            "once the request is admitted (default: %(default)s; 0 disables)"
        ),
    )
    parser.add_argument(
        "--fault-log",
        type=Path,
        metavar="PATH",
        default=os.environ.get("MLX2_FAULT_LOG") or None,
        help="append faulthandler stack dumps for fatal signals (SIGSEGV, "
        "SIGBUS, SIGABRT, ...) to PATH instead of stderr (env MLX2_FAULT_LOG)",
    )
    parser.add_argument("--max-context", type=int, default=262144)
    parser.add_argument(
        "--prefill-step", type=prefill_step_arg,
        help=(
            "prefill chunk step; explicit values override adapter policy, while "
            "models without an adapter override autoscale from prompt length; "
            "the resolved policy is bound to route qualification"
        ),
    )
    parser.add_argument(
        "--prefill-depth-budget", type=prefill_step_arg,
        help="shrink a prefill chunk once rows x (KV depth + rows) would exceed "
        "this many tokens squared, so deep prefill stays under the Metal "
        "watchdog (default: the adapter's, normally off); changes output bits "
        "of the chunks it shrinks and is bound to route qualification",
    )
    parser.add_argument(
        "--default-max-tokens",
        type=default_max_tokens_arg,
        default=DEFAULT_OUTPUT_TOKENS,
        help=(
            "output default when a request omits its cap, clamped to remaining "
            "context after tokenization (default: 65536)"
        ),
    )
    parser.add_argument(
        "--max-request-bytes",
        type=int,
        help=(
            "maximum JSON request body bytes; defaults to a bounded value "
            "derived from --max-context"
        ),
    )
    parser.add_argument("--execution-policy", type=Path, help="JSON file with adapter execution choices; included in qualification identity")
    # The geometry every handoff qualification ran at (settings in
    # qualification/runs/mtp-handoff-20260920/*-qualification.json).  The
    # MTP->ordinary handoff fires only above max_mtp_width 4, so the old
    # default of 4 lanes made it, and every B8/B16 number, unreachable from a
    # bare start.  Memory admission still bounds the real width per host.
    parser.add_argument(
        "--max-lanes",
        type=int,
        default=DEFAULT_MAX_LANES,
        help=f"decode lanes (default: {DEFAULT_MAX_LANES}; admission bounds real width)",
    )
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=DEFAULT_MAX_INFLIGHT,
        help=f"queued plus active requests (default: {DEFAULT_MAX_INFLIGHT})",
    )
    parser.add_argument(
        "--coalesce-window-ms",
        type=float,
        default=5.0,
        help="idle-to-active wait for B1-B4 admission (default: 5 ms)",
    )
    parser.add_argument(
        "--gpu-keep-warm-seconds",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help=(
            "after a request, keep the GPU out of its idle power state for "
            "this long by submitting a one-element kernel from the idle "
            "worker loop, so the next request skips the wake-up stall "
            "(default: %(default)s; 0 disables)"
        ),
    )
    parser.add_argument(
        "--gpu-keep-warm-interval",
        type=float,
        default=0.5,
        metavar="SECONDS",
        help="seconds between keep-warm ticks (default: %(default)s)",
    )
    parser.add_argument(
        "--power-telemetry",
        action="store_true",
        help=(
            "sample GPU/DRAM/CPU/ANE power, GPU P-state residency and die "
            "temperatures without privilege (IOReport) on a host thread; "
            "publishes /metrics power families, /v1/status.power and a "
            "per-request receipt energy estimate (default: off)"
        ),
    )
    parser.add_argument(
        "--power-telemetry-interval",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="seconds between power samples (default: %(default)s)",
    )
    parser.add_argument(
        "--power-telemetry-window",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help=(
            "rolling window for joules per output token and tokens/s "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--power-governor",
        choices=("off", "max_throughput", "efficient", "budget"),
        default="off",
        help=(
            "steer GPU+DRAM power by shaping work (requires --power-telemetry): "
            "max_throughput changes nothing but enables the runtime control "
            "endpoint POST /v1/admin/power; efficient holds the idle admission "
            "window for wider cohorts; budget holds rolling power under "
            "--power-budget-watts by pacing rounds, then capping lanes "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--power-budget-watts",
        type=float,
        default=None,
        metavar="WATTS",
        help="GPU+DRAM average power cap for --power-governor budget",
    )
    parser.add_argument(
        "--power-budget-window",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="window the power budget is averaged over (default: %(default)s)",
    )
    parser.add_argument(
        "--power-governor-hold-ms",
        type=float,
        default=50.0,
        metavar="MS",
        help=(
            "efficient/budget modes: hold the idle admission window this long "
            "so near-simultaneous arrivals start as one cohort; 0 disables "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--batch-cohort-timeout-ms",
        type=float,
        default=1000.0,
        help="deadline for a declared atomic HTTP batch cohort (default: 1000 ms)",
    )
    parser.add_argument(
        "--cache-bytes",
        type=int,
        default=default_cache_bytes(),
        action=_ExplicitCacheBytes,
        help=(
            "APCv2 resident prefix-cache cap (default: physical/8 up to a "
            "64 GiB host, then 5/8 of each further GiB, capped at 96 GiB = "
            f"{default_cache_bytes() / (1 << 30):.2f} GiB here; after the "
            "model loads the default is clamped to the headroom beside it, "
            "never below min(16 GiB, physical/8). An explicit value is used "
            "exactly as given)"
        ),
    )
    parser.set_defaults(cache_bytes_source="host_default")
    parser.add_argument("--cache-dir")
    parser.add_argument(
        "--apc-persist-dir",
        help="private 0700 APCv2 disk tier retained and rescanned across restarts",
    )
    parser.add_argument("--apc-persist-on-shutdown", action="store_true")
    parser.add_argument(
        "--apc-persist-shutdown-seconds", type=float, default=5.0
    )
    parser.add_argument(
        "--apc-persist-corruption",
        choices=("quarantine", "delete"),
        default="quarantine",
    )
    parser.add_argument("--apc-session-max-ttl-seconds", type=int, default=3600)
    parser.add_argument(
        "--apc-session-pinned-disk-bytes", type=int, default=64 << 30
    )
    parser.add_argument(
        "--apc-session-pinned-disk-bytes-global", type=int, default=64 << 30
    )
    parser.add_argument(
        "--apc-session-pinned-resident-bytes", type=int, default=12 << 30
    )
    parser.add_argument(
        "--apc-session-prefetch-ttl-seconds", type=int, default=30
    )
    parser.add_argument("--apc-quarantine-max-entries", type=int, default=128)
    parser.add_argument("--apc-quarantine-max-bytes", type=int, default=1 << 30)
    parser.add_argument(
        "--api-state-dir",
        help=(
            "directory for durable tenant-scoped Responses, Files, and Batch "
            "state (default: process-local state)"
        ),
    )
    parser.add_argument(
        "--semantic-memory",
        action="store_true",
        help=(
            "enable automatic capsule-backed concept memory for session requests; "
            "durable writes require configured tenant authentication"
        ),
    )
    parser.add_argument(
        "--semantic-memory-dir",
        help=(
            "owner-private semantic capsule root (default: semantic-memory below "
            "--api-state-dir or --apc-persist-dir)"
        ),
    )
    parser.add_argument(
        "--semantic-retrieval-limit",
        type=int,
        default=8,
        help="maximum concept tokens injected per request (1..32)",
    )
    parser.add_argument(
        "--semantic-bridge",
        choices=("rendered", "neural", "hybrid"),
        default="rendered",
        help=(
            "concept read bridge: rendered text baseline, learned neural prefill "
            "cross-attention, or both (default: rendered)"
        ),
    )
    parser.add_argument(
        "--neural-concept-artifact",
        help="trained, identity-bound recurrent concept bridge artifact directory",
    )
    parser.add_argument(
        "--lora-dir",
        help="allowlisted directory containing dynamically loadable LoRA adapters",
    )
    parser.add_argument(
        "--max-loras",
        type=int,
        default=0,
        help=(
            "concurrent multi-LoRA: resident adapter slots served in one mixed "
            "batch, selected per request by model=<lora_name> (default 0: off; "
            "ordinary route only; requires --lora-dir)"
        ),
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=16,
        help="concurrent multi-LoRA: padded slot rank (adapters above it are refused)",
    )
    parser.add_argument(
        "--tool-backend-config",
        type=Path,
        help="JSON allowlist for Responses MCP Streamable HTTP execution",
    )
    parser.add_argument(
        "--agent-compat",
        nargs="?",
        const="on",
        default="auto",
        choices=("off", "on", "opt-in", "auto"),
        help=(
            "coding-agent wire translation (Codex custom/namespace tools, "
            "message phase, hosted-tool drop; Claude Code adaptive thinking, "
            "output_config, context_management, mid-conversation system). "
            "auto (default): X-MLX2-Agent-Compat header > tenant policy > "
            "client detection; opt-in: header > tenant policy; on: always "
            "unless the header says off; off: never. A bare --agent-compat "
            "means on."
        ),
    )
    parser.add_argument(
        "--agent-compat-tenants",
        help=(
            "JSON file mapping tenant id to {agent_compat: on|off, optional "
            "custom_tool_grammar: validate|off}"
        ),
    )
    parser.add_argument(
        "--custom-tool-grammar",
        choices=("off", "validate"),
        default="off",
        help=(
            "default custom-tool grammar mode for agent-compat requests "
            "(X-MLX2-Custom-Tool-Grammar header > tenant policy > this); "
            "'validate' fails closed on non-matching inputs"
        ),
    )
    signing = parser.add_mutually_exclusive_group()
    signing.add_argument("--reasoning-signing-key-file")
    signing.add_argument(
        "--reasoning-signing-key-env",
        help="environment variable containing the shared reasoning signing secret",
    )
    signing.add_argument(
        "--reasoning-signing-ephemeral",
        action="store_true",
        help=(
            "use a random per-process reasoning signing key even with "
            "--api-state-dir (default there: <state>/reasoning-signing.key)"
        ),
    )
    parser.add_argument(
        "--persistent-block-bytes",
        type=int,
        default=0,
        help="qualification-only APCv2 persistent block size (0 disables)",
    )
    parser.add_argument(
        "--cache-capsules",
        action="store_true",
        help="qualification-only generation-safe warm fanout capsules",
    )
    parser.add_argument(
        "--cache-capsule-backend",
        choices=("cpu", "gpu"),
        default="gpu",
    )
    parser.add_argument("--cache-capsule-deadline-ms", type=float, default=50.0)
    parser.add_argument(
        "--host-prompt-cache-entries",
        type=int,
        default=128,
        help="maximum in-process rendered/tokenized prompts (0 disables)",
    )
    parser.add_argument(
        "--host-prompt-cache-tokens",
        type=int,
        default=1 << 20,
        help="maximum token IDs retained by the host prompt cache (0 disables)",
    )
    parser.add_argument(
        "--incremental-tokenizer-cache-entries",
        type=int,
        default=0,
        help=(
            "unqualified revision-bound incremental tokenizer entries; "
            "0 keeps the candidate off"
        ),
    )
    parser.add_argument(
        "--incremental-tokenizer-cache-characters",
        type=int,
        default=8 << 20,
        help="maximum rendered characters retained by the incremental tokenizer",
    )
    parser.add_argument(
        "--incremental-tokenizer-cache-tokens",
        type=int,
        default=1 << 20,
        help="maximum token IDs retained by the incremental tokenizer",
    )
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--ordinary", action="store_true", help="ordinary decode reference with APCv2")
    execution.add_argument(
        "--native-mtp",
        action="store_true",
        help="native self-MTP route; fails if the resolved artifact has no MTP head",
    )
    execution.add_argument("--external-draft", action="store_true", help="external draft/verify; requires draft_model execution policy")
    execution.add_argument(
        "--prompt-lookup",
        action="store_true",
        help="target-verified indexed prompt-lookup route",
    )
    owned = parser.add_argument_group("experimental TensorFold-owned token-ID route")
    owned.add_argument("--tensorfold-owned-live", action="store_true",
                       help="enable unqualified, default-off native worker endpoint")
    owned.add_argument("--tensorfold-owned-source")
    owned.add_argument("--tensorfold-owned-mlx-lm-source")
    owned.add_argument("--tensorfold-owned-drafter")
    parser.add_argument(
        "--tenant-scoped-cache",
        action="store_true",
        help=(
            "bind the APCv2 prefix cache to X-Tenant-ID so clients cannot warm-hit "
            "or probe another tenant's prompts; off by default for the shared lab cache"
        ),
    )
    parser.add_argument(
        "--adaptive-mtp-depth",
        action="store_true",
        help=(
            "cohort-wide adaptive native-MTP depth; exact, so it may run "
            "unqualified (labelled so) without a matching receipt "
            "(default: disabled)"
        ),
    )
    parser.add_argument(
        "--mtp-acceptance-log",
        type=Path,
        help=(
            "qualification-only JSONL of per-draft-position features and "
            "acceptance labels (default: disabled)"
        ),
    )
    parser.add_argument(
        "--mtp-acceptance-log-lookahead",
        type=int,
        default=0,
        help=(
            "unverified greedy lookahead drafts per cycle, so the acceptance "
            "log observes positions past the served depth (default: 0). "
            "Requires --mtp-acceptance-log; the drafts are dropped before "
            "verification, so output is unchanged"
        ),
    )
    parser.add_argument(
        "--spomin-live-surgery",
        type=json.loads,
        metavar="JSON",
        help=(
            "qualification-only approximate prompt compaction policy, e.g. "
            '\'{"enabled": true, "capacity_tokens": 32768}\' '
            "(default: disabled; ordinary route only)"
        ),
    )
    parser.add_argument(
        "--approximate-kv",
        metavar="JSON",
        help=(
            "approximate KV policy mapping, for example "
            '\'{"operation": "kv_k8v4", "enabled": true}\'; ordinary route '
            "only (default: disabled)"
        ),
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help=(
            "reasoning-token budget at medium effort for models that think; unset uses the "
            "model adapter's default (North-Mini-Code: 512), 0 disables "
            "(scaled by reasoning_effort: low 0.375x, high 4x; 0 disables). A run-on "
            "alarm or the budget ramps a bias onto the thinking-close token; the "
            "channel is closed outright only at the budget itself"
        ),
    )
    parser.add_argument(
        "--thinking-steer-alpha",
        type=float,
        default=None,
        help=(
            "strength of the adapter-calibrated commit-direction steering; unset uses the model "
            "adapter's default (North-Mini-Code: 0.0), 0 disables. Applied to the "
            "residual stream while a lane is reasoning (ordinary route only; "
            "North-Mini-Code keeps it off because it lengthened reasoning on the "
            "corrected body)"
        ),
    )
    parser.add_argument(
        "--thinking-steer-hammer",
        type=float,
        default=None,
        help="stronger steering strength used once the run-on alarm or soft budget has tripped (0 keeps alpha)",
    )
    parser.add_argument(
        "--lane-matmul",
        choices=("auto", "off", "crossover", "exact"),
        default="auto",
        help=(
            "row-invariant lane matmul for small multi-row projection calls "
            "(runtime/lane). auto: per-format thresholds detected from the model "
            "(dense on, mixture-of-experts off); crossover: force on with the "
            "policy's thresholds; exact: every 1..max_rows call; off: stock only"
        ),
    )
    parser.add_argument(
        "--lane-policy",
        default=None,
        help=(
            "JSON object (inline or a file path) overriding lane defaults: mode, "
            "min_rows per format (q2..q8, bf16, fp16), max_rows, grouping, skip "
            "name patterns, and moe (overrides applied to mixture-of-experts models)"
        ),
    )
    parser.add_argument(
        "--no-thinking-auto-calibration",
        action="store_true",
        help=(
            "never calibrate a commit direction at startup; steering then needs a stored "
            "calibration bound to this exact artifact (default: calibrate when steering is "
            "wanted and none is bound, and keep steering off if the result fails its gates)"
        ),
    )
    parser.add_argument(
        "--int8-prefill",
        choices=("off", "mlp", "all"),
        default="off",
        help=(
            "W8A8 int8 NAX prefill on M5-class GPUs for prefill-sized calls "
            "(rows >= 512): 'mlp' = dense/shared-expert MLPs, 'all' adds "
            "attention projections.  Approximate fidelity; the adapter must "
            "declare the scope; separate APCv2 namespace (default: off)"
        ),
    )
    parser.add_argument(
        "--recurrent-state-codec",
        choices=("off", "int8-row-v1"),
        default="off",
        help=(
            "store recurrent (GDN/SSM) state in APCv2 entries as int8 with one "
            "scale per row; restores decode to float32.  Live state and "
            "rollback records stay exact.  Approximate and unqualified: "
            "qualification mode only; separate APCv2 namespace (default: off)"
        ),
    )
    parser.add_argument(
        "--verify-bitexact",
        action="store_true",
        help=(
            "batch-invariant quantized matmuls: every verify/decode row gets "
            "the single-row (M=1) qmv arithmetic, so greedy output does not "
            "depend on how many lanes share a verify call.  Needs an mlx "
            "build with mx.metal.set_qmv_bitexact; slower at 4 lanes "
            "(bypasses NAX/split-K); requests may then set "
            "verify_bitexact=true (default: off)"
        ),
    )
    parser.add_argument("--native-packed-prefill-profile", help="explicit default-off source-bound short research serving profile")
    parser.add_argument("--native-packed-prefill-manifest", help="pinned artifact manifest for packed-prefill research")
    parser.add_argument("--native-packed-prefill-mlx-wheel", help="pinned MLX wheel for packed-prefill research")
    parser.add_argument("--qualification-mode", action="store_true")
    parser.add_argument("--qualification")
    parser.add_argument(
        "--otlp-traces-endpoint",
        help=(
            "optional OTLP/HTTP traces endpoint; enables lazy W3C request "
            "tracing with asynchronous export"
        ),
    )
    return parser


def approximate_kv_mode(args, native_mtp):
    """Parse ``--approximate-kv`` and refuse unqualified or speculative use."""
    raw = getattr(args, "approximate_kv", None)
    if raw is None:
        return None
    from .runtime.approximate_kv import ServingApproximateKVPolicy

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"--approximate-kv must be a JSON object: {error}") from error
    policy = ServingApproximateKVPolicy.from_value(value)
    if policy.enabled:
        if not args.qualification_mode and not args.qualification:
            raise ValueError(
                "--approximate-kv requires --qualification-mode or a "
                "--qualification record carrying its evidence"
            )
        if args.external_draft or getattr(args, "prompt_lookup", False):
            raise ValueError(
                "--approximate-kv requires the --ordinary route or native MTP "
                "with compose_mtp"
            )
        if native_mtp and not policy.compose_mtp:
            raise ValueError(
                "--approximate-kv on native MTP requires \"compose_mtp\": true "
                "(target-only quantization); otherwise use --ordinary"
            )
        if policy.compose_mtp and not native_mtp:
            raise ValueError("--approximate-kv compose_mtp requires native MTP")
    return policy.as_dict()


@dataclass(frozen=True)
class RouteSelection:
    route: str
    source: str

    @property
    def native_mtp(self):
        return self.route == "native_mtp"


def resolve_route_selection(args, policy, adapter_resolution=None):
    """Resolve explicit intent or an adapter default before weights are loaded."""
    external_policy = bool((policy or {}).get("draft_model"))
    if args.ordinary and external_policy:
        raise ValueError("--ordinary cannot select an external draft policy")
    if args.external_draft and not external_policy:
        raise ValueError("--external-draft requires an execution policy with draft_model")
    if external_policy and not args.external_draft:
        raise ValueError("External draft policy requires --external-draft")
    prompt_lookup = bool(getattr(args, "prompt_lookup", False))
    if prompt_lookup and external_policy:
        raise ValueError("--prompt-lookup cannot select an external draft policy")
    native_mtp = bool(getattr(args, "native_mtp", False))
    if args.ordinary:
        route = "ordinary"
    elif args.external_draft:
        route = "external_draft"
    elif prompt_lookup:
        route = "prompt_lookup"
    elif native_mtp:
        route = "native_mtp"
    else:
        route = (
            adapter_resolution.default_route
            if adapter_resolution is not None
            else "native_mtp"
        )
    source = (
        "explicit_flag"
        if args.ordinary or args.external_draft or prompt_lookup or native_mtp
        else "adapter_default"
    )
    if route == "native_mtp" and adapter_resolution is not None:
        from .contracts import Capability

        if Capability.MTP not in adapter_resolution.descriptor.capabilities:
            prefix = "--native-mtp requested, but" if native_mtp else "adapter default selected"
            raise ValueError(
                f"{prefix} {adapter_resolution.descriptor.family} artifact has "
                "no implemented native MTP route"
            )
    return RouteSelection(route, source)


# SRPT prefill order with the bypass cap, default on for every native
# self-MTP route.  Measured on Qwen3.6 35B native MTP
# (qualification/runs/srpt-prefill-20260920/mtp-{off,on}.bench.json): short
# request TTFT p50 0.95 -> 0.51 s and max 1.27 -> 0.73 s with the long
# request unchanged (8.96 -> 8.92 s).  The ordinary route stays FIFO: there
# p50 moved 1.88 -> 1.85 s but the max rose 1.95 -> 2.36 s (ord-*.bench.json).
NATIVE_MTP_PREFILL_SCHEDULING = {
    "order": "srpt",
    "max_bypass": 3,
    "one_slice_contention": True,
}

NATIVE_MTP_COPY_DRAFT_DEFAULT = {
    "enabled": True,
    "ngram_min": 3,
    "ngram_max": 6,
    "lookback": 256,
    "min_match": 4,
    # The existing cross-model safety gate: do not widen a physical cohort.
    "batched_max_span": 0,
}


def resolve_execution_policy_defaults(
    policy,
    route_selection,
    adapter_resolution,
    *,
    approximate_kv=False,
    max_lanes=None,
    skipped=None,
):
    """Apply adapter defaults only after the serving route is known.

    A default fills only a key the operator's policy does not name: any
    explicit value, including ``null`` or a disabled object, wins.  State
    checkpoint defaults are skipped when approximate KV was requested, because
    the engine refuses them there and a default must never stop a startup
    that would otherwise succeed.  A default that cannot engage at
    ``max_lanes`` steps aside; when ``skipped`` is a dict, each such default
    is recorded there as ``{key: reason}`` for the engine's settings.
    """
    from .adapters.registry import STATE_CHECKPOINT_POLICY_KEYS

    resolved = {} if policy is None else dict(policy)
    if (
        route_selection.native_mtp
        and "mtp_ordinary_handoff" not in resolved
        and adapter_resolution is not None
    ):
        handoff = adapter_resolution.default_mtp_ordinary_handoff
        if (
            handoff is not None
            and max_lanes is not None
            and max_lanes <= handoff["max_mtp_width"]
        ):
            # The handoff fires only when the cohort grows past its width, so
            # at --max-lanes <= width it can never engage.  Selecting it would
            # only demand qualification evidence no run can produce.
            if skipped is not None:
                skipped["mtp_ordinary_handoff"] = (
                    f"max_lanes {max_lanes} <= adapter default max_mtp_width "
                    f"{handoff['max_mtp_width']}: the handoff cannot engage"
                )
            handoff = None
        if handoff is not None:
            resolved["mtp_ordinary_handoff"] = handoff
    if adapter_resolution is not None:
        declared = adapter_resolution.default_execution_policy(
            route_selection.route,
            operator_policy=resolved,
        )
        external_route = route_selection.route == "external_draft"
        tree_geometry_overridden = (
            external_route
            and "batch_size_route" in resolved
            and resolved["batch_size_route"] != declared.get("batch_size_route")
        )
        # The external varlen pair is one default group: external varlen
        # prefill requires the varlen dense MLP selection, so an operator who
        # names either key owns both, and the adapter's (off) defaults never
        # contradict an explicit enable.
        varlen_operator_owned = external_route and bool(
            {"external_varlen_prefill", "varlen_dense_mlp"} & set(resolved)
        )
        for key, value in declared.items():
            if key in resolved:
                continue
            if tree_geometry_overridden and key == "tree_node_budget_by_lanes":
                continue
            if varlen_operator_owned and key in {
                "external_varlen_prefill",
                "varlen_dense_mlp",
            }:
                continue
            if approximate_kv and key in STATE_CHECKPOINT_POLICY_KEYS:
                continue
            resolved[key] = value
    if (
        route_selection.native_mtp
        and "self_mtp_copy_draft" not in resolved
    ):
        resolved["self_mtp_copy_draft"] = dict(NATIVE_MTP_COPY_DRAFT_DEFAULT)
    if (
        route_selection.native_mtp
        and "prefill_scheduling" not in resolved
        and max_lanes is not None
    ):
        # The bypass cap needs a self-MTP admission window (--max-lanes) of
        # at least max_bypass + 1; below that the engine refuses the policy,
        # so the default steps aside instead of breaking a small startup.
        if max_lanes > NATIVE_MTP_PREFILL_SCHEDULING["max_bypass"]:
            resolved["prefill_scheduling"] = dict(NATIVE_MTP_PREFILL_SCHEDULING)
        elif skipped is not None:
            skipped["prefill_scheduling"] = (
                f"max_lanes {max_lanes} <= max_bypass "
                f"{NATIVE_MTP_PREFILL_SCHEDULING['max_bypass']}: SRPT needs a "
                "wider admission window"
            )
    return resolved or None


def native_mtp_mode(args, policy, adapter_resolution=None):
    """Compatibility helper returning whether the resolved route is native MTP."""
    return resolve_route_selection(args, policy, adapter_resolution).native_mtp


def serving_engine_kwargs(
    args,
    policy,
    *,
    native_mtp,
    route_selection_source="engine_argument",
    approximate_kv,
    max_request_bytes,
):
    """Map a parsed server namespace to the production engine constructor."""
    return {
        "max_inflight": args.max_inflight,
        "max_lanes": args.max_lanes,
        "max_context": args.max_context,
        "prefill_step": getattr(args, "prefill_step", None),
        "prefill_depth_budget": getattr(args, "prefill_depth_budget", None),
        "default_max_tokens": args.default_max_tokens,
        "max_request_bytes": max_request_bytes,
        "cache_bytes": args.cache_bytes,
        # A namespace built without the parser has no source; treat its value
        # as the operator's, which is never clamped.
        "cache_bytes_source": getattr(args, "cache_bytes_source", "explicit"),
        "cache_dir": args.cache_dir,
        "host_prompt_cache_entries": args.host_prompt_cache_entries,
        "host_prompt_cache_tokens": args.host_prompt_cache_tokens,
        "incremental_tokenizer_cache_entries": getattr(
            args, "incremental_tokenizer_cache_entries", 0
        ),
        "incremental_tokenizer_cache_characters": getattr(
            args, "incremental_tokenizer_cache_characters", 8 << 20
        ),
        "incremental_tokenizer_cache_tokens": getattr(
            args, "incremental_tokenizer_cache_tokens", 1 << 20
        ),
        "coalesce_window_ms": args.coalesce_window_ms,
        "batch_cohort_timeout_ms": args.batch_cohort_timeout_ms,
        "mtp": native_mtp,
        "route_selection_source": route_selection_source,
        "skipped_route_defaults": getattr(args, "skipped_route_defaults", None),
        "prompt_lookup": args.prompt_lookup,
        "tenant_scoped_cache": args.tenant_scoped_cache,
        "qualification_mode": args.qualification_mode,
        "qualification": args.qualification,
        "execution_policy": policy,
        "adaptive_mtp_depth": args.adaptive_mtp_depth,
        "mtp_acceptance_log": (
            None
            if args.mtp_acceptance_log is None
            else {
                "path": str(args.mtp_acceptance_log),
                "lookahead": args.mtp_acceptance_log_lookahead,
            }
        ),
        "spomin_live_surgery": args.spomin_live_surgery,
        "thinking_budget": args.thinking_budget,
        "thinking_steer_alpha": args.thinking_steer_alpha,
        "thinking_steer_hammer": args.thinking_steer_hammer,
        "thinking_auto_calibration": not args.no_thinking_auto_calibration,
        "lane_matmul": args.lane_matmul,
        "lane_policy": args.lane_policy,
        "cache_capsules": {
            "enabled": args.cache_capsules,
            "backend": args.cache_capsule_backend,
            "fallback": args.cache_capsule_backend,
            "deadline_ms": args.cache_capsule_deadline_ms,
        },
        "persistent_block_bytes": args.persistent_block_bytes,
        "approximate_kv": approximate_kv,
        "lora_root": args.lora_dir,
        "max_loras": args.max_loras,
        "max_lora_rank": args.max_lora_rank,
        "reasoning_signing_key_file": args.reasoning_signing_key_file,
        "reasoning_signing_key_env": args.reasoning_signing_key_env,
        "apc_persist_dir": args.apc_persist_dir,
        "apc_persist_on_shutdown": args.apc_persist_on_shutdown,
        "apc_persist_shutdown_seconds": args.apc_persist_shutdown_seconds,
        "apc_persist_corruption": args.apc_persist_corruption,
        "apc_session_max_ttl_seconds": args.apc_session_max_ttl_seconds,
        "apc_session_pinned_disk_bytes": args.apc_session_pinned_disk_bytes,
        "apc_session_pinned_disk_bytes_global": (
            args.apc_session_pinned_disk_bytes_global
        ),
        "apc_session_pinned_resident_bytes": (
            args.apc_session_pinned_resident_bytes
        ),
        "apc_session_prefetch_ttl_seconds": (
            args.apc_session_prefetch_ttl_seconds
        ),
        "apc_quarantine_max_entries": args.apc_quarantine_max_entries,
        "apc_quarantine_max_bytes": args.apc_quarantine_max_bytes,
        "int8_prefill": args.int8_prefill,
        "verify_bitexact": bool(getattr(args, "verify_bitexact", False)),
        "recurrent_state_codec": getattr(args, "recurrent_state_codec", "off"),
        "gpu_keep_warm": (
            {
                "window_seconds": args.gpu_keep_warm_seconds,
                "interval_seconds": getattr(args, "gpu_keep_warm_interval", 0.5),
            }
            if getattr(args, "gpu_keep_warm_seconds", 0)
            else None
        ),
        "power_telemetry": (
            {
                "interval_seconds": getattr(args, "power_telemetry_interval", 1.0),
                "window_seconds": getattr(args, "power_telemetry_window", 60.0),
            }
            if getattr(args, "power_telemetry", False)
            else None
        ),
        "power_governor": power_governor_policy_value(args),
    }


def power_governor_policy_value(args):
    """The engine's ``power_governor`` value; None when the flag is off."""
    mode = getattr(args, "power_governor", "off")
    if mode in (None, "off"):
        return None
    return {
        "mode": mode,
        "budget_watts": getattr(args, "power_budget_watts", None),
        "window_seconds": getattr(args, "power_budget_window", 30.0),
        "efficient_hold_ms": getattr(args, "power_governor_hold_ms", 50.0),
    }


class SignalShutdownController:
    """First-signal graceful drain with a second-signal immediate escape hatch.

    A supervisor that signals the whole process group can deliver SIGINT and
    SIGTERM together (sglang #35202).  That is one request to stop, so a
    signal of a different kind within ``coalesce_seconds`` of the first is
    ignored; a repeat of the same kind, or any signal after the window,
    escalates to immediate shutdown.
    """

    def __init__(self, server, engine, drain_seconds=None, *,
                 coalesce_seconds=1.0, clock=time.monotonic):
        self.server = server
        self.engine = engine
        self.drain_seconds = drain_seconds
        self.coalesce_seconds = coalesce_seconds
        self._clock = clock
        self._first = None  # (signum, monotonic time) of the first signal

    def __call__(self, signum=None, _frame=None):
        now = self._clock()
        if self._first is None:
            self._first = (signum, now)
            target = self._graceful if self.drain_seconds is not None else self.server.shutdown
        else:
            first_signum, first_at = self._first
            if (
                signum is not None
                and first_signum is not None
                and signum != first_signum
                and now - first_at < self.coalesce_seconds
            ):
                logging.getLogger("mlx2.server").info(
                    "signal %s arrived %.3fs after signal %s; treating both as "
                    "one shutdown request", signum, now - first_at, first_signum,
                )
                return
            target = self.server.shutdown
        threading.Thread(target=target, daemon=True).start()

    def _graceful(self):
        try:
            self.engine.quiesce(
                drain_timeout_seconds=self.drain_seconds,
                suspend=False,
            )
            self.engine.wait_for_quiesce(self.drain_seconds + 5.0)
        except Exception:  # noqa: BLE001 - shutdown must still progress
            logging.getLogger("mlx2.server").exception(
                "signal drain failed; continuing immediate shutdown"
            )
        self.server.shutdown()


def main():
    parser = build_parser()
    args = parser.parse_args()
    packed_paths=(args.native_packed_prefill_profile,args.native_packed_prefill_manifest,args.native_packed_prefill_mlx_wheel)
    if any(packed_paths):
        if not all(packed_paths):parser.error("packed-prefill requires profile, artifact manifest and MLX wheel")
        if any(not Path(path).is_file() for path in packed_paths):parser.error("packed-prefill pinned input path missing")
        from .runtime.paged_packed_prefill_serving_profile import startup_environment
        try:os.environ.update(startup_environment(packed_paths[0]))
        except ValueError as error:parser.error(str(error))
        os.environ.update(MLX2_NATIVE_PACKED_PREFILL_B2_PROFILE=packed_paths[0],
            MLX2_NATIVE_PAGED_MANIFEST=packed_paths[1],MLX2_NATIVE_PAGED_MLX_WHEEL=packed_paths[2])

    owned_paths = (args.tensorfold_owned_source, args.tensorfold_owned_mlx_lm_source,
                   args.tensorfold_owned_drafter)
    if (args.tensorfold_owned_live and not all(owned_paths)) or (
        not args.tensorfold_owned_live and any(owned_paths)
    ):
        parser.error("TensorFold-owned live route requires its explicit flag and all three pinned paths")
    try:
        max_request_bytes = request_body_limit(
            args.max_context, args.max_request_bytes
        )
    except ValueError as error:
        parser.error(str(error))
    if not math.isfinite(args.sse_keepalive_seconds) or not (
        args.sse_keepalive_seconds == 0 or 1 <= args.sse_keepalive_seconds <= 280
    ):
        # Past 280 s the keepalive no longer beats undici's 300 s timeout.
        parser.error("--sse-keepalive-seconds must be 0 or between 1 and 280")
    if args.drain_on_sigterm is not None and (
        not math.isfinite(args.drain_on_sigterm)
        or not 0.1 <= args.drain_on_sigterm <= 3600
    ):
        parser.error("--drain-on-sigterm must be between 0.1 and 3600 seconds")
    if not 1 <= args.semantic_retrieval_limit <= 32:
        parser.error("--semantic-retrieval-limit must be between 1 and 32")
    if args.semantic_memory and not (
        args.semantic_memory_dir or args.api_state_dir or args.apc_persist_dir
    ):
        parser.error(
            "--semantic-memory requires --semantic-memory-dir, --api-state-dir, "
            "or --apc-persist-dir"
        )
    if args.semantic_bridge != "rendered":
        if not args.semantic_memory or not args.neural_concept_artifact:
            parser.error(
                "a neural semantic bridge requires --semantic-memory and "
                "--neural-concept-artifact"
            )
        if not args.qualification_mode and not args.qualification:
            parser.error(
                "a neural semantic bridge requires --qualification-mode or a "
                "matching qualification receipt"
            )
    elif args.neural_concept_artifact:
        parser.error(
            "--neural-concept-artifact requires --semantic-bridge neural or hybrid"
        )
    try:
        admin_token = (
            load_admin_token(args.admin_token_file)
            if args.admin_token_file is not None
            else None
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    try:
        http_security = policy_for_bind(
            args.host,
            allowed_hosts=args.allowed_host,
            api_key=load_api_key(
                key_file=args.api_key_file, key_env=args.api_key_env
            ),
        )
        tenant_authenticator = build_tenant_authenticator(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.power_governor != "off" and not args.power_telemetry:
        parser.error("--power-governor requires --power-telemetry (its power signal)")
    if args.power_budget_watts is not None and args.power_governor != "budget":
        parser.error("--power-budget-watts requires --power-governor budget")
    if args.power_governor != "off":
        from .power_governor import PowerGovernorPolicy

        try:
            PowerGovernorPolicy.from_value(power_governor_policy_value(args))
        except ValueError as error:
            parser.error(str(error))
    policy = json.loads(args.execution_policy.read_text()) if args.execution_policy else None
    try:
        from .adapters.registry import inspect_model

        adapter_resolution = inspect_model(args.model)
        route_selection = resolve_route_selection(args, policy, adapter_resolution)
        native_mtp = route_selection.native_mtp
        skipped_route_defaults = {}
        policy = resolve_execution_policy_defaults(
            policy,
            route_selection,
            adapter_resolution,
            approximate_kv=getattr(args, "approximate_kv", None) is not None,
            max_lanes=args.max_lanes,
            skipped=skipped_route_defaults,
        )
        args.skipped_route_defaults = skipped_route_defaults
    except ValueError as error:
        parser.error(str(error))
    from .runtime.adaptive_policy import (
        AdaptiveMTPDepthPolicy,
        MTPOrdinaryHandoffPolicy,
    )

    policy_adaptive = (policy or {}).get("adaptive_mtp_depth")
    try:
        adaptive_selected = args.adaptive_mtp_depth or (
            policy_adaptive is not None
            and AdaptiveMTPDepthPolicy.from_value(policy_adaptive).enabled
        )
    except ValueError as error:
        parser.error(str(error))
    if adaptive_selected and not native_mtp:
        parser.error("--adaptive-mtp-depth requires the native self-MTP route")
    try:
        handoff_selected = MTPOrdinaryHandoffPolicy.from_value(
            (policy or {}).get("mtp_ordinary_handoff")
        ).enabled
    except ValueError as error:
        parser.error(str(error))
    if handoff_selected and not native_mtp:
        parser.error("MTP ordinary handoff requires the native self-MTP route")
    if args.spomin_live_surgery and not args.qualification_mode:
        parser.error("--spomin-live-surgery is restricted to --qualification-mode")
    if args.mtp_acceptance_log_lookahead and args.mtp_acceptance_log is None:
        parser.error("--mtp-acceptance-log-lookahead requires --mtp-acceptance-log")
    if args.mtp_acceptance_log is not None and not args.qualification_mode:
        parser.error("--mtp-acceptance-log is restricted to --qualification-mode")
    if args.mtp_acceptance_log is not None and not native_mtp:
        parser.error("--mtp-acceptance-log requires the native self-MTP route")
    try:
        approximate_kv = approximate_kv_mode(args, native_mtp)
    except ValueError as error:
        parser.error(str(error))
    if args.power_telemetry:
        from .power_telemetry import PowerTelemetryPolicy

        try:
            PowerTelemetryPolicy.from_value(
                {
                    "interval_seconds": args.power_telemetry_interval,
                    "window_seconds": args.power_telemetry_window,
                }
            )
        except ValueError as error:
            parser.error(str(error))
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    from .exit_trace import ExitTrace

    try:
        exit_trace = ExitTrace().install(args.fault_log)
    except OSError as error:
        parser.error(f"cannot open --fault-log: {error}")
    logging.getLogger("mlx2.server").info(
        "selected %s route from %s",
        route_selection.route,
        route_selection.source,
    )
    if not args.qualification_mode and not args.qualification:
        # Qualification is confidence, not permission to run (AGENTS.md).
        logging.getLogger("mlx2.server").warning(
            "no --qualification receipt: serving UNQUALIFIED; /v1/status and "
            "every route receipt report qualification=unqualified"
        )
    try:
        signing_key_path = resolve_reasoning_signing_key(args)
    except (OSError, ValueError) as error:
        parser.error(f"cannot use persistent reasoning signing key: {error}")
    if signing_key_path is not None:
        logging.getLogger("mlx2.server").info(
            "reasoning signing key persisted at %s", signing_key_path
        )
    server = BoundedHTTPServer(
        (args.host, args.port),
        BaseHTTPRequestHandler,
        # Every in-flight request holds a connection: a fixed cap below
        # --max-inflight made the larger admission bound unreachable.
        max_connections=max(DEFAULT_MAX_CONNECTIONS, args.max_inflight),
    )
    request_tracer = None
    try:
        from .tracing import OptionalRequestTracer

        request_tracer = OptionalRequestTracer(args.otlp_traces_endpoint)
        engine = ServingEngine(
            args.model,
            **serving_engine_kwargs(
                args,
                policy,
                native_mtp=native_mtp,
                route_selection_source=route_selection.source,
                approximate_kv=approximate_kv,
                max_request_bytes=max_request_bytes,
            ),
        )
    except BaseException:
        if request_tracer is not None:
            request_tracer.close()
        server.server_close()
        raise
    if tenant_authenticator is not None and tenant_authenticator.uses_secret(
        getattr(getattr(engine, "reasoning_signer", None), "_secret", None)
    ):
        engine.close()
        if request_tracer is not None:
            request_tracer.close()
        server.server_close()
        parser.error(
            "the tenant token secret must differ from the reasoning signing key"
        )
    tool_backend = None
    if args.tool_backend_config is not None:
        from .tool_backend import ConfiguredToolBackend

        tool_backend = ConfiguredToolBackend(args.tool_backend_config)
    semantic_middleware = None
    if args.semantic_memory:
        from .semantic_sidecar import SemanticServingMiddleware

        root = args.semantic_memory_dir
        if root is None:
            root = str(
                Path(args.api_state_dir or args.apc_persist_dir).expanduser().resolve()
                / "semantic-memory"
            )
        from .serving import runtime_identity

        try:
            artifact_binding = adapter_resolution.artifact_fingerprint
        except ValueError as error:
            engine.close()
            if request_tracer is not None:
                request_tracer.close()
            server.server_close()
            parser.error(str(error))
        semantic_middleware = SemanticServingMiddleware.create(
            root,
            model_binding=artifact_binding,
            tokenizer_binding=artifact_binding,
            runtime_binding=runtime_identity()["source_sha256"],
            retrieval_limit=args.semantic_retrieval_limit,
            neural_artifact_root=args.neural_concept_artifact,
            bridge_mode=args.semantic_bridge,
        )
        if semantic_middleware.neural_memory is not None:
            try:
                engine.configure_neural_concept_bridge(
                    semantic_middleware.neural_memory.artifact
                )
                # Reuse the same revision-bound learned projector for the
                # default-off activation-capsule capability. Selection still
                # requires a trusted bus-issued request payload.
                engine.configure_activation_capsule_bridge(
                    semantic_middleware.neural_memory.artifact
                )
            except (RuntimeError, TimeoutError, ValueError) as error:
                engine.close()
                if request_tracer is not None:
                    request_tracer.close()
                server.server_close()
                parser.error(str(error))
    tensorfold_owned_router = None
    if args.tensorfold_owned_live:
        from .adapters.qwen38_tensorfold_owned import TensorfoldOwnedB1Profile
        from .runtime.tensorfold_owned_router import TensorfoldOwnedLiveRouter

        try:
            tensorfold_owned_router = TensorfoldOwnedLiveRouter(
                TensorfoldOwnedB1Profile(
                    args.tensorfold_owned_source, args.model, args.tensorfold_owned_drafter,
                    mlx_lm_source=args.tensorfold_owned_mlx_lm_source, enabled=True,
                ), max_streams=min(args.max_lanes, 8), max_context=args.max_context,
            )
        except (OSError, RuntimeError, ValueError) as error:
            engine.close()
            if request_tracer is not None:
                request_tracer.close()
            server.server_close()
            parser.error(str(error))
    server.RequestHandlerClass = handler_for(
        engine,
        max_request_bytes=max_request_bytes,
        request_tracer=request_tracer,
        api_state_dir=args.api_state_dir,
        tool_backend=tool_backend,
        admin_token=admin_token,
        http_security=http_security,
        tenant_authenticator=tenant_authenticator,
        agent_compat=AgentCompatPolicy(
            args.agent_compat,
            args.custom_tool_grammar,
            load_tenant_policy(args.agent_compat_tenants)
            if args.agent_compat_tenants
            else None,
        ),
        semantic_middleware=semantic_middleware,
        tensorfold_owned_router=tensorfold_owned_router,
        sse_keepalive_seconds=args.sse_keepalive_seconds or None,
    )

    stop = SignalShutdownController(server, engine, args.drain_on_sigterm)
    exit_trace.install_shutdown_signals(stop)

    def watch_worker():
        engine.thread.join()
        if engine.error:
            exit_trace.set_reason(f"generation worker failed: {engine.error}")
            server.shutdown()

    threading.Thread(target=watch_worker, daemon=True).start()
    logging.getLogger("mlx2.server").info(
        "serving on %s:%d (pid %d)", args.host, server.server_address[1], os.getpid()
    )
    try:
        server.serve_forever(poll_interval=0.1)
    except BaseException as error:
        exit_trace.set_reason(f"serve loop raised {type(error).__name__}")
        raise
    else:
        exit_trace.set_reason("server shutdown")
    finally:
        try:
            try:
                if tensorfold_owned_router is not None:
                    tensorfold_owned_router.close()
            finally:
                engine.close()
        finally:
            try:
                request_tracer.close()
            finally:
                server.server_close()
    if engine.error:
        raise SystemExit(engine.error)


if __name__ == "__main__":
    main()
