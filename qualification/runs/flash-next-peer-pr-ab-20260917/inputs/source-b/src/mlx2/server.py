"""Local OpenAI-compatible HTTP serving over the mlx2 execution lifecycle."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
from pathlib import Path
import queue
import signal
import threading

from .serving import Overloaded, ServingEngine
from .logprobs import MAX_TOP_LOGPROBS, wants_logprobs


MIN_REQUEST_BODY_BYTES = 2 << 20
MAX_REQUEST_BODY_BYTES = 32 << 20
REQUEST_BODY_BYTES_PER_CONTEXT_TOKEN = 32
REQUEST_BODY_JSON_OVERHEAD_BYTES = 64 << 10


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


class BoundedHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, *args, max_connections=32, **kwargs):
        self.connections = threading.BoundedSemaphore(max_connections)
        super().__init__(*args, **kwargs)

    def process_request(self, request, address):
        if not self.connections.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self.connections.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.connections.release()


def validate_request(body, chat=True):
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
        "n",
        "stream",
        "enable_thinking",
        "tools",
        "tool_choice",
        "stop",
        "stream_options",
        "reasoning_effort",
        "context_limit",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "response_format",
        "grammar",
        "batch_cohort",
        "mlx_fault",
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
            if not isinstance(message, dict) or message.get("role") not in {
                "system",
                "user",
                "assistant",
                "tool",
            }:
                raise ValueError("invalid text message role")
            if not isinstance(message.get("content"), str) and not (
                message["role"] == "assistant" and message.get("tool_calls")
            ):
                raise ValueError("message content must be text")
            if "tool_calls" in message:
                calls = message["tool_calls"]
                if not isinstance(calls, list) or any(
                    not isinstance(c, dict)
                    or not isinstance(c.get("function"), dict)
                    or not isinstance(c["function"].get("name"), str)
                    for c in calls
                ):
                    raise ValueError("invalid tool call history")
    elif not isinstance(body.get("prompt"), str) or not body["prompt"]:
        raise ValueError("prompt must be a non-empty string")
    if "response_format" in body or "grammar" in body:
        if "response_format" in body and "grammar" in body:
            raise ValueError("response_format and grammar are mutually exclusive")
        from .structured_output import compile_constraint

        compile_constraint(body.get("response_format"), body.get("grammar"))
        if body.get("response_format") == {"type": "text"}:
            body = {k: v for k, v in body.items() if k != "response_format"}
    if "logprobs" in body and not isinstance(body["logprobs"], bool):
        raise ValueError("logprobs must be boolean")
    top_logprobs = body.get("top_logprobs", 0)
    if isinstance(top_logprobs, bool) or not isinstance(top_logprobs, int) or not 0 <= top_logprobs <= MAX_TOP_LOGPROBS:
        raise ValueError("top_logprobs must be an integer from 0 to 11")
    if "tools" in body:
        tools = body["tools"]
        if not chat or not isinstance(tools, list) or not 1 <= len(tools) <= 64:
            raise ValueError("tools must contain 1 to 64 function definitions")
        names = set()
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
            if function.get("strict") or not isinstance(
                function.get("parameters", {}), dict
            ):
                raise ValueError("strict tool schema enforcement is not qualified")
    if body.get("tool_choice", "auto") not in ("auto", "none"):
        raise ValueError("tool_choice supports auto or none")
    if "stop" in body:
        stops = [body["stop"]] if isinstance(body["stop"], str) else body["stop"]
        if (
            not isinstance(stops, list)
            or not 1 <= len(stops) <= 4
            or any(not isinstance(s, str) or not 1 <= len(s) <= 256 for s in stops)
        ):
            raise ValueError(
                "stop must contain 1 to 4 nonempty strings of at most 256 characters"
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
    for key, default, lower, upper in (
        ("max_tokens", 512, 1, 8192),
        ("min_tokens", 0, 0, 8192),
        ("top_k", 20, 0, 2**31 - 1),
    ):
        value = body.get(key, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not lower <= value <= upper
        ):
            raise ValueError(f"{key} must be an integer between {lower} and {upper}")
    if body.get("min_tokens", 0) > body.get("max_tokens", 512):
        raise ValueError("min_tokens must not exceed max_tokens")
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
            or not math.isfinite(value)
            or not lower <= value <= upper
        ):
            raise ValueError(f"{key} must be finite and between {lower} and {upper}")
    if "logit_bias" in body:
        biases = body["logit_bias"]
        if not isinstance(biases, dict) or len(biases) > 4096:
            raise ValueError("logit_bias must be an object with at most 4096 tokens")
        for token, value in biases.items():
            if (not isinstance(token, str) or not token.isdecimal()
                or not 0 <= int(token) < 2**31 or isinstance(value, bool)
                or not isinstance(value, (int, float)) or not math.isfinite(value)
                or not -100 <= value <= 100):
                raise ValueError("invalid logit_bias token or value")
    for key in ("stream", "enable_thinking"):
        if key in body and not isinstance(body[key], bool):
            raise ValueError(f"{key} must be boolean")
    if "seed" in body and (
        isinstance(body["seed"], bool)
        or not isinstance(body["seed"], int)
        or not 0 <= body["seed"] < 2**32
    ):
        raise ValueError("seed must be an unsigned 32-bit integer")
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
    aliases = {"num_predict": "max_tokens", "num_ctx": "context_limit"}
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


def collect_nonstream_job(job, body, *, chat):
    """Collect one already-submitted sample without writing an HTTP response."""
    parts, reasoning, calls, probabilities = [], [], [], []
    while True:
        try:
            event = job.events.get(timeout=1800)
        except queue.Empty as exc:
            raise TimeoutError("generation timed out") from exc
        if "error" in event:
            raise RuntimeError(event["error"])
        if "logprob" in event:
            probabilities.append(event["logprob"])
        if "text" in event or "delta" in event:
            delta = event.get("delta", {"content": event.get("text", "")})
            parts.append(delta.get("content", ""))
            reasoning.append(delta.get("reasoning_content", ""))
            calls.extend(delta.get("tool_calls", []))
        if "finish_reason" not in event:
            continue
        choice = {"finish_reason": event["finish_reason"]}
        if wants_logprobs(body):
            choice["logprobs"] = {"content": probabilities}
        message = {"role": "assistant", "content": "".join(parts)}
        if any(reasoning):
            message["reasoning_content"] = "".join(reasoning)
        if calls:
            message["tool_calls"] = [
                {key: value for key, value in call.items() if key != "index"}
                for call in calls
            ]
        choice.update({"message": message} if chat else {"text": "".join(parts)})
        return choice, {
            "prompt_tokens": job.prompt_tokens,
            "completion_tokens": job.completion_tokens,
            "total_tokens": job.prompt_tokens + job.completion_tokens,
        }, event["receipt"]


def handler_for(engine, *, max_request_bytes: int | None = None):
    # Preserve the standalone/test embedding contract: callers that do not
    # configure a transport limit retain the historical 2 MiB bound. The CLI
    # always resolves and passes its context-derived value explicitly.
    body_limit = request_body_limit(
        1,
        MIN_REQUEST_BODY_BYTES
        if max_request_bytes is None
        else max_request_bytes,
    )

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.connection.settimeout(30)

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
            self.wfile.write(payload)

        def error(self, status, message):
            self.send_json(
                status,
                {
                    "error": {
                        "message": message,
                        "type": "invalid_request_error"
                        if status in {400, 413}
                        else "server_error",
                    }
                },
                headers={"Retry-After": "1"} if status == 429 else None,
            )

        def do_GET(self):
            status = engine.status()
            if self.path == "/health":
                self.send_json(
                    200 if status["healthy"] else 503,
                    {
                        "status": "ok" if status["healthy"] else "unavailable",
                        "error": status["error"],
                    },
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
                            }
                        ],
                    },
                )
            elif self.path == "/v1/status":
                self.send_json(200, status)
            elif self.path == "/v1/status/batching":
                self.send_json(200, engine.batching_status())
            else:
                self.error(404, "unknown endpoint")

        def do_POST(self):
            if self.path not in {"/v1/chat/completions", "/v1/completions"}:
                self.error(404, "unknown endpoint")
                return
            job = None
            jobs = []
            streaming = False
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= body_limit:
                    self.close_connection = True
                    self.error(
                        413,
                        f"body must contain at most {body_limit} bytes",
                    )
                    return
                body = json.loads(self.rfile.read(size))
                chat = self.path == "/v1/chat/completions"
                body = validate_request(body, chat)
                model = engine.status().get("model")
                if body.get("model", model) != model:
                    self.error(404, "unknown model")
                    return
                sample_count = body.get("n", 1)
                if sample_count > 1:
                    admission = engine.admit_parallel_samples(sample_count)
                    base_seed = body.get("seed")
                    samples = []
                    for index in range(sample_count):
                        sample = {**body, "n": 1}
                        if base_seed is not None:
                            sample["seed"] = (base_seed + index) % (2**32)
                        samples.append(sample)
                    jobs.extend(
                        engine.submit_many(
                            samples,
                            tenant_id=self.headers.get("X-Tenant-ID", "default"),
                        )
                    )
                    results = [
                        collect_nonstream_job(sample_job, body, chat=chat)
                        for sample_job in jobs
                    ]
                    choices, usages, receipts = zip(*results)
                    for index, choice in enumerate(choices):
                        choice["index"] = index
                    usage = {
                        "prompt_tokens": sum(item["prompt_tokens"] for item in usages),
                        "completion_tokens": sum(item["completion_tokens"] for item in usages),
                    }
                    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
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
                job = engine.submit(
                    body,
                    tenant_id=self.headers.get("X-Tenant-ID", "default"),
                )
                parts, reasoning, calls, probabilities = [], [], [], []
                while True:
                    try:
                        event = job.events.get(timeout=1800)
                    except queue.Empty:
                        raise TimeoutError("generation timed out")
                    if "error" in event:
                        if streaming:
                            self._sse({"error": {"message": event["error"]}})
                            self._sse("[DONE]")
                        else:
                            self.error(event.get("status", 503), event["error"])
                        return
                    if body.get("stream") and not streaming:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        self.close_connection = True
                        streaming = True
                    if "logprob" in event:
                        if streaming:
                            choice = {"index": 0, "finish_reason": None,
                                      "logprobs": {"content": [event["logprob"]]}}
                            choice.update({"delta": {}} if chat else {"text": ""})
                            self._sse({"id": job.id,
                                       "object": "chat.completion.chunk" if chat else "text_completion",
                                       "created": int(job.created), "model": model,
                                       "choices": [choice]})
                        else:
                            probabilities.append(event["logprob"])
                    if "text" in event or "delta" in event:
                        delta = event.get("delta", {"content": event.get("text", "")})
                        parts.append(delta.get("content", ""))
                        reasoning.append(delta.get("reasoning_content", ""))
                        calls.extend(delta.get("tool_calls", []))
                        if streaming:
                            choice = {"index": 0, "finish_reason": None}
                            choice.update(
                                {"delta": delta}
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
                        usage = {
                            "prompt_tokens": job.prompt_tokens,
                            "completion_tokens": job.completion_tokens,
                            "total_tokens": job.prompt_tokens + job.completion_tokens,
                            "prompt_tokens_details": {
                                "cached_tokens": job.cached_tokens
                            },
                        }
                        choice = {"index": 0, "finish_reason": event["finish_reason"]}
                        if streaming:
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
                                    "usage": usage,
                                    "mlx2": receipt,
                                }
                            )
                            self._sse("[DONE]")
                        else:
                            if wants_logprobs(body):
                                choice["logprobs"] = {"content": probabilities}
                            message = {"role": "assistant", "content": "".join(parts)}
                            if any(reasoning):
                                message["reasoning_content"] = "".join(reasoning)
                            if calls:
                                message["tool_calls"] = [
                                    {k: v for k, v in c.items() if k != "index"}
                                    for c in calls
                                ]
                            choice.update(
                                {"message": message}
                                if chat
                                else {"text": "".join(parts)}
                            )
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
                        return
            except (ValueError, json.JSONDecodeError) as exc:
                self.error(400, str(exc))
            except Overloaded as exc:
                self.error(429, str(exc))
            except (BrokenPipeError, ConnectionResetError):
                pass
            except (RuntimeError, TimeoutError) as exc:
                if not streaming:
                    self.error(503, str(exc))
            finally:
                if job is not None:
                    job.cancelled.set()
                for sample_job in jobs:
                    sample_job.cancelled.set()

        def _sse(self, value):
            data = (
                value if isinstance(value, str) else json.dumps(value, allow_nan=False)
            )
            self.wfile.write(f"data: {data}\n\n".encode())
            self.wfile.flush()

    return Handler


def build_parser():
    parser = argparse.ArgumentParser(description="mlx2 APCv2 Flash-Next server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8285)
    parser.add_argument("--max-context", type=int, default=262144)
    parser.add_argument(
        "--max-request-bytes",
        type=int,
        help=(
            "maximum JSON request body bytes; defaults to a bounded value "
            "derived from --max-context"
        ),
    )
    parser.add_argument("--execution-policy", type=Path, help="JSON file with adapter execution choices; included in qualification identity")
    parser.add_argument("--max-lanes", type=int, default=4)
    parser.add_argument("--max-inflight", type=int, default=8)
    parser.add_argument(
        "--coalesce-window-ms",
        type=float,
        default=5.0,
        help="idle-to-active wait for B1-B4 admission (default: 5 ms)",
    )
    parser.add_argument(
        "--batch-cohort-timeout-ms",
        type=float,
        default=1000.0,
        help="deadline for a declared atomic HTTP batch cohort (default: 1000 ms)",
    )
    parser.add_argument("--cache-bytes", type=int, default=12 << 30)
    parser.add_argument("--cache-dir")
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
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--ordinary", action="store_true", help="ordinary decode reference with APCv2")
    execution.add_argument("--external-draft", action="store_true", help="external draft/verify; requires draft_model execution policy")
    execution.add_argument(
        "--prompt-lookup",
        action="store_true",
        help="target-verified indexed prompt-lookup route",
    )
    parser.add_argument("--qualification-mode", action="store_true")
    parser.add_argument("--qualification")
    return parser


def native_mtp_mode(args, policy):
    """Validate route intent before binding a listener or allocating a model."""
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
    return not (args.ordinary or args.external_draft or prompt_lookup)


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        max_request_bytes = request_body_limit(
            args.max_context, args.max_request_bytes
        )
    except ValueError as error:
        parser.error(str(error))
    policy = json.loads(args.execution_policy.read_text()) if args.execution_policy else None
    try:
        native_mtp = native_mtp_mode(args, policy)
    except ValueError as error:
        parser.error(str(error))
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if not args.qualification_mode and not args.qualification:
        parser.error("provide --qualification or explicitly run --qualification-mode")
    server = BoundedHTTPServer((args.host, args.port), BaseHTTPRequestHandler)
    try:
        engine = ServingEngine(
            args.model,
            max_inflight=args.max_inflight,
            max_lanes=args.max_lanes,
            max_context=args.max_context,
            max_request_bytes=max_request_bytes,
            cache_bytes=args.cache_bytes,
            cache_dir=args.cache_dir,
            host_prompt_cache_entries=args.host_prompt_cache_entries,
            host_prompt_cache_tokens=args.host_prompt_cache_tokens,
            coalesce_window_ms=args.coalesce_window_ms,
            batch_cohort_timeout_ms=args.batch_cohort_timeout_ms,
            mtp=native_mtp,
            prompt_lookup=args.prompt_lookup,
            qualification_mode=args.qualification_mode,
            qualification=args.qualification,
            execution_policy=policy,
        )
    except BaseException:
        server.server_close()
        raise
    server.RequestHandlerClass = handler_for(
        engine, max_request_bytes=max_request_bytes
    )

    def stop(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def watch_worker():
        engine.thread.join()
        if engine.error:
            server.shutdown()

    threading.Thread(target=watch_worker, daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        engine.close()
        server.server_close()
    if engine.error:
        raise SystemExit(engine.error)


if __name__ == "__main__":
    main()
