"""Allowlisted Responses MCP execution over Streamable HTTP JSON-RPC."""

from __future__ import annotations

import json
import math
import re
import select
import socket
import ssl
import threading
import time
from collections import Counter
from contextlib import contextmanager
from http.client import HTTPConnection, HTTPException, HTTPSConnection
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import (
    HTTPHandler,
    HTTPRedirectHandler,
    HTTPSHandler,
    Request,
    build_opener,
)

from .api_resources import CapabilityUnavailable


class HostedToolError(RuntimeError):
    """An MCP server failed or broke protocol: a bad gateway, not a bad request.

    Every transport and protocol failure of a hosted tool call surfaces as
    this one type, so the API answers 502 instead of mapping a malformed
    reply to 400 or a server fault to 500.
    """

    status = 502
    code = "hosted_tool_error"


class _SessionExpired(HostedToolError):
    """The MCP server answered 404 for this client's session id.

    Streamable HTTP servers may end a session at any time (a restart, idle
    expiry); the client must then start a new one.  A subclass, so a request
    that still fails after one new session is a 502 like any other.
    """


class HostedToolCancelled(HostedToolError):
    """The local caller stopped waiting; a remote side effect may have run."""

    code = "hosted_tool_cancelled"


class HostedToolTimeout(HostedToolError):
    """One complete MCP operation exhausted its wall-clock budget."""

    code = "hosted_tool_timeout"


class _OperationBudget:
    """One budget across lock acquisition, handshake, retries and pagination.

    The watcher shuts down active sockets, rather than abandoning a worker
    that could continue reading and mutate session state after cancellation.
    OS DNS calls use a bounded helper pool; cancelling abandons only that
    lookup, never an HTTP request that could later execute a tool.
    """

    def __init__(self, client, cancel_event=None):
        self.deadline = time.monotonic() + client.timeout
        self.cancel_event = cancel_event
        self.max_bytes = client.max_response_bytes
        self.bytes = 0
        self.max_events = client.max_sse_events
        self.events = 0
        self.max_depth = client.max_json_depth
        self._sockets = set()
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._failure = None

    def _reason(self):
        if self.cancel_event is not None and self.cancel_event.is_set():
            return HostedToolCancelled("MCP operation cancelled; remote completion is unknown")
        if time.monotonic() >= self.deadline:
            return HostedToolTimeout("MCP operation exceeded its total timeout")
        return None

    def _abort(self, reason):
        with self._lock:
            if self._failure is None:
                self._failure = reason
            for connection in self._sockets:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    connection.close()
                except OSError:
                    pass

    def check(self):
        reason = self._failure or self._reason()
        if reason is not None:
            self._abort(reason)
            raise reason

    def track(self, connection):
        with self._lock:
            self._sockets.add(connection)
        self.check()

    def remaining(self):
        self.check()
        return max(0.001, self.deadline - time.monotonic())

    def _watch(self):
        while not self._done.wait(min(0.05, max(0, self.deadline - time.monotonic()))):
            reason = self._reason()
            if reason is not None:
                self._abort(reason)
                return

    def __enter__(self):
        self.check()
        self._watcher = threading.Thread(target=self._watch, name="mlx2-mcp-budget", daemon=True)
        self._watcher.start()
        return self

    def __exit__(self, *_args):
        self._done.set()
        self._watcher.join()


def _bounded_json(text, max_depth):
    # Check before json.loads: a bounded body can still overflow Python's
    # recursive JSON decoder. Count structure only outside escaped strings.
    depth, quoted, escaped = 0, False, False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > max_depth:
                raise HostedToolError("MCP response exceeded the JSON depth bound")
        elif char in "]}":
            depth -= 1
    return json.loads(text)


def _rpc_payload(raw, *, expected_id, budget=None):
    text = raw.decode("utf-8")
    max_depth = budget.max_depth if budget is not None else 128

    def event(value):
        if budget is not None:
            budget.check()
            budget.events += 1
            if budget.events > budget.max_events:
                raise HostedToolError("MCP operation exceeded the SSE event bound")
        return _bounded_json(value, max_depth)

    if text.lstrip().startswith("{"):
        events = [event(text)]
    else:
        events, data = [], []
        # SSE recognizes CR/LF only. Unicode line separators can be literal
        # characters inside the JSON data and must not split an event.
        for line in [*text.replace("\r\n", "\n").replace("\r", "\n").split("\n"), ""]:
            if not line:
                if data:
                    events.append(event("\n".join(data)))
                    data = []
            elif line.startswith("data:"):
                value = line[5:]
                data.append(value.removeprefix(" "))
    # JSON-RPC ids are per sender: a request the server sends on this stream
    # (``ping`` needs no capability) may carry the same id, but it has a
    # ``method`` and is not the reply.
    responses = [
        event for event in events
        if isinstance(event, dict)
        and "method" not in event
        and type(event.get("id")) is type(expected_id)
        and event.get("id") == expected_id
    ]
    if len(responses) != 1:
        raise HostedToolError("MCP server did not return one matching response")
    response = responses[0]
    if response.get("jsonrpc") != "2.0" or ("result" in response) == ("error" in response):
        raise HostedToolError("MCP server returned an invalid JSON-RPC response")
    return response


class _NoRedirect(HTTPRedirectHandler):
    """Refuse every redirect: the operator allowlisted one origin.

    urllib would otherwise follow a 3xx anywhere, copying the configured
    headers (an Authorization or X-Api-Key value) to the new host, so an
    allowlisted HTTPS server could forward the credentials to plain HTTP
    elsewhere.  Returning None makes the 3xx an HTTPError like any other.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = build_opener(_NoRedirect())
_RESOLVER_SLOTS = threading.BoundedSemaphore(4)


def _resolve_addresses(host, port, budget):
    # getaddrinfo has no cancellation API. Cap its outstanding daemon workers
    # globally, and keep network connects and session mutation in the caller.
    # A stuck OS resolver can consume a slot but cannot outlive a request by
    # subsequently sending it, nor create an unbounded thread backlog.
    while not _RESOLVER_SLOTS.acquire(timeout=min(0.05, budget.remaining())):
        budget.check()
    try:
        budget.check()
    except BaseException:
        _RESOLVER_SLOTS.release()
        raise
    outcome, done = {}, threading.Event()

    def resolve():
        try:
            outcome["value"] = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
        except Exception as error:  # noqa: BLE001 - re-raised on the waiting caller
            outcome["error"] = error
        finally:
            _RESOLVER_SLOTS.release()
            done.set()

    try:
        threading.Thread(target=resolve, name="mlx2-mcp-dns", daemon=True).start()
    except BaseException:
        _RESOLVER_SLOTS.release()
        raise
    while not done.wait(min(0.05, budget.remaining())):
        budget.check()
    budget.check()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def _wait_ready(sock, events, remaining):
    """Wait up to 50 ms (and the budget) for ``events`` on ``sock``.

    ``poll``, not ``select.select``: the latter raises ValueError for
    descriptors >= 1024, which failed every handshake on a busy server.
    """
    poller = select.poll()
    poller.register(sock, events)
    poller.poll(max(0, int(min(0.05, remaining) * 1000)))


class _BudgetConnection:
    """Register sockets before writes and response-header reads can block."""

    def __init__(self, *args, budget, **kwargs):
        self.budget = budget
        super().__init__(*args, **kwargs)
        self._create_connection = self._connect_socket

    def _connect_socket(self, address, timeout=None, source_address=None):
        failure = None
        for family, kind, protocol, _canonical, endpoint in _resolve_addresses(
            *address, self.budget
        ):
            self.budget.check()
            connection = socket.socket(family, kind, protocol)
            try:
                self.budget.track(connection)
                connection.settimeout(min(timeout, self.budget.remaining()))
                if source_address:
                    connection.bind(source_address)
                connection.connect(endpoint)
                self.budget.check()
                return connection
            except OSError as error:
                failure = error
                connection.close()
                self.budget.check()
        if failure is not None:
            raise failure
        raise OSError("MCP hostname resolved to no stream addresses")

    def send(self, data):
        self.budget.check()
        if self.sock is None:
            self.connect()
        # Also covers a proxy CONNECT's send, before its header read. The
        # TLS wrapper is registered again for the subsequent origin request.
        self.budget.track(self.sock)
        self.sock.settimeout(self.budget.remaining())
        return super().send(data)


class _HTTPBudgetConnection(_BudgetConnection, HTTPConnection):
    pass


class _HTTPSBudgetConnection(_BudgetConnection, HTTPSConnection):
    def connect(self):
        # Preserve HTTPConnection's TCP/proxy CONNECT handling, but transfer
        # watchdog ownership before TLS can block. wrap_socket detaches the
        # raw descriptor, so tracking only that socket cannot abort handshake.
        HTTPConnection.connect(self)
        try:
            self.budget.check()
            self.sock = self._context.wrap_socket(
                self.sock, server_hostname=self._tunnel_host or self.host,
                do_handshake_on_connect=False,
            )
            self.budget.track(self.sock)
            # A cross-thread shutdown can miss OpenSSL's blocking poll on
            # some platforms even after the peer observes EOF. Keep the
            # handshake nonblocking so cancellation never relies on that
            # wakeup; readiness waits consume the same operation deadline.
            self.sock.setblocking(False)
            while True:
                self.budget.check()
                try:
                    self.sock.do_handshake()
                except ssl.SSLWantReadError:
                    _wait_ready(self.sock, select.POLLIN, self.budget.remaining())
                except ssl.SSLWantWriteError:
                    _wait_ready(self.sock, select.POLLOUT, self.budget.remaining())
                else:
                    break
            # HTTP writes/header reads retain ordinary timeout semantics,
            # refreshed from the budget after all TLS work has completed.
            self.sock.settimeout(self.budget.remaining())
        except Exception:
            # Also close an unwrapped raw socket if context setup failed.
            # A watcher shutdown can surface as an SSL/descriptor exception;
            # preserve the operation's typed timeout/cancellation instead.
            self.close()
            self.budget.check()
            raise


def urlopen(request, *, timeout, budget=None):
    if budget is None:
        return _OPENER.open(request, timeout=timeout)

    class HTTPBudgetHandler(HTTPHandler):
        def http_open(self, req):
            return self.do_open(
                lambda *a, **kw: _HTTPBudgetConnection(*a, budget=budget, **kw), req
            )

    class HTTPSBudgetHandler(HTTPSHandler):
        def https_open(self, req):
            return self.do_open(
                lambda *a, **kw: _HTTPSBudgetConnection(*a, budget=budget, **kw),
                req, context=self._context,
            )

    return build_opener(
        _NoRedirect(), HTTPBudgetHandler(), HTTPSBudgetHandler()
    ).open(request, timeout=timeout)


def _read_response(response, budget):
    length = response.headers.get("Content-Length")
    if length is not None:
        try:
            length = int(length)
        except ValueError as error:
            raise HostedToolError("MCP response has invalid Content-Length") from error
        if length < 0 or length > budget.max_bytes - budget.bytes:
            raise HostedToolError("MCP operation exceeded the response byte bound")
    chunks = []
    while True:
        budget.check()
        # read1 returns after one raw read. A peer cannot keep this call open
        # indefinitely by delivering one byte just before each socket timeout.
        chunk = response.read1(min(65536, budget.max_bytes - budget.bytes + 1))
        budget.check()
        if not chunk:
            break
        budget.bytes += len(chunk)
        if budget.bytes > budget.max_bytes:
            raise HostedToolError("MCP operation exceeded the response byte bound")
        chunks.append(chunk)
    return b"".join(chunks)


class _HTTPMCPClient:
    def __init__(
        self, url, headers, *, timeout=30, max_response_bytes=8 * 1024 * 1024,
        max_sse_events=1024, max_json_depth=128,
    ):
        if (
            isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0
        ):
            raise ValueError("MCP timeout must be finite positive seconds")
        for name, limit in (
            ("max_response_bytes", max_response_bytes),
            ("max_sse_events", max_sse_events), ("max_json_depth", max_json_depth),
        ):
            if type(limit) is not int or limit <= 0:
                raise ValueError(f"MCP {name} must be a positive integer")
        self.url = url
        self.headers = dict(headers)
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes
        self.max_sse_events = max_sse_events
        self.max_json_depth = max_json_depth
        self.session_id = None
        self.initialized = False
        self.serial = 0
        self.lock = threading.Lock()
        self._budget = None

    def _call(self, method, params=None, *, notification=False):
        if self._budget is None:
            with _OperationBudget(self) as budget:
                return self._call_bounded(method, params, notification=notification, budget=budget)
        return self._call_bounded(
            method, params, notification=notification, budget=self._budget
        )

    def _call_bounded(self, method, params=None, *, notification=False, budget):
        budget.check()
        self.serial += 1
        message = {"jsonrpc": "2.0", "method": method}
        if not notification:
            message["id"] = self.serial
        if params is not None:
            message["params"] = params
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-03-26",
            **self.headers,
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        try:
            with urlopen(
                Request(self.url, data=json.dumps(message).encode(), headers=headers),
                timeout=budget.remaining(), budget=budget,
            ) as response:
                budget.check()
                session = response.headers.get("Mcp-Session-Id")
                if notification:
                    return None
                result = _rpc_payload(
                    _read_response(response, budget), expected_id=message["id"], budget=budget
                )
                budget.check()
                if session:
                    self.session_id = session
        except HostedToolError:
            raise
        except HTTPError as error:
            error.close()
            budget.check()
            if error.code == 404 and "Mcp-Session-Id" in headers:
                raise _SessionExpired(f"MCP {method} failed: {error}") from error
            raise HostedToolError(f"MCP {method} failed: {error}") from error
        except (OSError, ValueError, HTTPException, RecursionError) as error:
            # OSError covers URLError, HTTPError and timeouts; ValueError
            # covers undecodable or malformed JSON replies.
            budget.check()
            raise HostedToolError(f"MCP {method} failed: {error}") from error
        if not isinstance(result, dict) or result.get("jsonrpc") != "2.0":
            raise HostedToolError(f"MCP {method} reply is not a JSON-RPC 2.0 object")
        if "error" in result:
            # A response carries exactly one of result or error, so any error
            # member fails the call, whatever else the reply holds.
            raise HostedToolError(f"MCP error: {result['error']}")
        # The id must be this request's integer id: bool and float compare
        # equal to ints in Python, and a reply to another request is not
        # this call's result.
        reply_id = result.get("id")
        if type(reply_id) is not int or reply_id != message["id"]:
            raise HostedToolError(f"MCP {method} reply answers another request id")
        # MCP defines every request's result as an object; a missing or
        # null result must not reach the model as a tool output.
        if not isinstance(result.get("result"), dict):
            raise HostedToolError(f"MCP {method} reply has no result object")
        return result["result"]

    def _initialize(self):
        if self.initialized:
            return
        # A new InitializeRequest never carries an old session id: not after
        # the server ended it, nor after a half-finished handshake.
        self.session_id = None
        self._call(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "mlx2", "version": "0.1.0"},
            },
        )
        self._call("notifications/initialized", notification=True)
        self.initialized = True

    def _with_session(self, operation):
        """Run ``operation`` in a live session; called with the lock held.

        If the server ended the session, start a new one and retry once.  The
        404 means the server turned the request away unrun, so retrying a
        ``tools/call`` cannot run the tool twice.
        """
        self._initialize()
        try:
            return operation()
        except _SessionExpired:
            self.initialized = False
            self._initialize()
            return operation()

    def _list_tools(self):
        tools, seen, params = [], set(), None
        for _ in range(128):
            result = self._call("tools/list", params)
            if (
                not isinstance(result, dict)
                or not isinstance(result.get("tools"), list)
                or any(not isinstance(tool, dict) for tool in result["tools"])
            ):
                raise HostedToolError(
                    "MCP server returned an invalid tools/list response"
                )
            tools.extend(result["tools"])
            cursor = result.get("nextCursor")
            if cursor is None:
                return tools
            if not isinstance(cursor, str) or cursor in seen:
                raise HostedToolError("MCP server returned an invalid pagination cursor")
            seen.add(cursor)
            params = {"cursor": cursor}
        raise HostedToolError("MCP tools/list exceeded the pagination page bound")

    @contextmanager
    def _operation(self, cancel_event):
        with _OperationBudget(self, cancel_event) as budget:
            while not self.lock.acquire(timeout=min(0.05, budget.remaining())):
                budget.check()
            try:
                budget.check()
                self._budget = budget
                yield
                budget.check()
            finally:
                self._budget = None
                self.lock.release()

    def list_tools(self, *, cancel_event=None):
        with self._operation(cancel_event):
            # A new session restarts from the first page: the old session's
            # cursors may not be valid in it.
            return self._with_session(self._list_tools)

    def call_tool(self, name, arguments, *, cancel_event=None):
        with self._operation(cancel_event):
            return self._with_session(
                lambda: self._call("tools/call", {"name": name, "arguments": arguments})
            )


def _allowlisted_server_url(url) -> bool:
    """HTTPS to a named host, or plain HTTP to 127.0.0.1 with an explicit port.

    Parsed structurally: a string-prefix check admitted
    ``http://127.0.0.1:80@evil.example/mcp``, whose host is ``evil.example``
    (the prefix is userinfo), and would have sent the configured headers
    there in clear text.  Userinfo is refused in every scheme.
    """
    if not isinstance(url, str):
        return False
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return False
    if "@" in parts.netloc or not parts.hostname:
        return False
    if parts.scheme == "https":
        return True
    return parts.scheme == "http" and parts.hostname == "127.0.0.1" and port is not None


class ConfiguredToolBackend:
    """Resolve only MCP servers explicitly allowlisted by the operator."""

    def __init__(self, config):
        if isinstance(config, (str, Path)):
            config = json.loads(Path(config).read_text())
        servers = config.get("servers") if isinstance(config, dict) else None
        if not isinstance(servers, dict):
            raise ValueError(  # noqa: TRY004 - configuration errors share one API
                "tool backend config requires a servers object"
            )
        self.clients = {}
        self.counts = Counter()
        for label, value in servers.items():
            if (
                not isinstance(label, str)
                or not isinstance(value, dict)
                or not _allowlisted_server_url(value.get("server_url"))
            ):
                raise ValueError("MCP servers require labels and allowlisted HTTP URLs")
            headers = value.get("headers", {})
            if not isinstance(headers, dict) or any(
                not isinstance(key, str) or not isinstance(item, str)
                for key, item in headers.items()
            ):
                raise ValueError("MCP headers must be string mappings")
            self.clients[label] = _HTTPMCPClient(
                value["server_url"], headers,
                timeout=value.get("timeout_seconds", 30),
                max_response_bytes=value.get("max_response_bytes", 8 * 1024 * 1024),
                max_sse_events=value.get("max_sse_events", 1024),
                max_json_depth=value.get("max_json_depth", 128),
            )

    def prepare(self, tools):
        chat_tools, executors = [], {}
        names = set()
        for tool in tools:
            if tool.get("type") == "function":
                unknown = set(tool) - {"type", "name", "description", "parameters", "strict"}
                if unknown:
                    raise ValueError("unsupported Responses function fields: " + ", ".join(sorted(unknown)))
                name = tool.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError("function name must be nonempty text")
                if name in names:
                    raise ValueError("tool names collide after normalization")
                names.add(name)
                chat_tools.append(
                    {"type": "function", "function": {key: tool[key] for key in tool if key != "type"}}
                )
                continue
            if tool.get("type") != "mcp":
                raise CapabilityUnavailable(
                    "configured execution currently supports function and MCP tools"
                )
            unknown = set(tool) - {
                "type", "server_label", "server_description", "server_url",
                "allowed_tools", "require_approval",
            }
            if unknown:
                raise ValueError("unsupported MCP fields: " + ", ".join(sorted(unknown)))
            label = tool.get("server_label")
            client = self.clients.get(label)
            if client is None or tool.get("server_url") != client.url:
                raise CapabilityUnavailable("MCP server is not operator-allowlisted")
            if tool.get("require_approval", "always") != "never":
                raise CapabilityUnavailable("mlx2 has no interactive MCP approval channel")
            allowed = tool.get("allowed_tools")
            if allowed is not None and (
                not isinstance(allowed, list)
                or any(not isinstance(name, str) for name in allowed)
            ):
                raise ValueError("allowed_tools must be a list of tool names")
            for definition in client.list_tools():
                original = definition.get("name")
                if not isinstance(original, str) or (allowed is not None and original not in allowed):
                    continue
                safe = re.sub(r"[^A-Za-z0-9_-]", "_", f"mcp__{label}__{original}")[:128]
                if safe in names:
                    raise ValueError("tool names collide after normalization")
                names.add(safe)
                chat_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": safe,
                            "description": definition.get("description", ""),
                            "parameters": definition.get("inputSchema", {"type": "object"}),
                            "strict": False,
                        },
                    }
                )
                executors[safe] = (client, original)
        self.counts["preparations"] += 1
        self.counts["tools_exposed"] += len(executors)
        return chat_tools, executors

    def execute(self, binding, arguments, *, cancel_event=None):
        client, name = binding
        try:
            result = client.call_tool(name, arguments, cancel_event=cancel_event)
        except BaseException:
            self.counts["execution_failures"] += 1
            raise
        self.counts["executions"] += 1
        return result

    def status(self):
        return {
            "configured": True,
            "servers": len(self.clients),
            "operation_limits": {
                label: {
                    "timeout_seconds": client.timeout,
                    "max_response_bytes": client.max_response_bytes,
                    "max_sse_events": client.max_sse_events,
                    "max_json_depth": client.max_json_depth,
                }
                for label, client in self.clients.items()
            },
            "counts": dict(self.counts),
        }
