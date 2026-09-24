"""Allowlisted Responses MCP execution over Streamable HTTP JSON-RPC."""

from __future__ import annotations

import json
import re
import threading
from collections import Counter
from http.client import HTTPException
from pathlib import Path
from urllib.request import Request, urlopen

from .api_resources import CapabilityUnavailable


class HostedToolError(RuntimeError):
    """An MCP server failed or broke protocol: a bad gateway, not a bad request.

    Every transport and protocol failure of a hosted tool call surfaces as
    this one type, so the API answers 502 instead of mapping a malformed
    reply to 400 or a server fault to 500.
    """

    status = 502
    code = "hosted_tool_error"


def _rpc_payload(raw, *, expected_id):
    text = raw.decode("utf-8")
    if text.lstrip().startswith("{"):
        events = [json.loads(text)]
    else:
        events, data = [], []
        # SSE recognizes CR/LF only. Unicode line separators can be literal
        # characters inside the JSON data and must not split an event.
        for line in [*text.replace("\r\n", "\n").replace("\r", "\n").split("\n"), ""]:
            if not line:
                if data:
                    events.append(json.loads("\n".join(data)))
                    data = []
            elif line.startswith("data:"):
                value = line[5:]
                data.append(value.removeprefix(" "))
    responses = [
        event for event in events
        if isinstance(event, dict)
        and type(event.get("id")) is type(expected_id)
        and event.get("id") == expected_id
    ]
    if len(responses) != 1:
        raise HostedToolError("MCP server did not return one matching response")
    response = responses[0]
    if response.get("jsonrpc") != "2.0" or ("result" in response) == ("error" in response):
        raise HostedToolError("MCP server returned an invalid JSON-RPC response")
    return response


class _HTTPMCPClient:
    def __init__(self, url, headers, *, timeout=30):
        self.url = url
        self.headers = dict(headers)
        self.timeout = timeout
        self.session_id = None
        self.initialized = False
        self.serial = 0
        self.lock = threading.Lock()

    def _call(self, method, params=None, *, notification=False):
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
                timeout=self.timeout,
            ) as response:
                session = response.headers.get("Mcp-Session-Id")
                if session:
                    self.session_id = session
                if notification:
                    return None
                result = _rpc_payload(response.read(), expected_id=message["id"])
        except HostedToolError:
            raise
        except (OSError, ValueError, HTTPException) as error:
            # OSError covers URLError, HTTPError and timeouts; ValueError
            # covers undecodable or malformed JSON replies.
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

    def list_tools(self):
        with self.lock:
            self._initialize()
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

    def call_tool(self, name, arguments):
        with self.lock:
            self._initialize()
            return self._call("tools/call", {"name": name, "arguments": arguments})


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
                or not isinstance(value.get("server_url"), str)
                or not value["server_url"].startswith(("https://", "http://127.0.0.1:"))
            ):
                raise ValueError("MCP servers require labels and allowlisted HTTP URLs")
            headers = value.get("headers", {})
            if not isinstance(headers, dict) or any(
                not isinstance(key, str) or not isinstance(item, str)
                for key, item in headers.items()
            ):
                raise ValueError("MCP headers must be string mappings")
            self.clients[label] = _HTTPMCPClient(value["server_url"], headers)

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

    def execute(self, binding, arguments):
        client, name = binding
        try:
            result = client.call_tool(name, arguments)
        except BaseException:
            self.counts["execution_failures"] += 1
            raise
        self.counts["executions"] += 1
        return result

    def status(self):
        return {
            "configured": True,
            "servers": len(self.clients),
            "counts": dict(self.counts),
        }
