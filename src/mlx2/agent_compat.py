"""Opt-in wire compatibility for real coding-agent clients.

The OpenAI Codex CLI (Responses) and Claude Code (Anthropic Messages) send
request shapes beyond mlx2's bounded API subset: ``custom`` freeform tools with
a Lark grammar, ``namespace`` tool groups, hosted tools the local server cannot
execute, message ``phase`` markers, ``developer`` messages after the first
turn, adaptive thinking, ``output_config``, ``context_management`` and
mid-conversation system messages.  With ``--agent-compat`` those shapes are
translated deterministically onto the ordinary chat serving contract; without
it main's fail-closed behavior is unchanged.

Every translation increments a default-on integer counter in ``engine.counts``
so harnesses can refuse an arm whose mechanism never engaged.  Nothing here
touches the scheduler, cache or model math: it is request/response translation
only, and every rendering is a pure function of the request so APCv2 prefix
reuse across agent turns is preserved.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping

from .lark_regex import LarkGrammarError, lark_to_regex

_LOG = logging.getLogger(__name__)

GRAMMAR_MODES = frozenset({"off", "validate"})

#: Hosted Responses tools that need an OpenAI-side executor.  In agent-compat
#: mode they are removed from the request (never simulated) and reported.
DROPPABLE_HOSTED_TOOLS = frozenset(
    {
        "web_search",
        "web_search_preview",
        "tool_search",
        "image_generation",
        "local_shell",
        "file_search",
        "code_interpreter",
    }
)

MESSAGE_PHASES = frozenset({"commentary", "final_answer"})

#: Counter keys (``engine.counts``) and their Prometheus event labels.
COUNTERS = {
    "agent_compat_requests": "request",
    "agent_compat_custom_tools_declared": "custom_tool_declared",
    "agent_compat_custom_tool_calls": "custom_tool_call",
    "agent_compat_custom_tool_replays": "custom_tool_replay",
    "agent_compat_custom_tool_grammar_validated": "custom_tool_grammar_validated",
    "agent_compat_custom_tool_grammar_rejected": "custom_tool_grammar_rejected",
    "agent_compat_namespace_tools": "namespace_tool_declared",
    "agent_compat_namespace_calls": "namespace_call",
    "agent_compat_hosted_tools_dropped": "hosted_tool_dropped",
    "agent_compat_phase_inputs": "phase_input",
    "agent_compat_phase_outputs": "phase_output",
    "agent_compat_system_folded": "system_folded",
    "agent_compat_adaptive_thinking": "adaptive_thinking",
    "agent_compat_thinking_omitted": "thinking_omitted",
    "agent_compat_thinking_restored": "thinking_restored",
    "agent_compat_thinking_cleared": "thinking_cleared",
    "agent_compat_output_effort": "output_effort",
}


SERVER_MODES = ("off", "on", "opt-in", "auto")
SOURCES = ("header", "tenant", "detected", "server")
CLIENTS = ("codex", "claude_code")
COMPAT_HEADER = "X-MLX2-Agent-Compat"
GRAMMAR_HEADER = "X-MLX2-Custom-Tool-Grammar"

for _source in SOURCES:
    COUNTERS[f"agent_compat_source_{_source}"] = f"source_{_source}"
for _client in CLIENTS:
    COUNTERS[f"agent_compat_detected_{_client}"] = f"detected_{_client}"
COUNTERS["agent_compat_mode_conflicts"] = "mode_conflict"


class AgentCompatError(ValueError):
    """A request is inconsistent with its resolved agent-compat mode (HTTP 400)."""


class AgentCompat:
    """One resolved agent-compat decision (per request, or fixed)."""

    def __init__(
        self,
        enabled: bool = False,
        custom_tool_grammar: str = "off",
        *,
        source: str = "server",
        client: str | None = None,
    ):
        if custom_tool_grammar not in GRAMMAR_MODES:
            raise ValueError("custom_tool_grammar must be off or validate")
        if source not in SOURCES:
            raise ValueError(f"agent-compat source must be one of {SOURCES}")
        self.enabled = bool(enabled)
        self.custom_tool_grammar = custom_tool_grammar
        self.source = source
        self.client = client

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "custom_tool_grammar": self.custom_tool_grammar,
        }

    def receipt(self) -> dict:
        return {
            "enabled": self.enabled,
            "source": self.source,
            "client": self.client,
            "grammar": self.custom_tool_grammar,
        }

    @property
    def notable(self) -> bool:
        """Whether the decision differs from main's (so it is receipted).

        Plain SDK traffic resolves to ``enabled=False`` from the server
        default with no detected client; its responses stay byte-identical
        to main.
        """
        return self.enabled or self.source != "server" or self.client is not None

    def describe(self) -> str:
        state = "on" if self.enabled else "off"
        return f"agent-compat {state} (source: {self.source})"


# Detection rules derived only from headers observed on the wire
# (tests/fixtures/agent_clients/headers.json): codex-cli 0.145.0 sends
# ``originator: codex_exec`` and ``user-agent: codex_exec/0.145.0 (...)``;
# Claude Code 2.1.269 sends ``user-agent: claude-cli/2.1.269 (...)`` together
# with ``x-app: cli`` and an ``anthropic-beta`` list containing
# ``claude-code-20250219``.  Both signals must agree, versions are matched
# loosely, and each client is only recognized on its own API family.
_CODEX_ORIGINATOR = re.compile(r"^codex_[a-z0-9_]{1,32}$")
_VERSION = r"[0-9]+(?:\.[0-9A-Za-z-]+)*"
_CLAUDE_UA = re.compile(rf"^claude-cli/{_VERSION}(?:[ (]|$)")


def detect_client(headers, path: str) -> str | None:
    """Identify a coding-agent client from its identity headers, or None."""
    lowered = {}
    if headers is not None:
        for key, value in headers.items():
            lowered.setdefault(str(key).lower(), value)

    def get(name):
        value = lowered.get(name.lower())
        return value.strip() if isinstance(value, str) else ""

    agent = get("User-Agent")
    route = path.split("?", 1)[0]
    if route == "/v1/responses":
        originator = get("originator")
        if (
            _CODEX_ORIGINATOR.match(originator)
            and re.match(rf"^{re.escape(originator)}/{_VERSION}(?:[ (]|$)", agent)
        ):
            return "codex"
    elif route in {"/v1/messages", "/v1/messages/count_tokens"}:
        betas = {item.strip() for item in get("anthropic-beta").split(",")}
        if _CLAUDE_UA.match(agent) and (
            get("x-app") or any(item.startswith("claude-code-") for item in betas)
        ):
            return "claude_code"
    return None


def _switch(value, name):
    if value in {"on", "off"}:
        return value
    raise AgentCompatError(f"{name} must be on or off")


def load_tenant_policy(path) -> dict:
    """Load ``--agent-compat-tenants``: tenant id -> {agent_compat, grammar}."""
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("agent-compat tenant policy must be a JSON object")
    policy = {}
    for tenant, entry in raw.items():
        if not isinstance(tenant, str) or not tenant or not isinstance(entry, Mapping):
            raise ValueError("agent-compat tenant policy maps tenant ids to objects")
        unknown = set(entry) - {"agent_compat", "custom_tool_grammar"}
        if unknown or "agent_compat" not in entry:
            raise ValueError(
                f"tenant {tenant!r} policy requires agent_compat and may set "
                "custom_tool_grammar"
            )
        if entry["agent_compat"] not in {"on", "off"}:
            raise ValueError(f"tenant {tenant!r} agent_compat must be on or off")
        grammar = entry.get("custom_tool_grammar")
        if grammar is not None and grammar not in GRAMMAR_MODES:
            raise ValueError(f"tenant {tenant!r} custom_tool_grammar must be off or validate")
        policy[tenant] = {"agent_compat": entry["agent_compat"], "custom_tool_grammar": grammar}
    return policy


class AgentCompatPolicy:
    """Resolve one :class:`AgentCompat` per request.

    Modes: ``off`` never translates; ``on`` always does unless the request
    header says ``off``; ``opt-in`` uses header > tenant policy; ``auto`` (the
    CLI default) uses header > tenant policy > client detection.  The custom
    tool grammar resolves header > tenant policy > server default.
    """

    def __init__(self, mode="auto", custom_tool_grammar="off", tenants=None):
        if mode not in SERVER_MODES:
            raise ValueError(f"agent-compat mode must be one of {SERVER_MODES}")
        if custom_tool_grammar not in GRAMMAR_MODES:
            raise ValueError("custom_tool_grammar must be off or validate")
        self.mode = mode
        self.custom_tool_grammar = custom_tool_grammar
        self.tenants = dict(tenants or {})

    @classmethod
    def coerce(cls, value):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, AgentCompat):
            # Historical fixed setting: enabled == mode "on", else "off".
            return cls("on" if value.enabled else "off", value.custom_tool_grammar)
        raise TypeError("agent_compat must be an AgentCompat or AgentCompatPolicy")

    def status(self) -> dict:
        return {
            "mode": self.mode,
            "enabled": self.mode == "on",
            "custom_tool_grammar": self.custom_tool_grammar,
            "tenant_policies": len(self.tenants),
            "detection": self.mode == "auto",
        }

    def resolve(self, headers, tenant_id="default", path="", counts=None) -> AgentCompat:
        lowered = {}
        for key, value in (headers.items() if headers is not None else ()):
            lowered.setdefault(str(key).lower(), value)
        header = lowered.get(COMPAT_HEADER.lower())
        grammar_header = lowered.get(GRAMMAR_HEADER.lower())
        tenant = self.tenants.get(tenant_id)
        client = detect_client(headers, path) if self.mode == "auto" else None
        grammar = self.custom_tool_grammar
        if tenant is not None and tenant.get("custom_tool_grammar"):
            grammar = tenant["custom_tool_grammar"]
        if grammar_header is not None:
            grammar = grammar_header.strip().lower()
            if grammar not in GRAMMAR_MODES:
                raise AgentCompatError(f"{GRAMMAR_HEADER} must be off or validate")
        if header is not None:
            header = _switch(header.strip().lower(), COMPAT_HEADER)
        if self.mode == "off":
            enabled, source = False, "server"
        elif header is not None:
            enabled, source = header == "on", "header"
        elif self.mode == "on":
            enabled, source = True, "server"
        elif tenant is not None:
            enabled, source = tenant["agent_compat"] == "on", "tenant"
        elif client is not None:
            enabled, source = True, "detected"
        else:
            enabled, source = False, "server"
        resolved = AgentCompat(enabled, grammar, source=source, client=client)
        if resolved.notable:
            count(counts, f"agent_compat_source_{source}")
        if client is not None:
            count(counts, f"agent_compat_detected_{client}")
        return resolved


def require_compat(compat, what):
    """Raise a clear 400 for agent-compat-only history under a non-compat mode."""
    state = compat.describe() if compat is not None else "agent-compat off"
    raise AgentCompatError(
        f"{what} can only be replayed with agent-compat on, but this request "
        f"resolved {state}; a conversation must keep one agent-compat mode "
        "(set X-MLX2-Agent-Compat: on)"
    )


def count(counts, key, amount=1):
    """Increment one agent-compat counter when the engine exposes counts."""
    if counts is not None and amount:
        counts[key] += amount


# ---------------------------------------------------------------------------
# Responses tools


def _custom_format(tool, grammar_mode):
    """Return ``(description_suffix, compiled_constraint_or_None, syntax)``."""
    fmt = tool.get("format", {"type": "text"})
    if not isinstance(fmt, Mapping):
        raise ValueError("custom tool format must be an object")
    kind = fmt.get("type")
    if kind == "text" and set(fmt) == {"type"}:
        return "", None, "text"
    if kind != "grammar" or set(fmt) - {"type", "syntax", "definition"}:
        raise ValueError("custom tool format supports text or grammar")
    syntax, definition = fmt.get("syntax"), fmt.get("definition")
    if syntax not in ("lark", "regex"):
        raise ValueError("custom tool grammar syntax must be lark or regex")
    if not isinstance(definition, str) or not definition:
        raise ValueError("custom tool grammar definition must be nonempty text")
    suffix = (
        f"\n\nThe `input` string must match this {syntax} grammar exactly:\n"
        f"{definition}"
    )
    if grammar_mode == "off":
        return suffix, None, syntax
    from .structured_output import compile_constraint

    try:
        pattern = lark_to_regex(definition) if syntax == "lark" else definition
        constraint = compile_constraint(grammar=pattern)
    except (LarkGrammarError, ValueError) as error:
        raise ValueError(
            f"custom tool {tool.get('name')!r} grammar cannot be enforced: {error}"
        ) from error
    return suffix, constraint, syntax


def translate_responses_tools(value, compat: AgentCompat, counts=None):
    """Translate Responses tools; returns ``(chat_tools, tool_map)``.

    ``tool_map`` records custom tools (with their compiled grammar), namespace
    qualification and dropped hosted tools so output rendering can undo the
    shims exactly.
    """
    if not isinstance(value, list) or not 1 <= len(value) <= 128:
        raise ValueError("tools must contain 1 to 128 definitions")
    tools, custom, namespaces, dropped = [], {}, {}, []

    def function_tool(tool, qualified_name=None):
        unknown = set(tool) - {"type", "name", "description", "parameters", "strict"}
        if unknown:
            raise ValueError(
                "unsupported Responses function fields: " + ", ".join(sorted(unknown))
            )
        if not isinstance(tool.get("name"), str) or not tool["name"]:
            raise ValueError("function tool requires a nonempty name")
        function = {key: tool[key] for key in tool if key != "type"}
        if qualified_name is not None:
            function["name"] = qualified_name
        return {"type": "function", "function": function}

    for tool in value:
        if not isinstance(tool, Mapping):
            raise ValueError("Responses tools must be objects")
        kind = tool.get("type")
        if not isinstance(kind, str):
            # Request validation uses ValueError so the HTTP boundary returns 400.
            raise ValueError("Responses tool type must be text")  # noqa: TRY004
        if kind == "function":
            tools.append(function_tool(tool))
        elif kind == "custom":
            unknown = set(tool) - {"type", "name", "description", "format"}
            if unknown:
                raise ValueError(
                    "unsupported custom tool fields: " + ", ".join(sorted(unknown))
                )
            name = tool.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("custom tool requires a name")
            description = tool.get("description", "")
            if not isinstance(description, str):
                raise ValueError("custom tool description must be text")
            suffix, constraint, syntax = _custom_format(tool, compat.custom_tool_grammar)
            custom[name] = {"constraint": constraint, "syntax": syntax}
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": (
                            description
                            + "\n\nThis is a FREEFORM tool: put the raw text in the"
                            " `input` argument."
                            + suffix
                        ).strip(),
                        "parameters": {
                            "type": "object",
                            "properties": {"input": {"type": "string"}},
                            "required": ["input"],
                            "additionalProperties": False,
                        },
                    },
                }
            )
            count(counts, "agent_compat_custom_tools_declared")
        elif kind == "namespace":
            unknown = set(tool) - {"type", "name", "description", "tools"}
            if unknown:
                raise ValueError(
                    "unsupported namespace tool fields: " + ", ".join(sorted(unknown))
                )
            namespace, inner = tool.get("name"), tool.get("tools")
            if not isinstance(namespace, str) or not namespace or "." in namespace:
                raise ValueError("namespace tool requires a dot-free name")
            if not isinstance(inner, list) or not inner:
                raise ValueError("namespace tool requires nonempty tools")
            for member in inner:
                if not isinstance(member, Mapping) or member.get("type") != "function":
                    raise ValueError("namespace tools may only contain function tools")
                qualified = f"{namespace}.{member.get('name')}"
                namespaces[qualified] = (namespace, member.get("name"))
                tools.append(function_tool(member, qualified))
            count(counts, "agent_compat_namespace_tools")
        elif kind in DROPPABLE_HOSTED_TOOLS:
            dropped.append(kind)
            count(counts, "agent_compat_hosted_tools_dropped")
        else:
            from .api_resources import CapabilityUnavailable

            raise CapabilityUnavailable(
                f"the loaded route does not implement {kind!r} tools"
            )
    names = [tool["function"].get("name") for tool in tools]
    if len(names) != len(set(names)):
        raise ValueError("tool names must be unique after namespace flattening")
    return tools, {"custom": custom, "namespaces": namespaces, "dropped": dropped}


def shim_arguments(text: str) -> str:
    """Encode a freeform custom-tool input exactly as the function shim does."""
    return json.dumps({"input": text}, ensure_ascii=False)


def qualified_call_name(item: Mapping) -> str:
    namespace = item.get("namespace")
    if namespace is None:
        return item["name"]
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("function_call namespace must be nonempty text")
    return f"{namespace}.{item['name']}"


def tool_output_text(output, *, kind) -> str:
    """Flatten Codex's text-part array tool output; media parts fail closed."""
    if isinstance(output, str):
        return output
    if not isinstance(output, list):
        raise ValueError(f"{kind} requires text or a list of text parts")
    pieces = []
    for part in output:
        if not isinstance(part, Mapping):
            raise ValueError(f"{kind} parts must be objects")
        if part.get("type") in ("input_text", "output_text", "text") and isinstance(
            part.get("text"), str
        ):
            pieces.append(part["text"])
        else:
            raise ValueError(
                f"{kind} only supports text parts on this route"
            )
    return "".join(pieces)


def rewrite_responses_output(payload, tool_map, compat: AgentCompat, counts=None):
    """Undo the tool shims on a rendered Responses payload, in place.

    Raises ``ToolContractError`` when a custom-tool call does not carry a
    string ``input`` or (in validate mode) does not match the declared grammar.
    """
    from .openai_compat import ToolContractError

    custom = tool_map.get("custom", {})
    namespaces = tool_map.get("namespaces", {})
    output = payload["output"]
    has_calls = any(item.get("type") == "function_call" for item in output)
    for position, item in enumerate(output):
        if item.get("type") == "message":
            item["phase"] = "commentary" if has_calls else "final_answer"
            count(counts, "agent_compat_phase_outputs")
            continue
        if item.get("type") != "function_call":
            continue
        name = item["name"]
        if name in custom:
            try:
                arguments = json.loads(item["arguments"])
            except (TypeError, json.JSONDecodeError) as error:
                raise ToolContractError(
                    f"custom tool {name!r} call is not valid shim JSON"
                ) from error
            if (
                not isinstance(arguments, dict)
                or set(arguments) != {"input"}
                or not isinstance(arguments["input"], str)
            ):
                raise ToolContractError(
                    f"custom tool {name!r} call must carry one string input"
                )
            text = arguments["input"]
            constraint = custom[name].get("constraint")
            if constraint is not None:
                if constraint.fullmatch(text) is None:
                    count(counts, "agent_compat_custom_tool_grammar_rejected")
                    # The client only sees "does not match"; without the text
                    # a rejection cannot be diagnosed after the fact.
                    _LOG.warning(
                        "agent-compat: %s input rejected by its grammar: %.600r",
                        name, text,
                    )
                    raise ToolContractError(
                        f"custom tool {name!r} input does not match its declared grammar"
                    )
                count(counts, "agent_compat_custom_tool_grammar_validated")
            call_id = item["call_id"]
            output[position] = {
                "id": "ctc_" + str(item["id"]).removeprefix("call_").removeprefix("fc_"),
                "type": "custom_tool_call",
                "status": "completed",
                "call_id": call_id,
                "name": name,
                "input": text,
            }
            count(counts, "agent_compat_custom_tool_calls")
        elif name in namespaces:
            namespace, inner = namespaces[name]
            item["name"] = inner
            item["namespace"] = namespace
            count(counts, "agent_compat_namespace_calls")
    receipt = payload.get("mlx2")
    if isinstance(receipt, dict):
        receipt["agent_compat"] = {
            **(receipt.get("agent_compat") or compat.receipt()),
            "custom_tools": sorted(custom),
            "namespaces": sorted({pair[0] for pair in namespaces.values()}),
            "dropped_tools": list(tool_map.get("dropped", ())),
        }
    return payload


# ---------------------------------------------------------------------------
# Message placement shared by both APIs


def fold_system_messages(messages, counts=None):
    """Keep one leading system message; fold later ones into user turns.

    Codex sends several ``developer`` messages and Claude Code sends
    mid-conversation ``system`` messages.  Common chat templates reject a
    system message anywhere but first, so leading system text is joined and
    later system text is merged, deterministically, into the adjacent user
    turn (or becomes one).
    """
    result = []
    leading = True
    for message in messages:
        role = message.get("role")
        if role == "system":
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError("system messages must be text on this route")
            if leading:
                if result:
                    result[0] = {
                        **result[0],
                        "content": result[0]["content"] + "\n\n" + content,
                    }
                    count(counts, "agent_compat_system_folded")
                else:
                    result.append(dict(message))
                continue
            count(counts, "agent_compat_system_folded")
            if (
                result
                and result[-1].get("role") == "user"
                and isinstance(result[-1].get("content"), str)
            ):
                result[-1] = {
                    **result[-1],
                    "content": result[-1]["content"] + "\n\n" + content,
                }
            else:
                result.append({"role": "user", "content": content})
            continue
        leading = False
        result.append(message)
    return result
