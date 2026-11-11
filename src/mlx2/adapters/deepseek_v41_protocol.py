# SPDX-License-Identifier: MIT
"""V4.1 public text protocol layered over the pinned official encoder.

The shared OutputParser supplies incremental marker buffering, code-fence and
reasoning separation, truncation receipts, and bounded parse work. This module
owns only the V4.1 spaced DSML grammar and its typed parameter semantics.
"""
import copy
import json
import re

from ..output import OutputParser
from ..reasoning_options import normalize_reasoning_options
from ..runtime.tool_parsers._schema import (
    executable_schema,
    json_native,
    schema_value_matches,
)

CALLS_OPEN = '<｜DSML｜ calls>'
CALLS_CLOSE = '</｜DSML｜ calls>'
INVOKE_CLOSE = '</｜DSML｜ invoke>'
PARAM_CLOSE = '</｜DSML｜ parameter>'
_NAME = r'[^"<>\x00-\x1f]{1,255}'
_INVOKE = re.compile(r'<｜DSML｜ invoke name="(' + _NAME + r')">')
_PARAM = re.compile(r'<｜DSML｜ parameter name="(' + _NAME + r')" string="(true|false)">')


def _namespace(value):
    if isinstance(value, dict):
        if not isinstance(value.get('name'), str) or not isinstance(value.get('description', ''), str):
            raise ValueError('V4.1 tool namespace requires a name and text description')  # noqa: TRY004 - public request refusal contract
        value = value.get('name')
    if value is not None and (not isinstance(value, str) or not re.fullmatch(_NAME, value) or '::' in value):
        raise ValueError('V4.1 tool namespace must be a representable nonempty name')
    return value


def _function_name(function, namespace=None):
    name = function.get('name')
    if not isinstance(name, str) or not re.fullmatch(_NAME, name) or name.count('::') > 1:
        raise ValueError('V4.1 tool names must be representable DSML names with at most one namespace')
    explicit, nested = _namespace(namespace), _namespace(function.get('namespace'))
    if explicit is not None and nested is not None and explicit != nested:
        raise ValueError('V4.1 tool namespaces disagree')
    namespace = explicit if explicit is not None else nested
    if '::' in name:
        prefix, bare = name.split('::')
        if not prefix or not bare or namespace not in (None, prefix):
            raise ValueError('V4.1 qualified tool name and namespace disagree')
    elif namespace is not None:
        name = namespace + '::' + name
    if not re.fullmatch(_NAME, name):
        raise ValueError('V4.1 qualified tool name cannot be represented in DSML')
    return name


def _normalize_tools(tools):
    if not isinstance(tools, list):
        raise ValueError('V4.1 tools must be a list of function definitions')  # noqa: TRY004 - public request refusal contract
    normalized, names = [], set()
    for tool in tools:
        if not isinstance(tool, dict) or tool.get('type') != 'function' or not isinstance(tool.get('function'), dict):
            raise ValueError('V4.1 tools require function definitions')
        function = dict(tool['function'])
        if tool.get('defer_loading') or function.get('defer_loading'):
            raise ValueError('V4.1 deferred tool loading requires a tool-search protocol; supply activated function definitions')
        name = _function_name(function, tool.get('namespace'))
        if name in names:
            raise ValueError('V4.1 tool names must be unique after namespace normalization')
        names.add(name)
        if not isinstance(function.get('parameters', {}), dict) or type(function.get('strict', False)) is not bool:
            raise ValueError('V4.1 tool schema is invalid')
        # The official encoder folds namespace descriptions into the function.
        # Preserve that prompt information while using one public identity in
        # the renderer, parser, grammar, route receipt and terminal contract.
        namespace = tool.get('namespace', function.get('namespace'))
        if isinstance(namespace, dict) and namespace.get('description'):
            if not isinstance(namespace['description'], str) or not isinstance(function.get('description', ''), str):
                raise ValueError('V4.1 tool namespace description must be text')
            function['description'] = namespace['description'] + '\n' + function.get('description', '')
        function.pop('namespace', None)
        function['name'] = name
        normalized.append({key: value for key, value in {**tool, 'function': function}.items() if key != 'namespace'})
    return normalized


def normalize_request(request):
    result = normalize_reasoning_options(request)
    template = result.pop('chat_template_kwargs', {}) or {}
    if set(template) - {'enable_thinking', 'preserve_thinking'}:
        raise ValueError('V4.1 supports only enable_thinking/preserve_thinking template options')
    if 'preserve_thinking' in template:
        if 'preserve_thinking' in result and result['preserve_thinking'] != template['preserve_thinking']:
            raise ValueError('preserve_thinking options disagree')
        result['preserve_thinking'] = template['preserve_thinking']
    if 'preserve_thinking' in result and type(result['preserve_thinking']) is not bool:
        raise ValueError('preserve_thinking must be boolean')
    if result.get('verification') or 'tool_call_format' in result:
        raise ValueError('V4.1 does not support verification/tool_call_format template options')
    if result.get('tools') is not None:
        original = result['tools']
        result['tools'] = _normalize_tools(original)
        choice = result.get('tool_choice')
        if isinstance(choice, dict):
            function = choice.get('function')
            if (choice.get('type') != 'function' or set(choice) - {'type', 'function', 'namespace'}
                    or not isinstance(function, dict) or set(function) - {'name', 'namespace'}):
                raise ValueError('V4.1 named tool_choice requires a function')
            selected = _function_name(function, choice.get('namespace'))
            names = {tool['function']['name'] for tool in result['tools']}
            if selected not in names:
                aliases = {after['function']['name'] for before, after in zip(original, result['tools'])
                           if before['function']['name'] == selected}
                if len(aliases) != 1:
                    raise ValueError('V4.1 named tool_choice does not select a unique declared function')
                selected = aliases.pop()
            result['tool_choice'] = {'type': 'function', 'function': {'name': selected}}
    effort = result.get('reasoning_effort')
    # Pinned deepseek-recipe request conversion aliases, followed by the
    # V4.1 rendering law (xhigh intentionally has high's strength, 75).
    levels = {'minimal': 50, 'low': 50, 'medium': 75, 'high': 75,
              'xhigh': 75, 'max': 100, 'ultra': 100}
    result.setdefault('_mlx2_reasoning_effort_requested', effort)
    enabled = result.get('enable_thinking', False)
    result['reasoning_effort'] = (
        levels.get(effort, effort) if effort not in (None, 'none') else 75
    ) if enabled else None
    return result


def messages_for_encoding(request):
    """Validate/associate OpenAI text history before official DSML rendering."""
    if not isinstance(request.get('messages'), list):
        raise ValueError('V4.1 messages must be a list')  # noqa: TRY004 - public request refusal contract
    messages = copy.deepcopy(request['messages'])
    pending = set()
    seen = set()
    has_schema_history = any(isinstance(message, dict) and 'tool_schema_state' in message for message in messages)
    active = None
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get('role') not in ('system', 'user', 'assistant', 'tool'):
            raise ValueError('V4.1 requires text system/user/assistant/tool history')
        if 'tool_schema_state' in message:
            if message.get('role') != 'system' or not isinstance(message['tool_schema_state'], list):
                raise ValueError('V4.1 tool schema activation requires a system message with a schema list')
            schemas = _normalize_tools(message.pop('tool_schema_state'))
            active = {tool['function']['name']: tool for tool in schemas}
            if index == 0:
                message['tools'] = schemas
            else:
                from ..runtime.models.deepseek_v41_encoding import (
                    tools_from_openai_format,
                )

                message['content'] = message.get('content', '') + '\n\n### Updated Tool Schemas\n\n' + json.dumps(tools_from_openai_format(schemas), ensure_ascii=False) + '\nOnly tools in this updated list are currently available.'
        elif has_schema_history and active is None:
            raise ValueError('V4.1 tool history must declare its initial active schema state')
        if message.get('content') is None and message.get('role') == 'assistant' and message.get('tool_calls'):
            message['content'] = ''
        if not isinstance(message.get('content'), str):
            raise ValueError('V4.1 text-only route requires string message content')  # noqa: TRY004 - public request refusal contract
        if message.get('reasoning_content') is not None and not isinstance(message['reasoning_content'], str):
            raise ValueError('V4.1 history reasoning_content must be text')
        # DeepSeek clients must echo reasoning_content and send a turn that
        # had none as " " (the hosted API refuses ""). Rendered verbatim, that
        # space lands inside the think block as a token the model never
        # sampled, so the cached prefix breaks there and the rest of the
        # conversation re-prefills (idea from antirez/ds4#1296; see
        # docs/PROVENANCE.md). Only that pad is absent reasoning: other
        # whitespace may be what the model sampled and mlx2 returned
        # verbatim, and its exact echo must keep rendering as sampled.
        if message.get('reasoning_content') == ' ':
            del message['reasoning_content']
        if message.get('role') == 'tool':
            identity = message.get('tool_call_id')
            if not isinstance(identity, str) or identity not in pending:
                raise ValueError('V4.1 tool result must reference an outstanding assistant tool_call_id exactly once')
            pending.remove(identity)
        elif message.get('tool_calls'):
            if message['role'] != 'assistant' or pending or not isinstance(message['tool_calls'], list):
                raise ValueError('V4.1 assistant calls require resolved prior tool results')
            for call in message['tool_calls']:
                if not isinstance(call, dict) or not isinstance(call.get('function'), dict):
                    raise ValueError('V4.1 tool history requires function call objects')  # noqa: TRY004 - public request refusal contract
                identity = call.get('id')
                function = dict(call['function'])
                if not isinstance(identity, str) or not identity or identity in seen:
                    raise ValueError('V4.1 tool calls require unique nonempty ids')
                if call.get('type', 'function') != 'function':
                    raise ValueError('V4.1 tool history has an invalid function name')
                function['name'] = _function_name(function, call.pop('namespace', None))
                function.pop('namespace', None)
                if has_schema_history and function['name'] not in active:
                    raise ValueError('V4.1 tool history calls a function before activation or after removal')
                arguments = function.get('arguments', {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments, object_pairs_hook=_pairs, parse_constant=_constant)
                    except (ValueError, TypeError, RecursionError) as exc:
                        raise ValueError('V4.1 tool history arguments must be a JSON object') from exc
                if not isinstance(arguments, dict) or not json_native(arguments):
                    raise ValueError('V4.1 tool history arguments must be a JSON-native object')
                if any(not re.fullmatch(_NAME, name) for name in arguments):
                    raise ValueError('V4.1 history argument names must be representable DSML names')
                function['arguments'] = arguments
                call['function'] = function
                pending.add(identity)
                seen.add(identity)
        elif pending:
            raise ValueError('V4.1 tool results must immediately follow their assistant calls')
    if pending:
        raise ValueError('V4.1 history has unresolved assistant tool calls')
    tools = request.get('tools') if request.get('tool_choice', 'auto') != 'none' else None
    if has_schema_history:
        for tool in tools or ():
            name = tool['function']['name']
            if name not in active or active[name] != tool:
                raise ValueError('V4.1 callable tools differ from the final activated schema state')
    elif tools:
        if not messages or messages[0]['role'] != 'system':
            messages.insert(0, {'role': 'system', 'content': ''})
        messages[0]['tools'] = copy.deepcopy(tools)
    return messages


class IncompleteDSML(ValueError):
    # A whole-block closer may be a literal inside a typed JSON/string value.
    # The shared incremental scanner retries at the next block closer.
    tool_call_close_quoted = True


def _pairs(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError('duplicate JSON argument key')
        result[name] = value
    return result


def _constant(value):
    raise ValueError('nonfinite JSON argument: ' + value)


def partial_call_names(text, *, marker_allowed=lambda marker, position: True):
    """Read committed invoke identities, never markup inside argument data.

    A name belongs in the receipt even when it is undeclared: terminal tool
    validation must distinguish it from a truncation before any name. Once an
    argument is incomplete, nothing beyond it can establish another invoke.
    """
    cursor, names = 0, []
    decoder = json.JSONDecoder(object_pairs_hook=_pairs, parse_constant=_constant)
    invoke_prefix = '<｜DSML｜ invoke name="'
    while cursor < len(text):
        cursor += len(text[cursor:]) - len(text[cursor:].lstrip())
        if cursor == len(text):
            break
        match = _INVOKE.match(text, cursor)
        if match is None:
            tail = text[cursor:]
            if invoke_prefix.startswith(tail) or (tail.startswith(invoke_prefix) and '"' not in tail[len(invoke_prefix):]):
                break
            raise ValueError('V4.1 truncated tool call has an unreadable invoke name')
        if not marker_allowed('<｜DSML｜ invoke', cursor):
            raise ValueError('V4.1 truncated invoke lacks its structural token')
        names.append(match[1])
        cursor = match.end()
        while True:
            cursor += len(text[cursor:]) - len(text[cursor:].lstrip())
            if text.startswith(INVOKE_CLOSE, cursor) and marker_allowed(INVOKE_CLOSE, cursor):
                cursor += len(INVOKE_CLOSE)
                break
            parameter = _PARAM.match(text, cursor)
            if parameter is None or not marker_allowed('<｜DSML｜ parameter', cursor):
                return names
            cursor = parameter.end()
            if parameter[2] == 'true':
                end = text.find(PARAM_CLOSE, cursor)
                while end >= 0 and not marker_allowed(PARAM_CLOSE, end):
                    end = text.find(PARAM_CLOSE, end + 1)
                if end < 0:
                    return names
                cursor = end
            else:
                cursor += len(text[cursor:]) - len(text[cursor:].lstrip())
                try:
                    _, cursor = decoder.raw_decode(text, cursor)
                except (ValueError, RecursionError):
                    return names
                cursor += len(text[cursor:]) - len(text[cursor:].lstrip())
            if not text.startswith(PARAM_CLOSE, cursor) or not marker_allowed(PARAM_CLOSE, cursor):
                return names
            cursor += len(PARAM_CLOSE)
    return names


def parse_calls(text, tools, *, marker_allowed=lambda marker, position: True):
    functions = {tool['function']['name']: tool['function'] for tool in tools}
    cursor, calls = 0, []
    decoder = json.JSONDecoder(object_pairs_hook=_pairs, parse_constant=_constant)
    while cursor < len(text):
        cursor += len(text[cursor:]) - len(text[cursor:].lstrip())
        if cursor == len(text):
            break
        match = _INVOKE.match(text, cursor)
        if match is None or not marker_allowed('<｜DSML｜ invoke', cursor):
            raise ValueError('V4.1 calls block requires a spaced DSML invoke')
        name = match[1]
        if name not in functions:
            raise ValueError('V4.1 called an undeclared function: ' + name)
        cursor, arguments = match.end(), {}
        while True:
            cursor += len(text[cursor:]) - len(text[cursor:].lstrip())
            if text.startswith(INVOKE_CLOSE, cursor) and marker_allowed(INVOKE_CLOSE, cursor):
                cursor += len(INVOKE_CLOSE)
                break
            parameter = _PARAM.match(text, cursor)
            if parameter is None or not marker_allowed('<｜DSML｜ parameter', cursor):
                raise IncompleteDSML('V4.1 invoke has an incomplete or malformed parameter')
            key, string = parameter[1], parameter[2]
            if key in arguments:
                raise ValueError('V4.1 duplicate parameter: ' + key)
            cursor = parameter.end()
            if string == 'true':
                end = text.find(PARAM_CLOSE, cursor)
                while end >= 0 and not marker_allowed(PARAM_CLOSE, end):
                    end = text.find(PARAM_CLOSE, end + 1)
                if end < 0:
                    raise IncompleteDSML('V4.1 string parameter is incomplete')
                value = text[cursor:end]
                cursor = end
            else:
                cursor += len(text[cursor:]) - len(text[cursor:].lstrip())
                try:
                    value, cursor = decoder.raw_decode(text, cursor)
                except json.JSONDecodeError as exc:
                    raise IncompleteDSML('V4.1 parameter must contain valid JSON') from exc
                cursor += len(text[cursor:]) - len(text[cursor:].lstrip())
            if not text.startswith(PARAM_CLOSE, cursor) or not marker_allowed(PARAM_CLOSE, cursor):
                raise IncompleteDSML('V4.1 parameter is missing its closing tag')
            cursor += len(PARAM_CLOSE)
            arguments[key] = value
        function = functions[name]
        schema = function.get('parameters') or {'type': 'object'}
        if function.get('strict'):
            if not schema_value_matches(arguments, executable_schema(schema)):
                raise ValueError('V4.1 strict tool arguments violate the declared schema')
        elif set(schema.get('required', ())) - set(arguments):
            raise ValueError('V4.1 tool arguments omit a required parameter')
        calls.append({'name': name, 'arguments': arguments})
    if not calls:
        raise ValueError('V4.1 calls block must contain an invoke')
    return calls


class DeepSeekV41OutputParser(OutputParser):
    tool_open = CALLS_OPEN
    tool_close = CALLS_CLOSE

    def __init__(self, *, tokenizer=None, **kwargs):
        self._special_ids = {}
        self._special_spans = set()
        self._received_chars = 0
        self._token_aware = False
        if tokenizer is not None:
            for spelling in ('｜DSML｜', '<think>', '</think>'):
                ids = tokenizer.encode(spelling, add_special_tokens=False)
                if len(ids) == 1:
                    self._special_ids[int(ids[0])] = spelling
        kwargs['parse_tool'] = self._parse_calls
        kwargs['partial_names'] = self._partial_names
        super().__init__(**kwargs)

    def record_output_token(self, text, token):
        """Bind detokenized special segments to their actual generated ids.

        A final decoder flush may contribute plain text without a token. A
        dedicated special token must be the suffix of its decoder segment;
        delayed plain bytes before it retain their plain-token provenance.
        """
        self._token_aware = True
        spelling = self._special_ids.get(token)
        if spelling is not None:
            if not text.endswith(spelling):
                raise ValueError('V4.1 detokenizer did not preserve a dedicated special-token boundary')
            self._special_spans.add((self._received_chars + len(text) - len(spelling), spelling))
        self._received_chars += len(text)

    def _marker_allowed(self, marker, absolute_offset):
        if not self._token_aware:
            return True  # explicit text-only input, as in deepseek-recipe
        if '｜DSML｜' in marker:
            absolute_offset += marker.index('｜DSML｜')
            marker = '｜DSML｜'
        return (absolute_offset, marker) in self._special_spans

    def _parse_calls(self, text, tools):
        origin = self._buffer_origin
        return parse_calls(text, tools, marker_allowed=lambda marker, offset: self._marker_allowed(marker, origin + offset))

    def _partial_names(self, text, tools):
        return partial_call_names(text, marker_allowed=lambda marker, offset: self._marker_allowed(marker, self._buffer_origin + offset))

    def push(self, text, *, final=False):
        events = super().push(text, final=final)
        if self.channel != 'tool':
            self._special_spans = {span for span in self._special_spans if span[0] >= self._buffer_origin}
        return events


def constrained_tool_grammar(tools, tool_choice, *, parallel_tool_calls=True):
    """Canonical spaced DSML with typed values and bounded schema enforcement.

    Raw string values exclude structural closing tags: the text-only protocol
    cannot distinguish their literal spelling from a closing parameter. JSON values retain
    the JSON decoder's quote-aware delimiter handling.
    """
    import regex

    from ..runtime.tool_parsers._schema import raw_string_pattern
    from ..structured_output import _schema_pattern

    functions = [tool['function'] for tool in tools]
    if isinstance(tool_choice, dict):
        functions = [function for function in functions if function['name'] == tool_choice['function']['name']]
    if not functions:
        raise ValueError('V4.1 constrained tool_choice selects no function')
    invokes = []
    for function in functions:
        name = function['name']
        if not re.fullmatch(_NAME, name):
            raise ValueError('tool name cannot be represented in V4.1 DSML')
        schema = executable_schema(function.get('parameters') or {'type': 'object', 'properties': {}})
        if schema.get('type', 'object') != 'object' or schema.get('additionalProperties', False) not in (False, None):
            raise ValueError('V4.1 constrained arguments require a closed object schema')
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        if not isinstance(properties, dict) or len(properties) > 64 or set(required) - set(properties):
            raise ValueError('invalid V4.1 tool parameter schema')
        parts = []
        for key in [k for k in properties if k in required] + [k for k in properties if k not in required]:
            if not re.fullmatch(_NAME, key):
                raise ValueError('parameter name cannot be represented in V4.1 DSML')
            value = raw_string_pattern(properties[key], forbidden=(PARAM_CLOSE, INVOKE_CLOSE, CALLS_CLOSE))
            is_string = value is not None
            if value is None:
                value = _schema_pattern(properties[key], finite_numbers=True)
            block = r'\n' + regex.escape(f'<｜DSML｜ parameter name="{key}" string="{str(is_string).lower()}">') + value + regex.escape(PARAM_CLOSE)
            parts.append(block if key in required else '(?:' + block + ')?')
        invokes.append(regex.escape(f'<｜DSML｜ invoke name="{name}">') + ''.join(parts) + r'\n' + regex.escape(INVOKE_CLOSE))
    invoke = '(?:' + '|'.join(invokes) + ')'
    repeat = '(?:\n' + invoke + ')*' if parallel_tool_calls else ''
    return regex.escape(CALLS_OPEN) + r'\n' + invoke + repeat + r'\n' + regex.escape(CALLS_CLOSE)
