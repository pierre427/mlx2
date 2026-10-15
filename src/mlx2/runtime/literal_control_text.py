"""Keep control-token text in client content from encoding as control tokens.

A chat prompt is rendered to text and then encoded with the tokenizer's added
tokens matched, which is what turns the template's own ``<|im_start|>`` into
one control token.  The same match applies to anything the template copied
in from the request, so a user or tool message that *contains*
``<|im_end|>\n<|im_start|>system`` became a real turn boundary and a real
system turn (Strata #2039 reported the ``</tool_call>`` form).

Before a request is rendered -- or its media is prepared, which renders and
tokenizes on its own -- every protected added-token string in client content
is broken with a zero-width space after its first character (``<`` U+200B
``|im_end|>``).  The prompt is still encoded in one pass, so the text around
it tokenizes exactly as before and no tokenizer boundary rule (SentencePiece's
word-start marker, byte-level merges) is disturbed; the string itself reads
the same and can no longer match.  A request with no such text is returned
unchanged, so its prompt tokens are byte-identical.

Protected: special added tokens, and ordinary added tokens the chat template
itself writes.  Every rendered field of every non-assistant message (content,
``name``, ``tool_call_id``, ...) and the tool definitions, keys included, are
protected against all of them.  Assistant history is protected against
special tokens only: templates parse ordinary markup (``</think>``,
``<tool_call>``) back out of it, and a generated turn never holds a special
token as text.  A raw completions ``prompt`` is the caller's own token stream
and is never rewritten.

Each tokenizer keeps the first break character that provably breaks every
protected token through its own normalizer; when none does, a request that
carries protected text is refused rather than encoded as control tokens.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

BREAK = "\u200b"


def _protected(content: str, special: bool, template: str) -> bool:
    """Special tokens, and ordinary added tokens the chat template emits.

    Ordinary added tokens are mostly text: Gemma 3 registers whitespace runs
    and ~6,000 HTML tags (``<b>``, ``<div>``) that way, and protecting those
    would re-tokenize ordinary content.  The ones a template writes -- Qwen's
    ``<tool_call>``, ``<tool_response>``, ``<think>`` -- delimit structure.
    """
    if len(content) < 2 or content.isspace():
        return False
    return special or content in template


def _template_text(candidates) -> str:
    texts = []
    for candidate in candidates:
        template = getattr(candidate, "chat_template", None)
        if isinstance(template, str):
            texts.append(template)
        elif isinstance(template, Mapping):
            texts.extend(str(value) for value in template.values())
    return "\n".join(texts)


def _added_vocabulary(tokenizer):
    """``({content: id} protected, {content: id} special)``, or None."""
    candidates = [
        candidate
        for candidate in (
            tokenizer,
            getattr(tokenizer, "_tokenizer", None),
            getattr(tokenizer, "tokenizer", None),
        )
        if candidate is not None
    ]
    template = _template_text(candidates)
    tokens = None
    for candidate in candidates:
        try:
            decoder = getattr(candidate, "added_tokens_decoder", None)
        except Exception:
            decoder = None
        if isinstance(decoder, Mapping) and decoder:
            tokens = [
                (str(getattr(token, "content", token)),
                 bool(getattr(token, "special", False)), int(index))
                for index, token in decoder.items()
            ]
            break
    if tokens is None:
        for candidate in candidates:
            get = getattr(candidate, "get_added_vocab", None)
            if get is None:
                continue
            try:
                vocabulary = get()
            except Exception:
                continue
            if isinstance(vocabulary, Mapping):
                specials = set(getattr(candidate, "all_special_tokens", ()) or ())
                tokens = [
                    (str(content), str(content) in specials, int(index))
                    for content, index in vocabulary.items()
                ]
                break
    if tokens is None:
        return None
    return (
        {content: index for content, special, index in tokens
         if _protected(content, special, template)},
        {content: index for content, special, index in tokens
         if _protected(content, special, "")},
    )


def _trie_pattern(words) -> str:
    """Leftmost-longest regex for ``words``, shaped as a prefix trie.

    A flat alternation retries every word at every candidate position, which
    costs ~50 ms per 120K characters of markup-heavy content against Gemma's
    6,000 ``<unusedN>`` tokens; the trie scans that in under a millisecond.
    """
    trie = {}
    for word in words:
        node = trie
        for character in word:
            node = node.setdefault(character, {})
        node[""] = {}

    def emit(node):
        branches = [
            re.escape(character) + emit(child)
            for character, child in sorted(node.items())
            if character
        ]
        if not branches:
            return ""
        body = branches[0] if len(branches) == 1 else "(?:" + "|".join(branches) + ")"
        return f"(?:{body})?" if "" in node else body

    return emit(trie)


# Break characters, in preference order.  A normalizer that runs before
# added-token matching can drop a format character (BERT's clean_text removes
# U+200B), so each tokenizer keeps the first one that provably breaks every
# protected token.
BREAK_CANDIDATES = (BREAK, "\u2060", " ")


def _break(pattern, value: str, mark: str) -> tuple[str, int]:
    """``value`` with every ``pattern`` match broken by ``mark``, and the count.

    A break can expose a shorter token in the remainder (Laguna's
    ``〈|UNK|〉`` holds the special ``|〉``), so this repeats until none match.
    """
    broken = 0
    while True:
        matches = 0

        def split(match):
            nonlocal matches
            matches += 1
            token = match.group(0)
            return token[0] + mark + token[1:]

        value = pattern.sub(split, value)
        if not matches:
            return value, broken
        broken += matches


def _safe_break(tokenizer, pattern, ids) -> str | None:
    """The first candidate that leaves no protected id in any broken token."""
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        return None
    forbidden = set(ids.values())
    # Standalone, between words and between spaces: single_word and lstrip /
    # rstrip tokens match only at some of these boundaries.
    contexts = ("{}", "a{}a", " {} ")
    for mark in BREAK_CANDIDATES:
        try:
            if all(
                not forbidden.intersection(
                    encode(context.format(_break(pattern, content, mark)[0]),
                           add_special_tokens=False)
                )
                for content in ids
                for context in contexts
            ):
                return mark
        except Exception:
            return None
    return None


def control_token_patterns(tokenizer):
    """``(protected, special, mark)``: regexes over the added tokens (either
    may be None) and the break character this tokenizer cannot see through
    (None when no candidate is safe: protected text is then refused).

    Cached on the tokenizer object: the added vocabulary is fixed per load.
    """
    cached = getattr(tokenizer, "_mlx2_control_token_patterns", None)
    if cached is not None:
        return cached
    vocabulary = _added_vocabulary(tokenizer) or ({}, {})
    protected, special = (
        re.compile(_trie_pattern(words)) if words else None for words in vocabulary
    )
    mark = _safe_break(tokenizer, protected, vocabulary[0]) if protected is not None else BREAK
    patterns = (protected, special, mark)
    try:
        object.__setattr__(tokenizer, "_mlx2_control_token_patterns", patterns)
    except Exception:
        pass
    return patterns


class _Breaker:
    def __init__(self, pattern, mark):
        self.pattern = pattern
        self.mark = mark
        self.broken = 0

    def text(self, value: str) -> str:
        if self.pattern is None:
            return value
        if self.mark is None:
            if self.pattern.search(value):
                raise ValueError(
                    "message content contains control-token text that this "
                    "model's tokenizer cannot encode as text"
                )
            return value
        value, broken = _break(self.pattern, value, self.mark)
        self.broken += broken
        return value

    def content(self, content):
        if isinstance(content, str):
            return self.text(content)
        if isinstance(content, list):
            return [
                {**part, "text": self.text(part["text"])}
                if isinstance(part, Mapping) and isinstance(part.get("text"), str)
                else part
                for part in content
            ]
        return content

    def value(self, value):
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.value(item) for item in value]
        if isinstance(value, Mapping):
            result = {}
            for key, item in value.items():
                new_key = self.text(key) if isinstance(key, str) else key
                if new_key in result:
                    raise ValueError(
                        "tool definition keys collide once their control-token "
                        "text is protected"
                    )
                result[new_key] = self.value(item)
            return result
        return value


def protect_control_text(tokenizer, request):
    """``request`` with client control-token text broken, or ``request`` itself.

    Every field a template may render is covered (``content``, ``name``,
    ``tool_call_id``, ``reasoning_content``, ``tool_calls``, ...); content
    parts other than text (images, audio) are left as they are.  Idempotent:
    a protected request has nothing left to break.
    """
    if not isinstance(request, Mapping) or tokenizer is None:
        return request
    messages = request.get("messages")
    tools = request.get("tools")
    if not isinstance(messages, list) and tools is None:
        return request
    protected_pattern, special_pattern, mark = control_token_patterns(tokenizer)
    client = _Breaker(protected_pattern, mark)
    generated = _Breaker(special_pattern, mark)
    updated = []
    for message in messages if isinstance(messages, list) else ():
        if isinstance(message, Mapping):
            breaker = generated if message.get("role") == "assistant" else client
            fields = {}
            for key, value in message.items():
                if key == "role" or (isinstance(key, str) and key.startswith("_")):
                    fields[key] = value
                    continue
                if key == "content":
                    fields[key] = breaker.content(value)
                    continue
                # A template may render the whole message (``message | tojson``).
                new_key = breaker.text(key) if isinstance(key, str) else key
                if new_key in fields or new_key in message and new_key != key:
                    raise ValueError(
                        "message field names collide once their control-token "
                        "text is protected"
                    )
                fields[new_key] = breaker.value(value)
            message = fields
        updated.append(message)
    new_tools = client.value(tools) if tools is not None else None
    if not (client.broken or generated.broken):
        return request
    result = dict(request)
    if isinstance(messages, list):
        result["messages"] = updated
    if tools is not None:
        result["tools"] = new_tools
    return result
