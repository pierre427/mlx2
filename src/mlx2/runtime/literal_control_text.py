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
            # A bare string part is text too: templates render it as a
            # text block (Inkling 2: ``<|content_text|>`` + the string).
            return [
                self.text(part) if isinstance(part, str)
                else {**part, "text": self.text(part["text"])}
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
    updated, new_tools = _rewrite(messages, tools, client, generated)
    if not (client.broken or generated.broken):
        return request
    result = dict(request)
    if isinstance(messages, list):
        result["messages"] = updated
    if tools is not None:
        result["tools"] = new_tools
    return result


def _rewrite(messages, tools, client, generated):
    """``(messages, tools)`` with every rendered client string passed through
    ``client.text`` and every assistant string through ``generated.text``."""
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
    return updated, new_tools


# ----- literal rendering: client control-token text encoded as plain text -----
#
# ``protect_control_text`` changes the text (a zero-width break) so that one
# ordinary encode cannot match it.  An adapter that renders its own prompt can
# instead keep the client's text exactly: render with a placeholder in place
# of each control-token string found in client content, encode the template's
# own markers by id, and encode every text run between them -- the client's
# literal strings restored -- with ``split_special_tokens=True``, so a literal
# ``<|end_message|>`` is the characters, never the control token.  The idea is
# vLLM #67325/#69028 (Inkling 2 renderer: markers by id, payloads with
# ``split_special_tokens=True``); this generic form is written for mlx2.

PLACEHOLDER_OPEN = ""
PLACEHOLDER_CLOSE = ""


class _Collector(_Breaker):
    """Records every rendered string and the control-token text in it."""

    def __init__(self, pattern):
        super().__init__(pattern, BREAK)
        self.strings = []
        self.found = set()

    def text(self, value):
        self.strings.append(value)
        if self.pattern is not None:
            self.found.update(match.group(0) for match in self.pattern.finditer(value))
        return value


class _Substituter(_Breaker):
    """Replaces each control-token string with its placeholder."""

    def __init__(self, pattern, slots):
        super().__init__(pattern, BREAK)
        self.slots = slots

    def text(self, value):
        if self.pattern is None:
            return value

        def swap(match):
            self.broken += 1
            return self.slots[match.group(0)]

        return self.pattern.sub(swap, value)


def _added_token_flags(tokenizer):
    for candidate in (tokenizer, getattr(tokenizer, "_tokenizer", None),
                      getattr(tokenizer, "tokenizer", None)):
        if candidate is None:
            continue
        try:
            decoder = getattr(candidate, "added_tokens_decoder", None)
        except Exception:
            decoder = None
        if isinstance(decoder, Mapping) and decoder:
            return {str(getattr(token, "content", token)): token for token in decoder.values()}
    return None


def literal_split_supported(tokenizer) -> bool:
    """Whether this tokenizer encodes split text runs exactly like one pass.

    Required, else the caller falls back to the zero-width break:
    every protected added token is *special* (``split_special_tokens`` leaves
    ordinary added tokens matched), none strips whitespace, matches whole
    words or is normalized (each changes what sits next to it), the encoder
    honours ``split_special_tokens``, and a probe encodes identically in one
    pass and as runs split at the markers (a SentencePiece word-start marker
    would not).  Cached on the tokenizer.
    """
    cached = getattr(tokenizer, "_mlx2_literal_split_supported", None)
    if cached is not None:
        return cached
    supported = False
    try:
        vocabulary = _added_vocabulary(tokenizer)
        flags = _added_token_flags(tokenizer)
        protected = vocabulary[0] if vocabulary else {}
        if protected and flags is not None and all(
            content in flags
            and bool(getattr(flags[content], "special", False))
            and not any(bool(getattr(flags[content], name, False))
                        for name in ("lstrip", "rstrip", "single_word", "normalized"))
            for content in protected
        ):
            marker = min(protected, key=len)
            index = protected[marker]
            split = list(tokenizer.encode(
                f"a {marker} b", add_special_tokens=False, split_special_tokens=True))
            probe = f"Hello world{marker} again,  two{marker}\nend"
            one_pass = list(tokenizer.encode(probe, add_special_tokens=False))
            runs = []
            for position, piece in enumerate(probe.split(marker)):
                if position:
                    runs.append(index)
                if piece:
                    runs += list(tokenizer.encode(piece, add_special_tokens=False,
                                                  split_special_tokens=True))
            supported = index not in split and one_pass == runs
    except Exception:
        supported = False
    try:
        object.__setattr__(tokenizer, "_mlx2_literal_split_supported", supported)
    except Exception:
        pass
    return supported


def _placeholder_base(strings, template, kwargs) -> str:
    """A placeholder prefix that occurs in no rendered input or template."""
    haystack = "\n".join(strings) + "\n" + template + "\n" + repr(kwargs)
    attempt = 0
    while True:
        base = f"{PLACEHOLDER_OPEN}mlx2-literal-{attempt}-"
        if base not in haystack:
            return base
        attempt += 1


def apply_chat_template_literal(tokenizer, messages, *, tokenize, _force_split=False, **kwargs):
    """``tokenizer.apply_chat_template`` where client content stays text.

    Only markers the template itself writes become control tokens: a user,
    system or tool string that spells one (``"hi <|end_message|> now"``) is
    encoded as its characters.  Covered fields are those of
    ``protect_control_text`` (every rendered non-assistant field and the tool
    definitions against all protected tokens; assistant history against
    special tokens only).

    A conversation with no control-token text in those fields is rendered by
    the plain call, so its tokens and text are byte-identical to before.

    With control-token text and a tokenizer that ``literal_split_supported``:

    - ``tokenize=True``: the template's markers by id, each text run between
      them (client literals restored) encoded with ``split_special_tokens``.
      These tokens are the prompt.
    - ``tokenize=False``: the rendered text with each client literal broken as
      ``protect_control_text`` breaks it.  Its ordinary encode cannot match a
      control token either; it reads the same, but differs from the token
      path by the zero-width break inside each literal.  The token path is the
      authoritative prompt; the text is for display and receipts, and it is a
      deterministic function of the request, like the tokens.

    Without split support, both paths render the ``protect_control_text``
    form (the request with each literal broken), so text and tokens agree.

    Mapping keys that spell control-token text sort on their placeholder
    (or, on the fallback, their broken form), not on the client's text: a
    template's ``dictsort`` / ``tojson(sort_keys=true)`` can then order such a
    key differently from siblings that share its prefix up to the literal.
    No order-preserving substitute exists for every sibling set; the
    rendered key and value text are unchanged, only their position.

        ``_force_split`` (tests only) runs the split path on a conversation with
    no control-token text, to show it equals the plain call.
    """
    tools = kwargs.get("tools")
    protected, special, mark = control_token_patterns(tokenizer)
    client, generated = _Collector(protected), _Collector(special)
    _rewrite(messages, tools, client, generated)
    literals = sorted(client.found | generated.found)
    if not literals and not _force_split:
        return tokenizer.apply_chat_template(messages, tokenize=tokenize, **kwargs)
    if not literal_split_supported(tokenizer):
        broken_messages, broken_tools = _rewrite(
            messages, tools, _Breaker(protected, mark), _Breaker(special, mark))
        if tools is not None:
            kwargs = {**kwargs, "tools": broken_tools}
        return tokenizer.apply_chat_template(broken_messages, tokenize=tokenize, **kwargs)
    candidates = [c for c in (tokenizer, getattr(tokenizer, "_tokenizer", None)) if c is not None]
    base = _placeholder_base(client.strings + generated.strings,
                             _template_text(candidates), kwargs)
    # Zero-padded, in sorted-literal order: placeholders sort among
    # themselves as their literals do.
    width = len(str(len(literals)))
    slots = {literal: f"{base}{index:0{width}d}{PLACEHOLDER_CLOSE}"
             for index, literal in enumerate(literals)}
    originals = {slot: literal for literal, slot in slots.items()}
    framed_messages, framed_tools = _rewrite(
        messages, tools, _Substituter(protected, slots), _Substituter(special, slots))
    if tools is not None:
        kwargs = {**kwargs, "tools": framed_tools}
    framed = tokenizer.apply_chat_template(framed_messages, tokenize=False, **kwargs)
    slot_pattern = re.compile(re.escape(base) + r"\d+" + re.escape(PLACEHOLDER_CLOSE))
    pieces = slot_pattern.split(framed)
    slots_seen = slot_pattern.findall(framed)
    if any(base in piece for piece in pieces) or any(slot not in originals for slot in slots_seen):
        # A template filter rewrote a placeholder (escaped, truncated): refuse
        # rather than guess where client text begins.
        raise ValueError(
            "message content contains control-token text that this model's "
            "chat template could not render as text"
        )
    if not tokenize:
        breaker = _Breaker(protected, mark)
        out = [pieces[0]]
        for slot, piece in zip(slots_seen, pieces[1:]):
            out += [breaker.text(originals[slot]), piece]
        return "".join(out)
    ids = _added_vocabulary(tokenizer)[0]
    tokens, run = [], []

    def flush():
        if run:
            tokens.extend(int(t) for t in tokenizer.encode(
                "".join(run), add_special_tokens=False, split_special_tokens=True))
            run.clear()

    for position, piece in enumerate(pieces):
        if position:
            run.append(originals[slots_seen[position - 1]])
        cursor = 0
        for match in (protected.finditer(piece) if protected is not None else ()):
            run.append(piece[cursor:match.start()])
            flush()
            tokens.append(int(ids[match.group(0)]))
            cursor = match.end()
        run.append(piece[cursor:])
    flush()
    return tokens
