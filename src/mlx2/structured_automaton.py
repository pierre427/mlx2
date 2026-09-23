"""Exact token-level automaton for structured output.

The scanner in ``structured_output`` decides admissibility by partial-matching
``prefix + piece`` with the ``regex`` module for every candidate token.  This
module compiles the same pattern once into a deterministic automaton and
answers "which tokens keep the output inside the language" with one vectorized
walk of the automaton down a trie of the vocabulary, memoized per automaton
state.  The mask is exact and complete, so no tail bound is incurred.

Supported pattern subset (everything else raises ``AutomatonUnsupported`` and
the scanner stays in charge):

* literals, escaped punctuation, ``\\n \\t \\r \\f \\v \\a``, ``\\xHH``,
  ``\\uHHHH``, ``\\UHHHHHHHH``;
* ``.`` and character classes, including negation, ranges and the class escapes
  ``\\d \\w \\s \\D \\W \\S \\p{..} \\P{..}`` (their members are enumerated with
  the ``regex`` module itself, so the semantics are regex's by construction);
* groups (capturing, non-capturing and named are treated alike), alternation,
  ``? * + {m} {m,} {,n} {m,n}`` and their lazy forms (for full-match language
  membership lazy and greedy quantifiers are the same language);
* ``(?(DEFINE)...)`` and named group calls ``(?&name)``.  Non-recursive calls
  are inlined.  Recursive calls compile to a deterministic pushdown automaton
  and are accepted only when the recursion is LL(1)-deterministic: a called
  rule is never empty, ends only in states without continuations, and its
  first characters never overlap the alternatives at the call site.  The
  ``json_object`` grammar has exactly this shape.

Refused: lookaround, backreferences, anchors and boundaries, atomic groups and
possessive quantifiers (they change the language), inline or compile flags,
conditionals, numbered/whole-pattern recursion, fuzzy matching, nested sets and
POSIX classes, comments, unescaped ``{``/``}``/``]`` literals, and any pattern
whose automaton would exceed the state caps.
"""

from __future__ import annotations

import codecs
import itertools
import threading
from bisect import bisect_right
from collections import OrderedDict

import numpy as np
import regex

MAX_DFA_STATES = 20_000
MAX_NFA_STATES = 200_000
MAX_TABLE_ENTRIES = 8_000_000
MAX_RECURSIVE_RULES = 6
# Bound on the memoized token masks of one vocabulary (packed bits).
MASK_MEMO_BYTES = 64 << 20
_AUTOMATON_CACHE_ENTRIES = 64
_MAX_CODEPOINT = 0x10FFFF
_DEAD = -1
_SPECIAL = -2
_TRIE_ATTRIBUTE = "_mlx2_structured_token_trie_v1"
NO_CONTINUATION = "structured-output grammar has no valid token continuation"


class AutomatonUnsupported(ValueError):
    """The pattern is outside the exactly supported subset."""


# ---------------------------------------------------------------------------
# Character sets: sorted tuples of inclusive (lo, hi) code point intervals.


def _normalize(intervals):
    merged = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1] + 1:
            if hi > merged[-1][1]:
                merged[-1][1] = hi
        else:
            merged.append([lo, hi])
    return tuple((lo, hi) for lo, hi in merged)


def _negate(intervals):
    out = []
    cursor = 0
    for lo, hi in intervals:
        if lo > cursor:
            out.append((cursor, lo - 1))
        cursor = hi + 1
    if cursor <= _MAX_CODEPOINT:
        out.append((cursor, _MAX_CODEPOINT))
    return tuple(out)


_ALL_CHARS = None
_ENUMERATED = {}
_ENUMERATE_LOCK = threading.Lock()


def _enumerate_charset(source):
    """Members of a one-character pattern, decided by ``regex`` itself."""
    global _ALL_CHARS
    with _ENUMERATE_LOCK:
        cached = _ENUMERATED.get(source)
        if cached is not None:
            return cached
        if _ALL_CHARS is None:
            _ALL_CHARS = "".join(map(chr, range(_MAX_CODEPOINT + 1)))
        try:
            matched = regex.findall(source, _ALL_CHARS)
        except regex.error as exc:
            raise AutomatonUnsupported(f"character set {source!r}: {exc}") from exc
        if any(len(item) != 1 for item in matched[:1]) or (
            matched and not isinstance(matched[0], str)
        ):
            raise AutomatonUnsupported(f"character set {source!r} is not one character")
        points = np.frombuffer(
            "".join(matched).encode("utf-32-le", "surrogatepass"), dtype="<u4"
        ).astype(np.int64)
        if points.size != len(matched):
            raise AutomatonUnsupported(f"character set {source!r} is not one character")
        if points.size == 0:
            result = ()
        else:
            breaks = np.flatnonzero(np.diff(points) != 1)
            starts = np.concatenate(([points[0]], points[breaks + 1]))
            ends = np.concatenate((points[breaks], [points[-1]]))
            result = tuple((int(lo), int(hi)) for lo, hi in zip(starts, ends))
        _ENUMERATED[source] = result
        return result


# ---------------------------------------------------------------------------
# Parser: pattern source -> AST.
#   ("set", intervals) | ("cat", [nodes]) | ("alt", [nodes])
#   ("rep", node, minimum, maximum_or_None) | ("ref", name) | ("eps",)

_SIMPLE_ESCAPES = {"n": 10, "t": 9, "r": 13, "f": 12, "v": 11, "a": 7}
_CLASS_ESCAPES = "dDwWsS"
_HEX = "0123456789abcdefABCDEF"


class _Parser:
    def __init__(self, source):
        self.source = source
        self.index = 0
        self.rules = {}

    def fail(self, reason):
        raise AutomatonUnsupported(f"{reason} at offset {self.index}")

    def peek(self, offset=0):
        position = self.index + offset
        return self.source[position] if position < len(self.source) else ""

    def parse(self):
        node = self.alternation()
        if self.index != len(self.source):
            self.fail("unbalanced parenthesis")
        return node

    def alternation(self):
        branches = [self.sequence()]
        while self.peek() == "|":
            self.index += 1
            branches.append(self.sequence())
        return branches[0] if len(branches) == 1 else ("alt", branches)

    def sequence(self):
        items = []
        while self.index < len(self.source) and self.peek() not in "|)":
            items.append(self.quantified())
        if not items:
            return ("eps",)
        return items[0] if len(items) == 1 else ("cat", items)

    def quantified(self):
        atom = self.atom()
        char = self.peek()
        if char == "*":
            bounds = (0, None)
            self.index += 1
        elif char == "+":
            bounds = (1, None)
            self.index += 1
        elif char == "?":
            bounds = (0, 1)
            self.index += 1
        elif char == "{":
            bounds = self.counted()
        else:
            return atom
        follower = self.peek()
        if follower == "?":
            self.index += 1  # lazy: same language under full match
        elif follower == "+":
            self.fail("possessive quantifier")
        if self.peek() and self.peek() in "*+?{":
            self.fail("stacked quantifier")
        minimum, maximum = bounds
        if maximum is not None and maximum < minimum:
            self.fail("inverted repeat bounds")
        return ("rep", atom, minimum, maximum)

    def counted(self):
        match = regex.compile(r"\{([0-9]*)(?:(,)([0-9]*))?\}").match(self.source, self.index)
        if match is None or (not match.group(1) and not match.group(2)):
            self.fail("unsupported brace expression")
        low, comma, high = match.group(1), match.group(2), match.group(3)
        self.index = match.end()
        minimum = int(low) if low else 0
        if not comma:
            return (minimum, minimum)
        return (minimum, int(high) if high else None)

    def atom(self):
        char = self.peek()
        if char == "(":
            return self.group()
        if char == "[":
            return ("set", self.bracket())
        if char == ".":
            self.index += 1
            return ("set", _negate(((10, 10),)))
        if char == "\\":
            return ("set", self.escape(in_class=False))
        if char in "^$":
            self.fail("anchor")
        if char in "*+?":
            self.fail("quantifier without operand")
        if char in "{}]":
            self.fail("unescaped brace or bracket literal")
        self.index += 1
        return ("set", ((ord(char), ord(char)),))

    def group(self):
        self.index += 1  # (
        name = None
        if self.peek() == "?":
            rest = self.source[self.index + 1 :]
            if rest.startswith(":"):
                self.index += 2
            elif rest.startswith("&"):
                end = self.source.find(")", self.index)
                if end < 0:
                    self.fail("unterminated group call")
                target = self.source[self.index + 2 : end]
                if not target.isidentifier():
                    self.fail("unsupported group call")
                self.index = end + 1
                return ("ref", target)
            elif rest.startswith("(DEFINE)"):
                self.index += len("?(DEFINE)")
                self.alternation()  # registers the named rules; matches empty
                if self.peek() != ")":
                    self.fail("unterminated DEFINE group")
                self.index += 1
                return ("eps",)
            elif rest.startswith("P<") or (
                rest.startswith("<") and not rest.startswith(("<=", "<!"))
            ):
                start = self.index + (3 if rest.startswith("P<") else 2)
                end = self.source.find(">", start)
                if end < 0:
                    self.fail("unterminated group name")
                name = self.source[start:end]
                if not name.isidentifier() or name in self.rules:
                    self.fail("unsupported group name")
                self.index = end + 1
            else:
                self.fail("unsupported group construct")
        if name is not None:
            self.rules[name] = None  # reserve: duplicates are refused
        node = self.alternation()
        if self.peek() != ")":
            self.fail("unterminated group")
        self.index += 1
        if name is not None:
            self.rules[name] = node
        return node

    def escape(self, in_class):
        """One escape as intervals; ``self.index`` is on the backslash."""
        start = self.index
        char = self.peek(1)
        if not char:
            self.fail("dangling backslash")
        self.index += 2
        if char in _CLASS_ESCAPES:
            return _enumerate_charset("\\" + char)
        if char in "pP":
            if self.peek() != "{":
                self.fail("unsupported property escape")
            end = self.source.find("}", self.index)
            if end < 0:
                self.fail("unterminated property escape")
            self.index = end + 1
            return _enumerate_charset(self.source[start : self.index])
        if char in _SIMPLE_ESCAPES:
            point = _SIMPLE_ESCAPES[char]
        elif char in "xuU":
            width = {"x": 2, "u": 4, "U": 8}[char]
            digits = self.source[self.index : self.index + width]
            if len(digits) != width or any(d not in _HEX for d in digits):
                self.fail("malformed hex escape")
            self.index += width
            point = int(digits, 16)
            if point > _MAX_CODEPOINT:
                self.fail("hex escape out of range")
        elif char == "b" and in_class:
            point = 8
        elif char.isalnum() or char == "_":
            self.fail(f"unsupported escape \\{char}")
        else:
            point = ord(char)
        return ((point, point),)

    def bracket(self):
        self.index += 1  # [
        negate = self.peek() == "^"
        if negate:
            self.index += 1
        intervals = []
        first = True
        while True:
            char = self.peek()
            if not char:
                self.fail("unterminated character set")
            if char == "]" and not first:
                self.index += 1
                break
            if char == "[":
                self.fail("nested or POSIX character set")
            first = False
            low = self.set_item()
            if self.peek() == "-" and self.peek(1) not in ("]", ""):
                if len(low) != 1 or low[0][0] != low[0][1]:
                    self.fail("range from a character class")
                self.index += 1
                if self.peek() == "[":
                    self.fail("nested or POSIX character set")
                high = self.set_item()
                if len(high) != 1 or high[0][0] != high[0][1] or high[0][0] < low[0][0]:
                    self.fail("malformed range")
                intervals.append((low[0][0], high[0][0]))
            else:
                intervals.extend(low)
        result = _normalize(intervals)
        return _negate(result) if negate else result

    def set_item(self):
        if self.peek() == "\\":
            return self.escape(in_class=True)
        char = self.peek()
        if not char:
            self.fail("unterminated character set")
        self.index += 1
        return ((ord(char), ord(char)),)


# ---------------------------------------------------------------------------
# Thompson NFA over symbol sets, subset construction, minimization.


class _NFA:
    def __init__(self):
        self.eps = []
        self.edges = []  # per state: list of (symbols_tuple, target)

    def state(self):
        if len(self.eps) >= MAX_NFA_STATES:
            raise AutomatonUnsupported("pattern expands past the NFA state cap")
        self.eps.append([])
        self.edges.append([])
        return len(self.eps) - 1


def _build_fragment(nfa, node, symbols_of, call_symbol, rules, inlining):
    kind = node[0]
    if kind == "eps":
        start = nfa.state()
        return start, start
    if kind == "set":
        start, end = nfa.state(), nfa.state()
        symbols = symbols_of(node[1])
        if symbols:
            nfa.edges[start].append((symbols, end))
        return start, end
    if kind == "ref":
        name = node[1]
        if name in call_symbol:
            start, end = nfa.state(), nfa.state()
            nfa.edges[start].append(((call_symbol[name],), end))
            return start, end
        if name in inlining:
            raise AutomatonUnsupported("recursion outside the chosen call rules")
        body = rules.get(name)
        if body is None:
            raise AutomatonUnsupported(f"call to unknown group {name!r}")
        return _build_fragment(
            nfa, body, symbols_of, call_symbol, rules, inlining | {name}
        )
    if kind == "cat":
        start = end = None
        for child in node[1]:
            child_start, child_end = _build_fragment(
                nfa, child, symbols_of, call_symbol, rules, inlining
            )
            if start is None:
                start = child_start
            else:
                nfa.eps[end].append(child_start)
            end = child_end
        return start, end
    if kind == "alt":
        start, end = nfa.state(), nfa.state()
        for child in node[1]:
            child_start, child_end = _build_fragment(
                nfa, child, symbols_of, call_symbol, rules, inlining
            )
            nfa.eps[start].append(child_start)
            nfa.eps[child_end].append(end)
        return start, end
    if kind == "rep":
        _, child, minimum, maximum = node
        start = nfa.state()
        end = start
        for _ in range(minimum):
            child_start, child_end = _build_fragment(
                nfa, child, symbols_of, call_symbol, rules, inlining
            )
            nfa.eps[end].append(child_start)
            end = child_end
        if maximum is None:
            loop_start, loop_end = _build_fragment(
                nfa, child, symbols_of, call_symbol, rules, inlining
            )
            hub = nfa.state()
            nfa.eps[end].append(hub)
            nfa.eps[hub].append(loop_start)
            nfa.eps[loop_end].append(hub)
            end = hub
        else:
            final = nfa.state()
            for _ in range(maximum - minimum):
                child_start, child_end = _build_fragment(
                    nfa, child, symbols_of, call_symbol, rules, inlining
                )
                nfa.eps[end].append(final)
                nfa.eps[end].append(child_start)
                end = child_end
            nfa.eps[end].append(final)
            end = final
        return start, end
    raise AutomatonUnsupported(f"unknown node {kind}")


def _determinize(nfa, start, accept, symbol_count, state_budget):
    closures = {}

    def closure(state):
        cached = closures.get(state)
        if cached is None:
            seen = {state}
            stack = [state]
            while stack:
                for target in nfa.eps[stack.pop()]:
                    if target not in seen:
                        seen.add(target)
                        stack.append(target)
            cached = closures[state] = frozenset(seen)
        return cached

    first = closure(start)
    index = {first: 0}
    order = [first]
    rows = []
    cursor = 0
    while cursor < len(order):
        current = order[cursor]
        cursor += 1
        by_edge = {}
        for state in current:
            for symbols, target in nfa.edges[state]:
                by_edge.setdefault(symbols, set()).add(target)
        by_symbol = {}
        for symbols, targets in by_edge.items():
            for symbol in symbols:
                existing = by_symbol.get(symbol)
                if existing is None:
                    by_symbol[symbol] = set(targets)
                else:
                    existing.update(targets)
        row = [_DEAD] * symbol_count
        merged = {}
        for symbol, targets in by_symbol.items():
            key = frozenset(targets)
            target_index = merged.get(key)
            if target_index is None:
                closed = frozenset().union(*(closure(t) for t in key))
                target_index = index.get(closed)
                if target_index is None:
                    if len(order) >= state_budget:
                        raise AutomatonUnsupported(
                            f"pattern expands past the {MAX_DFA_STATES}-state DFA cap"
                        )
                    target_index = index[closed] = len(order)
                    order.append(closed)
                merged[key] = target_index
            row[symbol] = target_index
        rows.append(row)
    table = np.array(rows, dtype=np.int32).reshape(len(rows), symbol_count)
    accepting = np.array([accept in states for states in order], dtype=bool)
    return table, accepting


def _refinable_partition(order, sizes):
    """Partition of ``range(len(order))`` whose sets can be split in place.

    ``order`` lists the elements set by set and ``sizes`` gives each set's
    length.  Returns the lists ``(elements, location, set_of, first, past,
    marked)``: a set's members are ``elements[first[s]:past[s]]`` and the
    marked ones sit at the front of that slice.
    """
    size = len(order)
    sizes = np.asarray(sizes, dtype=np.int64)
    location = np.empty(size, dtype=np.int64)
    location[order] = np.arange(size)
    set_of = np.empty(size, dtype=np.int64)
    set_of[order] = np.repeat(np.arange(len(sizes)), sizes)
    past = np.cumsum(sizes)
    first = past - sizes
    return (
        order.tolist(),
        location.tolist(),
        set_of.tolist(),
        first.tolist(),
        past.tolist(),
        [0] * len(sizes),
    )


def _split_marked(elements, set_of, first, past, marked, touched):
    """Split every touched set into its marked and unmarked members.

    The smaller part becomes the new set, so a set that was already used as a
    splitter only has to be followed by that smaller part (Hopcroft's
    "process the smaller half").
    """
    for split in touched:
        middle = first[split] + marked[split]
        marked[split] = 0
        if middle == past[split]:
            continue  # every member was marked: nothing to separate
        new = len(first)
        if middle - first[split] <= past[split] - middle:
            first.append(first[split])
            past.append(middle)
            first[split] = middle
        else:
            first.append(middle)
            past.append(past[split])
            past[split] = middle
        marked.append(0)
        for position in range(first[new], past[new]):
            set_of[elements[position]] = new
    touched.clear()


def _minimize(table, accepting):
    """Coarsest stable partition of the states; state 0 stays the start state.

    Two states merge when they agree on acceptance and every symbol takes both
    to merged states or both to the dead state.  This is Hopcroft's algorithm
    in the form Valmari and Lehtinen give for partial transition functions
    ("Efficient Minimization of DFAs with Partial Transition Functions",
    STACS 2008), O(m log n) over the m defined transitions.  Moore refinement
    reached the same partition one distinguishing step per round, sorting the
    whole table each round, and a string ``maxLength`` of n is n steps deep:
    it spent 8 s of a request's compile at n=4096.
    """
    count, symbol_count = table.shape
    sources, symbols = np.nonzero(table >= 0)
    targets = table[sources, symbols]
    # States start as the rejecting and the accepting block.  Block 0 is the
    # one block never used as a splitter (its effect follows from the others),
    # so it is the larger.
    accept_states = np.flatnonzero(accepting)
    reject_states = np.flatnonzero(~accepting)
    larger, smaller = sorted((reject_states, accept_states), key=len, reverse=True)
    sizes = [size for size in (len(larger), len(smaller)) if size]
    states = _refinable_partition(np.concatenate((larger, smaller)), sizes)
    block_members, block_location, block_of, block_first, block_past, block_marked = states
    # Transitions are grouped into "cords" that share a symbol and a target
    # block.  They start as one cord per symbol; using those as splitters
    # separates states that define different symbols, which is what keeps a
    # transition to the dead state distinct from one to a real state.
    by_symbol = np.argsort(symbols, kind="stable")
    cords = _refinable_partition(
        by_symbol, [size for size in np.bincount(symbols, minlength=symbol_count) if size]
    )
    cord_members, cord_location, cord_of, cord_first, cord_past, cord_marked = cords
    incoming = np.argsort(targets, kind="stable").tolist()
    incoming_start = np.concatenate(
        ([0], np.cumsum(np.bincount(targets, minlength=count)))
    ).tolist()
    tail = sources.tolist()
    touched_blocks, touched_cords = [], []
    block_cursor, cord_cursor = 1, 0
    while cord_cursor < len(cord_first):
        # Split the blocks by which of their states have a transition in this
        # cord.  A cord holds one symbol, so it names each source at most once.
        for position in range(cord_first[cord_cursor], cord_past[cord_cursor]):
            state = tail[cord_members[position]]
            block = block_of[state]
            front = block_first[block] + block_marked[block]
            at = block_location[state]
            block_members[at] = displaced = block_members[front]
            block_location[displaced] = at
            block_members[front] = state
            block_location[state] = front
            if not block_marked[block]:
                touched_blocks.append(block)
            block_marked[block] += 1
        _split_marked(
            block_members, block_of, block_first, block_past, block_marked, touched_blocks
        )
        cord_cursor += 1
        # Split the cords by which of their transitions enter each block not
        # yet used as a splitter.
        while block_cursor < len(block_first):
            for member in range(block_first[block_cursor], block_past[block_cursor]):
                state = block_members[member]
                for edge in range(incoming_start[state], incoming_start[state + 1]):
                    transition = incoming[edge]
                    cord = cord_of[transition]
                    front = cord_first[cord] + cord_marked[cord]
                    at = cord_location[transition]
                    cord_members[at] = displaced = cord_members[front]
                    cord_location[displaced] = at
                    cord_members[front] = transition
                    cord_location[transition] = front
                    if not cord_marked[cord]:
                        touched_cords.append(cord)
                    cord_marked[cord] += 1
            _split_marked(
                cord_members, cord_of, cord_first, cord_past, cord_marked, touched_cords
            )
            block_cursor += 1
    block = np.array(block_of, dtype=np.int64)
    blocks = len(block_first)
    # Renumber blocks in order of first appearance so the start stays 0.
    _, first_index = np.unique(block, return_index=True)
    rank = np.empty(blocks, dtype=np.int64)
    rank[np.argsort(first_index)] = np.arange(blocks)
    block = rank[block]
    representative = np.empty(blocks, dtype=np.int64)
    representative[block[::-1]] = np.arange(count)[::-1]
    reduced = table[representative]
    remap = np.concatenate((block, [_DEAD]))
    reduced = remap[reduced].astype(np.int32)
    return reduced, accepting[representative]


# ---------------------------------------------------------------------------
# The compiled automaton.


class TokenAutomaton:
    """Deterministic (pushdown) automaton over code point classes.

    A configuration is ``(state, stack)`` where ``stack`` is a tuple of return
    states.  After trimming, every reachable configuration is live: an
    accepting configuration can be reached from it.
    """

    _uids = itertools.count(1)

    def __init__(self, bounds, class_of_segment, class_count, table, calls, first,
                 rule_start, final, accepting, start):
        self.uid = next(TokenAutomaton._uids)
        self.bounds = bounds  # sorted segment starts (python list, for bisect)
        self.bounds_array = np.array(bounds, dtype=np.int64)
        self.segment_class = class_of_segment  # python list
        self.segment_class_array = np.array(class_of_segment, dtype=np.int32)
        self.class_count = class_count
        self.table = table  # numpy [states, classes]
        self.rows = table.tolist()
        self.calls = calls  # state -> tuple of (rule, return_state)
        self.first = first  # rule -> frozenset of classes
        self.rule_start = rule_start
        self.final = final  # python list of bool: pop on arrival
        self.accepting = accepting  # python list of bool (root accept states)
        self.start = (start, ())
        self.state_count = table.shape[0]
        self.recursive = any(calls)
        walk = table.copy()
        if self.recursive:
            final_array = np.array(final, dtype=bool)
            targets = np.where(walk >= 0, walk, 0)
            walk[(walk >= 0) & final_array[targets]] = _SPECIAL
            for state, state_calls in enumerate(calls):
                for rule, _ in state_calls:
                    for symbol in first[rule]:
                        walk[state, symbol] = _SPECIAL
        self.walk_table = walk

    def class_of(self, codepoint):
        return self.segment_class[bisect_right(self.bounds, codepoint) - 1]

    def step(self, config, symbol):
        state, stack = config
        target = self.rows[state][symbol]
        if target < 0:
            for rule, return_state in self.calls[state]:
                if symbol in self.first[rule]:
                    return self.step((self.rule_start[rule], stack + (return_state,)), symbol)
            return None
        while self.final[target]:
            target = stack[-1]
            stack = stack[:-1]
        return (target, stack)

    def advance(self, config, text):
        """Configuration after reading ``text``, or None when it leaves the language."""
        for char in text:
            if config is None:
                return None
            config = self.step(config, self.class_of(ord(char)))
        return config

    def is_accepting(self, config):
        return config is not None and not config[1] and self.accepting[config[0]]

    def fullmatch(self, text, partial=False):
        config = self.advance(self.start, text)
        if config is None:
            return False
        return True if partial else self.is_accepting(config)


def _compile_rule(body, symbols_of, call_symbol, rules, symbol_count, budget):
    nfa = _NFA()
    start, end = _build_fragment(nfa, body, symbols_of, call_symbol, rules, frozenset())
    table, accepting = _determinize(nfa, start, end, symbol_count, budget)
    return _minimize(table, accepting)


def _references(node, out):
    kind = node[0]
    if kind == "ref":
        out.add(node[1])
    elif kind in ("cat", "alt"):
        for child in node[1]:
            _references(child, out)
    elif kind == "rep":
        _references(node[1], out)
    return out


def _collect_sets(node, out):
    kind = node[0]
    if kind == "set":
        out.add(node[1])
    elif kind in ("cat", "alt"):
        for child in node[1]:
            _collect_sets(child, out)
    elif kind == "rep":
        _collect_sets(node[1], out)


def _cyclic_rules(root, rules):
    graph = {name: _references(body, set()) for name, body in rules.items()}
    for name, targets in graph.items():
        for target in targets:
            if target not in rules:
                raise AutomatonUnsupported(f"call to unknown group {target!r}")
    for target in _references(root, set()):
        if target not in rules:
            raise AutomatonUnsupported(f"call to unknown group {target!r}")

    def reaches(origin):
        seen = set()
        stack = list(graph[origin])
        while stack:
            name = stack.pop()
            if name not in seen:
                seen.add(name)
                stack.extend(graph[name])
        return seen

    return sorted(name for name in rules if name in reaches(name)), graph


def _breaks_cycles(chosen, graph):
    remaining = {n: {t for t in ts if t not in chosen} for n, ts in graph.items() if n not in chosen}
    state = {}

    def visit(name):
        if state.get(name) == 1:
            return False
        if state.get(name) == 2:
            return True
        state[name] = 1
        for target in remaining.get(name, ()):
            if not visit(target):
                return False
        state[name] = 2
        return True

    return all(visit(name) for name in remaining)


def compile_pattern(source):
    """Compile a ``regex`` pattern source into a ``TokenAutomaton``."""
    if not isinstance(source, str):
        raise AutomatonUnsupported("pattern source is not text")
    parser = _Parser(source)
    root = parser.parse()
    rules = parser.rules
    if any(body is None for body in rules.values()):
        raise AutomatonUnsupported("unterminated named group")
    cyclic, graph = _cyclic_rules(root, rules)
    if len(cyclic) > MAX_RECURSIVE_RULES:
        raise AutomatonUnsupported("too many mutually recursive rules")

    # Alphabet: partition the code points by membership signature.
    charsets = set()
    _collect_sets(root, charsets)
    for body in rules.values():
        _collect_sets(body, charsets)
    charsets = sorted(charsets)
    cuts = {0}
    for intervals in charsets:
        for lo, hi in intervals:
            cuts.add(lo)
            if hi + 1 <= _MAX_CODEPOINT:
                cuts.add(hi + 1)
    bounds = sorted(cuts)
    bounds_array = np.array(bounds, dtype=np.int64)
    membership = np.zeros((len(charsets), len(bounds)), dtype=bool)
    for row, intervals in enumerate(charsets):
        for lo, hi in intervals:
            first = int(np.searchsorted(bounds_array, lo, side="left"))
            last = int(np.searchsorted(bounds_array, hi, side="right"))
            membership[row, first:last] = True
    if len(charsets):
        _, segment_class = np.unique(membership.T, axis=0, return_inverse=True)
        segment_class = segment_class.reshape(-1)
    else:
        segment_class = np.zeros(len(bounds), dtype=np.int64)
    class_count = int(segment_class.max()) + 1 if len(bounds) else 1
    set_symbols = {}
    for row, intervals in enumerate(charsets):
        set_symbols[intervals] = tuple(
            sorted(set(segment_class[membership[row]].tolist()))
        )

    def symbols_of(intervals):
        return set_symbols[intervals]

    last_error = None
    candidates = [()] if not cyclic else [
        combo
        for size in range(1, len(cyclic) + 1)
        for combo in itertools.combinations(cyclic, size)
        if _breaks_cycles(set(combo), graph)
    ]
    for chosen in candidates:
        try:
            return _assemble(
                root, rules, chosen, symbols_of, class_count, bounds, segment_class
            )
        except AutomatonUnsupported as exc:
            last_error = exc
    raise last_error or AutomatonUnsupported("recursion cannot be made deterministic")


def _assemble(root, rules, chosen, symbols_of, class_count, bounds, segment_class):
    call_symbol = {name: class_count + index for index, name in enumerate(chosen)}
    symbol_count = class_count + len(chosen)
    compiled = []
    budget = MAX_DFA_STATES
    for body in [root] + [rules[name] for name in chosen]:
        table, accepting = _compile_rule(
            body, symbols_of, call_symbol, rules, symbol_count, budget
        )
        budget -= table.shape[0]
        if budget <= 0:
            raise AutomatonUnsupported(
                f"pattern expands past the {MAX_DFA_STATES}-state DFA cap"
            )
        compiled.append((table, accepting))
    offsets = np.cumsum([0] + [table.shape[0] for table, _ in compiled])
    total = int(offsets[-1])
    if total * class_count > MAX_TABLE_ENTRIES:
        raise AutomatonUnsupported("transition table exceeds its size cap")
    table = np.full((total, class_count), _DEAD, dtype=np.int32)
    accepting = np.zeros(total, dtype=bool)
    final = np.zeros(total, dtype=bool)
    calls = [[] for _ in range(total)]
    rule_start = {}
    for rule, (rule_table, rule_accept) in enumerate(compiled):
        base = int(offsets[rule])
        chars = rule_table[:, :class_count]
        table[base : base + rule_table.shape[0]] = np.where(chars >= 0, chars + base, _DEAD)
        if rule == 0:
            accepting[base : base + rule_table.shape[0]] = rule_accept
        else:
            rule_start[rule] = base
            if rule_accept[0]:
                raise AutomatonUnsupported("a recursive rule may not match the empty string")
            if (rule_table[rule_accept] >= 0).any():
                raise AutomatonUnsupported(
                    "a recursive rule must end in states without continuations"
                )
            final[base : base + rule_table.shape[0]] = rule_accept
        for state, symbol in zip(*np.nonzero(rule_table[:, class_count:] >= 0)):
            target = int(rule_table[state, class_count + symbol]) + base
            calls[base + int(state)].append((int(symbol) + 1, target))

    # First sets of the called rules (left recursion is refused).
    first = {}

    def first_of(rule, active):
        if rule in first:
            return first[rule]
        if rule in active:
            raise AutomatonUnsupported("left-recursive rule")
        start = rule_start[rule]
        symbols = set(np.flatnonzero(table[start] >= 0).tolist())
        for callee, _ in calls[start]:
            nested = first_of(callee, active | {rule})
            if symbols & nested:
                raise AutomatonUnsupported("ambiguous rule call")
            symbols |= nested
        first[rule] = frozenset(symbols)
        return first[rule]

    for rule in rule_start:
        first_of(rule, frozenset())

    # Co-reachability with productive rules, then trim.
    reverse = {}
    sources, symbols = np.nonzero(table >= 0)
    for source, target in set(zip(sources.tolist(), table[sources, symbols].tolist())):
        reverse.setdefault(target, []).append(source)
    live = accepting | final
    productive = set()
    while True:
        work = list(np.flatnonzero(live))
        call_sites = {}
        for state, state_calls in enumerate(calls):
            for rule, target in state_calls:
                if rule in productive:
                    call_sites.setdefault(target, []).append(state)
        while work:
            state = work.pop()
            for source in reverse.get(state, ()):
                if not live[source]:
                    live[source] = True
                    work.append(source)
            for source in call_sites.get(state, ()):
                if not live[source]:
                    live[source] = True
                    work.append(source)
        grown = {rule for rule, start in rule_start.items() if live[start]} - productive
        if not grown:
            break
        productive |= grown
    if not live[0]:
        raise AutomatonUnsupported("the pattern matches nothing")
    live_extended = np.concatenate((live, [False]))
    table[~live_extended[table]] = _DEAD
    table[~live] = _DEAD
    calls = [
        tuple(
            (rule, target)
            for rule, target in state_calls
            if live[state] and rule in productive and live[target]
        )
        for state, state_calls in enumerate(calls)
    ]
    # Determinism at every call site.
    for state, state_calls in enumerate(calls):
        if not state_calls:
            continue
        taken = set(np.flatnonzero(table[state] >= 0).tolist())
        for rule, _ in state_calls:
            if taken & first[rule]:
                raise AutomatonUnsupported("rule call overlaps its alternatives")
            taken |= first[rule]
    return TokenAutomaton(
        bounds,
        segment_class.tolist(),
        class_count,
        table,
        calls,
        first,
        rule_start,
        final.tolist(),
        accepting.tolist(),
        0,
    )


_AUTOMATA = OrderedDict()
_AUTOMATA_LOCK = threading.Lock()
_DEFAULT_FLAGS = regex.compile("x").flags
# Sources being compiled now, so concurrent callers share one compile.
_COMPILING = {}


class _PendingCompile:
    __slots__ = ("done", "result")

    def __init__(self):
        self.done = threading.Event()
        self.result = None


def _compile_or_refusal(source):
    try:
        return compile_pattern(source)
    except AutomatonUnsupported as exc:
        return exc
    except RecursionError:
        return AutomatonUnsupported("pattern nests too deeply")


def automaton_for(pattern, prepared=None):
    """Cached automaton for a compiled ``regex`` pattern (raises when unsupported).

    ``prepared`` maps pattern sources to what an earlier call returned or
    raised for them.  An entry there is used as is: it was compiled before
    the caller's hot path, and the shared cache may have evicted it since.
    """
    source = getattr(pattern, "pattern", None)
    flags = getattr(pattern, "flags", None)
    if not isinstance(source, str) or flags != _DEFAULT_FLAGS:
        raise AutomatonUnsupported("pattern flags are not the defaults")
    cached = (prepared or {}).get(source)
    while cached is None:
        with _AUTOMATA_LOCK:
            cached = _AUTOMATA.get(source)
            if cached is not None:
                _AUTOMATA.move_to_end(source)
                break
            pending = _COMPILING.get(source)
            owner = pending is None
            if owner:
                pending = _COMPILING[source] = _PendingCompile()
        if owner:
            try:
                pending.result = _compile_or_refusal(source)
            finally:
                with _AUTOMATA_LOCK:
                    del _COMPILING[source]
                    if pending.result is not None:
                        _AUTOMATA[source] = pending.result
                        while len(_AUTOMATA) > _AUTOMATON_CACHE_ENTRIES:
                            _AUTOMATA.popitem(last=False)
                pending.done.set()
        else:
            # Requests are prepared on their own threads, so a burst sharing
            # one new schema would otherwise compile it once per request, all
            # contending for the interpreter lock with the generation worker.
            pending.done.wait()
        # None only when the compiling thread died unexpectedly: try again.
        cached = pending.result
    if isinstance(cached, Exception):
        raise AutomatonUnsupported(str(cached))
    return cached


# ---------------------------------------------------------------------------
# Vocabulary trie and the memoized mask walk.


class TokenTrie:
    """Level-ordered trie of the usable vocabulary pieces.

    Usable means what the scanner index means: not an EOS id, nonempty, and
    free of U+FFFD (byte-fallback and partial UTF-8 pieces are never admitted).
    """

    def __init__(self, pieces, eos_ids):
        self.vocab_size = len(pieces)
        order = sorted(
            (
                token
                for token, piece in enumerate(pieces)
                if token not in eos_ids and piece and "�" not in piece
            ),
            key=lambda token: pieces[token],
        )
        parents = [-1]
        chars = [0]
        depths = [0]
        token_nodes = []
        path = [0]  # node ids along the previous piece
        previous = ""
        for token in order:
            piece = pieces[token]
            common = 0
            limit = min(len(previous), len(piece))
            while common < limit and previous[common] == piece[common]:
                common += 1
            del path[common + 1 :]
            for depth in range(common, len(piece)):
                parents.append(path[-1])
                chars.append(ord(piece[depth]))
                depths.append(depth + 1)
                path.append(len(parents) - 1)
            token_nodes.append(path[-1])
            previous = piece
        depths = np.array(depths, dtype=np.int32)
        level_order = np.argsort(depths, kind="stable")
        position = np.empty(len(level_order), dtype=np.int64)
        position[level_order] = np.arange(len(level_order))
        parents = np.array(parents, dtype=np.int64)[level_order]
        parents[1:] = position[parents[1:]]
        self.parents = parents
        chars = np.array(chars, dtype=np.int64)[level_order]
        self.unique_chars, self.char_index = np.unique(chars, return_inverse=True)
        self.char_index = self.char_index.reshape(-1)
        counts = np.bincount(depths)
        self.level_offsets = np.concatenate(([0], np.cumsum(counts))).tolist()
        self.tokens = np.array(order, dtype=np.int64)
        self.token_nodes = position[np.array(token_nodes, dtype=np.int64)] if order else np.zeros(0, dtype=np.int64)
        self.node_count = len(parents)
        self._classes = OrderedDict()
        self.memo = OrderedDict()
        self.memo_bytes = 0
        self._memo_depths = {}
        self.lock = threading.Lock()
        self.walks = 0
        self.hits = 0

    def _node_classes(self, automaton):
        cached = self._classes.get(automaton.uid)
        if cached is None:
            segments = np.searchsorted(automaton.bounds_array, self.unique_chars, side="right") - 1
            cached = automaton.segment_class_array[segments][self.char_index]
            self._classes[automaton.uid] = cached
            while len(self._classes) > 8:
                self._classes.popitem(last=False)
        else:
            self._classes.move_to_end(automaton.uid)
        return cached

    def mask(self, automaton, config):
        """Packed admissible non-EOS token bits for ``config`` (memoized).

        EOS is the caller's business: it is admissible iff the configuration
        is accepting, and EOS ids may lie outside the tokenizer vocabulary.
        """
        state, stack = config
        with self.lock:
            for depth in self._memo_depths.get((automaton.uid, state), ()):
                if depth > len(stack):
                    continue
                key = (automaton.uid, state, stack[len(stack) - depth :] if depth else ())
                packed = self.memo.get(key)
                if packed is not None:
                    self.memo.move_to_end(key)
                    self.hits += 1
                    return packed
            allowed, depth = self._walk(automaton, config)
            packed = np.packbits(allowed)
            key = (automaton.uid, state, stack[len(stack) - depth :] if depth else ())
            self.memo[key] = packed
            self.memo_bytes += packed.nbytes
            self._memo_depths.setdefault((automaton.uid, state), set()).add(depth)
            while self.memo_bytes > MASK_MEMO_BYTES and len(self.memo) > 1:
                _, dropped = self.memo.popitem(last=False)
                self.memo_bytes -= dropped.nbytes
            return packed

    def allowed(self, automaton, config):
        """Boolean admissibility per token id (length ``vocab_size``)."""
        packed = self.mask(automaton, config)
        return np.unpackbits(packed, count=self.vocab_size).astype(bool)

    def _walk(self, automaton, config):
        self.walks += 1
        classes = self._node_classes(automaton)
        state_count = automaton.state_count
        walk_table = automaton.walk_table
        start_state, stack = config
        states = np.full(self.node_count, _DEAD, dtype=np.int64)
        states[0] = start_state
        # Configurations that pushed or popped inside one piece get ids past
        # the plain states: (state, frames popped from ``stack``, pushed).
        extra = []
        extra_index = {}
        resolved = {}
        deepest = 0

        def intern(state, popped, pushed):
            if not popped and not pushed:
                return state
            key = (state, popped, pushed)
            identifier = extra_index.get(key)
            if identifier is None:
                identifier = extra_index[key] = state_count + len(extra)
                extra.append(key)
            return identifier

        def resolve(identifier, symbol):
            nonlocal deepest
            if identifier < state_count:
                state, popped, pushed = identifier, 0, ()
            else:
                state, popped, pushed = extra[identifier - state_count]
            while True:
                target = automaton.rows[state][symbol]
                if target >= 0:
                    break
                for rule, return_state in automaton.calls[state]:
                    if symbol in automaton.first[rule]:
                        pushed = pushed + (return_state,)
                        state = automaton.rule_start[rule]
                        break
                else:
                    return _DEAD
            while automaton.final[target]:
                if pushed:
                    target = pushed[-1]
                    pushed = pushed[:-1]
                else:
                    target = stack[len(stack) - 1 - popped]
                    popped += 1
                    deepest = max(deepest, popped)
            return intern(target, popped, pushed)

        offsets = self.level_offsets
        for level in range(1, len(offsets) - 1):
            low, high = offsets[level], offsets[level + 1]
            parent_states = states[self.parents[low:high]]
            alive = np.flatnonzero(parent_states != _DEAD)
            if alive.size == 0:
                break
            sources = parent_states[alive]
            symbols = classes[low:high][alive]
            plain = sources < state_count
            if plain.all():
                targets = walk_table[sources, symbols].astype(np.int64)
            else:
                targets = np.full(alive.size, _SPECIAL, dtype=np.int64)
                targets[plain] = walk_table[sources[plain], symbols[plain]]
            special = np.flatnonzero(targets == _SPECIAL)
            if special.size:
                pairs = sources[special] * automaton.class_count + symbols[special]
                unique_pairs, inverse = np.unique(pairs, return_inverse=True)
                values = np.empty(unique_pairs.size, dtype=np.int64)
                for slot, pair in enumerate(unique_pairs.tolist()):
                    value = resolved.get(pair)
                    if value is None:
                        value = resolved[pair] = resolve(
                            pair // automaton.class_count, pair % automaton.class_count
                        )
                    values[slot] = value
                targets[special] = values[inverse.reshape(-1)]
            states[low + alive] = targets
        allowed = np.zeros(self.vocab_size, dtype=bool)
        if self.tokens.size:
            allowed[self.tokens[states[self.token_nodes] != _DEAD]] = True
        return allowed, deepest


def trie_for(tokenizer, pieces, eos_ids):
    """One trie per tokenizer vocabulary, cached on the tokenizer."""
    trie = getattr(tokenizer, _TRIE_ATTRIBUTE, None)
    if not isinstance(trie, TokenTrie) or trie.vocab_size != len(pieces):
        trie = TokenTrie(pieces, frozenset(eos_ids))
        try:
            setattr(tokenizer, _TRIE_ATTRIBUTE, trie)
        except (AttributeError, TypeError):
            pass
    return trie


# ---------------------------------------------------------------------------
# Byte-fallback pieces: tokens that are only part of a UTF-8 character.
#
# The trie above works on decoded text, so it can never admit a piece whose
# isolated decode is U+FFFD.  A character the vocabulary only reaches through
# such pieces (most of the astral planes on a byte-level BPE) would dead-end.
# The automaton stays a codepoint automaton; the incomplete UTF-8 tail is
# carried beside its configuration and these pieces are walked as bytes.

_BYTES_ATTRIBUTE = "_mlx2_structured_token_bytes"
_ADDED_VOCAB_ATTRIBUTE = "_mlx2_added_vocab_ids_v1"
_FRAGMENT_MEMO_ENTRIES = 4096


def _byte_level_alphabet():
    """GPT-2 printable-unicode alphabet for bytes (char -> byte)."""
    kept = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars = kept[:]
    extra = 0
    for value in range(256):
        if value not in kept:
            kept.append(value)
            chars.append(256 + extra)
            extra += 1
    return {chr(char): value for value, char in zip(kept, chars)}


def token_bytes_for(tokenizer, pieces):
    """Exact bytes per token id for a byte-level BPE vocabulary, else ``None``.

    ``None`` entries contribute no text (special tokens, skipped by the decode
    the tracker mirrors).  Any disagreement with the decoded pieces disables
    the byte path for the tokenizer: fragments then stay inadmissible.
    """
    cached = getattr(tokenizer, _BYTES_ATTRIBUTE, None)
    if isinstance(cached, tuple) and cached[0] == len(pieces):
        return cached[1]
    table = None
    try:
        names = tokenizer.convert_ids_to_tokens(list(range(len(pieces))))
        special = {int(item) for item in getattr(tokenizer, "all_special_ids", ())}
        added = {int(item) for item in tokenizer.get_added_vocab().values()}
        alphabet = _byte_level_alphabet()
        table = [None] * len(pieces)
        for token, name in enumerate(names):
            if token in special or name is None:
                continue
            if token in added:
                table[token] = pieces[token].encode("utf-8")
                continue
            raw = bytes(alphabet[char] for char in name)
            table[token] = raw
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if text != pieces[token]:
                raise ValueError("token bytes disagree with the decoded piece")
    except Exception:  # noqa: BLE001 - not a byte-level vocabulary
        table = None
    try:
        setattr(tokenizer, _BYTES_ATTRIBUTE, (len(pieces), table))
    except (AttributeError, TypeError):
        pass
    return table


def token_bytes_value(tokenizer, token_id: int) -> bytes:
    """Return one token's exact byte spelling using the structured byte logic.

    This is the small, per-token counterpart to :func:`token_bytes_for` used by
    API logprob serialization.  It consults an already-built full vocabulary
    table when available, understands both GPT-2 byte alphabets and explicit
    ``<0xNN>`` fallback pieces, and otherwise uses the token's decoded UTF-8.
    """
    token_id = int(token_id)
    cached = getattr(tokenizer, _BYTES_ATTRIBUTE, None)
    if isinstance(cached, tuple) and len(cached) == 2 and cached[1] is not None:
        table = cached[1]
        if 0 <= token_id < len(table) and table[token_id] is not None:
            return table[token_id]
    piece = tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    names = tokenizer.convert_ids_to_tokens([token_id])
    name = names[0] if names else None
    if isinstance(name, str) and len(name) == 6 and name.startswith("<0x") and name.endswith(">"):
        try:
            return bytes((int(name[3:5], 16),))
        except ValueError:
            pass
    special = {int(item) for item in getattr(tokenizer, "all_special_ids", ())}
    added = getattr(tokenizer, _ADDED_VOCAB_ATTRIBUTE, None)
    if added is None:
        added_vocab = getattr(tokenizer, "get_added_vocab", lambda: {})()
        added = frozenset(int(item) for item in added_vocab.values())
        try:
            setattr(tokenizer, _ADDED_VOCAB_ATTRIBUTE, added)
        except (AttributeError, TypeError):
            pass
    if token_id not in special and token_id not in added and isinstance(name, str):
        # SentencePiece spells word-boundary spaces as U+2581 in its piece
        # vocabulary. A one-token decode at the start of an isolated fragment
        # can drop that leading space, so the vocabulary piece is the exact
        # byte authority when this marker is present.
        if "▁" in name:
            return name.replace("▁", " ").encode("utf-8")
        try:
            raw = bytes(_byte_level_alphabet()[char] for char in name)
        except (KeyError, ValueError):
            pass
        else:
            try:
                decoded = raw.decode("utf-8")
            except UnicodeDecodeError:
                return raw
            if decoded == piece:
                return raw
    return piece.encode("utf-8")


def decode_token_bytes(table, token_ids):
    """``(text, pending)``: whole characters and the incomplete UTF-8 tail.

    Returns ``None`` for a byte sequence that can never become valid UTF-8.
    """
    data = b"".join(table[token] or b"" for token in token_ids if token < len(table))
    return _split_utf8(data)


def _split_utf8(data):
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    try:
        text = decoder.decode(data)
    except UnicodeDecodeError:
        return None
    return text, decoder.getstate()[0]


_UTF8_MINIMUM = (0, 0, 0x80, 0x800, 0x10000)


def _prefix_codepoints(prefix):
    """Inclusive codepoint ranges whose UTF-8 encoding starts with ``prefix``."""
    lead = prefix[0]
    if lead >> 5 == 0b110:
        length, value = 2, lead & 0x1F
    elif lead >> 4 == 0b1110:
        length, value = 3, lead & 0x0F
    elif lead >> 3 == 0b11110:
        length, value = 4, lead & 0x07
    else:
        return ()
    for byte in prefix[1:]:
        value = (value << 6) | (byte & 0x3F)
    shift = 6 * (length - len(prefix))
    low = max(value << shift, _UTF8_MINIMUM[length])
    high = min((value << shift) | ((1 << shift) - 1), _MAX_CODEPOINT)
    ranges = []
    for start, stop in ((low, min(high, 0xD7FF)), (max(low, 0xE000), high)):
        if start <= stop:
            ranges.append((start, stop))
    return tuple(ranges)


def prefix_viable(automaton, config, prefix):
    """Whether some character completing ``prefix`` keeps ``config`` alive."""
    tried = set()
    for low, high in _prefix_codepoints(prefix):
        index = bisect_right(automaton.bounds, low) - 1
        while index < len(automaton.bounds) and automaton.bounds[index] <= high:
            symbol = automaton.segment_class[index]
            if symbol not in tried:
                tried.add(symbol)
                if automaton.step(config, symbol) is not None:
                    return True
            index += 1
    return False


def advance_bytes(automaton, config, data):
    """``(config, pending)`` after ``data``, or ``None`` when it cannot match."""
    split = _split_utf8(data)
    if split is None:
        return None
    text, pending = split
    config = automaton.advance(config, text)
    if config is None:
        return None
    if pending and not prefix_viable(automaton, config, pending):
        return None
    return config, pending


class ByteFragments:
    """Admissibility of the pieces the text trie cannot represent."""

    def __init__(self, table, pieces, eos_ids):
        self.items = [
            (token, raw)
            for token, raw in enumerate(table)
            if raw and token not in eos_ids and "�" in pieces[token]
        ]
        self.memo = OrderedDict()
        self.lock = threading.Lock()

    def allowed_ids(self, automaton, config, pending):
        key = (automaton.uid, config, pending)
        with self.lock:
            cached = self.memo.get(key)
            if cached is not None:
                self.memo.move_to_end(key)
                return cached
        found = []
        for token, raw in self.items:
            if not pending and raw[0] & 0xC0 == 0x80:
                continue  # a continuation byte needs an open character
            if advance_bytes(automaton, config, pending + raw) is not None:
                found.append(token)
        result = np.array(found, dtype=np.int64)
        with self.lock:
            self.memo[key] = result
            while len(self.memo) > _FRAGMENT_MEMO_ENTRIES:
                self.memo.popitem(last=False)
        return result


def fragments_for(tokenizer, pieces, eos_ids):
    """``(byte table, ByteFragments)`` or ``(None, None)`` when unavailable."""
    table = token_bytes_for(tokenizer, pieces)
    if table is None:
        return None, None
    cached = getattr(tokenizer, _BYTES_ATTRIBUTE + "_index", None)
    if not isinstance(cached, ByteFragments):
        cached = ByteFragments(table, pieces, frozenset(eos_ids))
        try:
            setattr(tokenizer, _BYTES_ATTRIBUTE + "_index", cached)
        except (AttributeError, TypeError):
            pass
    return table, cached
