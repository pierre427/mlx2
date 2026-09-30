"""Compiled-pattern caches keyed by client grammars stay bounded per process."""

import mlx2.structured_automaton as automaton
import mlx2.structured_output as structured


def test_worker_pattern_cache_is_lru_bounded(monkeypatch):
    monkeypatch.setattr(structured, "_WORKER_PATTERN_LIMIT", 4)
    monkeypatch.setattr(structured, "_WORKER_PATTERNS", {})
    for i in range(10):
        structured._worker_pattern(f"a{i}", 0)
    assert len(structured._WORKER_PATTERNS) == 4
    assert ("a9", 0) in structured._WORKER_PATTERNS
    assert ("a0", 0) not in structured._WORKER_PATTERNS
    # A hit refreshes recency instead of recompiling.
    kept = structured._worker_pattern("a6", 0)
    structured._worker_pattern("a10", 0)
    assert structured._worker_pattern("a6", 0) is kept


def test_charset_enumeration_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(automaton, "_ENUMERATED_LIMIT", 3)
    monkeypatch.setattr(automaton, "_ENUMERATED", {})
    for source in ("[a]", "[b]", "[c]", "[d]", "[e]"):
        automaton._enumerate_charset(source)
    assert len(automaton._ENUMERATED) == 3
    assert "[e]" in automaton._ENUMERATED
