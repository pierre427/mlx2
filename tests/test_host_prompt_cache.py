import queue
import threading
from collections import Counter
from types import SimpleNamespace

from mlx2.serving import HostPromptCache, ServingEngine


def test_host_prompt_cache_ignores_sampling_fields_and_copies_values():
    cache = HostPromptCache(max_entries=2, max_tokens=8)
    first = {"messages": [{"role": "user", "content": "hello"}], "temperature": 0}
    second = {**first, "temperature": 1, "top_p": 0.5}
    tokens = [1, 2, 3]
    cache.put(first, tokens)
    tokens.append(4)

    hit = cache.get(second)
    assert hit == [1, 2, 3]
    hit.append(9)
    assert cache.get(first) == [1, 2, 3]
    assert cache.status()["hits"] == 2


def test_host_prompt_cache_ignores_every_token_invariant_internal_control():
    base = {
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "weather"}}],
        "tool_choice": "auto",
    }
    expected = HostPromptCache.key(base)
    for field, value in {
        "session_id": "conversation-2",
        "skip_writing_prefix_cache": True,
        "thinking_budget": 128,
        "thinking_budget_mode": "history",
        "thinking_steer_alpha": 0.2,
        "parallel_tool_calls": False,
        "stream_options": {"include_usage": True},
        "max_completion_tokens": 64,
    }.items():
        assert HostPromptCache.key({**base, field: value}) == expected, field

    # These controls are adapter inputs and can alter rendered prompt tokens.
    for field, value in {
        "messages": [{"role": "user", "content": "different"}],
        "enable_thinking": True,
        "reasoning_effort": "high",
        "tools": [],
        "tool_choice": "none",
    }.items():
        assert HostPromptCache.key({**base, field: value}) != expected, field


def test_responses_translation_only_fields_never_reach_the_host_prompt_cache():
    from mlx2.openai_compat import responses_to_chat_request

    translated, _options = responses_to_chat_request(
        {
            "model": "fixture",
            "input": "hello",
            "metadata": {"trace": "x"},
            "store": False,
            "user": "client",
            "prompt_cache_key": "hint",
            "truncation": "disabled",
            "service_tier": "default",
            "include": [],
            "stream_options": {
                "include_usage": True,
                "include_obfuscation": False,
            },
        }
    )
    translation_only = {
        "input",
        "metadata",
        "store",
        "user",
        "prompt_cache_key",
        "truncation",
        "service_tier",
        "include",
        "stream_options",
    }
    assert translation_only.isdisjoint(translated)


def test_host_prompt_cache_is_lru_bounded_by_entries_and_tokens():
    cache = HostPromptCache(max_entries=2, max_tokens=5)
    cache.put({"prompt": "a"}, [1, 2])
    cache.put({"prompt": "b"}, [3, 4])
    assert cache.get({"prompt": "a"}) == [1, 2]
    cache.put({"prompt": "c"}, [5, 6])

    assert cache.get({"prompt": "b"}) is None
    assert cache.get({"prompt": "a"}) == [1, 2]
    assert cache.get({"prompt": "c"}) == [5, 6]
    status = cache.status()
    assert status["entries"] == 2
    assert status["tokens"] == 4
    assert status["evictions"] == 1


def test_host_prompt_cache_can_be_disabled_and_skips_oversize_entries():
    disabled = HostPromptCache(max_entries=0, max_tokens=0)

    def must_not_iterate():
        raise AssertionError("disabled cache consumed its input")
        yield 1

    disabled.put({"prompt": "a"}, must_not_iterate())
    assert disabled.get({"prompt": "a"}) is None
    assert disabled.status()["disabled_skips"] == 1

    cache = HostPromptCache(max_entries=2, max_tokens=2)
    cache.put({"prompt": "large"}, [1, 2, 3])
    assert cache.get({"prompt": "large"}) is None
    assert cache.status()["oversize_skips"] == 1


def test_host_prompt_cache_bounds_oversize_generator_consumption():
    cache = HostPromptCache(max_entries=2, max_tokens=3)
    consumed = []

    def tokens():
        for token in range(103):
            consumed.append(token)
            yield token

    cache.put({"prompt": "large"}, tokens())

    assert consumed == [0, 1, 2, 3]
    assert cache.get({"prompt": "large"}) is None
    status = cache.status()
    assert status["entries"] == 0
    assert status["tokens"] == 0
    assert status["oversize_skips"] == 1


def test_host_prompt_cache_disabled_and_oversize_stats_are_thread_safe():
    disabled = HostPromptCache(max_entries=0, max_tokens=0)
    oversize = HostPromptCache(max_entries=2, max_tokens=1)
    errors = []

    def exercise():
        try:
            for _ in range(250):
                assert disabled.get({"prompt": "a"}) is None
                disabled.put({"prompt": "a"}, [1])
                disabled.status()
                oversize.put({"prompt": "large"}, [1, 2])
                oversize.status()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=exercise) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert disabled.status()["disabled_misses"] == 1000
    assert disabled.status()["disabled_skips"] == 1000
    assert oversize.status()["oversize_skips"] == 1000


def test_parallel_samples_publish_only_the_prefill_leader():
    engine = ServingEngine.__new__(ServingEngine)
    engine.ready = threading.Event()
    engine.ready.set()
    engine.error = None
    engine.thread = SimpleNamespace(is_alive=lambda: True)
    engine.host_prompt_cache = HostPromptCache()
    engine.route_capabilities = frozenset()
    engine.qualification_mode = False
    engine.slots = threading.BoundedSemaphore(4)
    engine.max_lanes = 4
    engine.submission_lock = threading.Lock()
    engine.lock = threading.Lock()
    engine.jobs = {}
    engine.fanout_waiting = {}
    engine.queued_jobs = 0
    engine.incoming = queue.Queue()
    engine.counts = Counter()
    engine.batch_metrics = SimpleNamespace(
        admitted=lambda *_args: None,
        rejected=lambda *_args: None,
    )

    jobs = engine.submit_many(
        [{"prompt": "same", "seed": 1}, {"prompt": "same", "seed": 2}]
    )
    assert engine.incoming.qsize() == 1
    assert len(engine.jobs) == 2
    assert jobs[0].fanout_role == "prefill_leader"
    assert jobs[1].fanout_role == "apcv2_sibling"
    assert engine.fanout_waiting[jobs[0].fanout_group] == (jobs[1],)


def test_tenant_scoped_cache_namespace():
    from mlx2.serving import cache_semantic_fingerprint

    assert cache_semantic_fingerprint(None) == "text-token-v1"
    assert cache_semantic_fingerprint("a") != cache_semantic_fingerprint("b")
    assert cache_semantic_fingerprint("a") == cache_semantic_fingerprint("a")
    assert cache_semantic_fingerprint("a") != cache_semantic_fingerprint(None)


def test_tenant_scoped_cache_is_part_of_the_serving_settings(monkeypatch):
    """The knob changes cache identity, so it must be in the qualification
    settings; a receipt for the shared cache cannot select a scoped server."""
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "src" / "mlx2" / "serving.py").read_text()
    assert '"tenant_scoped_cache": self.tenant_scoped_cache' in source
    assert "--tenant-scoped-cache" in (Path(__file__).resolve().parents[1] / "src" / "mlx2" / "server.py").read_text()
