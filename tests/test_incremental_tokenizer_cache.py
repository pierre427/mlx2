import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from mlx2.runtime.incremental_tokenizer_cache import IncrementalPromptTokenizerCache
from mlx2.serving import HostPromptCache, ServingEngine

LOCAL_QWEN38 = Path(
    "~/mlx-models/Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP"
)

TEMPLATE = """{% for message in messages %}<s>{{ message['role'] }}\n{{ message['content'] }}</s>\n{% endfor %}<s>assistant\n"""


def _backend(*, alternate=False, dropout=None):
    alphabet = list(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 <>/|_-'\n"
    )
    vocab = {character: index for index, character in enumerate(alphabet)}
    vocab["[UNK]"] = len(vocab)
    merges = []
    if alternate:
        # The replacement has the same type and vocabulary size, but a
        # different mapping: this is the mutation that defeated geometry-only
        # identity checks in the withdrawn prototype.
        first, second = vocab["a"], vocab["b"]
        vocab["a"], vocab["b"] = second, first
    tokenizer = Tokenizer(
        models.BPE(
            vocab=vocab,
            merges=merges,
            unk_token="[UNK]",
            dropout=dropout,
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    tokenizer.add_special_tokens([AddedToken("<s>", normalized=False, special=True)])
    return tokenizer


class _Wrapper:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._chat_template = None
        self._v1_encode_worker = None

    def encode(self, text, *, add_special_tokens=False):
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)


class _Adapter:
    incremental_tokenizer_cache_supported = True
    incremental_tokenizer_renderer_revision = "test-renderer-v1"

    def __init__(self, *, alternate=False, added_tokens=()):
        backend = _backend(alternate=alternate)
        if added_tokens:
            backend.add_tokens(list(added_tokens))
        fast = PreTrainedTokenizerFast(tokenizer_object=backend)
        fast.chat_template = TEMPLATE
        self.tokenizer = _Wrapper(fast)
        self.identity = {"fingerprint": "fixture-artifact-v1"}

    @staticmethod
    def render_incremental_prompt(tokenizer, request):
        if "messages" in request:
            return tokenizer.apply_chat_template(
                request["messages"], tokenize=False, add_generation_prompt=True
            )
        return request["prompt"]

    def render_prompt(self, request):
        return self.render_incremental_prompt(self.tokenizer._tokenizer, request)

    def prompt_tokens(self, request):
        return self.tokenizer.encode(
            self.render_prompt(request), add_special_tokens=False
        )


def _request(text):
    return {"messages": [{"role": "user", "content": text}]}


def _selected(*, entries=4, characters=100_000, tokens=100_000):
    adapter = _Adapter()
    cache = IncrementalPromptTokenizerCache(
        max_entries=entries,
        max_characters=characters,
        max_tokens=tokens,
    )
    assert cache.bind(adapter) is True
    return adapter, cache


def _tokenize(cache, adapter, request):
    prepared = cache.prepare(request)
    assert prepared is not None
    return prepared, cache.tokenize(prepared, lambda: adapter.prompt_tokens(request))


def test_default_off_and_non_opted_adapter_fail_closed():
    adapter = _Adapter()
    disabled = IncrementalPromptTokenizerCache()
    assert disabled.bind(adapter) is False
    assert disabled.status()["selected"] is False

    adapter.incremental_tokenizer_cache_supported = False
    selected = IncrementalPromptTokenizerCache(max_entries=2)
    assert selected.bind(adapter) is False
    assert selected.status()["refusal"] == "adapter_not_opted_in"


def test_bpe_dropout_refuses_nondeterministic_suffix_reuse():
    adapter = _Adapter()
    adapter.tokenizer._tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=_backend(dropout=0.1)
    )
    adapter.tokenizer._tokenizer.chat_template = TEMPLATE
    candidate = IncrementalPromptTokenizerCache(max_entries=2)
    assert candidate.bind(adapter) is False
    assert candidate.status()["refusal"] == "bpe_dropout"


def test_growing_edited_and_interleaved_prompts_match_bound_full_reference():
    adapter, cache = _selected()
    requests = [
        _request("alpha"),
        _request("alpha beta"),
        _request("conversation B"),
        _request("alpha beta gamma"),
        _request("edited earlier content"),
    ]
    actions = []
    for request in requests:
        prepared, (actual, receipt) = _tokenize(cache, adapter, request)
        assert actual == cache.bound_full_tokens(prepared)
        assert receipt["exact"] is True
        actions.append(receipt["action"])
    assert actions[0] == "ordinary_full_validation"
    assert "incremental_hit" in actions
    assert cache.status()["incremental_hits"] >= 1


def test_added_token_and_same_size_model_mutations_cannot_change_bound_revision():
    adapter, cache = _selected()
    first = _request("alpha")
    grown = _request("alpha beta")
    prepared, (original, _receipt) = _tokenize(cache, adapter, first)
    revision = prepared.revision

    live = adapter.tokenizer._tokenizer.backend_tokenizer
    live.add_tokens([AddedToken("alpha", normalized=False, special=True)])
    live.model = _backend(alternate=True).model
    assert adapter.prompt_tokens(first) != original

    prepared, (actual, receipt) = _tokenize(cache, adapter, grown)
    assert prepared.revision == revision
    assert actual == cache.bound_full_tokens(prepared)
    assert receipt["action"] == "incremental_hit"
    assert cache.status()["source_mutation_policy"] == (
        "isolated_by_immutable_bound_snapshots"
    )

    # A lifecycle clear requires a new ordinary parity gate.  The now-mutated
    # source cannot silently validate the old bound revision.
    cache.clear()
    prepared = cache.prepare(grown)
    live_tokens = adapter.prompt_tokens(grown)
    actual, receipt = cache.tokenize(prepared, lambda: live_tokens)
    assert actual == live_tokens
    assert receipt["action"] == "refused_full"
    assert cache.status()["refusal"] == "ordinary_reference_mismatch"


def test_non_special_added_token_extends_overlap_margin_but_not_cut_marks():
    adapter = _Adapter(
        added_tokens=(AddedToken("<s>abcdefghX", normalized=False, special=False),)
    )
    cache = IncrementalPromptTokenizerCache(max_entries=4)
    assert cache.bind(adapter)

    _tokenize(cache, adapter, {"prompt": "<s>abcdefghY tail"})
    prepared, (actual, receipt) = _tokenize(
        cache, adapter, {"prompt": "<s>abcdefghX tail"}
    )

    assert actual == adapter.prompt_tokens({"prompt": "<s>abcdefghX tail"})
    assert actual == cache.bound_full_tokens(prepared)
    assert receipt["exact"] is True


def test_same_length_live_template_replacement_does_not_change_frozen_render():
    adapter, cache = _selected()
    request = _request("alpha")
    prepared, (expected, _receipt) = _tokenize(cache, adapter, request)
    frozen_text = prepared.text

    replacement = TEMPLATE.replace("assistant", "assistanz")
    assert len(replacement) == len(TEMPLATE)
    adapter.tokenizer._tokenizer.chat_template = replacement
    assert adapter.render_prompt(request) != frozen_text

    prepared = cache.prepare(request)
    assert prepared.text == frozen_text
    actual, receipt = cache.tokenize(prepared, lambda: adapter.prompt_tokens(request))
    assert actual == expected
    assert receipt["tokenizer_revision"] == prepared.revision


def test_cold_parity_mismatch_refuses_and_preserves_ordinary_result():
    adapter, cache = _selected()
    request = _request("alpha")
    prepared = cache.prepare(request)
    ordinary = adapter.prompt_tokens(request) + [999]
    actual, receipt = cache.tokenize(prepared, lambda: ordinary)
    assert actual == ordinary
    assert receipt["action"] == "refused_full"
    assert receipt["refusal"] == "ordinary_reference_mismatch"
    assert cache.status()["selected"] is False


def test_tokenizers_v1_selected_after_bind_refuses_before_prepare():
    adapter, cache = _selected()
    adapter.tokenizer._v1_encode_worker = object()
    assert cache.prepare(_request("alpha")) is None
    assert cache.status()["refusal"] == "tokenizer_v1_selected_after_bind"


def test_origin_sensitive_plain_spans_fail_closed_to_ordinary_reference():
    adapter = _Adapter()
    adapter.incremental_plain_spans = lambda _request, text: (
        (text.index("alpha"), text.index("alpha") + len("alpha")),
    )
    cache = IncrementalPromptTokenizerCache(max_entries=2)
    assert cache.bind(adapter)
    request = _request("alpha")
    prepared = cache.prepare(request)
    ordinary = adapter.prompt_tokens(request)
    actual, receipt = cache.tokenize(prepared, lambda: ordinary)
    assert actual == ordinary
    assert receipt["action"] == "plain_span_full"
    assert cache.status()["entries"] == 0


def test_ordinary_exception_is_not_shaped_or_swallowed():
    _adapter, cache = _selected()
    prepared = cache.prepare(_request("alpha"))

    class OrdinaryError(RuntimeError):
        pass

    def fail():
        raise OrdinaryError("ordinary path failed")

    try:
        cache.tokenize(prepared, fail)
    except OrdinaryError as error:
        assert str(error) == "ordinary path failed"
    else:
        raise AssertionError("ordinary exception was swallowed")


def test_lru_and_character_token_bounds():
    adapter, cache = _selected(entries=2, characters=90, tokens=90)
    one = _request("a" * 8)
    two = _request("b" * 8)
    three = _request("c" * 8)
    for request in (one, two):
        _tokenize(cache, adapter, request)
    # Touch one through an exact re-tokenization, then add three: two is LRU.
    _tokenize(cache, adapter, one)
    _tokenize(cache, adapter, three)
    status = cache.status()
    assert status["entries"] <= 2
    assert status["characters"] <= 90
    assert status["tokens"] <= 90
    assert status["evictions"] >= 1

    oversize_adapter, oversize = _selected(entries=2, characters=8, tokens=8)
    _tokenize(oversize, oversize_adapter, _request("long prompt" * 8))
    assert oversize.status()["entries"] == 0
    assert oversize.status()["oversize_skips"] == 1


def test_concurrent_prepare_tokenize_status_and_clear_are_safe():
    adapter, cache = _selected(entries=8)
    errors = []

    def exercise(index):
        try:
            for step in range(20):
                request = _request(f"conversation {index} step {step}")
                prepared = cache.prepare(request)
                if prepared is not None:
                    actual, _receipt = cache.tokenize(
                        prepared, lambda request=request: adapter.prompt_tokens(request)
                    )
                    try:
                        expected = cache.bound_full_tokens(prepared)
                    except RuntimeError:
                        # clear() deliberately invalidates an in-flight epoch;
                        # tokenize still returned its immutable old-revision
                        # exact result, but a later reference lookup must not
                        # cross into the successor generation.
                        assert not cache.is_current(prepared)
                    else:
                        assert actual == expected
                cache.status()
                if index == 0 and step == 10:
                    cache.clear()
        except Exception as error:  # noqa: BLE001 - thread forwards failures
            errors.append(error)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(exercise, range(4)))
    assert errors == []


def test_rebind_cannot_commit_an_old_cold_validation():
    old_adapter, cache = _selected()
    request = _request("alpha")
    prepared = cache.prepare(request)
    entered = threading.Event()
    release = threading.Event()
    result = []

    def old_ordinary():
        entered.set()
        assert release.wait(timeout=5)
        return old_adapter.prompt_tokens(request)

    thread = threading.Thread(
        target=lambda: result.append(cache.tokenize(prepared, old_ordinary))
    )
    thread.start()
    assert entered.wait(timeout=5)
    new_adapter = _Adapter(alternate=True)
    assert cache.bind(new_adapter)
    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert result[0][1]["selected"] is False
    assert cache.status()["stale_validation_skips"] == 1

    new_calls = 0

    def new_ordinary():
        nonlocal new_calls
        new_calls += 1
        return new_adapter.prompt_tokens(request)

    current = cache.prepare(request)
    actual, receipt = cache.tokenize(current, new_ordinary)
    assert actual == new_adapter.prompt_tokens(request)
    assert receipt["action"] == "ordinary_full_validation"
    assert new_calls == 1


def test_clear_cannot_commit_an_old_cold_validation():
    adapter, cache = _selected()
    request = _request("alpha")
    prepared = cache.prepare(request)
    entered = threading.Event()
    release = threading.Event()

    def old_ordinary():
        entered.set()
        assert release.wait(timeout=5)
        return adapter.prompt_tokens(request)

    result = []
    thread = threading.Thread(
        target=lambda: result.append(cache.tokenize(prepared, old_ordinary))
    )
    thread.start()
    assert entered.wait(timeout=5)
    old_epoch = prepared.epoch
    cache.clear()
    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert result[0][1]["selected"] is False
    assert cache.status()["lifecycle_epoch"] != old_epoch

    current = cache.prepare(request)
    _actual, receipt = cache.tokenize(current, lambda: adapter.prompt_tokens(request))
    assert receipt["action"] == "ordinary_full_validation"


def test_stale_prepare_refusal_cannot_disable_a_rebound_revision():
    class BlockingAdapter(_Adapter):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def render_incremental_prompt(self, tokenizer, request):
            self.entered.set()
            assert self.release.wait(timeout=5)
            return object()

    old_adapter = BlockingAdapter()
    cache = IncrementalPromptTokenizerCache(max_entries=4)
    assert cache.bind(old_adapter)
    result = []
    thread = threading.Thread(
        target=lambda: result.append(cache.prepare(_request("old")))
    )
    thread.start()
    assert old_adapter.entered.wait(timeout=5)
    new_adapter = _Adapter(alternate=True)
    assert cache.bind(new_adapter)
    old_adapter.release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()

    assert result == [None]
    assert cache.status()["selected"] is True
    assert cache.status()["refusal"] is None
    assert cache.status()["stale_refusal_skips"] == 1


def test_host_prompt_cache_namespace_and_engine_validate_before_host_hit():
    request = _request("alpha")
    host = HostPromptCache(max_entries=4, max_tokens=100)
    host.put(request, [1], namespace="revision-a")
    assert host.get(request, namespace="revision-b") is None
    assert host.get(request, namespace="revision-a") == [1]

    adapter, incremental = _selected()
    engine = ServingEngine.__new__(ServingEngine)
    engine.adapter = adapter
    engine.prompt_lock = threading.Lock()
    engine.host_prompt_cache = HostPromptCache(max_entries=4, max_tokens=1000)
    engine.incremental_tokenizer_cache = incremental

    first, first_receipt = engine._tokenize_prompt(request)
    assert first_receipt["action"] == "ordinary_full_validation"
    live = adapter.tokenizer._tokenizer.backend_tokenizer
    live.add_tokens([AddedToken("alpha", normalized=False, special=True)])
    adapter.tokenizer._tokenizer.chat_template = TEMPLATE.replace(
        "assistant", "assistanz"
    )

    # The exact hit is validated by rendering with the immutable template and
    # namespaced by the immutable tokenizer revision before host lookup.
    again, hit_receipt = engine._tokenize_prompt(request)
    assert again == first
    assert hit_receipt["action"] == "host_prompt_cache_hit"

    grown = _request("alpha beta")
    grown_tokens, grown_receipt = engine._tokenize_prompt(grown)
    prepared = incremental.prepare(grown)
    assert grown_tokens == incremental.bound_full_tokens(prepared)
    assert grown_receipt["action"] == "incremental_hit"


def test_engine_retries_a_host_hit_when_tokenizer_rebinds_during_lookup():
    class BlockingHostPromptCache(HostPromptCache):
        def __init__(self):
            super().__init__(max_entries=4, max_tokens=1000)
            self.block = False
            self.entered = threading.Event()
            self.release = threading.Event()

        def get(self, request, *, namespace=None):
            tokens = super().get(request, namespace=namespace)
            if self.block and tokens is not None:
                self.entered.set()
                assert self.release.wait(timeout=5)
            return tokens

    request = _request("alpha")
    old_adapter, incremental = _selected()
    engine = ServingEngine.__new__(ServingEngine)
    engine.adapter = old_adapter
    engine.prompt_lock = threading.Lock()
    engine.host_prompt_cache = BlockingHostPromptCache()
    engine.incremental_tokenizer_cache = incremental
    old_tokens, _receipt = engine._tokenize_prompt(request)

    engine.host_prompt_cache.block = True
    result = []
    thread = threading.Thread(
        target=lambda: result.append(engine._tokenize_prompt(request))
    )
    thread.start()
    assert engine.host_prompt_cache.entered.wait(timeout=5)
    new_adapter = _Adapter(alternate=True)
    assert incremental.bind(new_adapter)
    engine.adapter = new_adapter
    engine.host_prompt_cache.release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()

    tokens, receipt = result[0]
    assert tokens == new_adapter.prompt_tokens(request)
    assert tokens != old_tokens
    assert receipt["action"] == "ordinary_full_validation"
    assert incremental.status()["stale_host_hit_skips"] == 1


def test_observed_used_is_current_epoch_while_hit_counts_are_cumulative():
    adapter, cache = _selected()
    _tokenize(cache, adapter, _request("alpha"))
    _tokenize(cache, adapter, _request("alpha beta"))
    used = cache.status()
    assert used["observed_used"] is True
    assert used["incremental_hits"] >= 1
    cumulative_hits = used["incremental_hits"]

    assert cache.bind(_Adapter(alternate=True))
    rebound = cache.status()
    assert rebound["observed_used"] is False
    assert rebound["incremental_hits"] == cumulative_hits


def test_server_cli_is_default_off_and_maps_explicit_bounds():
    from mlx2.server import build_parser, serving_engine_kwargs

    parser = build_parser()
    default = parser.parse_args(["--model", "fixture"])
    assert default.incremental_tokenizer_cache_entries == 0
    selected = parser.parse_args(
        [
            "--model",
            "fixture",
            "--incremental-tokenizer-cache-entries",
            "3",
            "--incremental-tokenizer-cache-characters",
            "4096",
            "--incremental-tokenizer-cache-tokens",
            "2048",
        ]
    )
    assert selected.incremental_tokenizer_cache_entries == 3
    assert selected.incremental_tokenizer_cache_characters == 4096
    assert selected.incremental_tokenizer_cache_tokens == 2048
    kwargs = serving_engine_kwargs(
        selected,
        None,
        native_mtp=False,
        approximate_kv=None,
        max_request_bytes=1024,
    )
    assert kwargs["incremental_tokenizer_cache_entries"] == 3
    assert kwargs["incremental_tokenizer_cache_characters"] == 4096
    assert kwargs["incremental_tokenizer_cache_tokens"] == 2048


def test_terminal_route_receipt_includes_prompt_tokenization_without_prompt_text():
    source = (Path(__file__).resolve().parents[1] / "src/mlx2/serving.py").read_text()
    assert '"prompt_tokenization": job.prompt_tokenization_receipt' in source
    adapter, cache = _selected()
    prepared, (_tokens, receipt) = _tokenize(cache, adapter, _request("secret alpha"))
    encoded = str(receipt)
    assert prepared.text not in encoded
    assert "secret alpha" not in encoded


@pytest.mark.skipif(
    not (LOCAL_QWEN38 / "tokenizer.json").is_file(),
    reason="local Qwen3.8 tokenizer absent",
)
def test_real_qwen38_candidate_is_exact_for_growth_edits_and_quoted_specials():
    from scripts.bench_incremental_tokenizer_cache import run

    report = run(LOCAL_QWEN38, base_characters=2000, turns=3, repeats=1)
    assert report["all_ids_exact"] is True
    assert report["environment"]["model_loaded"] is False
    assert report["environment"]["gpu_used"] is False
    rows = {row["name"]: row for row in report["cases"]}
    assert rows["growing-1"]["action"] == "incremental_hit"
    assert rows["edited-earlier-turn"]["action"] == "bound_full"
    assert rows["quoted-special-literal"]["exact"] is True
