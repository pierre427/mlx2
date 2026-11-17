"""Prompt-token logprobs: parsing, gate, fidelity, chunked scorer, shapes (CPU units).

Serving-path, concurrency and HTTP tests: test_prompt_logprobs_serving_cpu.py.
Real-tokenizer offsets: test_prompt_logprobs_tokenizers_cpu.py.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.prompt_logprobs import (
    PromptLogprobsPolicy,
    PromptLogprobsRefused,
    PromptScores,
    admit_prompt_length,
    admit_prompt_scoring,
    chunk_log_softmax,
    extract_prompt_scoring,
    int8_chunk_cap,
    legacy_echo_logprobs,
    prompt_scoring_binding,
    run_prompt_scoring,
    score_prompt_tokens,
    scoring_fidelity,
    vllm_prompt_logprobs,
)


def tiny_model(vocab=97, seed=3):
    from mlx2.runtime.models.standard_decoder import Model, ModelArgs

    args = ModelArgs(
        model_type="qwen3", hidden_size=64, num_hidden_layers=3,
        intermediate_size=64, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=vocab, num_key_value_heads=2, max_position_embeddings=4096,
        rope_theta=10000.0, tie_word_embeddings=True, head_dim=16,
    )
    mx.random.seed(seed)
    model = Model(args)
    model.eval()
    mx.eval(model.parameters())
    return model


class CharTokenizer:
    """Token id ``i`` spells ``chr(97 + i % 26)``; enough for shapes."""

    all_special_ids = ()

    def decode(self, ids, **_):
        return "".join(chr(97 + int(i) % 26) for i in ids)

    def convert_ids_to_tokens(self, ids):
        return [self.decode([i]) for i in ids]


def reference_logprobs(model, tokens):
    logits = model(mx.array(tokens)[None])[0].astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


# ---------------------------------------------------------------- parsing


def test_extract_leaves_ordinary_requests_untouched():
    body = {"prompt": "x", "max_tokens": 3}
    assert extract_prompt_scoring(body, chat=False) == (body, None)
    stripped, spec = extract_prompt_scoring({"prompt": "x", "echo": False, "max_tokens": 0}, chat=False)
    assert spec is None and stripped == {"prompt": "x", "max_tokens": 0}
    # Recognised nulls are absent fields.
    assert extract_prompt_scoring({"prompt": "x", "echo": None, "prompt_logprobs": None},
                                  chat=False)[1] is None


def test_extract_completions_echo_logprobs_and_max_tokens_zero():
    stripped, spec = extract_prompt_scoring(
        {"prompt": "abc", "echo": True, "logprobs": 3, "max_tokens": 0}, chat=False
    )
    assert stripped == {"prompt": "abc", "logprobs": 3}
    assert spec.echo and spec.echo_top == 3 and spec.prompt_top is None
    assert not spec.generate and spec.scoring and spec.top_k == 3
    _, spec = extract_prompt_scoring({"prompt": "abc", "echo": True}, chat=False)
    assert spec.echo and not spec.scoring and spec.generate


def test_extract_prompt_logprobs_on_both_endpoints_and_null_n_stream():
    _, spec = extract_prompt_scoring({"prompt": "abc", "prompt_logprobs": 0, "n": None,
                                      "stream": None}, chat=False)
    assert spec.prompt_top == 0 and spec.scoring and not spec.echo
    stripped, spec = extract_prompt_scoring(
        {"messages": [{"role": "user", "content": "hi"}], "prompt_logprobs": 5, "max_tokens": 0},
        chat=True,
    )
    assert "prompt_logprobs" not in stripped and "max_tokens" not in stripped
    assert spec.prompt_top == 5 and not spec.generate


@pytest.mark.parametrize(
    "body,chat,match",
    [
        ({"messages": [], "echo": True}, True, "Completions field"),
        ({"prompt": "a", "echo": "yes"}, False, "echo must be boolean"),
        ({"prompt": "a", "prompt_logprobs": -1}, False, "non-negative"),
        ({"prompt": "a", "prompt_logprobs": True}, False, "non-negative"),
        ({"prompt": "a", "echo": True, "logprobs": -2}, False, "non-negative"),
        ({"prompt": "a", "prompt_logprobs": 1, "max_tokens": 0, "max_completion_tokens": 4}, False, "disagree"),
        ({"prompt": "a", "echo": True, "logprobs": 1, "logprobs_start": 2}, False, "logprobs_start"),
    ],
)
def test_extract_rejects_malformed_fields(body, chat, match):
    with pytest.raises(ValueError, match=match):
        extract_prompt_scoring(body, chat=chat)


def test_extract_refuses_stream_and_parallel_samples():
    with pytest.raises(PromptLogprobsRefused) as error:
        extract_prompt_scoring({"prompt": "a", "prompt_logprobs": 1, "stream": True}, chat=False)
    assert error.value.code == "prompt_logprobs_stream_unsupported"
    with pytest.raises(PromptLogprobsRefused):
        extract_prompt_scoring({"prompt": "a", "echo": True, "logprobs": 1, "n": 2}, chat=False)


def test_policy_gate_caps_and_access():
    spec = extract_prompt_scoring({"prompt": "a", "prompt_logprobs": 4}, chat=False)[1]
    with pytest.raises(PromptLogprobsRefused) as error:
        admit_prompt_scoring(PromptLogprobsPolicy(), spec, {"prompt": "a"}, base_model="m")
    assert error.value.http_status == 400 and error.value.code == "prompt_logprobs_disabled"
    assert error.value.mlx2["prompt_logprobs"]["reason"] == "capability_disabled"
    on = PromptLogprobsPolicy(enabled=True, max_top=3, max_prompt_tokens=8)
    with pytest.raises(PromptLogprobsRefused, match="cap of 3"):
        admit_prompt_scoring(on, spec, {"prompt": "a"}, base_model="m")
    small = extract_prompt_scoring({"prompt": "a", "prompt_logprobs": 2}, chat=False)[1]
    admit_prompt_scoring(on, small, {"prompt": "a", "model": "m"}, base_model="m")
    for body, match in (({"model": "m:lora"}, "base model"), ({"weight_overlay": "w"}, "weight_overlay"),
                        ({"prefill_handoff": "h"}, "prefill_handoff")):
        with pytest.raises(PromptLogprobsRefused, match=match):
            admit_prompt_scoring(on, small, {"prompt": "a", **body}, base_model="m")
    admit_prompt_length(on, 8)
    with pytest.raises(PromptLogprobsRefused) as error:
        admit_prompt_length(on, 9)
    assert error.value.code == "prompt_logprobs_length_cap"
    # The served context is a cap too.
    with pytest.raises(PromptLogprobsRefused) as error:
        admit_prompt_length(on, 6, max_context=5)
    assert error.value.mlx2["prompt_logprobs"]["limit"] == 5
    # Access: without tenant auth every caller past the transport gate; with
    # it, the operator scope or a listed tenant.
    assert on.authorized(tenant_id="default", tenant_auth=False)
    assert not on.authorized(tenant_id="t1", tenant_auth=True)
    assert on.authorized(tenant_id="t1", tenant_auth=True, scoped=True)
    listed = PromptLogprobsPolicy(enabled=True, tenants=("t1",))
    assert listed.authorized(tenant_id="t1", tenant_auth=True)
    assert not listed.authorized(tenant_id="t2", tenant_auth=True)
    assert not listed.authorized(tenant_id="t1", tenant_auth=False)  # header-claimed


def test_exposure_validation_at_startup():
    from mlx2.prompt_logprobs import validate_exposure

    def check(policy, **kwargs):
        validate_exposure(policy, **{"loopback_bind": True, "tenant_auth": False, "api_key": False,
                                     "allow_unauthenticated": False, **kwargs})

    on = PromptLogprobsPolicy(enabled=True)
    check(on)  # loopback, no auth: the default stays allowed
    with pytest.raises(ValueError, match="tenant authentication"):
        check(PromptLogprobsPolicy(enabled=True, tenants=("research",)))
    check(PromptLogprobsPolicy(enabled=True, tenants=("research",)), tenant_auth=True)
    with pytest.raises(ValueError, match="non-loopback"):
        check(on, loopback_bind=False)
    check(on, loopback_bind=False, api_key=True)
    check(on, loopback_bind=False, tenant_auth=True)
    check(on, loopback_bind=False, allow_unauthenticated=True)
    check(PromptLogprobsPolicy(), loopback_bind=False)  # off: nothing to expose


def test_quantized_scoring_cache_is_refused():
    from mlx2.prompt_logprobs import scoring_cache_receipt
    from mlx2.runtime.models.cache import KVCache, QuantizedKVCache

    assert scoring_cache_receipt([KVCache(), KVCache()]) == {
        "kind": "fresh_request_private", "classes": ["KVCache"]}

    class Glm53LatentQ8Cache:
        pass

    class Glm53MLACache:
        caches = (Glm53LatentQ8Cache(), KVCache())

    for cache in ([QuantizedKVCache()], [Glm53MLACache()]):
        with pytest.raises(PromptLogprobsRefused) as error:
            scoring_cache_receipt(cache)
        assert error.value.mlx2["prompt_logprobs"]["reason"] == "route_not_exact"


def test_policy_validates_caps():
    with pytest.raises(ValueError):
        PromptLogprobsPolicy(chunk_tokens=1)
    with pytest.raises(ValueError):
        PromptLogprobsPolicy(enabled="on")
    with pytest.raises(ValueError):
        PromptLogprobsPolicy(tenants=("",))


# ---------------------------------------------------------------- fidelity


def test_fidelity_refuses_declared_approximate_routes():
    exact = SimpleNamespace(numeric_profile="official_bf16", numerics_revision="r1")
    receipt = scoring_fidelity(exact)
    assert receipt["fidelity"] == "exact" and receipt["numerics_revision"] == "r1"
    cases = [
        (SimpleNamespace(numeric_profile="p:ops_approximate_unqualified"), {}, "numeric_profile"),
        (SimpleNamespace(prompt_logprobs_fidelity=lambda: "yoco_v1"), {}, "adapter_declared:yoco_v1"),
        (SimpleNamespace(), {"apc_reuse_disabled_reason": "moe_keep_topk_8"}, "apcv2_reuse_disabled"),
        (SimpleNamespace(), {"numerical_contract": {"kv": "approximate_q8"}}, "contract"),
        (SimpleNamespace(), {"microscaling_active": True}, "microscaling"),
    ]
    for adapter, kwargs, needle in cases:
        with pytest.raises(PromptLogprobsRefused) as error:
            scoring_fidelity(adapter, **kwargs)
        assert error.value.http_status == 501
        receipt = error.value.mlx2["prompt_logprobs"]
        assert receipt["reason"] == "route_not_exact"
        assert any(needle in reason for reason in receipt["reasons"]), receipt
    # Batch-invariance-only APCv2 restrictions do not concern a width-1 forward.
    scoring_fidelity(SimpleNamespace(),
                     apc_reuse_disabled_reason="external_varlen_prefill_not_batch_invariant")


def test_deepseek_source_declares_approximate_expert_policies():
    from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter

    def fidelity(policy):
        adapter = DeepSeekV41SourceAdapter.__new__(DeepSeekV41SourceAdapter)
        adapter.kernel_policy = policy
        return adapter.prompt_logprobs_fidelity()

    assert fidelity({}) == "exact"
    assert "hot_original_cold_q1_experts" in fidelity(
        {"deepseek_source_experts": {"mode": "hot_original_q1"}})
    assert "dynamic_precision" in fidelity(
        {"deepseek_source_experts": {"mode": "hot_original_q1", "residency": "dynamic_precision_v1"}})


def test_int8_threshold_caps_the_scoring_chunk():
    handle = SimpleNamespace(policy=SimpleNamespace(row_threshold=512))
    assert int8_chunk_cap(2048, handle) == (511, {
        "active": True, "row_threshold": 512, "engaged": False,
        "mechanism": "chunk_below_row_threshold", "chunk_tokens": 511})
    assert int8_chunk_cap(256, handle)[0] == 256
    assert int8_chunk_cap(2048, None) == (2048, {"active": False})


# ---------------------------------------------------------------- scorer


def test_chunked_log_softmax_equals_the_full_computation():
    mx.random.seed(0)
    logits = mx.random.normal((37, 211)) * 6
    full = chunk_log_softmax(logits, mx)
    reference = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    assert mx.max(mx.abs(full - reference)).item() < 1e-5
    pieces = mx.concatenate([chunk_log_softmax(logits[i:i + 5], mx) for i in range(0, 37, 5)])
    assert mx.array_equal(pieces, full)


@pytest.mark.parametrize("chunk", [2, 3, 7, 64])
def test_chunked_scorer_matches_one_full_forward(chunk):
    from mlx2.runtime.models.cache import make_prompt_cache

    model = tiny_model()
    tokens = [5, 17, 2, 88, 41, 41, 9, 60, 3, 77, 12, 19, 30]
    reference = reference_logprobs(model, tokens)
    seen = []

    def forward(x, cache):
        seen.append(int(x.shape[1]))
        return model(x, cache=cache)

    scores = score_prompt_tokens(
        forward, make_prompt_cache(model), tokens, top_k=4, chunk_tokens=chunk, mx=mx
    )
    assert max(seen) <= chunk and sum(seen) in (len(tokens), len(tokens) - 1)
    assert len(scores.logprobs) == len(tokens) - 1
    for index, token in enumerate(tokens[1:]):
        row = reference[index]
        assert abs(scores.logprobs[index] - row[token].item()) < 1e-4
        assert scores.ranks[index] == int(mx.sum(row > row[token]).item()) + 1
        assert scores.top_ids[index] == mx.argsort(-row)[:4].tolist()


def _hybrid_and_sliding_models():
    from route_harness import tiny_qwen38_mtp

    hybrid, vocab = tiny_qwen38_mtp(vocab=97)
    yield "qwen3.8 hybrid GDN", hybrid
    from mlx2.adapters.muse_glimmer_config import ModelArgs
    from mlx2.runtime.models.muse_glimmer import Model

    mx.random.seed(8)
    sliding = Model(ModelArgs(
        hidden_size=16, intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        vocab_size=97, sliding_window=8, max_position_embeddings=512,
    ))
    sliding.eval()
    mx.eval(sliding.parameters())
    yield "muse sliding window", sliding


def test_recurrent_and_sliding_models_chunked_equal_one_shot():
    """GDN recurrent state and sliding windows ride the cache across chunks."""
    from mlx2.runtime.models.cache import make_prompt_cache

    tokens = [3, 14, 15, 92, 65, 35, 89, 79, 32, 38, 46, 26, 43, 38, 32, 79, 50, 28, 84, 19, 71]
    for name, model in _hybrid_and_sliding_models():
        # fp32 parameters: the comparison checks state plumbing, not bf16 shape noise.
        reference = reference_logprobs(model, tokens)
        for chunk in (4, 7):
            scores = score_prompt_tokens(lambda x, c: model(x, cache=c), make_prompt_cache(model),
                                         tokens, top_k=0, chunk_tokens=chunk, mx=mx)
            worst = max(abs(scores.logprobs[i] - reference[i, t].item())
                        for i, t in enumerate(tokens[1:]))
            assert worst < 2e-4, (name, chunk, worst)


def test_scorer_fails_closed_with_reasons():
    with pytest.raises(PromptLogprobsRefused) as error:
        score_prompt_tokens(lambda x, c: mx.zeros((1, 1, 11)), None, [1, 2, 3, 4],
                            top_k=0, chunk_tokens=8, mx=mx)
    assert (error.value.http_status, error.value.mlx2["prompt_logprobs"]["reason"]) == (
        501, "forward_not_full_sequence")
    with pytest.raises(PromptLogprobsRefused) as error:
        score_prompt_tokens(lambda x, c: (mx.zeros((1, 4, 11)), None), None, [1, 2, 3, 4],
                            top_k=0, chunk_tokens=8, mx=mx)
    assert (error.value.http_status, error.value.mlx2["prompt_logprobs"]["reason"]) == (
        501, "forward_returns_tuple")

    def nan_in_alternatives(x, cache):
        logits = mx.zeros((1, int(x.shape[1]), 11))
        return mx.concatenate([logits[..., :10], mx.full((1, int(x.shape[1]), 1), mx.nan)], axis=-1)

    with pytest.raises(PromptLogprobsRefused) as error:
        score_prompt_tokens(nan_in_alternatives, None, [1, 2, 3, 4], top_k=3, chunk_tokens=8, mx=mx)
    assert (error.value.http_status, error.value.mlx2["prompt_logprobs"]["reason"]) == (500, "nonfinite")


def test_single_token_prompt_runs_no_forward():
    def boom(x, cache):
        raise AssertionError("no row predicts anything")

    scores = score_prompt_tokens(boom, None, [7], top_k=2, chunk_tokens=4, mx=mx)
    assert scores.logprobs == [] and scores.chunks == 0


def test_binding_contract_and_refusals():
    model = tiny_model()
    assert prompt_scoring_binding(SimpleNamespace(model=model)).kind == "model_call"
    with pytest.raises(PromptLogprobsRefused) as error:
        prompt_scoring_binding(SimpleNamespace(model=model, prompt_logprobs_refusal="weights stream"))
    assert error.value.http_status == 501
    with pytest.raises(PromptLogprobsRefused, match="no in-process target"):
        prompt_scoring_binding(SimpleNamespace(model=None))
    custom = SimpleNamespace(prompt_logprobs_binding=lambda: ((lambda x, c: None), "cache"))
    assert prompt_scoring_binding(custom).kind == "adapter_binding"

    class WithPrefill:
        def __call__(self, x, cache=None):
            raise AssertionError("prefill_forward is the prompt forward")

        def prefill_forward(self, x, cache=None):
            return model(x, cache=cache)

        def make_cache(self):
            from mlx2.runtime.models.cache import make_prompt_cache
            return make_prompt_cache(model)

    scores, receipt = run_prompt_scoring(
        SimpleNamespace(model=WithPrefill()), [1, 2, 3, 4, 5], top_k=1, chunk_tokens=2
    )
    assert receipt["forward"] == "prefill_forward" and len(scores.logprobs) == 4
    assert receipt["apc"] == {"read": "bypassed", "write": "none", "cached_tokens": 0}
    assert receipt["lane_width"] == 1 and receipt["speculation"] == "not_used"
    assert receipt["qualified"] is False and receipt["observed_used"] is True


# ---------------------------------------------------------------- shapes


def _scores():
    return PromptScores(
        logprobs=[-1.5, -0.25], ranks=[3, 1],
        top_ids=[[4, 9, 1], [2, 4, 0]], top_logprobs=[[-0.5, -1.0, -1.5], [-0.25, -2.0, -3.0]],
        chunks=1, vocab=10,
    )


def test_vllm_prompt_logprobs_shape_includes_the_actual_token():
    rows = vllm_prompt_logprobs(CharTokenizer(), [0, 1, 2], _scores(), 2)
    assert rows[0] is None and len(rows) == 3
    assert rows[1]["1"] == {"logprob": -1.5, "rank": 3, "decoded_token": "b"}
    assert set(rows[1]) == {"1", "4", "9"}
    assert rows[2]["2"]["rank"] == 1 and set(rows[2]) == {"2", "4"}
    zero = vllm_prompt_logprobs(CharTokenizer(), [0, 1, 2], _scores(), 0)
    assert [set(row) for row in zero[1:]] == [{"1"}, {"2"}]


def test_legacy_echo_shape():
    generated = [{"id": 3, "logprob": -0.75, "top_logprobs": [{"id": 3, "logprob": -0.75}, {"id": 5, "logprob": -1.0}]}]
    out = legacy_echo_logprobs(CharTokenizer(), [0, 1, 2], _scores(), 2, generated)
    assert out["tokens"] == ["a", "b", "c", "d"]
    assert out["token_logprobs"] == [None, -1.5, -0.25, -0.75]
    assert out["top_logprobs"][0] is None
    assert out["top_logprobs"][1] == {"e": -0.5, "j": -1.0}
    assert out["top_logprobs"][3] == {"d": -0.75, "f": -1.0}
    assert out["text_offset"] == [0, 1, 2, 3]
