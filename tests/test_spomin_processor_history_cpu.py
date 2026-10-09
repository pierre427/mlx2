"""Live Spomin compaction must not shift the processors' prompt/generated split.

Serving builds every request processor (presence/frequency penalties,
``min_tokens``, Muse/North tool constraints, structured output) with the
original prompt length P and reads the generated span as ``tokens[P:]``.  A
post-prefill transform that retains R < P - 1 cached tokens must therefore not
shorten the history those processors see.
"""
import mlx.core as mx

from mlx2 import serving
from mlx2.adapters.muse_glimmer import MuseRecipientProcessor
from mlx2.runtime.generate import BatchGenerator
from mlx2.runtime.sample_utils import _generated_window, make_logits_processors
from mlx2.runtime.spomin_live_surgery import (
    ServingSpominPolicy,
    SpominLiveSurgeryManager,
)
from mlx2.runtime.spomin_standard_surgery import StandardAttentionSpominBackend

PROMPT = list(range(1, 14))  # 13 tokens: 12 cached, compacted to 9 below
MAX_TOKENS = 8


def _north(seed=7):
    from mlx2.runtime.models.cohere2_moe import Model, ModelArgs

    mx.random.seed(seed)
    model = Model(ModelArgs(
        hidden_size=16, head_dim=4, num_hidden_layers=2, intermediate_size=8,
        prefix_dense_intermediate_size=24, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=32, num_experts=4,
        num_experts_per_tok=2, first_k_dense_replace=1, sliding_window=4,
        layer_types=["full_attention", "sliding_attention"],
    ))
    model.eval()
    return model


def _run(processors, *, surgery=True):
    policy = ServingSpominPolicy(enabled=True, capacity_tokens=16, segment_tokens=3)
    manager = SpominLiveSurgeryManager(
        enabled=True, backend_factory=StandardAttentionSpominBackend
    )

    def transform(*, uid, model, prompt_cache, cached_token_ids):
        if not surgery:
            return None
        ledger = policy.transcript(
            cached_token_ids, tokenizer_identity="t", revision=f"r{uid}"
        )
        transaction = manager.prepare(
            request_id=str(uid), prompt_token_ids=cached_token_ids,
            transcript=ledger, capacity_tokens=16, strategy=policy.strategy,
            has_mtp_state=False, has_recurrent_state=False,
            cache_is_request_private=True,
            protected_segment_ids=(ledger.segments[0].segment_id,),
        )
        mx.synchronize()
        receipt = transaction.apply(
            model, prompt_cache, request_quiescent=True, device_work_drained=True
        )
        return {"receipt": receipt, "retained_token_ids": transaction.retained_token_ids,
                "prompt_cache": prompt_cache}

    batch = BatchGenerator(
        _north(), completion_batch_size=1, prefill_batch_size=1,
        prefill_step_size=32, post_prefill_transform=transform,
    )
    batch.insert([PROMPT], max_tokens=[MAX_TOKENS], logits_processors=[processors])
    tokens, receipt = [], None
    for _ in range(200):
        prompt_responses, responses = batch.next()
        for response in prompt_responses:
            if response.end_of_prompt:
                receipt = batch.pop_post_prefill_receipt(response.uid)
        tokens.extend(response.token for response in responses)
        if any(response.finish_reason for response in responses):
            break
    batch.close()
    return tokens, receipt


def test_compacted_lane_keeps_the_generated_window_processors_were_built_for():
    seen = []

    def recorder(tokens, logits):
        seen.append(len(_generated_window(tokens, 0, len(PROMPT))))
        return logits

    tokens, receipt = _run([recorder])
    assert receipt["status"] == "applied"
    assert receipt["retained_tokens"] < receipt["source_tokens"]
    assert len(tokens) == MAX_TOKENS
    # Step k has generated exactly k tokens, compacted or not.
    assert seen == list(range(MAX_TOKENS))


def test_compacted_lane_honours_min_tokens_and_presence_penalty():
    masked, penalized = [], []
    minimum = serving.minimum_tokens_processor(mx, [0], len(PROMPT), 2)
    (presence,) = make_logits_processors(
        presence_penalty=2.0, presence_context_size=0,
        penalty_generation_start=len(PROMPT),
    )

    def min_recorder(tokens, logits):
        out = minimum(tokens, logits)
        masked.append(bool(mx.isinf(out[0, 0]).item()))
        return out

    def presence_recorder(tokens, logits):
        before = logits + 0  # the processor edits logits in place
        out = presence(tokens, logits)
        penalized.append(int(mx.sum(out != before).item()))
        return out

    tokens, receipt = _run([min_recorder, presence_recorder])
    assert receipt["status"] == "applied"
    # EOS is suppressed for exactly min_tokens steps, not min_tokens + gap.
    assert masked == [True, True] + [False] * (MAX_TOKENS - 2)
    # The presence penalty covers every distinct token generated so far.
    assert penalized == [len(set(tokens[:k])) for k in range(MAX_TOKENS)]


def test_compacted_muse_lane_emits_the_recipient_header_once():
    header = [20, 21, 22]
    exact, _ = _run([MuseRecipientProcessor(len(PROMPT), [header])], surgery=False)
    compacted, receipt = _run([MuseRecipientProcessor(len(PROMPT), [header])])
    assert receipt["status"] == "applied"
    assert exact[:3] == header
    assert compacted[:3] == header
