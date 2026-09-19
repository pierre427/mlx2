import mlx.core as mx

from mlx2.runtime.models.cache import KVCache
from mlx2.runtime.pld import PromptLookupBatchGenerator


class _PatternModel:
    def __init__(self, *, reject=False):
        self.reject = reject

    def __call__(self, tokens, *, cache):
        values = tokens.astype(mx.float32)[:, None, :, None]
        cache[0].update_and_fetch(values, values)
        vocabulary = 8
        predicted = (
            mx.full(tokens.shape, 4, dtype=mx.int32)
            if self.reject
            else mx.where(tokens == 1, 2, 1)
        )
        return mx.where(
            mx.arange(vocabulary)[None, None, :] == predicted[..., None],
            20.0,
            -20.0,
        )


def _run(model, maximum):
    generator = PromptLookupBatchGenerator(
        model,
        prefill_step_size=16,
        prompt_lookup={"num_draft": 2, "ngram_min": 2, "ngram_max": 2},
    )
    uid = generator.insert(
        [[1, 2, 1, 2]],
        max_tokens=[maximum],
        caches=[[KVCache()]],
    )[0]
    prompts, responses = generator.next()
    assert not responses and prompts[0].uid == uid and prompts[0].end_of_prompt
    emitted = []
    final = None
    while final is None:
        _, responses = generator.next()
        emitted.extend(response.token for response in responses)
        final = next(
            (response for response in responses if response.finish_reason), None
        )
    return emitted, final


def test_live_prompt_lookup_accepts_exact_indexed_continuation():
    emitted, final = _run(_PatternModel(), 3)
    assert emitted == [1, 2, 1]
    assert final.speculative_receipt["execution"] == "prompt_lookup_verify"
    assert final.speculative_receipt["accepted"] == 2
    assert final.speculative_receipt["proposed"] == 2
    assert final.prompt_cache[0].offset == len(final.all_tokens) == 6


def test_live_prompt_lookup_rejection_rewinds_to_committed_boundary():
    emitted, final = _run(_PatternModel(reject=True), 2)
    assert emitted == [4, 4]
    assert final.speculative_receipt["accepted"] == 0
    assert final.prompt_cache[0].offset == len(final.all_tokens) == 5


def test_deferred_admission_probes_on_plain_path_then_activates():
    generator = PromptLookupBatchGenerator(
        _PatternModel(),
        prefill_step_size=16,
        prompt_lookup={
            "num_draft": 2,
            "ngram_min": 2,
            "ngram_max": 2,
            "deferred_admission": True,
            "admission_window": 1,
            "admission_gate": 1.0,
            "admission_reprobe_interval": 1,
            "adaptive_warmup": 100,
        },
    )
    uid = generator.insert(
        [[1, 2, 1, 2]], max_tokens=[4], caches=[[KVCache()]]
    )[0]
    generator.next()
    _, first = generator.next()
    assert first[0].uid == uid and not first[0].from_draft
    _, second = generator.next()
    receipts = [response.speculative_receipt for response in second]
    assert receipts[-1]["admission_activations"] == 1
    # The activation happens only after the ordinary probe commits; the next
    # closed boundary is the first one allowed to run a verify span.
    _, third = generator.next()
    assert any(response.from_draft for response in third)


def test_cliff_aware_pld_extends_past_the_configured_plateau():
    generator = PromptLookupBatchGenerator(
        _PatternModel(),
        prefill_step_size=32,
        prompt_lookup={
            "num_draft": 8,
            "ngram_min": 2,
            "ngram_max": 2,
            "cliff_aware_span": True,
        },
    )
    prompt = [1, 2] * 10
    generator.insert([prompt], max_tokens=[18], caches=[[KVCache()]])
    while True:
        _prompts, responses = generator.next()
        if responses:
            receipt = responses[-1].speculative_receipt
            assert max(map(int, receipt["verify_span_hist"])) == 16
            assert receipt["span_extend_cycles"] == 1
            break
