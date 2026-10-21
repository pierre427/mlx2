"""Prompt lookup on the standard decoder (Qwen3/Qwen2/Llama): exact greedy parity.

Tiny random Qwen3 weights on CPU.  The PLD route must emit exactly the
ordinary greedy continuation, including on a repetitive prompt where lookup
proposals are accepted and on rejections that rewind the KV cache.
"""
import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.models.standard_decoder import Model, ModelArgs
from mlx2.runtime.pld import PromptLookupBatchGenerator


def tiny(seed=3, vocab=64):
    mx.random.seed(seed)
    model = Model(ModelArgs(
        model_type="qwen3", hidden_size=32, num_hidden_layers=2, intermediate_size=64,
        num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=vocab, num_key_value_heads=2,
        max_position_embeddings=512, rope_theta=10000.0, tie_word_embeddings=True, head_dim=8,
    ))
    model.eval()
    mx.eval(model.parameters())
    return model


def greedy(model, prompt, count):
    cache = model.make_cache()
    tokens, out = list(prompt), []
    for step in range(count):
        logits = model(mx.array([tokens if step == 0 else [tokens[-1]]]), cache=cache)
        token = int(mx.argmax(logits[0, -1]).item())
        tokens.append(token)
        out.append(token)
    return out


def run_pld(model, prompt, count, policy):
    generator = PromptLookupBatchGenerator(model, prefill_step_size=8, prompt_lookup=policy)
    uid = generator.insert([prompt], max_tokens=[count], caches=[model.make_cache()])[0]
    emitted, final = [], None
    for _ in range(10 * count + 20):
        _, responses = generator.next()
        emitted.extend(r.token for r in responses if r.uid == uid)
        final = next((r for r in responses if r.finish_reason), None)
        if final is not None:
            return emitted, final
    raise AssertionError("PLD generator stalled")


@pytest.mark.parametrize("policy", [
    {"num_draft": 4, "ngram_min": 2, "ngram_max": 4},
    {"num_draft": 8, "ngram_min": 2, "ngram_max": 6, "cost_aware_admission": True},
])
def test_standard_decoder_pld_matches_ordinary_greedy(policy):
    model = tiny()
    reference_prompt = [5, 9, 11]
    reference = greedy(model, reference_prompt, 24)
    # Seed the prompt with the model's own continuation so lookup proposals
    # are accepted, then continue past it so rejections rewind too.
    prompt = reference_prompt + reference[:12] + reference_prompt
    expected = greedy(model, prompt, 20)
    emitted, final = run_pld(model, prompt, 20, policy)
    assert emitted == expected
    receipt = final.speculative_receipt
    assert receipt["execution"] == "prompt_lookup_verify"
    if not policy.get("cost_aware_admission"):
        assert receipt["proposed"] > 0 and receipt["accepted"] > 0


def test_standard_decoder_declares_pld_as_its_no_drafter_default():
    # Pierre (via coordinator, 2026-10-21): with no drafter specified and none
    # built in, the default route is adaptive prompt lookup with latches.
    from mlx2.adapters.registry import AdapterResolution
    from mlx2.adapters.standard_decoder import StandardDecoderAdapter, descriptor_for
    from mlx2.server import build_parser, resolve_route_selection

    resolution = AdapterResolution(StandardDecoderAdapter, descriptor_for("qwen3"), {"fingerprint": "f"})
    selected = resolve_route_selection(build_parser().parse_args(["--model", "x"]), None, resolution)
    assert (selected.route, selected.source) == ("prompt_lookup", "capability_default")
