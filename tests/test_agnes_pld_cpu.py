"""Prompt lookup on Agnes 3 Flash (GDN + global attention): exact greedy parity.

Tiny random weights on CPU.  Accepted and rejected lookups must leave the
GDN recurrent state and the KV cache exactly where ordinary decode would.
"""
import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.models.agnes import LAYER_DELTA, LAYER_GLOBAL, Model, ModelArgs, TextModelArgs
from mlx2.runtime.pld import PromptLookupBatchGenerator


def tiny(seed=4, vocab=64):
    mx.random.seed(seed)
    args = TextModelArgs(
        hidden_size=32, num_hidden_layers=4, intermediate_size=64, vocab_size=vocab,
        layer_types=[LAYER_DELTA, LAYER_DELTA, LAYER_DELTA, LAYER_GLOBAL],
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_conv_kernel_dim=3,
        rope_parameters={"type": "default", "partial_rotary_factor": 0.25, "rope_theta": 10000.0},
    )
    model = Model(ModelArgs(model_type="agnes", text_config=args.__dict__))
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


@pytest.mark.parametrize("batched_verify", [False, True])
def test_agnes_pld_matches_ordinary_greedy(batched_verify):
    model = tiny()
    seed_prompt = [3, 7, 11, 5]
    reference = greedy(model, seed_prompt, 16)
    prompt = seed_prompt + reference[:10] + seed_prompt
    expected = greedy(model, prompt, 18)
    generator = PromptLookupBatchGenerator(
        model, prefill_step_size=8,
        prompt_lookup={"num_draft": 4, "ngram_min": 2, "ngram_max": 4, "batched_verify": batched_verify},
    )
    uid = generator.insert([prompt], max_tokens=[18], caches=[model.make_cache()])[0]
    emitted, final = [], None
    for _ in range(200):
        _, responses = generator.next()
        emitted.extend(r.token for r in responses if r.uid == uid)
        final = next((r for r in responses if r.finish_reason), None)
        if final is not None:
            break
    assert final is not None
    assert emitted == expected
    assert final.speculative_receipt["proposed"] > 0
