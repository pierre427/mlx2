"""Prompt lookup on Ripple (sliding-window + global NoPE attention): greedy parity.

Tiny random weights on CPU with a four-token window, so drafts, rejections and
rollback cross the rotating-cache wrap many times. Accepted and rejected
lookups must leave every rotating and full KV cache where ordinary decode would.
"""
import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.models.ricursive import Model, ModelArgs
from mlx2.runtime.pld import PromptLookupBatchGenerator


def tiny(seed=4, vocab=64, quantize=False):
    mx.random.seed(seed)
    args = ModelArgs(hidden_size=32, intermediate_size=64, num_hidden_layers=5,
                     num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                     vocab_size=vocab, sliding_window=4, embed_scale=71.55417527999327,
                     ffn_clamp_threshold=20.0)
    model = Model(args, nvfp4=False)
    for layer, value in zip(model.layers, [-2.75, -.3, .25, 1.1, 4.0], strict=True):
        layer.self_attn.value_res_lambda = mx.array([value])
    if quantize:
        from mlx import nn
        nn.quantize(model, group_size=32, bits=4,
                    class_predicate=lambda _p, m: isinstance(m, nn.Linear) and m.weight.shape[-1] % 32 == 0)
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


@pytest.mark.parametrize("quantize", [False, True])
@pytest.mark.parametrize("batched_verify", [False, True])
def test_ripple_pld_matches_ordinary_greedy_across_window_wrap(batched_verify, quantize):
    model = tiny(quantize=quantize)
    seed_prompt = [3, 7, 11, 5]
    reference = greedy(model, seed_prompt, 16)
    prompt = seed_prompt + reference[:10] + seed_prompt
    expected = greedy(model, prompt, 24)
    generator = PromptLookupBatchGenerator(
        model, prefill_step_size=8,
        prompt_lookup={"num_draft": 4, "ngram_min": 2, "ngram_max": 4, "batched_verify": batched_verify},
    )
    uid = generator.insert([prompt], max_tokens=[24], caches=[model.make_cache()])[0]
    emitted, final = [], None
    for _ in range(300):
        _, responses = generator.next()
        emitted.extend(r.token for r in responses if r.uid == uid)
        final = next((r for r in responses if r.finish_reason), None)
        if final is not None:
            break
    assert final is not None
    assert emitted == expected
    receipt = final.speculative_receipt
    assert receipt["proposed"] > 0
    assert receipt.get("accepted", 0) > 0


def test_ripple_prompt_lookup_is_explicit_only_until_gpu_ab():
    from types import SimpleNamespace

    from mlx2.adapters.ripple import DESCRIPTOR, RippleAdapter
    from mlx2.contracts import Capability
    from mlx2.server import build_parser, resolve_route_selection

    assert Capability.PROMPT_LOOKUP in DESCRIPTOR.capabilities
    resolution = SimpleNamespace(adapter_type=RippleAdapter, descriptor=DESCRIPTOR,
                                 default_route=RippleAdapter.default_route)
    parser = build_parser()
    selected = resolve_route_selection(parser.parse_args(["--model", "fixture"]), None, resolution)
    assert (selected.route, selected.source) == ("ordinary", "adapter_default")
    selected = resolve_route_selection(
        parser.parse_args(["--model", "fixture", "--prompt-lookup"]), None, resolution)
    assert (selected.route, selected.source) == ("prompt_lookup", "explicit_flag")
