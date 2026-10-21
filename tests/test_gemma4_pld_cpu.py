"""Gemma 4 prompt lookup: segmented verify rows on mlx-vlm's attention, exactly.

model-matrix-20261021 found that PLD's batched verify hands mlx-vlm's Gemma 4
attention a ``SegmentedKVView`` (no batched keys) and an mlx2 ``_RowMask``,
which ``mx.fast.scaled_dot_product_attention`` rejects with a TypeError that
killed the worker (HTTP 503).  The adapter's ``pld_verify_forward`` lets the
view reduce its own rows.  These tests drive the real prompt-lookup generator
on a tiny random-weight Gemma 4 (mlx-vlm's language model, dense and sparse
MoE, a sliding window that wraps) and compare against ordinary greedy decode.
"""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

mx.set_default_device(mx.cpu)

pytest.importorskip("mlx_vlm")

from mlx2.adapters.gemma4 import (  # noqa: E402
    GEMMA4_31B,
    GEMMA4_A4B,
    Gemma431BAdapter,
    Gemma4A4BAdapter,
    _Gemma4LogitsModel,
)
from mlx2.contracts import Capability  # noqa: E402
from mlx2.runtime.models.cache import KVCache, RotatingKVCache  # noqa: E402
from mlx2.runtime.pld import PromptLookupBatchGenerator  # noqa: E402

V, WINDOW = 64, 8
LAYERS = ["sliding_attention", "sliding_attention", "full_attention"] * 2
TEXT = dict(
    hidden_size=32,
    num_hidden_layers=len(LAYERS),
    intermediate_size=64,
    num_attention_heads=2,
    head_dim=8,
    global_head_dim=16,
    vocab_size=V,
    vocab_size_per_layer_input=V,
    num_key_value_heads=1,
    num_global_key_value_heads=1,
    attention_k_eq_v=True,
    num_kv_shared_layers=0,
    hidden_size_per_layer_input=0,
    sliding_window=WINDOW,
    layer_types=LAYERS,
    use_double_wide_mlp=False,
    max_position_embeddings=4096,
)
SPARSE = dict(
    enable_moe_block=True, num_experts=4, top_k_experts=2, moe_intermediate_size=16
)
# Longer than the sliding window, so every ring has wrapped before decode.
PROMPTS = [
    [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3],
    [7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8],
    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
]
EXPLORE = {
    "num_draft": 4,
    "ngram_min": 2,
    "ngram_max": 3,
    "adaptive": False,
    "deferred_admission": False,
}


class TinyGemma4(nn.Module):
    """mlx-vlm's Gemma 4 text model, as the adapter's wrapper serves it."""

    def __init__(self, seed=2, **extra):
        super().__init__()
        from mlx_vlm.models.gemma4.config import TextConfig
        from mlx_vlm.models.gemma4.language import LanguageModel

        config = TextConfig(**{**TEXT, **extra})
        mx.random.seed(seed)
        self.language_model = LanguageModel(config)
        self.config = SimpleNamespace(text_config=config)
        mx.eval(self.parameters())

    def __call__(self, inputs, cache=None, **kwargs):
        return self.language_model(inputs, cache=cache, **kwargs)


@pytest.fixture(scope="module", params=["dense", "sparse_moe"])
def gemma(request):
    extra = SPARSE if request.param == "sparse_moe" else {}
    return _Gemma4LogitsModel(TinyGemma4(**extra))


def _greedy(model, prompt, tokens):
    cache = model.make_cache()
    logits = model(mx.array([prompt]), cache=cache)
    produced = []
    for _ in range(tokens):
        token = int(mx.argmax(logits[0, -1]).item())
        produced.append(token)
        logits = model(mx.array([[token]]), cache=cache)
    return produced


def _run(model, prompts, policy, tokens=24):
    generator = PromptLookupBatchGenerator(
        model,
        completion_batch_size=len(prompts),
        prefill_step_size=5,
        prompt_lookup=policy,
    )
    uids = generator.insert(prompts, max_tokens=[tokens] * len(prompts))
    out, finals = {uid: [] for uid in uids}, {}
    for _ in range(4000):
        _prompts, responses = generator.next()
        for response in responses:
            out[response.uid].append(response.token)
            if response.finish_reason:
                finals[response.uid] = response
        if len(finals) == len(uids):
            break
    generator.close()
    return [out[uid] for uid in uids], [finals[uid] for uid in uids], generator


def test_the_wrapper_owns_its_prompt_lookup_verify_forward(gemma):
    assert callable(gemma.pld_verify_forward)
    assert gemma.pld_verify_forward_kind == "gemma4_segmented_row_attention"
    # The generator reads both declarations off the wrapper, not mlx-vlm.
    assert getattr(gemma, "pld_verify_max_lanes", None) is None


def test_batched_verify_over_wrapped_rings_equals_ordinary_greedy(gemma):
    produced, finals, generator = _run(gemma, PROMPTS, EXPLORE)
    for prompt, tokens in zip(PROMPTS, produced):
        assert tokens == _greedy(gemma, prompt, 24)
    stats = generator.scheduler_stats
    # Shared segmented forwards, proposals, rejected tails: the crash path.
    assert stats["pld_batched_rounds"] > 0
    assert stats["pld_batched_max_width"] == len(PROMPTS)
    assert stats["pld_proposed"] > 0 and stats["pld_rollbacks"] > 0
    assert stats["pld_model_verify_forwards"] > 0
    for final in finals:
        live = final.prompt_cache
        assert {type(c) for c in live} == {KVCache, RotatingKVCache}
        assert all(int(c.offset) == len(final.all_tokens) for c in live)
        assert not any(getattr(c, "speculating", False) for c in live)


def test_per_lane_snapshot_rounds_equal_ordinary_greedy(gemma):
    # batched_verify off: each lane verifies on its own caches through the
    # snapshot/rewind/replay driver, wrapped rings included.
    produced, _finals, generator = _run(
        gemma, PROMPTS[:2], {**EXPLORE, "batched_verify": False}
    )
    for prompt, tokens in zip(PROMPTS, produced):
        assert tokens == _greedy(gemma, prompt, 24)
    assert generator.scheduler_stats["pld_batched_rounds"] == 0
    assert generator.scheduler_stats["pld_rollbacks"] > 0


def test_default_latched_policy_equals_ordinary_greedy(gemma):
    from mlx2.server import DEFAULT_PROMPT_LOOKUP_POLICY

    # A copy-heavy prompt: the model's own continuation of a prefix, then the
    # prefix again, so the latch's shadow probes match and it explores.
    proposed = 0
    for head in PROMPTS:
        prompt = head + _greedy(gemma, head, 30) + head
        produced, _finals, generator = _run(
            gemma, [prompt], dict(DEFAULT_PROMPT_LOOKUP_POLICY), tokens=40
        )
        assert produced[0] == _greedy(gemma, prompt, 40)
        assert generator.verify_forward == "gemma4_segmented_row_attention"
        proposed += generator.scheduler_stats["pld_proposed"]
    assert proposed > 0


def test_published_cache_continues_like_a_cold_prefill(gemma):
    _produced, finals, _generator = _run(gemma, PROMPTS[:1], EXPLORE, tokens=12)
    final = finals[0]
    follow = [4, 5, 6]
    logits = gemma(mx.array([follow]), cache=final.prompt_cache)
    reference = gemma(mx.array([final.all_tokens + follow]), cache=gemma.make_cache())
    assert int(mx.argmax(logits[0, -1]).item()) == int(
        mx.argmax(reference[0, -1]).item()
    )


def test_ordinary_forwards_keep_mlx_vlm_attention(gemma):
    # The installed router is exact outside segmented views: a plain cache
    # forward after a verify gives bit-identical logits to before it.
    prompt = mx.array([PROMPTS[2]])
    before = gemma(prompt, cache=gemma.make_cache())
    _run(gemma, PROMPTS[:1], EXPLORE, tokens=6)
    after = gemma(prompt, cache=gemma.make_cache())
    assert mx.array_equal(before, after).item()


def test_kv_shared_layers_fail_closed_on_a_segmented_verify():
    from mlx2.runtime.segmented_rotating_kv import SegmentedKVRows

    model = _Gemma4LogitsModel(TinyGemma4(num_kv_shared_layers=3))
    cache = model.make_cache()
    model(mx.array([PROMPTS[1]]), cache=cache)
    transaction = SegmentedKVRows([cache]).begin(lengths=[2])
    with pytest.raises(ValueError, match="KV-shared"):
        model.pld_verify_forward(mx.array([[7, 8]]), cache=transaction.caches)
    transaction.abort()


@pytest.mark.parametrize(
    ("adapter_type", "descriptor"),
    [(Gemma4A4BAdapter, GEMMA4_A4B), (Gemma431BAdapter, GEMMA4_31B)],
)
def test_gemma4_defaults_to_latched_prompt_lookup_and_keeps_ordinary(
    adapter_type, descriptor
):
    from mlx2.adapters.registry import AdapterResolution
    from mlx2.server import (
        DEFAULT_PROMPT_LOOKUP_POLICY,
        build_parser,
        resolve_execution_policy_defaults,
        resolve_route_selection,
    )

    assert Capability.PROMPT_LOOKUP in descriptor.capabilities
    resolution = AdapterResolution(adapter_type, descriptor, {"fingerprint": "fixture"})
    assert resolution.default_route == "ordinary"
    parser = build_parser()
    selected = resolve_route_selection(
        parser.parse_args(["--model", "fixture"]), None, resolution
    )
    assert (selected.route, selected.source) == ("prompt_lookup", "capability_default")
    assert (
        resolve_execution_policy_defaults(None, selected, resolution)["prompt_lookup"]
        == DEFAULT_PROMPT_LOOKUP_POLICY
    )
    selected = resolve_route_selection(
        parser.parse_args(["--model", "fixture", "--ordinary"]), None, resolution
    )
    assert (selected.route, selected.source) == ("ordinary", "explicit_flag")
    # Media stays on the ordinary route; the PLD route refuses it up front.
    assert adapter_type.multimodal_routes == frozenset({"ordinary"})
    contract = object.__new__(adapter_type).route_capability_contract()
    assert contract["prompt_lookup"]["implemented"] is True
    assert contract["prompt_lookup"]["media_routes"] == ("ordinary",)
    assert contract["prompt_lookup"]["qualified"] is False
