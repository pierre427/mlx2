"""Gemma 3n prompt-chunk row pruning before the KV-shared layers (opt-in).

A tiny random Gemma 3n text model runs the same prompt through the ordinary
mlx-vlm forward and the pruned forward, chunk by chunk, inside the checkpoint
requests the prefill loop opens.  Every cache byte and every later decode
logit must match exactly; the claimed checkpoint row must match the ordinary
last row up to matmul rounding.  Ineligible calls fall back to ordinary.
"""

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.adapters import gemma3n_prune  # noqa: E402
from mlx2.adapters.mlx_vlm import (  # noqa: E402
    _Gemma3nLogitsModel,
    _gemma3n_policy_options,
)

# Importing the mlx-vlm gemma3n package patches the process-wide
# transformers.AutoProcessor.from_pretrained (install_auto_processor_patch);
# restore it so this module leaves later real-processor tests
# (test_gemma3n_prompt_bos_cpu) the state they had.
transformers = pytest.importorskip("transformers")
_auto_from_pretrained = transformers.AutoProcessor.__dict__["from_pretrained"]
language = pytest.importorskip("mlx_vlm.models.gemma3n.language")
transformers.AutoProcessor.from_pretrained = _auto_from_pretrained
from mlx_vlm.models.gemma3n.config import TextConfig  # noqa: E402
from mlx2.runtime.models.cache import BatchKVCache, BatchRotatingKVCache  # noqa: E402
from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows  # noqa: E402

WINDOW = 8
LAYER_TYPES = ["sliding_attention", "sliding_attention", "full_attention"] * 2


def _text_model():
    config = TextConfig(
        model_type="gemma3n_text", hidden_size=32, num_hidden_layers=6,
        intermediate_size=[64] * 6, num_attention_heads=2, head_dim=16,
        num_key_value_heads=1, vocab_size=97, vocab_size_per_layer_input=97,
        laurel_rank=4, hidden_size_per_layer_input=8, sliding_window=WINDOW,
        num_kv_shared_layers=3, layer_types=list(LAYER_TYPES),
        activation_sparsity_pattern=[0.0] * 6, final_logit_softcapping=30.0,
    )
    mx.random.seed(7)
    model = language.LanguageModel(config)
    mx.eval(model.parameters())
    return model


class _Outer:
    """The text-only slice of the mlx-vlm Gemma3n ``Model`` the wrapper uses."""

    def __init__(self, language_model):
        self.language_model = language_model
        self.config = language_model.config

    def get_input_embeddings(self, input_ids, **kwargs):
        model = self.language_model.model
        embeds = model.embed_tokens(input_ids)
        per_layer = model.get_per_layer_inputs(
            mx.where(input_ids < model.vocab_size_per_layer_input, input_ids, 0)
        )

        class Features:
            inputs_embeds = embeds
            per_layer_inputs = per_layer

        return Features


def _caches(batch):
    return [
        BatchRotatingKVCache(WINDOW, [0] * batch) if kind == "sliding_attention"
        else BatchKVCache([0] * batch)
        for kind in LAYER_TYPES[:3]
    ]


def _states(cache):
    out = []
    for layer in cache:
        for array in layer.state:
            if isinstance(array, mx.array):
                out.append(np.array(array.astype(mx.float32)))
    return out


@pytest.mark.parametrize("batch", [1, 2])
def test_pruned_prefill_keeps_every_cache_byte_and_decode_logit(batch):
    language_model = _text_model()
    ordinary = _Gemma3nLogitsModel(_Outer(language_model))
    counters = gemma3n_prune.PruneCounters()
    pruned = _Gemma3nLogitsModel(_Outer(language_model), kv_shared_prune=counters)
    mx.random.seed(1)
    prompt = mx.random.randint(1, 97, (batch, 23))
    a, b = _caches(batch), _caches(batch)
    # Chunks wider than the sliding window and a short tail chunk.
    for start, stop in ((0, 10), (10, 20), (20, 22)):
        chunk = prompt[:, start:stop]
        width = stop - start
        with prefill_checkpoint_rows([width] * batch, width=width) as request_a:
            out_a = ordinary.prefill_forward(chunk, cache=a)
        with prefill_checkpoint_rows([width] * batch, width=width) as request_b:
            out_b = pruned.prefill_forward(chunk, cache=b)
        assert request_a.applied == 0 and request_b.applied == 1
        assert out_a.shape == (batch, width, 97) and out_b.shape == (batch, 1, 97)
        assert np.allclose(np.array(out_a[:, -1]), np.array(out_b[:, 0]), atol=1e-4)
        for left, right in zip(_states(a), _states(b)):
            assert np.array_equal(left, right)
    # The prompt's final token and three greedy steps take the decode path.
    token = prompt[:, 22:23]
    for step in range(4):
        forward_a = ordinary.prefill_forward if step == 0 else ordinary
        forward_b = pruned.prefill_forward if step == 0 else pruned
        logits_a = forward_a(token, cache=a)
        logits_b = forward_b(token, cache=b)
        assert np.array_equal(np.array(logits_a), np.array(logits_b))
        token = mx.argmax(logits_a[:, -1:], axis=-1)
    assert counters.pruned_forwards == 3
    assert counters.rows_in == 22 * batch and counters.rows_kept == 3 * batch
    # The one-row prompt tail is not eligible and runs the ordinary forward.
    assert counters.ordinary_forwards == 1


def test_ineligible_calls_run_the_ordinary_forward():
    language_model = _text_model()
    counters = gemma3n_prune.PruneCounters()
    pruned = _Gemma3nLogitsModel(_Outer(language_model), kv_shared_prune=counters)
    prompt = mx.random.randint(1, 97, (2, 6))
    # No checkpoint request open.
    assert pruned.prefill_forward(prompt, cache=_caches(2)).shape == (2, 6, 97)
    # A ragged chunk: lane 1 ends before the last row.
    with prefill_checkpoint_rows([6, 4], width=6) as request:
        out = pruned.prefill_forward(prompt, cache=_caches(2))
    assert request.applied == 0 and out.shape == (2, 6, 97)
    assert counters.pruned_forwards == 0 and counters.ordinary_forwards == 2


def test_mirrored_source_guard():
    language_model = _text_model()
    gemma3n_prune.check_mirrored_source(language_model)
    original = dict(gemma3n_prune.MIRRORED_SOURCES)
    try:
        gemma3n_prune.MIRRORED_SOURCES["Gemma3Model.__call__"] = "0" * 64
        with pytest.raises(ValueError, match="refusing"):
            gemma3n_prune.check_mirrored_source(language_model)
    finally:
        gemma3n_prune.MIRRORED_SOURCES.update(original)


def test_opt_in_route_identity_extends_the_video_policy():
    from mlx2.adapters.mlx_vlm import Gemma3nAdapterPolicy
    from mlx2.adapters.multimodal import Gemma3nVideoPolicy

    video = Gemma3nVideoPolicy()
    policy = Gemma3nAdapterPolicy(video)
    assert policy.as_dict() == {**video.as_dict(), "kv_shared_row_pruning": True}


def test_policy_option_parsing():
    # Default on (Pierre, 2026-10-31); false selects the ordinary reference.
    assert _gemma3n_policy_options(None) is True
    assert _gemma3n_policy_options({}) is True
    assert _gemma3n_policy_options({"gemma3n_prefill": {}}) is True
    assert _gemma3n_policy_options({"gemma3n_prefill": {"kv_shared_row_pruning": False}}) is False
    for bad in ({"speculative": {}}, {"gemma3n_prefill": {"other": 1}},
                {"gemma3n_prefill": {"kv_shared_row_pruning": "yes"}}, []):
        with pytest.raises(ValueError):
            _gemma3n_policy_options(bad)
