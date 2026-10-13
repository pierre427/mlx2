"""The prefill checkpoint guard projects one row per lane, not the whole chunk.

Sweep 2026-10-12 L2-02: the guard indexed the last real row out of the
forward's [batch, sequence, vocabulary] logits.  MLX has no slice push-down,
so every ordinary prefill chunk evaluated the full sequence x vocabulary head
projection that prompt processing otherwise discards.
"""

import inspect

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.runtime.generate import PromptChunkCorruption, PromptProcessingBatch
from mlx2.runtime.models.cache import KVCache
from mlx2.runtime.models.standard_decoder import Model, ModelArgs


class _RecordingHead(nn.Linear):
    def __call__(self, x):
        self.seen.append(tuple(x.shape))
        return super().__call__(x)


def _tiny(vocab=9, hidden=8):
    mx.random.seed(13)
    model = Model(
        ModelArgs(
            model_type="qwen3",
            hidden_size=hidden,
            num_hidden_layers=2,
            intermediate_size=12,
            num_attention_heads=2,
            rms_norm_eps=1e-6,
            vocab_size=vocab,
            num_key_value_heads=1,
            head_dim=hidden // 2,
            max_position_embeddings=4096,
            rope_theta=10000,
            tie_word_embeddings=False,
        )
    )
    head = _RecordingHead(hidden, vocab, bias=False)
    head.seen = []
    model.lm_head = head
    mx.eval(model.parameters())
    return model


def _batch(model, uids, step):
    batch = object.__new__(PromptProcessingBatch)
    batch.uids = list(uids)
    batch.tokens = [[] for _ in uids]
    batch.prompt_cache = [KVCache() for _ in model.layers]
    batch.prefill_step_size = step
    batch.model = model
    batch.decode_active = False
    return batch


def test_runtime_model_prefill_projects_only_the_checkpoint_row():
    model = _tiny()
    _batch(model, [5], step=4).prompt([[1, 2, 3, 4, 5, 6, 7, 8, 1, 2]])

    # Three chunks (4, 4, 2): one head row each, never the whole chunk.
    assert model.lm_head.seen == [(1, 1, 8), (1, 1, 8), (1, 1, 8)]


def test_runtime_model_prefill_peak_memory_does_not_scale_with_vocabulary():
    vocab, hidden, step = 32768, 64, 512
    model = _tiny(vocab=vocab, hidden=hidden)
    batch = _batch(model, [9], step=step)
    mx.clear_cache()
    base = mx.get_active_memory()
    mx.reset_peak_memory()
    batch.prompt([list(range(1, 2 * step + 1))])
    rise = mx.get_peak_memory() - base

    chunk_logits = step * vocab * 4  # one chunk of fp32 logits: 64 MiB
    assert rise < chunk_logits // 4, (rise, chunk_logits)


class _NarrowedModel:
    """Test double that honours the checkpoint-row request like a runtime model."""

    def __call__(self, tokens, *, cache):
        from mlx2.runtime.prefill_checkpoint_rows import select_checkpoint_rows

        hidden = tokens.astype(mx.float32)[..., None] * mx.ones((1, 1, 3))
        hidden = select_checkpoint_rows(hidden)
        return mx.where(hidden == 99.0, mx.array(float("nan")), hidden)


@pytest.mark.parametrize("bad_lane", [0, 1])
def test_narrowed_projection_still_refuses_the_corrupt_lane_only(bad_lane):
    class Recorder:
        state = ()

        def state_checkpoint(self, positions, force=False):
            return None

        def prepare(self, **_kwargs):
            return None

        def finalize(self):
            return None

    prompts = [[1, 2, 3, 4], [5, 6, 7]]
    prompts[bad_lane][-1] = 99  # the last real row of the bad lane only
    batch = object.__new__(PromptProcessingBatch)
    batch.uids = [10, 11]
    batch.tokens = [[], []]
    batch.prompt_cache = [Recorder()]
    batch.prefill_step_size = 8
    batch.model = _NarrowedModel()
    batch.decode_active = False

    with pytest.raises(PromptChunkCorruption) as raised:
        batch.prompt(prompts)
    assert raised.value.uids == (10 + bad_lane,)


def test_poison_before_the_last_row_is_outside_the_narrowed_row():
    """Right padding (lane 1's row 3) and earlier rows are never selected."""

    class Recorder:
        state = ()

        def state_checkpoint(self, positions, force=False):
            return None

        def prepare(self, **_kwargs):
            return None

        def finalize(self):
            return None

    batch = object.__new__(PromptProcessingBatch)
    batch.uids = [10, 11]
    batch.tokens = [[], []]
    batch.prompt_cache = [Recorder()]
    batch.prefill_step_size = 8
    batch.model = _NarrowedModel()
    batch.decode_active = False
    batch.prompt([[99, 2, 3, 4], [5, 6, 7]])


@pytest.mark.parametrize(
    "module, cls",
    [
        ("agnes", "TextModel"),
        ("gpt_oss", "Model"),
        ("gpt_oss_puzzle", "Model"),
        ("granitemoe_swa", "Model"),
        ("hy_v3", "Model"),
        ("laguna", "Model"),
        ("mellum", "Model"),
        ("muse_glimmer", "Model"),
        ("nemotron_h", "Model"),
        ("olmo_hils", "Model"),
        ("qwen38_27b", "TextModel"),
        ("qwen4_exp", "TextModel"),
        ("standard_decoder", "Model"),
        ("xing4_0", "Model"),
    ],
)
def test_runtime_model_heads_honour_the_checkpoint_row_request(module, cls):
    import importlib

    source = inspect.getsource(
        getattr(importlib.import_module(f"mlx2.runtime.models.{module}"), cls).__call__
    )
    assert "select_checkpoint_rows(" in source
