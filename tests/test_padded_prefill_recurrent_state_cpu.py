"""Right-padded batched prefill leaves every recurrent row exactly as if alone.

``PromptProcessingBatch.prompt`` right-pads ragged prompts and calls
``prepare(lengths=..., right_padding=...)`` on every cache plane.  The merged
recurrent cache of fresh rows carries ``left_padding == 0`` beside those
lengths, which is the layout the pinned mlx-vlm Qwen SSM mask helper turns
into "no mask" (jundot/omlx#4526): padded tokens then advance the GDN state
and the conv tail.  mlx2 builds its own SSM mask (``ArraysCache.make_mask``
intersects both bounds), so for every native recurrent family each row's
conv state, recurrent state and next-token logits must match running that
row alone.  The planted-defect test proves the comparison would see a leak.
"""

import mlx.core as mx
import pytest

from mlx2.runtime.generate import PromptProcessingBatch
from mlx2.runtime.models.cache import ArraysCache

# Batched vs single-row float32 rounding is ~5e-6 here; the mlx-vlm mask
# shape planted below moves the same quantities by 7e-4 to 0.85.
TOL = 5e-5
CASES = (([5, 9], 16), ([9, 4, 7], 3))


def _qwen38():
    import route_harness as rh

    model, vocab = rh.tiny_qwen38_mtp()
    return model, vocab


def _qwen36():
    from mlx2.runtime.models.qwen36_35b import Model, ModelArgs
    from test_qwen36_35b_port import TINY_MTP_MOE

    mx.random.seed(5)
    model = Model(ModelArgs.from_dict(
        {"model_type": "qwen3_5_moe", "text_config": TINY_MTP_MOE}
    ))
    return model, 128


def _qwen4():
    from test_batched_mtp import _tiny_qwen4_model

    mx.random.seed(41)
    return _tiny_qwen4_model(), 64


def _nemotron():
    from test_nemotron_external_taps_cpu import tiny_model

    return tiny_model(), 16


def _agnes():
    from mlx2.runtime.models.agnes import (
        LAYER_DELTA, LAYER_GLOBAL, Model, ModelArgs, TextModelArgs,
    )

    args = TextModelArgs(
        hidden_size=32, num_hidden_layers=4, intermediate_size=32,
        vocab_size=64, layer_types=[LAYER_DELTA] * 3 + [LAYER_GLOBAL],
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        rope_parameters={"type": "default", "partial_rotary_factor": 0.25},
    )
    mx.random.seed(9)
    return Model(ModelArgs(model_type="agnes", text_config=args.__dict__)), 64


FAMILIES = {
    "qwen38_27b": _qwen38,
    "qwen36_35b": _qwen36,
    "qwen4_exp": _qwen4,
    "nemotron_h": _nemotron,
    "agnes": _agnes,
}


@pytest.fixture(scope="module", params=sorted(FAMILIES))
def family(request):
    model, vocab = FAMILIES[request.param]()
    model.eval()
    mx.eval(model.parameters())
    return request.param, model, vocab


def _ids(n, salt, vocab):
    return [(i * 7 + salt) % (vocab - 1) + 1 for i in range(n)]


def _diff(a, b):
    return float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max())


def _worst_row_error(model, vocab, lengths, step):
    """Largest per-row deviation of logits and recurrent slots from B=1."""
    prompts = [_ids(n, 3 * j + 1, vocab) for j, n in enumerate(lengths)]
    batch = PromptProcessingBatch(
        model, list(range(len(prompts))), [model.make_cache() for _ in prompts],
        prefill_step_size=step,
    )
    batch.prompt([p[:-1] for p in prompts])
    batched = model(mx.array([[p[-1]] for p in prompts]), cache=batch.prompt_cache)
    mx.eval(batched)
    worst = {}
    for row, prompt in enumerate(prompts):
        cache = model.make_cache()
        for start in range(0, len(prompt) - 1, step):
            model(mx.array([prompt[start:min(start + step, len(prompt) - 1)]]),
                  cache=cache)
        alone = model(mx.array([[prompt[-1]]]), cache=cache)
        mx.eval(alone)
        worst["logits"] = max(worst.get("logits", 0.0),
                              _diff(batched[row:row + 1], alone))
        for joined, single in zip(batch.prompt_cache, cache):
            if not isinstance(joined, ArraysCache):
                continue
            for slot, (a, b) in enumerate(zip(joined.cache, single.cache)):
                if a is None or b is None:
                    continue
                key = f"recurrent[{slot}]"
                worst[key] = max(worst.get(key, 0.0), _diff(a[row:row + 1], b))
    assert any(k.startswith("recurrent") for k in worst), "no recurrent plane compared"
    return worst


@pytest.mark.parametrize("lengths,step", CASES)
def test_padded_rows_match_running_alone(family, lengths, step):
    name, model, vocab = family
    worst = _worst_row_error(model, vocab, lengths, step)
    assert max(worst.values()) < TOL, (name, worst)


def _mlx_vlm_pinned_make_mask(self, N):
    """The pinned mlx-vlm ``ArraysCache.make_mask``: one bound, never both."""
    pos = mx.arange(N)
    if self.left_padding is not None:
        return pos >= self.left_padding[:, None]
    if self.lengths is not None:
        return pos < self.lengths[:, None]
    return None


def test_the_comparison_sees_the_mlx_vlm_mask_leak(family, monkeypatch):
    """Falsifier: the omlx#4526 mask shape must fail the test above."""
    name, model, vocab = family
    monkeypatch.setattr(ArraysCache, "make_mask", _mlx_vlm_pinned_make_mask)
    worst = _worst_row_error(model, vocab, [5, 9], 16)
    assert max(worst.values()) > 10 * TOL, (name, worst)


def test_merged_fresh_rows_keep_the_right_bound():
    """The layout the batch generator builds: left padding 0, lengths set."""
    cache = ArraysCache.merge([ArraysCache(2), ArraysCache(2)])
    cache.prepare(lengths=[4, 8])
    assert cache.make_mask(8).tolist() == [[True] * 4 + [False] * 4, [True] * 8]


def test_both_bounds_intersect():
    cache = ArraysCache(2, left_padding=[0, 2])
    cache.lengths = mx.array([5, 8])
    assert cache.make_mask(8).tolist() == [
        [True] * 5 + [False] * 3,
        [False] * 2 + [True] * 6,
    ]
