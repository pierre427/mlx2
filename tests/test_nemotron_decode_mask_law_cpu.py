"""Nemotron one-row attention follows the ordinary decode mask law.

Ordinary single-token decode on ``BatchKVCache`` hands SDPA an all-true
``(B, 1, 1, keys)`` boolean mask under ``MLX2_DECODE_MASK=array`` (the
default) and no mask under ``none``.  MTP verify rows and prompt-tail tokens
reach ``NemotronHAttention`` unmasked on a plain/segmented KV cache; MLX
0.32.4 gives other bits without the mask from 8192 keys on, so they must
take the same law (``test_nemotron_decode_mask_law_metal.py`` pins the bits).
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import nemotron_h
from mlx2.runtime.models.cache import BatchKVCache, KVCache
from mlx2.runtime.models.nemotron_h import ModelArgs, NemotronHAttention


def _attention(head_dim=8):
    mx.random.seed(3)
    args = ModelArgs.from_dict({
        "model_type": "nemotron_h", "vocab_size": 16, "hidden_size": 32,
        "intermediate_size": 32, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": head_dim, "attention_bias": False, "mamba_num_heads": 2,
        "mamba_head_dim": 8, "ssm_state_size": 32, "conv_kernel": 4, "n_groups": 1,
        "layers_block_type": ["attention"], "num_hidden_layers": 1,
        "layer_norm_epsilon": 1e-5, "max_position_embeddings": 64,
        "mamba_proj_bias": False, "mlp_bias": False, "use_bias": False,
        "use_conv_bias": True,
    })
    layer = NemotronHAttention(args)
    mx.eval(layer.parameters())
    return layer


@pytest.fixture
def masks(monkeypatch):
    seen = []
    original = nemotron_h.scaled_dot_product_attention

    def record(queries, keys, values, *, cache, scale, mask):
        packed = keys[0] if isinstance(keys, tuple) else keys
        seen.append((int(queries.shape[2]), int(packed.shape[2]), mask))
        return original(queries, keys, values, cache=cache, scale=scale, mask=mask)

    monkeypatch.setattr(nemotron_h, "scaled_dot_product_attention", record)
    return seen


def _prefilled(cache, layer, rows=5):
    mx.eval(layer(mx.random.normal((1, rows, 32)), mask="causal", cache=cache))


@pytest.mark.parametrize("law", ["array", None])
def test_unmasked_one_row_takes_the_batch_decode_mask(monkeypatch, masks, law):
    if law is None:
        monkeypatch.delenv("MLX2_DECODE_MASK", raising=False)
    else:
        monkeypatch.setenv("MLX2_DECODE_MASK", law)
    layer = _attention()
    plain = KVCache()
    _prefilled(plain, layer)
    batch = BatchKVCache([0])
    _prefilled(batch, layer)
    masks.clear()
    reference = batch.make_mask(1)
    x = mx.random.normal((1, 1, 32))
    mx.eval(layer(x, mask=None, cache=plain))
    rows, keys, mask = masks[-1]
    assert (rows, keys) == (1, 6)
    assert isinstance(mask, mx.array) and mask.dtype == mx.bool_
    assert mask.shape == reference.shape == (1, 1, 1, 6)
    assert np.array_equal(np.asarray(mask), np.asarray(reference))


def test_no_mask_law_keeps_the_one_row_unmasked(monkeypatch, masks):
    monkeypatch.setenv("MLX2_DECODE_MASK", "none")
    layer = _attention()
    plain = KVCache()
    _prefilled(plain, layer)
    batch = BatchKVCache([0])
    _prefilled(batch, layer)
    masks.clear()
    assert batch.make_mask(1) is None
    mx.eval(layer(mx.random.normal((1, 1, 32)), mask=None, cache=plain))
    assert masks[-1][2] is None


def test_multi_row_and_explicit_masks_are_untouched(monkeypatch, masks):
    monkeypatch.delenv("MLX2_DECODE_MASK", raising=False)
    layer = _attention()
    plain = KVCache()
    _prefilled(plain, layer)
    assert masks[-1][2] == "causal"
    explicit = mx.array([[[[True] * 5 + [False]]]])
    mx.eval(layer(mx.random.normal((1, 1, 32)), mask=explicit, cache=plain))
    assert masks[-1][2] is explicit


def test_quantized_one_row_keeps_its_own_law(monkeypatch, masks):
    """Packed keys come back as tuples: no crash, and the quantized decode
    kernel's no-mask admission is left as it was."""
    from mlx2.runtime.models.cache import QuantizedKVCache

    monkeypatch.delenv("MLX2_DECODE_MASK", raising=False)
    layer = _attention(head_dim=32)
    cache = QuantizedKVCache(group_size=32, bits=8)
    _prefilled(cache, layer)
    masks.clear()
    mx.eval(layer(mx.random.normal((1, 1, 32)), mask=None, cache=cache))
    assert masks[-1][2] is None


def _padded_batch(layer):
    import copy

    cache = BatchKVCache([2, 0])
    prompt = mx.random.normal((2, 5, 32), key=mx.random.key(9))
    mx.eval(layer(prompt, mask=cache.make_mask(5, return_array=True), cache=cache))
    return cache, copy.deepcopy(cache)


def test_padded_batch_one_row_equals_the_ordinary_masked_call(monkeypatch):
    """B>1 left padding: the derived mask is the cache's own decode mask."""
    monkeypatch.delenv("MLX2_DECODE_MASK", raising=False)
    layer = _attention()
    reference, candidate = _padded_batch(layer)
    x = mx.random.normal((2, 1, 32), key=mx.random.key(10))
    want = layer(x, mask=reference.make_mask(1), cache=reference)
    have = layer(x, mask=None, cache=candidate)
    mx.eval(want, have)
    assert np.array_equal(
        np.asarray(have).view(np.uint32), np.asarray(want).view(np.uint32)
    )


class _ProbeStub:
    def __init__(self):
        self.masks = []

    def attend(self, ordinal, queries, keys, values, scale, mask=None, sinks=None):
        self.masks.append(mask)
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask
        )


def test_view_caches_keep_the_mask_they_were_given(monkeypatch, masks):
    from mlx2.runtime.quantized_key_index import QuantizedKeyIndexCache
    from mlx2.runtime.topk_index_reuse import ReuseProbeKVCache

    monkeypatch.delenv("MLX2_DECODE_MASK", raising=False)
    layer = _attention()
    probe = _ProbeStub()
    for cache in (ReuseProbeKVCache(probe, 0), QuantizedKeyIndexCache()):
        _prefilled(cache, layer)
        masks.clear()
        probe.masks.clear()
        mx.eval(layer(mx.random.normal((1, 1, 32)), mask=None, cache=cache))
        assert masks[-1][2] is None, type(cache).__name__
        if isinstance(cache, ReuseProbeKVCache):
            assert probe.masks == [None]


def test_one_mask_per_key_count_is_shared_across_layers(monkeypatch, masks):
    monkeypatch.delenv("MLX2_DECODE_MASK", raising=False)
    first, second = _attention(), _attention()
    caches = [KVCache(), KVCache()]
    for layer, cache in zip((first, second), caches):
        _prefilled(cache, layer)
    masks.clear()
    x = mx.random.normal((1, 1, 32))
    mx.eval(first(x, mask=None, cache=caches[0]), second(x, mask=None, cache=caches[1]))
    assert len(masks) == 2 and masks[0][2] is masks[1][2]
    mx.eval(first(x, mask=None, cache=caches[0]))
    assert masks[-1][2].shape == (1, 1, 1, 7) and masks[-1][2] is not masks[0][2]


def test_mask_memo_survives_an_eviction_between_lookup_and_refresh(monkeypatch):
    """Codex round 9: another engine thread may evict a key between this
    thread's lookup and its LRU refresh; that must not raise mid-decode."""
    from collections import OrderedDict

    from mlx2.runtime.models import nemotron_h

    class EvictedAfterRead(OrderedDict):
        def get(self, key, default=None):
            value = super().get(key, default)
            if key in self:
                del self[key]  # the other thread's eviction lands here
            return value

    memo = EvictedAfterRead()
    monkeypatch.setattr(nemotron_h, "_DECODE_MASK_MEMO", memo)
    first = nemotron_h._all_true_decode_mask(1, 64)
    again = nemotron_h._all_true_decode_mask(1, 64)
    assert again.shape == first.shape == (1, 1, 1, 64)
    assert bool(again.all().item())


def test_mask_memo_is_bounded_and_consistent_under_threads(monkeypatch):
    import sys
    import threading
    from collections import OrderedDict

    from mlx2.runtime.models import nemotron_h

    monkeypatch.setattr(nemotron_h, "_DECODE_MASK_MEMO", OrderedDict())
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    errors = []

    def hammer(seed):
        try:
            for i in range(400):
                keys = 8 + (seed * 7 + i) % 40
                mask = nemotron_h._all_true_decode_mask(1, keys)
                assert mask.shape == (1, 1, 1, keys)
        except BaseException as error:  # noqa: BLE001 - surfaced below
            errors.append(error)

    try:
        threads = [threading.Thread(target=hammer, args=(seed,)) for seed in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        sys.setswitchinterval(previous)
    assert not errors, errors[:1]
    assert len(nemotron_h._DECODE_MASK_MEMO) <= nemotron_h._DECODE_MASK_MEMO_ENTRIES
