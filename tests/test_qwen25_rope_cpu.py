"""Qwen2.5-VL M-RoPE cache lifecycle without a real MLX import."""

import importlib.abc
import copy
import sys
import types

import numpy as np
import pytest

from mlx2.adapters.qwen25_rope import (
    Qwen25RoPECache, RequestPrivateQwen25Model, _batch_mask_attention,
)


class _BlockMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise AssertionError("real MLX import in CPU-only Qwen2.5 test")
        return None


@pytest.fixture(autouse=True)
def block_real_mlx(monkeypatch):
    assert "mlx.core" not in sys.modules
    monkeypatch.setattr(sys, "meta_path", [_BlockMLX(), *sys.meta_path])


@pytest.fixture
def fake_mx(monkeypatch):
    module = types.ModuleType("mlx")
    core = types.ModuleType("mlx.core")
    for name in ("array", "arange", "broadcast_to"):
        setattr(core, name, getattr(np, name))
    core.int32 = np.int32
    module.core = core
    monkeypatch.setitem(sys.modules, "mlx", module)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    cache = types.ModuleType("mlx2.runtime.models.cache")
    cache.BatchKVCache = _FakeKVCache
    monkeypatch.setitem(sys.modules, "mlx2.runtime.models.cache", cache)
    return core


class _FakeLanguage:
    def __init__(self, seen):
        self._rope_deltas = "stale-other-request"
        self._position_ids = "stale-other-request"
        self.model = _FakeDecoder(seen)
        self.args = types.SimpleNamespace(tie_word_embeddings=False)
        self.lm_head = lambda hidden: hidden

    def make_cache(self):
        return [_FakeKVCache()]

    def get_rope_index(self, ids, image_grid, video_grid, mask):
        assert image_grid == "grid"
        return np.array([[[0, 1, 2, 3]], [[0, 1, 1, 2]], [[0, 0, 1, 2]]]), np.array([[-1]])


class _FakePinnedModel:
    def __init__(self):
        self.seen = []
        self.language_model = _FakeLanguage(self.seen)

    def __call__(self, ids, *, cache, position_ids, **kwargs):
        # This models the pinned top-level __call__ forwarding **kwargs into
        # language_model. The computed InputEmbeddingsFeatures fields are not
        # forwarded, so the explicit position_ids argument is essential.
        assert len(cache) == 1
        self.seen.append(position_ids.copy())
        return types.SimpleNamespace(logits=np.array([1]))


class _FakeKVCache:
    _idx = 1
    left_padding = np.zeros(2, dtype=np.int32)

    def make_mask(self, width, return_array=False):
        return np.ones((2, 1, width, self._idx + width), dtype=bool)


class _FakeDecoder:
    def __init__(self, seen):
        self.layers = []
        self.seen = seen

    def __call__(self, ids, *, cache, position_ids, mask):
        assert mask.shape == (2, 1, ids.shape[1], 1 + ids.shape[1])
        self.seen.append(position_ids.copy())
        return np.array([1])


def test_batch_mask_shape_refuses_before_cache_mutation(fake_mx):
    pinned = _FakePinnedModel()
    model = RequestPrivateQwen25Model(pinned)
    rope = Qwen25RoPECache.merge([Qwen25RoPECache(), Qwen25RoPECache()])

    class WrongMask(_FakeKVCache):
        def make_mask(self, width, return_array=False):
            return np.ones((2, 1, width, 1), dtype=bool)

    with pytest.raises(ValueError, match="mask/cache length mismatch"):
        model(np.array([[1], [2]]), cache=[WrongMask(), rope])
    assert [row["length"] for row in rope.rows] == [0, 0]
    assert pinned.seen == []

    class QuantizedBatchCache(_FakeKVCache):
        bits = 8

    with pytest.raises(ValueError, match="mask/cache length mismatch"):
        model(np.array([[1], [2]]), cache=[QuantizedBatchCache(), rope])
    assert [row["length"] for row in rope.rows] == [0, 0]

    with pytest.raises(ValueError, match="mask/cache length mismatch"):
        model(np.array([[1], [2]]), cache=[_FakeKVCache(), rope],
              mask=np.ones((2, 1, 2), dtype=bool))
    assert [row["length"] for row in rope.rows] == [0, 0]


def test_patched_attention_uses_full_cache_mask_after_append(fake_mx, monkeypatch):
    language = types.ModuleType("mlx_vlm.models.qwen2_5_vl.language")
    seen = []

    def attention(queries, keys, values, cache, scale, mask):
        seen.append((keys.shape[-2], mask.shape[-1]))
        return np.zeros_like(queries)

    language.apply_multimodal_rotary_pos_emb = lambda q, k, *args, **kwargs: (q, k)
    language.scaled_dot_product_attention = attention
    monkeypatch.setitem(sys.modules, "mlx_vlm", types.ModuleType("mlx_vlm"))
    monkeypatch.setitem(sys.modules, "mlx_vlm.models", types.ModuleType("mlx_vlm.models"))
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.qwen2_5_vl",
                        types.ModuleType("mlx_vlm.models.qwen2_5_vl"))
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.qwen2_5_vl.language", language)

    class Attention:
        n_heads = n_kv_heads = 1
        head_dim = 2
        scale = 1.0
        q_proj = k_proj = v_proj = o_proj = staticmethod(lambda x: x)
        rotary_emb = types.SimpleNamespace(apply_rotary=lambda q, k, p, **kw: (q, k))

    class Cache:
        def update_and_fetch(self, keys, values):
            pad = np.zeros((2, 1, 3, 2))
            return np.concatenate((pad, keys), axis=2), np.concatenate((pad, values), axis=2)

    patched = _batch_mask_attention(Attention)()
    patched(np.ones((2, 1, 2)), mask=np.ones((2, 1, 1, 4), dtype=bool),
            cache=Cache(), position_ids=np.zeros((3, 2, 1), dtype=np.int32))
    assert seen == [(4, 4)]


def test_media_prefill_decode_and_global_state_isolation(fake_mx):
    pinned = _FakePinnedModel()
    model = RequestPrivateQwen25Model(pinned)
    cache = model.make_cache()
    assert model(np.array([[10, 11, 12, 13]]), cache=cache,
                 pixel_values="image", image_grid_thw="grid",
                 _mlx2_rope_media_end=3).tolist() == [1]
    assert cache[-1].rows == [{"length": 4, "delta": -1, "media": True,
                               "media_end": 3}]
    assert pinned.language_model._rope_deltas is None
    assert pinned.language_model._position_ids is None
    model(np.array([[14]]), cache=cache)
    np.testing.assert_array_equal(pinned.seen[-1], np.full((3, 1, 1), 3))


def test_merge_extract_trim_and_serialization(fake_mx):
    model = RequestPrivateQwen25Model(_FakePinnedModel())
    media = model.make_cache()[-1]
    media.rows[0].update(length=8, delta=-2, media=True, media_end=3)
    text = Qwen25RoPECache([{"length": 8, "delta": 0, "media": False,
                            "media_end": 0}])
    joined = Qwen25RoPECache.merge([media, text])
    assert joined.rows[0]["delta"] == -2
    assert joined.extract(1).rows == text.rows
    model(np.array([[1], [2]]), cache=[_FakeKVCache(), joined])
    np.testing.assert_array_equal(model._model.seen[-1][:, :, 0],
                                  np.array([[6, 8]] * 3))
    joined.trim_ragged([3, 1])
    assert [row["length"] for row in joined.rows] == [6, 8]
    restored = Qwen25RoPECache.from_state(joined.state, joined.meta_state)
    assert restored.rows == joined.rows
    branch = copy.deepcopy(restored)
    branch.rows[0]["delta"] = 7
    assert restored.rows[0]["delta"] == -2
    restored.trim_ragged([6, 0])
    assert restored.rows[0] == {"length": 0, "delta": 0, "media": False,
                                "media_end": 0}


def test_media_cannot_enter_warm_or_batched_prefill(fake_mx):
    model = RequestPrivateQwen25Model(_FakePinnedModel())
    cache = model.make_cache()
    cache[-1].rows[0]["length"] = 3
    with pytest.raises(ValueError, match="cold isolated"):
        model(np.array([[1, 2]]), cache=cache, pixel_values="image",
              image_grid_thw="grid", _mlx2_rope_media_end=2)
    assert cache[-1].rows[0]["length"] == 3
    with pytest.raises(ValueError, match="cannot override"):
        model(np.array([[1]]), cache=cache, position_ids=np.array([1]))


def test_padded_text_rows_restore_exact_lengths(fake_mx):
    model = RequestPrivateQwen25Model(_FakePinnedModel())
    rope = Qwen25RoPECache.merge([Qwen25RoPECache(), Qwen25RoPECache()])
    rope.prepare(lengths=[2, 2], right_padding=[1, 1])
    model(np.array([[1, 2, 0], [4, 5, 0]]), cache=[_FakeKVCache(), rope])
    rope.finalize()
    assert [row["length"] for row in rope.rows] == [2, 2]


def test_unequal_current_width_rejected_before_mask_or_model(fake_mx):
    pinned = _FakePinnedModel()
    model = RequestPrivateQwen25Model(pinned)
    rope = Qwen25RoPECache.merge([Qwen25RoPECache(), Qwen25RoPECache()])
    rope.prepare(lengths=[3, 1], right_padding=[0, 2])

    class CountingKV(_FakeKVCache):
        mask_calls = 0

        def make_mask(self, width, return_array=False):
            self.mask_calls += 1
            return super().make_mask(width, return_array=return_array)

    kv = CountingKV()
    with pytest.raises(ValueError, match="unequal-length batch is not qualified"):
        model(np.array([[1, 2, 3], [4, 0, 0]]), cache=[kv, rope])
    assert kv.mask_calls == 0
    assert pinned.seen == []
    assert rope.rows == [
        {"length": 0, "delta": 0, "media": False, "media_end": 0},
        {"length": 0, "delta": 0, "media": False, "media_end": 0},
    ]
    assert rope._right_padding == [0, 2]


def test_invalid_padding_cannot_mutate_replayed_rows():
    left = Qwen25RoPECache([{"length": 6, "delta": -2,
                            "media": True, "media_end": 3}])
    right = Qwen25RoPECache([{"length": 4, "delta": 0,
                             "media": False, "media_end": 0}])
    joined = Qwen25RoPECache.merge([left, right])
    before = copy.deepcopy(joined.rows)
    for invalid in ([1, -1], [1, 1.5], [1], [1, 2, 3]):
        with pytest.raises(ValueError, match="right padding"):
            joined.prepare(right_padding=invalid)
        assert joined.rows == before
    joined.prepare(right_padding=[1, 5])
    with pytest.raises(ValueError, match="exceeds row length"):
        joined.finalize()
    assert joined.rows == before
    joined.prepare(right_padding=[4, 0])
    with pytest.raises(ValueError, match="enters media span"):
        joined.finalize()
    assert joined.rows == before
    joined.prepare(right_padding=[1, 2])
    joined.finalize()
    assert [row["length"] for row in joined.rows] == [5, 2]
    assert left.rows[0]["length"] == 6
    joined.prepare(right_padding=[5, 2])
    joined.finalize()
    assert joined.rows == [
        {"length": 0, "delta": 0, "media": False, "media_end": 0},
        {"length": 0, "delta": 0, "media": False, "media_end": 0},
    ]


def test_restored_unequal_media_and_text_rows_fail_closed_before_replay(fake_mx):
    model = RequestPrivateQwen25Model(_FakePinnedModel())
    media = Qwen25RoPECache([{"length": 8, "delta": -2,
                              "media": True, "media_end": 3}])
    text = Qwen25RoPECache([{"length": 4, "delta": 0,
                             "media": False, "media_end": 0}])
    restored = [Qwen25RoPECache.from_state(c.state, c.meta_state)
                for c in (media, text)]
    joined = Qwen25RoPECache.merge(restored)
    joined.prepare(lengths=[3, 1], right_padding=[0, 2])
    with pytest.raises(ValueError, match="unequal-length batch is not qualified"):
        model(np.array([[31, 32, 33], [41, 0, 0]]),
              cache=[_FakeKVCache(), joined])
    assert joined.rows == [media.rows[0], text.rows[0]]
    assert model._model.seen == []
    joined.prepare(right_padding=None)
    media_branch, text_branch = joined.extract(0), joined.extract(1)
    media_branch.trim(3)
    text_branch.trim(1)
    assert [row["length"] for row in joined.rows] == [8, 4]
    model(np.array([[51, 52]]), cache=[_FakeKVCache(), media_branch])
    model(np.array([[61, 62]]), cache=[_FakeKVCache(), text_branch])
    np.testing.assert_array_equal(model._model.seen[-2],
                                  np.array([[[3, 4]], [[3, 4]], [[3, 4]]]))
    np.testing.assert_array_equal(model._model.seen[-1],
                                  np.array([[[3, 4]], [[3, 4]], [[3, 4]]]))
    assert [row["length"] for row in (media_branch.rows[0], text_branch.rows[0])] == [7, 5]


def test_invalid_or_divergent_state_fails_closed():
    cache = Qwen25RoPECache()
    with pytest.raises(ValueError, match="incompatible"):
        cache.meta_state = ("other-v1", "[]")
    with pytest.raises(ValueError, match="invalid"):
        cache.trim_ragged([1])
    cache.rows[0].update(length=6, delta=-1, media=True, media_end=4)
    with pytest.raises(ValueError, match="inside media span"):
        cache.trim(3)
    assert cache.rows[0]["length"] == 6


def test_descriptor_declares_implemented_unqualified_prefix_state():
    from mlx2.adapters.qwen25_vl import DESCRIPTOR
    from mlx2.contracts import Capability

    assert Capability.APC_V2 in DESCRIPTOR.capabilities
    assert Capability.PREFIX_REUSE in DESCRIPTOR.capabilities
    assert DESCRIPTOR.metadata["qualification"] == "pending"
    assert DESCRIPTOR.cache_layout == "qwen25-vl-request-private-mrope-v1"


def test_final_media_placeholder_fails_before_serving(monkeypatch):
    from mlx2.adapters.pinned_vlm_candidate import PinnedVisionCandidateAdapter
    from mlx2.adapters.qwen25_vl import Qwen25VLCandidateAdapter

    request = {"messages": []}
    prepared = {**request, "_mlx2_prompt_tokens": [1, 2, 3],
                "_mlx2_media_token_end": 3, "_mlx2_prefill_inputs": {}}
    monkeypatch.setattr(PinnedVisionCandidateAdapter, "prepare_multimodal_request",
                        lambda self, request, file_loader=None: prepared)
    adapter = object.__new__(Qwen25VLCandidateAdapter)
    with pytest.raises(ValueError, match="final prompt token"):
        adapter.prepare_multimodal_request(request)
    prepared["_mlx2_prompt_tokens"].append(4)
    result = adapter.prepare_multimodal_request(request)
    assert result["_mlx2_prefill_inputs"]["_mlx2_rope_media_end"] == 3
