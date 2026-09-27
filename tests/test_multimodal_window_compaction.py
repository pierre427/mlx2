"""One-chunk multimodal prefill and sliding-window cache memory (CPU only).

An isolated multimodal request prefills its whole prompt in one chunk
(``PromptProcessingBatch.prompt``).  A sliding layer concatenates that chunk
onto its window, so during that forward every sliding layer holds all ``P``
prompt tokens, and the APCv2 prompt boundary extracted from the batch cache
after prefill kept a full-length sliding copy for the life of the request
(and published it).  Admission charged a sliding layer ``window +
prefill_step`` tokens.  The live cache itself is trimmed by the first decode
step, which ``PromptProcessingBatch.generate`` runs before ``extend``.

The fix trims sliding caches back to the window at each chunk boundary
(exact: it is the trim the next update performs anyway), so the boundary
holds the window, and charges the in-forward concatenation as a prefill
transient.  These tests drive the real ``BatchGenerator`` with mlx2's cache
classes (Gemma 4's) and mlx-vlm's (Gemma 3n's) on a tiny model with a
KV-shared layer.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2 import serving
from mlx2.adapters.mlx_vlm_memory import SlidingKVCacheBudget
from mlx2.runtime import generate as generate_module
from mlx2.runtime.generate import BatchGenerator, GenerationBatch
from mlx2.runtime.models.base import create_attention_mask

V, D, W, STEP = 32, 16, 8, 8
LAYERS = ["sliding_attention", "full_attention", "sliding_attention"]


def _cache_types(family):
    if family == "mlx2":
        from mlx2.runtime.models.cache import KVCache, RotatingKVCache
    else:
        pytest.importorskip("mlx_vlm")
        from mlx_vlm.models.cache import KVCache, RotatingKVCache

        from mlx2.adapters.mlx_vlm import register_mlx_vlm_window_compaction

        register_mlx_vlm_window_compaction()
    return KVCache, RotatingKVCache


class TinyWindowModel(nn.Module):
    """Sliding, full, sliding, plus a fourth layer that reuses layer 0's K/V
    (Gemma 3n/Gemma 4 KV sharing reads the source cache during the forward).
    ``pixel_values`` shifts the embedding, standing in for the encoder."""

    def __init__(self, family):
        super().__init__()
        self.kv_types = _cache_types(family)
        mx.random.seed(0)
        self.embed = nn.Embedding(V, D)
        self.qs = [nn.Linear(D, D, bias=False) for _ in range(4)]
        self.ks = [nn.Linear(D, D, bias=False) for _ in range(3)]
        self.vs = [nn.Linear(D, D, bias=False) for _ in range(3)]
        self.out = nn.Linear(D, V, bias=False)
        self.forward_sliding_bytes = []
        mx.eval(self.parameters())

    @property
    def layers(self):
        return [0, 1, 2]

    def make_cache(self):
        KVCache, RotatingKVCache = self.kv_types
        return [
            RotatingKVCache(max_size=W, keep=0) if kind == "sliding_attention" else KVCache()
            for kind in LAYERS
        ]

    def __call__(self, inputs, cache=None, pixel_values=None, **_kwargs):
        h = self.embed(inputs)
        if pixel_values is not None:
            h = h + 3.0 * pixel_values.sum()
        B, L, _ = h.shape
        sliding = create_attention_mask(h, cache[0], window_size=W, return_array=True)
        full = create_attention_mask(h, cache[1], return_array=True)

        def attend(i, keys, values, mask):
            q = self.qs[i](h).reshape(B, L, 1, D).transpose(0, 2, 1, 3)
            o = mx.fast.scaled_dot_product_attention(q, keys, values, scale=D**-0.5, mask=mask)
            return o.transpose(0, 2, 1, 3).reshape(B, L, D)

        for i, kind in enumerate(LAYERS):
            k = self.ks[i](h).reshape(B, L, 1, D).transpose(0, 2, 1, 3)
            v = self.vs[i](h).reshape(B, L, 1, D).transpose(0, 2, 1, 3)
            k, v = cache[i].update_and_fetch(k, v)
            h = h + attend(i, k, v, sliding if kind == "sliding_attention" else full)
        shared = cache[0].state
        h = h + attend(3, shared[0], shared[1], sliding)
        self.forward_sliding_bytes.append(_sliding_bytes(cache))
        return self.out(h)


def _budget():
    text = {
        "num_hidden_layers": 3,
        "layer_types": LAYERS,
        "num_key_value_heads": 1,
        "head_dim": D,
        "sliding_window": W,
        "dtype": "float32",
    }
    return SlidingKVCacheBudget.from_gemma4_config(text, mtp=False, prefill_step=STEP)


def _generator(model):
    return BatchGenerator(
        model,
        completion_batch_size=4,
        prefill_batch_size=2,
        prefill_step_size=STEP,
        prefill_batch_window=1,
    )


def _media():
    pixels = mx.ones((1, 3))
    mx.eval(pixels)
    return {"pixel_values": pixels}


TEXT = [(5 * i + 2) % (V - 1) + 1 for i in range(13)]
MEDIA = [(3 * i + 1) % (V - 1) + 1 for i in range(96)]  # 12x the window


def _sliding_bytes(caches):
    return sum(c.nbytes for c, kind in zip(caches, LAYERS) if kind == "sliding_attention")


def _sliding_window_bytes(caches):
    """Sliding K/V without the restore snapshots ``nbytes`` also counts."""
    return sum(
        c.keys.nbytes + c.values.nbytes
        for c, kind in zip(caches, LAYERS)
        if kind == "sliding_attention" and c.keys is not None
    )


def _drive(gen, uids, record, limit=400):
    done = set()
    for _ in range(limit):
        _, responses = gen.next()
        for r in responses:
            record.setdefault(r.uid, []).append(
                (int(r.token), np.array(r.logprobs.astype(mx.float32)))
            )
            if r.finish_reason:
                done.add(r.uid)
        if done >= set(uids):
            return record
    raise AssertionError("lanes did not finish")


# ---------------------------------------------------------------------------
# Memory: what the real caches hold versus what admission charges.
# ---------------------------------------------------------------------------


def test_one_chunk_media_prefill_memory_stays_within_what_admission_charges(monkeypatch):
    budget = _budget()
    model = TinyWindowModel("mlx2")
    gen = _generator(model)
    extended = []
    original_extend = GenerationBatch.extend

    def extend(self, batch):
        original_extend(self, batch)
        extended.append(_sliding_bytes(self.prompt_cache))

    monkeypatch.setattr(GenerationBatch, "extend", extend)
    text_uid = gen.insert([TEXT], max_tokens=[40])[0]
    for _ in range(4):
        gen.next()
    model.forward_sliding_bytes.clear()
    extended.clear()
    media_uid = gen.insert([MEDIA], max_tokens=[4], prefill_inputs=[_media()])[0]
    for _ in range(10):
        if media_uid in gen._prompt_boundaries:
            break
        gen.next()
    # The media prompt ran as a single forward of len(MEDIA) - 1 rows.
    prefill_peak = max(model.forward_sliding_bytes)
    row = budget.sliding_bytes_per_token
    assert prefill_peak == (len(MEDIA) - 1) * row
    boundary = gen._prompt_boundaries[media_uid]["target_cache"]

    # What admission charges the lane's sliding layers: ``project``'s live
    # window-plus-chunk term and its restore-snapshot term, plus (new) the
    # one-chunk prefill transient.
    context = len(MEDIA) + 4
    live_charge = min(budget._capacity(context), budget.sliding_token_cap) * row
    snapshot_charge = budget.sliding_snapshots(context) * budget.sliding_snapshot_bytes(
        context
    )
    sliding_charge = live_charge + snapshot_charge
    transient = getattr(budget, "prefill_transient_bytes", lambda *_: 0)(
        len(MEDIA), len(MEDIA)
    )

    # 1. During the one-chunk forward every sliding layer holds the whole
    #    prompt.  That transient is now charged; before, it was not.
    assert prefill_peak > sliding_charge
    assert prefill_peak <= sliding_charge + transient
    # 2. The APCv2 prompt boundary retained for the request's lifetime held a
    #    full-length sliding copy; it is now the window, plus any exact
    #    restore snapshots (each cropped to the window), which the snapshot
    #    term charges.
    assert _sliding_window_bytes(boundary) <= W * row <= live_charge
    snapshots = _sliding_bytes(boundary) - _sliding_window_bytes(boundary)
    assert 0 <= snapshots <= snapshot_charge
    # 3. Joining the decode batch keeps each lane inside its sliding charge
    #    (the first decode step has already trimmed the lane before extend).
    assert extended and extended[-1] <= 2 * sliding_charge
    # 4. And the lane's own live cache, after prefill, is back to the window.
    live = gen._generation_batch.extract_cache(
        gen._generation_batch.uids.index(media_uid)
    )
    assert _sliding_window_bytes(live) <= W * row
    assert _sliding_bytes(live) <= W * row + snapshot_charge


def test_serving_charges_the_whole_media_tail_as_one_chunk():
    budget = _budget()
    tail = len(MEDIA)
    ordinary = serving.prefill_transient_gib(
        budget, context_tokens=tail, uncached_tokens=tail, prefill_step=STEP
    )
    media = serving.prefill_transient_gib(
        budget, context_tokens=tail, uncached_tokens=tail, prefill_step=STEP,
        single_chunk=True,
    )
    assert ordinary == 0.0
    expected = (min(budget._capacity(tail), W + tail) - (W + STEP)) * budget.sliding_bytes_per_token
    assert media == pytest.approx(expected / (1 << 30))
    # A full-attention-only family (MiniCPM-o's Qwen2) has no sliding term.
    qwen2 = SlidingKVCacheBudget.from_qwen2_config(
        {
            "num_attention_heads": 4, "hidden_size": 64, "num_hidden_layers": 2,
            "num_key_value_heads": 2, "dtype": "bfloat16",
        },
        mtp=False, prefill_step=STEP,
    )
    assert qwen2.prefill_transient_bytes(tail, tail) == 0


def test_31b_geometry_16k_image_prompt_numbers():
    """The 31B's sliding term at a 16K one-chunk image prompt (bf16)."""
    # ~/mlx-models/gemma-4-31B-MLX-8bit/config.json geometry.
    text = {
        "num_hidden_layers": 60,
        "layer_types": [
            "full_attention" if i % 6 == 5 else "sliding_attention" for i in range(60)
        ],
        "num_key_value_heads": 16,
        "num_global_key_value_heads": 4,
        "attention_k_eq_v": True,
        "head_dim": 256,
        "global_head_dim": 512,
        "sliding_window": 1024,
        "dtype": "bfloat16",
    }
    budget = SlidingKVCacheBudget.from_gemma4_config(text, mtp=False)
    prompt = 16384
    per_token = budget.sliding_bytes_per_token
    assert per_token == 800 * 1024
    charged = budget.sliding_token_cap * per_token
    full_length = prompt * per_token
    assert full_length / (1 << 30) == pytest.approx(12.5)
    assert charged / (1 << 30) == pytest.approx(2.34375)
    transient = budget.prefill_transient_bytes(prompt, prompt)
    assert charged + transient >= full_length
    # Capped at the allocation capacity (prompt + 256), less the charged cap.
    assert transient == (prompt + 256 - 3072) * per_token
    assert transient / (1 << 30) == pytest.approx(10.3515625)


# ---------------------------------------------------------------------------
# Exactness: compaction changes no token, no logit, and no APCv2 reuse.
# ---------------------------------------------------------------------------


def _scenario(family, *, compact, monkeypatch):
    calls = []
    real = generate_module.compact_prompt_cache_windows

    def spy(cache):
        trimmed = real(cache) if compact else 0
        calls.append(trimmed)
        return trimmed

    monkeypatch.setattr(generate_module, "compact_prompt_cache_windows", spy)
    model = TinyWindowModel(family)
    gen = _generator(model)
    record = {}
    first = gen.insert([TEXT], max_tokens=[30])[0]
    for _ in range(3):
        _, responses = gen.next()
        for r in responses:
            record.setdefault(r.uid, []).append(
                (int(r.token), np.array(r.logprobs.astype(mx.float32)))
            )
    media = gen.insert([MEDIA], max_tokens=[6], prefill_inputs=[_media()])[0]
    # Two ragged text prompts prefilled together: left padding plus several
    # chunks, compacted between chunks.
    pair = gen.insert([TEXT[:11] * 2, TEXT[:5] * 3], max_tokens=[5, 5])
    _drive(gen, [first, media, *pair], record)
    # APCv2-style reuse of the media prompt boundary: a follow-up turn
    # restores the extracted boundary and prefills only its new tokens.
    boundary = gen.pop_prompt_boundary(media)
    follow = gen.insert(
        [[7, 9, 11, 13, 2]], max_tokens=[5],
        caches=[boundary["target_cache"]], all_tokens=[boundary["tokens"]],
    )[0]
    _drive(gen, [follow], record)
    return record, [first, media, *pair, follow], sum(calls)


@pytest.mark.parametrize("family", ["mlx2", "mlx_vlm"])
def test_compaction_changes_no_token_or_logit(family, monkeypatch):
    on, uids_on, trimmed = _scenario(family, compact=True, monkeypatch=monkeypatch)
    off, uids_off, untouched = _scenario(family, compact=False, monkeypatch=monkeypatch)
    assert trimmed > 0 and untouched == 0
    assert uids_on == uids_off
    for uid in uids_on:
        assert [t for t, _ in on[uid]] == [t for t, _ in off[uid]], uid
        for (_, a), (_, b) in zip(on[uid], off[uid]):
            assert np.array_equal(a, b), uid


@pytest.mark.parametrize("family", ["mlx2", "mlx_vlm"])
def test_batch_compaction_is_the_trim_the_next_update_performs(family):
    """State after compaction equals the state the uncompacted cache reaches
    after the same next update, for decode and for another chunk."""
    KVCache, RotatingKVCache = _cache_types(family)
    from mlx2.runtime.models.cache import compact_prompt_cache_windows

    def filled(rows):
        cache = RotatingKVCache(max_size=W, keep=0).merge([RotatingKVCache(max_size=W, keep=0)])
        k = mx.arange(rows * 4, dtype=mx.float32).reshape(1, 1, rows, 4)
        cache.update_and_fetch(k, k + 0.5)
        return cache

    for rows, nxt in ((37, 1), (37, 5), (W + 1, 1), (W, 1)):
        a, b = filled(rows), filled(rows)
        trimmed = compact_prompt_cache_windows([a])
        assert trimmed == int(rows > W)
        if trimmed:
            assert a.keys.shape[2] == W
        k = mx.full((1, 1, nxt, 4), -1.0)
        mask_a = a.make_mask(nxt, return_array=True)
        mask_b = b.make_mask(nxt, return_array=True)
        ka, va = a.update_and_fetch(k, k)
        kb, vb = b.update_and_fetch(k, k)
        assert np.array_equal(np.array(ka), np.array(kb))
        assert np.array_equal(np.array(va), np.array(vb))
        assert np.array_equal(np.array(mask_a), np.array(mask_b))
        assert a.left_padding.tolist() == b.left_padding.tolist()
        assert (a._idx, a.rotated) == (b._idx, b.rotated)


def test_unbatched_rotating_snapshot_holds_the_window(monkeypatch):
    """mlx2's RotatingKVCache (used unbatched) records a restore snapshot at a
    chunk boundary; after compaction it copies the window, not the chunk."""
    from mlx2.runtime.models.cache import (
        RotatingKVCache,
        compact_prompt_cache_windows,
        record_state_checkpoints,
    )

    monkeypatch.setenv("MLX_LM_STATE_CHECKPOINT_STRIDE", "1")
    a, b = RotatingKVCache(max_size=W), RotatingKVCache(max_size=W)
    k = mx.arange(40 * 4, dtype=mx.float32).reshape(1, 1, 40, 4)
    for cache in (a, b):
        cache.update_and_fetch(k, k)
    assert compact_prompt_cache_windows([a]) == 1
    record_state_checkpoints([a], [40], force=True)
    assert a._checkpoints[-1][1].shape[2] == W
    assert a.nbytes == 4 * W * 4 * 4
    # Restoring the snapshot and continuing matches the uncompacted cache.
    a.trim_to_position(40, 0)
    step = mx.full((1, 1, 1, 4), -1.0)
    ka, _ = a.update_and_fetch(step, step)
    kb, _ = b.update_and_fetch(step, step)
    assert np.array_equal(np.array(ka), np.array(kb))
