"""A multi-token step after the batched sliding ring rotated must not hide a
short row's oldest in-window token.

``BatchRotatingKVCache._update_concat`` first restores temporal order (which
sets ``_idx`` to the ring width, ``max_size``) and then trims one column, so
``left_padding`` drops by one.  ``make_mask`` worked the N > 1 trim out from the
rotated write cursor ``_idx`` instead, leaving the padding boundary one column
to the right: every row with ``left_padding > 0`` lost its oldest real token
for the early queries of that chunk.  In serving this follows a prefill round
in which every prompt row's slice is one token (checkpoint clamps), after a
longer row filled the window.
"""

from __future__ import annotations

import copy

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.muse_glimmer_config import ModelArgs
from mlx2.runtime.generate import BatchGenerator
from mlx2.runtime.models.cache import (
    BatchRotatingKVCache,
    BatchRotatingQuantizedKVCache,
    RotatingKVCache,
)
from mlx2.runtime.models.muse_glimmer import Model

W = 4


def _kv(values, dim=1):
    a = mx.array(values, dtype=mx.float32)[None, None, :, None]
    return mx.repeat(a, dim, axis=-1)


def _row(tokens, dim=1):
    cache = RotatingKVCache(max_size=W)
    a = _kv(tokens, dim)
    cache.update_and_fetch(a, a)
    return cache


def _rotated_batch(dim=1):
    """Rows of 6 and 2 tokens merged, then one decode step rotates the ring."""
    long_row, short_row = _row([1, 2, 3, 4, 5, 6], dim), _row([11, 12], dim)
    reference = copy.deepcopy(short_row)
    batch = BatchRotatingKVCache.merge([long_row, short_row])
    assert batch.left_padding.tolist() == [0, 2]
    step = mx.repeat(mx.array([[7.0], [13.0]])[:, None, :, None], dim, axis=-1)
    batch.make_mask(1, window_size=W)
    batch.update_and_fetch(step, step)
    reference.update_and_fetch(_kv([13], dim), _kv([13], dim))
    assert batch.rotated and 0 < batch._idx < W
    return batch, reference


def _visible(mask_row, values):
    return [int(v) for v, seen in zip(values, mask_row) if seen]


def test_rotated_ring_multi_token_mask_keeps_short_row_window():
    batch, reference = _rotated_batch()
    mask = batch.make_mask(2, window_size=W)
    chunk = mx.array([[8.0, 9.0], [14.0, 15.0]])[:, None, :, None]
    keys, _ = batch.update_and_fetch(chunk, chunk)

    want_mask = reference.make_mask(2, window_size=W, return_array=True)
    want_keys, _ = reference.update_and_fetch(_kv([14, 15]), _kv([14, 15]))

    got_values = np.array(keys[1, 0, :, 0]).tolist()
    want_values = np.array(want_keys[0, 0, :, 0]).tolist()
    for query in range(2):
        assert _visible(np.array(mask[1, 0, query]), got_values) == _visible(
            np.array(want_mask[query]), want_values
        )
    # The long row (no padding) is unaffected either way.
    assert _visible(np.array(mask[0, 0, 0]), np.array(keys[0, 0, :, 0]).tolist()) == [
        5, 6, 7, 8,
    ]


def test_rotated_quantized_ring_multi_token_mask_keeps_short_row_window():
    batch, _reference = _rotated_batch(dim=64)
    quantized = batch.to_quantized(group_size=64, bits=8)
    assert isinstance(quantized, BatchRotatingQuantizedKVCache)
    assert quantized.rotated and quantized._idx == batch._idx
    mask = quantized.make_mask(2, window_size=W)
    chunk = mx.repeat(
        mx.array([[8.0, 9.0], [14.0, 15.0]])[:, None, :, None], 64, axis=-1
    )
    quantized.update_and_fetch(chunk, chunk)
    # Row 1 holds [11, 12, 13] then the chunk: after the concat its padding
    # boundary is column 0, so query 0 must see all four columns 0..3.
    assert quantized.left_padding.tolist()[1] == 0
    assert np.array(mask[1, 0, 0]).tolist()[:4] == [True, True, True, True]


def _muse():
    mx.random.seed(3)
    model = Model(
        ModelArgs(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=128,
            sliding_window=W,
            max_position_embeddings=2048,
        )
    )
    model.eval()
    mx.eval(model.parameters())
    return model


def _drive(gen, uids, limit=200):
    record, done = {}, set()
    for _ in range(limit):
        _, responses = gen.next()
        for r in responses:
            record.setdefault(r.uid, []).append(np.array(r.logprobs.astype(mx.float32)))
            if r.finish_reason:
                done.add(r.uid)
        if done >= set(uids):
            return record
    raise AssertionError("lanes did not finish")


def _generator(model):
    return BatchGenerator(
        model, completion_batch_size=4, prefill_batch_size=4, prefill_step_size=64
    )


def test_one_token_prefill_round_then_chunk_matches_solo_rows():
    """Scheduler level: checkpoint clamps give a round where every prompt row
    takes one token after the long row filled the window; the short row's
    next chunk must still match the same prompt prefilled alone."""
    model = _muse()
    long_prompt = list(range(5, 15))
    short_prompt = list(range(20, 30))
    positions = [[5, 6], [1, 2]]
    gen = _generator(model)
    uids = gen.insert(
        [long_prompt, short_prompt], max_tokens=[2, 2], apc_interior_positions=positions
    )
    batched = _drive(gen, uids)
    for uid, prompt, interior in zip(uids, (long_prompt, short_prompt), positions):
        solo_gen = _generator(model)
        solo_uid = solo_gen.insert(
            [prompt], max_tokens=[2], apc_interior_positions=[interior]
        )[0]
        solo = _drive(solo_gen, [solo_uid])[solo_uid]
        np.testing.assert_allclose(batched[uid][0], solo[0], atol=1e-4)
