"""DeviceRingKVCache (spec-round slice 1a-4 DC-3), CPU only, tiny DFlash2 drafter.

Pins: a drafter whose context caches are device rings fed every verify feature
row with a device-held committed count drafts what the host rotating caches
drafts when fed only the committed rows (hidden states within float32
reduction noise; the tokens equal), across ring wrap-around; rows past the
count are never attended; the context-only append and ``from_host`` adoption
keep the same law; offset readers refuse.
"""

import mlx.core as mx
import numpy as np
import pytest

from test_external_dflash2_cpu import _tiny_dflash_args, tiny as _tiny

from mlx2.runtime.drafters.device_ring_kv import DeviceRingKVCache

mx.set_default_device(mx.cpu)

SLACK = 6  # the widest round below writes exactly SLACK rows


def tiny():
    """The shared tiny pair with two KV heads (a head/slot mix-up must not pass)."""
    import dataclasses

    from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel

    model, _ = _tiny()
    mx.random.seed(8)
    args = dataclasses.replace(_tiny_dflash_args(), num_attention_heads=2, num_key_value_heads=2)
    return model, DFlash2DraftModel(args).bind(model)


def _feature_width(draft):
    return draft.config.hidden_size * len(draft.config.target_layer_ids)


def _rings(draft, caches):
    return [DeviceRingKVCache.from_host(c, window=c.max_size, slack=SLACK) for c in caches]


def _run(device, rounds, seed=4, adopt_after=0, rows_wrapper=True):
    _, draft = tiny()
    rng = np.random.default_rng(seed)
    caches = draft.make_cache()
    width = _feature_width(draft)
    outs = []
    for index, (rows, keep) in enumerate(rounds):
        if device and index == adopt_after:
            caches = _rings(draft, caches)
        features = mx.array(rng.normal(size=(1, rows, width)).astype(np.float32))
        block = mx.array([[int(rng.integers(1, 30))] + [draft.config.mask_token_id] * 3])
        if device and index >= adopt_after:
            for c in caches:
                c.pending_count = mx.array([keep], dtype=mx.int32)
            hidden = draft._hidden(block, features,
                                   draft.batch_caches([caches]) if rows_wrapper else caches)
        else:
            hidden = draft._hidden(block, features[:, :keep],
                                   draft.batch_caches([caches]) if rows_wrapper else caches)
        outs.append(hidden)
    return outs


ROUNDS = [(1, 1), (4, 2), (4, 4), (3, 1), (5, 3), (2, 2), (6, 1), (4, 3)]


@pytest.mark.parametrize("rows_wrapper", [True, False])
@pytest.mark.parametrize("adopt_after", [0, 3])
def test_device_rings_draft_like_rotating_caches(adopt_after, rows_wrapper):
    host = _run(False, ROUNDS, adopt_after=adopt_after, rows_wrapper=rows_wrapper)
    dev = _run(True, ROUNDS, adopt_after=adopt_after, rows_wrapper=rows_wrapper)
    for a, b in zip(host, dev):
        assert a.shape == b.shape
        assert float(mx.max(mx.abs(a - b))) < 1e-5
        _, draft = tiny()
        la, lb = draft._logits(a[:, 1:]), draft._logits(b[:, 1:])
        assert np.array_equal(np.argmax(np.array(la), -1), np.argmax(np.array(lb), -1))


@pytest.mark.parametrize("invalid_value", [None, np.nan, np.inf])
def test_rows_past_the_count_are_never_attended(invalid_value):
    """Garbage in the uncommitted rows of every append changes nothing."""
    _, draft = tiny()
    width = _feature_width(draft)
    rng = np.random.default_rng(9)

    def run(poison):
        caches = _rings(draft, draft.make_cache())
        r = np.random.default_rng(1)
        outs = []
        for rows, keep in ROUNDS:
            features = r.normal(size=(1, rows, width)).astype(np.float32)
            if poison:
                features[:, keep:] = (rng.normal(scale=1e3, size=features[:, keep:].shape)
                                      if invalid_value is None else invalid_value)
            block = mx.array([[5, draft.config.mask_token_id, draft.config.mask_token_id]])
            for c in caches:
                c.pending_count = mx.array([keep], dtype=mx.int32)
            outs.append(draft._hidden(block, mx.array(features), draft.batch_caches([caches])))
        return outs

    for a, b in zip(run(False), run(True)):
        assert mx.array_equal(a, b).item()


def test_context_only_append_matches_the_host_cache():
    _, draft = tiny()
    width = _feature_width(draft)
    rng = np.random.default_rng(2)
    host, dev = draft.make_cache(), _rings(draft, draft.make_cache())
    for rows, keep in [(3, 2), (4, 4), (2, 1)]:
        features = mx.array(rng.normal(size=(1, rows, width)).astype(np.float32))
        draft.append_context(features[:, :keep], host)
        for c in dev:
            c.pending_count = mx.array([keep], dtype=mx.int32)
        draft.append_context(features, dev)
    block = mx.array([[7, draft.config.mask_token_id, draft.config.mask_token_id]])
    tail = mx.zeros((1, 0, width))
    a = draft._hidden(block, tail, draft.batch_caches([host]))
    b = draft._hidden(block, tail, draft.batch_caches([dev]))
    assert float(mx.max(mx.abs(a - b))) < 1e-5


def test_ring_bookkeeping_and_refusals():
    ring = DeviceRingKVCache(window=3, slack=4)
    k = mx.ones((1, 1, 4, 2))
    keys, values, visible = ring.append(k, k, mx.array([2], dtype=mx.int32))
    assert int(ring.pos.item()) == 2 and ring.bound == 4
    assert visible.tolist() == [True, True, False, False, False, False, False]
    ring.append(k, k, mx.array([4], dtype=mx.int32))
    # Last three of absolute positions 0..5 are visible: 3, 4, 5.
    assert sorted(int(p) for p, v in zip(ring.slot_pos.tolist(), ring.context_mask().tolist()) if v) == [3, 4, 5]
    with pytest.raises(ValueError, match="slack"):
        ring.append(mx.ones((1, 1, 5, 2)), mx.ones((1, 1, 5, 2)))
    with pytest.raises(RuntimeError, match="no host offset"):
        ring.offset
    with pytest.raises(ValueError):
        DeviceRingKVCache(window=0, slack=1)



def test_adoption_and_counts_refuse_silent_context_changes():
    from mlx2.runtime.models.cache import KVCache, RotatingKVCache

    with pytest.raises(ValueError, match="sliding RotatingKVCache"):
        DeviceRingKVCache.from_host(KVCache(), window=4, slack=2)
    with pytest.raises(ValueError, match="sliding RotatingKVCache"):
        DeviceRingKVCache.from_host(RotatingKVCache(max_size=8, keep=0), window=4, slack=2)
    with pytest.raises(ValueError, match="keep=0"):
        DeviceRingKVCache.from_host(RotatingKVCache(max_size=4, keep=1), window=4, slack=2)
    ring = DeviceRingKVCache(window=3, slack=4)
    k = mx.ones((1, 2, 3, 2))
    with pytest.raises(ValueError, match="committed count"):
        ring.append(k, k)
    ring.pending_count = mx.array([1], dtype=mx.int32)
    ring.append(k[:, :, :0], k[:, :, :0])  # an empty append keeps the pending count
    ring.append(k, k)
    assert int(ring.pos.item()) == 1 and ring.pending_count is None
