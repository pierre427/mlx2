"""DeviceOffsetKVCache (spec-round slice 1a-4 DC-1), CPU only, on the tiny Raon model.

Pins: host mode is KVCache bit for bit; in device mode, a round of verify rows
committed by a device count gives the same logits as the host KVCache trimmed
by the same count (within float32 reduction-order noise; Pierre's bar for
device-commit rounds is the near-tie contract), rejected rows in the slack
are never attended, and host-int operations refuse in device mode.
"""

import mlx.core as mx
import numpy as np
import pytest

from test_raon1_mini_cpu import _model

from mlx2.runtime.models.cache import KVCache
from mlx2.runtime.models.device_offset_kv import DeviceOffsetKVCache

mx.set_default_device(mx.cpu)


def _caches(model, device):
    caches = model.make_cache()
    if device:
        caches = [DeviceOffsetKVCache() if type(c) is KVCache else c for c in caches]
    return caches


def test_host_mode_is_kvcache_bit_for_bit():
    model = _model()
    tokens = mx.array([[3, 14, 15, 9, 26, 5, 35, 8, 9, 79, 4, 41]])
    ref, dev = _caches(model, False), _caches(model, True)
    a = [model(tokens[:, :7], ref)] + [model(tokens[:, i:i + 1], ref) for i in range(7, 12)]
    b = [model(tokens[:, :7], dev)] + [model(tokens[:, i:i + 1], dev) for i in range(7, 12)]
    for x, y in zip(a, b):
        assert mx.array_equal(x, y).item()


def _assert_close(ref, dev):
    for x, y in zip(ref, dev):
        assert x.shape == y.shape
        assert float(mx.max(mx.abs(x - y))) < 1e-5
        assert np.array_equal(np.argmax(np.array(x), -1), np.argmax(np.array(y), -1))


def _configured(mla):
    from mlx2.adapters import raon1_mini as A

    model = _model()
    model.configure(A.parse_policy({"mla": mla}))
    return model


@pytest.mark.parametrize("mla", ["auto", "latent_single", "expanded"])
@pytest.mark.parametrize("rounds", [
    [(3, 4), (3, 1), (5, 2), (2, 3)],
    [(5, 6), (5, 1), (5, 2)],
    [(1, 1), (4, 2), (4, 5), (3, 1), (6, 3)],
    [(0, 1), (15, 3), (0, 1), (15, 16), (2, 1)],  # one-row decode and 16-row verify
])
def test_device_commit_matches_host_trim(rounds, mla):
    model = _configured(mla)
    ref = _speculative_rounds_kv_only(model, False, rounds)
    dev = _speculative_rounds_kv_only(model, True, rounds)
    _assert_close(ref, dev)


def _speculative_rounds_kv_only(model, device, rounds):
    # The tiny Raon's RDA layers keep their own exact host replay (DC-2 is a
    # separate slice); drive them through the same speculating window on both
    # arms so only the KV path differs.
    from mlx2.runtime.models.cache import ArraysCache

    rng = np.random.default_rng(11)
    caches = _caches(model, device)
    model(mx.array([[3, 14, 15, 9, 26, 5, 35, 8]]), caches)
    kv = [c for c in caches if isinstance(c, DeviceOffsetKVCache)]
    for c in kv:
        c.enter_device_mode()
    outs = []
    for k, keep in rounds:
        rows = mx.array([rng.integers(1, 96, size=k + 1).tolist()])
        for c in caches:
            if isinstance(c, ArraysCache):
                c.start_speculation()
        logits = model(rows, caches)
        outs.append(logits[:, :keep])
        for c in caches:
            if isinstance(c, DeviceOffsetKVCache):
                c.device_commit(mx.array([keep], dtype=mx.int32))
            elif isinstance(c, ArraysCache):
                c.trim(k + 1 - keep)
                c.stop_speculation()
            else:
                c.trim(k + 1 - keep)
    return outs


def test_slack_rows_are_never_attended():
    model = _model()
    rounds = [(4, 2), (4, 1), (3, 3)]
    clean = _speculative_rounds_kv_only(model, True, rounds)

    # Poison every row past each cache's true fill after every commit.
    from mlx2.runtime.models import device_offset_kv as D

    real = D.DeviceOffsetKVCache.device_commit

    def poisoning(self, count):
        out = real(self, count)
        width = self.keys.shape[2]
        live = mx.arange(width)[None, None, :, None] < self.pos.reshape(1, 1, 1, 1)
        self.keys = mx.where(live, self.keys, mx.array(1e4, dtype=self.keys.dtype))
        self.values = mx.where(live, self.values, mx.array(-1e4, dtype=self.values.dtype))
        return out

    D.DeviceOffsetKVCache.device_commit = poisoning
    try:
        poisoned = _speculative_rounds_kv_only(model, True, rounds)
    finally:
        D.DeviceOffsetKVCache.device_commit = real
    for x, y in zip(clean, poisoned):
        assert mx.array_equal(x, y).item()


def test_observe_and_leave_device_mode():
    cache = DeviceOffsetKVCache()
    k = mx.ones((1, 1, 5, 4))
    cache.update_and_fetch(k, k)
    cache.enter_device_mode()
    cache.update_and_fetch(k[:, :, :3], k[:, :, :3])
    cache.device_commit(mx.array([2], dtype=mx.int32))
    assert cache.bound == 8 and int(cache.pos.item()) == 7
    with pytest.raises(RuntimeError):
        cache.trim(1)
    with pytest.raises(RuntimeError):
        cache.meta_state
    with pytest.raises(ValueError):
        cache.observe(9)
    cache.leave_device_mode(7)
    assert cache.offset == 7 and not cache.device_mode
    assert cache.meta_state == ("7",)
    keys, _ = cache.update_and_fetch(k[:, :, :1], k[:, :, :1])
    assert keys.shape[2] == 8


def test_commit_count_is_clamped_to_the_written_rows():
    cache = DeviceOffsetKVCache().enter_device_mode()
    k = mx.ones((1, 1, 4, 2))
    cache.update_and_fetch(k, k)
    cache.device_commit(mx.array([9], dtype=mx.int32))
    assert int(cache.pos.item()) == 4
    with pytest.raises(RuntimeError):
        cache.device_commit(mx.array([1], dtype=mx.int32))  # no pending write


# -- DC-2: RDA (Kimi KDA) device-count rollback -------------------------------------

def _rda_states_after(model, keep, t, device):
    from mlx2.runtime.models.cache import ArraysCache

    rng = np.random.default_rng(5)
    caches = model.make_cache()
    model(mx.array([[3, 14, 15, 9, 26, 5, 35, 8]]), caches)
    rows = mx.array([rng.integers(1, 96, size=t).tolist()])
    for c in caches:
        if isinstance(c, ArraysCache):
            c.start_speculation()
    model(rows, caches)
    out = []
    for c in caches:
        if not isinstance(c, ArraysCache):
            continue
        if device:
            c.device_rollback(mx.array([keep], dtype=mx.int32))
        else:
            c.trim(t - keep)
            c.stop_speculation()
        out.append([None if s is None else mx.array(s) for s in c.cache])
    return out


@pytest.mark.parametrize("t", [1, 4, 6])
def test_rda_device_rollback_is_bitwise_host_trim(t):
    model = _model()
    for keep in range(0, t + 1):
        host = _rda_states_after(model, keep, t, device=False)
        dev = _rda_states_after(model, keep, t, device=True)
        for layer_h, layer_d in zip(host, dev):
            for a, b in zip(layer_h, layer_d):
                assert (a is None) == (b is None)
                if a is not None:
                    assert a.shape == b.shape and mx.array_equal(a, b).item(), (t, keep)


def test_rda_device_rollback_refuses_outside_a_one_forward_window():
    from mlx2.runtime.models.cache import ArraysCache

    model = _model()
    caches = model.make_cache()
    model(mx.array([[3, 14, 15, 9]]), caches)
    rda = next(c for c in caches if isinstance(c, ArraysCache))
    with pytest.raises(RuntimeError, match="active speculation"):
        rda.device_rollback(mx.array([1], dtype=mx.int32))
    for c in caches:
        if isinstance(c, ArraysCache):
            c.start_speculation()
    model(mx.array([[5, 6]]), caches)
    model(mx.array([[7]]), caches)
    with pytest.raises(RuntimeError, match="exactly one recorded forward"):
        rda.device_rollback(mx.array([1], dtype=mx.int32))


def test_whole_rounds_with_kv_and_rda_committed_on_the_device():
    """KV by device_commit and RDA by device_rollback against the host path."""
    from mlx2.runtime.models.cache import ArraysCache

    model = _model()

    def run(device):
        rng = np.random.default_rng(23)
        caches = _caches(model, device)
        model(mx.array([[3, 14, 15, 9, 26, 5, 35, 8]]), caches)
        if device:
            for c in caches:
                if isinstance(c, DeviceOffsetKVCache):
                    c.enter_device_mode()
        outs = []
        for k, keep in [(5, 2), (5, 6), (3, 1), (4, 3), (2, 0 + 1)]:
            rows = mx.array([rng.integers(1, 96, size=k + 1).tolist()])
            for c in caches:
                if isinstance(c, ArraysCache):
                    c.start_speculation()
            logits = model(rows, caches)
            outs.append(logits[:, :keep])
            count = mx.array([keep], dtype=mx.int32)
            for c in caches:
                if isinstance(c, DeviceOffsetKVCache):
                    c.device_commit(count)
                elif isinstance(c, ArraysCache):
                    if device:
                        c.device_rollback(count)
                    else:
                        c.trim(k + 1 - keep)
                        c.stop_speculation()
                else:
                    c.trim(k + 1 - keep)
        return outs

    _assert_close(run(False), run(True))



def test_device_mode_refuses_silent_host_operations():
    cache = DeviceOffsetKVCache()
    k = mx.ones((1, 1, 4, 2))
    cache.update_and_fetch(k, k)
    cache.enter_device_mode()
    cache.update_and_fetch(k[:, :, :2], k[:, :, :2])
    with pytest.raises(RuntimeError):
        cache.meta_state = ("3",)
    with pytest.raises(RuntimeError):
        cache.state = (k, k)
    with pytest.raises(RuntimeError):
        cache.to_quantized()
    with pytest.raises(RuntimeError):
        DeviceOffsetKVCache.merge([cache])
    assert cache.state[0].shape[2] >= 6  # evaluation still sees the storage (fill + slack)


def test_observe_clears_a_pending_write():
    cache = DeviceOffsetKVCache().enter_device_mode()
    k = mx.ones((1, 1, 3, 2))
    cache.update_and_fetch(k, k)
    cache.observe(3)
    with pytest.raises(RuntimeError, match="needs a device-mode write"):
        cache.device_commit(mx.array([1], dtype=mx.int32))


def test_host_mode_state_round_trips_through_from_state():
    from mlx2.runtime.models.cache import load_prompt_cache, save_prompt_cache

    import tempfile, os
    cache = DeviceOffsetKVCache()
    k = mx.arange(8, dtype=mx.float32).reshape(1, 1, 4, 2)
    cache.update_and_fetch(k, k)
    path = os.path.join(tempfile.mkdtemp(), "c.safetensors")
    save_prompt_cache(path, [cache])
    loaded = load_prompt_cache(path)[0]
    assert type(loaded) is DeviceOffsetKVCache and loaded.offset == 4 and not loaded.device_mode
