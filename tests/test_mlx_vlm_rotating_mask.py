"""mlx-vlm BatchRotatingKVCache.make_mask after the ring wraps (CPU only).

Keys carry their absolute position (pads are negative), so the true mask for
each update can be read off the buffer ``update_and_fetch`` returns: slot j
is visible to query q iff 0 <= pos_j <= q and pos_j > q - window.  The mask is
built before the update, the way models call it.  The pinned mlx-vlm method
mis-trimmed a multi-token update that follows a wrap while a row's left
padding is still in the window; mlx2 patches it in
``patch_mlx_vlm_batch_rotating_mask``.
"""

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

PAD = -1e6

CASES = [
    (8, [0, 2], [5] + [1] * 20),
    (8, [0, 2], [5] + [1] * 10 + [3] + [1] * 5),
    (16, [0, 3], [10] + [1] * 12 + [4, 4] + [1] * 6),
    (8, [0, 1], [12] + [1] * 12),
    (8, [0, 2], [6] + [1] * 9 + [2, 2, 2, 2]),
    # left padding still in the window when the ring wraps, then a chunk
    (8, [0, 6], [8, 1, 3, 1, 1]),
    (8, [0, 5], [8, 1, 1, 2, 1]),
    (16, [0, 12], [16, 1, 1, 4, 1, 1]),
]
WRAP_CASES = CASES[5:]


def _mismatches(cls, window, pads, schedule):
    batch = len(pads)
    cache = cls(window, pads)
    nxt = [0] * batch
    bad = []
    for step, n in enumerate(schedule):
        rows = []
        for b in range(batch):
            if step == 0:
                real = n - pads[b]
                rows.append([PAD] * pads[b] + list(range(real)))
                nxt[b] = real
            else:
                rows.append(list(range(nxt[b], nxt[b] + n)))
                nxt[b] += n
        k = mx.array(rows, dtype=mx.float32)[:, None, :, None]
        mask = cache.make_mask(n)
        keys, _ = cache.update_and_fetch(k, k)
        pos = keys[:, 0, :, 0][:, None, :]
        q = mx.array(rows)[:, :, None]
        truth = (pos >= 0) & (pos <= q) & (pos > q - window)
        got = mx.broadcast_to(mask, (batch, 1) + mask.shape[-2:])[:, 0]
        assert got.shape == truth.shape, (step, got.shape, truth.shape)
        if ((got != truth) & (q >= 0)).any().item():
            bad.append(step)
    return bad


@pytest.fixture
def vlm_cls():
    pytest.importorskip("mlx_vlm")
    from mlx_vlm.models.cache import BatchRotatingKVCache

    from mlx2.adapters.mlx_vlm import patch_mlx_vlm_batch_rotating_mask

    patch_mlx_vlm_batch_rotating_mask(BatchRotatingKVCache)
    return BatchRotatingKVCache


@pytest.mark.parametrize("window,pads,schedule", CASES)
def test_patched_mlx_vlm_mask_matches_buffer(vlm_cls, window, pads, schedule):
    assert _mismatches(vlm_cls, window, pads, schedule) == []


@pytest.mark.parametrize("window,pads,schedule", CASES)
def test_mlx2_mask_matches_buffer(window, pads, schedule):
    from mlx2.runtime.models.cache import BatchRotatingKVCache

    assert _mismatches(BatchRotatingKVCache, window, pads, schedule) == []


@pytest.mark.parametrize("window,pads,schedule", WRAP_CASES)
def test_unpatched_mlx_vlm_mask_is_wrong_after_wrap(vlm_cls, window, pads, schedule):
    """The probe is sensitive: the pinned method fails these cases."""
    import mlx_vlm.models.cache as vlm_cache

    class Unpatched(vlm_cls):
        pass

    Unpatched.make_mask = _pinned(vlm_cache)
    assert _mismatches(Unpatched, window, pads, schedule) != []


def _pinned(vlm_cache):
    """Rebuild the pinned make_mask from source (the class attribute is patched)."""
    import inspect
    import textwrap

    src = inspect.getsource(vlm_cache)
    start = src.index("class BatchRotatingKVCache")
    body = src[start:]
    m_start = body.index("    def make_mask(")
    m_end = body.index("    def filter(", m_start)
    ns = {"mx": mx, "Optional": __import__("typing").Optional}
    exec(textwrap.dedent(body[m_start:m_end]), ns)
    return ns["make_mask"]


def test_patch_is_idempotent(vlm_cls):
    from mlx2.adapters.mlx_vlm import patch_mlx_vlm_batch_rotating_mask

    assert patch_mlx_vlm_batch_rotating_mask(vlm_cls) is False
    assert getattr(vlm_cls.make_mask, "_mlx2_wrap_fix", False)
