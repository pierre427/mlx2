"""Actual APCv2 endpoint publication/restore, CPU only."""

from dataclasses import replace

import pytest

mx = pytest.importorskip("mlx.core")
from mlx.utils import tree_flatten

from mlx2.experimental.hysparse2.apc import EndpointAPC
from mlx2.experimental.hysparse2.config import Config
from mlx2.experimental.hysparse2.model import Model
from mlx2.runtime.apc_v2 import APCv2


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(42)
    yield
    mx.set_default_device(previous)


def test_real_apcv2_endpoint_and_independent_restore():
    c = replace(Config.smoke(), local_window=4)
    model = Model(c)
    model.eval()
    tokens = list(range(1, 15))
    _, cache = model.prefill(mx.array([tokens]))
    reference = model.decode(mx.array([[15]]), cache)
    mx.eval(reference)
    _, cache = model.prefill(mx.array([tokens]))
    engine = APCv2(max_size=4, layout_name="hysparse2-endpoint-v1")
    try:
        bridge = EndpointAPC(
            model,
            engine,
            checkpoint_revision="fixture-r1",
            tokenizer_fingerprint="fixture-t1",
        )
        capability = bridge.publish(tokens, cache)
        assert capability.exact_prefix
        a, hit = bridge.restore(tokens + [15])
        assert hit.hit and hit.cached_tokens == len(tokens)
        b, _ = bridge.restore(tokens + [15])
        assert a.owner is model._cache_owner
        assert a is not b and a.self_kv is not b.self_kv
        actual = model.decode(mx.array([[15]]), a)
        assert float(mx.max(mx.abs(actual - reference)).item()) == 0
        assert b.length == len(tokens)
        again = model.decode(mx.array([[15]]), b)
        assert float(mx.max(mx.abs(again - reference)).item()) == 0
        # Reuse on an independently reconstructed, exact-weight model.
        clone = Model(c)
        clone.load_weights(tree_flatten(model.parameters()), strict=True)
        clone.eval()
        other = EndpointAPC(
            clone,
            engine,
            checkpoint_revision="fixture-r1",
            tokenizer_fingerprint="fixture-t1",
        )
        restored, _ = other.restore(tokens + [15])
        assert (
            float(
                mx.max(
                    mx.abs(clone.decode(mx.array([[15]]), restored) - reference)
                ).item()
            )
            == 0
        )
        changed = EndpointAPC(
            model,
            engine,
            checkpoint_revision="fixture-r2",
            tokenizer_fingerprint="fixture-t1",
        )
        assert changed.restore(tokens + [15])[0] is None
        # A branch before the only recorded endpoint cannot reuse approximate state.
        branched, result = bridge.restore(tokens[:-2] + [31])
        assert branched is None and not result.hit
    finally:
        engine.close()


def test_parameter_replacement_invalidates_kv_and_existing_checkpoint_bridge():
    model = Model(Config.smoke())
    model.eval()
    tokens = [1, 2, 3, 4]
    _, cache = model.prefill(mx.array([tokens]))
    engine = APCv2(max_size=4, layout_name="hysparse2-endpoint-v1")
    try:
        bridge = EndpointAPC(model, engine, checkpoint_revision="old", tokenizer_fingerprint="t")
        bridge.publish(tokens, cache)
        model.update({"embedding": {"weight": model.embedding.weight + 0.01}})
        before = cache.length
        with pytest.raises(ValueError, match="another model"):
            model.decode(mx.array([[5]]), cache)
        assert cache.length == before
        with pytest.raises(ValueError, match="parameter update"):
            bridge.restore(tokens + [5])
        _, fresh = model.prefill(mx.array([tokens]))
        with pytest.raises(ValueError, match="parameter update"):
            bridge.publish(tokens, fresh)
        rebound = EndpointAPC(model, engine, checkpoint_revision="updated", tokenizer_fingerprint="t")
        _, miss = rebound.restore(tokens + [5])
        assert not miss.hit
        rebound.publish(tokens, fresh)
        restored, hit = rebound.restore(tokens + [5])
        assert hit.hit
        expected = model.decode(mx.array([[5]]), fresh)
        actual = model.decode(mx.array([[5]]), restored)
        assert float(mx.max(mx.abs(expected - actual)).item()) == 0
        if hasattr(hit.cache, "close"):
            hit.cache.close()
    finally:
        engine.close()


def test_fail_closed_stale_and_disabled_checkpoint(monkeypatch):
    model = Model(Config.smoke())
    model.eval()
    tokens = [1, 2, 3, 4]
    _, cache = model.prefill(mx.array([tokens]))
    engine = APCv2(max_size=4, layout_name="hysparse2-endpoint-v1")
    try:
        bridge = EndpointAPC(
            model, engine, checkpoint_revision="r1", tokenizer_fingerprint="t1"
        )
        monkeypatch.setenv("MLX_LM_STATE_CHECKPOINT_MAX", "0")
        with pytest.raises(ValueError, match="disabled"):
            bridge.publish(tokens, cache)
        model.adapter_revision = "changed"
        with pytest.raises(ValueError, match="revision"):
            bridge.publish(tokens, cache)
    finally:
        engine.close()


def test_spill_restore_exact_endpoint(tmp_path):
    now = [0.0]
    model = Model(replace(Config.smoke(), local_window=4))
    model.eval()
    tokens = list(range(1, 15))
    _, cache = model.prefill(mx.array([tokens]))
    expected = model.decode(mx.array([[15]]), cache)
    _, cache = model.prefill(mx.array([tokens]))
    engine = APCv2(
        max_size=4,
        layout_name="hysparse2-endpoint-v1",
        idle_disk_seconds=1,
        idle_disk_dir=str(tmp_path),
        now_fn=lambda: now[0],
    )
    try:
        bridge = EndpointAPC(
            model, engine, checkpoint_revision="r1", tokenizer_fingerprint="t1"
        )
        bridge.publish(tokens, cache)
        now[0] = 2.0
        assert engine.spill_idle_entries(now=2.0) == 1
        restored, hit = bridge.restore(tokens + [15])
        assert hit.hit and restored.length == len(tokens)
        actual = model.decode(mx.array([[15]]), restored)
        assert float(mx.max(mx.abs(actual - expected)).item()) == 0
        if hasattr(hit.cache, "close"):
            hit.cache.close()
    finally:
        engine.close()


def test_failed_restore_releases_lookup_lease():
    from types import SimpleNamespace

    from mlx2.runtime.models.cache import ArraysCache

    class Lease(list):
        closed = False

        def close(self):
            self.closed = True

    leaf = ArraysCache(1)
    leaf[0] = mx.ones((1, 2), dtype=mx.float32)
    lease = Lease([leaf])

    class BrokenEngine:
        def lookup(self, _key, _tokens):
            return SimpleNamespace(hit=True, cache=lease)

    model = Model(Config.smoke())
    model.eval()
    bridge = EndpointAPC(
        model, BrokenEngine(), checkpoint_revision="r", tokenizer_fingerprint="t"
    )
    with pytest.raises(ValueError, match="metadata"):
        bridge.restore([1, 2])
    assert lease.closed


@pytest.mark.parametrize("damage", ["missing_layer", "offset", "boundary", "history"])
def test_incomplete_endpoint_cannot_be_published(damage):
    model = Model(replace(Config.smoke(), local_window=4))
    model.eval()
    tokens = list(range(1, 15))
    _, cache = model.prefill(mx.array([tokens]))
    if damage == "missing_layer":
        del cache.self_kv[0]
    elif damage == "offset":
        k, v, start = cache.cross_kv[0][0]
        cache.cross_kv[0][0] = (k, v, start + 1)
    elif damage == "boundary":
        cache.boundary = cache.boundary[:, :, :1]
    else:
        cache.ple_history = None
    engine = APCv2(max_size=4, layout_name="hysparse2-endpoint-v1")
    try:
        bridge = EndpointAPC(
            model, engine, checkpoint_revision="r", tokenizer_fingerprint="t"
        )
        with pytest.raises(ValueError, match="endpoint state"):
            bridge.publish(tokens, cache)
    finally:
        engine.close()


@pytest.mark.parametrize("damage", ["missing_layer", "offset", "boundary", "history", "alias", "orphan"])
def test_malformed_restored_geometry_releases_lease(damage, monkeypatch):
    import json
    from types import SimpleNamespace

    from mlx2.runtime.semantic_capsules import canonical_json

    class Lease(list):
        closed = False

        def close(self):
            self.closed = True

    model = Model(replace(Config.smoke(), local_window=4))
    model.eval()
    tokens = list(range(1, 15))
    _, cache = model.prefill(mx.array([tokens]))
    engine = APCv2(max_size=4, layout_name="hysparse2-endpoint-v1")
    try:
        bridge = EndpointAPC(
            model, engine, checkpoint_revision="r", tokenizer_fingerprint="t"
        )
        bridge.publish(tokens, cache)
        hit = engine.lookup(bridge.key(), tokens + [15])
        leaf = hit.cache[0]
        header = json.loads(bytes(leaf.cache[0][0].tolist()))
        if damage == "missing_layer":
            del header["groups"]["self_kv"]["0"]
        elif damage == "offset":
            header["groups"]["cross_kv"]["0"][0][2] += 1
        elif damage == "boundary":
            header["boundary"] = header["groups"]["cross_kv"]["0"][0][0]
        elif damage == "history":
            header["ple_history"] = None
        elif damage == "alias":
            header["groups"]["self_kv"]["0"][0][1] = header["groups"]["self_kv"]["0"][0][0]
        else:
            extra = mx.zeros_like(leaf.cache[1])
            leaf.cache.append(extra)
            header["arrays"].append([list(extra.shape), str(extra.dtype)])
        leaf.cache[0] = mx.array(list(canonical_json(header)), dtype=mx.uint8)[None]
        lease = Lease([leaf])
        monkeypatch.setattr(
            engine,
            "lookup",
            lambda *_: SimpleNamespace(
                hit=True, cache=lease, cached_tokens=len(tokens)
            ),
        )
        with pytest.raises(ValueError, match="endpoint state"):
            bridge.restore(tokens)
        assert lease.closed
        if hasattr(hit.cache, "close"):
            hit.cache.close()
    finally:
        engine.close()
