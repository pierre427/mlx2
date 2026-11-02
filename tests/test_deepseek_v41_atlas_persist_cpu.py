"""Advisory expert-atlas persistence must not wedge the provider (CPU only)."""
from collections import OrderedDict
from unittest import mock

import pytest

from mlx2.adapters import deepseek_v41_mlx_weights as W
from mlx2.adapters import deepseek_v41_source_lfu as L


def _store(tmp_path):
    store = L.LFUSourceExpertStore.__new__(L.LFUSourceExpertStore)
    pool = mock.Mock()
    pool.counters.return_value = {"weight_read_inflight": 0}
    store.__dict__.update(
        atlas_path=tmp_path / "atlas.json", binding="B", index=L.LFUIndex(),
        _atlas_saved=L.Counter(), _atlas_saved_at=0, stats={}, pool=pool,
        _synchronize=lambda: None, _leased={}, _pending_uploads={}, _pending_host={},
        _cache=OrderedDict(), _cache_bytes=0, _active=False)
    store.index.counts[(0, 1)] += 3
    return store


def _enospc(*args, **kwargs):
    raise OSError(28, "No space left on device")


def test_failed_atlas_write_leaves_no_temporary_and_next_persist_succeeds(tmp_path):
    store = _store(tmp_path)
    with mock.patch.object(L.json, "dump", _enospc), pytest.raises(OSError):
        store.persist_atlas()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["atlas.json.lock"]
    store.index.counts[(0, 2)] += 1
    store.persist_atlas()
    rows = L.read_atlas(tmp_path / "atlas.json", "B")["counts"]
    assert {(r["layer"], r["expert"]): r["token_selections"] for r in rows} == {(0, 1): 3, (0, 2): 1}


def test_stale_own_temporary_does_not_block_persist(tmp_path):
    store = _store(tmp_path)
    stale = tmp_path / f"atlas.json.{L.os.getpid()}.tmp"
    stale.write_text("partial")
    store.persist_atlas()
    assert not stale.exists() and (tmp_path / "atlas.json").exists()


def test_end_context_atlas_failure_is_advisory(tmp_path):
    store = _store(tmp_path)
    with mock.patch.object(L.json, "dump", _enospc):
        store.end_context()
    assert store.stats["atlas_persist_failures"] == 1
    assert store.atlas_persist_error["type"] == "OSError"


def test_atlas_failure_at_close_does_not_quarantine_weights(tmp_path):
    store = _store(tmp_path)
    weights = W.NativeV41Weights.__new__(W.NativeV41Weights)
    read_pool = mock.Mock()
    read_pool.counters.return_value = {"weight_read_inflight": 0}
    weights.__dict__.update(expert_stream=store, _read_pool=read_pool, _dense={"x": 1},
                            _quant={}, _pending_dense={}, _pending_dense_host=None)
    with mock.patch.object(L.json, "dump", _enospc):
        weights.close()
    assert id(weights) not in W._UPLOAD_QUARANTINE
    W._refuse_quarantined_uploads()
    assert store.atlas_persist_error["type"] == "OSError"


@pytest.mark.parametrize("malformed", [
    [],
    {"schema": "mlx2.expert-frequency-atlas.v1", "source": "B", "clock": 0},
    {"schema": "mlx2.expert-frequency-atlas.v1", "source": "B", "clock": 0,
     "counts": None},
    {"schema": "mlx2.expert-frequency-atlas.v1", "source": "B", "clock": 0,
     "counts": [{}]},
    {"schema": "mlx2.expert-frequency-atlas.v1", "source": "B", "clock": 0,
     "counts": [None]},
    {"schema": "mlx2.expert-frequency-atlas.v1", "source": "B", "clock": 0,
     "counts": [{"layer": [], "expert": 1, "token_selections": 3, "last_access": 0}]},
])
def test_malformed_atlas_is_a_schema_error_and_remains_advisory(tmp_path, malformed):
    store = _store(tmp_path)
    store.atlas_path.write_text(L.json.dumps(malformed))
    with pytest.raises(ValueError):
        L.read_atlas(store.atlas_path, "B")
    store.end_context()
    assert store.stats["atlas_persist_failures"] == 1
    assert store.atlas_persist_error["type"] == "ValueError"
    assert not store._atlas_saved
    # A later repair can persist all observations: a failed advisory write
    # must not mark the delta as saved.
    store.atlas_path.unlink()
    store.persist_atlas()
    assert L.read_atlas(store.atlas_path, "B")["counts"][0]["token_selections"] == 3
