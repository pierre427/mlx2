"""The GPU diagnostic's digests compare committed state, including raw keys."""

import runpy
from pathlib import Path

import mlx.core as mx
import pytest
from test_external_dflash2_cpu import drain, generator, tiny

from mlx2.runtime.apc_v2 import APCKey, APCv2
from mlx2.runtime.drafters.fold import FOLDKVCache, configure
from mlx2.runtime.sample_utils import LaneRNG


@pytest.fixture
def probe(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return runpy.run_path(str(scripts / "probe_fold_phases.py"))


def test_probe_digest_ignores_spare_capacity_but_binds_raw_keys_and_boundary(probe):
    cache = FOLDKVCache()
    values = mx.ones((1, 1, 3, 4))
    cache.append_raw_keys(values)
    cache.update_and_fetch(values, values)
    record = probe["cache_record"]
    expected = record([cache])
    cache.raw_keys[..., 3:, :] = 7
    cache.keys[..., 3:, :] = 9
    assert record([cache]) == expected
    cache.raw_keys[..., 1, :] = 2
    assert record([cache]) != expected
    cache.raw_keys[..., 1, :] = 1
    assert record([cache]) == expected
    cache.offset = 2
    assert record([cache]) != expected
    assert record([cache, cache]) != expected


def test_probe_digests_match_actual_fold_rollback_and_retry(probe, monkeypatch):
    def run(fail):
        target, draft = tiny()
        configure(draft, {"window": 3, "alpha": 2, "sliding_layers": "fold"})
        batch = generator(target, draft)
        batch.insert(
            [list(range(1, 14))],
            max_tokens=[4],
            sampling_configs=[{"sampling_temp": 0.8}],
            lane_rngs=[LaneRNG(53)],
        )
        lane = batch.lanes[0]
        while lane.anchor is None:
            batch._prefill(lane)

        def snapshot():
            return {
                "target": probe["cache_record"](lane.cache),
                "draft": probe["cache_record"](lane.draft_cache),
                "tail": probe["state_record"](lane.tail),
                "rng": probe["state_record"](lane.rng.snapshot()),
                "history": list(lane.history),
            }

        try:
            if fail:
                before = snapshot()
                original = batch._propose

                def injected(*args, **kwargs):
                    original(*args, **kwargs)
                    raise RuntimeError("post-proposal")

                with monkeypatch.context() as patch:
                    patch.setattr(batch, "_propose", injected)
                    with pytest.raises(RuntimeError, match="post-proposal"):
                        batch._round([lane])
                assert snapshot() == before
                assert not lane.ready
            tokens, ends = drain(batch)
            return tokens, probe["terminal_record"](ends[0])
        finally:
            batch.close()

    assert run(True) == run(False)


def test_probe_checkpoint_digest_survives_fresh_disk_owner(probe, tmp_path):
    target, draft = tiny()
    configure(draft, {"window": 3, "sliding_layers": "fold"})
    batch = generator(target, draft)
    batch.insert([list(range(1, 14))], max_tokens=[3], lane_rngs=[LaneRNG(53)])
    _, ends = drain(batch)
    end = ends[0]
    expected = probe["checkpoint_record"](end.prompt_cache, end.cache_sidecar)
    key = APCKey("fold-probe", revision="test", cache_layout_fingerprint="test")
    options = {
        "layout_name": "test",
        "idle_disk_seconds": 1,
        "idle_disk_dir": str(tmp_path),
        "persist_dir": str(tmp_path),
        "persist_identity": key,
    }
    apc = APCv2(**options)
    try:
        apc.store(key, end.all_tokens, end.prompt_cache, sidecar=end.cache_sidecar)
        apc.close(persist_resident=True, time_budget_seconds=10, release_memory=False)
        apc = APCv2(**options)
        hit = apc.lookup(key, end.all_tokens + [end.token])
        try:
            assert hit.hit
            assert probe["checkpoint_record"](hit.cache, hit.sidecar) == expected
        finally:
            if hasattr(hit.cache, "close"):
                hit.cache.close()
    finally:
        apc.close(release_memory=False)
        batch.close()
