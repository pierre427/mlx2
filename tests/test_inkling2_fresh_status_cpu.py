"""``_fresh_status`` in scripts/research/inkling2_attn_decode_serve.py, CPU only.

The serving worker republishes ``/v1/status`` at most once a second, so a
read right after a response can be up to a second stale (the 2026-11-24
step-count probe's missing decode steps). The helper waits for two
``host_snapshot_timing.collections`` increments after its first read.
"""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "research"))

import inkling2_attn_decode_serve as serve  # noqa: E402


def _fake(monkeypatch, statuses):
    reads = iter(statuses)
    seen = []

    def get(port, path, token=None):
        assert path == "/v1/status"
        status = next(reads)
        seen.append(status)
        return status

    monkeypatch.setattr(serve, "_get", get)
    monkeypatch.setattr(serve.time, "sleep", lambda s: None)
    return seen


def _status(collections, calls):
    return {"host_snapshot_timing": {"collections": collections},
            "execution": {"kernel_counters": {"attn_decode": {"calls": calls}}}}


def test_waits_for_two_collections(monkeypatch):
    seen = _fake(monkeypatch, [_status(10, 100), _status(11, 140), _status(12, 148)])
    status, fresh = serve._fresh_status(8000)
    assert fresh is True
    assert status["execution"]["kernel_counters"]["attn_decode"]["calls"] == 148
    assert len(seen) == 3


def test_reports_stale_on_timeout(monkeypatch):
    _fake(monkeypatch, itertools.repeat(_status(10, 100)))
    status, fresh = serve._fresh_status(8000, timeout=0.05)
    assert fresh is False
    assert status["host_snapshot_timing"]["collections"] == 10


def test_no_collection_count_is_not_fresh(monkeypatch):
    _fake(monkeypatch, [{"execution": {}}])
    status, fresh = serve._fresh_status(8000)
    assert fresh is False and status == {"execution": {}}
