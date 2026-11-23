"""scripts/amoeba_staged_window.py gating rules (CPU, no GPU, no radio)."""
import datetime as dt
import importlib.util
import os
import time
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "amoeba_staged_window", Path(__file__).resolve().parents[1] / "scripts" / "amoeba_staged_window.py")
S = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(S)

T = dt.time


def test_night_window_wraps_midnight_and_honours_latest_start():
    s, e, latest = T(23, 0), T(7, 0), T(5, 30)
    assert S.in_window(T(23, 30), s, e, latest)
    assert S.in_window(T(3, 0), s, e, latest)
    assert not S.in_window(T(6, 0), s, e, latest)      # inside the night, past the latest start
    assert not S.in_window(T(12, 0), s, e, latest)
    assert not S.in_window(T(22, 59), s, e, latest)
    assert S.in_window(T(6, 0), s, e)                   # no latest start: the whole night
    assert not S.in_window(T(7, 0), s, e)


def test_day_window_without_wrap():
    assert S.in_window(T(10, 0), T(9, 0), T(17, 0))
    assert not S.in_window(T(17, 0), T(9, 0), T(17, 0))


def test_live_waiters_skip_dead_stale_self_and_malformed(tmp_path):
    me = os.getpid()
    live = tmp_path / f"sess.job.{me}.{int(time.time())}"
    live.touch()
    stale = tmp_path / f"pierre-hold.hy4.{me}.1795217026"
    stale.touch()
    old = time.time() - 3600
    os.utime(stale, (old, old))
    dead = tmp_path / "sess.gone.999999.1"
    dead.touch()
    (tmp_path / "not-a-waiter").touch()
    assert S.live_waiters(waiters=tmp_path) == [live.name]
    assert S.live_waiters(waiters=tmp_path, me=me) == []      # its own wrapper never blocks it
    assert S.live_waiters(waiters=tmp_path / "missing") == []
