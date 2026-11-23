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


def test_latest_start_before_midnight_also_cuts_off_the_following_morning():
    assert S.in_window(T(23, 10), T(23), T(7), T(23, 30))
    assert not S.in_window(T(1), T(23), T(7), T(23, 30))


def test_prelaunch_refusal_is_distinct_from_a_child_exiting_75(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    a = SimpleNamespace(log_dir=str(tmp_path), tree=str(tmp_path), session="test", cpg_task="test")
    receipt = {"rc": None, "child_pid": None, "refusal": "lease held"}

    def fake_cpg(cmd, **kwargs):
        log = Path(cmd[cmd.index("--out") + 1])
        Path(f"{log}.done").write_text(json.dumps(receipt))
        return 75

    monkeypatch.setattr(S.subprocess, "call", fake_cpg)
    assert S.run_leased(a, "stage", ["child"], {}) is None
    receipt.update(rc=75, child_pid=123, refusal=None)
    assert S.run_leased(a, "stage", ["child"], {}) == 75


def test_relative_paths_name_same_files_when_child_changes_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tree").mkdir()
    (tmp_path / "build").mkdir()
    (tmp_path / "build/run.json").touch()
    calls = iter([{"map": "pending"}, {"map": "done"}, {"map": "done"}])

    def fake_status(py, out, env):
        assert py == str(tmp_path / "venv/python")
        assert out == str(tmp_path / "build")
        return next(calls)

    def fake_lease(a, label, argv, env):
        assert a.tree == str(tmp_path / "tree")
        assert a.log_dir == str(tmp_path / "logs")
        assert argv[argv.index("--out") + 1] == str(tmp_path / "build")
        return 0

    monkeypatch.setattr(S, "stage_status", fake_status)
    monkeypatch.setattr(S, "wait_turn", lambda a: None)
    monkeypatch.setattr(S, "run_leased", fake_lease)
    assert S.main(["--tree", "tree", "--out", "build", "--python", "venv/python",
                   "--label", "test", "--session", "test", "--cpg-task", "test",
                   "--log-dir", "logs", "--yield-s", "0"]) == 0


def test_stage_status_honors_the_recorded_map_only_plan(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    (tmp_path / "build.json").write_text(json.dumps({"stages": ["map"]}))
    result = {"stages": {"core": {"status": None}, "map": {"status": "done"}, "gates": {"status": None}}}
    monkeypatch.setattr(S.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=json.dumps(result)))
    assert S.stage_status("python", str(tmp_path), {}) == {"map": "done"}
