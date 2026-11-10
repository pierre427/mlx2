"""Bug sweep 2026-11-09, serving lane: World identity/cap/number handling and the
companion request lane. CPU only; no model, server or Metal kernel."""
import copy
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx2.desktop_world import MAX_SAVED_WORLDS, Worlds, validate_world

SRC = str(Path(__file__).resolve().parents[1] / "src")


def _world(worlds, identity, name):
    document = copy.deepcopy(worlds.active)
    document["id"], document["name"] = identity, name
    return document


def test_srv1_world_ids_are_case_safe_on_case_insensitive_volumes(tmp_path):
    worlds = Worlds(SimpleNamespace(home=tmp_path))
    worlds.dispatch({"command": "world_save", "world": _world(worlds, "alpha", "Harbour study")})
    other = _world(worlds, "ALPHA", "Orchard plan")
    # Refused before approval (check) and at apply time; alpha's file is untouched.
    with pytest.raises(ValueError, match="different letter case"):
        worlds.check({"command": "world_save", "world": other})
    with pytest.raises(ValueError, match="different letter case"):
        worlds.dispatch({"command": "world_save", "world": other})
    with pytest.raises(ValueError, match="does not exist"):
        worlds.dispatch({"command": "world_load", "world_id": "ALPHA"})
    assert json.loads((tmp_path / "worlds/alpha.json").read_text())["name"] == "Harbour study"
    listed = worlds.snapshot()
    assert {"id": "alpha", "name": "Harbour study"} in listed["worlds"] and listed["errors"] == []
    assert worlds.dispatch({"command": "world_load", "world_id": "alpha"})["active"]["name"] == "Harbour study"
    # The reserved application file names are reserved in every spelling.
    for reserved in ("STATE", "State", "Assets"):
        with pytest.raises(ValueError, match="reserved"):
            validate_world(_world(worlds, reserved, "Lighthouse"))
    # An existing mixed-case id keeps working under its own spelling (no rename).
    worlds.dispatch({"command": "world_save", "world": _world(worlds, "Beta", "Granary")})
    worlds.dispatch({"command": "world_save", "world": _world(worlds, "Beta", "Granary at dusk")})
    assert json.loads((tmp_path / "worlds/Beta.json").read_text())["name"] == "Granary at dusk"


def test_srv2_world_save_under_a_new_id_respects_the_saved_world_cap(tmp_path):
    worlds = Worlds(SimpleNamespace(home=tmp_path))
    folder = tmp_path / "worlds"
    have = sum(1 for p in folder.glob("*.json") if p.name != "state.json")
    for index in range(MAX_SAVED_WORLDS - have):
        document = _world(worlds, f"filler{index}", f"Filler {index}")
        (folder / f"filler{index}.json").write_text(json.dumps(document))
    fresh = _world(worlds, "newcomer", "Over the cap")
    with pytest.raises(ValueError, match="At most"):
        worlds.check({"command": "world_save", "world": fresh})
    with pytest.raises(ValueError, match="At most"):
        worlds.dispatch({"command": "world_save", "world": fresh})
    assert not (folder / "newcomer.json").exists()
    # Saving over an existing world is not a new file and still succeeds.
    renamed = _world(worlds, worlds.active["id"], "Renamed active")
    assert worlds.dispatch({"command": "world_save", "world": renamed})["active"]["name"] == "Renamed active"


def test_srv3_out_of_float_range_integers_are_set_aside_not_raised(tmp_path):
    worlds = Worlds(SimpleNamespace(home=tmp_path))
    folder = tmp_path / "worlds"
    bad = _world(worlds, "bigint", "Comet")
    bad["objects"][0]["position"] = [10 ** 400, 0, 0]
    (folder / "bigint.json").write_text(json.dumps(bad))
    snapshot = worlds.snapshot()
    assert any(error.startswith("Could not read bigint.json") for error in snapshot["errors"])
    for field, value in (("theme", {"background": "#071321", "accent": "#38CBFF", "exposure": 10 ** 400}),
                         ("transitions", {"duration": -(10 ** 400), "style": "ease"})):
        with pytest.raises(ValueError):
            validate_world({**_world(worlds, "edge", "Edge"), field: value})
    # The same value in the active state is set aside at startup.
    state = json.loads((folder / "state.json").read_text())
    state["active"]["objects"][0]["animation"]["speed"] = 10 ** 400
    (folder / "state.json").write_text(json.dumps(state))
    restarted = Worlds(SimpleNamespace(home=tmp_path))
    assert restarted.startup_notice and list(folder.glob("state.json.invalid-*"))
    # A deeply nested state file is set aside the same way.
    (folder / "state.json").write_text('{"active": ' + "[" * 200000 + "]" * 200000 + "}")
    assert Worlds(SimpleNamespace(home=tmp_path)).startup_notice
    assert len(list(folder.glob("state.json.invalid-*"))) == 2


_COMPANION = textwrap.dedent('''
    import json, signal, sys, threading, time, types
    from types import SimpleNamespace
    from mlx2 import desktop
    from mlx2.desktop_lane import LANE
    mode = sys.argv[1]
    ran = []
    inside = threading.Event()
    class Controller:
        def __init__(self, *args):
            pass
        def dispatch(self, request):
            ran.append(request["command"])
            assert "bridge_deadline" not in request, request
            if request["command"] == "busy":
                time.sleep(0.6)
            if request["command"] == "network":
                with LANE.released():
                    inside.set()
                    time.sleep(0.5)
            return {"command": request["command"]}
        def close(self):
            print(json.dumps({"ran": ran}), file=sys.stderr, flush=True)
    sys.modules["mlx2.desktop_control"] = types.SimpleNamespace(DesktopController=Controller)
    if mode == "terminate":
        original, calls = LANE.ticket, [0]
        def ticket():
            value = original()
            if threading.current_thread() is threading.main_thread():
                calls[0] += 1
                if calls[0] == 2:
                    # The first request is now waiting on the network outside the lane.
                    assert inside.wait(5)
                    signal.raise_signal(signal.SIGTERM)  # between ticket() and worker.start()
            return value
        LANE.ticket = ticket
    started = time.monotonic()
    desktop._serve_companion(SimpleNamespace(home=sys.argv[2], trash_helper="/never-executed"), sys.stdout, shutdown_grace=5.0)
    print(json.dumps({"elapsed": time.monotonic() - started}), file=sys.stderr, flush=True)
''')


def _run_companion(mode, tmp_path, lines):
    environment = {**os.environ, "PYTHONPATH": SRC}
    process = subprocess.Popen([sys.executable, "-c", _COMPANION, mode, str(tmp_path)], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment)
    for line in lines:
        process.stdin.write(json.dumps(line) + "\n")
        process.stdin.flush()
    out, err = process.communicate(timeout=30)
    replies = {reply["id"]: reply for reply in map(json.loads, out.splitlines())}
    notes = [json.loads(line) for line in err.splitlines() if line.startswith("{")]
    return replies, notes


def test_srv4_request_that_expires_while_queued_is_refused_unrun(tmp_path):
    now = time.time()
    replies, notes = _run_companion("serve", tmp_path, [
        {"id": "busy", "command": "busy", "bridge_deadline": now + 60},
        {"id": "stale", "command": "world_create", "bridge_deadline": now + 0.2},
        {"id": "garbage", "command": "agent_delegate", "bridge_deadline": "soon"},
        {"id": "fresh", "command": "media_generate", "bridge_deadline": now + 60},
        {"id": "legacy", "command": "snapshot"},
    ])
    assert replies["busy"]["ok"] and replies["fresh"]["ok"] and replies["legacy"]["ok"], replies
    assert not replies["stale"]["ok"] and "not run" in replies["stale"]["error"]
    assert not replies["garbage"]["ok"] and "not run" in replies["garbage"]["error"]
    ran = next(note["ran"] for note in notes if "ran" in note)
    assert ran == ["busy", "media_generate", "snapshot"]


def test_srv_termination_between_ticket_and_start_does_not_stall_shutdown(tmp_path):
    replies, notes = _run_companion("terminate", tmp_path, [
        {"id": "network", "command": "network"},
        {"id": "second", "command": "second"},
    ])
    elapsed = next(note["elapsed"] for note in notes if "elapsed" in note)
    # Before the fix the second ticket was orphaned: the network dispatch could
    # not rejoin the lane and shutdown waited the whole 5 s grace.
    assert elapsed < 4.0, (elapsed, replies)
    # The request handed off at the signal still got its single answer.
    assert "second" in replies, replies
