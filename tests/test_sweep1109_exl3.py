"""Bug sweep 2026-11-09, lane exl3: research converter and harness repairs (CPU, no checkpoint, no Metal)."""
import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/research/ds41_exl3"))
import convert_bank as CB  # noqa: E402
import exl3_worker as XW  # noqa: E402
import bench_encoder as BE  # noqa: E402

spec = importlib.util.spec_from_file_location("ablation_ladder_sweep1109", ROOT / "scripts/research/ds41_atlas/ablation_ladder.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)


def row(layer, expert, words, ok=True, err=0.17, selection=None):
    r = {"layer": layer, "expert": expert, "words": words, "decode_matches_encoder": ok, "expert_rel_err": err,
         "file": f"w{words}/layers.{layer}.ffn.experts.{expert}.safetensors"}
    if selection:
        r["selection"] = selection
    return r


def put(out, layer, expert, words):
    p = out / f"w{words}" / f"layers.{layer}.ffn.experts.{expert}.safetensors"
    p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(os.urandom(64))


def manifest(out):
    return json.loads((out / "manifest.json").read_text())


def test_exl1_failed_check_is_not_done_nor_in_manifest(tmp_path):
    put(tmp_path, 3, 7, 48); put(tmp_path, 3, 9, 48); put(tmp_path, 4, 2, 48)
    (tmp_path / "results.json").write_text(json.dumps(
        [row(3, 7, 48), row(3, 9, 48, ok=False, err=float("nan")), row(4, 2, 48, ok=True, err=float("nan"))]))
    assert sorted(CB.done_rows(tmp_path, 48)) == [(3, 7, 48)]
    CB.write_manifest(tmp_path, 48)
    assert sorted(manifest(tmp_path)["files"]) == ["w48/layers.3.ffn.experts.7.safetensors"]


def test_exl2_resume_and_manifest_keyed_by_words(tmp_path):
    put(tmp_path, 3, 7, 32)
    (tmp_path / "results_a.json").write_text(json.dumps([row(3, 7, 32)]))
    assert (3, 7, 48) not in CB.done_rows(tmp_path, 48)      # a w48 run still plans the expert
    put(tmp_path, 3, 7, 48)
    (tmp_path / "results_a.json").write_text(json.dumps([row(3, 7, 48)]))
    (tmp_path / "results_b.json").write_text(json.dumps([row(3, 7, 32)]))
    CB.write_manifest(tmp_path, 48)
    assert list(manifest(tmp_path)["files"]) == ["w48/layers.3.ffn.experts.7.safetensors"]
    CB.write_manifest(tmp_path, 32)
    assert list(manifest(tmp_path)["files"]) == ["w32/layers.3.ffn.experts.7.safetensors"]


def test_exl3_manifest_selection_describes_every_contributing_selection(tmp_path):
    for e in (1, 2, 5):
        put(tmp_path, 0, e, 48)
    (tmp_path / "results.json").write_text(json.dumps([row(0, 1, 48)]))   # legacy row, no selection recorded
    (tmp_path / "results_m3.json").write_text(json.dumps([row(0, 2, 48, selection="top 50 per layer")]))
    (tmp_path / "results_m5.json").write_text(json.dumps([row(0, 5, 48, selection="explicit set union.json")]))
    CB.write_manifest(tmp_path, 48)
    sel = manifest(tmp_path)["selection"]
    assert isinstance(sel, str)
    assert "top 50 per layer: 1 rows" in sel and "explicit set union.json: 1 rows" in sel
    assert "unrecorded selection (results.json): 1 rows" in sel


def _fake_worker(reply_meta, close_without_reply=False):
    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)

    def serve():
        conn, _ = srv.accept()
        with conn:
            XW.recv_msg(conn); XW.recv_msg(conn)
            if not close_without_reply:
                XW.send_msg(conn, {"meta": np.array(json.dumps(reply_meta))})
        srv.close()
    threading.Thread(target=serve, daemon=True).start()
    return f"127.0.0.1:{srv.getsockname()[1]}"


def _ctx():
    rng = np.random.default_rng(11)
    return {"W": {k: rng.standard_normal((4, 3)).astype(np.float32) for k in ("gate", "up", "down")},
            "h_in": np.eye(3), "h_mid": np.eye(4), "layer": 2, "expert": 5}


def test_exl4_lean_reply_without_decode_match_fails(monkeypatch):
    monkeypatch.delenv("MLX2_EXL3_WORKER_TOKEN", raising=False); monkeypatch.delenv("MLX2_EXL3_WORKER_TOKEN_FILE", raising=False)
    addr = _fake_worker({"ok": True, "host": "w", "wait_s": 0, "encode_s": 1, "stats": [{}, {}, {}]})
    with pytest.raises(RuntimeError, match="remote decode check"):
        CB.remote_encode(addr, _ctx(), 48, 32, lean=True)
    addr = _fake_worker({"ok": True, "stats": [{}, {}, {}], "remote_decode_match": [True, True]})
    with pytest.raises(RuntimeError, match="remote decode check"):
        CB.remote_encode(addr, _ctx(), 48, 32, lean=True)
    # The lean row's decode_matches_encoder is the worker's check, not the local file round-trip.
    r = CB.lean_row({"decode_matches_encoder": True}, [{"stats": {"remote_decode_match": [True, False, True]}}])
    assert r["decode_matches_encoder"] is False and r["saved_file_matches_reply"] is True
    assert not CB.checked(dict(r, expert_rel_err=0.1))


def test_exl5_zero_resident_passes_quote_no_ttft(tmp_path):
    tok = [{"token": "x", "logprob": 0.0, "top": [("x", 0.0), ("y", -9.0)]}]
    ex = {"v41_expert_cache_misses": 5, "v41_expert_demand_wait_ns": 0}
    for arm in ("base", "exl3"):
        (tmp_path / arm).mkdir()
        L._write(tmp_path / arm / "replies.json", {"p": {
            "reply": "the answer is 41", "finish": "stop", "tokens": tok, "decode_tok_s": None, "ttft_s": 41.7,
            "completion_tokens": 1, "experts": ex, "resident_passes": [0, 3], "tokens_stable": True}})
        L._write(tmp_path / arm / "done.json", {})
    rows = L.summarize(tmp_path, ["base", "exl3"], {"p": ("any", lambda _r: True)})
    assert rows[1]["median_ttft_s"] is None and rows[1]["median_decode_tok_s"] is None


def test_exl6_bench_time_reports_every_projection_on_v2():
    shared = {"prep_s": 1.0, "quant_s": 5.0}
    assert BE._timing([{"stats": {"timing": shared}}] * 3) == {"timing": shared, "timing_scope": "expert"}
    units = [{"stats": {"timing": {"load_s": 0.1 * i, "unit_wall_s": 2.0 + i}}} for i in range(3)]
    t = BE._timing(units)
    assert t["timing_scope"] == "per_projection" and [u["unit_wall_s"] for u in t["timing"]] == [2.0, 3.0, 4.0]


def _dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"]); p.wait()
    return p.pid


def test_exl7_foreign_claim_reported(tmp_path, capsys):
    (tmp_path / "claims").mkdir()
    (tmp_path / "claims" / "3.8.w48").write_text("other-host.example 4242")
    skipped = []
    assert CB.claim(tmp_path, 3, 8, skipped, words=48) is False
    assert skipped == [(3, 8, "other-host.example 4242")]
    left = CB.unconverted(tmp_path, [(3, 8, 0, 10)], 48, skipped)
    assert left == [(3, 8, 0, 10)] and "other-host.example 4242" in capsys.readouterr().out


def test_exl7_takeover_of_a_dead_claim_is_single(tmp_path):
    host = socket.gethostname()
    for rnd in range(30):
        (tmp_path / "claims").mkdir(exist_ok=True)
        (tmp_path / "claims" / f"1.{rnd}.w48").write_text(f"{host} {_dead_pid()}")
        wins, gate = [], threading.Barrier(6)

        def go():
            gate.wait(); wins.append(CB.claim(tmp_path, 1, rnd, words=48))
        ts = [threading.Thread(target=go) for _ in range(6)]
        [t.start() for t in ts]; [t.join() for t in ts]
        assert wins.count(True) == 1, (rnd, wins)
        assert (tmp_path / "claims" / f"1.{rnd}.w48").read_text() == f"{host} {os.getpid()}"
        assert not (tmp_path / "claims" / f"1.{rnd}.w48.takeover").exists()


def test_exl7_permission_error_means_alive(tmp_path, monkeypatch):
    (tmp_path / "claims").mkdir()
    (tmp_path / "claims" / "2.4.w48").write_text(f"{socket.gethostname()} 1")

    def kill(pid, sig):
        raise PermissionError(1, "Operation not permitted")
    monkeypatch.setattr(CB.os, "kill", kill)
    assert CB.claim(tmp_path, 2, 4, words=48) is False
    assert (tmp_path / "claims" / "2.4.w48").read_text().endswith(" 1")


# -- review follow-ups ---------------------------------------------------------

def test_review_stale_takeover_lock_of_a_dead_local_process_is_reclaimed(tmp_path):
    host = socket.gethostname()
    claims = tmp_path / "claims"; claims.mkdir()
    (claims / "5.17.w48").write_text(f"{host} {_dead_pid()}")
    (claims / "5.17.w48.takeover").write_text(f"{host} {_dead_pid()}")   # taker killed mid-takeover
    skipped = []
    assert CB.claim(tmp_path, 5, 17, skipped, words=48) is True and skipped == []
    assert (claims / "5.17.w48").read_text() == f"{host} {os.getpid()}"
    assert not (claims / "5.17.w48.takeover").exists()
    # A lock held by a live local process, or by another host, is still respected.
    (claims / "5.18.w48").write_text(f"{host} {_dead_pid()}")
    for holder in (f"{host} {os.getppid()}", "other-host.example 4242"):
        (claims / "5.18.w48.takeover").write_text(holder)
        assert CB.claim(tmp_path, 5, 18, skipped, words=48) is False
        assert (claims / "5.18.w48.takeover").read_text() == holder


def test_review_claims_are_keyed_by_words(tmp_path):
    (tmp_path / "claims").mkdir()
    (tmp_path / "claims" / "3.8.w32").write_text("other-host.example 4242")   # a w32 run holds this expert
    assert CB.claim(tmp_path, 3, 8, words=48) is True
    assert CB.claim(tmp_path, 3, 8, [], words=32) is False


def test_review_worker_drops_a_silent_peer_before_auth():
    srv = XW.Server(("127.0.0.1", 0), XW.Handler); srv.token = "s3cret-7f2"; srv.auth_timeout_s = 0.3
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with socket.create_connection(srv.server_address, timeout=5) as c:
            assert c.recv(1) == b""   # closed by the worker, not held open
    finally:
        srv.shutdown(); srv.server_close()
    assert XW.AUTH_TIMEOUT_S <= 30


def test_exl8_frame_cap_none_reply_and_token(monkeypatch):
    a, b = socket.socketpair()
    with a, b:
        a.sendall((XW.MAX_FRAME + 1).to_bytes(8, "little"))
        with pytest.raises(ConnectionError, match="limit"):
            XW.recv_msg(b)
    monkeypatch.delenv("MLX2_EXL3_WORKER_TOKEN_FILE", raising=False)
    monkeypatch.setenv("MLX2_EXL3_WORKER_TOKEN", "")
    with pytest.raises(ConnectionError, match="without a reply"):
        CB.remote_encode(_fake_worker({}, close_without_reply=True), _ctx(), 48, 32)
    with pytest.raises(SystemExit):
        XW.check_bind("192.0.2.10", "")
    XW.check_bind("127.0.0.1", ""); XW.check_bind("192.0.2.10", "s3cret-7f2")
    assert XW.check_auth({"token": np.array("s3cret-7f2")}, "s3cret-7f2")
    assert not XW.check_auth({"token": np.array("guess")}, "s3cret-7f2")
    assert not XW.check_auth({"w0": np.zeros(1)}, "s3cret-7f2")
    # A worker with a token drops a client that sends the wrong one, before reading its request.
    srv = XW.Server(("127.0.0.1", 0), XW.Handler); srv.token = "s3cret-7f2"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("MLX2_EXL3_WORKER_TOKEN", "guess")
        with pytest.raises((ConnectionError, OSError)):
            CB.remote_encode(f"127.0.0.1:{srv.server_address[1]}", _ctx(), 48, 32)
    finally:
        srv.shutdown(); srv.server_close()
    tokfile = Path(os.environ.get("TMPDIR", "/tmp")) / f"sweep1109-exl3-token-{os.getpid()}"
    tokfile.write_text("from-file-91\n")
    try:
        monkeypatch.delenv("MLX2_EXL3_WORKER_TOKEN"); monkeypatch.setenv("MLX2_EXL3_WORKER_TOKEN_FILE", str(tokfile))
        assert XW.load_token() == "from-file-91"
    finally:
        tokfile.unlink()
