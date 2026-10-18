from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from mlx2.fabric import FabricError
from mlx2.fabric.link import LinkBudget, LinkReceipt
from mlx2.fabric.placement import PeerView, place
from mlx2.fabric.registry import list_local_servers, pid_alive, register_local_server


def _peer(nid, *, state="alive", queue=0, free=100, rtt=2.0):
    link = LinkReceipt(nid, "tls1.3/tcp", rtt, rtt, 3, 1.0, 1.0, 64, 1.0)
    return PeerView(
        nid, state, {"m"}, queue, free, "Apple_M", "runtime", link, nid, (link,)
    )


def test_local_wins_and_receipt_states_are_truthful():
    result = place(
        request_id="r",
        model="m",
        local_nid="local",
        local_models={"m"},
        peers=[_peer("remote")],
        link_budget=LinkBudget(),
    )
    receipt = result.receipt()
    assert result.kind == "local"
    assert receipt["states"] == {
        "implemented": True,
        "qualified": False,
        "selected": True,
        "observed_used": False,
    }
    assert receipt["exactness"] == "whole_model_one_node" and not receipt["approximate"]


def test_remote_ranking_and_ineligible_peer_reasons():
    peers = [
        _peer("queue", queue=2, free=1000, rtt=1),
        _peer("winner", queue=1, free=10, rtt=9),
        _peer("dead", state="dead", queue=0),
        _peer("revoked", state="revoked", queue=0),
    ]
    result = place(
        request_id="r",
        model="m",
        local_nid="local",
        local_models=set(),
        peers=peers,
        link_budget=LinkBudget(),
    )
    assert result.kind == "remote_resident" and result.served_by.nid == "winner"
    rejected = {
        (item.get("nid"), item["reason"])
        for item in result.candidates
        if item["verdict"] == "rejected"
    }
    assert ("dead", "dead") in rejected and ("revoked", "revoked") in rejected


def test_refusal_lists_unimplemented_and_approximate_options():
    result = place(
        request_id="r",
        model="missing",
        local_nid="local",
        local_models=set(),
        peers=[],
        link_budget=LinkBudget(),
    )
    assert (
        result.kind == "refused" and result.reason == "no node has the model resident"
    )
    options = {item["option"]: item for item in result.candidates}
    assert options["pipeline"]["verdict"] == "not_implemented"
    assert options["pipeline"]["reason"] == "not implemented"
    assert options["streaming"]["verdict"] == "not_implemented"
    assert options["approximate_prefill_offload"] == {
        "option": "approximate_prefill_offload",
        "verdict": "refused",
        "reason": "unqualified approximate state operation (AGENTS.md); not selectable",
    }


def test_refusal_pipeline_reason_uses_best_alive_peers_link_history():
    best = _peer("best", queue=0, rtt=2)
    old = LinkReceipt("best", "tls1.3/tcp", 1, 9, 20, None, None, 0, 0)
    best = PeerView(
        best.nid,
        best.state,
        set(),
        best.queue_depth,
        best.free_bytes,
        best.device_class,
        best.runtime_digest,
        best.link,
        best.hostname,
        (old, best.link),
    )
    result = place(
        request_id="r",
        model="missing",
        local_nid="local",
        local_models=set(),
        peers=[best, _peer("dead", state="dead", queue=-1, rtt=99)],
        link_budget=LinkBudget(),
    )
    pipeline = next(item for item in result.candidates if item["option"] == "pipeline")
    assert pipeline["reason"] == (
        "not implemented; worst RTT p90(last 2) 9 ms > budget 5 ms"
    )


def test_registry_drops_dead_process_and_rejects_non_loopback(tmp_path, monkeypatch):
    path = register_local_server(tmp_path, port=8285, pid=os.getpid())
    assert list_local_servers(tmp_path)[0]["url"] == "http://127.0.0.1:8285"
    path.write_text(
        json.dumps(
            {
                "url": "http://127.0.0.1:8285",
                "pid": 99999999,
                "started_at": 1,
                "api_key_file": None,
            }
        )
    )
    assert list_local_servers(tmp_path) == [] and not path.exists()
    monkeypatch.setattr("mlx2.fabric.registry._loopback", lambda _url: False)
    with pytest.raises(FabricError, match="loopback"):
        register_local_server(tmp_path, port=8286, pid=os.getpid())


def test_pid_alive_treats_unreaped_zombie_as_dead(monkeypatch):
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        time.sleep(0.1)
        assert child.returncode is None

        def zombie_stat(command, **_kwargs):
            assert command == ["ps", "-o", "stat=", "-p", str(child.pid)]
            return subprocess.CompletedProcess(command, 0, stdout="Z+\n", stderr="")

        monkeypatch.setattr("mlx2.fabric.registry.subprocess.run", zombie_stat)
        assert pid_alive(child.pid) is False
        assert child.returncode is None
    finally:
        child.wait()
