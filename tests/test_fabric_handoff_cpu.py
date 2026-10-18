from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mlx2.fabric import FabricError
from mlx2.fabric.capability import local_models
from mlx2.fabric.community import create_community, sha256_hex
from mlx2.fabric.discovery import Candidate
from mlx2.fabric.link import LinkReceipt
from mlx2.fabric.node import FabricNode, _Peer
from mlx2.fabric.registry import register_local_server

MODEL = "handoff-model"
LOCAL_RUNTIME = "receiver-runtime"
PEER_RUNTIME = "producer-runtime"
EXECUTION_DIGEST = "receiver-execution"


def _capability(*, export: bool, import_: bool, runtime: str) -> dict:
    return {
        "export": export,
        "import": import_,
        "operation": "prefill-handoff-approx-v1",
        "operation_revision": "revision",
        "runtime_digest": runtime,
        "execution_digest": EXECUTION_DIGEST,
        "artifact_fingerprint": "artifact",
        "layout_fingerprint": "layout",
        "tokenizer_digest": "tokenizer",
        "plan_chunk_law": "prompt_length_prefill_step",
    }


class _ModelServer(BaseHTTPRequestHandler):
    requests: list[tuple[str, dict | bytes, str | None]]
    decode_failure: bool
    record_state: str | None

    def do_GET(self):
        body = json.dumps(
            {
                "data": [
                    {
                        "id": MODEL,
                        "loaded": True,
                        "prefill_handoff": _capability(
                            export=True, import_=True, runtime=LOCAL_RUNTIME
                        ),
                    }
                ]
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            value = raw
        recorded = dict(value) if isinstance(value, dict) else value
        self.requests.append((self.path, recorded, self.headers.get("Authorization")))
        status = 200
        if self.path.endswith("/plan"):
            assert value.pop("producer_runtime_digest") == PEER_RUNTIME
            response = {
                "token_ids": [1, 2, 3],
                "prompt_tokens": 3,
                "plan": {"prompt_tokens": 3, "prefix_tokens": 2, "chunk": 512},
                "plan_digest": "plan",
                "token_digest": "tokens",
                "artifact_fingerprint": "artifact",
                "tokenizer_digest": "tokenizer",
                "layout_fingerprint": "layout",
                "runtime_digest": LOCAL_RUNTIME,
                "record": (
                    {
                        "record_id": "qualified-record",
                        "record_state": self.record_state,
                        "max_prompt_tokens": 1024,
                    }
                    if self.record_state is not None
                    else None
                ),
            }
        elif self.path.endswith("/export"):
            response = b"handoff"
        elif self.path.endswith("/import"):
            assert self.headers.get("X-Mlx2-Handoff-Expected")
            response = {"handle": "a" * 32, "receipt": {"qualified": True}}
        else:
            handoff = isinstance(value, dict) and value.get("prefill_handoff")
            if handoff and self.decode_failure:
                status = 500
                response = {"error": {"reason": "decode_broke"}}
            else:
                response = {
                    "choices": [{"message": {"content": "ok"}}],
                    "mlx2": {"prefill_handoff": {"observed_used": bool(handoff)}},
                }
        body = (
            response if isinstance(response, bytes) else json.dumps(response).encode()
        )
        self.send_response(status)
        self.send_header(
            "Content-Type",
            "application/octet-stream"
            if isinstance(response, bytes)
            else "application/json",
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args):
        pass


@pytest.fixture
def model_server():
    _ModelServer.requests = []
    _ModelServer.decode_failure = False
    _ModelServer.record_state = "final"
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ModelServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _node(tmp_path, model_server, monkeypatch) -> tuple[FabricNode, str]:
    home = tmp_path / "fabric"
    community = create_community(home, hosts=("127.0.0.1",))
    token = tmp_path / "admin-token"
    token.write_text("admin-secret")
    register_local_server(
        home,
        port=model_server.server_port,
        pid=os.getpid(),
        admin_token_file=token,
    )
    node = FabricNode(home, use_zeroconf=False)
    node.community = community
    url = f"http://127.0.0.1:{model_server.server_port}"
    local = {
        "id": MODEL,
        "server_url": url,
        "loaded": True,
        "prefill_handoff": _capability(
            export=False, import_=True, runtime=LOCAL_RUNTIME
        ),
    }
    node._local = [local]
    node._signed_advert = {
        "advert": {
            "host": {"hostname": "local", "device_class": "M5"},
            "runtime": {"digest": LOCAL_RUNTIME},
        }
    }
    link = LinkReceipt("peer-nid", "tls", 1, 1, 3, 1, 1, 1, 1)
    peer = _Peer(
        Candidate("peer-nid", ("127.0.0.1",), 1, "", 0, "", "static"),
        state="alive",
        advert={
            "models": [
                {
                    "id": MODEL,
                    "loaded": True,
                    "prefill_handoff": _capability(
                        export=True, import_=False, runtime=PEER_RUNTIME
                    ),
                }
            ],
            "host": {"hostname": "peer", "device_class": "M3"},
            "runtime": {"digest": PEER_RUNTIME},
        },
        link=link,
    )
    peer.link_history.append(link)
    node._peers[peer.candidate.nid] = peer
    monkeypatch.setattr(node, "_export_from_peer", lambda **_kwargs: b"handoff")
    return node, url


def _body(**extra) -> bytes:
    return json.dumps({"model": MODEL, "messages": [], **extra}).encode()


def test_capability_advertises_prefill_handoff(model_server):
    url = f"http://127.0.0.1:{model_server.server_port}"
    entry = local_models([{"url": url, "api_key": None}])[0]
    assert entry["prefill_handoff"]["runtime_digest"] == LOCAL_RUNTIME


def test_opted_in_qualified_handoff_is_observed_and_strips_opt_in(
    tmp_path, model_server, monkeypatch
):
    node, _ = _node(tmp_path, model_server, monkeypatch)
    header_clean, header_requested = node._request_opt_in(
        _body(), "prefill-handoff-approx-v1"
    )
    assert header_requested is True and json.loads(header_clean)["model"] == MODEL
    raw = _body(mlx2_fabric={"allow_approximate_prefill": True})
    clean, requested = node._request_opt_in(raw, None)
    status, _, output, placement = node._route(
        "/v1/chat/completions", clean, allow_approximate_prefill=requested
    )
    receipt = json.loads(output)["mlx2_fabric"]
    assert status == 200 and placement.kind == "approximate_prefill_offload"
    assert receipt["prefill_handoff"]["producer"] == "peer-nid"
    assert receipt["prefill_handoff"]["record_state"] == "final"
    assert receipt["served_by"]["nid"] == node.community.nid
    assert receipt["states"] == {
        "implemented": True,
        "qualified": True,
        "selected": True,
        "observed_used": True,
    }
    assert all(
        not isinstance(value, dict) or "mlx2_fabric" not in value
        for _, value, _ in _ModelServer.requests
    )
    assert all(auth == "Bearer admin-secret" for _, _, auth in _ModelServer.requests)


def test_not_requested_and_no_record_use_ordinary_local(
    tmp_path, model_server, monkeypatch
):
    node, _ = _node(tmp_path, model_server, monkeypatch)
    status, _, output, placement = node._route("/v1/chat/completions", _body())
    assert status == 200 and placement.kind == "local"
    candidate = json.loads(output)["mlx2_fabric"]["candidates"][0]
    assert candidate["verdict"] == "not_requested"

    _ModelServer.record_state = None
    status, _, output, placement = node._route(
        "/v1/chat/completions", _body(), allow_approximate_prefill=True
    )
    assert status == 200 and placement.kind == "local"
    candidate = json.loads(output)["mlx2_fabric"]["candidates"][0]
    assert candidate["reason"] == "no_qualification_record"


def test_provisional_record_is_selected_but_not_qualified(
    tmp_path, model_server, monkeypatch
):
    _ModelServer.record_state = "provisional"
    node, _ = _node(tmp_path, model_server, monkeypatch)
    status, _, output, placement = node._route(
        "/v1/chat/completions", _body(), allow_approximate_prefill=True
    )
    receipt = json.loads(output)["mlx2_fabric"]
    assert status == 200 and placement.kind == "approximate_prefill_offload"
    assert receipt["prefill_handoff"]["record_state"] == "provisional"
    assert receipt["states"]["qualified"] is False


def test_logprobs_fields_reach_plan_and_decode_unchanged(
    tmp_path, model_server, monkeypatch
):
    node, _ = _node(tmp_path, model_server, monkeypatch)
    body = _body(logprobs=True, top_logprobs=2)
    status, _, _, _ = node._route(
        "/v1/chat/completions", body, allow_approximate_prefill=True
    )
    assert status == 200
    plan = next(value for path, value, _ in _ModelServer.requests if path.endswith("/plan"))
    decode = next(value for path, value, _ in _ModelServer.requests if "chat" in path)
    assert plan["logprobs"] is True and plan["top_logprobs"] == 2
    assert plan["producer_runtime_digest"] == PEER_RUNTIME
    assert decode["logprobs"] is True and decode["top_logprobs"] == 2


def test_origin_refuses_export_blob_not_bound_by_signed_receipt(
    tmp_path, model_server, monkeypatch
):
    node, _ = _node(tmp_path, model_server, monkeypatch)
    peer = node._peers["peer-nid"]

    class Response:
        status = 200

        @staticmethod
        def read():
            return b"received-blob"

        @staticmethod
        def getheader(name):
            return "signed" if name == "X-Mlx2-Serve-Receipt" else None

    class Connection:
        def request(self, *_args, **_kwargs):
            pass

        @staticmethod
        def getresponse():
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(node, "_connect", lambda *_args, **_kwargs: Connection())
    monkeypatch.setattr(
        node,
        "_verify_serve",
        lambda *_args, **_kwargs: {"blob_sha256": sha256_hex(b"different-blob")},
    )
    plan = {
        "token_ids": [1, 2, 3],
        "plan": {"prompt_tokens": 3, "prefix_tokens": 2, "chunk": 512},
        "plan_digest": "plan",
        "artifact_fingerprint": "artifact",
        "tokenizer_digest": "tokenizer",
        "layout_fingerprint": "layout",
    }
    with pytest.raises(FabricError, match="blob_digest_mismatch"):
        FabricNode._export_from_peer(
            node,
            peer=peer,
            request_id="request",
            model=MODEL,
            plan_response=plan,
        )


def test_link_budget_refuses_handoff(
    tmp_path, model_server, monkeypatch
):
    node, _ = _node(tmp_path, model_server, monkeypatch)
    peer = node._peers["peer-nid"]
    peer.link = LinkReceipt("peer-nid", "tls", 9, 9, 3, 1, 1, 1, 1)
    peer.link_history.clear()
    peer.link_history.append(peer.link)
    status, _, output, placement = node._route(
        "/v1/chat/completions", _body(), allow_approximate_prefill=True
    )
    assert status == 200 and placement.kind == "local"
    assert json.loads(output)["mlx2_fabric"]["candidates"][0]["reason"].startswith(
        "link_budget_refused:"
    )


def test_export_failure_falls_back_but_decode_failure_does_not_retry(
    tmp_path, model_server, monkeypatch
):
    node, _ = _node(tmp_path, model_server, monkeypatch)

    def fail_export(**_kwargs):
        raise FabricError("peer_export_failed")

    monkeypatch.setattr(node, "_export_from_peer", fail_export)
    status, _, output, _ = node._route(
        "/v1/chat/completions", _body(), allow_approximate_prefill=True
    )
    receipt = json.loads(output)["mlx2_fabric"]
    assert status == 200
    assert receipt["fallback"] == "ordinary_local"
    assert receipt["failed_step"] == "export"
    assert receipt["reason"] == "peer_export_failed"
    assert receipt["kind"] == "local"
    assert receipt["approximate"] is False
    assert receipt["states"] == {
        "implemented": True,
        "qualified": False,
        "selected": True,
        "observed_used": True,
    }
    assert receipt["candidates"][0]["verdict"] == "fallback"

    monkeypatch.setattr(node, "_export_from_peer", lambda **_kwargs: b"handoff")
    _ModelServer.requests.clear()
    _ModelServer.decode_failure = True
    status, _, output, _ = node._route(
        "/v1/chat/completions", _body(), allow_approximate_prefill=True
    )
    receipt = json.loads(output)["mlx2_fabric"]
    chat_requests = [item for item in _ModelServer.requests if "chat" in item[0]]
    assert status == 500 and len(chat_requests) == 1
    assert receipt["failed_step"] == "decode"
    assert "fallback" not in receipt


def test_plan_failure_builds_truthful_ordinary_local_fallback(
    tmp_path, model_server, monkeypatch
):
    node, _ = _node(tmp_path, model_server, monkeypatch)
    monkeypatch.setattr(
        node,
        "_plan_handoff",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FabricError("plan_broke")),
    )
    status, _, output, placement = node._route(
        "/v1/chat/completions", _body(), allow_approximate_prefill=True
    )
    receipt = json.loads(output)["mlx2_fabric"]
    assert status == 200 and placement.kind == "local"
    assert receipt["fallback"] == "ordinary_local"
    assert receipt["failed_step"] == "plan"
    assert receipt["approximate"] is False
    assert receipt["states"]["selected"] is True
    assert receipt["states"]["observed_used"] is True
    assert receipt["candidates"][0]["verdict"] == "fallback"


def test_not_ready_or_digestless_capability_is_not_selectable(
    tmp_path, model_server, monkeypatch
):
    node, _ = _node(tmp_path, model_server, monkeypatch)
    local = node._local[0]["prefill_handoff"]
    local["ready"] = False
    status, _, output, _ = node._route(
        "/v1/chat/completions", _body(), allow_approximate_prefill=True
    )
    assert status == 200
    assert json.loads(output)["mlx2_fabric"]["candidates"][0]["reason"] == (
        "local_import_unavailable"
    )

    local["ready"] = True
    del node._peers["peer-nid"].advert["models"][0]["prefill_handoff"][
        "execution_digest"
    ]
    status, _, output, _ = node._route(
        "/v1/chat/completions", _body(), allow_approximate_prefill=True
    )
    assert status == 200
    assert json.loads(output)["mlx2_fabric"]["candidates"][0]["reason"] == (
        "peer_execution_digest_missing"
    )


def test_peer_refuses_export_for_model_not_resident(tmp_path):
    node = FabricNode(tmp_path, use_zeroconf=False)
    with pytest.raises(FabricError, match="model_not_resident_locally"):
        node._forward_handoff_export(
            json.dumps({"model": "missing", "token_ids": []}).encode()
        )
