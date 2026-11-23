"""CPU checks for ``scripts/research/kv_growth_gpu_check.py`` (no model, no GPU)."""

import io
import json
import socket
import sys
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(ROOT / "scripts"))

import kv_growth_gpu_check as K


def _run(index, arm, tokens, *, growth=None, peak=100, ttft=1.0, decode=50.0, cached=(0, 900, 0)):
    names = ("long", "long_shared_prefix", "short")
    return {
        "index": index, "arm": arm, "checkpoint": "M@abc",
        "kv_planned_growth": (arm == "1") if growth is None else growth,
        "requests": [{"name": n, "token_ids": list(t), "ttft_s": ttft, "decode_tok_s": decode,
                      "cached_tokens": c, "prompt_tokens": 1000}
                     for n, t, c in zip(names, tokens, cached)],
        "memory_max": {"metal_peak_bytes": peak, "metal_active_bytes": peak - 10,
                       "process_physical_footprint_bytes": peak + 5, "group_rss_bytes": peak + 7},
        "final_memory": {"metal_peak_bytes": peak},
        "stop": "stopped",
    }


TOKENS = ([1, 2, 3], [4, 5], [6])


def test_verdict_identical_arms():
    v = K.build_verdict([_run(0, "0", TOKENS, peak=100, ttft=2.0, decode=40.0),
                         _run(1, "1", TOKENS, peak=90, ttft=1.5, decode=41.0)])
    assert v["tokens_identical"] and v["token_differences"] == [] and v["arm_receipts_match"]
    assert v["kv_planned_growth_observed"] == {"0": [False], "1": [True]}
    assert v["peak_memory"]["0"]["metal_peak_bytes"] == 100
    assert v["peak_memory_delta_bytes"]["metal_peak_bytes"] == -10
    assert v["timing_delta"]["long"] == {"ttft_s": -0.5, "decode_tok_s": 1.0}
    assert v["same_checkpoint"] and v["checkpoints"] == ["M@abc"]
    assert v["apc_warm_reuse"] == {"0": True, "1": True}


@pytest.mark.parametrize("other", [
    ([1, 2, 9], [4, 5], [6]),       # a changed token
    ([1, 2, 3], [4, 5, 7], [6]),    # a longer reply
    ([1, 2, 3], [4], [6]),          # a shorter reply
    ([1, 2, 3], [4, 5], [7]),       # the short request
])
def test_verdict_flags_any_token_difference(other):
    v = K.build_verdict([_run(0, "0", TOKENS), _run(1, "1", other)])
    assert v["tokens_identical"] is False
    assert len(v["token_differences"]) == 1 and v["token_differences"][0]["arm"] == "1"


def test_verdict_flags_a_difference_in_the_second_order():
    runs = [_run(0, "0", TOKENS), _run(1, "1", TOKENS), _run(2, "1", TOKENS),
            _run(3, "0", ([1, 2, 3], [4, 5], [8]))]
    v = K.build_verdict(runs)
    assert not v["tokens_identical"]
    assert v["token_differences"][0]["run"] == 3
    assert v["kv_planned_growth_observed"] == {"0": [False, False], "1": [True, True]}


def test_verdict_refuses_vacuous_and_mislabelled_runs():
    v = K.build_verdict([_run(0, "0", ([], [4], [6])), _run(1, "1", ([], [4], [6]))])
    assert not v["tokens_identical"] and v["empty_token_ids"]
    v = K.build_verdict([_run(0, "0", TOKENS), _run(1, "1", TOKENS, growth=False)])
    assert v["tokens_identical"] and v["arm_receipts_match"] is False
    v = K.build_verdict([_run(0, "0", TOKENS), _run(1, "1", TOKENS, growth=None) | {"kv_planned_growth": None}])
    assert v["arm_receipts_match"] is False
    with pytest.raises(ValueError):
        K.build_verdict([])


def test_port_checks_refuse_reserved_and_busy_ports():
    with pytest.raises(SystemExit, match="reserved"):
        K.check_port(8282)
    with pytest.raises(SystemExit, match="in use"):
        K.check_port(8399, free=lambda port: False)
    K.check_port(8399, free=lambda port: True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        with pytest.raises(SystemExit, match="in use"):
            K.check_port(sock.getsockname()[1])


def test_out_dir_must_be_empty(tmp_path):
    fresh = tmp_path / "new"
    assert K.prepare_out_dir(fresh) == fresh and fresh.is_dir()
    (fresh / "x").write_text("1")
    with pytest.raises(SystemExit, match="not empty"):
        K.prepare_out_dir(fresh)


def test_main_refuses_before_any_server(tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(K, "start", lambda *a, **k: started.append(a))
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "old.json").write_text("{}")
    with pytest.raises(SystemExit, match="not empty"):
        K.main(["--model", str(tmp_path), "--out-dir", str(tmp_path / "out")])
    with pytest.raises(SystemExit, match="reserved"):
        K.main(["--model", str(tmp_path), "--out-dir", str(tmp_path / "o2"), "--port", "8282"])
    assert started == [] and not (tmp_path / "o2").exists()


def test_atomic_write_and_arms(tmp_path):
    path = tmp_path / K.REPORT_NAME
    K.write_atomic(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1}
    assert not list(tmp_path.glob("*.partial"))
    assert K.arms_for(False) == ["0", "1"] and K.arms_for(True) == ["0", "1", "1", "0"]


def test_server_command_and_env(tmp_path):
    a = SimpleNamespace(model=Path("/m/Qwen"), port=8399, context=16384, max_context=None)
    cmd = K.server_command(a, tmp_path)
    assert "--ordinary" in cmd and cmd[cmd.index("--max-lanes") + 1] == "1"
    assert cmd[cmd.index("--cache-dir") + 1] == str(tmp_path)
    assert cmd[cmd.index("--max-context") + 1] == str(16384 + 4096)
    assert K.server_env("0")[K.ENV] == "0" and K.server_env("1")[K.ENV] == "1"


def test_request_set_shares_the_long_prefix():
    specs = K.request_set("DOC " * 50)
    long_a, long_b = (s["messages"][-1]["content"] for s in specs[:2])
    prefix = long_a[: long_a.index(K.QUESTION_A)]
    assert long_b.startswith(prefix) and long_a != long_b
    assert [s["name"] for s in specs] == ["long", "long_shared_prefix", "short"]
    body = K.request_body(specs[2]["messages"], max_tokens=64, reasoning_effort="low")
    assert body["temperature"] == 0 and body["logprobs"] is True and body["reasoning_effort"] == "low"
    assert "reasoning_effort" not in K.request_body([], max_tokens=1, reasoning_effort=None)


def test_ask_drops_reasoning_effort_when_refused():
    seen = []

    def send(body):
        seen.append(dict(body))
        if "reasoning_effort" in body:
            raise urllib.error.HTTPError("u", 400, "bad", {}, io.BytesIO(b""))
        return {"token_ids": [3, 4], "ttft_s": 0.1, "decode_tok_s": 9.0,
                "receipt": {"prompt_tokens": 10, "cached_tokens": 0}}

    a = SimpleNamespace(reasoning_effort="low", max_tokens=8, request_timeout=5)
    got = K.ask("http://x", {"name": "short", "messages": []}, a, send=send)
    assert got["token_ids"] == [3, 4] and got["reasoning_effort_sent"] is None
    assert len(seen) == 2 and "reasoning_effort" not in seen[1]


def test_memory_sampler_keeps_maxima():
    values = iter([{"metal_peak_bytes": 5, "metal_active_bytes": 3},
                   {"metal_peak_bytes": 9, "metal_active_bytes": 2}])
    rss = iter([100, 50])
    sampler = K.MemorySampler("http://x", 1, 1.0, status=lambda: next(values),
                              rss=lambda pgid: next(rss))
    sampler.sample(), sampler.sample()
    assert sampler.max["metal_peak_bytes"] == 9 and sampler.max["metal_active_bytes"] == 3
    assert sampler.max["group_rss_bytes"] == 100 and sampler.samples == 2
    assert K.group_rss_bytes(-12345) is None


class _StubChat(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        ids = [11, 12, 13] if "Reference" in body["messages"][-1]["content"] else [21]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for token in ids:
            event = {"choices": [{"delta": {"content": "t"},
                                  "logprobs": {"content": [{"id": token, "logprob": -0.1}]}}]}
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        final = {"choices": [{"delta": {}, "finish_reason": "length"}],
                 "mlx2": {"prompt_tokens": 40, "cached_tokens": 30, "route": "ordinary"}}
        self.wfile.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode())


def test_ask_against_a_stub_server():
    server = HTTPServer(("127.0.0.1", 0), _StubChat)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        a = SimpleNamespace(reasoning_effort="none", max_tokens=8, request_timeout=10)
        got = [K.ask(base, spec, a) for spec in K.request_set("doc")]
    finally:
        server.shutdown()
        server.server_close()
    assert [g["token_ids"] for g in got] == [[11, 12, 13], [11, 12, 13], [21]]
    assert got[0]["cached_tokens"] == 30 and got[0]["prompt_tokens"] == 40
    runs = [{"index": i, "arm": arm, "checkpoint": "M@f", "kv_planned_growth": arm == "1",
             "requests": got, "stop": "stopped"} for i, arm in enumerate("01")]
    assert K.build_verdict(runs)["tokens_identical"]
