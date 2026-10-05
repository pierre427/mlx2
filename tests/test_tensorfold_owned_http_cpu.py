"""Host-only admission and HTTP contracts for the default-off native profile."""

import json
import importlib.util
import io
import hashlib
import socket
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from mlx2.runtime.tensorfold_owned_router import (
    TensorfoldOwnedLiveRouter, _Request, _timing_receipt,
)
from mlx2.server import handler_for

PRIVATE_GATE_ROOT = (Path(__file__).resolve().parents[1] / "qualification/runs"
                     / "tree15-b1-discriminator-20261003")
private_gate_test = pytest.mark.skipif(
    not PRIVATE_GATE_ROOT.is_dir(),
    reason="private machine-specific gate sources are not in the public export",
)


class FakeProfile:
    identity = "fixture"
    cache_layout = "qwen38-tensorfold-owned:fixture"

    def __init__(self):
        self.sessions = {}
        self.closed = False
        self.next_id = 0

    def start(self, *, timeout):
        return {"identity": self.identity, "cache_layout": self.cache_layout}

    def call(self, method, *, timeout, **params):
        if method == "live_open":
            prefill_start_ns = time.monotonic_ns()
            self.next_id += 1
            sid = str(self.next_id)
            self.sessions[sid] = {"token": params["prompt"][-1],
                                  "remaining": params["max_new_tokens"],
                                  "route": params["route"],
                                  "eos_ids": params.get("eos_ids", [])}
            prefill_end_ns = time.monotonic_ns()
            return {"session": sid, "tokens": [], "finished": False,
                    "prefill_start_ns": prefill_start_ns,
                    "prefill_end_ns": max(prefill_end_ns, prefill_start_ns + 1),
                    "first_token_ns": None, "prompt_tokens": len(params["prompt"]),
                    "cached_prompt_tokens": 0,
                    "route_requested": params["route"],
                    "cache_layout": self.cache_layout, "cache_policy": "disabled",
                    "qualified": False,
                    "apcv2_lookup": False, "apcv2_store": False}
        if method == "live_step":
            mode = ("b2plus_shared_ordinary" if len(self.sessions) > 1 else
                    "b1_serial_ordinary" if next(iter(self.sessions.values()))["route"] == "serial"
                    else "b1_tree_eligible")
            landed, finished, reasons = {}, {}, {}
            for sid, row in self.sessions.items():
                row["token"] += 1
                row["remaining"] -= 1
                landed[sid] = [row["token"]]
                finished[sid] = row["remaining"] == 0 or row["token"] in row["eos_ids"]
                reasons[sid] = ("stop" if row["token"] in row["eos_ids"] else
                                "length" if finished[sid] else "")
            return {"mode": mode, "tokens": landed, "finished": finished,
                    "token_ready_ns": time.monotonic_ns(),
                    "active_width": len(self.sessions),
                    "finish_reasons": reasons,
                    "drafted": int(mode == "b1_tree_eligible"), "accepted": 0,
                    "per_stream": {sid: {"drafted": int(mode == "b1_tree_eligible"),
                                         "accepted": 0} for sid in landed},
                    "cache_policy": "disabled", "qualified": False,
                    "apcv2_lookup": False, "apcv2_store": False}
        if method == "live_close":
            del self.sessions[params["session"]]
            return {"closed": params["session"]}
        raise AssertionError(method)

    def close(self):
        self.closed = True


def test_native_timing_excludes_queue_wait_and_one_token_decode_is_undefined():
    request = _Request([1] * 100, 3, "auto")
    request.queued_ns = 1_000_000_000
    request.admission_start_ns = 3_000_000_000
    request.prefill_start_ns = 4_000_000_000
    request.prefill_end_ns = 5_000_000_000
    request.first_token_ns = 5_000_000_000
    request.final_token_ns = 5_500_000_000
    request.tokens = [2, 3, 4]
    timing = _timing_receipt(request)
    assert timing["queue_wait_seconds"] == 2
    assert timing["prefill_to_first_token_seconds"] == 1
    assert timing["effective_prefill_tokens_per_second"] == 100
    assert timing["decode_tokens_per_second"] == 4
    assert timing["first_token_offset_seconds"] == 4
    request.tokens = [2]
    request.final_token_ns = request.first_token_ns
    timing = _timing_receipt(request)
    assert timing["decode_rate_defined"] is False
    assert timing["decode_tokens_per_second"] is None
    request.tokens = []
    request.first_token_ns = request.final_token_ns = None
    timing = _timing_receipt(request)
    assert timing["decode_tokens_per_second"] is None
    assert timing["first_token_offset_seconds"] is None


def test_router_batches_two_greedy_requests_and_keeps_worker_cache_opaque():
    profile = FakeProfile()
    router = TensorfoldOwnedLiveRouter(profile, coalesce_ms=30)
    results = []
    barrier = threading.Barrier(3)

    def request(prompt):
        barrier.wait()
        results.append(router.generate(prompt, 3, timeout=2))

    threads = [threading.Thread(target=request, args=([1, 2],)),
               threading.Thread(target=request, args=([3, 4],))]
    try:
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=3)
        assert len(results) == 2
        assert sorted(row["token_ids"] for row in results) == [[3, 4, 5], [5, 6, 7]]
        for result in results:
            receipt = result["route_receipt"]
            assert receipt["qualified"] is False
            assert receipt["cache_policy"] == "disabled"
            assert receipt["apcv2_lookup"] is False and receipt["apcv2_store"] is False
            assert receipt["active_width_modes"]["b2plus_shared_ordinary"] >= 1
            assert receipt["drafted"] == 0
            assert receipt["ordinary_compute_width"] == 2
            assert receipt["active_width_counts"]["2"] >= 1
            assert receipt["timing"]["clock"] == "host_monotonic_ns"
            assert receipt["timing"]["effective_prefill_tokens_per_second"] > 0
            assert receipt["timing"]["decode_rate_defined"] is True
        assert not profile.sessions
    finally:
        router.close()
    assert profile.closed


def test_router_timeout_closes_native_session_then_serves_next_request():
    class SlowOnceProfile(FakeProfile):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.resume = threading.Event()
            self.pause_once = True

        def call(self, method, *, timeout, **params):
            if method == "live_step" and self.pause_once:
                self.pause_once = False
                self.entered.set()
                assert self.resume.wait(2)
            return super().call(method, timeout=timeout, **params)

    profile = SlowOnceProfile()
    router = TensorfoldOwnedLiveRouter(profile, coalesce_ms=0)
    failures = []

    def expire():
        try:
            router.generate([1, 2], 10, timeout=.05)
        except TimeoutError as exc:
            failures.append(str(exc))

    thread = threading.Thread(target=expire)
    try:
        thread.start()
        assert profile.entered.wait(2)
        thread.join(timeout=2)
        assert failures == ["TensorFold-owned request timed out"]
        profile.resume.set()
        for _ in range(200):
            if not profile.sessions:
                break
            time.sleep(.01)
        assert not profile.sessions
        assert router.generate([3, 4], 1, timeout=2)["token_ids"] == [5]
        assert not profile.sessions
    finally:
        profile.resume.set()
        router.close()


def test_router_disconnect_during_prefill_closes_session_before_round():
    class SlowOpenProfile(FakeProfile):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.resume = threading.Event()
            self.steps = 0
            self.pause_once = True

        def call(self, method, *, timeout, **params):
            if method == "live_step":
                self.steps += 1
            result = super().call(method, timeout=timeout, **params)
            if method == "live_open" and self.pause_once:
                self.pause_once = False
                self.entered.set()
                assert self.resume.wait(2)
            return result

    profile = SlowOpenProfile()
    router = TensorfoldOwnedLiveRouter(profile, coalesce_ms=0)
    disconnected = threading.Event()
    failures = []

    def submit():
        try:
            router.generate([1, 2], 10, timeout=2,
                            disconnect_probe=disconnected.is_set)
        except ConnectionAbortedError as exc:
            failures.append(str(exc))

    thread = threading.Thread(target=submit)
    try:
        thread.start()
        assert profile.entered.wait(2)
        disconnected.set()
        thread.join(timeout=2)
        assert failures == ["TensorFold-owned client disconnected"]
        profile.resume.set()
        for _ in range(200):
            if not profile.sessions:
                break
            time.sleep(.01)
        assert not profile.sessions
        assert profile.steps == 0
        assert router.generate([3, 4], 1, timeout=2)["token_ids"] == [5]
    finally:
        profile.resume.set()
        router.close()


def test_router_rejects_unsupported_capability_without_worker_call():
    profile = FakeProfile()
    router = TensorfoldOwnedLiveRouter(profile, max_context=8)
    try:
        with pytest.raises(ValueError, match="Qwen3.8 token_ids"):
            router.generate([True], 2)
        with pytest.raises(ValueError, match="1..192"):
            router.generate([1], 193)
        with pytest.raises(ValueError, match="service context"):
            router.generate([1, 2, 3, 4, 5], 4)
        with pytest.raises(ValueError, match="180"):
            router.generate([1], 1, timeout=840)
        with pytest.raises(ValueError, match="900"):
            router.generate([1], 1, timeout=901, allow_long=True)
        assert router.generate([1], 1, timeout=840, allow_long=True)["token_ids"] == [2]
        assert not profile.sessions
    finally:
        router.close()


def test_router_queues_twenty_requests_with_eight_active_stream_bound():
    profile = FakeProfile()
    router = TensorfoldOwnedLiveRouter(profile, max_streams=8, coalesce_ms=30)
    barrier = threading.Barrier(21)
    outcomes = []

    def submit(index):
        barrier.wait(timeout=3)
        outcomes.append(router.generate([index + 1], 2, timeout=5))

    threads = [threading.Thread(target=submit, args=(i,)) for i in range(20)]
    try:
        for thread in threads:
            thread.start()
        barrier.wait(timeout=3)
        for thread in threads:
            thread.join(timeout=6)
        assert len(outcomes) == 20
        assert not profile.sessions
        assert all(result["route_receipt"]["apcv2_store"] is False
                   for result in outcomes)
    finally:
        router.close()


def test_router_changes_width_between_rounds_without_cache_handoff():
    class PausingProfile(FakeProfile):
        def __init__(self):
            super().__init__()
            self.first_round = threading.Event()
            self.resume = threading.Event()
            self.steps = 0

        def call(self, method, *, timeout, **params):
            result = super().call(method, timeout=timeout, **params)
            if method == "live_step":
                self.steps += 1
                if self.steps == 1:
                    self.first_round.set()
                    assert self.resume.wait(2)
            return result

    profile = PausingProfile()
    router = TensorfoldOwnedLiveRouter(profile, coalesce_ms=0)
    results = {}

    def request(name, prompt, budget):
        results[name] = router.generate(prompt, budget, timeout=3)

    first = threading.Thread(target=request, args=("first", [1, 2], 5))
    second = threading.Thread(target=request, args=("second", [3, 4], 2))
    try:
        first.start()
        assert profile.first_round.wait(2)
        second.start()
        with router._condition:
            assert router._condition.wait_for(lambda: len(router._pending) == 1, timeout=2)
        profile.resume.set()
        first.join(timeout=4)
        second.join(timeout=4)
        assert len(results) == 2
        modes = results["first"]["route_receipt"]["active_width_modes"]
        assert modes["b1_tree_eligible"] >= 2
        assert modes["b2plus_shared_ordinary"] >= 1
        assert results["second"]["route_receipt"]["active_width_modes"] == {
            "b2plus_shared_ordinary": 2,
        }
        transitions = results["first"]["route_receipt"]["mode_transitions"]
        assert [(row["mode"], row["active_width"]) for row in transitions] == [
            ("b1_tree_eligible", 1), ("b2plus_shared_ordinary", 2),
            ("b1_tree_eligible", 1)]
        assert [row["completion_tokens_before_round"] for row in transitions] == [0, 1, 3]
        assert [row["sequence"] for row in transitions] == [1, 2, 4]
        assert router.status()["last_round"] == {
            "sequence": 5, "mode": "b1_tree_eligible", "active_width": 1}
        assert results["first"]["route_receipt"]["drafted"] == 3
        assert results["second"]["route_receipt"]["drafted"] == 0
        assert not profile.sessions
    finally:
        profile.resume.set()
        router.close()


def test_router_closes_native_session_when_admission_receipt_is_invalid():
    class BadReceiptProfile(FakeProfile):
        def call(self, method, *, timeout, **params):
            receipt = super().call(method, timeout=timeout, **params)
            if method == "live_open":
                receipt["apcv2_store"] = True
            return receipt

    profile = BadReceiptProfile()
    router = TensorfoldOwnedLiveRouter(profile, coalesce_ms=0)
    try:
        with pytest.raises(RuntimeError, match="worker request failed"):
            router.generate([1, 2], 2, timeout=2)
        assert not profile.sessions
    finally:
        router.close()


class FakeEngine:
    model_path = "fixture"
    api_resources = {}

    def __init__(self):
        self.prompt_lock = threading.RLock()
        self.adapter = SimpleNamespace(tokenizer=SimpleNamespace(
            eos_token_ids={12},
            decode=lambda ids, skip_special_tokens: " ".join(
                str(token) for token in ids if not skip_special_tokens or token != 12),
        ))

    def status(self):
        return {"healthy": True, "error": None, "model": "fixture"}

    def render_prompt(self, request):
        assert request.get("enable_thinking") is False or "prompt" in request
        return [10]


def _post(port, body):
    request = Request(
        f"http://127.0.0.1:{port}/v1/experimental/tensorfold-owned/generate",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=3) as response:
        return response.status, json.load(response)


def test_explicit_http_endpoint_is_off_by_default_and_returns_route_receipt():
    router = TensorfoldOwnedLiveRouter(FakeProfile(), coalesce_ms=0)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(FakeEngine(), tensorfold_owned_router=router))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, payload = _post(server.server_port, {"token_ids": [10], "max_new_tokens": 2})
        assert status == 200 and payload["token_ids"] == [11, 12]
        assert payload["route_receipt"]["selected_by"] == "explicit_token_id_endpoint"
        status, serial = _post(server.server_port, {
            "token_ids": [10], "max_new_tokens": 2, "route": "serial",
        })
        assert status == 200
        assert serial["route_receipt"]["active_width_modes"] == {"b1_serial_ordinary": 2}
        with pytest.raises(HTTPError) as error:
            _post(server.server_port, {"token_ids": [10], "max_new_tokens": 2, "tools": []})
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        router.close()

    off = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(FakeEngine()))
    off_thread = threading.Thread(target=off.serve_forever, daemon=True)
    off_thread.start()
    try:
        with pytest.raises(HTTPError) as error:
            _post(off.server_port, {"token_ids": [10], "max_new_tokens": 2})
        assert error.value.code == 404
    finally:
        off.shutdown()
        off.server_close()
        off_thread.join()


def test_http_disconnect_closes_native_session_and_next_request_succeeds():
    class PausedStepProfile(FakeProfile):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.resume = threading.Event()
            self.pause_once = True

        def call(self, method, *, timeout, **params):
            if method == "live_step" and self.pause_once:
                self.pause_once = False
                self.entered.set()
                assert self.resume.wait(3)
            return super().call(method, timeout=timeout, **params)

    profile = PausedStepProfile()
    router = TensorfoldOwnedLiveRouter(profile, coalesce_ms=0)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(
        FakeEngine(), tensorfold_owned_router=router))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = socket.create_connection(("127.0.0.1", server.server_port), timeout=2)
    body = json.dumps({"token_ids": [10], "max_new_tokens": 20}).encode()
    wire = (b"POST /v1/experimental/tensorfold-owned/generate HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
    try:
        client.sendall(wire)
        assert profile.entered.wait(2)
        with urlopen(f"http://127.0.0.1:{server.server_port}/v1/status", timeout=2) as response:
            native = json.load(response)["tensorfold_owned"]
        assert native["cache_policy"] == "disabled"
        assert native["active_requests"] == 1
        client.close()
        for _ in range(200):
            with router._condition:
                cancelled = any(req.cancelled for req in router._active.values())
            if cancelled:
                break
            time.sleep(.01)
        assert cancelled
        profile.resume.set()
        for _ in range(200):
            with urlopen(f"http://127.0.0.1:{server.server_port}/v1/status", timeout=2) as response:
                native = json.load(response)["tensorfold_owned"]
            if native["active_requests"] == 0:
                break
            time.sleep(.01)
        assert native["active_requests"] == 0
        for _ in range(200):
            if not profile.sessions:
                break
            time.sleep(.01)
        assert not profile.sessions
        status, payload = _post(server.server_port, {"token_ids": [10],
                                                    "max_new_tokens": 2})
        assert status == 200 and payload["token_ids"] == [11, 12]
    finally:
        client.close()
        profile.resume.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        router.close()


def test_standard_completion_header_uses_native_eos_and_rejects_unsupported_options():
    router = TensorfoldOwnedLiveRouter(FakeProfile(), coalesce_ms=0)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(
        FakeEngine(), tensorfold_owned_router=router))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def post(path, body, *, header="1", route="auto"):
        request = Request(
            f"http://127.0.0.1:{server.server_port}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "X-MLX2-TensorFold-Owned": header,
                     "X-MLX2-TensorFold-Route": route},
        )
        with urlopen(request, timeout=3) as response:
            return json.load(response)

    try:
        chat = post("/v1/chat/completions", {
            "model": "fixture", "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0, "top_p": 1, "top_k": 0,
            "enable_thinking": False, "max_tokens": 5,
        })
        assert chat["object"] == "chat.completion"
        assert chat["choices"] == [{"index": 0, "finish_reason": "stop",
                                    "message": {"role": "assistant", "content": "11"}}]
        assert chat["usage"] == {"prompt_tokens": 1, "completion_tokens": 2,
                                 "total_tokens": 3}
        assert chat["mlx2"]["selected_by"] == "explicit_standard_header"
        assert chat["mlx2"]["apcv2_lookup"] is False
        serial = post("/v1/chat/completions", {
            "model": "fixture", "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0, "enable_thinking": False, "max_tokens": 5,
        }, route="serial")
        assert serial["mlx2"]["route_requested"] == "serial"
        assert serial["mlx2"]["active_width_modes"] == {"b1_serial_ordinary": 2}
        assert serial["mlx2"]["output_token_ids_sha256"] == chat["mlx2"]["output_token_ids_sha256"]
        default_chat = post("/v1/chat/completions", {
            "model": "fixture", "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 5,
        })
        assert default_chat["mlx2"]["output_token_ids_sha256"] == chat["mlx2"]["output_token_ids_sha256"]
        completion = post("/v1/completions", {
            "model": "fixture", "prompt": "hello", "temperature": 0,
            "max_tokens": 2,
        })
        assert completion["choices"][0]["text"] == "11"
        default_completion = post("/v1/completions", {
            "model": "fixture", "prompt": "hello", "max_tokens": 2,
        })
        assert default_completion["choices"][0]["text"] == "11"
        for body in (
            {"prompt": "hello", "temperature": 0, "stream": True},
            {"prompt": "hello", "temperature": 0.7},
            {"prompt": "hello", "temperature": 0, "stop": ["x"]},
            {"prompt": "hello", "temperature": 0, "max_tokens": 193},
            {"prompt": "hello", "temperature": 0, "top_p": 0.9},
            {"prompt": "hello", "temperature": 0, "top_k": 5},
        ):
            with pytest.raises(HTTPError) as error:
                post("/v1/completions", body)
            assert error.value.code == 400
        with pytest.raises(HTTPError) as error:
            post("/v1/chat/completions", {
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0, "enable_thinking": True,
            })
        assert error.value.code == 400
        with pytest.raises(HTTPError) as error:
            post("/v1/completions", {"prompt": "hello", "temperature": 0}, header="0")
        assert error.value.code == 400
        with pytest.raises(HTTPError) as error:
            post("/v1/completions", {"prompt": "hello", "temperature": 0}, route="unknown")
        assert error.value.code == 400
        unowned_route = Request(
            f"http://127.0.0.1:{server.server_port}/v1/completions",
            data=json.dumps({"prompt": "hello", "temperature": 0}).encode(),
            headers={"Content-Type": "application/json",
                     "X-MLX2-TensorFold-Route": "serial"},
        )
        with pytest.raises(HTTPError) as error:
            urlopen(unowned_route, timeout=3)
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        router.close()


@private_gate_test
def test_service_gate_cpu_drives_shared_http_stage_before_paired_screen(monkeypatch):
    script = (Path(__file__).resolve().parents[1] / "qualification/runs"
              / "tree15-b1-discriminator-20261003/tensorfold_owned_service.py")
    spec = importlib.util.spec_from_file_location("tensorfold_owned_service_gate_cpu", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    class Process:
        returncode = None

        def poll(self):
            return None

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

    class Response(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    def fake_request(base, prompt, tokens, route):
        calls.append((tuple(prompt), tokens, route))
        mode = ("b2plus_shared_ordinary" if tokens == 32 and route == "auto" else
                "b1_serial_ordinary" if route == "serial" else "b1_tree_eligible")
        return {"elapsed_seconds": 1.0, "tokens": [7] * tokens,
                "receipt": {"qualified": False, "apcv2_lookup": False,
                            "apcv2_store": False, "drafted": int(route == "auto"),
                            "active_width_modes": {mode: 1}}}

    monkeypatch.setattr(module, "free_port", lambda: 12345)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(module, "urlopen", lambda url, timeout: Response(
        json.dumps({"route_receipt": "ordinary"}).encode()))
    monkeypatch.setattr(module, "request", fake_request)
    import psutil

    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(
        total=128 << 30, available=96 << 30))
    result = module.run(SimpleNamespace(
        source="source", target="target", drafter="drafter", mlx_lm_source="mlx_lm",
        dry_run=False,
    ))
    assert result["status"] == "completed"
    assert result["b2_token_ids_equal"] == [True, True]
    assert len(result["b1_pairs"]) == 3
    assert len([call for call in calls if call[1] == 32 and call[2] == "auto"]) == 2
    assert calls.index((tuple([44, 382, 991, 144]), 16, "auto")) < next(
        i for i, call in enumerate(calls) if call[1] == 48)


@private_gate_test
def test_spomin_report_requires_frozen_cell_order_and_source_identity(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / "qualification/runs/tree15-b1-discriminator-20261003/tensorfold_owned_spomin_report.py"
    spec = importlib.util.spec_from_file_location("tensorfold_owned_spomin_report_cpu", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    corpus = json.loads(module.CORPUS.read_text())
    source = {"cases": 400, "source_preparation_reconciliation": {
        "cases_compared": 400, "mismatches": 0}}
    source_file = tmp_path / "prepared.json"
    source_file.write_text(json.dumps(source))
    source_sha = hashlib.sha256(source_file.read_bytes()).hexdigest()
    source_commit = "a" * 40
    domain = corpus["domain_order"][0]
    case_ids = [case["case_id"] for case in corpus["cases"] if case["domain"] == domain]
    payload = {"schema": "mlx2.tensorfold-owned-spomin-cell.v2",
               "cell_index": 0, "domain": domain, "transcript_arm": "full",
               "corpus_sha256": module.FROZEN_CORPUS_SHA256,
               "prepared_report_sha256": source_sha,
               "status": "completed", "worker_identities": ["native"],
               "batch_wall_seconds": 2.0, "completion_tokens_total": 40,
               "aggregate_completion_tokens_per_second": 20.0,
               "observed_active_widths": [2],
               "rows": [{"case_id": case_id, "transcript_arm": "full",
                         "prompt_tokens": 10, "completion_tokens": 2,
                         "receipt": {"ordinary_compute_width": 2},
                         "timing": {"clock": "host_monotonic_ns", "prompt_tokens": 10,
                                    "completion_tokens": 2, "http_wall_seconds": 1.0,
                                    "effective_prefill_tokens_per_second": 100.0,
                                    "decode_tokens_per_second": 5.0,
                                    "decode_rate_defined": True}}
                        for case_id in case_ids]}
    cell = tmp_path / f"tree15-owned-spomin-cell-00-{source_commit[:8]}.json"
    cell.write_text(json.dumps({"source_commit": source_commit,
                                "gate": "spomin-cell", "cell_index": 0,
                                "status": "completed", "payload": payload}))
    report = module.collect(tmp_path, source_commit, source_file, source_sha, "unused")
    assert report["completed_cells"] == 1 and report["status"] == "incomplete"
    assert len(report["rows"]) == 20
    payload["transcript_arm"] = "compacted"
    cell.write_text(json.dumps({"source_commit": source_commit,
                                "gate": "spomin-cell", "cell_index": 0,
                                "status": "completed", "payload": payload}))
    with pytest.raises(RuntimeError, match="provenance or domain order"):
        module.collect(tmp_path, source_commit, source_file, source_sha, "unused")


@private_gate_test
def test_spomin_long_lease_does_not_expand_other_gate_caps():
    root = Path(__file__).resolve().parents[1]
    script = (root / "qualification/runs/tree15-b1-discriminator-20261003"
              / "tensorfold-owned-b1/run_gate.py")
    spec = importlib.util.spec_from_file_location("tensorfold_owned_gate_caps_cpu", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.cap_error("spomin-cell", 900, 48) is None
    assert "900" in module.cap_error("spomin-cell", 901, 48)
    assert module.cap_error("service", 180, 48) is None
    assert "180" in module.cap_error("service", 181, 48)
    assert "resident" in module.cap_error("spomin-cell", 900, 49)


@private_gate_test
def test_spomin_b1_normal_endpoint_pairing_and_exact_token_oracle(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    script = root / "qualification/runs/tree15-b1-discriminator-20261003/tensorfold_owned_spomin_b1.py"
    spec = importlib.util.spec_from_file_location("tensorfold_owned_spomin_b1_cpu", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prepared = [{"case": {"case_id": "software_architecture.01"},
                 "full_messages": [], "compacted_messages": []}]
    monkeypatch.setattr(module, "source_cell", lambda index, *_: (
        "software_architecture", "full" if index == 0 else "compacted",
        prepared, "system"))
    monkeypatch.setattr(module, "free_port", lambda: 12345)
    import psutil

    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(
        available=96 << 30))

    class Process:
        returncode = None

        def poll(self):
            return None

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

    class Response(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(module, "urlopen", lambda *a, **kw: Response(b"{}"))
    calls = []

    def fake_post(base, body, route):
        calls.append(route)
        mode = "b1_serial_ordinary" if route == "serial" else "b1_tree_eligible"
        return {"wall_seconds": 2.0 if route == "serial" else 1.0,
                "text": "same", "finish_reason": "stop",
                "usage": {"completion_tokens": 3},
                "receipt": {"output_token_ids_sha256": "equal", "identity": "native",
                            "active_width_modes": {mode: 1}, "route_requested": route,
                            "qualified": False, "apcv2_lookup": False,
                            "apcv2_store": False}}

    monkeypatch.setattr(module, "post", fake_post)
    result = module.run(SimpleNamespace(
        target="target", source="source", mlx_lm_source="mlx_lm",
        drafter="drafter", prepared_report="prepared", prepared_sha="hash",
        dry_run=False))
    assert result["status"] == "completed"
    assert calls == ["serial", "auto", "serial", "auto", "auto", "serial",
                     "serial", "auto"] * 2
    assert all(arm["tree_over_serial_ratio"] == 0.5
               and all(pair["token_ids_exact"] for pair in arm["pairs"])
               for arm in result["arms"].values())


@private_gate_test
def test_owned_feature_gate_cpu_dry_run_and_call_order(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    script = root / "qualification/runs/tree15-b1-discriminator-20261003/tensorfold_owned_feature_gate.py"
    spec = importlib.util.spec_from_file_location("tensorfold_owned_feature_gate_cpu", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    import transformers
    import psutil
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=96 << 30))
    monkeypatch.setattr(module, "free_port", lambda: 12345)

    class Process:
        returncode = None

        def poll(self):
            return None

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

    class Response(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(module, "urlopen", lambda *a, **kw: Response(b"{}"))
    calls = []

    def fake_post(base, path, body, route="auto", **kwargs):
        calls.append((path, route, body.get("max_tokens")))
        if (route == "invalid" or body.get("temperature", 0) != 0
                or body.get("stream") or body.get("stop") is not None
                or body.get("tools") is not None or body.get("max_tokens", 0) > 192):
            raise HTTPError(base + path, 400, "unsupported", {}, None)
        shared = threading.current_thread().name.startswith("ThreadPoolExecutor")
        mode = ("b2plus_shared_ordinary" if shared else
                "b1_serial_ordinary" if route == "serial" else "b1_tree_eligible")
        round_modes = {} if body["max_tokens"] == 1 else {mode: 1}
        marker = next((name for name in ("AXR7319", "BVK4826", "CPL5942")
                       if name in body.get("prompt", "")), "violet")
        prompt_count = len(Tokenizer().encode(body.get("prompt", "")))
        return {"wall_seconds": 1.2 if route == "serial" else 1.0,
                "text": marker, "finish_reason": "length" if body["max_tokens"] == 1 else "stop",
                "usage": {"prompt_tokens": prompt_count, "completion_tokens": 1},
                "receipt": {"identity": "native", "cache_policy": "disabled",
                            "output_token_ids_sha256": marker, "qualified": False,
                            "apcv2_lookup": False, "apcv2_store": False,
                            "selected_by": "explicit_standard_header",
                            "route_requested": route, "active_width_modes": round_modes,
                            "active_width_counts": {"2": 1} if shared else {"1": 1},
                            "timing": {
                                "clock": "host_monotonic_ns", "prompt_tokens": prompt_count,
                                "completion_tokens": 1, "cached_prompt_tokens": 0,
                                "queue_wait_seconds": 0.01,
                                "effective_prefill_tokens_per_second": 100,
                                "rate_definition": "tokens_after_first / first_to_final_token_seconds",
                                "decode_rate_defined": shared,
                                "decode_tokens_per_second": 100 if shared else None},
                            "drafted": 0, "accepted": 0}}

    monkeypatch.setattr(module, "post", fake_post)
    args = SimpleNamespace(target="target", source="source", mlx_lm_source="mlx_lm",
                           drafter="drafter", dry_run=True)
    dry = module.run(args)
    assert dry["status"] == "dry_run"
    assert dry["context_targets"]["near8k"] >= 7520
    args.dry_run = False
    result = module.run(args)
    assert result["status"] == "completed"
    assert all(result["checks"].values())
    assert result["cache_policy"] == "disabled" and result["qualified"] is False
    assert len([row for row in calls if row[1] == "invalid"]) == 1
    assert result["cancellation_cleanup"].startswith("CPU timeout/close")
    for route in ("serial", "auto"):
        one = result["length"][route]
        assert one["receipt"]["active_width_modes"] == {}
        assert module.check_receipt(one, "native", route, None)
        changed = {**one, "receipt": {**one["receipt"], "apcv2_store": True}}
        assert not module.check_receipt(changed, "native", route, None)
        changed = {**one, "receipt": {**one["receipt"], "active_width_modes": {"b1_tree_eligible": 1}}}
        assert not module.check_receipt(changed, "native", route, None)


@private_gate_test
def test_owned_feature_gate_cap_is_separate_from_other_gates():
    script = (Path(__file__).resolve().parents[1] / "qualification/runs"
              / "tree15-b1-discriminator-20261003/tensorfold-owned-b1/run_gate.py")
    spec = importlib.util.spec_from_file_location("owned_feature_wrapper_cpu", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.cap_error("feature", 600, 48) is None
    assert "600" in module.cap_error("feature", 601, 48)
    assert "180" in module.cap_error("service", 181, 48)


@private_gate_test
def test_owned_disconnect_gate_cpu_packet(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    script = root / "qualification/runs/tree15-b1-discriminator-20261003/tensorfold_owned_disconnect.py"
    spec = importlib.util.spec_from_file_location("tensorfold_owned_disconnect_cpu", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "free_port", lambda: 12345)
    result = module.run(SimpleNamespace(target="target", source="source",
                                        mlx_lm_source="mlx_lm", drafter="drafter",
                                        dry_run=True))
    assert result["status"] == "dry_run"
    assert result["cache_policy"] == "disabled"
    assert result["qualified"] is False
    assert result["command"][-2:] == ["--tensorfold-owned-drafter", "drafter"]
