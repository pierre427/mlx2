"""CPU HTTP/event-wait contract; no model loading or GPU work."""

import argparse
import json
import queue
import threading
from collections import Counter
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from mlx2 import server
from mlx2.generation_timeout import validate_generation_event_timeout
from mlx2.serving import Job, ServingEngine


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        None,
        "7200",
        0,
        -1,
        0.5,
        28801,
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
)
def test_timeout_validator_rejects_invalid_and_bool(value):
    with pytest.raises(ValueError, match="finite number"):
        validate_generation_event_timeout(value)


@pytest.mark.parametrize("value", [1, 1800, 7200.5, 28800])
def test_timeout_validator_valid_bounds(value):
    assert validate_generation_event_timeout(value) == float(value)


@pytest.mark.parametrize("value", ["0", "28801", "nan", "inf", "true"])
def test_timeout_cli_invalid(value):
    with pytest.raises(SystemExit):
        server.build_parser().parse_args(
            ["--model", "fixture", "--generation-event-timeout-seconds", value]
        )


def test_timeout_cli_default_and_explicit():
    parser = server.build_parser()
    assert (
        parser.parse_args(["--model", "fixture"]).generation_event_timeout_seconds
        == 1800
    )
    assert (
        parser.parse_args(
            ["--model", "fixture", "--generation-event-timeout-seconds", "7200"]
        ).generation_event_timeout_seconds
        == 7200
    )
    with pytest.raises(argparse.ArgumentTypeError):
        server.generation_event_timeout_arg(True)


class Clock:
    now = 0.0

    def monotonic(self):
        return self.now


class DelayedEvents:
    def __init__(self, clock, ready):
        self.clock, self.ready = clock, list(ready)

    def get(self, timeout):
        self.clock.now += timeout
        if self.ready and self.clock.now >= self.ready[0][0]:
            return self.ready.pop(0)[1]
        raise queue.Empty


@pytest.mark.parametrize("timeout,succeeds", [(1800, False), (7200, True)])
def test_long_silent_prefill_uses_explicit_bound(monkeypatch, timeout, succeeds):
    clock = Clock()
    monkeypatch.setattr(server.time, "monotonic", clock.monotonic)
    events = DelayedEvents(clock, [(1900, {"text": "ready"})])
    if succeeds:
        assert server.wait_event(events, deadline_seconds=timeout) == {"text": "ready"}
    else:
        with pytest.raises(TimeoutError, match="generation timed out"):
            server.wait_event(events)
        assert clock.now == 1800


def test_nonstream_gap_deadline_resets_per_event_not_total(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(server.time, "monotonic", clock.monotonic)
    job = SimpleNamespace(
        events=DelayedEvents(
            clock,
            [
                (1700, {"text": "Hello"}),
                (3400, {"finish_reason": "stop", "receipt": {"route": "ordinary"}}),
            ],
        ),
        prompt_tokens=1,
        completion_tokens=1,
    )
    choice, _, _ = server.collect_nonstream_job(job, {}, chat=True)
    assert choice["message"]["content"] == "Hello" and clock.now == 3400


def test_parallel_collectors_receive_same_bound_and_cancel_siblings(monkeypatch):
    seen = []

    def fail(job, body, *, chat, generation_event_timeout_seconds):
        seen.append(generation_event_timeout_seconds)
        raise TimeoutError("generation timed out")

    monkeypatch.setattr(server, "collect_nonstream_job", fail)
    jobs = [SimpleNamespace(cancelled=threading.Event()) for _ in range(2)]
    with pytest.raises(TimeoutError):
        server.collect_parallel_samples(
            jobs, {}, chat=True, generation_event_timeout_seconds=7200
        )
    assert seen == [7200, 7200] and all(job.cancelled.is_set() for job in jobs)


class Engine:
    generation_event_timeout_seconds = 7200
    model_path = "fixture"
    max_context = 16384
    max_request_bytes = server.MIN_REQUEST_BODY_BYTES

    def __init__(self):
        self.lock = threading.Lock()
        self.counts = Counter()
        self.job = None

    def status(self):
        return {
            "healthy": True,
            "model": "fixture",
            "http": {"max_request_bytes": self.max_request_bytes},
            "settings": {"generation_event_timeout_seconds": 7200},
        }

    def batching_status(self):
        return {"schema": "mlx2.batch-runtime.v1", "gauges": {"queue_depth": 0}}

    def submit(self, body, *, tenant_id="default"):
        self.job = Job(body)
        self.job.tenant_id = tenant_id
        self.job.prompt_tokens = 1
        self.job.completion_tokens = 1
        self.job.events.put({"text": "Hello"})
        self.job.events.put({"finish_reason": "stop", "receipt": {"route": "ordinary"}})
        return self.job


@pytest.mark.parametrize("stream", [False, True])
def test_http_single_stream_propagates_bound_and_timeout_cancels(monkeypatch, stream):
    engine = Engine()
    seen = []

    def timed_out(events, **kwargs):
        seen.append(kwargs["deadline_seconds"])
        raise TimeoutError("generation timed out")

    monkeypatch.setattr(server, "wait_event", timed_out)
    http = ThreadingHTTPServer(("127.0.0.1", 0), server.handler_for(engine))
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        body = {"messages": [{"role": "user", "content": "hello"}], "stream": stream}
        req = Request(
            f"http://127.0.0.1:{http.server_port}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(HTTPError) as failure:
            urlopen(req, timeout=5)
        assert failure.value.code == 503
        assert seen == [7200]
        # HTTP headers can reach the client before the handler's finally runs.
        # Synchronize on cancellation rather than racing the response write.
        assert engine.job.cancelled.wait(timeout=5)
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=5)


def test_handler_profile_mismatch_refused():
    with pytest.raises(ValueError, match="match engine serving profile"):
        server.handler_for(Engine(), generation_event_timeout_seconds=3600)


@pytest.mark.parametrize("value", [True, 0, float("nan"), 28801])
def test_engine_rejects_bad_timeout_before_thread_or_model(value):
    with pytest.raises(ValueError, match="generation event timeout"):
        ServingEngine("unused", generation_event_timeout_seconds=value)


def test_keepalive_does_not_extend_engine_event_deadline(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(server.time, "monotonic", clock.monotonic)
    writes = []
    with pytest.raises(TimeoutError):
        server.wait_event(
            DelayedEvents(clock, []),
            deadline_seconds=1800,
            idle_seconds=15,
            on_idle=lambda: writes.append(clock.now),
            idle_since=lambda: 0,
        )
    assert clock.now == 1800 and len(writes) > 100


def test_cli_timeout_enters_engine_profile_kwargs():
    args = server.build_parser().parse_args(
        ["--model", "fixture", "--generation-event-timeout-seconds", "7200"]
    )
    kwargs = server.serving_engine_kwargs(
        args,
        None,
        native_mtp=False,
        approximate_kv=False,
        max_request_bytes=server.MIN_REQUEST_BODY_BYTES,
    )
    assert kwargs["generation_event_timeout_seconds"] == 7200


def test_qualified_settings_reject_different_event_timeout(tmp_path, monkeypatch):
    from mlx2 import qualification

    monkeypatch.setattr(qualification, "unqualifiable_candidate", lambda _: None)
    path = tmp_path / "qualification.json"
    path.write_text(
        json.dumps(
            {
                "qualification_harness": qualification.APPROVED_QUALIFICATION_HARNESS,
                "runtime": {},
                "artifact": "fixture",
                "settings": {"generation_event_timeout_seconds": 1800},
            }
        )
    )
    with pytest.raises(ValueError, match="does not match serving settings"):
        qualification.load_qualified_route(
            path,
            runtime={},
            artifact="fixture",
            settings={"generation_event_timeout_seconds": 7200},
            descriptor=None,
            name="fixture",
        )
