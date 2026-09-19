import queue
import threading
from types import SimpleNamespace

import pytest

from mlx2.server import collect_nonstream_job, validate_request
from mlx2.serving import Overloaded, ServingEngine


def test_parallel_sampling_validation_is_bounded_and_nonstreaming():
    body = validate_request({"messages": [{"role": "user", "content": "x"}], "n": 4})
    assert body["n"] == 4
    with pytest.raises(ValueError, match="streaming"):
        validate_request({"messages": [{"role": "user", "content": "x"}], "n": 2, "stream": True})
    with pytest.raises(ValueError, match="1 to 8"):
        validate_request({"messages": [{"role": "user", "content": "x"}], "n": 9})


def test_nonstream_sample_collector_keeps_receipt_and_usage():
    events = queue.Queue()
    events.put({"delta": {"content": "yes"}})
    events.put({"finish_reason": "stop", "receipt": {"route": "ordinary"}})
    job = SimpleNamespace(events=events, prompt_tokens=3, completion_tokens=1)
    choice, usage, receipt = collect_nonstream_job(job, {}, chat=True)
    assert choice["message"]["content"] == "yes"
    assert usage == {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
    assert receipt == {"route": "ordinary"}


def test_parallel_submission_reserves_the_whole_cohort():
    engine = object.__new__(ServingEngine)
    engine.ready = threading.Event()
    engine.ready.set()
    engine.error = None
    engine.thread = SimpleNamespace(is_alive=lambda: True)
    engine.slots = threading.BoundedSemaphore(2)
    engine.submission_lock = threading.Lock()
    engine.incoming = queue.Queue(maxsize=2)
    # Raw fixtures must preserve the logical queue-depth invariant established
    # by ServingEngine.__init__. Published cohort envelopes may represent more
    # than one queued job, so physical Queue.qsize() is not authoritative.
    engine.queued_jobs = 0
    engine.lock = threading.Lock()
    engine.jobs = {}
    engine.qualification_mode = True
    engine.route_capabilities = frozenset()
    engine.batch_metrics = SimpleNamespace(
        rejected=lambda *_: None, admitted=lambda *_: None
    )

    with pytest.raises(Overloaded, match="whole|available|capacity|inflight"):
        engine.submit_many([{"n": 1}, {"n": 1}, {"n": 1}])
    assert engine.incoming.empty()
    assert engine.queued_jobs == 0
    assert not engine.jobs
    assert engine.slots.acquire(blocking=False)
    assert engine.slots.acquire(blocking=False)
