"""Prompt logprobs through the real ServingEngine task path and HTTP handler (CPU).

The engine here is a ``ServingEngine`` whose scoring methods are the real ones
(``submit_prompt_scoring``, ``_step_prompt_scoring``, fidelity, memory and
length gates); a small worker thread stands in for the scheduler loop, calling
``_step_prompt_scoring`` between simulated decode steps.  Ordinary generation
is the FakeEngine contract (``submit``).  Covers review findings H1-H3, M2-M4
and the reviewer's drain probe as a regression.
"""

import json
import threading
import time
from collections import Counter
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import mlx.core as mx
import pytest
from test_prompt_logprobs_cpu import CharTokenizer, reference_logprobs, tiny_model

from mlx2.prompt_logprobs import PromptLogprobsPolicy, PromptLogprobsRefused, busy
from mlx2.server import handler_for, validate_request, wait_prompt_scoring
from mlx2.serving import Job, ServingEngine

PROMPT = "abcdefghij"


def char_ids(text):
    return [ord(char) - 97 for char in text]


class _NoAPC:
    """Any APCv2 store access fails, except the headroom accounting read."""

    def unleased_resident_nbytes(self):
        return 0

    def __getattr__(self, name):
        raise AssertionError(f"prompt scoring touched APCv2 ({name})")


class ScoringServingEngine(ServingEngine):
    """Real scoring path; FakeEngine-style generation and status."""

    model_path = "fixture"
    hard_reserve_gib = 0.0

    def admission_headroom_fn(self):
        return getattr(self, "headroom", lambda: 1 << 40)

    def status(self):
        return {"healthy": True, "error": None, "model": "fixture",
                "http": {"max_request_bytes": 1 << 20}}

    def batching_status(self, tenant_id=None):
        return {"schema": "mlx2.batch-runtime.v1", "gauges": {"queue_depth": 0}}

    def submit(self, request, *, tenant_id="default", admitted=False):
        with self._admission_section():
            self._ensure_admission("generation", admitted=admitted)
        job = Job(request)
        job.prompt_tokens, job.completion_tokens = len(PROMPT), 2
        if request.get("logprobs") or request.get("top_logprobs"):
            for token in (5, 6):
                job.events.put({"logprob": {"id": token, "token": str(token), "logprob": -0.5,
                                            "top_logprobs": [{"id": token, "logprob": -0.5}]}})
        job.events.put({"text": "fg"})
        job.events.put({"finish_reason": "stop", "receipt": {"cache": "apcv2",
                                                              "model_revision": self.model_revision}})
        self.submitted.append(request)
        return job


def make_engine(adapter=None, *, worker=True):
    model = tiny_model()
    adapter = adapter or SimpleNamespace(
        model=model, tokenizer=CharTokenizer(), identity={"fingerprint": "f" * 16},
        prompt_tokens=lambda request: char_ids(request["prompt"]),
    )
    engine = ScoringServingEngine.__new__(ScoringServingEngine)
    engine.ready = threading.Event()
    engine.ready.set()
    engine.error = None
    engine.stop_event = threading.Event()
    engine.lock = threading.RLock()
    engine.submission_lock = threading.RLock()
    engine.prompt_lock = threading.Lock()
    engine._exclusive_operation_done = threading.Condition(engine.submission_lock)
    engine._service_state = "serving"
    engine._quiesce_worker_owned = False
    engine._admission_leases = set()
    engine.jobs = {}
    engine._admissions_preparing = 0
    engine.counts = Counter()
    engine.operation_receipts = []
    engine.model_revision = 7
    engine.max_context = 4096
    engine.adapter = adapter
    engine.apc = _NoAPC()
    engine.submitted = []
    engine.decode_steps = 0
    engine.step_gate = threading.Event()
    engine.step_gate.set()

    def loop():
        while not engine.stop_event.is_set():
            engine.step_gate.wait()
            engine._step_prompt_scoring(mx)
            engine.decode_steps += 1  # a live lane keeps decoding between chunks
            time.sleep(0.002)

    engine.thread = threading.Thread(target=loop, daemon=True)
    if worker:
        engine.thread.start()
    else:
        engine.thread = SimpleNamespace(is_alive=lambda: True)
    return engine, model


@pytest.fixture
def served():
    started = []

    def start(policy=None, handler_kwargs=None, **kwargs):
        engine, model = make_engine(**kwargs)
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(
            engine, prompt_logprobs=policy or PromptLogprobsPolicy(enabled=True),
            **(handler_kwargs or {})))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        started.append((engine, server, thread))
        return engine, model, f"http://127.0.0.1:{server.server_port}"

    yield start
    for engine, server, thread in started:
        engine.stop_event.set()
        engine.step_gate.set()
        server.shutdown()
        server.server_close()
        thread.join()


def post(base, path, body, headers=None):
    request = Request(base + path, data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urlopen(request, timeout=60) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        return exc.code, json.load(exc)


# ---------------------------------------------------------------- HTTP, real path


def test_http_echo_max_tokens_zero_matches_reference(served):
    engine, model, base = served(PromptLogprobsPolicy(enabled=True, chunk_tokens=3))
    status, payload = post(base, "/v1/completions", {"prompt": PROMPT, "echo": True, "logprobs": 2,
                                                      "max_tokens": 0, "prompt_logprobs": 2})
    assert status == 200, payload
    choice = payload["choices"][0]
    assert choice["text"] == PROMPT and choice["finish_reason"] == "length"
    lp = choice["logprobs"]
    assert lp["tokens"] == list(PROMPT) and lp["text_offset"] == list(range(len(PROMPT)))
    assert lp["token_logprobs"][0] is None and lp["top_logprobs"][0] is None
    reference = reference_logprobs(model, char_ids(PROMPT))
    tokens = char_ids(PROMPT)
    worst = max(abs(lp["token_logprobs"][p] - reference[p - 1, tokens[p]].item())
                for p in range(1, len(tokens)))
    assert worst < 1e-4
    receipt = payload["mlx2"]["prompt_logprobs"]
    assert receipt["apc"]["read"] == "bypassed" and receipt["lane_width"] == 1
    assert receipt["chunks"] == 3 and receipt["model_revision"] == 7
    assert receipt["numerics"]["fidelity"] == "exact"
    assert engine.submitted == [] and engine.counts["prompt_logprobs_completed"] == 1
    assert engine._prompt_scoring_task is None


def test_http_echo_with_generation_takes_the_ordinary_path(served):
    engine, _model, base = served()
    status, payload = post(base, "/v1/completions", {"prompt": PROMPT, "echo": True, "logprobs": 1,
                                                      "max_tokens": 2})
    assert status == 200, payload
    choice = payload["choices"][0]
    assert choice["text"] == PROMPT + "fg"
    lp = choice["logprobs"]
    assert lp["token_logprobs"][-2:] == [-0.5, -0.5]
    # Generated offsets continue from the prompt tokens, increasing.
    assert lp["text_offset"][len(PROMPT):] == [len(PROMPT), len(PROMPT) + 1]
    assert len(lp["content"]) == 2
    assert payload["mlx2"]["cache"] == "apcv2" and "prompt_logprobs" in payload["mlx2"]
    # Generation went through ordinary submit with an admission lease.
    assert len(engine.submitted) == 1 and "echo" not in engine.submitted[0]


def test_http_chat_prompt_logprobs(served):
    adapter = SimpleNamespace(model=tiny_model(), tokenizer=CharTokenizer(), identity={},
                              prompt_tokens=lambda request: char_ids("chat" + request["messages"][0]["content"]))
    _engine, _model, base = served(adapter=adapter)
    status, payload = post(base, "/v1/chat/completions",
                           {"messages": [{"role": "user", "content": "hi"}],
                            "prompt_logprobs": 0, "max_tokens": 0})
    assert status == 200, payload
    rows = payload["prompt_logprobs"]
    assert rows[0] is None and len(rows) == len("chathi")
    assert set(rows[1]) == {str(char_ids("chathi")[1])}
    assert payload["choices"][0]["message"] == {"role": "assistant", "content": ""}


def test_flag_off_echo_matches_origin_main(served):
    """origin/main refused echo/prompt_logprobs as unknown fields (400)."""
    engine, _model, base = served(PromptLogprobsPolicy())
    for path, body in (
        ("/v1/completions", {"prompt": PROMPT, "echo": True}),
        ("/v1/completions", {"prompt": PROMPT, "echo": True, "logprobs": 1, "max_tokens": 0}),
        ("/v1/completions", {"prompt": PROMPT, "echo": False}),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "prompt_logprobs": 0}),
    ):
        status, payload = post(base, path, body)
        with pytest.raises(ValueError) as main_error:
            validate_request(body, path.endswith("chat/completions"))
        assert (status, payload["error"]["message"]) == (400, str(main_error.value)), payload
        assert payload["mlx2"]["prompt_logprobs"]["reason"] == "capability_disabled"
    # A null echo is the dropped-unknown case main had: still the same 400.
    status, _ = post(base, "/v1/completions", {"prompt": PROMPT, "max_tokens": 0})
    assert status == 400
    assert engine.submitted == [] and engine._prompt_scoring_task is None
    status, payload = post(base, "/v1/completions", {"prompt": PROMPT})
    assert status == 200 and payload["choices"][0]["text"] == "fg"


def test_http_refusals_carry_receipts(served):
    engine, _model, base = served(PromptLogprobsPolicy(enabled=True, max_top=1, max_prompt_tokens=4))
    cases = [
        ({"prompt_logprobs": 3, "max_tokens": 0}, 400, "prompt_logprobs_top_cap"),
        ({"prompt_logprobs": 1, "max_tokens": 0}, 400, "prompt_logprobs_length_cap"),
        ({"prompt_logprobs": 1, "stream": True}, 400, "prompt_logprobs_stream_unsupported"),
        ({"echo": True, "prefill_handoff": "h"}, 400, "prompt_logprobs_weight_program_unsupported"),
    ]
    for extra, status_code, code in cases:
        status, payload = post(base, "/v1/completions", {"prompt": PROMPT, **extra})
        assert (status, payload["error"]["code"]) == (status_code, code), payload
        assert payload["mlx2"]["prompt_logprobs"]["status"] == "refused"
    assert engine.submitted == []


SCORE = {"prompt": PROMPT, "prompt_logprobs": 0, "max_tokens": 0}


def test_tenant_list_needs_tenant_auth_so_a_header_cannot_spoof_it():
    """The reviewer's tenant-spoof probe: X-Tenant-ID is client-set without auth."""
    engine, _model = make_engine(worker=False)
    with pytest.raises(ValueError, match="tenant authentication"):
        handler_for(engine, prompt_logprobs=PromptLogprobsPolicy(enabled=True, tenants=("research",)))


class FakeTenantAuth:
    """Bearer ``<tenant>:<scope>[,<scope>]`` -> a verified principal."""

    header_policy = "ignore"

    def authenticate(self, headers, *, required_scope="inference"):
        from mlx2.tenant_auth import TenantAuthError

        value = (headers.get("Authorization") or "").removeprefix("Bearer ")
        if ":" not in value:
            raise TenantAuthError("missing")
        tenant, scopes = value.split(":", 1)
        scopes = frozenset(scopes.split(","))
        if required_scope is not None and required_scope not in scopes:
            raise TenantAuthError("scope")
        return SimpleNamespace(tenant=tenant, method="api_key", scopes=scopes)


def test_tenant_auth_admits_operator_scope_and_listed_tenants(served):
    engine, _model, base = served(PromptLogprobsPolicy(enabled=True, tenants=("research",)),
                                  handler_kwargs={"tenant_authenticator": FakeTenantAuth()})
    cases = [
        ("research:inference", 200),
        ("ops:inference,prompt-logprobs", 200),
        # usage-admin is tenant self-service for usage views, not a scoring grant.
        ("self:inference,usage-admin", 403),
        ("other:inference", 403),
    ]
    for credential, expected in cases:
        status, payload = post(base, "/v1/completions", SCORE,
                               headers={"Authorization": f"Bearer {credential}",
                                        "X-Tenant-ID": "research"})
        assert status == expected, (credential, payload)
        if expected == 403:
            assert payload["error"]["code"] == "prompt_logprobs_forbidden"
            assert payload["mlx2"]["prompt_logprobs"]["reason"] == "tenant_not_authorized"
    # Echo without logprobs scores nothing and needs no scoring access.
    status, _ = post(base, "/v1/completions", {"prompt": PROMPT, "echo": True, "max_tokens": 1},
                     headers={"Authorization": "Bearer other:inference"})
    assert status == 200


def test_semantic_session_memory_is_refused(served):
    class Middleware:
        classifier_token_ids = ()

        def prepare(self, body, *, tenant_id, authenticated_tenant):
            return {**body, "_mlx2_semantic_fingerprint": "f"}, object()

        def receipt(self, state):
            return {}

        def complete(self, *args, **kwargs):
            return None

    adapter = SimpleNamespace(model=tiny_model(), tokenizer=CharTokenizer(), identity={},
                              prompt_tokens=lambda request: char_ids("chathi"))
    engine, _model, base = served(adapter=adapter, handler_kwargs={"semantic_middleware": Middleware()})
    status, payload = post(base, "/v1/chat/completions",
                           {"messages": [{"role": "user", "content": "hi"}], "session_id": "s1",
                            "prompt_logprobs": 0, "max_tokens": 0})
    assert (status, payload["error"]["code"]) == (400, "prompt_logprobs_semantic_unsupported"), payload
    receipt = payload["mlx2"]["prompt_logprobs"]
    assert receipt["internal_fields"] == ["_mlx2_semantic_fingerprint"] and receipt["semantic_state"]
    assert engine._prompt_scoring_task is None


def test_template_error_in_the_worker_render_is_a_400(served):
    def broken(request):
        raise ValueError("this model's chat template requires a field the request does not provide")

    adapter = SimpleNamespace(model=tiny_model(), tokenizer=CharTokenizer(), identity={},
                              prompt_tokens=broken)
    engine, _model, base = served(adapter=adapter)
    status, payload = post(base, "/v1/completions", SCORE)
    assert (status, payload["error"]["code"]) == (400, "prompt_logprobs_invalid_request"), payload
    assert payload["mlx2"]["prompt_logprobs"]["reason"] == "invalid_request"
    assert engine.counts["prompt_logprobs_failed"] == 1 and engine._prompt_scoring_task is None


# ---------------------------------------------------------------- concurrency (H1)


def test_scoring_neither_drains_nor_blocks_admission():
    """The reviewer's drain probe: an in-flight job no longer stalls arrivals."""
    engine, _model = make_engine()
    try:
        engine.jobs = {"long": object()}  # a long generation that never finishes
        policy = PromptLogprobsPolicy(enabled=True, chunk_tokens=2)
        result = {}

        def scorer():
            started = time.monotonic()
            scored = wait_prompt_scoring(engine, {"prompt": PROMPT}, SimpleNamespace(top_k=0), policy)
            result["scorer"] = (len(scored["scores"].logprobs), time.monotonic() - started)

        def arrival():
            time.sleep(0.05)
            started = time.monotonic()
            with engine._admission_section():
                pass
            result["arrival_wait"] = time.monotonic() - started

        threads = [threading.Thread(target=scorer), threading.Thread(target=arrival)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        assert result["scorer"][0] == len(PROMPT) - 1 and result["scorer"][1] < 10
        assert result["arrival_wait"] < 0.5
        assert engine.decode_steps > 5  # lanes kept stepping beside the scoring chunks
        assert engine._exclusive_operation_active is False
    finally:
        engine.stop_event.set()


def test_second_scoring_is_busy_and_slot_frees():
    engine, _model = make_engine()
    try:
        engine.step_gate.clear()  # the worker is mid-step elsewhere
        policy = PromptLogprobsPolicy(enabled=True)
        first = engine.submit_prompt_scoring({"prompt": PROMPT}, top_k=0, policy=policy)
        with pytest.raises(PromptLogprobsRefused) as error:
            engine.submit_prompt_scoring({"prompt": PROMPT}, top_k=0, policy=policy)
        assert error.value.http_status == 503 and error.value.mlx2["prompt_logprobs"]["reason"] == "busy"
        engine.step_gate.set()
        assert first.done.wait(10) and first.error is None
        assert engine._prompt_scoring_task is None
        engine.submit_prompt_scoring({"prompt": PROMPT}, top_k=0, policy=policy).done.wait(10)
    finally:
        engine.stop_event.set()
        engine.step_gate.set()


def test_timeout_is_a_503_receipt_and_frees_an_unstarted_slot():
    engine, _model = make_engine(worker=False)  # no worker ever steps
    policy = PromptLogprobsPolicy(enabled=True, timeout_seconds=0.3)
    with pytest.raises(PromptLogprobsRefused) as error:
        wait_prompt_scoring(engine, {"prompt": PROMPT}, SimpleNamespace(top_k=0), policy,
                            poll_seconds=0.05)
    assert error.value.http_status == 503
    assert error.value.mlx2["prompt_logprobs"]["reason"] == "timeout"
    assert engine._prompt_scoring_task is None


def test_failure_mid_scoring_leaves_no_stuck_state():
    model = tiny_model()
    calls = []

    class Flaky:
        def __call__(self, x, cache=None):
            calls.append(int(x.shape[1]))
            if len(calls) == 2:
                raise RuntimeError("device fault in chunk 2")
            return model(x, cache=cache)

        def make_cache(self):
            from mlx2.runtime.models.cache import make_prompt_cache
            return make_prompt_cache(model)

    adapter = SimpleNamespace(model=Flaky(), tokenizer=CharTokenizer(), identity={},
                              prompt_tokens=lambda request: char_ids(request["prompt"]))
    engine, _ = make_engine(adapter)
    try:
        policy = PromptLogprobsPolicy(enabled=True, chunk_tokens=3)
        with pytest.raises(PromptLogprobsRefused) as error:
            wait_prompt_scoring(engine, {"prompt": PROMPT}, SimpleNamespace(top_k=0), policy)
        assert error.value.http_status == 500
        assert error.value.mlx2["prompt_logprobs"]["reason"] == "scoring_failed"
        assert engine._prompt_scoring_task is None and engine.counts["prompt_logprobs_failed"] == 1
        scored = wait_prompt_scoring(engine, {"prompt": PROMPT}, SimpleNamespace(top_k=0), policy)
        assert len(scored["scores"].logprobs) == len(PROMPT) - 1
    finally:
        engine.stop_event.set()


def test_cancel_mid_run_frees_the_slot():
    engine, _model = make_engine()
    try:
        policy = PromptLogprobsPolicy(enabled=True, chunk_tokens=2)
        task = engine.submit_prompt_scoring({"prompt": PROMPT * 20}, top_k=0, policy=policy)
        while task.run is None and not task.done.is_set():
            time.sleep(0.001)
        engine.cancel_prompt_scoring(task)
        assert task.done.wait(10)
        assert task.error is not None and task.error.mlx2["prompt_logprobs"]["reason"] == "cancelled"
        assert engine._prompt_scoring_task is None
    finally:
        engine.stop_event.set()


def test_model_change_during_scoring_fails_closed():
    engine, _model = make_engine()
    try:
        engine.step_gate.clear()
        policy = PromptLogprobsPolicy(enabled=True, chunk_tokens=2)
        task = engine.submit_prompt_scoring({"prompt": PROMPT}, top_k=0, policy=policy)
        engine._step_prompt_scoring(mx)  # first chunk renders and binds at revision 7
        engine.model_revision = 8
        engine.step_gate.set()
        assert task.done.wait(10)
        assert task.error.mlx2["prompt_logprobs"]["reason"] == "model_changed"
    finally:
        engine.stop_event.set()
        engine.step_gate.set()


# ---------------------------------------------------------------- admission and fidelity (H2/H3)


def _score(engine, body=None, policy=None):
    return wait_prompt_scoring(engine, body or {"prompt": PROMPT}, SimpleNamespace(top_k=0),
                               policy or PromptLogprobsPolicy(enabled=True))


def test_context_and_memory_admission():
    engine, _model = make_engine()
    try:
        engine.max_context = 5
        with pytest.raises(PromptLogprobsRefused) as error:
            _score(engine)
        assert error.value.code == "prompt_logprobs_length_cap"
        engine.max_context = 4096
        engine.headroom = lambda: 1024
        with pytest.raises(PromptLogprobsRefused) as error:
            _score(engine)
        receipt = error.value.mlx2["prompt_logprobs"]
        assert error.value.http_status == 503 and receipt["reason"] == "memory_admission_refused"
        assert receipt["required_bytes"] > receipt["available_bytes"] == 1024
        engine.headroom = lambda: 1 << 40
        assert _score(engine)["receipt"]["memory"]["checked"] is True
    finally:
        engine.stop_event.set()


def test_approximate_routes_are_refused():
    engine, _model = make_engine()
    try:
        engine.apc_reuse_disabled_reason = "moe_keep_topk_8"
        with pytest.raises(PromptLogprobsRefused) as error:
            _score(engine)
        assert error.value.http_status == 501
        assert error.value.mlx2["prompt_logprobs"]["reason"] == "route_not_exact"
        engine.apc_reuse_disabled_reason = None
        engine.adapter.numeric_profile = "official:ops_approximate_unqualified"
        with pytest.raises(PromptLogprobsRefused):
            _score(engine)
    finally:
        engine.stop_event.set()


def test_int8_prefill_never_engages_during_scoring():
    model = tiny_model()
    rows = []

    class Watch:
        def __call__(self, x, cache=None):
            rows.append(int(x.shape[1]))
            return model(x, cache=cache)

        def make_cache(self):
            from mlx2.runtime.models.cache import make_prompt_cache
            return make_prompt_cache(model)

    adapter = SimpleNamespace(model=Watch(), tokenizer=CharTokenizer(), identity={},
                              prompt_tokens=lambda request: char_ids(request["prompt"]))
    engine, _ = make_engine(adapter)
    try:
        engine.int8_prefill_handle = SimpleNamespace(policy=SimpleNamespace(row_threshold=4))
        scored = _score(engine, {"prompt": PROMPT * 2}, PromptLogprobsPolicy(enabled=True, chunk_tokens=256))
        assert max(rows) < 4
        assert scored["receipt"]["int8_prefill"]["chunk_tokens"] == 3
        assert scored["receipt"]["int8_prefill"]["engaged"] is False
    finally:
        engine.stop_event.set()


def test_disconnect_cancels_between_chunks(monkeypatch):
    import mlx2.server as server_module

    engine, _model = make_engine()
    try:
        engine.step_gate.clear()
        monkeypatch.setattr(server_module, "client_disconnected", lambda connection: True)
        with pytest.raises(server_module.ClientGone):
            wait_prompt_scoring(engine, {"prompt": PROMPT}, SimpleNamespace(top_k=0),
                                PromptLogprobsPolicy(enabled=True), connection=object(),
                                poll_seconds=0.01)
        assert engine._prompt_scoring_task is None  # not started: slot freed at once
    finally:
        engine.stop_event.set()
        engine.step_gate.set()


def test_busy_refusal_shape():
    error = busy("x")
    assert error.http_status == 503 and error.code == "prompt_logprobs_busy"
    assert error.mlx2["prompt_logprobs"]["reason"] == "busy"


def test_cancel_after_the_worker_claims_keeps_one_owner_of_the_slot():
    engine, _model = make_engine(worker=False)
    policy = PromptLogprobsPolicy(enabled=True, chunk_tokens=2)
    task = engine.submit_prompt_scoring({"prompt": PROMPT}, top_k=0, policy=policy)
    entered, release = threading.Event(), threading.Event()
    original = engine._start_prompt_scoring

    def slow_start(task, mx, **kwargs):
        entered.set()
        release.wait(10)
        return original(task, mx, **kwargs)

    engine._start_prompt_scoring = slow_start
    worker = threading.Thread(target=engine._step_prompt_scoring, args=(mx,))
    worker.start()
    assert entered.wait(10)
    engine.cancel_prompt_scoring(task)  # the worker owns the task now
    assert engine._prompt_scoring_task is task
    with pytest.raises(PromptLogprobsRefused):
        engine.submit_prompt_scoring({"prompt": PROMPT}, top_k=0, policy=policy)
    release.set()
    worker.join(10)
    while not task.done.is_set():
        engine._step_prompt_scoring(mx)
    assert task.error.mlx2["prompt_logprobs"]["reason"] == "cancelled"
    assert engine._prompt_scoring_task is None
    assert engine.counts["prompt_logprobs_cancelled"] == 1
    assert engine.counts["prompt_logprobs_failed"] == 0 and engine.counts["prompt_logprobs_completed"] == 0


def test_exclusive_operation_preempts_a_running_scoring_task():
    engine, _model = make_engine()
    try:
        engine._kernel_route_receipt = lambda receipt: receipt
        policy = PromptLogprobsPolicy(enabled=True, chunk_tokens=2)
        task = engine.submit_prompt_scoring({"prompt": PROMPT * 50}, top_k=0, policy=policy)
        while task.run is None and not task.done.is_set():
            time.sleep(0.001)
        started = time.monotonic()
        assert engine._exclusive_adapter_operation("embeddings", lambda adapter: "ok", timeout=5) == "ok"
        assert time.monotonic() - started < 2
        assert task.done.wait(5)
        assert task.error.mlx2["prompt_logprobs"]["reason"] == "preempted_by_embeddings"
    finally:
        engine.stop_event.set()


def test_memory_is_charged_through_lane_admission_and_held_until_the_end(monkeypatch):
    from mlx2 import memory

    engine, _model = make_engine(worker=False)
    policy = PromptLogprobsPolicy(enabled=True, chunk_tokens=2)
    task = engine.submit_prompt_scoring({"prompt": PROMPT}, top_k=0, policy=policy)
    seen = []
    engine._step_prompt_scoring(mx, admit=lambda required: seen.append(required) or False)
    assert task.error.mlx2["prompt_logprobs"]["reason"] == "memory_admission_refused"
    assert task.error.mlx2["prompt_logprobs"]["method"] == "lane_admission"
    assert task.error.mlx2["prompt_logprobs"]["kv_charge"] == "lane_envelope_gib_per_1k_tokens"
    assert engine._prompt_scoring_reserved_bytes == 0 and engine._prompt_scoring_task is None

    task = engine.submit_prompt_scoring({"prompt": PROMPT}, top_k=0, policy=policy)
    engine._step_prompt_scoring(mx, admit=lambda required: True)
    reserved = engine._prompt_scoring_reserved_bytes
    reservation = task.state["memory"]["reservation"]
    # No hard reserve is held (lane admission adds its own), and after the
    # first chunk only the KV not yet filled stays reserved.
    assert reservation["transient_bytes"] <= reserved < seen[0]
    monkeypatch.setattr(memory, "execution_headroom", lambda: 10**12)
    assert ServingEngine.admission_headroom_fn(engine)() == 10**12 - reserved
    engine._step_prompt_scoring(mx)
    assert engine._prompt_scoring_reserved_bytes < reserved
    while not task.done.is_set():
        engine._step_prompt_scoring(mx)
    assert task.error is None and engine._prompt_scoring_reserved_bytes == 0
    assert ServingEngine.admission_headroom_fn(engine)() == 10**12


def test_headroom_shortfall_clears_the_allocator_once_then_rereads(monkeypatch):
    engine, _model = make_engine(worker=False)
    cleared = []
    monkeypatch.setattr(mx, "clear_cache", lambda: cleared.append(1))
    engine.headroom = lambda: (1 << 40) if cleared else 0
    task = engine.submit_prompt_scoring({"prompt": PROMPT}, top_k=0,
                                        policy=PromptLogprobsPolicy(enabled=True))
    while not task.done.is_set():
        engine._step_prompt_scoring(mx)
    assert task.error is None
    memory = task.result[1]["memory"]
    assert memory["allocator_cache_cleared"] is True and memory["method"] == "headroom"


def test_quantized_kv_route_is_refused_by_the_engine():
    from mlx2.runtime.models.cache import QuantizedKVCache

    model = tiny_model()

    class QuantizedRoute:
        def __call__(self, x, cache=None):
            return model(x, cache=cache)

        def make_cache(self):
            return [QuantizedKVCache() for _ in model.layers]

    adapter = SimpleNamespace(model=QuantizedRoute(), tokenizer=CharTokenizer(), identity={},
                              prompt_tokens=lambda request: char_ids(request["prompt"]))
    engine, _ = make_engine(adapter)
    try:
        with pytest.raises(PromptLogprobsRefused) as error:
            _score(engine)
        assert error.value.mlx2["prompt_logprobs"]["reasons"][0].startswith("quantized_kv_cache:")
    finally:
        engine.stop_event.set()


def test_receipt_labels_the_cache_actually_built():
    engine, _model = make_engine()
    try:
        receipt = _score(engine)["receipt"]
        assert receipt["kv_cache"] == {"kind": "fresh_request_private", "classes": ["KVCache"]}
    finally:
        engine.stop_event.set()


def test_cancel_or_preempt_before_the_worker_claims_answers_with_a_receipt():
    """The reviewer's preempt probe: an unclaimed task's waiter gets a 503 receipt."""
    engine, _model = make_engine(worker=False)
    policy = PromptLogprobsPolicy(enabled=True)

    def preempt():
        time.sleep(0.2)
        engine.cancel_prompt_scoring(engine._prompt_scoring_task, reason="preempted_by_embeddings")

    threading.Thread(target=preempt).start()
    with pytest.raises(PromptLogprobsRefused) as error:
        wait_prompt_scoring(engine, {"prompt": PROMPT}, SimpleNamespace(top_k=0), policy,
                            poll_seconds=0.05)
    assert error.value.http_status == 503
    assert error.value.mlx2["prompt_logprobs"]["reason"] == "preempted_by_embeddings"
    assert engine._prompt_scoring_task is None and engine._prompt_scoring_reserved_bytes == 0


def test_headroom_reading_is_untouched_without_a_reservation(monkeypatch):
    """Flag off (no task): admission_headroom_fn returns the reading object itself."""
    from mlx2 import memory

    engine, _model = make_engine(worker=False)
    reading = SimpleNamespace(bytes=7)
    monkeypatch.setattr(memory, "execution_headroom", lambda: reading)
    assert ServingEngine.admission_headroom_fn(engine)() is reading
