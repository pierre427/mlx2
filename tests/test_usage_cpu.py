"""Per-tenant/per-session usage accounting and opt-in limits (CPU only)."""

import datetime as dt
import json
import os
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace as NS
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pytest

from mlx2 import usage
from mlx2.api_resources import FileStore, ResponseStore
from mlx2.server import handler_for
from mlx2.serving import Job
from mlx2.tenant_auth import TenantAuthenticator, TenantPrincipal, hash_api_key
from mlx2.usage import (
    UsageCalendar,
    UsageLedger,
    UsageLimitExceeded,
    UsageLimits,
)

ZONE = "America/Toronto"


def _ts(*args):
    return dt.datetime(*args, tzinfo=ZoneInfo(ZONE)).timestamp()


class Clock:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


def _limits(mode="admission", **windows):
    return UsageLimits.from_document({"version": 1, "mode": mode, "default": windows})


@pytest.fixture
def ledgers(tmp_path):
    opened = []

    def open_(directory=tmp_path / "usage", **kwargs):
        kwargs.setdefault("timezone", ZONE)
        kwargs.setdefault("writer", False)
        ledger = UsageLedger(directory, **kwargs)
        opened.append(ledger)
        return ledger

    yield open_
    for ledger in opened:
        try:
            ledger.close()
        except Exception:
            pass


# -- calendar -------------------------------------------------------------------


def test_calendar_windows_are_local_and_weeks_start_monday():
    calendar = UsageCalendar(ZONE)
    # Wednesday 2026-10-14, 23:30 local, already 2026-10-15 in UTC.
    now = _ts(2026, 10, 14, 23, 30)
    starts = calendar.starts(now)
    assert starts == {
        "today": dt.date(2026, 10, 14),
        "week": dt.date(2026, 10, 12),
        "month": dt.date(2026, 10, 1),
        "year": dt.date(2026, 1, 1),
    }
    resets = calendar.resets(now)
    assert resets["today"] == 30 * 60
    assert resets["week"] == 30 * 60 + 4 * 86400
    # 2026-11-01 is the DST end in Toronto: the month reset is an exact
    # local midnight, not a fixed 24 h multiple.
    assert resets["month"] == int(_ts(2026, 11, 1) - now)
    assert calendar.resets(_ts(2026, 12, 31, 12))["year"] == 12 * 3600


# -- ledger ---------------------------------------------------------------------


def test_windows_accumulate_and_survive_a_restart(ledgers, tmp_path):
    clock = Clock(_ts(2026, 10, 13, 9))  # Tuesday
    ledger = ledgers(clock=clock)
    ledger.record("acme", "s1", prompt_tokens=100, cached_tokens=40, completion_tokens=10,
                  reasoning_tokens=3)
    clock.value = _ts(2026, 10, 14, 9)  # Wednesday
    ledger.record("acme", "s1", prompt_tokens=50, completion_tokens=5, status="cancelled")
    ledger.record("beta", None, prompt_tokens=7, completion_tokens=1)
    windows = ledger.windows("view", "acme")
    assert windows["today"]["total_tokens"] == 55
    assert windows["week"]["total_tokens"] == 165
    assert windows["overall"]["requests"] == 2
    assert windows["overall"]["completed"] == 1
    assert windows["overall"]["cancelled"] == 1
    assert windows["overall"]["prefill_tokens"] == 110
    assert windows["overall"]["reasoning_tokens"] == 3
    assert windows["overall"]["since"] == "2026-10-13"
    ledger.close()

    clock.value = _ts(2026, 11, 2, 9)  # next month, next week
    reopened = ledgers(clock=clock)
    windows = reopened.windows("accounting", "acme")
    assert windows["today"]["total_tokens"] == 0
    assert windows["month"]["total_tokens"] == 0
    assert windows["year"]["total_tokens"] == 165
    assert windows["overall"]["total_tokens"] == 165
    assert reopened.tenants() == ["acme", "beta"]
    # Sessions are memory-only: a restart starts them over.
    assert reopened.sessions("view", "acme") == []


def test_background_writer_persists_without_close(tmp_path):
    ledger = UsageLedger(tmp_path / "u", flush_seconds=0.05, timezone=ZONE)
    try:
        ledger.record("acme", prompt_tokens=3, completion_tokens=4)
        deadline = 200
        while ledger.counts["flushes"] == 0 and deadline:
            threading.Event().wait(0.02)
            deadline -= 1
        assert ledger.counts["flushes"] >= 1
    finally:
        ledger.close()


def test_records_before_a_book_loads_are_not_lost_or_doubled(ledgers):
    ledger = ledgers()
    ledger.record("acme", prompt_tokens=1)  # pending, book not loaded
    ledger.flush()
    ledger.record("acme", prompt_tokens=2)  # pending again
    assert ledger.windows("view", "acme")["overall"]["prompt_tokens"] == 3
    ledger.record("acme", prompt_tokens=4)  # book loaded now
    ledger.flush()
    assert ledger.windows("view", "acme")["overall"]["prompt_tokens"] == 7
    ledger.close()
    assert ledgers().windows("view", "acme")["overall"]["prompt_tokens"] == 7


def test_tenant_clear_resets_the_view_but_not_accounting(ledgers):
    ledger = ledgers(limits=_limits(today={"total_tokens": 10}))
    ledger.record("acme", "s1", prompt_tokens=8, completion_tokens=4)
    ledger.record("beta", prompt_tokens=1)
    ledger.clear("acme", ledger="view")
    assert ledger.windows("view", "acme")["overall"]["total_tokens"] == 0
    assert ledger.sessions("view", "acme")[0]["total_tokens"] == 0
    assert ledger.windows("accounting", "acme")["overall"]["total_tokens"] == 12
    assert ledger.windows("view", "beta")["overall"]["total_tokens"] == 1
    assert ledger.collection("acme", "view")["cleared_at"] is not None
    with pytest.raises(UsageLimitExceeded):
        ledger.admit("acme")
    ledger.flush()
    ledger.close()
    reopened = ledgers()
    assert reopened.windows("view", "acme")["overall"]["total_tokens"] == 0
    assert reopened.windows("accounting", "acme")["overall"]["total_tokens"] == 12


def test_operator_clear_all_ledgers_lifts_the_limit(ledgers):
    ledger = ledgers(limits=_limits(today={"requests": 1}))
    ledger.record("acme")
    with pytest.raises(UsageLimitExceeded):
        ledger.admit("acme")
    ledger.clear("acme", ledger="all")
    assert ledger.admit("acme") is not None


def test_pause_is_per_ledger_and_survives_restart(ledgers):
    ledger = ledgers()
    ledger.set_paused(True, "acme", ledger="view")
    ledger.record("acme", prompt_tokens=5)
    assert ledger.windows("view", "acme")["overall"]["total_tokens"] == 0
    assert ledger.windows("accounting", "acme")["overall"]["total_tokens"] == 5
    ledger.close()
    reopened = ledgers()
    assert reopened.collection("acme", "view")["paused"] is True
    reopened.set_paused(False, "acme", ledger="view")
    reopened.record("acme", prompt_tokens=1)
    assert reopened.windows("view", "acme")["overall"]["total_tokens"] == 1


def test_global_pause_covers_every_tenant(ledgers):
    ledger = ledgers()
    ledger.set_paused(True, None, ledger="all")
    ledger.record("acme", prompt_tokens=5)
    assert ledger.windows("accounting", "acme")["overall"]["total_tokens"] == 0


def test_sessions_are_tenant_scoped_and_bounded(ledgers):
    ledger = ledgers(max_sessions=2)
    ledger.record("acme", "shared", prompt_tokens=1)
    ledger.record("beta", "shared", prompt_tokens=9)
    assert ledger.sessions("view", "acme")[0]["total_tokens"] == 1
    assert ledger.sessions("view", "beta")[0]["total_tokens"] == 9
    ledger.record("acme", "s2")
    ledger.record("acme", "s3")
    assert [row["session_id"] for row in ledger.sessions("view", "acme")] == ["s3", "s2"]
    assert ledger.counts["sessions_evicted"] == 1
    assert ledger.sessions("view", "acme", "missing") == []


def test_tenant_cardinality_is_bounded_but_configured_tenants_always_fit(ledgers):
    ledger = ledgers(max_tenants=1, known_tenants={"vip"})
    ledger.record("first")
    ledger.record("second")
    ledger.record("vip")
    assert set(ledger.tenants()) == {"first", usage.OVERFLOW_TENANT, "vip"}
    assert ledger.counts["overflow_records"] == 1


def test_directory_is_exclusive_and_private(ledgers, tmp_path):
    ledgers()
    with pytest.raises(ValueError, match="in use"):
        UsageLedger(tmp_path / "usage", writer=False)
    assert oct(os.stat(tmp_path / "usage").st_mode & 0o777) == "0o700"
    assert oct(os.stat(tmp_path / "usage" / usage.DB_NAME).st_mode & 0o777) == "0o600"


# -- limits ---------------------------------------------------------------------


def test_limits_file_is_validated(tmp_path):
    with pytest.raises(ValueError, match="version"):
        UsageLimits.from_document({"default": {}})
    with pytest.raises(ValueError, match="unknown window"):
        UsageLimits.from_document({"version": 1, "default": {"hour": {"requests": 1}}})
    with pytest.raises(ValueError, match="unknown counter"):
        UsageLimits.from_document({"version": 1, "default": {"today": {"bytes": 1}}})
    with pytest.raises(ValueError, match="nonnegative"):
        UsageLimits.from_document({"version": 1, "default": {"today": {"requests": -1}}})
    with pytest.raises(ValueError, match="tenant ids"):
        UsageLimits.from_document({"version": 1, "tenants": {"bad tenant": {}}})
    path = tmp_path / "limits.json"
    path.write_text(json.dumps({"version": 1, "tenants": {"acme": {"today": {"requests": 2}}}}))
    os.chmod(path, 0o666)
    with pytest.raises(ValueError, match="writable"):
        UsageLimits.load(path)
    os.chmod(path, 0o600)
    limits = UsageLimits.load(path)
    assert limits.for_tenant("acme") == {"today": {"requests": 2}}
    assert limits.for_tenant("other") == {}


def test_without_limits_admission_never_refuses(ledgers):
    ledger = ledgers()
    ledger.record("acme", prompt_tokens=10**9)
    assert ledger.admit("acme") is None


def test_admission_refuses_at_the_limit_with_window_reset(ledgers):
    clock = Clock(_ts(2026, 10, 14, 22))
    ledger = ledgers(clock=clock, limits=_limits(today={"total_tokens": 100}))
    first = ledger.admit("acme")
    ledger.record("acme", prompt_tokens=60, completion_tokens=40, reservation=first)
    with pytest.raises(UsageLimitExceeded) as caught:
        ledger.admit("acme")
    assert caught.value.retry_after == 2 * 3600
    assert caught.value.receipt()["counter"] == "total_tokens"
    assert ledger.admit("beta") is not None  # the default applies per tenant
    clock.value = _ts(2026, 10, 15, 0, 1)
    assert ledger.admit("acme") is not None


def test_request_limits_count_in_flight_requests(ledgers):
    ledger = ledgers(limits=_limits(today={"requests": 2}))
    first = ledger.admit("acme")
    ledger.admit("acme")
    with pytest.raises(UsageLimitExceeded):
        ledger.admit("acme")
    ledger.release(first)
    assert ledger.admit("acme") is not None


def test_reserve_mode_caps_output_and_holds_it_until_finish(ledgers):
    ledger = ledgers(limits=_limits("reserve", today={"completion_tokens": 100}))
    first = ledger.admit("acme", requested_output=60)
    second = ledger.admit("acme", requested_output=60)
    assert (first.output_cap, second.output_cap) == (60, 40)
    with pytest.raises(UsageLimitExceeded):
        ledger.admit("acme", requested_output=1)
    # Finishing below the cap returns the unused hold.
    ledger.record("acme", completion_tokens=10, reservation=first)
    assert ledger.admit("acme", requested_output=500).output_cap == 50
    # Embeddings produce no output and hold none.
    assert ledger.admit("acme", requested_output=0).output_cap is None


def test_report_mode_counts_but_never_refuses(ledgers):
    ledger = ledgers(limits=_limits("report", today={"requests": 0}))
    assert ledger.admit("acme") is not None
    assert ledger.counts["limit_reports"] == 1
    assert ledger.counts["limit_refusals"] == 0


def test_session_limits_apply_per_session(ledgers):
    ledger = ledgers(limits=_limits(session={"total_tokens": 10}))
    ledger.record("acme", "s1", prompt_tokens=10)
    with pytest.raises(UsageLimitExceeded) as caught:
        ledger.admit("acme", "s1")
    assert caught.value.scope == "session" and caught.value.retry_after is None
    assert ledger.admit("acme", "s2") is not None
    assert ledger.admit("acme") is not None


def test_tenant_entry_replaces_the_default(ledgers):
    limits = UsageLimits.from_document(
        {
            "version": 1,
            "default": {"today": {"requests": 0}},
            "tenants": {"vip": {"today": {"requests": 5}}},
        }
    )
    ledger = ledgers(limits=limits)
    with pytest.raises(UsageLimitExceeded):
        ledger.admit("anyone")
    assert ledger.admit("vip") is not None
    status = ledger.limit_status("vip")
    # The admission above is still in flight and holds one request.
    assert status["limits"][0]["remaining"] == 4


def test_tenant_prometheus_is_scoped_and_operator_sees_all(ledgers):
    ledger = ledgers(limits=_limits(today={"requests": 9}))
    ledger.record("acme", "sess-1", prompt_tokens=3)
    ledger.record("beta", "sess-2", prompt_tokens=4)
    scoped = ledger.prometheus("acme")
    assert 'tenant="acme"' in scoped and "beta" not in scoped
    assert 'session="sess-1"' in scoped and "sess-2" not in scoped
    assert 'mlx2_tenant_usage_window{counter="total_tokens",ledger="view",tenant="acme",window="today"} 3' in scoped
    assert "mlx2_tenant_usage_limit_remaining" in scoped
    operator = ledger.prometheus()
    assert 'tenant="beta"' in operator and 'ledger="accounting"' in operator
    assert "mlx2_session_usage_total" not in operator


# -- engine integration ---------------------------------------------------------


def _bare_engine(ledger):
    from mlx2 import serving

    engine = serving.ServingEngine.__new__(serving.ServingEngine)
    engine.lock = threading.Lock()
    engine.jobs = {}
    engine.slots = NS(release=lambda: None)
    engine.batch_metrics = NS(terminal=lambda job_id, status: None)
    engine.counts = {"cancelled": 0}
    engine._emit = lambda job, event: None
    engine.usage_ledger = ledger
    engine.default_max_tokens = 4096
    return engine


def test_engine_finish_records_once_and_releases_the_hold(ledgers):
    ledger = ledgers(limits=_limits("reserve", today={"total_tokens": 1000, "requests": 5}))
    engine = _bare_engine(ledger)
    (reservation,) = engine._usage_admit(
        [{"session_id": "s1", "max_tokens": 300}], "acme"
    )
    job = Job({"session_id": "s1"}, tenant_id="acme")
    engine._usage_attach(job, reservation)
    assert job.usage_output_cap == 300
    assert engine._usage_capped(job, 4096) == 300
    job.prompt_tokens, job.cached_tokens, job.completion_tokens = 20, 5, 7
    engine.jobs[job.id] = job
    engine._finish(job, {"finish_reason": "stop"})
    engine._finish(job, {"finish_reason": "stop"})  # duplicate terminal
    overall = ledger.windows("accounting", "acme")["overall"]
    assert (overall["requests"], overall["total_tokens"], overall["prefill_tokens"]) == (1, 27, 15)
    assert ledger.sessions("view", "acme")[0]["requests"] == 1
    assert reservation.released
    # The hold is gone: the next request may reserve all that remains.
    assert ledger.admit("acme").output_cap == 1000 - 27


def test_engine_cancelled_finish_is_counted_as_cancelled(ledgers):
    ledger = ledgers()
    engine = _bare_engine(ledger)
    job = Job({}, tenant_id="acme")
    job.completion_tokens = 3
    engine.jobs[job.id] = job
    engine._finish(job, {"error": "cancelled"})
    overall = ledger.windows("view", "acme")["overall"]
    assert overall["cancelled"] == 1 and overall["completion_tokens"] == 3


def test_engine_admission_is_all_or_none(ledgers):
    ledger = ledgers(limits=_limits(today={"requests": 2}))
    engine = _bare_engine(ledger)
    with pytest.raises(UsageLimitExceeded):
        engine._usage_admit([{}, {}, {}], "acme")
    # The two holds taken before the refusal were returned.
    assert len(engine._usage_admit([{}, {}], "acme")) == 2


def test_engine_without_ledger_is_untouched():
    engine = _bare_engine(None)
    assert engine._usage_admit([{}], "acme") == [None]
    job = Job({})
    engine._usage_attach(job, None)
    assert engine._usage_capped(job, 77) == 77


# -- HTTP -----------------------------------------------------------------------

KEY_A = "mlx2k_" + "a" * 43
KEY_A_ADMIN = "mlx2k_" + "e" * 43
KEY_B = "mlx2k_" + "b" * 43


def _authenticator():
    return TenantAuthenticator(
        keys={
            hash_api_key(KEY_A): (TenantPrincipal("tenant-a", "api_key", "a1"), False),
            hash_api_key(KEY_A_ADMIN): (
                TenantPrincipal(
                    "tenant-a", "api_key", "a-admin",
                    scopes=frozenset({"inference", "usage-admin"}),
                ),
                False,
            ),
            hash_api_key(KEY_B): (TenantPrincipal("tenant-b", "api_key", "b1"), False),
        }
    )


class UsageEngine:
    model_path = "fixture"

    def __init__(self):
        from collections import Counter

        self.counts = Counter()

    def status(self):
        return {"healthy": True, "error": None, "model": "fixture"}

    def batching_status(self):
        return {}

    def prometheus_metrics(self):
        return "# HELP mlx2_up up\n# TYPE mlx2_up gauge\nmlx2_up 1\n"

    def submit(self, request, *, tenant_id="default", **_kwargs):
        # The real engine's admission hook, reduced to its observable effect.
        self.usage_ledger.admit(tenant_id, request.get("session_id"))
        job = Job(request, tenant_id=tenant_id)
        job.prompt_tokens = job.completion_tokens = 1
        job.events.put({"delta": {"content": "ok"}})
        job.events.put({"finish_reason": "stop", "receipt": {}})
        return job


@pytest.fixture
def served(tmp_path):
    servers = []

    def start(ledger, authenticator=None, **kwargs):
        engine = UsageEngine()
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            handler_for(
                engine,
                response_store=ResponseStore(),
                file_store=FileStore(),
                tenant_authenticator=authenticator,
                usage_ledger=ledger,
                **kwargs,
            ),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        return f"http://127.0.0.1:{server.server_port}"

    yield start
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join()


def _call(base, method, path, body=None, key=None, headers=None):
    request_headers = dict(headers or {})
    if key:
        request_headers["Authorization"] = f"Bearer {key}"
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        request_headers["Content-Type"] = "application/json"
    try:
        with urlopen(Request(base + path, method=method, data=data,
                             headers=request_headers)) as response:
            return response.status, dict(response.headers), response.read()
    except HTTPError as error:
        return error.code, dict(error.headers), error.read()


def test_tenants_see_only_their_own_usage(served, ledgers):
    ledger = ledgers()
    ledger.record("tenant-a", "a-session", prompt_tokens=5)
    ledger.record("tenant-b", "b-session", prompt_tokens=7)
    base = served(ledger, _authenticator())
    status, _, raw = _call(base, "GET", "/v1/usage", key=KEY_A)
    report = json.loads(raw)
    assert status == 200 and report["tenant"] == "tenant-a"
    assert report["windows"]["today"]["total_tokens"] == 5
    assert [row["session_id"] for row in report["sessions"]] == ["a-session"]
    assert "tenant-b" not in raw.decode() and "b-session" not in raw.decode()
    # X-Tenant-ID cannot redirect the view.
    status, _, _ = _call(base, "GET", "/v1/usage", key=KEY_A,
                         headers={"X-Tenant-ID": "tenant-b"})
    assert status == 403
    status, _, raw = _call(base, "GET", "/v1/usage/metrics", key=KEY_B)
    assert status == 200 and "tenant-a" not in raw.decode()
    assert 'tenant="tenant-b"' in raw.decode()
    status, _, raw = _call(base, "GET", "/v1/usage?session=a-session", key=KEY_B)
    assert json.loads(raw)["sessions"] == []
    assert _call(base, "GET", "/v1/usage")[0] == 401


def test_clearing_needs_the_usage_admin_scope_and_is_view_only(served, ledgers):
    ledger = ledgers()
    ledger.record("tenant-a", prompt_tokens=5)
    ledger.record("tenant-b", prompt_tokens=7)
    base = served(ledger, _authenticator())
    assert _call(base, "POST", "/v1/usage/clear", key=KEY_A)[0] == 403
    status, _, _ = _call(base, "POST", "/v1/usage/clear", key=KEY_A_ADMIN)
    assert status == 200
    assert ledger.windows("view", "tenant-a")["overall"]["total_tokens"] == 0
    assert ledger.windows("accounting", "tenant-a")["overall"]["total_tokens"] == 5
    assert ledger.windows("view", "tenant-b")["overall"]["total_tokens"] == 7
    # A tenant admin cannot aim the action elsewhere.
    status, _, _ = _call(base, "POST", "/v1/usage/clear", {"tenant": "tenant-b"},
                         key=KEY_A_ADMIN)
    assert status == 400
    assert _call(base, "POST", "/v1/usage/pause", key=KEY_A_ADMIN)[0] == 200
    assert ledger.collection("tenant-a", "view")["paused"]
    assert not ledger.collection("tenant-a", "accounting")["paused"]


def test_tenant_views_refuse_unverified_tenants(served, ledgers):
    base = served(ledgers())
    status, _, raw = _call(base, "GET", "/v1/usage", headers={"X-Tenant-ID": "acme"})
    assert status == 403 and b"tenant authentication" in raw


def test_operator_admin_usage_and_metrics(served, ledgers):
    ledger = ledgers()
    ledger.record("tenant-a", prompt_tokens=5)
    ledger.record("tenant-b", prompt_tokens=7)
    base = served(ledger, _authenticator(), admin_token="admin-secret")
    status, _, raw = _call(base, "GET", "/v1/admin/usage", key="admin-secret")
    report = json.loads(raw)
    assert status == 200 and set(report["tenant_usage"]) == {"tenant-a", "tenant-b"}
    assert _call(base, "GET", "/v1/admin/usage")[0] == 401
    # Loopback /metrics without the admin bearer carries no tenant names.
    status, _, raw = _call(base, "GET", "/metrics")
    assert status == 200 and "tenant-a" not in raw.decode()
    status, _, raw = _call(base, "GET", "/metrics", key="admin-secret")
    assert 'tenant="tenant-a"' in raw.decode() and 'tenant="tenant-b"' in raw.decode()
    status, _, raw = _call(base, "POST", "/v1/admin/usage/clear",
                           {"tenant": "tenant-b", "ledger": "all"}, key="admin-secret")
    assert status == 200
    assert ledger.windows("accounting", "tenant-b")["overall"]["total_tokens"] == 0
    assert ledger.windows("accounting", "tenant-a")["overall"]["total_tokens"] == 5
    assert _call(base, "POST", "/v1/admin/usage/explode", {}, key="admin-secret")[0] == 404
    status, _, raw = _call(base, "POST", "/v1/admin/usage/reload-limits", key="admin-secret")
    assert status == 400 and b"no --usage-limits-file" in raw


def test_limit_refusal_is_a_429_with_reset_and_receipt(served, ledgers):
    ledger = ledgers(limits=_limits(today={"requests": 0}))
    base = served(ledger, _authenticator())
    chat = {"model": "fixture", "messages": [{"role": "user", "content": "hi"}]}
    status, headers, raw = _call(base, "POST", "/v1/chat/completions", chat, key=KEY_A)
    body = json.loads(raw)
    assert status == 429
    assert body["error"]["code"] == "insufficient_quota"
    assert body["mlx2"]["usage_limit"]["window"] == "today"
    assert int(headers["Retry-After"]) > 1


def test_disabled_usage_answers_404(served):
    base = served(None, _authenticator())
    assert _call(base, "GET", "/v1/usage", key=KEY_A)[0] == 404


def test_usage_admin_is_a_known_scope():
    from mlx2.tenant_auth import SCOPES

    assert "usage-admin" in SCOPES


# -- command line ---------------------------------------------------------------


def test_cli_operates_on_a_stopped_server_and_refuses_a_running_one(tmp_path, capsys):
    directory = tmp_path / "usage"
    ledger = UsageLedger(directory, writer=False)
    ledger.record("acme", prompt_tokens=5)
    with pytest.raises(SystemExit, match="use --url"):
        usage._main(["--usage-dir", str(directory), "show"])
    ledger.close()
    assert usage._main(["--usage-dir", str(directory), "show", "--tenant", "acme"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["tenant_usage"]["acme"]["accounting"]["windows"]["overall"]["total_tokens"] == 5
    assert usage._main(["--usage-dir", str(directory), "clear", "--tenant", "acme"]) == 0
    capsys.readouterr()
    reopened = UsageLedger(directory, writer=False)
    try:
        assert reopened.windows("accounting", "acme")["overall"]["total_tokens"] == 0
    finally:
        reopened.close()


def test_server_arguments_and_default_directory(tmp_path):
    from mlx2.server import build_parser

    parser = build_parser()
    args = parser.parse_args(["--model", "m", "--api-state-dir", str(tmp_path / "state")])
    ledger = usage.from_args(args)
    try:
        assert ledger.persistent and ledger.directory == (tmp_path / "state" / "usage").resolve()
        assert ledger.limits is None
    finally:
        ledger.close()
    assert usage.from_args(parser.parse_args(["--model", "m", "--no-usage-stats"])) is None
    memory = usage.from_args(parser.parse_args(["--model", "m"]))
    try:
        assert not memory.persistent
    finally:
        memory.close()


# -- regressions from the independent review ------------------------------------


def test_admission_sees_a_request_that_finishes_while_the_book_loads(ledgers):
    ledger = ledgers(limits=_limits(today={"requests": 2}))
    ledger.record("acme")  # one recorded
    running = ledger.admit("acme")  # one in flight: the limit is now full
    original = ledger._book

    def book_then_finish(ledger_name, tenant):
        book = original(ledger_name, tenant)
        # The in-flight request finishes between the load and the check.
        ledger.record("acme", reservation=running)
        return book

    ledger._book = book_then_finish
    with pytest.raises(UsageLimitExceeded):
        ledger.admit("acme")


def test_reload_into_reserve_is_refused_while_requests_run(ledgers, tmp_path):
    path = tmp_path / "limits.json"
    path.write_text(json.dumps({"version": 1, "mode": "admission"}))
    os.chmod(path, 0o600)
    ledger = ledgers(limits=UsageLimits.load(path), limits_path=path)
    # A tenant with no limits still holds a reservation while limits are on.
    running = ledger.admit("acme")
    assert running is not None and ledger.in_flight() == 1
    path.write_text(json.dumps(
        {"version": 1, "mode": "reserve", "default": {"today": {"completion_tokens": 10}}}
    ))
    with pytest.raises(usage.UsageReloadBusy):
        ledger.reload_limits()
    assert ledger.limits.mode == "admission"
    ledger.record("acme", reservation=running)
    assert ledger.reload_limits()["mode"] == "reserve"


def test_finish_releases_the_hold_when_cleanup_raises(ledgers):
    ledger = ledgers(limits=_limits(today={"requests": 1}))
    engine = _bare_engine(ledger)
    (reservation,) = engine._usage_admit([{}], "acme")
    job = Job({}, tenant_id="acme")
    engine._usage_attach(job, reservation)

    def broken_close():
        raise RuntimeError("branch close failed")

    job.cache_branch = NS(close=broken_close)
    engine.jobs[job.id] = job
    with pytest.raises(RuntimeError, match="branch close"):
        engine._finish(job, {"finish_reason": "stop"})
    assert reservation.released
    assert ledger.admit("acme") is not None


def test_post_acceptance_failures_are_failed_requests(ledgers):
    from mlx2.serving import Overloaded

    ledger = ledgers(limits=_limits(today={"requests": 9}))
    engine = _bare_engine(ledger)
    held = engine._usage_admit([{"session_id": "s"}], "acme")
    engine._usage_failed("acme", [{"session_id": "s"}], held, ValueError("bad media"))
    overall = ledger.windows("accounting", "acme")["overall"]
    assert (overall["requests"], overall["failed"]) == (1, 1)
    assert held[0].released
    refused = engine._usage_admit([{}], "acme")
    engine._usage_failed("acme", [{}], refused, Overloaded("full"))
    assert ledger.windows("accounting", "acme")["overall"]["requests"] == 1
    assert refused[0].released
    # A registered job is left to _finish.
    job = Job({}, tenant_id="acme")
    engine.jobs[job.id] = job
    registered = engine._usage_admit([{}], "acme")
    engine._usage_failed("acme", [{}], registered, ValueError("x"), [job])
    assert not registered[0].released


def test_pooled_failures_are_recorded_as_failed(ledgers):
    ledger = ledgers()

    def explode():
        raise RuntimeError("adapter failed")

    with pytest.raises(RuntimeError):
        usage.metered_pooled(ledger, "acme", explode)
    assert ledger.windows("view", "acme")["overall"]["failed"] == 1
    usage.metered_pooled(ledger, "acme", lambda: {"usage": {"prompt_tokens": 12}})
    assert ledger.windows("view", "acme")["overall"]["embedding_tokens"] == 12


def test_rerank_payload_reports_its_input_tokens():
    from mlx2.server import rerank_payload

    class Engine:
        def status(self):
            return {"model": "fixture"}

        def rerank(self, query, documents, *, with_usage=False):
            scores = [0.5 for _ in documents]
            return (scores, 17) if with_usage else scores

    payload = rerank_payload(Engine(), {"query": "q", "documents": ["a", "b"]})
    assert payload["usage"] == {"prompt_tokens": 17, "total_tokens": 17}


def test_limit_status_counts_in_flight_holds(ledgers):
    ledger = ledgers(limits=_limits("reserve", today={"completion_tokens": 100}))
    ledger.admit("acme", requested_output=100)
    (row,) = ledger.limit_status("acme")["limits"]
    assert (row["recorded"], row["in_flight"], row["remaining"]) == (0, 100, 0)


@pytest.mark.parametrize("mode", [0o755, 0o750])
def test_shared_usage_directories_are_refused(tmp_path, mode):
    directory = tmp_path / "usage"
    directory.mkdir()
    os.chmod(directory, mode)
    with pytest.raises(ValueError, match="no group or world"):
        UsageLedger(directory, writer=False)


def test_a_readable_existing_database_is_refused(tmp_path):
    directory = tmp_path / "usage"
    UsageLedger(directory, writer=False).close()
    os.chmod(directory / usage.DB_NAME, 0o644)
    with pytest.raises(ValueError, match="no group or world"):
        UsageLedger(directory, writer=False)


def test_non_resetting_limits_send_no_retry_after(served, ledgers):
    ledger = ledgers(limits=_limits(overall={"requests": 0}))
    base = served(ledger, _authenticator())
    chat = {"model": "fixture", "messages": [{"role": "user", "content": "hi"}]}
    status, headers, raw = _call(base, "POST", "/v1/chat/completions", chat, key=KEY_A)
    assert status == 429 and "Retry-After" not in headers
    assert json.loads(raw)["mlx2"]["usage_limit"]["retry_after_seconds"] is None
