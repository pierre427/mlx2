"""Per-tenant and per-session token usage, with opt-in limits.

Every terminal generation (and every embeddings/rerank call) is recorded
against the tenant that made it.  Tenant totals are kept per local calendar
day in a small SQLite ledger, so they survive engine restarts and can be read
back as today / week / month / year / overall.  Session totals (keyed by the
request ``session_id``) are kept in memory only, bounded per tenant.

Two ledgers are written side by side:

* ``accounting`` is what limits are enforced against.  Only the operator
  (loopback admin API, or the CLI against a stopped server) can clear or
  pause it.
* ``view`` is what tenants see.  A tenant admin (a credential with the
  ``usage-admin`` scope) can clear or pause their own tenant's view; that
  never lifts a limit, because limits read ``accounting``.

The recording path never touches the database: :meth:`UsageLedger.record`
updates in-memory books and per-key deltas under one lock, and a writer
thread adds the deltas to SQLite every ``flush_seconds``.  A crash loses at
most that interval.

Limits are disabled unless a limits file is configured.  Tenant names never
reach a Prometheus label on a scrape that is not either the operator's or
scoped to that tenant's own verified credential.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import json
import logging
import os
import sqlite3
import stat
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("mlx2.usage")

SCHEMA_VERSION = 1
# Counter order is the column order of both ledger tables.
COUNTERS = (
    "requests",
    "completed",
    "cancelled",
    "failed",
    "prompt_tokens",
    "cached_tokens",
    "prefill_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "embedding_tokens",
    "total_tokens",
    "service_ms",
)
WINDOWS = ("today", "week", "month", "year", "overall")
LIMIT_WINDOWS = WINDOWS + ("session",)
LIMIT_COUNTERS = ("requests", "prompt_tokens", "completion_tokens", "total_tokens")
# Counters a request's output can grow, and so the ones ``reserve`` caps.
OUTPUT_COUNTERS = ("completion_tokens", "total_tokens")
LEDGERS = ("accounting", "view")
LIMIT_MODES = ("report", "admission", "reserve")
STATUSES = ("completed", "cancelled", "failed")
# Not a valid tenant id (tenant ids start alphanumeric), so it cannot collide.
OVERFLOW_TENANT = "_overflow"
ALL_TENANTS = "*"
DEFAULT_MAX_TENANTS = 1024
DEFAULT_MAX_SESSIONS = 256
DEFAULT_RETENTION_DAYS = 800
MIN_RETENTION_DAYS = 372  # a full year plus the week that straddles Jan 1
DEFAULT_FLUSH_SECONDS = 2.0
DB_NAME = "usage.sqlite3"
LOCK_NAME = "usage.lock"


class UsageLimitExceeded(Exception):
    """A request would exceed a configured usage limit.

    Deliberately not :class:`~mlx2.serving.Overloaded`: the batch runner
    retries ``Overloaded`` rows for minutes, and a spent quota does not free
    up in minutes.
    """

    status = 429
    code = "insufficient_quota"

    def __init__(self, *, scope, window, counter, limit, used, retry_after):
        self.scope = scope
        self.window = window
        self.counter = counter
        self.limit = int(limit)
        self.used = int(used)
        # Seconds until the window resets; None for windows that never do.
        self.retry_after = retry_after
        where = "session" if scope == "session" else "tenant"
        super().__init__(
            f"{where} usage limit reached: {counter} {self.used}/{self.limit} "
            f"for window {window}"
        )

    def receipt(self):
        return {
            "scope": self.scope,
            "window": self.window,
            "counter": self.counter,
            "limit": self.limit,
            "used": self.used,
            "retry_after_seconds": self.retry_after,
        }


class UsageReloadBusy(RuntimeError):
    """A limits reload that is unsafe while requests are in flight."""


def _zero():
    return dict.fromkeys(COUNTERS, 0)


def _add(target, delta):
    for name, value in delta.items():
        if value:
            target[name] = target.get(name, 0) + value


# -- calendar ----------------------------------------------------------------


class UsageCalendar:
    """Local calendar windows; ``timezone=None`` follows the host zone."""

    def __init__(self, timezone=None):
        self.name = timezone
        if timezone is None:
            self._zone = None
        else:
            from zoneinfo import ZoneInfo

            self._zone = ZoneInfo(str(timezone))

    @property
    def label(self):
        if self.name is not None:
            return self.name
        return time.strftime("%Z") or "local"

    def date(self, ts):
        if self._zone is None:
            return _dt.date.fromtimestamp(ts)
        return _dt.datetime.fromtimestamp(ts, self._zone).date()

    def _midnight(self, day):
        if self._zone is None:
            # A naive datetime's timestamp() is read in the host zone, DST
            # included.
            return _dt.datetime.combine(day, _dt.time.min).timestamp()
        return _dt.datetime.combine(day, _dt.time.min, tzinfo=self._zone).timestamp()

    def starts(self, ts):
        """``{window: first local date}`` for the calendar windows at ``ts``."""
        today = self.date(ts)
        return {
            "today": today,
            "week": today - _dt.timedelta(days=today.weekday()),
            "month": today.replace(day=1),
            "year": today.replace(month=1, day=1),
        }

    def resets(self, ts):
        """``{window: seconds until it next resets}`` at ``ts``."""
        starts = self.starts(ts)
        today = starts["today"]
        month = starts["month"]
        following = {
            "today": today + _dt.timedelta(days=1),
            "week": starts["week"] + _dt.timedelta(days=7),
            "month": (month.replace(year=month.year + 1, month=1)
                      if month.month == 12 else month.replace(month=month.month + 1)),
            "year": starts["year"].replace(year=today.year + 1),
        }
        return {
            window: max(1, int(self._midnight(day) - ts + 0.999))
            for window, day in following.items()
        }

    def horizon(self, ts):
        """Earliest local date any calendar window at ``ts`` needs."""
        starts = self.starts(ts)
        return min(starts["week"], starts["year"])


# -- limits ------------------------------------------------------------------


def _limit_table(value, where):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object of windows")
    table = {}
    for window, counters in value.items():
        if window not in LIMIT_WINDOWS:
            raise ValueError(f"{where} has unknown window {window!r}")
        if not isinstance(counters, dict) or not counters:
            raise ValueError(f"{where}.{window} must be a nonempty object of counters")
        row = {}
        for counter, limit in counters.items():
            if counter not in LIMIT_COUNTERS:
                raise ValueError(f"{where}.{window} has unknown counter {counter!r}")
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                raise ValueError(f"{where}.{window}.{counter} must be a nonnegative integer")
            row[counter] = limit
        table[window] = row
    return table


@dataclass(frozen=True)
class UsageLimits:
    """Validated limits file contents.  A tenant entry replaces ``default``."""

    mode: str = "admission"
    default: dict = field(default_factory=dict)
    tenants: dict = field(default_factory=dict)

    def for_tenant(self, tenant):
        return self.tenants.get(tenant, self.default)

    @classmethod
    def from_document(cls, document):
        if not isinstance(document, dict):
            raise ValueError("usage limits file must be a JSON object")
        unknown = set(document) - {"version", "mode", "default", "tenants"}
        if unknown:
            raise ValueError(f"usage limits file has unknown fields: {', '.join(sorted(unknown))}")
        if document.get("version") != 1:
            raise ValueError("usage limits file version must be 1")
        mode = document.get("mode", "admission")
        if mode not in LIMIT_MODES:
            raise ValueError(f"usage limits mode must be one of {LIMIT_MODES}")
        tenants = document.get("tenants", {})
        if not isinstance(tenants, dict):
            raise ValueError("usage limits tenants must be an object")
        from .tenant_auth import validate_tenant

        return cls(
            mode=mode,
            default=_limit_table(document.get("default"), "default"),
            tenants={
                validate_tenant(tenant): _limit_table(table, f"tenants.{tenant}")
                for tenant, table in tenants.items()
            },
        )

    @classmethod
    def load(cls, path):
        candidate = Path(path).expanduser().resolve()
        metadata = os.stat(candidate)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("--usage-limits-file must be a regular file")
        if stat.S_IMODE(metadata.st_mode) & 0o022:
            raise ValueError("--usage-limits-file must not be group- or world-writable")
        return cls.from_document(json.loads(candidate.read_text(encoding="utf-8")))


@dataclass
class UsageReservation:
    """Admission's hold on a tenant's (and session's) limits until finish."""

    keys: tuple
    output: int
    output_cap: int | None
    released: bool = False


# -- ledger ------------------------------------------------------------------


class _Book:
    __slots__ = ("days", "total", "since", "loaded")

    def __init__(self):
        self.days = {}
        self.total = _zero()
        self.since = None
        self.loaded = False


class _Session:
    __slots__ = ("books", "first_seen", "last_seen")

    def __init__(self, now):
        self.books = {ledger: _zero() for ledger in LEDGERS}
        self.first_seen = now
        self.last_seen = now


def _connect(path):
    if path is None:
        connection = sqlite3.connect(":memory:", check_same_thread=False)
    else:
        connection = sqlite3.connect(str(path), check_same_thread=False, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
    columns = ", ".join(f"{name} INTEGER NOT NULL DEFAULT 0" for name in COUNTERS)
    connection.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS usage_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS usage_day (
            ledger TEXT NOT NULL, tenant TEXT NOT NULL, day TEXT NOT NULL, {columns},
            PRIMARY KEY (ledger, tenant, day));
        CREATE TABLE IF NOT EXISTS usage_total (
            ledger TEXT NOT NULL, tenant TEXT NOT NULL, since REAL NOT NULL, {columns},
            PRIMARY KEY (ledger, tenant));
        CREATE TABLE IF NOT EXISTS usage_control (
            ledger TEXT NOT NULL, tenant TEXT NOT NULL,
            paused INTEGER NOT NULL DEFAULT 0, cleared_at REAL,
            PRIMARY KEY (ledger, tenant));
        """
    )
    row = connection.execute(
        "SELECT value FROM usage_meta WHERE key='schema_version'"
    ).fetchone()
    if row is None:
        connection.execute(
            "INSERT INTO usage_meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
        )
    elif row[0] != str(SCHEMA_VERSION):
        connection.close()
        raise ValueError(f"usage ledger schema {row[0]} is not {SCHEMA_VERSION}")
    connection.commit()
    return connection


def _check_private(path, *, kind):
    """A process-owned ``kind`` with no group or world access, not a symlink."""
    metadata = os.lstat(path)
    expected = stat.S_ISDIR if kind == "directory" else stat.S_ISREG
    if not expected(metadata.st_mode):
        raise ValueError(f"usage {kind} {path} must be a real {kind}, not a link")
    if metadata.st_uid != os.getuid():
        raise ValueError(f"usage {kind} {path} must be owned by the process uid")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ValueError(
            f"usage {kind} {path} must have no group or world permissions "
            f"(it holds tenant names); chmod {'700' if kind == 'directory' else '600'} it"
        )


def _prepare_directory(directory):
    directory = Path(directory).expanduser().absolute()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    # The directory guards the SQLite WAL/SHM side files too, which SQLite
    # creates with the process umask.
    _check_private(directory, kind="directory")
    database = directory / DB_NAME
    try:
        os.close(os.open(database, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    except FileExistsError:
        pass
    for path in (database, directory / LOCK_NAME):
        if path.exists() or path.is_symlink():
            _check_private(path, kind="file")
    return directory, database


def acquire_directory_lock(directory, *, blocking=False):
    """Exclusive lock on a usage directory; None when another holder has it."""
    directory, _database = _prepare_directory(directory)
    handle = os.open(
        directory / LOCK_NAME, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
    except BlockingIOError:
        os.close(handle)
        return None
    return handle


class UsageLedger:
    """Thread-safe usage books with a background SQLite writer."""

    def __init__(
        self,
        directory=None,
        *,
        timezone=None,
        limits=None,
        limits_path=None,
        max_tenants=DEFAULT_MAX_TENANTS,
        max_sessions=DEFAULT_MAX_SESSIONS,
        retention_days=DEFAULT_RETENTION_DAYS,
        flush_seconds=DEFAULT_FLUSH_SECONDS,
        known_tenants=(),
        clock=time.time,
        writer=True,
    ):
        if int(max_tenants) < 1 or int(max_sessions) < 1:
            raise ValueError("usage tenant and session bounds must be positive")
        if int(retention_days) < MIN_RETENTION_DAYS:
            raise ValueError(f"usage retention must be at least {MIN_RETENTION_DAYS} days")
        if not float(flush_seconds) > 0:
            raise ValueError("usage flush interval must be positive")
        self.calendar = UsageCalendar(timezone)
        self.limits = limits
        self.limits_path = limits_path
        self.max_tenants = int(max_tenants)
        self.max_sessions = int(max_sessions)
        self.retention_days = int(retention_days)
        self.flush_seconds = float(flush_seconds)
        self.known_tenants = frozenset(known_tenants)
        self._clock = clock
        self._lock_handle = None
        self.directory = None
        database = None
        if directory is not None:
            self._lock_handle = acquire_directory_lock(directory)
            if self._lock_handle is None:
                raise ValueError(
                    f"usage directory {directory} is in use by another process"
                )
            self.directory, database = _prepare_directory(directory)
        self.persistent = database is not None
        try:
            self._db = _connect(database)
        except BaseException:
            self._release_directory()
            raise
        # Lock order: _db_lock before _lock.  record() takes only _lock.
        self._db_lock = threading.Lock()
        self._lock = threading.Lock()
        self._books = {}
        self._pending_days = {}
        self._pending_totals = {}
        self._sessions = {}
        self._inflight = {}
        self._overflow_logged = False
        self.counts = {
            "flushes": 0,
            "flush_errors": 0,
            "records": 0,
            "overflow_records": 0,
            "limit_refusals": 0,
            "limit_reports": 0,
            "sessions_evicted": 0,
        }
        self._refusals = {}
        self._tenants = {
            tenant for (tenant,) in self._db.execute(
                "SELECT DISTINCT tenant FROM usage_total"
            )
        }
        self._paused = set()
        self._cleared_at = {}
        for ledger, tenant, paused, cleared_at in self._db.execute(
            "SELECT ledger, tenant, paused, cleared_at FROM usage_control"
        ):
            if paused:
                self._paused.add((ledger, tenant))
            if cleared_at is not None:
                self._cleared_at[(ledger, tenant)] = cleared_at
        self._pruned_on = None
        self._stop = threading.Event()
        self._writer = None
        if writer:
            self._writer = threading.Thread(
                target=self._write_loop, name="mlx2-usage-writer", daemon=True
            )
            self._writer.start()

    # -- lifecycle --------------------------------------------------------

    def _release_directory(self):
        if self._lock_handle is not None:
            try:
                fcntl.flock(self._lock_handle, fcntl.LOCK_UN)
            finally:
                os.close(self._lock_handle)
                self._lock_handle = None

    def close(self):
        self._stop.set()
        if self._writer is not None:
            self._writer.join(timeout=10)
        try:
            self.flush()
        finally:
            with self._db_lock:
                self._db.close()
            self._release_directory()

    def _write_loop(self):
        while not self._stop.wait(self.flush_seconds):
            try:
                self.flush()
                self._prune()
            except Exception:  # the writer must outlive one bad flush
                log.exception("usage ledger flush failed")

    def flush(self):
        """Add pending deltas to SQLite; safe to call from any thread."""
        with self._db_lock:
            with self._lock:
                days, self._pending_days = self._pending_days, {}
                totals, self._pending_totals = self._pending_totals, {}
            if not days and not totals:
                return
            names = ", ".join(COUNTERS)
            marks = ", ".join("?" for _ in COUNTERS)
            updates = ", ".join(f"{name} = {name} + excluded.{name}" for name in COUNTERS)
            try:
                with self._db:
                    self._db.executemany(
                        f"INSERT INTO usage_day (ledger, tenant, day, {names}) "
                        f"VALUES (?, ?, ?, {marks}) "
                        f"ON CONFLICT (ledger, tenant, day) DO UPDATE SET {updates}",
                        [
                            (ledger, tenant, day, *(delta.get(n, 0) for n in COUNTERS))
                            for (ledger, tenant, day), delta in days.items()
                        ],
                    )
                    self._db.executemany(
                        f"INSERT INTO usage_total (ledger, tenant, since, {names}) "
                        f"VALUES (?, ?, ?, {marks}) "
                        f"ON CONFLICT (ledger, tenant) DO UPDATE SET {updates}, "
                        "since = MIN(since, excluded.since)",
                        [
                            (ledger, tenant, since, *(delta.get(n, 0) for n in COUNTERS))
                            for (ledger, tenant), (since, delta) in totals.items()
                        ],
                    )
            except sqlite3.Error:
                # Put the deltas back so the next flush retries them.
                with self._lock:
                    for key, delta in days.items():
                        _add(self._pending_days.setdefault(key, {}), delta)
                    for key, (since, delta) in totals.items():
                        held = self._pending_totals.get(key)
                        if held is None:
                            self._pending_totals[key] = (since, dict(delta))
                        else:
                            _add(held[1], delta)
                            self._pending_totals[key] = (min(held[0], since), held[1])
                    self.counts["flush_errors"] += 1
                raise
            with self._lock:
                self.counts["flushes"] += 1

    def _prune(self):
        today = self.calendar.date(self._clock())
        if self._pruned_on == today:
            return
        cutoff = (today - _dt.timedelta(days=self.retention_days)).isoformat()
        with self._db_lock:
            with self._db:
                self._db.execute("DELETE FROM usage_day WHERE day < ?", (cutoff,))
        horizon = self.calendar.horizon(self._clock()).isoformat()
        with self._lock:
            for book in self._books.values():
                for day in [day for day in book.days if day < horizon]:
                    del book.days[day]
        self._pruned_on = today

    # -- recording --------------------------------------------------------

    def _resolve_tenant(self, tenant):
        """The book a tenant records into; holds ``_lock``."""
        tenant = str(tenant or "default")
        if tenant in self._tenants or tenant in self.known_tenants:
            self._tenants.add(tenant)
            return tenant
        if len(self._tenants) < self.max_tenants:
            self._tenants.add(tenant)
            return tenant
        if not self._overflow_logged:
            log.warning(
                "usage ledger tracks at most %d tenants; further tenants are "
                "recorded as %s", self.max_tenants, OVERFLOW_TENANT,
            )
            self._overflow_logged = True
        self.counts["overflow_records"] += 1
        self._tenants.add(OVERFLOW_TENANT)
        return OVERFLOW_TENANT

    def is_paused(self, ledger, tenant):
        return (ledger, tenant) in self._paused or (ledger, ALL_TENANTS) in self._paused

    def _session(self, tenant, session_id, now):
        """The live session record, evicting the least recent; holds ``_lock``."""
        sessions = self._sessions.setdefault(tenant, OrderedDict())
        current = sessions.get(session_id)
        if current is None:
            current = sessions[session_id] = _Session(now)
            while len(sessions) > self.max_sessions:
                sessions.popitem(last=False)
                self.counts["sessions_evicted"] += 1
        else:
            sessions.move_to_end(session_id)
        current.last_seen = now
        return current

    def record(
        self,
        tenant,
        session_id=None,
        *,
        status="completed",
        prompt_tokens=0,
        cached_tokens=0,
        completion_tokens=0,
        reasoning_tokens=0,
        embedding_tokens=0,
        service_ms=0,
        reservation=None,
    ):
        """Account one terminal request.  Never blocks on the database."""
        if status not in STATUSES:
            status = "failed"
        prompt_tokens = max(0, int(prompt_tokens or 0))
        cached_tokens = min(prompt_tokens, max(0, int(cached_tokens or 0)))
        completion_tokens = max(0, int(completion_tokens or 0))
        embedding_tokens = max(0, int(embedding_tokens or 0))
        delta = {
            "requests": 1,
            status: 1,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "prefill_tokens": prompt_tokens - cached_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": max(0, int(reasoning_tokens or 0)),
            "embedding_tokens": embedding_tokens,
            "total_tokens": prompt_tokens + completion_tokens + embedding_tokens,
            "service_ms": max(0, int(service_ms or 0)),
        }
        now = self._clock()
        day = self.calendar.date(now).isoformat()
        with self._lock:
            self._release(reservation)
            tenant = self._resolve_tenant(tenant)
            self.counts["records"] += 1
            session = (
                self._session(tenant, session_id, now) if session_id else None
            )
            for ledger in LEDGERS:
                if self.is_paused(ledger, tenant):
                    continue
                book = self._books.get((ledger, tenant))
                if book is not None and book.loaded:
                    _add(book.days.setdefault(day, _zero()), delta)
                    _add(book.total, delta)
                    if book.since is None:
                        book.since = now
                _add(self._pending_days.setdefault((ledger, tenant, day), {}), delta)
                held = self._pending_totals.get((ledger, tenant))
                if held is None:
                    self._pending_totals[(ledger, tenant)] = (now, dict(delta))
                else:
                    _add(held[1], delta)
                if session is not None:
                    _add(session.books[ledger], delta)

    # -- reading ----------------------------------------------------------

    def _book(self, ledger, tenant):
        """A loaded book; takes ``_db_lock`` then ``_lock``.  Not under _lock."""
        with self._lock:
            book = self._books.get((ledger, tenant))
            if book is not None and book.loaded:
                return book
        with self._db_lock:
            horizon = self.calendar.horizon(self._clock()).isoformat()
            names = ", ".join(COUNTERS)
            days = {
                row[0]: dict(zip(COUNTERS, row[1:]))
                for row in self._db.execute(
                    f"SELECT day, {names} FROM usage_day "
                    "WHERE ledger = ? AND tenant = ? AND day >= ?",
                    (ledger, tenant, horizon),
                )
            }
            row = self._db.execute(
                f"SELECT since, {names} FROM usage_total WHERE ledger = ? AND tenant = ?",
                (ledger, tenant),
            ).fetchone()
            with self._lock:
                book = self._books.get((ledger, tenant))
                if book is not None and book.loaded:
                    return book
                book = self._books[(ledger, tenant)] = _Book()
                book.days = days
                if row is not None:
                    book.since = row[0]
                    book.total = dict(zip(COUNTERS, row[1:]))
                # Deltas not yet flushed belong to the book too.  Holding
                # _db_lock keeps a flush from moving them meanwhile.
                for (pledger, ptenant, day), delta in self._pending_days.items():
                    if pledger == ledger and ptenant == tenant and day >= horizon:
                        _add(book.days.setdefault(day, _zero()), delta)
                held = self._pending_totals.get((ledger, tenant))
                if held is not None:
                    _add(book.total, held[1])
                    book.since = held[0] if book.since is None else min(book.since, held[0])
                book.loaded = True
                return book

    @staticmethod
    def _sum(book, first, counters):
        """Sum ``counters`` over a book's days from ``first``; holds ``_lock``."""
        total = dict.fromkeys(counters, 0)
        for day, values in book.days.items():
            if day >= first:
                for name in counters:
                    total[name] += values.get(name, 0)
        return total

    def windows(self, ledger, tenant, now=None, *, counters=COUNTERS, only=None):
        """``{window: {"since": date, counters...}}`` for one tenant book."""
        now = self._clock() if now is None else now
        book = self._book(ledger, tenant)
        starts = self.calendar.starts(now)
        result = {}
        with self._lock:
            for window, start in starts.items():
                if only is None or window in only:
                    first = start.isoformat()
                    result[window] = {"since": first, **self._sum(book, first, counters)}
            if only is None or "overall" in only:
                result["overall"] = {
                    "since": (
                        self.calendar.date(book.since).isoformat()
                        if book.since is not None else None
                    ),
                    **{name: book.total.get(name, 0) for name in counters},
                }
        return result

    def sessions(self, ledger, tenant, session_id=None):
        with self._lock:
            sessions = self._sessions.get(str(tenant), {})
            items = (
                [(session_id, sessions[session_id])]
                if session_id is not None and session_id in sessions
                else [] if session_id is not None
                else reversed(list(sessions.items()))
            )
            return [
                {
                    "session_id": key,
                    "first_seen": int(value.first_seen),
                    "last_seen": int(value.last_seen),
                    **value.books[ledger],
                }
                for key, value in items
            ]

    def tenants(self):
        with self._lock:
            return sorted(self._tenants)

    def _live_sessions(self, tenant):
        with self._lock:
            return len(self._sessions.get(tenant, ()))

    # -- limits -----------------------------------------------------------

    def _book_tenant(self, tenant):
        """The book ``tenant`` records into, without registering it."""
        with self._lock:
            if (
                tenant in self._tenants
                or tenant in self.known_tenants
                or len(self._tenants) < self.max_tenants
            ):
                return tenant
            return OVERFLOW_TENANT

    def _limit_rows(self, limits, table, book, tenant_key, session_key, session,
                    now, *, requested=None):
        """``(row, ...)`` of recorded-plus-held usage per limit; holds ``_lock``.

        Admission and the reports read the same numbers, so a report never
        shows headroom that admission would refuse.
        """
        starts = self.calendar.starts(now)
        resets = self.calendar.resets(now)
        sums = {}
        rows = []
        for window, counters in table.items():
            if window == "session":
                if session_key is None:
                    continue
                source = session.books["accounting"] if session else _zero()
                hold = self._inflight.get(session_key, (0, 0))
                scope = "session"
            else:
                if window not in sums:
                    sums[window] = (
                        book.total if window == "overall"
                        else self._sum(book, starts[window].isoformat(), LIMIT_COUNTERS)
                    )
                source = sums[window]
                hold = self._inflight.get(tenant_key, (0, 0))
                scope = "tenant"
            for counter, limit in counters.items():
                if counter == "completion_tokens" and requested == 0:
                    continue  # embeddings/rerank produce no completion
                recorded = source.get(counter, 0)
                held = (
                    hold[0] if counter == "requests"
                    else hold[1] if counter in OUTPUT_COUNTERS else 0
                )
                rows.append(
                    {
                        "scope": scope,
                        "window": window,
                        "counter": counter,
                        "limit": limit,
                        "recorded": recorded,
                        "in_flight": held,
                        "used": recorded + held,
                        "remaining": max(0, limit - recorded - held),
                        "resets_in_seconds": resets.get(window),
                    }
                )
        return rows

    def admit(self, tenant, session_id=None, *, requested_output=None):
        """Check limits before a request runs; returns a reservation or None.

        Raises :class:`UsageLimitExceeded` when a limit is already reached
        (``admission``/``reserve``), and in ``reserve`` mode caps the request's
        output at what remains.  ``report`` mode refuses nothing.  With limits
        enabled every admitted request holds a reservation (so a live reload
        knows what is in flight), even for a tenant with no limits.
        """
        limits = self.limits
        if limits is None:
            return None
        tenant = self._book_tenant(str(tenant or "default"))
        table = limits.for_tenant(tenant)
        # None: output unknown (hold all that remains); 0: produces no output.
        requested = None if requested_output is None else max(0, int(requested_output))
        # Load first (it may read SQLite); then read usage, check holds and
        # take this hold under one _lock, so a request finishing in between
        # cannot be counted in neither place.
        book = self._book("accounting", tenant)
        tenant_key = ("tenant", tenant)
        session_key = ("session", tenant, session_id) if session_id else None
        now = self._clock()
        with self._lock:
            session = (
                self._sessions.get(tenant, {}).get(session_id) if session_id else None
            )
            rows = self._limit_rows(
                limits, table, book, tenant_key, session_key, session, now,
                requested=requested,
            )
            remaining_output = None
            for row in rows:
                if row["used"] >= row["limit"]:
                    if limits.mode == "report":
                        self.counts["limit_reports"] += 1
                        continue
                    self.counts["limit_refusals"] += 1
                    key = (tenant, row["window"], row["counter"])
                    self._refusals[key] = self._refusals.get(key, 0) + 1
                    raise UsageLimitExceeded(
                        scope=row["scope"],
                        window=row["window"],
                        counter=row["counter"],
                        limit=row["limit"],
                        used=row["used"],
                        retry_after=row["resets_in_seconds"],
                    )
                if row["counter"] in OUTPUT_COUNTERS:
                    left = row["remaining"]
                    remaining_output = (
                        left if remaining_output is None else min(remaining_output, left)
                    )
            output = 0
            cap = None
            if limits.mode == "reserve" and remaining_output is not None and requested != 0:
                cap = remaining_output if requested is None else min(remaining_output, requested)
                output = cap
            keys = tuple(key for key in (tenant_key, session_key) if key is not None)
            for key in keys:
                count, held = self._inflight.get(key, (0, 0))
                self._inflight[key] = (count + 1, held + output)
            return UsageReservation(keys=keys, output=output, output_cap=cap)

    def _release(self, reservation):
        """Drop an admission hold; idempotent; holds ``_lock``."""
        if reservation is None or reservation.released:
            return
        reservation.released = True
        for key in reservation.keys:
            count, held = self._inflight.get(key, (0, 0))
            count, held = count - 1, held - reservation.output
            if count <= 0:
                self._inflight.pop(key, None)
            else:
                self._inflight[key] = (count, max(0, held))

    def release(self, reservation):
        """Release a hold for a request that never ran."""
        with self._lock:
            self._release(reservation)

    def limit_status(self, tenant, session_id=None):
        """Configured limits for ``tenant``: accounting usage, in-flight holds
        and headroom, computed exactly as admission computes them."""
        limits = self.limits
        if limits is None:
            return {"enabled": False}
        tenant = str(tenant)
        table = limits.for_tenant(tenant)
        book = self._book("accounting", tenant) if table else None
        now = self._clock()
        with self._lock:
            session = (
                self._sessions.get(tenant, {}).get(session_id) if session_id else None
            )
            rows = (
                self._limit_rows(
                    limits, table, book, ("tenant", tenant),
                    ("session", tenant, session_id) if session_id else None,
                    session, now,
                )
                if table else []
            )
        return {"enabled": True, "mode": limits.mode, "limits": rows}

    def in_flight(self):
        with self._lock:
            return sum(count for key, (count, _held) in self._inflight.items()
                       if key[0] == "tenant")

    def reload_limits(self):
        """Re-read the limits file.

        Switching into ``reserve`` while requests are in flight is refused:
        they were admitted without output holds, so a new request could
        reserve budget they will also spend.
        """
        if self.limits_path is None:
            raise ValueError("no --usage-limits-file is configured")
        limits = UsageLimits.load(self.limits_path)
        with self._lock:
            current = self.limits
            busy = sum(count for key, (count, _held) in self._inflight.items()
                       if key[0] == "tenant")
            if (
                limits.mode == "reserve"
                and (current is None or current.mode != "reserve")
                and busy
            ):
                raise UsageReloadBusy(
                    f"cannot switch usage limits to reserve with {busy} request(s) "
                    "in flight; retry when idle"
                )
            self.limits = limits
        return {"mode": limits.mode, "tenants": len(limits.tenants)}

    # -- control ----------------------------------------------------------

    def _ledgers(self, ledger):
        if ledger == "all":
            return LEDGERS
        if ledger not in LEDGERS:
            raise ValueError(f"ledger must be one of {LEDGERS + ('all',)}")
        return (ledger,)

    def _control(self, ledger, tenant, **fields):
        """Upsert one control row; holds ``_db_lock``."""
        current = self._db.execute(
            "SELECT paused, cleared_at FROM usage_control WHERE ledger = ? AND tenant = ?",
            (ledger, tenant),
        ).fetchone()
        paused, cleared_at = current if current is not None else (0, None)
        paused = int(fields.get("paused", paused))
        cleared_at = fields.get("cleared_at", cleared_at)
        self._db.execute(
            "INSERT INTO usage_control VALUES (?, ?, ?, ?) "
            "ON CONFLICT (ledger, tenant) DO UPDATE SET paused = excluded.paused, "
            "cleared_at = excluded.cleared_at",
            (ledger, tenant, paused, cleared_at),
        )

    def clear(self, tenant=None, *, ledger="view"):
        """Delete one tenant's (or every tenant's) books in ``ledger``."""
        ledgers = self._ledgers(ledger)
        now = self._clock()
        target = ALL_TENANTS if tenant is None else str(tenant)
        with self._db_lock:
            with self._lock:
                for name in ledgers:
                    for key in [key for key in self._books if key[0] == name
                                and (tenant is None or key[1] == target)]:
                        del self._books[key]
                    for key in [key for key in self._pending_days if key[0] == name
                                and (tenant is None or key[1] == target)]:
                        del self._pending_days[key]
                    for key in [key for key in self._pending_totals if key[0] == name
                                and (tenant is None or key[1] == target)]:
                        del self._pending_totals[key]
                    for owner, sessions in self._sessions.items():
                        if tenant is None or owner == target:
                            for session in sessions.values():
                                session.books[name] = _zero()
                    self._cleared_at[(name, target)] = now
                if tenant is None and ledgers == LEDGERS:
                    self._sessions.clear()
            with self._db:
                for name in ledgers:
                    for table in ("usage_day", "usage_total"):
                        if tenant is None:
                            self._db.execute(f"DELETE FROM {table} WHERE ledger = ?", (name,))
                        else:
                            self._db.execute(
                                f"DELETE FROM {table} WHERE ledger = ? AND tenant = ?",
                                (name, target),
                            )
                    self._control(name, target, cleared_at=now)
            if tenant is None and ledgers == LEDGERS:
                with self._lock:
                    self._tenants = set()
        return {"cleared": list(ledgers), "tenant": tenant, "cleared_at": int(now)}

    def set_paused(self, paused, tenant=None, *, ledger="view"):
        ledgers = self._ledgers(ledger)
        target = ALL_TENANTS if tenant is None else str(tenant)
        with self._db_lock:
            with self._lock:
                for name in ledgers:
                    if paused:
                        self._paused.add((name, target))
                    else:
                        self._paused.discard((name, target))
            with self._db:
                for name in ledgers:
                    self._control(name, target, paused=1 if paused else 0)
        return {"paused": bool(paused), "ledgers": list(ledgers), "tenant": tenant}

    def collection(self, tenant, ledger):
        with self._lock:
            cleared = [
                self._cleared_at.get((ledger, key)) for key in (tenant, ALL_TENANTS)
            ]
            cleared = [value for value in cleared if value is not None]
            return {
                "paused": self.is_paused(ledger, tenant),
                "cleared_at": int(max(cleared)) if cleared else None,
            }

    # -- reports ----------------------------------------------------------

    def tenant_report(self, tenant, *, sessions=True, session_id=None):
        """What a tenant may see about itself: its view ledger and its limits."""
        tenant = str(tenant)
        report = {
            "object": "usage",
            "tenant": tenant,
            "timezone": self.calendar.label,
            "ledger": "view",
            "collection": self.collection(tenant, "view"),
            "windows": self.windows("view", tenant),
            "limits": self.limit_status(tenant, session_id),
        }
        if sessions or session_id is not None:
            report["sessions"] = self.sessions("view", tenant, session_id)
            report["sessions_tracked_max"] = self.max_sessions
        return report

    def status(self):
        with self._lock:
            counts = dict(self.counts)
            tenants = len(self._tenants)
        return {
            "enabled": True,
            "persistent": self.persistent,
            "directory": str(self.directory) if self.directory else None,
            "timezone": self.calendar.label,
            "tenants": tenants,
            "max_tenants": self.max_tenants,
            "max_sessions_per_tenant": self.max_sessions,
            "retention_days": self.retention_days,
            "limits": (
                {"enabled": True, "mode": self.limits.mode,
                 "tenant_overrides": len(self.limits.tenants)}
                if self.limits is not None else {"enabled": False}
            ),
            "counts": counts,
        }

    def public_status(self):
        """``/v1/status.usage``: every tenant reads it, so no tenant counts."""
        return {
            "enabled": True,
            "persistent": self.persistent,
            "timezone": self.calendar.label,
            "limits": (
                {"enabled": True, "mode": self.limits.mode}
                if self.limits is not None else {"enabled": False}
            ),
        }

    def operator_report(self, tenant=None):
        tenants = [str(tenant)] if tenant is not None else self.tenants()
        return {
            "object": "usage.admin",
            **self.status(),
            "tenant_usage": {
                name: {
                    ledger: {
                        "collection": self.collection(name, ledger),
                        "windows": self.windows(ledger, name),
                    }
                    for ledger in LEDGERS
                }
                | {
                    "limits": self.limit_status(name),
                    "sessions_live": self._live_sessions(name),
                }
                for name in tenants
            },
        }

    def prometheus(self, tenant=None):
        """Usage families.  ``tenant=None`` is the operator scrape (every
        tenant, both ledgers, no sessions); a tenant gets only its own view
        ledger, its sessions, and its limits."""
        from .prometheus import PrometheusBuilder

        builder = PrometheusBuilder()
        scoped = tenant is not None
        tenants = [str(tenant)] if scoped else self.tenants()
        ledgers = ("view",) if scoped else LEDGERS
        for name in tenants:
            for ledger in ledgers:
                windows = self.windows(ledger, name)
                for counter in COUNTERS:
                    labels = {"tenant": name, "ledger": ledger, "counter": counter}
                    builder.counter(
                        "mlx2_tenant_usage_total",
                        "Lifetime tenant usage since the ledger was last cleared.",
                        windows["overall"][counter],
                        labels,
                    )
                    for window in WINDOWS[:-1]:
                        builder.gauge(
                            "mlx2_tenant_usage_window",
                            "Tenant usage in the current local calendar window.",
                            windows[window][counter],
                            {**labels, "window": window},
                        )
            for row in self.limit_status(name).get("limits", ()):
                labels = {"tenant": name, "window": row["window"], "counter": row["counter"]}
                builder.gauge(
                    "mlx2_tenant_usage_limit", "Configured usage limit.", row["limit"], labels
                )
                builder.gauge(
                    "mlx2_tenant_usage_limit_remaining",
                    "Usage left before the limit refuses requests.",
                    row["remaining"],
                    labels,
                )
            if scoped:
                for session in self.sessions("view", name):
                    for counter in ("requests", "prompt_tokens", "cached_tokens",
                                    "completion_tokens", "total_tokens"):
                        builder.counter(
                            "mlx2_session_usage_total",
                            "Usage of a live session (in memory; bounded per tenant).",
                            session[counter],
                            {"tenant": name, "session": session["session_id"],
                             "counter": counter},
                        )
        with self._lock:
            refusals = {key: value for key, value in self._refusals.items()
                        if not scoped or key[0] == str(tenant)}
            counts = dict(self.counts)
        for (name, window, counter), value in sorted(refusals.items()):
            builder.counter(
                "mlx2_tenant_usage_limit_refusals_total",
                "Requests refused by a usage limit.",
                value,
                {"tenant": name, "window": window, "counter": counter},
            )
        if not scoped:
            builder.gauge(
                "mlx2_usage_ledger_persistent",
                "Whether tenant usage survives a restart.",
                1 if self.persistent else 0,
            )
            builder.counter(
                "mlx2_usage_ledger_flush_errors_total",
                "Usage ledger writes that failed and were retried.",
                counts["flush_errors"],
            )
            builder.gauge(
                "mlx2_usage_ledger_tenants", "Tenants tracked by the usage ledger.",
                len(tenants),
            )
        return builder.render()


def metered_pooled(ledger, tenant, produce):
    """Run an embeddings/rerank call under ``tenant``'s limits and record it.

    The payload's ``usage.prompt_tokens`` is recorded as ``embedding_tokens``.
    A call that fails after admission is recorded as ``failed``, like a
    generation request that fails after the engine accepted it.
    """
    if ledger is None:
        return produce()
    reservation = ledger.admit(tenant, requested_output=0)
    try:
        payload = produce()
    except BaseException:
        ledger.record(tenant, status="failed", reservation=reservation)
        raise
    usage = (payload.get("usage") if isinstance(payload, dict) else None) or {}
    ledger.record(
        tenant,
        embedding_tokens=usage.get("prompt_tokens", usage.get("total_tokens", 0)),
        reservation=reservation,
    )
    return payload


def requested_output_tokens(request, default):
    """The output budget a request asks for, before context resolution."""
    for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        value = request.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return default


def from_args(args, *, known_tenants=()):
    """Build the server's ledger from CLI settings, or None when disabled."""
    if getattr(args, "no_usage_stats", False):
        if args.usage_limits_file:
            raise ValueError("--usage-limits-file needs usage stats enabled")
        return None
    directory = args.usage_dir
    if directory is None and getattr(args, "api_state_dir", None):
        directory = str(Path(args.api_state_dir).expanduser() / "usage")
    limits = UsageLimits.load(args.usage_limits_file) if args.usage_limits_file else None
    return UsageLedger(
        directory,
        timezone=args.usage_timezone,
        limits=limits,
        limits_path=args.usage_limits_file,
        max_tenants=args.usage_max_tenants,
        max_sessions=args.usage_max_sessions,
        retention_days=args.usage_retention_days,
        known_tenants=known_tenants,
    )


def add_server_arguments(parser):
    group = parser.add_argument_group(
        "usage accounting",
        "per-tenant and per-session token usage; tenant totals persist under "
        "--usage-dir (default <--api-state-dir>/usage, else memory only); "
        "limits stay off unless --usage-limits-file is given",
    )
    group.add_argument("--no-usage-stats", action="store_true",
                       help="do not record tenant or session usage")
    group.add_argument("--usage-dir", help="directory for the persistent usage ledger")
    group.add_argument("--usage-timezone", metavar="ZONE",
                       help="IANA zone for today/week/month/year (default: host zone)")
    group.add_argument("--usage-limits-file",
                       help="JSON usage limits (version 1); enforcement is off without it")
    group.add_argument("--usage-max-tenants", type=int, default=DEFAULT_MAX_TENANTS)
    group.add_argument("--usage-max-sessions", type=int, default=DEFAULT_MAX_SESSIONS,
                       help="live sessions tracked per tenant (least recent evicted)")
    group.add_argument("--usage-retention-days", type=int, default=DEFAULT_RETENTION_DAYS)


# -- command line --------------------------------------------------------------


def _http(url, method, path, *, token=None, body=None):
    import urllib.error
    import urllib.request

    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url.rstrip("/") + path, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise SystemExit(f"{method} {path} failed with HTTP {error.code}: {detail}") from None


def _read_secret(path=None, env=None):
    if path:
        return Path(path).expanduser().read_text(encoding="utf-8").strip()
    if env:
        value = os.environ.get(env, "").strip()
        if not value:
            raise SystemExit(f"environment variable {env} is empty")
        return value
    return None


def _main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m mlx2.usage",
        description="show, clear, pause or resume mlx2 usage stats",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--url", help="running server, e.g. http://127.0.0.1:8285")
    target.add_argument("--usage-dir", help="ledger directory of a STOPPED server")
    parser.add_argument("--admin-token-file", help="operator: server admin bearer token")
    tenant_key = parser.add_mutually_exclusive_group()
    tenant_key.add_argument("--api-key-file",
                            help="tenant: credential file (tenant endpoints; usage-admin to change)")
    tenant_key.add_argument("--api-key-env", help="tenant: env var with the credential")
    parser.add_argument("--timezone", help="with --usage-dir: zone for the windows")
    commands = parser.add_subparsers(dest="command", required=True)
    show = commands.add_parser("show")
    show.add_argument("--tenant")
    show.add_argument("--session")
    for name in ("clear", "pause", "resume"):
        command = commands.add_parser(name)
        command.add_argument("--tenant", help="operator: one tenant (default: all)")
        command.add_argument(
            "--ledger", choices=LEDGERS + ("all",), default=None,
            help="operator default: all for clear/pause/resume; tenants: view only",
        )
    commands.add_parser("reload-limits")
    args = parser.parse_args(argv)
    tenant_token = _read_secret(args.api_key_file, args.api_key_env)

    if args.url and tenant_token:
        if args.command == "show":
            query = f"?session={args.session}" if args.session else ""
            result = _http(args.url, "GET", f"/v1/usage{query}", token=tenant_token)
        elif args.command == "reload-limits":
            raise SystemExit("reload-limits is an operator command")
        else:
            if args.tenant or args.ledger not in (None, "view"):
                raise SystemExit("tenant credentials act on their own view ledger only")
            result = _http(args.url, "POST", f"/v1/usage/{args.command}",
                           token=tenant_token, body={})
    elif args.url:
        token = _read_secret(args.admin_token_file)
        if args.command == "show":
            query = f"?tenant={args.tenant}" if args.tenant else ""
            result = _http(args.url, "GET", f"/v1/admin/usage{query}", token=token)
        else:
            body = {}
            if args.command != "reload-limits":
                body = {"tenant": args.tenant, "ledger": args.ledger or "all"}
            result = _http(args.url, "POST", f"/v1/admin/usage/{args.command}",
                           token=token, body=body)
    else:
        if tenant_token:
            raise SystemExit("--usage-dir is operator-only; use --url with a tenant key")
        if args.command == "reload-limits":
            raise SystemExit("reload-limits needs a running server (--url)")
        try:
            ledger = UsageLedger(args.usage_dir, timezone=args.timezone, writer=False)
        except ValueError as error:
            raise SystemExit(f"{error}; while the server runs, use --url") from None
        try:
            if args.command == "show":
                # Sessions live in the server's memory; a stopped server has none.
                result = ledger.operator_report(args.tenant)
            elif args.command == "clear":
                result = ledger.clear(args.tenant, ledger=args.ledger or "all")
            else:
                result = ledger.set_paused(
                    args.command == "pause", args.tenant, ledger=args.ledger or "all"
                )
        finally:
            ledger.close()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
