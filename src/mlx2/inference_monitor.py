"""Read-only inference telemetry from running mlx2 servers for :mod:`mlx2.top`.

Every number here comes from a server's own exports: the Prometheus
exposition at ``/metrics``, the batch runtime snapshot at
``/v1/status/batching``, and the route receipts in ``/v1/status``.  The
collector only issues GET requests; it never submits work.

Rates are deltas between consecutive samples.  The "window" figures use the
oldest retained sample inside ``window_seconds``, so a quiet server reports
zero rather than a stale lifetime average.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

_SAMPLE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?\s+(\S+)")
_LABEL = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')
_SERVER_ARGS = re.compile(
    r"(?:-m\s+mlx2\.(?:server|image_mcp\.server)\b|/mlx2-(?:serve|image-mcp)(?:\s|$))"
)
_MODEL_ARG = re.compile(r"--model[=\s]+(\S+)")

# Counter families whose per-event rates make up the "hot paths" list.  Each
# is a bounded enum on the server side, so the list stays small.
MECHANISM_FAMILIES = (
    ("advanced_path_events_total", "event"),
    ("capability_engagements_total", "capability"),
    ("fused_gdn_events_total", "event"),
    ("indexed_qsa_attestations_total", None),
    ("indexed_qsa_fallbacks_total", None),
    ("int8_prefill_calls_total", "outcome"),
    ("lane_matmul_calls_total", "path"),
    ("moe_dispatches_total", "mode"),
    ("moe_router_calls_total", None),
    ("ple_compile_events_total", "event"),
    ("qsa_mtp_amendment_events_total", "event"),
    ("runtime_events_total", "component,event"),
    ("scheduler_events_total", "event"),
    ("segmented_mtp_events_total", "event"),
    ("verify_bitexact_dispatches_total", None),
)


class Metrics:
    """A parsed Prometheus text exposition, keyed by metric name."""

    def __init__(self, samples: dict[str, list[tuple[dict[str, str], float]]]):
        self.samples = samples

    @classmethod
    def parse(cls, text: str) -> Metrics:
        samples: dict[str, list[tuple[dict[str, str], float]]] = {}
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            match = _SAMPLE.match(line)
            if match is None:
                continue
            name, raw_labels, raw_value = match.groups()
            try:
                value = float(raw_value)
            except ValueError:
                continue
            labels = {
                key: value_.replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")
                for key, value_ in _LABEL.findall(raw_labels or "")
            }
            samples.setdefault(name, []).append((labels, value))
        return cls(samples)

    def series(self, name: str) -> list[tuple[dict[str, str], float]]:
        return self.samples.get("mlx2_" + name, [])

    def value(self, name: str, **labels: str) -> float | None:
        """Sum the series matching ``labels``; ``None`` when none exist."""
        matched = [
            value
            for series_labels, value in self.series(name)
            if all(series_labels.get(key) == wanted for key, wanted in labels.items())
        ]
        return sum(matched) if matched else None

    def labels(self, name: str) -> dict[str, str]:
        series = self.series(name)
        return series[0][0] if series else {}

    def buckets(self, name: str) -> list[tuple[float, float]]:
        result = []
        for labels, value in self.series(name + "_bucket"):
            try:
                result.append((float(labels.get("le", "nan")), value))
            except ValueError:
                continue
        return sorted(result)


def histogram_quantile(quantile: float, buckets: list[tuple[float, float]]) -> float | None:
    """Prometheus-style linear interpolation over cumulative buckets."""
    if not buckets or buckets[-1][1] <= 0:
        return None
    rank = quantile * buckets[-1][1]
    lower, below = 0.0, 0.0
    for upper, count in buckets:
        if count >= rank:
            if math.isinf(upper):
                return lower
            if count == below:
                return upper
            return lower + (upper - lower) * (rank - below) / (count - below)
        lower, below = upper, count
    return lower


def _bucket_delta(
    current: list[tuple[float, float]], before: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    previous = dict(before)
    return [(upper, max(0.0, count - previous.get(upper, 0.0))) for upper, count in current]


@dataclass(frozen=True)
class ServerEndpoint:
    url: str
    pid: int | None = None
    model_path: str | None = None
    discovered: bool = False


@dataclass(frozen=True)
class LatencyStat:
    """Seconds.  ``exact`` marks percentiles taken from raw samples rather
    than interpolated from power-of-two histogram buckets."""

    p50: float | None = None
    p95: float | None = None
    p99: float | None = None
    average: float | None = None
    count: int = 0
    exact: bool = False
    window_p50: float | None = None
    window_p95: float | None = None
    window_count: int = 0


@dataclass(frozen=True)
class MechanismRate:
    name: str
    per_second: float
    total: float


@dataclass(frozen=True)
class ActiveRequest:
    request_id: str
    phase: str
    age_seconds: float | None
    queue_ms: float | None = None
    ttft_ms: float | None = None
    mechanism: str | None = None


@dataclass(frozen=True)
class CompletedRequest:
    request_id: str
    status: str | None
    prompt_tokens: int | None
    cached_tokens: int | None
    completion_tokens: int | None
    ttft_seconds: float | None
    elapsed_seconds: float | None
    route: str | None
    thinking: bool | None
    draft_acceptance: float | None = None
    tokens_per_cycle: float | None = None

    @property
    def decode_tokens_per_second(self) -> float | None:
        if (
            self.completion_tokens is None
            or self.completion_tokens < 2
            or self.ttft_seconds is None
            or self.elapsed_seconds is None
            or self.elapsed_seconds <= self.ttft_seconds
        ):
            return None
        return (self.completion_tokens - 1) / (self.elapsed_seconds - self.ttft_seconds)


@dataclass(frozen=True)
class SpeculationReading:
    mode: str | None = None
    num_draft: int | None = None
    receipt_kind: str | None = None
    cycles: int = 0
    accepted_all: int = 0
    accepted_partial: int = 0
    accepted_zero: int = 0
    window_cycles: int = 0
    window_accepted_all: int = 0
    receipt_proposed: int = 0
    receipt_accepted: int = 0
    receipt_emitted: int = 0
    receipt_cycles: int = 0
    receipt_requests: int = 0
    copy_proposed: int = 0
    copy_accepted: int = 0
    progressive_tile: int | None = None
    progressive_rounds: int = 0
    progressive_launches: int = 0
    progressive_target_rows: int = 0
    progressive_padding_rows: int = 0
    progressive_ragged_launches: int = 0
    progressive_rejections: int = 0
    progressive_full_tiles: int = 0


@dataclass(frozen=True)
class ServerReading:
    endpoint: ServerEndpoint
    reachable: bool
    error: str | None = None
    status_error: str | None = None
    status_age_seconds: float | None = None
    # identity
    model: str | None = None
    profile: str | None = None
    qualification: str | None = None
    route: str | None = None
    state: str | None = None
    ready: bool | None = None
    uptime_seconds: float | None = None
    revision: str | None = None
    mlx_version: str | None = None
    max_lanes: int | None = None
    max_inflight: int | None = None
    max_context: int | None = None
    selected_capabilities: tuple[str, ...] = ()
    qualified_capabilities: tuple[str, ...] = ()
    # load
    running: int | None = None
    waiting: int | None = None
    active_lanes: int | None = None
    peak_active_lanes: int | None = None
    memory_waiting: int | None = None
    batch_width_window: float | None = None
    batch_width_lifetime: float | None = None
    batch_composition: tuple[tuple[int, int], ...] = ()
    scheduler_cycles_per_second: float | None = None
    # throughput
    generation_tps: float | None = None
    generation_tps_window: float | None = None
    prompt_tps: float | None = None
    prompt_tps_window: float | None = None
    cached_prompt_tps_window: float | None = None
    requests_per_second_window: float | None = None
    generation_tokens_total: float | None = None
    prompt_tokens_total: float | None = None
    cached_prompt_tokens_total: float | None = None
    outcomes: tuple[tuple[str, int], ...] = ()
    tenant_token_rates: tuple[tuple[str, float], ...] = ()
    jain_fairness: float | None = None
    # latency
    latency: tuple[tuple[str, LatencyStat], ...] = ()
    # speculation
    speculation: SpeculationReading = field(default_factory=SpeculationReading)
    # prefix cache
    cache_lookups: float | None = None
    cache_hits: float | None = None
    cache_window_lookups: float | None = None
    cache_window_hits: float | None = None
    cache_query_tokens: float | None = None
    cache_entries: float | None = None
    cache_resident_bytes: float | None = None
    cache_capacity_bytes: float | None = None
    cache_disk_entries: float | None = None
    cache_disk_bytes: float | None = None
    cache_interior_hits: float | None = None
    cache_rolling_hits: float | None = None
    cache_junction_hits: float | None = None
    cache_evictions_window: float | None = None
    host_prompt_cache: dict[str, Any] | None = None
    # memory
    metal_active_bytes: float | None = None
    metal_peak_bytes: float | None = None
    footprint_bytes: float | None = None
    headroom_bytes: float | None = None
    host_available_bytes: float | None = None
    host_pressure_level: float | None = None
    wired_limit_bytes: int | None = None
    weights_bytes: int | None = None
    # http and failures
    http_requests_per_second: float | None = None
    http_errors: tuple[tuple[str, int], ...] = ()
    failures: tuple[tuple[str, int], ...] = ()
    # mechanisms and requests
    mechanisms: tuple[MechanismRate, ...] = ()
    active_requests: tuple[ActiveRequest, ...] = ()
    recent_requests: tuple[CompletedRequest, ...] = ()


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dig(document: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(document, dict):
            return None
        document = document.get(key)
    return document


def discover_servers(timeout: float = 2.0) -> tuple[ServerEndpoint, ...]:
    """Find listening mlx2 inference and image control-plane processes."""
    try:
        listing = subprocess.run(
            ["/bin/ps", "-axo", "pid=,args="],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ()
    processes: dict[int, str] = {}
    for line in listing.splitlines():
        pid_text, _, args = line.strip().partition(" ")
        if _SERVER_ARGS.search(args):
            pid = _int(pid_text)
            if pid is not None:
                processes[pid] = args
    if not processes:
        return ()
    try:
        sockets = subprocess.run(
            [
                "/usr/sbin/lsof",
                "-nP",
                "-a",
                "-iTCP",
                "-sTCP:LISTEN",
                "-p",
                ",".join(str(pid) for pid in sorted(processes)),
                "-Fpn",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ()
    return parse_lsof_listeners(sockets, processes)


def parse_lsof_listeners(
    output: str, processes: dict[int, str]
) -> tuple[ServerEndpoint, ...]:
    """Turn ``lsof -Fpn`` listener records into endpoint URLs."""
    result: dict[str, ServerEndpoint] = {}
    pid: int | None = None
    for line in output.splitlines():
        if line.startswith("p"):
            pid = _int(line[1:])
        elif line.startswith("n") and pid is not None:
            host, _, port = line[1:].rpartition(":")
            if not port.isdigit():
                continue
            if host in {"*", "0.0.0.0", ""}:
                host = "127.0.0.1"
            elif host in {"[::]"}:
                host = "[::1]"
            url = f"http://{host}:{port}"
            model = _MODEL_ARG.search(processes.get(pid, ""))
            result.setdefault(
                url,
                ServerEndpoint(
                    url=url,
                    pid=pid,
                    model_path=model.group(1) if model else None,
                    discovered=True,
                ),
            )
    return tuple(sorted(result.values(), key=lambda endpoint: endpoint.url))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def local_server_api_key(endpoint: ServerEndpoint) -> str | None:
    """Resolve auth for a fresh, same-user loopback listener only.

    Credentials stay out of endpoint/readout dataclasses and are never cached.
    Explicit URLs without a discovered PID cannot opt into local credential reads.
    """
    from .http_security import load_secret_file

    parsed = urllib.parse.urlsplit(endpoint.url)
    if (
        not endpoint.discovered
        or endpoint.pid is None
        or parsed.scheme != "http"
        or not _is_loopback(endpoint.url)
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        return None
    if not any(
        fresh.url == endpoint.url and fresh.pid == endpoint.pid
        for fresh in discover_servers()
    ):
        return None
    process = psutil.Process(endpoint.pid)
    if process.uids().real != os.getuid():
        return None
    args = process.cmdline()
    for index, argument in enumerate(args):
        flag, _, inline = argument.partition("=")
        if flag not in {"--api-key-file", "--api-key-env"}:
            continue
        value = inline or (args[index + 1] if index + 1 < len(args) else "")
        if not value:
            return None
        if flag == "--api-key-env":
            return process.environ().get(value)
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path(process.cwd()) / path
        return load_secret_file(path, flag="--api-key-file")
    return None


def _get(url: str, path: str, api_key: str | None, timeout: float) -> bytes:
    headers = {"Accept": "application/json, text/plain"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url.rstrip("/") + path, headers=headers)
    # A redirect must never forward a discovered server's private credential.
    opener = urllib.request.build_opener(_NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        return response.read()


def _error_text(error: Exception) -> str:
    if isinstance(error, urllib.error.HTTPError):
        return f"HTTP {error.code}"
    if isinstance(error, urllib.error.URLError):
        return str(error.reason)
    return str(error) or type(error).__name__


def _is_loopback(url: str) -> bool:
    host = urllib.parse.urlsplit(url).hostname or ""
    return host in {"127.0.0.1", "localhost", "::1"}


@dataclass
class _ServerState:
    history: deque = field(default_factory=lambda: deque(maxlen=600))
    status: dict | None = None
    status_at: float | None = None
    status_error: str | None = None
    uptime: float | None = None


class InferenceCollector:
    """Polls every known mlx2 server and turns counters into rates."""

    def __init__(
        self,
        *,
        urls: tuple[str, ...] = (),
        fallback_urls: tuple[str, ...] = (),
        discover: bool = True,
        api_key: str | None = None,
        window_seconds: float = 30.0,
        status_every_seconds: float = 5.0,
        discover_every_seconds: float = 10.0,
        timeout_seconds: float = 1.5,
        recent_requests: int = 8,
    ):
        self.urls = tuple(dict.fromkeys(url.rstrip("/") for url in urls))
        self.fallback_urls = tuple(url.rstrip("/") for url in fallback_urls)
        self.discover = discover
        self.api_key = api_key
        self.window_seconds = window_seconds
        self.status_every_seconds = status_every_seconds
        self.discover_every_seconds = discover_every_seconds
        self.timeout_seconds = timeout_seconds
        self.recent_requests = recent_requests
        self._states: dict[str, _ServerState] = {}
        self._discovered: tuple[ServerEndpoint, ...] = ()
        self._discovered_at: float | None = None
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="mlx2-top-llm")

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def endpoints(self) -> tuple[ServerEndpoint, ...]:
        now = time.monotonic()
        if self.discover and (
            self._discovered_at is None
            or now - self._discovered_at >= self.discover_every_seconds
        ):
            self._discovered = discover_servers()
            self._discovered_at = now
        found = {endpoint.url: endpoint for endpoint in self._discovered}
        for url in self.urls:
            found.setdefault(url, ServerEndpoint(url=url))
        if not found:
            for url in self.fallback_urls:
                found.setdefault(url, ServerEndpoint(url=url))
        return tuple(found.values())

    def sample(self) -> tuple[ServerReading, ...]:
        endpoints = self.endpoints()
        live = {endpoint.url for endpoint in endpoints}
        for url in list(self._states):
            if url not in live:
                del self._states[url]
        futures = [self._pool.submit(self._sample_one, endpoint) for endpoint in endpoints]
        return tuple(future.result() for future in futures)

    def _sample_one(self, endpoint: ServerEndpoint) -> ServerReading:
        state = self._states.setdefault(endpoint.url, _ServerState())
        api_key = self.api_key
        try:
            try:
                raw = _get(endpoint.url, "/metrics", api_key, self.timeout_seconds)
            except urllib.error.HTTPError as error:
                if error.code != 401 or api_key is not None:
                    raise
                try:
                    api_key = local_server_api_key(endpoint)
                except (OSError, ValueError, psutil.Error):
                    # Never include credential paths, values or process environment
                    # in a public reading when local resolution is unavailable.
                    api_key = None
                if not api_key:
                    raise
                raw = _get(endpoint.url, "/metrics", api_key, self.timeout_seconds)
            metrics = Metrics.parse(
                raw.decode("utf-8", "replace")
            )
        except (OSError, ValueError) as error:
            state.history.clear()
            return ServerReading(endpoint=endpoint, reachable=False, error=_error_text(error))
        if metrics.value("build_info") is None:
            return ServerReading(
                endpoint=endpoint, reachable=False, error="not an mlx2 /metrics endpoint"
            )
        batching = None
        try:
            batching = json.loads(
                _get(endpoint.url, "/v1/status/batching", api_key, self.timeout_seconds)
            )
        except (OSError, ValueError):
            batching = None
        now = time.monotonic()
        if state.status_at is None or now - state.status_at >= self.status_every_seconds:
            try:
                state.status = json.loads(
                    _get(endpoint.url, "/v1/status", api_key, 2 * self.timeout_seconds)
                )
                state.status_error = None
            except (OSError, ValueError) as error:
                state.status_error = _error_text(error)
            state.status_at = now
        uptime = metrics.value("process_uptime_seconds")
        if uptime is not None and state.uptime is not None and uptime < state.uptime:
            state.history.clear()  # the server restarted; counters reset
        state.uptime = uptime
        state.history.append((now, metrics))
        while len(state.history) > 2 and now - state.history[1][0] > self.window_seconds:
            state.history.popleft()
        previous = state.history[-2] if len(state.history) > 1 else None
        window = state.history[0] if len(state.history) > 1 else None
        return build_reading(
            endpoint,
            metrics,
            now=now,
            previous=previous,
            window=window,
            batching=batching if isinstance(batching, dict) else None,
            status=state.status,
            status_error=state.status_error,
            status_age_seconds=None if state.status_at is None else now - state.status_at,
            recent_requests=self.recent_requests,
            clock_comparable=_is_loopback(endpoint.url),
        )


def _rate(
    name: str,
    metrics: Metrics,
    baseline: tuple[float, Metrics] | None,
    now: float,
    **labels: str,
) -> float | None:
    if baseline is None:
        return None
    seconds = now - baseline[0]
    current = metrics.value(name, **labels)
    before = baseline[1].value(name, **labels)
    if seconds <= 0 or current is None:
        return None
    return max(0.0, current - (before or 0.0)) / seconds


def _delta(
    name: str, metrics: Metrics, baseline: tuple[float, Metrics] | None, **labels: str
) -> float | None:
    if baseline is None:
        return None
    current = metrics.value(name, **labels)
    if current is None:
        return None
    return max(0.0, current - (baseline[1].value(name, **labels) or 0.0))


def _histogram_stat(
    name: str,
    metrics: Metrics,
    window: tuple[float, Metrics] | None,
    exact: dict | None = None,
) -> LatencyStat:
    buckets = metrics.buckets(name)
    count = int(metrics.value(name + "_count") or 0)
    total = metrics.value(name + "_sum")
    window_p50 = window_p95 = None
    window_count = 0
    if window is not None:
        delta = _bucket_delta(buckets, window[1].buckets(name))
        window_count = int(delta[-1][1]) if delta else 0
        if window_count:
            window_p50 = histogram_quantile(0.5, delta)
            window_p95 = histogram_quantile(0.95, delta)
    if exact and _int(exact.get("count")):
        return LatencyStat(
            p50=None if exact.get("p50") is None else exact["p50"] / 1000.0,
            p95=None if exact.get("p95") is None else exact["p95"] / 1000.0,
            p99=None if exact.get("p99") is None else exact["p99"] / 1000.0,
            average=None if not count or total is None else total / count,
            count=count or int(exact["count"]),
            exact=True,
            window_p50=window_p50,
            window_p95=window_p95,
            window_count=window_count,
        )
    return LatencyStat(
        p50=histogram_quantile(0.5, buckets),
        p95=histogram_quantile(0.95, buckets),
        p99=histogram_quantile(0.99, buckets),
        average=None if not count or total is None else total / count,
        count=count,
        window_p50=window_p50,
        window_p95=window_p95,
        window_count=window_count,
    )


def _mechanisms(
    metrics: Metrics, previous: tuple[float, Metrics] | None, now: float
) -> tuple[MechanismRate, ...]:
    result = []
    seconds = None if previous is None else now - previous[0]
    for family, keys in MECHANISM_FAMILIES:
        before = {} if previous is None else {
            tuple(sorted(labels.items())): value
            for labels, value in previous[1].series(family)
        }
        stem = family.removesuffix("_total")
        for labels, value in metrics.series(family):
            if value <= 0:
                continue
            parts = [labels.get(key, "") for key in (keys or "").split(",") if key]
            name = stem + (":" + "/".join(part for part in parts if part) if parts else "")
            rate = 0.0
            if seconds:
                rate = max(0.0, value - before.get(tuple(sorted(labels.items())), 0.0)) / seconds
            result.append(MechanismRate(name=name, per_second=rate, total=value))
    result.sort(key=lambda item: (-item.per_second, -item.total, item.name))
    return tuple(result)


def _active_requests(
    batching: dict | None, now: float, clock_comparable: bool
) -> tuple[ActiveRequest, ...]:
    if not batching:
        return ()
    active = [str(item) for item in batching.get("active_requests") or ()]
    if not active:
        return ()
    events: dict[str, dict[str, Any]] = {request_id: {} for request_id in active}
    mechanisms: dict[str, str] = {}
    for event in batching.get("events") or ():
        if not isinstance(event, dict):
            continue
        request_id = str(event.get("request_id"))
        if request_id in events:
            events[request_id][str(event.get("kind"))] = _float(event.get("at"))
            if event.get("mechanism"):
                mechanisms[request_id] = str(event["mechanism"])
    result = []
    for request_id in active:
        seen = events[request_id]
        admitted = seen.get("admitted")
        dequeued = seen.get("dequeued")
        first = seen.get("first_token")
        if first is not None:
            phase = "decode"
        elif seen.get("lane_attached") is not None:
            phase = "prefill"
        elif dequeued is not None:
            phase = "attach"
        else:
            phase = "queued"
        age = None
        if clock_comparable and admitted is not None and 0 <= now - admitted < 86_400:
            age = now - admitted
        result.append(
            ActiveRequest(
                request_id=request_id,
                phase=phase,
                age_seconds=age,
                queue_ms=None
                if admitted is None or dequeued is None
                else 1000.0 * (dequeued - admitted),
                ttft_ms=None
                if admitted is None or first is None
                else 1000.0 * (first - admitted),
                mechanism=mechanisms.get(request_id),
            )
        )
    return tuple(result)


def _recent_requests(
    status: dict | None, batching: dict | None, limit: int
) -> tuple[CompletedRequest, ...]:
    receipts = (status or {}).get("recent_receipts") or []
    outcomes = {
        str(item.get("request_id")): item.get("status")
        for item in (batching or {}).get("completed_requests") or ()
        if isinstance(item, dict)
    }
    result = []
    for receipt in reversed(receipts[-limit:] if limit else []):
        if not isinstance(receipt, dict):
            continue
        request_id = str(receipt.get("request_id") or "?")
        stats = _dig(receipt, "mtp", "stats")
        speculation = receipt.get("speculation")
        if isinstance(stats, dict):
            proposed = _int(stats.get("draft_proposed"))
            accepted = _int(stats.get("draft_accepted"))
            cycles = _int(stats.get("cycles"))
            emitted = _int(stats.get("total_emitted"))
        elif isinstance(speculation, dict):
            proposed = _int(speculation.get("proposed"))
            accepted = _int(speculation.get("accepted"))
            cycles = _int(speculation.get("external_rounds"))
            # External draft receipts intentionally do not infer emitted
            # tokens from accepts plus rounds: an early stop can suppress the
            # target bonus token.  Leave tokens/cycle unavailable until the
            # executor publishes an exact emitted count.
            emitted = _int(speculation.get("total_emitted"))
        else:
            proposed = accepted = cycles = emitted = None
        result.append(
            CompletedRequest(
                request_id=request_id,
                status=outcomes.get(request_id),
                prompt_tokens=_int(receipt.get("prompt_tokens")),
                cached_tokens=_int(receipt.get("cached_tokens")),
                completion_tokens=_int(receipt.get("completion_tokens")),
                ttft_seconds=_float(receipt.get("ttft_seconds")),
                elapsed_seconds=_float(receipt.get("elapsed_seconds")),
                route=(
                    _dig(receipt, "mtp", "route")
                    or receipt.get("route")
                    or (
                        speculation.get("kind")
                        if isinstance(speculation, dict)
                        else None
                    )
                ),
                thinking=_dig(receipt, "request_controls", "thinking"),
                draft_acceptance=None if not proposed else (accepted or 0) / proposed,
                tokens_per_cycle=None if not cycles or emitted is None else emitted / cycles,
            )
        )
    return tuple(result)


def _speculation(
    metrics: Metrics,
    window: tuple[float, Metrics] | None,
    status: dict | None,
) -> SpeculationReading:
    settings = (status or {}).get("settings") or {}
    num_draft = _int(
        _dig(settings, "adapter_policy", "num_draft")
        or _dig(settings, "execution_policy", "num_draft")
    )
    proposed = accepted = emitted = cycles = requests = 0
    receipt_kinds: set[str] = set()
    receipt_progressive: dict[str, Any] = {}
    emitted_complete = True
    for receipt in (status or {}).get("recent_receipts") or ():
        stats = _dig(receipt, "mtp", "stats")
        speculation = receipt.get("speculation") if isinstance(receipt, dict) else None
        if isinstance(stats, dict):
            requests += 1
            proposed += _int(stats.get("draft_proposed")) or 0
            accepted += _int(stats.get("draft_accepted")) or 0
            emitted += _int(stats.get("total_emitted")) or 0
            cycles += _int(stats.get("cycles")) or 0
        elif isinstance(speculation, dict):
            requests += 1
            proposed += _int(speculation.get("proposed")) or 0
            accepted += _int(speculation.get("accepted")) or 0
            exact_emitted = _int(speculation.get("total_emitted"))
            if exact_emitted is None:
                emitted_complete = False
            else:
                emitted += exact_emitted
            cycles += _int(speculation.get("external_rounds")) or 0
            receipt_kind = str(
                receipt.get("route") or speculation.get("kind") or ""
            )
            if receipt_kind:
                receipt_kinds.add(receipt_kind)
            progressive = speculation.get("progressive_verification")
            if isinstance(progressive, dict):
                receipt_progressive = progressive
    scheduler = (status or {}).get("scheduler") or {}
    receipt_kind = (
        next(iter(receipt_kinds))
        if len(receipt_kinds) == 1
        else "mixed"
        if receipt_kinds
        else None
    )

    def progressive_value(scheduler_key: str, receipt_key: str) -> int:
        value = _int(scheduler.get(scheduler_key))
        if value is None:
            value = _int(receipt_progressive.get(receipt_key))
        return value or 0

    def event(name: str) -> int:
        return int(metrics.value("segmented_mtp_events_total", event=name) or 0)

    def window_event(name: str) -> int:
        return int(_delta("segmented_mtp_events_total", metrics, window, event=name) or 0)

    window_cycles = sum(
        window_event(name) for name in ("accepted_all", "accepted_partial", "accepted_zero")
    )
    return SpeculationReading(
        mode=settings.get("speculation") or None,
        num_draft=num_draft,
        receipt_kind=receipt_kind,
        cycles=event("accepted_all") + event("accepted_partial") + event("accepted_zero"),
        accepted_all=event("accepted_all"),
        accepted_partial=event("accepted_partial"),
        accepted_zero=event("accepted_zero"),
        window_cycles=window_cycles,
        window_accepted_all=window_event("accepted_all"),
        receipt_proposed=proposed,
        receipt_accepted=accepted,
        receipt_emitted=emitted if emitted_complete else 0,
        receipt_cycles=cycles,
        receipt_requests=requests,
        copy_proposed=_int(scheduler.get("self_mtp_copy_proposed_tokens")) or 0,
        copy_accepted=_int(scheduler.get("self_mtp_copy_accepted_tokens")) or 0,
        progressive_tile=(
            _int(scheduler.get("external_progressive_verify_tile"))
            or _int(receipt_progressive.get("verification_tile"))
        ),
        progressive_rounds=progressive_value(
            "external_progressive_verify_rounds", "rounds"
        ),
        progressive_launches=progressive_value(
            "external_progressive_verify_launches", "target_launches"
        ),
        progressive_target_rows=progressive_value(
            "external_progressive_verify_target_rows", "target_rows"
        ),
        progressive_padding_rows=progressive_value(
            "external_progressive_verify_padding_rows", "padding_rows"
        ),
        progressive_ragged_launches=progressive_value(
            "external_progressive_verify_ragged_launches", "ragged_launches"
        ),
        progressive_rejections=progressive_value(
            "external_progressive_verify_rejections", "rejections"
        ),
        progressive_full_tiles=progressive_value(
            "external_progressive_verify_full_tiles", "full_nonfinal_tiles"
        ),
    )


def build_reading(
    endpoint: ServerEndpoint,
    metrics: Metrics,
    *,
    now: float,
    previous: tuple[float, Metrics] | None = None,
    window: tuple[float, Metrics] | None = None,
    batching: dict | None = None,
    status: dict | None = None,
    status_error: str | None = None,
    status_age_seconds: float | None = None,
    recent_requests: int = 8,
    clock_comparable: bool = True,
) -> ServerReading:
    """Derive one server reading from current and earlier exports."""
    status = status if isinstance(status, dict) else None
    settings = (status or {}).get("settings") or {}
    model_info = metrics.labels("model_info")
    build = metrics.labels("build_info")
    state = next(
        (labels.get("state") for labels, value in metrics.series("server_state") if value == 1),
        None,
    )
    latency_exact = (batching or {}).get("latency_ms") or {}
    latency = (
        ("TTFT", _histogram_stat("time_to_first_token_seconds", metrics, window, latency_exact.get("ttft"))),
        ("ITL", _histogram_stat("inter_token_latency_seconds", metrics, window, latency_exact.get("itl"))),
        ("Queue", _histogram_stat("request_queue_time_seconds", metrics, window, latency_exact.get("queue"))),
        ("Prefill", _histogram_stat("request_prefill_time_seconds", metrics, window)),
        ("Decode", _histogram_stat("request_decode_time_seconds", metrics, window)),
        ("TPOT", _histogram_stat("request_time_per_output_token_seconds", metrics, window)),
        ("E2E", _histogram_stat("e2e_request_latency_seconds", metrics, window)),
    )
    batch_count = _delta("batch_size_count", metrics, window)
    batch_sum = _delta("batch_size_sum", metrics, window)
    lifetime_count = metrics.value("batch_size_count")
    outcomes: dict[str, int] = {}
    for labels, value in metrics.series("requests_total"):
        outcome = labels.get("outcome", "?")
        reason = labels.get("finished_reason", "none")
        key = outcome if reason in {"none", ""} else f"{outcome}:{reason}"
        if value > 0:
            outcomes[key] = outcomes.get(key, 0) + int(value)
    http_errors: dict[str, int] = {}
    for labels, value in metrics.series("http_requests_total"):
        status_class = labels.get("status_class", "")
        if status_class in {"4xx", "5xx"} and value > 0:
            key = f"{labels.get('route', '?')} {status_class}"
            http_errors[key] = http_errors.get(key, 0) + int(value)
    failures: dict[str, int] = {}
    for family, keys in (
        ("fail_closed_total", ("capability", "reason")),
        ("admissions_rejected_total", ("endpoint_class",)),
        ("admission_decisions_total", ("decision", "reason")),
        ("telemetry_export_errors_total", ()),
        ("trace_export_errors_total", ()),
        ("suspend_failures_total", ()),
        ("drain_timeouts_total", ()),
    ):
        for labels, value in metrics.series(family):
            if value <= 0 or labels.get("decision") == "admitted":
                continue
            parts = [labels[key] for key in keys if labels.get(key)]
            name = family.removesuffix("_total") + (":" + "/".join(parts) if parts else "")
            failures[name] = failures.get(name, 0) + int(value)
    composition = []
    for width, count in ((batching or {}).get("batch_composition") or {}).items():
        if _int(width) is not None and _int(count):
            composition.append((int(width), int(count)))
    tenants = (batching or {}).get("fairness") or {}
    evictions = None
    if window is not None:
        evictions = _delta("prefix_cache_eviction_age_seconds_count", metrics, window)
    return ServerReading(
        endpoint=endpoint,
        reachable=True,
        status_error=status_error,
        status_age_seconds=status_age_seconds,
        model=(status or {}).get("model") or model_info.get("model_name"),
        profile=(status or {}).get("profile") or model_info.get("profile"),
        qualification=(status or {}).get("qualification") or model_info.get("qualification"),
        route=settings.get("route") or (status or {}).get("route_receipt"),
        state=state,
        ready=None if metrics.value("ready") is None else bool(metrics.value("ready")),
        uptime_seconds=metrics.value("process_uptime_seconds"),
        revision=(build.get("revision") or "")[:12] or None,
        mlx_version=_dig(status, "runtime", "mlx"),
        max_lanes=_int((status or {}).get("max_lanes") or settings.get("max_lanes")),
        max_inflight=_int(settings.get("max_inflight")),
        max_context=_int((status or {}).get("max_context") or settings.get("max_context")),
        selected_capabilities=tuple((status or {}).get("selected_capabilities") or ()),
        qualified_capabilities=tuple((status or {}).get("qualified_capabilities") or ()),
        running=_int(metrics.value("num_requests_running")),
        waiting=_int(metrics.value("num_requests_waiting")),
        active_lanes=_int(metrics.value("active_lanes")),
        peak_active_lanes=_int(metrics.value("peak_active_lanes")),
        memory_waiting=_int(metrics.value("memory_waiting_requests")),
        batch_width_window=None if not batch_count else (batch_sum or 0.0) / batch_count,
        batch_width_lifetime=None
        if not lifetime_count
        else (metrics.value("batch_size_sum") or 0.0) / lifetime_count,
        batch_composition=tuple(sorted(composition)),
        scheduler_cycles_per_second=_rate(
            "scheduler_cycles_total", metrics, window, now, condition="all"
        ),
        generation_tps=_rate("generation_tokens_total", metrics, previous, now),
        generation_tps_window=_rate("generation_tokens_total", metrics, window, now),
        prompt_tps=_rate("prompt_tokens_total", metrics, previous, now),
        prompt_tps_window=_rate("prompt_tokens_total", metrics, window, now),
        cached_prompt_tps_window=_rate("cached_prompt_tokens_total", metrics, window, now),
        requests_per_second_window=_rate("requests_total", metrics, window, now),
        generation_tokens_total=metrics.value("generation_tokens_total"),
        prompt_tokens_total=metrics.value("prompt_tokens_total"),
        cached_prompt_tokens_total=metrics.value("cached_prompt_tokens_total"),
        outcomes=tuple(sorted(outcomes.items(), key=lambda item: -item[1])),
        tenant_token_rates=tuple(
            sorted(
                (
                    (str(name), float(rate))
                    for name, rate in (tenants.get("tenant_token_rates") or {}).items()
                    if _float(rate) is not None
                ),
                key=lambda item: -item[1],
            )
        ),
        jain_fairness=metrics.value("jain_tenant_token_rate"),
        latency=latency,
        speculation=_speculation(metrics, window, status),
        cache_lookups=metrics.value("prefix_cache_lookups_total"),
        cache_hits=metrics.value("prefix_cache_hits_total"),
        cache_window_lookups=_delta("prefix_cache_lookups_total", metrics, window),
        cache_window_hits=_delta("prefix_cache_hits_total", metrics, window),
        cache_query_tokens=metrics.value("prefix_cache_query_tokens_total"),
        cache_entries=metrics.value("prefix_cache_entries"),
        cache_resident_bytes=metrics.value("prefix_cache_resident_bytes"),
        cache_capacity_bytes=metrics.value("prefix_cache_capacity_bytes"),
        cache_disk_entries=metrics.value("prefix_cache_disk_entries"),
        cache_disk_bytes=metrics.value("prefix_cache_disk_bytes"),
        cache_interior_hits=metrics.value("prefix_cache_interior_hits_total"),
        cache_rolling_hits=metrics.value("prefix_cache_rolling_hits_total"),
        cache_junction_hits=metrics.value("prefix_cache_junction_hits_total"),
        cache_evictions_window=evictions,
        host_prompt_cache=(status or {}).get("host_prompt_cache"),
        metal_active_bytes=metrics.value("memory_active_bytes"),
        metal_peak_bytes=metrics.value("memory_peak_bytes"),
        footprint_bytes=metrics.value("process_physical_footprint_bytes"),
        headroom_bytes=metrics.value("memory_headroom_bytes"),
        host_available_bytes=metrics.value("host_memory_available_bytes"),
        host_pressure_level=metrics.value("host_memory_pressure_level"),
        wired_limit_bytes=_int(_dig(status, "weight_residency", "wired_limit_bytes")),
        weights_bytes=_int(_dig(status, "weight_residency", "active_bytes_at_wire")),
        http_requests_per_second=_rate("http_requests_total", metrics, window, now),
        http_errors=tuple(sorted(http_errors.items(), key=lambda item: -item[1])),
        failures=tuple(sorted(failures.items(), key=lambda item: -item[1])),
        mechanisms=_mechanisms(metrics, previous, now),
        active_requests=_active_requests(batching, now, clock_comparable),
        recent_requests=_recent_requests(status, batching, recent_requests),
    )


__all__ = [
    "ActiveRequest",
    "CompletedRequest",
    "InferenceCollector",
    "LatencyStat",
    "MechanismRate",
    "Metrics",
    "ServerEndpoint",
    "ServerReading",
    "SpeculationReading",
    "build_reading",
    "discover_servers",
    "histogram_quantile",
    "parse_lsof_listeners",
]
