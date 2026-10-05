"""Explicit, default-off HTTP admission coordinator for a native TF worker.

One thread exclusively owns worker IPC. Other server threads submit token-ID
requests; new requests join between native rounds, never inside a cache
transaction. No worker cache is exposed to mlx2 or APCv2.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass(eq=False)
class _Request:
    prompt: list[int]
    budget: int
    route: str
    eos_ids: list[int] = field(default_factory=list)
    done: threading.Event = field(default_factory=threading.Event)
    session: str | None = None
    tokens: list[int] = field(default_factory=list)
    modes: dict[str, int] = field(default_factory=dict)
    active_width_counts: dict[int, int] = field(default_factory=dict)
    mode_transitions: list[dict] = field(default_factory=list)
    drafted: int = 0
    accepted: int = 0
    error: BaseException | None = None
    cancelled: bool = False
    finish_reason: str = ""
    observed_compute_width: int = 0
    queued_ns: int = 0
    admission_start_ns: int = 0
    prefill_start_ns: int = 0
    prefill_end_ns: int = 0
    first_token_ns: int | None = None
    final_token_ns: int | None = None
    cached_prompt_tokens: int = 0


def _timing_receipt(request):
    """Monotonic worker boundaries; no HTTP or admission time enters model rates."""
    prefill_ns = request.prefill_end_ns - request.prefill_start_ns
    if prefill_ns <= 0 or request.prefill_start_ns < request.admission_start_ns:
        raise RuntimeError("invalid native prefill timing boundaries")
    count = len(request.tokens)
    decode_ns = (request.final_token_ns - request.first_token_ns
                 if count > 1 and request.first_token_ns is not None
                 and request.final_token_ns is not None else None)
    return {
        "clock": "host_monotonic_ns", "prompt_tokens": len(request.prompt),
        "cached_prompt_tokens": request.cached_prompt_tokens,
        "queue_wait_seconds": (request.admission_start_ns - request.queued_ns) / 1e9,
        "prefill_to_first_token_seconds": prefill_ns / 1e9,
        "effective_prefill_tokens_per_second": (
            (len(request.prompt) - request.cached_prompt_tokens) * 1e9 / prefill_ns),
        "first_token_offset_seconds": (request.first_token_ns - request.queued_ns) / 1e9
        if request.first_token_ns is not None else None,
        "final_token_offset_seconds": (request.final_token_ns - request.queued_ns) / 1e9
        if request.final_token_ns is not None else None,
        "completion_tokens": count,
        "decode_seconds": decode_ns / 1e9 if decode_ns is not None and decode_ns > 0 else None,
        "decode_tokens_per_second": (count - 1) * 1e9 / decode_ns
        if decode_ns is not None and decode_ns > 0 else None,
        "decode_rate_defined": decode_ns is not None and decode_ns > 0,
        "rate_definition": "tokens_after_first / first_to_final_token_seconds",
    }


class TensorfoldOwnedLiveRouter:
    """Route one to eight greedy requests through one TF-owned lane engine."""

    def __init__(self, profile, *, max_streams=8, max_context=16384,
                 coalesce_ms=5, call_timeout=60):
        if not 1 <= max_streams <= 8 or not 0 <= coalesce_ms <= 50:
            raise ValueError("invalid TensorFold live admission bounds")
        if type(max_context) is not int or not 1 <= max_context <= 131072:
            raise ValueError("invalid TensorFold live context bound")
        self.profile = profile
        self.max_streams = max_streams
        self.max_context = max_context
        self.coalesce_seconds = coalesce_ms / 1000
        self.call_timeout = call_timeout
        self._condition = threading.Condition()
        self._pending: list[_Request] = []
        self._opening: list[_Request] = []
        self._active: dict[str, _Request] = {}
        self._closed = False
        self._started = False
        self._round_sequence = 0
        self._last_round = None
        self._thread = threading.Thread(target=self._drive, name="tensorfold-owned-router", daemon=True)
        self._thread.start()

    def status(self):
        with self._condition:
            return {"identity": self.profile.identity, "cache_policy": "disabled",
                    "qualified": False, "max_streams": self.max_streams,
                    "pending_requests": len(self._pending),
                    "opening_requests": len(self._opening),
                    "active_requests": len(self._active),
                    "last_round": dict(self._last_round) if self._last_round else None}

    def generate(self, token_ids, max_new_tokens, *, route="auto", eos_ids=None,
                 timeout=180, allow_long=False, disconnect_probe=None):
        if (not isinstance(token_ids, list) or not token_ids
                or any(type(token) is not int or not 0 <= token < 248320 for token in token_ids)):
            raise ValueError("TensorFold-owned route requires Qwen3.8 token_ids")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 192:
            raise ValueError("TensorFold-owned route supports max_new_tokens 1..192")
        if route not in {"auto", "serial"}:
            raise ValueError("TensorFold-owned route must be auto or serial")
        if (eos_ids is not None and (not isinstance(eos_ids, list)
                or any(type(token) is not int or not 0 <= token < 248320
                       for token in eos_ids))):
            raise ValueError("TensorFold-owned EOS IDs must be Qwen3.8 token IDs")
        if len(token_ids) + max_new_tokens > self.max_context:
            raise ValueError("TensorFold-owned request exceeds service context")
        limit = 900 if allow_long else 180
        if timeout <= 0 or timeout > limit:
            raise ValueError(f"TensorFold-owned request timeout must be in (0, {limit}]")
        request = _Request(list(token_ids), max_new_tokens, route, list(eos_ids or ()))
        request.queued_ns = time.monotonic_ns()
        with self._condition:
            if self._closed:
                raise RuntimeError("TensorFold-owned router is closed")
            if len(self._pending) >= 32:
                raise RuntimeError("TensorFold-owned admission queue is full")
            self._pending.append(request)
            self._condition.notify()
        if disconnect_probe is None:
            completed = request.done.wait(timeout)
        else:
            deadline = time.monotonic() + timeout
            while not request.done.is_set():
                if disconnect_probe():
                    with self._condition:
                        request.cancelled = True
                        self._condition.notify()
                    raise ConnectionAbortedError("TensorFold-owned client disconnected")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                request.done.wait(min(.05, remaining))
            completed = request.done.is_set()
        if not completed:
            with self._condition:
                request.cancelled = True
                self._condition.notify()
            raise TimeoutError("TensorFold-owned request timed out")
        if request.error is not None:
            raise RuntimeError("TensorFold-owned worker request failed") from request.error
        return {"token_ids": request.tokens, "finish_reason": request.finish_reason or "length",
                "route_receipt": {
            "route": "tensorfold_owned_live_v1", "identity": self.profile.identity,
            "cache_layout": self.profile.cache_layout, "cache_policy": "disabled",
            "qualified": False,
            "selected_by": "explicit_token_id_endpoint", "apcv2_lookup": False,
            "apcv2_store": False, "route_requested": request.route,
            "active_width_modes": dict(request.modes),
            "mode_transitions": list(request.mode_transitions),
            "active_width_counts": {str(width): count for width, count
                                    in sorted(request.active_width_counts.items())},
            "drafted": request.drafted, "accepted": request.accepted,
            "ordinary_compute_width": max(1, request.observed_compute_width),
            "timing": _timing_receipt(request),
        }}

    def _call(self, method, **params):
        return self.profile.call(method, timeout=self.call_timeout, **params)

    def _admit(self, request):
        request.admission_start_ns = time.monotonic_ns()
        opened = self._call("live_open", prompt=request.prompt,
                            max_new_tokens=request.budget, route=request.route,
                            eos_ids=request.eos_ids)
        if (opened.get("cache_layout") != self.profile.cache_layout
                or opened.get("cache_policy") != "disabled"
                or opened.get("qualified") is not False
                or opened.get("apcv2_lookup") is not False
                or opened.get("apcv2_store") is not False
                or opened.get("route_requested") != request.route):
            if isinstance(opened.get("session"), str):
                self._call("live_close", session=opened["session"])
            raise RuntimeError("native worker admission receipt violates cache isolation")
        request.session = opened["session"]
        request.prefill_start_ns = opened["prefill_start_ns"]
        request.prefill_end_ns = opened["prefill_end_ns"]
        request.cached_prompt_tokens = opened["cached_prompt_tokens"]
        if (opened["prompt_tokens"] != len(request.prompt)
                or not 0 <= request.cached_prompt_tokens <= len(request.prompt)):
            raise RuntimeError("native prompt timing count mismatch")
        request.tokens.extend(opened["tokens"])
        if request.tokens:
            request.first_token_ns = opened["first_token_ns"]
            request.final_token_ns = opened["first_token_ns"]
        request.finish_reason = opened.get("finish_reason", "")
        if opened["finished"]:
            self._call("live_close", session=request.session)
            request.done.set()
        else:
            with self._condition:
                self._active[request.session] = request

    def _finish(self, session, request, *, error=None):
        with self._condition:
            self._active.pop(session, None)
        try:
            self._call("live_close", session=session)
        except Exception as exc:
            error = error or exc
        request.error = error
        request.done.set()

    def _drive(self):
        try:
            while True:
                with self._condition:
                    while not self._closed and not self._pending and not self._active:
                        self._condition.wait()
                    if self._closed:
                        break
                    if self._pending and not self._active and self.coalesce_seconds:
                        self._condition.wait(timeout=self.coalesce_seconds)
                    entering = self._pending[:max(0, self.max_streams - len(self._active))]
                    del self._pending[:len(entering)]
                    self._opening.extend(entering)
                    cancelled = [sid for sid, req in self._active.items() if req.cancelled]
                if (entering or self._active) and not self._started:
                    ready = self.profile.start(timeout=self.call_timeout)
                    if (ready.get("identity") != self.profile.identity
                            or ready.get("cache_layout") != self.profile.cache_layout):
                        raise RuntimeError("native worker identity disagrees with admission profile")
                    self._started = True
                for sid in cancelled:
                    self._finish(sid, self._active[sid], error=TimeoutError("cancelled request"))
                for request in entering:
                    if request.cancelled:
                        request.error = TimeoutError("cancelled before admission")
                        request.done.set()
                        with self._condition:
                            self._opening.remove(request)
                        continue
                    self._admit(request)
                    with self._condition:
                        self._opening.remove(request)
                # A client may leave while native prefill is in progress.
                # Close its newly opened session before scheduling a round.
                with self._condition:
                    cancelled_after_open = [(sid, req) for sid, req in self._active.items()
                                            if req.cancelled]
                for sid, request in cancelled_after_open:
                    self._finish(sid, request, error=ConnectionAbortedError(
                        "TensorFold-owned client disconnected"))
                with self._condition:
                    if self._closed:
                        break
                if not self._active:
                    continue
                round_receipt = self._call("live_step")
                if (round_receipt.get("qualified") is not False
                        or round_receipt.get("cache_policy") != "disabled"
                        or round_receipt.get("apcv2_lookup") is not False
                        or round_receipt.get("apcv2_store") is not False):
                    raise RuntimeError("native round receipt violates cache isolation")
                mode = round_receipt["mode"]
                landed = round_receipt["tokens"]
                finished = round_receipt["finished"]
                finish_reasons = round_receipt.get("finish_reasons", {})
                per_stream = round_receipt["per_stream"]
                active_width = round_receipt["active_width"]
                if (set(landed) != set(self._active) or set(finished) != set(self._active)
                        or set(per_stream) != set(self._active)
                        or type(active_width) is not int
                        or active_width != len(self._active)):
                    raise RuntimeError("native round stream set differs from admitted requests")
                with self._condition:
                    self._round_sequence += 1
                    self._last_round = {"sequence": self._round_sequence,
                                        "mode": mode, "active_width": active_width}
                for sid, request in list(self._active.items()):
                    if (not request.mode_transitions or
                            request.mode_transitions[-1]["mode"] != mode or
                            request.mode_transitions[-1]["active_width"] != active_width):
                        request.mode_transitions.append({
                            "sequence": self._round_sequence, "mode": mode,
                            "active_width": active_width,
                            "completion_tokens_before_round": len(request.tokens)})
                    request.tokens.extend(landed[sid])
                    if landed[sid]:
                        ready_ns = round_receipt["token_ready_ns"]
                        if request.first_token_ns is None:
                            request.first_token_ns = ready_ns
                        request.final_token_ns = ready_ns
                    request.observed_compute_width = max(request.observed_compute_width,
                                                         active_width)
                    request.active_width_counts[active_width] = (
                        request.active_width_counts.get(active_width, 0) + 1)
                    request.finish_reason = finish_reasons.get(sid, request.finish_reason)
                    request.modes[mode] = request.modes.get(mode, 0) + 1
                    request.drafted += per_stream[sid]["drafted"]
                    request.accepted += per_stream[sid]["accepted"]
                    if finished[sid] or request.cancelled:
                        self._finish(sid, request)
        except BaseException as exc:
            with self._condition:
                requests = [*self._pending, *self._opening, *self._active.values()]
                self._pending.clear()
                self._opening.clear()
                self._active.clear()
                self._closed = True
            for request in requests:
                request.error = exc
                request.done.set()
        finally:
            self.profile.close()

    def close(self):
        with self._condition:
            self._closed = True
            requests = [*self._pending, *self._opening, *self._active.values()]
            self._pending.clear()
            self._condition.notify_all()
        for request in requests:
            request.error = RuntimeError("TensorFold-owned router closed")
            request.done.set()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            self.profile.close()
