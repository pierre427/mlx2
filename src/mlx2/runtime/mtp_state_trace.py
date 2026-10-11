# SPDX-License-Identifier: MIT
"""Caller-scoped, bounded MTP state diagnostics; never enabled implicitly.

Activate on the model worker thread, around the bounded diagnostic operation.
Readback can perturb scheduling. These receipts are diagnostic evidence, not
qualification or performance evidence. No tensors, prompt text, RNG keys,
model paths, or exception messages are retained.
"""

import hashlib
import math
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np

_ACTIVE = ContextVar("mlx2_mtp_state_trace", default=None)


class MTPStateTrace:
    def __init__(
        self,
        *,
        max_events=64,
        max_leaves=128,
        max_bytes_per_event=8 * 1024 * 1024,
        token_index=None,
    ):
        for name, value, ceiling in (
            ("max_events", max_events, 256),
            ("max_leaves", max_leaves, 512),
            ("max_bytes_per_event", max_bytes_per_event, 64 * 1024 * 1024),
        ):
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"{name} must be an integer in [1, {ceiling}]")
        if token_index is not None and (
            type(token_index) is not int or token_index < 1
        ):
            raise ValueError("token_index must be a positive one-based output position")
        self.max_events = max_events
        self.max_leaves = max_leaves
        self.max_bytes_per_event = max_bytes_per_event
        self.token_index = token_index
        self.events = []
        self.dropped_events = 0

    def _record(self, stage, build):
        if len(self.events) >= self.max_events:
            self.dropped_events = min(self.dropped_events + 1, 2**31 - 1)
            return
        event = {"stage": stage}
        try:
            build(event)
        except Exception as error:  # noqa: BLE001 - diagnostics cannot interrupt decode
            # Observation failure must not change decoding or expose payloads.
            event["diagnostic_error"] = type(error).__name__
        self.events.append(event)

    def state(
        self,
        stage,
        *,
        target,
        draft=None,
        seed=None,
        uid=None,
        generated=None,
        pending_hidden=None,
    ):
        def build(event):
            event.update(uid=uid, generated=generated, tensors=[])
            remaining = self.max_bytes_per_event
            leaves = 0
            truncated = False

            def walk(value, path, depth=0):
                nonlocal remaining, leaves, truncated
                if leaves >= self.max_leaves or depth > 8:
                    truncated = True
                    return
                if value is None:
                    return
                if isinstance(value, (list, tuple)):
                    for index, item in enumerate(value):
                        walk(item, f"{path}.{index}", depth + 1)
                        if leaves >= self.max_leaves:
                            truncated = True
                            break
                    return
                leaves += 1
                if hasattr(value, "shape") and hasattr(value, "dtype"):
                    shape = tuple(int(n) for n in value.shape)
                    nbytes = int(value.nbytes)
                    item = {
                        "path": path,
                        "shape": shape,
                        "dtype": str(value.dtype),
                        "nbytes": nbytes,
                    }
                    event["tensors"].append(item)
                    if nbytes > remaining:
                        item["omitted"] = "readback_budget"
                        truncated = True
                        return
                    remaining -= nbytes
                    # MLX bfloat16 has no NumPy dtype. A byte view retains its
                    # exact bits, including NaN payloads, without a float cast.
                    if type(value).__module__.startswith("mlx"):
                        import mlx.core as mx

                        host = np.asarray(value.reshape(-1).view(mx.uint8))
                    else:
                        host = np.asarray(value)
                    item["sha256"] = hashlib.sha256(host.tobytes()).hexdigest()
                    return
                if isinstance(value, (int, bool)):
                    event.setdefault("offsets", {})[path] = int(value)
                    return
                # Cache state is the public persistence seam, so adapters
                # retain ownership of tensor topology and model-specific math.
                if hasattr(value, "state"):
                    event.setdefault("cache_types", {})[path] = type(value).__name__
                    walk(value.state, f"{path}.state", depth + 1)
                    for name in ("offset", "lengths", "left_padding"):
                        offset = getattr(value, name, None)
                        if offset is None:
                            continue
                        walk(offset, f"{path}.{name}", depth + 1)
                        if (
                            hasattr(offset, "size")
                            and int(offset.size) <= 16
                            and int(offset.nbytes) <= remaining
                        ):
                            remaining -= int(offset.nbytes)
                            event.setdefault("offsets", {})[f"{path}.{name}"] = (
                                np.asarray(offset).reshape(-1).tolist()
                            )
                    return
                truncated = True
                event.setdefault("unsupported_paths", []).append(path)

            walk(pending_hidden, "pending_hidden")
            walk(seed, "tail_hidden")
            walk(draft, "draft")
            walk(target, "target")
            event["readback_bytes"] = self.max_bytes_per_event - remaining
            event["complete"] = not truncated

        self._record(stage, build)

    def verification(self, *, lane, drafts, accepted, logprobs, bonus):
        position = self.token_index
        if position is None or not lane.ntoks < position <= lane.ntoks + accepted + 1:
            return

        def build(event):
            row = position - lane.ntoks - 1
            values = logprobs[row]
            # Read only two selected values/indices; vocabulary-sized host
            # copies and records for downstream positions are unnecessary.
            import mlx.core as mx

            ids = mx.argpartition(values, kth=max(0, int(values.size) - 2))[-2:]
            scores = values[ids]
            mx.eval(ids, scores)
            pairs = sorted(
                zip(ids.tolist(), scores.tolist()), key=lambda p: (-p[1], p[0])
            )
            event.update(
                uid=int(lane.uid),
                token_index=position,
                top_two=[
                    {"token": int(i), "logprob": float(v) if math.isfinite(v) else None}
                    for i, v in pairs
                ],
                margin=float(pairs[0][1] - pairs[1][1])
                if len(pairs) == 2 and all(math.isfinite(v) for _, v in pairs)
                else None,
                selected=int(drafts[row]) if row < accepted else int(bonus),
                law="processed_target_logprobs",
            )

        self._record("selected_token_logprobs", build)

    def receipt(self):
        import copy

        return {
            "schema": "mlx2.mtp_state_trace.v1",
            "diagnostic_only": True,
            "token_index": self.token_index,
            "events": copy.deepcopy(self.events),
            "dropped_events": self.dropped_events,
            "limits": {
                "events": self.max_events,
                "leaves": self.max_leaves,
                "bytes_per_event": self.max_bytes_per_event,
            },
        }


@contextmanager
def activate(trace):
    if not isinstance(trace, MTPStateTrace):
        raise TypeError("trace must be MTPStateTrace")
    token = _ACTIVE.set(trace)
    try:
        yield trace
    finally:
        _ACTIVE.reset(token)


def state(stage, **kwargs):
    trace = _ACTIVE.get()
    if trace is not None:
        trace.state(stage, **kwargs)


def sidecar_state(stage, target, sidecar):
    trace = _ACTIVE.get()
    if trace is None:
        return
    try:
        value = getattr(sidecar, "state", sidecar)
        draft, seed = (
            (value[0], value[1])
            if isinstance(value, (list, tuple)) and len(value) >= 2
            else (None, None)
        )
        trace.state(stage, target=target, draft=draft, seed=seed)
    except Exception as error:  # noqa: BLE001 - diagnostics cannot interrupt decode
        error_name = type(error).__name__
        trace._record(stage, lambda event: event.update(diagnostic_error=error_name))


def round_state(stage, batch, drafts, accepted, logprobs=None, bonuses=None):
    trace = _ACTIVE.get()
    if trace is None:
        return
    for row, lane in enumerate(batch.lanes):
        selected = (
            trace.token_index is not None
            and lane.ntoks < trace.token_index <= lane.ntoks + accepted[row] + 1
        )
        if lane.stats.cycles != 0 and not selected:
            continue
        trace.state(
            stage,
            target=batch.caches.target,
            draft=batch.caches.draft,
            seed=lane.seed_h,
            uid=int(lane.uid),
            generated=int(lane.ntoks),
        )
        if logprobs is not None:

            def build(event, lane=lane, row=row):
                event.update(
                    uid=int(lane.uid),
                    generated=int(lane.ntoks),
                    drafts=[int(x) for x in drafts[row][:32]],
                    accepted=int(accepted[row]),
                    accepted_ids=[
                        int(x) for x in drafts[row][: min(accepted[row], 32)]
                    ],
                    bonus=int(bonuses[row]),
                    draft_ids_truncated=len(drafts[row]) > 32,
                )

            trace._record("verify_ids", build)
            trace.verification(
                lane=lane,
                drafts=drafts[row],
                accepted=accepted[row],
                logprobs=logprobs[row],
                bonus=bonuses[row],
            )


def committed_state(batch, emitted):
    trace = _ACTIVE.get()
    if trace is None:
        return
    for row, lane in enumerate(batch.lanes):
        start = lane.ntoks - emitted[row]
        selected = (
            trace.token_index is not None and start < trace.token_index <= lane.ntoks
        )
        if lane.stats.cycles != 1 and not selected:
            continue
        trace.state(
            "commit_complete",
            target=batch.caches.target,
            draft=batch.caches.draft,
            seed=lane.seed_h,
            pending_hidden=lane.pending_hs,
            uid=int(lane.uid),
            generated=int(lane.ntoks),
        )


def initial_token(lane, logprobs):
    trace = _ACTIVE.get()
    if trace is None or trace.token_index != 1:
        return
    from types import SimpleNamespace

    trace.verification(
        lane=SimpleNamespace(uid=lane.uid, ntoks=0),
        drafts=[],
        accepted=0,
        logprobs=logprobs[None],
        bonus=lane.cur,
    )
