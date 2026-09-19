"""Default-off undo/replay compatibility transaction for rotating KV caches."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Sequence

import mlx.core as mx

from .models.cache import RotatingKVCache


class RotatingReplayError(RuntimeError):
    pass


@dataclass(frozen=True)
class RotatingReplayPolicy:
    enabled: bool = False
    max_proposal_tokens: int = 0

    def __post_init__(self):
        if self.enabled and self.max_proposal_tokens <= 0:
            raise ValueError("enabled rotating replay requires a positive proposal bound")


class RotatingReplayTransaction:
    """Restore the pre-verify ring, then replay only its accepted prefix.

    This is a compatibility boundary for stock single-row ``RotatingKVCache``
    objects.  It is intentionally separate from mlx2's segmented transactional
    caches and is never selected implicitly.
    """

    def __init__(
        self,
        caches: Sequence[RotatingKVCache],
        proposal_token_ids: Sequence[int],
        *,
        request_id: str,
        state_revision: str,
        policy: RotatingReplayPolicy,
    ):
        if not policy.enabled:
            raise RotatingReplayError("rotating replay is disabled")
        tokens = tuple(int(token) for token in proposal_token_ids)
        if not tokens:
            raise ValueError("rotating replay requires proposed tokens")
        if len(tokens) > policy.max_proposal_tokens:
            raise RotatingReplayError("proposal exceeds the qualified replay bound")
        if not caches or any(type(cache) is not RotatingKVCache for cache in caches):
            raise RotatingReplayError(
                "rotating replay supports only stock single-row RotatingKVCache"
            )
        if any(cache.speculating for cache in caches):
            raise RotatingReplayError("rotating cache already has an armed transaction")
        self.caches = list(caches)
        self.tokens = tokens
        self.request_id = str(request_id)
        self.source_revision = str(state_revision)
        self.source_offsets = tuple(int(cache.offset) for cache in caches)
        self.closed = False
        armed = []
        try:
            for cache in self.caches:
                cache.start_speculation(rollback_window=len(tokens))
                armed.append(cache)
        except BaseException:
            for cache in armed:
                cache.stop_speculation()
            raise

    def _snapshot_caches(self, *, copy_arrays: bool):
        def snapshot_array(value):
            if value is None or not copy_arrays:
                return value
            return mx.array(value)

        return [
            (
                snapshot_array(cache.keys),
                snapshot_array(cache.values),
                cache._idx,
                cache.offset,
                deque(cache._rollbacks),
                cache.speculating,
                cache._rollback_window,
            )
            for cache in self.caches
        ]

    def _restore_snapshots(self, snapshots):
        for cache, snapshot in zip(self.caches, snapshots):
            (
                cache.keys,
                cache.values,
                cache._idx,
                cache.offset,
                rollbacks,
                cache.speculating,
                cache._rollback_window,
            ) = snapshot
            cache._rollbacks = deque(rollbacks)

    def _restore(self):
        for cache, source in zip(self.caches, self.source_offsets):
            advanced = int(cache.offset) - source
            if advanced != len(self.tokens):
                raise RotatingReplayError(
                    "verify advance does not match the declared proposal"
                )
            recorded = sum(item[0] for item in cache._rollbacks)
            if recorded < len(self.tokens):
                raise RotatingReplayError(
                    "rotating cache cannot exactly restore the declared proposal"
                )
        snapshots = self._snapshot_caches(copy_arrays=False)
        try:
            for cache in self.caches:
                cache.trim(len(self.tokens))
        except BaseException as error:
            self._restore_snapshots(snapshots)
            raise RotatingReplayError("rotating replay rollback was not atomic") from error
        else:
            for cache in self.caches:
                cache.stop_speculation()

    def commit(self, accepted_tokens: int, replay: Callable[[tuple[int, ...]], None]):
        if self.closed:
            raise RotatingReplayError("rotating replay transaction is closed")
        if isinstance(accepted_tokens, bool) or not 0 <= accepted_tokens <= len(self.tokens):
            raise ValueError("accepted token count is outside the proposal")
        self._restore()
        accepted = self.tokens[:accepted_tokens]
        if accepted:
            source_snapshots = self._snapshot_caches(copy_arrays=True)
            for cache in self.caches:
                cache.start_speculation(rollback_window=len(accepted))
            try:
                replay(accepted)
                expected = tuple(source + len(accepted) for source in self.source_offsets)
                actual = tuple(int(cache.offset) for cache in self.caches)
                if actual != expected:
                    raise RotatingReplayError(
                        "accepted-prefix replay advanced the wrong cache extent"
                    )
            except BaseException:
                self._restore_snapshots(source_snapshots)
                for cache in self.caches:
                    cache.stop_speculation()
                self.closed = True
                raise
            else:
                for cache in self.caches:
                    cache.stop_speculation()
        expected = tuple(source + len(accepted) for source in self.source_offsets)
        actual = tuple(int(cache.offset) for cache in self.caches)
        if actual != expected:
            raise RotatingReplayError("accepted-prefix replay publication mismatch")
        self.closed = True
        return self._receipt(len(accepted), replayed=bool(accepted))

    def commit_verified(self):
        """Publish a wholly accepted verify forward as it stands.

        When every declared token is kept there is nothing to restore, so the
        ring written by the verify forward is already the committed state and
        no replay forward is owed.  Partial acceptance must use ``commit``.
        """
        if self.closed:
            raise RotatingReplayError("rotating replay transaction is closed")
        for cache, source in zip(self.caches, self.source_offsets):
            if int(cache.offset) - source != len(self.tokens):
                raise RotatingReplayError(
                    "verify advance does not match the declared proposal"
                )
        for cache in self.caches:
            cache.stop_speculation()
        self.closed = True
        return self._receipt(len(self.tokens), replayed=False)

    def _receipt(self, accepted_tokens: int, *, replayed: bool):
        return {
            "schema": "mlx2.rotating-undo-replay.v1",
            "request_id": self.request_id,
            "source_revision": self.source_revision,
            "proposed_tokens": len(self.tokens),
            "accepted_tokens": int(accepted_tokens),
            "replayed": bool(replayed),
            "status": "committed",
            "selected": True,
        }

    def rollback(self):
        if self.closed:
            return
        self._restore()
        self.closed = True
