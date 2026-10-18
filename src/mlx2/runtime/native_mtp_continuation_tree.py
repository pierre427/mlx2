"""Request-private cache ownership for greedy native-MTP continuation trees.

The native head can rank more than one complete continuation, but a live
self-MTP request owns more than token IDs: its target cache, head cache and
pending teacher-forcing state form one sidecar.  This module gives adapters a
small transaction for materialising those alternatives from one pre-round
checkpoint.  No branch is authoritative until the caller selects and commits
it; abort and failure discard every private branch.

This is intentionally a greedy-only seam.  A sampled native-head tree needs an
exact joint proposal law and a request-RNG schedule before it can share this
lifecycle contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .cow_cache import restore_recovery_descriptors, snapshot_recovery_descriptors
from .hybrid_speculative import SelfMTPCachePair


GREEDY_NATIVE_MTP_TREE_PROTOCOL = "greedy-native-mtp-sidecar-tree-v1"


@dataclass
class NativeMTPSidecar:
    """All request-private state that must move together at publication."""

    caches: SelfMTPCachePair
    seed_h: Any
    pending_hs: Any
    pending_ts: list[int]
    cur: int
    token_prefix: Any


@dataclass(frozen=True)
class NativeMTPContinuationBranch:
    ordinal: int
    tokens: tuple[int, ...]
    sidecar: NativeMTPSidecar


def sidecar_from_lane(lane: Any, caches: SelfMTPCachePair) -> NativeMTPSidecar:
    """Capture the capability-owned fields needed by a private MTP branch."""

    if not isinstance(caches, SelfMTPCachePair):
        raise TypeError("native MTP continuation requires a SelfMTPCachePair")
    pending_ts = getattr(lane, "pending_ts", None)
    if not isinstance(pending_ts, list) or any(
        type(token) is not int for token in pending_ts
    ):
        raise ValueError("native MTP pending tokens must be a list of integers")
    cur = getattr(lane, "cur", None)
    if type(cur) is not int or cur < 0:
        raise ValueError("native MTP continuation requires a valid current token")
    return NativeMTPSidecar(
        caches=caches,
        seed_h=getattr(lane, "seed_h"),
        pending_hs=getattr(lane, "pending_hs"),
        pending_ts=list(pending_ts),
        cur=cur,
        token_prefix=getattr(lane, "token_prefix"),
    )


def _freeze_sidecar(sidecar: NativeMTPSidecar):
    pair = sidecar.caches
    target, draft, borrowed = snapshot_recovery_descriptors(pair.target, pair.draft)
    return (
        target,
        draft,
        borrowed,
        sidecar.seed_h,
        sidecar.pending_hs,
        tuple(sidecar.pending_ts),
        sidecar.cur,
        sidecar.token_prefix,
    )


def _restore_sidecar(frozen) -> NativeMTPSidecar:
    target, draft, borrowed, seed_h, pending_hs, pending_ts, cur, token_prefix = frozen
    target, draft = restore_recovery_descriptors(target, draft, borrowed)
    # Array objects are immutable in MLX, but the host containers are not.
    return NativeMTPSidecar(
        SelfMTPCachePair(target, draft),
        seed_h,
        pending_hs,
        list(pending_ts),
        cur,
        token_prefix,
    )


def _dispose_sidecar(sidecar: NativeMTPSidecar | None) -> None:
    if sidecar is None:
        return
    first_error = None
    for cache_group in (sidecar.caches.target, sidecar.caches.draft):
        for cache in cache_group:
            close = getattr(cache, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as error:  # disposal is best-effort
                    if first_error is None:
                        first_error = error
    if first_error is not None:
        raise first_error


class NativeMTPContinuationTreeTransaction:
    """Own private sidecars for ranked complete greedy native-MTP paths."""

    protocol = GREEDY_NATIVE_MTP_TREE_PROTOCOL
    proposal_distribution = "deterministic_point_mass"

    def __init__(
        self,
        checkpoint: NativeMTPSidecar,
        paths: Sequence[Sequence[int]],
        materialize: Callable[[NativeMTPSidecar, tuple[int, ...], int], None],
        *,
        sampling_temperature: float = 0.0,
        proposal_distribution: str = "deterministic_point_mass",
    ):
        if (
            sampling_temperature != 0
            or proposal_distribution != self.proposal_distribution
        ):
            raise ValueError(
                "native MTP continuation trees are greedy-only until an exact-law "
                "sampled tree protocol is implemented"
            )
        if not isinstance(checkpoint, NativeMTPSidecar):
            raise TypeError("native MTP continuation requires a sidecar checkpoint")
        normalized = tuple(tuple(path) for path in paths)
        if (
            not normalized
            or len(normalized) > 15
            or any(
                not path
                or len(path) > 15
                or any(type(token) is not int or token < 0 for token in path)
                for path in normalized
            )
            or len(set(normalized)) != len(normalized)
        ):
            raise ValueError("native MTP continuation paths are invalid")
        if not callable(materialize):
            raise TypeError("native MTP continuation materializer must be callable")
        self.paths = normalized
        self._branches: list[NativeMTPContinuationBranch] = []
        self.closed = False
        frozen = _freeze_sidecar(checkpoint)
        try:
            for ordinal, path in enumerate(normalized):
                sidecar = _restore_sidecar(frozen)
                result = materialize(sidecar, path, ordinal)
                if result is not None:
                    raise TypeError(
                        "native MTP continuation materializer must mutate its "
                        "private sidecar in place"
                    )
                if not isinstance(sidecar.caches, SelfMTPCachePair):
                    raise TypeError("native MTP continuation lost its cache pair")
                self._branches.append(
                    NativeMTPContinuationBranch(ordinal, path, sidecar)
                )
        except BaseException:
            try:
                self._discard_all()
            except BaseException:
                # Preserve the materialization fault. No branch is published,
                # and disposal failure cannot make one authoritative.
                pass
            self.closed = True
            raise

    @property
    def branches(self) -> tuple[NativeMTPContinuationBranch, ...]:
        return tuple(self._branches)

    def commit(
        self, selected_index: int, *, selected_tokens: Sequence[int]
    ) -> NativeMTPSidecar:
        """Atomically hand off one complete sidecar and destroy every sibling.

        Publication is the returned object-reference handoff: target cache,
        draft cache and pending state are never returned separately.  The
        caller replaces its authoritative sidecar as one unit.
        """

        if self.closed:
            raise RuntimeError("native MTP continuation transaction is closed")
        if type(selected_index) is not int or not 0 <= selected_index < len(self._branches):
            raise IndexError("native MTP continuation selected branch is invalid")
        selected = self._branches[selected_index]
        if tuple(selected_tokens) != selected.tokens:
            raise ValueError("native MTP continuation selection/path mismatch")
        kept = selected.sidecar
        errors = []
        for branch in self._branches:
            if branch is selected:
                continue
            try:
                _dispose_sidecar(branch.sidecar)
            except BaseException as error:
                errors.append(error)
        self._branches.clear()
        self.closed = True
        if errors:
            # The selected state has not escaped yet. Dispose it too rather
            # than publishing after an incomplete sibling retirement.
            try:
                _dispose_sidecar(kept)
            finally:
                raise RuntimeError(
                    "native MTP continuation sibling disposal failed"
                ) from errors[0]
        return kept

    def abort(self) -> None:
        if self.closed:
            return
        try:
            self._discard_all()
        finally:
            self.closed = True

    def _discard_all(self) -> None:
        errors = []
        for branch in self._branches:
            try:
                _dispose_sidecar(branch.sidecar)
            except BaseException as error:
                errors.append(error)
        self._branches.clear()
        if errors:
            raise RuntimeError(
                "native MTP continuation branch disposal failed"
            ) from errors[0]

