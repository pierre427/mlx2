"""Exact target sampling over ranked complete continuation paths.

Paths contain proposed tokens, never a joint draft distribution. Each target
prefix draws once from a canonical matching branch. Only that branch's reached
input prefix is committed; all other request-private branches are discarded.
"""

from __future__ import annotations

from dataclasses import dataclass

from .continuation_strategy import LongestFirstExactPrefix
from .cow_cache import restore_recovery_descriptors, snapshot_recovery_descriptors


@dataclass(frozen=True)
class ContinuationOutcome:
    selected_index: int
    accepted: int
    emitted: tuple[int, ...]
    physical_width: int
    physical_span: int
    input_lengths: tuple[int, ...]
    matched_indices: tuple[int, ...]
    attempted_indices: tuple[int, ...] = ()
    pruned_siblings: int = 0
    shared_prefix_tokens_reused: int = 0
    algorithm: str = "parallel_complete_paths_v1"
    launches: int = 1
    target_rows: int | None = None
    shared_prefix_reused_tokens: int = 0

    @property
    def executed_target_rows(self):
        if self.target_rows is not None:
            return self.target_rows
        return self.physical_width * self.physical_span


class MaterializedContinuationTransaction:
    """One already-pruned private branch ready for authoritative publication."""

    def __init__(self, cache, expected_rows):
        self.cache = cache
        self.expected_rows = int(expected_rows)
        self.closed = False

    def commit(self, accepted_lengths):
        if self.closed:
            raise RuntimeError("materialized continuation transaction is closed")
        if list(accepted_lengths) != [self.expected_rows]:
            raise ValueError("materialized continuation commit length mismatch")
        self.closed = True
        cache, self.cache = self.cache, None
        return [cache]

    def abort(self):
        self.cache = None
        self.closed = True


@dataclass(frozen=True)
class PrefixCascadeAttempt:
    """One longest-first branch beyond an authoritative target frontier."""

    index: int
    path: tuple[int, ...]
    suffix: tuple[int, ...]
    reused_prefix_tokens: int


class ExactPrefixFrontier:
    """Plan staged complete-path verification without replaying common tokens.

    The frontier contains target tokens that have already been committed to the
    request's exact cache state.  A sibling remains possible only when its
    proposal starts with that complete frontier.  The planner never mutates a
    cache; callers must advance it with an exact transaction before recording
    the emitted target tokens here.
    """

    def __init__(self, paths):
        paths = tuple(tuple(path) for path in paths)
        if not paths or any(
            not path or any(type(token) is not int or token < 0 for token in path)
            for path in paths
        ):
            raise ValueError("exact prefix cascade requires nonempty token paths")
        self.paths = paths
        self.order = tuple(
            index
            for index, _path in sorted(
                enumerate(paths), key=lambda item: (-len(item[1]), item[0])
            )
        )
        self.frontier = ()
        self.attempted = set()
        self.pruned = set()
        self.reused_prefix_tokens = 0

    def next_attempt(self):
        """Return the next viable longest path, or ``None`` after pruning."""
        progress = len(self.frontier)
        for index in self.order:
            if index in self.attempted or index in self.pruned:
                continue
            path = self.paths[index]
            if len(path) <= progress or path[:progress] != self.frontier:
                self.pruned.add(index)
                continue
            self.attempted.add(index)
            self.reused_prefix_tokens += progress
            return PrefixCascadeAttempt(
                index=index,
                path=path,
                suffix=path[progress:],
                reused_prefix_tokens=progress,
            )
        return None

    def record(self, attempt, outcome):
        """Advance only after the corresponding exact cache commit succeeds."""
        if not isinstance(attempt, PrefixCascadeAttempt):
            raise TypeError("exact prefix cascade requires its attempt record")
        if attempt.index not in self.attempted or attempt.path != self.paths[attempt.index]:
            raise ValueError("stale or foreign exact prefix attempt")
        if tuple(attempt.path[: len(self.frontier)]) != self.frontier:
            raise ValueError("exact prefix attempt no longer matches the frontier")
        emitted = tuple(int(token) for token in outcome.emitted)
        accepted = int(outcome.accepted)
        if (
            accepted < 0
            or accepted > len(attempt.suffix)
            or accepted > len(emitted)
            or emitted[:accepted] != attempt.suffix[:accepted]
            or not emitted
        ):
            raise ValueError("continuation outcome does not prove an exact prefix")
        self.frontier = (*self.frontier, *emitted)
        for index, path in enumerate(self.paths):
            if index in self.attempted or index in self.pruned:
                continue
            if len(path) <= len(self.frontier) or path[: len(self.frontier)] != self.frontier:
                self.pruned.add(index)

    def receipt(self):
        return {
            "algorithm": "longest-first-exact-prefix-v1",
            "frontier_tokens": len(self.frontier),
            "attempted_paths": len(self.attempted),
            "pruned_paths": len(self.pruned),
            "shared_prefix_tokens_reused": self.reused_prefix_tokens,
        }


class SelectedContinuationTransaction:
    """Translate the ordinary one-lane commit to one kept branch epoch."""

    def __init__(self, transaction, selected_index, count):
        self.transaction = transaction
        self.selected_index = selected_index
        self.count = count

    @property
    def closed(self):
        return self.transaction.closed

    def commit(self, accepted_lengths):
        if len(accepted_lengths) != 1:
            raise ValueError("selected continuation commit requires one request")
        # Hybrid owners require at least the reached anchor for each private
        # epoch. Close unused epochs there, then discard those cache objects;
        # only the selected branch is returned to the authoritative request.
        kept = [1] * self.count
        kept[self.selected_index] = accepted_lengths[0]
        rows = self.transaction.commit(accepted_lengths=kept)
        return [rows[self.selected_index]]

    def abort(self):
        self.transaction.abort()


class LongestFirstContinuationTransaction:
    """Finalize the last private suffix after earlier prefixes were committed."""

    def __init__(self, transaction, *, final_count, total_count):
        self.transaction = transaction
        self.final_count = final_count
        self.total_count = total_count

    @property
    def closed(self):
        return self.transaction.closed

    def commit(self, accepted_lengths):
        if accepted_lengths != [self.total_count]:
            raise ValueError(
                "longest-first continuation commit must cover its exact reached prefix"
            )
        return self.transaction.commit(accepted_lengths=[self.final_count])

    def abort(self):
        self.transaction.abort()


def prepare_continuations(
    model,
    mx,
    cache,
    anchor,
    paths,
    capture_layers,
    owner_factory,
    *,
    max_sequences=15,
    max_depth=15,
):
    """One physical target forward for all bounded paths of one request."""
    paths = tuple(tuple(path) for path in paths)
    if (
        not paths
        or not 1 <= len(paths) <= max_sequences <= 15
        or not 1 <= max_depth <= 15
        or any(
            not path
            or len(path) > max_depth
            or any(type(t) is not int or t < 0 for t in path)
            for path in paths
        )
    ):
        raise ValueError("invalid complete continuation paths")
    vocab = getattr(getattr(model, "args", None), "vocab_size", None)
    if vocab is None:
        vocab = getattr(
            getattr(getattr(model, "args", None), "text_config", None),
            "vocab_size",
            None,
        )
    if vocab is not None and any(token >= vocab for path in paths for token in path):
        raise ValueError("continuation token outside target vocabulary")
    lengths = tuple(len(path) + 1 for path in paths)
    span = max(lengths)
    snapshots = snapshot_recovery_descriptors(cache)
    branches = [restore_recovery_descriptors(*snapshots)[0] for _ in paths]
    transaction = owner_factory(branches).begin(lengths=lengths)
    inputs = [[int(anchor), *path] + [0] * (span - len(path) - 1) for path in paths]
    try:
        logits, features = model.forward_with_taps(
            mx.array(inputs), transaction.caches, capture_layers
        )
        mx.eval(logits, features)
    except BaseException:
        transaction.abort()
        raise
    return paths, logits, features, transaction


def sample_continuations(paths, logits, sample_row, *, maximum, stop_tokens=()):
    """Walk actual history; ``sample_row(logits,prefix)`` performs one draw."""
    if maximum < 1:
        raise ValueError("continuation output budget must be positive")
    active = list(range(len(paths)))
    emitted = []
    accepted = 0
    selected = active[0]
    matched = []
    stops = set(stop_tokens)
    for position in range(min(max(map(len, paths)) + 1, maximum)):
        # Every active path shares exactly the emitted prefix. Rank order is
        # fixed before target draws and cannot select a target law by its draw.
        selected = active[0]
        token = int(sample_row(logits[selected, position], tuple(emitted)))
        emitted.append(token)
        survivors = [
            index
            for index in active
            if len(paths[index]) > position and paths[index][position] == token
        ]
        if survivors:
            accepted = position + 1
            matched = survivors
        else:
            accepted = position
            matched = []
        if token in stops or not survivors or len(emitted) >= maximum:
            break
        active = survivors
    if matched:
        # A stop or the budget can end the walk on a draw only a lower-ranked
        # survivor proposed.  Name the highest-ranked path that contains the
        # emitted tokens; its branch holds the same committed prefix rows.
        selected = matched[0]
    return ContinuationOutcome(
        selected,
        accepted,
        tuple(emitted),
        len(paths),
        int(logits.shape[1]),
        tuple(len(path) + 1 for path in paths),
        tuple(matched),
    )


def verify_longest_first_continuations(
    paths,
    cache,
    anchor,
    prepare_attempt,
    sample_row,
    *,
    maximum,
    stop_tokens=(),
):
    """Verify exact complete paths serially, reusing each reached prefix.

    ``prepare_attempt(cache, anchor, suffix)`` returns ``(logits, features,
    transaction[, settle])`` for one request-private branch.  A failed path
    commits only its reached input prefix.  Siblings that cannot match the
    target-drawn prefix are then removed, and the next longest viable sibling
    starts after that exact checkpoint rather than recomputing common tokens.

    The caller must gate this helper on a target/cache contract that proves
    contextual prefix equivalence.  This function never publishes to APCv2.
    """
    paths = tuple(tuple(path) for path in paths)
    if (
        not paths
        or type(maximum) is not int
        or maximum < 1
        or any(not path or any(type(token) is not int or token < 0 for token in path)
               for path in paths)
    ):
        raise ValueError("invalid longest-first continuation paths")
    order = tuple(
        sorted(range(len(paths)), key=lambda index: (-len(paths[index]), index))
    )
    stops = set(stop_tokens)
    emitted = []
    attempted = []
    remaining = set(order)
    feature_slices = []
    current_cache = cache
    current_anchor = int(anchor)
    target_rows = 0
    reused = 0
    selected = order[0]
    try:
        while len(emitted) < maximum:
            prefix = tuple(emitted)
            viable = [
                index
                for index in order
                if index in remaining
                and len(paths[index]) > len(prefix)
                and paths[index][: len(prefix)] == prefix
            ]
            if not viable:
                break
            selected = viable[0]
            remaining.remove(selected)
            attempted.append(selected)
            progress = len(prefix)
            # ``prefix[-1]`` is the new anchor and is evaluated by this
            # attempt. Earlier target-drawn proposal tokens already exist in
            # ``current_cache`` and are the rows actually spared recompute.
            reused += max(0, progress - 1)
            suffix = paths[selected][progress:]
            prepared = prepare_attempt(current_cache, current_anchor, suffix)
            if len(prepared) not in (3, 4):
                raise ValueError("prepare_attempt must return three or four values")
            logits, features, transaction = prepared[:3]
            settle = prepared[3] if len(prepared) == 4 else None
            local = sample_continuations(
                (suffix,),
                logits,
                lambda row, local_prefix: sample_row(
                    row, tuple(emitted) + tuple(local_prefix)
                ),
                maximum=maximum - len(emitted),
                stop_tokens=stops,
            )
            consumed = min(local.accepted + 1, len(local.emitted))
            try:
                current_cache = transaction.commit([consumed])[0]
            except BaseException:
                if not transaction.closed:
                    transaction.abort()
                raise
            if settle is not None:
                settle(consumed)
            feature_slices.append(features[:, :consumed])
            target_rows += int(logits.shape[1])
            emitted.extend(local.emitted)
            current_anchor = int(emitted[-1])
            # Once the target draw disagrees, no sibling outside the exact
            # reached prefix can become authoritative in this round.
            reached = tuple(emitted)
            remaining.intersection_update(
                index
                for index in remaining
                if len(paths[index]) >= len(reached)
                and paths[index][: len(reached)] == reached
            )
            if (
                current_anchor in stops
                or len(emitted) >= maximum
                or local.accepted == len(suffix)
                or not remaining
            ):
                break
    except BaseException:
        current_cache = None
        raise
    selected_path = paths[selected]
    accepted = 0
    while (
        accepted < len(selected_path)
        and accepted < len(emitted)
        and selected_path[accepted] == emitted[accepted]
    ):
        accepted += 1
    matched = tuple(
        index
        for index, path in enumerate(paths)
        if path[:accepted] == tuple(emitted[:accepted])
    )
    pruned = len(paths) - len(remaining) - len(attempted)
    outcome = ContinuationOutcome(
        selected_index=selected,
        accepted=accepted,
        emitted=tuple(emitted),
        physical_width=1,
        physical_span=target_rows,
        input_lengths=tuple(len(paths[index]) + 1 for index in attempted),
        matched_indices=matched,
        attempted_indices=tuple(attempted),
        pruned_siblings=pruned,
        shared_prefix_tokens_reused=reused,
        algorithm="longest_first_exact_prefix_v1",
        launches=len(attempted),
        target_rows=target_rows,
        shared_prefix_reused_tokens=reused,
    )
    return (
        outcome,
        tuple(feature_slices),
        MaterializedContinuationTransaction(current_cache, len(emitted)),
    )


def prepare_longest_first_continuations(
    model,
    mx,
    cache,
    anchor,
    paths,
    capture_layers,
    owner_factory,
    sample_row,
    *,
    maximum,
    stop_tokens=(),
    max_sequences=15,
    max_depth=15,
):
    """Verify longest paths serially while reusing exact reached-prefix state.

    Every launch owns a private branch cloned from the current reached prefix.
    A mismatch commits only ``anchor + accepted proposal`` inputs; the target
    correction becomes the next anchor.  Consequently a surviving sibling is
    evaluated only from its first not-yet-seen token.  No intermediate state is
    published to APCv2 or to the live lane before the returned transaction is
    committed by the ordinary serving boundary.
    """
    paths = tuple(tuple(path) for path in paths)
    if len(paths) > max_sequences or any(len(path) > max_depth for path in paths):
        raise ValueError("longest-first continuation geometry exceeds its bound")
    controller = LongestFirstExactPrefix(
        paths, maximum=maximum, stop_tokens=stop_tokens
    )
    current_cache = cache
    current_anchor = int(anchor)
    feature_parts = []
    attempts = []
    final_transaction = None
    final_count = 0
    pruned = set()
    while True:
        attempt = controller.next_attempt()
        if attempt is None:
            raise RuntimeError("longest-first verification made no terminal attempt")
        _paths, logits, features, transaction = prepare_continuations(
            model,
            mx,
            current_cache,
            current_anchor,
            (attempt.suffix,),
            capture_layers,
            owner_factory,
            max_sequences=1,
            max_depth=max_depth,
        )

        prefix = controller.emitted

        def draw(row, local_prefix, prefix=prefix):
            return sample_row(row, prefix + tuple(local_prefix))

        try:
            local = sample_continuations(
                (attempt.suffix,),
                logits,
                draw,
                maximum=maximum - len(controller.emitted),
                stop_tokens=stop_tokens,
            )
            observation = controller.observe(attempt, local.emitted)
        except BaseException:
            if not transaction.closed:
                transaction.abort()
            raise
        used = len(local.emitted)
        pruned.update(observation.pruned_indices)
        attempts.append(attempt)
        if observation.terminal:
            final_transaction = transaction
            final_count = used
            feature_parts.append(features[:, :used])
            break
        try:
            current_cache = transaction.commit(accepted_lengths=[used])[0]
        except BaseException:
            if not transaction.closed:
                transaction.abort()
            raise
        feature_parts.append(features[:, :used])
        current_anchor = observation.emitted[-1]

    try:
        hidden = (
            feature_parts[0]
            if len(feature_parts) == 1
            else mx.concatenate(feature_parts, axis=1)
        )
        total_count = len(controller.emitted)
        selected_path = paths[attempts[-1].path_index]
        accepted = 0
        while (
            accepted < len(selected_path)
            and accepted < total_count
            and selected_path[accepted] == controller.emitted[accepted]
        ):
            accepted += 1
        outcome = ContinuationOutcome(
            selected_index=attempts[-1].path_index,
            accepted=accepted,
            emitted=controller.emitted,
            physical_width=1,
            physical_span=max(attempt.input_rows for attempt in attempts),
            input_lengths=tuple(attempt.input_rows for attempt in attempts),
            matched_indices=tuple(controller.viable),
            algorithm="longest_first_exact_prefix_v1",
            launches=len(attempts),
            target_rows=sum(attempt.input_rows for attempt in attempts),
            pruned_siblings=len(pruned),
            shared_prefix_reused_tokens=sum(
                attempt.start for attempt in attempts[1:]
            ),
        )
    except BaseException:
        if final_transaction is not None and not final_transaction.closed:
            final_transaction.abort()
        raise
    return (
        outcome,
        hidden,
        LongestFirstContinuationTransaction(
            final_transaction,
            final_count=final_count,
            total_count=total_count,
        ),
    )
