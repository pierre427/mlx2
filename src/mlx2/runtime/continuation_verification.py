"""Exact target sampling over ranked complete continuation paths.

Paths contain proposed tokens, never a joint draft distribution. Each target
prefix draws once from a canonical matching branch. Only that branch's reached
input prefix is committed; all other request-private branches are discarded.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    return ContinuationOutcome(
        selected,
        accepted,
        tuple(emitted),
        len(paths),
        int(logits.shape[1]),
        tuple(len(path) + 1 for path in paths),
        tuple(matched),
    )
