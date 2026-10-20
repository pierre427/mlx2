"""Append-only private AST history without copying the committed prefix."""

from collections.abc import Sequence
from operator import index


class FrozenReceipt(dict):
    """JSON-compatible diagnostic snapshot shared by tokens of one lane-round."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("committed AST receipt is immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable
    __ior__ = _immutable

    def __deepcopy__(self, memo):
        return self


class FrozenReceiptList(list):
    def _immutable(self, *args, **kwargs):
        raise TypeError("committed AST receipt is immutable")

    __setitem__ = __delitem__ = append = extend = insert = pop = remove = _immutable
    clear = reverse = sort = __iadd__ = __imul__ = _immutable

    def __deepcopy__(self, memo):
        return self


def freeze_receipt(value, memo=None):
    memo = {} if memo is None else memo
    if isinstance(value, (FrozenReceipt, FrozenReceiptList)):
        return value
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, dict):
        result = FrozenReceipt(
            (key, freeze_receipt(item, memo)) for key, item in value.items()
        )
    elif isinstance(value, list):
        result = FrozenReceiptList(freeze_receipt(item, memo) for item in value)
    elif isinstance(value, tuple):
        result = tuple(freeze_receipt(item, memo) for item in value)
    else:
        return value
    memo[id(value)] = result
    return result


class PrivateHistory(Sequence):
    """Pinned committed prefix plus a private, short speculative suffix.

    AST runs exclusively inside the outer round's append-only history journal.
    The source may grow after publication; this view never exposes those later
    entries. It is never published as the authoritative request history.
    """

    __slots__ = ("_boundary", "_prefix", "_suffix")

    def __init__(self, prefix):
        self._prefix = prefix
        self._boundary = len(prefix)
        self._suffix = []

    def __len__(self):
        return self._boundary + len(self._suffix)

    def __getitem__(self, key):
        if len(self._prefix) < self._boundary:
            raise RuntimeError(
                "AST committed history was truncated during a private round"
            )
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if step == 1 and stop <= self._boundary:
                return self._prefix[start:stop]
            return [self[position] for position in range(start, stop, step)]
        position = index(key)
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError("AST private history index out of range")
        if position < self._boundary:
            return self._prefix[position]
        return self._suffix[position - self._boundary]

    def extend(self, values):
        self._suffix.extend(values)

    def __eq__(self, other):
        if not isinstance(other, Sequence):
            return NotImplemented
        return len(self) == len(other) and all(
            left == right for left, right in zip(self, other, strict=True)
        )
