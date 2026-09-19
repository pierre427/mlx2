# SPDX-License-Identifier: Apache-2.0
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
import mlx.core as mx
from .models.cache import KVCache
from .models.base import create_causal_mask
from .models.qwen4_exp import QSAKVCache


class SharedSuffixQSAError(RuntimeError):
    """The shared-prefix/suffix storage contract was violated."""


def _coverage(identity: dict[str, Any] | None) -> int:
    if identity is None:
        return 0
    try:
        value = int(identity.get("complete_blocks", 0))
    except (TypeError, ValueError) as error:
        raise SharedSuffixQSAError("invalid QSA summary coverage") from error
    if value < 0:
        raise SharedSuffixQSAError("QSA summary coverage cannot be negative")
    return value


def _with_coverage(
    identity: dict[str, Any] | None, complete_blocks: int
) -> dict[str, Any] | None:
    result = {} if identity is None else dict(identity)
    result["complete_blocks"] = int(complete_blocks)
    return result


@dataclass(frozen=True)
class QSAImmutableBase:
    """One aligned, batch-one QSA prefix shared by row adapters by identity."""

    keys: mx.array
    values: mx.array
    index_keys: mx.array
    length: int
    layout_id: str
    pooled_keys: mx.array | None = None
    pooled_ratio: int | None = None
    summary_identity: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        length = int(self.length)
        if length <= 0 or length % 4:
            raise ValueError("shared QSA base length must be positive and 4-aligned")
        if not self.layout_id:
            raise ValueError("shared QSA base requires a layout identity")
        if self.keys.ndim != 4 or self.values.ndim != 4:
            raise ValueError("shared QSA K/V must be rank four")
        if int(self.keys.shape[0]) != 1 or int(self.values.shape[0]) != 1:
            raise ValueError("shared QSA K/V must have one source row")
        if self.keys.shape[:3] != self.values.shape[:3]:
            raise ValueError("shared QSA K/V geometries differ")
        if int(self.keys.shape[2]) != length:
            raise ValueError("shared QSA K/V width differs from its length")
        if self.index_keys.ndim != 3 or int(self.index_keys.shape[0]) != 1:
            raise ValueError("shared QSA raw-index ledger must be rank three, B=1")
        if int(self.index_keys.shape[1]) != length:
            raise ValueError("shared QSA raw-index ledger differs from its length")
        if self.pooled_keys is None:
            if self.pooled_ratio is not None or _coverage(self.summary_identity):
                raise ValueError("QSA summary metadata exists without pooled keys")
            return
        if self.pooled_keys.ndim != 3 or int(self.pooled_keys.shape[0]) != 1:
            raise ValueError("shared QSA pooled keys must be rank three, B=1")
        if self.pooled_ratio is None or int(self.pooled_ratio) <= 0:
            raise ValueError("shared QSA pooled keys require a positive ratio")
        complete = _coverage(self.summary_identity)
        if complete != int(self.pooled_keys.shape[1]):
            raise ValueError("QSA summary identity and pooled coverage disagree")
        if complete > length // int(self.pooled_ratio):
            raise ValueError("QSA pooled coverage exceeds the shared base")

    @classmethod
    def from_cache(
        cls, cache: QSAKVCache, *, layout_id: str, length: int | None = None
    ) -> "QSAImmutableBase":
        """Alias a complete prefix from a stock cache without copying it."""
        if type(cache) is not QSAKVCache:
            raise TypeError("shared QSA base requires a plain QSAKVCache")
        stop = int(cache.offset if length is None else length)
        if stop > int(cache.offset):
            raise ValueError("shared QSA base exceeds the source cache")
        if cache.keys is None or cache.values is None or cache.index_keys is None:
            raise ValueError("cannot share an unpopulated QSA cache")
        if int(cache.index_keys.shape[1]) < stop:
            raise ValueError("QSA raw-index ledger is shorter than the shared base")
        pooled = getattr(cache, "_qsa_pooled_keys", None)
        identity = getattr(cache, "_qsa_summary_identity", None)
        ratio = getattr(cache, "_qsa_pooled_ratio", None)
        if pooled is not None and ratio:
            keep = min(int(pooled.shape[1]), stop // int(ratio))
            pooled = pooled[:, :keep]
            identity = _with_coverage(identity, keep)
        elif pooled is None:
            ratio = None
            identity = _with_coverage(identity, 0)
        return cls(
            keys=cache.keys[..., :stop, :],
            values=cache.values[..., :stop, :],
            index_keys=cache.index_keys[:, :stop],
            length=stop,
            layout_id=layout_id,
            pooled_keys=pooled,
            pooled_ratio=ratio,
            summary_identity=identity,
        )

    @property
    def nbytes(self) -> int:
        return int(self.keys.nbytes + self.values.nbytes + self.index_keys.nbytes) + (
            0 if self.pooled_keys is None else int(self.pooled_keys.nbytes)
        )


@dataclass(frozen=True)
class QSAMaterializationReceipt:
    layout_id: str
    base_length: int
    suffix_length: int
    materialized_bytes: int
    explicit: bool = True


class SharedSuffixQSAKVCache:
    """One row-local, physically allocated suffix over an immutable QSA base."""

    supports_shared_qsa_suffix = True

    def __init__(
        self, base: QSAImmutableBase, *, note: Callable[[str, int], None] | None = None
    ) -> None:
        if not isinstance(base, QSAImmutableBase):
            raise TypeError("shared QSA row requires a QSAImmutableBase")
        self.base = base
        self._kv = KVCache()
        self.index_keys: mx.array | None = None
        self._suffix_pooled_keys: mx.array | None = None
        self._note = note
        self._mtp_share_topk = False
        self._mtp_shared_topk = None
        self._mtp_shared_topk_n_blocks = None

    @classmethod
    def from_cache(
        cls,
        cache: QSAKVCache,
        base: QSAImmutableBase,
        *,
        note: Callable[[str, int], None] | None = None,
    ) -> "SharedSuffixQSAKVCache":
        """Split one attested stock row at ``base.length`` without joining it."""
        if type(cache) is not QSAKVCache:
            raise TypeError("shared QSA suffix requires a plain QSAKVCache")
        if int(cache.offset) < int(base.length):
            raise ValueError("source QSA cache is shorter than the shared base")
        if cache.keys is None or cache.values is None or cache.index_keys is None:
            raise ValueError("cannot split an unpopulated QSA cache")
        row = cls(base, note=note)
        suffix = int(cache.offset) - int(base.length)
        if suffix:
            row.append_index_keys(cache.index_keys[:, base.length : cache.offset])
            row.append_kv(
                cache.keys[..., base.length : cache.offset, :],
                cache.values[..., base.length : cache.offset, :],
            )
        return row

    def _bump(self, key: str, amount: int = 1) -> None:
        if self._note is not None:
            self._note(key, int(amount))

    @property
    def offset(self) -> int:
        return int(self.base.length + self._kv.offset)

    @property
    def suffix_length(self) -> int:
        return int(self._kv.offset)

    def size(self) -> int:
        return self.offset

    def start_speculation(self, rollback_window=None) -> None:
        del rollback_window

    def stop_speculation(self) -> None:
        self.release_qsa_cycle("SharedSuffixQSAKVCache.stop_speculation")

    def is_trimmable(self) -> bool:
        return True

    def empty(self) -> bool:
        return False

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        del return_array
        return create_causal_mask(N, offset=self.offset, **kwargs)

    def release_qsa_cycle(self, who: str, *, keep_pooled: bool = True) -> None:
        del keep_pooled
        self._mtp_share_topk = False
        self._mtp_shared_topk = None
        self._mtp_shared_topk_n_blocks = None
        ledger = 0 if self.index_keys is None else int(self.index_keys.shape[1])
        if ledger < self._kv.offset:
            raise SharedSuffixQSAError(
                f"{who}: the suffix raw-key ledger holds {ledger} positions but suffix K/V holds {self._kv.offset}"
            )
        if ledger > self._kv.offset:
            self.index_keys = mx.contiguous(self.index_keys[:, : self._kv.offset])

    def append_index_keys(self, keys: mx.array) -> mx.array:
        """Append only to the private ledger; never form ``[base, suffix]``."""
        if keys.ndim != 3 or int(keys.shape[0]) != 1:
            raise ValueError("QSA suffix raw keys must have shape [1, M, D]")
        if int(keys.shape[2]) != int(self.base.index_keys.shape[2]):
            raise ValueError("QSA suffix raw-key layout differs from the base")
        self.index_keys = (
            keys
            if self.index_keys is None
            else mx.concatenate([self.index_keys[:, : self._kv.offset], keys], axis=1)
        )
        return self.index_keys

    def append_kv(
        self, keys: mx.array, values: mx.array, *, allow_unledgered: bool = False
    ) -> tuple[mx.array, mx.array]:
        """Append to physical suffix slabs and return only the live suffix."""
        if keys.ndim != 4 or values.ndim != 4 or int(keys.shape[0]) != 1:
            raise ValueError("QSA suffix K/V must be rank four with batch one")
        if keys.shape[:3] != values.shape[:3]:
            raise ValueError("QSA suffix K/V geometries differ")
        if keys.shape[1:] != (
            self.base.keys.shape[1],
            keys.shape[2],
            self.base.keys.shape[3],
        ):
            raise ValueError("QSA suffix key layout differs from the base")
        if values.shape[1:] != (
            self.base.values.shape[1],
            values.shape[2],
            self.base.values.shape[3],
        ):
            raise ValueError("QSA suffix value layout differs from the base")
        ledger_width = 0 if self.index_keys is None else int(self.index_keys.shape[1])
        expected = self._kv.offset + int(keys.shape[2])
        if ledger_width != expected:
            if not (allow_unledgered and ledger_width == self._kv.offset):
                raise SharedSuffixQSAError(
                    "QSA suffix raw-index ledger and K/V append disagree"
                )
        return self._kv.update_and_fetch(keys, values)

    def update_index_keys(self, keys: mx.array):
        raise SharedSuffixQSAError("shared-suffix QSA requires the split-aware indexer")

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        del keys, values
        raise SharedSuffixQSAError(
            "shared-suffix QSA requires the two-source attention consumer"
        )

    def suffix_keys_and_values(self) -> tuple[mx.array | None, mx.array | None]:
        if self._kv.keys is None:
            return (None, None)
        return self._kv.keys_and_values()

    def set_suffix_pooled_keys(self, pooled: mx.array | None) -> None:
        """Install contiguous pooled blocks immediately after base coverage."""
        if pooled is not None:
            if pooled.ndim != 3 or int(pooled.shape[0]) != 1:
                raise ValueError("suffix pooled keys must have shape [1, N, D]")
            if self.base.pooled_keys is None or self.base.pooled_ratio is None:
                raise SharedSuffixQSAError(
                    "suffix pooled keys require a contiguous pooled base"
                )
            if int(pooled.shape[2]) != int(self.base.pooled_keys.shape[2]):
                raise ValueError("suffix pooled-key layout differs from the base")
            ledger = 0 if self.index_keys is None else int(self.index_keys.shape[1])
            logical = int(self.base.length) + max(int(self._kv.offset), ledger)
            maximum = logical // int(self.base.pooled_ratio)
            complete = _coverage(self.base.summary_identity) + int(pooled.shape[1])
            if complete > maximum:
                raise ValueError("suffix pooled coverage exceeds the live cache")
        self._suffix_pooled_keys = pooled

    def trim(self, count: int) -> int:
        """Trim exactly inside the private suffix; the base is never mutable."""
        count = int(count)
        if count < 0:
            raise ValueError("QSA suffix trim cannot be negative")
        if count > self._kv.offset:
            raise SharedSuffixQSAError("QSA suffix trim would cross the shared base")
        self._kv.trim(count)
        if self.index_keys is not None:
            self.index_keys = mx.contiguous(self.index_keys[:, : self._kv.offset])
        if self._suffix_pooled_keys is not None:
            ratio = int(self.base.pooled_ratio)
            keep_total = self.offset // ratio
            keep_suffix = max(0, keep_total - _coverage(self.base.summary_identity))
            self._suffix_pooled_keys = mx.contiguous(
                self._suffix_pooled_keys[:, :keep_suffix]
            )
        return count

    def materialize_to_qsa(self) -> tuple[QSAKVCache, QSAMaterializationReceipt]:
        """Cross the stock-consumer boundary with one explicit full-row copy."""
        (suffix_k, suffix_v) = self.suffix_keys_and_values()
        if suffix_k is None:
            (keys, values) = (self.base.keys, self.base.values)
        else:
            keys = mx.concatenate([self.base.keys, suffix_k], axis=2)
            values = mx.concatenate([self.base.values, suffix_v], axis=2)
        ledger_width = 0 if self.index_keys is None else int(self.index_keys.shape[1])
        if ledger_width != self._kv.offset:
            raise SharedSuffixQSAError(
                "QSA suffix raw-index ledger and K/V width disagree"
            )
        if self.index_keys is None:
            index_keys = self.base.index_keys
        else:
            index_keys = mx.concatenate([self.base.index_keys, self.index_keys], axis=1)
        cache = QSAKVCache(self.base.summary_identity)
        cache.keys = keys
        cache.values = values
        cache.index_keys = index_keys
        cache.offset = self.offset
        pooled = self.base.pooled_keys
        if pooled is not None and self._suffix_pooled_keys is not None:
            pooled = mx.concatenate([pooled, self._suffix_pooled_keys], axis=1)
        cache._qsa_pooled_keys = pooled
        cache._qsa_pooled_ratio = self.base.pooled_ratio if pooled is not None else None
        cache._qsa_summary_identity = _with_coverage(
            self.base.summary_identity, 0 if pooled is None else int(pooled.shape[1])
        )
        materialized_bytes = int(keys.nbytes + values.nbytes + index_keys.nbytes)
        if pooled is not None:
            materialized_bytes += int(pooled.nbytes)
        self._bump("shared_qsa_materializations")
        self._bump("shared_qsa_materialized_bytes", materialized_bytes)
        return (
            cache,
            QSAMaterializationReceipt(
                layout_id=self.base.layout_id,
                base_length=self.base.length,
                suffix_length=self.suffix_length,
                materialized_bytes=materialized_bytes,
            ),
        )

    @property
    def state(self):
        return (
            self.base.keys,
            self.base.values,
            self.base.index_keys,
            self._kv.keys,
            self._kv.values,
            int(self._kv.offset),
            self.index_keys,
            self.base.pooled_keys,
            self._suffix_pooled_keys,
        )

    @property
    def nbytes(self) -> int:
        return int(self.base.nbytes + self.private_nbytes)

    @property
    def private_nbytes(self) -> int:
        total = int(self._kv.nbytes)
        if self.index_keys is not None:
            total += int(self.index_keys.nbytes)
        if self._suffix_pooled_keys is not None:
            total += int(self._suffix_pooled_keys.nbytes)
        return total


def split_attested_qsa_rows(
    caches, *, layout_id: str, note: Callable[[str, int], None] | None = None
) -> list[SharedSuffixQSAKVCache]:
    """Replace equal-offset attested stock rows with one aligned base.

    Equality of the prefix contents is intentionally not inferred here: the
    caller must hold the live-tip host attestation that established it.
    Geometry and summary provenance are still checked before any row changes.
    """
    caches = list(caches)
    if len(caches) < 2 or not all((type(cache) is QSAKVCache for cache in caches)):
        raise TypeError("shared QSA splitting requires at least two stock rows")
    offsets = {int(cache.offset) for cache in caches}
    if len(offsets) != 1:
        raise ValueError("attested QSA rows have unequal offsets")
    stop = offsets.pop()
    base_length = stop // 4 * 4
    if base_length <= 0:
        raise ValueError("attested QSA rows have no aligned shared base")
    first = caches[0]
    geometry = (
        tuple(first.keys.shape[1:]) if first.keys is not None else None,
        tuple(first.values.shape[1:]) if first.values is not None else None,
        tuple(first.index_keys.shape[2:]) if first.index_keys is not None else None,
        str(first.keys.dtype) if first.keys is not None else None,
        str(first.values.dtype) if first.values is not None else None,
        str(first.index_keys.dtype) if first.index_keys is not None else None,
    )
    for cache in caches[1:]:
        candidate = (
            tuple(cache.keys.shape[1:]) if cache.keys is not None else None,
            tuple(cache.values.shape[1:]) if cache.values is not None else None,
            tuple(cache.index_keys.shape[2:]) if cache.index_keys is not None else None,
            str(cache.keys.dtype) if cache.keys is not None else None,
            str(cache.values.dtype) if cache.values is not None else None,
            str(cache.index_keys.dtype) if cache.index_keys is not None else None,
        )
        if candidate != geometry:
            raise ValueError("attested QSA row layouts differ")
        if getattr(cache, "_qsa_summary_identity", None) != getattr(
            first, "_qsa_summary_identity", None
        ):
            raise ValueError("attested QSA summary identities differ")
    base = QSAImmutableBase.from_cache(first, layout_id=layout_id, length=base_length)
    expected_blocks = base_length // 4
    if (
        base.pooled_keys is None
        or int(base.pooled_ratio or 0) != 4
        or int(base.pooled_keys.shape[1]) != expected_blocks
    ):
        raise ValueError(
            "shared QSA splitting requires complete 4-token pooled summaries"
        )
    rows = [
        SharedSuffixQSAKVCache.from_cache(cache, base, note=note) for cache in caches
    ]
    if note is not None:
        note("shared_qsa_rows", len(rows))
        note("shared_qsa_base_bytes", base.nbytes)
        note("shared_qsa_private_bytes", sum((row.private_nbytes for row in rows)))
    return rows
