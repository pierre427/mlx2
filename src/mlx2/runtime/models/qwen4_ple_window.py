"""Bounded device-ID PLE gather without joining or staging whole shards.

Original mlx2 composition, inspired by oMLX #4880. Full advertised arrays
remain bound to ordinary gather ops; no borrowed undersized pointer views.
"""
from collections import Counter
from weakref import ref

import mlx.core as mx
from mlx import nn

MAX_SHARDS = 9
MAX_ROWS = 64
_COUNTS = Counter()


def _resident_parts(table):
    layout = [table.vocab_size, table.dims, table.num_shards, table.rows_per_shard]
    sources = [table]
    for index in range(table.num_shards):
        shard = getattr(table, f"shard_{index}")
        if type(shard) not in (nn.Embedding, nn.QuantizedEmbedding):
            raise ValueError("unsupported resident PLE shard")
        layout.append((type(shard), getattr(shard, "bits", None),
                       getattr(shard, "group_size", None), getattr(shard, "mode", None)))
        sources.extend((shard, shard.weight))
        if type(shard) is nn.QuantizedEmbedding:
            sources.extend(value for name in ("scales", "biases")
                           if (value := getattr(shard, name, None)) is not None)
    return tuple(layout), tuple(sources)


class ResidentBinding:
    """Identity seal for an independently prepared, immutable shard table.

    Weak references neither keep replaced table weights resident nor let a
    recycled Python identity inherit the old residency assertion. This does
    not measure residency. Live in-place weight mutation is unsupported.
    """

    def __init__(self, table):
        self.layout, sources = _resident_parts(table)
        self.sources = tuple(ref(value) for value in sources)

    def matches(self, table):
        try:
            layout, sources = _resident_parts(table)
        except (AttributeError, ValueError):
            return False
        return (layout == self.layout and len(sources) == len(self.sources)
                and all(old() is new for old, new in zip(self.sources, sources)))


def plan_windows(vocab_size, num_shards, ranges):
    if (type(vocab_size) is not int or type(num_shards) is not int
            or num_shards < 1 or vocab_size < 1 or vocab_size % num_shards):
        raise ValueError("PLE requires uniform nonempty shards")
    size = vocab_size // num_shards
    windows = []
    for low, high in ranges:
        if (type(low) is not int or type(high) is not int
                or not 0 <= low < high <= vocab_size):
            raise ValueError("PLE head range outside the table")
        first, stop = low // size, (high - 1) // size + 1
        if stop - first > MAX_SHARDS:
            raise ValueError("PLE head exceeds bounded shard window")
        windows.append((first, stop))
    if not windows:
        raise ValueError("PLE needs head ranges")
    return tuple(windows)


def eligible(table, shape, ranges, *, metal=True):
    if not shape or shape[-1] != len(ranges):
        return False
    rows = 1
    for dim in shape[:-1]:
        rows *= dim
    if not 1 <= rows <= MAX_ROWS:
        return False
    if metal and (mx.default_device() != mx.gpu or not mx.metal.is_available()):
        return False
    try:
        plan_windows(table.vocab_size, table.num_shards, ranges)
    except ValueError:
        return False
    return True


def lookup(table, indices, ranges, *, metal=True):
    """Caller proves each head's IDs lie in its immutable hash range.

    This function is private to the n-gram producer, not an arbitrary indexed
    embedding API. The caller must establish shard residency independently.
    This helper does not explicitly join or stage tables, but ordinary gather
    ops bind full arrays and framework residency accounting still applies.
    """
    if not eligible(table, indices.shape, ranges, metal=metal):
        _COUNTS["declined"] += 1
        return None
    if indices.dtype not in (mx.int32, mx.int64):
        raise ValueError("PLE device IDs must be signed integers")
    windows = plan_windows(table.vocab_size, table.num_shards, ranges)
    pieces = []
    for head, (first, stop) in enumerate(windows):
        ids = indices[..., head]
        result = None
        for shard_id in range(first, stop):
            offset = shard_id * table.rows_per_shard
            local = mx.clip(ids - offset, 0, table.rows_per_shard - 1)
            values = getattr(table, f"shard_{shard_id}")(local)
            active = ((ids >= offset) & (ids < offset + table.rows_per_shard))[..., None]
            result = mx.where(active, values, mx.zeros_like(values) if result is None else result)
        pieces.append(result)
    _COUNTS["dispatches"] += 1
    _COUNTS["shard_gathers"] += sum(stop - first for first, stop in windows)
    return mx.stack(pieces, axis=-2)


def status():
    return {"schema": "mlx2.ple-windowed-gather.v1", "qualified": False,
            "max_shards_per_head": MAX_SHARDS, "max_rows": MAX_ROWS,
            "counts": dict(_COUNTS)}
