"""Which PLE n-gram table tensors a Flash-Next route consumes (MLX-free)."""

from __future__ import annotations

import re

_TABLE_PARTS = ("weight", "scales", "biases")


def check_ngram_table_tensors(keys, *, prefix=None, num_shards=None) -> None:
    """Refuse n-gram table tensors a Flash-Next route would not consume.

    Both routes read each shard's 4-bit affine ``weight``, ``scales`` and
    ``biases`` and nothing else: the file-backed table serves rows built from
    exactly those, and its loader discards every ``.shard_`` tensor before
    the strict load.  A table tensor outside that set can still carry math:
    oQ conversions of the FP8 checkpoint keep the table's ``weight_scale``,
    which multiplies every looked-up row (TensorFold#774), and an unquantized
    table (omlx#4476) stores 16-bit rows with no scales.  Both refuse here by
    name instead of serving unscaled or misread rows.

    ``keys`` are checkpoint tensor names (an index's weight map).  With
    ``prefix`` and ``num_shards`` (a sidecar manifest), every table tensor
    must sit under that prefix at a shard index below ``num_shards``.  Shards
    that are absent altogether are left to the spot check.
    """
    shard_part = re.compile(r"\.shard_(\d+)\.([A-Za-z_]+)$")
    table = sorted(
        key
        for key in keys
        if ".ngram_embedding." in key or (prefix is not None and key.startswith(prefix + "."))
    )
    unexpected = []
    parts: dict[tuple[str, int], set] = {}
    for key in table:
        match = shard_part.search(key)
        if (
            match is None
            or match.group(2) not in _TABLE_PARTS
            or (prefix is not None and key[: match.start()] != prefix)
            or (num_shards is not None and int(match.group(1)) >= num_shards)
        ):
            unexpected.append(key)
            continue
        parts.setdefault((key[: match.start()], int(match.group(1))), set()).add(match.group(2))
    scales = [key for key in unexpected if key.rsplit(".", 1)[-1] == "weight_scale"]
    if scales:
        raise ValueError(
            f"the PLE n-gram table stores a weight_scale ({', '.join(scales[:3])}) that must "
            "multiply every looked-up row (an oQ conversion of the FP8 checkpoint); no "
            "Flash-Next route applies it, so it would serve unscaled rows; the artifact "
            "is refused"
        )
    if unexpected:
        raise ValueError(
            f"{len(unexpected)} PLE n-gram table tensor(s) no Flash-Next route "
            f"consumes: {', '.join(unexpected[:4])}; it reads only "
            f"<prefix>.shard_<i>.{{weight,scales,biases}}"
            + (f" under {prefix} for shards 0..{num_shards - 1}" if prefix is not None and num_shards else "")
        )
    incomplete = sorted(
        f"{table_prefix}.shard_{index} has {sorted(found)}"
        for (table_prefix, index), found in parts.items()
        if found != set(_TABLE_PARTS)
    )
    if incomplete:
        raise ValueError(
            "PLE n-gram table shards are not 4-bit affine rows (weight, scales, biases): "
            + "; ".join(incomplete[:3])
            + "; an unquantized or differently packed table is not supported"
        )
