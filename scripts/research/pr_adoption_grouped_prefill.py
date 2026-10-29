"""Evidence gate for layer-major streamed-prefill research.

Plans an idealized IO ceiling only. Does not execute a grouped forward or
publish KV/APCv2 state. Inputs must be per-chunk observed expert reads.
"""

import re


def assess(chunks, *, extra_activation_bytes, free_bytes, resident_experts=False):
    if (type(extra_activation_bytes) is not int or extra_activation_bytes < 0
            or type(free_bytes) is not int or free_bytes < 0
            or type(resident_experts) is not bool):
        raise ValueError("bounded owned memory accounting required")
    if not isinstance(chunks, list) or not 2 <= len(chunks) <= 8:
        raise ValueError("two to eight observed plain prefill chunks required")
    binding = None
    frontier = None
    reads = []
    chunk_uniques = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("observed chunk object required")  # noqa: TRY004
        current = (chunk.get("checkpoint"), chunk.get("source_revision"),
                   chunk.get("expert_cache_bytes"))
        if (not all(isinstance(x, str) and x for x in current[:2])
                or "@" not in current[0] or not all(current[0].split("@", 1))
                or current[0].startswith("*@") or re.fullmatch("[0-9a-f]{40}", current[1]) is None
                or type(current[2]) is not int or current[2] < 0):
            raise ValueError("checkpoint/source/cache binding required")
        if binding is not None and binding != current:
            raise ValueError("different grouped-prefill source or cache profile")
        binding = current
        start, rows = chunk.get("start"), chunk.get("rows")
        if (type(start) is not int or start < 0 or type(rows) is not int or rows < 1
                or (frontier is not None and frontier != start)):
            raise ValueError("chunks must form a contiguous prefill span")
        frontier = start + rows
        if any(chunk.get(flag) is not False for flag in
               ("samples", "image_rows", "grammar_rows", "interior_checkpoint")):
            return {"admitted": False, "reason": "non-plain chunk boundary",
                    "selected": False, "qualified": False}
        entries = chunk.get("expert_reads")
        if not isinstance(entries, list):
            raise ValueError("per-chunk observed expert reads required")  # noqa: TRY004
        distinct = {}
        for entry in entries:
            if (not isinstance(entry, dict)
                    or any(type(entry.get(k)) is not int or entry[k] < 0
                           for k in ("layer", "expert", "bytes"))
                    or entry["bytes"] == 0):
                raise ValueError("invalid observed expert read")
            reads.append(entry)
            distinct[(entry["layer"], entry["expert"])] = entry["bytes"]
        chunk_uniques.append(distinct)
    if resident_experts or not reads:
        return {"admitted": False, "reason": "no observed expert reread opportunity",
                "selected": False, "qualified": False}
    unique = {}
    for entry in reads:
        key = (entry["layer"], entry["expert"])
        if key in unique and unique[key] != entry["bytes"]:
            raise ValueError("expert byte identity changed")
        unique[key] = entry["bytes"]
    current_bytes, ideal_bytes = sum(x["bytes"] for x in reads), sum(unique.values())
    # Existing within-forward expert batching is a different mechanism. Only
    # overlap between chunks can justify grouping those chunks layer-major.
    per_chunk_bytes = sum(sum(chunk.values()) for chunk in chunk_uniques)
    saved = per_chunk_bytes - ideal_bytes
    layer_bytes = {}
    for (layer, _), size in unique.items():
        layer_bytes[layer] = layer_bytes.get(layer, 0) + size
    concurrent_bytes = max(layer_bytes.values())
    # Pinned experts and new activations must fit simultaneously; using the
    # original cache budget after spending it on group slots is not valid.
    fits = extra_activation_bytes <= free_bytes and concurrent_bytes <= binding[2]
    return {"admitted": bool(saved > 0 and fits),
            "reason": "bounded research opportunity" if saved > 0 and fits else
                      "no repeated reads or insufficient group memory",
            "observed_read_bytes": current_bytes, "ideal_union_read_bytes": ideal_bytes,
            "ideal_saved_bytes": saved, "extra_activation_bytes": extra_activation_bytes,
            "within_chunk_repeat_bytes": current_bytes - per_chunk_bytes,
            "max_layer_pinned_bytes": concurrent_bytes,
            "selected": False, "qualified": False,
            "limit": "ideal IO ceiling; no grouped execution or speed evidence"}
