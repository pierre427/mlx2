"""Every receipt field on the speculative stats objects is written somewhere.

On 2026-09-30 thirteen fields (rate gate, depth router, external-cache
reconciliation, retrieval corpus) were declared, serialised into 9,046 lab
receipts at their defaults, and assigned by no line of product code.  A
receipt field nothing writes tells a reader the feature exists.
"""

import dataclasses
import re
from pathlib import Path

from mlx2.runtime import hybrid_speculative, prompt_lookup

SRC = Path(hybrid_speculative.__file__).parents[1]


def _written_names():
    names = set()
    for path in SRC.rglob("*.py"):
        text = path.read_text()
        # ``x.name =``, ``x.name +=``, ``name=`` in a constructor/replace call,
        # and ``setattr(x, "name"`` all count as writes.
        for m in re.finditer(r"\.(\w+)\s*(?:=|\+=|-=)(?!=)", text):
            names.add(m.group(1))
        for m in re.finditer(r"[(,]\s*(\w+)\s*=(?!=)", text):
            names.add(m.group(1))
        for m in re.finditer(r'setattr\(\s*\w+\s*,\s*"(\w+)"', text):
            names.add(m.group(1))
        # container fields are written in place: ``x.hist[k] = ``,
        # ``x.hist.update(...)`` and the like.
        for m in re.finditer(r"\.(\w+)\s*\[[^\]]*\]\s*(?:=|\+=)(?!=)", text):
            names.add(m.group(1))
        for m in re.finditer(r"\.(\w+)\.(?:update|setdefault|append|extend|pop|clear)\(", text):
            names.add(m.group(1))
    return names


def test_every_stats_field_has_a_writer():
    written = _written_names()
    unwritten = []
    for cls in (prompt_lookup.HybridStats, hybrid_speculative.HybridStats):
        for field in dataclasses.fields(cls):
            if field.name not in written:
                unwritten.append(f"{cls.__module__}.{cls.__name__}.{field.name}")
    assert not unwritten, f"receipt fields nothing writes: {unwritten}"
