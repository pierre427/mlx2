"""ENS H54 connector: a system of record -> (entity, attribute, value) records, and an entity index for mentions.

Pierre (2026-11-24): "a tile that acts as an interface between a system of record (markdown, psql db, etc) like
system-of-record <-> record port tile <-> bus <-> kimi". This module is the deterministic, CPU-only first stage:
it reads a source, never writes to it, and stamps every record with the revision it was read at, so a receipt names
exactly what the port saw. Aggregates and set queries ("which customers churned") are out of scope (SQL or tool
calls); writes back to a system of record go through tool calls, never the bus.

Sources:
  markdown  frontmatter (`---` / `key: value` lines / `---`): the file is one entity, named by `name` or `title`, else
            the file stem; pipe tables: the first column names the entity, the header row names the attributes;
            heading sections: a `#`-heading names an entity and the `- attribute: value` bullets under it are its
            records. The revision is the file's SHA-256.
  sqlite    one table and its key column; every other non-NULL column is an attribute. The revision of a row is a
            hash of its values (column order fixed by the table), so a changed row changes its revision.

EntityIndex finds mentions in a prompt: case-folded exact names and aliases, longest match first, left to right,
whole words only (a name never matches inside a longer word).
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Record:
    entity: str
    attribute: str
    value: str
    source: str          # e.g. "md:people.md#table" or "sqlite:crm.db/customers"
    revision: str        # sha256 hex (file) or row hash

    def as_text(self) -> str:
        """The record as one prompt line (the pasted-text control arm uses exactly this)."""
        return f"{self.entity} | {self.attribute}: {self.value}"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _clean(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1]
    return s.strip()


# ------------------------------------------------------------------ markdown
_FM_LINE = re.compile(r"^([A-Za-z_][\w \-]*?)\s*:\s*(.*)$")
_BULLET = re.compile(r"^\s*[-*+]\s+([^:]+?)\s*:\s*(.+?)\s*$")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+#+)?\s*$")        # a closing #-run only after a space ("C#" stays)
_SEP_CELL = re.compile(r"^:?-{3,}:?$")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


class MarkdownError(ValueError):
    """The file is not in a shape this reader can read without guessing (fail closed)."""


def _label(s: str) -> str:
    """An attribute or header name: emphasis and code marks removed, whitespace folded."""
    s = re.sub(r"[*`]+", "", s)
    s = re.sub(r"^__(.+)__$", r"\1", s.strip())          # __bold__; inner underscores (start_date) stay
    return " ".join(s.split())


def _table_cells(line: str) -> list[str] | None:
    s = line.strip()
    if not (s.startswith("|") and s.endswith("|") and len(s) > 1):
        return None
    cells = re.split(r"(?<!\\)\|", s[1:-1])                   # an escaped \| is cell text, not a separator
    return [c.strip().replace("\\|", "|") for c in cells]


def read_markdown(path: str | Path) -> list[Record]:
    """Frontmatter, pipe tables (with outer pipes and a --- separator row) and heading sections. Fenced code
    blocks are skipped. Raises MarkdownError on unclosed frontmatter, a table row with the wrong number of cells, or
    two different values for one (entity, attribute) in the file; it never guesses."""
    p = Path(path)
    raw = p.read_bytes()
    rev = _sha(raw)
    lines = raw.decode("utf-8-sig").splitlines()                  # utf-8-sig drops a byte-order mark
    out: list[Record] = []
    seen: dict[tuple, str] = {}

    def add(ent, attr, val, kind):
        k = (ent, attr)
        if k in seen and seen[k] != val:
            raise MarkdownError(f"{p.name}: {ent!r} has two values for {attr!r}")
        if k not in seen:
            seen[k] = val
            out.append(Record(ent, attr, val, f"md:{p.name}#{kind}", rev))

    i, doc_entity = 0, None
    if lines and lines[0].strip() == "---":
        end = next((k for k in range(1, len(lines)) if lines[k].strip() == "---"), None)
        if end is None:
            raise MarkdownError(f"{p.name}: frontmatter has no closing ---")
        kv = []
        for ln in lines[1:end]:
            m = _FM_LINE.match(ln)
            if m and _clean(m.group(2)):
                kv.append((_label(m.group(1)), _clean(m.group(2))))
        doc_entity = next((v for k, v in kv if k.lower() in ("name", "title")), p.stem)
        for k, v in kv:
            if k.lower() not in ("name", "title"):
                add(doc_entity, k, v, "frontmatter")
        i = end + 1
    entity, fence = doc_entity, None                 # bullets before any heading belong to the frontmatter entity
    while i < len(lines):
        ln = lines[i]
        f = _FENCE.match(ln)
        if fence is None and f:
            fence = (f.group(1)[0], len(f.group(1)))   # a fence closes only with the same marker, at least as long
            i += 1
            continue
        if fence is not None:
            if f and f.group(1)[0] == fence[0] and len(f.group(1)) >= fence[1] and not ln.strip()[len(f.group(1)):].strip():
                fence = None
            i += 1
            continue
        cells = _table_cells(ln)
        nxt = _table_cells(lines[i + 1]) if i + 1 < len(lines) else None
        if cells and nxt and len(nxt) == len(cells) and all(_SEP_CELL.match(c) for c in nxt):
            header = [_label(c) for c in cells]
            i += 2
            while i < len(lines) and (row := _table_cells(lines[i])) is not None:
                if len(row) != len(header):
                    raise MarkdownError(f"{p.name}:{i + 1}: table row has {len(row)} cells, header has {len(header)}")
                ent = _clean(row[0])
                if ent:
                    for c in range(1, len(header)):
                        if _clean(row[c]):
                            add(ent, header[c], _clean(row[c]), "table")
                i += 1
            continue                                 # bullets after a table stay with the current heading
        h = _HEADING.match(ln)
        if h:
            entity = _clean(h.group(2))
        else:
            b = _BULLET.match(ln)
            if b and entity and not b.group(2).startswith("//"):     # "- https://x" is a link, not a record
                add(entity, _label(b.group(1)), _clean(b.group(2)), "section")
        i += 1
    if fence is not None:
        raise MarkdownError(f"{p.name}: code fence is never closed")
    return out


# ------------------------------------------------------------------ sqlite
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _q(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def read_sqlite(db: str | Path, table: str, key: str) -> list[Record]:
    """Rows of `table` keyed by column `key` (matched case-insensitively, as SQLite does). Read-only connection;
    the table and key must be plain identifiers; column names from the database are quoted. Raises ValueError on
    a duplicate key value (two rows would merge into one entity)."""
    if not (_IDENT.match(table) and _IDENT.match(key)):
        raise ValueError("table and key must be plain SQL identifiers")
    con = sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({_q(table)})")]
        kc = next((c for c in cols if c.lower() == key.lower()), None)
        if kc is None:
            raise ValueError(f"{table} has no column {key}")
        out: list[Record] = []
        keys: set[str] = set()
        for row in con.execute(f"SELECT {', '.join(_q(c) for c in cols)} FROM {_q(table)} ORDER BY {_q(kc)}"):
            vals = dict(zip(cols, row))
            if vals[kc] is None:
                continue
            ent = str(vals[kc])
            if ent in keys:
                raise ValueError(f"{table}.{kc} has duplicate key {ent!r}")
            keys.add(ent)
            rev = _sha(repr([(c, vals[c]) for c in cols]).encode())[:16]
            out += [Record(ent, c, str(vals[c]), f"sqlite:{Path(db).name}/{table}", rev)
                    for c in cols if c != kc and vals[c] is not None and str(vals[c]).strip()]
        return out
    finally:
        con.close()


# ------------------------------------------------------------------ entity index
def _fold(s: str) -> str:
    return " ".join(s.casefold().split())


@dataclass
class Mention:
    entity: str
    start: int           # char offsets in the original prompt, [start, end)
    end: int


@dataclass
class EntityIndex:
    names: dict[str, str] = field(default_factory=dict)        # folded surface form -> canonical entity
    surfaces: dict[str, str] = field(default_factory=dict)     # folded form -> an original spelling (for the regex)
    _rx: object = None

    @classmethod
    def build(cls, records: Iterable[Record], aliases: dict[str, Iterable[str]] | None = None) -> "EntityIndex":
        idx = cls()
        for r in records:
            idx.add(r.entity, r.entity)
        for ent, al in (aliases or {}).items():
            for a in al:
                idx.add(a, ent)
        return idx

    def add(self, surface: str, entity: str):
        f = _fold(surface)
        if not f:
            return
        prev = self.names.get(f)
        if prev is not None and prev != entity:
            raise ValueError(f"ambiguous name {surface!r}: {prev!r} and {entity!r}")
        self.names[f] = entity
        self.surfaces.setdefault(f, " ".join(surface.split()))
        self._rx = None

    def _regex(self):
        if self._rx is None:
            forms = sorted(self.surfaces.values(), key=len, reverse=True)       # longest first, original spelling
            alt = "|".join(r"\s+".join(re.escape(w) for w in f.split(" ")) for f in forms)
            self._rx = re.compile(r"(?<!\w)(?:" + alt + r")(?!\w)", re.IGNORECASE) if forms else None
        return self._rx

    def find(self, text: str) -> list[Mention]:
        """Longest match first, left to right, whole words, no overlaps. Matching runs on the original text
        (case-insensitive), so spans index the prompt exactly."""
        rx = self._regex()
        if rx is None:
            return []
        out = []
        for m in rx.finditer(text):
            ent = self.names.get(_fold(m.group(0)))
            if ent is not None:                  # re's case-insensitivity can differ from casefold on rare letters
                out.append(Mention(ent, m.start(), m.end()))
        return out


def records_for(records: Iterable[Record], entities: Iterable[str]) -> dict[str, list[Record]]:
    want = list(dict.fromkeys(entities))
    by = {e: [] for e in want}
    for r in records:
        if r.entity in by:
            by[r.entity].append(r)
    return by


def receipt(records: Iterable[Record]) -> dict:
    """What a route receipt records about the records a request used: sources, revisions, counts. No values."""
    rs = list(records)
    return {"n_records": len(rs), "entities": len({r.entity for r in rs}),
            "sources": sorted({r.source for r in rs}),
            "revisions": sorted({f"{r.source}@{r.revision}" for r in rs}),
            "digest": _sha("\n".join(sorted(f"{r.source}@{r.revision}|{r.entity}|{r.attribute}"
                                            for r in rs)).encode())[:16]}
