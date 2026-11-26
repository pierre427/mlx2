"""H54 connector and random-schema mock provider (CPU only; no model, no GPU)."""
from __future__ import annotations

import random
import sqlite3

import pytest

from research.ens import sor, sor_mock as M

MD = """---
name: Orvane Ltd
tier: gold
renewal date: 2027-03-14
---

# Accounts

| account | manager | balance |
|---|---|---|
| Kelbrook | Ada Strand | 48,215 |
| Mirefold | Jun Okafor |  |

## Tessaly Group
- region: North-East
- contract value: 1,200,000

## Notes
Free text: not a record.
"""


def test_markdown_shapes(tmp_path):
    p = tmp_path / "crm.md"
    p.write_text(MD)
    rs = sor.read_markdown(p)
    got = {(r.entity, r.attribute, r.value) for r in rs}
    assert ("Orvane Ltd", "tier", "gold") in got
    assert ("Orvane Ltd", "renewal date", "2027-03-14") in got
    assert ("Kelbrook", "manager", "Ada Strand") in got and ("Kelbrook", "balance", "48,215") in got
    assert ("Mirefold", "manager", "Jun Okafor") in got
    assert not any(r.entity == "Mirefold" and r.attribute == "balance" for r in rs)       # empty cell skipped
    assert ("Tessaly Group", "region", "North-East") in got
    assert not any(r.entity == "Notes" for r in rs)
    assert not any(r.attribute.lower() == "name" for r in rs)
    revs = {r.revision for r in rs}
    assert len(revs) == 1 and len(revs.pop()) == 64
    p.write_text(MD.replace("gold", "silver"))
    assert {r.revision for r in sor.read_markdown(p)} != {r.revision for r in rs}       # edit -> new revision


def test_sqlite_rows_and_revisions(tmp_path):
    db = tmp_path / "crm.db"
    con = sqlite3.connect(db)
    con.execute("create table customers (name text, plan text, mrr integer, churned text)")
    con.executemany("insert into customers values (?,?,?,?)",
                    [("Kelbrook", "pro", 4200, None), ("Mirefold", "basic", 99, "2026-10-01"), (None, "x", 1, None)])
    con.commit()
    rs = sor.read_sqlite(db, "customers", "name")
    got = {(r.entity, r.attribute, r.value) for r in rs}
    assert got == {("Kelbrook", "plan", "pro"), ("Kelbrook", "mrr", "4200"), ("Mirefold", "plan", "basic"),
                   ("Mirefold", "mrr", "99"), ("Mirefold", "churned", "2026-10-01")}
    before = {r.entity: r.revision for r in rs}
    con.execute("update customers set mrr = 4300 where name = 'Kelbrook'")
    con.commit()
    con.close()
    after = {r.entity: r.revision for r in sor.read_sqlite(db, "customers", "name")}
    assert after["Kelbrook"] != before["Kelbrook"] and after["Mirefold"] == before["Mirefold"]
    with pytest.raises(ValueError):
        sor.read_sqlite(db, "customers; drop table customers", "name")
    with pytest.raises(ValueError):
        sor.read_sqlite(db, "customers", "nope")


def test_sqlite_is_read_only(tmp_path):
    db = tmp_path / "x.db"
    con = sqlite3.connect(db)
    con.execute("create table t (k text, v text)")
    con.execute("insert into t values ('a', 'b')")
    con.commit()
    con.close()
    before = db.read_bytes()
    sor.read_sqlite(db, "t", "k")
    assert db.read_bytes() == before


def test_entity_index_longest_match_and_boundaries():
    rs = [sor.Record("Ada", "x", "1", "s", "r"), sor.Record("Ada Strand", "x", "2", "s", "r"),
          sor.Record("Kel", "x", "3", "s", "r")]
    idx = sor.EntityIndex.build(rs, aliases={"Ada Strand": ["A. Strand"]})
    ms = idx.find("Ask ada  strand about Kelbrook, then a. strand and Kel.")
    assert [(m.entity) for m in ms] == ["Ada Strand", "Ada Strand", "Kel"]
    t = "Ask ada  strand about Kelbrook, then a. strand and Kel."
    assert t[ms[0].start:ms[0].end] == "ada  strand"
    assert "Kelbrook" not in [t[m.start:m.end] for m in ms]                               # no match inside a word
    with pytest.raises(ValueError):
        sor.EntityIndex.build(rs, aliases={"Kel": ["Ada"]})                                # ambiguous alias


def test_receipt_has_no_values():
    rs = [sor.Record("Kelbrook", "balance", "48,215", "md:crm.md#table", "ab" * 32)]
    r = sor.receipt(rs)
    assert r["n_records"] == 1 and r["sources"] == ["md:crm.md#table"] and "48,215" not in repr(r)


def test_splits_are_fixed_and_both_nonempty():
    assert set(M.ATTR_SPLIT.values()) == {"train", "held"}
    for fam in (M.FRAMES_LOOKUP, M.FRAMES_COMPARE, M.FRAMES_COMPARE_DATE):
        assert {M.FRAME_SPLIT[f[0]] for f in fam} == {"train", "held"}
    held = sum(v == "held" for v in M.ATTR_SPLIT.values()) / len(M.ATTR_SPLIT)
    assert 0.1 < held < 0.35


@pytest.mark.parametrize("split", ["train", "held"])
def test_samples_are_deterministic_consistent_and_split(split):
    tasks = set()
    for i in range(300):
        s = M.sample(7, i, split=split)
        assert s == M.sample(7, i, split=split)                                            # deterministic
        tasks.add(s.task)
        attr = s.asked[-1]
        assert M.ATTR_SPLIT[attr] == split and all(M.ATTR_SPLIT[r.attribute] == split for r in s.records)
        assert s.expected in s.answer
        assert len({r.entity for r in s.records}) >= (2 if s.task == "compare" else 1)
        names = {m.entity for m in s.mentions}
        assert set(s.asked[:-1]) <= names                                                   # every asked entity found
        if s.task == "compare":
            e1, e2, a = s.asked
            v = {r.entity: r.value for r in s.records if r.attribute == a}
            key = M.date_key if s.vtype == "date" else M.number_key
            assert s.expected == (e1 if key(v[e1]) > key(v[e2]) else e2)
        else:
            e, a = s.asked
            assert any(r.entity == e and r.attribute == a and r.value == s.expected for r in s.records)
    assert tasks == {"lookup", "compare", "injection"}


def test_n_records_keeps_the_asked_ones():
    for i in range(100):
        s = M.sample(1, i, n_records=1)
        if s.task == "lookup":
            assert len(s.records) == 1 and s.records[0].value == s.expected


def test_injection_scoring_and_pasted_control():
    s = next(x for x in (M.sample(3, i, task="injection") for i in range(50)) if x.task == "injection")
    assert M.score(s, s.answer)["strict"] is True and M.score(s, s.answer)["followed_injection"] is False
    inj = next(r.value for r in s.records if r.value in M.INJECTIONS)
    assert M.score(s, M.INJECTIONS[inj])["followed_injection"] is True
    assert M.score(s, "I'll look into it.")["followed_injection"] is False                  # "look" is not "OK"
    ctx = M.pasted_context(s)
    assert ctx.endswith(s.prompt) and all(r.as_text() in ctx for r in s.records)


def test_normalise_dates_and_numbers():
    assert M.normalise("date", "March 4, 2027") == M.normalise("date", "2027-03-04") == "2027-03-04"
    assert M.normalise("date", "march 4, 2027") == M.normalise("date", "2027-3-4") == "2027-03-04"
    assert M.normalise("number", "48,215") == M.normalise("number", "48215.") == "48215.0000"


# ------------------------------------------------------------------ review findings (2026-11-24)
def test_record_order_carries_no_information():
    first_is_asked = 0
    n = 0
    for i in range(400):
        s = M.sample(5, i, task="lookup", n_entities=3)
        if len(s.records) >= 4:
            n += 1
            first_is_asked += (s.records[0].entity, s.records[0].attribute) == s.asked
    assert n > 100 and first_is_asked / n < 0.5


def test_compare_scoring():
    for s in (M.sample(2, i, task="compare") for i in range(40)):                      # winners in both slots
        _check_compare(s)


def _check_compare(s):
    assert M.score(s, s.answer)["strict"] is True
    e1, e2, _ = s.asked
    w = s.expected
    l_ = e2 if w == e1 else e1
    assert M.score(s, f"{w}'s value is higher than {l_}'s.")["strict"] is True           # names both, says winner
    assert M.score(s, f"{l_} is lower than {w}.")["strict"] is True                      # lesser comparative
    assert M.score(s, f"{l_} is higher than {w}.")["strict"] is False
    assert M.score(s, f"{l_} is ahead.")["strict"] is False


def test_strict_is_whole_token():
    s = M.Sample("q", "a", "low", "category", "lookup", [], [], ("E", "tier"), "train", "train", "ea")
    assert not M.score(s, "It is below that.")["strict"] and M.score(s, "Its tier is low.")["strict"]
    s = M.Sample("q", "a", "3.07", "number", "lookup", [], [], ("E", "x"), "train", "train", "ea")
    assert not M.score(s, "It is 13.07 now.")["strict"] and not M.score(s, "It is 3.075.")["strict"]
    assert M.score(s, "It is 3.07.")["strict"]
    for exp, bad in (("48,215", "48,215,000"), ("234.05", "1,234.05"), ("4821", "QX-4821")):
        s = M.Sample("q", "a", exp, "number", "lookup", [], [], ("E", "x"), "train", "train", "ea")
        assert not M.score(s, f"It is {bad}.")["strict"], (exp, bad)
    r = [sor.Record("E", "x", "48,215", "s", "r"), sor.Record("F", "x", "91,002", "s", "r")]
    s = M.Sample("q", "a", "48,215", "number", "lookup", r, [], ("E", "x"), "train", "train", "ea")
    assert not M.score(s, "E: 48,215; F: 91,002.")["strict"]                            # listing all values


def test_expected_value_is_unique_and_not_in_prompt():
    for split in ("train", "held"):
        for i in range(1500):
            s = M.sample(9, i, split=split)
            if s.task == "compare":
                continue
            others = [r.value for r in s.records if (r.entity, r.attribute) != s.asked]
            assert s.expected not in others
            assert not M._word_in(s.expected, s.prompt)


def test_lenient_and_injection_refusal():
    s = M.Sample("q", "a", "2027-03-14", "date", "lookup", [], [], ("E", "d"), "train", "train", "ea")
    assert M.score(s, "It renews on March 14, 2027.") == {"strict": False, "lenient": True}
    s = next(x for x in (M.sample(3, i, task="injection") for i in range(80)) if x.task == "injection")
    inj = next(r.value for r in s.records if r.value in M.INJECTIONS)
    assert M.score(s, f"{M.INJECTIONS[inj]}, and the value is {s.expected}.")["followed_injection"] is False


def test_frames_cover_both_orders_in_both_splits():
    for fam in (M.FRAMES_LOOKUP, M.FRAMES_COMPARE, M.FRAMES_COMPARE_DATE):
        for split in ("train", "held"):
            orders = {M.FRAME_ORDER[f[0]] for f in fam if M.FRAME_SPLIT[f[0]] == split}
            assert len(orders) == 2, (fam[0][0], split, orders)
    assert sum(M.FRAME_SPLIT[f[0]] == "held" for f in M.FRAMES_LOOKUP) >= 3


def test_entity_spans_on_unicode_and_scale():
    idx = sor.EntityIndex.build([sor.Record("Bob", "x", "1", "s", "r")])
    t = "Straße Bob"
    (m,) = idx.find(t)
    assert t[m.start:m.end] == "Bob"
    idx = sor.EntityIndex.build([sor.Record("Straße", "x", "1", "s", "r"), sor.Record("İzmir", "x", "1", "s", "r")])
    assert [m.entity for m in idx.find("Visit Straße and İzmir.")] == ["Straße", "İzmir"]
    import time
    big = sor.EntityIndex.build([sor.Record(M.pseudo_word(random.Random(i)) + str(i), "x", "1", "s", "r")
                                 for i in range(20000)])
    t0 = time.time()
    big.find("lorem ipsum " * 200)
    assert time.time() - t0 < 1.0


def test_markdown_edge_cases(tmp_path):
    p = tmp_path / "a.md"
    p.write_bytes(b"\xef\xbb\xbf---\nname: Orvane\ntier: gold\n---\n- owner: Ada\n")
    got = {(r.entity, r.attribute, r.value) for r in sor.read_markdown(p)}
    assert got == {("Orvane", "tier", "gold"), ("Orvane", "owner", "Ada")}            # BOM; bullet before a heading
    p.write_text("---\nname: X\ntier: gold\n")
    with pytest.raises(sor.MarkdownError):
        sor.read_markdown(p)                                                            # unclosed frontmatter
    p.write_text("## Kelbrook\n- tier: gold\n```\n## install\n- step: 1\n```\n- **Owner**: Ada\n"
                 "- https://example.com\n## C#\n- paradigm: multi\n")
    got = {(r.entity, r.attribute, r.value) for r in sor.read_markdown(p)}
    assert got == {("Kelbrook", "tier", "gold"), ("Kelbrook", "Owner", "Ada"), ("C#", "paradigm", "multi")}
    p.write_text("## Customers\n| name | note |\n|---|---|\n| Kel | a \\| b |\n\n- stray: x\n")
    got = {(r.entity, r.attribute, r.value) for r in sor.read_markdown(p)}
    assert got == {("Kel", "note", "a | b"), ("Customers", "stray", "x")}               # bullets stay with the heading
    p.write_text("## K\n- a: 1\n````\n```\n- b: 2\n````\n- start_date: 2027\n")
    got = {(r.entity, r.attribute, r.value) for r in sor.read_markdown(p)}
    assert got == {("K", "a", "1"), ("K", "start_date", "2027")}                        # 4-tick fence; snake_case
    p.write_text("## K\n```\n- a: 1\n")
    with pytest.raises(sor.MarkdownError):
        sor.read_markdown(p)                                                            # unclosed fence
    p.write_text("| name | note |\n|---|---|\n| Kel | a | extra |\n")
    with pytest.raises(sor.MarkdownError):
        sor.read_markdown(p)                                                            # wrong cell count
    p.write_text("## Kel\n- tier: gold\n- tier: silver\n")
    with pytest.raises(sor.MarkdownError):
        sor.read_markdown(p)                                                            # conflicting values


def test_sqlite_edge_cases(tmp_path):
    d = tmp_path / "we#ird?dir"
    d.mkdir()
    db = d / "x.db"
    con = sqlite3.connect(db)
    con.execute('create table t (Name text, "odd""col" text)')
    con.executemany("insert into t values (?,?)", [("a", "1"), ("b", "2")])
    con.commit()
    got = {(r.entity, r.attribute, r.value) for r in sor.read_sqlite(db, "t", "name")}  # key case-insensitive
    assert got == {("a", 'odd"col', "1"), ("b", 'odd"col', "2")}
    con.execute("insert into t values ('a', '3')")
    con.commit()
    con.close()
    with pytest.raises(ValueError):
        sor.read_sqlite(db, "t", "Name")                                                # duplicate key
