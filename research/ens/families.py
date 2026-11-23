"""A0 task families (spec v0.4 section 6.1, step 1).

A family is a (generator class, structure knobs) pair with a family-seeded hidden
law H: query key -> intermediate z. The provider knows H. The prompt carries a
visible operation op, and the answer is op(z). Training, development and
confirmation families reuse a *training structure* the core can already do and
carry a fresh hidden law H the core never saw: the A0 extension test is new
knowledge on seen structure (re-scoped 2026-11-21 after pilot 2, which showed
the core cannot do unseen structure even with the oracle message). Families on
*unseen* structure (domain size, hop count, automaton size, program shape) are
kept as a separate, labelled generalization axis: splits gen_dev / gen_confirm.

During core training z is drawn uniformly at random per row and the oracle mock
provider supplies it, so the core cannot memorise any H. A family's H only
matters after the freeze, when a contributor module has to learn it from the
family's training pool.
"""
from __future__ import annotations

import functools
import hashlib
import json
import random
import zlib
from dataclasses import dataclass
from pathlib import Path

from research.ens import vocab as V

SEQ_LEN = 32
PROGRAM_Z = 16          # PROGRAM families draw z from 0..15
TRAIN_POOL = 4096       # contributor training rows per family (fixed pool)
HELD_POOL = 512         # held rows per family, disjoint from the training pool

_SOURCE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
# Hidden laws and pools are seeded from the family record plus this law version, not from the file
# hash, so a comment edit cannot silently reseed every H. Bump it whenever generation changes.
LAW_VERSION = "ens.a0.families.v2"   # v2: GRAPH/FST ops are non-constant in z


@dataclass(frozen=True)
class Family:
    fid: str
    cls: str            # PERM | PROGRAM | GRAPH | FST
    knobs: tuple        # class-specific structure knobs, see z_domain / _sample_op
    split: str          # train | dev | confirm | gen_dev | gen_confirm
    law: str = ""       # law tag for extension families (seen structure, fresh H); "" otherwise
    held_div: int = 8   # an op is held iff crc32(op) % held_div == 0; 2 for small op spaces

    @property
    def z_domain(self) -> int:
        if self.cls == "PERM":
            return self.knobs[0]
        if self.cls == "PROGRAM":
            return PROGRAM_Z
        if self.cls == "GRAPH":
            return self.knobs[0]
        if self.cls == "FST":
            return self.knobs[0]
        raise ValueError(self.cls)

    def record(self) -> dict:
        rec = {"fid": self.fid, "cls": self.cls, "knobs": list(self.knobs), "split": self.split,
               "law_version": LAW_VERSION, "generator_sha256": _SOURCE_SHA}
        if self.law:                      # absent otherwise, so training-family digests are unchanged
            rec["law"] = self.law
        if self.split != "train":
            rec["held_div"] = self.held_div
        return rec

    @property
    def digest(self) -> str:
        law = {k: v for k, v in self.record().items() if k != "generator_sha256"}
        blob = json.dumps(law, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(b"ens.family.v0\0" + blob).hexdigest()

    # ------------------------------------------------------------ hidden law
    @functools.cached_property
    def hidden(self) -> tuple[int, ...]:
        """Balanced, family-seeded map key -> z (each z hit floor/ceil(16/m) times)."""
        rng = random.Random(int(self.digest[:16], 16))
        m = self.z_domain
        zs = [i % m for i in range(V.N_KEYS)]
        rng.shuffle(zs)
        return tuple(zs)


@dataclass(frozen=True)
class Row:
    x: int              # query key
    z: int              # intermediate (H(x) at contributor time, random during core training)
    op: tuple[int, ...] # visible operation tokens
    answer: int         # answer value

    @property
    def signature(self) -> tuple:
        return (self.x, self.op)


# ------------------------------------------------------------------ operators
def _apply_prog(z: int, prog) -> int:
    for name, c in prog:
        if name == "ADD":
            z = (z + c) % V.N_VALUES
        elif name == "MUL":
            z = (z * c) % V.N_VALUES
        elif name == "XOR":
            z = z ^ c
        else:
            raise ValueError(name)
    return z


def _rand_const(rng, name):
    if name == "MUL":
        return rng.randrange(3, 64, 2)
    return rng.randrange(1, 64)


_ALL_OPS = [("ADD", c) for c in range(1, 64)] + [("MUL", c) for c in range(3, 64, 2)] + \
           [("XOR", c) for c in range(1, 64)]


@functools.lru_cache(maxsize=None)
def _program_tables_up_to(n: int) -> frozenset:
    """Function tables over z in 0..15 of every program with at most n ops (n <= 2)."""
    tables = {tuple(range(PROGRAM_Z))}
    if n >= 1:
        for a in _ALL_OPS:
            tables.add(tuple(_apply_prog(z, [a]) for z in range(PROGRAM_Z)))
    if n >= 2:
        for a in _ALL_OPS:
            for b in _ALL_OPS:
                tables.add(tuple(_apply_prog(z, [a, b]) for z in range(PROGRAM_Z)))
    if n > 2:
        raise ValueError("only programs up to two ops are enumerated")
    return frozenset(tables)


def _short_program_tables() -> frozenset:
    return _program_tables_up_to(2)


def _sample_op(fam: Family, rng: random.Random):
    """Return (op tokens, f) where f maps z to the answer value.

    Every op must depend on z: an instance whose answer is the same for every z in the domain
    would be answerable from the prompt alone and is redrawn.
    """
    while True:
        toks, f = _sample_op_once(fam, rng)
        if len({f(z) for z in range(fam.z_domain)}) > 1:
            return toks, f


def _sample_op_once(fam: Family, rng: random.Random):
    cls, k = fam.cls, fam.knobs
    if cls == "PERM":
        m = k[0]
        perm = list(range(m))
        rng.shuffle(perm)
        order = list(range(m))
        rng.shuffle(order)
        toks = [V.tok("PERM")]
        for a in order:
            toks += [V.val(a), V.val(perm[a])]
        return tuple(toks), (lambda z, p=tuple(perm): p[z])
    if cls == "PROGRAM":
        pattern = k[0]
        while True:   # witness filter at every length: the function must need all len(pattern) ops
            prog = tuple((name, _rand_const(rng, name)) for name in pattern)
            table = tuple(_apply_prog(z, prog) for z in range(PROGRAM_Z))
            if table not in _program_tables_up_to(len(prog) - 1):
                break
        toks = []
        for name, c in prog:
            toks += [V.tok(name), V.val(c)]
        return tuple(toks), (lambda z, p=prog: _apply_prog(z, p))
    if cls == "GRAPH":
        m, hops = k
        succ = tuple(rng.randrange(m) for _ in range(m))
        toks = [V.tok("GR")] + [V.val(s) for s in succ] + [V.tok("HOP"), V.val(hops)]

        def f(z, s=succ, h=hops):
            for _ in range(h):
                z = s[z]
            return z
        return tuple(toks), f
    if cls == "FST":
        s, n = k
        table = tuple((rng.randrange(s), rng.randrange(s)) for _ in range(s))
        bits = tuple(rng.randrange(2) for _ in range(n))
        toks = [V.tok("FST")]
        for n0, n1 in table:
            toks += [V.val(n0), V.val(n1)]
        toks += [V.tok("IN")] + [V.tok("B1" if b else "B0") for b in bits] + [V.tok("END")]

        def f(z, t=table, bs=bits):
            for b in bs:
                z = t[z][b]
            return z
        return tuple(toks), f
    raise ValueError(cls)


def sample_row(fam: Family, rng: random.Random, *, x: int | None = None, z: int | None = None) -> Row:
    """One row. With z=None and x given, z = H(x) (contributor/eval); with both None, z is random."""
    op, f = _sample_op(fam, rng)
    if x is None:
        x = rng.randrange(V.N_KEYS)
    if z is None:
        z = rng.randrange(fam.z_domain)
    return Row(x=x, z=z, op=op, answer=f(z))


def answer_for(fam: Family, row: Row, z: int) -> int:
    """Re-score a row's visible operation for a different intermediate (counterfactual target)."""
    return _eval_op(fam, row.op, z)


def _eval_op(fam: Family, op: tuple[int, ...], z: int) -> int:
    cls = fam.cls
    vals = [t - V.V0 for t in op]
    if cls == "PERM":
        pairs = vals[1:]
        mapping = {pairs[i]: pairs[i + 1] for i in range(0, len(pairs), 2)}
        return mapping[z]
    if cls == "PROGRAM":
        prog = [(V.TOKENS[op[i]], vals[i + 1]) for i in range(0, len(op), 2)]
        return _apply_prog(z, prog)
    if cls == "GRAPH":
        m = fam.knobs[0]
        succ, hops = vals[1:1 + m], vals[2 + m]
        for _ in range(hops):
            z = succ[z]
        return z
    if cls == "FST":
        s = fam.knobs[0]
        table = [(vals[1 + 2 * i], vals[2 + 2 * i]) for i in range(s)]
        bits = [1 if op[j] == V.tok("B1") else 0 for j in range(2 + 2 * s, len(op) - 1)]
        for b in bits:
            z = table[z][b]
        return z
    raise ValueError(cls)


MIN_HELD_OPS = 15        # distinct held operations per contributor family (one-op MUL has 31 ops, holds 15)
_STALE = 4000            # stop looking for new signatures after this many draws without one


def _is_held(fam: Family, sig) -> bool:
    return zlib.crc32(repr(sig[1]).encode()) % fam.held_div == 0


def family_rows(fam: Family, split: str) -> list[Row]:
    """Fixed contributor pools, disjoint by op through a hash partition, so held rows
    pair every key with operations no *contributor* training row used. The core itself saw these
    operations during core training (with random z): held means new to the module, not to the core.

    'held' is up to HELD_POOL distinct held-bucket rows. 'train' is TRAIN_POOL rows from the other
    buckets: distinct while new signatures keep turning up, then cycled. Small structures (one-op
    PROGRAM, FST 2-x) have only a few hundred signatures, so their held pools are smaller; the
    module reads only the query key through the causal tap, so cycling cannot leak an answer.
    """
    if fam.split == "train":
        raise ValueError(f"{fam.fid}: core-training families have no contributor pools")
    if split not in ("train", "held"):
        raise ValueError(split)
    want_held = split == "held"
    n = HELD_POOL if want_held else TRAIN_POOL
    rng = random.Random(int(hashlib.sha256(f"{fam.digest}:{split}".encode()).hexdigest()[:16], 16))
    out, seen, stale = [], set(), 0
    while len(out) < n and stale < _STALE:
        x = rng.randrange(V.N_KEYS)
        r = sample_row(fam, rng, x=x, z=fam.hidden[x])
        if _is_held(fam, r.signature) != want_held or r.signature in seen:
            stale += 1
            continue
        stale = 0
        seen.add(r.signature)
        out.append(r)
    if not out:
        raise RuntimeError(f"{fam.fid}: no {split} rows")
    if not want_held:
        out = [out[i % len(out)] for i in range(n)]
    return out


# ------------------------------------------------------------------ layout
def layout(row: Row, mode: str) -> tuple[list[int], int, int, int]:
    """Token ids plus (query_idx, field_idx, ans_idx).

    mode 'bus'      : [BOS Q x op.. SLOT NEUTRAL ANS]   z arrives through the bus at field_idx
    mode 'typed'    : [BOS Q x op.. SLOT V(z)   ANS]    z arrives as an ordinary typed input
    mode 'ordinary' : [BOS GIVEN V(z) op.. SLOT NEUTRAL ANS]  no provider; z is in the prompt
    The geometry is identical across modes; the answer is predicted at ans_idx.
    """
    if mode == "ordinary":
        head = [V.BOS, V.GIVEN, V.val(row.z)]
    else:
        head = [V.BOS, V.Q, V.key(row.x)]
    field = V.val(row.z) if mode == "typed" else V.NEUTRAL
    ids = head + list(row.op) + [V.SLOT, field, V.ANS]
    if len(ids) > SEQ_LEN:
        raise ValueError(f"row longer than SEQ_LEN: {len(ids)}")
    field_idx = len(ids) - 2
    ans_idx = len(ids) - 1
    return ids + [V.PAD] * (SEQ_LEN - len(ids)), 2, field_idx, ans_idx


# ------------------------------------------------------------------ registry
def _f(cls, knobs, split, law="", held_div=8):
    kn = "-".join(str(k) if not isinstance(k, tuple) else ".".join(k) for k in knobs)
    return Family(f"{cls.lower()}.{kn}" + (f".{law}" if law else ""), cls, knobs, split, law, held_div)


SEALED = ("confirm", "gen_confirm")      # held rows and contributor training need --unseal-confirm
CONTRIBUTOR_SPLITS = ("dev", "confirm", "gen_dev", "gen_confirm", "route_dev")


FAMILIES: tuple[Family, ...] = (
    # train (17): structure classes the core sees during core training
    _f("PERM", (4,), "train"), _f("PERM", (5,), "train"), _f("PERM", (6,), "train"),
    _f("PROGRAM", (("ADD",),), "train"), _f("PROGRAM", (("MUL",),), "train"),
    _f("PROGRAM", (("XOR",),), "train"), _f("PROGRAM", (("ADD", "MUL"),), "train"),
    _f("PROGRAM", (("MUL", "ADD"),), "train"), _f("PROGRAM", (("ADD", "XOR"),), "train"),
    _f("PROGRAM", (("XOR", "MUL"),), "train"),
    _f("GRAPH", (4, 2), "train"), _f("GRAPH", (6, 1), "train"), _f("GRAPH", (6, 2), "train"),
    _f("GRAPH", (8, 1), "train"),
    _f("FST", (2, 1), "train"), _f("FST", (2, 2), "train"), _f("FST", (3, 1), "train"),
    # dev (4) and confirm (4): extension on seen structure, a fresh hidden law per family. Structures were
    # picked where both pilot-2 cores reach an oracle ceiling >= 0.9
    # (research/ens/evidence/pilot2-20261121/ceilings-train-20261121.txt); PERM and multi-op PROGRAM are
    # excluded because neither core can do them at this size. Dev and confirm share no structure, so
    # choices made on dev rows cannot tune confirm rows (review F3, 2026-11-21). One-op PROGRAM has few
    # operations and holds half of them, for >= MIN_HELD_OPS held ops.
    _f("PROGRAM", (("ADD",),), "dev", "h1", 2), _f("GRAPH", (6, 2), "dev", "h1"),
    _f("FST", (3, 1), "dev", "h1"), _f("GRAPH", (4, 2), "dev", "h1"),
    # confirm: sealed until the decision record is written
    _f("PROGRAM", (("MUL",),), "confirm", "h1", 2), _f("PROGRAM", (("XOR",),), "confirm", "h1", 2),
    _f("GRAPH", (6, 1), "confirm", "h1"), _f("GRAPH", (8, 1), "confirm", "h1"),
    # No FST confirmation family: fst.2-x has <= 8 held ops even at half, and fst.3-1 is in dev. An FST
    # confirmation needs a larger FST training structure in the next core-training round.
    # route_dev (2): a second fresh law on a dev structure, for the H8 same-layout routing test (two modules
    # whose prompts look alike, so only content - the key - can tell the router which one writes)
    _f("GRAPH", (6, 2), "route_dev", "h2"), _f("FST", (3, 1), "route_dev", "h2"),
    # gen_dev (4) / gen_confirm (6): unseen structure, the separate generalization axis
    _f("PERM", (8,), "gen_dev"), _f("PROGRAM", (("XOR", "ADD"),), "gen_dev"),
    _f("GRAPH", (8, 2), "gen_dev"), _f("FST", (3, 2), "gen_dev"),
    _f("PERM", (10,), "gen_confirm"), _f("PROGRAM", (("XOR", "ADD", "MUL"),), "gen_confirm"),
    _f("GRAPH", (8, 3), "gen_confirm"), _f("GRAPH", (10, 2), "gen_confirm"),
    _f("FST", (3, 3), "gen_confirm"), _f("FST", (4, 2), "gen_confirm"),
)
BY_ID = {f.fid: f for f in FAMILIES}


def split(name: str) -> list[Family]:
    return [f for f in FAMILIES if f.split == name]


def registry_record() -> dict:
    return {"schema": "ens.a0.families.v0", "generator_sha256": _SOURCE_SHA,
            "families": [dict(f.record(), digest=f.digest) for f in FAMILIES]}


def check_structure() -> list[str]:
    """Return problems: duplicate ids, a gen family on a train structure, an extension family off train
    structure or without a law tag, an extension law equal to another family's, overlong rows."""
    problems = []
    ids = [f.fid for f in FAMILIES]
    if len(ids) != len(set(ids)):
        problems.append("duplicate family ids")
    train_struct = {(f.cls, f.knobs) for f in split("train")}
    dev_struct = {(f.cls, f.knobs) for f in split("dev") + split("gen_dev")}
    for f in split("confirm") + split("gen_confirm"):
        if (f.cls, f.knobs) in dev_struct:
            problems.append(f"{f.fid}: confirmation structure also used in development")
    laws = {}
    for f in FAMILIES:
        if f.split in ("gen_dev", "gen_confirm") and (f.cls, f.knobs) in train_struct:
            problems.append(f"{f.fid}: generalization structure also in train")
        if f.split == "route_dev" and (f.cls, f.knobs) not in {(g.cls, g.knobs) for g in split("dev")}:
            problems.append(f"{f.fid}: a routing family must share a dev structure")
        if f.split in ("dev", "confirm", "route_dev"):
            if (f.cls, f.knobs) not in train_struct:
                problems.append(f"{f.fid}: extension family off training structure")
            if not f.law:
                problems.append(f"{f.fid}: extension family without a law tag")
        if f.split in ("dev", "confirm"):
            ops = {r.op for r in family_rows(f, "held")}
            if len(ops) < MIN_HELD_OPS:
                problems.append(f"{f.fid}: only {len(ops)} distinct held operations")
        if f.split != "train" and laws.setdefault((f.cls, f.knobs, f.hidden), f.fid) != f.fid:
            problems.append(f"{f.fid}: hidden law equals {laws[(f.cls, f.knobs, f.hidden)]}")
        rng = random.Random(0)
        for _ in range(8):
            try:
                layout(sample_row(f, rng), "typed")
            except ValueError as e:
                problems.append(f"{f.fid}: {e}")
                break
    return problems
