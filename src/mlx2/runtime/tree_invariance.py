"""Bitwise tree-size invariance check for the Qwen3.8 TensorFold tree verify.

In the spirit of Kapitsa ``tools/moe_tree_check`` (nace-ai/kapitsa 55b6153,
Apache-2.0; see ``provenance/tree-position-law.json``): a position's logits
must not depend on the tree that verified it (its size, its shape, the slot
its path takes) nor on the round's anchor (where the committed-prefix/window
split of the attention and the replay/in-tree split of the GDN fall).

1. ``teacher_stream``: the prompt plus ``n_tok`` greedy tokens, decoded
   through one-row tree rounds, as the teacher-forced token stream.
2. ``verify_schedule``: a fresh cache, the same ordinary prefill, then the
   stream verified round by round through ``lane_multi.multi_tree_forward``
   trees that the schedule builds (anchor row plus path, junk siblings,
   jump rows), committing the true path through ``commit_streams`` each
   round.  Each path row's fp32 logits are hashed under its absolute
   position.
3. ``compare``: every pair of schedules must agree on every common
   position's logits bit for bit.  The report gives the classes per
   position, the first differing position per pair and its max |diff|.

Schedules (``parse_schedule``): ``cN`` a chain of N rows; ``tN`` N rows, the
path first then junk siblings; ``jN`` junk rows interleaved before each path
row (the path sits in other slots); ``rN:S`` a random N-row tree with a
random root path (seed S); ``mixS`` a size 1..16 and shape changing every
round (seed S).
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import numpy as np

DEFAULT_SCHEDULES = ("c1", "c4", "c8", "c16", "t8", "t16", "j8", "j16", "r16:1", "r12:2", "mix1", "mix2")


@dataclass(frozen=True)
class Shape:
    parents: tuple[int, ...]
    path: tuple[int, ...]          # the true path's rows, root (anchor) first


def chain_shape(n: int) -> Shape:
    return Shape(tuple(range(-1, n - 1)), tuple(range(n)))


def tail_shape(n: int) -> Shape:
    """The path first (rows 0..k-1), then junk siblings of the path's nodes."""

    k = max(1, (n + 1) // 2)
    parents = list(range(-1, k - 1)) + [max(0, r - k) for r in range(k, n)]
    return Shape(tuple(parents), tuple(range(k)))


def jump_shape(n: int) -> Shape:
    """Junk before every path row: 0 root, 1 junk, 2 path, 3 junk, 4 path, ..."""

    parents, path, last = [-1], [0], 0
    for row in range(1, n):
        parents.append(last)
        if row % 2 == 0:
            last = row
            path.append(row)
    return Shape(tuple(parents), tuple(path))


def random_shape(n: int, rng: random.Random) -> Shape:
    parents = [-1] + [rng.randrange(row) for row in range(1, n)]
    path, node = [0], 0
    while True:
        kids = [row for row in range(1, n) if parents[row] == node]
        if not kids:
            return Shape(tuple(parents), tuple(path))
        node = rng.choice(kids)
        path.append(node)


@dataclass
class Schedule:
    name: str
    kind: str
    n: int = 8
    seed: int = 0
    rng: random.Random = field(default_factory=random.Random)

    def next_shape(self) -> Shape:
        if self.kind == "c":
            return chain_shape(self.n)
        if self.kind == "t":
            return tail_shape(self.n)
        if self.kind == "j":
            return jump_shape(self.n)
        if self.kind == "r":
            return random_shape(self.n, self.rng)
        size = self.rng.randint(1, 16)
        maker = self.rng.choice((chain_shape, tail_shape, jump_shape, None))
        return random_shape(size, self.rng) if maker is None else maker(size)


def parse_schedule(word: str) -> Schedule:
    if word.startswith("mix"):
        seed = int(word[3:] or 0)
        return Schedule(word, "mix", seed=seed, rng=random.Random(seed))
    kind, rest = word[0], word[1:]
    if kind not in "ctjr":
        raise ValueError(f"unknown schedule {word!r}")
    size, _, seed = rest.partition(":")
    n = int(size)
    if not 1 <= n <= 16:
        raise ValueError(f"schedule {word!r}: 1 to 16 rows")
    return Schedule(word, kind, n=n, seed=int(seed or 0), rng=random.Random(int(seed or 0)))


def _lane_multi():
    """The served tree forward: hash-validated vendored source with mlx2's owned GDN installed."""

    from .qwen38_tensorfold import _modules
    from .tensorfold_qwen38 import lane_multi

    _modules(None, cached=True)
    return lane_multi


def _prefill(model: Any, prompt: Sequence[int], cache: list[Any]) -> None:
    """The ordinary prefill of all but the last prompt token (the first anchor)."""

    if len(prompt) > 1:
        mx.eval(model(mx.array([list(prompt[:-1])], dtype=mx.uint32), cache=cache))


def _round(model: Any, cache: list[Any], start: int, tokens: list[int], parents: Sequence[int],
           path: Sequence[int]) -> mx.array:
    lane_multi = _lane_multi()
    logits, records, _ = lane_multi.multi_tree_forward(
        model.model, model.logits, [tokens], [list(parents)], [cache], [start])
    rows = mx.array(list(path), dtype=mx.int32)
    picked = mx.take(logits[0], rows, axis=0).astype(mx.float32)
    mx.eval(picked)
    lane_multi.commit_streams([cache], records, [list(path)], [len(tokens)], [start])
    return picked


def teacher_stream(model: Any, prompt: Sequence[int], n_tok: int) -> list[int]:
    """Prompt plus ``n_tok`` greedy tokens through one-row tree rounds."""

    cache = model.make_cache()
    _prefill(model, prompt, cache)
    stream = [int(t) for t in prompt]
    start = len(stream) - 1
    for _ in range(n_tok):
        picked = _round(model, cache, start, [stream[start]], [-1], [0])
        stream.append(int(mx.argmax(picked[0]).item()))
        start += 1
    return stream


def _junk(token: int, salt: int, vocab: int) -> int:
    return (token * 7919 + 104729 * (salt + 1)) % vocab


def verify_schedule(model: Any, prompt: Sequence[int], stream: Sequence[int], schedule: Schedule,
                    vocab: int) -> dict[int, tuple[str, np.ndarray]]:
    """Per absolute position: (sha256, fp32 logits) of the row verified there."""

    cache = model.make_cache()
    _prefill(model, prompt, cache)
    start = len(prompt) - 1
    last = len(stream) - 1                       # rows up to here have a next token to predict
    out: dict[int, tuple[str, np.ndarray]] = {}
    rounds = 0
    while start < last:
        shape = schedule.next_shape()
        depth = {row: d for d, row in enumerate(shape.path)}
        # never verify past the stream: cut the path where the stream ends
        keep = min(len(shape.path), last - start)
        tokens = []
        for row in range(len(shape.parents)):
            d = depth.get(row)
            if d is not None and start + d < len(stream):
                tokens.append(int(stream[start + d]))
            else:
                tokens.append(_junk(int(stream[start]), row + 31 * rounds, vocab))
        path = list(shape.path[:keep])
        picked = np.array(_round(model, cache, start, tokens, shape.parents, path))
        for d in range(keep):
            row = picked[d]
            out[start + d] = (hashlib.sha256(row.tobytes()).hexdigest(), row)
        start += keep
        rounds += 1
    return out


def compare(results: dict[str, dict[int, tuple[str, np.ndarray]]]) -> dict:
    """Classes per position and the first divergence of every schedule pair."""

    names = list(results)
    common = sorted(set.intersection(*(set(r) for r in results.values())))
    split = [pos for pos in common if len({results[n][pos][0] for n in names}) > 1]
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            first = next((pos for pos in common if results[a][pos][0] != results[b][pos][0]), None)
            entry = {"a": a, "b": b, "equal": first is None}
            if first is not None:
                ra, rb = results[a][first][1], results[b][first][1]
                entry.update(first_position=first, max_abs_diff=float(np.max(np.abs(ra - rb))),
                             argmax_equal=bool(int(np.argmax(ra)) == int(np.argmax(rb))),
                             differing_positions=sum(1 for pos in common
                                                     if results[a][pos][0] != results[b][pos][0]))
            pairs.append(entry)
    return {
        "schedules": names,
        "positions": len(common),
        "first_position": common[0] if common else None,
        "split_positions": len(split),
        "first_split_position": split[0] if split else None,
        "pairs_total": len(pairs),
        "pairs_differing": sum(1 for p in pairs if not p["equal"]),
        "pairs": pairs,
        "invariant": not split,
    }


def run(model: Any, prompt: Sequence[int], n_tok: int, schedules: Sequence[str] = DEFAULT_SCHEDULES,
        *, vocab: int | None = None, stream: Sequence[int] | None = None) -> dict:
    """Teacher stream, every schedule, and the comparison."""

    if vocab is None:
        vocab = int(model.args.vocab_size)
    if stream is None:
        stream = teacher_stream(model, prompt, n_tok)
    results = {word: verify_schedule(model, prompt, stream, parse_schedule(word), vocab) for word in schedules}
    report = compare(results)
    report["stream_tokens"] = len(stream)
    report["prompt_tokens"] = len(prompt)
    report["stream_sha256"] = hashlib.sha256(np.array(stream, dtype=np.int64).tobytes()).hexdigest()
    return report


__all__ = ["DEFAULT_SCHEDULES", "Schedule", "Shape", "chain_shape", "compare", "jump_shape", "parse_schedule",
           "random_shape", "run", "tail_shape", "teacher_stream", "verify_schedule"]
