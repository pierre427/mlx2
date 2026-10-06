#!/usr/bin/env python3
"""CPU-only viability bench for exact shared-prefix prompt tokenization.

This is a research harness, not a serving path.  It implements the safety
contract from Strata #567: resume only after a special-token boundary whose
look-ahead decision is wholly inside the shared text.  It also carries the
later #537/plain-span review finding: a change in which source ranges are
plain text limits reuse even when the rendered strings are identical.

With ``--tokenizer`` the harness uses a local Hugging Face tokenizer without
remote code or network access.  Without it, a deterministic UTF-8 byte/special
token tokenizer makes the exactness tests and benchmark dependency-free.
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
import statistics
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path


def common_prefix_len(left: str, right: str) -> int:
    """Return the character length of the common prefix using slice compares."""
    limit = min(len(left), len(right))
    low, step = 0, 4096
    while low < limit:
        high = min(limit, low + step)
        if left[low:high] != right[low:high]:
            break
        low, step = high, min(step * 2, 1 << 20)
    else:
        return limit
    while high - low > 1:
        middle = (low + high) // 2
        if left[low:middle] == right[low:middle]:
            low = middle
        else:
            high = middle
    return low


def _normalize_spans(spans: Iterable[Sequence[int]], length: int) -> tuple[tuple[int, int], ...]:
    result = []
    for raw_start, raw_stop in spans:
        start, stop = int(raw_start), int(raw_stop)
        if not 0 <= start <= stop <= length:
            raise ValueError("plain span lies outside rendered prompt")
        if start != stop:
            result.append((start, stop))
    result.sort()
    if any(left[1] > right[0] for left, right in pairwise(result)):
        raise ValueError("plain spans must not overlap")
    return tuple(result)


def _span_policy_agreement_limit(
    old: Sequence[tuple[int, int]], new: Sequence[tuple[int, int]], limit: int
) -> int:
    """First character where old/new special-token eligibility differs."""
    points = {0, max(0, int(limit))}
    for start, stop in (*old, *new):
        points.add(min(max(start, 0), limit))
        points.add(min(max(stop, 0), limit))
    points = sorted(points)

    def plain(spans, position):
        return any(start <= position < stop for start, stop in spans)

    for start, stop in pairwise(points):
        if start < stop and plain(old, start) != plain(new, start):
            return start
    return limit


class ByteSpecialTokenizer:
    """Small exact tokenizer fixture: UTF-8 bytes plus longest special literals."""

    def __init__(self, special_tokens=("<|im_start|>", "<|im_end|>", "<think>", "</think>")):
        self.special_tokens = tuple(sorted(set(special_tokens), key=lambda value: (-len(value), value)))
        self.special_ids = {token: 256 + index for index, token in enumerate(self.special_tokens)}
        self.max_special_len = max((len(token) for token in self.special_tokens), default=1)

    @staticmethod
    def _plain(spans, start, stop):
        return any(left < stop and start < right for left, right in spans)

    def encode_marked(self, text: str, *, plain_spans=()):
        spans = _normalize_spans(plain_spans, len(text))
        output, marks = [], []
        plain_start = 0
        position = 0
        while position < len(text):
            match = next(
                (
                    token
                    for token in self.special_tokens
                    if text.startswith(token, position)
                    and not self._plain(spans, position, position + len(token))
                ),
                None,
            )
            if match is None:
                position += 1
                continue
            output.extend(text[plain_start:position].encode("utf-8"))
            output.append(self.special_ids[match])
            position += len(match)
            plain_start = position
            marks.append((position, len(output)))
        output.extend(text[plain_start:].encode("utf-8"))
        return output, marks

    def encode(self, text: str, *, plain_spans=()):
        return self.encode_marked(text, plain_spans=plain_spans)[0]


class HFMarkedTokenizer:
    """Local-only adapter that exposes Strata-style special-token marks."""

    def __init__(self, path: Path):
        from tokenizers import Tokenizer
        from transformers import AutoTokenizer

        path = Path(path)
        self.backend = AutoTokenizer.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False
        )
        tokens = set(getattr(self.backend, "all_special_tokens", ()))
        tokens.update(getattr(self.backend, "added_tokens_encoder", {}))
        self.special_tokens = tuple(
            sorted((str(token) for token in tokens if token), key=lambda value: (-len(value), value))
        )
        if not self.special_tokens:
            raise ValueError("tokenizer exposes no special-token literals")
        self.special_ids = {
            token: int(self.backend.convert_tokens_to_ids(token)) for token in self.special_tokens
        }
        self.max_special_len = max(map(len, self.special_tokens))
        self._pattern = re.compile("|".join(map(re.escape, self.special_tokens)))
        # The normal fast tokenizer always recognizes AddedToken literals,
        # including those whose JSON ``special`` field is false.  Strata's
        # later plain-span contract needs the opposite behavior inside quoted
        # ranges.  Build a tokenizer-only backend with the same normalizer,
        # pre-tokenizer and BPE model but no AddedToken matcher or post-
        # processor.  This reads tokenizer.json only; no model is constructed.
        payload = json.loads((path / "tokenizer.json").read_text())
        payload["added_tokens"] = []
        payload["post_processor"] = None
        self.plain_backend = Tokenizer.from_str(json.dumps(payload))

    @staticmethod
    def _overlaps(spans, start, stop):
        return any(left < stop and start < right for left, right in spans)

    def encode_marked(self, text: str, *, plain_spans=()):
        spans = _normalize_spans(plain_spans, len(text))
        output, marks, position = [], [], 0
        for match in self._pattern.finditer(text):
            if self._overlaps(spans, match.start(), match.end()):
                continue
            output.extend(
                self.plain_backend.encode(
                    text[position : match.start()], add_special_tokens=False
                ).ids
            )
            output.append(self.special_ids[match.group(0)])
            position = match.end()
            marks.append((position, len(output)))
        output.extend(
            self.plain_backend.encode(text[position:], add_special_tokens=False).ids
        )
        return list(map(int, output)), marks

    def encode(self, text: str, *, plain_spans=()):
        if plain_spans:
            return self.encode_marked(text, plain_spans=plain_spans)[0]
        return list(map(int, self.backend.encode(text, add_special_tokens=False)))


@dataclass(frozen=True)
class _Entry:
    text: str
    plain_spans: tuple[tuple[int, int], ...]
    token_ids: tuple[int, ...]
    mark_ends: tuple[int, ...]
    mark_counts: tuple[int, ...]


class MarkedPrefixEncoder:
    """Experimental exact cache for tokenization prefixes; not used by mlx2 serving."""

    def __init__(self, tokenizer, *, keep: int = 4, margin_override: int | None = None):
        if keep < 1:
            raise ValueError("keep must be positive")
        self.tokenizer, self.keep = tokenizer, int(keep)
        self.margin_override = margin_override
        self.entries: list[_Entry] = []
        self.last_reused_chars = 0
        self.last_reused_tokens = 0

    def encode(self, text: str, *, plain_spans=()) -> list[int]:
        spans = _normalize_spans(plain_spans, len(text))
        margin = (
            self.tokenizer.max_special_len - 1
            if self.margin_override is None
            else int(self.margin_override)
        )
        chosen, chosen_index = None, -1
        for entry in self.entries:
            limit = max(0, common_prefix_len(entry.text, text) - margin)
            limit = _span_policy_agreement_limit(entry.plain_spans, spans, limit)
            index = bisect.bisect_right(entry.mark_ends, limit) - 1
            if index >= 0 and (
                chosen is None or entry.mark_ends[index] > chosen.mark_ends[chosen_index]
            ):
                chosen, chosen_index = entry, index
        if chosen is None:
            token_ids, marks = self.tokenizer.encode_marked(text, plain_spans=spans)
            self.last_reused_chars = self.last_reused_tokens = 0
        else:
            cut = chosen.mark_ends[chosen_index]
            count = chosen.mark_counts[chosen_index]
            tail_spans = tuple(
                (max(0, start - cut), stop - cut)
                for start, stop in spans
                if stop > cut
            )
            tail, marks = self.tokenizer.encode_marked(text[cut:], plain_spans=tail_spans)
            token_ids = list(chosen.token_ids[:count]) + tail
            marks = list(zip(chosen.mark_ends[: chosen_index + 1], chosen.mark_counts[: chosen_index + 1])) + [
                (cut + end, count + token_count) for end, token_count in marks
            ]
            self.last_reused_chars, self.last_reused_tokens = cut, count
        entry = _Entry(
            text=text,
            plain_spans=spans,
            token_ids=tuple(map(int, token_ids)),
            mark_ends=tuple(end for end, _ in marks),
            mark_counts=tuple(count for _, count in marks),
        )
        self.entries = [entry] + [item for item in self.entries if item is not chosen][: self.keep - 1]
        return list(entry.token_ids)


def run_benchmark(tokenizer, *, base_chars: int, turns: int, repeats: int) -> dict:
    unit = "English code: for i in range(8): value += i; CJK 你好.\n"
    document = (unit * (base_chars // len(unit) + 1))[:base_chars]
    prompts = []
    text = f"<|im_start|>user\n{document}<|im_end|>\n"
    for turn in range(turns):
        text += (
            f"<|im_start|>assistant\n<think>step {turn}</think>answer {turn}<|im_end|>\n"
            f"<|im_start|>user\nfollow-up {turn}<|im_end|>\n"
        )
        prompts.append(text + "<|im_start|>assistant\n")

    full_samples, incremental_samples, reused = [], [], []
    for _ in range(repeats):
        encoder = MarkedPrefixEncoder(tokenizer)
        full_run, incremental_run, run_reused = [], [], []
        for prompt in prompts:
            start = time.perf_counter_ns()
            expected = tokenizer.encode(prompt)
            full_run.append(time.perf_counter_ns() - start)
            start = time.perf_counter_ns()
            actual = encoder.encode(prompt)
            incremental_run.append(time.perf_counter_ns() - start)
            if actual != expected:
                raise RuntimeError("incremental tokenizer changed token IDs")
            run_reused.append(encoder.last_reused_chars)
        full_samples.append(full_run)
        incremental_samples.append(incremental_run)
        reused.append(run_reused)
    full_hot = [value for run in full_samples for value in run[1:]]
    incremental_hot = [value for run in incremental_samples for value in run[1:]]
    return {
        "schema": "mlx2.tokenizer-prefix-viability.v1",
        "gpu_used": False,
        "mechanism_selected": False,
        "turns": turns,
        "base_chars": base_chars,
        "repeats": repeats,
        "tokenizer": type(tokenizer).__name__,
        "ids_exact": True,
        "hot_full_median_ns": int(statistics.median(full_hot)),
        "hot_incremental_median_ns": int(statistics.median(incremental_hot)),
        "hot_median_speedup": statistics.median(full_hot) / max(statistics.median(incremental_hot), 1),
        "last_reused_chars": reused[-1][-1],
        "source": {
            "repository": "Niko1221/Strata",
            "pull_request": 567,
            "revision": "f29e527856b85e37bda90626ccc28e002b4989dd",
            "review_followup": "plain-span changes in 0.1.39 must limit reuse",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, help="local Hugging Face tokenizer directory")
    parser.add_argument("--base-chars", type=int, default=100_000)
    parser.add_argument("--turns", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if min(args.base_chars, args.turns, args.repeats) < 1:
        parser.error("benchmark sizes must be positive")
    tokenizer = HFMarkedTokenizer(args.tokenizer.resolve()) if args.tokenizer else ByteSpecialTokenizer()
    report = run_benchmark(
        tokenizer, base_chars=args.base_chars, turns=args.turns, repeats=args.repeats
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
