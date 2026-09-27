"""Deterministic diarization scoring for qualification preflight.

The final corpus score must also be checked with NVIDIA NeMo's scorer.  This
implementation keeps local comparisons and failure diagnosis independent of
that optional evaluation environment.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path


def read_rttm(path: str | Path, *, recording_id: str) -> list[tuple[float, float, str]]:
    segments = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 8 or fields[0] != "SPEAKER" or fields[1] != recording_id:
            raise ValueError(f"invalid RTTM at {path}:{number}")
        start, duration = float(fields[3]), float(fields[4])
        if not 0 <= start < float("inf") or not 0 < duration < float("inf"):
            raise ValueError(f"invalid RTTM timing at {path}:{number}")
        segments.append((start, start + duration, fields[7]))
    return segments


def write_rttm(path: str | Path, *, recording_id: str, segments: list[dict]) -> None:
    lines = []
    for item in segments:
        start, end = float(item["Start"]), float(item["End"])
        if not 0 <= start < end:
            raise ValueError("invalid predicted segment")
        lines.append(f"SPEAKER {recording_id} 1 {start:.2f} {end-start:.2f} "
                     f"<NA> <NA> speaker_{item['Speaker']} <NA> <NA>\n")
    Path(path).write_text("".join(lines))


def read_uem(path: str | Path, *, recording_id: str) -> list[tuple[float, float]]:
    intervals = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 4 or fields[0] != recording_id:
            raise ValueError(f"invalid UEM at {path}:{number}")
        start, end = float(fields[2]), float(fields[3])
        if not 0 <= start < end < float("inf"):
            raise ValueError(f"invalid UEM timing at {path}:{number}")
        intervals.append((start, end))
    return intervals


def _best_mapping(weight: dict[tuple[str, str], float], refs: set[str], hyps: set[str]):
    """Maximize global speaker co-occurrence with a one-to-one assignment."""
    reference, hypothesis = sorted(refs), sorted(hyps)
    if len(hypothesis) > 8:
        raise ValueError("hypothesis exceeds the checkpoint's eight speakers")
    states = {0: (0.0, ())}
    for name in reference:
        next_states = dict(states)  # leaving a reference speaker unmatched is legal
        for mask, (score, pairs) in states.items():
            for index, speaker in enumerate(hypothesis):
                if mask & (1 << index):
                    continue
                next_mask = mask | (1 << index)
                candidate = (score + weight.get((name, speaker), 0.0),
                             pairs + ((speaker, name),))
                if candidate[0] > next_states.get(next_mask, (-1.0, ()))[0]:
                    next_states[next_mask] = candidate
        states = next_states
    return dict(max(states.values(), key=lambda value: value[0])[1])


def score_segments(reference: list[tuple[float, float, str]],
                   hypothesis: list[tuple[float, float, str]], *,
                   duration: float, collar: float = 0.0,
                   uem: list[tuple[float, float]] | None = None) -> dict:
    """Score overlap, miss, false alarm and speaker confusion in seconds.

    No-score collars have NIST half-width semantics.  The same UEM and collar
    are applied to both arms; results are percentages only after division by
    scored reference speaker-time.
    """
    if duration <= 0 or collar < 0:
        raise ValueError("invalid scoring duration or collar")
    events = defaultdict(list)

    def add(start, end, kind, speaker=""):
        start, end = max(0.0, start), min(duration, end)
        if start < end:
            events[start].append((kind, speaker, 1))
            events[end].append((kind, speaker, -1))

    reference_names, hypothesis_names = set(), set()
    for start, end, speaker in reference:
        if not 0 <= start < end:
            raise ValueError("invalid reference segment")
        reference_names.add(speaker)
        add(start, end, "reference", speaker)
        if collar:
            add(start - collar, start + collar, "collar")
            add(end - collar, end + collar, "collar")
    for start, end, speaker in hypothesis:
        if not 0 <= start < end:
            raise ValueError("invalid hypothesis segment")
        hypothesis_names.add(speaker)
        add(start, end, "hypothesis", speaker)
    for start, end in (uem if uem is not None else [(0.0, duration)]):
        add(start, end, "uem")
    events[0.0]
    events[duration]
    active = {"reference": defaultdict(int), "hypothesis": defaultdict(int),
              "collar": defaultdict(int), "uem": defaultdict(int)}
    spans = []
    times = sorted(events)
    for index, now in enumerate(times[:-1]):
        for kind, speaker, delta in events[now]:
            active[kind][speaker] += delta
        length = times[index + 1] - now
        if (length <= 0 or active["uem"][""] <= 0 or
                active["collar"][""] > 0):
            continue
        refs = frozenset(name for name, count in active["reference"].items() if count)
        hyps = frozenset(name for name, count in active["hypothesis"].items() if count)
        spans.append((length, refs, hyps))
    weight = defaultdict(float)
    for length, refs, hyps in spans:
        for ref in refs:
            for hyp in hyps:
                weight[(ref, hyp)] += length
    mapping = _best_mapping(weight, reference_names, hypothesis_names)
    miss = false_alarm = confusion = reference_time = 0.0
    for length, refs, hyps in spans:
        reference_time += length * len(refs)
        miss += length * max(0, len(refs) - len(hyps))
        false_alarm += length * max(0, len(hyps) - len(refs))
        correct = sum(mapping.get(hyp) in refs for hyp in hyps)
        confusion += length * (min(len(refs), len(hyps)) - correct)
    error = miss + false_alarm + confusion
    return {"reference_speaker_seconds": reference_time,
            "miss_seconds": miss, "false_alarm_seconds": false_alarm,
            "confusion_seconds": confusion, "error_seconds": error,
            "der": error / reference_time if reference_time else None,
            "reference_speakers": len(reference_names),
            "predicted_speakers": len(hypothesis_names),
            "speaker_count_correct": len(reference_names) == len(hypothesis_names),
            "speaker_count_abs_error": abs(len(reference_names) - len(hypothesis_names))}
