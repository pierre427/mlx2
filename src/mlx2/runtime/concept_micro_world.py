"""Deterministic semantic-memory micro-world with held-out surface forms."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass

MICRO_WORLD_SCHEMA = "mlx2-concept-micro-world-v1"


@dataclass(frozen=True, slots=True)
class ConceptEpisode:
    episode_id: str
    subject: str
    subject_alias: str
    relation: str
    intermediate: str | None
    object: str
    query: str
    answer: str
    hops: int
    distractors: tuple[tuple[str, str, str], ...]

    def as_dict(self) -> dict:
        return asdict(self)


PREFIXES = (
    "Cedar", "Amber", "Silver", "Moss", "Birch", "Willow", "Juniper",
    "Granite", "Copper", "Maple", "Quiet", "North", "South", "Hidden",
    "Rain", "Fox", "Owl", "Pine", "Fern", "River",
)
SUFFIXES = (
    "Loop", "Path", "Gate", "Bridge", "Glen", "Ridge", "Hollow", "Crossing",
    "Meadow", "Landing",
)
AROMAS = (
    "earthy cedar", "rain-washed mint", "warm pine resin", "crushed fern",
    "smoky juniper", "wet stone", "sweet birch", "peppery moss", "cold spruce",
    "sunlit bark", "wild thyme", "damp leaf litter", "river clay", "apple wood",
    "wintergreen", "forest loam", "orange lichen", "sage needles", "maple smoke",
    "violet rain",
)
MARKERS = (
    "blue cairn", "bronze owl", "three white stones", "split cedar post",
    "red lantern", "moss arch", "silver bell", "carved fox", "glass acorn",
    "copper feather",
)


def _identifier(seed: int, index: int, subject: str, object_: str) -> str:
    payload = f"{seed}:{index}:{subject}:{object_}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def generate_micro_world(*, seed: int, count: int = 200) -> tuple[ConceptEpisode, ...]:
    if type(seed) is not int or type(count) is not int or not 12 <= count <= 2000:
        raise ValueError("micro-world requires an integer seed and 12..2000 episodes")
    rng = random.Random(seed)
    subjects = [f"{prefix} {suffix}" for prefix in PREFIXES for suffix in SUFFIXES]
    rng.shuffle(subjects)
    aromas = list(AROMAS)
    markers = list(MARKERS)
    episodes = []
    for index in range(count):
        subject = subjects[index % len(subjects)]
        prefix, suffix = subject.split(" ", 1)
        subject_alias = f"the {prefix.casefold()} {suffix.casefold()} route"
        answer = aromas[(index * 7 + seed) % len(aromas)]
        hops = 2 if index % 2 else 1
        intermediate = markers[(index * 3 + seed) % len(markers)] if hops == 2 else None
        if hops == 1:
            query = f"What aroma is associated with {subject_alias}?"
        else:
            query = f"Following the marker for {subject_alias}, what aroma should I notice?"
        distractors = []
        for offset in (1, 5, 11):
            other = subjects[(index + offset) % len(subjects)]
            other_answer = aromas[(index + offset * 3 + seed) % len(aromas)]
            distractors.append((other, "has_property", other_answer))
        episodes.append(
            ConceptEpisode(
                episode_id=_identifier(seed, index, subject, answer),
                subject=subject,
                subject_alias=subject_alias,
                relation="has_property",
                intermediate=intermediate,
                object=answer,
                query=query,
                answer=answer,
                hops=hops,
                distractors=tuple(distractors),
            )
        )
    return tuple(episodes)


def split_micro_world(
    episodes: Iterable[ConceptEpisode], *, train_fraction: float = 0.70, validation_fraction: float = 0.15
) -> dict[str, tuple[ConceptEpisode, ...]]:
    rows = tuple(episodes)
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("micro-world split fractions must be inside (0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("micro-world split must leave a test partition")
    train_end = round(len(rows) * train_fraction)
    validation_end = train_end + round(len(rows) * validation_fraction)
    return {
        "train": rows[:train_end],
        "validation": rows[train_end:validation_end],
        "test": rows[validation_end:],
    }


__all__ = [
    "MICRO_WORLD_SCHEMA",
    "ConceptEpisode",
    "generate_micro_world",
    "split_micro_world",
]
