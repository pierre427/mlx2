import pytest

from mlx2.runtime.concept_micro_world import (
    MAX_EPISODES,
    generate_micro_world,
    split_micro_world,
)


def test_micro_world_is_deterministic_and_has_held_out_test_partition():
    first = generate_micro_world(seed=7, count=200)
    assert first == generate_micro_world(seed=7, count=200)
    assert first != generate_micro_world(seed=8, count=200)
    split = split_micro_world(first)
    assert {name: len(rows) for name, rows in split.items()} == {
        "train": 140,
        "validation": 30,
        "test": 30,
    }
    assert all(row.subject.casefold() in row.subject_alias for row in first)
    assert {row.hops for row in first} == {1, 2}


def test_micro_world_distractors_never_repeat_the_gold_subject():
    for row in generate_micro_world(seed=11, count=64):
        assert len(row.distractors) == 3
        assert all(subject != row.subject for subject, _, _ in row.distractors)


def test_micro_world_test_split_never_repeats_a_training_episode():
    def content(row):
        return (row.subject, row.relation, row.intermediate, row.object, row.query)

    split = split_micro_world(generate_micro_world(seed=7, count=MAX_EPISODES))
    train = {content(row) for row in split["train"]}
    assert not [row for row in split["test"] if content(row) in train]
    # More episodes than distinct subjects would repeat episodes verbatim,
    # putting training rows into the held-out partition.
    with pytest.raises(ValueError, match="distinct episodes"):
        generate_micro_world(seed=7, count=MAX_EPISODES + 1)
    with pytest.raises(ValueError):
        generate_micro_world(seed=7, count=400)
