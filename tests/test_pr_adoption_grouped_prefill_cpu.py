"""Observed IO prerequisite tests; synthetic metadata, no model run."""
import copy
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("group_adoption",
    Path(__file__).parents[1] / "scripts/research/pr_adoption_grouped_prefill.py")
group = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(group)


def chunks():
    return [{"checkpoint": "synthetic@fixture", "source_revision": "a" * 40,
             "expert_cache_bytes": 512, "start": i * 128, "rows": 128,
             "samples": False, "image_rows": False, "grammar_rows": False,
             "interior_checkpoint": False,
             "expert_reads": [{"layer": 0, "expert": 3, "bytes": 128}]}
            for i in range(2)]


def test_resident_evidence_cannot_justify_grouping_and_memory_is_not_free():
    rows = chunks()
    args = {"extra_activation_bytes": 64, "free_bytes": 64}
    assert not group.assess(rows, **args, resident_experts=True)["admitted"]
    assert not group.assess(rows, **{**args, "free_bytes": 63})["admitted"]
    for row in rows:
        row["expert_reads"] = []
    assert not group.assess(rows, **args)["admitted"]


def test_boundaries_and_source_identity_prevent_unsafe_groups():
    rows = chunks()
    rows[-1]["samples"] = True
    assert not group.assess(rows, extra_activation_bytes=0, free_bytes=64)["admitted"]
    rows = chunks()
    rows[-1]["source_revision"] = "b" * 40
    with pytest.raises(ValueError, match="different"):
        group.assess(rows, extra_activation_bytes=0, free_bytes=64)


def test_union_is_only_an_ideal_read_ceiling():
    rows = chunks()
    result = group.assess(rows, extra_activation_bytes=64, free_bytes=64)
    assert result["admitted"]
    assert result["observed_read_bytes"] == 256 and result["ideal_saved_bytes"] == 128
    assert not result["selected"] and not result["qualified"]
    changed = copy.deepcopy(rows)
    changed[-1]["expert_reads"][0]["bytes"] = 129
    with pytest.raises(ValueError, match="identity"):
        group.assess(changed, extra_activation_bytes=0, free_bytes=64)


def test_existing_within_chunk_rereads_cannot_justify_cross_chunk_grouping():
    rows = chunks()
    rows[0]["expert_reads"] *= 2
    rows[1]["expert_reads"][0]["expert"] = 7
    result = group.assess(rows, extra_activation_bytes=0, free_bytes=64)
    assert result["within_chunk_repeat_bytes"] == 128
    assert result["ideal_saved_bytes"] == 0 and not result["admitted"]


def test_layer_major_pins_only_one_layer_union_at_once():
    rows = chunks()
    for row in rows:
        row["expert_cache_bytes"] = 128
        row["expert_reads"] += [{"layer": 1, "expert": 7, "bytes": 128}]
    result = group.assess(rows, extra_activation_bytes=0, free_bytes=64)
    assert result["ideal_union_read_bytes"] == 256
    assert result["max_layer_pinned_bytes"] == 128 and result["admitted"]
