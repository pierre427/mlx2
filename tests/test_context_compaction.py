import pytest

from mlx2.context_compaction import CompactionPressurePolicy, TranscriptTurn, plan_turn_compaction, publish_compacted_state
from mlx2.contracts import Capability, Fidelity, QualifiedProfile, StatePlane
from mlx2.state import InvalidStatePublication, StateIdentity, StateManifest


def test_pressure_and_turn_plan_preserve_system_recent_and_tool_pairs():
    policy = CompactionPressurePolicy()
    assert policy.decide(context_tokens=90, max_context=100, footprint_bytes=1, memory_limit_bytes=10).reason == "context"
    turns = (
        TranscriptTurn("system", (1,)),
        TranscriptTurn("user", (2, 3)),
        TranscriptTurn("assistant", (4,), tool_pair="a"),
        TranscriptTurn("tool", (5,), tool_pair="a"),
        TranscriptTurn("user", (6,)),
        TranscriptTurn("assistant", (7,)),
        TranscriptTurn("user", (8,)),
        TranscriptTurn("assistant", (9,)),
    )
    plan = plan_turn_compaction(turns, target_tokens=6)
    assert turns[0] in plan["kept"]
    assert (turns[2] in plan["removed"]) == (turns[3] in plan["removed"])
    assert all(turn in plan["kept"] for turn in turns[-4:])


def test_compaction_publication_requires_qualified_profile_and_retains_transcript():
    current = StateManifest(StateIdentity("r", 0), {StatePlane.ATTENTION_KV: "old"})
    unqualified = QualifiedProfile("wrong", frozenset({Capability.TEXT}), Fidelity.EXACT, ("e",), "x")
    with pytest.raises(InvalidStatePublication):
        publish_compacted_state(current, profile=unqualified, target_state="new", uncompacted_transcript="ledger")
    approximate = QualifiedProfile(
        "approximate",
        frozenset({Capability.COMPACTION}),
        Fidelity.APPROXIMATE,
        ("e",),
        "x",
    )
    with pytest.raises(InvalidStatePublication, match="exceeds"):
        publish_compacted_state(
            current,
            profile=approximate,
            target_state="new",
            uncompacted_transcript="ledger",
        )
    qualified = QualifiedProfile("compact", frozenset({Capability.COMPACTION}), Fidelity.NUMERICALLY_BOUNDED, ("e",), "x")
    published, prepared = publish_compacted_state(current, profile=qualified, target_state="new", uncompacted_transcript="ledger")
    assert published is prepared
    assert published.fidelity is Fidelity.NUMERICALLY_BOUNDED
    assert published.planes[StatePlane.TRANSCRIPT] == "ledger"
