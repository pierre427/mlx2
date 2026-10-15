"""Conflict policy over the actual capsule store and session CAS (CPU only)."""

import copy
from dataclasses import replace

import pytest

from mlx2.runtime.hyper_directory import DirectoryContext, HyperDirectory, Scope
from mlx2.runtime.semantic_capsules import CapsuleStore
from mlx2.runtime.semantic_memory import SemanticMemory, SemanticProposal, concept_token


@pytest.fixture
def memory(tmp_path):
    capsules = CapsuleStore(tmp_path / "capsules")
    return SemanticMemory(
        capsules=capsules,
        directory=HyperDirectory(tmp_path / "directory", capsules),
        model_binding="model",
        tokenizer_binding="tokenizer",
        runtime_binding="runtime",
    )


@pytest.fixture
def context():
    return DirectoryContext(model="model", tenant="synthetic-tenant", session="facts")


def fact(value, *, tier="session_committed", key="current gate code", supersedes=()):
    return SemanticProposal(
        "gate code", "has_property", value, .99, .8, f"observation-{value}",
        authority_tier=tier, conflict_key=key, supersedes=supersedes,
    )


def commit(memory, context, *proposals, **kwargs):
    return memory.commit_after_delivery(
        context, proposals, response_delivered=True, authenticated_tenant=True, **kwargs,
    )


def edges(memory, context):
    return {edge["object"]: edge for edge in memory.load(context)[0]["edges"]}


def test_equal_authority_conflicts_are_contested_and_retry_is_idempotent(memory, context):
    old, new = fact("alpha"), fact("beta")
    commit(memory, context, old)
    commit(memory, context, new)
    result = memory.retrieve(context, "gate code")
    assert {edge["status"] for edge in result.edges} == {"contested"}
    assert {edge["authority"] for edge in result.edges} == {"proposal"}
    assert "Conflicting semantic memory (not authoritative)" in result.preamble()
    assert "Verified semantic memory" not in result.preamble()
    before = memory.load(context)
    assert commit(memory, context, old)["reason"] == "unchanged"
    assert memory.load(context) == before


def test_stronger_fact_supersedes_older_fact_without_reactivating_on_retry(memory, context):
    old = fact("alpha", tier="legacy")
    commit(memory, context, old)
    commit(memory, context, fact("beta", tier="signed_current"))
    by_object = edges(memory, context)
    assert by_object[concept_token("alpha")]["status"] == "superseded"
    assert by_object[concept_token("alpha")]["authority"] == "proposal"
    assert by_object[concept_token("beta")]["authority"] == "committed"
    result = memory.retrieve(context, "gate code")
    assert [edge["object"] for edge in result.edges] == [concept_token("beta")]
    assert "gate code has_property alpha" not in result.preamble()
    assert "gate code has_property beta" in result.preamble()
    assert commit(memory, context, old)["reason"] == "unchanged"


def test_existing_capsule_consumer_receives_only_active_committed_edges(memory, context):
    from mlx2.experimental.hysparse2.capsule_memory import CapsuleMemory

    commit(memory, context, fact("alpha", tier="signed_current"))
    receipt = commit(memory, context, fact("beta", tier="legacy"))
    selected = CapsuleMemory.from_store(
        memory.capsules, receipt["capsule"], **memory.bindings,
        encode=lambda text: [ord(c) for c in text], vocab_size=256,
    )
    assert [item[0] for item in selected.records] == ["gate code has_property alpha"]
    receipt = commit(memory, context, fact("gamma", tier="signed_current"))
    with pytest.raises(ValueError, match="nonempty committed edge set"):
        CapsuleMemory.from_store(
            memory.capsules, receipt["capsule"], **memory.bindings,
            encode=lambda text: [ord(c) for c in text], vocab_size=256,
        )


def test_weaker_implicit_conflict_cannot_demote_stronger_active_fact(memory, context):
    commit(memory, context, fact("alpha", tier="signed_current"))
    commit(memory, context, fact("beta", tier="legacy"))
    by_object = edges(memory, context)
    assert by_object[concept_token("alpha")]["status"] == "active"
    assert by_object[concept_token("beta")]["status"] == "contested"
    preamble = memory.retrieve(context, "gate code").preamble()
    verified, contested = preamble.split("Conflicting semantic memory")
    assert "gate code has_property alpha" in verified
    assert "gate code has_property beta" not in verified
    assert "gate code has_property beta" in contested


def test_weaker_explicit_supersession_refuses_without_any_publication(memory, context):
    commit(memory, context, fact("alpha", tier="signed_current"))
    before = memory.load(context)
    target = before[0]["edges"][0]["edge_id"]
    prepared = []
    # A successful earlier proposal in the same batch must not leak on failure.
    with pytest.raises(ValueError, match="weaker.*supersede.*stronger"):
        commit(
            memory, context, fact("unrelated", key="other fact"),
            fact("beta", tier="legacy", supersedes=(target,)),
            prepare_derived_handles=lambda *args: prepared.append(args) or {},
        )
    assert memory.load(context) == before
    assert not prepared


def test_explicit_same_authority_supersession_resolves_a_contested_fact(memory, context):
    commit(memory, context, fact("alpha"), fact("beta"))
    targets = tuple(edge["edge_id"] for edge in memory.load(context)[0]["edges"])
    commit(memory, context, fact("gamma", supersedes=targets))
    by_object = edges(memory, context)
    assert by_object[concept_token("gamma")]["status"] == "active"
    assert {by_object[concept_token(v)]["status"] for v in ("alpha", "beta")} == {"superseded"}


@pytest.mark.parametrize("change", ["unknown", "key", "subject", "relation", "missing_key"])
def test_explicit_supersession_is_bound_to_the_same_fact(memory, context, change):
    commit(memory, context, fact("alpha"))
    before = memory.load(context)
    target = before[0]["edges"][0]["edge_id"]
    candidate = fact("beta", tier="signed_current", supersedes=(target,))
    updates = {
        "unknown": {"supersedes": ("0" * 64,)},
        "key": {"conflict_key": "different code"},
        "subject": {"subject": "other gate"},
        "relation": {"relation": "related_to"},
        "missing_key": {"conflict_key": None},
    }
    with pytest.raises(ValueError, match="semantic supersedes references"):
        commit(memory, context, replace(candidate, **updates[change]))
    assert memory.load(context) == before


def test_lower_authority_same_fact_cannot_replace_stronger_evidence(memory, context):
    commit(memory, context, fact("alpha", tier="signed_current"))
    before = memory.load(context)
    with pytest.raises(ValueError, match="weaker.*replace.*stronger"):
        commit(memory, context, replace(fact("alpha"), evidence="new weaker evidence"))
    assert memory.load(context) == before


@pytest.mark.parametrize("key", [None, "different fact"])
def test_replacement_cannot_discard_a_retired_facts_conflict_binding(memory, context, key):
    commit(memory, context, fact("alpha", tier="legacy"))
    commit(memory, context, fact("beta", tier="signed_current"))
    before = memory.load(context)
    with pytest.raises(ValueError, match="cannot replace its conflict binding"):
        commit(memory, context, replace(fact("alpha", tier="legacy", key=key),
                                        evidence="new unscoped evidence"))
    assert memory.load(context) == before


@pytest.mark.parametrize("changed", ["subject", "relation", "no_key"])
def test_unrelated_or_multivalued_facts_do_not_become_conflicts(memory, context, changed):
    first, second = fact("alpha"), fact("beta")
    if changed == "subject":
        second = replace(second, subject="other gate")
    elif changed == "relation":
        second = replace(second, relation="related_to")
    else:
        first, second = replace(first, conflict_key=None), replace(second, conflict_key=None)
    commit(memory, context, first, second)
    assert {edge["status"] for edge in memory.load(context)[0]["edges"]} == {"active"}


def test_legacy_capsule_reads_without_mutation_and_committed_consumers_still_work(memory, context):
    commit(memory, context, fact("alpha", key=None))
    graph, _, revision = memory.load(context)
    legacy = copy.deepcopy(graph)
    for edge in legacy["edges"]:
        for key in ("status", "authority_tier", "conflict_key_digest", "supersedes", "edge_id"):
            edge.pop(key)
    capsule = memory.capsules.put(kind="semantic_base", data=legacy,
                                  provenance={"fixture": "legacy graph"}, **memory.bindings)
    memory.directory.update(Scope.SESSION, context, expected_revision=revision,
                            handles={"semantic-memory": capsule.digest})
    before = memory.load(context)
    result = memory.retrieve(context, "gate code")
    assert result.edges[0]["status"] == "active"
    assert result.edges[0]["authority_tier"] == "session_committed"
    assert result.edges[0]["authority"] == "committed"
    assert memory.load(context) == before


@pytest.mark.parametrize("status", ["superseded", "retracted"])
def test_retired_edges_are_not_expanded_or_rendered(memory, context, status):
    commit(memory, context, fact("alpha"))
    graph, _, revision = memory.load(context)
    graph["edges"][0]["status"] = status
    capsule = memory.capsules.put(kind="semantic_delta", data=graph,
                                  provenance={"fixture": "retired graph"}, **memory.bindings)
    memory.directory.update(Scope.SESSION, context, expected_revision=revision,
                            handles={"semantic-memory": capsule.digest})
    result = memory.retrieve(context, "gate code")
    assert not result.edges
    assert {c["label"] for c in result.concepts} == {"gate code"}
    assert "alpha" not in result.preamble()


@pytest.mark.parametrize("bad", ["digest-string", ["not-a-digest"], [None], ["0" * 64] * 65])
def test_supersession_references_are_bounded_digests(bad):
    with pytest.raises(ValueError):
        fact("alpha", supersedes=bad)
