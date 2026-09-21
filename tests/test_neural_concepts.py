"""CPU oracles for recurrent concept state and learned cross-attention."""

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mlx2.runtime.hyper_directory import DirectoryContext, HyperDirectory
from mlx2.runtime.neural_concepts import (
    NEURAL_CONCEPT_SCHEMA,
    ConceptCrossAttention,
    NeuralConceptArtifact,
    NeuralConceptMemory,
    RecurrentConceptEncoder,
    label_features,
    validate_state_document,
)
from mlx2.runtime.semantic_capsules import CapsuleStore
from mlx2.runtime.semantic_memory import (
    RELATIONS,
    SemanticMemory,
    SemanticProposal,
    concept_token,
)
from mlx2.semantic_sidecar import SemanticServingMiddleware


def write_artifact(root: Path, *, hidden_dim=16, state_dim=8, feature_dim=16):
    rng = np.random.default_rng(7)
    arrays = {
        "input_weight": rng.normal(0, 0.1, (feature_dim, state_dim)).astype("f4"),
        "input_bias": np.zeros(state_dim, dtype="f4"),
        "relation_weight": np.stack(
            [np.eye(state_dim, dtype="f4") for _ in sorted(RELATIONS)]
        ),
        "gru_z_weight": np.zeros((2 * state_dim, state_dim), dtype="f4"),
        "gru_z_bias": np.zeros(state_dim, dtype="f4"),
        "gru_r_weight": np.zeros((2 * state_dim, state_dim), dtype="f4"),
        "gru_r_bias": np.zeros(state_dim, dtype="f4"),
        "gru_h_weight": np.concatenate(
            [np.eye(state_dim, dtype="f4"), np.zeros((state_dim, state_dim), dtype="f4")]
        ),
        "gru_h_bias": np.zeros(state_dim, dtype="f4"),
        "key_projection": rng.normal(0, 0.1, (state_dim, hidden_dim)).astype("f4"),
        "value_projection": rng.normal(0, 0.1, (state_dim, hidden_dim)).astype("f4"),
        "output_gate": np.asarray([0.0], dtype="f4"),
    }
    weights = root / "weights.npz"
    np.savez(weights, **arrays)
    manifest = {
        "schema": NEURAL_CONCEPT_SCHEMA,
        "bindings": {"model": "model", "tokenizer": "tokenizer", "runtime": "runtime"},
        "feature_dim": feature_dim,
        "state_dim": state_dim,
        "hidden_dim": hidden_dim,
        "message_rounds": 2,
        "attention_temperature": 0.07,
        "max_graph_concepts": 64,
        "relations": sorted(RELATIONS),
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "training": {"kind": "test-fixture"},
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["fingerprint"] = hashlib.sha256(encoded + weights.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    return root


@pytest.fixture
def artifact():
    with tempfile.TemporaryDirectory() as directory:
        yield NeuralConceptArtifact.load(
            write_artifact(Path(directory)),
            model_binding="model",
            tokenizer_binding="tokenizer",
            runtime_binding="runtime",
        )


def graph():
    forest = concept_token("forest walk")
    leaves = concept_token("wet leaves")
    scent = concept_token("earthy scent")
    return {
        "concepts": {
            forest: {"id": forest, "label": "forest walk", "aliases": []},
            leaves: {"id": leaves, "label": "wet leaves", "aliases": []},
            scent: {"id": scent, "label": "earthy scent", "aliases": []},
        },
        "edges": [
            {"subject": forest, "relation": "related_to", "object": leaves, "authority": "committed"},
            {"subject": leaves, "relation": "has_property", "object": scent, "authority": "committed"},
        ],
    }


def test_label_features_are_stable_bounded_and_surface_sensitive():
    first = label_features("Walking in a Forest", 32)
    assert np.array_equal(first, label_features(" walking  in a forest ", 32))
    assert not np.array_equal(first, label_features("smelling leaves", 32))
    assert np.linalg.norm(first) == pytest.approx(1.0)


def test_recurrent_edges_change_only_message_receivers(artifact):
    encoder = RecurrentConceptEncoder(artifact)
    encoded = {item.concept_id: item for item in encoder.encode_graph(graph())}
    forest = concept_token("forest walk")
    scent = concept_token("earthy scent")
    assert encoded[forest].key_state != encoded[forest].value_state
    assert encoded[scent].key_state == encoded[scent].value_state
    document = encoder.state_document(graph(), semantic_capsule="a" * 64)
    assert validate_state_document(document, artifact) == tuple(encoded.values())


def test_cross_attention_is_bounded_and_finite(artifact):
    concepts = RecurrentConceptEncoder(artifact).encode_graph(graph())
    queries = np.ones((5, artifact.hidden_dim), dtype=np.float32)
    output, weights = ConceptCrossAttention(artifact).apply(queries, concepts)
    assert output.shape == queries.shape
    assert weights.shape == (5, 3)
    assert np.isfinite(output).all()
    assert np.allclose(weights.sum(axis=-1), 1.0)


def test_artifact_and_state_bindings_fail_closed(artifact):
    with pytest.raises(ValueError, match="binding mismatch"):
        NeuralConceptArtifact.load(
            artifact.root,
            model_binding="wrong",
            tokenizer_binding="tokenizer",
            runtime_binding="runtime",
        )
    document = RecurrentConceptEncoder(artifact).state_document(
        graph(), semantic_capsule="b" * 64
    )
    document["artifact_fingerprint"] = "wrong"
    with pytest.raises(ValueError, match="artifact mismatch"):
        validate_state_document(document, artifact)


def test_neural_state_is_derived_from_and_bound_to_semantic_capsule(artifact, tmp_path):
    capsules = CapsuleStore(tmp_path / "capsules")
    directory = HyperDirectory(tmp_path / "directory", capsules)
    semantic = SemanticMemory(
        capsules=capsules,
        directory=directory,
        model_binding="model",
        tokenizer_binding="tokenizer",
        runtime_binding="runtime",
    )
    context = DirectoryContext(model="qwen9b", tenant="tenant", session="session")
    semantic.commit_after_delivery(
        context,
        [SemanticProposal("forest walk", "has_property", "earthy scent", 0.99, 0.8, "turn")],
        response_delivered=True,
        authenticated_tenant=True,
    )
    memory = NeuralConceptMemory(semantic, artifact)
    receipt = memory.rebuild(context)
    assert receipt["committed"] and receipt["concepts"] == 2
    assert len(memory.load(context)) == 2

    semantic.commit_after_delivery(
        context,
        [SemanticProposal("forest walk", "related_to", "wet leaves", 0.99, 0.8, "turn-2")],
        response_delivered=True,
        authenticated_tenant=True,
    )
    with pytest.raises(ValueError, match="stale"):
        memory.load(context)


def test_neural_middleware_rebuilds_after_delivery_then_injects_state(artifact, tmp_path):
    capsules = CapsuleStore(tmp_path / "capsules")
    directory = HyperDirectory(tmp_path / "directory", capsules)
    semantic = SemanticMemory(
        capsules=capsules,
        directory=directory,
        model_binding="model",
        tokenizer_binding="tokenizer",
        runtime_binding="runtime",
    )
    middleware = SemanticServingMiddleware(
        semantic,
        model_scope="model",
        neural_memory=NeuralConceptMemory(semantic, artifact),
        bridge_mode="neural",
    )
    write, state = middleware.prepare(
        {
            "session_id": "walk",
            "messages": [
                {"role": "user", "content": "Remember that Cedar Loop is earthy scent."}
            ],
        },
        tenant_id="alice",
        authenticated_tenant=True,
    )
    assert "_mlx2_neural_concepts" not in write
    committed = middleware.complete(state, "Noted.")
    assert committed["neural_state"]["committed"]

    prepared, recalled = middleware.prepare(
        {
            "session_id": "walk",
            "messages": [
                {"role": "user", "content": "What scent belongs to Cedar Loop?"}
            ],
        },
        tenant_id="alice",
        authenticated_tenant=True,
    )
    assert recalled.neural_concepts == 2
    assert prepared["_mlx2_neural_concepts"]["artifact_fingerprint"] == artifact.fingerprint
    assert prepared["messages"][0]["role"] == "user"
    assert middleware.receipt(recalled)["neural_observed_used"] is True
