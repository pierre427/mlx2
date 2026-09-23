"""CPU oracles for recurrent concept state and learned cross-attention."""

import hashlib
import json
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

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
from mlx2.serving import ServingEngine


def write_artifact(
    root: Path, *, hidden_dim=16, state_dim=8, feature_dim=16, decode_steps=0
):
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
    if decode_steps:
        arrays["decode_value_projections"] = rng.normal(
            0, 0.1, (decode_steps, state_dim, hidden_dim)
        ).astype("f4")
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
        "training": {
            "kind": "test-fixture",
            **({"max_decode_steps": decode_steps} if decode_steps else {}),
        },
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
    assert weights.shape == (1, 3)
    assert np.isfinite(output).all()
    assert np.allclose(weights.sum(axis=-1), 1.0)
    assert np.array_equal(output[:-1], queries[:-1])
    assert 0 < np.linalg.norm(output[-1] - queries[-1]) < 1


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


def test_artifact_accepts_declared_bounded_decode_capsule(tmp_path):
    loaded = NeuralConceptArtifact.load(
        write_artifact(tmp_path, decode_steps=4),
        model_binding="model",
        tokenizer_binding="tokenizer",
        runtime_binding="runtime",
    )
    assert loaded.arrays["decode_value_projections"].shape == (4, 8, 16)


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
    assert committed["revision"] == committed["neural_state"]["revision"] == 1

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


def test_neural_prepare_failure_does_not_publish_semantic_state(artifact, tmp_path, monkeypatch):
    capsules = CapsuleStore(tmp_path / "capsules")
    directory = HyperDirectory(tmp_path / "directory", capsules)
    semantic = SemanticMemory(
        capsules=capsules, directory=directory,
        model_binding="model", tokenizer_binding="tokenizer", runtime_binding="runtime",
    )
    neural = NeuralConceptMemory(semantic, artifact)
    middleware = SemanticServingMiddleware(
        semantic, model_scope="model", neural_memory=neural, bridge_mode="neural"
    )
    _, state = middleware.prepare(
        {"session_id": "walk", "messages": [
            {"role": "user", "content": "Remember that Cedar Loop is earthy scent."}
        ]},
        tenant_id="alice", authenticated_tenant=True,
    )

    def fail_prepare(_graph, _digest):
        raise RuntimeError("injected neural preparation failure")

    monkeypatch.setattr(neural, "prepare", fail_prepare)
    result = middleware.complete(state, "Noted.")
    assert result == {"committed": False, "reason": "post-delivery-commit-failed"}
    graph, digest, revision = semantic.load(state.context)
    assert not graph["concepts"] and digest is None and revision == 0


def test_engine_binds_neural_bridge_only_after_adapter_is_ready():
    configured = []

    class Adapter:
        def configure_neural_concept_bridge(self, artifact):
            configured.append(artifact)

        def diagnostics(self):
            return {"neural_concept_bridge": {"state": "configured-unqualified"}}

    engine = object.__new__(ServingEngine)
    engine.ready = threading.Event()
    engine.ready.set()
    engine.prompt_lock = threading.Lock()
    engine.lock = threading.Lock()
    engine.adapter = Adapter()
    engine.snapshot = {"state": "ready", "settings": {"route": "ordinary"}}
    artifact = SimpleNamespace(fingerprint="artifact")
    engine.configure_neural_concept_bridge(artifact, timeout=0.1)
    assert configured == [artifact]
    assert engine.snapshot["execution"]["neural_concept_bridge"]["state"] == (
        "configured-unqualified"
    )


@pytest.mark.parametrize("route", ["prompt_lookup", "native_mtp", "external_draft"])
def test_engine_refuses_a_neural_bridge_its_route_cannot_apply(route):
    # Only ordinary decode applies concept prefill and decode inputs; the
    # server turns this ValueError into a startup refusal.
    configured = []

    class Adapter:
        def configure_neural_concept_bridge(self, artifact):
            configured.append(artifact)

        def diagnostics(self):
            return {}

    engine = object.__new__(ServingEngine)
    engine.ready = threading.Event()
    engine.ready.set()
    engine.prompt_lock = threading.Lock()
    engine.lock = threading.Lock()
    engine.adapter = Adapter()
    engine.snapshot = {"state": "ready", "settings": {"route": route}}
    with pytest.raises(ValueError, match=f"{route} route cannot apply"):
        engine.configure_neural_concept_bridge(SimpleNamespace(fingerprint="a"), timeout=0.1)
    assert configured == []


def test_neural_concept_request_is_refused_where_the_route_would_drop_it(monkeypatch):
    # Prompt lookup swallowed the concept prefill inputs while the receipt
    # said "applied" and the engagement counter moved.
    from route_harness import (
        make_engine, make_external_engine, patch_host, run, tiny_muse_dflash,
        tiny_qwen38_mtp,
    )

    patch_host(monkeypatch)
    inner, vocab = tiny_qwen38_mtp()
    seen = []

    class ConceptModel:
        def __init__(self, model):
            self.model_ = model

        def __call__(self, inputs, cache=None, deep_concept_memory=None, **kwargs):
            if deep_concept_memory is not None:
                seen.append(True)
            return self.model_(inputs, cache=cache, **kwargs)

        def __getattr__(self, name):
            return getattr(self.model_, name)

    class Bridge:
        def neural_concept_prefill(self, tokens, payload, *, prefill_step):
            return {
                "deep_concept_memory": {"keys": 1},
                "receipt": {"schema": "tiny.neural-concept.v1", "status": "applied"},
            }

    muse, draft, muse_vocab = tiny_muse_dflash()
    model = ConceptModel(inner)
    engines = {
        "ordinary": lambda: make_engine(model, vocab, mtp=False, adapter_mixin=Bridge),
        "prompt_lookup": lambda: make_engine(
            model, vocab, mtp=False, prompt_lookup=True, adapter_mixin=Bridge
        ),
        "native_mtp": lambda: make_engine(model, vocab, mtp=True, adapter_mixin=Bridge),
        "external_draft": lambda: make_external_engine(muse, draft, muse_vocab),
    }
    prompt = [(7 * i + 3) % (vocab - 2) + 1 for i in range(12)]
    request = {"tokens": prompt, "max_tokens": 4, "temperature": 0,
               "_mlx2_neural_concepts": {"concepts": [{"id": "c"}]}}
    for route, factory in engines.items():
        seen.clear()
        engine = factory()
        if route == "external_draft":
            type(engine.adapter).neural_concept_prefill = Bridge.neural_concept_prefill
        try:
            output = run(engine, request)
            engagements = engine.counts.get("neural_concept_bridge_engagements", 0)
            alive = engine.thread.is_alive()
        finally:
            engine.close()
        assert alive, route
        if route == "ordinary":
            assert output["receipt"]["neural_concept_bridge"]["status"] == "applied"
            assert engagements == 1 and seen
        else:
            assert output.get("status") == 400, (route, output)
            assert f"{route} route cannot apply" in output["error"]
            assert engagements == 0 and not seen
