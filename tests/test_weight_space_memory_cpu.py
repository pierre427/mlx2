"""Host-only activation fitting and semantic-memory lifecycle oracles."""

import dataclasses

import numpy as np
import pytest

from mlx2.experimental.weight_space.memory import (
    CalibrationSplit,
    CapturedActivations,
    MemoryProgramStore,
    MemorySource,
    calibrate_memory_program,
)
from mlx2.experimental.weight_space.programs import create_program
from mlx2.runtime.hyper_directory import DirectoryContext, HyperDirectory, Scope
from mlx2.runtime.semantic_capsules import CapsuleStore
from mlx2.runtime.semantic_memory import SemanticMemory, SemanticProposal

CHECKPOINT = "captured-memory-fixture@sha256:" + "a" * 64
SOURCE_REVISION = "e448f070b84642bfc5d11dc04719e40336326a68"
IDS = ("cal-a", "cal-b", "hold-a", "hold-b")
SPLIT = CalibrationSplit(IDS[:2], IDS[2:])


@pytest.fixture
def memory_setup(tmp_path):
    capsules = CapsuleStore(tmp_path / "capsules")
    directory = HyperDirectory(tmp_path / "directory", capsules)
    memory = SemanticMemory(capsules=capsules, directory=directory, model_binding=CHECKPOINT,
                            tokenizer_binding="fixture-tokenizer-v1", runtime_binding=SOURCE_REVISION)
    context = DirectoryContext(model="fixture", tenant="tenant-a", session="session-a")
    result = memory.commit_after_delivery(
        context, [SemanticProposal("project", "has_property", "blue", 1.0, 1.0,
                                   "reviewed source blue", conflict_key="colour")],
        response_delivered=True, authenticated_tenant=True, expected_revision=0,
    )
    graph, digest, revision = memory.load(context)
    store = MemoryProgramStore(memory)
    source = store.snapshot(context, expected_revision=revision, expected_semantic_digest=digest,
                            edge_ids=(graph["edges"][0]["edge_id"],))
    return memory, context, store, source, result


def captured(target="model.layers.0.proj", *, heldout_scale=1.0):
    inputs = np.array([[1, 0], [0, 1], [2, 0], [0, 2]], dtype=np.float32)
    outputs = inputs @ np.array([[1, -2], [0, 0.5]], dtype=np.float32)
    outputs[2:] *= heldout_scale
    return CapturedActivations(target, IDS, inputs, outputs)


def fitted(source, **kwargs):
    return calibrate_memory_program([captured()], split=SPLIT, source=source,
                                    source_revision=SOURCE_REVISION, rank=2, ridge=0, **kwargs)


def published(memory_setup):
    memory, context, store, source, _ = memory_setup
    calibration = fitted(source)
    receipt = store.publish(context, calibration.program, expected_revision=source.session_revision,
                            expires_at_ns=200, now_ns=100)
    return memory, context, store, source, calibration, receipt


def mount(store, context, receipt, **kwargs):
    return store.mount(context, checkpoint=CHECKPOINT,
                       expected_directory_revision=receipt["directory_revision"],
                       expected_capsule_digest=receipt["capsule_digest"], now_ns=150, **kwargs)


def test_calibration_fits_supplied_activations_and_does_not_claim_recall(memory_setup):
    source = memory_setup[3]
    result = fitted(source)
    a, b = result.program.factor_map["model.layers.0.proj"]
    np.testing.assert_allclose(captured().inputs @ a @ b, captured().target_deltas, atol=1e-6)
    assert result.report["heldout"]["residual_mse"] < 1e-12
    assert result.report["evidence_scope"] == "supplied_activation_reconstruction_only"
    assert result.report["task_quality_evaluated"] is False
    assert result.program.provenance["promotion_scope"] == "session"
    assert result.program.provenance["contains_private_memory"] is True
    assert result.program.approximate is True
    assert MemorySource.from_dict(source.to_dict()) == source


def test_heldout_cannot_choose_layer_or_change_fitted_factors(memory_setup):
    source = memory_setup[3]
    good = captured()
    wrong = CapturedActivations("model.layers.1.proj", IDS, np.zeros_like(good.inputs), good.target_deltas)
    normal = calibrate_memory_program([wrong, good], split=SPLIT, source=source,
                                      source_revision=SOURCE_REVISION, rank=2, ridge=0)
    poisoned = calibrate_memory_program([wrong, captured(heldout_scale=1e6)], split=SPLIT, source=source,
                                        source_revision=SOURCE_REVISION, rank=2, ridge=0)
    assert normal.report["selected_target"] == poisoned.report["selected_target"] == good.target
    for a, b in zip(normal.program.factor_map[good.target], poisoned.program.factor_map[good.target]):
        np.testing.assert_array_equal(a, b)
    assert normal.report["calibration_digest"] == poisoned.report["calibration_digest"]
    assert normal.report["heldout_digest"] != poisoned.report["heldout_digest"]
    assert poisoned.report["heldout"]["residual_mse"] > 1e8


def test_zero_strength_is_exact_identity(memory_setup):
    result = fitted(memory_setup[3], strength=0)
    a, b = result.program.factor_map["model.layers.0.proj"]
    delta = captured().inputs @ a @ b
    np.testing.assert_array_equal(delta, np.zeros_like(delta))
    original = captured().target_deltas
    np.testing.assert_array_equal(original + delta, original)
    assert result.report["strength"] == 0


def test_capture_is_immutable_and_split_leakage_refused(memory_setup):
    capture = captured()
    with pytest.raises(ValueError):
        capture.inputs.setflags(write=True)
    with pytest.raises(ValueError, match="disjoint"):
        CalibrationSplit(("same",), ("same",))
    with pytest.raises(ValueError, match="duplicate"):
        CalibrationSplit(("a", "a"), ("b",))
    with pytest.raises(ValueError, match="exactly"):
        calibrate_memory_program([capture], split=CalibrationSplit(("cal-a",), ("hold-a",)),
                                 source=memory_setup[3], source_revision=SOURCE_REVISION)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_capture_is_refused(bad):
    with pytest.raises(ValueError, match="finite"):
        CapturedActivations("projection", ("a",), [[bad]], [[1]])


def test_publication_and_mount_preserve_authoritative_graph(memory_setup):
    memory, context, store, source, calibration, receipt = published(memory_setup)
    graph, digest, revision = memory.load(context)
    assert digest == source.semantic_digest
    assert revision == source.session_revision + 1
    assert graph["edges"][0]["status"] == "active"
    selected = mount(store, context, receipt)
    assert selected.program.program_id == calibration.program.program_id
    assert selected.receipt["selected"] is True
    assert selected.receipt["engaged"] is selected.receipt["qualified"] is False
    assert selected.receipt["requires_fresh_cache"] is True
    assert store.validate_mount(context, selected, now_ns=151).program.program_id == selected.program.program_id


def test_wrong_memory_checkpoint_and_tenant_refused(memory_setup):
    _, context, store, _, _, receipt = published(memory_setup)
    with pytest.raises(ValueError, match="checkpoint"):
        store.mount(context, checkpoint="wrong@sha256:" + "b" * 64,
                    expected_directory_revision=receipt["directory_revision"],
                    expected_capsule_digest=receipt["capsule_digest"], now_ns=150)
    with pytest.raises(ValueError, match="handle"):
        store.mount(context, checkpoint=CHECKPOINT,
                    expected_directory_revision=receipt["directory_revision"],
                    expected_capsule_digest="f" * 64, now_ns=150)
    with pytest.raises(ValueError, match="revision"):
        mount(store, dataclasses.replace(context, tenant="tenant-b"), receipt)


def test_copied_cross_tenant_handle_still_refuses(memory_setup):
    memory, context, store, _, _, receipt = published(memory_setup)
    other = dataclasses.replace(context, tenant="tenant-b")
    source_digest = memory.load(context)[1]
    memory.directory.update(Scope.SESSION, other, expected_revision=0,
                            handles={"semantic-memory": source_digest, "weight-memory": receipt["capsule_digest"]})
    memory.directory.update(Scope.SESSION, other, expected_revision=1, handles={})
    with pytest.raises(ValueError, match="scope"):
        mount(store, other, receipt)


def test_expiry_and_revocation_invalidate_mounted_program(memory_setup):
    memory, context, store, source, _, receipt = published(memory_setup)
    selected = mount(store, context, receipt)
    with pytest.raises(ValueError, match="expired"):
        store.validate_mount(context, selected, now_ns=200)
    revoked = store.revoke(context, expected_revision=receipt["directory_revision"],
                            expected_capsule_digest=receipt["capsule_digest"])
    assert memory.load(context)[1] == source.semantic_digest
    with pytest.raises(ValueError, match="revision"):
        store.validate_mount(context, selected, now_ns=150)
    with pytest.raises(ValueError, match="revoked"):
        mount(store, context, revoked)


def test_correction_requires_recompile_and_never_rewrites_facts(memory_setup):
    memory, context, store, source, _, receipt = published(memory_setup)
    memory.commit_after_delivery(
        context, [SemanticProposal("project", "has_property", "green", 1.0, 1.0,
                                   "operator correction green", authority_tier="user_confirmed", conflict_key="colour")],
        response_delivered=True, authenticated_tenant=True, expected_revision=receipt["directory_revision"],
    )
    graph, digest, revision = memory.load(context)
    assert digest != source.semantic_digest
    with pytest.raises(ValueError, match="stale"):
        mount(store, context, {**receipt, "directory_revision": revision})
    active = [edge["edge_id"] for edge in graph["edges"] if edge["status"] == "active"]
    current = store.snapshot(context, expected_revision=revision, expected_semantic_digest=digest, edge_ids=active)
    program = fitted(current).program
    replacement = store.publish(context, program, expected_revision=revision, expires_at_ns=250, now_ns=150)
    assert mount(store, context, replacement).program.program_id == program.program_id
    assert memory.load(context)[1] == digest


def test_conflicting_and_retracted_sources_cannot_compile_or_mount(memory_setup):
    memory, context, store, source, _, receipt = published(memory_setup)
    memory.commit_after_delivery(
        context, [SemanticProposal("project", "has_property", "red", 1.0, 1.0,
                                   "conflicting source red", conflict_key="colour")],
        response_delivered=True, authenticated_tenant=True, expected_revision=receipt["directory_revision"],
    )
    graph, digest, revision = memory.load(context)
    with pytest.raises(ValueError, match="conflicting"):
        store.snapshot(context, expected_revision=revision, expected_semantic_digest=digest, edge_ids=source.edge_ids)
    graph["edges"][0]["status"] = "retracted"
    graph["edges"][0]["authority"] = "retracted"
    retracted = memory.capsules.put(kind="semantic_delta", data=graph, parents=(digest,),
                                    provenance={"operation": "test-retraction"}, **memory.bindings)
    memory.directory.update(Scope.SESSION, context, expected_revision=revision,
                            handles={"semantic-memory": retracted.digest})
    with pytest.raises(ValueError, match="retracted"):
        store.snapshot(context, expected_revision=revision + 1, expected_semantic_digest=retracted.digest,
                        edge_ids=(graph["edges"][0]["edge_id"],))


def test_offline_claim_cannot_override_live_source(memory_setup):
    _, context, store, source, _ = memory_setup
    forged = dataclasses.replace(source, tenant="tenant-b")
    program = fitted(forged).program
    with pytest.raises(ValueError, match="source, scope"):
        store.publish(context, program, expected_revision=source.session_revision, expires_at_ns=200, now_ns=100)


def test_stale_publication_and_proposed_source_are_refused(memory_setup):
    memory, context, store, source, _ = memory_setup
    program = fitted(source).program
    memory.directory.update(Scope.SESSION, context, expected_revision=source.session_revision, handles={})
    with pytest.raises(ValueError, match="revision"):
        store.publish(context, program, expected_revision=source.session_revision, expires_at_ns=200, now_ns=100)
    graph, digest, revision = memory.load(context)
    graph["edges"][0]["authority"] = "proposal"
    graph["edges"][0]["authority_tier"] = "legacy"
    graph["edges"][0]["status"] = "contested"
    proposed = memory.capsules.put(kind="semantic_delta", data=graph, parents=(digest,),
                                   provenance={"operation": "test-proposal"}, **memory.bindings)
    memory.directory.update(Scope.SESSION, context, expected_revision=revision,
                            handles={"semantic-memory": proposed.digest})
    with pytest.raises(ValueError, match="uncommitted"):
        store.snapshot(context, expected_revision=revision + 1, expected_semantic_digest=proposed.digest,
                        edge_ids=source.edge_ids)


def test_wrong_program_source_digest_is_not_a_valid_memory_binding(memory_setup):
    _, context, store, source, _ = memory_setup
    program = fitted(source).program
    metadata = program.provenance
    metadata["memory_source"]["semantic_digest"] = "d" * 64
    wrong = create_program(program.factor_map, checkpoint=program.checkpoint, source_revision=program.source_revision,
                           construction=program.construction, provenance=metadata, approximate=True)
    with pytest.raises(ValueError, match="digest"):
        store.publish(context, wrong, expected_revision=source.session_revision, expires_at_ns=200, now_ns=100)


def test_private_scope_cannot_be_stripped_before_publication(memory_setup):
    _, context, store, source, _ = memory_setup
    program = fitted(source).program
    provenance = program.provenance
    provenance["promotion_scope"] = "deployment"
    stripped = create_program(program.factor_map, checkpoint=program.checkpoint, source_revision=program.source_revision,
                              construction=program.construction, provenance=provenance, approximate=True)
    with pytest.raises(ValueError, match="private session scope"):
        store.publish(context, stripped, expected_revision=source.session_revision, expires_at_ns=200, now_ns=100)


def test_active_status_alone_does_not_promote_uncommitted_input(memory_setup):
    memory, context, store, source, _ = memory_setup
    graph, digest, revision = memory.load(context)
    graph["edges"][0]["authority"] = "proposal"
    candidate = memory.capsules.put(kind="semantic_delta", data=graph, parents=(digest,),
                                    provenance={"operation": "test-uncommitted"}, **memory.bindings)
    memory.directory.update(Scope.SESSION, context, expected_revision=revision,
                            handles={"semantic-memory": candidate.digest})
    with pytest.raises(ValueError, match="uncommitted"):
        store.snapshot(context, expected_revision=revision + 1, expected_semantic_digest=candidate.digest,
                        edge_ids=source.edge_ids)


def test_reserved_semantic_handle_cannot_be_published_or_revoked(memory_setup):
    memory, context, store, source, _ = memory_setup
    graph, digest, revision = memory.load(context)
    with pytest.raises(ValueError, match="reserved"):
        store.publish(context, fitted(source).program, expected_revision=revision,
                      expires_at_ns=200, now_ns=100, handle="semantic-memory")
    with pytest.raises(ValueError, match="reserved"):
        store.revoke(context, expected_revision=revision, expected_capsule_digest=digest, handle="semantic-memory")
    assert memory.load(context) == (graph, digest, revision)


def test_revoke_refuses_a_handle_that_is_not_a_derived_program(memory_setup):
    memory, context, store, _, _ = memory_setup
    _, digest, revision = memory.load(context)
    memory.directory.update(Scope.SESSION, context, expected_revision=revision,
                            handles={"weight-memory": digest})
    with pytest.raises(ValueError, match="not a derived memory program"):
        store.revoke(context, expected_revision=revision + 1, expected_capsule_digest=digest)
    with pytest.raises(ValueError, match="not a derived memory program"):
        store.revoke(context, expected_revision=revision + 1, expected_capsule_digest=None, handle="absent")
    assert memory.load(context)[1] == digest


def test_unregularized_fit_ignores_rounding_noise_singular_values():
    from mlx2.experimental.weight_space.memory import _fit

    inputs = np.ones((3, 2), dtype=np.float32)
    residuals = np.array([[1], [2], [3]], dtype=np.float32)
    a, b = _fit(inputs, residuals, rank=8, ridge=0)
    np.testing.assert_allclose(inputs @ a @ b, np.full((3, 1), 2.0), rtol=1e-6)


def test_regularized_fit_keeps_small_singular_directions():
    from mlx2.experimental.weight_space.memory import _fit

    # The rank cutoff is for ridge=0 only; ridge>0 keeps the 0.01 direction.
    inputs = np.diag([1e14, 0.01]).astype(np.float32)
    residuals = np.array([[0], [1]], dtype=np.float32)
    a, b = _fit(inputs, residuals, rank=8, ridge=1e-4)
    np.testing.assert_allclose(inputs.astype(np.float64) @ a @ b, [[0.0], [0.5]], atol=1e-6)
