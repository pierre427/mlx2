"""Real private config/NPZ/SSD slice tools without model or GPU imports."""

import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest

from mlx2.experimental.weight_space.experiment_config import (
    BoundExpertFiles,
    configure_experiment_assets,
)
from mlx2.experimental.weight_space.experts import (
    ExpertKey,
    ExpertMaskPlan,
    ExpertRecord,
    LayerExpertMask,
)
from mlx2.experimental.weight_space.tool_service import (
    CONFIG_SCHEMA,
    WeightSpaceToolService,
    service_from_operator_config,
)

CHECKPOINT = "config-fixture@" + "a" * 64
SOURCE = "e448f070b84642bfc5d11dc04719e40336326a68"


def private_json(root, name, value):
    path = root / name
    path.write_text(json.dumps(value))
    path.chmod(0o600)
    return path


def private_arrays(root, name, **arrays):
    path = root / name
    np.savez(path, **arrays)
    path.chmod(0o600)


def assets(root, **kwargs):
    root.chmod(0o700)
    private_json(root, "experiments.json", {"schema": "mlx2.weight-experiment-assets.v1", **kwargs})
    private_json(root, "operator.json", {"schema": CONFIG_SCHEMA, "session_id": "s", "captures": [],
        "programs": {}, "experiments": {"assets": "experiments.json", "checkpoint": CHECKPOINT,
                                         "source_revision": SOURCE}})
    return service_from_operator_config(root, "operator.json")


def test_offline_config_registers_real_excision_healing_and_quality_tools(tmp_path):
    x = np.array([[1, 0], [0, 1], [1, 1], [2, 3]], dtype="f4")
    private_arrays(tmp_path, "captures.npz", x=x, reference=x * 1.5,
                   logits=np.array([[2., 0.], [0., 2.]]), worse=np.zeros((2, 2)), labels=np.array([0, 1]))
    service = assets(tmp_path, excision_inventories=[{"inventory_id": "decoder", "layer_count": 24, "hidden_size": 2}],
        excision_captures=[{"capture_id": "boundary", "inventory_id": "decoder", "removed_layers": [21],
                           "arrays": "captures.npz", "excised_outputs": "x", "reference_outputs": "reference",
                           "sample_ids": ["a", "b", "c", "d"], "calibration_ids": ["a", "b"],
                           "heldout_ids": ["c", "d"], "input_digest": "c" * 64}],
        quality_captures=[{"capture_id": "ppl", "arrays": "captures.npz", "reference_logits": "logits",
                           "candidate_logits": "worse", "next_token_ids": "labels",
                           "tokenizer_fingerprint": "toy-tokenizer", "input_digest": "d" * 64}])
    service.compile_layer_excision("decoder", "remove21", [21])
    result = service.fit_excision_healing("remove21", "boundary", "heal22", rank=2, ridge=0)
    assert not result.get("qualified", False)
    assert service.inspect_layer_excision("heal22")["healing_boundary"] == 22
    quality = service.evaluate_quantization_quality("ppl")
    assert quality["delta_perplexity"] > 0
    assert not quality["qualified"]


def expert_fixture(root):
    data = b"ABCDEFGHabcdefgh"
    path = root / "expert.bin"
    path.write_bytes(data)
    plan = ExpertMaskPlan(CHECKPOINT, SOURCE, "b" * 64, (LayerExpertMask(23, 2, 1, (0, 1)),),
                          "native-test-bytes", "cpu-byte-cache", "c" * 64)
    records = tuple(ExpertRecord(ExpertKey(23, i), "shard", i * 8, 8,
        hashlib.sha256(data[i * 8:i * 8 + 8]).hexdigest(), plan.source_fingerprint) for i in range(2))
    return path, plan, records


def test_expert_config_reads_real_ssd_slices_with_budget_and_mutation_checks(tmp_path):
    path, plan, records = expert_fixture(tmp_path)
    private_arrays(tmp_path, "scores.npz", importance=np.array([2., 1.]))
    service = assets(tmp_path, expert_sources=[{"source_id": "model", "plan": plan.to_dict(),
        "arrays": "scores.npz", "importance": {"23": "importance"}, "records": [asdict(r) for r in records],
        "files": {"shard": str(path)}, "residency": {"budget_bytes": 12, "chunk_bytes": 4}}])
    service.prefetch_experts("model", [{"layer": 23, "expert": 0, "projection": "bundle"}])
    service.advance_expert_stream("model", max_chunks=2)
    controller = service._expert_sources["model"]["controller"]
    with controller.acquire(records[0].key) as lease:
        assert lease.read(0, 4) == b"ABCD"
    path.write_bytes(b"changed contents")
    with pytest.raises(Exception, match="source"):
        service.advance_expert_stream("model")


def test_bound_reader_refuses_unregistered_ranges_symlinks_and_replacement(tmp_path):
    path, plan, records = expert_fixture(tmp_path)
    bound = BoundExpertFiles({"shard": str(path)}, records, plan.source_fingerprint)
    with pytest.raises(ValueError, match="bounded indexed"):
        bound.read(records[0], 7, 2)
    linked = tmp_path / "link"
    linked.symlink_to(path)
    with pytest.raises(ValueError, match="regular"):
        BoundExpertFiles({"shard": str(linked)}, records, plan.source_fingerprint)
    path.unlink()
    path.write_bytes(b"ABCDEFGHabcdefgh")
    with pytest.raises(ValueError, match="revision"):
        bound.validate()


def test_asset_config_refuses_paths_permissions_and_unknown_fields(tmp_path):
    tmp_path.chmod(0o700)
    service = WeightSpaceToolService(session_id="s")
    with pytest.raises(ValueError, match="relative"):
        configure_experiment_assets(service, tmp_path, {"assets": "../outside.json"},
                                    checkpoint=CHECKPOINT, source_revision=SOURCE)
    file = private_json(tmp_path, "bad.json", {"schema": "mlx2.weight-experiment-assets.v1", "python": "ignored"})
    with pytest.raises(ValueError, match="fields"):
        configure_experiment_assets(service, tmp_path, {"assets": "bad.json"},
                                    checkpoint=CHECKPOINT, source_revision=SOURCE)
    file.chmod(0o644)
    with pytest.raises(ValueError, match="owner-private"):
        configure_experiment_assets(service, tmp_path, {"assets": "bad.json"},
                                    checkpoint=CHECKPOINT, source_revision=SOURCE)
