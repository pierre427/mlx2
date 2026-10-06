"""Persisted tiny artifacts exercise adapter preflight, loading, policy and identity."""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)
from mlx.utils import tree_flatten
from test_lilicorr_artifact_metadata import artifact as lilicorr_artifact  # noqa: F401
from test_xpress_artifact_metadata import artifact  # noqa: F401 - shared pytest fixture

from mlx2.adapters.standard_decoder import StandardDecoderAdapter
from mlx2.contracts import Capability, StatePlane
from mlx2.runtime.models.standard_decoder import Model, ModelArgs


@pytest.fixture
def pack(artifact, monkeypatch):  # noqa: F811 - pytest fixture dependency
    from transformers import AutoTokenizer

    draft, target, _, _ = artifact
    config = json.loads((target / "config.json").read_text())
    config.update(
        intermediate_size=7,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        dtype="bfloat16",
        eos_token_id=8,
    )
    (target / "config.json").write_text(json.dumps(config))
    mx.random.seed(19)
    model = Model(ModelArgs.from_dict(config))
    mx.save_safetensors(
        str(target / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    tokenizer = SimpleNamespace(
        eos_token_id=8,
        chat_template=None,
        get_vocab=lambda: {chr(65 + i): i for i in range(9)},
    )
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(mx, "clear_cache", lambda: None)
    return draft, target


def test_tiny_standard_adapter_loads_xpress_and_forwards_adaptive_policy(pack):
    draft, target = pack
    policy = {
        "verification_costs": [1, 2, 4],
        "mode": "per_request",
        "verification_costs_by_cohort": {"2": [1, 3, 8]},
    }
    adapter = StandardDecoderAdapter(
        str(target),
        execution_policy={
            "draft_model": str(draft),
            "num_draft": 2,
            "xpress_num_passes": 2,
            "adaptive_verification": policy,
        },
    )
    assert Capability.EXTERNAL_DRAFT in adapter.descriptor.capabilities
    assert StatePlane.DRAFT in adapter.descriptor.state_planes
    assert adapter.descriptor.variant == "external-xpress"
    assert adapter.diagnostics()["qualification"] == "unqualified"
    assert adapter.profile_name(False) == "qwen3-apcv2-xpress"
    assert (
        adapter.execution_config(max_lanes=2, prefill_step=3)["backend"]
        == "external_draft"
    )
    engine = adapter.create_external_batch(completion_batch_size=2)
    assert engine.adaptive_policy.mode == "per_request"
    assert engine.adaptive_policy.costs(2) == (1, 3, 8)
    engine.insert([[1, 2]], max_tokens=[2])
    responses = []
    for _ in range(10):
        _, steps = engine.next()
        responses.extend(steps)
        if not engine.lanes:
            break
    assert len(responses) == 2
    assert responses[-1].speculative_receipt["draft_settings"]["xpress_num_passes"] == 2
    engine.close()


@pytest.mark.parametrize(
    "settings,error",
    [
        ({"draft_quantization": {"bits": 4, "group_size": 64}}, "quantization"),
        ({"xpress_num_passes": True}, "passes"),
        ({"num_draft": 4}, "block size"),
        ({"adaptive_verification": {"verification_costs": [1, 2]}}, "cost"),
        (
            {
                "adaptive_verification": {
                    "verification_costs": [1, 2, 4],
                    "draft_cost": "oops",
                }
            },
            "cost",
        ),
        ({"draft_attention_windows": []}, "one entry"),
        ({"draft_attention_windows": [True]}, "positive integers"),
        ({"draft_attention_windows": [0]}, "positive integers"),
        ({"draft_revision": "0" * 64}, "revision mismatch"),
        ({"target_fingerprint": "0" * 64}, "fingerprint mismatch"),
    ],
)
def test_bad_policy_rejected_before_target_allocation(
    pack, monkeypatch, settings, error
):
    import mlx2.runtime.models.standard_decoder as runtime

    draft, target = pack

    def forbidden(*a, **kw):
        raise AssertionError("target allocation preceded policy rejection")

    monkeypatch.setattr(runtime, "Model", forbidden)
    with pytest.raises(ValueError, match=error):
        StandardDecoderAdapter(
            str(target),
            execution_policy={"draft_model": str(draft), "num_draft": 2, **settings},
        )


def test_longest_first_requires_exact_target_geometry_before_allocation(
    pack, monkeypatch
):
    import mlx2.runtime.models.standard_decoder as runtime

    draft, target = pack

    def forbidden(*_args, **_kwargs):
        raise AssertionError("target allocation preceded exact-prefix rejection")

    monkeypatch.setattr(runtime, "Model", forbidden)
    with pytest.raises(ValueError, match="target_verify_row_exact"):
        StandardDecoderAdapter(
            str(target),
            execution_policy={
                "draft_model": str(draft),
                "num_draft": 2,
                "continuation_pool": {
                    "verification_algorithm": "longest-first-exact-prefix-v1"
                },
            },
        )


@pytest.mark.parametrize(
    "architecture", ["Qwen3XPressModel", "DFlashDraftModel", "DFlash2DraftModel"]
)
def test_qwen3_sidecar_artifacts_are_not_target_adapters(tmp_path, architecture):
    from mlx2.adapters.standard_decoder import inspect_artifact

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "architectures": [architecture]})
    )
    with pytest.raises(ValueError, match="sidecars"):
        inspect_artifact(tmp_path)


def test_pass_changes_bind_distinct_paired_cache_identities(pack):
    draft, target = pack
    first = StandardDecoderAdapter(
        str(target),
        execution_policy={
            "draft_model": str(draft),
            "num_draft": 2,
            "xpress_num_passes": 1,
        },
    )
    second = StandardDecoderAdapter(
        str(target),
        execution_policy={
            "draft_model": str(draft),
            "num_draft": 2,
            "xpress_num_passes": 2,
        },
    )
    assert first.identity["target_fingerprint"] == second.identity["target_fingerprint"]
    assert first.identity["draft_fingerprint"] == second.identity["draft_fingerprint"]
    assert first.identity["fingerprint"] != second.identity["fingerprint"]
    assert first.layout != second.layout
    engine = first.create_external_batch()
    engine.insert([[1, 2]], max_tokens=[2])
    lane = engine.lanes[0]
    engine._prefill(lane)
    paired = engine._sidecar(lane)
    with pytest.raises(ValueError, match="revision"):
        paired.validate(second.identity["fingerprint"], len(lane.history))
    engine.close()


def test_qwen38_preflight_validates_adaptive_policy_and_forwarding(
    tmp_path, monkeypatch
):
    from test_qwen38_dflash2_adapter import _pins, _write_artifacts

    import mlx2.runtime.external_speculative as execution
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter, inspect_external_policy

    target, draft = _write_artifacts(tmp_path)
    policy = {"verification_costs": [1, 2, 4], "mode": "per_request"}
    inspect_external_policy(
        {**_pins(target, draft), "num_draft": 2, "adaptive_verification": policy},
        target,
    )
    with pytest.raises(ValueError, match="cost"):
        inspect_external_policy(
            {
                **_pins(target, draft),
                "num_draft": 2,
                "adaptive_verification": {"verification_costs": [1, 2]},
            },
            target,
        )
    adapter = object.__new__(Qwen3827BAdapter)
    adapter.draft_model = object()
    adapter.model = object()
    adapter.identity = {"fingerprint": "test-revision"}
    adapter.external_policy = {"num_draft": 2, "adaptive_verification": policy}
    captured = []
    monkeypatch.setattr(
        execution, "ExternalDraftBatchGenerator", lambda *a, **kw: captured.append(kw)
    )
    adapter.create_external_batch()
    assert captured[0]["adaptive_verification"] == policy
    assert captured[0]["ready_drain"] == "all"
    assert captured[0]["binding"] == "test-revision"


def test_valid_window_override_binds_settings_and_distinct_cache_identity(pack):
    draft, target = pack
    ordinary = StandardDecoderAdapter(
        str(target), execution_policy={"draft_model": str(draft), "num_draft": 2}
    )
    windowed = StandardDecoderAdapter(
        str(target),
        execution_policy={
            "draft_model": str(draft),
            "num_draft": 2,
            "draft_attention_windows": [2],
        },
    )
    assert ordinary.identity["fingerprint"] != windowed.identity["fingerprint"]
    assert windowed.draft_model.receipt_settings["draft_attention_windows"] == [2]
    assert windowed.draft_model.make_cache()[0].max_size == 2


def test_prepared_revision_pin_detects_same_size_same_mtime_tensor_mutation(
    pack, monkeypatch
):
    import mlx2.runtime.models.standard_decoder as runtime

    draft, target = pack
    path = Path(__file__).parents[1] / "scripts/prepare_parallel_draft_policy.py"
    spec = importlib.util.spec_from_file_location("tiny_parallel_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = module.prepare(
        target,
        draft,
        num_draft=2,
        attention_windows=[2],
        adaptive={"verification_costs": [1, 2, 4], "mode": "per_request"},
    )
    assert len(policy["draft_revision"]) == len(policy["target_fingerprint"]) == 64
    shard = draft / "model.safetensors"
    old = shard.stat()
    with shard.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"\x01")
    os.utime(shard, ns=(old.st_atime_ns, old.st_mtime_ns))

    def forbidden(*a, **k):
        raise AssertionError("target allocated before source revision rejection")

    monkeypatch.setattr(runtime, "Model", forbidden)
    with pytest.raises(ValueError, match="source revision mismatch"):
        StandardDecoderAdapter(str(target), execution_policy=policy)


@pytest.fixture
def lili_pack(lilicorr_artifact, monkeypatch):  # noqa: F811 - pytest fixture dependency
    from transformers import AutoTokenizer

    draft, target, _, _ = lilicorr_artifact
    config = json.loads((target / "config.json").read_text())
    config.update(
        intermediate_size=7,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        dtype="bfloat16",
        eos_token_id=8,
    )
    (target / "config.json").write_text(json.dumps(config))
    mx.random.seed(19)
    model = Model(ModelArgs.from_dict(config))
    mx.save_safetensors(
        str(target / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    tokenizer = SimpleNamespace(
        eos_token_id=8,
        chat_template=None,
        get_vocab=lambda: {chr(65 + i): i for i in range(9)},
    )
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(mx, "clear_cache", lambda: None)
    return draft, target


def test_tiny_lilicorr_adapter_dispatch_serves_and_keeps_pending_prefill(lili_pack):
    draft, target = lili_pack
    adapter = StandardDecoderAdapter(
        str(target),
        execution_policy={
            "draft_model": str(draft),
            "num_draft": 2,
            "draft_attention_windows": [2],
            "adaptive_verification": {
                "verification_costs": [1, 2, 4],
                "mode": "per_request",
            },
        },
    )
    assert adapter.descriptor.variant == "external-lilicorr"
    assert adapter.draft_model.requires_pending_context_at_prefill
    engine = adapter.create_external_batch()
    engine.insert([[1, 2, 3]], max_tokens=[4])
    lane = engine.lanes[0]
    engine._prefill(lane)
    assert lane.tail.shape[1] == 2
    engine._sidecar(lane).validate(adapter.identity["fingerprint"], len(lane.history))
    responses = []
    for _ in range(10):
        _, steps = engine.next()
        responses.extend(steps)
        if not engine.lanes:
            break
    assert len(responses) == 4
    assert responses[-1].speculative_receipt["kind"] == "external_lilicorr"
    assert responses[-1].speculative_receipt["draft_settings"][
        "draft_attention_windows"
    ] == [2]
    engine.close()
