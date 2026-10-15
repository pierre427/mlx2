"""GPU-free dispatch and policy gates for the two Gemma 4 layouts."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from mlx2.adapters.gemma4 import (
    GEMMA4_31B,
    GEMMA4_A4B,
    Gemma4A4BAdapter,
    Gemma431BAdapter,
    _Gemma4LogitsModel,
    inspect_gemma4_artifact,
)
from mlx2.adapters.mlx_vlm import _media_token_end
from mlx2.adapters.registry import inspect_model, resolve_adapter
from mlx2.contracts import Capability


def _artifact(path, *, sparse):
    layers = 30 if sparse else 60
    config = {
        "model_type": "gemma4",
        "video_token_id": 258884,
        "audio_config": None,
        "vision_config": {"model_type": "gemma4_vision"},
        "text_config": {
            "num_hidden_layers": layers,
            "hidden_size": 2816 if sparse else 5376,
            "enable_moe_block": sparse,
            "num_experts": 128 if sparse else None,
            "sliding_window": 1024,
            "num_kv_shared_layers": 0,
            "max_position_embeddings": 262144,
            "layer_types": [
                "full_attention" if index % 6 == 5 else "sliding_attention"
                for index in range(layers)
            ],
        },
    }
    (path / "config.json").write_text(json.dumps(config))
    (path / "model.safetensors").write_bytes(b"metadata-only fixture")
    return config


@pytest.mark.parametrize("sparse,expected,adapter_name", [
    (True, GEMMA4_A4B, "Gemma4A4BAdapter"),
    (False, GEMMA4_31B, "Gemma431BAdapter"),
])
def test_exact_topologies_resolve_without_loading_weights(tmp_path, sparse, expected, adapter_name):
    _artifact(tmp_path, sparse=sparse)
    result = inspect_model(tmp_path)
    assert result.adapter_type.__name__ == adapter_name
    assert result.descriptor == expected
    assert result.artifact["max_context"] == 262144
    assert result.artifact["full_attention_layers"] == (5 if sparse else 10)
    assert result.default_route == "ordinary"
    identity = result.adapter_type._inspect_adapter_artifact(tmp_path)
    assert identity["variant"] == expected.variant
    assert Capability.VISION in expected.capabilities
    assert Capability.VIDEO in expected.capabilities
    assert Capability.AUDIO not in expected.capabilities
    with pytest.raises(ValueError, match="native MTP"):
        resolve_adapter(tmp_path, mtp=True)


def test_mismatched_model_or_quantization_fails_before_load(tmp_path):
    config = _artifact(tmp_path, sparse=True)
    config["text_config"]["num_experts"] = 64
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="topology"):
        inspect_gemma4_artifact(tmp_path)

    config["text_config"]["num_experts"] = 128
    config["audio_config"] = {"hidden_size": 1024}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="without an audio tower"):
        inspect_gemma4_artifact(tmp_path)
    config["audio_config"] = None
    config["quantization"] = {"bits": 4, "group_size": 64, "mode": "affine"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="affine 8-bit or mxfp8"):
        inspect_gemma4_artifact(tmp_path)


@pytest.mark.parametrize("adapter_type", [Gemma4A4BAdapter, Gemma431BAdapter])
def test_execution_policy_is_refused_before_artifact_inspection(adapter_type):
    with pytest.raises(ValueError, match="do not accept speculative policy"):
        adapter_type(
            "/missing-gemma4-artifact",
            execution_policy={"draft_model": "/missing-draft"},
        )


def test_conversion_provenance_must_match_resolved_variant(tmp_path):
    config = _artifact(tmp_path, sparse=False)
    config["quantization"] = {"bits": 8, "group_size": 64, "mode": "affine"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="source provenance is required"):
        inspect_gemma4_artifact(tmp_path)
    (tmp_path / "source-and-quantization.json").write_text(json.dumps({
        "source_repo": "google/gemma-4-26B-A4B",
        "source_revision": "24548b62aa021d562695c04aaf7758a1ea47990b",
    }))
    with pytest.raises(ValueError, match="conversion source"):
        inspect_gemma4_artifact(tmp_path)


def test_instruct_conversions_are_accepted_per_variant(tmp_path):
    # Pierre, 2026-10-14: serve the instruct 26B-A4B as MLX mxfp8 (group 32).
    config = _artifact(tmp_path, sparse=True)
    config["quantization"] = {"bits": 8, "group_size": 32, "mode": "mxfp8"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "source-and-quantization.json").write_text(json.dumps({
        "source_repo": "google/gemma-4-26B-A4B-it",
        "source_revision": "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
    }))
    artifact = inspect_gemma4_artifact(tmp_path)
    assert artifact["precision"] == "mlx-mxfp8"
    assert artifact["source_revision"] == "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
    dense = tmp_path / "dense"
    dense.mkdir()
    config = _artifact(dense, sparse=False)
    config["quantization"] = {"bits": 8, "group_size": 32, "mode": "mxfp8"}
    (dense / "config.json").write_text(json.dumps(config))
    (dense / "source-and-quantization.json").write_text(json.dumps({
        "source_repo": "google/gemma-4-26B-A4B-it",
        "source_revision": "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
    }))
    with pytest.raises(ValueError, match="conversion source"):
        inspect_gemma4_artifact(dense)
    # The 31B instruct source is pinned for the dense variant (affine q8).
    config["quantization"] = {"bits": 8, "group_size": 64, "mode": "affine"}
    (dense / "config.json").write_text(json.dumps(config))
    (dense / "source-and-quantization.json").write_text(json.dumps({
        "source_repo": "google/gemma-4-31B-it",
        "source_revision": "842da3794eaa0b77d5f08bae87a17459d91ff475",
    }))
    assert inspect_gemma4_artifact(dense)["precision"] == "mlx-affine-8bit"


def test_pinned_prequantized_download_is_accepted_only_at_its_revision(tmp_path):
    # Pierre, 2026-10-14: serve mlx-community's 8-bit build as downloaded.
    config = _artifact(tmp_path, sparse=True)
    config["quantization"] = {"bits": 8, "group_size": 64, "mode": "affine"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    record = {
        "prequantized_repo": "mlx-community/gemma-4-26b-a4b-it-8bit",
        "prequantized_revision": "33c6d23798a0af159529890f79329206dbfbd73c",
    }
    (tmp_path / "source-and-quantization.json").write_text(json.dumps(record))
    artifact = inspect_gemma4_artifact(tmp_path)
    assert artifact["precision"] == "mlx-affine-8bit"
    assert artifact["source_revision"] == record["prequantized_revision"]
    record["prequantized_revision"] = "0" * 40
    (tmp_path / "source-and-quantization.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="conversion source"):
        inspect_gemma4_artifact(tmp_path)


@pytest.mark.parametrize("quant", [
    {"bits": 8, "group_size": 64, "mode": "mxfp8"},
    {"bits": 4, "group_size": 32, "mode": "mxfp4"},
])
def test_other_quantizations_are_refused(tmp_path, quant):
    config = _artifact(tmp_path, sparse=True)
    config["quantization"] = quant
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="affine 8-bit or mxfp8"):
        inspect_gemma4_artifact(tmp_path)


@pytest.mark.parametrize("change", [
    lambda c: c["text_config"]["layer_types"].reverse(),
    lambda c: c["text_config"].update(num_kv_shared_layers=20),
    lambda c: c["text_config"].pop("num_kv_shared_layers"),
])
def test_mismatched_cache_layout_fails_before_load(tmp_path, change):
    config = _artifact(tmp_path, sparse=True)
    change(config)
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="full/sliding attention layout"):
        inspect_gemma4_artifact(tmp_path)


@pytest.mark.parametrize("field,value", [
    ("vision_config", None), ("video_token_id", 7),
])
def test_mismatched_media_artifact_fails_before_load(tmp_path, field, value):
    config = _artifact(tmp_path, sparse=False)
    config[field] = value
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Expected vision/video"):
        inspect_gemma4_artifact(tmp_path)


def test_mixed_attention_uses_mlx2_cache_types_and_media_boundary():
    from mlx2.runtime.models.cache import KVCache, RotatingKVCache

    text = SimpleNamespace(
        sliding_window=1024,
        layer_types=["sliding_attention", "full_attention", "sliding_attention"],
    )
    model = SimpleNamespace(
        config=SimpleNamespace(text_config=text),
        language_model=SimpleNamespace(model=SimpleNamespace(first_kv_shared_layer_idx=3)),
    )
    cache = _Gemma4LogitsModel(model).make_cache()
    assert [type(item) for item in cache] == [RotatingKVCache, KVCache, RotatingKVCache]
    assert cache[0].max_size == cache[2].max_size == 1024
    assert _media_token_end({"mm_token_type_ids": np.array([[0, 1, 2, 0]])}) == 3


def test_both_variants_use_the_pinned_vendor_sampling_defaults():
    for adapter in (Gemma4A4BAdapter, Gemma431BAdapter):
        defaults = adapter.sampling_defaults.profiles["general"]
        assert defaults.values() == {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
        assert defaults.source == "generation_config.json"


def test_route_contract_separates_dense_and_sparse_optimization_surfaces():
    sparse = object.__new__(Gemma4A4BAdapter)
    dense = object.__new__(Gemma431BAdapter)
    sparse_contract = sparse.route_capability_contract()
    dense_contract = dense.route_capability_contract()

    for adapter, contract in ((sparse, sparse_contract), (dense, dense_contract)):
        assert contract["route"] == "ordinary"
        assert contract["ordinary_reference_preserved"] is True
        assert contract["apcv2"]["exact_only"] is True
        assert contract["apcv2"]["cache_layout"] == adapter.descriptor.cache_layout
        assert contract["apcv2"]["candidate_verify_publication"] is False
        assert contract["unsupported_capabilities"] == (
            "audio", "mtp", "external_draft"
        )
        assert contract["qualified"] is False

    sparse_opt = sparse_contract["optimization"]
    dense_opt = dense_contract["optimization"]
    assert sparse_opt["topology_kind"] == "sparse_moe"
    assert sparse_opt["dense_int8_prefill"]["eligible"] is False
    assert sparse_opt["sparse_moe_prefill"]["eligible"] is False
    assert dense_opt["topology_kind"] == "dense"
    assert dense_opt["dense_int8_prefill"]["eligible"] is True
    assert dense_opt["dense_int8_prefill"]["scopes"] == ("mlp", "all")
    assert dense_opt["sparse_moe_prefill"]["eligible"] is False
    assert sparse_opt["speculative_decode"]["eligible"] is False
    assert dense_opt["speculative_decode"]["eligible"] is False


def test_prepared_media_route_requires_complete_apcv2_identity():
    adapter = object.__new__(Gemma4A4BAdapter)
    prepared = {
        "_mlx2_prompt_tokens": [2, 258884, 7],
        "_mlx2_prefill_inputs": {"pixel_values": object()},
        "_mlx2_media_token_end": 2,
        "_mlx2_media_fingerprint": "ordered-media-proof",
    }
    assert adapter.validate_prepared_media_route(prepared) == {
        "route": "ordinary",
        "cache_layout": adapter.descriptor.cache_layout,
        "media_token_end": 2,
        "media_fingerprint": "ordered-media-proof",
        "apcv2_exact": True,
    }
    for field in tuple(prepared):
        broken = dict(prepared)
        broken.pop(field)
        with pytest.raises(ValueError, match="incomplete"):
            adapter.validate_prepared_media_route(broken)
    with pytest.raises(ValueError, match="outside the prompt"):
        adapter.validate_prepared_media_route(
            {**prepared, "_mlx2_media_token_end": 4}
        )
    with pytest.raises(ValueError, match="payload is missing"):
        adapter.validate_prepared_media_route(
            {**prepared, "_mlx2_prefill_inputs": {"input_ids": [1, 2]}}
        )
    video = {
        **prepared,
        "_mlx2_prefill_inputs": {"pixel_values_videos": object()},
    }
    assert adapter.validate_prepared_media_route(video)["apcv2_exact"] is True


def _served_adapter(cls, encoded, chat_template="{{ messages }}"):
    """An adapter shell with a fake tokenizer: no weights, no mlx-vlm."""
    adapter = object.__new__(cls)
    tokenizer = SimpleNamespace(bos_token_id=2, chat_template=None)
    adapter.processor = SimpleNamespace(
        tokenizer=tokenizer,
        chat_template=chat_template,
        apply_chat_template=lambda messages, **_: messages[-1]["content"],
    )
    adapter.tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens: list(encoded))
    return adapter


@pytest.mark.parametrize("cls", [Gemma4A4BAdapter, Gemma431BAdapter])
def test_text_prompts_start_with_bos_exactly_once(cls):
    # Base checkpoints have no chat template; without <bos> the served text
    # path degenerated (results/smoke-26b-q8-prefix-nobos.json).
    adapter = _served_adapter(cls, [818, 5279])
    assert adapter.prompt_tokens({"prompt": "The capital"}) == [2, 818, 5279]
    assert adapter.prompt_tokens(
        {"messages": [{"role": "user", "content": "The capital"}]}
    ) == [2, 818, 5279]
    already = _served_adapter(cls, [2, 818])
    assert already.prompt_tokens({"prompt": "<bos>The"}) == [2, 818]
    # A base checkpoint (no chat template) still serves completions with
    # <bos>, but refuses chat: mlx-vlm would join the raw message text and the
    # base model would continue it (Gemma 4 26B-A4B base, 2026-10-14).
    base = _served_adapter(cls, [818, 5279], chat_template=None)
    assert base.prompt_tokens({"prompt": "The capital"}) == [2, 818, 5279]
    with pytest.raises(ValueError, match="no chat template"):
        base.prompt_tokens({"messages": [{"role": "user", "content": "The capital"}]})
    # Processor-built media prompts already carry their own ids untouched.
    prepared = {
        "messages": [],
        "_mlx2_prompt_tokens": [2, 255999, 7],
        "_mlx2_prefill_inputs": {"pixel_values": object()},
        "_mlx2_media_token_end": 2,
        "_mlx2_media_fingerprint": "ordered-media-proof",
    }
    assert adapter.prompt_tokens(prepared) == [2, 255999, 7]


def test_prefill_step_defaults_are_per_variant_measured_values():
    # qualification/runs/gemma4-defaults-20260925/results/prefill-*-q8.json
    assert Gemma431BAdapter.default_prefill_step == 512
    assert Gemma4A4BAdapter.default_prefill_step == 2048
    for cls in (Gemma4A4BAdapter, Gemma431BAdapter):
        adapter = object.__new__(cls)
        assert adapter.prefill_step_default() == cls.default_prefill_step
        # Before the engine configures a step, the budget uses the default.
        assert adapter._budget_prefill_step() == cls.default_prefill_step
        adapter.execution_config(max_lanes=2, prefill_step=1024)
        assert adapter._budget_prefill_step() == 1024
