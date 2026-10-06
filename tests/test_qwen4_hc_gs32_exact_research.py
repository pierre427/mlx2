"""CPU/static gates for the deferred exact gs32 Qwen4 HC candidate."""

import json
from pathlib import Path

import pytest

from scripts.research.bench_qwen4_hc_gs32_exact import (
    require_gpu_lease,
    selected_modules,
)
from scripts.research.qwen4_hc_gs32_exact import (
    SOURCE_HASHES,
    describe,
    exact_gs32_sources,
    inspect_mapping,
    pointer_geometry,
    source_constants,
    verify_source_pin,
)

SOURCE = Path("src/mlx2/runtime/models/qwen4_hc_decode.py")


def _weights(*, group_size=32, mixed_up=None, partial=False):
    prefix = "language_model.model.layers.0.attn_hyper_connection"
    mapping = {}
    for projection in (
        "input_mix_weight_down",
        "input_mix_weight_up",
        "block_inject_weight",
    ):
        for part in ("weight", "scales", "biases"):
            if partial and projection == "input_mix_weight_up" and part == "biases":
                continue
            mapping[f"{prefix}.{projection}.{part}"] = "model.safetensors"
    quant = {"bits": 4, "group_size": group_size, "mode": "affine"}
    if mixed_up is not None:
        quant[f"{prefix}.input_mix_weight_up"] = {
            "bits": 4,
            "group_size": mixed_up,
            "mode": "affine",
        }
    return {"model_type": "qwen4_exp", "quantization": quant}, mapping


def test_current_hc_source_is_pinned_without_importing_mlx():
    values = source_constants(SOURCE)
    assert verify_source_pin(values) == SOURCE_HASHES


def test_rewrite_changes_only_group_addressing_and_template_binding():
    original = source_constants(SOURCE)
    candidate = exact_gs32_sources(original)

    header = candidate["HEADER"]
    assert "int BITS, int GS" in header
    assert "g * GS + sc * 8" in header
    assert "sc < GS / 8" in header
    assert "g * 64 + sc * 8" not in header

    for name in ("NORM_DOWN_SOURCE", "UP_MIX_SOURCE"):
        assert " / 64" not in candidate[name]
        assert "64 / " not in candidate[name]

    # Reverse exactly the admitted substitutions. Equality proves that the
    # candidate did not inherit #4248's different FP32 epilogue law.
    restored_header = header.replace(
        "template <typename T, int BITS, int GS, typename P>\ninline float hcd_wide_row(",
        "template <typename T, int BITS, typename P>\ninline float hcd_wide_row(",
    ).replace("g * GS + sc * 8", "g * 64 + sc * 8").replace(
        "sc < GS / 8", "sc < 8"
    )
    assert restored_header == original["HEADER"]
    for name, tags in (
        ("NORM_DOWN_SOURCE", ("DB", "IB")),
        ("UP_MIX_SOURCE", ("UB",)),
    ):
        restored = candidate[name]
        for tag in tags:
            restored = restored.replace(
                f"hcd_wide_row<T, {tag}, GS>(", f"hcd_wide_row<T, {tag}>("
            )
        restored = restored.replace(" / GS", " / 64").replace("GS / ", "64 / ")
        assert restored == original[name]


def test_complete_homogeneous_gs32_artifact_passes_metadata_gate():
    config, weights = _weights()
    report = inspect_mapping(config, weights)
    assert report["artifact_gate"] == "pass"
    assert report["all_modules_metadata_eligible"] is True
    assert report["eligible_module_count"] == 1
    assert report["modules"][0]["native_checks_remaining"]


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("width,fast", [(10240, True), (320, False)])
def test_synthetic_gs32_group_pointer_fixture_covers_every_value(bits, width, fast):
    result = pointer_geometry(width=width, bits=bits, group_size=32, fast=fast)
    assert result["cells"] == width
    assert result["mismatches"] == []
    assert result["wide_subchunks_per_group"] == 4


def test_gs64_mixed_and_partial_artifacts_fail_closed():
    config, weights = _weights(group_size=64)
    report = inspect_mapping(config, weights)
    assert report["artifact_gate"] == "blocked"
    assert any("group_size=64" in item for item in report["modules"][0]["decline_reasons"])

    config, weights = _weights(mixed_up=64)
    report = inspect_mapping(config, weights)
    assert report["artifact_gate"] == "blocked"
    assert "input_mix_weight_up group_size=64" in report["modules"][0]["decline_reasons"]

    config, weights = _weights(partial=True)
    report = inspect_mapping(config, weights)
    assert report["artifact_gate"] == "blocked"
    assert "partial input_mix_weight_up" in report["modules"][0]["decline_reasons"]


def test_no_hc_modules_is_blocked_not_vacuously_supported():
    report = inspect_mapping(
        {"model_type": "qwen4_exp", "quantization": {"bits": 4, "group_size": 32}},
        {"language_model.model.embed_tokens.weight": "model.safetensors"},
    )
    assert report["module_count"] == 0
    assert report["artifact_gate"] == "blocked"


def test_receipt_separates_exact_4245_law_from_text_changing_4248_law():
    receipt = describe(SOURCE, [])
    assert receipt["upstream"]["exact_law"]["pr"] == 4245
    assert receipt["upstream"]["non_reference_law"]["pr"] == 4248
    assert receipt["upstream"]["related_open_issue"]["issue"] == 4209
    assert "not isolated" in receipt["upstream"]["related_open_issue"]["finding"]
    assert "FP32" in receipt["law"]["excluded"]
    assert receipt["route"] == {
        "production_changed": False,
        "default_enabled": False,
        "artifact_gate_required": True,
        "qualified": False,
        "selected": False,
        "observed_used": False,
    }


def test_module_selection_is_deterministic_and_combined_only():
    modules = [
        {
            "path": f"layer.{index}.attn_hyper_connection",
            "metadata_eligible": True,
            "projections": {"block_inject_weight": {}},
        }
        for index in range(9)
    ]
    modules.append(
        {
            "path": "model.hyper_connection_mixer",
            "metadata_eligible": True,
            "projections": {},
        }
    )
    chosen = selected_modules({"modules": modules}, 3)
    assert chosen == [
        "layer.0.attn_hyper_connection",
        "layer.4.attn_hyper_connection",
        "layer.8.attn_hyper_connection",
    ]


def test_native_run_requires_matching_gpu_lease_receipts(tmp_path):
    first, second = tmp_path / "host.json", tmp_path / "local.json"
    with pytest.raises(RuntimeError, match="missing"):
        require_gpu_lease((first, second))
    first.write_text(json.dumps({"lease_id": "a"}))
    second.write_text(json.dumps({"lease_id": "b"}))
    with pytest.raises(RuntimeError, match="matching"):
        require_gpu_lease((first, second))
    second.write_text(first.read_text())
    receipt = require_gpu_lease((first, second))
    assert receipt["lease_id"] == "a"
