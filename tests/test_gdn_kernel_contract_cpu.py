"""Host/CPU contracts for model-supplied fused-GDN arithmetic."""

from dataclasses import replace
from pathlib import Path

from mlx2.runtime.models.gdn_kernel_contract import (
    FusedGdnContract,
    GdnGeometry,
    GdnQkNormalization,
    arithmetic_key_for_contract,
    resolve_fused_gdn_contract,
)
from mlx2.runtime.models.qwen_gdn_contracts import (
    QWEN4_GDN_CONTRACT,
    QWEN36_GDN_CONTRACT,
    QWEN38_GDN_CONTRACT,
    resolve_qwen_gdn_contract,
    uses_activation_dtype_arithmetic,
)


def test_each_model_contract_selects_explicit_generic_arithmetic():
    assert arithmetic_key_for_contract(QWEN4_GDN_CONTRACT) == (
        "sigmoid", True, False, GdnQkNormalization.ACTIVATION_L2,
    )
    assert uses_activation_dtype_arithmetic(QWEN38_GDN_CONTRACT)
    assert uses_activation_dtype_arithmetic(QWEN36_GDN_CONTRACT)


def test_shared_contract_requires_an_explicit_typed_value():
    assert "invalid fused GDN contract" in resolve_fused_gdn_contract(object())[1]
    assert resolve_fused_gdn_contract(None)[1] == "missing fused GDN contract"


def test_legacy_qwen_names_resolve_only_at_model_owned_edge():
    assert resolve_qwen_gdn_contract(None, "qwen38")[0] == QWEN38_GDN_CONTRACT
    assert "unsupported Qwen GDN architecture" in resolve_qwen_gdn_contract(
        None, "future-gdn"
    )[1]
    selected, refusal = resolve_qwen_gdn_contract(QWEN36_GDN_CONTRACT, "qwen4")
    assert selected is None
    assert refusal == "fused GDN contract disagrees with legacy Qwen architecture"


def test_shared_geometry_validation_has_no_model_family_allowlist():
    generic = FusedGdnContract(
        "generic-gdn-v1",
        GdnGeometry(8, 24, 64, 80, 3),
        "sigmoid",
        True,
        False,
        GdnQkNormalization.ACTIVATION_L2,
        "batch-hv-dv-dk-f32-accum-v1",
        ("float32", "float16"),
    )
    assert generic.refusal() is None


def test_qwen_kernel_refuses_unimplemented_generic_arithmetic_combination():
    unsupported = replace(
        QWEN4_GDN_CONTRACT,
        identity="sigmoid-activation-decay",
        decay_input_float32=False,
    )
    assert unsupported.refusal() is None
    try:
        uses_activation_dtype_arithmetic(unsupported)
    except ValueError as error:
        assert "does not implement arithmetic" in str(error)
    else:
        raise AssertionError("Qwen kernel accepted an unimplemented arithmetic tuple")


def test_unrecognized_normalization_refuses_even_on_valid_geometry():
    invalid = replace(
        QWEN4_GDN_CONTRACT,
        identity="unknown-normalization",
        qk_normalization="not-a-normalization",  # type: ignore[arg-type]
    )
    assert "unsupported QK normalization" in invalid.refusal()


def test_unknown_state_layout_refuses_before_source_selection():
    invalid = replace(
        QWEN36_GDN_CONTRACT,
        identity="unknown-state-layout",
        state_layout="flat-state-without-head-ownership",
    )
    assert "state layout" in invalid.refusal()


def test_shared_contract_source_contains_no_qwen_identity_or_architecture_map():
    source = (
        Path(__file__).resolve().parents[1]
        / "src/mlx2/runtime/models/gdn_kernel_contract.py"
    ).read_text()
    assert "QWEN" not in source.upper()
    assert "architecture" not in source
    assert "source_profile" not in source


def test_qwen38_and_qwen36_wrappers_never_select_qwen4_contract():
    models = Path(__file__).resolve().parents[1] / "src/mlx2/runtime/models"
    q38 = (models / "qwen38_fused_gdn.py").read_text()
    q36 = (models / "qwen36_35b.py").read_text()
    assert "QWEN4_GDN_CONTRACT" not in q38
    assert "QWEN4_GDN_CONTRACT" not in q36
    assert "contract=self.fused_gdn_contract" in q38
    assert "contract=QWEN36_GDN_CONTRACT" in q36
