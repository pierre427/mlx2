"""Evidence-backed per-model performance defaults (2026-09-24 defaults audit).

CPU-only: these tests resolve policies and parse arguments; nothing loads
weights or touches the GPU.
"""
import pytest

from mlx2.adapters.flash_next import FlashNextAdapter
from mlx2.adapters.nemotron3_super import DESCRIPTOR as NEMOTRON, Nemotron3SuperAdapter
from mlx2.adapters.qwen import QWEN4_FLASH_NEXT
from mlx2.adapters.qwen35_9b import Qwen359BAdapter, descriptor_for as qwen35_9b
from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter, descriptor_for as qwen36
from mlx2.adapters.qwen38_27b import QWEN38_27B, Qwen3827BAdapter
from mlx2.adapters.registry import AdapterResolution
from mlx2.adapters.xing import XingAdapter, descriptor_for as xing
from mlx2.server import RouteSelection, resolve_execution_policy_defaults

MTP = RouteSelection("native_mtp", "adapter_default")
ORDINARY = RouteSelection("ordinary", "explicit_flag")


def _resolution(adapter_type, descriptor):
    return AdapterResolution(adapter_type, descriptor, {"fingerprint": "fixture"})


HYBRID_MTP = [
    (FlashNextAdapter, QWEN4_FLASH_NEXT),
    (Qwen3827BAdapter, QWEN38_27B),
    (Qwen3635BA3BAdapter, qwen36(has_mtp=True)),
]


@pytest.mark.parametrize(("adapter_type", "descriptor"), HYBRID_MTP)
def test_hybrid_mtp_adapters_default_interior_checkpoints_auto(
    adapter_type, descriptor
):
    policy = resolve_execution_policy_defaults(
        None, MTP, _resolution(adapter_type, descriptor)
    )
    assert policy["apc_interior_checkpoints"] == "auto"
    # Declared by the class itself, not inherited from Flash-Next.
    assert "default_route_execution_policy" in adapter_type.__dict__


@pytest.mark.parametrize(("adapter_type", "descriptor"), HYBRID_MTP)
def test_interior_default_is_native_mtp_only_and_explicit_policy_wins(
    adapter_type, descriptor
):
    resolution = _resolution(adapter_type, descriptor)
    ordinary = resolve_execution_policy_defaults(None, ORDINARY, resolution) or {}
    assert "apc_interior_checkpoints" not in ordinary
    for explicit in (None, {"count": 2, "min_stride": 512}):
        policy = resolve_execution_policy_defaults(
            {"apc_interior_checkpoints": explicit}, MTP, resolution
        )
        assert policy["apc_interior_checkpoints"] == explicit
    # Prompt lookup and external draft never receive adapter defaults.
    for route in ("prompt_lookup", "external_draft"):
        assert resolution.default_execution_policy(route) == {}


@pytest.mark.parametrize(("adapter_type", "descriptor"), HYBRID_MTP)
def test_checkpoint_defaults_skip_approximate_kv(adapter_type, descriptor):
    policy = resolve_execution_policy_defaults(
        None, MTP, _resolution(adapter_type, descriptor), approximate_kv=True
    )
    assert "apc_interior_checkpoints" not in (policy or {})


@pytest.mark.parametrize(
    ("adapter_type", "descriptor"),
    [
        (Nemotron3SuperAdapter, NEMOTRON),
        (Qwen359BAdapter, qwen35_9b(has_mtp=False)),
        (XingAdapter, xing(has_mtp=True)),
    ],
)
def test_unmeasured_subclasses_do_not_inherit_checkpoint_defaults(
    adapter_type, descriptor
):
    resolution = _resolution(adapter_type, descriptor)
    for route in (MTP, ORDINARY):
        policy = resolve_execution_policy_defaults(None, route, resolution) or {}
        assert "apc_interior_checkpoints" not in policy


def test_interior_default_passes_engine_argument_validation():
    from mlx2.serving import ServingEngine

    policy = resolve_execution_policy_defaults(
        None, MTP, _resolution(Qwen3827BAdapter, QWEN38_27B)
    )
    ServingEngine.validate_arguments("unused", execution_policy=policy)


def test_adapter_default_policy_rejects_non_defaultable_keys():
    class Bad(Qwen3827BAdapter):
        default_route_execution_policy = {"native_mtp": {"num_draft": 3}}

    with pytest.raises(ValueError, match="non-default-able"):
        _resolution(Bad, QWEN38_27B).default_execution_policy("native_mtp")
