import json

import pytest

from mlx2.adapters.flash_next import (
    DEFAULT_MTP_ORDINARY_HANDOFF_MAX_WIDTH as FLASH_NEXT_HANDOFF_WIDTH,
    FlashNextAdapter,
)
from mlx2.adapters.laguna_xs21 import LAGUNA_XS21, LagunaXS21Adapter
from mlx2.adapters.mlx_vlm import (
    GEMMA3N,
    MINICPMO,
    Gemma3nAdapter,
    MiniCPMOAdapter,
)
from mlx2.adapters.muse_glimmer import MUSE_GLIMMER, MuseGlimmerAdapter
from mlx2.adapters.north_mini_code import NORTH_MINI_CODE, NorthMiniCodeAdapter
from mlx2.adapters.qwen import QWEN4_FLASH_NEXT
from mlx2.adapters.qwen36_35b import (
    DEFAULT_MTP_ORDINARY_HANDOFF_MAX_WIDTH as QWEN36_HANDOFF_WIDTH,
    Qwen3635BA3BAdapter,
    descriptor_for as qwen36,
)
from mlx2.adapters.qwen38_27b import (
    DEFAULT_MTP_ORDINARY_HANDOFF_MAX_WIDTH as QWEN38_HANDOFF_WIDTH,
    QWEN38_27B,
    QWEN38_27B_ORDINARY,
    Qwen3827BAdapter,
)
from mlx2.adapters.registry import AdapterResolution
from mlx2.adapters.xing import (
    DEFAULT_MTP_ORDINARY_HANDOFF_MAX_WIDTH as XING_HANDOFF_WIDTH,
    XingAdapter,
    descriptor_for as xing,
)
from mlx2.qualification import APPROVED_QUALIFICATION_HARNESS, REQUIRED_CHECKS


def _resolution(adapter_type, descriptor):
    return AdapterResolution(adapter_type, descriptor, {"fingerprint": "fixture"})


@pytest.mark.parametrize(
    ("adapter_type", "descriptor"),
    [
        (MuseGlimmerAdapter, MUSE_GLIMMER),
        (NorthMiniCodeAdapter, NORTH_MINI_CODE),
        (LagunaXS21Adapter, LAGUNA_XS21),
        (Gemma3nAdapter, GEMMA3N),
        (MiniCPMOAdapter, MINICPMO),
    ],
)
def test_non_mtp_adapters_declare_ordinary_default(adapter_type, descriptor):
    assert _resolution(adapter_type, descriptor).default_route == "ordinary"


@pytest.mark.parametrize(
    ("adapter_type", "descriptor"),
    [
        (FlashNextAdapter, QWEN4_FLASH_NEXT),
        (Qwen3827BAdapter, QWEN38_27B),
        (XingAdapter, xing(has_mtp=True)),
    ],
)
def test_mtp_adapters_declare_native_mtp_default(adapter_type, descriptor):
    assert _resolution(adapter_type, descriptor).default_route == "native_mtp"


def test_qwen36_defaults_to_native_mtp_with_the_handoff():
    # Measured 2026-09-20: native MTP gives Qwen3.6 no single-stream gain, but
    # with the wide-cohort ordinary handoff it beats ordinary batched (B8 243.8
    # vs 234.5, B16 305.0 vs 282.6).  The route is only sound with the handoff,
    # so assert both together.
    resolution = _resolution(Qwen3635BA3BAdapter, qwen36(has_mtp=True))
    assert resolution.default_route == "native_mtp"
    assert resolution.default_mtp_ordinary_handoff == {
        "enabled": True,
        "max_mtp_width": QWEN36_HANDOFF_WIDTH,
    }


@pytest.mark.parametrize(
    ("adapter_type", "descriptor", "constant"),
    [
        (Qwen3827BAdapter, QWEN38_27B, QWEN38_HANDOFF_WIDTH),
        (Qwen3635BA3BAdapter, qwen36(has_mtp=True), QWEN36_HANDOFF_WIDTH),
    ],
)
def test_qualified_mtp_adapters_declare_default_handoff_width_four(
    adapter_type, descriptor, constant
):
    assert constant == 4
    assert "default_mtp_ordinary_handoff_max_width" in adapter_type.__dict__
    assert _resolution(
        adapter_type, descriptor
    ).default_mtp_ordinary_handoff == {
        "enabled": True,
        "max_mtp_width": constant,
    }


@pytest.mark.parametrize(
    ("adapter_type", "descriptor", "constant"),
    [
        (FlashNextAdapter, QWEN4_FLASH_NEXT, FLASH_NEXT_HANDOFF_WIDTH),
        (XingAdapter, xing(has_mtp=True), XING_HANDOFF_WIDTH),
    ],
)
def test_unqualified_mtp_adapters_declare_handoff_default_off(
    adapter_type, descriptor, constant
):
    assert constant is None
    assert "default_mtp_ordinary_handoff_max_width" in adapter_type.__dict__
    assert _resolution(adapter_type, descriptor).default_mtp_ordinary_handoff is None


def test_handoff_default_resolves_only_for_native_mtp_and_can_be_disabled():
    from mlx2.server import (
        RouteSelection,
        resolve_execution_policy_defaults,
    )

    resolution = _resolution(Qwen3827BAdapter, QWEN38_27B)
    selected = resolve_execution_policy_defaults(
        None, RouteSelection("native_mtp", "adapter_default"), resolution
    )
    assert selected == {
        "mtp_ordinary_handoff": {"enabled": True, "max_mtp_width": 4}
    }
    assert resolve_execution_policy_defaults(
        None, RouteSelection("ordinary", "explicit_flag"), resolution
    ) is None
    assert resolve_execution_policy_defaults(
        {"mtp_ordinary_handoff": False},
        RouteSelection("native_mtp", "explicit_flag"),
        resolution,
    ) == {"mtp_ordinary_handoff": False}
    explicit = {"mtp_ordinary_handoff": {"enabled": True, "max_mtp_width": 4}}
    assert resolve_execution_policy_defaults(
        explicit,
        RouteSelection("native_mtp", "explicit_flag"),
        _resolution(FlashNextAdapter, QWEN4_FLASH_NEXT),
    ) == explicit


@pytest.mark.parametrize(
    ("adapter_type", "descriptor"),
    [
        (Qwen3827BAdapter, QWEN38_27B),
        (Qwen3635BA3BAdapter, qwen36(has_mtp=True)),
    ],
)
def test_default_handoff_without_qualification_still_fails_closed(
    adapter_type, descriptor
):
    from mlx2.server import RouteSelection, resolve_execution_policy_defaults
    from mlx2.serving import ServingEngine

    resolution = _resolution(adapter_type, descriptor)
    policy = resolve_execution_policy_defaults(
        None, RouteSelection("native_mtp", "adapter_default"), resolution
    )
    with pytest.raises(ValueError, match="observed handoff evidence"):
        ServingEngine.validate_arguments("unused", execution_policy=policy)
    ServingEngine.validate_arguments(
        "unused", mtp=False, execution_policy=resolve_execution_policy_defaults(
            None, RouteSelection("ordinary", "explicit_flag"), resolution
        )
    )


def test_adapter_default_falls_back_to_ordinary_for_artifact_without_mtp_head():
    resolution = _resolution(Qwen3827BAdapter, QWEN38_27B_ORDINARY)
    assert resolution.default_route == "ordinary"
    assert resolution.default_mtp_ordinary_handoff is None


def test_no_flag_uses_adapter_default_and_explicit_flags_win():
    from mlx2.server import build_parser, resolve_route_selection

    parser = build_parser()
    mtp = _resolution(Qwen3827BAdapter, QWEN38_27B)
    ordinary = _resolution(MuseGlimmerAdapter, MUSE_GLIMMER)

    selected = resolve_route_selection(
        parser.parse_args(["--model", "fixture"]), None, mtp
    )
    assert (selected.route, selected.source) == ("native_mtp", "adapter_default")
    selected = resolve_route_selection(
        parser.parse_args(["--model", "fixture"]), None, ordinary
    )
    assert (selected.route, selected.source) == ("ordinary", "adapter_default")

    for flag, route in (
        ("--ordinary", "ordinary"),
        ("--native-mtp", "native_mtp"),
        ("--prompt-lookup", "prompt_lookup"),
    ):
        selected = resolve_route_selection(
            parser.parse_args(["--model", "fixture", flag]), None, mtp
        )
        assert (selected.route, selected.source) == (route, "explicit_flag")

    policy = {"draft_model": "fixture-draft"}
    selected = resolve_route_selection(
        parser.parse_args(["--model", "fixture", "--external-draft"]),
        policy,
        ordinary,
    )
    assert (selected.route, selected.source) == (
        "external_draft",
        "explicit_flag",
    )


def test_explicit_native_mtp_fails_clearly_for_non_mtp_adapter():
    from mlx2.server import build_parser, resolve_route_selection

    args = build_parser().parse_args(["--model", "fixture", "--native-mtp"])
    with pytest.raises(ValueError, match=r"--native-mtp.*no implemented native MTP"):
        resolve_route_selection(
            args, None, _resolution(MuseGlimmerAdapter, MUSE_GLIMMER)
        )
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--model", "fixture", "--ordinary", "--native-mtp"]
        )


def test_qualification_matches_resolved_route_not_selection_spelling(tmp_path):
    from mlx2.qualification import load_qualified_route

    explicit = {
        "mtp": False,
        "route": "ordinary",
        "speculation": "ordinary",
        "route_selection_source": "explicit_flag",
    }
    record = {
        "passed": True,
        "runtime": {"source": "fixture"},
        "artifact": "weights",
        "settings": explicit,
        "qualification_harness": APPROVED_QUALIFICATION_HARNESS,
        "checks": {
            name: {"passed": True}
            for name in REQUIRED_CHECKS | {"structured_output"}
        },
    }
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(record))

    defaulted = {**explicit, "route_selection_source": "adapter_default"}
    route = load_qualified_route(
        path,
        runtime=record["runtime"],
        artifact=record["artifact"],
        settings=defaulted,
        descriptor=QWEN4_FLASH_NEXT,
        name="ordinary",
    )
    assert route.profile.name == "ordinary"

    with pytest.raises(ValueError, match="serving settings"):
        load_qualified_route(
            path,
            runtime=record["runtime"],
            artifact=record["artifact"],
            settings={**defaulted, "route": "native_mtp", "mtp": True},
            descriptor=QWEN4_FLASH_NEXT,
            name="native-mtp",
        )
