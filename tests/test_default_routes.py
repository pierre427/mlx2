import json

import pytest

from mlx2.adapters.gemma4 import GEMMA4_31B, GEMMA4_A4B, Gemma431BAdapter, Gemma4A4BAdapter
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
        (Gemma4A4BAdapter, GEMMA4_A4B),
        (Gemma431BAdapter, GEMMA4_31B),
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
        (XingAdapter, xing(has_mtp=True), XING_HANDOFF_WIDTH),
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


def test_flash_next_declares_default_handoff_width_three():
    # Width 3 beat width 4 by 9.9% at 4 lanes on the served artifact
    # (options-sweep-20261001); the other MTP adapters were not measured.
    assert FLASH_NEXT_HANDOFF_WIDTH == 3
    assert "default_mtp_ordinary_handoff_max_width" in FlashNextAdapter.__dict__
    assert _resolution(
        FlashNextAdapter, QWEN4_FLASH_NEXT
    ).default_mtp_ordinary_handoff == {"enabled": True, "max_mtp_width": 3}


def test_qwen36_declares_default_handoff_width_one():
    # With the decode wins on, static width 1 beat width 3 by +10.0% at 2
    # lanes and +22.0% at 3, neutral at 4 (options-sweep-qwen36-20261002,
    # handoff-dwnowin.json).  Native MTP stays the default route.
    assert QWEN36_HANDOFF_WIDTH == 1
    assert "default_mtp_ordinary_handoff_max_width" in Qwen3635BA3BAdapter.__dict__
    resolution = _resolution(Qwen3635BA3BAdapter, qwen36(has_mtp=True))
    assert resolution.default_route == "native_mtp"
    assert resolution.default_mtp_ordinary_handoff == {"enabled": True, "max_mtp_width": 1}


def test_handoff_default_resolves_only_for_native_mtp_and_can_be_disabled():
    from mlx2.server import (
        RouteSelection,
        resolve_execution_policy_defaults,
    )

    resolution = _resolution(Qwen3827BAdapter, QWEN38_27B)
    selected = resolve_execution_policy_defaults(
        None, RouteSelection("native_mtp", "adapter_default"), resolution
    )
    assert selected["mtp_ordinary_handoff"] == {
        "enabled": True, "max_mtp_width": 4
    }
    assert "mtp_ordinary_handoff" not in resolve_execution_policy_defaults(
        None, RouteSelection("ordinary", "explicit_flag"), resolution
    )
    assert resolve_execution_policy_defaults(
        {"mtp_ordinary_handoff": False},
        RouteSelection("native_mtp", "explicit_flag"),
        resolution,
    )["mtp_ordinary_handoff"] is False
    explicit = {"mtp_ordinary_handoff": {"enabled": True, "max_mtp_width": 4}}
    assert resolve_execution_policy_defaults(
        explicit,
        RouteSelection("native_mtp", "explicit_flag"),
        _resolution(FlashNextAdapter, QWEN4_FLASH_NEXT),
    )["mtp_ordinary_handoff"] == explicit["mtp_ordinary_handoff"]


@pytest.mark.parametrize(
    ("adapter_type", "descriptor"),
    [
        (Qwen3827BAdapter, QWEN38_27B),
        (Qwen3635BA3BAdapter, qwen36(has_mtp=True)),
    ],
)
def test_default_handoff_without_qualification_runs_unqualified(
    adapter_type, descriptor
):
    # Qualification is confidence, not permission to run (AGENTS.md): the
    # adapter-default handoff is exact, so an unqualified MTP model keeps it
    # and is labelled unqualified instead of being refused.
    from mlx2.server import RouteSelection, resolve_execution_policy_defaults
    from mlx2.serving import ServingEngine

    resolution = _resolution(adapter_type, descriptor)
    policy = resolve_execution_policy_defaults(
        None, RouteSelection("native_mtp", "adapter_default"), resolution
    )
    assert policy["mtp_ordinary_handoff"]["enabled"] is True
    ServingEngine.validate_arguments("unused", execution_policy=policy)
    # Handoff still needs the native self-MTP route.
    with pytest.raises(ValueError, match="native self-MTP"):
        ServingEngine.validate_arguments("unused", mtp=False, execution_policy=policy)
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


DECODE_FIRST_ORDER = {"enabled": True, "shared_prefill_budget": False}


@pytest.mark.parametrize(
    ("adapter_type", "descriptor"),
    [(FlashNextAdapter, QWEN4_FLASH_NEXT), (Qwen3827BAdapter, QWEN38_27B)],
)
@pytest.mark.parametrize("route", ["ordinary", "native_mtp"])
def test_decode_first_order_is_the_default_on_flash_next_and_the_27b(
    adapter_type, descriptor, route
):
    # Publication order only: decode-to-client lag max ~0.5-0.8 s -> 2-101 ms
    # with TTFT and throughput unchanged (options-sweep-*-20261002).  An
    # explicit value (false included) wins; prompt-lookup gets nothing.
    from mlx2.runtime.adaptive_policy import DecodeFirstPublish
    from mlx2.server import RouteSelection, resolve_execution_policy_defaults

    resolution = _resolution(adapter_type, descriptor)
    resolved = resolve_execution_policy_defaults(
        None, RouteSelection(route, "adapter_default"), resolution
    )
    assert resolved["decode_first"] == DECODE_FIRST_ORDER
    parsed = DecodeFirstPublish.from_value(resolved["decode_first"])
    assert parsed.mode({}) == "order"
    assert parsed.mode({"MLX2_DECODE_FIRST": "0"}) == "off"  # kill switch kept
    assert resolve_execution_policy_defaults(
        {"decode_first": False}, RouteSelection(route, "explicit_flag"), resolution
    )["decode_first"] is False
    assert resolution.default_execution_policy("prompt_lookup") == {}


@pytest.mark.parametrize(
    ("adapter_type", "descriptor"),
    [
        (Qwen3635BA3BAdapter, qwen36(has_mtp=True)),
        (XingAdapter, xing(has_mtp=True)),
    ],
)
def test_decode_first_stays_off_where_it_was_not_approved(adapter_type, descriptor):
    from mlx2.server import RouteSelection, resolve_execution_policy_defaults

    for route in ("ordinary", "native_mtp"):
        resolved = resolve_execution_policy_defaults(
            None, RouteSelection(route, "adapter_default"), _resolution(adapter_type, descriptor)
        ) or {}
        assert "decode_first" not in resolved


def test_flash_next_copy_draft_strong_span_defaults_to_fourteen():
    # 14 vs 16 on copy-heavy B1: +11.4% and +11.1% (8/8 reps each), tokens
    # identical (options-sweep-flashnext-20261002 item 8h).
    from mlx2.runtime.copy_draft import CopyDraftPolicy
    from mlx2.server import RouteSelection, resolve_execution_policy_defaults

    resolved = resolve_execution_policy_defaults(
        None, RouteSelection("native_mtp", "adapter_default"),
        _resolution(FlashNextAdapter, QWEN4_FLASH_NEXT),
    )
    copy = resolved["self_mtp_copy_draft"]
    assert copy["strong_max_span"] == 14
    assert CopyDraftPolicy.from_value(copy).strong_max_span == 14


def test_qwen38_27b_defaults_fused_gdn_and_num_draft_three(monkeypatch):
    # options-sweep-27b-20261002: fused_gdn bit-identical, ordinary +1.5..3.2%;
    # num_draft 3 served B2 +15%, B4 +11%.  Own-namespace defaults: the 9B
    # and Qwen3.6 subclasses keep fused_gdn off and num_draft 2.
    from mlx2.adapters.qwen35_9b import Qwen359BAdapter

    class _Stop(Exception):
        pass

    def built(adapter_type, policy=None):
        adapter = object.__new__(adapter_type)

        def stop(_path):
            raise _Stop

        monkeypatch.setattr(adapter_type, "artifact_inspector", staticmethod(stop))
        with pytest.raises(_Stop):
            adapter_type.__init__(adapter, "/unused", execution_policy=policy)
        return adapter

    default = built(Qwen3827BAdapter)
    assert default.fused_gdn is True and default._num_draft == 3
    explicit = built(Qwen3827BAdapter, {"fused_gdn": False, "num_draft": 2})
    assert explicit.fused_gdn is False and explicit._num_draft == 2
    nine = built(Qwen359BAdapter)
    assert nine.fused_gdn is False and nine._num_draft == 2
    assert "default_fused_gdn" not in vars(Qwen3635BA3BAdapter)
