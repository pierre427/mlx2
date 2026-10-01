"""Evidence-backed per-model performance defaults (2026-09-24 defaults audit).

CPU-only: these tests resolve policies and parse arguments; nothing loads
weights or touches the GPU.
"""
import os

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


def test_qwen38_27b_native_mtp_defaults_copy_drafts_single_lane():
    from mlx2.runtime.copy_draft import CopyDraftPolicy
    from mlx2.serving import ServingEngine

    resolution = _resolution(Qwen3827BAdapter, QWEN38_27B)
    policy = resolve_execution_policy_defaults(None, MTP, resolution)
    parsed = CopyDraftPolicy.from_value(policy["self_mtp_copy_draft"])
    assert parsed.enabled is True
    # The GO verdict was measured with cohort copies refused.
    assert parsed.batched_max_span == 0
    ServingEngine.validate_arguments("unused", execution_policy=policy)
    # Ordinary route: copy drafts need self-MTP, so no default there.
    ordinary = resolve_execution_policy_defaults(None, ORDINARY, resolution) or {}
    assert "self_mtp_copy_draft" not in ordinary
    # An explicit disable wins.
    off = resolve_execution_policy_defaults(
        {"self_mtp_copy_draft": {"enabled": False}}, MTP, resolution
    )
    assert off["self_mtp_copy_draft"] == {"enabled": False}


def test_flash_next_native_mtp_defaults_match_gated_copy_drafts():
    from mlx2.runtime.copy_draft import CopyDraftPolicy
    from mlx2.serving import ServingEngine

    resolution = _resolution(FlashNextAdapter, QWEN4_FLASH_NEXT)
    policy = resolve_execution_policy_defaults(None, MTP, resolution)
    assert policy["apc_interior_checkpoints"] == "auto"
    assert policy["host_memory_signals"] == {"enabled": True}
    parsed = CopyDraftPolicy.from_value(policy["self_mtp_copy_draft"])
    assert parsed.enabled and parsed.batched_max_span == 0
    # The measured policy: fused-verify width, match gate, strong spans.
    assert (parsed.max_span, parsed.min_match, parsed.strong_match,
            parsed.strong_max_span, parsed.initial_span) == (7, 8, 32, 16, 7)
    ServingEngine.validate_arguments("unused", execution_policy=policy)
    ordinary = resolve_execution_policy_defaults(None, ORDINARY, resolution) or {}
    assert "self_mtp_copy_draft" not in ordinary
    assert ordinary["host_memory_signals"] == {"enabled": True}
    assert "host_memory_signals" not in resolve_execution_policy_defaults(
        None, MTP, _resolution(Qwen3827BAdapter, QWEN38_27B)
    )
    assert resolve_execution_policy_defaults(
        {"host_memory_signals": {"enabled": False}}, MTP, resolution
    )["host_memory_signals"] == {"enabled": False}
    off = resolve_execution_policy_defaults(
        {"self_mtp_copy_draft": {"enabled": False}}, MTP, resolution
    )
    assert off["self_mtp_copy_draft"] == {"enabled": False}


@pytest.mark.parametrize(
    ("adapter_type", "descriptor"),
    [
        (Qwen3635BA3BAdapter, qwen36(has_mtp=True)),
        (XingAdapter, xing(has_mtp=True)),
    ],
)
def test_copy_drafts_stay_off_where_unproven(adapter_type, descriptor):
    policy = resolve_execution_policy_defaults(
        None, MTP, _resolution(adapter_type, descriptor)
    ) or {}
    assert "self_mtp_copy_draft" not in policy


ALL_MTP = HYBRID_MTP + [
    (XingAdapter, xing(has_mtp=True)),
    (Nemotron3SuperAdapter, NEMOTRON),
]


@pytest.mark.parametrize(("adapter_type", "descriptor"), ALL_MTP)
def test_native_mtp_routes_default_srpt_prefill_scheduling(adapter_type, descriptor):
    from mlx2.runtime.adaptive_policy import PrefillOrder

    resolution = _resolution(adapter_type, descriptor)
    policy = resolve_execution_policy_defaults(None, MTP, resolution, max_lanes=16)
    order = PrefillOrder.from_value(policy["prefill_scheduling"])
    assert order.enabled and order.order == "srpt"
    assert order.max_bypass == 3 and order.one_slice_contention is True
    # Ordinary stays FIFO: SRPT raised its max TTFT 1.95 -> 2.36 s.
    ordinary = resolve_execution_policy_defaults(
        None, ORDINARY, resolution, max_lanes=16
    ) or {}
    assert "prefill_scheduling" not in ordinary


def test_srpt_default_steps_aside_below_the_bypass_window_and_for_explicit():
    resolution = _resolution(Qwen3635BA3BAdapter, qwen36(has_mtp=True))
    for lanes in (1, 2, 3):
        policy = resolve_execution_policy_defaults(
            None, MTP, resolution, max_lanes=lanes
        )
        assert "prefill_scheduling" not in policy
    assert "prefill_scheduling" in resolve_execution_policy_defaults(
        None, MTP, resolution, max_lanes=4
    )
    explicit = resolve_execution_policy_defaults(
        {"prefill_scheduling": None}, MTP, resolution, max_lanes=16
    )
    assert explicit["prefill_scheduling"] is None


def test_server_main_passes_max_lanes_to_the_defaults(monkeypatch):
    # The CLI path is what makes the default reach a bare server start.
    import inspect

    from mlx2 import server

    source = inspect.getsource(server.main)
    assert "max_lanes=args.max_lanes" in source


def test_muse_dflash2_unqualified_default_depth_is_three_qualified_pins_four():
    import json
    from pathlib import Path

    from mlx2.adapters import muse_glimmer

    adapter = muse_glimmer.MuseGlimmerAdapter.__new__(muse_glimmer.MuseGlimmerAdapter)
    adapter.draft_model = object()
    adapter.external_policy = {"draft_model": "x"}
    assert adapter.execution_config(max_lanes=4, prefill_step=8)["num_draft"] == 3
    # The qualified profile keeps its explicit, receipt-bound depth.
    root = Path(__file__).resolve().parents[1]
    policy = json.loads((root / "qualification/policies/muse-dflash2.json").read_text())
    assert policy["num_draft"] == 4


def test_selected_prompt_lookup_policy_enables_only_qualified_candidate():
    import json
    from pathlib import Path

    from mlx2.runtime.pld import PromptLookupBatchGenerator

    root = Path(__file__).resolve().parents[1]
    policy = json.loads(
        (root / "qualification/policies/prompt-lookup.json").read_text()
    )["prompt_lookup"]
    validated = PromptLookupBatchGenerator.validate_policy(policy)
    assert validated["cliff_aware_span"] is True
    assert validated["deferred_admission"] is False
    assert validated["cost_aware_admission"] is False
    assert validated["recent_prompt_segments"] == 0
    assert validated["rotating_replay"] is False
    assert validated["batched_verify"] is False
    assert "retrieval_segments" not in validated


def test_bare_server_cli_reaches_the_qualified_handoff_geometry():
    from mlx2.server import build_parser

    args = build_parser().parse_args(["--model", "fixture"])
    # The handoff fires only above width 4; every handoff receipt ran 16/32.
    assert args.max_lanes == 16
    assert args.max_inflight == 32
    assert args.max_lanes > 4


def test_server_prefill_step_can_match_a_bound_media_qualification():
    from mlx2.server import build_parser, serving_engine_kwargs

    parser = build_parser()
    default = parser.parse_args(["--model", "fixture"])
    explicit = parser.parse_args(["--model", "fixture", "--prefill-step", "256"])
    assert default.prefill_step is None
    assert explicit.prefill_step == 256
    kwargs = serving_engine_kwargs(
        explicit, None, native_mtp=False, approximate_kv=None,
        max_request_bytes=1 << 20,
    )
    assert kwargs["prefill_step"] == 256
    for invalid in ("0", "-1", "not-an-int"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--model", "fixture", "--prefill-step", invalid])


@pytest.mark.parametrize(
    ("physical_gib", "expected_gib"),
    [
        (8, 1.0),
        (16, 2.0),
        (36, 4.5),
        (64, 8.0),
        (96, 28.0),
        (128, 48.0),
        (192, 88.0),
        (256, 96.0),
        (512, 96.0),
        (None, 16.0),
    ],
)
def test_default_cache_bytes_scales_with_host_memory(
    monkeypatch, physical_gib, expected_gib
):
    from mlx2 import cache_sizing
    from mlx2.server import default_cache_bytes

    if physical_gib is None:
        # Unknown host: keep the legacy qualified 128 GiB geometry, not the
        # large-host curve it cannot be shown to have.
        monkeypatch.setattr(cache_sizing, "physical_memory_bytes", lambda: None)
        assert default_cache_bytes() == 16 << 30
        return
    assert default_cache_bytes(physical_gib << 30) == int(expected_gib * (1 << 30))


def test_default_cache_bytes_is_continuous_and_monotone():
    from mlx2.cache_sizing import default_cache_bytes, legacy_default_cache_bytes

    gib = 1 << 30
    previous = 0
    for physical in range(4 * gib, 1024 * gib, gib // 4):
        value = default_cache_bytes(physical)
        assert value >= previous
        # One step of host RAM never moves the cache by more than 5/8 of it.
        assert value - previous <= (gib // 4) * 5 // 8 + 1 or previous == 0
        previous = value
        # Never below what the legacy rule gave the same host.
        assert value >= legacy_default_cache_bytes(physical)
    # Up to 64 GiB the curve *is* the legacy rule.
    for physical_gib in (8, 16, 24, 32, 36, 48, 64):
        physical = physical_gib * gib
        assert default_cache_bytes(physical) == legacy_default_cache_bytes(physical)


def test_default_cache_bytes_never_exceeds_the_m3_advisory_share():
    from mlx2.server import default_cache_bytes

    # 36 GiB M3: Metal advisory 28.08 GiB; a 19 GiB model must still fit
    # with the default cache full.
    assert 19 * (1 << 30) + default_cache_bytes(36 << 30) < 28.08 * (1 << 30)


def test_cache_bytes_parser_default_is_gpu_free_and_records_its_source():
    import subprocess
    import sys

    code = (
        "import sys; from mlx2.server import build_parser, default_cache_bytes; "
        "a = build_parser().parse_args(['--model', 'm']); "
        "assert a.cache_bytes == default_cache_bytes(), a.cache_bytes; "
        "assert a.cache_bytes_source == 'host_default'; "
        "b = build_parser().parse_args(['--model', 'm', '--cache-bytes', '8589934592']); "
        "assert b.cache_bytes == 8589934592 and b.cache_bytes_source == 'explicit'; "
        "assert 'mlx.core' not in sys.modules and 'mlx' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr


def test_engine_kwargs_carry_the_cache_bytes_source():
    from types import SimpleNamespace

    from mlx2.server import build_parser, serving_engine_kwargs

    for argv, source in (
        (["--model", "m"], "host_default"),
        (["--model", "m", "--cache-bytes", "12884901888"], "explicit"),
    ):
        args = build_parser().parse_args(argv)
        kwargs = serving_engine_kwargs(
            args,
            None,
            native_mtp=False,
            approximate_kv=None,
            max_request_bytes=1 << 20,
        )
        assert kwargs["cache_bytes_source"] == source
    # A hand-built namespace has no source: its value is the operator's.
    bare = SimpleNamespace(**vars(build_parser().parse_args(["--model", "m"])))
    del bare.cache_bytes_source
    assert (
        serving_engine_kwargs(
            bare, None, native_mtp=False, approximate_kv=None, max_request_bytes=1
        )["cache_bytes_source"]
        == "explicit"
    )


def test_clamp_host_default_cache_bytes_arithmetic():
    from mlx2.cache_sizing import clamp_host_default_cache_bytes

    gib = 1 << 30

    def clamp(resident_gib, limit_gib=92, requested_gib=48, floor_gib=16):
        return clamp_host_default_cache_bytes(
            requested_gib * gib,
            floor_bytes=floor_gib * gib,
            admission_limit_bytes=None if limit_gib is None else limit_gib * gib,
            resident_bytes=None if resident_gib is None else int(resident_gib * gib),
            stream_reserve_bytes=0,
            lane_need_bytes=1 * gib,
            lanes=4,
        )

    # Small model: 92 - 18 - 4 = 70 GiB of room, the full 48 GiB default fits.
    assert clamp(18) == (48 * gib, clamp(18)[1])
    assert clamp(18)[1]["reason"] == "fits"
    # Mid model: 92 - 60 - 4 = 28 GiB.
    (value, detail) = clamp(60)
    assert value == 28 * gib and detail["reason"] == "headroom"
    # Flash-Next's 72.5 GiB: 15.5 GiB of room is below the legacy 16 GiB, so
    # the legacy geometry holds (it is what that model already runs with).
    (value, detail) = clamp(72.5)
    assert value == 16 * gib and detail["reason"] == "legacy_floor"
    # Unmeasurable headroom falls back to the floor, never to the new curve.
    (value, detail) = clamp(None)
    assert value == 16 * gib and detail["reason"] == "headroom_unmeasured"
    (value, detail) = clamp(18, limit_gib=None)
    assert value == 16 * gib and detail["reason"] == "headroom_unmeasured"
    # The clamp never raises a request, including one below the floor.
    assert clamp(0, requested_gib=4, floor_gib=16)[0] == 4 * gib


def _cache_engine(monkeypatch, *, resident_gib, guarded=False, **overrides):
    from types import SimpleNamespace as NS

    from mlx2 import cache_sizing, memory, serving
    from mlx2.runtime import apc_v2, generate, os_memory

    gib = 1 << 30
    seen = {}

    class APC:
        def __init__(self, **kwargs):
            seen["max_bytes"] = kwargs.get("max_bytes")
            self.apc_stats = {}

        def key(self, *_a, **_kw):
            return "key"

        def spill_idle_entries(self):
            pass

        def clear(self):
            pass

    class Batch:
        scheduler_stats = {}

        def __init__(self, *_a, **_kw):
            pass

        def next(self):
            return [], []

        def close(self):
            pass

    class Adapter:
        apc_cache_headroom_guard = guarded
        max_context = 1000
        identity = {"fingerprint": "fake"}
        environment = {}
        layout = "fake"
        model = None
        tokenizer = NS(vocab_size=10, eos_token_ids=[])

        def __init__(self, _path):
            pass

        def profile_name(self, _mtp):
            return "fake"

        def execution_config(self, *, max_lanes, prefill_step):
            return {"num_draft": 0}

        def diagnostics(self):
            return {}

        def close(self):
            pass

    # The 128 GiB calibration host: advisory 112, reserves 16 + 4, so the
    # MLX admission limit is 92 GiB.  The adapter footprint is faked.
    monkeypatch.setattr(cache_sizing, "physical_memory_bytes", lambda: 128 * gib)
    monkeypatch.setattr(memory, "host_memory_gib", lambda: 128.0)
    monkeypatch.setattr(memory, "metal_advisory_gib", lambda: 112.0)
    monkeypatch.setattr(
        serving, "resident_device_bytes", lambda: int(resident_gib * gib)
    )
    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "fake"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * gib)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    monkeypatch.setattr(apc_v2, "APCv2", APC)
    monkeypatch.setattr(generate, "BatchGenerator", Batch)
    engine = serving.ServingEngine(
        "fake",
        adapter_factory=Adapter,
        qualification_mode=True,
        mtp=False,
        max_lanes=2,
        max_inflight=2,
        **overrides,
    )
    try:
        assert engine.ready.wait(5), engine.error
        status = engine.status()
    finally:
        engine.close()
    return status["settings"], seen


def _lane_need_gib():
    # Fake adapter: no cache_budget, so the envelope applies at its 1000-token
    # max_context; depth floor 0 pays one third of the k=2 transient.
    return 0.44 * 1000 / 1024 + 1.76 / 3


def test_post_load_clamp_lowers_a_host_default_beside_a_large_model(monkeypatch):
    gib = 1 << 30
    (settings, seen) = _cache_engine(
        monkeypatch,
        resident_gib=60,
        cache_bytes=48 * gib,
        cache_bytes_source="host_default",
    )
    # min(requested, 92 - 60 - 2 lanes x need); max_lanes=2 bounds the lanes.
    expected = 92 * gib - 60 * gib - 2 * int(_lane_need_gib() * gib)
    assert settings["cache_bytes"] == seen["max_bytes"] == expected
    assert settings["cache_bytes_source"] == "host_default"
    assert settings["cache_bytes_clamped_from"] == 48 * gib
    headroom = settings["cache_bytes_headroom"]
    assert headroom["reason"] == "headroom"
    assert headroom["admission_limit_bytes"] == 92 * gib
    assert headroom["resident_bytes"] == 60 * gib
    assert headroom["lanes"] == 2
    assert headroom["floor_bytes"] == 16 * gib


def test_post_load_clamp_never_goes_below_the_legacy_default(monkeypatch):
    gib = 1 << 30
    # 92 - 80 - 2 lanes leaves ~10 GiB: below the legacy 16 GiB the host
    # already ran with, so the legacy geometry holds.
    (settings, seen) = _cache_engine(
        monkeypatch,
        resident_gib=80,
        cache_bytes=48 * gib,
        cache_bytes_source="host_default",
    )
    assert settings["cache_bytes"] == seen["max_bytes"] == 16 * gib
    assert settings["cache_bytes_clamped_from"] == 48 * gib
    assert settings["cache_bytes_headroom"]["reason"] == "legacy_floor"


def test_post_load_clamp_leaves_room_for_a_small_model(monkeypatch):
    gib = 1 << 30
    (settings, seen) = _cache_engine(
        monkeypatch,
        resident_gib=18,
        cache_bytes=48 * gib,
        cache_bytes_source="host_default",
    )
    assert settings["cache_bytes"] == seen["max_bytes"] == 48 * gib
    assert settings["cache_bytes_source"] == "host_default"
    assert settings["cache_bytes_clamped_from"] is None
    assert settings["cache_bytes_headroom"]["reason"] == "fits"


def test_explicit_cache_bytes_is_never_clamped(monkeypatch):
    gib = 1 << 30
    (settings, seen) = _cache_engine(
        monkeypatch,
        resident_gib=80,
        cache_bytes=48 * gib,
        cache_bytes_source="explicit",
    )
    assert settings["cache_bytes"] == seen["max_bytes"] == 48 * gib
    assert settings["cache_bytes_source"] == "explicit"
    assert settings["cache_bytes_clamped_from"] is None
    assert settings["cache_bytes_headroom"] is None


def test_flash_guard_clamps_explicit_cache_below_legacy_floor(monkeypatch):
    gib = 1 << 30
    (settings, seen) = _cache_engine(
        monkeypatch,
        resident_gib=80,
        guarded=True,
        cache_bytes=16 * gib,
        cache_bytes_source="explicit",
    )
    expected = 92 * gib - 80 * gib - 2 * int(_lane_need_gib() * gib)
    assert 0 < expected < 16 * gib
    assert settings["cache_bytes"] == seen["max_bytes"] == expected
    assert settings["cache_bytes_source"] == "explicit"
    assert settings["cache_bytes_clamped_from"] == 16 * gib
    assert settings["cache_bytes_headroom"]["floor_bytes"] == 0
    assert settings["cache_bytes_headroom"]["reason"] == "headroom"


def test_cache_bytes_provenance_is_not_route_identity(tmp_path):
    import json

    from mlx2.qualification import (
        APPROVED_QUALIFICATION_HARNESS,
        PROVENANCE_ONLY_SETTINGS,
        REQUIRED_CHECKS,
        load_qualified_route,
    )

    assert {
        "route_selection_source",
        "cache_bytes_source",
        "cache_bytes_clamped_from",
        "cache_bytes_headroom",
    } <= PROVENANCE_ONLY_SETTINGS
    assert "cache_bytes" not in PROVENANCE_ONLY_SETTINGS

    # A receipt recorded with an explicit 12 GiB, before the provenance
    # fields existed, still matches a server given --cache-bytes 12 GiB (or a
    # host default that resolves to 12 GiB); a different value never does.
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "passed": True,
                "qualification_harness": APPROVED_QUALIFICATION_HARNESS,
                "runtime": {"source": "abc"},
                "artifact": "weights",
                "settings": {"mtp": True, "cache_bytes": 12 << 30},
                "checks": {
                    c: {"passed": True}
                    for c in REQUIRED_CHECKS | {"mtp_execution", "structured_output"}
                },
            }
        )
    )

    def load(settings):
        return load_qualified_route(
            receipt,
            runtime={"source": "abc"},
            artifact="weights",
            settings=settings,
            descriptor=QWEN4_FLASH_NEXT,
            name="test",
        )

    served = {
        "mtp": True,
        "cache_bytes": 12 << 30,
        "cache_bytes_source": "explicit",
        "cache_bytes_clamped_from": None,
        "cache_bytes_headroom": None,
    }
    assert "profile=test" in load(served).receipt
    clamped = {
        **served,
        "cache_bytes_source": "host_default",
        "cache_bytes_clamped_from": 48 << 30,
        "cache_bytes_headroom": {"reason": "headroom"},
    }
    assert "profile=test" in load(clamped).receipt
    with pytest.raises(ValueError, match="does not match serving settings"):
        load({**served, "cache_bytes": 48 << 30, "cache_bytes_source": "host_default"})


def test_nemotron_environment_strips_inherited_lab_switches(monkeypatch):
    from mlx2.adapters import nemotron3_super

    for name in ("MLX_LM_BATCH_ATTENTION_BACKEND", "MLX_QWEN4_MEGAKERNEL",
                 "MLXUAG_EXPERIMENT", "MLX_GDN_CORE"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("MLX2_UNRELATED", "kept")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "MLX_ENABLE_TF32"):
        monkeypatch.delenv(name, raising=False)  # restored after the test
    profile = nemotron3_super.configure_environment()
    import os

    for name in ("MLX_LM_BATCH_ATTENTION_BACKEND", "MLX_QWEN4_MEGAKERNEL",
                 "MLXUAG_EXPERIMENT", "MLX_GDN_CORE"):
        assert name not in os.environ
    assert os.environ["MLX2_UNRELATED"] == "kept"
    # The pinned profile, and so the qualification identity, is unchanged.
    assert profile == {
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "MLX_ENABLE_TF32": "0",
    }


# --- Kernel A/B plumbing: switches exist, default to the qualified profile ---


def test_qwen36_kernel_switches_default_to_stock_and_toggle_by_policy(monkeypatch):
    from mlx2.adapters import qwen36_35b

    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "MLX_ENABLE_TF32"):
        monkeypatch.delenv(name, raising=False)
    for variable in qwen36_35b.KERNEL_POLICY_ENV.values():
        monkeypatch.delenv(variable, raising=False)
    stock = qwen36_35b.configure_environment()
    assert stock == qwen36_35b.configure_environment({})
    assert stock["MLX_QWEN36_FUSED_GDN_DECODE"] == "0"
    assert stock["MLX_QWEN4_MOE_FUSED_GATE_UP"] == "0"
    assert stock["MLX_GDN_CORE"] == "0"
    fused = qwen36_35b.configure_environment(
        {"fused_gdn_decode": True, "moe_fused_gate_up": True, "gdn_core": True}
    )
    import os

    assert fused["MLX_QWEN36_FUSED_GDN_DECODE"] == "1"
    assert os.environ["MLX_QWEN36_FUSED_GDN_DECODE"] == "1"
    assert fused["MLX_QWEN4_MOE_FUSED_GATE_UP"] == "1"
    assert fused["MLX_GDN_CORE"] == "1"
    # Only the three switches differ.
    assert {k for k in stock if stock[k] != fused[k]} == set(
        qwen36_35b.KERNEL_POLICY_ENV.values()
    ) - {qwen36_35b.KERNEL_POLICY_ENV["moe_routed_candidate"]}


def test_qwen36_adapter_accepts_kernel_switches_and_validates_them():
    from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter

    with pytest.raises(ValueError, match="must be boolean"):
        Qwen3635BA3BAdapter("/missing", execution_policy={"fused_gdn_decode": "on"})
    with pytest.raises(ValueError, match="kernel switches"):
        Qwen3635BA3BAdapter("/missing", execution_policy={"moe_router_kernel": True})
    # A valid switch passes validation and fails only on the missing artifact.
    with pytest.raises(Exception) as error:
        Qwen3635BA3BAdapter("/missing", execution_policy={"fused_gdn_decode": True})
    assert "kernel switches" not in str(error.value)
    assert "must be boolean" not in str(error.value)


def test_qwen38_adapter_accepts_gdn_core_switch():
    with pytest.raises(ValueError, match="gdn_core must be boolean"):
        Qwen3827BAdapter("/missing", execution_policy={"gdn_core": 1})
    with pytest.raises(Exception) as error:
        Qwen3827BAdapter("/missing", execution_policy={"gdn_core": True})
    assert "supports only" not in str(error.value)


def test_flash_next_policy_kernel_switches_are_opt_in_and_receipt_neutral():
    from mlx2.adapters.flash_next_policy import FlashNextPolicy

    default = FlashNextPolicy()
    for name in ("moe_router_kernel", "qsa_nax_decode", "gdn_core"):
        assert name not in default.as_dict()
    env = default.environment()
    assert "MLX_QWEN4_MOE_ROUTER_KERNEL" not in env
    assert "MLX_QWEN4_QSA_NAX_DECODE" not in env
    assert "MLX_GDN_CORE" not in env
    selected = FlashNextPolicy.from_mapping(
        {"moe_router_kernel": True, "qsa_nax_decode": True, "gdn_core": True}
    )
    env = selected.environment()
    assert env["MLX_QWEN4_MOE_ROUTER_KERNEL"] == "1"
    assert env["MLX_QWEN4_QSA_NAX_DECODE"] == "1"
    assert env["MLX_GDN_CORE"] == "1"
    assert selected.as_dict()["qsa_nax_decode"] is True
    with pytest.raises(ValueError):
        FlashNextPolicy.from_mapping({"gdn_core": "yes"})
