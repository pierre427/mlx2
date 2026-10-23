import importlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("run_nemotron35_super_quick_probe")


def test_head_depth_is_independent_of_one_session_server_capacity(probe, tmp_path):
    for route, depth in (("lilicorr", 7), ("dspark", 8), ("native_mtp", 16)):
        head = None if route == "native_mtp" else tmp_path / route
        policy = probe.policy_for(route, head, depth)
        assert policy["num_draft"] == depth
        args = SimpleNamespace(route=route, model=tmp_path / "target", port=18959)
        command = probe.server_command(args, tmp_path / "policy.json")
        assert command[command.index("--max-lanes") + 1] == "1"
        assert command[command.index("--max-inflight") + 1] == "1"
    assert probe.policy_for("lilicorr", tmp_path, None)["num_draft"] == 7
    assert probe.policy_for("dspark", tmp_path, None)["num_draft"] == 8
    with pytest.raises(ValueError):
        probe.policy_for("lilicorr", tmp_path, 8)
    tiled = probe.policy_for("lilicorr", tmp_path, 7, verification_tile=4)
    assert tiled["num_draft"] == 7 and tiled["progressive_verification_tile"] == 4
    assert tiled["progressive_verification_fidelity"] == "exact"
    alternatives = probe.policy_for(
        "lilicorr",
        tmp_path,
        7,
        verification_tile=4,
        alternative_branches=4,
    )
    assert alternatives["progressive_alternative_branches"] == 4
    shared = probe.policy_for(
        "lilicorr",
        tmp_path,
        7,
        verification_tile=4,
        alternative_branches=4,
        alternative_strategy="shared_prefix",
    )
    assert shared["progressive_alternative_strategy"] == "shared_prefix"
    paired = probe.policy_for(
        "lilicorr",
        tmp_path,
        7,
        verification_tile=4,
        alternative_branches=4,
        alternative_strategy="paired_continuations",
    )
    assert paired["progressive_alternative_strategy"] == "paired_continuations"
    confidence_tail = probe.policy_for(
        "lilicorr",
        tmp_path,
        7,
        verification_tile=4,
        alternative_branches=4,
        alternative_strategy="confidence_tree",
        alternative_depth_policy="confidence_tail",
    )
    assert confidence_tail["progressive_alternative_depth_policy"] == (
        "confidence_tail"
    )
    dspark_alternatives = probe.policy_for(
        "dspark",
        tmp_path,
        8,
        verification_tile=4,
        alternative_branches=4,
        alternative_strategy="shared_prefix",
    )
    assert dspark_alternatives["progressive_alternative_branches"] == 4
    assert dspark_alternatives["progressive_alternative_strategy"] == "shared_prefix"
    with pytest.raises(ValueError):
        probe.policy_for("ordinary", None, None, verification_tile=4)


def test_server_command_exposes_prefill_chunk_without_changing_historical_default(
    probe, tmp_path
):
    base = SimpleNamespace(
        route="ordinary", model=tmp_path / "target", port=18959
    )
    command = probe.server_command(base, None)
    assert command[command.index("--prefill-step") + 1] == "512"

    tuned = SimpleNamespace(
        route="ordinary",
        model=tmp_path / "target",
        port=18959,
        prefill_step=2048,
    )
    command = probe.server_command(tuned, None)
    assert command[command.index("--prefill-step") + 1] == "2048"


def test_switchmlp_candidate_policy_composes_without_changing_defaults(probe, tmp_path):
    assert probe.policy_for("ordinary", None, None) is None
    assert probe.policy_for(
        "ordinary", None, None, expert_gather_sort="unsorted_decode"
    ) == {"expert_gather_sort": "unsorted_decode"}
    assert probe.policy_for(
        "native_mtp", None, 3, expert_gather_sort="unsorted_decode"
    ) == {"num_draft": 3, "expert_gather_sort": "unsorted_decode"}
    external = probe.policy_for(
        "lilicorr",
        tmp_path,
        7,
        verification_tile=4,
        expert_gather_sort="unsorted_decode",
    )
    assert external["expert_gather_sort"] == "unsorted_decode"
    with pytest.raises(ValueError, match="expert_gather_sort"):
        probe.policy_for("ordinary", None, None, expert_gather_sort="always")


def test_mamba_conv_candidate_policy_composes_without_changing_defaults(
    probe, tmp_path
):
    assert probe.policy_for("ordinary", None, None, mamba_conv="fused_b1") == {
        "mamba_conv": "fused_b1"
    }
    assert probe.policy_for("native_mtp", None, 3, mamba_conv="fused_b1") == {
        "num_draft": 3,
        "mamba_conv": "fused_b1",
    }
    external = probe.policy_for(
        "lilicorr",
        tmp_path,
        7,
        verification_tile=4,
        mamba_conv="fused_b1",
    )
    assert external["mamba_conv"] == "fused_b1"
    assert probe.policy_for("ordinary", None, None, mamba_conv="fused_short") == {
        "mamba_conv": "fused_short"
    }
    with pytest.raises(ValueError, match="mamba_conv"):
        probe.policy_for("ordinary", None, None, mamba_conv="always")


def test_mamba_ssm_candidate_policy_composes_without_changing_defaults(probe, tmp_path):
    assert probe.policy_for("ordinary", None, None, mamba_ssm="fused_short") == {
        "mamba_ssm": "fused_short"
    }
    external = probe.policy_for(
        "lilicorr",
        tmp_path,
        7,
        verification_tile=4,
        mamba_ssm="fused_short",
    )
    assert external["mamba_ssm"] == "fused_short"
    assert probe.policy_for("ordinary", None, None, mamba_ssm="fused_short_gate") == {
        "mamba_ssm": "fused_short_gate"
    }
    with pytest.raises(ValueError, match="mamba_ssm"):
        probe.policy_for("ordinary", None, None, mamba_ssm="always")


def test_mamba_norm_short_policy_composes_for_lilicorr(probe, tmp_path):
    external = probe.policy_for(
        "lilicorr",
        tmp_path,
        7,
        verification_tile=4,
        mamba_norm="fused_short",
    )
    assert external["mamba_norm"] == "fused_short"
    with pytest.raises(ValueError, match="mamba_norm"):
        probe.policy_for("ordinary", None, None, mamba_norm="always")


def test_mamba_projection_policy_is_adapter_owned(probe, tmp_path):
    assert probe.policy_for(
        "ordinary", None, None, mamba_projection="adaptive_row_exact"
    ) == {"mamba_projection": "adaptive_row_exact"}
    assert probe.policy_for(
        "ordinary", None, None, mamba_projection="short_row_exact"
    ) == {"mamba_projection": "short_row_exact"}
    assert probe.policy_for(
        "ordinary", None, None, mamba_projection="native_qmm_residual_b1"
    ) == {"mamba_projection": "native_qmm_residual_b1"}
    external = probe.policy_for(
        "lilicorr",
        tmp_path,
        7,
        verification_tile=4,
        mamba_projection="compact_row_exact",
    )
    assert external["mamba_projection"] == "compact_row_exact"
    with pytest.raises(ValueError, match="mamba_projection"):
        probe.policy_for("ordinary", None, None, mamba_projection="bad")


def test_mamba_segment_policy_is_adapter_owned(probe, tmp_path):
    assert probe.policy_for(
        "ordinary", None, None, mamba_segment="layerwise_short_exact"
    ) == {"mamba_segment": "layerwise_short_exact"}
    external = probe.policy_for(
        "lilicorr",
        tmp_path,
        7,
        verification_tile=4,
        mamba_segment="layerwise_short_exact",
    )
    assert external["mamba_segment"] == "layerwise_short_exact"
    with pytest.raises(ValueError, match="mamba_segment"):
        probe.policy_for("ordinary", None, None, mamba_segment="bad")
    assert probe.policy_for(
        "ordinary",
        None,
        None,
        mamba_segment="layerwise_short_exact_routed_batch",
    ) == {"mamba_segment": "layerwise_short_exact_routed_batch"}


def test_external_draft_q8_policy_is_explicit_and_external_only(probe, tmp_path):
    policy = probe.policy_for(
        "lilicorr",
        tmp_path,
        7,
        verification_tile=4,
        draft_weight_quantization="affine_q8_g64",
    )
    assert policy["draft_weight_quantization"] == "affine_q8_g64"
    with pytest.raises(ValueError, match="Ordinary reference"):
        probe.policy_for(
            "ordinary",
            None,
            None,
            draft_weight_quantization="affine_q8_g64",
        )
    with pytest.raises(ValueError, match="Native MTP"):
        probe.policy_for(
            "native_mtp",
            None,
            3,
            draft_weight_quantization="affine_q8_g64",
        )


def test_route_binding_rejects_extra_sessions_wrong_head_or_qualification(probe):
    status = {
        "artifact": "bound",
        "qualification": "candidate",
        "settings": {"route": "external_draft", "max_lanes": 1, "max_inflight": 1},
    }
    assert probe.binding_errors(status, artifact="bound", route="dspark") == []
    for field in ("max_lanes", "max_inflight"):
        changed = {**status, "settings": {**status["settings"], field: 4}}
        assert probe.binding_errors(changed, artifact="bound", route="dspark")
    assert probe.binding_errors(status, artifact="other_head", route="dspark")
    assert probe.binding_errors(status, artifact="bound", route="ordinary")
    assert probe.binding_errors(
        {**status, "qualification": "qualified"}, artifact="bound", route="dspark"
    )


def test_chat_response_binds_to_live_server_identity_without_invented_fields(probe):
    initial = {
        "runtime": {"source_sha256": "runtime"},
        "artifact": "artifact",
        "profile": "profile",
        "settings": {"route": "ordinary", "max_lanes": 1, "max_inflight": 1},
    }
    # This matches the real API: /v1/status owns artifact/runtime identity;
    # chat's mlx2 receipt exposes route and profile, without those fields.
    response = {"mlx2": {"route": "ordinary", "profile": "profile"}}
    assert probe.response_binding_errors(response, dict(initial), initial) == []
    for key in ("runtime", "artifact", "profile", "settings"):
        assert probe.response_binding_errors(
            response, {**initial, key: "changed"}, initial
        )
    for key in ("route", "profile"):
        assert probe.response_binding_errors(
            {"mlx2": {**response["mlx2"], key: "changed"}}, initial, initial
        )


def test_external_binding_matches_observed_super_native_server_namespace(probe):
    # Exact immutable target/head identity and the independently observed
    # real-server /v1/status fingerprint from the first native-head load.
    artifact = {
        "identity": {
            "fingerprint": "c9750f0f58e0111da4efc88fda1728ed422519ffbc702643c7784668ee37842a"
        },
        "conversion": {"recipe": "nemotron35-super-vl-affine-experts4-protected8-v1"},
    }
    head = {
        "fingerprint": "fe4c4719c7aa42294f9d6d440d3f285ec23b468b511d8ed21f08a0095e7c94d5"
    }
    assert (
        probe.expected_artifact(artifact, head)
        == "9b763b126eb84439d1eca5a895198c2e50f6dcf2790d6b7bb86c2351227dd35b"
    )
    assert (
        probe.expected_artifact(artifact, None) == artifact["identity"]["fingerprint"]
    )
    q8 = probe.expected_artifact(
        artifact,
        head,
        {"draft_weight_quantization": "affine_q8_g64"},
    )
    assert q8 != probe.expected_artifact(artifact, head)
    assert q8 == probe.expected_artifact(
        artifact,
        head,
        {"draft_weight_quantization": "affine_q8_g64"},
    )


def test_independent_vision_smoke_uses_expected_pixels_and_explicit_oracle(probe):
    import base64
    import io

    from PIL import Image

    case = probe.image_smoke_case()
    raw = base64.b64decode(case["prompt"][0]["image_url"]["url"].split(",", 1)[1])
    with Image.open(io.BytesIO(raw)) as image:
        assert image.size == (512, 512)
        assert image.getpixel((256, 256)) == (255, 0, 0)
        assert image.getpixel((0, 0)) == (255, 255, 255)
    response = lambda text: {"choices": [{"message": {"content": text}}]}
    assert probe.functional_check(case["id"], response("Red."))["passed"]
    assert not probe.functional_check(case["id"], response("blue"))["passed"]


def test_memory_and_thermal_guards_reject_missing_or_contaminated_evidence(probe):
    memory = probe.common.parse_memory(
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free:  100.\nPages inactive: 200.\nPages speculative: 30.\nSwapouts: 7.\n"
    )
    assert memory["available_bytes"] == 330 * 16384
    assert memory["swapouts"] == 7
    thermal = {
        "thermal_state": 0,
        "pmset_no_thermal_warning": True,
        "pmset_no_performance_warning": True,
        "pmset_no_cpu_power_warning": True,
        "battery_temperature_c": 35.0,
        "virtual_temperature_c": 40.0,
    }
    host = {
        "memory": {"available_bytes": 100 * probe.common.GIB, "swapouts": 7},
        "swap": {"used_bytes": 0},
        "memory_pressure_level": 1,
        "thermal": thermal,
        "foreign_model_processes": [],
        "foreign_live_services": [],
    }
    assert (
        probe.common.admission_errors(
            host, minimum_available_bytes=90 * probe.common.GIB, performance=True
        )
        == []
    )
    hot = {**host, "thermal": {**thermal, "thermal_state": 1}}
    assert probe.common.admission_errors(
        hot, minimum_available_bytes=1, performance=True
    )
    assert probe.common.admission_errors(hot, minimum_available_bytes=1) == []
    swapped = {**host, "memory": {**host["memory"], "swapouts": 8}}
    assert "swap-outs increased" in probe.common.contamination(host, swapped)
    assert probe.common.admission_errors(
        {**host, "foreign_live_services": [{"pid": 123}]}, minimum_available_bytes=1
    )
    with pytest.raises(RuntimeError, match="unavailable"):
        probe.common.parse_memory("missing counters")
    assert probe.common.idle_audio_service(
        "/installed/bin/audiocpp_server --config anything", 9000
    )
    assert not probe.common.idle_audio_service(
        "/installed/bin/audiocpp_server --config anything", 600000
    )
    assert not probe.common.idle_audio_service("/installed/bin/llama-server", 9000)
    assert probe.common.remote_ssh_client("ssh gpu-host mlx2.server --model remote")
    assert probe.common.remote_ssh_client("/usr/bin/ssh gpu-host mlx_lm.server")
    assert not probe.common.remote_ssh_client("/installed/bin/python -m mlx2.server")
    assert probe.common.cpu_seconds("1-02:03:04.50") == 93784.5
    observed = {"pid": 123, "cpu_seconds": 20.0, "rss_kib": 9000}
    idle = {**host, "observed_idle_audio_services": [observed]}
    active = {
        **host,
        "observed_idle_audio_services": [{**observed, "cpu_seconds": 20.8}],
    }
    assert "otherwise idle audio service became active" in probe.common.contamination(
        idle, active
    )


def test_ignore_swap_growth_keeps_other_contamination_fail_closed(probe, monkeypatch):
    monkeypatch.setattr(
        probe.common,
        "contamination",
        lambda before, after: [
            "swap-outs increased",
            "swap usage increased",
            "foreign model process appeared",
        ],
    )
    kept, ignored = probe.contamination_with_swap_policy(
        {}, {}, ignore_swap_growth=True
    )
    assert kept == ["foreign model process appeared"]
    assert ignored == ["swap-outs increased", "swap usage increased"]
    kept, ignored = probe.contamination_with_swap_policy(
        {}, {}, ignore_swap_growth=False
    )
    assert kept == [
        "swap-outs increased",
        "swap usage increased",
        "foreign model process appeared",
    ]
    assert ignored == []


def test_batched_methodology_is_separate_from_functional_outcome(probe):
    progressive = {
        "max_cohort_width": 4,
        "last_proposal_sha256": "placeholder",
    }
    results = [
        {
            "mlx2": {
                "speculation": {
                    "progressive_verification": {
                        **progressive,
                        "last_proposal_sha256": f"proposal-{index}",
                    }
                }
            }
        }
        for index in range(4)
    ]
    before = {
        "external_progressive_verify_batched_rounds": 10,
        "external_progressive_verify_b4_aligned_tile_launches": 2,
        "external_progressive_verify_b4_aligned_continued_launches": 0,
        "external_progressive_verify_b4_disjoint_first_tiles": 0,
    }
    after = {
        **before,
        "external_progressive_verify_batched_rounds": 11,
        "external_progressive_verify_b4_aligned_tile_launches": 3,
        "external_progressive_verify_b2_aligned_continued_launches": 1,
    }
    findings, evidence = probe.batched_methodology_assessment(
        results, before, after, active_sessions=4
    )
    assert evidence["observed_widths"] == [4, 4, 4, 4]
    assert evidence["proposal_hashes_distinct"] is True
    assert (
        evidence["counter_deltas"][
            "external_progressive_verify_b4_aligned_tile_launches"
        ]
        == 1
    )
    assert findings == []
    assert evidence["any_branch_continued"] is True
    assert evidence["surviving_continuation_launches"] == 1
    assert evidence["first_tiles_pairwise_disjoint"] is False


def test_cleanup_kills_children_even_after_group_leader_exits(probe):
    # The original server can exit while an inherited subprocess survives.
    # This CPU-only process fixture catches cleanup that checks only poll().
    code = (
        "import subprocess,sys\n"
        'subprocess.Popen([sys.executable,"-c",'
        '"import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
        'print(os.getpid(),flush=True); time.sleep(90)"])\n'
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    child = int(process.stdout.readline())
    try:
        process.wait(timeout=5)
        assert process.poll() == 0
        probe.common.stop_group(process)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            state = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(child)],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            if not state or state.startswith("Z"):
                break
            time.sleep(0.05)
        else:
            pytest.fail("Owned grandchild survived process-group cleanup")
    finally:
        probe.common.stop_group(process)
        process.stdout.close()
        try:
            os.kill(child, 9)
        except ProcessLookupError:
            pass


def test_cleanup_tolerates_permission_errors_only_after_group_exit(probe, monkeypatch):
    process = SimpleNamespace(pid=123456789, wait=lambda **kw: 0, poll=lambda: 0)
    calls = []

    def denied(pgid, signum):
        calls.append((pgid, signum))
        raise PermissionError("group contains only unreaped dead members")

    monkeypatch.setattr(probe.common.os, "killpg", denied)
    monkeypatch.setattr(probe.common, "group_running", lambda pgid: [])
    probe.common.stop_group(process)
    assert calls == [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)]


@pytest.mark.parametrize("denied_signal", [signal.SIGTERM, signal.SIGKILL])
def test_cleanup_does_not_hide_denied_live_group(probe, monkeypatch, denied_signal):
    process = SimpleNamespace(pid=123456789, wait=lambda **kw: 0, poll=lambda: 0)

    def signal_group(pgid, signum):
        if signum == denied_signal:
            raise PermissionError("live group cannot be signaled")

    monkeypatch.setattr(probe.common.os, "killpg", signal_group)
    monkeypatch.setattr(probe.common, "group_running", lambda pgid: [process.pid + 1])
    with pytest.raises(PermissionError, match="live group"):
        probe.common.stop_group(process)


def test_native_binary_identity_detects_payload_changes_at_the_same_version(
    probe, monkeypatch, tmp_path
):
    binary = tmp_path / "core.so"
    binary.write_bytes(b"native build one")

    class Distribution:
        version = "same-version"
        files = (Path("mlx/core.so"),)

        def locate_file(self, item):
            return binary

    def distribution(package):
        if package == "mlx":
            return Distribution()
        raise probe.common.importlib.metadata.PackageNotFoundError(package)

    monkeypatch.setattr(probe.common.importlib.metadata, "distribution", distribution)
    first = probe.common.runtime_dependencies()
    binary.write_bytes(b"native build two")
    second = probe.common.runtime_dependencies()
    assert first["versions"] == second["versions"]
    assert first["mlx_native_files"] != second["mlx_native_files"]
