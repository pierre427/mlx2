"""CPU regressions for the peer-informed September 29 audit."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest
from test_apcv2_sessions_persistence import _identity, _persistent, _state

from mlx2.adapters.generative_media import LTX25Adapter
from mlx2.adapters.multimodal import (
    Gemma3nVideoPolicy,
    MiniCPMOExecutionPolicy,
    install_gemma3n_vision_batching,
    install_minicpmo_vision_batching,
)
from mlx2.runtime.apc_v2 import APCv2
from mlx2.runtime.memory_policy import (
    SelfMTPLaneAdmissionController,
    _make_self_mtp_admission_callback,
)
from mlx2.runtime.models.nemotron_h import MoEGate, group_expert_select
from mlx2.runtime.models.switch_layers import SwitchLinear
from mlx2.runtime.weight_stream import ExpertLRU, StreamStats


@pytest.mark.parametrize("failure", ["payload", "manifest"])
@pytest.mark.parametrize("resident", [False, True])
def test_failed_parked_republication_preserves_previous_snapshot(
    tmp_path, monkeypatch, failure, resident
):
    key, tag, tokens = _identity(), ("tenant-a", "retry"), list(range(16))
    apc = _persistent(tmp_path)
    apc.store(key, tokens, [_state(16)], session_tag=tag)
    apc.park_session(*tag, ttl_seconds=3600)
    if resident:
        hit = apc.lookup(key, tokens + [99])
        hit.cache.close()
    previous = apc._trie.get(key, tokens)
    before_bytes = apc.nbytes
    disk_bytes = apc._disk_bytes
    manifest = Path(previous._apc_disk["manifest"])
    with monkeypatch.context() as patch:
        if failure == "payload":

            def fail(*args):
                raise OSError("injected disk full")

            patch.setattr(apc, "_atomic_save_cache", fail)
        else:
            patch.setattr(apc, "_write_manifest_locked", lambda *args: False)
        result = apc.store(key, tokens, [_state(16, 500)], session_tag=tag)
    assert result.stored is False
    assert apc._trie.get(key, tokens) is previous
    assert apc.nbytes == before_bytes
    assert apc._disk_bytes == disk_bytes
    assert manifest.exists()
    # Model a crash before orderly shutdown has a chance to repair the disk.
    apc._release_persist_lock()
    restarted = _persistent(tmp_path)
    try:
        hit = restarted.lookup(key, tokens + [99], session_tag=tag)
        assert hit.hit
        assert hit.cache[0].keys[0, 0, :16, 0].tolist() == list(range(16))
        hit.cache.close()
        assert restarted.store(key, tokens, [_state(16, 500)], session_tag=tag).stored
        assert not manifest.exists()
    finally:
        restarted.close()


def test_switch_quantization_does_not_allocate_discarded_random_experts(monkeypatch):
    linear = SwitchLinear(64, 8, 3, bias=True)
    expected = mx.quantize(linear.weight, group_size=32, bits=4)

    def forbidden(*args, **kwargs):
        raise AssertionError("conversion allocated unused random expert weights")

    monkeypatch.setattr(mx.random, "uniform", forbidden)
    quantized = linear.to_quantized(group_size=32, bits=4)
    for actual, reference in zip(
        (quantized.weight, quantized.scales, quantized.biases), expected
    ):
        assert mx.array_equal(actual, reference).item()
    assert quantized.bias is linear.bias
    assert quantized.input_dims == 64 and quantized.output_dims == 8
    assert quantized.num_experts == 3
    assert not quantized.trainable_parameters()


def test_gemma_frame_batches_are_evaluated_before_constructing_next(monkeypatch):
    events = []
    real_eval = mx.eval

    def forward(pixels, *args):
        events.append(("forward", pixels.shape[0]))
        return pixels + 1

    def evaluate(value):
        events.append(("eval", value.shape[0]))
        real_eval(value)

    model = SimpleNamespace(get_image_features=forward)
    install_gemma3n_vision_batching(model, Gemma3nVideoPolicy(frame_batch_size=2))
    monkeypatch.setattr(mx, "eval", evaluate)
    pixels = mx.arange(5).reshape(5, 1)
    result = model.get_image_features(pixels, None, None, None)
    assert result.tolist() == [[1], [2], [3], [4], [5]]
    assert events == [
        (kind, size) for size in (2, 2, 1) for kind in ("forward", "eval")
    ]


def test_minicpm_chunks_evaluate_and_restore_mixed_sample_image_order(monkeypatch):
    events = []
    real_eval = mx.eval

    class Tower:
        embeddings = SimpleNamespace(
            patch_embedding=SimpleNamespace(weight=mx.zeros((1,)))
        )

        def __call__(self, pixels, **kwargs):
            events.append(("forward", pixels.shape[0]))
            return pixels[:, 0, 0, :1]

    def evaluate(value):
        events.append(("eval", value.shape[0]))
        real_eval(value)

    model = SimpleNamespace(
        vision_tower=Tower(),
        resampler=lambda hidden, targets: hidden,
        config=SimpleNamespace(patch_size=1),
    )
    install_minicpmo_vision_batching(
        model, MiniCPMOExecutionPolicy(vision_batch_size=2)
    )
    monkeypatch.setattr(mx, "eval", evaluate)
    result = model.get_vision_embedding(
        [
            [mx.full((3, 2, 2), 1), mx.full((3, 4, 4), 2)],
            [mx.full((3, 2, 2), 3), mx.full((3, 2, 2), 4)],
        ],
        [[[2, 2], [4, 4]], [[2, 2], [2, 2]]],
    )
    assert [value.tolist() for value in result] == [[[1.0], [2.0]], [[3.0], [4.0]]]
    assert events == [
        (kind, size) for size in (2, 1, 1) for kind in ("forward", "eval")
    ]


@pytest.mark.parametrize("width", [3, 8, 20])
def test_verify_row_budget_counts_only_lanes_selected_to_execute(width):
    controller = SelfMTPLaneAdmissionController(
        verification_row_cap=6,
        saturation_lane_cap=16,
    )
    plan = controller.decide([100] * width, 1000, max_draft=2)
    assert plan.modes.count("self_mtp") == 2
    assert plan.modes.count("queue") == width - 2
    assert plan.primary_rows == 2
    assert plan.speculative_rows == 4


def test_plain_row_budget_also_limits_selected_execution_width():
    controller = SelfMTPLaneAdmissionController(verification_row_cap=2)
    plan = controller.decide([100] * 5, 1000, max_draft=0)
    assert plan.modes.count("plain") == 2
    assert plan.primary_rows == 2 and plan.speculative_rows == 0


def test_readmission_hold_reports_only_executing_rows_and_memory():
    controller = SelfMTPLaneAdmissionController(
        host_memory_gib=16,
        advisory_gib=12,
        transient_gib_per_lane=1.0,
    )
    rows = [(0, 100, 2, True, 0.0001), (1, 100, 2, True, 0.0001)]
    cost = controller.lane_gib(100, 2, 0.0001, resident_cache=True)
    free = [controller.hard_reserve_gib + 1.5 * cost]
    receipts = []
    admit = _make_self_mtp_admission_callback(
        controller,
        free_memory=lambda: free[0],
        max_draft=2,
        observer=receipts.append,
    )
    assert admit(rows) == {0: 2, 1: "queue"}
    # Both fit, but the second lane must wait for the re-admission margin.
    free[0] = controller.hard_reserve_gib + 2.2 * cost
    assert admit(rows) == {0: 2, 1: "queue"}
    assert receipts[-1].primary_rows == 1
    assert receipts[-1].speculative_rows == 2
    assert receipts[-1].estimated_gib == pytest.approx(cost)
    # Hysteresis must eventually re-enable a lane; it is not a permanent latch.
    free[0] += controller.READMIT_MARGIN_GIB
    assert admit(rows) == {0: 2, 1: 2}
    assert receipts[-1].primary_rows == 2
    assert receipts[-1].estimated_gib == pytest.approx(2 * cost)


@pytest.mark.parametrize("persistent", [False, True])
def test_closed_apc_invalidates_capsules_and_refuses_new_work(tmp_path, persistent):
    apc = (
        _persistent(tmp_path)
        if persistent
        else APCv2(max_bytes=1 << 20, layout_name="test")
    )
    generation = apc.capsule_generation.current
    apc.close(release_memory=False)
    assert apc.capsule_generation.current > generation
    assert apc.reserve_capsule_bytes(16) is None
    with pytest.raises(RuntimeError, match="closed"):
        apc.store(_identity(), [1], [_state(1)])
    with pytest.raises(RuntimeError, match="closed"):
        apc.lookup(_identity(), [1, 2])


@pytest.mark.parametrize("fail", [False, True])
def test_expert_overflow_reclaims_unneeded_rows_and_bounds_failure(fail):
    cache = None

    class Reader:
        num_experts = 8
        expert_bytes = 16

        def read_many(self, experts):
            # Old expert 0 is not used by the new forward and must not overlap
            # the entire temporary bank needed by this wider request.
            assert 0 not in cache._entries
            return {i: i for i in experts}

        def materialize(self, expert):
            if fail and expert == 3:
                raise RuntimeError("injected materialization failure")
            return (expert,)

    cache = ExpertLRU(Reader(), capacity_experts=1, stats=StreamStats())
    cache._entries[0] = (0,)
    if fail:
        with pytest.raises(RuntimeError, match="materialization failure"):
            cache.acquire([1, 2, 3])
    else:
        result = cache.acquire([1, 2, 3])
        assert result == {1: (1,), 2: (2,), 3: (3,)}
    assert cache.resident <= cache.capacity
    assert cache.stats.resident_bytes == cache.resident * Reader.expert_bytes


def test_grouped_moe_never_selects_a_masked_group_with_negative_corrected_scores():
    # Both retained experts have negative corrected scores, but beat group 1.
    # Masking excluded experts with zero erroneously makes them the winners.
    gates = mx.zeros((2, 3, 4), dtype=mx.float32)
    bias = mx.array([-1.0, -1.5, -3.0, -4.0])
    indices, weights = group_expert_select(gates, bias, 2, 2, 1, 1.0, True)
    assert mx.array_equal(
        mx.sort(indices, axis=-1), mx.broadcast_to(mx.array([0, 1]), (2, 3, 2))
    ).item()
    assert mx.allclose(weights, mx.full((2, 3, 2), 0.5)).item()


@pytest.mark.parametrize(
    "groups,selected,experts,topk",
    [(0, 1, 4, 2), (2, 3, 4, 2), (3, 1, 4, 2), (2, 1, 4, 3), (4, 1, 4, 1)],
)
def test_nemotron_rejects_invalid_routing_geometry(groups, selected, experts, topk):
    config = SimpleNamespace(
        num_experts_per_tok=topk,
        norm_topk_prob=True,
        n_routed_experts=experts,
        n_group=groups,
        topk_group=selected,
        routed_scaling_factor=1.0,
        hidden_size=2,
    )
    with pytest.raises(ValueError, match="routing geometry"):
        MoEGate(config)


def test_nemotron_all_groups_and_explicit_zero_scale():
    config = SimpleNamespace(
        num_experts_per_tok=2,
        norm_topk_prob=True,
        n_routed_experts=4,
        n_group=4,
        topk_group=4,
        routed_scaling_factor=0.0,
        hidden_size=2,
    )
    gate = MoEGate(config)
    indices, weights = gate(mx.ones((1, 2)))
    assert indices.shape == (1, 2)
    assert mx.array_equal(weights, mx.zeros_like(weights)).item()


@pytest.mark.parametrize("outcome", ["timeout", "success", "competing_writer"])
def test_ltx_output_is_published_only_on_success_without_clobbering(
    tmp_path, monkeypatch, outcome
):
    adapter = object.__new__(LTX25Adapter)
    from mlx2.adapters.generative_media import LTX_RUNTIME_REVISION

    adapter._init_lora("ltx-2.5", LTX_RUNTIME_REVISION)
    adapter._conversion_identity = {}
    adapter.runtime_root = tmp_path
    adapter.executable = Path("unused-python")
    adapter.mlx_model = tmp_path / "model"
    adapter.artifact = SimpleNamespace(fingerprint="test")
    target = tmp_path / "video.mp4"

    def run(command, **kwargs):
        if command[0] == "git":
            return SimpleNamespace(
                stdout=LTX_RUNTIME_REVISION + "\n" if "rev-parse" in command else ""
            )
        if "--output" not in command:
            return SimpleNamespace(returncode=0, stdout="")
        staged = Path(command[command.index("--output") + 1])
        staged.write_bytes(b"rendered video")
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, 1)
        if outcome == "competing_writer":
            target.write_bytes(b"other writer")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    if outcome == "timeout":
        with pytest.raises(subprocess.TimeoutExpired):
            adapter.generate_video("test", output=target, timeout_seconds=1)
        assert not target.exists()
    elif outcome == "competing_writer":
        with pytest.raises(FileExistsError):
            adapter.generate_video("test", output=target)
        assert target.read_bytes() == b"other writer"
    else:
        result = adapter.generate_video("test", output=target)
        assert result.path == target
        assert target.read_bytes() == b"rendered video"
    assert set(tmp_path.iterdir()) == ({target} if target.exists() else set())
