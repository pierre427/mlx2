from types import SimpleNamespace
import pytest
from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.serving import shared_prefix_attestation


def test_policy_is_explicit_and_measured_thresholds_retained():
    policy = FlashNextPolicy()
    assert policy.shared_qsa_suffix == "auto"
    assert policy.private_delta_min_context == 65536
    assert policy.batch_config(max_lanes=4, prefill_step=2048)["prefetch_known_tail_ple"]
    assert policy.environment()["MLX_LM_SEGMENTED_ASYNC_QSA_PROMOTION"] == "1"
    assert policy.environment()["MLX_QWEN4_EAGER_DISPATCH"] == "1"
    assert policy.environment()["MLX_QWEN4_EAGER_DISPATCH_MAX_ROWS"] == "64"
    assert policy.environment()["MLX_QWEN4_EAGER_DISPATCH_STRIDE"] == "2"


def test_eager_dispatch_policy_controls_all_runtime_inputs():
    policy = FlashNextPolicy(
        eager_dispatch=False,
        eager_dispatch_max_rows=8,
        eager_dispatch_stride=2,
    )
    environment = policy.environment()
    assert environment["MLX_QWEN4_EAGER_DISPATCH"] == "0"
    assert environment["MLX_QWEN4_EAGER_DISPATCH_MAX_ROWS"] == "8"
    assert environment["MLX_QWEN4_EAGER_DISPATCH_STRIDE"] == "2"


def test_flash_adapter_explicitly_selects_observable_parity_paths(tmp_path, monkeypatch):
    from mlx2.adapters.flash_next import configure_environment

    monkeypatch.setenv("MLX_GDN_CORE", "1")
    monkeypatch.setenv("MLX_GDN_UNBOUND_EXPERIMENT", "1")
    environment = configure_environment(tmp_path, FlashNextPolicy())
    assert environment["MLX_QWEN4_PLE_NVME"] == str(tmp_path / "ple_rows.bin")
    assert environment["MLX_QWEN4_PLE_NVME_LRU_MB"] == "2048"
    assert environment["MLX_QWEN4_PLE_COMPILE"] == "1"
    assert environment["MLX_QWEN4_QSA_SCATTER_CHOSEN"] == "1"
    assert environment["MLX_QWEN4_MOE_FUSED_GATE_UP"] == "0"
    assert environment["MLX_QWEN4_FUSED_EXPERT_KERNEL"] == "auto"
    assert environment["MLX_QWEN4_FUSED_GDN_REPLAY_ROLLBACK"] == "1"
    assert environment["MLX_GDN_PACKED"] == "1"
    assert environment["MLX_GDN_CORE"] == "0"
    assert "MLX_GDN_UNBOUND_EXPERIMENT" not in __import__("os").environ


@pytest.mark.parametrize("settings", [{"num_draft": 0}, {"num_draft": True},
    {"shared_qsa_suffix": "yes"}, {"async_qsa_promotion": 1},
    {"private_delta_min_context": -1}, {"eager_dispatch": 1},
    {"eager_dispatch_max_rows": 0}, {"eager_dispatch_stride": 0},
    {"bogus": True}])
def test_invalid_policy_rejected(settings):
    with pytest.raises(ValueError):
        FlashNextPolicy.from_mapping(settings)


def test_prefix_attestation_requires_same_immutable_generation_and_seed():
    def hit(lineage="one", generation=0, tail=(7,), sidecar=True):
        return SimpleNamespace(cache=SimpleNamespace(cow_lineage_id=lineage,
            cow_generation=generation), cached_tokens=100, remaining_tokens=tail,
            sidecar=object() if sidecar else None)
    one = shared_prefix_attestation(hit())
    assert one and one == shared_prefix_attestation(hit())
    assert one != shared_prefix_attestation(hit(lineage="two"))
    assert one != shared_prefix_attestation(hit(generation=1))
    assert one != shared_prefix_attestation(hit(tail=(8,)))
    assert shared_prefix_attestation(hit(tail=(7, 8))) is None
    assert shared_prefix_attestation(hit(sidecar=False)) is None


def test_empty_scheduler_poll_preserves_warm_cache_without_memory_queue():
    from mlx2.serving import reclaim_deferred_cache
    class Cache:
        size = 3
        def __len__(self): return self.size
        def evict_oldest_unleased(self): self.size -= 1; return True
    cache = Cache()
    scratch = []
    assert not reclaim_deferred_cache(cache, {"stage": "full"}, lambda: scratch.append(1))
    assert cache.size == 3
    assert not reclaim_deferred_cache(cache, {"stage": "fewer_lanes"}, lambda: None)
    assert cache.size == 3
    assert reclaim_deferred_cache(cache, {"stage": "queue"}, lambda: None)
    assert cache.size == 2


def test_native_indexed_output_gate_does_not_require_slower_sequential_merge():
    policy = FlashNextPolicy(indexed_output_gate=True)
    assert policy.environment()["MLX_QWEN4_QSA_INDEXED_FUSED_GATE"] == "1"
    assert policy.environment()["MLX_QWEN4_QSA_INDEXED_FUSED_MERGE"] == "0"


@pytest.mark.parametrize("mode,enabled", [("auto", True), ("on", True), ("off", False)])
def test_shared_auto_policy_diagnostics_and_admission(monkeypatch, mode, enabled):
    from mlx2.runtime.segmented_self_mtp import segmented_self_mtp_stats, shared_qsa_suffix_admission
    for key, value in FlashNextPolicy(shared_qsa_suffix=mode).environment().items():
        monkeypatch.setenv(key, value)
    status = segmented_self_mtp_stats()
    assert status["shared_qsa_suffix_mode"] == mode
    assert status["shared_qsa_suffix_environment_enabled"] is enabled
    assert shared_qsa_suffix_admission(base_tokens=131000, remaining_tokens=64)[0] is enabled
    if mode == "auto":
        assert not shared_qsa_suffix_admission(base_tokens=100, remaining_tokens=64)[0]


def test_finishing_job_releases_closed_cache_object():
    import gc
    import threading
    import weakref
    from mlx2.serving import Job, ServingEngine
    class Branch:
        def close(self): pass
    engine = ServingEngine.__new__(ServingEngine)
    engine.lock = threading.Lock()
    engine.slots = threading.BoundedSemaphore(1)
    assert engine.slots.acquire()
    job = Job({})
    branch = Branch()
    reference = weakref.ref(branch)
    job.cache_branch = branch
    engine.jobs = {job.id: job}
    del branch
    engine._finish(job, {"finish_reason": "stop"})
    gc.collect()
    assert job.cache_branch is None
    assert reference() is None
    assert engine.slots.acquire(blocking=False)


def test_flash_next_receipt_round_trips_every_non_default_choice():
    # as_dict omits a field only at its default, so a receipt read back
    # reproduces the policy: an explicit "off" of a default-on kernel used to
    # be dropped and read back as "on" (2026-10-01).
    from dataclasses import fields

    from mlx2.adapters.flash_next_policy import FlashNextPolicy

    alternatives = {
        "hc_decode_kernels": False,
        "attn_fused_rows": False,
        "moe_routed_decode": "off",
        "fused_gdn_batch_decode": "off",
        "moe_topk_fold": "off",
        "fused_gdn_verify_max_steps": 8,
        "hc_decode_multi_row": "off",
        "moe_window_batch_decode": True,
        "row_exact_verify": True,
        "qsa_fused_scores": False,
        "fused_gdn_batch_verify": "off",
    }
    assert FlashNextPolicy.from_mapping(FlashNextPolicy().as_dict()) == FlashNextPolicy()
    names = {f.name for f in fields(FlashNextPolicy)}
    for name, value in alternatives.items():
        assert name in names, name
        policy = FlashNextPolicy.from_mapping({name: value})
        assert FlashNextPolicy.from_mapping(policy.as_dict()) == policy, name
        assert policy.as_dict()[name] == value, name


def test_flash_next_router_kernel_refuses_the_router_topk_launch_and_fold():
    # The fused router kernel and the top-k launch/fold replace the same
    # routing step; with both selected every one-token call declined the
    # launch ("fused router kernel selected") and ordinary B1 lost 9.6%
    # (qualification/runs/options-sweep-20261001).  The policy refuses it.
    import pytest

    from mlx2.adapters.flash_next_policy import FlashNextPolicy

    for topk in ("launch", "fold"):
        with pytest.raises(ValueError, match="moe_router_kernel.*moe_topk_fold"):
            FlashNextPolicy.from_mapping({"moe_router_kernel": True, "moe_topk_fold": topk})
    with pytest.raises(ValueError, match="moe_topk_fold"):
        FlashNextPolicy.from_mapping({"moe_router_kernel": True})
    policy = FlashNextPolicy.from_mapping({"moe_router_kernel": True, "moe_topk_fold": "off"})
    assert policy.environment()["MLX_QWEN4_MOE_ROUTER_KERNEL"] == "1"
    assert "MLX_QWEN4_MOE_TOPK_FOLD" not in policy.environment()
    assert FlashNextPolicy.from_mapping(policy.as_dict()) == policy
