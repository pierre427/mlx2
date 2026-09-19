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


def test_flash_adapter_explicitly_selects_observable_parity_paths(tmp_path, monkeypatch):
    from mlx2.adapters.flash_next import configure_environment

    monkeypatch.setenv("MLX_GDN_CORE", "1")
    monkeypatch.setenv("MLX_GDN_UNBOUND_EXPERIMENT", "1")
    environment = configure_environment(tmp_path, FlashNextPolicy())
    assert environment["MLX_QWEN4_QSA_SCATTER_CHOSEN"] == "1"
    assert environment["MLX_QWEN4_MOE_FUSED_GATE_UP"] == "1"
    assert environment["MLX_QWEN4_FUSED_EXPERT_KERNEL"] == "auto"
    assert environment["MLX_QWEN4_FUSED_GDN_REPLAY_ROLLBACK"] == "1"
    assert environment["MLX_GDN_PACKED"] == "1"
    assert environment["MLX_GDN_CORE"] == "0"
    assert "MLX_GDN_UNBOUND_EXPERIMENT" not in __import__("os").environ


@pytest.mark.parametrize("settings", [{"num_draft": 0}, {"num_draft": True},
    {"shared_qsa_suffix": "yes"}, {"async_qsa_promotion": 1},
    {"private_delta_min_context": -1}, {"bogus": True}])
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
