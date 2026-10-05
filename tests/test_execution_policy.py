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


def test_flash_next_profile_pins_the_nax_gather_at_every_value(tmp_path, monkeypatch):
    # Default "fused" (port-nax-gather-20261002); an inherited MLX2_* value is
    # not stripped, so the profile pins the variable even at "off".
    import os

    from mlx2.adapters.flash_next import configure_environment

    monkeypatch.setenv("MLX2_MOE_NAX_GATHER", "gather")
    assert FlashNextPolicy().moe_nax_gather == "fused"
    assert "moe_nax_gather" not in FlashNextPolicy().as_dict()
    environment = configure_environment(tmp_path, FlashNextPolicy())
    assert environment["MLX2_MOE_NAX_GATHER"] == "fused"
    assert os.environ["MLX2_MOE_NAX_GATHER"] == "fused"
    off = FlashNextPolicy.from_mapping({"moe_nax_gather": "off"})
    assert off.as_dict()["moe_nax_gather"] == "off"
    environment = configure_environment(tmp_path, off)
    assert environment["MLX2_MOE_NAX_GATHER"] == "off"
    assert os.environ["MLX2_MOE_NAX_GATHER"] == "off"


@pytest.mark.parametrize("settings", [{"num_draft": 0}, {"num_draft": True},
    {"shared_qsa_suffix": "yes"}, {"async_qsa_promotion": 1},
    {"private_delta_min_context": -1}, {"eager_dispatch": 1},
    {"eager_dispatch_max_rows": 0}, {"eager_dispatch_stride": 0},
    {"moe_nax_gather": "on"}, {"moe_nax_gather": True},
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
        "moe_nax_gather": "off",
        "moe_rhs_pad_policy": "floor",
        "qsa_nax_min_physical_kv": 16384,
    }
    assert FlashNextPolicy.from_mapping(FlashNextPolicy().as_dict()) == FlashNextPolicy()
    names = {f.name for f in fields(FlashNextPolicy)}
    for name, value in alternatives.items():
        assert name in names, name
        policy = FlashNextPolicy.from_mapping({name: value})
        assert FlashNextPolicy.from_mapping(policy.as_dict()) == policy, name
        assert policy.as_dict()[name] == value, name


def test_varlen_sparse_moe_policy_is_default_off_and_round_trips():
    default = FlashNextPolicy()
    assert "varlen_sparse_moe" not in default.as_dict()

    selected = FlashNextPolicy.from_mapping(
        {
            "varlen_sparse_moe": {
                "minimum_padding_rows": 32,
                "minimum_padding_fraction": 0.2,
            }
        }
    )
    assert selected.as_dict()["varlen_sparse_moe"] == {
        "enabled": True,
        "minimum_padding_rows": 32,
        "minimum_padding_fraction": 0.2,
    }
    assert FlashNextPolicy.from_mapping(selected.as_dict()) == selected

    with pytest.raises(ValueError, match="varlen_sparse_moe"):
        FlashNextPolicy.from_mapping({"varlen_sparse_moe": 1})
    with pytest.raises(ValueError, match="cannot be combined with varlen_sparse_moe"):
        FlashNextPolicy.from_mapping(
            {"invariant_prefill": True, "varlen_sparse_moe": True}
        )


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


def test_flash_next_defaults_to_the_adaptive_moe_pad_and_pins_it(tmp_path, monkeypatch):
    # Default "adaptive" on Flash-Next (options-sweep-flashnext-20261002): the
    # profile pins MLX2_MOE_RHS_PAD_POLICY at every value (MLX2_* is not
    # stripped), an explicit "floor" round-trips, and the module default the
    # other models reach stays "floor".
    import os

    from mlx2.adapters.flash_next import configure_environment
    from mlx2.runtime.models import switch_layers

    assert switch_layers._rhs_pad_policy({}) == "floor"
    policy = FlashNextPolicy()
    assert policy.moe_rhs_pad_policy == "adaptive"
    assert "moe_rhs_pad_policy" not in policy.as_dict()
    monkeypatch.setenv("MLX2_MOE_RHS_PAD_POLICY", "always")
    environment = configure_environment(tmp_path, policy)
    assert environment["MLX2_MOE_RHS_PAD_POLICY"] == "adaptive"
    assert os.environ["MLX2_MOE_RHS_PAD_POLICY"] == "adaptive"
    floor = FlashNextPolicy.from_mapping({"moe_rhs_pad_policy": "floor"})
    assert floor.as_dict()["moe_rhs_pad_policy"] == "floor"
    assert FlashNextPolicy.from_mapping(floor.as_dict()) == floor
    assert configure_environment(tmp_path, floor)["MLX2_MOE_RHS_PAD_POLICY"] == "floor"
    with pytest.raises(ValueError, match="moe_rhs_pad_policy"):
        FlashNextPolicy.from_mapping({"moe_rhs_pad_policy": "sometimes"})


def test_flash_next_adaptive_pad_reaches_route_and_apc_identity(monkeypatch):
    # The pad law is process-wide numerics: serving binds moe_rhs_pad into the
    # route identity and the APCv2 execution identity whenever the effective
    # policy is not the floor, so the Flash-Next default is bound and an
    # explicit "floor" is not.
    from mlx2.runtime import apc_numerics
    from mlx2.runtime.models import switch_layers

    monkeypatch.setattr(switch_layers, "_RHS_PAD_POLICY", "floor")
    switch_layers.set_pad_policy(FlashNextPolicy().moe_rhs_pad_policy)
    assert apc_numerics.moe_rhs_pad_identity() == {
        "policy": "adaptive", "min_rows_per_expert": switch_layers._RHS_PAD_MIN_ROWS_PER_EXPERT,
    }
    identity = apc_numerics.execution_numerics_identity({})
    assert identity is None or "MLX2_MOE_RHS_PAD_POLICY" not in identity  # env-only read
    assert apc_numerics.execution_numerics_identity(None)["MLX2_MOE_RHS_PAD_POLICY"] == "adaptive"
    switch_layers.set_pad_policy("floor")
    assert apc_numerics.moe_rhs_pad_identity() is None


def test_flash_next_qsa_nax_crossover_defaults_to_8192_and_is_pinned():
    # 8192 since 2026-10-02 (qsa-nax-prefill-20261002: TTFT -2.6% at 8K,
    # neutral at 16K/32K).  The module default stays 16384, so the policy
    # default must reach the environment; an explicit 16384 round-trips and
    # leaves the variable unset (the module default then applies).
    policy = FlashNextPolicy()
    assert policy.qsa_nax_min_physical_kv == 8192
    assert "qsa_nax_min_physical_kv" not in policy.as_dict()
    assert policy.environment()["MLX_QWEN4_QSA_NAX_AUTO_MIN_PHYSICAL_KV"] == "8192"
    old = FlashNextPolicy.from_mapping({"qsa_nax_min_physical_kv": 16384})
    assert old.as_dict()["qsa_nax_min_physical_kv"] == 16384
    assert "MLX_QWEN4_QSA_NAX_AUTO_MIN_PHYSICAL_KV" not in old.environment()
    assert FlashNextPolicy.from_mapping(old.as_dict()) == old


def test_flash_next_refuses_qsa_nax_decode_beside_fused_attention_rows():
    # qsa_nax_decode takes the B1 QSA-mask calls from the default fused
    # attention rows, fails qualification on both routes and measured -15.6%
    # (options-sweep-flashnext-20261002 item 9b).  Refused unless the fused
    # rows are off, which keeps a deliberate A/B possible.
    with pytest.raises(ValueError, match="qsa_nax_decode"):
        FlashNextPolicy.from_mapping({"qsa_nax_decode": True})
    arm = FlashNextPolicy.from_mapping({"qsa_nax_decode": True, "attn_fused_rows": False})
    assert arm.environment()["MLX_QWEN4_QSA_NAX_DECODE"] == "1"
    assert FlashNextPolicy.from_mapping(arm.as_dict()) == arm
