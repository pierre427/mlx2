"""Synthetic exact-artifact CPU evidence across distinct adapter math/layouts.

These generated checkpoints are named and hashed in pytest properties. They
exercise architecture plumbing, not qualification of a released checkpoint.
"""

import hashlib
import subprocess

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models.cache import make_prompt_cache
from mlx2.runtime.streamed_load import (
    WeightStreamRequest,
    force_stock_expert_arithmetic,
    load_streamed,
)
from mlx2.runtime.weight_stream import apc_weight_stream_fingerprint


@pytest.mark.parametrize(
    "family", ["qwen-hybrid", "laguna", "nemotron-latent", "hy-v3"]
)
@pytest.mark.parametrize("adaptive", [False, True])
def test_exact_logits_and_cache_state_survive_shrink_topic_change_and_regrowth(
    tmp_path, monkeypatch, record_property, family, adaptive
):
    import test_disk_weight_streaming as qwen
    import test_weight_stream_unquantized_auto_cpu as other

    if family == "qwen-hybrid":
        root, weight_map, records = qwen._moe_checkpoint(tmp_path, monkeypatch)
        resident = qwen._resident(
            qwen._moe_model(monkeypatch, fused=True), root, weight_map
        )
        model = qwen._moe_model(monkeypatch, fused=True)
        quantize = qwen._quantizer(model)
    else:
        if family == "laguna":
            root, weight_map, records = other._laguna_checkpoint(tmp_path)
            build = other._laguna
        elif family == "nemotron-latent":
            root, weight_map, records = other._nemotron_checkpoint(tmp_path)
            build = other._nemotron
        else:
            root, weight_map, records = other._hy_checkpoint(tmp_path, per_expert=True)
            build = other._hy
        resident = build()
        resident = other._resident(
            resident,
            root,
            weight_map,
            other._hy_quantize(resident) if family == "hy-v3" else lambda _: None,
        )
        model = build()
        quantize = other._hy_quantize(model) if family == "hy-v3" else lambda _: None
    digest = hashlib.sha256()
    for name in sorted(set(weight_map.values())):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    record_property("checkpoint", f"synthetic-{family}@{digest.hexdigest()}")
    record_property(
        "source_revision",
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    record_property("evidence_kind", "synthetic_cpu_functional_not_qualification")
    kwargs = {"shard_prune": model.shard_prune} if hasattr(model, "shard_prune") else {}
    loaded = load_streamed(
        model,
        root,
        sorted(set(weight_map.values())),
        request=WeightStreamRequest(
            mode="moe_experts",
            budget_bytes=1 << 30,
            read_workers=2,
            elastic=True,
            adaptive_residency={"enabled": adaptive, "decay_every": 2},
        ),
        sanitize=model.sanitize,
        quantize=quantize,
        records=records,
        weight_map=weight_map,
        top_k=2,
        on_installed=lambda manager: setattr(
            manager, "forced_stock", force_stock_expert_arithmetic(model)
        ),
        **kwargs,
    )
    m = loaded.manager
    m.begin_serving()
    try:
        from mlx.utils import tree_flatten

        reference_cache = make_prompt_cache(resident)
        streamed_cache = make_prompt_cache(model)
        namespace = apc_weight_stream_fingerprint("test", m)
        rounds = [[3, 7, 1, 12], [9], [30], [4], [18]]
        for index, tokens in enumerate(rounds):
            if index == 1:
                m.rebalance_headroom(available_bytes=0, protected_bytes=0)
                assert all(c.capacity == 2 for c in m.caches.values())
            if index == 3:
                m.rebalance_headroom(available_bytes=1 << 30, protected_bytes=0)
                assert all(
                    c.capacity == m._maximum_capacities[path]
                    for path, c in m.caches.items()
                )
            expected = resident(mx.array([tokens]), cache=reference_cache)
            actual = model(mx.array([tokens]), cache=streamed_cache)
            mx.eval(expected, actual)
            assert np.array_equal(np.array(actual), np.array(expected))
            for left, right in zip(reference_cache, streamed_cache):
                ls = tree_flatten(left.state)
                rs = tree_flatten(right.state)
                assert [p for p, _ in ls] == [p for p, _ in rs]
                for (_, a), (_, b) in zip(ls, rs):
                    if isinstance(a, mx.array):
                        assert np.array_equal(np.array(a), np.array(b))
            assert apc_weight_stream_fingerprint("test", m) == namespace
        assert m.counters()["stream_page_ins_total"] > 0
        assert m.receipt()["elastic_memory"]["qualification"] == "unqualified"
        assert m.stats.peak_transient_bytes <= m.plan.transient_bound_bytes
    finally:
        m.close()


@pytest.mark.parametrize("lookup", [False, True])
def test_live_serving_worker_lends_and_recovers_expert_capacity_with_apcv2(
    tmp_path, monkeypatch, record_property, lookup
):
    import test_disk_weight_streaming as qwen
    from route_harness import collect, make_adapter, patch_host

    from mlx2.serving import ServingEngine

    patch_host(monkeypatch)
    root, weight_map, records = qwen._moe_checkpoint(tmp_path, monkeypatch)
    model = qwen._moe_model(monkeypatch, fused=True)
    digest = hashlib.sha256()
    for name in sorted(set(weight_map.values())):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    record_property("checkpoint", f"synthetic-qwen-hybrid@{digest.hexdigest()}")
    record_property(
        "source_revision",
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    observations = []
    base = make_adapter(model, 64)

    class Adapter(base):
        weight_streaming_modes = ("moe_experts",)

        def __init__(self, path, *, weight_streaming):
            super().__init__(path)
            loaded = load_streamed(
                self.model,
                root,
                sorted(set(weight_map.values())),
                request=weight_streaming,
                sanitize=model.sanitize,
                quantize=qwen._quantizer(model),
                records=records,
                weight_map=weight_map,
                shard_prune=model.shard_prune,
                top_k=2,
                on_installed=lambda manager: setattr(
                    manager, "forced_stock", force_stock_expert_arithmetic(model)
                ),
            )
            self.weight_stream = loaded.manager
            self.weight_stream.begin_serving()
            original = self.weight_stream.rebalance_headroom

            def tracked(**kwargs):
                # Simulated pressure at the first real admission boundary;
                # subsequent calls can recover capacity on the same worker.
                if not observations:
                    kwargs["available_bytes"] = 0
                result = original(**kwargs)
                observations.append(
                    (
                        result["revision"],
                        tuple(c.capacity for c in self.weight_stream.caches.values()),
                    )
                )
                return result

            self.weight_stream.rebalance_headroom = tracked

        def close(self):
            self.weight_stream.close()

    engine = ServingEngine(
        str(root),
        adapter_factory=Adapter,
        max_lanes=1,
        max_inflight=2,
        qualification_mode=True,
        mtp=False,
        prompt_lookup=lookup,
        prefill_step=8,
        execution_policy={
            "moe_expert_streaming": {
                "enabled": True,
                "cache_gib": 1,
                "elastic": True,
                "adaptive_residency": {"enabled": True},
            }
        },
    )
    try:
        assert engine.ready.wait(30), engine.error
        assert engine.error is None, engine.error
        request = {"tokens": [3, 7, 1, 12] * 4, "max_tokens": 4, "temperature": 0}
        cold = collect(engine.submit(dict(request)), timeout=30)
        assert "error" not in cold, cold
        warm = collect(engine.submit(dict(request)), timeout=30)
        assert "error" not in warm, warm
        assert cold["tokens"] == warm["tokens"]
        assert warm["receipt"]["cached_tokens"] > 0
        assert min(observations[0][1]) == 2
        assert any(max(capacities) > 2 for _, capacities in observations[1:])
        assert len(observations) >= 4  # ingress plus the live execution rounds
        status = engine.weight_streaming_status()["receipt"]
        assert status["elastic_memory"]["revision"] == observations[-1][0]
        assert status["adaptive_residency"]["qualification"] == "unqualified"
    finally:
        engine.close()
