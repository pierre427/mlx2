import time
from types import SimpleNamespace as NS

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mlx2 import memory, serving
from mlx2.runtime import apc_v2, generate, os_memory
from mlx2.runtime.models.cache import KVCache, RotatingKVCache
from mlx2.server import collect_parallel_samples
from mlx2.serving import ServingEngine, cache_capsule_policy


def test_bind_failure_removes_inserted_lane_and_clears_pending_group():
    class Prepared:
        def __init__(self): self.closes = 0
        def close(self): self.closes += 1
    class Batch:
        def __init__(self): self.removed = []; self.cancelled = []
        def bind_cache_capsule(self, *_args): raise RuntimeError("bind failed")
        def remove(self, uids): self.removed.extend(uids)
        def cancel_cache_capsule_group(self, group, reason):
            self.cancelled.append((group, reason)); return True
    prepared, batch = Prepared(), Batch()
    first = NS(
        uid=17, fanout_group="g", cache_capsule=prepared, cache_capsule_rows=2
    )
    sibling = NS(
        uid=None, fanout_group="g", cache_capsule=prepared, cache_capsule_rows=2
    )
    engine = object.__new__(ServingEngine)
    engine.lock = __import__("threading").Lock()
    engine.jobs = {"first": first, "sibling": sibling}
    engine.fanout_capsules = {"g": prepared}
    with pytest.raises(RuntimeError, match="bind failed"):
        engine._bind_pending_cache_capsule(batch, first)
    assert batch.removed == [17]
    assert batch.cancelled == [("g", "attachment_failed")]
    assert prepared.closes == 1 and not engine.fanout_capsules
    assert first.cache_capsule is None and sibling.cache_capsule is None


def test_queued_sibling_cancellation_declines_prepared_group():
    class Prepared:
        def __init__(self): self.closes = 0
        def close(self): self.closes += 1
    class Batch:
        def __init__(self): self.cancelled = []
        def cancel_cache_capsule_group(self, group, reason):
            self.cancelled.append((group, reason)); return True
    prepared, batch = Prepared(), Batch()
    jobs = [
        NS(fanout_group="g", cache_capsule=prepared, cache_capsule_rows=2)
        for _ in range(2)
    ]
    engine = object.__new__(ServingEngine)
    engine.lock = __import__("threading").Lock()
    engine.jobs = {str(i): job for i, job in enumerate(jobs)}
    engine.fanout_capsules = {"g": prepared}
    assert engine._cancel_pending_cache_capsule(
        batch, jobs[0], "queued_member_cancelled"
    )
    assert prepared.closes == 1 and not engine.fanout_capsules
    assert batch.cancelled == [("g", "queued_member_cancelled")]
    assert all(job.cache_capsule is None for job in jobs)


def test_deferred_admission_timeout_declines_prepared_group():
    class Prepared:
        def __init__(self): self.closes = 0
        def close(self): self.closes += 1
    class Batch:
        def __init__(self): self.cancelled = []
        def cancel_cache_capsule_group(self, group, reason):
            self.cancelled.append((group, reason)); return True
    prepared, batch = Prepared(), Batch()
    job = NS(fanout_group="g", cache_capsule=prepared, cache_capsule_rows=2)
    engine = object.__new__(ServingEngine)
    engine.lock = __import__("threading").Lock()
    engine.jobs = {"job": job}
    engine.fanout_capsules = {"g": prepared}
    engine.counts = {"memory_admission_timeouts": 0}
    terminal = []
    engine._finish = lambda finished, event: terminal.append((finished, event))
    engine._fail_deferred_admission_timeout(batch, job)
    assert prepared.closes == 1 and not engine.fanout_capsules
    assert batch.cancelled == [("g", "memory_admission_timeout")]
    assert job.cache_capsule is None
    assert terminal[0][1]["status"] == 429
    assert engine.counts["memory_admission_timeouts"] == 1


def test_occupied_scheduler_width_declines_capsule_before_preparation():
    engine = object.__new__(ServingEngine)
    engine.max_lanes = 3
    engine.counts = {"cache_capsule_width_fallbacks": 0}
    siblings = [NS(cache_capsule_receipt=None) for _ in range(2)]
    # The leader plus one unrelated lane are active, so both siblings cannot
    # attach before the next scheduler step at width three.
    assert not engine._cache_capsule_fanout_fits({1: object(), 2: object()}, siblings)
    assert engine.counts["cache_capsule_width_fallbacks"] == 1
    assert all(
        sibling.cache_capsule_receipt == {
            "schema": "mlx2.cache-capsule.v1",
            "status": "fallback",
            "reason": "scheduler_width_unavailable",
            "rows": 2,
        }
        for sibling in siblings
    )


def test_cache_capsule_policy_rejects_unknown_or_unqualified_selection(tmp_path):
    with pytest.raises(ValueError, match="unknown cache capsule"):
        cache_capsule_policy({"enabled": True, "mystery": 1})
    with pytest.raises(ValueError, match="qualification mode"):
        ServingEngine("unused", cache_capsules=True)
    with pytest.raises(ValueError, match="qualification mode"):
        ServingEngine(
            "unused", cache_dir=str(tmp_path), persistent_block_bytes=4096
        )
    with pytest.raises(ValueError, match="requires cache_dir"):
        ServingEngine(
            "unused", qualification_mode=True, persistent_block_bytes=4096
        )


def test_cache_capsule_policy_is_ordinary_only():
    with pytest.raises(ValueError, match="ordinary decode"):
        ServingEngine(
            "unused", qualification_mode=True, cache_capsules=True, mtp=True
        )


def test_qualification_serving_fanout_engages_cache_capsule(monkeypatch):
    def cache():
        value = mx.arange(6, dtype=mx.float32).astype(mx.bfloat16).reshape(1, 1, 3, 2)
        result = KVCache(); result.update_and_fetch(value, value); mx.eval(result.state)
        return result

    def rotating_cache():
        value = mx.arange(6, dtype=mx.float32).reshape(1, 1, 3, 2)
        result = RotatingKVCache(max_size=8, keep=0)
        result.update_and_fetch(value, value)
        mx.eval(result.state)
        return result

    class Batch:
        scheduler_stats = {}
        def __init__(self, *_args, **_kwargs):
            self.uid = 0; self.lanes = {}; self.boundaries = {}; self.receipts = {}
        def insert(self, *_args, **_kwargs):
            uid = self.uid; self.uid += 1
            self.lanes[uid] = {"prompted": False}; return [uid]
        def bind_cache_capsule(self, group, uid, prepared, expected):
            members = getattr(self, "members", [])
            members.append(uid); self.members = members
            if len(members) == expected:
                for member in members:
                    self.receipts[member] = {
                        "schema": "mlx2.cache-capsule.v1", "status": "engaged",
                        "rows": expected, "planes": 2,
                        "capsule_planes": 1, "ordinary_planes": 1,
                    }
                return True
            return False
        def cancel_cache_capsule_group(self, *_args): return True
        def pop_cache_capsule_receipt(self, uid): return self.receipts.pop(uid, None)
        def pop_post_prefill_receipt(self, _uid): return None
        def pop_prompt_boundary(self, uid): return self.boundaries.pop(uid, None)
        def take_atomic_cohort_failures(self): return ()
        def next(self):
            prompts, responses = [], []
            for uid, lane in list(self.lanes.items()):
                if not lane["prompted"]:
                    lane["prompted"] = True
                    if uid == 0:
                        self.boundaries[uid] = {
                            "committed_only": True, "tokens": [1, 2, 3],
                            # A supported model may mix global plain KV planes
                            # with local rotating planes. Capsule the former and
                            # preserve the latter through its ordinary merge.
                            "target_cache": [cache(), rotating_cache()],
                            "mtp_state": None,
                        }
                    prompts.append(NS(uid=uid, end_of_prompt=True)); continue
                responses.append(NS(
                    uid=uid, execution_width=len(self.lanes), finish_reason="length",
                    token=4, mtp_state=None, all_tokens=[1, 2, 3, 4],
                    prompt_cache=[cache()], mtp_receipt=None,
                ))
                del self.lanes[uid]
            time.sleep(0.001)
            return prompts, responses
        def remove(self, uids):
            for uid in uids: self.lanes.pop(uid, None)
        def close(self): pass

    class Detok:
        last_segment = "x"
        def reset(self): pass
        def add_token(self, _token): pass
        def finalize(self): pass
    class Parser:
        stopped = False; tool_count = 0
        def push(self, text, final=False): return [{"content": text}]
    class Adapter:
        max_context = 64; identity = {"fingerprint": "fake"}; environment = {}
        layout = "fake-layout"; model = None
        tokenizer = NS(vocab_size=32, eos_token_ids=[], detokenizer=Detok())
        def __init__(self, _path): pass
        def profile_name(self, _mtp): return "fake"
        def execution_config(self, **_kwargs): return {"num_draft": 0}
        def prompt_tokens(self, _request): return [1, 2, 3]
        def output_parser(self, _request): return Parser()
        def diagnostics(self): return {}
        def close(self): pass

    # Latest-main fanout hardening requires each sibling to lease the exact
    # committed APCv2 boundary before optional capsule replacement. Preserve
    # the real APC/capsule pool while making those resident test branches
    # deterministic (the tiny synthetic cache can otherwise be reclaimed by
    # the real byte estimator before the worker performs its boundary lookup).
    original_store = apc_v2.APCv2.store
    original_lookup = apc_v2.APCv2.lookup

    def observed_store(instance, key, tokens, prompt_cache, **kwargs):
        result = original_store(instance, key, tokens, prompt_cache, **kwargs)
        if result.stored:
            instance._capsule_test_tokens = list(tokens)
        return result

    class Branch(list):
        def close(self):
            for plane in self:
                close = getattr(plane, "close", None)
                if callable(close):
                    close()

    def exact_lookup(instance, key, tokens, **kwargs):
        requested = list(tokens)
        committed = getattr(instance, "_capsule_test_tokens", None)
        if committed and requested[:len(committed)] == committed:
            return apc_v2.APCLookup(
                cache=Branch([cache(), rotating_cache()]),
                remaining_tokens=requested[len(committed):],
                cached_tokens=len(committed),
                hit=True,
                hit_kind="exact",
                miss_reason=None,
                capsule_generation=instance.capsule_generation.current,
            )
        return original_lookup(instance, key, requested, **kwargs)

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "source"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    monkeypatch.setattr(apc_v2.APCv2, "store", observed_store)
    monkeypatch.setattr(apc_v2.APCv2, "lookup", exact_lookup)
    monkeypatch.setattr(generate, "BatchGenerator", Batch)
    engine = ServingEngine(
        "fake", adapter_factory=Adapter, qualification_mode=True, mtp=False,
        max_lanes=3, max_inflight=3,
        cache_capsules={"enabled": True, "backend": "cpu", "fallback": "cpu"},
    )
    assert engine.ready.wait(5)
    try:
        body = {"max_tokens": 1, "n": 1, "temperature": 0}
        jobs = engine.submit_many([dict(body, seed=i) for i in range(3)])
        results = collect_parallel_samples(jobs, body, chat=True)
    finally:
        engine.close()
    assert engine.counts["cache_capsule_prepared"] == 1
    sibling_receipts = [result[2]["cache_capsule"] for result in results[1:]]
    assert all(receipt["status"] == "engaged" for receipt in sibling_receipts)
    assert all(receipt["planes"] == 2 for receipt in sibling_receipts)
    assert all(receipt["capsule_planes"] == 1 for receipt in sibling_receipts)
    assert all(receipt["ordinary_planes"] == 1 for receipt in sibling_receipts)
    with pytest.raises(ValueError, match="ordinary decode"):
        ServingEngine(
            "unused", qualification_mode=True, cache_capsules=True,
            mtp=False, prompt_lookup=True,
        )
