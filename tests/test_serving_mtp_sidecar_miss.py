"""The self-MTP route safely handles warm target hits without draft state."""

from types import SimpleNamespace as NS

from mlx2 import memory, serving
from mlx2.runtime import apc_v2, generate, os_memory
from mlx2.server import collect_nonstream_job


def test_mtp_route_marks_prompt_boundary_hit_for_plain_fallback(monkeypatch):
    inserted = []
    closed = []

    class Branch(list):
        def close(self):
            closed.append(True)

    class APC:
        def __init__(self, **_kw):
            self.apc_stats = {}

        def key(self, *_a, **_kw):
            return "key"

        def lookup(self, _key, tokens, **_kw):
            # An ordinary-route checkpoint: warm target prefix, no sidecar.
            return NS(
                cache=Branch([object()]),
                cached_tokens=len(tokens) - 1,
                remaining_tokens=list(tokens[-1:]),
                miss_reason=None,
                sidecar=None,
                hit=True,
            )

        def store(self, *_a, **_kw):
            pass

        def spill_idle_entries(self):
            pass

        def evict_oldest_unleased(self):
            return False

        def clear(self):
            pass

    class Batch:
        scheduler_stats = {}

        def __init__(self, *_a, **_kw):
            self.lanes = {}

        def insert(self, prompts, *, caches, all_tokens, **kw):
            inserted.append(
                (
                    prompts,
                    caches,
                    all_tokens,
                    kw.get("mtp_states"),
                    kw.get("self_mtp_configs"),
                )
            )
            self.lanes[0] = 0
            return [0]

        def pop_prompt_boundary(self, _uid):
            return None

        def next(self):
            if not self.lanes:
                return [], []
            self.lanes.pop(0)
            return [], [
                NS(
                    uid=0, execution_width=1, finish_reason="length", token=3,
                    mtp_state=None, all_tokens=[1, 2, 3], prompt_cache=[],
                    mtp_receipt=None,
                )
            ]

        def remove(self, uids):
            for uid in uids:
                self.lanes.pop(uid, None)

        def close(self):
            pass

    class Detok:
        last_segment = "x"

        def reset(self):
            pass

        def add_token(self, _t):
            pass

        def finalize(self):
            pass

    class Parser:
        stopped = False
        tool_count = 0

        def push(self, text, final=False):
            return [{"content": text}]

    class Adapter:
        max_context = 100000
        identity = {"fingerprint": "fake"}
        environment = {}
        layout = "fake"
        model = None
        tokenizer = NS(vocab_size=100, eos_token_ids=[], detokenizer=Detok())

        def __init__(self, _path):
            pass

        def profile_name(self, _mtp):
            return "fake"

        def execution_config(self, **_kw):
            return {"num_draft": 2}

        def prompt_tokens(self, _request):
            return [1, 2, 3, 4]

        def output_parser(self, _request):
            return Parser()

        def diagnostics(self):
            return {}

        def close(self):
            pass

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "fake"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    monkeypatch.setattr(apc_v2, "APCv2", APC)
    monkeypatch.setattr(generate, "BatchGenerator", Batch)
    engine = serving.ServingEngine(
        "fake", adapter_factory=Adapter, qualification_mode=True, mtp=True,
        max_lanes=2, max_inflight=4,
    )
    try:
        assert engine.ready.wait(5)
        body = {"prompt": "hi", "max_tokens": 4}
        job = engine.submit(body)
        _choice, _usage, receipt = collect_nonstream_job(job, body, chat=False)
    finally:
        engine.close()
    prompts, caches, all_tokens, mtp_states, configs = inserted[0]
    assert prompts == [[4]] and caches[0] is not None and all_tokens == [[1, 2, 3]]
    assert mtp_states == [None]
    assert configs[0]["target_only_plain_fallback"] is True
    assert closed == [True], "the warm lease is released only after the request"
    assert receipt["cached_tokens"] == 3
    assert engine.counts["mtp_sidecar_missing_plain_fallbacks"] == 1
    assert engine.counts["mtp_sidecar_missing_misses"] == 0


def test_checkpoint_publication_failure_does_not_escape():
    from collections import Counter

    engine = serving.ServingEngine.__new__(serving.ServingEngine)
    engine.counts = Counter()

    class APC:
        def store(self, *_a, **_kw):
            raise ValueError("cannot freeze a speculating cache")

    engine._publish_checkpoint(APC(), "key", [1, 2], [object()], sidecar=None)
    assert engine.counts["apcv2_store_failures"] == 1
