"""Real APCv2, native serving and mixed external draft cohorts on CPU.

Tiny synthetic heads establish lifecycle integration, not model qualification
or performance. DPara is deliberately absent: it has no batched serving route.
"""

import copy
import hashlib
import json
from threading import Event
from typing import ClassVar

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from test_lilicorr_serving_cpu import pair as lilicorr_pair
from test_standard_xpress_serving_cpu import drain, reference, tiny

from mlx2.runtime.apc_v2 import APCKey, APCv2
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.sample_utils import LaneRNG


def pair(kind, windows=None):
    model, draft = tiny() if kind == "xpress" else lilicorr_pair()
    if windows is not None:
        draft = type(draft)(draft.config, draft_attention_windows=windows).bind(model)
    mx.eval(model.parameters(), draft.parameters())
    return model, draft


def binding(kind, draft, revision="synthetic-head-v1"):
    return hashlib.sha256(
        json.dumps([kind, revision, draft.receipt_settings], sort_keys=True).encode()
    ).hexdigest()


def scheduler(model, draft, revision, **kwargs):
    return ExternalDraftBatchGenerator(
        model,
        draft_model=draft,
        binding=revision,
        num_draft=2,
        prefill_step_size=2,
        completion_batch_size=4,
        **kwargs,
    )


def serving_engine(monkeypatch, kind, model, draft):
    """Actual ServingEngine worker, APCv2 owner, admission and batch lifecycle."""
    from test_approximate_kv_serving import Detok, Parser

    from mlx2 import memory, serving
    from mlx2.contracts import Capability, ModelDescriptor, StatePlane
    from mlx2.runtime import os_memory

    monkeypatch.setattr(
        serving, "runtime_identity", lambda: {"source_sha256": "cpu-test-source"}
    )
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    revision = binding(kind, draft)
    batches = []

    class Tokenizer:
        vocab_size = 9
        eos_token_ids: ClassVar[list] = []

        @property
        def detokenizer(self):
            return Detok()

    class Adapter:
        max_context = 128
        identity: ClassVar[dict] = {"fingerprint": revision}
        environment: ClassVar[dict] = {}
        layout = revision
        tokenizer = Tokenizer()
        descriptor = ModelDescriptor(
            model_type="qwen3",
            family="qwen3",
            variant=kind,
            state_planes=frozenset({StatePlane.ATTENTION_KV, StatePlane.DRAFT}),
            capabilities=frozenset(
                {
                    Capability.TEXT,
                    Capability.CONTINUOUS_BATCH,
                    Capability.PREFIX_REUSE,
                    Capability.APC_V2,
                    Capability.EXTERNAL_DRAFT,
                    Capability.STREAMING,
                }
            ),
            cache_layout=revision,
        )

        def __init__(self, _path):
            # Device placement is explicit in the serving worker as well.
            mx.set_default_device(mx.cpu)
            self.model = model

        def profile_name(self, _mtp):
            return "tiny-" + kind

        def execution_config(self, *, max_lanes, prefill_step):
            return {
                "persistent": True,
                "num_draft": 2,
                "backend": "external_draft",
                "rate_gate": False,
                "prefill_step_size": prefill_step,
            }

        def create_external_batch(self, **kwargs):
            batch = ExternalDraftBatchGenerator(
                model, draft_model=draft, binding=revision, num_draft=2, **kwargs
            )
            batches.append(batch)
            return batch

        def prompt_tokens(self, request):
            return list(request["tokens"])

        def output_parser(self, _request):
            return Parser()

        def diagnostics(self):
            return {}

        def close(self):
            pass

    engine = serving.ServingEngine(
        "tiny",
        adapter_factory=Adapter,
        qualification_mode=True,
        mtp=False,
        max_lanes=4,
        prefill_step=2,
    )
    assert engine.ready.wait(30), engine.error
    assert engine.error is None and isinstance(engine.apc, APCv2)
    return engine, batches


def collect(job):
    text = ""
    while True:
        event = job.events.get(timeout=30)
        if "delta" in event:
            text += event["delta"].get("content", "")
        if "error" in event:
            return {
                "tokens": [int(token) for token in text.split()],
                "error": event["error"],
            }
        if "finish_reason" in event:
            return {
                "tokens": [int(token) for token in text.split()],
                "receipt": event["receipt"],
            }


@pytest.mark.parametrize("kind", ["xpress", "lilicorr"])
@pytest.mark.parametrize("windows", [None, [2]])
def test_native_serving_cold_then_actual_paired_apcv2_warm_hit(
    monkeypatch, kind, windows
):
    model, draft = pair(kind, windows)
    engine, batches = serving_engine(monkeypatch, kind, model, draft)
    prompt = [1, 2, 3, 4, 5]
    try:
        cold = collect(
            engine.submit(
                {"tokens": prompt, "max_tokens": 5, "temperature": 0, "top_k": 0}
            )
        )
        assert "error" not in cold and cold["receipt"]["cached_tokens"] == 0
        assert cold["tokens"] == reference(model, prompt, 5)
        # Continue the exact finished transcript, including its pending bonus.
        warm_prompt = prompt + cold["tokens"]
        before = engine.apc.apc_stats["hits"]
        warm = collect(
            engine.submit(
                {"tokens": warm_prompt, "max_tokens": 4, "temperature": 0, "top_k": 0}
            )
        )
        assert "error" not in warm
        assert warm["receipt"]["cached_tokens"] == len(warm_prompt) - 1
        assert engine.apc.apc_stats["hits"] > before
        assert warm["tokens"] == reference(model, warm_prompt, 4)
        assert batches[0].scheduler_stats["paired_cache_resumes"] >= 1
        assert engine.error is None and engine.thread.is_alive()
        assert batches[0].scheduler_stats["draft_fallbacks"] == 0
    finally:
        engine.close()


@pytest.mark.parametrize("kind", ["xpress", "lilicorr"])
def test_mixed_lengths_temperatures_budgets_and_depth_groups_keep_lane_laws(
    monkeypatch, kind
):
    model, draft = pair(kind, [2])
    revision = binding(kind, draft)
    prompts = [[1, 2, 3, 4, 5], [2, 3], [4], [5, 4, 3]]
    budgets, temperatures = [9, 9, 1, 2], [0, 0.8, 0, 0.6]
    calls = []
    real = model.forward_with_taps

    def forward(tokens, *args, **kwargs):
        calls.append(tuple(tokens.shape))
        return real(tokens, *args, **kwargs)

    monkeypatch.setattr(model, "forward_with_taps", forward)
    batch = scheduler(model, draft, revision)
    uids = batch.insert(
        prompts,
        max_tokens=budgets,
        lane_rngs=[LaneRNG(100 + i) for i in range(4)],
        sampling_configs=[
            {"sampling_temp": temperature} for temperature in temperatures
        ],
    )
    # Stage the different prefill chunks before launching the common cohort.
    for lane in batch.lanes.values():
        while lane.remaining:
            batch._prefill(lane)
    output, final = drain(batch)
    assert (2, 3) in calls  # Two mixed greedy/sampled rows, each two proposals.
    assert any(length == 1 for _, length in calls)  # Final budget group.
    assert any(length == 2 for _, length in calls)  # One-proposal budget group.
    for i, uid in enumerate(uids):
        assert len(output[uid]) == budgets[i]
        final[uid].cache_sidecar.validate(revision, len(final[uid].all_tokens))
        assert final[uid].speculative_receipt["draft_settings"][
            "draft_attention_windows"
        ] == [2]
        # Cohort membership must preserve each lane's random stream and output.
        solo = scheduler(model, draft, revision)
        solo.insert(
            [prompts[i]],
            max_tokens=[budgets[i]],
            lane_rngs=[LaneRNG(100 + i)],
            sampling_configs=[{"sampling_temp": temperatures[i]}],
        )
        expected, _ = drain(solo)
        assert output[uid] == expected[0]
        if temperatures[i] == 0:
            assert output[uid] == reference(model, prompts[i], budgets[i])


@pytest.mark.parametrize("kind", ["xpress", "lilicorr"])
def test_actual_apcv2_target_only_hit_rebuilds_both_planes_without_stale_pair(kind):
    model, draft = pair(kind, [2])
    revision = binding(kind, draft)
    prompt = [1, 2, 3, 4, 5]
    target_cache = model.make_cache()
    model.prefill_body(mx.array([prompt[:-1]]), target_cache, [0, 2])
    mx.eval([cache.state for cache in target_cache])
    apc = APCv2(max_size=2, layout_name=revision)
    key = APCKey(kind, revision=revision, cache_layout_fingerprint=revision)
    apc.store(key, prompt[:-1], target_cache)  # Explicitly no draft sidecar.
    hit = apc.lookup(key, prompt)
    assert hit.hit and hit.cached_tokens == len(prompt) - 1 and hit.sidecar is None
    batch = scheduler(model, draft, revision)
    try:
        uid = batch.insert(
            [hit.remaining_tokens],
            max_tokens=[5],
            caches=[hit.cache],
            all_tokens=[prompt[: hit.cached_tokens]],
            cache_states=[hit.sidecar],
        )[0]
        lane = batch.lanes[uid]
        assert lane.history == [] and list(lane.remaining) == prompt
        assert lane.cache is not hit.cache
        assert all(cache.offset == 0 for cache in lane.cache + lane.draft_cache)
        output, final = drain(batch)
        assert output[uid] == reference(model, prompt, 5)
        assert batch.scheduler_stats["paired_cache_resumes"] == 0
        final[uid].cache_sidecar.validate(revision, len(final[uid].all_tokens))
    finally:
        hit.cache.close()
        batch.close()
        assert apc.apc_stats["cow"]["active_leases"] == 0
        apc.clear(release_memory=False)


@pytest.mark.parametrize(
    "kind,changed",
    [
        ("xpress", "window"),
        ("xpress", "head"),
        ("xpress", "passes"),
        ("lilicorr", "window"),
        ("lilicorr", "head"),
    ],
)
def test_apcv2_sidecar_refuses_different_window_pass_or_head_revision(kind, changed):
    model, draft = pair(kind, [2])
    revision = binding(kind, draft)
    first = scheduler(model, draft, revision)
    first.insert([[1, 2, 3]], max_tokens=[4])
    _, final = drain(first)
    end = final[0]
    apc = APCv2(max_size=2, layout_name=revision)
    key = APCKey(kind, revision=revision, cache_layout_fingerprint=revision)
    apc.store(key, end.all_tokens, end.prompt_cache, sidecar=end.cache_sidecar)
    hit = apc.lookup(key, end.all_tokens + [end.token])
    assert hit.hit and hit.hit_kind == "external_draft_sidecar"
    if changed == "window":
        changed_draft = type(draft)(draft.config, draft_attention_windows=[3]).bind(
            model
        )
        changed_revision = binding(kind, changed_draft)
    elif changed == "passes":
        args = copy.deepcopy(draft.config)
        args.xpress_num_passes = 1
        changed_draft = type(draft)(args, draft_attention_windows=[2]).bind(model)
        changed_revision = binding(kind, changed_draft)
    else:
        changed_draft = draft
        changed_revision = binding(kind, draft, revision="synthetic-head-v2")
    try:
        assert not apc.lookup(
            APCKey(
                kind,
                revision=changed_revision,
                cache_layout_fingerprint=changed_revision,
            ),
            end.all_tokens + [end.token],
        ).hit
        resumed = scheduler(model, changed_draft, changed_revision)
        with pytest.raises(ValueError, match="revision"):
            resumed.insert(
                [[end.token]],
                max_tokens=[4],
                caches=[hit.cache],
                all_tokens=[end.all_tokens],
                cache_states=[hit.sidecar],
            )
        assert not resumed.lanes
    finally:
        hit.cache.close()
        first.close()
        apc.clear(release_memory=False)


@pytest.mark.parametrize("kind", ["xpress", "lilicorr"])
def test_native_serving_cancellation_during_mixed_cohort_preserves_healthy_lane(
    monkeypatch, kind
):
    model, draft = pair(kind, [2])
    cohort_started, release = Event(), Event()
    calls = []
    original = model.forward_with_taps

    def forward(tokens, *args, **kwargs):
        calls.append(tuple(tokens.shape))
        if tokens.shape[0] >= 2 and not cohort_started.is_set():
            cohort_started.set()
            assert release.wait(10), "test never released the CPU verification cohort"
        return original(tokens, *args, **kwargs)

    monkeypatch.setattr(model, "forward_with_taps", forward)
    engine, batches = serving_engine(monkeypatch, kind, model, draft)
    try:
        cancelled = engine.submit(
            {
                "tokens": [1, 2, 3, 4, 5],
                "max_tokens": 32,
                "temperature": 0.8,
                "top_k": 0,
                "seed": 44,
            }
        )
        healthy = engine.submit(
            {"tokens": [2, 3], "max_tokens": 32, "temperature": 0, "top_k": 0}
        )
        assert cohort_started.wait(10), "native serving never formed a mixed CPU cohort"
        cancelled.cancelled.set()
        release.set()
        abandoned, retained = collect(cancelled), collect(healthy)
        assert abandoned.get("error") == "cancelled"
        assert "error" not in retained and retained["tokens"] == reference(
            model, [2, 3], 32
        )
        assert any(width >= 2 for width, _ in calls)
        assert batches[0].scheduler_stats["cancelled"] >= 1
        assert engine.thread.is_alive() and engine.error is None
    finally:
        release.set()
        engine.close()
