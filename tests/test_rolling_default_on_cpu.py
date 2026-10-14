"""Rolling prefill checkpoints are on by default (Pierre, 2026-10-14).

A default-constructed engine selects them on every route that supports them,
so a prefill cancelled between chunks keeps its progress for the retry (gufo
#491 shape).  Routes that cannot support them step aside and say why; an
explicit request on such a route still fails closed.  CPU only.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx2 import serving as serving_module
from mlx2.serving import ServingEngine

from test_apc_hits_hybrid_gdn_self_mtp import tiny_qwen38_mtp
from test_rolling_prefill_checkpoints import (  # noqa: F401 - fixture
    _adapter,
    _cancel_mid_prefill,
    _roles,
    _run,
    host,
)

LONG = 16384
PREFILL_STEP = 512


def _long_adapter(model, vocab):
    base = _adapter(model, vocab)

    class Adapter(base):
        max_context = LONG

    return Adapter


def _kv_only_model(*, mtp=False):
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel

    args = TextModelArgs(
        model_type="qwen3_5", hidden_size=64, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, vocab_size=128, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=3, full_attention_interval=1,
        partial_rotary_factor=0.5, rope_parameters=None,
        max_position_embeddings=LONG,
        **({"mtp_num_hidden_layers": 1} if mtp else {}),
    )
    mx.random.seed(5)
    model = TextModel(args)
    model.eval()
    mx.eval(model.parameters())
    return model, 128


def _default_engine(model, vocab, **kw):
    engine = ServingEngine(
        "tiny", adapter_factory=_long_adapter(model, vocab),
        qualification_mode=True, max_lanes=1, prefill_step=PREFILL_STEP, **kw,
    )
    assert engine.ready.wait(60), engine.error
    return engine


def test_default_interval_is_splash_4096():
    default = getattr(serving_module, "APC_ROLLING_CHECKPOINT_DEFAULT", None)
    assert default == {"interval_tokens": 4096}


@pytest.mark.parametrize("mtp", [False, True], ids=["ordinary", "self_mtp"])
def test_default_engine_selects_rolling_checkpoints_on_hybrid(host, mtp):  # noqa: F811
    model, vocab = tiny_qwen38_mtp()
    engine = _default_engine(model, vocab, mtp=mtp)
    try:
        assert engine.apc_rolling_route == "hybrid"
        settings = engine.snapshot["settings"]
        assert settings["apc_rolling_checkpoints"] == {"interval_tokens": 4096}
        assert settings["apc_rolling_checkpoints_default"] is True
        assert "apc_rolling_checkpoints" not in settings.get(
            "skipped_route_defaults", {}
        )
    finally:
        engine.close()


def test_default_cancelled_hybrid_prefill_keeps_its_progress(host):  # noqa: F811
    model, vocab = tiny_qwen38_mtp()
    prompt = [(7 * i + 3) % (vocab - 2) + 1 for i in range(4096 + 1024)]
    engine = _default_engine(model, vocab, mtp=False)
    try:
        _cancel_mid_prefill(
            engine, prompt,
            lambda engine, _calls: (
                engine.counts["apc_rolling_checkpoints_published"] >= 1
            ),
        )
        assert (4096, "prefill_rolling") in _roles(engine)
        warm, receipt, job = _run(engine, prompt)
        assert job.cached_tokens == 4096
        assert receipt["cache_checkpoint_role"] == "prefill_rolling"
        # The committed boundary retires the last progress point.
        assert not [r for r in _roles(engine) if r[1] == "prefill_rolling"]
    finally:
        engine.close()
    cold = _default_engine(
        model, vocab, mtp=False,
        execution_policy={"apc_rolling_checkpoints": {"interval_tokens": 0}},
    )
    try:
        assert cold.apc_rolling_route is None
        assert "apc_rolling_checkpoints" not in cold.snapshot["settings"]
        assert _run(cold, prompt)[0] == warm
    finally:
        cold.close()


def test_default_cancelled_kv_prefill_keeps_its_progress(host):  # noqa: F811
    model, vocab = _kv_only_model()
    prompt = [(11 * i + 5) % 126 + 1 for i in range(4096 + 2048)]
    engine = _default_engine(model, vocab, mtp=False)
    try:
        assert engine.apc_rolling_route == "kv"
        # Hold the eleventh step: ten 512-token chunks (5120 tokens) are done.
        _cancel_mid_prefill(engine, prompt, lambda _engine, calls: calls == 11)
        assert engine.counts["apc_rolling_checkpoints_cancel_published"] == 1
        _warm, receipt, job = _run(engine, prompt)
        assert job.cached_tokens >= 4096
        assert receipt["cache_checkpoint_role"] == "prefill_rolling"
    finally:
        engine.close()


def test_default_steps_aside_on_an_unsupported_route(host):  # noqa: F811
    # Self-MTP on a KV-only target: neither the hybrid capture nor the
    # ordinary-route cancel publication applies.
    model, vocab = _kv_only_model(mtp=True)
    engine = _default_engine(model, vocab, mtp=True)
    try:
        assert engine.apc_rolling_route is None
        settings = engine.snapshot["settings"]
        assert "apc_rolling_checkpoints" not in settings
        reason = settings["skipped_route_defaults"]["apc_rolling_checkpoints"]
        assert "adapter cache route" in reason
    finally:
        engine.close()
    explicit = ServingEngine(
        "tiny", adapter_factory=_long_adapter(model, vocab),
        qualification_mode=True, max_lanes=1, prefill_step=PREFILL_STEP, mtp=True,
        execution_policy={"apc_rolling_checkpoints": {"interval_tokens": 4096}},
    )
    try:
        explicit.thread.join(30)
        assert not explicit.ready.is_set()
        assert "APCv2 rolling checkpoints" in explicit.error
    finally:
        explicit.close()


def test_explicit_null_turns_the_default_off(host):  # noqa: F811
    model, vocab = tiny_qwen38_mtp()
    engine = _default_engine(
        model, vocab, mtp=False,
        execution_policy={"apc_rolling_checkpoints": None},
    )
    try:
        assert engine.apc_rolling_route is None
        assert "apc_rolling_checkpoints" not in engine.snapshot["settings"]
    finally:
        engine.close()


def test_default_selection_requires_observed_engagement():
    """A default option must be shown working (AGENTS.md): the default
    selection is a required feature, never selected-not-observed, so a run
    without cancel-and-retry evidence cannot qualify the route."""
    from mlx2.qualification import (
        required_feature_checks as required_features,
        selected_not_observed_features,
    )

    settings = {
        "apc_rolling_checkpoints": {"interval_tokens": 4096},
        "apc_rolling_checkpoints_default": True,
    }
    assert "feature_apc_rolling_checkpoints" in required_features(settings)
    assert "feature_apc_rolling_checkpoints" not in selected_not_observed_features(
        settings
    )
    explicit = {"apc_rolling_checkpoints": {"interval_tokens": 4096}}
    assert "feature_apc_rolling_checkpoints" in required_features(explicit)


def test_default_lane_holds_one_rolling_entry_at_a_time(host):  # noqa: F811
    """Retire-after-successor bounds the default: a lane that published three
    progress points holds only the newest, and its APCv2 bytes stay inside
    the resident budget."""
    model, vocab = tiny_qwen38_mtp()
    prompt = [(5 * i + 2) % (vocab - 2) + 1 for i in range(3 * 4096 + 512)]
    engine = _default_engine(model, vocab, mtp=False)
    try:
        _cancel_mid_prefill(
            engine, prompt,
            lambda engine, _calls: (
                engine.counts["apc_rolling_checkpoints_published"] >= 3
            ),
        )
        rolling = [r for r in _roles(engine) if r[1] == "prefill_rolling"]
        assert rolling == [(12288, "prefill_rolling")]
        assert engine.counts["apc_rolling_checkpoints_retired"] == 2
        assert engine.apc.nbytes <= engine.apc.max_bytes
    finally:
        engine.close()


def test_default_interval_is_a_multiple_of_the_prefill_step():
    from mlx2.serving import default_rolling_interval

    assert default_rolling_interval(512) == 4096
    assert default_rolling_interval(2048) == 4096
    assert default_rolling_interval(4096) == 4096
    assert default_rolling_interval(8192) == 8192   # Flash-Next, autoscale top
    assert default_rolling_interval(3000) == 6000
    assert default_rolling_interval(None) == 4096


def test_default_engine_with_a_large_prefill_step_spaces_by_the_step(host):  # noqa: F811
    model, vocab = tiny_qwen38_mtp()
    # The context must hold a prompt past the 8192 boundary by another 8192
    # slice, or the default steps aside (see the short-context test below).
    engine = ServingEngine(
        "tiny", adapter_factory=_context_adapter(model, vocab, 32768),
        qualification_mode=True, max_lanes=1, prefill_step=8192, mtp=False,
        max_context=32768,
    )
    assert engine.ready.wait(60), engine.error
    try:
        assert engine.snapshot["settings"]["apc_rolling_checkpoints"] == {
            "interval_tokens": 8192
        }
    finally:
        engine.close()


def _rolling(engine):
    return [r for r in _roles(engine) if r[1] == "prefill_rolling"]


def test_repeated_kv_cancellation_keeps_one_rolling_entry(host, tmp_path):  # noqa: F811
    """cancel -> retry hits C1 -> cancel again publishes C2: C1 must be
    retired (codex review 2026-10-14), so cancel/hit/cancel cycles never
    accumulate disposable entries, resident or spilled."""
    from test_rolling_prefill_checkpoints import _engine

    model, vocab = _kv_only_model()
    prompt = [(11 * i + 5) % 126 + 1 for i in range(200)]
    engine = _engine(model, vocab, mtp=False, interval=32, cache_dir=str(tmp_path))
    try:
        assert engine.apc_rolling_route == "kv"
        _cancel_mid_prefill(engine, prompt, lambda _engine, calls: calls == 6)
        first = _rolling(engine)
        assert len(first) == 1
        # The retry resumes at C1 and is cancelled again three chunks later.
        _cancel_mid_prefill(engine, prompt, lambda _engine, calls: calls == 4)
        assert engine.counts["apc_rolling_checkpoints_cancel_published"] == 2
        second = _rolling(engine)
        assert len(second) == 1 and second[0][0] > first[0][0], second
        _cancel_mid_prefill(engine, prompt, lambda _engine, calls: calls == 3)
        assert len(_rolling(engine)) == 1
        apc = engine.apc
        stats = apc.apc_stats
        assert stats["cow"]["active_leases"] == 0
        with apc._apc_lock:
            on_disk = [
                len(tokens)
                for _key, tokens, entry in apc._entry_records_locked()
                if getattr(entry, "_apc_disk", None)
                and entry._apc_retention_role == "prefill_rolling"
            ]
            resident = sum(
                int(entry.nbytes)
                for _key, _tokens, entry in apc._entry_records_locked()
            )
        assert len(on_disk) <= 1
        assert apc.nbytes == resident
        _warm, _receipt, job = _run(engine, prompt)
        assert job.cached_tokens >= second[0][0]
        assert not _rolling(engine)
    finally:
        engine.close()


def test_default_steps_aside_without_a_cache_projection(host):  # noqa: F811
    """Admission budgets hybrid captures against the adapter's cache
    projection; with none it would plan no rolling boundary, so the default
    must not be reported selected."""
    from test_apc_hits_hybrid_gdn_self_mtp import make_adapter

    model, vocab = tiny_qwen38_mtp()
    engine = ServingEngine(
        "tiny", adapter_factory=make_adapter(model, vocab),
        qualification_mode=True, max_lanes=1, prefill_step=16, mtp=False,
    )
    assert engine.ready.wait(60), engine.error
    try:
        assert engine.apc_rolling_route is None
        settings = engine.snapshot["settings"]
        assert "apc_rolling_checkpoints" not in settings
        assert "cache projection" in (
            settings["skipped_route_defaults"]["apc_rolling_checkpoints"]
        )
    finally:
        engine.close()


def _context_adapter(model, vocab, context):
    base = _adapter(model, vocab)

    class Adapter(base):
        max_context = context

    return Adapter


def test_default_steps_aside_when_the_context_cannot_show_reuse(host):  # noqa: F811
    """max_context 8192 at prefill step 8192 selects an 8192 interval, but no
    prompt can pass that boundary by another slice: the default would be
    selected yet never publish and reuse a checkpoint, and qualification
    could never observe it (codex re-review 2026-10-14).  An explicit
    request keeps today's behaviour."""
    from mlx2.runtime.state_boundaries import rolling_reuse_prompt_tokens

    assert rolling_reuse_prompt_tokens(
        interval_tokens=8192, prefill_step=8192, max_context=8192
    ) is None
    model, vocab = tiny_qwen38_mtp()
    engine = ServingEngine(
        "tiny", adapter_factory=_context_adapter(model, vocab, 8192),
        qualification_mode=True, max_lanes=1, prefill_step=8192, mtp=False,
    )
    assert engine.ready.wait(60), engine.error
    try:
        assert engine.apc_rolling_route is None
        settings = engine.snapshot["settings"]
        assert "apc_rolling_checkpoints" not in settings
        reason = settings["skipped_route_defaults"]["apc_rolling_checkpoints"]
        assert "max_context 8192" in reason
    finally:
        engine.close()
    explicit = ServingEngine(
        "tiny", adapter_factory=_context_adapter(model, vocab, 8192),
        qualification_mode=True, max_lanes=1, prefill_step=8192, mtp=False,
        execution_policy={"apc_rolling_checkpoints": {"interval_tokens": 8192}},
    )
    assert explicit.ready.wait(60), explicit.error
    try:
        assert explicit.apc_rolling_route == "hybrid"
        assert explicit.snapshot["settings"]["apc_rolling_checkpoints"] == {
            "interval_tokens": 8192
        }
    finally:
        explicit.close()


def test_default_selection_and_probe_share_one_reach_rule():
    """Serving's default and the qualifier's probe size from the same helper,
    so a selected default always admits the probe."""
    import importlib.util
    from pathlib import Path

    from mlx2.runtime.state_boundaries import (
        ROLLING_REUSE_CONTEXT_RESERVE,
        rolling_reuse_prompt_tokens,
    )

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "qualify_serving_reach", root / "scripts" / "qualify_serving.py"
    )
    qualify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qualify)
    assert ROLLING_REUSE_CONTEXT_RESERVE == (
        qualify.LONG_CONTEXT_HEADROOM + qualify.ROLLING_PROBE_COMPLETION_TOKENS
    )
    for interval, step, context, source in (
        (4096, 512, 16384, "engine_argument"),
        (8192, 8192, 8192, "adapter"),
        (8192, 8192, 16384, "adapter"),
        (8192, 8192, 16384, "prompt_length_autoscale"),
        (4096, 2048, 4500, "adapter"),
    ):
        settings = {
            "apc_rolling_checkpoints": {"interval_tokens": interval},
            "prefill_step": step,
            "prefill_step_source": source,
            "max_context": context,
        }
        assert qualify.rolling_probe_prompt_words(settings) == (
            rolling_reuse_prompt_tokens(
                interval_tokens=interval, prefill_step=step, max_context=context,
                autoscale=source == "prompt_length_autoscale",
            )
        ), settings


def test_autoscaled_probe_keeps_three_execution_slices_across_a_rung():
    """Execution picks the autoscaled step from the whole prompt
    (BatchGenerator._prompt_prefill_step); sizing the probe from the step at
    ``interval + 1`` straddled the 32K rung: 32700 + 3 x 512 = 34236 tokens
    prefill at 2048, leaving one partial slice past the boundary (codex
    re-review 2026-10-14).  The probe must leave three slices of the step
    execution will use for the final prompt, template overhead included."""
    import importlib.util
    from pathlib import Path

    from mlx2.runtime.prefill_plan import prompt_length_prefill_step
    from mlx2.runtime.state_boundaries import (
        ROLLING_REUSE_PROMPT_OVERHEAD,
        ROLLING_REUSE_TAIL_SLICES,
        rolling_reuse_prompt_tokens,
    )

    def execution_tail_slices(length, interval):
        # Worst case over the prompt's possible template overhead.
        step = max(
            prompt_length_prefill_step(length + extra, maximum=8192)
            for extra in (0, ROLLING_REUSE_PROMPT_OVERHEAD)
        )
        return (length - interval) // step

    for interval, context in ((32700, 65536), (65000, 131072), (8192, 16384)):
        length = rolling_reuse_prompt_tokens(
            interval_tokens=interval, prefill_step=8192, max_context=context,
            autoscale=True,
        )
        assert length is not None, interval
        assert execution_tail_slices(length, interval) >= ROLLING_REUSE_TAIL_SLICES, (
            interval, length,
        )
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "qualify_serving_rung", root / "scripts" / "qualify_serving.py"
    )
    qualify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qualify)
    words = qualify.rolling_probe_prompt_words({
        "apc_rolling_checkpoints": {"interval_tokens": 32700},
        "prefill_step": 8192,
        "prefill_step_source": "prompt_length_autoscale",
        "max_context": 65536,
    })
    assert words == 32700 + 3 * 2048
