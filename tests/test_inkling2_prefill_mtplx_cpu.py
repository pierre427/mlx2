"""MTPLX ingest, prefill items (2026-11-23), CPU only (no Metal, no model).

* inkling2_prefill_handover: bit-identical, counted.
* inkling2_prefill_transient_pricing: off charges nothing; on is a shape
  bound that grows with rows and keys.
* inkling2_prefill_wide_chunk: the width schedule (a function of a chunk's
  absolute depth, fixed at load: up to 2,048 rows below 18,432 keys, 512
  past it), its load-time decline, the generic ``_mlx2_prefill_rows_at``
  hook as the real planner and execution read it, the APCv2 namespace term,
  cold vs restored reads, admission pricing and the executed-chunk counter.
* Every option defaults off.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from test_inkling2_cpu import TINY

GIB = 1 << 30


@pytest.fixture(autouse=True)
def _cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def _tiny_model(seed=0, config=None):
    from mlx2.runtime.models.inkling2 import Model, TextConfig

    mx.random.seed(seed)
    model = Model(TextConfig.from_dict(config or TINY))
    weights = []
    for key, value in tree_flatten(model.parameters()):
        noise = mx.random.normal(value.shape) * (0.05 if value.ndim > 1 else 0.3)
        weights.append((key, noise + (1.0 if key.endswith("norm.weight") else 0.0)))
    model.load_weights(weights, strict=True)
    mx.eval(model.parameters())
    return model


def _run(model, tokens, plan, pads=None):
    from mlx2.runtime.models.cache import BatchKVCache, BatchRotatingKVCache, KVCache

    cache = model.make_cache()
    if pads is not None:
        window = model.config.sliding_window_size
        for i in range(0, len(cache), 2):
            cache[i] = (BatchKVCache(list(pads)) if type(cache[i]) is KVCache
                        else BatchRotatingKVCache(window, list(pads)))
    outs = []
    for a, b in plan:
        outs.append(model(tokens[:, a:b], cache=cache))
        mx.eval(outs[-1])
    return mx.concatenate(outs, axis=1)


# ----- defaults -----


def test_every_option_defaults_off():
    from mlx2.adapters.inkling2 import Inkling2Adapter, validate_policy

    kernels, rest = validate_policy(None)
    for name in ("prefill_handover", "prefill_wide_chunk", "prefill_transient_pricing"):
        assert kernels[name] is False, name
    assert rest == {}
    adapter = Inkling2Adapter.__new__(Inkling2Adapter)
    assert adapter.prefill_step_default() == 512
    assert adapter.prefill_step_reference() == 512
    model = _tiny_model()
    policy = model.kernel_policy if hasattr(model, "kernel_policy") else {}
    assert not policy.get("prefill_handover")
    model.configure_kernels()
    assert model.kernel_policy["prefill_handover"] is False
    assert "_mlx2_prefill_rows_at" not in model.__dict__
    assert model.model.__dict__.get("_prefill_wide_rows") is None


def test_policy_validation():
    from mlx2.adapters.inkling2 import validate_policy

    kernels, _ = validate_policy({"inkling2_prefill_wide_chunk": True})
    assert kernels["prefill_wide_chunk"] and kernels["prefill_transient_pricing"]
    kernels, _ = validate_policy({"inkling2_prefill_transient_pricing": True})
    assert kernels["prefill_transient_pricing"] and not kernels["prefill_wide_chunk"]
    with pytest.raises(ValueError):
        validate_policy({"inkling2_prefill_wide_chunk": True,
                         "inkling2_prefill_transient_pricing": False})
    for key in ("inkling2_prefill_handover",
                "inkling2_prefill_wide_chunk", "inkling2_prefill_transient_pricing"):
        with pytest.raises(TypeError):
            validate_policy({key: 1})


def test_wide_chunk_refuses_the_dspark_route_before_loading():
    from mlx2.adapters.inkling2 import Inkling2Adapter

    with pytest.raises(ValueError, match="ordinary-route only"):
        Inkling2Adapter("/nonexistent/inkling2", execution_policy={
            "draft_model": "embedded_dspark", "inkling2_prefill_wide_chunk": True})


# ----- handover -----


def test_handover_is_bit_identical_and_counted():
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 140), key=mx.random.key(2))
    plan = [(0, 70), (70, 139), (139, 140)]
    base = _run(model, tokens, plan)
    model.configure_kernels(prefill_handover=True)
    got = _run(model, tokens, plan)
    assert mx.array_equal(got, base)
    visits = len(model.model.layer_indices)
    # Two prompt chunks of >= 64 rows hand over every 4 visits (not after
    # the last); the one-row step does not.
    expected = 2 * ((visits - 1) // 4)
    assert model.kernel_receipt()["prefill_handover_calls"] == expected


# ----- transient pricing -----


def _real_text():
    from mlx2.runtime.models.inkling2 import TextConfig

    return TextConfig()  # the Mini's geometry (config defaults)


def test_transient_pricing_off_charges_nothing():
    from mlx2.adapters.inkling2_memory import Inkling2CacheBudget
    from mlx2.serving import prefill_transient_gib

    budget = Inkling2CacheBudget.from_config(_real_text(), prefill_rows=512)
    assert budget.prefill_transient_bytes(4096, 512) == 0
    assert prefill_transient_gib(budget, context_tokens=4096, uncached_tokens=4096,
                                 prefill_step=512) == 0.0


def test_transient_pricing_is_a_shape_bound():
    from mlx2.adapters.inkling2_memory import Inkling2CacheBudget
    from mlx2.serving import prefill_transient_gib

    text = _real_text()
    on = Inkling2CacheBudget.from_config(text, prefill_rows=2048, transient_pricing=True)
    maskless = Inkling2CacheBudget.from_config(text, prefill_rows=2048, transient_pricing=True,
                                               mask_planes=False)
    at = on.prefill_transient_bytes
    assert 0.2 * GIB < at(4096, 512) < 1.0 * GIB  # MTPLX measured 0.625 GiB at 512 rows
    assert at(4096, 2048) > at(4096, 512) and at(65536, 512) > at(4096, 512)
    assert maskless.prefill_transient_bytes(65536, 2048) < at(65536, 2048)
    terms = on.transient_terms(65536, 2048)
    assert set(terms) == {"residual", "attention", "mask", "moe", "growth"}
    assert maskless.transient_terms(65536, 2048)["mask"] == 0
    # A larger mask budget prices larger slices.
    wide_mask = Inkling2CacheBudget.from_config(text, prefill_rows=512, transient_pricing=True,
                                                mask_budget_mb=512)
    assert wide_mask.prefill_transient_bytes(65536, 512) > at(65536, 512)
    gib = prefill_transient_gib(on, context_tokens=8192, uncached_tokens=8192,
                                prefill_step=2048)
    assert gib == pytest.approx(at(8192, 2048) / GIB)
    # Pricing does not touch the lane projection.
    off = Inkling2CacheBudget.from_config(text, prefill_rows=2048)
    assert off.project(65536) == on.project(65536)


# ----- the wide-chunk schedule -----


def test_schedule_is_a_function_of_depth():
    from mlx2.adapters.inkling2_prefill_plan import CROSSOVER, rows_at

    assert CROSSOVER == 18_432
    assert [rows_at(2048, d) for d in (0, 2048, 16384, 17000, 18431, 18432, 50000)] == [
        2048, 2048, 2048, 1432, 1, 512, 512]
    # A configured maximum below 2,048 is never exceeded.
    assert [rows_at(1024, d) for d in (0, 18000, 18432)] == [1024, 432, 512]
    assert rows_at(256, 30000) == 256


def _bare_adapter(policy=None, headroom=40 * GIB):
    from mlx2.adapters.inkling2 import Inkling2Adapter, validate_policy

    adapter = Inkling2Adapter.__new__(Inkling2Adapter)
    kernels, _ = validate_policy(policy or {})
    adapter.prefill_wide_chunk = kernels.pop("prefill_wide_chunk")
    adapter.prefill_transient_pricing = kernels.pop("prefill_transient_pricing")
    adapter.kernel_policy = kernels
    adapter.model = _tiny_model()
    adapter.prefill_schedule = None
    if adapter.prefill_wide_chunk:
        adapter._install_prefill_schedule(headroom_bytes=headroom)
    return adapter


WIDE = {"inkling2_prefill_wide_chunk": True}


def test_route_declines_at_load_below_the_headroom_floor():
    from mlx2.runtime.generate import _declared_model_callback
    from mlx2.serving import prefill_step_namespace_term

    for headroom, reason in ((8 * GIB, "headroom"), (None, "no_gpu")):
        adapter = _bare_adapter(WIDE, headroom=headroom)
        assert adapter.prefill_schedule.receipt()["decline"] == reason
        assert adapter.prefill_step_default() == 512
        assert _declared_model_callback(adapter.model, "_mlx2_prefill_rows_at") is None
        assert adapter.prefill_schedule_identity(512) is None
        assert prefill_step_namespace_term(adapter, 512, autoscale=False) is None


def test_namespace_terms_are_distinct_for_the_three_configurations():
    from mlx2.serving import prefill_step_namespace_term

    off, wide = _bare_adapter(), _bare_adapter(WIDE)
    assert off.prefill_step_default() == 512 and wide.prefill_step_default() == 2048
    terms = [prefill_step_namespace_term(off, 512, autoscale=False),
             prefill_step_namespace_term(off, 2048, autoscale=False),  # --prefill-step 2048
             prefill_step_namespace_term(wide, 2048, autoscale=False)]
    assert terms[0] is None
    assert terms[1] == {"schema": "mlx2.prefill-step.v1", "prefill_step": 2048}
    assert terms[2]["schedule"] == {"schema": "mlx2.inkling2-prefill-schedule.v1",
                                    "max_rows": 2048, "narrow_rows": 512, "crossover": 18432}
    assert len({repr(t) for t in terms}) == 3


class _Generator:
    """The BatchGenerator planning and execution methods on a minimal host."""

    PREFILL_DEPTH_FLOOR = 128

    def __init__(self, model, step):
        from mlx2.runtime.generate import BatchGenerator

        for name in ("_model_rows_at", "_depth_bounded_step", "_planned_prefill_rows",
                     "_prompt_prefill_step", "_planned_prompt_prefill_step"):
            setattr(self, name, getattr(BatchGenerator, name).__get__(self))
        self.model, self.prefill_step_size = model, step
        self.prefill_step_autoscale = False
        self.prefill_depth_budget = None
        self.scheduler_stats = {}


def _chunks(gen, start, end, cuts=()):
    """One lane's slices from ``start`` to ``end`` as the ordinary round runs
    them (``_round_impl``: round slice, ``_prompt_prefill_step``,
    ``_depth_bounded_step``, the checkpoint cut), each asserted equal to the
    idle plan's slice (``_planned_prefill_rows``)."""
    from mlx2.runtime.prefill_plan import checkpoint_bounded_prefill_rows, next_prefill_checkpoint

    out, covered = [], start
    while covered < end:
        cut = next_prefill_checkpoint(cuts, covered)
        round_slice = gen.prefill_step_size  # one lane: the shared width is the step
        step = gen._depth_bounded_step(min(round_slice, gen._prompt_prefill_step(end)), covered)
        rows = checkpoint_bounded_prefill_rows(min(end - covered, step), covered, cut)
        planned = gen._planned_prefill_rows(gen._planned_prompt_prefill_step(end), covered,
                                            end - covered, cut)
        assert rows == planned, (covered, rows, planned)
        out.append((covered, covered + rows))
        covered += rows
    return out


def test_follow_up_turn_across_the_cap_with_a_junction_cut():
    """Turn A (10,000 tokens) runs at 2,048; turn B extends it past 18,432.
    A cold read of B equals a warm B resumed from A's end only when a
    junction checkpoint cuts the cold read at A's end: the same guarantee as
    the 512-row route. With the cut, cold B runs A's chunks and then exactly
    the warm read's chunks; without it the chunk at A's end differs."""
    wide = _bare_adapter(WIDE)
    gen = _Generator(wide.model, wide.prefill_step_default())
    turn_a = _chunks(gen, 0, 10_000)
    cold_b = _chunks(gen, 0, 30_000, cuts=(10_000,))
    warm_b = _chunks(gen, 10_000, 30_000)
    assert cold_b[:len(turn_a)] == turn_a
    assert cold_b[len(turn_a):] == warm_b
    assert {b - a for a, b in turn_a[:-1]} == {2048}
    assert all(b <= 18432 or a >= 18432 for a, b in cold_b)  # no wide chunk crosses the cap
    assert {b - a for a, b in warm_b[:-1] if a >= 18432} == {512}  # last: the prompt end
    uncut = _chunks(gen, 0, 30_000)
    assert uncut[:len(turn_a)] != turn_a
    # Without the switch the schedule is the plain step.
    off = _bare_adapter()
    assert {b - a for a, b in _chunks(_Generator(off.model, 512), 0, 4096)} == {512}


def _run_on(model, tokens, plan, cache):
    outs = []
    for a, b in plan:
        outs.append(model(tokens[:, a:b], cache=cache))
        mx.eval(outs[-1])
    return mx.concatenate(outs, axis=1)


@pytest.mark.parametrize("restore_at", [30, 48], ids=["junction", "interior_past_cap"])
def test_restored_read_is_bitwise_the_cold_read(tmp_path, restore_at):
    """A cold read of a 60-token prompt, cut at ``restore_at``, is snapshotted
    there (the serialized prompt cache, as an APCv2 entry persists it); a
    warm read restores the snapshot into a new cache and continues. Every row
    after the restore point is bitwise the cold read's. Scaled-down schedule:
    16 rows below 40 keys, 4 past it."""
    from mlx2.adapters.inkling2_prefill_plan import WidePrefillSchedule
    from mlx2.runtime.models.cache import load_prompt_cache, save_prompt_cache

    model = _tiny_model()
    schedule = WidePrefillSchedule(headroom_bytes=40 * GIB, crossover=40, narrow=4, wide=16)
    object.__setattr__(model, "_mlx2_prefill_rows_at", schedule.rows_at)
    gen = _Generator(model, 16)
    tokens = mx.random.randint(0, 500, (1, 61), key=mx.random.key(9))
    cache, outs = model.make_cache(), []
    for a, b in _chunks(gen, 0, 60, cuts=(restore_at,)) + [(60, 61)]:
        if a == restore_at:
            save_prompt_cache(str(tmp_path / "entry.safetensors"), cache)
        outs.append(model(tokens[:, a:b], cache=cache))
        mx.eval(outs[-1])
    cold = mx.concatenate(outs, axis=1)[:, restore_at:]
    restored = load_prompt_cache(str(tmp_path / "entry.safetensors"))
    warm = _run_on(model, tokens, _chunks(gen, restore_at, 60) + [(60, 61)], restored)
    assert mx.array_equal(cold, warm)


def test_wide_chunk_counter_follows_the_resolved_step_and_counts_lanes():
    from mlx2.runtime.models.inkling2 import prefill_route

    wide = _bare_adapter(WIDE)
    inner = wide.model.model
    assert inner.__dict__["_prefill_wide_rows"] == 2048
    wide.bind_prefill_step(1024)  # --prefill-step 1024
    assert inner.__dict__["_prefill_wide_rows"] == 1024
    for shape in ((2, 1024, 1), (2, 700, 1), (1, 2048, 1), (1, 1024, 1)):
        prefill_route.handover_every(inner, mx.zeros(shape))
    # Two lanes at the step, then one; the near-cap 700-row slice and the
    # 2,048 rows of another step are not this step's chunks.
    assert wide.model.kernel_receipt()["prefill_wide_chunks"] == 3
    wide.bind_prefill_step(512)
    assert inner.__dict__["_prefill_wide_rows"] is None
    off = _bare_adapter()
    off.bind_prefill_step(2048)
    assert off.model.model.__dict__.get("_prefill_wide_rows") is None


def test_admission_charges_the_largest_chunk_the_schedule_runs():
    from mlx2.serving import prefill_transient_gib

    wide = _bare_adapter(WIDE)
    wide.bind_prefill_step(2048)
    budget = wide.cache_budget(mtp=False)
    assert not hasattr(budget, "recommended_prefill_rows")
    assert (budget.schedule_crossover, budget.schedule_narrow_rows) == (18432, 512)

    def cost(keys, rows):
        return sum(budget.transient_terms(keys, rows).values())

    deep = prefill_transient_gib(budget, context_tokens=131072, uncached_tokens=131072,
                                 prefill_step=2048)
    expected = max(cost(18432, 2048), cost(131072, 512)) / GIB
    assert deep == pytest.approx(expected)
    assert deep < cost(131072, 2048) / GIB
    # On the Mini's geometry: a 128K request is charged about 1.7 GiB (the
    # 512-row chunk at full depth), not the 3.4 GiB of a 2,048-row chunk there.
    from mlx2.adapters.inkling2_memory import Inkling2CacheBudget

    mini = Inkling2CacheBudget.from_config(_real_text(), prefill_rows=2048,
                                           transient_pricing=True, schedule=(18432, 512))
    charged = mini.prefill_transient_bytes(131072, 2048) / GIB
    assert 1.5 < charged < 1.8, charged
    assert sum(mini.transient_terms(131072, 2048).values()) / GIB > 3.0
    shallow = prefill_transient_gib(budget, context_tokens=8192, uncached_tokens=8192,
                                    prefill_step=2048)
    assert shallow == pytest.approx(cost(8192, 2048) / GIB)
    # A configured maximum below 2,048 prices that maximum.
    wide.bind_prefill_step(1024)
    narrow = wide.cache_budget(mtp=False)
    priced = prefill_transient_gib(narrow, context_tokens=8192, uncached_tokens=8192,
                                   prefill_step=1024)
    assert priced == pytest.approx(sum(narrow.transient_terms(8192, 1024).values()) / GIB)
    # A fixed step (switch off) prices its own chunk at the full depth.
    off = _bare_adapter({"inkling2_prefill_transient_pricing": True})
    off.bind_prefill_step(512)
    fixed = off.cache_budget(mtp=False)
    assert fixed.schedule_crossover == 0
    assert fixed.prefill_transient_bytes(131072, 512) == sum(
        fixed.transient_terms(131072, 512).values())
