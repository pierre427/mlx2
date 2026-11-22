"""Flint's opt-in prefill block (MLX2_FLINT_PREFILL_STEP) and its APCv2 geometry.

The schedule helper below reproduces the scheduler's cold slice law from its
pure parts: a slice is the step, clamped to the next planned state boundary
(``checkpoint_bounded_prefill_rows``), over the prompt less its final token
(the generation boundary).  Rotating-window restore snapshots follow
``_window_checkpoint_due`` at slice ends.
"""
import pytest

from mlx2.adapters import flint
from mlx2.adapters.flint import FlintAdapter
from mlx2.runtime.generate import interior_checkpoint_positions
from mlx2.runtime.models.cache import _window_checkpoint_due
from mlx2.runtime.prefill_plan import (
    checkpoint_bounded_prefill_rows,
    next_prefill_checkpoint,
)
from mlx2.runtime.state_boundaries import BoundaryPurpose, plan_state_boundaries
from mlx2.serving import default_rolling_interval

WINDOW = 513


def adapter():
    result = FlintAdapter.__new__(FlintAdapter)
    result.config = {
        "layer_types": ["full_attention"] * 14 + ["sliding_attention"] * 35,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "sliding_window": WINDOW,
        "torch_dtype": "bfloat16",
    }
    return result


def boundaries(prompt, step, cached=0):
    # The served hybrid plan: pow2 interior lattice (auto policy count 4,
    # min stride 256) plus the step-derived default rolling interval.
    return plan_state_boundaries(
        prompt_tokens=prompt,
        cached_tokens=cached,
        interior=interior_checkpoint_positions(prompt, count=4, min_stride=256),
        rolling_interval=default_rolling_interval(step),
    )


def schedule(prompt, step, planned, cached=0):
    positions = [bound.position for bound in planned]
    covered, end, slices = cached, prompt - 1, []
    while covered < end:
        rows = checkpoint_bounded_prefill_rows(
            min(step, end - covered), covered, next_prefill_checkpoint(positions, covered)
        )
        slices.append((covered, rows))
        covered += rows
    return slices


def window_snapshots(slices):
    lane = []
    for start, rows in slices:
        if _window_checkpoint_due(lane, start + rows):
            lane.append((start + rows,))
    return [entry[0] for entry in lane]


@pytest.fixture(autouse=True)
def _stride(monkeypatch):
    monkeypatch.setenv("MLX_LM_STATE_CHECKPOINT_MAX", "4")
    monkeypatch.setenv("MLX_LM_STATE_CHECKPOINT_STRIDE", "2048")
    monkeypatch.delenv(flint.PREFILL_STEP_ENV, raising=False)


def test_default_stays_512_and_env_is_honored(monkeypatch):
    assert flint.prefill_step_choice({}) == 512
    assert adapter().prefill_step_default() == 512
    monkeypatch.setenv(flint.PREFILL_STEP_ENV, "2048")
    assert adapter().prefill_step_default() == 2048
    assert adapter().cache_budget(mtp=False).prefill_step == 2048
    assert flint.prefill_step_choice({flint.PREFILL_STEP_ENV: " 2052 "}) == 2052


@pytest.mark.parametrize("raw", ["0", "-512", "2k", "1.5", "16385"])
def test_invalid_env_fails_closed(raw):
    with pytest.raises(ValueError, match=flint.PREFILL_STEP_ENV):
        flint.prefill_step_choice({flint.PREFILL_STEP_ENV: raw})


def test_cache_budget_follows_the_executed_step():
    model = adapter()
    assert model.cache_budget(mtp=False).sliding_token_cap == WINDOW + 512
    config = model.execution_config(max_lanes=1, prefill_step=2048)
    assert config["prefill_step_size"] == 2048
    budget = model.cache_budget(mtp=False)
    assert budget.sliding_token_cap == WINDOW + 2048
    # The wider chunk is charged in the live bound, not as a transient.
    assert budget.prefill_transient_bytes(131072, 2048) == 0
    assert budget.project(131072) > adapter().cache_budget(mtp=False).project(131072)


@pytest.mark.parametrize(
    "prompt,slices_512,slices_2048", [(32768, 64, 16), (131072, 256, 64)]
)
def test_cold_schedule_and_apc_boundaries(prompt, slices_512, slices_2048):
    plans = {step: boundaries(prompt, step) for step in (512, 2048)}
    # Boundary placement does not depend on the step between 512 and 2048.
    assert plans[512] == plans[2048]
    assert default_rolling_interval(512) == default_rolling_interval(2048) == 4096
    planned = {bound.position for bound in plans[2048]}
    # Rolling multiples of 4096 plus the pow2 interior lattice (a shared
    # position keeps the higher-priority purpose).
    assert planned == set(range(4096, prompt - 1, 4096)) | set(
        interior_checkpoint_positions(prompt, count=4, min_stride=256)
    )
    assert any(b.purpose == BoundaryPurpose.ROLLING for b in plans[2048])
    for step, expected in ((512, slices_512), (2048, slices_2048)):
        slices = schedule(prompt, step, plans[step])
        assert len(slices) == expected
        assert [rows for _, rows in slices] == [step] * (expected - 1) + [step - 1]
        ends = {start + rows for start, rows in slices}
        # Every APC boundary is a slice end: a capture lands exactly.
        assert planned <= ends
        assert sum(rows for _, rows in slices) == prompt - 1
    # Rotating-window restore snapshots land on the same positions.
    assert window_snapshots(schedule(prompt, 512, plans[512])) == window_snapshots(
        schedule(prompt, 2048, plans[2048])
    ) == list(range(2048, prompt - 1, 2048))


@pytest.mark.parametrize("prompt", [32768, 131072])
def test_warm_continuation_slices_as_cold_above_a_boundary(prompt):
    for step in (512, 2048):
        cold = schedule(prompt, step, boundaries(prompt, step))
        cached = 16384
        warm = schedule(prompt, step, boundaries(prompt, step, cached), cached)
        assert warm == [entry for entry in cold if entry[0] >= cached]


def test_window_aligned_2052_moves_rolling_and_snapshot_positions():
    # Why the documented A/B value is 2048, not omlx's window multiple.
    prompt = 32768
    assert default_rolling_interval(2052) == 4104
    assert boundaries(prompt, 2052) != boundaries(prompt, 512)
    plan = boundaries(prompt, 2052)
    slices = schedule(prompt, 2052, plan)
    assert {b.position for b in plan} <= {start + rows for start, rows in slices}
    assert max(rows for _, rows in slices) == 2052
    assert window_snapshots(slices) != list(range(2048, prompt - 1, 2048))


def test_non_reference_step_gets_its_own_apc_namespace(monkeypatch):
    from mlx2.serving import apc_semantic_namespace, prefill_step_namespace_term

    model = adapter()
    base = ("semantic", "template")
    today = apc_semantic_namespace(base)
    at = lambda step, autoscale=False: apc_semantic_namespace(  # noqa: E731
        base, prefill_step=prefill_step_namespace_term(model, step, autoscale=autoscale)
    )
    # The default (and an explicit 512) keeps today's namespace exactly.
    assert prefill_step_namespace_term(model, 512, autoscale=False) is None
    assert at(512) == today == base
    assert at(2048) != today and at(2048) != at(4096)
    # The opt-in env moves prefill_step_default, not the reference.
    monkeypatch.setenv(flint.PREFILL_STEP_ENV, "2048")
    assert model.prefill_step_default() == 2048
    assert at(2048) != today
    # Autoscaled engines are unchanged, as are adapters at their own default.

    class Plain:
        def prefill_step_default(self):
            return 2048

    assert prefill_step_namespace_term(model, 8192, autoscale=True) is None
    assert prefill_step_namespace_term(Plain(), 2048, autoscale=False) is None
    assert prefill_step_namespace_term(Plain(), 512, autoscale=False) is not None


def test_resolved_step_is_bound_before_the_host_cache_clamp():
    import inspect

    from mlx2 import serving

    model = adapter()
    serving.bind_adapter_prefill_step(model, 2048)
    assert model.cache_budget(mtp=False).sliding_token_cap == WINDOW + 2048
    serving.bind_adapter_prefill_step(object(), 2048)  # no hook: no-op
    source = inspect.getsource(serving)
    bind = source.index("bind_adapter_prefill_step(adapter, self.prefill_step)")
    clamp = source.index("self._clamp_host_default_cache_bytes(adapter)")
    resolved = source.index('self.prefill_step_source = "adapter"')
    assert resolved < bind < clamp
