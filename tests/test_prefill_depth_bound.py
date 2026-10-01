"""jundot/omlx#4149: an opt-in, depth-aware bound on the prefill chunk.

At deep KV one fixed prefill chunk's attention grows with rows x depth, and
omlx lost a 245K prompt when one chunk's command buffer outran the Metal
watchdog.  ``prefill_depth_budget`` shrinks a chunk only once
``rows x (depth + rows)`` would exceed the budget, so a request whose
prefill ends at or before ``budget // step`` tokens keeps its schedule (and
its output bits).  CPU tests over the tiny Flash-Next (qwen4_exp) hybrid, on
both the ordinary and the self-MTP prefill paths, and through a served route.
"""

import mlx.core as mx
import pytest

from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.runtime.generate import BatchGenerator, interior_checkpoint_positions
from mlx2.runtime.prefill_plan import depth_bounded_prefill_rows
from mlx2.runtime.sample_utils import LaneRNG
from route_harness import collect, make_engine, patch_host, tiny_qwen38_mtp
from test_batched_mtp import _tiny_qwen4_model

STEP = 32
BUDGET = STEP * 128  # the old chunk is kept for every chunk ending <= 128
FLOOR = 4


@pytest.fixture(scope="module")
def model():
    mx.random.seed(41)
    return _tiny_qwen4_model()


@pytest.fixture
def small_floor(monkeypatch):
    # Production floors at 128 rows; the tiny model's prompts are shorter.
    monkeypatch.setattr(BatchGenerator, "PREFILL_DEPTH_FLOOR", FLOOR)


def test_rows_keep_the_step_until_the_budget_and_then_halve():
    budget = 8192 * 131072
    # Every chunk that ends at or before budget // step is unchanged.
    for depth in (0, 65536, 131072 - 8192):
        assert depth_bounded_prefill_rows(8192, depth, budget) == 8192
    assert depth_bounded_prefill_rows(8192, 131072 - 8192 + 1, budget) == 4096
    assert depth_bounded_prefill_rows(8192, 253952, budget) == 4096
    # A non-power-of-two step drops to the power of two below it.
    assert depth_bounded_prefill_rows(3000, 10**6, 2048 * 10**6) == 1024
    # Never below min(step, floor), never above step.
    assert depth_bounded_prefill_rows(8192, 10**9, 1) == 128
    assert depth_bounded_prefill_rows(64, 10**9, 1) == 64
    assert depth_bounded_prefill_rows(2048, 10**9, None) == 2048
    for depth in range(0, 300000, 977):
        rows = depth_bounded_prefill_rows(2048, depth, 2048 * 100000)
        assert rows == 2048 or (
            rows * (depth + rows) <= 2048 * 100000 or rows == 128
        )
    with pytest.raises(ValueError):
        depth_bounded_prefill_rows(2048, 0, 0)
    with pytest.raises(ValueError):
        BatchGenerator(None, prefill_step_size=64, prefill_depth_budget=-1)


def _expected(prompt_len, *, budget, checkpoints=(), floor=FLOOR):
    widths, depth = [], 0
    while depth < prompt_len - 1:
        rows = depth_bounded_prefill_rows(STEP, depth, budget, floor=floor)
        rows = min(rows, prompt_len - 1 - depth)
        later = [c for c in checkpoints if c > depth]
        if later:
            rows = min(rows, later[0] - depth)
        widths.append(rows)
        depth += rows
    return widths


def _run(model, prompt, *, mtp, budget, checkpoints=None):
    kwargs = dict(
        prefill_step_size=STEP, prefill_depth_budget=budget,
        completion_batch_size=1, prefill_batch_size=1,
    )
    insert = dict(max_tokens=[4], lane_rngs=[LaneRNG(5)])
    if mtp:
        kwargs["self_mtp"] = {"num_draft": 2, "persistent": True}
        insert.update(
            mtp_states=[None],
            self_mtp_configs=[{"sampling_temp": 0.0, "num_draft": 2}],
        )
    if checkpoints is not None:
        kwargs["apc_interior_checkpoints"] = checkpoints
    gen = BatchGenerator(model, **kwargs)
    widths = []
    record = gen._record_prefill_chunk

    def recorded(uid, width):
        widths.append(int(width))
        return record(uid, width)

    gen._record_prefill_chunk = recorded
    try:
        uid = gen.insert([prompt], **insert)[0]
        tokens = []
        for _ in range(1000):
            _, responses = gen.next()
            tokens.extend(r.token for r in responses)
            if any(r.finish_reason for r in responses):
                break
        else:
            raise AssertionError("request did not finish")
        interiors = gen.pop_interior_checkpoints(uid) if checkpoints else []
        trace = gen.pop_prefill_chunk_trace(uid)
        stats = dict(gen.scheduler_stats)
    finally:
        gen.close()
    return widths, tokens, interiors, trace, stats


@pytest.mark.parametrize("mtp", [False, True])
def test_bound_shrinks_only_chunks_past_the_budget(model, small_floor, mtp):
    prompt = [(5 * i + 3) % 60 + 2 for i in range(300)]
    off, off_tokens, _, off_trace, off_stats = _run(model, prompt, mtp=mtp, budget=None)
    on, _, _, trace, stats = _run(model, prompt, mtp=mtp, budget=BUDGET)
    assert set(off[:-1]) == {STEP}
    assert off_stats.get("prefill_depth_bounded_chunks", 0) == 0
    assert off_trace["depth_budget"] is None
    expected = _expected(len(prompt), budget=BUDGET)
    # The self-MTP lane consumes its last <= step + 1 tokens in the final
    # preparation call, which records no chunk.
    assert on == (expected if not mtp else expected[: len(on)])
    assert sum(on) >= len(prompt) - 1 - (STEP + 1)
    # The first four chunks (ending at 128 = budget // step) are unchanged.
    assert on[:4] == off[:4] == [STEP] * 4
    assert 16 in on and 8 in on and max(on[4:]) < STEP
    assert stats["prefill_depth_bounded_chunks"] >= len(on) - 4
    assert trace["depth_budget"] == BUDGET and trace["varied"]


@pytest.mark.parametrize("mtp", [False, True])
def test_requests_inside_the_budget_are_bit_identical(model, small_floor, mtp):
    prompt = [(7 * i + 1) % 60 + 2 for i in range(120)]
    off = _run(model, prompt, mtp=mtp, budget=None)
    on = _run(model, prompt, mtp=mtp, budget=BUDGET)
    assert on[0] == off[0]
    assert on[1] == off[1]
    assert on[4].get("prefill_depth_bounded_chunks", 0) == 0


def test_bound_keeps_apc_interior_checkpoint_geometry(model, small_floor):
    """Checkpoints still land exactly on their lattice positions."""
    prompt = [(3 * i + 5) % 60 + 2 for i in range(300)]
    policy = {"count": 3, "min_stride": 4}
    positions = interior_checkpoint_positions(len(prompt), **policy)
    assert positions[-1] > 128  # at least one checkpoint past the budget
    widths, _, interiors, _, stats = _run(
        model, prompt, mtp=True, budget=BUDGET, checkpoints=policy
    )
    assert [item["covered_tokens"] for item in interiors] == list(positions)
    expected = _expected(len(prompt), budget=BUDGET, checkpoints=positions)
    assert widths == expected[: len(widths)]
    assert stats["prefill_depth_bounded_chunks"] > 0


def test_flash_next_policy_declares_the_bound_and_keeps_it_out_of_receipts():
    assert FlashNextPolicy().prefill_depth_budget is None
    assert "prefill_depth_budget" not in FlashNextPolicy().as_dict()
    policy = FlashNextPolicy.from_mapping({"prefill_depth_budget": 1 << 30})
    assert policy.as_dict()["prefill_depth_budget"] == 1 << 30
    with pytest.raises(ValueError, match="prefill_depth_budget"):
        FlashNextPolicy.from_mapping({"prefill_depth_budget": 0})


def test_served_route_binds_the_budget_and_reports_the_chunks(monkeypatch):
    patch_host(monkeypatch)
    monkeypatch.setattr(BatchGenerator, "PREFILL_DEPTH_FLOOR", FLOOR)
    model, vocab = tiny_qwen38_mtp()
    prompt = [(7 * i + 2) % (vocab - 2) + 1 for i in range(200)]
    body = {"tokens": prompt, "max_tokens": 4, "temperature": 0}
    plain = make_engine(model, vocab, mtp=False)
    try:
        assert "prefill_depth_budget" not in plain.snapshot["settings"]
        reference = collect(plain.submit(dict(body)))
    finally:
        plain.close()
    # make_engine's step is 16: chunks ending past 64 tokens shrink.
    bounded = make_engine(model, vocab, mtp=False, prefill_depth_budget=16 * 64)
    try:
        assert bounded.snapshot["settings"]["prefill_depth_budget"] == 16 * 64
        assert bounded.prefill_depth_budget_source == "engine_argument"
        served = collect(bounded.submit(dict(body)))
    finally:
        bounded.close()
    assert "error" not in served, served
    chunk = served["receipt"]["prefill_chunk"]
    assert chunk["depth_budget"] == 16 * 64 and chunk["varied"]
    assert chunk["widths"]["16"] == 4 and set(chunk["widths"]) >= {"16", "8", "4"}
    assert reference["receipt"]["prefill_chunk"]["depth_budget"] is None


def test_an_explicit_budget_fails_closed_on_routes_that_chunk_on_their_own(monkeypatch):
    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    with pytest.raises((AssertionError, RuntimeError), match="prefill depth budget"):
        make_engine(model, vocab, mtp=False, prompt_lookup=True,
                    prefill_depth_budget=1024)


def test_cli_flag_reaches_the_engine():
    from mlx2.server import build_parser, serving_engine_kwargs

    def kwargs(*extra):
        args = build_parser().parse_args(["--model", "fixture", *extra])
        return serving_engine_kwargs(
            args, None, native_mtp=False, approximate_kv=None,
            max_request_bytes=1 << 20,
        )

    assert kwargs("--prefill-depth-budget", "1048576")["prefill_depth_budget"] == 1048576
    assert kwargs()["prefill_depth_budget"] is None
    with pytest.raises(SystemExit):
        kwargs("--prefill-depth-budget", "0")


def test_prefill_input_request_that_cannot_be_bounded_is_refused():
    from mlx2.serving import refuse_unbounded_prefill_input

    # Off by default: nothing is refused.
    refuse_unbounded_prefill_input(None, 200_000, 0)
    # Within the budget the single chunk is allowed (rows * rows <= budget).
    refuse_unbounded_prefill_input(4096 * 4096, 4097, 0)
    # A prefill-input chunk the bound would have to shrink fails closed.
    with pytest.raises(ValueError, match="prefill depth budget"):
        refuse_unbounded_prefill_input(4096 * 4096, 8193, 0)
