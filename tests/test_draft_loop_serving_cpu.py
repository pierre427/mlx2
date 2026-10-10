"""Serving defaults for the draft loop: CLI, selection guard, admission charge."""

from types import SimpleNamespace

import pytest

from mlx2.runtime import verify_topology as vt
from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController as Controller


def test_cli_defaults_to_auto_and_reaches_the_engine():
    from mlx2.server import build_parser, serving_engine_kwargs

    parser = build_parser()
    args = parser.parse_args(["--model", "fixture"])
    assert args.draft_loop == "auto"
    kwargs = serving_engine_kwargs(
        args, None, native_mtp=True, approximate_kv=None, max_request_bytes=1 << 20
    )
    assert kwargs["draft_loop"] == "auto"
    assert parser.parse_args(["--model", "m", "--draft-loop", "off"]).draft_loop == "off"
    assert args.draft_loop_threshold is None and kwargs["draft_loop_threshold"] is None
    custom = parser.parse_args(["--model", "m", "--draft-loop-threshold", "-0.3"])
    assert serving_engine_kwargs(
        custom, None, native_mtp=True, approximate_kv=None, max_request_bytes=1 << 20
    )["draft_loop_threshold"] == -0.3
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "m", "--draft-loop", "on"])


def _engine(**overrides):
    state = dict(
        draft_loop_mode="auto", draft_loop_threshold=-0.4, draft_loop_widths=[1, 2],
        mtp=True, max_lanes=1,
        verify_topology_receipt="stale",
        lane_matmul_receipt={"covered": {"q4": 3}, "law_id": "lane-simd-v1"},
    )
    state.update(overrides)
    return SimpleNamespace(**state)


def _adapter():
    tokenizer = SimpleNamespace(encode=lambda text: [1, 2, 3])
    model = SimpleNamespace(args=SimpleNamespace(text_config={"vocab_size": 64}))
    return SimpleNamespace(model=model, tokenizer=tokenizer, identity={"fingerprint": "f"})


@pytest.mark.parametrize(
    "overrides, config",
    [
        ({"draft_loop_mode": "off"}, {"num_draft": 3}),
        ({"mtp": False}, {"num_draft": 3}),
        ({"lane_matmul_receipt": {"covered": {}, "law_id": "stock"}}, {"num_draft": 3}),
        ({}, {"num_draft": 3, "backend": "external_draft"}),
        ({}, {"num_draft": 3, "draft_loop": {"stage": 2, "threshold": -1.0}}),
    ],
)
def test_selection_leaves_the_policy_alone_when_it_does_not_apply(monkeypatch, overrides, config):
    from mlx2.serving import ServingEngine

    monkeypatch.setattr(vt, "resolve_topology", lambda *a, **k: pytest.fail("probed"))
    engine = _engine(**overrides)
    assert ServingEngine._select_draft_loop(engine, _adapter(), config) is config
    assert engine.verify_topology_receipt is None


def test_selection_adds_the_probed_topology(monkeypatch):
    from mlx2.serving import ServingEngine

    seen = {}

    def resolve(model, **kwargs):
        seen.update(kwargs)
        return {"selected": {"by_width": {"1": [3, 7]}, "threshold": -0.4, "cohort": "any"},
                "tile_edges": [8]}

    monkeypatch.setattr(vt, "resolve_topology", resolve)
    engine = _engine()
    config = {"num_draft": 3}
    out = ServingEngine._select_draft_loop(engine, _adapter(), config)
    assert out == {"num_draft": 3,
                   "draft_loop": {"by_width": {"1": [3, 7]}, "threshold": -0.4, "cohort": "any"}}
    assert config == {"num_draft": 3}  # the adapter's mapping is not mutated
    assert seen["identity"]["lane_law"] == "lane-simd-v1" and seen["base"] == 3
    assert seen["max_end"] == vt.MAX_END and seen["vocab_size"] == 64
    assert seen["threshold"] == -0.4
    assert seen["widths"] == [1]  # max_lanes 1 caps the probed widths


def test_selection_probes_two_lanes_when_the_engine_serves_them(monkeypatch):
    from mlx2.serving import ServingEngine

    seen = {}
    monkeypatch.setattr(vt, "resolve_topology",
                        lambda model, **kw: seen.update(kw) or {"selected": None})
    ServingEngine._select_draft_loop(_engine(max_lanes=4), _adapter(), {"num_draft": 3})
    assert seen["widths"] == [1, 2]


def test_selection_passes_the_operator_threshold(monkeypatch):
    from mlx2.serving import ServingEngine

    seen = {}
    monkeypatch.setattr(vt, "resolve_topology",
                        lambda model, **kw: seen.update(kw) or {"selected": None})
    ServingEngine._select_draft_loop(_engine(draft_loop_threshold=-0.3), _adapter(), {"num_draft": 3})
    assert seen["threshold"] == -0.3


@pytest.mark.parametrize("bad", [0.1, -2.5, -0.01, float("nan"), True, "x"])
def test_engine_refuses_a_threshold_outside_the_prior(bad):
    from mlx2.serving import ServingEngine

    with pytest.raises(ValueError, match="draft_loop_threshold"):
        ServingEngine("unused", mtp=True, max_lanes=1, draft_loop_threshold=bad)


def test_selection_without_a_prior_for_the_base_depth_records_why(monkeypatch):
    from mlx2.serving import ServingEngine

    monkeypatch.setattr(vt, "resolve_topology", lambda *a, **k: pytest.fail("probed"))
    engine = _engine()
    config = {"num_draft": 2}
    assert ServingEngine._select_draft_loop(engine, _adapter(), config) is config
    assert engine.verify_topology_receipt["selected"] is None


def test_admission_charges_the_loop_ceiling():
    from mlx2.serving import draft_loop_transient_ratio

    assert draft_loop_transient_ratio({"num_draft": 3}) == 1.0
    ratio = draft_loop_transient_ratio(
        {"num_draft": 3, "draft_loop": {"boundaries": [3, 7], "threshold": -0.4}}
    )
    assert ratio == Controller.TRANSIENT_SCALE[7] / Controller.TRANSIENT_SCALE[3] > 1.0
    # A per-width topology is charged for its deepest width.
    assert draft_loop_transient_ratio(
        {"num_draft": 3, "draft_loop": {"by_width": {"1": [3, 9], "2": [3, 7]},
                                        "threshold": -0.4, "cohort": "any"}}
    ) == Controller.TRANSIENT_SCALE[9] / Controller.TRANSIENT_SCALE[3]
    with pytest.raises(ValueError):
        Controller.depth_transient_ratio(3, 99)


def test_admission_row_cap_counts_the_loop_ceiling():
    """The admission controller's row cap is the most rows one round may
    execute: the widest MTP cohort at its loop-aware depth (+ the pending
    target row per lane) beside the ordinary lanes that fill the remaining
    slots.  A looped lane executes ceiling + 1, not num_draft + 1."""
    from mlx2.runtime.copy_draft import CopyDraftPolicy
    from mlx2.serving import self_mtp_verification_row_cap as cap

    loop = {"boundaries": [3, 9], "threshold": -0.4}
    assert cap({"num_draft": 3}, max_lanes=4) == 16
    assert cap({"num_draft": 3, "draft_loop": loop}, max_lanes=4) == 40
    # The probed topology (max_width 1) loops solo lanes only: one looped
    # lane (10 rows) beside ordinary siblings, or a full cohort at depth 3.
    probed = {"num_draft": 3, "draft_loop": {**loop, "max_width": 1}}
    assert cap(probed, max_lanes=4) == max(10 + 3, 4 * 4) == 16
    assert cap(probed, max_lanes=2) == max(10 + 1, 2 * 4) == 11
    assert cap(probed, max_lanes=1) == 10
    assert cap({"num_draft": 2, "draft_loop": loop}, max_lanes=4) == 12
    # A solo copy lane executes its copied span (8 + 1) beside its siblings.
    copying = CopyDraftPolicy(enabled=True)
    assert cap({"num_draft": 3}, max_lanes=2, copy_draft_policy=copying) == max(9 + 1, 8) == 10
    assert cap({"num_draft": 3}, max_lanes=4, copy_draft_policy=copying) == 16


def test_admission_row_cap_admits_a_looped_lane_beside_an_ordinary_one():
    """Mixed round at max_lanes 2: one ineligible lane decodes plainly (one
    row) while the eligible lane loops to 10 rows.  The cap must hold both,
    and the controller must still admit the eligible lane at full depth."""
    from mlx2.serving import self_mtp_verification_row_cap

    config = {"num_draft": 3, "draft_loop": {"boundaries": [3, 9], "threshold": -0.4, "max_width": 1}}
    cap = self_mtp_verification_row_cap(config, max_lanes=2)
    looped_rows, plain_rows = 9 + 1, 1
    assert cap >= looped_rows + plain_rows
    controller = Controller(verification_row_cap=cap, saturation_lane_cap=2)
    plan = controller.decide([100, 100], 1000, max_draft=3, eligible=(False, True))
    assert plan.modes == ("plain", "self_mtp") and plan.draft_depths == (0, 3)
    assert plan.primary_rows == 2 and plan.speculative_rows == 3


def test_loop_depth_helpers_follow_the_executor():
    from mlx2.runtime.draft_loop import (
        cohort_draft_depth,
        cohort_proposal_depths,
        draft_depth_ceiling,
    )

    loop = {"boundaries": [3, 7], "threshold": -0.4, "max_width": 2}
    config = {"num_draft": 3, "draft_loop": loop}
    assert [cohort_draft_depth(config, lanes=n) for n in (1, 2, 3)] == [7, 7, 3]
    assert draft_depth_ceiling(config) == 7 and draft_depth_ceiling({"num_draft": 3}) == 3
    assert draft_depth_ceiling({"num_draft": 2, "draft_loop": loop}) == 2
    assert cohort_proposal_depths(config, max_lanes=3) == {1: 7, 2: 7, 3: 3}
    assert cohort_proposal_depths(None, max_lanes=2) == {1: 0, 2: 0}


def test_transient_scale_is_calibrated_to_the_prior_and_monotone():
    scale = Controller.TRANSIENT_SCALE
    assert all(depth in scale for depth in range(0, vt.MAX_END + 1))
    values = [scale[d] for d in sorted(scale)]
    assert values == sorted(values)


# --------------------------------------------------------------------------
# The adapter's exact self-MTP row window bounds the loop ceiling
# --------------------------------------------------------------------------


class _WindowedAdapter:
    """An adapter whose verify/rollback tensor path is exact through 9 rows
    (proposer depth 8), like Qwen3.8-27B."""

    max_exact_self_mtp_verification_rows = 9
    max_exact_self_mtp_rollback_rows = 9

    def __init__(self):
        self.tokenizer = SimpleNamespace(encode=lambda text: [1, 2, 3])
        self.model = SimpleNamespace(args=SimpleNamespace(text_config={"vocab_size": 64}))
        self.identity = {"fingerprint": "f"}


def _resolve_recording(seen, selected):
    def resolve(model, **kwargs):
        seen.update(kwargs)
        return {"selected": selected, "tile_edges": [4, 5, 6], "key": {"max_end": kwargs["max_end"]}}

    return resolve


def test_probe_ceiling_is_capped_by_the_adapter_exact_row_window(monkeypatch):
    from mlx2.serving import ServingEngine

    seen = {}
    selected = {"boundaries": [3, 8], "threshold": -0.4, "max_width": 1}
    monkeypatch.setattr(vt, "resolve_topology", _resolve_recording(seen, selected))
    engine = _engine()
    out = ServingEngine._select_draft_loop(engine, _WindowedAdapter(), {"num_draft": 3})
    # 9 exact rows = one pending target row + at most 8 drafts.
    assert seen["max_end"] == 8 < vt.MAX_END
    assert seen["identity"]["exact_self_mtp_rows"] == 9
    assert out["draft_loop"] == selected
    assert engine.verify_topology_receipt["selected"] == selected


def test_an_undeclared_adapter_keeps_the_prior_ceiling(monkeypatch):
    from mlx2.serving import ServingEngine

    seen = {}
    monkeypatch.setattr(vt, "resolve_topology", _resolve_recording(seen, None))
    ServingEngine._select_draft_loop(_engine(), _adapter(), {"num_draft": 3})
    assert seen["max_end"] == vt.MAX_END
    assert seen["identity"]["exact_self_mtp_rows"] is None


def test_a_selection_past_the_exact_window_is_refused(monkeypatch):
    from mlx2.serving import ServingEngine

    seen = {}
    # A probe (or a stale cache entry) answering with an end the window
    # cannot verify exactly: 9 drafts are 10 verify rows against 9 declared.
    bad = {"boundaries": [3, 9], "threshold": -0.4, "max_width": 1}
    monkeypatch.setattr(vt, "resolve_topology", _resolve_recording(seen, bad))
    engine = _engine()
    with pytest.raises(ValueError, match="10 verify rows.*9"):
        ServingEngine._select_draft_loop(engine, _WindowedAdapter(), {"num_draft": 3})
    receipt = engine.verify_topology_receipt
    assert receipt["selected"] is None and "10 verify rows" in receipt["refused"]


@pytest.mark.parametrize("mode", ["auto", "off"])
def test_an_operator_loop_past_the_exact_window_is_refused(monkeypatch, mode):
    from mlx2.serving import ServingEngine

    monkeypatch.setattr(vt, "resolve_topology", lambda *a, **k: pytest.fail("probed"))
    engine = _engine(draft_loop_mode=mode)
    config = {"num_draft": 3, "draft_loop": {"boundaries": [3, 9], "threshold": -0.4}}
    with pytest.raises(ValueError, match="10 verify rows.*9"):
        ServingEngine._select_draft_loop(engine, _WindowedAdapter(), config)
    # A loop inside the window, or one the base depth never reaches, is honoured.
    inside = {"num_draft": 3, "draft_loop": {"boundaries": [3, 8], "threshold": -0.4}}
    assert ServingEngine._select_draft_loop(engine, _WindowedAdapter(), inside) is inside


def test_an_operator_loop_refusal_leaves_a_receipt(monkeypatch):
    """The docstring promises the reason in the topology receipt for an
    operator loop too, not only for a probed one (seam review A, N1)."""
    from mlx2.serving import ServingEngine

    monkeypatch.setattr(vt, "resolve_topology", lambda *a, **k: pytest.fail("probed"))
    engine = _engine(draft_loop_mode="off")
    config = {"num_draft": 3, "draft_loop": {"boundaries": [3, 9], "threshold": -0.4}}
    with pytest.raises(ValueError):
        ServingEngine._select_draft_loop(engine, _WindowedAdapter(), config)
    receipt = engine.verify_topology_receipt
    assert receipt["selected"] is None
    assert receipt["source"] == "operator"
    assert "10 verify rows" in receipt["refused"]
    below = {"num_draft": 2, "draft_loop": {"boundaries": [3, 9], "threshold": -0.4}}
    assert ServingEngine._select_draft_loop(engine, _WindowedAdapter(), below) is below
    # External-draft routes do not carry the self-MTP window.
    external = {**config, "backend": "external_draft"}
    assert ServingEngine._select_draft_loop(engine, _WindowedAdapter(), external) is external


def test_a_window_with_no_end_above_the_base_depth_skips_the_probe(monkeypatch):
    from mlx2.serving import ServingEngine

    class _Narrow(_WindowedAdapter):
        max_exact_self_mtp_verification_rows = 4
        max_exact_self_mtp_rollback_rows = 4

    monkeypatch.setattr(vt, "resolve_topology", lambda *a, **k: pytest.fail("probed"))
    engine = _engine()
    config = {"num_draft": 3}
    assert ServingEngine._select_draft_loop(engine, _Narrow(), config) is config
    receipt = engine.verify_topology_receipt
    assert receipt["selected"] is None and "exact self-MTP" in receipt["reason"]


def test_production_adapters_cap_the_probe_at_their_declared_window(monkeypatch):
    from mlx2.adapters.flash_next import FlashNextAdapter
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter
    from mlx2.serving import ServingEngine

    expected = {Qwen3827BAdapter: 8, FlashNextAdapter: vt.MAX_END}
    for adapter_type, max_end in expected.items():
        adapter = object.__new__(adapter_type)
        adapter.tokenizer = SimpleNamespace(encode=lambda text: [1, 2, 3])
        adapter.model = SimpleNamespace(args=SimpleNamespace(text_config={"vocab_size": 64}))
        adapter.identity = {"fingerprint": "f"}
        seen = {}
        monkeypatch.setattr(vt, "resolve_topology", _resolve_recording(seen, None))
        ServingEngine._select_draft_loop(_engine(), adapter, {"num_draft": 3})
        assert seen["max_end"] == max_end, adapter_type.__name__


def test_exact_rows_receipt_reports_the_loop_ceiling():
    from mlx2.adapters.self_mtp_rows import (
        constrain_self_mtp_draft_loop,
        constrain_self_mtp_proposers,
        declared_exact_self_mtp_rows,
    )
    from mlx2.runtime.copy_draft import CopyDraftPolicy

    rows = declared_exact_self_mtp_rows(_WindowedAdapter())
    _, _, receipt = constrain_self_mtp_proposers(
        rows, self_mtp_num_draft=3, self_mtp_copy_draft_policy=CopyDraftPolicy()
    )
    fixed = constrain_self_mtp_draft_loop(rows, receipt, {"num_draft": 3})
    assert fixed["effective_self_mtp_draft_ceiling"] == 3
    assert fixed["effective_self_mtp_verify_rows"] == 4
    looped = constrain_self_mtp_draft_loop(
        rows, receipt, {"num_draft": 3, "draft_loop": {"boundaries": [3, 8], "threshold": -0.4}}
    )
    assert looped["effective_self_mtp_draft_ceiling"] == 8
    assert looped["effective_self_mtp_verify_rows"] == 9
    assert looped["effective_max_self_mtp_rows"] == 9 and looped["clamped"] is False
    # A copy round verifies the copied span in place of the head drafts: the
    # verify-row figure is the wider of the two (8-token copy at depth 3 = 9 rows).
    _, _, copying = constrain_self_mtp_proposers(
        rows, self_mtp_num_draft=3, self_mtp_copy_draft_policy=CopyDraftPolicy(enabled=True)
    )
    assert copying["effective_self_mtp_copy_max_span"] == 8
    with_copy = constrain_self_mtp_draft_loop(rows, copying, {"num_draft": 3})
    assert with_copy["effective_self_mtp_draft_ceiling"] == 3
    assert with_copy["effective_self_mtp_verify_rows"] == 9
    with pytest.raises(ValueError, match="10 verify rows.*9"):
        constrain_self_mtp_draft_loop(
            rows, receipt, {"num_draft": 3, "draft_loop": {"boundaries": [3, 9], "threshold": -0.4}}
        )


def test_async_qsa_reserve_tail_counts_the_loop_ceiling():
    """The promotion's first commit may advance a looped lane by ceiling + 1
    rows; a tail sized from num_draft declined every extended commit."""
    from mlx2.runtime.draft_loop import DraftLoopPolicy
    from mlx2.runtime.generate import MTPGenerationBatch

    def tail(lanes, loop):
        batch = SimpleNamespace(
            draft_loop=DraftLoopPolicy.from_value(loop),
            state=SimpleNamespace(lanes=[SimpleNamespace(num_draft=d) for d in lanes]),
        )
        return MTPGenerationBatch._async_qsa_reserve_tail(batch)

    loop = {"boundaries": [3, 9], "threshold": -0.4, "max_width": 1}
    assert tail([3], None) == 4
    assert tail([3], loop) == 10
    assert tail([3, 3], loop) == 4  # wider than the probed width: fixed depth
    assert tail([3, 3], {**loop, "max_width": None}) == 10
    assert tail([2, 3], {**loop, "max_width": None}) == 10
    assert tail([2], loop) == 3  # below the first boundary: not gated


def test_async_qsa_reserve_tail_counts_the_copy_span():
    from mlx2.runtime.copy_draft import CopyDraftPolicy
    from mlx2.runtime.generate import MTPGenerationBatch

    def tail(lanes, policy):
        state = SimpleNamespace(policy=policy)
        batch = SimpleNamespace(
            draft_loop=None,
            state=SimpleNamespace(
                lanes=[SimpleNamespace(num_draft=d, copy_draft=state) for d in lanes]
            ),
        )
        return MTPGenerationBatch._async_qsa_reserve_tail(batch)

    # A solo copy lane may commit its whole copied span (8 + 1 rows).
    assert tail([3], CopyDraftPolicy(enabled=True)) == 9
    # The default cohort policy refuses batched copies: head depth + 1.
    assert tail([3, 3], CopyDraftPolicy(enabled=True)) == 4
    assert tail([3, 3], CopyDraftPolicy(enabled=True, max_span=16, batched_max_span=16)) == 17


def test_operator_widths_limit_the_probe(monkeypatch):
    from mlx2.serving import ServingEngine

    seen = {}
    monkeypatch.setattr(vt, "resolve_topology",
                        lambda model, **kw: seen.update(kw) or {"selected": None})
    ServingEngine._select_draft_loop(
        _engine(max_lanes=4, draft_loop_widths=[1]), _adapter(), {"num_draft": 3})
    assert seen["widths"] == [1]


@pytest.mark.parametrize("bad", ["0", "1,x", "5", [], [True]])
def test_engine_refuses_bad_widths(bad):
    from mlx2.serving import ServingEngine

    with pytest.raises(ValueError, match="draft_loop_widths"):
        ServingEngine("unused", mtp=True, max_lanes=1, draft_loop_widths=bad)


def test_cli_widths_reach_the_engine():
    from mlx2.server import build_parser, serving_engine_kwargs

    args = build_parser().parse_args(["--model", "m", "--draft-loop-widths", "1"])
    kwargs = serving_engine_kwargs(
        args, None, native_mtp=True, approximate_kv=None, max_request_bytes=1 << 20
    )
    assert kwargs["draft_loop_widths"] == "1"
