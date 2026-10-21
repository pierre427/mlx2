"""--ast route observability: /v1/status ast_route and bounded /metrics series."""

import json
import re
from types import SimpleNamespace

import pytest

from mlx2.ast_observability import (
    DRAFTER_KINDS,
    NODE_LABELS,
    WIDTH_BUCKETS,
    ast_scheduler_series,
    route_info_labels,
    route_status,
    route_view,
)
from mlx2.prometheus import render_engine_metrics
from mlx2.runtime.ast_tile_budget import TileBudgetController
from mlx2.server import apply_ast_invocation, serving_engine_kwargs, build_parser

from test_prometheus import FakeEngine, assert_valid_prometheus_text

FINGERPRINT = "ab12cd34ef56" + "0" * 52


def _engine(route=None, *, scheduler=None, draft_fingerprint=FINGERPRINT):
    engine = FakeEngine()
    if scheduler is not None:
        engine.snapshot["scheduler"] = scheduler
    engine.ast_route = route
    engine.adapter = SimpleNamespace(identity={"draft_fingerprint": draft_fingerprint})
    return engine


def _drafter(tmp_path, architecture=None):
    path = tmp_path / "drafter"
    path.mkdir()
    config = {} if architecture is None else {"architectures": [architecture]}
    (path / "config.json").write_text(json.dumps(config))
    return str(path)


def _ast_series(text):
    return [line for line in text.splitlines() if "mlx2_ast_" in line]


def _info_line(text):
    lines = [line for line in text.splitlines() if line.startswith("mlx2_ast_route_info{")]
    assert len(lines) == 1, lines
    return lines[0]


def _old_receipt():
    controller = TileBudgetController(warmup=1, probe=1)
    for width, nodes, ranks, seconds in (
        (1, 15, [0, 1, 2], 0.020), (1, 7, [0, 1], 0.012), (1, 15, [0], 0.021),
        (2, 15, [0, 3], 0.030), (2, 7, [0], 0.018),
    ):
        controller.observe(width, nodes, ranks, seconds)
        controller.choose(width, 15)
    return controller.receipt()


def _new_shape_scheduler():
    """The calibration / row-sharing lane's receipts (branch a664d3a4)."""
    receipt = _old_receipt()
    receipt["1"]["fit"] = {"intercept_ms": 9.5, "slope_ms_per_row": 0.6}
    return {
        "external_tree_tile_budget": receipt,
        "external_tree_node_budget_histogram": {"15": 4, "7": 2},
        "external_tree_calibration": {
            "schema": "x", "mode": "order_size", "law": "sigmoid(...)",
            "window": 512, "min_records": 64, "refit_every": 16,
            "trees": 40, "fits": 2,
            "predicted_tokens_per_tree": 3.25, "realized_tokens_per_tree": 3.1,
            "rate_tokens_per_second": 61.5,
            "truncated_rounds": 5, "truncated_nodes": 17,
            "tables": {
                "greedy": {"trees": 30, "alpha": {"1": 0.4, "2": -0.1},
                           "beta": {"1": 1.1, "2": 0.9}, "trials": {"1": 30, "2": 22},
                           "hit_rate": {"1": 0.8, "2": 0.55}},
                "sampled": {"trees": 10, "alpha": {"1": 0.2}, "beta": {"1": 1.0},
                            "trials": {"1": 10}, "hit_rate": {"1": 0.6}},
            },
        },
        "external_tree_row_sharing_rounds": 6,
        "external_tree_row_sharing_lane_nodes": {"11": 3, "4": 3},
        "ast_planner_source_budgets": {"fixed_tree_source_budget": 9, "secret-reason": 1},
        "ast_planner_launches": {"primary": 7, "alternative": 2},
        "ast_planner_launch_rows": {"logical": 50, "physical": 64, "padding": 14},
    }


def test_old_shape_tile_budget_receipt_is_exported():
    text = render_engine_metrics(
        _engine(scheduler={"external_tree_tile_budget": _old_receipt()})
    )
    assert_valid_prometheus_text(text)
    assert 'mlx2_ast_tile_rounds_total{width_bucket="1"} 3' in text
    assert 'mlx2_ast_tile_rounds_total{width_bucket="2"} 2' in text
    assert re.search(r'mlx2_ast_tile_cost_seconds\{nodes="15",width_bucket="1"\} 0\.02', text)
    assert 'mlx2_ast_tile_tokens_per_round{nodes="15",width_bucket="1"}' in text
    assert "# TYPE mlx2_ast_tile_rounds_total counter" in text
    assert "# TYPE mlx2_ast_tile_cost_seconds gauge" in text
    assert "mlx2_ast_tile_chosen_nodes{" in text


def test_new_shape_receipts_are_exported():
    text = render_engine_metrics(_engine(scheduler=_new_shape_scheduler()))
    assert_valid_prometheus_text(text)
    for expected in (
        'mlx2_ast_tile_cost_fit_seconds{term="intercept",width_bucket="1"} 0.009',
        'mlx2_ast_tile_cost_fit_seconds{term="slope",width_bucket="1"} 0.000',
        'mlx2_ast_tile_decisions_total{nodes="15"} 4',
        'mlx2_ast_tree_calibration_enabled{mode="order_size"} 1',
        "mlx2_ast_tree_calibration_trees_total 40",
        'mlx2_ast_tree_calibration_truncations_total{unit="nodes"} 17',
        'mlx2_ast_tree_calibration_tokens_per_tree{kind="realized"} 3.1',
        'mlx2_ast_tree_calibration_parameter{depth="2",parameter="beta",table="greedy"} 0.9',
        'mlx2_ast_tree_calibration_window_trials{depth="1",table="sampled"} 10',
        'mlx2_ast_tree_calibration_hit_ratio{depth="1",table="greedy"} 0.8',
        "mlx2_ast_row_sharing_rounds_total 6",
        'mlx2_ast_row_sharing_lane_nodes_total{nodes="11"} 3',
        'mlx2_ast_planner_source_budgets_total{reason="fixed_tree_source_budget"} 9',
        'mlx2_ast_planner_source_budgets_total{reason="other"} 1',
        'mlx2_ast_planner_launches_total{backend_role="alternative"} 2',
        'mlx2_ast_planner_launch_rows_total{kind="padding"} 14',
    ):
        assert expected in text, expected
    assert "secret-reason" not in text


@pytest.mark.parametrize(
    "drafter, architecture, kind",
    [
        ("mtp", None, "mtp"),
        ("pld", None, "pld"),
        ("artifact", None, "dflash2"),
        ("artifact", "LlamaForCausalLMEagle4", "eagle4"),
        ("artifact", "Qwen3DSparkModel", "dspark"),
        ("missing", None, "external"),
    ],
)
def test_route_info_for_each_drafter_kind(tmp_path, drafter, architecture, kind):
    path = (
        _drafter(tmp_path, architecture) if drafter == "artifact"
        else str(tmp_path / "absent") if drafter == "missing" else drafter
    )
    route = {"tree_tiles": True, "drafter": path, "topology": "5x3",
             "graft_pld": drafter == "pld", "tiles": None, "tile_rows": 8,
             "drafter_policy_defaults": {"draft_kernel": "lane"}}
    policy = {"draft_kernel": "lane", "draft_quantization": {"bits": 4, "group_size": 64}}
    status = route_status(route, policy)
    assert status["drafter_kind"] == kind
    text = render_engine_metrics(_engine(status))
    line = _info_line(text)
    assert str(tmp_path) not in text
    assert f'drafter="{kind}"' in line
    assert 'tile_rows="8"' in line and 'tile_rows_mode="adaptive"' in line
    assert 'drafter_kernel="lane"' in line and 'drafter_quantization="q4g64"' in line
    assert 'topology="5x3"' in line and 'tree_tiles="on"' in line
    if kind in ("mtp", "pld"):
        assert 'drafter_fingerprint="none"' in line
    else:
        assert 'drafter_fingerprint="ab12cd34ef56"' in line
    assert line.endswith(" 1")


def test_refused_route_keeps_its_reason_in_status_only():
    reason = "tree tiles need bf16 conv weights; /secret/model/path has float16"
    status = route_status(
        {"tree_tiles": False, "drafter": "mtp", "topology": None, "graft_pld": False,
         "tiles": None, "tile_rows": None, "tree_tiles_refused": reason},
    )
    assert status["tree_tiles"] == "refused"
    assert status["tree_tiles_refused_reason"] == reason
    assert status["tile_rows_mode"] == "not_applicable"
    engine = _engine(status)
    text = render_engine_metrics(engine)
    line = _info_line(text)
    assert 'tree_tiles="refused"' in line
    assert "secret" not in text and "bf16" not in text
    assert route_view(engine)["tree_tiles_refused_reason"] == reason


def test_free_form_values_collapse_to_bounded_labels():
    status = route_status(
        {"tree_tiles": True, "drafter": "/m/d", "topology": "evil\"label",
         "graft_pld": False, "tiles": 2, "tile_rows": 99},
        {"draft_kernel": "secret-kernel", "draft_quantization": {"bits": 3, "group_size": 7}},
    )
    labels = route_info_labels({**status, "drafter_fingerprint": "/m/d"})
    assert labels["topology"] == "other"
    assert labels["tile_rows"] == "other"
    assert labels["drafter_kernel"] == "other"
    assert labels["drafter_quantization"] == "other"
    assert labels["drafter_fingerprint"] == "unknown"
    assert labels["drafter"] in DRAFTER_KINDS


def test_label_cardinality_stays_bounded():
    widths = range(1, 200)
    receipt = {
        str(width): {
            "rounds": 1, "chosen": width % 40,
            "cost_ms": {str(nodes): 1.0 for nodes in range(0, 120)},
            "tokens_per_round": {str(nodes): 1.0 for nodes in range(0, 120)},
        }
        for width in widths
    }
    scheduler = {
        "external_tree_tile_budget": receipt,
        "external_tree_node_budget_histogram": {str(n): 1 for n in range(500)},
        "external_tree_row_sharing_lane_nodes": {str(n): 1 for n in range(500)},
        "ast_planner_source_budgets": {f"reason-{n}": 1 for n in range(500)},
    }
    series = ast_scheduler_series(scheduler)
    by_name = {}
    for item in series:
        by_name.setdefault(item.name, set()).add(item.labels)
        labels = dict(item.labels)
        assert labels.get("width_bucket", "1") in WIDTH_BUCKETS
        assert labels.get("nodes", "1") in NODE_LABELS
    assert len(by_name["mlx2_ast_tile_rounds_total"]) == len(WIDTH_BUCKETS)
    assert len(by_name["mlx2_ast_tile_cost_seconds"]) <= len(WIDTH_BUCKETS) * 33
    assert len(by_name["mlx2_ast_tile_decisions_total"]) == len(NODE_LABELS)
    assert len(by_name["mlx2_ast_planner_source_budgets_total"]) == 1
    # Collapsed widths sum their counters.
    rounds = {dict(s.labels)["width_bucket"]: s.value for s in series
              if s.name == "mlx2_ast_tile_rounds_total"}
    assert rounds["17+"] == len(range(17, 200))
    text = render_engine_metrics(_engine(scheduler=scheduler))
    assert_valid_prometheus_text(text)


def test_non_ast_server_exports_no_ast_series():
    engine = FakeEngine()
    text = render_engine_metrics(engine)
    assert not _ast_series(text)
    # Empty initialised receipts (a tree route before its first round) too.
    engine.snapshot["scheduler"] = {
        **engine.snapshot.get("scheduler", {}),
        "external_tree_node_budget_histogram": {},
    }
    assert not _ast_series(render_engine_metrics(engine))


def test_tenant_scrape_omits_ast_series():
    engine = _engine(route_status({"tree_tiles": True, "drafter": "pld"}),
                     scheduler=_new_shape_scheduler())
    text = render_engine_metrics(engine, tenant_id="tenant-a")
    assert not _ast_series(text)


def _invocation_args(**overrides):
    values = dict(model="/m/target", ast=True, ast_drafter=None, ast_topology=None,
                  ast_graft_pld=False, draft_model=None, external_draft=False,
                  native_mtp=False, prompt_lookup=False)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_status_carries_ast_route_from_the_invocation(tmp_path):
    drafter = _drafter(tmp_path)

    class Adapter:
        ast_tree_tiles = True

        @classmethod
        def ast_drafter_policy_defaults(cls, _drafter):
            return {"draft_kernel": "lane",
                    "draft_quantization": {"bits": 4, "group_size": 64}}

    resolution = SimpleNamespace(adapter_type=Adapter, artifact={"has_mtp": False},
                                 default_route="ordinary")
    args = _invocation_args(ast_drafter=drafter)
    policy = apply_ast_invocation(args, None, resolution)
    status = route_status(args.ast_route, policy, Adapter)
    assert status["drafter_kind"] == "dflash2"
    assert status["drafter_kernel"] == "lane"
    assert status["drafter_policy_defaults"] == ["draft_kernel", "draft_quantization"]
    assert drafter not in json.dumps(status)

    import inspect

    from mlx2.serving import ServingEngine

    engine = object.__new__(ServingEngine)
    engine.ast_route = status
    engine.adapter = SimpleNamespace(identity={"draft_fingerprint": FINGERPRINT})
    view = engine._ast_route_status()["ast_route"]
    assert view["drafter_fingerprint"] == FINGERPRINT[:12]
    assert view["drafter_identity"] == "loaded"
    assert view["tree_tiles"] == "on" and view["selected"] is True
    assert "**self._ast_route_status()" in inspect.getsource(ServingEngine.status)
    engine.ast_route = None
    assert engine._ast_route_status() == {}


def test_engine_kwargs_carry_the_route_status():
    args = build_parser().parse_args(["--model", "/m"])
    kw = dict(native_mtp=False, approximate_kv=None, max_request_bytes=1 << 20)
    assert serving_engine_kwargs(args, None, **kw)["ast_route"] is None
    args.ast_route_status = {"selected": True}
    assert serving_engine_kwargs(args, None, **kw)["ast_route"] == {"selected": True}
