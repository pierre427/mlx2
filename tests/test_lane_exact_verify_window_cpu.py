"""Lane ``exact`` mode bound to the self-MTP verify width (Pierre 2026-10-14, D2).

``exact`` gives every 1..max_rows-row call the lane arithmetic, so a verify
row equals the same row decoded alone.  A call wider than ``max_rows`` (32 by
default) silently ran stock MLX (installer: ``stock_above_max_rows``), so a
cohort of lanes x (draft ceiling + 1) rows past the window lost that
equality without any refusal.  Serving now computes the widest self-MTP
verify call the route can issue and refuses an exact route whose window
does not hold it, with a receipt naming the rows and the fixes.
"""

import re
from pathlib import Path

import pytest

from mlx2.serving import lane_exact_verify_window


def _receipt(mode="exact", max_rows=32, chunk=False, covered=True):
    return {
        "law_id": "lane-matmul-v1",
        "covered": {"affine-q4-g64": 10} if covered else {},
        "max_rows": max_rows,
        "chunk_above_max": chunk,
        "policy": {"mode": mode, "max_rows": max_rows, "chunk_above_max": chunk},
    }


def test_a_window_that_holds_every_verify_call_is_admitted_and_reported():
    window = lane_exact_verify_window(_receipt(), {"num_draft": 3}, max_lanes=4)
    # w lanes verifying beside max_lanes - w plain one-row lanes.
    assert window["verify_rows"] == 16
    assert window["by_width"] == {"1": 7, "2": 10, "3": 13, "4": 16}
    assert window["max_rows"] == 32 and window["refused"] is None


def test_a_verify_wider_than_the_window_is_refused_with_its_rows():
    config = {"num_draft": 3, "draft_loop": {"boundaries": [3, 9], "threshold": -0.4}}
    window = lane_exact_verify_window(_receipt(), config, max_lanes=4)
    assert window["verify_rows"] == 40 and window["by_width"]["4"] == 40
    assert window["refused"] is not None
    assert "40" in window["refused"] and '"max_rows": 40' in window["refused"]


def test_chunking_or_a_wide_enough_window_admits_it():
    config = {"num_draft": 3, "draft_loop": {"boundaries": [3, 9], "threshold": -0.4}}
    assert lane_exact_verify_window(_receipt(max_rows=40), config, max_lanes=4)["refused"] is None
    # Chunks of max_rows keep each row's bits (split-K is shape-only).
    assert lane_exact_verify_window(_receipt(chunk=True), config, max_lanes=4)["refused"] is None


def test_past_the_kernel_limit_only_chunking_can_hold_it():
    window = lane_exact_verify_window(_receipt(), {"num_draft": 7}, max_lanes=32)
    assert window["verify_rows"] == 256
    assert "chunk_above_max" in window["refused"] and '"max_rows": 256' not in window["refused"]


@pytest.mark.parametrize(
    "receipt",
    [None, _receipt(mode="crossover"), _receipt(covered=False), {"law_id": "stock", "covered": {}}],
)
def test_only_an_installed_exact_law_is_bound(receipt):
    assert lane_exact_verify_window(receipt, {"num_draft": 3}, max_lanes=4) is None


def test_serving_refuses_after_the_draft_loop_is_selected():
    """The loop can raise the ceiling, so the bound is taken after it."""
    source = Path(__file__).resolve().parents[1].joinpath("src/mlx2/serving.py").read_text()
    loop = source.index("config = self._select_draft_loop(adapter, config)")
    bound = source.index("lane_exact_verify_window(", loop)
    assert re.search(r'raise ValueError\(\s*window\["refused"\]\s*\)', source[bound:bound + 1500])


def test_plain_sibling_rows_count_toward_the_physical_call():
    """Codex 2026-10-14 MEDIUM 5: a solo-only loop (max_width 1) reaching
    depth 9 verifies 10 rows beside one plain row at max_lanes 2: an
    11-row call.  max_rows 10 must refuse it."""
    config = {"num_draft": 3,
              "draft_loop": {"boundaries": [3, 9], "threshold": -0.4, "max_width": 1}}
    window = lane_exact_verify_window(_receipt(max_rows=10), config, max_lanes=2)
    assert window["by_width"] == {"1": 11, "2": 8}
    assert window["verify_rows"] == 11 and window["refused"] is not None
    assert lane_exact_verify_window(_receipt(max_rows=11), config, max_lanes=2)["refused"] is None


# External drafters (2026-10-15 peer-PR audit): the self-MTP bound skipped
# ``backend == "external_draft"``, so DFlash / tree routes wider than the
# window silently ran stock MLX under an "exact" receipt.

def _external(num_draft, *, tree=False, tree_width=4, budgets=None, fixed_tree=False,
              continuation_limit=None, execution="provider", cohort_limit=None,
              continuation_algorithm="parallel"):
    from types import SimpleNamespace

    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator as G

    generator = SimpleNamespace(
        num_draft=num_draft,
        dynamic_singleton_tree=tree,
        dynamic_tree_max_width=tree_width if tree else 1,
        tree_node_budget_by_lanes=budgets or {1: 15},
        draft_topology="tree15" if tree or fixed_tree else "chain",
        continuation_policy=(
            SimpleNamespace(limit=continuation_limit, verification_algorithm=continuation_algorithm)
            if continuation_limit else None
        ),
        continuation_strategy=SimpleNamespace(algorithm=None),
        target_execution=execution,
        target_tree_cohort_limit=cohort_limit or (tree_width if tree else 1),
        _auto_active_width=1,
    )
    generator._chain_draft_cap = lambda width=None: G._chain_draft_cap(generator, width)
    generator._tree_node_budget = lambda width=None: G._tree_node_budget(generator, width)
    return lambda max_lanes: G.verify_rows_by_width(generator, max_lanes)


def test_external_chain_rows_are_lanes_times_block():
    assert _external(15)(4) == {"1": 16, "2": 32, "3": 48, "4": 64}


def test_external_dynamic_tree_rows_take_the_node_budget_then_capped_chains():
    rows = _external(15, tree=True, budgets={1: 15, 2: 7, 3: 4, 4: 3})(8)
    # B1-B4 verify the width's tree; past the widest tree the chain keeps
    # the tree's 16-row budget (rows // lanes - 1 drafts).
    assert rows == {"1": 16, "2": 16, "3": 15, "4": 16,
                    "5": 15, "6": 12, "7": 14, "8": 16}


def test_a_fixed_tree15_topology_verifies_its_tree_at_every_width():
    # MLX2_DFLASH_TOPOLOGY=tree15 without the dynamic route: num_draft 3
    # would bound 4 rows a lane, but the round verifies 15 nodes + root,
    # one lane a forward under reference execution (codex round 2).
    assert _external(3, fixed_tree=True, execution="reference")(4) == {
        "1": 16, "2": 16, "3": 16, "4": 16}
    assert _external(3, fixed_tree=True, cohort_limit=2)(4) == {
        "1": 16, "2": 32, "3": 32, "4": 32}


def test_continuation_paths_share_one_forward_per_lane():
    # Up to ``limit`` paths of num_draft tokens plus the anchor, one lane.
    assert _external(7, continuation_limit=8)(2) == {"1": 64, "2": 64}


def test_longest_first_continuations_verify_one_path_a_forward():
    from mlx2.runtime.proposal_providers import LONGEST_FIRST_EXACT_PREFIX

    rows = _external(7, continuation_limit=8,
                     continuation_algorithm=LONGEST_FIRST_EXACT_PREFIX)(1)
    assert rows == {"1": 8}


def test_an_external_route_wider_than_the_window_is_refused():
    window = lane_exact_verify_window(
        _receipt(), {"num_draft": 15, "backend": "external_draft"},
        max_lanes=4, rows_by_width=_external(15)(4),
    )
    assert window["verify_rows"] == 64
    assert "external-draft route" in window["refused"]
    assert "plain lanes" not in window["refused"]
    admitted = lane_exact_verify_window(
        _receipt(), {"num_draft": 15, "backend": "external_draft"},
        max_lanes=2, rows_by_width=_external(15)(2),
    )
    assert admitted["refused"] is None and admitted["verify_rows"] == 32
    chunked = lane_exact_verify_window(
        _receipt(chunk=True), {}, max_lanes=4, rows_by_width=_external(15)(4)
    )
    assert chunked["refused"] is None


def test_serving_binds_the_external_window_after_the_generator_exists():
    source = Path(__file__).resolve().parents[1].joinpath("src/mlx2/serving.py").read_text()
    built = source.index("batch = build_batch()\n")
    tail = source[built:built + 1500]
    assert "if external_draft and lane_policy.get(\"mode\") == \"exact\":" in tail
    assert "rows_by_width=" in tail
    assert re.search(r'raise ValueError\(window\["refused"\]\)', tail)
