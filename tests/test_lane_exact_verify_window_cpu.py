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
