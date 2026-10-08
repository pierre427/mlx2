from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/research/probe_dflash_split_shape.py"


def load_module():
    spec = importlib.util.spec_from_file_location("probe_dflash_split_shape", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("count", "tile", "widths"),
    [
        (3, 1, (1, 1, 2)),
        (7, 2, (2, 2, 2, 2)),
        (15, 3, (3, 3, 3, 3, 4)),
        (15, 4, (4, 4, 4, 4)),
        (15, 8, (8, 8)),
        (15, 15, (16,)),
    ],
)
def test_progressive_partition_covers_fixed_inputs_once(count, tile, widths):
    module = load_module()
    proposals = tuple(range(1, count + 1))
    chunks = module.progressive_input_chunks(0, proposals, tile)
    assert tuple(map(len, chunks)) == widths
    assert tuple(token for chunk in chunks for token in chunk) == (0, *proposals)


@pytest.mark.parametrize(
    ("anchor", "proposals", "tile"),
    [(-1, (1,), 1), (0, (), 1), (0, (1,), 0), (0, (1,), 2), (True, (1,), 1)],
)
def test_progressive_partition_rejects_invalid_geometry(anchor, proposals, tile):
    module = load_module()
    with pytest.raises(ValueError):
        module.progressive_input_chunks(anchor, proposals, tile)


def test_tiny_row_exact_target_closes_fixed_and_progressive_shapes(tmp_path):
    module = load_module()
    output = tmp_path / "row-exact.json"
    assert (
        module.main(
            [
                "--tiny",
                "--target-verify-row-exact",
                "--verify-proposal",
                "--num-draft",
                "7",
                "--tiles",
                "2,3",
                "--prompts",
                "2",
                "--out",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text())
    assert payload["verdict"] == "strict_equal"
    assert all(
        comparison["strict_equal"]
        for result in payload["results"]
        for comparison in result["comparisons_vs_ordinary_rows"].values()
    )
    assert all(
        comparison["strict_equal"]
        for result in payload["results"]
        for comparison in result["proposal_verification"][
            "comparisons_vs_fixed"
        ].values()
    )
