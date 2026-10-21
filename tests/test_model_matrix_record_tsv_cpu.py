"""CPU tests for the model-matrix TSV recorder's route-to-row mapping."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "model_matrix"


def _load():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("mm_record_tsv", SCRIPTS / "record_tsv.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


record_tsv = _load()


@pytest.mark.parametrize(
    ("default_kind", "row"),
    [
        ("prompt_lookup", ("feature", "prompt_lookup")),
        ("native_mtp", ("feature", "native_mtp")),
        ("external_draft", ("feature", "external_draft")),
        ("ordinary", ("server_cli", "--ordinary")),
    ],
)
def test_no_flag_cell_is_recorded_under_the_route_its_default_serves(default_kind, row):
    assert record_tsv.flag_for("no_flag_default", default_kind) == row
    assert record_tsv.flag_for("no_flag_default-r2", default_kind) == row


def test_no_flag_cell_keeps_the_standard_decoder_mapping_by_default():
    assert record_tsv.flag_for("no_flag_default") == ("feature", "prompt_lookup")


def test_no_flag_cell_with_unknown_default_is_refused():
    with pytest.raises(ValueError):
        record_tsv.flag_for("no_flag_default", "speculative_magic")


@pytest.mark.parametrize(
    ("route", "row"),
    [
        ("ordinary", ("server_cli", "--ordinary")),
        ("pld_advanced-fix-abc", ("server_cli", "--prompt-lookup")),
        ("native_mtp", ("server_cli", "--native-mtp")),
        ("ast_auto", ("server_cli", "--ast")),
        ("external_dflash2-x", ("server_cli", "--external-draft")),
    ],
)
def test_explicit_routes_ignore_the_default_kind(route, row):
    assert record_tsv.flag_for(route, "native_mtp") == row


def test_served_route_decides_the_no_flag_row_over_the_inventory_expectation():
    cell = {"route": "no_flag_default-r2", "expected_route_kind": "ordinary",
            "routes_seen": ["prompt_lookup"]}
    assert record_tsv.flag_for(cell["route"], record_tsv.default_route_kind(cell)) == (
        "feature", "prompt_lookup")
    unloaded = {"route": "no_flag_default", "expected_route_kind": "native_mtp"}
    assert record_tsv.default_route_kind(unloaded) == "native_mtp"


def test_a_skip_superseded_by_a_later_run_is_dropped():
    skipped = ("a", {"checkpoint": "m@x", "route": "pld_advanced", "verdict": "skipped",
                     "finished_at": "2026-10-21T13:00:00Z"})
    rerun = ("b", {"checkpoint": "m@x", "route": "pld_advanced", "verdict": "fail",
                   "finished_at": "2026-10-21T15:00:00Z"})
    lone = ("c", {"checkpoint": "m@x", "route": "ast_auto", "verdict": "skipped",
                  "finished_at": "2026-10-21T13:00:00Z"})
    kept = record_tsv.drop_superseded_skips([skipped, rerun, lone])
    assert [path for path, _ in kept] == ["b", "c"]
