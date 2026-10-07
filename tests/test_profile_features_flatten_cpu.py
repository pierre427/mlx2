"""profile_features.flatten must keep list-valued diagnostics (e.g. execution.ple_tables).

It used to drop lists silently, so per-table PLE timing never reached the
status deltas of a profiling run (p620 intake, 2026-10-07).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from profile_features import flatten  # noqa: E402


def test_flatten_indexes_lists_and_still_skips_bools():
    status = {"execution": {"ple_tables": [{"elapsed_seconds": 1.5, "lookups": 3}], "on": True}}
    assert flatten(status) == {
        "execution.ple_tables.0.elapsed_seconds": 1.5,
        "execution.ple_tables.0.lookups": 3,
    }
