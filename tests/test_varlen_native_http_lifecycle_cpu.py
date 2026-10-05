"""CPU checks for exact native HTTP lifecycle retirement evidence."""

import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "research"))
from varlen_native_http_lifecycle_gate import _retirement


def _state(*, pending=0, pages=0, poisoned=False, closed=True):
    writer = NS(pending_epochs=set(range(pending)), ledger=NS(pending_count=0),
                pool=NS(allocated_count=pages), poisoned=poisoned,
                failed_arena_torn_down=False)
    backend = NS(writer=writer, read_submissions=29, terminal_successes=29)
    candidate = NS(backend=backend)
    owner = NS(fully_retired=closed, _quarantine=[], _open_branches=set(),
               _readers={})
    return owner, candidate


def test_terminal_proof_releases_exact_owner_state():
    owner, candidate = _state()
    result = _retirement(owner, candidate)
    assert result["owner_fully_retired"]
    assert result["native_read_submissions"] == result["native_terminal_successes"] == 29
    assert result["retained_pages"] == result["pending_native_epochs"] == 0


@pytest.mark.parametrize("change", ["pending", "pages", "poisoned"])
def test_terminal_proof_rejects_unreleased_or_poisoned_arena(change):
    owner, candidate = _state(**{change: 1 if change != "poisoned" else True})
    with pytest.raises(RuntimeError, match="retained state"):
        _retirement(owner, candidate)
