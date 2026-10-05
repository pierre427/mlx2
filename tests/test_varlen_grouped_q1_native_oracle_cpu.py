"""CPU preflight for the source layout and terminal identity oracle."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research"))
from varlen_grouped_q1_native_oracle import _source_planes, _terminal_write


def test_grouped_oracle_sources_have_two_distinct_exact_strided_rows():
    with mx.stream(mx.cpu):
        keys, values, expected_keys, expected_values = _source_planes()
        np.testing.assert_array_equal(np.asarray(keys), expected_keys)
        np.testing.assert_array_equal(np.asarray(values), expected_values)
    assert keys.shape == (2, 8, 128)
    assert not np.array_equal(expected_keys[0], expected_keys[1])
    assert not np.array_equal(expected_keys, expected_values)


def test_grouped_oracle_terminal_requires_exact_successful_ticket():
    ticket = object()
    owner = SimpleNamespace(poll_completions=lambda: [
        SimpleNamespace(ticket=ticket, succeeded=True)])
    _terminal_write(owner, ticket)
    with pytest.raises(AssertionError):
        _terminal_write(owner, object())
    owner.poll_completions = lambda: [SimpleNamespace(ticket=ticket, succeeded=False)]
    with pytest.raises(AssertionError):
        _terminal_write(owner, ticket)
