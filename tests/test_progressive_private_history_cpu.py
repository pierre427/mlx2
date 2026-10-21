"""P9: a processor-free progressive lane pins its committed history.

``list(lane.history)`` copied the whole token history per lane per round for
a value only read for its length and private suffix.  The lane state after
promotion is unchanged.
"""

import mlx.core as mx
import pytest

from mlx2.runtime import external_speculative
from mlx2.runtime.ast_private_state import PrivateHistory
from tests.test_external_dflash2_cpu import (
    _assert_lane_target_state_equal,
    _run_forced_progressive_round,
)


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.mark.parametrize("reject_at", [None, 1])
def test_progressive_round_pins_history_without_copying(monkeypatch, reject_at):
    created = []

    class Counted(PrivateHistory):
        __slots__ = ()

        def __init__(self, prefix):
            created.append(prefix)
            super().__init__(prefix)

    monkeypatch.setattr(external_speculative, "PrivateHistory", Counted)
    _fixed_batch, fixed = _run_forced_progressive_round(None, reject_at)
    _batch, progressive = _run_forced_progressive_round(2, reject_at)
    _assert_lane_target_state_equal(progressive, fixed)
    # The private view aliases the lane's own committed history list.
    assert created and all(prefix is progressive.history for prefix in created)
