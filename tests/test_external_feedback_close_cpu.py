"""Optional trainer failure cannot skip target adapter resource teardown."""

import pytest

from mlx2.adapters.external_draft_policy import ExternalDraftAdapterMixin


class Parent:
    def close(self):
        self.parent_close_calls += 1


class Adapter(ExternalDraftAdapterMixin, Parent):
    def __init__(self, manager=None):
        self.parent_close_calls = 0
        self._external_feedback_manager = manager


def test_worker_receipt_failure_still_closes_target_resources():
    class FailedFeedback:
        def close(self):
            raise OSError("worker receipt unavailable")

    adapter = Adapter(FailedFeedback())
    with pytest.raises(OSError, match="worker receipt unavailable"):
        adapter.close()
    assert adapter.parent_close_calls == 1


def test_absent_or_successful_feedback_closes_parent_once():
    class Feedback:
        closed = False

        def close(self):
            self.closed = True

    manager = Feedback()
    for value in (None, manager):
        adapter = Adapter(value)
        adapter.close()
        assert adapter.parent_close_calls == 1
    assert manager.closed
