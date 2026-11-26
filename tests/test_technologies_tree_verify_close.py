"""What a refused commit path does to a packed tree transaction (CPU).

Ling, Inkling (and Hybrid, attention, Flash-Next) leave the transaction open
so the scheduler can publish a valid path or abort; Nemotron closes it.
Both are kept as ``TreeLaneTransaction.close_on_refused_path``.
"""

import pytest

from mlx2.runtime.inkling_tree_executor import InklingTreeTransaction
from mlx2.runtime.ling_tree_executor import LingTreeTransaction
from mlx2.runtime.nemotron_tree_executor import NemotronTreeTransaction


class _Tree:
    live = ()

    def validate_path(self, path):
        raise ValueError("bad path")

    def prepare_commit(self, path):  # pragma: no cover - never reached
        raise AssertionError("prepared after a refused path")

    def release(self):
        self.released = True


@pytest.mark.parametrize("cls,closes", [
    (LingTreeTransaction, False), (InklingTreeTransaction, False),
    (NemotronTreeTransaction, True),
])
def test_refused_path_keeps_each_executors_contract(cls, closes):
    tree = _Tree()
    transaction = cls([tree], "token")
    with pytest.raises(ValueError, match="bad path"):
        transaction.commit_paths([[0, 5]])
    assert transaction.closed is closes
    assert getattr(tree, "released", False) is closes
    transaction.abort()
    assert transaction.closed
