import copy
import importlib
import sys
import types

import numpy as np


class FakeMX:
    @staticmethod
    def array(value):
        return np.asarray(value)

    @staticmethod
    def eval(*_values):
        return None

    @staticmethod
    def concatenate(values, axis=0):
        return np.concatenate(values, axis=axis)


class Transaction:
    def __init__(self, caches, lengths):
        self.caches = caches
        self.lengths = lengths
        self.closed = False

    def commit(self, accepted_lengths):
        assert not self.closed and len(accepted_lengths) == len(self.caches)
        self.closed = True
        for cache, accepted, limit in zip(
            self.caches, accepted_lengths, self.lengths
        ):
            assert 1 <= accepted <= limit
            cache["offset"] += accepted
        return self.caches

    def abort(self):
        self.closed = True


class Owner:
    def __init__(self, caches):
        self.caches = caches

    def begin(self, lengths):
        return Transaction(self.caches, lengths)


class Model:
    class Args:
        vocab_size = 32

    args = Args()

    def __init__(self, target):
        self.target = tuple(target)
        self.calls = []

    def forward_with_taps(self, inputs, caches, _capture_layers):
        self.calls.append(
            {
                "shape": tuple(inputs.shape),
                "offsets": tuple(cache["offset"] for cache in caches),
            }
        )
        rows, span = inputs.shape
        logits = np.zeros((rows, span, self.args.vocab_size), dtype=np.float32)
        for row, cache in enumerate(caches):
            for position in range(span):
                logits[row, position, self.target[cache["offset"] + position]] = 1
        return logits, inputs[..., None].astype(np.float32)


def _module(monkeypatch):
    # Import the production host/cache controller without constructing Metal.
    mlx = types.ModuleType("mlx")
    core = types.ModuleType("mlx.core")
    core.array = np.ndarray
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    module = importlib.import_module("mlx2.runtime.continuation_verification")
    monkeypatch.setattr(
        module,
        "snapshot_recovery_descriptors",
        lambda cache: (copy.deepcopy(cache),),
    )
    monkeypatch.setattr(
        module,
        "restore_recovery_descriptors",
        lambda cache: (copy.deepcopy(cache),),
    )
    return module


def test_executor_reuses_exact_cache_boundary_and_does_not_recompute_prefix(
    monkeypatch,
):
    module = _module(monkeypatch)
    target = (1, 2, 3, 4, 9, 6, 7, 8)
    paths = (
        (1, 2, 3, 4, 5, 6, 7),
        (0, 2, 3, 4, 9, 6),
        (1, 2, 0, 4, 9, 6),
        (1, 2, 3, 4, 9, 6),
    )
    model = Model(target)
    cache = {"offset": 0}
    outcome, hidden, transaction = module.prepare_longest_first_continuations(
        model,
        FakeMX,
        cache,
        19,
        paths,
        (),
        Owner,
        lambda row, _prefix: int(np.argmax(row)),
        maximum=8,
        max_depth=7,
    )

    assert outcome.emitted == target[:7]
    assert outcome.accepted == 6
    assert outcome.input_lengths == (8, 2)
    assert outcome.target_rows == 10 and outcome.launches == 2
    assert outcome.pruned_siblings == 2
    assert outcome.shared_prefix_reused_tokens == 5
    assert model.calls == [
        {"shape": (1, 8), "offsets": (0,)},
        {"shape": (1, 2), "offsets": (5,)},
    ]
    assert hidden.shape == (1, 7, 1)
    assert cache == {"offset": 0}
    assert transaction.commit([7]) == [{"offset": 7}]


def test_executor_aborts_private_transaction_when_sampling_fails(monkeypatch):
    module = _module(monkeypatch)
    transactions = []

    class TrackingOwner(Owner):
        def begin(self, lengths):
            transaction = super().begin(lengths)
            transactions.append(transaction)
            return transaction

    try:
        module.prepare_longest_first_continuations(
            Model((1, 2, 3)),
            FakeMX,
            {"offset": 0},
            19,
            ((1, 2),),
            (),
            TrackingOwner,
            lambda _row, _prefix: (_ for _ in ()).throw(RuntimeError("draw failed")),
            maximum=3,
            max_depth=2,
        )
    except RuntimeError as error:
        assert str(error) == "draw failed"
    else:
        raise AssertionError("expected sampling failure")
    assert len(transactions) == 1 and transactions[0].closed
