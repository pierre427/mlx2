# Mined from unified 1e2bc604, MIT; see docs/PROVENANCE.md.
"""CPU-only contracts for cached-prefix known-tail PLE prefetch."""

import unittest

from mlx2.runtime import round_levers
from mlx2.runtime.generate import _prefetch_known_mtp_tail


class _Model:
    def __init__(self, result=3, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def ple_prefetch_verify(self, previous, tokens):
        self.calls.append((previous, tokens))
        if self.error is not None:
            raise self.error
        return self.result


class TestKnownTailPLEPrefetch(unittest.TestCase):
    def setUp(self):
        round_levers.reset_counters()

    def test_default_off_does_nothing(self):
        model = _Model()
        self.assertEqual(_prefetch_known_mtp_tail(model, [1, 2], [3], {}), 0)
        self.assertEqual(model.calls, [])
        self.assertEqual(round_levers.counters()["ple_tail_prefetch_requests"], 0)

    def test_submits_complete_known_tail(self):
        model = _Model(result=7)
        result = _prefetch_known_mtp_tail(
            model,
            [1, 2, 3],
            [4, 5, 6],
            {"prefetch_known_tail_ple": True},
        )
        self.assertEqual(result, 7)
        self.assertEqual(model.calls, [([1, 2, 3], [4, 5, 6])])
        stats = round_levers.counters()
        self.assertEqual(stats["ple_tail_prefetch_requests"], 1)
        self.assertEqual(stats["ple_tail_prefetch_tables"], 7)
        self.assertEqual(stats["ple_tail_prefetch_declined"], 0)
        self.assertEqual(stats["ple_tail_prefetch_failures"], 0)

    def test_unsupported_and_failure_fall_back(self):
        config = {"prefetch_known_tail_ple": True}
        self.assertEqual(_prefetch_known_mtp_tail(object(), [1], [2], config), 0)
        self.assertEqual(
            _prefetch_known_mtp_tail(
                _Model(error=RuntimeError("synthetic")), [1], [2], config
            ),
            0,
        )
        stats = round_levers.counters()
        self.assertEqual(stats["ple_tail_prefetch_declined"], 1)
        self.assertEqual(stats["ple_tail_prefetch_failures"], 1)


if __name__ == "__main__":
    unittest.main()
