import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "varlen_n20_model_ragged_gate",
    ROOT / "scripts/research/varlen_n20_model_ragged_gate.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_plan_keeps_weighted_route_evidence_narrow():
    receipt = MODULE.plan("prompt_lookup")
    assert receipt["proposal_source_requested"] == "prompt_lookup"
    assert receipt["target_reference"] == "stock_merged_ordinary_dynamic_shrink"
    assert receipt["qualified"] is False
    assert receipt["route_observed_used"] is False
    assert receipt["prompt_lookup_source_observed_used"] is False
    assert receipt["performance_claim"] is False


def test_controlled_proposals_force_full_partial_and_rejection():
    rows = MODULE.controlled_proposals(
        (10, 20, 30), ((11, 12, 13), (21, 22), (31,)), 100
    )
    assert rows == ((10, 11, 12), (20, 21, 23), (30, 32, 33))
    assert rows[0][1:] == (11, 12)
    assert rows[1][1] == 21 and rows[1][2] != 22
    assert rows[2][1] != 31


def test_reference_round_uses_stock_merged_batch_geometry():
    class Cache:
        def __init__(self, rows):
            self.rows = list(rows)

        @property
        def batch_size(self):
            return len(self.rows)

        @property
        def state(self):
            return ()

    class MX:
        array = staticmethod(np.array)
        isfinite = staticmethod(np.isfinite)
        all = staticmethod(np.all)
        argmax = staticmethod(np.argmax)

        @staticmethod
        def eval(*_values):
            return None

    class Model:
        def __init__(self):
            self.widths = []
            self.model = self

        def __call__(self, inputs, cache):
            self.widths.append((inputs.shape[0], cache[0].batch_size))
            return np.stack(
                [np.array([[token, token + 1]]) for token in inputs[:, 0]]
            )

        @staticmethod
        def logits(hidden):
            return hidden

    model = Model()
    caches = [Cache((0, 2))]
    logits, sampled = MODULE.advance_reference_round(
        model, caches, (10, 30), MX
    )
    assert model.widths == [(2, 2)]
    assert logits.shape == (2, 2)
    assert sampled == (1, 1)


def test_reference_batch_size_supports_kv_offset_vectors():
    class Cache:
        batch_size = None
        offset = np.array([17, 19, 23])

    assert MODULE.reference_batch_size(Cache()) == 3


def test_reference_retirement_snapshots_rows_before_filtering():
    class Cache:
        def __init__(self, rows):
            self.rows = list(rows)

        @property
        def batch_size(self):
            return len(self.rows)

        @property
        def state(self):
            return ()

        def extract(self, row):
            return Cache((self.rows[row],))

        def filter(self, keep):
            self.rows = [self.rows[row] for row in keep]

    class MX:
        @staticmethod
        def eval(*_values):
            return None

    caches = [Cache(("lane0", "lane1", "lane2"))]
    retired = MODULE.retire_reference_lanes(caches, (0, 1, 2), (0, 1), MX)
    assert tuple(retired) == (2,)
    assert retired[2][0].rows == ["lane2"]
    assert caches[0].rows == ["lane0", "lane1"]


def test_natural_reference_follows_real_proposal_acceptance_geometry():
    class Cache:
        def __init__(self, rows):
            self.rows = [list(row) for row in rows]

        @classmethod
        def merge(cls, caches):
            return cls([cache.rows[0] for cache in caches])

        @property
        def batch_size(self):
            return len(self.rows)

        @property
        def state(self):
            return ()

        def extract(self, row):
            return Cache((self.rows[row],))

        def filter(self, keep):
            self.rows = [self.rows[row] for row in keep]

    class MX:
        array = staticmethod(np.array)
        isfinite = staticmethod(np.isfinite)
        all = staticmethod(np.all)
        argmax = staticmethod(np.argmax)

        @staticmethod
        def eval(*_values):
            return None

    class Model:
        def __init__(self):
            self.model = self

        def __call__(self, inputs, cache):
            values = inputs[:, 0]
            for row, token in enumerate(values):
                cache[0].rows[row].append(int(token))
            hidden = np.zeros((len(values), 1, 128), dtype=np.float32)
            for row, token in enumerate(values):
                hidden[row, 0, int(token) + 1] = 1
            return hidden

        @staticmethod
        def logits(hidden):
            return hidden

    references = tuple((Cache(((lane,),)),) for lane in range(3))
    proposals = ((10, 11, 12), (20, 21, 99), (30, 31))
    logits, tokens, boundaries, widths = MODULE.natural_reference(
        Model(), references, (10, 20, 30), proposals, MX)
    assert tuple(map(len, logits)) == (3, 2, 2)
    assert tokens == ((11, 12, 13), (21, 22), (31, 32))
    assert widths == (3, 3, 1)
    assert all(boundary[0].batch_size == 1 for boundary in boundaries)


@pytest.mark.parametrize(
    "anchors,targets,vocab",
    [
        ((1, 2), ((3, 4, 5), (6, 7), (8,)), 10),
        ((1, 2, 3), ((3, 4), (6, 7), (8,)), 10),
        ((1, 2, 3), ((3, 4, 5), (6, 7), (10,)), 10),
        ((1, 2, 3), ((3, 4, 5), (6, 7), (8,)), 1),
    ],
)
def test_controlled_proposals_fail_closed(anchors, targets, vocab):
    with pytest.raises(ValueError, match="depths 3,2,1"):
        MODULE.controlled_proposals(anchors, targets, vocab)
