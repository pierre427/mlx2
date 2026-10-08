from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from mlx2.runtime.speculative_sampling import RequestRNG

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/research/progressive_dflash_verify.py"


def load_module():
    spec = importlib.util.spec_from_file_location("progressive_dflash_verify", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class ToyTransaction:
    base: tuple[int, ...]
    inputs: tuple[int, ...]
    closed: bool = False

    def commit(self, accepted_lengths):
        assert not self.closed and len(accepted_lengths) == 1
        used = accepted_lengths[0]
        self.closed = True
        return [self.base + self.inputs[:used]]

    def abort(self):
        self.closed = True


def one_hot(token, vocab=32):
    law = np.zeros(vocab, dtype=np.float64)
    law[token] = 1.0
    return law


def target_callback(module, initial_history, expected, *, reject_at=None):
    def prepare(cache, inputs, history):
        assert history[: len(initial_history)] == initial_history
        start = len(history) - len(initial_history)
        laws = []
        for row in range(len(inputs)):
            index = start + row
            token = expected[index]
            if reject_at == index:
                token = (token + 9) % 32
            laws.append(one_hot(token))
        features = np.asarray(inputs, dtype=np.float32).reshape(1, -1, 1)
        return module.PreparedStage(
            tuple(laws), features, ToyTransaction(tuple(cache), tuple(inputs))
        )

    return prepare


def flatten(parts):
    return np.concatenate(parts, axis=1)


@pytest.mark.parametrize("tile", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("reject_at", [None, 0, 2, 5])
def test_progressive_tiles_match_fixed_tokens_state_taps_and_rng(tile, reject_at):
    module = load_module()
    history = (11, 12)
    anchor = 13
    tokens = (1, 2, 3, 4, 5, 6, 7)
    bonus = 8
    laws = tuple(one_hot(token) for token in tokens)
    prepare = target_callback(
        module, history, (*tokens, bonus), reject_at=reject_at
    )
    start = RequestRNG(919).snapshot()
    fixed = module.fixed_verify(
        tokens,
        laws,
        cache=(),
        anchor=anchor,
        history=history,
        rng=RequestRNG(state=start),
        prepare_stage=prepare,
    )
    progressive = module.progressive_verify(
        tokens,
        laws,
        cache=(),
        anchor=anchor,
        history=history,
        rng=RequestRNG(state=start),
        verification_tile=tile,
        prepare_stage=prepare,
    )
    assert progressive.emitted == fixed.emitted
    assert progressive.accepted == fixed.accepted
    assert progressive.committed_inputs == fixed.committed_inputs
    assert progressive.cache == fixed.cache
    assert progressive.rng_state == fixed.rng_state
    np.testing.assert_array_equal(
        flatten(progressive.feature_parts), flatten(fixed.feature_parts)
    )
    assert progressive.target_rows <= fixed.target_rows


def test_intermediate_full_tile_does_not_draw_a_bonus():
    module = load_module()
    tokens = (1, 2, 3, 4)
    laws = tuple(one_hot(token) for token in tokens)
    rng = RequestRNG(7)
    before = rng.draws
    outcome = module.verify_proposal_tile(
        tokens[:2], laws[:2], laws[:2], rng, final=False
    )
    assert outcome.emitted == tokens[:2]
    assert outcome.accepted == 2 and not outcome.rejected
    assert rng.draws - before == 2


def test_invalid_future_proposal_law_fails_before_target_or_rng_work():
    module = load_module()
    tokens = (1, 2, 3)
    laws = [one_hot(token) for token in tokens]
    laws[-1][:] = 0
    calls = []
    rng = RequestRNG(7)
    before = rng.snapshot()
    with pytest.raises(ValueError, match="probability distribution"):
        module.progressive_verify(
            tokens,
            laws,
            cache=(),
            anchor=9,
            history=(8,),
            rng=rng,
            verification_tile=2,
            prepare_stage=lambda *_: calls.append(True),
        )
    assert not calls and rng.snapshot() == before


def test_cli_declares_research_only_scope(capsys):
    module = load_module()
    assert module.main([]) == 0
    payload = capsys.readouterr().out
    assert '"apcv2_publication": false' in payload
    assert '"implemented": false' in payload
    assert '"selected": false' in payload
