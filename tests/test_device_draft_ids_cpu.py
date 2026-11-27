"""Exercise the embedding boundary before lazy proposal packets can be read."""
from dataclasses import replace
from functools import wraps

import mlx.core as mx
import pytest
from test_device_commit_raon_cpu import _decode as decode_raon
from test_device_pipeline_raon_cpu import PIPE
from test_device_round_dflash2_cpu import (
    BOTH, PROC, _decode as decode_chain, _decode_processed, _presence,
)
from test_device_round_tree_cpu import DECIDE, PROMPT, _assert_rounds_equal, _decode as decode_tree
from test_nemotron_native_mtp_external_cpu import aligned

from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel
from mlx2.runtime.drafters.native_mtp import NativeMTPDraftModel
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.proposal_providers import PointMassLaw
from mlx2.runtime.speculative_device_round import DECLINE_REASONS
from mlx2.runtime.speculative_device_tree import DECLINES, DeviceTree


def _check_embedding(owner, inputs):
    vocab = owner._target_args.vocab_size
    for token in inputs.reshape(-1).tolist():
        assert 0 <= token < vocab, f"target input {token} outside embedding size {vocab}"


def _inject_chain(monkeypatch, token):
    forward = ExternalDraftBatchGenerator._cohort_target_forward
    draft = DFlash2DraftModel.draft_lazy
    injected = []

    def checked(owner, inputs, *args, **kwargs):
        _check_embedding(owner, inputs)
        logits, features = forward(owner, inputs, *args, **kwargs)
        if token == owner._target_args.vocab_size and kwargs["site"] == "standard_verify":
            # Padded heads can expose a logit column beyond the input table.
            pad = mx.full((*logits.shape[:-1], 1), -float("inf"), dtype=logits.dtype)
            logits = mx.concatenate([logits, pad], axis=-1)
        return logits, features

    def invalid_once(self, *args, **kwargs):
        originals = draft(self, *args, **kwargs)
        if injected:
            return originals
        injected.append(token)
        return mx.full(originals.shape, token, dtype=mx.int32)

    monkeypatch.setattr(ExternalDraftBatchGenerator, "_cohort_target_forward", checked)
    monkeypatch.setattr(DFlash2DraftModel, "draft_lazy", invalid_once)
    return injected


@pytest.mark.parametrize("policy", [{"lazy_draft": True}, BOTH], ids=["host-decide", "device-decide"])
@pytest.mark.parametrize("token", [-1, 32])
def test_chain_refuses_original_invalid_draft_after_safe_forward(monkeypatch, policy, token):
    injected = _inject_chain(monkeypatch, token)
    with pytest.raises(ValueError, match="outside the vocabulary"):
        decode_chain(steps=8, device_round=policy)
    assert injected == [token]


def test_pipeline_host_redo_sentinel_stays_out_of_embedding(monkeypatch):
    expected = decode_raon(PIPE, steps=16)
    injected = _inject_chain(monkeypatch, -2)
    tokens, state, _, stats = decode_raon(PIPE, steps=16)
    assert injected == [-2]
    assert (tokens, state) == expected[:2]
    assert stats[DECLINE_REASONS]["host_redo"] == 1
    assert stats["external_device_round_pipeline_followers"] > 0


def test_tree_restores_original_out_of_range_draft_after_safe_forward(monkeypatch):
    monkeypatch.setenv("MLX2_DFLASH_TOPOLOGY", "tree15")
    monkeypatch.setenv("MLX2_AST_CONTROLLER_MODE", "deterministic")
    monkeypatch.setenv("MLX2_AST_CONTROLLER_COST", '{"a_ms":5,"b_ms":0.5,"ordinary_ms":4}')
    host = decode_tree(aligned, PROMPT, steps=8)
    forward = DeviceTree._reference_forward
    draft = NativeMTPDraftModel.draft_tree_lazy
    injected = []

    def checked(owner, lane, inputs, geo):
        _check_embedding(owner, inputs)
        return forward(owner, lane, inputs, geo)

    def invalid_once(self, *args, **kwargs):
        proposal = draft(self, *args, **kwargs)
        if injected:
            return proposal
        injected.append(12345)
        return replace(proposal, tokens=mx.full(proposal.tokens.shape, 12345, dtype=mx.int32))

    monkeypatch.setattr(DeviceTree, "_reference_forward", staticmethod(checked))
    monkeypatch.setattr(NativeMTPDraftModel, "draft_tree_lazy", invalid_once)
    device = decode_tree(aligned, PROMPT, steps=8, device_round=DECIDE)
    assert injected == [12345]
    assert device[2][DECLINES]["draft_outside_vocabulary"] == 1
    assert device[0] == host[0]
    _assert_rounds_equal(host[1], device[1])


def test_processors_get_bounded_history_until_original_packet_is_read(monkeypatch):
    def make(prompt):
        (processor,) = _presence(prompt)

        @wraps(processor)
        def checked(tokens, logits):
            assert all(0 <= t < logits.shape[-1] for t in tokens.tolist())
            return processor(tokens, logits)

        return [checked]

    _inject_chain(monkeypatch, 12345)
    with pytest.raises(ValueError, match="outside the vocabulary"):
        _decode_processed(steps=8, make=make, device_round=PROC)


def test_pipeline_refuses_unembeddable_last_after_safe_follower_dispatch(monkeypatch):
    forward = ExternalDraftBatchGenerator._cohort_target_forward
    draft = DFlash2DraftModel.draft_lazy
    anchors = []

    def checked(owner, inputs, *args, **kwargs):
        _check_embedding(owner, inputs)
        logits, features = forward(owner, inputs, *args, **kwargs)
        if kwargs["site"] == "standard_verify":
            pad = mx.full((*logits.shape[:-1], 1), 1e6, dtype=logits.dtype)
            logits = mx.concatenate([logits, pad], axis=-1)
        return logits, features

    def checked_draft(self, anchor, *args, **kwargs):
        token = int(anchor.item())
        anchors.append(token)
        assert 0 <= token < self.config.vocab_size
        return draft(self, anchor, *args, **kwargs)

    monkeypatch.setattr(ExternalDraftBatchGenerator, "_cohort_target_forward", checked)
    monkeypatch.setattr(DFlash2DraftModel, "draft_lazy", checked_draft)
    with pytest.raises(ValueError, match="outside the vocabulary"):
        decode_raon(PIPE, steps=16)
    # The leading and following draft were both dispatched before the refusal.
    assert len(anchors) == 2 and all(0 <= token < 97 for token in anchors)


@pytest.mark.parametrize("route", ["lazy_draft", "decide", "tree-host-decide", "tree-device-decide"])
def test_refuses_unembeddable_target_token_before_commit(monkeypatch, route):
    commit = ExternalDraftBatchGenerator._commit
    forwards = []

    def checked_commit(owner, cohort, decisions, *args, **kwargs):
        for decision in decisions:
            assert all(t < owner._target_args.vocab_size for t in decision.emitted)
        return commit(owner, cohort, decisions, *args, **kwargs)

    def pad_head(logits):
        forwards.append(int(logits.shape[-1]))
        padding = mx.full((*logits.shape[:-1], 1), 1e6, dtype=logits.dtype)
        return mx.concatenate([logits, padding], axis=-1)

    monkeypatch.setattr(ExternalDraftBatchGenerator, "_commit", checked_commit)
    if route in {"lazy_draft", "decide"}:
        forward = ExternalDraftBatchGenerator._cohort_target_forward

        def widened(owner, inputs, *args, **kwargs):
            logits, features = forward(owner, inputs, *args, **kwargs)
            if kwargs["site"] == "standard_verify":
                logits = pad_head(logits)
            return logits, features

        monkeypatch.setattr(ExternalDraftBatchGenerator, "_cohort_target_forward", widened)
        if route == "decide":
            # The public decide-only route accepts sparse point-mass proposals.
            host_draft = DFlash2DraftModel.draft_distributions

            def point_mass_draft(self, *args, **kwargs):
                tokens, _ = host_draft(self, *args, **kwargs)
                return tokens, [[PointMassLaw(t) for t in row] for row in tokens]

            monkeypatch.setattr(DFlash2DraftModel, "draft_distributions", point_mass_draft)
        with pytest.raises(ValueError, match="outside the vocabulary"):
            decode_chain(steps=8, device_round={route: True})
    else:
        monkeypatch.setenv("MLX2_DFLASH_TOPOLOGY", "tree15")
        monkeypatch.setenv("MLX2_AST_CONTROLLER_MODE", "deterministic")
        monkeypatch.setenv("MLX2_AST_CONTROLLER_COST", '{"a_ms":5,"b_ms":0.5,"ordinary_ms":4}')
        tree_forward = DeviceTree._reference_forward

        def widened_tree(owner, lane, inputs, geo):
            logits, features, transaction = tree_forward(owner, lane, inputs, geo)
            return pad_head(logits), features, transaction

        monkeypatch.setattr(DeviceTree, "_reference_forward", staticmethod(widened_tree))
        policy = {"tree_lazy": True, "tree_decide": route == "tree-device-decide"}
        with pytest.raises(ValueError, match="outside the vocabulary"):
            decode_tree(aligned, PROMPT, steps=8, device_round=policy)
    assert forwards == [32]
