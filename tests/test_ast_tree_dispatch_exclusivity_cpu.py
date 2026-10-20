"""Selected tree tiles cannot escape to older ordinary/progressive verifiers.

CPU routing/recovery evidence only; no TensorFold numerical or GPU claim.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from test_ast_terminal_ordinary_cpu import _committed_digest
from test_ast_tree_cohort_flow_cpu import _cohort

from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def test_selected_tree_tiles_refuse_direct_legacy_progressive_entry():
    batch = SimpleNamespace(ast_tree_tile_rows=16)
    with pytest.raises(RuntimeError, match="legacy progressive verification"):
        ExternalDraftBatchGenerator._progressive_verify_many(batch, [], [])


def test_selected_tree_tiles_refuse_ordinary_multirow_entry_before_transaction():
    batch = SimpleNamespace(ast_tree_tile_rows=16)
    with pytest.raises(RuntimeError, match="ordinary multi-row verification"):
        ExternalDraftBatchGenerator._cohort_target_forward(
            batch, SimpleNamespace(shape=(4, 4)), None, [], site="refused"
        )


@pytest.mark.parametrize("unsupported", ["sampling", "processors", "cross_session"])
def test_unsupported_tile_request_refuses_legacy_dispatch_and_restores_state(
    monkeypatch, unsupported
):
    _draft, batch, lanes = _cohort()
    if unsupported == "sampling":
        for lane in lanes:
            lane.sampling["sampling_temp"] = 0.5
    elif unsupported == "processors":
        monkeypatch.setattr(
            batch, "_progressive_processors_supported", lambda lane: False
        )
    else:
        batch.progressive_cross_session_pack_rows = 4
        propose = batch._propose

        def alternatives(group, **kwargs):
            blocks = propose(group, **kwargs)
            for lane, block in zip(group, blocks, strict=True):
                batch._last_alternative_blocks[lane.uid] = [block, block]
                batch._last_alternative_prefix_tokens[lane.uid] = 0
            return blocks

        monkeypatch.setattr(batch, "_propose", alternatives)

    def forbidden(*args, **kwargs):
        raise AssertionError("selected AST tile reached a legacy verifier")

    for name in (
        "_progressive_verify_many",
        "_progressive_verify_shared_prefix",
        "_progressive_verify_cross_session_ast",
        "_cohort_target_forward",
    ):
        monkeypatch.setattr(batch, name, forbidden)
    before = [
        (_committed_digest(lane.cache), lane.generated, lane.rng.snapshot())
        for lane in lanes
    ]
    try:
        with pytest.raises(RuntimeError, match="no admitted target-tree dispatch"):
            batch._run_round(lanes)
        assert [
            (_committed_digest(lane.cache), lane.generated, lane.rng.snapshot())
            for lane in lanes
        ] == before
        assert not batch._open
    finally:
        batch.close()
