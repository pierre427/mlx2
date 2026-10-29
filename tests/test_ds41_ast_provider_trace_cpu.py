"""Bounded provider trace metadata only, synthetic CPU cache/model."""
import importlib.util
from pathlib import Path
import mlx.core as mx
import numpy as np
import pytest

from test_deepseek_v41_ast_target_cpu import fixture, PATHS
from mlx2.adapters.deepseek_v41_ast_target import SourceASTTargetTreeExecutor, PARENTS


def load():
    spec = importlib.util.spec_from_file_location("ast_provider_trace", Path(__file__).resolve().parents[1]/"scripts/research/ds41_ast_provider_trace.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trace_keeps_bfs_outputs_and_original_canonical_commit_paths(monkeypatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast, "metal_kernel", lambda *a, **k: (_ for _ in ()).throw(AssertionError("GPU forbidden")))
    model, _, cache = fixture()
    provider = SourceASTTargetTreeExecutor(model, cpu_reference=True)
    tokens = [3] + [v for path in PATHS for v in path]
    bfs = (0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15)
    inv = {old: new for new, old in enumerate(bfs)}
    parents = tuple(-1 if PARENTS[old] < 0 else inv[PARENTS[old]] for old in bfs)
    with load().instrument(provider, lambda: True) as report:
        logits, _, transaction = provider.forward_many(model, [[tokens[i] for i in bfs]],
            [parents], [cache], (4,), cached=True, configured_limit=1)
        assert provider.dispatch_metrics(1, transaction, cached=True).width == 1
        transaction.prepared_cache((0, 2, 5, 8))
        result = transaction.commit_paths([(0, 2, 5, 8)])[0]
        assert all(row.offset == 6 for row in result)
        row = report["tiles"][0]
        assert row["base_offset"] == 2 and row["inputs"] == [tokens[i] for i in bfs]
        assert row["canonical_inputs"] == tokens
        assert row["greedy_tokens"] == np.asarray(mx.argmax(logits, axis=-1)).reshape(-1).tolist()
        event = row["events"][-1]
        assert event["caller_paths"] == [[0, 2, 5, 8]]
        assert event["canonical_paths"] == [[0, 6, 7, 8]]
        assert event["consumed_tokens"] == [[3, 9, 10, 11]]
        assert event["expected_frontier_offsets"] == [6] and event["closed_after"]
        with pytest.raises(ValueError, match="exactlyone"):
            provider.forward(model, tokens, PARENTS, cache, (4,))
    assert "forward" not in provider.__dict__
