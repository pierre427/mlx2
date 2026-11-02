"""Synthetic exact hybrid AST isolation and publication contracts."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.glm53_ast_target import Glm53TargetTreeExecutor, stamp
from mlx2.runtime.models.glm53_gguf import Glm53KDACache, Glm53MLACache


class Toy:
    identity = "synthetic-glm53-tree"
    args = SimpleNamespace(vocab_size=16)

    def make_cache(self):
        return [Glm53KDACache(self.identity), Glm53MLACache()]

    def forward_with_taps(self, inputs, cache, layers):
        token = inputs.astype(mx.float32).reshape(1, 1, 1, 1)
        c = cache[0]
        old = c.recurrent if c.recurrent is not None else mx.zeros((1, 64, 128, 128))
        c.recurrent = old + token
        c.conv = mx.broadcast_to(token.reshape(1, 1, 1), (1, 3, 24576))
        c.offset += 1
        for kv in cache[1].caches:
            kv.update_and_fetch(token, mx.zeros((1, 1, 1, 0)))
        scalar = c.recurrent[0, 0, 0, 0]
        return mx.broadcast_to(scalar, (1, 1, 16)), mx.broadcast_to(scalar, (1, 1, 4))


def test_exact_private_tree_commit_abort_and_stale_frontier():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        model = Toy()
        live = model.make_cache()
        before = stamp(live)
        provider = Glm53TargetTreeExecutor(model)
        tokens = [1, 2, 4, 3, 5]
        parents = [-1, 0, 0, 1, 2]
        logits, _features, transaction = provider.forward(
            model, tokens, parents, live, [0]
        )
        assert stamp(live) == before
        np.testing.assert_array_equal(np.array(logits)[0, :, 0], [1, 3, 5, 6, 10])
        for path in ([0], [0, 1, 3], [0, 2, 4]):
            expected = model.make_cache()
            for index in path:
                model.forward_with_taps(mx.array([[tokens[index]]]), expected, [0])
            prepared = transaction.prepared_cache(path)
            assert prepared[0].offset == len(path)
            np.testing.assert_array_equal(
                np.array(prepared[0].recurrent), np.array(expected[0].recurrent)
            )
            for a, b in zip(prepared[1].caches, expected[1].caches):
                np.testing.assert_array_equal(
                    np.array(a.state[0]), np.array(b.state[0])
                )
        with pytest.raises(ValueError, match="root-connected"):
            transaction.prepared_cache([0, 1, 4])
        result = transaction.commit_paths([[0, 2, 4]])[0]
        assert transaction.closed and result[0].offset == 3 and stamp(live) == before
        _, _, aborted = provider.forward(model, [1, 2], [-1, 0], live, [0])
        aborted.abort()
        assert stamp(live) == before
        _, _, stale = provider.forward(model, [1, 2], [-1, 0], live, [0])
        live[0].offset = 1
        with pytest.raises(ValueError, match="frontier changed"):
            stale.prepared_cache([0])
        stale.abort()
    finally:
        mx.set_default_device(previous)
