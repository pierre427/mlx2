"""Row-order oracle for the shared-layer candidate; no checkpoint or GPU run."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mlx2.adapters.glm53_layer_tree import mlp_rows
from mlx2.runtime.models.glm53_gguf import Glm53GGUFModel


class Weights:
    def __init__(self):
        rng = np.random.default_rng(199)
        self.values = {}
        self.tensors = {}
        self.routing = []
        for prefix in ["blk.0", "blk.3"]:
            suffix = "" if prefix == "blk.0" else "_shexp"
            for name, shape in [
                ("ffn_gate" + suffix + ".weight", (7, 4)),
                ("ffn_up" + suffix + ".weight", (7, 4)),
                ("ffn_down" + suffix + ".weight", (4, 7)),
            ]:
                key = prefix + "." + name
                self.values[key] = mx.array(rng.normal(size=shape).astype(np.float32))
                self.tensors[key] = SimpleNamespace(ggml_type=0)
        self.values["blk.3.ffn_gate_inp.weight"] = mx.array(
            rng.normal(size=(9, 4)).astype(np.float32)
        )
        self.values["blk.3.exp_probs_b.bias"] = mx.array(
            rng.normal(size=(9,)).astype(np.float32)
        )

    def array(self, name):
        return self.values[name]

    def linear(self, name):
        return lambda x: x @ self.values[name].T

    def moe(self, x, ids, scores, prefix):
        self.routing.append((np.array(ids), np.array(scores)))
        # Make slot order observable, rather than a commutative expert sum.
        out = mx.zeros_like(x)
        for slot in range(8):
            out = (
                out * 0.9
                + x * (ids[:, slot : slot + 1] + 1) * scores[:, slot : slot + 1]
            )
        return out


def test_each_rows_own_routing_and_shared_order_are_preserved():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        weights = Weights()
        model = object.__new__(Glm53GGUFModel)
        model.weights = weights
        x = mx.array(
            np.random.default_rng(77).normal(size=(1, 5, 4)).astype(np.float32)
        )
        for index in (0, 3):
            prefix = f"blk.{index}"
            weights.routing = []
            expected = mx.concatenate(
                [model._mlp(x[:, i : i + 1], prefix, index) for i in range(5)], axis=1
            )
            expected_routes = list(weights.routing)
            weights.routing = []
            actual = mlp_rows(model, x, prefix, index)
            np.testing.assert_array_equal(np.array(actual), np.array(expected))
            if index == 3:
                ids, scores = weights.routing[0]
                np.testing.assert_array_equal(
                    ids, np.concatenate([r[0] for r in expected_routes])
                )
                np.testing.assert_array_equal(
                    scores, np.concatenate([r[1] for r in expected_routes])
                )
    finally:
        mx.set_default_device(previous)
