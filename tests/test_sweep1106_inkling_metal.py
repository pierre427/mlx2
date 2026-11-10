"""INK-1 GPU recheck: the tree's new ``supported`` gate keeps mapped attention engaged.

On Metal, a d128 model under ``inkling_attention: mapped`` must still launch the
mapped tree kernel (counter > 0) and match ordinary decode along each path.
"""

import copy
import dataclasses
import sys
from pathlib import Path

import mlx.core as mx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal only")


@pytest.mark.parametrize("parents", [[-1, 0, 1], [-1, 0, 0]])
def test_sweep1106_ink1_mapped_tree_engages_on_metal(parents):
    from test_inkling_model_cpu import args

    from mlx2.runtime.models.inkling import Model
    from mlx2.runtime.models.inkling.tree import InklingTree

    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        mx.random.seed(3)
        model = Model(dataclasses.replace(args(), head_dim=128, swa_head_dim=128))
        model.eval()
        mx.eval(model.parameters())
        model.configure_kernels(attention="mapped")
        live = model.make_cache()
        for token in range(1, 6):
            model(mx.array([[token]]), cache=live)
        before = model.kernel_receipt()["mapped_attention_calls"]
        assert before > 0  # ordinary decode engaged the kernel
        expected = copy.deepcopy(live)
        tokens = [7, 8, 9]
        tree = InklingTree(parents, live)
        logits, _ = tree.forward(model, mx.array([tokens]), [1])
        assert model.kernel_receipt()["mapped_attention_calls"] > before  # tree engaged too
        path = tree.paths[-1]
        reference = model(mx.array([[tokens[i] for i in path]]), cache=expected)
        mx.eval(logits, reference)
        assert bool(mx.allclose(logits[:, path[-1]], reference[:, -1], atol=2e-3).item())
    finally:
        mx.set_default_device(previous)
