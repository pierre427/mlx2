"""DSpine backbone on the real lane draft kernel (Metal; MLX2_RUN_METAL_TESTS=1).

The lane law is row-invariant, not stock-bit-equal, so the check is that it
covers every backbone projection, leaves the code bridge on stock modules,
and gives edge laws within BF16 noise of the stock drafter.
"""
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX2_RUN_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="Metal lane kernel test",
)


@pytest.fixture(autouse=True)
def _gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


def _draft(seed, bits):
    from mlx2.adapters.dspine import DSpineConfig
    from mlx2.runtime.drafters.dspine import DSpineDraftModel
    from test_dspine_drafter_cpu import TinyTarget, VOCAB, _randomize, tiny_raw

    # That module pins the CPU device at import; this test runs on the GPU.
    mx.set_default_device(mx.gpu)
    raw = tiny_raw()
    raw.update(hidden_size=128, intermediate_size=256, head_dim=32)
    cfg = DSpineConfig.from_dict(raw)
    draft = DSpineDraftModel(cfg)
    _randomize(draft, seed)
    if bits is not None:
        nn.quantize(draft, group_size=64, bits=bits,
                    class_predicate=lambda name, m: isinstance(m, nn.Linear)
                    and not name.startswith("selected_prefix_bridge"))
    mx.random.seed(seed + 1)
    target = TinyTarget(cfg)
    target.update({
        "embed_tokens": {"weight": mx.random.normal((VOCAB, cfg.hidden_size)).astype(mx.bfloat16)},
        "lm_head": {"weight": mx.random.normal((VOCAB, cfg.hidden_size)).astype(mx.bfloat16)},
    })
    return cfg, draft, target


@pytest.mark.parametrize("bits", [None, 8, 4])
def test_lane_backbone_matches_stock_and_leaves_the_bridge(bits):
    from mlx2.runtime.drafters.dflash2 import configure_draft_kernel
    from mlx2.runtime.lane import installer

    cfg, stock, target = _draft(3, bits)
    _, lane, _ = _draft(3, bits)
    from mlx2.runtime.drafters.dflash2_kernel import draft_kernel_law

    receipt = configure_draft_kernel(lane, "lane", drafter="dspine")
    assert receipt["kernel"] == "lane" and sum(receipt["covered"].values()) > 0
    # The installed law is the one inspection binds into the drafter identity.
    assert receipt["law_id"] == draft_kernel_law("lane", drafter="dspine")
    assert "_block_topk" not in lane.__dict__
    for name, module in lane.named_modules():
        if isinstance(module, (nn.Linear, nn.QuantizedLinear)):
            on_lane = installer.installed_lane(module) is not None
            assert on_lane != name.startswith("selected_prefix_bridge"), name
    stock.bind(target)
    lane.bind(target)
    mx.random.seed(9)
    hidden = mx.random.normal((2, 11, len(cfg.target_layer_ids) * cfg.hidden_size)).astype(mx.bfloat16)
    laws = []
    for draft in (stock, lane):
        state = draft.start_tree([5, 7], hidden, draft.batch_caches([draft.make_cache(), draft.make_cache()]), 15)
        mx.eval(*state["pending"])
        laws.append([np.asarray(x.astype(mx.float32)) for x in state["pending"]])
        trees = draft.finish_tree(state)
        assert all(len(tokens) == 15 for tokens, _p, _i in trees)
    (cand_s, law_s), (cand_l, law_l) = laws
    # Same candidates on most rows; where they agree the laws agree closely.
    same = (np.sort(cand_s, -1) == np.sort(cand_l, -1)).all(-1)
    assert same.mean() >= 0.8
    np.testing.assert_allclose(np.exp(law_l).sum(-1), 1.0, atol=1e-4)
    # A row's law is indexed by its own and its predecessor row's candidates.
    exact = (cand_s == cand_l).all(-1)
    checked = 0
    for b, i in np.argwhere(exact):
        if i == 0 or exact[b, i - 1]:
            np.testing.assert_allclose(law_l[b, i], law_s[b, i], atol=0.15)
            checked += 1
    assert checked >= 0.5 * exact.size
