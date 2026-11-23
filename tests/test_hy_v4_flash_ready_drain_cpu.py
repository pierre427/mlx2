"""Hy4-flash ``ready_drain`` (opt-in B>1 lockstep, CPU).

The engine default without an AST verification tile is "one": a lane that
accepted several tokens sits out later verifies while it emits them one per
call.  ``ready_drain: "all"`` emits every queued token at once; the committed
tokens must stay ordinary greedy at B=1/2/4.
"""

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from test_hy_v4_flash_ast_multilane_cpu import PROMPTS  # noqa: E402
from test_hy_v4_flash_cpu import _drain, _greedy, _tiny_artifact, build, checkpoint, tiny_config  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    config = tiny_config()
    W = checkpoint(config)
    return config, W, build(config, W)[0]


def test_policy_parse():
    from mlx2.adapters import hy_v4_flash as A

    assert A._parse_policy({"draft_model": "embedded_dfly", "ready_drain": "all"})[1]["ready_drain"] == "all"
    assert A._parse_policy({"draft_model": "embedded_dfly"})[1]["ready_drain"] == "all"   # served default
    assert "ready_drain" not in A._parse_policy({"draft_model": "embedded_dfly", "ready_drain": None})[1]
    with pytest.raises(ValueError, match="ready_drain must be"):
        A._parse_policy({"draft_model": "embedded_dfly", "ready_drain": "some"})


@pytest.mark.parametrize("drain", ["all", "one"])
@pytest.mark.parametrize("width", [1, 2, 4])
def test_ready_drain_commits_ordinary_greedy(tmp_path, tiny, drain, width):
    from mlx2.adapters import hy_v4_flash as A

    config, W, model = tiny
    path = str(_tiny_artifact(tmp_path, config, W))
    adapter = A.HYV4FlashAdapter(path, execution_policy={"draft_model": "embedded_dfly", "num_draft": 3,
                                                         "ready_drain": drain})
    try:
        n = 12
        batch = adapter.create_external_batch(completion_batch_size=width, prefill_step_size=4)
        assert batch.ready_drain == drain
        prompts = [PROMPTS[i % len(PROMPTS)] for i in range(width)]
        uids = batch.insert(prompts, max_tokens=[n] * width, sampling_configs=[{"sampling_temp": 0}] * width)
        out = _drain(batch, uids, n)
        for uid, prompt in zip(uids, prompts):
            assert out[uid][:n] == _greedy(adapter.model, prompt, n), prompt
    finally:
        adapter.close()
