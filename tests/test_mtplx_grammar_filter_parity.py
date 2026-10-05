"""Regression for grammar plus top-k/top-p speculative law composition.

MTPLX v2.12.1 reported a mismatched verify-position mask. These checks
exercise mlx2's own draft and target transforms; they do not imply that the
upstream bug was present here.
"""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mlx2.runtime.hybrid_speculative import (
    _lane_mtp_draft_logprobs,
    _lane_mtp_logprobs,
    _probe_logits_processors,
)
from mlx2.runtime.sample_utils import make_transformed_logprobs


def test_position_specific_grammar_precedes_topk_topp_on_draft_and_target():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        def grammar(tokens, logits):
            # Row zero permits 1,2,3; after drafted token 1 only 2 is legal.
            allowed = (1, 2, 3) if len(tokens) == 2 else (2,)
            mask = mx.array([i in allowed for i in range(4)])
            return mx.where(mask[None, :], logits, -mx.inf)

        grammar.probe = grammar
        grammar.history_pure = True
        lane = SimpleNamespace(
            token_prefix=mx.array([0], mx.uint32), cur=1,
            logits_processors=[grammar], sampling_temp=0.8,
            logprob_transform=make_transformed_logprobs(
                0.8, top_p=0.8, top_k=2, min_p=0.0
            ),
        )
        raw = mx.array([4.0, 3.0, 2.0, 1.0])
        for drafted, expected_support in (([], {1, 2}), ([1], {2})):
            draft = _lane_mtp_draft_logprobs(lane, raw, [mx.array(x) for x in drafted])
            history = mx.array([0, 1, *drafted], mx.uint32)
            target = _lane_mtp_logprobs(
                lane, _probe_logits_processors([grammar], history, raw)
            )
            mx.eval(draft, target)
            np.testing.assert_array_equal(np.asarray(draft), np.asarray(target))
            support = set(np.flatnonzero(np.isfinite(np.asarray(draft))).tolist())
            assert support == expected_support
    finally:
        mx.set_default_device(previous)
