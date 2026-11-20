"""Speculative target laws filter in the same dtype as plain sampling.

Rapid-MLX #5871 shape: an MTP verifier that built its top-p mask on bf16
logprobs while plain serving masked in fp32 sampled from a different
distribution at temperature > 0.  In mlx2 the batch generator promotes logits
to float32 before normalising (``generate.py``) and the shared draft/target
transform (``make_transformed_logprobs``) does the same, so the two kept sets
and laws agree on bf16 rows with knife-edge ties.  This pins it.
"""

import pytest

mx = pytest.importorskip("mlx.core")

from mlx2.runtime.sample_utils import (  # noqa: E402
    apply_min_p,
    apply_top_k,
    apply_top_p,
    make_transformed_logprobs,
)


def _plain_law(logits, temp, top_p=0.0, top_k=0, min_p=0.0):
    """The law ``make_sampler`` draws from on the batch generator's rows."""
    logits = logits.astype(mx.float32)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    if 0 < top_p < 1:
        lp = apply_top_p(lp, top_p)
    if min_p:
        lp = apply_min_p(lp, min_p)
    if top_k:
        lp = apply_top_k(lp, top_k)
    scaled = lp * (1 / temp)
    return scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)


def _bf16_rows(seed):
    key = mx.random.key(seed)
    base = mx.random.normal((4, 4096), key=key) * 3.0
    # bf16 rounding creates many exact ties near the nucleus boundary.
    return base.astype(mx.bfloat16)


@pytest.mark.parametrize("params", [
    {"top_p": 0.9}, {"top_p": 0.5}, {"top_p": 0.95, "top_k": 40},
    {"min_p": 0.05}, {"top_p": 0.8, "min_p": 0.02, "top_k": 100},
])
@pytest.mark.parametrize("temp", [0.6, 1.0])
def test_spec_transform_matches_plain_sampler_on_bf16_rows(params, temp):
    for seed in range(3):
        logits = _bf16_rows(seed)
        plain = _plain_law(logits, temp, **params)
        spec = make_transformed_logprobs(temp, **params)(logits)
        mx.eval(plain, spec)
        kept_plain = mx.isfinite(plain)
        kept_spec = mx.isfinite(spec)
        assert mx.array_equal(kept_plain, kept_spec).item()
        diff = mx.abs(mx.where(kept_plain, plain - spec, 0.0)).max().item()
        assert diff < 1e-5
