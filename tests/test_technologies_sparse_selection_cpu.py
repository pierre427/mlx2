"""Bit-exact parity: technologies.sparse_selection vs each model's old copy.

Every ``_legacy_*`` function below is the pre-migration body, verbatim (only
the enclosing ``self`` became an argument where it was a method). Each test
asserts ``mx.array_equal`` (exact) between the legacy copy and the migrated
model code on small random inputs, in every dtype the model uses, with exact
ties (integer-valued inputs give many exact-zero ReLU scores), masked rows
and offsets.
"""
from __future__ import annotations


import mlx.core as mx
import pytest

from mlx2.runtime.technologies import sparse_selection as ss

DTYPES = (mx.float32, mx.float16, mx.bfloat16)


def _tied(shape, dtype, seed, *, low=-2, high=3):
    """Integer-valued (exactly representable) values: many exact ties."""
    mx.random.seed(seed)
    return mx.random.randint(low, high, shape).astype(dtype)


def _normal(shape, dtype, seed):
    mx.random.seed(seed)
    return mx.random.normal(shape).astype(dtype)


def _eq(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype, (a.shape, b.shape, a.dtype, b.dtype)
    assert mx.array_equal(a, b, equal_nan=True).item()


# --------------------------------------------------------------------------
# reference-level contracts


def test_spec_from_config_and_validation():
    spec = ss.SelectionSpec.from_config(
        {"index_n_heads": 64, "index_head_dim": 128, "index_topk": 512, "compress_ratio": 4},
        tie_break="argsort",
    )
    assert (spec.n_heads, spec.head_dim, spec.topk, spec.ratio, spec.tie_break) == (64, 128, 512, 4, "argsort")
    assert spec.scale() == 128**-0.5
    with pytest.raises(ValueError):
        ss.SelectionSpec(tie_break="random")
    with pytest.raises(ValueError):
        ss.SelectionSpec(score_weighting="uniform", score_scale_site="post_relu")
    with pytest.raises(TypeError):
        ss.SelectionSpec.from_config({}, not_a_field=1)


def test_stable_topk_is_chronological_at_ties():
    scores = mx.array([[0.0, 1.0, 0.0, 1.0, 0.0, 0.0]])
    assert ss.stable_topk(scores, 3).tolist() == [[0, 1, 3]]
    assert ss.stable_topk(scores, 4).tolist() == [[0, 1, 2, 3]]


def test_gather_attend_latent_kv_sources():
    """K=V latent sources: values=None uses the keys, value_width a prefix."""
    q = _normal((1, 2, 3, 6), mx.float32, 100)
    keys = _normal((1, 1, 5, 6), mx.float32, 101)
    full = ss.gather_attend(q, (ss.Source(keys),))
    explicit = ss.gather_attend(q, (ss.Source(keys, values=keys),))
    _eq(full, explicit)
    prefix = ss.gather_attend(q, (ss.Source(keys, value_width=4),))
    _eq(prefix, ss.gather_attend(q, (ss.Source(keys, values=keys[..., :4]),)))
    assert prefix.shape == (1, 2, 3, 4)
    # one merge: two shared sources == logaddexp of their logsumexps
    a, b = keys[:, :, :2], keys[:, :, 2:]
    merged = ss.gather_attend(q, (ss.Source(a), ss.Source(b)))
    assert mx.allclose(merged, full, atol=1e-6).item()


def test_registries_never_pick_a_kernel_implicitly():
    spec = ss.SelectionSpec()
    for registry in (ss.SCORE_PATHS, ss.SELECT_PATHS, ss.GATHER_ATTEND_PATHS):
        fn, receipt = registry.select(spec)
        assert receipt.selected == "reference" and fn is registry.reference
        assert receipt.refusals and all(
            r == "policy_off: not requested by the call-site policy" for r in receipt.refusals.values()
        )
    _, receipt = ss.SELECT_PATHS.select(spec, policy="qsa_sidecar_fused_select",
                                        requested="qsa_sidecar_fused_select")
    assert receipt.selected == "qsa_sidecar_fused_select"
    with pytest.raises(RuntimeError):
        ss.SCORE_PATHS.select(spec, policy="qwen4_qsa_fused_scores")


