"""CPU/static contract for Xing longest-first exact-prefix strategy transfer."""

import copy
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.xing import XingAdapter
from mlx2.runtime.models.cache import KVCache
from mlx2.runtime.models.xing4_0 import Model, ModelArgs
from mlx2.runtime.proposal_cascade import (
    ALGORITHM,
    LongestFirstPrefixCascade,
    PrefixCascadeError,
)

FIXTURE = Path(__file__).parent / "fixtures" / "xing4_0_tiny"


def _model():
    config = json.loads((FIXTURE / "config.json").read_text())
    model = Model(ModelArgs.from_dict(config))
    weights = model.sanitize(mx.load(str(FIXTURE / "weights.safetensors")))
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model


def test_xing_owns_qualified_prefill_step_and_candidate_policy():
    adapter = object.__new__(XingAdapter)
    assert adapter.prefill_step_default() == 2048
    assert adapter.proposal_verification_policy() == {
        "algorithm": ALGORITHM,
        "order": "longest_first",
        "prune": "authoritative_target_prefix",
        "shared_prefix_reuse": "exact_adapter_geometry_only",
        "implemented": True,
        "qualified": False,
        "selected": False,
        "observed_used": False,
    }


def test_longest_first_prunes_impossible_siblings_and_resumes_exact_suffix():
    model = _model()
    cache = model.make_cache()
    model(mx.array([[1, 2, 3]], mx.int32), cache=cache)
    adapter = object.__new__(XingAdapter)
    adapter.model = model
    adapter.identity = {"fingerprint": "a" * 64}
    cascade = adapter.proposal_verification_plan(
        (
            (1, 2, 3, 4, 5, 6, 7),
            (0, 2, 3, 4, 9, 6),
            (1, 2, 0, 4, 9, 6),
            (1, 2, 3, 4, 9, 6),
        ),
        cache,
    )

    first = cascade.next_attempt()
    assert first.candidate_index == 0
    assert first.verification_suffix == (1, 2, 3, 4, 5, 6, 7)
    observation = cascade.observe(first, matched_prefix_tokens=4, correction_token=9)
    assert observation.frontier == (1, 2, 3, 4, 9)
    assert observation.pruned_candidates == (1, 2)
    assert observation.surviving_candidates == (3,)
    assert not observation.terminal

    second = cascade.next_attempt()
    assert second.candidate_index == 3
    # [1,2,3,4] remains in the exact cache.  Token 9 is emitted but pending,
    # so the next model input begins at the correction and does not recompute
    # the four cached common proposal tokens.
    assert second.shared_prefix_tokens == second.proposal_start == 4
    assert second.verification_suffix == (9, 6)
    final = cascade.observe(second, matched_prefix_tokens=6)
    assert final.full_match and final.terminal
    assert cascade.receipt()["reused_prefix_tokens"] == 4


def test_without_exact_geometry_cascade_recomputes_instead_of_claiming_reuse():
    cascade = LongestFirstPrefixCascade(((1, 2, 3), (1, 9, 4)))
    first = cascade.next_attempt()
    cascade.observe(first, matched_prefix_tokens=1, correction_token=9)
    second = cascade.next_attempt()
    assert second.shared_prefix_tokens == second.proposal_start == 0
    assert second.verification_suffix == (1, 9, 4)
    assert not cascade.receipt()["shared_prefix_reuse"]


def test_xing_geometry_is_derived_from_compressed_mla_not_dense_gqa():
    model = _model()
    cache = model.make_cache()
    model(mx.array([[1, 2, 3, 4]], mx.int32), cache=cache)
    geometry = model.exact_prefix_reuse_geometry(cache, state_revision="a" * 64)
    assert geometry.cache_layout == model.apc_v2_layout
    assert geometry.position == 4 and geometry.layer_count == len(model.layers)
    assert geometry.state_components == ("mla_latent", "rope_key")
    assert geometry.component_widths == (
        model.args.kv_lora_rank,
        model.args.qk_rope_head_dim,
    )
    assert geometry.component_widths != (
        model.args.num_key_value_heads,
        model.args.v_head_dim,
    )


def test_xing_geometry_refuses_segmented_approximate_or_misaligned_state():
    model = _model()
    empty = model.make_cache()
    with pytest.raises(PrefixCascadeError, match="populated"):
        model.exact_prefix_reuse_geometry(empty, state_revision="a" * 64)

    cache = model.make_cache()
    model(mx.array([[1, 2, 3, 4]], mx.int32), cache=cache)
    segmented = [KVCache.merge([layer]) for layer in cache]
    with pytest.raises(PrefixCascadeError, match="unsegmented"):
        model.exact_prefix_reuse_geometry(segmented, state_revision="a" * 64)

    misaligned = copy.deepcopy(cache)
    misaligned[0].trim(1)
    with pytest.raises(PrefixCascadeError, match="aligned"):
        model.exact_prefix_reuse_geometry(misaligned, state_revision="a" * 64)


def test_exact_shared_prefix_cache_matches_full_recomputation():
    model = _model()
    prefix = [1, 2, 3]
    shared = [4, 5]
    suffix = [6, 7]
    reused = model.make_cache()
    model(mx.array([prefix], mx.int32), cache=reused)
    model(mx.array([shared], mx.int32), cache=reused)
    model.exact_prefix_reuse_geometry(reused, state_revision="a" * 64)
    got = model(mx.array([suffix], mx.int32), cache=copy.deepcopy(reused))

    fresh = model.make_cache()
    expected = model(mx.array([[*prefix, *shared, *suffix]], mx.int32), cache=fresh)
    np.testing.assert_allclose(
        np.asarray(got.astype(mx.float32)),
        np.asarray(expected[:, -len(suffix) :].astype(mx.float32)),
        rtol=1e-5,
        atol=1e-5,
    )
