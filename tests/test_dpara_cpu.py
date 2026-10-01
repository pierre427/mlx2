"""Actual tiny M-DFlash tensor forwards on CPU; no checkpoint qualification."""

import dataclasses
import hashlib
import json

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.dpara import DParaBinding, DParaPrepared, DParaVerification
from mlx2.runtime.drafters.dpara import (
    MDFlashDParaDraftModel,
    _position_rope,
    branch_layout,
    load_dpara_artifact,
)
from mlx2.runtime.drafters.dpara_config import DParaConfig
from mlx2.runtime.speculative_sampling import RequestRNG, softmax, verify_proposals


def tiny(d=3, context_length=2):
    mx.random.seed(83)
    config = DParaConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=7,
        markov_rank=3,
        mask_token_id=6,
        block_size=d + 1,
        target_layer_ids=[0, 2],
        num_target_layers=3,
    )
    model = MDFlashDParaDraftModel(config)
    binding = DParaBinding("request-a", "synthetic-target", "synthetic", 2)
    features = mx.random.normal((1, context_length, 16))
    context = model.context(features, binding)
    mx.eval(model.parameters(), *[value for pair in context.layers for value in pair])
    return model, context, tuple(range(d + 1))


def separate_prefix_oracle(model, context, spine, r):
    """Independent single-branch graph, never the flattened tree implementation."""
    d = len(spine) - 1
    anchors = r + 1
    ids = mx.array([(*spine[:anchors], *((model.config.mask_token_id,) * d))])
    length = anchors + d
    allowed = np.zeros((length, context.length + length), dtype=bool)
    allowed[:, : context.length] = True
    for i in range(anchors):
        allowed[i, context.length : context.length + i + 1] = True
    allowed[anchors:, context.length :] = True
    mask = mx.array(allowed)
    x = model.embed_tokens(ids)
    for layer, (ck, cv) in zip(model.layers, context.layers, strict=True):
        attn = layer.self_attn
        normed = layer.input_layernorm(x)
        q = attn.q_norm(
            attn.q_proj(normed).reshape(1, length, attn.n_heads, attn.head_dim)
        ).transpose(0, 2, 1, 3)
        k = attn.k_norm(
            attn.k_proj(normed).reshape(1, length, attn.n_kv_heads, attn.head_dim)
        ).transpose(0, 2, 1, 3)
        v = (
            attn.v_proj(normed)
            .reshape(1, length, attn.n_kv_heads, attn.head_dim)
            .transpose(0, 2, 1, 3)
        )
        q = mx.fast.rope(
            q,
            model.config.head_dim,
            traditional=False,
            base=model.config.rope_theta,
            scale=1.0,
            offset=context.length,
        )
        k = mx.fast.rope(
            k,
            model.config.head_dim,
            traditional=False,
            base=model.config.rope_theta,
            scale=1.0,
            offset=context.length,
        )
        output = mx.fast.scaled_dot_product_attention(
            q,
            mx.concatenate((ck, k), axis=2),
            mx.concatenate((cv, v), axis=2),
            scale=attn.scale,
            mask=mask,
        )
        x = x + attn.o_proj(output.transpose(0, 2, 1, 3).reshape(1, length, -1))
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    hidden = model.norm(x)[0, anchors:]
    return hidden, model._logits(hidden)


@pytest.mark.parametrize("d", [1, 2, 4])
def test_tree_mask_isolates_every_branch_and_causal_spine(d):
    context = 2
    allowed, positions = branch_layout(d, context)
    spine = d + 1
    assert allowed[:, :context].all()
    for i in range(spine):
        assert allowed[i, context : context + i + 1].all()
        assert not allowed[i, context + i + 1 :].any()
    for r in range(spine):
        start = spine + r * d
        visible = set(np.flatnonzero(allowed[start]))
        expected = set(range(context + r + 1)) | set(
            range(context + start, context + start + d)
        )
        assert visible == expected
        assert np.all(allowed[start : start + d] == allowed[start])
        assert positions[start : start + d].tolist() == list(
            range(context + r + 1, context + r + 1 + d)
        )


def test_vector_position_rope_matches_separate_offsets_with_repeats():
    model, _, _ = tiny()
    x = mx.random.normal((1, 2, 5, 4))
    positions = mx.array([2, 3, 2, 4, 3], dtype=mx.int32)
    got = _position_rope(x, positions, model.config)
    expected = mx.concatenate(
        [
            mx.fast.rope(
                x[:, :, i : i + 1],
                4,
                traditional=False,
                base=model.config.rope_theta,
                scale=1.0,
                offset=int(position),
            )
            for i, position in enumerate(positions.tolist())
        ],
        axis=2,
    )
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), atol=2e-6)


@pytest.mark.parametrize("context_length", [0, 2, 5])
def test_all_branches_match_independent_prefix_oracle(context_length):
    model, context, spine = tiny(context_length=context_length)
    prepared = model.prepare(spine, context)
    for r in range(len(spine)):
        hidden, logits = separate_prefix_oracle(model, context, spine, r)
        np.testing.assert_allclose(
            np.asarray(prepared.payload.hidden[r]), np.asarray(hidden), atol=3e-6
        )
        np.testing.assert_allclose(
            np.asarray(prepared.payload.logits[r]), np.asarray(logits), atol=3e-6
        )


def test_rejected_spine_suffix_cannot_change_selected_branch():
    model, context, spine = tiny()
    original = model.prepare(spine, context).payload
    changed = model.prepare((spine[0], spine[1], 5, 4), context).payload
    np.testing.assert_allclose(
        np.asarray(original.logits[1]), np.asarray(changed.logits[1]), atol=2e-6
    )
    assert not np.allclose(
        np.asarray(original.logits[3]), np.asarray(changed.logits[3])
    )


@pytest.mark.parametrize("r", [0, 1, 2, 3])
@pytest.mark.parametrize("bonus", [0, 2, 5])
def test_every_acceptance_and_bonus_selects_real_head_and_commits_only_verified_prefix(
    r, bonus
):
    model, context, spine = tiny()
    prepared = model.prepare(spine, context)
    base = np.asarray(prepared.payload.logits[r]).copy()
    previous = bonus
    expected = []
    for position in range(3):
        correction = np.asarray(model.markov_head(mx.array(previous)))
        previous = int(np.argmax(base[position] + correction))
        expected.append(previous)
    before = [np.asarray(value).copy() for pair in context.layers for value in pair]
    features = mx.random.normal((1, 4, 16))
    verified = DParaVerification(context.binding, r, bonus, features)
    result = model.resolve(prepared, verified)
    assert result.draft_tokens == tuple(expected)
    assert result.next_spine == (bonus, *expected)
    assert result.accepted_tokens == spine[1 : r + 1]
    assert result.context.length == context.length + r + 1
    assert result.binding == context.binding.next()
    assert all(
        law[token] == 1
        for token, law in zip(
            result.draft_tokens, result.proposal_probabilities, strict=True
        )
    )
    for original, current in zip(
        before, [value for pair in context.layers for value in pair], strict=True
    ):
        np.testing.assert_array_equal(original, np.asarray(current))
    new = model._project_features(features[:, : r + 1], context.length)
    for old, added, committed in zip(
        context.layers, new, result.context.layers, strict=True
    ):
        for axis in (0, 1):
            expected_kv = mx.concatenate((old[axis], added[axis]), axis=2)
            np.testing.assert_allclose(
                np.asarray(committed[axis]), np.asarray(expected_kv), atol=2e-6
            )
    assert prepared.state == "resolved"
    with pytest.raises(ValueError, match="no longer ready"):
        model.resolve(prepared, verified)


def test_exact_markov_conditional_laws_can_use_ordinary_rejection_verifier():
    model, context, spine = tiny(d=2)
    prepared = model.prepare(spine, context)
    logits = np.asarray(prepared.payload.logits[1])
    result = model.resolve(
        prepared,
        DParaVerification(context.binding, 1, 3, mx.zeros((1, 3, 16))),
        temperature=0.8,
        rng=RequestRNG(98),
    )
    previous = 3
    for i, (token, law) in enumerate(
        zip(result.draft_tokens, result.proposal_probabilities, strict=True)
    ):
        expected = softmax(
            logits[i] + np.asarray(model.markov_head(mx.array(previous))), 0.8
        )
        np.testing.assert_allclose(law, expected, atol=2e-7)
        assert law[token] > 0
        previous = token
    target = np.array([0.15, 0.05, 0.1, 0.3, 0.2, 0.12, 0.08])
    verified = verify_proposals(
        result.draft_tokens, result.proposal_probabilities, [target] * 3, RequestRNG(22)
    )
    assert 0 <= verified.accepted <= 2


def test_markov_proposal_plus_rejection_has_target_first_token_distribution():
    model, context, spine = tiny(d=1)
    base = np.asarray(model.prepare(spine, context).payload.logits[0, 0])
    q = softmax(base + np.asarray(model.markov_head(mx.array(3))))
    target = np.array([0.15, 0.05, 0.1, 0.3, 0.2, 0.12, 0.08])
    rng = RequestRNG(184)
    counts = np.zeros(7)
    # Host Monte Carlo once tensor-generated q is independently checked above.
    for _ in range(12000):
        token = rng.sample(q)
        result = verify_proposals((token,), (q,), (target, target), rng)
        counts[result.emitted[0]] += 1
    np.testing.assert_allclose(counts / counts.sum(), target, atol=0.013)


def test_transformed_law_is_exact_and_invalid_transform_rolls_back_rng():
    model, context, spine = tiny(d=2)
    law = np.array([0.1, 0.3, 0.2, 0, 0.1, 0.1, 0.2])
    calls = []

    def transform(scores, position, previous):
        calls.append((position, previous))
        return law

    result = model.resolve(
        model.prepare(spine, context),
        DParaVerification(context.binding, 0, 4, mx.zeros((1, 3, 16))),
        probability_transform=transform,
        rng=RequestRNG(8),
    )
    assert calls[1][1] == (4, result.draft_tokens[0])
    for q in result.proposal_probabilities:
        np.testing.assert_array_equal(q, law)
        assert not q.flags.writeable
    rng = RequestRNG(8)
    snapshot = rng.snapshot()
    prepared = model.prepare(spine, context)
    with pytest.raises(ValueError, match="Invalid probability"):
        model.resolve(
            prepared,
            DParaVerification(context.binding, 0, 4, mx.zeros((1, 3, 16))),
            rng=rng,
            probability_transform=lambda scores, i, previous: (
                law if i == 0 else np.zeros(7)
            ),
        )
    assert rng.snapshot() == snapshot
    assert prepared.state == "discarded"


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_id", "other"),
        ("target_revision", "changed"),
        ("draft_revision", "changed"),
        ("generation", 8),
    ],
)
def test_stale_verification_fails_closed(field, value):
    model, context, spine = tiny()
    prepared = model.prepare(spine, context)
    bad = dataclasses.replace(context.binding, **{field: value})
    with pytest.raises(ValueError, match="mismatch"):
        model.resolve(prepared, DParaVerification(bad, 0, 1, mx.zeros((1, 4, 16))))
    assert context.length == 2
    prepared.discard()
    with pytest.raises(ValueError, match="no longer ready"):
        model.resolve(
            prepared, DParaVerification(context.binding, 0, 1, mx.zeros((1, 4, 16)))
        )


def test_non_dpara_artifacts_and_unsupported_context_regimes_refused():
    with pytest.raises(ValueError, match="Missing compatible"):
        DParaConfig.from_dict(
            {"model_type": "qwen3", "dflash_config": {"mask_token_id": 6}}
        )
    config = dataclasses.asdict(tiny()[0].config)
    with pytest.raises(ValueError, match="Missing compatible"):
        DParaConfig.from_dict(config)
    with pytest.raises(ValueError, match="full attention"):
        DParaConfig(**{**config, "layer_types": ["sliding_attention"] * 2})


def test_nonfinite_accepted_features_refused_but_rejected_suffix_is_never_projected():
    model, context, spine = tiny()
    features = mx.concatenate(
        (mx.zeros((1, 1, 16)), mx.full((1, 3, 16), float("nan"))), axis=1
    )
    result = model.resolve(
        model.prepare(spine, context),
        DParaVerification(context.binding, 0, 1, features),
    )
    assert result.context.length == 3
    prepared = model.prepare(spine, context)
    with pytest.raises(ValueError, match="must be finite"):
        model.resolve(prepared, DParaVerification(context.binding, 1, 1, features))
    assert prepared.state == "discarded"


def test_public_handle_rejects_different_request_context_at_construction():
    model, context, spine = tiny()
    payload = model.prepare(spine, context).payload
    other = dataclasses.replace(context.binding, request_id="other-request")
    with pytest.raises(ValueError, match="handle/context binding mismatch"):
        DParaPrepared(other, spine, context, payload)


@pytest.mark.parametrize("changed", ["request", "spine", "context"])
def test_repacked_public_branch_payload_cannot_publish_an_unrelated_state(changed):
    model, context, spine = tiny()
    prepared = model.prepare(spine, context)
    binding, attached_context, attached_spine = context.binding, context, spine
    if changed == "request":
        binding = dataclasses.replace(binding, request_id="other-request")
        attached_context = dataclasses.replace(context, binding=binding)
    elif changed == "spine":
        attached_spine = (5, *spine[1:])
    else:
        attached_context = model.context(mx.zeros((1, 2, 16)), binding)
    mixed = DParaPrepared(binding, attached_spine, attached_context, prepared.payload)
    rng = RequestRNG(3)
    before = rng.snapshot()
    with pytest.raises(ValueError, match="branch binding/spine/context mismatch"):
        model.resolve(
            mixed, DParaVerification(binding, 0, 1, mx.zeros((1, 4, 16))), rng=rng
        )
    assert rng.snapshot() == before
    assert mixed.state == "discarded"
    assert context.binding.request_id == "request-a" and context.length == 2


@pytest.mark.parametrize("pin", ["expected_target_revision", "expected_draft_revision"])
def test_expected_artifact_revision_checked_before_allocation_or_manifest(
    tmp_path, monkeypatch, pin
):
    import mlx2.runtime.drafters.dpara as module

    model, _, _ = tiny()
    config = dataclasses.asdict(model.config)
    config.update(
        model_type="dpara",
        dpara_config={
            "training_regime": "multiple_anchor_featureless",
            "backbone_revision": "trained-backbone",
            "target_revision": "trained-target",
            "markov_rank": 3,
            "markov_head": "low_rank_previous_token",
        },
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(
        module,
        "MDFlashDParaDraftModel",
        lambda *args: pytest.fail("mismatched identity reached model allocation"),
    )
    monkeypatch.setattr(
        mx,
        "load",
        lambda *args: pytest.fail("mismatched identity reached tensor loading"),
    )
    with pytest.raises(ValueError, match="expected .* revision mismatch"):
        load_dpara_artifact(tmp_path, **{pin: "wrong-revision"})


def test_pinned_artifact_roundtrip_and_hash_failure_before_tensor_load(
    tmp_path, monkeypatch
):
    model, _, _ = tiny()
    config = dataclasses.asdict(model.config)
    config.update(
        model_type="dpara",
        dpara_config={
            "training_regime": "multiple_anchor_featureless",
            "backbone_revision": "test-trained-revision",
            "target_revision": "test-target-revision",
            "markov_rank": 3,
            "markov_head": "low_rank_previous_token",
        },
    )
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config))
    from mlx.utils import tree_flatten

    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    hashes = {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in ("config.json", "model.safetensors")
    }
    (tmp_path / "dpara-manifest.json").write_text(
        json.dumps(
            {
                "backbone_revision": "test-trained-revision",
                "target_revision": "test-target-revision",
                "sha256": hashes,
            }
        )
    )
    restored = load_dpara_artifact(
        tmp_path,
        expected_target_revision="test-target-revision",
        expected_draft_revision="test-trained-revision",
    )
    assert restored.config.backbone_revision == "test-trained-revision"
    for (name, expected), (_, actual) in zip(
        tree_flatten(model.parameters()),
        tree_flatten(restored.parameters()),
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.asarray(actual), np.asarray(expected), err_msg=name
        )
    config_file.write_text(config_file.read_text() + "\n")
    monkeypatch.setattr(
        mx, "load", lambda *args: pytest.fail("hash failure must precede tensor load")
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        load_dpara_artifact(tmp_path)
