"""Inkling 2 community DFlash2 route: config, admission and drafter law (CPU).

Tiny tensors only; the 19 GB target is never loaded.  The real drafter's
config and weights are inspected (header + sha256, no tensor load) when
``~/mlx-models`` holds both artifacts, and skipped otherwise.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

mx.set_default_device(mx.cpu)

from mlx2.adapters import dflash2 as dflash2_adapter
from mlx2.adapters import inkling2_dflash2
from mlx2.runtime.drafters.dflash2 import (
    CandidateSelector,
    DFlash2DraftModel,
    cap_selector_scores,
)
from mlx2.runtime.drafters.dflash2_config import SELECTOR_SOFTCAP_MODES, DFlash2Config

TARGET = Path("~/mlx-models/Inkling-2-Mini").expanduser()
DRAFT = Path("~/mlx-models/Inkling-2-Mini-DFlash2").expanduser()
CAP = 30.0


def _hf_config(**dflash):
    base = {
        "architectures": ["DFlash2DraftModel"], "model_type": "qwen3", "hidden_size": 8,
        "intermediate_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 4, "vocab_size": 32, "num_target_layers": 4,
        "max_position_embeddings": 128, "sliding_window": 3, "rms_norm_eps": 1e-6,
        "layer_types": ["sliding_attention"] * 2,
        "dflash_config": {"block_size": 4, "mask_token_id": 29, "target_layer_ids": [0, 3],
                          "conv_kernel_size": 2, "conv_group_size": 2, "selector_rank": 4,
                          "selector_top_k": 4, "final_logit_softcapping": CAP},
    }
    base["dflash_config"].update(dflash)
    return base


# ---------------------------------------------------------------- config


def test_selector_softcap_mode_absent_is_todays_law():
    args = DFlash2Config.from_dict(_hf_config())
    assert args.selector_softcap_mode == "before_correction"
    assert SELECTOR_SOFTCAP_MODES == ("before_correction", "after_correction")


@pytest.mark.parametrize("mode", ["before_correction", "after_correction"])
def test_selector_softcap_mode_known_values_parse(mode):
    assert DFlash2Config.from_dict(_hf_config(selector_softcap_mode=mode)).selector_softcap_mode == mode


@pytest.mark.parametrize("mode", ["after", "AFTER_CORRECTION", "", None, 1, ["after_correction"]])
def test_selector_softcap_mode_unknown_values_refuse(mode):
    with pytest.raises(ValueError, match="selector_softcap_mode"):
        DFlash2Config.from_dict(_hf_config(selector_softcap_mode=mode))


def test_after_correction_needs_a_softcap():
    config = _hf_config(selector_softcap_mode="after_correction")
    del config["dflash_config"]["final_logit_softcapping"]
    with pytest.raises(ValueError, match="final_logit_softcapping"):
        DFlash2Config.from_dict(config)


def test_final_norm_tap_index_parses_but_beyond_refuses():
    args = DFlash2Config.from_dict(_hf_config(target_layer_ids=[0, 4]))
    assert args.taps_final_norm
    assert not DFlash2Config.from_dict(_hf_config()).taps_final_norm
    with pytest.raises(ValueError, match="target_layer_ids"):
        DFlash2Config.from_dict(_hf_config(target_layer_ids=[0, 5]))


# ---------------------------------------------------------------- admission


def test_check_target_taps_by_visit_count_and_final_norm():
    args = DFlash2Config.from_dict({**_hf_config(target_layer_ids=[1, 47, 48]),
                                    "num_target_layers": 48})
    # Looped target: 24 physical layers, 48 visits, final-norm tap declared.
    dflash2_adapter.check_target_taps(args, layer_count=24, tap_count=48, final_norm_tap=True)
    with pytest.raises(ValueError, match="layer count mismatch"):
        dflash2_adapter.check_target_taps(args, layer_count=24)
    with pytest.raises(ValueError, match="visit count mismatch"):
        dflash2_adapter.check_target_taps(args, layer_count=24, tap_count=47, final_norm_tap=True)
    with pytest.raises(ValueError, match="final-norm"):
        dflash2_adapter.check_target_taps(args, layer_count=24, tap_count=48)
    plain = DFlash2Config.from_dict(_hf_config())
    # Today's rule for a target that declares nothing.
    dflash2_adapter.check_target_taps(plain, layer_count=4)
    with pytest.raises(ValueError, match="layer count mismatch"):
        dflash2_adapter.check_target_taps(plain, layer_count=5)


def _text_config():
    return json.loads((TARGET / "config.json").read_text())["text_config"]


needs_artifacts = pytest.mark.skipif(
    not (TARGET / "config.json").is_file() or not (DRAFT / "model.safetensors").is_file(),
    reason="Inkling-2-Mini and Inkling-2-Mini-DFlash2 artifacts not present",
)


@needs_artifacts
def test_real_drafter_admitted_against_inkling2_by_visit_count():
    text = _text_config()
    assert inkling2_dflash2.target_visit_count(text) == 48
    record = inkling2_dflash2.inspect_drafter(
        DRAFT, target_path=TARGET, text_config=text, target_fingerprint="t" * 64)
    args = record["args"]
    assert record["weight_sha256"] == [inkling2_dflash2.WEIGHT_SHA256]
    assert record["checkpoint"] == f"Inkling-2-Mini-DFlash2@sha256:{inkling2_dflash2.WEIGHT_SHA256}"
    assert (args.block_size, args.mask_token_id, args.selector_top_k, args.selector_rank,
            args.conv_group_size, args.conv_kernel_size) == (8, 200056, 16, 256, 16, 2)
    assert args.selector_softcap_mode == "after_correction" and args.taps_final_norm
    assert list(args.target_layer_ids) == [5, 23, 41, 48] and args.num_target_layers == 48
    # The generic inspector refuses it for a target that counts physical layers.
    with pytest.raises(ValueError, match="layer count mismatch"):
        dflash2_adapter.inspect_drafter(DRAFT, TARGET, target_text=text)
    with pytest.raises(ValueError, match="final-norm"):
        dflash2_adapter.inspect_drafter(DRAFT, TARGET, target_text=text, target_tap_count=48)
    # The raw-embedding diagnostic arm is a different drafter identity.
    raw = inkling2_dflash2.inspect_drafter(
        DRAFT, target_path=TARGET, text_config=text, target_fingerprint="t" * 64,
        embed_norm=False)
    assert record["embed_norm"] is True and raw["embed_norm"] is False
    assert raw["embedding_law"] != record["embedding_law"]
    assert raw["fingerprint"] != record["fingerprint"]
    assert raw["draft_revision"] != record["draft_revision"]
    assert raw["artifact_fingerprint"] == record["artifact_fingerprint"]


def test_inspection_io_and_type_failures_are_counted_refusals(tmp_path):
    before = inkling2_dflash2.REFUSALS["inspection_error"]
    with pytest.raises(inkling2_dflash2.DFlash2Refusal, match="inspection_error"):
        inkling2_dflash2.inspect_drafter(tmp_path / "missing", target_path=tmp_path,
                                         text_config={}, target_fingerprint="t")
    (tmp_path / "config.json").write_text("{not json")
    with pytest.raises(inkling2_dflash2.DFlash2Refusal, match="inspection_error"):
        inkling2_dflash2.inspect_drafter(tmp_path, target_path=tmp_path,
                                         text_config={}, target_fingerprint="t")
    assert inkling2_dflash2.REFUSALS["inspection_error"] == before + 2


@needs_artifacts
@pytest.mark.parametrize("mutate,reason", [
    (lambda c: c["dflash_config"].__setitem__("mask_token_id", None), "mask_token"),
    (lambda c: c["dflash_config"].__setitem__("mask_token_id", "200056"), "mask_token"),
    (lambda c: c["dflash_config"].__setitem__("mask_token_id", 200100), "mask_token"),
    (lambda c: c.__setitem__("num_target_layers", 47), "visit_count"),
    (lambda c: c["dflash_config"].__setitem__("selector_softcap_mode", "mystery"), "softcap_mode"),
    (lambda c: c["dflash_config"].pop("selector_softcap_mode"), "softcap_mode"),
    (lambda c: c["dflash_config"].__setitem__("target_layer_ids", [5, 23, 41, 47]), "tap_ids"),
    (lambda c: c.__setitem__("vocab_size", 200058), "geometry"),
])
def test_declared_contract_refusals_are_counted(mutate, reason):
    config = json.loads((DRAFT / "config.json").read_text())
    mutate(config)
    before = inkling2_dflash2.REFUSALS[reason]
    with pytest.raises(ValueError, match=f"dflash2_refusal={reason}"):
        inkling2_dflash2._check_declared_contract(config, _text_config())
    assert inkling2_dflash2.REFUSALS[reason] == before + 1


def test_wrong_visit_count_target_refused_by_generic_inspector(tmp_path):
    """A drafter of 48 visits against a hypothetical 40-visit target."""
    config = {**_hf_config(target_layer_ids=[1, 48]), "num_target_layers": 48}
    args = DFlash2Config.from_dict(config)
    with pytest.raises(ValueError, match="visit count mismatch"):
        dflash2_adapter.check_target_taps(args, layer_count=20, tap_count=40, final_norm_tap=True)


# ---------------------------------------------------------------- selector law


def _selector(mode):
    mx.random.seed(3)
    args = DFlash2Config.from_dict(_hf_config(selector_softcap_mode=mode))
    selector = CandidateSelector(args)
    selector.predecessor_codebook.weight = mx.random.normal((32, 4)) * 4
    selector.successor_codebook.weight = mx.random.normal((32, 4)) * 4
    selector.hidden_projection.weight = mx.random.normal((4, 8)) * 4
    return args, selector


def test_post_correction_cap_math_on_a_tiny_example():
    raw = mx.array([[40.0, 10.0, -50.0]])
    edge = mx.array([[5.0, 25.0, -5.0]])
    before = (mx.tanh(raw / CAP) * CAP) + edge
    after = cap_selector_scores(raw + edge, CAP)
    np.testing.assert_allclose(np.asarray(after), np.tanh((np.asarray(raw) + np.asarray(edge)) / CAP) * CAP,
                               rtol=1e-6)
    # The placement changes the ranking: capping the unary first lets the
    # correction win; capping the sum keeps the large unary ahead.
    assert int(mx.argmax(before[0]).item()) == 1
    assert int(mx.argmax(after[0]).item()) == 0
    assert float(mx.max(mx.abs(after)).item()) < CAP
    assert float(mx.max(before).item()) > CAP  # before-mode scores are unbounded
    assert cap_selector_scores(raw, None) is raw


@pytest.mark.parametrize("mode", ["before_correction", "after_correction"])
def test_selector_select_uses_the_declared_cap_placement(mode):
    args, selector = _selector(mode)
    mx.random.seed(5)
    hidden = mx.random.normal((1, 3, 8))
    raw_logits = mx.random.normal((1, 3, 32)) * 40
    logits = raw_logits if mode == "after_correction" else mx.tanh(raw_logits / CAP) * CAP
    anchor = mx.array([7])
    got = selector.select(hidden, logits, anchor, lambda x: mx.argmax(x, -1))
    # Independent host reference of the selector walk.
    candidates = np.asarray(mx.argpartition(logits, -4, axis=-1)[..., -4:])
    unary = np.take_along_axis(np.asarray(logits), candidates, axis=-1)
    projected = np.asarray(selector.hidden_projection(hidden))
    pred = np.asarray(selector.predecessor_codebook.weight)
    succ = np.asarray(selector.successor_codebook.weight)
    previous, path = 7, []
    for position in range(3):
        edges = (pred[previous] * projected[0, position] * succ[candidates[0, position]]).sum(-1)
        scores = unary[0, position] + edges
        if mode == "after_correction":
            scores = np.tanh(scores / CAP) * CAP
        previous = int(candidates[0, position, int(np.argmax(scores))])
        path.append(previous)
    assert np.asarray(got)[0].tolist() == path
    assert (selector.post_softcap == CAP) is (mode == "after_correction")


# ---------------------------------------------------------------- target declarations


class _Target(nn.Module):
    """A minimal target exposing what DFlash2 ``bind`` reads."""

    def __init__(self, layers=4):
        super().__init__()
        mx.random.seed(11)
        self.model = SimpleNamespace(embed_tokens=nn.Embedding(32, 8))
        self.model.embed_tokens.weight = mx.random.normal((32, 8)) * 3
        self.lm_head = nn.Linear(8, 32, bias=False)
        self.lm_head.weight = mx.random.normal((32, 8)) * 20
        self.args = SimpleNamespace(hidden_size=8, vocab_size=32, num_hidden_layers=layers)


def _draft(mode="after_correction", **dflash):
    mx.random.seed(13)
    draft = DFlash2DraftModel(DFlash2Config.from_dict(_hf_config(selector_softcap_mode=mode, **dflash)))
    mx.eval(draft.parameters())
    return draft


def _declare(target, *, rows=30):
    norm = nn.RMSNorm(8)
    norm.weight = mx.arange(1, 9).astype(mx.float32) / 4
    embed = lambda inputs: norm(target.model.embed_tokens(inputs))  # noqa: E731
    object.__setattr__(target, "speculative_token_embedding", embed)
    object.__setattr__(target, "speculative_token_embedding_law", "test-norm")
    object.__setattr__(target, "speculative_vocab_rows", rows)
    return norm


def test_undeclared_target_binds_as_before():
    target = _Target()
    draft = _draft("before_correction").bind(target)
    tokens = mx.array([[1, 2, 29]])
    np.testing.assert_array_equal(np.asarray(draft._embed_input_tokens(tokens)),
                                  np.asarray(target.model.embed_tokens(tokens)))
    hidden = mx.random.normal((1, 2, 8))
    logits = draft._logits(hidden)
    assert logits.shape[-1] == 32
    np.testing.assert_allclose(np.asarray(logits),
                               np.asarray(mx.tanh(target.lm_head(hidden) / CAP) * CAP), rtol=1e-6)
    assert draft.proposal_vocab_size == 32
    settings = draft.receipt_settings
    assert not {"selector_softcap_mode", "target_token_embedding", "target_vocab_rows"} & set(settings)


def test_declared_target_norms_embeddings_and_crops_padded_rows():
    target = _Target()
    norm = _declare(target, rows=30)
    draft = _draft("after_correction").bind(target)
    tokens = mx.array([[1, 2, 29]])
    np.testing.assert_allclose(np.asarray(draft._embed_input_tokens(tokens)),
                               np.asarray(norm(target.model.embed_tokens(tokens))), rtol=1e-6)
    hidden = mx.random.normal((1, 2, 8))
    logits = draft._logits(hidden)
    # Cropped to the real rows and, in after_correction mode, not capped.
    assert logits.shape[-1] == 30
    np.testing.assert_allclose(np.asarray(logits), np.asarray(target.lm_head(hidden)[..., :30]),
                               rtol=1e-6)
    assert draft.proposal_vocab_size == 30
    settings = draft.receipt_settings
    assert settings["selector_softcap_mode"] == "after_correction"
    assert settings["target_token_embedding"] == "test-norm"
    assert settings["target_vocab_rows"] == 30
    draft.use_raw_token_embedding()
    np.testing.assert_array_equal(np.asarray(draft._embed_input_tokens(tokens)),
                                  np.asarray(target.model.embed_tokens(tokens)))
    # Sticky across a later bind/reset.
    draft.reset(target)
    np.testing.assert_array_equal(np.asarray(draft._embed_input_tokens(tokens)),
                                  np.asarray(target.model.embed_tokens(tokens)))
    assert draft.receipt_settings["target_token_embedding"] == "raw_table_diagnostic"


def test_declared_rows_refuse_a_mask_token_in_the_padding():
    target = _Target()
    _declare(target, rows=29)
    with pytest.raises(ValueError, match="mask_token_id"):
        _draft().bind(target)


def test_after_correction_laws_and_pairwise_scores_use_real_rows_and_post_cap():
    target = _Target()
    _declare(target, rows=30)
    draft = _draft("after_correction").bind(target)
    hidden = mx.random.normal((1, 3, 16))
    cache = draft.make_cache()
    rng = SimpleNamespace(uniform=lambda: 0.5, sample=lambda q: int(np.argmax(q)))
    tokens, laws = draft.draft_distributions([7], hidden, cache, 3, [rng], [0.0])
    assert len(tokens[0]) == 3 and all(len(law) == 30 for law in laws[0])
    assert all(0 <= token < 30 for token in tokens[0])
    # The pair table is capped after the correction term.
    features = mx.random.normal((1, 3, 8))
    logits = draft._logits(features)
    candidates = mx.argpartition(logits, -4, axis=-1)[..., -4:]
    unary = mx.take_along_axis(logits, candidates, axis=-1)
    scores = draft.pairwise_score_table(mx.array([7]), features, candidates, unary)
    assert float(mx.max(mx.abs(scores)).item()) <= CAP
    draft.candidate_selector.post_softcap = None
    uncapped = draft.pairwise_score_table(mx.array([7]), features, candidates, unary)
    np.testing.assert_allclose(np.asarray(scores), np.tanh(np.asarray(uncapped) / CAP) * CAP,
                               rtol=1e-5, atol=1e-5)


def test_rebind_clears_previous_target_terms_and_rejects_invalid_target_atomically():
    target = _Target()
    _declare(target, rows=30)
    draft = _draft("after_correction").bind(target)
    invalid = _Target()
    _declare(invalid, rows=1)
    with pytest.raises(ValueError):
        draft.reset(invalid)
    assert draft.embed_tokens is target.model.embed_tokens
    assert draft.proposal_vocab_size == 30

    plain = _Target()
    draft.reset(plain)
    assert draft.proposal_vocab_size == draft.config.vocab_size
    assert "target_token_embedding" not in draft.receipt_settings
    assert "target_vocab_rows" not in draft.receipt_settings
    tokens = mx.array([[1, 2, 29]])
    np.testing.assert_array_equal(draft._embed_input_tokens(tokens), plain.model.embed_tokens(tokens))


def test_after_correction_refuses_tree_and_lattice_paths():
    target = _Target()
    draft = _draft("after_correction").bind(target)
    assert draft.supports_continuation_tree_reuse is False
    assert draft.supports_tree_calibration is False
    settings = draft.receipt_settings
    assert settings["proposal_distribution"] == "categorical_chain"
    assert "tree_ranking" not in settings and "tree_max_nodes" not in settings
    assert _draft("before_correction").supports_continuation_tree_reuse is True
    hidden = mx.random.normal((1, 3, 16))
    with pytest.raises(ValueError, match="tree drafting"):
        draft.start_tree([7], hidden, draft.make_cache(), 7)
    rng = SimpleNamespace(uniform=lambda: 0.5, sample=lambda q: int(np.argmax(q)))
    with pytest.raises(ValueError, match="continuation lattices"):
        draft.draft_distributions([7], hidden, draft.make_cache(), 3, [rng], [0.0],
                                  capture_continuation_tree=True)


def test_before_correction_draft_is_bitwise_todays_law():
    """Absent key and explicit before_correction give identical drafts."""
    target = _Target()
    explicit = _draft("before_correction").bind(target)
    config = _hf_config()
    absent = DFlash2DraftModel(DFlash2Config.from_dict(config))
    absent.load_weights(tree_flatten(explicit.parameters()), strict=False)
    absent.bind(target)
    hidden = mx.random.normal((1, 3, 16))
    rng = SimpleNamespace(uniform=lambda: 0.5, sample=lambda q: int(np.argmax(q)))
    a = explicit.draft_distributions([7], hidden, explicit.make_cache(), 3, [rng], [0.0])
    b = absent.draft_distributions([7], hidden, absent.make_cache(), 3, [rng], [0.0])
    assert a[0] == b[0]
    for x, y in zip(a[1][0], b[1][0]):
        np.testing.assert_array_equal(x, y)


# ---------------------------------------------------------------- adapter policy


def _adapter():
    from mlx2.adapters.inkling2 import Inkling2Adapter

    return Inkling2Adapter.__new__(Inkling2Adapter)


def test_dflash2_route_selected_only_by_a_dflash2_path(tmp_path):
    adapter = _adapter()
    assert adapter._select_dflash2_route({}) is False
    assert adapter._select_dflash2_route({"draft_model": "embedded_dspark"}) is False
    (tmp_path / "config.json").write_text(json.dumps(_hf_config()))
    assert adapter._select_dflash2_route({"draft_model": str(tmp_path), "num_draft": 7}) is True
    assert adapter.dflash2_embed_norm is True
    policy = {"draft_model": str(tmp_path), inkling2_dflash2.EMBED_NORM_OPTION: False}
    assert adapter._select_dflash2_route(policy) is True
    assert adapter.dflash2_embed_norm is False and inkling2_dflash2.EMBED_NORM_OPTION not in policy


@pytest.mark.parametrize("key", ["continuation_pool", "progressive_verification_tile",
                                 "device_round", "prompt_copy_round", "ast_physical_target"])
def test_dflash2_route_refuses_unimplemented_policy_keys(tmp_path, key):
    (tmp_path / "config.json").write_text(json.dumps(_hf_config()))
    before = inkling2_dflash2.REFUSALS["unsupported_policy_key"]
    with pytest.raises(ValueError, match="does not implement"):
        _adapter()._select_dflash2_route({"draft_model": str(tmp_path), key: {}})
    assert inkling2_dflash2.REFUSALS["unsupported_policy_key"] == before + 1


def test_embed_norm_option_needs_the_dflash2_route():
    with pytest.raises(ValueError, match="requires the DFlash2 route"):
        _adapter()._select_dflash2_route({"draft_model": "embedded_dspark",
                                          inkling2_dflash2.EMBED_NORM_OPTION: True})


def test_budget_draft_prices_the_sliding_drafter():
    config = copy.deepcopy(_hf_config())
    draft = inkling2_dflash2.budget_draft(config)
    assert draft["local_layer_ids"] == [0, 1] and draft["num_hidden_layers"] == 2
    assert draft["sliding_window_size"] == 3 and draft["target_layer_ids"] == [0, 3]


# ---------------------------------------------------------------- engine


def _tiny_engine_pair():
    from mlx2.adapters.muse_glimmer_config import ModelArgs
    from mlx2.runtime.models.muse_glimmer import Model

    mx.random.seed(8)
    model = Model(ModelArgs(hidden_size=8, intermediate_size=16, num_hidden_layers=4,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=4,
                            vocab_size=32, sliding_window=16, max_position_embeddings=128))
    norm = nn.RMSNorm(8)
    object.__setattr__(model, "speculative_token_embedding",
                       lambda inputs: norm(model.model.embed_tokens(inputs)))
    draft = DFlash2DraftModel(DFlash2Config.from_dict(_hf_config(
        selector_softcap_mode="after_correction"))).bind(model)
    return model, draft


def _greedy_reference(model, prompt, count):
    cache = model.make_cache()
    tokens, out = list(prompt), []
    for i in range(count):
        logits = model(mx.array([tokens if i == 0 else [tokens[-1]]]), cache=cache)
        token = int(mx.argmax(logits[0, -1]).item())
        tokens.append(token)
        out.append(token)
    return out


@pytest.mark.parametrize("composed", [False, True])
def test_after_correction_drafter_serves_greedy_exact_on_the_external_engine(composed):
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from mlx2.runtime.proposal_composition import ComposedDraftModel

    model, draft = _tiny_engine_pair()
    backend = ComposedDraftModel(draft, {"prompt_lookup": True}) if composed else draft
    batch = ExternalDraftBatchGenerator(model, draft_model=backend, binding="inkling2-dflash2-test",
                                        num_draft=3, prefill_step_size=3)
    prompt = [1, 5, 9, 2, 7]
    try:
        uid = batch.insert([prompt], max_tokens=[10])[0]
        output = []
        for _ in range(100):
            _, responses = batch.next()
            output += [r.token for r in responses if r.uid == uid]
            if not batch.lanes:
                break
        else:
            raise AssertionError("scheduler stalled")
    finally:
        batch.close()
    assert output == _greedy_reference(model, prompt, 10)


# ---------------------------------------------------------------- review follow-ups


def _np_softmax(x, temperature):
    z = np.asarray(x, dtype=np.float64) / temperature
    z = np.exp(z - z.max())
    return z / z.sum()


def _reference_walk(draft, hidden, anchor, tokens, *, processor=None):
    """Independent host law per position: softmax(cap(unary + edges) / T)."""
    inputs = mx.concatenate([mx.array([[anchor]]), mx.full((1, len(tokens)),
                             draft.config.mask_token_id, dtype=mx.int32)], axis=1)
    features = draft._hidden(inputs, hidden, draft.make_cache())[:, 1:]
    logits = draft._logits(features)
    candidates = np.asarray(mx.argpartition(logits, -4, axis=-1)[..., -4:])[0]
    dense = np.asarray(logits.astype(mx.float32))[0]
    projected = np.asarray(draft.candidate_selector.hidden_projection(features).astype(mx.float32))[0]
    pred = np.asarray(draft.candidate_selector.predecessor_codebook.weight.astype(mx.float32))
    succ = np.asarray(draft.candidate_selector.successor_codebook.weight.astype(mx.float32))
    previous, rows = anchor, []
    for position, token in enumerate(tokens):
        ids = candidates[position]
        row = dense[position] if processor is None else processor(dense[position])
        edges = (pred[previous] * projected[position] * succ[ids]).sum(-1)
        raw = row[ids] + edges
        rows.append((ids, raw, np.tanh(raw / CAP) * CAP))
        previous = token
    return rows


@pytest.mark.parametrize("processed", [False, True])
def test_sampled_laws_are_softmax_of_post_correction_cap_at_the_cropped_width(processed):
    target = _Target()
    _declare(target, rows=30)
    draft = _draft("after_correction").bind(target)
    mx.random.seed(21)
    hidden = mx.random.normal((1, 3, 16))
    temperature = 1.3
    bias = np.zeros(30, dtype=np.float32)
    bias[[3, 7, 11]] = 12.0
    processor = (lambda _tokens, logits: logits + mx.array(bias)) if processed else None
    rng = SimpleNamespace(uniform=lambda: 0.37, sample=lambda q: int(np.argmax(q)))
    tokens, laws = draft.draft_distributions(
        [7], hidden, draft.make_cache(), 3, [rng], [temperature],
        **({"logits_processors": [[processor]], "processor_histories": [[1, 2]]} if processed else {}))
    assert all(len(law) == 30 for law in laws[0])
    reference = _reference_walk(draft, hidden, 7, tokens[0],
                                processor=(lambda row: row + bias) if processed else None)
    for law, (ids, raw, capped) in zip(laws[0], reference):
        np.testing.assert_allclose(law[ids], _np_softmax(capped, temperature), rtol=1e-4, atol=1e-6)
        assert abs(float(law.sum()) - 1.0) < 1e-6
    # The cap matters here: an uncapped law differs.
    assert any(np.abs(_np_softmax(raw, temperature) - law[ids]).max() > 1e-3
               for law, (ids, raw, _capped) in zip(laws[0], reference))


class _CaptureSampler:
    def __init__(self):
        self.scores = []

    def __call__(self, logits):
        return mx.argmax(logits, -1)

    def sample_candidate(self, scores, candidates, vocab):
        self.scores.append(scores)
        self.vocab = vocab
        return mx.argmax(scores, axis=-1)


def test_select_caps_bf16_scores_in_float32_without_ties():
    args = DFlash2Config.from_dict(_hf_config(selector_softcap_mode="after_correction"))
    selector = CandidateSelector(args)
    # Zero correction: the score is the capped unary alone.
    selector.predecessor_codebook.weight = mx.zeros((32, 4), dtype=mx.bfloat16)
    selector.successor_codebook.weight = mx.zeros((32, 4), dtype=mx.bfloat16)
    selector.hidden_projection.weight = mx.zeros((4, 8), dtype=mx.bfloat16)
    logits = mx.full((1, 1, 32), -100.0).astype(mx.bfloat16)
    logits[0, 0, 5] = 45.0
    logits[0, 0, 9] = 46.0
    hidden = mx.zeros((1, 1, 8), dtype=mx.bfloat16)
    sampler = _CaptureSampler()
    logits = logits[..., :30]  # The target head has two padding rows.
    path = selector.select(hidden, logits, mx.array([1]), sampler)
    assert sampler.vocab == 30
    scores = sampler.scores[0]
    assert scores.dtype == mx.float32
    candidates = np.asarray(mx.argpartition(logits, -4, axis=-1)[..., -4:])[0, 0]
    values = dict(zip(candidates.tolist(), np.asarray(scores)[0].tolist()))
    # BF16 tanh would give 27.25 for both; float32 keeps them apart.
    np.testing.assert_allclose(values[9], np.tanh(46 / CAP) * CAP, rtol=1e-6)
    np.testing.assert_allclose(values[5], np.tanh(45 / CAP) * CAP, rtol=1e-6)
    assert values[9] > values[5] and values[9] < CAP
    assert int(np.asarray(path)[0, 0]) == 9


def test_pairwise_scores_cap_bf16_in_float32():
    target = _Target()
    draft = _draft("after_correction").bind(target)
    # A BF16 drafter: codebooks, projection, features and unary all BF16.
    for module in (draft.candidate_selector.predecessor_codebook,
                   draft.candidate_selector.successor_codebook,
                   draft.candidate_selector.hidden_projection):
        module.weight = mx.zeros_like(module.weight).astype(mx.bfloat16)
    candidates = mx.array([[[5, 9, 1, 2]]], dtype=mx.uint32)
    unary = mx.array([[[45.0, 46.0, -100.0, -100.0]]]).astype(mx.bfloat16)
    features = mx.zeros((1, 1, 8), dtype=mx.bfloat16)
    scores = draft.pairwise_score_table(mx.array([7]), features, candidates, unary)
    row = np.asarray(scores)[0, 0, 0]
    assert row[1] > row[0]
    np.testing.assert_allclose(row[:2], np.tanh(np.array([45.0, 46.0]) / CAP) * CAP, rtol=1e-6)


def _diag_adapter(**attrs):
    adapter = _adapter()
    adapter._select_dflash2_route({})
    adapter.layout = "test-layout"
    adapter.identity = None
    adapter.kernel_policy = {}
    adapter.external_policy = {}
    for key, value in attrs.items():
        setattr(adapter, key, value)
    return adapter


def test_ordinary_and_dspark_diagnostics_carry_no_dflash2_key_after_another_refusal(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_hf_config()))
    refused = _diag_adapter()
    with pytest.raises(ValueError):
        refused._select_dflash2_route({"draft_model": str(tmp_path), "device_round": {}})
    assert inkling2_dflash2.REFUSALS["unsupported_policy_key"] >= 1
    ordinary = _diag_adapter()
    dspark = _diag_adapter(draft_model=object(), external_policy={"num_draft": 7})
    assert "dflash2" not in ordinary.diagnostics()
    dspark_diagnostics = dspark.diagnostics()
    assert "dflash2" not in dspark_diagnostics and dspark_diagnostics["dspark"]["selected"] is True
    receipt = refused.diagnostics()["dflash2"]
    assert receipt["selected"] is False and receipt["refusals"] == {"unsupported_policy_key": 1}


def test_dflash2_route_diagnostics_keep_drafter_counters_out_of_dspark():
    adapter = _diag_adapter(draft_model=object(), external_policy={"num_draft": 7},
                            dflash2_route=True, has_mtp=True)
    diagnostics = adapter.diagnostics()
    assert diagnostics["dspark"] == {"implemented": True, "selected": False, "qualified": False,
                                     "weights_present": True,
                                     "not_selected_reason": "dflash2_route_selected"}
    assert diagnostics["dflash2"]["selected"] is True
    assert diagnostics["dflash2"]["num_draft"] == 7


# ---------------------------------------------------------------- tiny real Inkling 2


def _tiny_inkling_pair():
    from test_inkling2_cpu import TINY

    from mlx2.runtime.models.inkling2 import Model, TextConfig

    mx.random.seed(0)
    target = Model(TextConfig.from_dict({**TINY, "final_logit_softcapping": CAP}))
    weights = []
    for key, value in tree_flatten(target.parameters()):
        noise = mx.random.normal(value.shape) * (0.08 if value.ndim > 1 else 0.3)
        weights.append((key, noise + (1.0 if key.endswith("norm.weight") else 0.0)))
    target.load_weights(weights, strict=True)
    mx.eval(target.parameters())
    config = {
        "architectures": ["DFlash2DraftModel"], "model_type": "qwen3", "hidden_size": 128,
        "intermediate_size": 96, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 32, "vocab_size": 512, "num_target_layers": 48,
        "max_position_embeddings": 4096, "sliding_window": 16, "rms_norm_eps": 1e-6,
        "layer_types": ["sliding_attention"] * 2,
        "dflash_config": {"block_size": 8, "mask_token_id": 499, "target_layer_ids": [5, 23, 41, 48],
                          "conv_kernel_size": 2, "conv_group_size": 16, "selector_rank": 16,
                          "selector_top_k": 8, "final_logit_softcapping": CAP,
                          "selector_softcap_mode": "after_correction"},
    }
    draft = DFlash2DraftModel(DFlash2Config.from_dict(config))
    mx.random.seed(1)
    draft.load_weights([(key, mx.random.normal(value.shape) * (0.08 if value.ndim > 1 else 0.3)
                         + (1.0 if key.endswith("norm.weight") else 0.0))
                        for key, value in tree_flatten(draft.parameters())], strict=True)
    with pytest.raises(ValueError, match="layer count mismatch"):
        DFlash2DraftModel(DFlash2Config.from_dict(config)).bind(target)  # undeclared: refused
    declared = inkling2_dflash2.declare_target(target)
    draft.bind(target)
    mx.eval(draft.parameters())
    return target, draft, declared


def test_declare_target_and_bind_against_a_tiny_inkling2_model():
    target, draft, declared = _tiny_inkling_pair()
    assert declared == {"tap_count": 48, "final_norm_tap": True, "vocab_rows": 500,
                        "embedding_law": inkling2_dflash2.EMBEDDING_LAW}
    assert draft.proposal_vocab_size == 500
    tokens = mx.array([[3, 17, 499]])
    np.testing.assert_array_equal(np.asarray(draft._embed_input_tokens(tokens)),
                                  np.asarray(target.model.embed(tokens)))
    logits, features = target.forward_with_taps(mx.array([[1, 2, 3, 4]]), target.make_cache(),
                                                [5, 23, 41, 48])
    assert logits.shape[-1] == 500 and features.shape[-1] == 4 * 128
    rng = SimpleNamespace(uniform=lambda: 0.5, sample=lambda q: int(np.argmax(q)))
    drafted, laws = draft.draft_distributions([3], features, draft.make_cache(), 7, [rng], [0.8])
    assert len(drafted[0]) == 7 and all(len(law) == 500 for law in laws[0])
    assert all(0 <= token < 500 for token in drafted[0])


def _inkling_run(target, draft, prompt, temperature, count=12):
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    batch = ExternalDraftBatchGenerator(target, draft_model=draft, binding="inkling2-dflash2-test",
                                        num_draft=7, prefill_step_size=16)
    try:
        uid = batch.insert([prompt], max_tokens=[count],
                           sampling_configs=[{"sampling_temp": temperature}])[0]
        got = []
        for _ in range(400):
            _, responses = batch.next()
            got += [r.token for r in responses if r.uid == uid]
            if not batch.lanes:
                break
        return got, dict(batch.scheduler_stats)
    finally:
        batch.close()


def test_tiny_inkling2_engine_greedy_exact_and_sampled_through_the_crop():
    target, draft, _ = _tiny_inkling_pair()
    mx.random.seed(4)
    prompt = [int(t) for t in mx.random.randint(1, 400, (24,)).tolist()]
    cache = target.make_cache()
    tokens, expect = list(prompt), []
    for step in range(12):
        logits = target(mx.array([tokens if step == 0 else [tokens[-1]]]), cache=cache)
        tokens.append(int(mx.argmax(logits[0, -1]).item()))
        expect.append(tokens[-1])
    greedy, _stats = _inkling_run(target, draft, prompt, 0.0)
    assert greedy == expect
    # T > 0: the 500-wide drafter laws meet the 500-wide target laws in the
    # sampled verifier (a 512-wide law would be refused there).
    sampled, stats = _inkling_run(target, draft, prompt, 1.0)
    assert len(sampled) == 12 and all(0 <= token < 500 for token in sampled)
    assert stats.get("external_hybrid_transactions", 0) > 0


def test_post_correction_cap_keeps_processor_exclusions():
    target = _Target()
    _declare(target, rows=30)
    draft = _draft("after_correction").bind(target)
    hidden = mx.zeros((1, 3, 16))
    # The grammar permits only a token outside the selector's top-k.
    def outside_top_k(_tokens, logits):
        ids = mx.arange(logits.shape[-1])
        return mx.where(ids == mx.argmin(logits, axis=-1), logits, -mx.inf)

    tokens, laws = draft.draft_distributions(
        [7], hidden, draft.make_cache(), 2, [SimpleNamespace(sample=np.argmax)], [1.0],
        logits_processors=[[outside_top_k]], processor_histories=[[1, 2]],
    )
    assert tokens == [[]] and laws == [[]]


@pytest.mark.parametrize("cap", [-1.0, float("inf"), float("nan"), True, "3.0"])
def test_after_correction_requires_finite_positive_numeric_cap(cap):
    with pytest.raises(ValueError, match="final_logit_softcapping"):
        DFlash2Config.from_dict(_hf_config(selector_softcap_mode="after_correction",
                                           final_logit_softcapping=cap))


# --- best-route default (Pierre 2026-11-25: "Yes dflash2") -----------------

INKLING2_FP = "440956c8d7f103c1b409b197f150faff54ae2e4233f1e64398f335745a93ff9f"


def _inkling2_entry():
    from mlx2 import best_route_defaults

    table = best_route_defaults.load_table()
    return best_route_defaults.find_entry(table, "Inkling2Adapter", INKLING2_FP)


def test_best_route_entry_is_the_tested_one_lane_dflash2_chain():
    entry = _inkling2_entry()
    assert entry["status"] == "selected_unqualified"
    spec = entry["default"]
    assert spec["kind"] == "ast" and spec["max_lanes"] == 1
    # Exactly the policy GSM8K measured (b4e6be2eb): depth 7, composition off.
    # APCv2 interior checkpoints joined the default (Pierre 2026-11-26): a
    # server-owned key, consumed before the adapter sees the policy.
    assert spec["execution_policy"] == {"num_draft": 7, "proposal_composition": False,
                                        "apc_interior_checkpoints": "auto"}
    from mlx2.serving import SERVER_OWNED_EXECUTION_POLICY_KEYS

    adapter_keys = set(spec["execution_policy"]) - SERVER_OWNED_EXECUTION_POLICY_KEYS
    assert adapter_keys | {"draft_model"} <= inkling2_dflash2.ALLOWED_POLICY_KEYS
    assert spec["drafter"]["architectures"] == ["DFlash2DraftModel"]
    assert spec["drafter"]["config_sha256"] == [inkling2_dflash2.CONFIG_SHA256]
    assert entry["fallback"] == [{"name": "adapter_default", "kind": "adapter_default"}]
    assert entry["checkpoint"] == f"Inkling-2-Mini@sha256:{INKLING2_FP}"
    # An unmeasured Inkling 2 checkpoint has no entry (no family fallback).
    from mlx2 import best_route_defaults

    assert best_route_defaults.find_entry(best_route_defaults.load_table(), "Inkling2Adapter", "0" * 64) is None


def test_auto_pairing_uses_the_adapter_inspector_and_adds_no_pins(monkeypatch):
    from mlx2 import best_route_defaults
    from mlx2.adapters import drafter_pairing, inkling2

    seen = {}
    monkeypatch.setattr(inkling2, "inspect_artifact", lambda path: {
        "identity": {"path": "/target", "fingerprint": "t" * 64}, "text_config": {"k": 1}})

    def inspect(path, **kwargs):
        seen.update(kwargs, path=path)
        return {"draft_revision": "d" * 64}

    monkeypatch.setattr(inkling2_dflash2, "inspect_drafter", inspect)
    pairing = drafter_pairing.pair_drafter(
        {"adapter": "Inkling2Adapter", "path": "/target"},
        {"path": "/draft", "architectures": ["DFlash2DraftModel"], "name": "Inkling-2-Mini-DFlash2"},
        target_revision=False)
    assert seen == {"path": "/draft", "target_path": "/target", "text_config": {"k": 1},
                    "target_fingerprint": "t" * 64}  # default embed_norm (on)
    assert pairing["pins"] == {} and pairing["identity"] == "d" * 64
    assert pairing["draft"] == "Inkling-2-Mini-DFlash2@sha256:" + "d" * 64
    assert pairing["execution_policy"] == {"draft_model": "/draft", "num_draft": 7}
    # The seam merges the entry policy and no pins: the route admits the result.
    record = {"applied": True, "route": {"execution_policy": {"num_draft": 7, "proposal_composition": False}},
              "drafter": {"path": "/draft", "pins": pairing["pins"]}}
    policy = best_route_defaults.finish_best_route_default(
        SimpleNamespace(route_default_selection=record), {"draft_model": "/draft"})
    assert policy == {"num_draft": 7, "proposal_composition": False, "draft_model": "/draft"}
    # The adapter's own route selection admits it (not only the key constant).
    from mlx2.adapters.inkling2 import Inkling2Adapter

    monkeypatch.setattr(inkling2_dflash2, "is_dflash2_artifact", lambda path: path == "/draft")
    adapter = object.__new__(Inkling2Adapter)
    assert adapter._select_dflash2_route(dict(policy)) is True
    assert adapter.dflash2_embed_norm is True


def test_auto_pairing_refuses_other_architectures():
    from mlx2.adapters import drafter_pairing

    with pytest.raises(ValueError, match="pinned Inkling-2-Mini-DFlash2"):
        drafter_pairing.pair_drafter(
            {"adapter": "Inkling2Adapter", "path": "/target"},
            {"path": "/draft", "architectures": ["DSparkDraftModel"], "name": "x"},
            target_revision=False)


def test_best_route_identity_is_the_real_drafter_revision():
    """Skips without the local artifacts; otherwise the table's evidence
    identity must be what the adapter's inspector computes today."""
    target = Path.home() / "mlx-models/Inkling-2-Mini"
    draft = Path.home() / "mlx-models/Inkling-2-Mini-DFlash2"
    if not (target / "config.json").exists() or not (draft / "config.json").exists():
        pytest.skip("Inkling 2 Mini or its DFlash2 drafter is not on this host")
    from mlx2.adapters import drafter_pairing

    pairing = drafter_pairing.pair_drafter(
        {"adapter": "Inkling2Adapter", "path": str(target)},
        {"path": str(draft), "architectures": ["DFlash2DraftModel"], "name": "Inkling-2-Mini-DFlash2"},
        target_revision=False)
    assert pairing["identity"] in _inkling2_entry()["default"]["drafter"]["identities"]


def test_no_flag_server_selects_dflash2_only_at_one_lane():
    """The server seam on the real artifacts (skips without them)."""
    target = Path.home() / "mlx-models/Inkling-2-Mini"
    draft = Path.home() / "mlx-models/Inkling-2-Mini-DFlash2"
    if not (target / "config.json").exists() or not (draft / "config.json").exists():
        pytest.skip("Inkling 2 Mini or its DFlash2 drafter is not on this host")
    from mlx2 import server as S
    from mlx2.adapters.registry import inspect_model
    from mlx2.best_route_defaults import finish_best_route_default

    def resolve(*flags):
        args = S.build_parser().parse_args(["--model", str(target), *flags])
        resolution = inspect_model(args.model)
        S.apply_adapter_lane_default(args, resolution)
        S.apply_best_route_default(args, resolution, policy=None)
        policy, resolution = S.apply_external_mtp_head(args, None, resolution)
        policy = S.apply_ast_invocation(args, policy, resolution)
        policy = finish_best_route_default(args, policy)
        selection = S.resolve_route_selection(args, policy, resolution)
        return args.route_default_selection["status"], selection.route, policy

    status, route, policy = resolve("--max-lanes", "1")
    assert (status, route) == ("selected_unqualified", "external_draft")
    assert policy == {"num_draft": 7, "proposal_composition": False, "draft_model": str(draft),
                      "apc_interior_checkpoints": "auto"}
    assert resolve()[:2] == ("width_keyed_adapter_default", "ordinary")
    assert resolve("--max-lanes", "1", "--ordinary")[:2] == ("explicit_route", "ordinary")
    assert resolve("--max-lanes", "1", "--no-auto-route")[0] == "opted_out"


def _greedy_ordinary(target, prompt, count):
    cache, tokens, out = target.make_cache(), list(prompt), []
    for step in range(count):
        logits = target(mx.array([tokens if step == 0 else [tokens[-1]]]), cache=cache)
        tokens.append(int(mx.argmax(logits[0, -1]).item()))
        out.append(tokens[-1])
    return out


def _lane_visits(target):
    return sum(layer.self_attn.__dict__.get("segmented_lane_counts", {}).get("visits", 0)
               for layer in target.model.layers)


@pytest.mark.parametrize("lengths,stagger", [((24, 9), False), ((30, 12, 20), False),
                                             ((24, 9, 17), True)])
def test_tiny_inkling2_engine_serves_several_lanes_greedy_exact(lengths, stagger):
    """Several lanes verify in one batched round (wrapped and unwrapped rings,
    lanes joining mid-flight); each lane's greedy output equals ordinary decode."""
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    target, draft, _ = _tiny_inkling_pair()
    mx.random.seed(7)
    prompts = [[int(t) for t in mx.random.randint(1, 400, (n,)).tolist()] for n in lengths]
    count = 14
    expect = [_greedy_ordinary(target, p, count) for p in prompts]
    batch = ExternalDraftBatchGenerator(target, draft_model=draft, binding="inkling2-dflash2-test",
                                        num_draft=7, prefill_step_size=16,
                                        completion_batch_size=4)
    try:
        first = prompts[:1] if stagger else prompts
        uids = list(batch.insert(first, max_tokens=[count] * len(first)))
        got = {}
        for step in range(800):
            if stagger and step == 3:
                uids += batch.insert(prompts[1:], max_tokens=[count] * (len(prompts) - 1))
            _, responses = batch.next()
            for r in responses:
                got.setdefault(r.uid, []).append(r.token)
            if len(uids) == len(prompts) and not batch.lanes:
                break
        assert [got[u] for u in uids] == expect
        assert _lane_visits(target) > 0  # some rounds really verified lanes together
        assert batch.scheduler_stats.get(
            "external_cohort_target_batched_standard_verify_rounds", 0) > 0
    finally:
        batch.close()


@pytest.mark.parametrize("lanes", [2, 16])
def test_dflash2_route_accepts_more_than_one_lane_and_dspark_still_refuses(lanes, monkeypatch):
    from mlx2.adapters.inkling2 import Inkling2Adapter

    adapter = object.__new__(Inkling2Adapter)
    adapter.draft_model = object()
    adapter.dflash2_route = True
    monkeypatch.setattr(Inkling2Adapter, "bind_prefill_step", lambda self, step: None)
    monkeypatch.setattr(Inkling2Adapter, "_external_execution_config",
                        lambda self, *, max_lanes, prefill_step: {"backend": "external_draft",
                                                                  "segment_aware_cohort_size": max_lanes})
    config = adapter.execution_config(max_lanes=lanes, prefill_step=512)
    assert config["backend"] == "external_draft" and config["segment_aware_cohort_size"] == lanes
    assert "max_lanes" not in adapter.__dict__.get("dflash2_refusals", {})
    adapter.dflash2_route = False  # the DSpark route drafts one lane at a time
    with pytest.raises(ValueError, match="B1 only"):
        adapter.execution_config(max_lanes=lanes, prefill_step=512)


class _LaneKey:
    def __init__(self, seed):
        self.key = [seed, 7]


def _serve_sampled(target, draft, prompts, seeds, count=16):
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    batch = ExternalDraftBatchGenerator(target, draft_model=draft, binding="inkling2-dflash2-test",
                                        num_draft=7, prefill_step_size=16, completion_batch_size=4)
    try:
        uids = batch.insert(prompts, max_tokens=[count] * len(prompts),
                            sampling_configs=[{"sampling_temp": 1.0}] * len(prompts),
                            lane_rngs=[_LaneKey(s) for s in seeds])
        got = {}
        for _ in range(2000):
            _, responses = batch.next()
            for r in responses:
                got.setdefault(r.uid, []).append(r.token)
            if not batch.lanes:
                break
        return [got[u] for u in uids], dict(batch.scheduler_stats)
    finally:
        batch.close()


def test_tiny_inkling2_engine_sampled_lanes_equal_each_lane_served_alone():
    """Per-lane keyed draws: a sampled lane's tokens do not depend on its co-lanes."""
    target, draft, _ = _tiny_inkling_pair()
    mx.random.seed(9)
    prompts = [[int(t) for t in mx.random.randint(1, 400, (n,)).tolist()] for n in (24, 11, 30)]
    seeds = [11, 22, 33]
    together, stats = _serve_sampled(target, draft, prompts, seeds)
    assert stats.get("external_cohort_target_batched_standard_verify_rounds", 0) > 0
    alone = [_serve_sampled(target, draft, [p], [s])[0][0] for p, s in zip(prompts, seeds)]
    assert together == alone


@pytest.mark.parametrize("kernels", [{}, {"verify_deferred_commit": True},
                                     {"sink_sdpa": True, "sink_slot": True, "fuse_controllers": True,
                                      "fuse_residual": True, "fuse_sconv": True,
                                      "fuse_attention": True, "verify_deferred_commit": True}])
@pytest.mark.parametrize("lengths,stagger", [((30, 12, 20), False), ((40, 33, 5, 60), False),
                                             ((24, 9, 17), True)])
def test_tiny_inkling2_engine_partial_accepts_across_lanes_greedy_exact(kernels, lengths, stagger):
    """An oracle drafter proposes each lane's true greedy continuation for a
    number of positions that varies by round, then wrong tokens: lanes accept
    different prefixes in the same batched round (multi-row commits, partial
    ring trims after a wrap). Output must equal ordinary decode."""
    import inspect

    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    target, draft, _ = _tiny_inkling_pair()
    if kernels:
        target.configure_kernels(**kernels)
    mx.random.seed(7)
    prompts = [[int(t) for t in mx.random.randint(1, 400, (n,)).tolist()] for n in lengths]
    count, num_draft = 30, 7
    expect = [_greedy_ordinary(target, p, count + num_draft + 2) for p in prompts]
    full = [p + e for p, e in zip(prompts, expect)]
    rounds = [0]
    original = draft.draft_distributions

    def oracle(anchors, hidden, cache, n, rngs, temps, **kw):
        tokens, laws = original(anchors, hidden, cache, n, rngs, temps, **kw)
        # The generator's local cohort names each row's lane (history + anchor).
        lanes = inspect.currentframe().f_back.f_locals.get("lanes")
        assert lanes is not None and len(lanes) == len(tokens)
        out_tokens, out_laws = [], []
        for lane, row in zip(lanes, tokens):
            seq = list(lane.history) + [lane.anchor]
            truth = next(f for f in full if f[: len(seq)] == seq)[len(seq) : len(seq) + len(row)]
            good = (rounds[0] * 3) % (len(row) + 1)
            row = [truth[j] if j < good else (truth[j] + 1) % 500 for j in range(len(row))]
            out_tokens.append(row)
            one_hot = []
            for token, law in zip(row, laws[len(out_laws)]):
                q = np.zeros_like(np.asarray(law))
                q[token] = 1.0
                one_hot.append(q)
            out_laws.append(one_hot)
        return out_tokens, out_laws

    draft.draft_distributions = oracle
    batch = ExternalDraftBatchGenerator(target, draft_model=draft, binding="inkling2-dflash2-test",
                                        num_draft=num_draft, prefill_step_size=16,
                                        completion_batch_size=4)
    try:
        first = prompts[:1] if stagger else prompts
        uids = list(batch.insert(first, max_tokens=[count] * len(first)))
        got = {}
        for step in range(3000):
            if stagger and step == 2:
                uids += batch.insert(prompts[1:], max_tokens=[count] * (len(prompts) - 1))
            rounds[0] += 1
            _, responses = batch.next()
            for r in responses:
                got.setdefault(r.uid, []).append(r.token)
            if len(uids) == len(prompts) and not batch.lanes:
                break
        stats = batch.scheduler_stats
        assert [got[u] for u in uids] == [e[:count] for e in expect]
        assert stats.get("accepted_proposals", 0) > 0
        assert stats.get("external_cohort_target_batched_standard_verify_rounds", 0) > 0
    finally:
        batch.close()


def test_tiny_inkling2_engine_sampled_lanes_complete():
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    target, draft, _ = _tiny_inkling_pair()
    mx.random.seed(9)
    prompts = [[int(t) for t in mx.random.randint(1, 400, (n,)).tolist()] for n in (24, 11)]
    batch = ExternalDraftBatchGenerator(target, draft_model=draft, binding="inkling2-dflash2-test",
                                        num_draft=7, prefill_step_size=16, completion_batch_size=4)
    try:
        uids = batch.insert(prompts, max_tokens=[12, 12],
                            sampling_configs=[{"sampling_temp": 1.0}] * 2)
        got = {}
        for _ in range(800):
            _, responses = batch.next()
            for r in responses:
                got.setdefault(r.uid, []).append(r.token)
            if not batch.lanes:
                break
        assert [len(got[u]) for u in uids] == [12, 12]
        assert all(0 <= t < 500 for u in uids for t in got[u])
        assert _lane_visits(target) > 0
    finally:
        batch.close()


def test_multilane_draft_cap_option(tmp_path, monkeypatch):
    from mlx2.adapters import inkling2_dflash2 as d2
    from mlx2.adapters.external_draft_policy import ExternalDraftAdapterMixin
    from mlx2.adapters.inkling2 import Inkling2Adapter

    path = tmp_path / "draft"
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"architectures": ["DFlash2DraftModel"]}))
    adapter = object.__new__(Inkling2Adapter)
    external = {"draft_model": str(path), "num_draft": 7, d2.MULTILANE_DRAFT_CAP_OPTION: 3}
    assert adapter._select_dflash2_route(external)
    assert adapter.dflash2_multilane_draft_cap == 3
    assert d2.MULTILANE_DRAFT_CAP_OPTION not in external
    seen = {}
    monkeypatch.setattr(ExternalDraftAdapterMixin, "create_external_batch",
                        lambda self, **kwargs: seen.update(kwargs))
    adapter.create_external_batch(completion_batch_size=4)
    assert seen["multilane_draft_cap"] == 3
    unset = object.__new__(Inkling2Adapter)
    assert unset._select_dflash2_route({"draft_model": str(path), "num_draft": 7})
    seen.clear()
    unset.create_external_batch()
    assert "multilane_draft_cap" not in seen
    for bad in (0, 2, 8, 3.0, True):  # the DFlash2 drafter proposes at least 3
        with pytest.raises((ValueError, TypeError)):
            object.__new__(Inkling2Adapter)._select_dflash2_route(
                {"draft_model": str(path), "num_draft": 7, d2.MULTILANE_DRAFT_CAP_OPTION: bad})
    with pytest.raises(ValueError, match="requires the DFlash2 route"):
        object.__new__(Inkling2Adapter)._select_dflash2_route(
            {"draft_model": "embedded_dspark", d2.MULTILANE_DRAFT_CAP_OPTION: 2})


def test_tiny_engine_multilane_cap_keeps_greedy_exact_and_caps_rounds():
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    target, draft, _ = _tiny_inkling_pair()
    mx.random.seed(7)
    prompts = [[int(t) for t in mx.random.randint(1, 400, (n,)).tolist()] for n in (24, 9, 17)]
    count = 14
    expect = [_greedy_ordinary(target, p, count) for p in prompts]
    batch = ExternalDraftBatchGenerator(target, draft_model=draft, binding="inkling2-dflash2-test",
                                        num_draft=7, prefill_step_size=16,
                                        completion_batch_size=4, multilane_draft_cap=3)
    try:
        uids = list(batch.insert(prompts, max_tokens=[count] * len(prompts)))
        got = {}
        for _ in range(800):
            _, responses = batch.next()
            for r in responses:
                got.setdefault(r.uid, []).append(r.token)
            if not batch.lanes:
                break
        assert [got[u] for u in uids] == expect
        assert batch.scheduler_stats.get("external_multilane_draft_cap_rounds", 0) > 0
    finally:
        batch.close()


def test_multilane_dflash2_keeps_lanes_in_lockstep(tmp_path, monkeypatch):
    from mlx2.adapters import inkling2_dflash2 as d2
    from mlx2.adapters.external_draft_policy import ExternalDraftAdapterMixin
    from mlx2.adapters.inkling2 import Inkling2Adapter

    path = tmp_path / "draft"
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"architectures": ["DFlash2DraftModel"]}))
    seen = {}
    monkeypatch.setattr(ExternalDraftAdapterMixin, "create_external_batch",
                        lambda self, **kwargs: seen.update(kwargs))
    adapter = object.__new__(Inkling2Adapter)
    assert adapter._select_dflash2_route({"draft_model": str(path), "num_draft": 7})
    adapter.create_external_batch(completion_batch_size=16)
    assert seen["ready_drain"] == "all"
    seen.clear()
    adapter.create_external_batch(completion_batch_size=1)  # the one-lane default: engine's own
    assert "ready_drain" not in seen
    opted = object.__new__(Inkling2Adapter)
    assert opted._select_dflash2_route({"draft_model": str(path), "num_draft": 7,
                                        d2.READY_DRAIN_OPTION: "one"})
    seen.clear()
    opted.create_external_batch(completion_batch_size=16)
    assert seen["ready_drain"] == "one"
    with pytest.raises(ValueError, match="'one' or 'all'"):
        object.__new__(Inkling2Adapter)._select_dflash2_route(
            {"draft_model": str(path), d2.READY_DRAIN_OPTION: "some"})
    with pytest.raises(ValueError, match="requires the DFlash2 route"):
        object.__new__(Inkling2Adapter)._select_dflash2_route(
            {"draft_model": "embedded_dspark", d2.READY_DRAIN_OPTION: "all"})


def test_tiny_engine_ready_drain_all_keeps_full_cohorts_greedy_exact():
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    target, draft, _ = _tiny_inkling_pair()
    mx.random.seed(7)
    prompts = [[int(t) for t in mx.random.randint(1, 400, (n,)).tolist()] for n in (24, 9, 17, 30)]
    count = 14
    expect = [_greedy_ordinary(target, p, count) for p in prompts]
    batch = ExternalDraftBatchGenerator(target, draft_model=draft, binding="inkling2-dflash2-test",
                                        num_draft=7, prefill_step_size=16,
                                        completion_batch_size=4, ready_drain="all")
    try:
        uids = list(batch.insert(prompts, max_tokens=[count] * len(prompts)))
        got = {}
        for _ in range(800):
            _, responses = batch.next()
            for r in responses:
                got.setdefault(r.uid, []).append(r.token)
            if not batch.lanes:
                break
        assert [got[u] for u in uids] == expect
        assert batch.scheduler_stats.get("external_decode_draining_lanes_skipped", 0) == 0
    finally:
        batch.close()
