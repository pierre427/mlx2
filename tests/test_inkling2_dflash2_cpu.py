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
