"""Vendor sampling defaults: merge semantics, per-adapter declarations,
penalty correctness against a reference, speculative exactness under
penalties, and receipt recording.  CPU only; no artifact loads."""

import json

import mlx.core as mx
import numpy as np
import pytest

from mlx2.sampling_defaults import (
    LEGACY_FALLBACK,
    NEUTRAL,
    SAMPLING_FIELDS,
    XING4_SAMPLING,
    SamplingDefaults,
    VendorSampling,
    generation_config_drift,
    resolve_sampling,
    vendor_sampling,
)
from mlx2.runtime.sample_utils import make_logits_processors


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _adapter_classes():
    from mlx2.adapters.flash_next import FlashNextAdapter
    from mlx2.adapters.muse_glimmer import MuseGlimmerAdapter
    from mlx2.adapters.north_mini_code import NorthMiniCodeAdapter
    from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter
    from mlx2.adapters.laguna_xs21 import LagunaXS21Adapter
    from mlx2.adapters.mlx_vlm import Gemma3nAdapter, MiniCPMOAdapter
    from mlx2.adapters.xing import XingAdapter

    return {
        "qwen4_exp": FlashNextAdapter,
        "qwen3_5": Qwen3827BAdapter,
        "qwen3_5_moe": Qwen3635BA3BAdapter,
        "muse_glimmer": MuseGlimmerAdapter,
        "cohere2_moe": NorthMiniCodeAdapter,
        "xing4_0": XingAdapter,
        "laguna": LagunaXS21Adapter,
        "gemma3n": Gemma3nAdapter,
        "minicpmo": MiniCPMOAdapter,
    }


QWEN36 = lambda: _adapter_classes()["qwen3_5_moe"].sampling_defaults  # noqa: E731


# --- merge semantics --------------------------------------------------------


def test_unset_fields_take_the_selected_profile_and_record_sources():
    effective, record = resolve_sampling({"messages": []}, XING4_SAMPLING, thinking=None)
    assert effective == {
        "temperature": 1.0, "top_p": 0.95, "top_k": 0, "min_p": 0.0,
        "repetition_penalty": 1.05, "presence_penalty": 0.0, "frequency_penalty": 0.0,
    }
    assert record["profile"] == "general" and record["profile_reason"] == "general"
    assert record["explicit"] == []
    assert record["applied"] == {"temperature": 1.0, "top_p": 0.95, "repetition_penalty": 1.05}
    assert set(record["sources"].values()) == {"generation_config.json"}
    assert record["fallback_kind"] == "neutral"
    assert record["fallback"] == {
        "top_k": 0, "min_p": 0.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
    }
    assert json.loads(json.dumps(record)) == record  # receipt-serializable


@pytest.mark.parametrize(
    "explicit",
    [
        {"temperature": 0},
        {"temperature": 0, "repetition_penalty": 1.0},
        {"top_p": 1.0, "presence_penalty": 0.0},
        {"temperature": 0.3, "top_p": 0.5, "top_k": 7, "min_p": 0.1,
         "repetition_penalty": 1.2, "presence_penalty": -0.5, "frequency_penalty": 0.25},
    ],
)
def test_explicit_request_values_always_win_including_zero(explicit):
    effective, record = resolve_sampling(explicit, QWEN36(), thinking=False)
    for name, value in explicit.items():
        assert effective[name] == value and type(effective[name]) is type(value)
        assert name not in record["applied"] and name not in record["fallback"]
    assert record["explicit"] == [name for name in SAMPLING_FIELDS if name in explicit]
    instruct = QWEN36().profiles["instruct"].values()
    for name, value in instruct.items():
        if name not in explicit:
            assert effective[name] == value and record["applied"][name] == value


def test_no_vendor_declaration_keeps_the_historical_engine_fallback():
    effective, record = resolve_sampling({"top_k": 3}, None, thinking=True)
    assert effective == {**LEGACY_FALLBACK, "top_k": 3}
    assert record["profile"] is None and record["applied"] == {}
    assert record["fallback_kind"] == "legacy"
    with pytest.raises(ValueError, match="declares no sampling profiles"):
        resolve_sampling({"sampling_profile": "coding"}, None, thinking=None)


def test_thinking_mode_selects_the_matching_vendor_profile():
    vendor = QWEN36()
    on, record_on = resolve_sampling({}, vendor, thinking=True)
    off, record_off = resolve_sampling({}, vendor, thinking=False)
    unknown, record_unknown = resolve_sampling({}, vendor, thinking=None)
    assert (record_on["profile"], record_on["profile_reason"]) == ("thinking", "thinking")
    assert (record_off["profile"], record_off["profile_reason"]) == ("instruct", "non_thinking")
    # A raw completion's mode is unknown: the vendor's general (default-mode)
    # profile, which for Qwen is thinking.
    assert (record_unknown["profile"], record_unknown["profile_reason"]) == ("thinking", "general")
    assert on["temperature"] == 1.0 and on["top_p"] == 0.95 and on["presence_penalty"] == 1.5
    assert off["temperature"] == 0.7 and off["top_p"] == 0.8 and off["presence_penalty"] == 1.5
    assert on["top_k"] == off["top_k"] == 20
    assert unknown == on


def test_requested_profile_wins_over_mode_and_unknown_names_fail_closed():
    effective, record = resolve_sampling(
        {"sampling_profile": "coding"}, QWEN36(), thinking=False
    )
    assert record["profile"] == "coding" and record["profile_reason"] == "requested"
    assert effective["temperature"] == 0.6 and effective["presence_penalty"] == 0.0
    with pytest.raises(ValueError, match="unknown sampling_profile"):
        resolve_sampling({"sampling_profile": "creative"}, QWEN36(), thinking=None)
    with pytest.raises(ValueError, match="sampling_profile must be"):
        resolve_sampling({"sampling_profile": ""}, QWEN36(), thinking=None)


def test_xing_profiles_general_for_both_modes_and_coding_agent_by_request():
    for thinking in (True, False, None):
        effective, record = resolve_sampling({}, XING4_SAMPLING, thinking=thinking)
        assert record["profile"] == "general"
        assert (effective["temperature"], effective["top_p"], effective["repetition_penalty"]) == (1.0, 0.95, 1.05)
    for name in ("coding", "agent"):
        effective, record = resolve_sampling({"sampling_profile": name}, XING4_SAMPLING, thinking=True)
        assert (effective["temperature"], effective["top_p"], effective["repetition_penalty"]) == (0.8, 0.95, 1.05)
        assert "model card" in record["sources"]["temperature"]


def test_declarations_validate_their_values_and_citations():
    with pytest.raises(ValueError, match="source"):
        SamplingDefaults(temperature=1.0)
    with pytest.raises(ValueError, match="at least one field"):
        SamplingDefaults(source="x")
    with pytest.raises(ValueError, match="top_p"):
        SamplingDefaults(top_p=1.5, source="x")
    with pytest.raises(ValueError, match="not declared"):
        VendorSampling({"general": SamplingDefaults(top_p=0.9, source="x")}, thinking="thinking")
    single = SamplingDefaults(top_p=0.9, source="x")
    assert vendor_sampling(type("A", (), {"sampling_defaults": single})()).profiles["general"] is single
    assert vendor_sampling(object()) is None


def test_request_validation_accepts_the_profile_extension():
    from mlx2.server import validate_request

    body = {"messages": [{"role": "user", "content": "hi"}], "sampling_profile": "coding"}
    assert validate_request(body)["sampling_profile"] == "coding"
    with pytest.raises(ValueError, match="sampling_profile"):
        validate_request({**body, "sampling_profile": 3})


def test_sampling_profile_is_generation_only_for_prompt_caching():
    from mlx2.serving import HostPromptCache

    base = {"messages": [{"role": "user", "content": "hi"}]}
    assert HostPromptCache.key(base) == HostPromptCache.key({**base, "sampling_profile": "coding"})


# --- per-adapter declarations ------------------------------------------------

EXPECTED = {
    # model_type: {profile: values}
    "qwen4_exp": {
        "thinking": dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, repetition_penalty=1.0, presence_penalty=0.0),
        "instruct": dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, repetition_penalty=1.0, presence_penalty=1.5),
    },
    "qwen3_5": {
        "thinking": dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, repetition_penalty=1.0, presence_penalty=0.0),
        "instruct": dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, repetition_penalty=1.0, presence_penalty=1.5),
    },
    "qwen3_5_moe": {
        "thinking": dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, repetition_penalty=1.0, presence_penalty=1.5),
        "coding": dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, repetition_penalty=1.0, presence_penalty=0.0),
        "instruct": dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, repetition_penalty=1.0, presence_penalty=1.5),
    },
    "muse_glimmer": {"general": dict(temperature=1.0, top_p=0.95, top_k=64)},
    "cohere2_moe": {"general": dict(temperature=1.0, top_p=0.95)},
    "laguna": {"general": dict(temperature=1.0, top_p=1.0, top_k=20, min_p=0.0)},
    "xing4_0": {
        "general": dict(temperature=1.0, top_p=0.95, repetition_penalty=1.05),
        "coding": dict(temperature=0.8, top_p=0.95, repetition_penalty=1.05),
        "agent": dict(temperature=0.8, top_p=0.95, repetition_penalty=1.05),
    },
    "gemma3n": {"general": dict(top_p=0.95, top_k=64)},
    "minicpmo": {"general": dict(temperature=0.5)},
}


def test_every_target_adapter_declares_cited_vendor_defaults():
    from mlx2.adapters.registry import _RESOLVERS

    classes = _adapter_classes()
    assert set(EXPECTED) <= set(_RESOLVERS) and set(_RESOLVERS) - set(EXPECTED) == {"muse_glimmer_text"}
    for model_type, profiles in EXPECTED.items():
        vendor = vendor_sampling(classes[model_type])
        assert vendor is not None and vendor.model, model_type
        assert {name: p.values() for name, p in vendor.profiles.items()} == profiles, model_type
        for profile in vendor.profiles.values():
            assert all(profile.source_of(name) for name in profile.values())
        assert json.loads(json.dumps(vendor.as_dict()))["profiles"]
    qwen = vendor_sampling(classes["qwen3_5"])
    # generation_config.json supplies the thinking-mode core fields.
    assert {qwen.profiles["thinking"].source_of(n) for n in ("temperature", "top_p", "top_k")} == {"generation_config.json"}
    assert "model card" in qwen.profiles["thinking"].source_of("presence_penalty")
    for model_type in ("qwen4_exp", "qwen3_5", "qwen3_5_moe"):
        vendor = vendor_sampling(classes[model_type])
        assert (vendor.general, vendor.thinking, vendor.non_thinking) == ("thinking", "thinking", "instruct")


def test_generation_config_drift_is_reported_not_applied(tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps(
        {"do_sample": True, "temperature": 1.0, "top_p": 0.9, "top_k": 20}
    ))
    vendor = vendor_sampling(_adapter_classes()["qwen3_5"])
    drift = generation_config_drift(tmp_path, vendor)
    assert drift["mismatches"] == {"top_p": {"declared": 0.95, "artifact": 0.9}}
    (tmp_path / "generation_config.json").write_text(json.dumps({"do_sample": False}))
    muse = vendor_sampling(_adapter_classes()["muse_glimmer"])
    assert generation_config_drift(tmp_path, muse)["mismatches"] == {
        "do_sample": {"declared": True, "artifact": False}
    }
    assert generation_config_drift(tmp_path / "missing", muse)["generation_config"] is None


# --- penalty correctness vs a reference -------------------------------------


def _reference_penalties(logits, history, prompt_len, *, rep, presence, frequency):
    """HF RepetitionPenaltyLogitsProcessor over prompt+generated, then the
    OpenAI/vLLM presence/frequency penalties over generated tokens only."""
    out = np.array(logits, dtype=np.float64)
    if rep != 1.0:
        for token in set(history):
            score = out[token]
            out[token] = score * rep if score < 0 else score / rep
    generated = history[prompt_len:]
    for token in set(generated):
        out[token] -= presence + frequency * generated.count(token)
    return out


@pytest.mark.parametrize("rep,presence,frequency", [(1.3, 0.0, 0.0), (0.8, 0.0, 0.0), (1.05, 0.7, 0.2), (2.0, -0.5, 0.0)])
def test_penalties_match_the_reference_semantics(rep, presence, frequency):
    rng = np.random.default_rng(5)
    logits = rng.normal(size=16).astype(np.float32)
    prompt = [1, 2, 2, 3, 9]
    generated = [3, 4, 4, 4, 11]
    history = prompt + generated
    processors = make_logits_processors(
        repetition_penalty=rep if rep != 1.0 else None, repetition_context_size=0,
        presence_penalty=presence, presence_context_size=0,
        frequency_penalty=frequency, frequency_context_size=0,
        penalty_generation_start=len(prompt),
    )
    value = mx.array(logits)[None]
    for processor in processors:
        value = processor(mx.array(history, mx.uint32), value)
    expected = _reference_penalties(logits, history, len(prompt), rep=rep, presence=presence, frequency=frequency)
    np.testing.assert_allclose(np.array(value)[0], expected, rtol=1e-6, atol=1e-6)
    # Prompt-only history: presence/frequency are inert, repetition is not.
    value = mx.array(logits)[None]
    for processor in processors:
        value = processor(mx.array(prompt, mx.uint32), value)
    expected = _reference_penalties(logits, prompt, len(prompt), rep=rep, presence=presence, frequency=frequency)
    np.testing.assert_allclose(np.array(value)[0], expected, rtol=1e-6, atol=1e-6)


# --- speculative exactness under penalties ----------------------------------

PENALTIES = dict(rep=1.6, presence=0.8, frequency=0.3)


def _processors(prompt_len, *, rep, presence, frequency):
    return make_logits_processors(
        repetition_penalty=rep, repetition_context_size=0,
        presence_penalty=presence, presence_context_size=0,
        frequency_penalty=frequency, frequency_context_size=0,
        penalty_generation_start=prompt_len,
    )


def _greedy_reference(next_logits, prompt, steps, processors):
    history = list(prompt)
    for _ in range(steps):
        logits = next_logits(history)[None]
        for processor in processors:
            logits = processor(mx.array(history, mx.uint32), logits)
        history.append(int(mx.argmax(logits[0]).item()))
    return history[len(prompt):]


class _TableModel:
    """Next-token logits depend on the last token only; close margins so the
    penalties change the greedy path."""

    vocabulary = 8

    def __init__(self):
        rng = np.random.default_rng(17)
        table = rng.normal(scale=0.4, size=(self.vocabulary, self.vocabulary))
        for last, nxt in ((1, 2), (2, 1), (3, 4), (4, 3)):
            table[last, nxt] += 1.0  # a 1,2,1,2 / 3,4,3,4 habit that PLD will draft
        self.table = mx.array(table.astype(np.float32))

    def __call__(self, tokens, *, cache):
        values = tokens.astype(mx.float32)[:, None, :, None]
        cache[0].update_and_fetch(values, values)
        return self.table[tokens]

    def next_logits(self, history):
        return self.table[history[-1]]


def _run_pld(model, prompt, maximum, processors):
    from mlx2.runtime.models.cache import KVCache
    from mlx2.runtime.pld import PromptLookupBatchGenerator

    generator = PromptLookupBatchGenerator(
        model, prefill_step_size=16,
        prompt_lookup={"num_draft": 3, "ngram_min": 1, "ngram_max": 2},
    )
    generator.insert([prompt], max_tokens=[maximum], caches=[[KVCache()]], logits_processors=[processors])
    emitted, final = [], None
    for _ in range(200):
        _, responses = generator.next()
        emitted.extend(response.token for response in responses)
        final = next((r for r in responses if r.finish_reason), None)
        if final is not None:
            return emitted, final
    raise AssertionError("prompt lookup did not finish")


def test_prompt_lookup_verification_is_exact_under_penalties():
    model = _TableModel()
    prompt = [1, 2, 1, 2, 3, 4, 3, 4, 1, 2]
    steps = 12
    plain = _greedy_reference(model.next_logits, prompt, steps, [])
    reference = _greedy_reference(model.next_logits, prompt, steps, _processors(len(prompt), **PENALTIES))
    assert reference != plain  # the penalties matter on this path
    emitted, final = _run_pld(model, prompt, steps, _processors(len(prompt), **PENALTIES))
    assert emitted == reference
    # Drafts were verified (and some accepted past a penalized prefix), not
    # bypassed through an ordinary fallback.
    assert final.speculative_receipt["proposed"] > 0
    assert final.speculative_receipt["accepted"] > 0
    unpenalized, _ = _run_pld(model, prompt, steps, [])
    assert unpenalized == plain


def test_external_draft_verification_is_exact_under_penalties():
    from test_external_dflash2_cpu import drain, generator, tiny

    m, d = tiny()
    prompts = [[1, 2, 3, 4, 5, 1, 2], [7, 7, 3]]
    steps = 8

    def next_logits(history):
        return m(mx.array([history]), cache=m.make_cache())[0, -1]

    b = generator(m, d)
    ids = b.insert(
        prompts, max_tokens=[steps] * 2, sampling_configs=[{"sampling_temp": 0}] * 2,
        logits_processors=[_processors(len(p), **PENALTIES) for p in prompts],
    )
    got, _ = drain(b)
    changed = False
    for uid, prompt in zip(ids, prompts):
        reference = _greedy_reference(next_logits, prompt, steps, _processors(len(prompt), **PENALTIES))
        changed |= reference != _greedy_reference(next_logits, prompt, steps, [])
        assert got[uid] == reference
    assert changed


def test_external_target_law_samples_the_penalized_transformed_distribution():
    from types import SimpleNamespace

    from mlx2.runtime.sample_utils import make_transformed_logprobs
    from test_external_dflash2_cpu import generator, tiny

    m, d = tiny()
    b = generator(m, d)
    history = [1, 2, 3, 3, 5]
    processors = _processors(3, **PENALTIES)
    lane = SimpleNamespace(processors=processors, sampling={"sampling_temp": 0.8, "top_p": 0.9, "top_k": 6, "min_p": 0.0})
    logits = mx.array(np.random.default_rng(3).normal(size=32).astype(np.float32))
    law = b._target_law(lane, logits, history)
    value = logits[None]
    for processor in processors:
        value = processor(mx.array(history, mx.int32), value)
    expected = np.exp(np.array(make_transformed_logprobs(0.8, top_p=0.9, top_k=6)(value)[0], dtype=np.float64))
    np.testing.assert_allclose(law, expected / expected.sum(), atol=1e-6)


def _drain_generator(batch, limit=200):
    emitted = {}
    finished = set()
    for _ in range(limit):
        _, responses = batch.next()
        for response in responses:
            emitted.setdefault(response.uid, []).append(response.token)
            if response.finish_reason:
                finished.add(response.uid)
        if emitted and finished == set(emitted):
            return emitted
    raise AssertionError("batch failed to complete")


def test_self_mtp_matches_ordinary_and_reference_under_penalties():
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.models.cache import make_prompt_cache
    from mlx2.runtime.sample_utils import LaneRNG
    from test_batched_mtp import _tiny_qwen4_model

    mx.random.seed(71)
    model = _tiny_qwen4_model()
    prompt = [1, 7, 3, 9, 2, 8, 4, 6, 5]
    steps = 10

    def next_logits(history):
        return model(mx.array([history], mx.uint32), cache=make_prompt_cache(model))[0, -1]

    reference = _greedy_reference(next_logits, prompt, steps, _processors(len(prompt), **PENALTIES))
    assert reference != _greedy_reference(next_logits, prompt, steps, [])

    ordinary = BatchGenerator(model, prefill_step_size=4)
    try:
        ordinary.insert(
            [prompt], max_tokens=[steps],
            samplers=[lambda logprobs: mx.argmax(logprobs, axis=-1)],
            logits_processors=[_processors(len(prompt), **PENALTIES)],
        )
        ordinary_out = next(iter(_drain_generator(ordinary).values()))
    finally:
        ordinary.close()
    assert ordinary_out == reference

    mtp = BatchGenerator(model, prefill_step_size=4, self_mtp={"num_draft": 2, "persistent": True})
    try:
        mtp.insert(
            [prompt], max_tokens=[steps], mtp_states=[None], lane_rngs=[LaneRNG(3)],
            self_mtp_configs=[{"sampling_temp": 0.0}],
            logits_processors=[_processors(len(prompt), **PENALTIES)],
        )
        mtp_out = next(iter(_drain_generator(mtp).values()))
    finally:
        mtp.close()
    assert mtp_out == reference


# --- receipts ---------------------------------------------------------------

from test_structured_deferral import _collect, scripted_engine  # noqa: E402,F401


def test_engine_applies_defaults_and_records_them_in_the_receipt(scripted_engine):
    build, state = scripted_engine
    state["sampling_defaults"] = XING4_SAMPLING
    engine = build(declare_marker=True)
    try:
        status = engine.status()["sampling_defaults"]
        assert status["model"] == XING4_SAMPLING.model and set(status["profiles"]) == {"general", "coding", "agent"}
        state["script"] = [7, 11]
        request = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "temperature": 0}
        _, _, final = _collect(engine.submit(dict(request)))
        controls = final["receipt"]["request_controls"]
        assert controls["sampling"] == {"temperature": 0}
        assert controls["effective_sampling"]["temperature"] == 0
        assert controls["effective_sampling"]["repetition_penalty"] == 1.05
        assert controls["sampling_defaults"]["applied"] == {"top_p": 0.95, "repetition_penalty": 1.05}
        assert controls["sampling_defaults"]["sources"]["repetition_penalty"] == "generation_config.json"
        config = state["inserted"][-1]["self_mtp_configs"][0]
        assert (config["sampling_temp"], config["top_p"], config["top_k"]) == (0, 0.95, 0)

        _, _, final = _collect(engine.submit({**request, "sampling_profile": "coding", "temperature": 0.5}))
        controls = final["receipt"]["request_controls"]
        assert controls["sampling_defaults"]["profile"] == "coding"
        assert controls["effective_sampling"]["temperature"] == 0.5  # explicit wins
        with pytest.raises(ValueError, match="unknown sampling_profile"):
            engine.submit({**request, "sampling_profile": "poetry"})
    finally:
        state["sampling_defaults"] = None


def test_engine_without_declaration_keeps_legacy_defaults(scripted_engine):
    build, state = scripted_engine
    engine = build(declare_marker=True)
    state["script"] = [7]
    # The scripted vocabulary (20) is too small for the legacy top_k of 20.
    _, _, final = _collect(engine.submit({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2, "top_k": 5}))
    controls = final["receipt"]["request_controls"]
    assert engine.status()["sampling_defaults"] is None
    assert controls["effective_sampling"] == {**LEGACY_FALLBACK, "top_k": 5}
    assert controls["sampling_defaults"]["fallback_kind"] == "legacy"
    with pytest.raises(ValueError, match="declares no sampling profiles"):
        engine.submit({"messages": [{"role": "user", "content": "hi"}], "sampling_profile": "coding"})


def test_neutral_values_are_the_disabled_filters():
    assert NEUTRAL["top_k"] == 0 and NEUTRAL["top_p"] == 1.0 and NEUTRAL["repetition_penalty"] == 1.0


def test_responses_api_forwards_the_profile_extension():
    from mlx2.openai_compat import responses_to_chat_request

    request, _ = responses_to_chat_request({"input": "hi", "sampling_profile": "coding"})
    assert request["sampling_profile"] == "coding"
