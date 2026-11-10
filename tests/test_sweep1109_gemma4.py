"""Bug sweep 2026-11-09, lane gemma4 (GEM-1..GEM-3). CPU, tiny random weights.

No checkpoint is loaded (``synthetic@n/a``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx2.runtime import prefill_tail_skip as pts  # noqa: E402
from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows  # noqa: E402

RESEARCH = Path(__file__).resolve().parents[1] / "scripts" / "research"


@pytest.fixture(autouse=True)
def _cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _scripts():
    if str(RESEARCH) not in sys.path:
        sys.path.insert(0, str(RESEARCH))
    import gemma4_native_ab as ab
    import gemma4_native_pld_check as pld

    return ab, pld


# -- GEM-1 --------------------------------------------------------------------


def _claimed(wrapped, width, salt, vocab):
    rng = np.random.default_rng(salt)
    tokens = mx.array(rng.integers(1, vocab - 1, size=(1, width)), dtype=mx.int32)
    with prefill_checkpoint_rows([width], width=width) as request:
        out = wrapped(tokens, cache=wrapped.make_cache())
        mx.eval(out)
    return out, request.applied


def test_gem1_native_tail_counter_follows_the_model_decision(monkeypatch):
    from mlx2.adapters.gemma4 import _Gemma4LogitsModel
    from test_gemma4_native_cpu import TEXT, _pair

    monkeypatch.delenv(pts.ENV, raising=False)  # Gemma 4 default: mode 1
    _, ours = _pair({}, seed=13)
    wrapped = _Gemma4LogitsModel(ours)
    vocab = TEXT["vocab_size"]
    split0 = pts.STATS["gemma4_native_tail_split"]
    full0 = pts.STATS["gemma4_native_tail_full"]
    out, applied = _claimed(wrapped, 9, 51, vocab)
    assert out.shape == (1, 1, vocab) and applied == 1
    assert pts.STATS["gemma4_native_tail_split"] == split0 + 1
    assert pts.STATS["gemma4_native_tail_full"] == full0
    # A hook that swaps the last layer's class makes the model run the full
    # layer; the split counter must not move.
    last = ours.language_model.model.layers[-1]
    last.__class__ = type("Hooked", (type(last),), {})
    engaged0 = pts.STATS["tail_engaged"]
    out, applied = _claimed(wrapped, 9, 52, vocab)
    assert out.shape == (1, 1, vocab) and applied == 1
    assert pts.STATS["tail_engaged"] == engaged0 + 1  # the adapter still claimed
    assert pts.STATS["gemma4_native_tail_split"] == split0 + 1
    assert pts.STATS["gemma4_native_tail_full"] == full0 + 1


# -- GEM-2 --------------------------------------------------------------------


def _payload(text, tokens=None, completion=None):
    choice = {"message": {"content": text}}
    if tokens is not None:
        choice["logprobs"] = {"content": [
            {"token": t, "logprob": -0.1, "top_logprobs": [{"token": t, "logprob": -0.1}]}
            for t in tokens]}
    payload = {"choices": [choice]}
    if completion is not None:
        payload["usage"] = {"completion_tokens": completion}
    return payload


def test_gem2_no_token_trace_is_never_token_parity():
    ab, _ = _scripts()
    no_trace = {"choices": [{"message": {"content": "x"}, "logprobs": None}]}
    other = {"choices": [{"message": {"content": "completely different"}}]}
    verdict = ab._classify(ab._first_divergence(no_trace, other))
    assert verdict["verdict"] == "unclassified_no_token_trace"
    same = {"choices": [{"message": {"content": "x"}}]}
    verdict = ab._classify(ab._first_divergence(no_trace, same))
    assert verdict["verdict"] == "text_equal_no_token_trace"
    # A trace shorter than the counted completion is not a full comparison.
    short = _payload("alpha beta", ["alpha"], completion=2)
    full = _payload("alpha gamma", ["alpha", " gamma"], completion=2)
    assert ab._classify(ab._first_divergence(short, full))["verdict"] == (
        "unclassified_no_token_trace")
    # Complete equal traces still classify as parity.
    a = _payload("alpha beta", ["alpha", " beta"], completion=2)
    b = _payload("alpha beta", ["alpha", " beta"], completion=2)
    assert ab._classify(ab._first_divergence(a, b))["verdict"] == "exact_token_parity"


# -- GEM-3 --------------------------------------------------------------------


def test_gem3_default_arm_must_serve_the_pld_profile_and_accept_drafts():
    _, pld = _scripts()
    ordinary = {"profile": "gemma4-native-apcv2-ordinary"}
    with pytest.raises(RuntimeError):
        pld._arm_gate("default", ordinary)  # substring of the PLD profile
    with pytest.raises(RuntimeError):
        pld._arm_gate("ordinary", {"profile": "gemma4-native-apcv2-ordinary-pld"})
    with pytest.raises(RuntimeError):
        pld._arm_gate("ordinary", {"profile": "mlx-vlm-apcv2-ordinary"})
    assert pld._arm_gate("ordinary", ordinary) == "gemma4-native-apcv2-ordinary"
    assert pld._arm_gate("default", {"profile": "gemma4-native-apcv2-ordinary-pld"}) == (
        "gemma4-native-apcv2-ordinary-pld")
    for final in ({"scheduler": {"pld_accepted": 0}}, {"scheduler": {}}, {}):
        with pytest.raises(RuntimeError):
            pld._pld_gate(final)
    assert pld._pld_gate({"scheduler": {"pld_accepted": 425}}) == 425


def test_review_arm_gate_accepts_suffixed_profiles_but_not_the_other_arm():
    _, pld = _scripts()
    suffix = "-int8-prefill-mlp"
    assert pld._arm_gate("ordinary", {"profile": "gemma4-native-apcv2-ordinary" + suffix})
    assert pld._arm_gate("default", {"profile": "gemma4-native-apcv2-ordinary-pld" + suffix})
    for name, profile in (("default", "gemma4-native-apcv2-ordinary" + suffix),
                          ("ordinary", "gemma4-native-apcv2-ordinary-pld" + suffix),
                          ("ordinary", "gemma4-native-apcv2-ordinaryx"),
                          ("default", None)):
        with pytest.raises(RuntimeError):
            pld._arm_gate(name, {"profile": profile})
