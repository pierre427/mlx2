"""The media qualification producer rejects incomplete or unsafe receipts."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/qualify_media_serving.py"
SPEC = importlib.util.spec_from_file_location("qualify_media_serving", SCRIPT)
producer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(producer)


def row(cached_tokens, *, prompt_tokens=10, output="blue"):
    receipt = {
        "cache": "apcv2", "route": "ordinary", "qualification": "candidate",
        "prompt_tokens": prompt_tokens, "cached_tokens": cached_tokens,
        "cache_checkpoint_role": "committed_prompt_boundary",
    }
    return {"finish_reason": "length", "output": output, "reasoning": "",
            "receipt": receipt,
            **{key: receipt[key] for key in ("route", "qualification",
               "prompt_tokens", "cached_tokens", "cache_checkpoint_role")}}


def serving_rows():
    return {
        "cold": row(0), "warm1": row(9), "warm2": row(9),
        "changed_tail": row(7, output="other"),
        "changed_lead": row(0, output="other"),
        "changed_pixels": row(0, output="other"),
        "return_original": row(9),
    }


def test_serving_predicates_detect_media_prefix_reuse_and_drift():
    rows = serving_rows()
    assert all(producer.check_serving_rows(rows, media_end=7).values())

    rows["changed_pixels"]["receipt"]["cached_tokens"] = 7
    checks = producer.check_serving_rows(rows, media_end=7)
    assert not checks["changed_pixels_refuses_media_reuse"]
    assert checks["post_media_branch"]

    rows = serving_rows()
    rows["warm2"]["output"] = "red"
    assert not producer.check_serving_rows(rows, media_end=7)["warm_apcv2_restore"]

    rows = serving_rows()
    rows["cold"]["receipt"]["qualification"] = "qualified"
    assert not producer.check_serving_rows(rows, media_end=7)["route_receipts"]


def test_prompt_alignment_requires_last_media_token_before_decode_reserve():
    prepared = {"_mlx2_prompt_tokens": [1, 42, 42, 9],
                "_mlx2_media_token_end": 3,
                "_mlx2_media_fingerprint": "media-id"}
    record = producer.prompt_alignment(prepared, 42)
    assert record["media_token_positions"] == [1, 2]
    assert record["media_token_end"] == 3
    assert record["prompt_tokens"] == 4

    prepared["_mlx2_media_token_end"] = 2
    with pytest.raises(AssertionError, match="boundary"):
        producer.prompt_alignment(prepared, 42)
    prepared["_mlx2_media_token_end"] = 3
    prepared["_mlx2_prompt_tokens"] = [1, 42, 42]
    with pytest.raises(AssertionError, match="boundary"):
        producer.prompt_alignment(prepared, 42)


def test_text_trace_uses_typed_prompt_and_source_greedy_chain(monkeypatch):
    fake_mx = SimpleNamespace(int32="int32", array=lambda value, dtype: value)
    fake_cache = SimpleNamespace(KVCache=lambda: object())
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=fake_mx))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setitem(sys.modules, "mlx_vlm", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mlx_vlm.models", SimpleNamespace(cache=fake_cache))
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.cache", fake_cache)
    calls = []

    class Source:
        language_model = SimpleNamespace(layers=[object(), object()])

        def __call__(self, input_ids, *, pixel_values, cache):
            assert pixel_values is None and len(cache) == 2
            calls.append(input_ids)
            return SimpleNamespace(logits=SimpleNamespace(shape=(1, len(input_ids[0]), 1000)))

    class Candidate:
        _model = Source()

        def make_cache(self):
            return [object(), object()]

        def __call__(self, input_ids, *, pixel_values, cache):
            assert pixel_values is None and len(cache) == 2
            return SimpleNamespace(shape=(1, len(input_ids[0]), 1000))

    adapter = SimpleNamespace(
        mlx_vlm_runtime={"revision": producer.SOURCE_REVISION},
        model=Candidate(), _eos_ids=lambda: [2],
        prompt_tokens=lambda request: ([1, 10, 11, 12]
            if request == {"messages": [{"role": "user", "content": "test prompt"}]}
            else pytest.fail("wrong text request")),
    )
    next_id = iter(range(100, 116))

    def compare(_mx, source, candidate):
        assert source.shape == candidate.shape
        token_id = next(next_id)
        return {"shape": list(source.shape), "max_abs": 0.0,
                "argmax_match": True, "source_argmax": token_id,
                "adapter_argmax": token_id}

    monkeypatch.setattr(producer, "compare_logits", compare)
    trace = producer.text_parity_arm(adapter, "test prompt")
    assert trace["prompt_token_ids"] == [1, 10, 11, 12]
    assert trace["generated_token_ids"] == list(range(100, 116))
    assert trace["prefill"]["sampled_argmax"] == 100
    assert trace["decode"][-1]["sampled_argmax"] == 115
    assert trace["stop_token_ids"] == [2]
    assert len(trace["decode"]) == 15
    assert [row["input_token"] for row in trace["decode"]] == list(range(100, 115))
    assert len(calls) == 16
