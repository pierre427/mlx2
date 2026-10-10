"""A media companion must carry trace evidence, not only passed booleans."""

from copy import deepcopy
import importlib.util
from types import SimpleNamespace
import hashlib
import json
from pathlib import Path

from mlx2.media_qualification import SMOL_MEDIA_CHECKS, evaluate_smol_media_report
from mlx2 import qualification
import pytest


def test_approved_media_producer_hashes_match_reviewed_scripts():
    root = Path(__file__).resolve().parents[1]
    for harness, _evaluator, _checks in qualification.APPROVED_MEDIA_PRODUCERS.values():
        assert hashlib.sha256((root / harness["name"]).read_bytes()).hexdigest() == harness["sha256"]


REV = "a" * 40
IDENTITY = {"schema": "mlx2.vlm-dependencies.v1", "family": "smolvlm",
            "source_sha256": "c" * 64, "dependency_files": 110,
            "reference_revision": REV}
SETTINGS = {"mtp": False, "mlx_vlm": dict(IDENTITY)}


def _row(cached, output="same"):
    return {"route": "ordinary", "qualification": "candidate",
            "finish_reason": "length", "prompt_tokens": 100,
            "cached_tokens": cached, "output": output}


def _arm(kind="image"):
    positions = list(range(41, 60))
    return {
        "fixture": {"trusted": True, "media_sha256": "a" * 64,
                    "changed_media_sha256": "d" * 64,
                    "ordered_frame_sha256": ["e" * 64] * (2 if kind == "video" else 1),
                    "changed_ordered_frame_sha256": ["f" * 64] * (2 if kind == "video" else 1),
                    "prompt_tokens_sha256": "b" * 64,
                    "media_token_positions": positions,
                    "media_token_positions_sha256": hashlib.sha256(
                        json.dumps(positions, separators=(",", ":")).encode()).hexdigest(),
                    "media_token_positions_count": len(positions),
                    "media_token_end": 60, "prompt_tokens": 100},
        "parity": {"prefill_max_abs": 0.0, "prefill_argmax_match": True,
                   "decode": [{"max_abs": 0.0, "argmax_match": True} for _ in range(4)],
                   "source_revision": REV, "source_sha256": IDENTITY["source_sha256"]},
        "mlx_vlm_runtime": dict(IDENTITY),
        "serving": {"cold": _row(0), "warm1": _row(99), "warm2": _row(99),
                    "changed_tail": _row(60), "changed_lead": _row(0),
                    "changed_pixels": _row(0), "return_original": _row(99),
                    "apcv2_hits_delta": 4},
    }


def _text_source(artifact="artifact-fingerprint", revision=REV):
    def case(prompt, length, count=16, start=100):
        prompt_ids = list(range(length))
        digest = hashlib.sha256(json.dumps(prompt_ids, separators=(",", ":")).encode()).hexdigest()
        generated = list(range(start, start + count))
        rows = [{"shape": [1, length, 1000], "max_abs": 0.0,
                 "argmax_match": True, "source_argmax": generated[0],
                 "adapter_argmax": generated[0], "sampled_argmax": generated[0]}]
        rows.extend({"shape": [1, 1, 1000], "max_abs": 0.0,
                     "argmax_match": True, "source_argmax": generated[i + 1],
                     "adapter_argmax": generated[i + 1],
                     "sampled_argmax": generated[i + 1], "step": i,
                     "input_token": generated[i]} for i in range(count - 1))
        return {"prompt": prompt, "prompt_tokens": length,
                "prompt_token_ids": prompt_ids, "prompt_tokens_sha256": digest,
                "generated_token_ids": generated,
                "stop_token_ids": [2], "prefill": rows[0], "decode": rows[1:],
                "source_revision": revision,
                "source_sha256": IDENTITY["source_sha256"],
                "sampling": {"temperature": 0, "repetition_penalty": 1.0,
                             "presence_penalty": 0.0, "frequency_penalty": 0.0},
                "max_tokens": count, "min_tokens": count if count == 64 else 0}

    source = Path(__file__).resolve().parents[1] / "scripts/qualify_serving.py"
    spec = importlib.util.spec_from_file_location("media_test_prompts", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    filler = module.long_context_filler(module.near_limit_prompt_floor(4096))
    near_prompt = (filler + "\nQuestion: What topic should the answer discuss? "
                   "Answer: compiler optimization. Start your answer with compiler.")
    control_prompt = (filler + "\nQuestion: What is the final secret word? "
                      "The final secret word is SAPPHIRE. Answer: SAPPHIRE.")

    return {"artifact": artifact, "cases": {
        "hermes_client": case("Reply with exactly HERMES_READY", 17),
        "cold_text": case("Reply with exactly MLX2_READY", 18),
        "near_context": case(near_prompt, 3869, 64),
        "near_context_control": case(control_prompt, 3875, 64, start=200),
    }}


def test_media_checks_recomputed_from_both_live_arms():
    report = {"artifact": "artifact-fingerprint", "source_revision": REV,
              "settings": deepcopy(SETTINGS),
              "text_source": _text_source(),
              "arms": {"image": _arm(), "video": _arm("video")},
              "checks": {name: {"passed": True} for name in SMOL_MEDIA_CHECKS}}
    assert all(evaluate_smol_media_report(report).values())

    missing_video = deepcopy(report)
    del missing_video["arms"]["video"]
    assert not all(evaluate_smol_media_report(missing_video).values())

    forged_parity = deepcopy(report)
    forged_parity["arms"]["image"]["parity"]["decode"][2]["max_abs"] = 0.2
    assert evaluate_smol_media_report(forged_parity)["image_parity"] is False

    forged_cache = deepcopy(report)
    forged_cache["arms"]["video"]["serving"]["warm1"]["cached_tokens"] = 0
    assert evaluate_smol_media_report(forged_cache)["multimodal_apcv2_reuse"] is False

    untrusted_alignment = deepcopy(report)
    untrusted_alignment["arms"]["image"]["fixture"]["trusted"] = False
    assert evaluate_smol_media_report(untrusted_alignment)["media_token_alignment"] is False

    for alteration in ("missing", "short", "chain", "sampling", "artifact",
                       "prompt_hash", "source_token", "adapter_logit", "early_stop",
                       "near_prompt", "near_short", "near_no_tail_effect"):
        forged = deepcopy(report)
        if alteration == "missing":
            del forged["text_source"]
        elif alteration == "short":
            forged["text_source"]["cases"]["cold_text"]["decode"].pop()
        elif alteration == "chain":
            forged["text_source"]["cases"]["cold_text"]["decode"][0]["input_token"] += 1
        elif alteration == "sampling":
            forged["text_source"]["cases"]["hermes_client"]["sampling"]["temperature"] = 0.7
        elif alteration == "prompt_hash":
            forged["text_source"]["cases"]["cold_text"]["prompt_token_ids"][0] += 1
        elif alteration == "source_token":
            forged["text_source"]["cases"]["cold_text"]["generated_token_ids"][0] += 1
        elif alteration == "adapter_logit":
            forged["text_source"]["cases"]["hermes_client"]["decode"][3]["max_abs"] = 0.01
        elif alteration == "early_stop":
            forged["text_source"]["cases"]["cold_text"]["stop_token_ids"] = [100]
        elif alteration == "near_prompt":
            forged["text_source"]["cases"]["near_context"]["prompt"] += "!"
        elif alteration == "near_short":
            forged["text_source"]["cases"]["near_context"]["decode"].pop()
        elif alteration == "near_no_tail_effect":
            near = forged["text_source"]["cases"]["near_context"]
            control = forged["text_source"]["cases"]["near_context_control"]
            control["generated_token_ids"] = list(near["generated_token_ids"])
            for index, row in enumerate([control["prefill"], *control["decode"]]):
                row["source_argmax"] = row["adapter_argmax"] = near["generated_token_ids"][index]
                row["sampled_argmax"] = near["generated_token_ids"][index]
                if index:
                    row["input_token"] = near["generated_token_ids"][index - 1]
        else:
            forged["text_source"]["artifact"] = "other"
        assert evaluate_smol_media_report(forged)["text_source_parity"] is False


def test_near_context_minimum_length_records_masked_stop_selection():
    report = {"artifact": "artifact-fingerprint", "source_revision": REV,
              "settings": deepcopy(SETTINGS),
              "text_source": _text_source(),
              "arms": {"image": _arm(), "video": _arm("video")}}
    control = report["text_source"]["cases"]["near_context_control"]
    control["decode"][5]["source_argmax"] = 2
    control["decode"][5]["adapter_argmax"] = 2
    assert evaluate_smol_media_report(report)["text_source_parity"] is True
    control["decode"][5]["sampled_argmax"] = 2
    assert evaluate_smol_media_report(report)["text_source_parity"] is False


def test_companion_binds_producer_source_runtime_artifact_and_settings(monkeypatch):
    producer = {"name": "scripts/qualify_media_serving.py", "sha256": "d" * 64}
    monkeypatch.setitem(qualification.APPROVED_MEDIA_PRODUCERS, "smolvlm",
                        (producer, evaluate_smol_media_report, SMOL_MEDIA_CHECKS))
    monkeypatch.setattr(qualification, "PINNED_MEDIA_SOURCE_REVISION", REV)
    descriptor = SimpleNamespace(model_type="smolvlm", metadata={"source_revision": REV})
    runtime, artifact, settings = {"revision": "runtime"}, "artifact-fingerprint", deepcopy(SETTINGS)
    report = {"schema": "mlx2.media-serving-qualification.v1",
              "qualification_harness": producer, "model_type": "smolvlm",
              "source_revision": REV, "runtime": runtime, "artifact": artifact,
              "settings": settings, "text_source": _text_source(),
              "arms": {"image": _arm(), "video": _arm("video")},
              "checks": {name: {"passed": True} for name in SMOL_MEDIA_CHECKS},
              "passed": True}
    args = {"runtime": runtime, "artifact": artifact,
            "settings": settings, "descriptor": descriptor}
    assert all(qualification.validate_adapter_qualification(report, **args).values())
    for field, replacement in (("artifact", "other"), ("source_revision", "other"),
                               ("qualification_harness", {"sha256": "other"}),
                               ("settings", {**SETTINGS, "mtp": True})):
        corrupted = {**report, field: replacement}
        with pytest.raises(ValueError):
            qualification.validate_adapter_qualification(corrupted, **args)
    corrupted = deepcopy(report)
    corrupted["arms"]["image"]["parity"]["prefill_max_abs"] = 1.0
    with pytest.raises(ValueError, match="traces"):
        qualification.validate_adapter_qualification(corrupted, **args)


def test_changed_tail_may_not_reuse_the_whole_prompt():
    """changed_tail must branch after the media but before the changed tail;
    reusing all 100 prompt tokens (nothing recomputed) passed."""
    report = {"artifact": "artifact-fingerprint", "source_revision": REV,
              "settings": deepcopy(SETTINGS),
              "text_source": _text_source(),
              "arms": {"image": _arm(), "video": _arm("video")},
              "checks": {name: {"passed": True} for name in SMOL_MEDIA_CHECKS}}
    report["arms"]["image"]["serving"]["changed_tail"]["cached_tokens"] = 100
    assert evaluate_smol_media_report(report)["multimodal_apcv2_reuse"] is False


def test_smol_arms_and_text_traces_bind_the_served_source_identity():
    """The reader accepted arms with no source identity at all, and text
    traces bound to nothing but the report's own ``source_revision``."""
    def report():
        return {"artifact": "artifact-fingerprint", "source_revision": REV,
                "settings": deepcopy(SETTINGS), "text_source": _text_source(),
                "arms": {"image": _arm(), "video": _arm("video")}}

    assert all(evaluate_smol_media_report(report()).values())
    stripped = report()
    for arm in stripped["arms"].values():
        del arm["mlx_vlm_runtime"]
        del arm["parity"]["source_sha256"]
        del arm["parity"]["source_revision"]
    for case in stripped["text_source"]["cases"].values():
        del case["source_sha256"]
    # Unbound arms are not evidence for anything derived from them.
    assert not any(evaluate_smol_media_report(stripped).values())
    one_arm = report()
    del one_arm["arms"]["video"]["mlx_vlm_runtime"]
    assert not any(evaluate_smol_media_report(one_arm).values())
    for path, value in (
        (("arms", "image", "mlx_vlm_runtime", "family"), "qwen2_5_vl"),
        (("arms", "video", "mlx_vlm_runtime", "reference_revision"), "b" * 40),
        (("arms", "video", "parity", "source_sha256"), "d" * 64),
        (("arms", "image", "parity", "source_revision"), "b" * 40),
        (("settings", "mlx_vlm", "source_sha256"), "d" * 64),
        (("settings", "mlx_vlm"), None),
        (("arms", "image", "mlx_vlm_runtime"), {"version": "0.6.17", "source": "index",
                                                "editable": False, "revision": REV}),
        (("text_source", "cases", "cold_text", "source_sha256"), "d" * 64),
    ):
        tampered = report()
        item = tampered
        for key in path[:-1]:
            item = item[key]
        item[path[-1]] = value
        checks = evaluate_smol_media_report(tampered)
        if path[0] == "text_source":
            # The arms stay bound; only the text trace lost its binding.
            assert not checks["text_source_parity"], path
        else:
            assert not any(checks.values()), path
        assert SETTINGS["mlx_vlm"]["source_sha256"] == "c" * 64
