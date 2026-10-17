"""Pure, fail-closed evaluation of source-bound media qualification traces.

The live producer gathers observations. Both it and the route loader use this
module so a hand-edited ``passed`` field cannot replace missing trace data.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from collections import Counter

SMOL_MEDIA_CHECKS = frozenset(
    {
        "multimodal_image",
        "multimodal_video",
        "image_parity",
        "video_parity",
        "media_token_alignment",
        "multimodal_state_replay",
        "multimodal_apcv2_reuse",
        "text_source_parity",
    }
)
LOGIT_MAX_ABS = 1e-4
TEXT_TOKENS = 16
NEAR_CONTEXT_TOKENS = 64
TEXT_PROMPTS = {
    "hermes_client": "Reply with exactly HERMES_READY",
    "cold_text": "Reply with exactly MLX2_READY",
}
TEXT_SAMPLING = {
    "temperature": 0,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}
# The near-context probe is one shared filler followed by one of these pinned
# instructions.  The filler used to be pinned byte for byte (a fixed 54-word
# LCG word salad); since 2026-10-14 it is varied repository text sized by the
# served tokenizer (reasoning models fail on repeated input), which changes
# with the snapshot, so its shape is checked instead of its bytes.
NEAR_CONTEXT_SUFFIXES = {
    "near_context": (
        "\nQuestion: What topic should the answer discuss? "
        "Answer: compiler optimization. Start your answer with compiler."
    ),
    "near_context_control": (
        "\nQuestion: What is the final secret word? "
        "The final secret word is SAPPHIRE. Answer: SAPPHIRE."
    ),
}
# Answers the two instructions name; the filler may not contain any of them
# (compared case-insensitively), so the only answer is at the far end.
NEAR_CONTEXT_ANSWER_MARKERS = ("compiler", "sapphire")
# Varied-text floor for the filler, measured on 16K-character windows of the
# realistic haystack corpus (README, docs, runtime sources; 2026-10-14):
# distinct-word share >= 0.113, top word <= 0.117, top ten words >= 0.127,
# repeated 8-word runs <= 0.65 anywhere, and 0.40 / 0.05 / 0.15 / 0.0 at the
# corpus head the producer reads.  The old word salad scores 0.02 distinct;
# a repeated phrase puts its top word far above 0.12 or its 8-word runs at
# 1.0; real text pasted twice repeats every run; a flat random draw from a
# large vocabulary has its top ten words near 0.01.
NEAR_CONTEXT_FILLER_MIN_WORDS = 512
NEAR_CONTEXT_FILLER_MIN_DISTINCT_SHARE = 0.10
NEAR_CONTEXT_FILLER_MAX_TOP_WORD_SHARE = 0.12
NEAR_CONTEXT_FILLER_MIN_TOP10_SHARE = 0.08
NEAR_CONTEXT_FILLER_RUN_WORDS = 8
NEAR_CONTEXT_FILLER_MAX_REPEATED_RUN_SHARE = 0.5
_WORD = re.compile(r"[a-z0-9]+")


# The identity ``adapters/vlm_runtime.bind_backend`` publishes as
# ``adapter.mlx_vlm_runtime`` and the engine records as ``settings.mlx_vlm``:
# the digest of the executed mlx-vlm source closure plus the reviewed
# reference revision.  The pre-contract ``{version, source, editable,
# revision}`` shape named an install, not the executed bytes, and is refused.
VLM_DEPENDENCY_SCHEMA = "mlx2.vlm-dependencies.v1"


def _sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(digit in "0123456789abcdef" for digit in value)
    )


def source_contract_identity(runtime, revision, family=None):
    """True when ``runtime`` is a dependency-content identity bound to
    ``revision`` (and, when given, to the contract ``family``).  Both are the
    reviewed values the caller is written against, never read from the
    report itself."""
    return (
        isinstance(runtime, dict)
        and runtime.get("schema") == VLM_DEPENDENCY_SCHEMA
        and isinstance(runtime.get("family"), str)
        and bool(runtime["family"])
        and (family is None or runtime["family"] == family)
        and _sha256(runtime.get("source_sha256"))
        and type(runtime.get("dependency_files")) is int
        and runtime["dependency_files"] > 0
        and isinstance(revision, str)
        and len(revision) == 40
        and runtime.get("reference_revision") == revision
    )


def parity_source_bound(parity, runtime, revision):
    """A parity trace is evidence only against the source bytes that produced
    it: its recorded revision and digest must equal the arm's bound identity."""
    return (
        source_contract_identity(runtime, revision)
        and isinstance(parity, dict)
        and parity.get("source_revision") == revision
        and parity.get("source_sha256") == runtime["source_sha256"]
    )


def arm_source_bound(arm, *, settings, family, revision):
    """An arm is evidence only for the identity the route served under: its
    ``mlx_vlm_runtime`` must equal the report's served ``settings.mlx_vlm``,
    name the family the report qualifies, and bind its parity trace.  A
    self-asserted identity with a matching fabricated digest is not evidence."""
    if not isinstance(arm, dict) or not isinstance(settings, dict):
        return False
    runtime = arm.get("mlx_vlm_runtime")
    return (
        source_contract_identity(runtime, revision)
        and runtime["family"] == family
        and settings.get("mlx_vlm") == runtime
        and parity_source_bound(arm.get("parity"), runtime, revision)
    )


def _finished(row):
    return (
        isinstance(row, dict)
        and row.get("route") == "ordinary"
        and row.get("qualification") == "candidate"
        and row.get("finish_reason") in {"length", "stop"}
        and type(row.get("prompt_tokens")) is int
        and row["prompt_tokens"] > 1
        and type(row.get("cached_tokens")) is int
        and row["cached_tokens"] >= 0
        and isinstance(row.get("output"), str)
    )


def _aligned(fixture, kind):
    if not isinstance(fixture, dict):
        return False
    prompt, end, count = (
        fixture.get(name)
        for name in ("prompt_tokens", "media_token_end", "media_token_positions_count")
    )
    positions = fixture.get("media_token_positions")
    frames = fixture.get("ordered_frame_sha256")
    changed_frames = fixture.get("changed_ordered_frame_sha256")
    return (
        fixture.get("trusted") is True
        and type(prompt) is int
        and type(end) is int
        and type(count) is int
        and 0 < count <= end < prompt
        and isinstance(positions, list)
        and len(positions) == count
        and all(type(position) is int and 0 <= position < end for position in positions)
        and positions == sorted(set(positions))
        and positions[-1] + 1 == end
        and hashlib.sha256(
            json.dumps(positions, separators=(",", ":")).encode()
        ).hexdigest()
        == fixture.get("media_token_positions_sha256")
        and isinstance(frames, list)
        and len(frames) >= (2 if kind == "video" else 1)
        and isinstance(changed_frames, list)
        and len(changed_frames) == len(frames)
        and all(_sha256(value) for value in frames + changed_frames)
        and _sha256(fixture.get("changed_media_sha256"))
        and fixture["changed_media_sha256"] != fixture.get("media_sha256")
        and all(
            _sha256(fixture.get(name))
            for name in (
                "media_sha256",
                "prompt_tokens_sha256",
                "media_token_positions_sha256",
            )
        )
    )


def _parity(parity):
    if not isinstance(parity, dict):
        return False
    rows = [
        {
            "max_abs": parity.get("prefill_max_abs"),
            "argmax_match": parity.get("prefill_argmax_match"),
        },
        *(parity.get("decode") or []),
    ]
    return len(rows) == 5 and all(
        isinstance(row, dict)
        and row.get("argmax_match") is True
        and type(row.get("max_abs")) in (int, float)
        and math.isfinite(row["max_abs"])
        and 0 <= row["max_abs"] <= LOGIT_MAX_ABS
        for row in rows
    )


def _arm(report, kind):
    arms = report.get("arms")
    arm = arms.get(kind) if isinstance(arms, dict) else None
    return arm if isinstance(arm, dict) else {}


def _text_case(case, prompt, *, max_tokens=TEXT_TOKENS, min_tokens=0):
    if not isinstance(case, dict) or case.get("prompt") != prompt:
        return False
    count = case.get("prompt_tokens")
    prompt_ids = case.get("prompt_token_ids")
    generated = case.get("generated_token_ids")
    stop_ids = case.get("stop_token_ids")
    prefill = case.get("prefill")
    decode = case.get("decode")
    if not (
        type(count) is int
        and count > 1
        and isinstance(prompt_ids, list)
        and len(prompt_ids) == count
        and all(type(value) is int and value >= 0 for value in prompt_ids)
        and hashlib.sha256(
            json.dumps(prompt_ids, separators=(",", ":")).encode()
        ).hexdigest()
        == case.get("prompt_tokens_sha256")
        and case.get("max_tokens") == max_tokens
        and case.get("min_tokens") == min_tokens
        and case.get("sampling") == TEXT_SAMPLING
        and isinstance(stop_ids, list)
        and all(type(value) is int and value >= 0 for value in stop_ids)
        and stop_ids == sorted(set(stop_ids))
        and isinstance(generated, list)
        and len(generated) == max_tokens
        and all(
            type(value) is int and value >= 0 and value not in stop_ids
            for value in generated
        )
        and isinstance(prefill, dict)
        and isinstance(decode, list)
        and len(decode) == max_tokens - 1
    ):
        return False
    rows = [prefill, *decode]
    if not all(
        isinstance(row, dict)
        and row.get("argmax_match") is True
        and type(row.get("source_argmax")) is int
        and row.get("source_argmax") == row.get("adapter_argmax")
        and type(row.get("max_abs")) in (int, float)
        and math.isfinite(row["max_abs"])
        and 0 <= row["max_abs"] <= LOGIT_MAX_ABS
        and isinstance(row.get("shape"), list)
        and len(row["shape"]) == 3
        and row["shape"][0] == 1
        and row["shape"][1] == (count if index == 0 else 1)
        and type(row["shape"][2]) is int
        and row["shape"][2] > 1
        and row["source_argmax"] < row["shape"][2]
        and type(row.get("sampled_argmax")) is int
        and 0 <= row["sampled_argmax"] < row["shape"][2]
        and (
            row["sampled_argmax"] == row["source_argmax"]
            if row["source_argmax"] not in stop_ids or not min_tokens
            else row["sampled_argmax"] != row["source_argmax"]
        )
        and generated[index] == row["sampled_argmax"]
        for index, row in enumerate(rows)
    ):
        return False
    return all(
        row.get("step") == step and row.get("input_token") == generated[step]
        for step, row in enumerate(decode)
    )


def near_context_filler_passes(filler):
    """True when ``filler`` is varied text that names no probe answer.

    Neither a repeated string, a small-vocabulary word salad, real text
    pasted twice, nor a flat random draw from a large vocabulary passes.
    """
    if not isinstance(filler, str):
        return False
    lowered = filler.lower()
    if any(marker in lowered for marker in NEAR_CONTEXT_ANSWER_MARKERS):
        return False
    words = _WORD.findall(lowered)
    if len(words) < NEAR_CONTEXT_FILLER_MIN_WORDS:
        return False
    counts = Counter(words)
    top10 = sum(count for _, count in counts.most_common(10))
    span = NEAR_CONTEXT_FILLER_RUN_WORDS
    runs = Counter(
        tuple(words[index : index + span]) for index in range(len(words) - span + 1)
    )
    repeated = sum(count for count in runs.values() if count > 1)
    return (
        len(counts) / len(words) >= NEAR_CONTEXT_FILLER_MIN_DISTINCT_SHARE
        and counts.most_common(1)[0][1] / len(words)
        <= NEAR_CONTEXT_FILLER_MAX_TOP_WORD_SHARE
        and top10 / len(words) >= NEAR_CONTEXT_FILLER_MIN_TOP10_SHARE
        and repeated / sum(runs.values()) <= NEAR_CONTEXT_FILLER_MAX_REPEATED_RUN_SHARE
    )


def near_context_filler(cases):
    """The one filler both near-context prompts share, or ``None``.

    Each prompt must be the filler followed by its pinned instruction, the
    two fillers must be identical, and the filler must pass
    ``near_context_filler_passes``.
    """
    fillers = set()
    for name, suffix in NEAR_CONTEXT_SUFFIXES.items():
        case = cases.get(name) if isinstance(cases, dict) else None
        prompt = case.get("prompt") if isinstance(case, dict) else None
        if not isinstance(prompt, str) or not prompt.endswith(suffix):
            return None
        fillers.add(prompt[: -len(suffix)])
    if len(fillers) != 1:
        return None
    (filler,) = fillers
    return filler if near_context_filler_passes(filler) else None


def _text_source(report, digest):
    trace = report.get("text_source")
    if not isinstance(trace, dict) or trace.get("artifact") != report.get("artifact"):
        return False
    if not _sha256(digest):
        return False
    cases = trace.get("cases")
    if not isinstance(cases, dict) or set(cases) != (
        set(TEXT_PROMPTS) | set(NEAR_CONTEXT_SUFFIXES)
    ):
        return False
    if not all(
        _text_case(cases[name], prompt)
        and cases[name].get("source_revision") == report.get("source_revision")
        and cases[name].get("source_sha256") == digest
        for name, prompt in TEXT_PROMPTS.items()
    ):
        return False
    if (
        cases["hermes_client"]["prompt_tokens_sha256"]
        == cases["cold_text"]["prompt_tokens_sha256"]
    ):
        return False
    if near_context_filler(cases) is None:
        return False
    for name in NEAR_CONTEXT_SUFFIXES:
        case = cases[name]
        prompt = case["prompt"]
        if not (
            _text_case(
                case,
                prompt,
                max_tokens=NEAR_CONTEXT_TOKENS,
                min_tokens=NEAR_CONTEXT_TOKENS,
            )
            and case.get("source_revision") == report.get("source_revision")
            and case.get("source_sha256") == digest
            and 3840 < case["prompt_tokens"] <= 4096 - NEAR_CONTEXT_TOKENS
        ):
            return False
    return (
        cases["near_context"]["generated_token_ids"]
        != cases["near_context_control"]["generated_token_ids"]
    )


def evaluate_smol_media_report(report):
    """Recompute eight required checks from media and source-text traces."""
    if not isinstance(report, dict):
        return {name: False for name in SMOL_MEDIA_CHECKS}
    arms = {kind: _arm(report, kind) for kind in ("image", "video")}
    aligned = {kind: _aligned(arm.get("fixture"), kind) for kind, arm in arms.items()}
    # ``source_revision`` is pinned by the route validator; here every arm and
    # text trace must be bound to the one served identity for that revision.
    bound = {
        kind: arm_source_bound(
            arm,
            settings=report.get("settings"),
            family="smolvlm",
            revision=report.get("source_revision"),
        )
        for kind, arm in arms.items()
    }
    if not all(bound.values()):
        return {name: False for name in SMOL_MEDIA_CHECKS}
    digest = arms["image"]["mlx_vlm_runtime"]["source_sha256"]
    parity = {kind: _parity(arm.get("parity")) for kind, arm in arms.items()}
    serving = {
        kind: arm.get("serving") if isinstance(arm.get("serving"), dict) else {}
        for kind, arm in arms.items()
    }
    completed = {
        kind: all(
            _finished(rows.get(name))
            for name in (
                "cold",
                "warm1",
                "warm2",
                "changed_tail",
                "changed_lead",
                "changed_pixels",
                "return_original",
            )
        )
        for kind, rows in serving.items()
    }

    def replay(kind):
        rows = serving[kind]
        if not completed[kind]:
            return False
        cold = rows["cold"]
        return all(
            rows[name]["output"] == cold["output"]
            for name in ("warm1", "warm2", "return_original")
        ) and all(
            rows[name]["prompt_tokens"] == cold["prompt_tokens"]
            for name in ("warm1", "warm2", "return_original")
        )

    def apcv2(kind):
        rows = serving[kind]
        if not completed[kind] or not aligned[kind]:
            return False
        prompt = rows["cold"]["prompt_tokens"]
        media_end = arms[kind]["fixture"]["media_token_end"]
        return (
            rows["cold"]["cached_tokens"] == 0
            and all(
                rows[name]["cached_tokens"] == prompt - 1
                for name in ("warm1", "warm2", "return_original")
            )
            and media_end
            <= rows["changed_tail"]["cached_tokens"]
            < rows["changed_tail"]["prompt_tokens"]
            and rows["changed_lead"]["cached_tokens"] == 0
            and rows["changed_pixels"]["cached_tokens"] == 0
            and type(rows.get("apcv2_hits_delta")) is int
            and rows["apcv2_hits_delta"] >= 3
        )

    return {
        "multimodal_image": aligned["image"] and completed["image"],
        "multimodal_video": aligned["video"] and completed["video"],
        "image_parity": aligned["image"] and parity["image"],
        "video_parity": aligned["video"] and parity["video"],
        "media_token_alignment": all(aligned.values()),
        "multimodal_state_replay": all(replay(kind) for kind in arms),
        "multimodal_apcv2_reuse": all(apcv2(kind) for kind in arms),
        "text_source_parity": _text_source(report, digest),
    }


def _hex(s):
    return (
        isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)
    )


# Short chat prompts: the instruct checkpoints answer chat, while raw
# completions take them off-template, where low-confidence tokens make the
# batched-versus-single comparison meaningless (Gemma 4 26B-A4B, 2026-10-15).
NATIVE_CONTINUOUS_BATCH_PROMPTS = (
    "What is two plus two? Answer with one word.",
    "What is three plus three? Answer with one word.",
)
# Functional oracle per batch prompt: a near-tie row passes only when both
# continuations still answer correctly (AGENTS.md near_tie_equivalent rule).
NATIVE_CONTINUOUS_BATCH_ANSWERS = {
    NATIVE_CONTINUOUS_BATCH_PROMPTS[0]: ("four", "4"),
    NATIVE_CONTINUOUS_BATCH_PROMPTS[1]: ("six", "6"),
}
CONTINUOUS_BATCH_EQUIVALENCE_SCHEMA = "mlx2.continuous-batch-equivalence.v1"
# A batched-versus-single first divergence is accepted only at a near tie
# (Pierre 2026-10-15: "we always accept near flips"). Same ceiling and
# semantics as scripts/assess_apcv2_replay_equivalence.py DEFAULT_MARGIN_NATS.
CONTINUOUS_BATCH_NEAR_TIE_NATS = 0.5
_LOGPROB_TOLERANCE = 1e-6


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _continuous_batch_token_trace(row):
    """Parse one row's per-token evidence, or ``None`` when it is unusable.

    Each entry is ``{"id", "logprob", "top": [[id, logprob], [id, logprob]]}``
    from the serving engine's per-token ``logprob`` event. The trace must cover
    every generated token the receipt counts.
    """
    tokens = row.get("tokens")
    count = (row.get("receipt") or {}).get("completion_tokens")
    if (
        not isinstance(tokens, list)
        or not tokens
        or type(count) is not int
        or count != len(tokens)
    ):
        return None
    trace = []
    for entry in tokens:
        if not isinstance(entry, dict):
            return None
        token = entry.get("id")
        logprob = entry.get("logprob")
        top = entry.get("top")
        if (
            type(token) is not int
            or token < 0
            or not _finite_number(logprob)
            or not isinstance(top, list)
            or len(top) != 2
        ):
            return None
        alternatives = {}
        for pair in top:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                return None
            alternative, alternative_logprob = pair
            if (
                type(alternative) is not int
                or alternative < 0
                or alternative in alternatives
                or not _finite_number(alternative_logprob)
            ):
                return None
            alternatives[alternative] = float(alternative_logprob)
        if (
            token in alternatives
            and abs(alternatives[token] - logprob) > _LOGPROB_TOLERANCE
        ):
            return None
        trace.append((token, alternatives))
    return trace


def _continuous_batch_answer_ok(prompt, text):
    # The prompt asks for one word: the whole answer must be that word (an
    # accepted token inside "Four is wrong; it is five" is not a pass).
    words = re.findall(r"[a-z0-9]+", text.casefold())
    return len(words) == 1 and words[0] in NATIVE_CONTINUOUS_BATCH_ANSWERS[prompt]


def _classify_continuous_batch_row(prompt, reference, batched):
    if (
        reference["text"] == batched["text"]
        and reference["finish_reason"] == batched["finish_reason"]
    ):
        return {"classification": "exact"}

    def failed(reason, **evidence):
        return {"classification": "failed", "reason": reason, **evidence}

    reference_trace = _continuous_batch_token_trace(reference)
    batched_trace = _continuous_batch_token_trace(batched)
    if reference_trace is None or batched_trace is None:
        return failed("token evidence is missing or malformed")
    index = next(
        (
            position
            for position, (left, right) in enumerate(
                zip(reference_trace, batched_trace)
            )
            if left[0] != right[0]
        ),
        None,
    )
    if index is None:
        # Identical ids up to the shorter trace cannot explain differing
        # output; a length-only divergence is not near-tie evidence.
        return failed("token traces do not contain a token flip")
    reference_token, reference_top = reference_trace[index]
    batched_token, batched_top = batched_trace[index]
    flip = {reference_token, batched_token}
    evidence = {
        "first_divergence_index": index,
        "reference_token_id": reference_token,
        "batched_token_id": batched_token,
    }
    if set(reference_top) != flip or set(batched_top) != flip:
        return failed(
            "flipped tokens are not each other's recorded top-2 alternatives",
            **evidence,
        )
    margins = {
        "reference": reference_top[reference_token] - reference_top[batched_token],
        "batched": batched_top[batched_token] - batched_top[reference_token],
    }
    evidence["margins_nats"] = margins
    if any(margin < -_LOGPROB_TOLERANCE for margin in margins.values()):
        return failed("a selected token is below its recorded alternative", **evidence)
    if any(margin > CONTINUOUS_BATCH_NEAR_TIE_NATS for margin in margins.values()):
        return failed("first divergence is not a near tie", **evidence)
    if not (
        _continuous_batch_answer_ok(prompt, reference["text"])
        and _continuous_batch_answer_ok(prompt, batched["text"])
    ):
        return failed(
            "near-tie continuation fails the functional oracle", **evidence
        )
    return {"classification": "near_tie", **evidence}


def _classify_continuous_batch_trace_unchecked(batch):
    result = {
        "schema": CONTINUOUS_BATCH_EQUIVALENCE_SCHEMA,
        "margin_ceiling_nats": CONTINUOUS_BATCH_NEAR_TIE_NATS,
        "passed": False,
        "rows": {},
    }
    if not isinstance(batch, dict):
        return {**result, "reason": "batch trace is missing"}
    width = batch.get("peak_observed_width")
    if type(width) is not int or width < 2:
        return {**result, "reason": "batch never reached physical width two"}
    prompts = set(NATIVE_CONTINUOUS_BATCH_PROMPTS)

    def index_rows(rows, *, cold):
        if not isinstance(rows, list) or len(rows) != len(prompts):
            return None
        indexed = {}
        request_ids = set()
        for row in rows:
            if not isinstance(row, dict):
                return None
            prompt = row.get("prompt")
            receipt = row.get("receipt")
            request_id = row.get("request_id")
            if (
                prompt not in prompts
                or row.get("prompt_sha256")
                != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                or type(request_id) is not str
                or not request_id
                or request_id in request_ids
                or prompt in indexed
                or not isinstance(receipt, dict)
                or receipt.get("request_id") != request_id
                or receipt.get("route") != "ordinary"
                or receipt.get("qualification") != "candidate"
                or row.get("finish_reason") not in ("length", "stop")
                or not isinstance(row.get("text"), str)
                or not row["text"]
                or (
                    cold
                    and (
                        type(receipt.get("cached_tokens")) is not int
                        or receipt["cached_tokens"] != 0
                    )
                )
            ):
                return None
            indexed[prompt] = row
            request_ids.add(request_id)
        if set(indexed) != prompts:
            return None
        return indexed, request_ids

    reference = index_rows(batch.get("reference_rows"), cold=True)
    concurrent = index_rows(batch.get("rows"), cold=False)
    if reference is None or concurrent is None:
        return {**result, "reason": "batch rows are malformed"}
    reference_rows, reference_ids = reference
    concurrent_rows, concurrent_ids = concurrent
    if reference_ids & concurrent_ids:
        return {**result, "reason": "reference and batched request ids overlap"}
    rows = {
        prompt: _classify_continuous_batch_row(
            prompt, reference_rows[prompt], concurrent_rows[prompt]
        )
        for prompt in NATIVE_CONTINUOUS_BATCH_PROMPTS
    }
    return {
        **result,
        "passed": all(row["classification"] != "failed" for row in rows.values()),
        "rows": rows,
    }


def classify_continuous_batch_trace(batch):
    """Recompute per-row batched-versus-single equivalence from raw traces.

    Rows whose text and finish reason match the isolated reference are
    ``exact``. Otherwise a row passes only as ``near_tie``: both token traces
    are present and cover the receipt's completion tokens, the generated ids
    agree before the first differing token, the two selected tokens are
    exactly each side's recorded top-2, and both selected-minus-alternate
    margins are within ``CONTINUOUS_BATCH_NEAR_TIE_NATS``, and both texts
    pass the prompt's answer oracle. Anything else fails closed. Never trusts a producer-written classification.
    """
    try:
        return _classify_continuous_batch_trace_unchecked(batch)
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return {
            "schema": CONTINUOUS_BATCH_EQUIVALENCE_SCHEMA,
            "margin_ceiling_nats": CONTINUOUS_BATCH_NEAR_TIE_NATS,
            "passed": False,
            "rows": {},
            "reason": "batch trace is malformed",
        }


def _valid_continuous_batch_trace(batch):
    return classify_continuous_batch_trace(batch)["passed"] is True


NATIVE_VLM_MODALITIES = {
    "gemma3n": ("image", "video", "audio"),
    "gemma4": ("image", "video"),
    "minicpmo": ("image", "audio"),
}


def _evaluate_native_vlm_traces_unchecked(
    report,
    *,
    producer_sha256,
    expected_source_revision,
    expected_source_sha256,
    expected_artifact,
    expected_runtime,
    expected_settings,
    expected_serving_runtime=None,
):
    family = report.get("family") if isinstance(report, dict) else None
    modalities = NATIVE_VLM_MODALITIES.get(family, ())
    if not modalities or not isinstance(report, dict):
        return {}
    ordinary = report.get("ordinary_reference", {})
    checks = {
        k: False
        for k in (
            "ordinary_reference",
            "continuous_batch",
            "receipts",
            "source_route_parity",
            "prefix_reuse",
            "replay",
            *modalities,
        )
    }
    if (
        not _hex(producer_sha256)
        or report.get("producer_sha256") != producer_sha256
        or report.get("reference_revision") != expected_source_revision
        or report.get("reference_source_sha256") != expected_source_sha256
        or report.get("artifact_sha256") != expected_artifact
        or report.get("source_runtime") != expected_runtime
        or report.get("settings") != expected_settings
        or (
            expected_serving_runtime is not None
            and report.get("runtime") != expected_serving_runtime
        )
        or not isinstance(report.get("ordinary_reference"), dict)
        or not isinstance(report.get("direct"), dict)
        or not isinstance(report.get("arms"), dict)
        or set(report["arms"]) != set(modalities)
        or any(
            not isinstance(report["arms"].get(kind), dict)
            or set(report["arms"][kind])
            != {"cold", "warm", "changed", "return_original"}
            or any(
                not isinstance(row, dict) or not isinstance(row.get("receipt"), dict)
                for row in report["arms"][kind].values()
            )
            for kind in modalities
        )
        or any(
            not isinstance((report.get("direct") or {}).get(kind), dict)
            or not isinstance(report["direct"][kind].get("fixture"), dict)
            or not isinstance(report["direct"][kind].get("parity"), dict)
            for kind in modalities
        )
        or not isinstance((ordinary.get("receipt")), dict)
        or not isinstance(
            (report.get("direct") or {}).get("ordinary_text_parity"), dict
        )
        or not isinstance(report.get("batch"), dict)
        or not isinstance(report["batch"].get("rows"), list)
        or not isinstance(report["batch"].get("reference_rows"), list)
        or any(
            not isinstance(row, dict) or not isinstance(row.get("receipt"), dict)
            for row in report["batch"]["rows"]
        )
    ):
        return checks
    direct = report["direct"]
    ordinary = report["ordinary_reference"]
    defaults = ((ordinary.get("receipt") or {}).get("request_controls") or {}).get(
        "sampling_defaults"
    ) or {}
    tp = direct.get("ordinary_text_parity") or {}
    checks["ordinary_reference"] = (
        ordinary.get("finish_reason") in ("length", "stop")
        and bool(ordinary.get("text"))
        and (ordinary.get("receipt") or {}).get("route") == "ordinary"
        and (ordinary.get("receipt") or {}).get("qualification") == "candidate"
        and defaults.get("schema") == "mlx2.sampling-defaults.v1"
        and tp.get("reference_revision") == expected_source_revision
        and tp.get("reference_source_sha256") == expected_source_sha256
        and len(tp.get("rows", [])) == 5
        and all(
            r.get("argmax_match") is True
            and type(r.get("max_abs")) in (int, float)
            and math.isfinite(r["max_abs"])
            and 0 <= r["max_abs"] <= 1e-4
            for r in tp["rows"]
        )
    )
    batch = report.get("batch") or {}
    checks["continuous_batch"] = _valid_continuous_batch_trace(batch)
    checks["receipts"] = all(
        isinstance(report["arms"].get(k), dict)
        and all(
            r.get("finish_reason") in ("length", "stop")
            and bool(r.get("text"))
            and (r.get("receipt") or {}).get("route") == "ordinary"
            and (r.get("receipt") or {}).get("qualification") == "candidate"
            for r in report["arms"][k].values()
        )
        for k in modalities
    )
    parity = [tp]
    parity.extend((direct.get(k) or {}).get("parity") or {} for k in modalities)
    checks["source_route_parity"] = all(
        p.get("reference_revision") == expected_source_revision
        and p.get("reference_source_sha256") == expected_source_sha256
        and len(p.get("rows", [])) == 5
        and all(
            r.get("argmax_match") is True
            and type(r.get("max_abs")) in (int, float)
            and math.isfinite(r["max_abs"])
            and 0 <= r["max_abs"] <= 1e-4
            for r in p["rows"]
        )
        for p in parity
    )
    prefix = []
    replay = []
    for kind in modalities:
        f = (direct.get(kind) or {}).get("fixture") or {}
        rows = report["arms"].get(kind) or {}
        try:
            raw = base64.b64decode(f.get("payload_base64", ""), validate=True)
            changed_raw = base64.b64decode(
                f.get("changed_payload_base64", ""), validate=True
            )
            payload_ok = (
                hashlib.sha256(raw).hexdigest() == f.get("sha256")
                and hashlib.sha256(changed_raw).hexdigest() == f.get("changed_sha256")
                and len(raw) == f.get("bytes")
            )
        except (ValueError, TypeError):
            payload_ok = False
        checks[kind] = (
            payload_ok
            and _hex(f.get("sha256"))
            and _hex(f.get("changed_sha256"))
            and f["sha256"] != f["changed_sha256"]
            and type(f.get("media_token_end")) is int
            and type(f.get("prompt_tokens")) is int
            and 0 < f["media_token_end"] < f["prompt_tokens"]
            and type(f.get("media_token_start")) is int
            and 0 <= f["media_token_start"] < f["media_token_end"]
            and type(f.get("changed_media_token_start")) is int
            and type(f.get("changed_media_token_end")) is int
            and 0 <= f["changed_media_token_start"] < f["changed_media_token_end"]
            and bool(f.get("media_fingerprint"))
            and bool(f.get("changed_media_fingerprint"))
            and f["media_fingerprint"] != f["changed_media_fingerprint"]
            and bool((rows.get("cold") or {}).get("text"))
        )
        cold = rows.get("cold", {}).get("receipt") or {}
        warm = rows.get("warm", {}).get("receipt") or {}
        changed = rows.get("changed", {}).get("receipt") or {}
        returned = rows.get("return_original", {}).get("receipt") or {}
        prefix.append(
            cold.get("cached_tokens") == 0
            and type(warm.get("cached_tokens")) is int
            and warm["cached_tokens"] >= f.get("media_token_end", 1)
        )
        replay.append(
            rows.get("cold", {}).get("text")
            == rows.get("warm", {}).get("text")
            == rows.get("return_original", {}).get("text")
            and returned.get("cached_tokens", 0) > 0
            and changed.get("route") == "ordinary"
            and type(changed.get("cached_tokens")) is int
            and changed["cached_tokens"] <= f.get("changed_media_token_start", -1)
        )
    delta = report.get("apc_stats_delta")
    if not isinstance(delta, dict) or any(
        not isinstance(key, str) or type(value) is not int or value < 0
        for key, value in delta.items()
    ):
        delta = {}
    apc_hits = sum(value for key, value in delta.items() if "hit" in key.lower())
    checks["prefix_reuse"] = all(prefix) and apc_hits >= len(modalities)
    checks["replay"] = all(replay)
    return checks


def evaluate_native_vlm_traces(report, **kwargs):
    """Fail closed on malformed nested producer data instead of raising."""
    try:
        return _evaluate_native_vlm_traces_unchecked(report, **kwargs)
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        family = report.get("family") if isinstance(report, dict) else None
        modalities = NATIVE_VLM_MODALITIES.get(family, ())
        if not modalities:
            return {}
        return {
            key: False
            for key in (
                "ordinary_reference",
                "continuous_batch",
                "receipts",
                "source_route_parity",
                "prefix_reuse",
                "replay",
                *modalities,
            )
        }


NATIVE_VLM_MEDIA_CHECKS = {
    "gemma3n": frozenset(
        {
            "multimodal_image",
            "multimodal_video",
            "multimodal_audio_input",
            "multimodal_encoder_batching",
            "multimodal_encoder_unsplit",
            "multimodal_continuous_batch",
            "multimodal_apcv2_reuse",
        }
    ),
    "gemma4": frozenset(
        {
            "multimodal_image",
            "multimodal_video",
            "multimodal_continuous_batch",
            "multimodal_apcv2_reuse",
        }
    ),
    "minicpmo": frozenset(
        {
            "multimodal_image",
            "multimodal_audio_input",
            "multimodal_encoder_batching",
            "multimodal_encoder_sequential",
            "multimodal_continuous_batch",
            "multimodal_apcv2_reuse",
        }
    ),
}


def resolve_conditional_qualification_checks(rules, settings):
    """Resolve descriptor-declared mechanism checks against exact settings."""
    if not isinstance(rules, (list, tuple)) or not isinstance(settings, dict):
        raise TypeError("conditional qualification policy is malformed")
    selected = set()
    by_path = {}
    all_checks = set()
    for rule in rules:
        if not isinstance(rule, dict) or set(rule) != {"setting", "equals", "check"}:
            raise ValueError("conditional qualification rule is malformed")
        path, expected, check = rule["setting"], rule["equals"], rule["check"]
        if (
            not isinstance(path, (list, tuple))
            or not path
            or any(not isinstance(key, str) or not key for key in path)
            or type(expected) is not bool
            or not isinstance(check, str)
            or not check
        ):
            raise ValueError("conditional qualification rule is malformed")
        path = tuple(path)
        if check in all_checks:
            raise ValueError("conditional qualification checks must be unique")
        all_checks.add(check)
        by_path.setdefault(path, []).append((expected, check))
    for path, cases in by_path.items():
        value = settings
        for key in path:
            if not isinstance(value, dict) or key not in value:
                raise ValueError("conditional qualification setting is missing")
            value = value[key]
        matches = [check for expected, check in cases if type(value) is bool and value is expected]
        if len(matches) != 1 or len(cases) != 2 or {expected for expected, _ in cases} != {True, False}:
            raise ValueError("conditional qualification setting is invalid or ambiguous")
        selected.add(matches[0])
    return selected


def _canonical_policy_check_rules(rules):
    if not isinstance(rules, (list, tuple)):
        return None
    canonical = []
    for rule in rules:
        if not isinstance(rule, dict) or set(rule) != {"setting", "equals", "check"}:
            return None
        path = rule.get("setting")
        if not isinstance(path, (list, tuple)):
            return None
        canonical.append(
            {
                "setting": list(path),
                "equals": rule.get("equals"),
                "check": rule.get("check"),
            }
        )
    return canonical


def _native_vlm_check_names(family, report, expected_policy_checks):
    checks = set(NATIVE_VLM_MEDIA_CHECKS[family])
    if not expected_policy_checks:
        return checks
    all_conditional = {
        rule.get("check")
        for rule in expected_policy_checks
        if isinstance(rule, dict) and isinstance(rule.get("check"), str)
    }
    checks.difference_update(all_conditional)
    settings = report.get("settings") if isinstance(report, dict) else None
    checks.update(
        resolve_conditional_qualification_checks(expected_policy_checks, settings)
    )
    return checks


def evaluate_native_vlm_report(
    report, *, expected_family, expected_harness, expected_policy_checks=()
):
    if expected_family not in NATIVE_VLM_MEDIA_CHECKS:
        raise ValueError(
            f"unsupported native VLM qualification family: {expected_family!r}"
        )
    try:
        return _evaluate_native_vlm_report_unchecked(
            report,
            expected_family=expected_family,
            expected_harness=expected_harness,
            expected_policy_checks=expected_policy_checks,
        )
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        names = set(NATIVE_VLM_MEDIA_CHECKS[expected_family])
        names.update(
            rule.get("check")
            for rule in expected_policy_checks
            if isinstance(rule, dict) and isinstance(rule.get("check"), str)
        )
        return {name: False for name in names}


NATIVE_VLM_SOURCE_REVISIONS = {
    "gemma3n": "67599f2e8ec31bf35cbb7b02794114f20844f0bb",
    "gemma4": "67599f2e8ec31bf35cbb7b02794114f20844f0bb",
    "minicpmo": "67599f2e8ec31bf35cbb7b02794114f20844f0bb",
}


def evaluate_encoder_batching_observation(value):
    """Accept only observed encoder calls that exceed and obey a chunk policy."""
    if (
        not isinstance(value, dict)
        or value.get("schema") != "mlx2.encoder-batching-observation.v1"
    ):
        return False
    limit = value.get("policy_limit")
    calls = value.get("calls")
    if (
        value.get("enabled") is not True
        or type(limit) is not int
        or limit < 2
        or not isinstance(calls, list)
        or not calls
    ):
        return False
    observed_overflow = False
    for call in calls:
        if not isinstance(call, dict):
            return False
        count, chunks = call.get("input_count"), call.get("chunk_sizes")
        if (
            type(count) is not int
            or count < 1
            or not isinstance(chunks, list)
            or not chunks
        ):
            return False
        if any(type(size) is not int or size < 1 or size > limit for size in chunks):
            return False
        if sum(chunks) != count:
            return False
        observed_overflow |= count > limit and len(chunks) > 1 and max(chunks) > 1
    return observed_overflow


def evaluate_encoder_unsplit_observation(value, policy):
    """Accept only observed encoder calls that each ran as one batch under a
    selected non-splitting Gemma 3n video policy."""
    if (
        not isinstance(value, dict)
        or value.get("schema") != "mlx2.encoder-batching-observation.v1"
        or not isinstance(policy, dict)
        or policy.get("video_frame_batch_splits") is not False
    ):
        return False
    limit = value.get("policy_limit")
    calls = value.get("calls")
    if (
        value.get("enabled") is not True
        or type(limit) is not int
        or limit < 1
        or type(policy.get("video_frame_batch_size")) is not int
        or limit != policy["video_frame_batch_size"]
        or not isinstance(calls, list)
        or not calls
    ):
        return False
    for call in calls:
        if not isinstance(call, dict):
            return False
        count, chunks = call.get("input_count"), call.get("chunk_sizes")
        if type(count) is not int or not 1 <= count <= limit or chunks != [count]:
            return False
    return True


def evaluate_encoder_sequential_observation(observation, policy, feature_parity):
    """Require the selected single-image path to execute and match source."""
    if (
        not isinstance(observation, dict)
        or observation.get("schema") != "mlx2.encoder-execution-observation.v1"
        or not isinstance(policy, dict)
        or observation.get("policy") != policy
        or policy.get("batch_vision_input") is not False
        or type(policy.get("vision_batch_size")) is not int
        or policy["vision_batch_size"] != 1
        or type(observation.get("input_count")) is not int
        or observation["input_count"] < 2
        or not isinstance(observation.get("tower_call_batch_sizes"), list)
        or len(observation["tower_call_batch_sizes"]) != observation["input_count"]
        or any(type(size) is not int or size != 1 for size in observation["tower_call_batch_sizes"])
    ):
        return False
    return feature_parity is None or not (
        not isinstance(feature_parity, dict)
        or feature_parity.get("passed") is not True
        or feature_parity.get("shapes_match") is not True
        or type(feature_parity.get("tensor_count")) is not int
        or feature_parity["tensor_count"] < 1
        or type(feature_parity.get("max_abs")) not in (int, float)
        or not math.isfinite(feature_parity["max_abs"])
        or not 0 <= feature_parity["max_abs"] <= 1e-4
    )


def _evaluate_native_vlm_report_unchecked(
    report, *, expected_family, expected_harness, expected_policy_checks=()
):
    """Recompute native VLM adapter checks from the source/media/cache traces."""
    if expected_family not in NATIVE_VLM_MEDIA_CHECKS:
        raise ValueError(
            f"unsupported native VLM qualification family: {expected_family!r}"
        )
    checks = {
        name: False
        for name in _native_vlm_check_names(
            expected_family, report, expected_policy_checks
        )
    }
    if not isinstance(report, dict) or not isinstance(expected_harness, dict):
        return checks
    if _canonical_policy_check_rules(
        report.get("conditional_qualification_checks", [])
    ) != _canonical_policy_check_rules(expected_policy_checks):
        return checks
    runtime = report.get("source_runtime")
    revision = NATIVE_VLM_SOURCE_REVISIONS[expected_family]
    source_sha = runtime.get("source_sha256") if isinstance(runtime, dict) else None
    if (
        not isinstance(report, dict)
        or report.get("model_type") != expected_family
        or report.get("qualification_harness") != expected_harness
        or report.get("source_revision") != revision
        or not source_contract_identity(runtime, revision, family=expected_family)
        or report.get("mlx_vlm_runtime") != runtime
        or not isinstance(report.get("settings"), dict)
        or report["settings"].get("mlx_vlm") != runtime
    ):
        return checks
    traces = evaluate_native_vlm_traces(
        report,
        producer_sha256=expected_harness.get("sha256"),
        expected_source_revision=revision,
        expected_source_sha256=source_sha,
        expected_artifact=report.get("artifact"),
        expected_runtime=runtime,
        expected_settings=report.get("settings"),
        expected_serving_runtime=report.get("runtime"),
    )
    for kind in ("image", "video", "audio"):
        mapped = {"audio": "multimodal_audio_input"}.get(kind, f"multimodal_{kind}")
        if mapped in checks:
            checks[mapped] = (
                traces.get(kind) is True and traces.get("source_route_parity") is True
            )
    if "multimodal_continuous_batch" in checks:
        checks["multimodal_continuous_batch"] = traces.get("continuous_batch") is True
    if "multimodal_encoder_batching" in checks:
        probe_kind = "video" if expected_family == "gemma3n" else "image"
        probe = (report.get("direct") or {}).get(probe_kind) or {}
        parity = probe.get("parity") or {}
        feature_parity = parity.get("feature_parity") or {}
        checks["multimodal_encoder_batching"] = (
            evaluate_encoder_batching_observation(probe.get("encoder_batching"))
            and evaluate_encoder_batching_observation(
                report.get("serving_encoder_batching")
            )
            and feature_parity.get("passed") is True
            and feature_parity.get("shapes_match") is True
            and type(feature_parity.get("tensor_count")) is int
            and feature_parity["tensor_count"] > 0
            and type(feature_parity.get("max_abs")) in (int, float)
            and math.isfinite(feature_parity["max_abs"])
            and 0 <= feature_parity["max_abs"] <= 1e-4
        )
    if "multimodal_encoder_unsplit" in checks:
        probe = (report.get("direct") or {}).get("video") or {}
        parity = probe.get("parity") or {}
        feature_parity = parity.get("feature_parity") or {}
        policy = (report.get("settings") or {}).get("adapter_policy") or {}
        checks["multimodal_encoder_unsplit"] = (
            evaluate_encoder_unsplit_observation(probe.get("encoder_batching"), policy)
            and evaluate_encoder_unsplit_observation(
                report.get("serving_encoder_batching"), policy
            )
            and feature_parity.get("passed") is True
            and feature_parity.get("shapes_match") is True
            and type(feature_parity.get("tensor_count")) is int
            and feature_parity["tensor_count"] > 0
            and type(feature_parity.get("max_abs")) in (int, float)
            and math.isfinite(feature_parity["max_abs"])
            and 0 <= feature_parity["max_abs"] <= 1e-4
        )
    if "multimodal_encoder_sequential" in checks:
        probe = (report.get("direct") or {}).get("image") or {}
        parity = probe.get("parity") or {}
        feature_parity = parity.get("feature_parity")
        direct_observation = (
            feature_parity.get("encoder_execution")
            if isinstance(feature_parity, dict)
            else None
        )
        serving_observation = report.get("serving_encoder_execution") or {}
        policy = (report.get("settings") or {}).get("adapter_policy") or {}
        checks["multimodal_encoder_sequential"] = (
            isinstance(feature_parity, dict)
            and evaluate_encoder_sequential_observation(
                direct_observation, policy, feature_parity
            )
            and evaluate_encoder_sequential_observation(
                serving_observation,
                policy,
                None,
            )
        )
    checks["multimodal_apcv2_reuse"] = (
        traces.get("prefix_reuse") is True
        and traces.get("replay") is True
        and traces.get("receipts") is True
        and traces.get("ordinary_reference") is True
    )
    return checks
