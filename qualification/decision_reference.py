"""Qualification-only reference calculations derived from pinned upstream code.

This module is not packaged or imported by serving. Clef and JEV execute the
actual pinned mlx-vlm source definitions; Decision2 and pplx use small direct
translations of their pinned C++ and SGLang scoring contracts.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace


def _source_record(repository: Path, revision: str, relative: str) -> tuple[str, dict]:
    pins = {
        (
            "01d6ebaeaa4f2dc2394204798a2032e38d8a2841",
            "mlx_vlm/models/clef/clef.py",
        ): "2a21346024251085512a7fe2c68c0c0660e04d9e6abadc9f950ba635751911cd",
        (
            "681af1cc55bea1acd1d4d2b5ce1f530ce8e232d1",
            "mlx_vlm/models/jev/jev.py",
        ): "1ac07c2d3dc0e8f66987150eb813369ff517f4fea0773e0b7814f2fcd208355d",
        (
            "3c64a581d4f95bb2de409d383741fa5b18d84345",
            "tools/server/server-decision.cpp",
        ): "0b9ef91fe0d7f14f7c9d53ec839c998c359138462500a0ca7d2f2e0f90bf9f53",
        (
            "3c64a581d4f95bb2de409d383741fa5b18d84345",
            "conversion/decision2.py",
        ): "7575180de1192f8ada501dd94513fe99ec42dc6085561f9f10ecd5c10ad3a910",
        (
            "6d7712222798f9728eba8bd603afe50a037e063c",
            "python/sglang/srt/entrypoints/systemone/serving.py",
        ): "4acef7a496288c68fe5d6debd0201d1bec515e8881a84a0b6dc7a4d0b5b1d0f1",
    }
    expected = pins.get((revision, relative))
    if expected is None:
        raise RuntimeError("qualification reference is not allowlisted")
    payload = subprocess.check_output(
        ["git", "-C", str(repository), "show", f"{revision}:{relative}"]
    )
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"pinned reference object is {actual}, expected {expected}"
        )
    return payload.decode(), {
        "repository": str(repository),
        "revision": revision,
        "path": relative,
        "sha256": actual,
    }


def _probabilities(answer: dict) -> dict[str, float]:
    if answer["type"] in {"noul", "bool"}:
        value = float(answer["probability"])
        return {"true": value, "false": round(1.0 - value, 4)}
    return {str(key): float(value) for key, value in answer["probabilities"].items()}


def _format_answers(rows) -> dict:
    answers = {}
    for name, question, labels, probabilities in rows:
        values = dict(zip(labels, (float(value) for value in probabilities)))
        kind = question["type"]
        if kind == "noul":
            probability = round(values["true"], 4)
            answers[name] = {
                "type": "noul",
                "value": probability >= 0.5,
                "probability": probability,
                "confidence": round(max(values.values()), 4),
            }
            continue
        best = max(labels, key=values.__getitem__)
        answer = {
            "type": kind,
            "value": (
                best
                if kind == "choice"
                else round(
                    sum(index * values[label] for index, label in enumerate(labels)),
                    4,
                )
            ),
            "probabilities": {
                label: round(values[label], 4) for label in labels
            },
            "confidence": round(values[best], 4),
        }
        if kind == "score":
            answer["legend"] = dict(zip(labels, question["criteria"]))
        answers[name] = answer
    return answers


def _comparison(production: dict, reference: dict, sources: list[dict]) -> dict:
    errors = {}
    for name, answer in production["answers"].items():
        left = _probabilities(answer)
        right = _probabilities(reference["answers"][name])
        if set(left) != set(right):
            errors[name] = {"labels": [sorted(left), sorted(right)]}
            continue
        errors[name] = {
            key: abs(left[key] - right[key]) for key in left
        }
    maximum = max(
        (value for row in errors.values() for value in row.values() if isinstance(value, float)),
        default=0.0,
    )
    return {
        "passed": all(
            left == _probabilities(reference["answers"][name])
            for name, left in (
                (name, _probabilities(answer))
                for name, answer in production["answers"].items()
            )
        ),
        "maximum_probability_error": maximum,
        "production_input_tokens": production["usage"]["input_tokens"],
        "reference_input_tokens": reference["usage"]["input_tokens"],
        "input_tokens_match": (
            production["usage"]["input_tokens"]
            == reference["usage"]["input_tokens"]
        ),
        "errors": errors,
        "sources": sources,
    }


def _clef(engine, request, upstream: Path) -> tuple[dict, list[dict]]:
    import mlx.core as mx
    import numpy as np
    from mlx import nn
    from mlx.utils import tree_flatten

    revision = "01d6ebaeaa4f2dc2394204798a2032e38d8a2841"
    relative = "mlx_vlm/models/clef/clef.py"
    source, record = _source_record(upstream, revision, relative)
    namespace = {"json": json, "math": math, "mx": mx, "nn": nn, "np": np}
    start = source.index("QUESTION_TYPES =")
    model = source.index("class Model(", start)
    render = source.index("def _render_criterion", model)
    exec(source[start:model], namespace)  # noqa: S102 - pinned source reference
    exec(source[render:], namespace)  # noqa: S102 - pinned source reference

    head = namespace["JointSchemaHead"](**engine.artifact["config"]["head_config"])
    head.load_weights(tree_flatten(engine.head.parameters()), strict=True)
    head.eval()
    mx.eval(head.parameters())

    class ReferenceModel:
        def __init__(self, reference_head):
            self.head = reference_head

        def __call__(self, input_ids, question_spans, option_spans, qtype, **_media):
            hidden = engine.model.model(input_ids)[0]
            hidden = self.head.hidden_norm(hidden)
            flat = input_ids[0].tolist()
            lexical_ids = [
                token for start_, end_ in option_spans for token in flat[start_:end_]
            ]
            boundaries = [0]
            for start_, end_ in option_spans:
                boundaries.append(boundaries[-1] + end_ - start_)
            lm_head = engine.model.language_model.lm_head
            ids = mx.array(lexical_ids)
            embeddings = lm_head.weight[ids]
            if hasattr(lm_head, "scales"):
                biases = getattr(lm_head, "biases", None)
                embeddings = mx.dequantize(
                    embeddings,
                    lm_head.scales[ids],
                    None if biases is None else biases[ids],
                    group_size=lm_head.group_size,
                    bits=lm_head.bits,
                    mode=lm_head.mode,
                )
            spans = list(pairwise(boundaries))
            lexical = namespace["_span_means"](spans, len(lexical_ids)) @ embeddings
            return self.head(
                hidden,
                [span for span, _count in question_spans],
                option_spans,
                lexical.astype(hidden.dtype),
                qtype,
                [count for _span, count in question_spans],
            )

    result = namespace["Clef"](ReferenceModel(head), engine.tokenizer).predict(
        request.state, dict(request.questions)
    )
    del head
    mx.clear_cache()
    return result, [record]


def _jev(engine, request, upstream: Path) -> tuple[dict, list[dict]]:
    import mlx.core as mx

    revision = "681af1cc55bea1acd1d4d2b5ce1f530ce8e232d1"
    relative = "mlx_vlm/models/jev/jev.py"
    source, record = _source_record(upstream, revision, relative)
    namespace = {"json": json, "mx": mx, "string": __import__("string")}
    start = source.index("def _render_criterion")
    exec(source[start:], namespace)  # noqa: S102 - pinned source reference
    artifact_settings = json.loads(
        (engine.artifact["path"] / "config.json").read_text()
    )["decision_config"]
    if artifact_settings != engine.settings:
        raise RuntimeError("JEV loaded calibration differs from pinned artifact metadata")

    class ReferenceModel:
        config = SimpleNamespace(
            decision_config=artifact_settings,
            model_type="jev_text",
        )

        def __call__(self, ids, **_media):
            hidden = engine.model.model(ids)[0, -1]
            return engine.model.logits(hidden)[None]

    result = namespace["Jev"](ReferenceModel(), engine.tokenizer).predict(
        request.state, dict(request.questions)
    )
    mx.clear_cache()
    return result, [record]


def _canonical(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _decision2(engine, request, upstream: Path) -> tuple[dict, list[dict]]:
    import mlx.core as mx
    from mlx import nn

    revision = "3c64a581d4f95bb2de409d383741fa5b18d84345"
    server_source, server_record = _source_record(
        upstream, revision, "tools/server/server-decision.cpp"
    )
    for marker in (
        "Select the single option best supported by the context and instructions.",
        "score_bias",
    ):
        if marker not in server_source:
            raise RuntimeError(f"Decision2 pinned server source is missing {marker!r}")
    conversion_source, conversion_record = _source_record(
        upstream, revision, "conversion/decision2.py"
    )
    for marker in (
        '"candidate_mlp.weight": (self.head_dim, hidden)',
        '"query_mlp.weight": (self.head_dim, hidden)',
        '"scalar.weight": (1, self.head_dim)',
    ):
        if marker not in conversion_source:
            raise RuntimeError(
                f"Decision2 pinned conversion source is missing {marker!r}"
            )
    rows = []
    tokens_total = 0
    for name, question in request.questions.items():
        criteria = question.get("criteria")
        if question["type"] == "noul":
            criteria = dict(criteria or {})
            options = (
                [(str(key), value) for key, value in criteria.items()]
                if len(criteria) == 2
                else [
                    ("false", criteria.get("false", "No")),
                    ("true", criteria.get("true", "Yes")),
                ]
            )
        elif question["type"] == "score":
            options = [(str(index), value) for index, value in enumerate(criteria)]
        else:
            options = [(str(key), value) for key, value in criteria.items()]
        state = _canonical(request.state)
        suffix = (
            f"\n\nTask type: {question['type']}\nQuestion:\n"
            f"{_canonical(question['instructions'])}\nOptions:"
        )
        encode = lambda text: list(engine.tokenizer.encode(text, add_special_tokens=False))
        sequence = encode("Context:\n" + state + suffix)
        positions = []
        labels = []
        for label, description in options:
            sequence.extend(
                encode(
                    "\n<option>\n"
                    + _canonical({"description": description, "key": label})
                    + "\n</option>"
                )
            )
            positions.append(len(sequence) - 1)
            labels.append(label)
        sequence.extend(
            encode(
                "\n\nSelect the single option best supported by the context and "
                "instructions.\nDecision:"
            )
        )
        tokens_total += len(sequence)
        hidden = engine.model.model(mx.array([sequence]))[0]
        candidates = engine.head.candidate_norm(
            hidden[mx.array(positions)].astype(mx.float32)
        )
        query = engine.head.query_norm(hidden[-1].astype(mx.float32))
        bilinear = mx.sum(
            engine.head.key(candidates) * engine.head.query(query), axis=-1
        ) / math.sqrt(256)
        nonlinear = nn.gelu(
            engine.head.candidate_mlp(candidates) + engine.head.query_mlp(query)
        )
        scores = bilinear + engine.head.scalar(nonlinear).squeeze(-1)
        if question["type"] == "score":
            offsets = engine.artifact["score_bias"].get(len(labels))
            if offsets is not None:
                scores = scores + mx.array(offsets)
        values = mx.softmax(scores.astype(mx.float32)).tolist()
        rows.append((name, question, labels, values))
    return {
        "answers": _format_answers(rows),
        "usage": {"input_tokens": tokens_total, "output_tokens": 0},
    }, [server_record, conversion_record]


def _pplx(engine, request, upstream: Path) -> tuple[dict, list[dict]]:
    import mlx.core as mx

    revision = "6d7712222798f9728eba8bd603afe50a037e063c"
    relative = "python/sglang/srt/entrypoints/systemone/serving.py"
    source, record = _source_record(upstream, revision, relative)
    namespace = {
        "json": json,
        "Any": object,
        "List": list,
        "Tuple": tuple,
        "QuestionView": SimpleNamespace,
    }
    start = source.index("def _describe")
    end = source.index("def _legend", start)
    exec(source[start:end], namespace)  # noqa: S102 - pinned source reference
    system_start = source.index("_DECIDER_SYSTEM = (")
    system_end = source.index("\n\n\nclass SystemOneServing", system_start)
    exec(source[system_start:system_end], namespace)  # noqa: S102 - pinned source
    system = namespace["_DECIDER_SYSTEM"]
    rows = []
    token_total = 0
    for name, question in request.questions.items():
        criteria = question.get("criteria")
        if question["type"] == "noul":
            criteria = dict(criteria or {})
            labels = ["true", "false"]
            selected = [1, 0]
            view = SimpleNamespace(
                kind="yes_no",
                question=question.get("instructions"),
                names=["yes", "no"],
                details=[criteria.get("true"), criteria.get("false")],
            )
        elif question["type"] == "score":
            labels = [str(index) for index in range(len(criteria))]
            selected = list(range(len(criteria)))
            view = SimpleNamespace(
                kind="score",
                question=question.get("instructions"),
                names=labels,
                details=list(criteria),
            )
        else:
            labels = list(criteria)
            selected = list(range(len(labels)))
            view = SimpleNamespace(
                kind="choice",
                question=question.get("instructions"),
                names=labels,
                details=[criteria[label] for label in labels],
            )
        _codes, user = namespace["_decider_question"](
            text=namespace["_describe"](request.state),
            view=view,
            codes=engine.codes,
        )
        sequence = engine.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if isinstance(sequence, Mapping):
            sequence = sequence["input_ids"]
        sequence = [int(token) for token in sequence]
        token_total += len(sequence)
        hidden = engine.model.model(mx.array([sequence]))[0, -1].astype(mx.float32)
        weights = engine.readout[mx.array(selected)].astype(mx.float32)
        values = mx.softmax((hidden @ weights.T) / engine.temperature).tolist()
        rows.append((name, question, labels, values))
    return {
        "answers": _format_answers(rows),
        "usage": {"input_tokens": token_total, "output_tokens": 0},
    }, [record]


def _describe(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def compare_with_upstream(engine, request, sources: dict[str, Path]) -> dict:
    production = engine.predict(request)
    if engine.family == "clef":
        reference, records = _clef(engine, request, sources["mlx-vlm"])
    elif engine.family == "jev":
        reference, records = _jev(engine, request, sources["mlx-vlm"])
    elif engine.family == "decision2":
        reference, records = _decision2(engine, request, sources["llama.cpp"])
    elif engine.family == "pplx-decider":
        reference, records = _pplx(engine, request, sources["sglang"])
    else:
        raise ValueError(f"no decision reference for {engine.family}")
    result = _comparison(production, reference, records)
    result["passed"] = result["passed"] and result["input_tokens_match"]
    return result
