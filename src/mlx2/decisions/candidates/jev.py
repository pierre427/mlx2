"""Text-only JEV 9B calibrated verbalizer adapter.

Prompt, calibration, and verbalizers follow mlx-vlm PR 2466. Provenance is
recorded in ``provenance/candidate-decisions.json``.
"""

from __future__ import annotations

import json
import math
import string
from pathlib import Path
from typing import Any

from ..schema import DecisionInputTooLong, DecisionRequestError
from .base import (
    CandidateEngine,
    add_request_tokens,
    format_answers,
    inspect_index,
    load_qwen_backbone,
    read_object,
)

SOURCE_REVISION = "681af1cc55bea1acd1d4d2b5ce1f530ce8e232d1"
ARTIFACT_REVISION = "c0eafb30e090cb94cff6a762bd2c67baba29f131"


def inspect_artifact(model_path: str | Path) -> dict[str, Any]:
    root = Path(model_path).expanduser().resolve()
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    config = read_object(config_path)
    if config.get("model_type") != "jev_text":
        raise ValueError("JEV 9B requires a prepared model_type 'jev_text' artifact")
    topology = (config.get("num_hidden_layers"), config.get("hidden_size"))
    if topology != (32, 4096):
        raise ValueError(f"unsupported JEV 9B topology {topology!r}")
    decision = config.get("decision_config")
    if not isinstance(decision, dict):
        raise TypeError("JEV 9B requires embedded decision_config")
    if decision.get("ranges") != {
        "noul": [0, 2],
        "score": [2, 8],
        "choice": [8, 24],
    }:
        raise ValueError("JEV decision ranges do not match the published layout")
    verbalizers = decision.get("verbalizer_ids")
    bias = decision.get("bias")
    temperatures = decision.get("temperature_by_type")
    if (
        not isinstance(verbalizers, list)
        or len(verbalizers) != 24
        or not isinstance(bias, list)
        or len(bias) != 24
        or not isinstance(temperatures, dict)
        or set(temperatures) != {"noul", "choice", "score"}
    ):
        raise ValueError("JEV calibration metadata is incomplete")
    if any(type(value) is not int or value < 0 for value in verbalizers):
        raise ValueError("JEV verbalizer token ids are invalid")
    vocab_size = config.get("vocab_size")
    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("JEV vocab_size must be a positive integer")
    if any(value >= vocab_size for value in verbalizers):
        raise ValueError("JEV verbalizer token ids exceed the vocabulary")
    if any(
        type(value) not in (int, float) or not math.isfinite(value) for value in bias
    ):
        raise ValueError("JEV bias must be finite")
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or value <= 0
        for value in temperatures.values()
    ):
        raise ValueError("JEV temperatures must be positive and finite")
    if not (root / "tokenizer.json").is_file():
        raise ValueError("JEV 9B requires a local tokenizer")
    max_positions = config.get("max_position_embeddings")
    if type(max_positions) is not int or max_positions <= 0:
        raise ValueError("JEV max_position_embeddings must be a positive integer")
    indexed = inspect_index(
        root,
        index_path,
        metadata_paths=(
            config_path,
            index_path,
            root / "tokenizer.json",
            root / "tokenizer_config.json",
        ),
        allowed_prefixes=("language_model.",),
        required_prefixes=(
            "language_model.model.embed_tokens.",
            "language_model.model.layers.",
            "language_model.model.norm.",
            "language_model.lm_head.weight",
        ),
    )
    if any("mtp." in key for key in indexed["weight_map"]):
        raise ValueError("the JEV decision route does not accept MTP tensors")
    return {
        **indexed,
        "path": root,
        "config": config,
        "text_config": config,
        "decision_config": decision,
        "quantization": config.get("quantization", config.get("quantization_config")),
        "variant": "jev-9b",
        "max_context": min(16384, max_positions),
    }


def _criterion(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _options(question):
    kind = question["type"]
    criteria = question.get("criteria")
    if kind == "noul":
        if criteria:
            raise DecisionRequestError("JEV noul questions do not accept criteria")
        return ["false", "true"], ["false", "true"]
    if kind == "score":
        if len(criteria) != 6:
            raise DecisionRequestError("JEV score questions require six levels")
        labels = [str(index) for index in range(6)]
        return labels, labels
    if not 2 <= len(criteria) <= 256:
        raise DecisionRequestError("JEV choice needs 2 to 256 options")
    labels = list(criteria)
    options = [
        label
        if criteria[label] in (None, "")
        else f"{label}: {_criterion(criteria[label])}"
        for label in labels
    ]
    return labels, options


def render_prompt(
    tokenizer, state, name, question, option_codes, *, max_length, truncate
):
    kind = question["type"]
    labels, options = _options(question)
    instruction = _criterion(question.get("instructions") or name)
    lines = (
        [f"{option_codes[index]}) {option}" for index, option in enumerate(options)]
        if kind == "choice"
        else options
    )
    prefix = f"[kind] {kind}\n[state] "
    suffix = (
        f"\n[question] {instruction}\n[options]\n" + "\n".join(lines) + "\n[decision]:"
    )
    state_text = _criterion(state)
    text = prefix + state_text + suffix
    tokens = list(tokenizer.encode(text, add_special_tokens=False))
    if len(tokens) <= max_length:
        return tokens, labels
    if not truncate:
        raise DecisionInputTooLong(
            f"JEV request requires {len(tokens)} tokens; maximum is {max_length}"
        )
    fixed = list(tokenizer.encode(prefix + suffix, add_special_tokens=False))
    state_ids = list(tokenizer.encode(state_text, add_special_tokens=False))
    keep = max(0, max_length - len(fixed) - 8)
    while True:
        shortened = tokenizer.decode(state_ids[:keep], skip_special_tokens=False)
        tokens = list(
            tokenizer.encode(prefix + shortened + suffix, add_special_tokens=False)
        )
        if len(tokens) <= max_length:
            if state_ids and keep == 0:
                raise DecisionInputTooLong(
                    "JEV truncation would remove the entire nonempty state"
                )
            return tokens, labels
        overflow = len(tokens) - max_length
        if keep == 0:
            raise DecisionInputTooLong("JEV fixed prompt exceeds its context")
        keep = max(0, keep - overflow)


class JevEngine(CandidateEngine):
    family = "jev"
    source_revision = SOURCE_REVISION
    inspect_artifact = staticmethod(inspect_artifact)

    def _load(self) -> None:
        def validate_tokenizer(tokenizer):
            labels = []
            names = list(string.ascii_uppercase) + [
                first + second
                for first in string.ascii_uppercase
                for second in string.ascii_uppercase
            ]
            for label in names:
                ids = tokenizer.encode(label, add_special_tokens=False)
                context = tokenizer.encode(
                    f"x\n{label}) y", add_special_tokens=False
                )
                if len(ids) == 1 and ids[0] in context:
                    labels.append((label, ids[0]))
                if len(labels) == 256:
                    break
            if len(labels) != 256 or len({token for _, token in labels}) != 256:
                raise ValueError(
                    "JEV tokenizer does not provide 256 distinct label tokens"
                )
            vocab_size = self.artifact["config"]["vocab_size"]
            if any(token < 0 or token >= vocab_size for _, token in labels):
                raise ValueError("JEV option-label token ids exceed the vocabulary")
            self.option_codes, self.option_token_ids = zip(*labels)

        self.model, self.tokenizer, self.pretokenizer_receipt = load_qwen_backbone(
            self.artifact,
            normalize=lambda weights: weights,
            keep=lambda name: name.startswith("language_model.") and "mtp." not in name,
            need_lm_head=True,
            validate_tokenizer=validate_tokenizer,
        )
        self.settings = self.artifact["decision_config"]
        self.head = self.model.language_model.lm_head

    def _predict_locked(self, request):
        import mlx.core as mx

        prepared = []
        token_count = 0
        for name, question in request.questions.items():
            tokens, labels = render_prompt(
                self.tokenizer,
                request.state,
                name,
                question,
                self.option_codes,
                max_length=self.artifact["max_context"],
                truncate=request.truncate,
            )
            token_count = add_request_tokens(
                token_count, tokens, max_context=self.artifact["max_context"]
            )
            kind = question["type"]
            start, end = self.settings["ranges"][kind]
            if kind == "choice":
                token_ids = self.option_token_ids[: len(labels)]
            else:
                token_ids = self.settings["verbalizer_ids"][start : start + len(labels)]
            bias = [
                self.settings["bias"][start + index] if start + index < end else 0.0
                for index in range(len(labels))
            ]
            prepared.append((name, question, tokens, labels, token_ids, bias, kind))
        rows = []
        self._execution_started = True
        for name, question, tokens, labels, token_ids, bias, kind in prepared:
            hidden = self.model.model(mx.array([tokens]))[0, -1]
            logits = self.model.logits(hidden)[mx.array(token_ids)].astype(mx.float32)
            logits = logits + mx.array(bias)
            temperature = float(self.settings["temperature_by_type"][kind])
            probabilities = mx.softmax(logits / temperature)
            mx.eval(probabilities)
            rows.append((name, question, labels, probabilities.tolist()))
        return {
            "model": self.model_name,
            "answers": format_answers(rows),
            "usage": {"input_tokens": token_count, "output_tokens": 0},
        }
