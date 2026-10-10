"""CPU-safe Clef artifact, prompt, and answer contracts.

The joint-head algorithm is adapted from mlx-vlm PR 2459.  See
``provenance/clef-decision.json`` and ``provenance/clef-decision.NOTICE``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..adapters.artifact_paths import (
    hub_blob_identity,
    hub_snapshot_revision,
    shard_within_artifact,
)
from .schema import DecisionInputTooLong

SOURCE_REVISION = "01d6ebaeaa4f2dc2394204798a2032e38d8a2841"
SOURCE_PATH = "mlx_vlm/models/clef/clef.py"
ARTIFACT_REVISION = "8f9beb2e63474d547c70810b41b468d3b4a84932"
MAX_CONTEXT = 16_384
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}
SUPPORTED_TOPOLOGIES = {
    (32, 4096): "clef-flash-9b",
    (64, 5120): "clef-27b",
}
REQUIRED_HEAD_KEYS = frozenset(
    {
        "head.hidden_norm.weight",
        "head.hidden_norm.bias",
        "head.memory_projection.weight",
        "head.question_projection.weight",
        "head.option_question_projection.weight",
        "head.global_projection.weight",
        "head.option_context_projection.weight",
        "head.option_lexical_projection.weight",
        "head.type_embedding.weight",
        "head.option_summary_norm.weight",
        "head.field_norm.weight",
        "head.option_norm.weight",
        "head.prior_logit_scale",
        "head.joint_logit_scale",
        "head.residual_gate",
    }
)


def _digest_part(digest, label: str, payload: bytes) -> None:
    """Hash labelled, length-delimited data without concatenation ambiguity."""
    label_bytes = label.encode()
    digest.update(len(label_bytes).to_bytes(8, "big"))
    digest.update(label_bytes)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _object(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r} in {path.name}")
            result[key] = value
        return result

    value = json.loads(path.read_text(), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def inspect_artifact(model_path: str | Path) -> dict:
    """Inspect an indexed, prepared Clef artifact without importing MLX."""
    path = Path(model_path).expanduser().resolve()
    config = _object(path / "config.json")
    if config.get("model_type") != "clef":
        raise ValueError("Clef decision serving requires model_type 'clef'")
    text = config.get("text_config")
    head = config.get("head_config")
    if not isinstance(text, dict) or not isinstance(head, dict):
        raise TypeError("Clef artifact requires text_config and head_config objects")
    topology = (text.get("num_hidden_layers"), text.get("hidden_size"))
    variant = SUPPORTED_TOPOLOGIES.get(topology)
    if variant is None:
        raise ValueError(f"unsupported Clef Qwen topology {topology!r}")
    expected_head = {
        "hidden_size": topology[1],
        "width": 1024,
        "routing_layers": 2,
        "layers": 4,
        "heads": 16,
        "feedforward": 4096,
    }
    if head != expected_head:
        raise ValueError("Clef joint-head geometry does not match the published layout")
    if text.get("mtp_num_hidden_layers", 0) != 0:
        raise ValueError("Clef decision artifacts must not contain an MTP route")
    index = _object(path / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Clef artifact has no indexed weights")
    if not REQUIRED_HEAD_KEYS <= set(weight_map):
        missing = sorted(REQUIRED_HEAD_KEYS - set(weight_map))
        raise ValueError(f"Clef joint head is incomplete: {missing}")
    if not any(key.startswith("language_model.model.layers.") for key in weight_map):
        raise ValueError("Clef artifact has no Qwen language-model layers")
    if any(".mtp." in key for key in weight_map):
        raise ValueError("Clef decision artifacts must not contain MTP tensors")
    if "language_model.lm_head.weight" not in weight_map:
        raise ValueError("Clef lexical scoring requires language_model.lm_head.weight")
    unknown = [
        key
        for key in weight_map
        if not key.startswith(("head.", "language_model.", "vision_tower."))
    ]
    if unknown:
        raise ValueError(
            f"Clef artifact contains unsupported tensor keys: {unknown[:3]}"
        )
    shard_names = sorted(set(weight_map.values()))
    digest = hashlib.sha256()
    revision = hub_snapshot_revision(path)
    if revision is not None:
        _digest_part(digest, "hub-snapshot-revision", revision.encode())
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        item = path / name
        if item.is_file():
            _digest_part(digest, f"metadata:{name}", item.read_bytes())
    records = []
    content_addressed = revision is not None
    for name in shard_names:
        if (
            not isinstance(name, str)
            or Path(name).is_absolute()
            or ".." in Path(name).parts
        ):
            raise ValueError("Clef shard paths must stay within the artifact")
        item = (path / name).resolve()
        if not shard_within_artifact(path, item) or not item.is_file():
            raise ValueError(f"missing or escaped Clef weight shard: {name}")
        stat = item.stat()
        if stat.st_size < 8:
            raise ValueError(f"empty Clef weight shard: {name}")
        blob = hub_blob_identity(path, item)
        content_addressed = content_addressed and blob is not None
        record = (name, stat.st_size, blob)
        records.append(record)
        _digest_part(
            digest,
            "weight-shard",
            json.dumps(record, separators=(",", ":")).encode(),
        )
    tokenizer = path / "tokenizer.json"
    if not tokenizer.is_file():
        raise ValueError("Clef artifact requires tokenizer.json")
    max_positions = text.get("max_position_embeddings")
    if type(max_positions) is not int or max_positions <= 0:
        raise ValueError("Clef max_position_embeddings must be a positive integer")
    return {
        "path": path,
        "config": config,
        "weight_map": weight_map,
        "variant": variant,
        "max_context": min(MAX_CONTEXT, max_positions),
        "identity": {
            "path": str(path),
            "revision": revision,
            "fingerprint": digest.hexdigest(),
            "fingerprint_kind": (
                "hub-blob-identity" if content_addressed else "layout-metadata"
            ),
            "files": records,
        },
    }


def render_criterion(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def options(question: Mapping[str, Any]) -> list[tuple[str, Any]]:
    kind = question["type"]
    criteria = question.get("criteria")
    if kind == "noul":
        defaults = {
            "true": "The proposition is true or the answer is yes.",
            "false": "The proposition is false or the answer is no.",
        }
        defaults.update(criteria or {})
        return [(key, defaults[key]) for key in ("true", "false")]
    if kind == "choice":
        return sorted((str(key), value) for key, value in criteria.items())
    if kind == "score":
        return [(str(index), value) for index, value in enumerate(criteria)]
    raise ValueError(f"unsupported Clef question type {kind!r}")


@dataclass(frozen=True, slots=True)
class RenderedQuestion:
    name: str
    question: Mapping[str, Any]
    question_span: tuple[int, int]
    option_spans: tuple[tuple[int, int], ...]
    labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RenderedDecision:
    token_ids: tuple[int, ...]
    questions: tuple[RenderedQuestion, ...]
    # State tokens cut to fit the context; zero when the state was rendered whole.
    state_tokens_dropped: int = 0


def _encode(tokenizer, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    return [int(token) for token in encoded]


def render_decision(
    tokenizer,
    state: Any,
    questions: Mapping[str, Mapping[str, Any]],
    *,
    max_length: int = MAX_CONTEXT,
    truncate: bool = True,
) -> RenderedDecision:
    """Render the exact published Clef text prompt and its head spans."""
    tokens = lambda value: _encode(tokenizer, value)
    schema = tokens("\n\nSCHEMA FIELDS:\n")
    relative = []
    for index, (name, question) in enumerate(questions.items()):
        schema += tokens(
            f"\nFIELD {index + 1}\nID: {name}\nTYPE: {question['type']}\nINSTRUCTION: "
        )
        start = len(schema)
        schema += tokens(render_criterion(question.get("instructions") or str(name)))
        question_span = (start, len(schema))
        schema += tokens("\nALLOWED OPTIONS:\n")
        spans, labels = [], []
        for number, (label, description) in enumerate(options(question), 1):
            schema += tokens(f"OPTION {number}: ")
            start = len(schema)
            semantics = {"option_id": label}
            if description is not None:
                semantics["description"] = description
            schema += tokens(render_criterion(semantics))
            spans.append((start, len(schema)))
            labels.append(label)
            schema += tokens("\n")
        schema += tokens("END FIELD\n")
        relative.append((name, question, question_span, spans, labels))
    prefix = tokens(
        "<|im_start|>system\nRead the complete state and schema. Decide every "
        "field jointly. Each answer must be exactly one of that field's allowed "
        "options.<|im_end|>\n<|im_start|>user\nSTATE:\n"
    )
    suffix = tokens(
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        "JOINT SCHEMA DECISIONS:"
    )
    state_ids = tokens(render_criterion(state))
    fixed = len(prefix) + len(schema) + len(suffix)
    if fixed > max_length:
        raise DecisionInputTooLong(
            f"schema requires {fixed} tokens before state; maximum is {max_length}"
        )
    available = max_length - fixed
    if len(state_ids) > available and not truncate:
        raise DecisionInputTooLong(
            f"request requires {fixed + len(state_ids)} tokens; maximum is {max_length}"
        )
    if state_ids and available <= 0:
        raise DecisionInputTooLong(
            "Clef truncation would remove the entire nonempty state"
        )
    dropped = max(0, len(state_ids) - available)
    state_ids = state_ids[:available]
    offset = len(prefix) + len(state_ids)
    rows = tuple(
        RenderedQuestion(
            name=name,
            question=question,
            question_span=(span[0] + offset, span[1] + offset),
            option_spans=tuple((start + offset, end + offset) for start, end in spans),
            labels=tuple(labels),
        )
        for name, question, span, spans, labels in relative
    )
    return RenderedDecision(tuple(prefix + state_ids + schema + suffix), rows, dropped)


def format_answers(rendered: RenderedDecision, distributions) -> dict:
    """Map per-option probabilities back to the transport-neutral response."""
    if len(distributions) != len(rendered.questions):
        raise RuntimeError("Clef distribution count does not match question count")
    answers = {}
    for row, values in zip(rendered.questions, distributions):
        if len(values) != len(row.labels):
            raise RuntimeError("Clef distribution width does not match option count")
        probabilities = dict(zip(row.labels, (float(value) for value in values)))
        if not all(math.isfinite(value) for value in probabilities.values()):
            raise RuntimeError("Clef probabilities must be finite")
        kind = row.question["type"]
        if kind == "noul":
            probability = round(probabilities["true"], 4)
            answers[row.name] = {
                "type": "noul",
                "value": probability >= 0.5,
                "probability": probability,
                "confidence": round(max(probabilities.values()), 4),
            }
            continue
        labels = (
            list(row.question["criteria"])
            if kind == "choice"
            else [str(index) for index in range(len(row.question["criteria"]))]
        )
        best = max(labels, key=probabilities.__getitem__)
        answer = {
            "type": kind,
            "value": (
                best
                if kind == "choice"
                else round(
                    sum(
                        index * probabilities[label]
                        for index, label in enumerate(labels)
                    ),
                    4,
                )
            ),
            "probabilities": {
                label: round(probabilities[label], 4) for label in labels
            },
            "confidence": round(probabilities[best], 4),
        }
        if kind == "score":
            answer["legend"] = dict(zip(labels, row.question["criteria"]))
            # Clef renders every level description as an option (see
            # render_decision), so the legend is what the model scored.
            answer["legend_source"] = "prompt"
        answers[row.name] = answer
    return answers
