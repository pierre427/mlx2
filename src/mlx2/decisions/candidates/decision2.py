"""Decision 2.0 Lux candidate-head adapter.

The prompt and head math follow llama.cpp PR 30158. Provenance is recorded in
``provenance/candidate-decisions.json``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ...adapters.artifact_paths import shard_within_artifact
from ..schema import DecisionInputTooLong, DecisionRequestError
from .base import (
    CandidateEngine,
    add_request_tokens,
    format_answers,
    inspect_index,
    load_qwen_backbone,
    read_object,
    usage,
)

SOURCE_REVISION = "3c64a581d4f95bb2de409d383741fa5b18d84345"
ARTIFACT_REVISION = "e0cd13890ba2995c8ad1a464ff0cbd817f1aed7a"
PROMPT_VERSION = "decision2-segmented-options-global-query-v1"
HEAD_KEYS = {
    "candidate_norm.weight",
    "candidate_norm.bias",
    "query_norm.weight",
    "query_norm.bias",
    "key.weight",
    "query.weight",
    "candidate_mlp.weight",
    "candidate_mlp.bias",
    "query_mlp.weight",
    "scalar.weight",
}


def _inside(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Decision 2.0 package paths must stay within the artifact")
    # Keep the lexical snapshot path so nested indexes continue to resolve
    # their shard names relative to the package.  Hugging Face snapshots use
    # symlinks into the repository's sibling blobs directory; accept only that
    # established cache layout when the resolved target leaves the snapshot.
    path = root / relative
    if not shard_within_artifact(root, path.resolve()) or not path.is_file():
        raise ValueError(f"missing or escaped Decision 2.0 package file: {relative}")
    return path


def inspect_artifact(model_path: str | Path) -> dict[str, Any]:
    root = Path(model_path).expanduser().resolve()
    config_path = root / "config.json"
    config = read_object(config_path)
    if config.get("model_type") != "decision2":
        raise ValueError("Decision 2.0 requires model_type 'decision2'")
    if config.get("package_schema") != "dev2-package/1":
        raise ValueError("unsupported Decision 2.0 package schema")
    if config.get("calibration") is not None:
        raise ValueError("calibrated Decision 2.0 packages are not implemented")
    if "adapter" in config:
        raise ValueError("Decision 2.0 LoRA packages must be merged before serving")
    backbone = config.get("backbone")
    decision_weights = config.get("decision_weights")
    if not isinstance(backbone, dict) or not isinstance(decision_weights, dict):
        raise TypeError("Decision 2.0 package metadata is incomplete")
    backbone_config_path = _inside(root, backbone.get("config", ""))
    backbone_index_path = _inside(root, backbone.get("index", ""))
    decision_config_path = _inside(root, config.get("model_config", ""))
    head_path = _inside(root, decision_weights.get("decision_head", ""))
    text = read_object(backbone_config_path)
    decision = read_object(decision_config_path)
    if text.get("model_type") != "qwen3_5_text":
        raise ValueError("Lux requires a Qwen3.5 text backbone")
    topology = (text.get("num_hidden_layers"), text.get("hidden_size"))
    if topology != (32, 4096):
        raise ValueError(f"unsupported Decision 2.0 Lux topology {topology!r}")
    if decision.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("unsupported Decision 2.0 prompt version")
    if decision.get("head_variant", "shared") != "shared":
        raise ValueError("only the shared Decision 2.0 candidate head is supported")
    if decision.get("head_dim") != 256 or decision.get("max_options") != 255:
        raise ValueError("Decision 2.0 candidate-head geometry is invalid")
    if not (root / "tokenizer.json").is_file():
        raise ValueError("Decision 2.0 requires a local tokenizer")
    max_positions = text.get("max_position_embeddings")
    max_input_tokens = config.get("max_input_tokens", 16384)
    if type(max_positions) is not int or max_positions <= 0:
        raise ValueError("Lux max_position_embeddings must be a positive integer")
    if type(max_input_tokens) is not int or max_input_tokens <= 0:
        raise ValueError("Lux max_input_tokens must be a positive integer")

    from safetensors import safe_open

    with safe_open(head_path, framework="numpy") as head:
        keys = head.keys()
        if set(keys) != HEAD_KEYS:
            raise ValueError("unexpected Decision 2.0 head tensors")
        shapes = {key: tuple(head.get_slice(key).get_shape()) for key in keys}
        if any(head.get_slice(key).get_dtype() != "F32" for key in keys):
            raise ValueError("Decision 2.0 head tensors must be F32")
        if any(not np.isfinite(head.get_tensor(key)).all() for key in keys):
            raise ValueError("Decision 2.0 head contains non-finite values")
    expected = {
        "candidate_norm.weight": (4096,),
        "candidate_norm.bias": (4096,),
        "query_norm.weight": (4096,),
        "query_norm.bias": (4096,),
        "key.weight": (256, 4096),
        "query.weight": (256, 4096),
        "candidate_mlp.weight": (256, 4096),
        "candidate_mlp.bias": (256,),
        "query_mlp.weight": (256, 4096),
        "scalar.weight": (1, 256),
    }
    if shapes != expected:
        raise ValueError("Decision 2.0 head tensor shapes are invalid")
    score_bias_path = root / "score_bias.json"
    indexed = inspect_index(
        root,
        backbone_index_path,
        metadata_paths=(
            config_path,
            backbone_config_path,
            backbone_index_path,
            decision_config_path,
            head_path,
            root / "tokenizer.json",
            root / "tokenizer_config.json",
            score_bias_path,
        ),
        allowed_prefixes=("embed_tokens.", "layers.", "norm."),
        required_prefixes=("embed_tokens.", "layers.", "norm."),
    )
    score_bias = {}
    if score_bias_path.is_file():
        report = read_object(score_bias_path)
        if report.get("format") != "dev2-score-bias-v1":
            raise ValueError("unknown Decision 2.0 score-bias format")
        for key, offsets in report.get("offsets", {}).items():
            if (
                not key.isdigit()
                or key != str(int(key))
                or not 2 <= int(key) <= 10
            ):
                raise ValueError("invalid Decision 2.0 score-bias level count")
            if not isinstance(offsets, list) or len(offsets) != int(key):
                raise ValueError("incomplete Decision 2.0 score-bias offsets")
            if any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in offsets
            ):
                raise ValueError("non-finite Decision 2.0 score-bias offset")
            score_bias[int(key)] = [float(value) for value in offsets]
    return {
        **indexed,
        "path": root,
        "config": config,
        "text_config": text,
        "decision_config": decision,
        "head_path": head_path,
        "score_bias": score_bias,
        "variant": "decision2-lux-9b",
        "max_context": min(max_input_tokens, max_positions),
    }


def canonical(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _payload(value: Any) -> bool:
    return isinstance(value, (str, list, tuple, Mapping))


def _options(question: Mapping[str, Any]) -> list[tuple[str, Any]]:
    kind = question["type"]
    criteria = question.get("criteria")
    if kind == "noul":
        criteria = dict(criteria or {})
        if any(value is None or not _payload(value) for value in criteria.values()):
            raise DecisionRequestError(
                "Decision 2.0 noul descriptions must be text, objects, or arrays"
            )
        if len(criteria) == 2:
            return [(str(key), value) for key, value in criteria.items()]
        return [
            ("false", criteria.get("false", "No")),
            ("true", criteria.get("true", "Yes")),
        ]
    if kind == "score":
        if not 2 <= len(criteria) <= 10:
            raise DecisionRequestError("Decision 2.0 score needs 2 to 10 levels")
        if any(not _payload(value) for value in criteria):
            raise DecisionRequestError(
                "Decision 2.0 score levels must be text, objects, or arrays"
            )
        return [(str(index), value) for index, value in enumerate(criteria)]
    if not 2 <= len(criteria) <= 255:
        raise DecisionRequestError("Decision 2.0 choice needs 2 to 255 options")
    if any(value is not None and not _payload(value) for value in criteria.values()):
        raise DecisionRequestError(
            "Decision 2.0 choice descriptions must be null, text, objects, or arrays"
        )
    return [(str(key), value) for key, value in criteria.items()]


def render_prompt(tokenizer, state, name, question, *, max_length, truncate):
    encode = lambda text: list(tokenizer.encode(text, add_special_tokens=False))
    if not _payload(state):
        raise DecisionRequestError(
            "Decision 2.0 state must be text, an object, or an array"
        )
    instruction = question.get("instructions")
    if instruction in (None, ""):
        raise DecisionRequestError("Decision 2.0 instructions must be provided")
    if not _payload(instruction) or (isinstance(instruction, str) and not instruction):
        raise DecisionRequestError(
            "Decision 2.0 instructions must be nonempty text, an object, or an array"
        )
    state_text = canonical(state)
    head_suffix = (
        f"\n\nTask type: {question['type']}\nQuestion:\n{canonical(instruction)}"
        "\nOptions:"
    )
    head = encode("Context:\n" + state_text + head_suffix)
    option_parts = []
    labels = []
    for label, description in _options(question):
        value = {"description": description, "key": label}
        part = encode("\n<option>\n" + canonical(value) + "\n</option>")
        if not part:
            raise DecisionRequestError("Decision 2.0 produced an empty option segment")
        option_parts.append(part)
        labels.append(label)
    query = encode(
        "\n\nSelect the single option best supported by the context and "
        "instructions.\nDecision:"
    )
    if not query:
        raise DecisionRequestError("Decision 2.0 produced an empty query segment")
    fixed = (
        len(encode("Context:\n" + head_suffix))
        + sum(len(part) for part in option_parts)
        + len(query)
    )
    if fixed > max_length:
        raise DecisionInputTooLong(
            f"Decision 2.0 question requires {fixed} tokens before state; maximum is {max_length}"
        )
    total = len(head) + sum(len(part) for part in option_parts) + len(query)
    if total > max_length and not truncate:
        raise DecisionInputTooLong(
            f"Decision 2.0 request exceeds the {max_length}-token context"
        )
    dropped = 0
    if total > max_length:
        state_ids = encode(state_text)
        keep = max(0, max_length - fixed)
        while True:
            shortened = tokenizer.decode(state_ids[:keep], skip_special_tokens=False)
            head = encode("Context:\n" + shortened + head_suffix)
            total = len(head) + sum(len(part) for part in option_parts) + len(query)
            if total <= max_length:
                if state_ids and keep == 0:
                    raise DecisionInputTooLong(
                        "Decision 2.0 truncation would remove the entire nonempty state"
                    )
                dropped = max(0, len(state_ids) - keep)
                break
            overflow = total - max_length
            if keep == 0:
                raise DecisionInputTooLong(
                    "Decision 2.0 fixed prompt exceeds its context"
                )
            keep = max(0, keep - overflow)
    tokens = list(head)
    positions = []
    for part in option_parts:
        tokens.extend(part)
        positions.append(len(tokens) - 1)
    tokens.extend(query)
    return tokens, positions, len(tokens) - 1, labels, dropped


class Decision2Head:
    def __new__(cls, hidden_size=4096, width=256):
        from mlx import nn

        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.candidate_norm = nn.LayerNorm(hidden_size)
                self.query_norm = nn.LayerNorm(hidden_size)
                self.key = nn.Linear(hidden_size, width, bias=False)
                self.query = nn.Linear(hidden_size, width, bias=False)
                self.candidate_mlp = nn.Linear(hidden_size, width)
                self.query_mlp = nn.Linear(hidden_size, width, bias=False)
                self.scalar = nn.Linear(width, 1, bias=False)

            def __call__(self, candidates, query):
                import mlx.core as mx

                candidates = self.candidate_norm(candidates.astype(mx.float32))
                query = self.query_norm(query.astype(mx.float32))
                bilinear = mx.sum(self.key(candidates) * self.query(query), axis=-1)
                bilinear = bilinear / math.sqrt(width)
                nonlinear = nn.gelu(
                    self.candidate_mlp(candidates) + self.query_mlp(query)
                )
                return bilinear + self.scalar(nonlinear).squeeze(-1)

        return Head()


class Decision2Engine(CandidateEngine):
    family = "decision2"
    source_revision = SOURCE_REVISION
    inspect_artifact = staticmethod(inspect_artifact)

    def _load(self) -> None:
        import mlx.core as mx

        self.model, self.tokenizer, self.pretokenizer_receipt = load_qwen_backbone(
            self.artifact,
            normalize=lambda weights: {
                f"model.{key}": value for key, value in weights.items()
            },
            keep=lambda _name: True,
            need_lm_head=False,
        )
        self.head = Decision2Head()
        self.head.load_weights(
            list(mx.load(str(self.artifact["head_path"])).items()), strict=True
        )
        self.head.eval()
        mx.eval(self.head.parameters())

    def _predict_locked(self, request):
        import mlx.core as mx

        prepared = []
        token_count = 0
        dropped = []
        for name, question in request.questions.items():
            tokens, candidates, query, labels, cut = render_prompt(
                self.tokenizer,
                request.state,
                name,
                question,
                max_length=self.artifact["max_context"],
                truncate=request.truncate,
            )
            token_count = add_request_tokens(
                token_count, tokens, max_context=self.artifact["max_context"]
            )
            dropped.append(cut)
            prepared.append((name, question, tokens, candidates, query, labels))
        rows = []
        self._begin_execution()
        for name, question, tokens, candidates, query, labels in prepared:
            hidden = self.model.model(mx.array([tokens]))[0]
            scores = self.head(hidden[mx.array(candidates)], hidden[query])
            if question["type"] == "score":
                offsets = self.artifact["score_bias"].get(len(labels))
                if offsets is not None:
                    scores = scores + mx.array(offsets)
            probabilities = mx.softmax(scores.astype(mx.float32))
            mx.eval(probabilities)
            rows.append((name, question, labels, probabilities.tolist()))
        return {
            "model": self.model_name,
            "answers": format_answers(rows),
            "usage": usage(token_count, dropped),
        }
