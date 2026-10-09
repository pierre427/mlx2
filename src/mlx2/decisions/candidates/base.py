"""Shared lifecycle and Qwen trunk loader for candidate-scoring models."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from ...adapters.artifact_paths import (
    hub_blob_identity,
    hub_snapshot_revision,
    shard_within_artifact,
)
from ...batch_metrics import HttpRuntimeMetrics
from ..metrics import DecisionRuntimeMetrics, render_decision_metrics
from ..schema import (
    DecisionExecutionFailure,
    DecisionInputTooLong,
    DecisionRequestError,
)

MAX_REQUEST_CONTEXT_MULTIPLIER = 8


def _digest_part(digest, label: str, payload: bytes) -> None:
    """Hash labelled, length-delimited data without concatenation ambiguity."""
    label_bytes = label.encode()
    digest.update(len(label_bytes).to_bytes(8, "big"))
    digest.update(label_bytes)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def add_request_tokens(total: int, tokens, *, max_context: int) -> int:
    """Accumulate candidate prompts under one bounded multi-question budget."""
    total += len(tokens)
    maximum = MAX_REQUEST_CONTEXT_MULTIPLIER * max_context
    if total > maximum:
        raise DecisionInputTooLong(
            f"candidate request requires {total} aggregate tokens; maximum is {maximum}"
        )
    return total


def read_object(path: Path) -> dict[str, Any]:
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


def inspect_index(
    root: Path,
    index_path: Path,
    *,
    metadata_paths: Iterable[Path],
    allowed_prefixes: tuple[str, ...],
    required_prefixes: tuple[str, ...],
) -> dict[str, Any]:
    """Validate one local indexed artifact without importing MLX."""
    index = read_object(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index_path.name} has no indexed weights")
    unknown = [key for key in weight_map if not key.startswith(allowed_prefixes)]
    if unknown:
        raise ValueError(f"unsupported tensor keys: {unknown[:3]}")
    for prefix in required_prefixes:
        if not any(key.startswith(prefix) for key in weight_map):
            raise ValueError(f"artifact is missing {prefix!r} tensors")

    digest = hashlib.sha256()
    revision = hub_snapshot_revision(root)
    if revision is not None:
        _digest_part(digest, "hub-snapshot-revision", revision.encode())
    for path in metadata_paths:
        if path.is_file():
            relative = str(path.relative_to(root))
            _digest_part(digest, f"metadata:{relative}", path.read_bytes())
    records = []
    shard_paths = []
    content_addressed = revision is not None
    for name in sorted(set(weight_map.values())):
        if not isinstance(name, str):
            raise TypeError("weight-map shard names must be strings")
        relative = index_path.parent.relative_to(root) / name
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("weight shards must stay within the artifact")
        path = root / relative
        resolved = path.resolve()
        if not shard_within_artifact(root, resolved) or not path.is_file():
            raise ValueError(f"missing or escaped weight shard: {relative}")
        stat = path.stat()
        if stat.st_size < 8:
            raise ValueError(f"empty weight shard: {relative}")
        blob = hub_blob_identity(root, resolved)
        content_addressed = content_addressed and blob is not None
        record = (str(relative), stat.st_size, blob)
        records.append(record)
        # Preserve the lexical filename (and therefore its .safetensors
        # suffix) for MLX format dispatch.  Hugging Face cache targets are
        # content-addressed blobs without filename extensions.
        shard_paths.append(path)
        _digest_part(
            digest,
            "weight-shard",
            json.dumps(record, separators=(",", ":")).encode(),
        )
    return {
        "weight_map": weight_map,
        "shard_files": shard_paths,
        "identity": {
            "path": str(root),
            "revision": revision,
            "fingerprint": digest.hexdigest(),
            "fingerprint_kind": (
                "hub-blob-identity" if content_addressed else "layout-metadata"
            ),
            "files": records,
        },
    }


def _configure_environment() -> dict[str, str]:
    from ...adapters.qwen35_9b import configure_environment

    return configure_environment()


def load_qwen_backbone(
    artifact: Mapping[str, Any],
    *,
    normalize: Callable[[dict], dict],
    keep: Callable[[str], bool],
    need_lm_head: bool,
    validate_tokenizer: Callable[[Any], None] | None = None,
):
    """Load an existing mlx2 Qwen trunk under a decision-only owner."""
    import mlx.core as mx
    from mlx import nn

    from ...runtime.models.import_env import assert_profile_applied
    from ...runtime.models.qwen38_27b import Model, ModelArgs
    from ...runtime.ubc_evict import load_shards_evicting
    from ..tokenizer import load_local_tokenizer

    assert_profile_applied("the candidate decision adapter")
    tokenizer, tokenizer_receipt = load_local_tokenizer(artifact["path"])
    if validate_tokenizer is not None:
        validate_tokenizer(tokenizer)
    text_config = dict(artifact["text_config"])
    text_config["mtp_num_hidden_layers"] = 0
    if not need_lm_head:
        text_config["tie_word_embeddings"] = True
    config = {"model_type": "qwen3_5", "text_config": text_config}
    model = Model(ModelArgs.from_dict(config))

    def selected(shard):
        return {name: value for name, value in shard.items() if keep(name)}

    raw = load_shards_evicting(artifact["shard_files"], sanitize=selected)
    weights = model.sanitize(normalize(raw))
    quantization = artifact.get("quantization")
    if quantization:
        required = {"group_size", "bits"}
        if not required <= set(quantization):
            raise ValueError("decision quantization config is incomplete")

        def predicate(name, module):
            override = quantization.get(name)
            if isinstance(override, dict):
                return override
            return hasattr(module, "to_quantized") and f"{name}.scales" in weights

        nn.quantize(
            model,
            group_size=int(quantization["group_size"]),
            bits=int(quantization["bits"]),
            mode=quantization.get("mode", "affine"),
            class_predicate=predicate,
        )
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    raw.clear()
    weights.clear()
    mx.clear_cache()
    return model, tokenizer, tokenizer_receipt


def format_answers(
    rows: Iterable[tuple[str, Mapping[str, Any], list[str], list[float]]],
) -> dict[str, Any]:
    """Map calibrated family probabilities onto mlx2's stable response shape."""
    answers = {}
    for name, question, labels, probabilities in rows:
        if len(labels) != len(probabilities):
            raise RuntimeError("candidate probability width does not match labels")
        values = dict(zip(labels, (float(value) for value in probabilities)))
        if not all(math.isfinite(value) for value in values.values()):
            raise RuntimeError("candidate probabilities must be finite")
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
            "probabilities": {label: round(values[label], 4) for label in labels},
            "confidence": round(values[best], 4),
        }
        if kind == "score":
            answer["legend"] = dict(zip(labels, question["criteria"]))
        answers[name] = answer
    return answers


class CandidateEngine:
    """Common status, locking, and receipt contract for candidate scorers."""

    family = "candidate"
    variant = "unknown"
    source_revision = "unknown"
    capabilities = ("text", "noul", "choice", "score")

    def __init__(self, model_path: str | Path, *, served_model_name: str | None = None):
        self.artifact = self.inspect_artifact(model_path)
        self.model_name = served_model_name or self.artifact["path"].name
        self.variant = self.artifact["variant"]
        self.environment = _configure_environment()
        self._lock = threading.Lock()
        self._counts_lock = threading.Lock()
        self._qualification = {
            "qualification": "unqualified",
            "qualified": False,
        }
        self.http_metrics = HttpRuntimeMetrics()
        self.decision_metrics = DecisionRuntimeMetrics()
        self._counts = {
            "requests": 0,
            "failures": 0,
            "refusals": 0,
            "input_tokens": 0,
        }
        loading_started = self.decision_metrics.loading_started()
        self._load()
        self.decision_metrics.loaded(loading_started)

    @staticmethod
    def inspect_artifact(model_path: str | Path) -> dict[str, Any]:
        raise NotImplementedError

    def _load(self) -> None:
        raise NotImplementedError

    def _predict_locked(self, request):
        raise NotImplementedError

    def predict(self, request):
        input_tokens = 0
        with self._lock:
            started_at = self.decision_metrics.execution_started()
            try:
                self._execution_started = False
                try:
                    response = self._predict_locked(request)
                except DecisionRequestError:
                    with self._counts_lock:
                        self._counts["refusals"] += 1
                    raise
                except Exception as error:
                    with self._counts_lock:
                        self._counts["failures"] += 1
                    raise DecisionExecutionFailure(
                        observed_used=self._execution_started
                    ) from error
                input_tokens = response["usage"]["input_tokens"]
                with self._counts_lock:
                    self._counts["requests"] += 1
                    self._counts["input_tokens"] += input_tokens
                response["mlx2"] = self.route_receipt(observed_used=True)
                return response
            finally:
                self.decision_metrics.execution_finished(
                    started_at, input_tokens=input_tokens
                )

    def record_refusal(self) -> None:
        with self._counts_lock:
            self._counts["refusals"] += 1

    def route_receipt(self, *, observed_used: bool) -> dict[str, Any]:
        return {
            "route": "decision",
            "family": self.family,
            "variant": self.variant,
            "artifact_fingerprint": self.artifact["identity"]["fingerprint"],
            "artifact_revision": self.artifact["identity"].get("revision"),
            "artifact_fingerprint_kind": self.artifact["identity"][
                "fingerprint_kind"
            ],
            "qualification": self._qualification["qualification"],
            "implemented": True,
            "qualified": self._qualification["qualified"],
            "selected": True,
            "observed_used": bool(observed_used),
            "capabilities": list(self.capabilities),
            "unsupported": ["images", "videos", "generation", "APCv2"],
            "source_revision": self.source_revision,
            **(
                {"qualification_receipt_sha256": self._qualification["receipt_sha256"]}
                if self._qualification["qualified"]
                else {}
            ),
        }

    def status(self) -> dict[str, Any]:
        with self._counts_lock:
            counters = dict(self._counts)
        return {
            "ready": True,
            "model": self.model_name,
            "decision_family": self.family,
            "qualification": self._qualification["qualification"],
            "route": self.route_receipt(observed_used=counters["requests"] > 0),
            "tokenizer_integrity": self.pretokenizer_receipt,
            "counters": counters,
        }

    def prometheus_metrics(self) -> str:
        return render_decision_metrics(self)

    def close(self) -> None:
        import mlx.core as mx

        self.model = None
        self.head = None
        if hasattr(self, "readout"):
            self.readout = None
        mx.clear_cache()
