"""Decision-model runtime registry and the first Clef execution backend."""

from __future__ import annotations

import json
import threading
from itertools import pairwise
from pathlib import Path

from ..batch_metrics import HttpRuntimeMetrics
from .clef import (
    QUESTION_TYPES,
    format_answers,
    inspect_artifact,
    render_decision,
)
from .metrics import DecisionRuntimeMetrics, new_counters, render_decision_metrics
from .schema import DecisionExecutionFailure, DecisionRequestError


def _configure_qwen_decision_environment() -> dict[str, str]:
    # Clef is prefill-only and carries no MTP route.  Reuse the ordinary 9B
    # profile for both supported dense Qwen geometries before importing the
    # runtime model modules.
    from ..adapters.qwen35_9b import configure_environment

    return configure_environment()


class ClefEngine:
    """One resident, serialized, text-only Clef decision model."""

    family = "clef"
    capabilities = ("text", "noul", "choice", "score")

    def __init__(self, model_path: str | Path, *, served_model_name: str | None = None):
        artifact = inspect_artifact(model_path)
        self.artifact = artifact
        self.variant = artifact["variant"]
        self.model_name = served_model_name or artifact["path"].name
        self.environment = _configure_qwen_decision_environment()
        self._lock = threading.Lock()
        self._counts_lock = threading.Lock()
        self._qualification = {
            "qualification": "unqualified",
            "qualified": False,
        }
        self.http_metrics = HttpRuntimeMetrics()
        self.decision_metrics = DecisionRuntimeMetrics()
        self._counts = new_counters()
        loading_started = self.decision_metrics.loading_started()
        self._load(artifact)
        # Strings the bound tokenizer folds into control ids; the request
        # contract refuses them (decisions.tokenizer.reserved_token_strings).
        self.reserved_tokens = frozenset(self.pretokenizer_receipt["reserved_tokens"])
        self.decision_metrics.loaded(loading_started)

    def _load(self, artifact) -> None:
        import mlx.core as mx
        from mlx import nn

        from ..runtime.models.import_env import assert_profile_applied
        from ..runtime.models.qwen38_27b import Model, ModelArgs
        from ..runtime.ubc_evict import load_shards_evicting
        from .clef_head import JointSchemaHead
        from .tokenizer import load_local_tokenizer

        assert_profile_applied("the Clef decision adapter")
        self.tokenizer, self.pretokenizer_receipt = load_local_tokenizer(
            artifact["path"]
        )
        config = dict(artifact["config"])
        config["text_config"] = dict(config["text_config"])
        config["text_config"]["mtp_num_hidden_layers"] = 0
        self.model = Model(ModelArgs.from_dict(config))
        self.head = JointSchemaHead(**config["head_config"])
        files = [
            artifact["path"] / name
            for name in sorted(set(artifact["weight_map"].values()))
        ]

        def text_and_head_only(shard):
            return {
                name: value
                for name, value in shard.items()
                if name.startswith(("language_model.", "head."))
            }

        weights = load_shards_evicting(files, sanitize=text_and_head_only)
        head_weights = {
            name.removeprefix("head."): value
            for name, value in weights.items()
            if name.startswith("head.")
        }
        backbone_weights = self.model.sanitize(
            {
                name: value
                for name, value in weights.items()
                if name.startswith("language_model.")
            }
        )
        quantization = config.get("quantization", config.get("quantization_config"))
        if quantization:
            required = {"group_size", "bits"}
            if not required <= set(quantization):
                raise ValueError("Clef quantization config is incomplete")

            def predicate(name, module):
                override = quantization.get(name)
                if isinstance(override, dict):
                    return override
                return (
                    hasattr(module, "to_quantized")
                    and f"{name}.scales" in backbone_weights
                )

            nn.quantize(
                self.model,
                group_size=int(quantization["group_size"]),
                bits=int(quantization["bits"]),
                mode=quantization.get("mode", "affine"),
                class_predicate=predicate,
            )
        self.model.load_weights(list(backbone_weights.items()), strict=True)
        self.head.load_weights(list(head_weights.items()), strict=True)
        self.model.eval()
        self.head.eval()
        mx.eval(self.model.parameters(), self.head.parameters())
        weights.clear()
        backbone_weights.clear()
        head_weights.clear()
        mx.clear_cache()

    def _lexical_embeddings(self, rendered):
        import mlx.core as mx

        flat = rendered.token_ids
        option_spans = [span for row in rendered.questions for span in row.option_spans]
        lexical_ids = [
            token for start, end in option_spans for token in flat[start:end]
        ]
        boundaries = [0]
        for start, end in option_spans:
            boundaries.append(boundaries[-1] + end - start)
        head = getattr(self.model.language_model, "lm_head", None)
        if head is None:
            raise RuntimeError("Clef lexical scoring requires the trained LM head")
        vocabulary = int(head.weight.shape[0])
        if any(token < 0 or token >= vocabulary for token in lexical_ids):
            raise RuntimeError("Clef lexical token id exceeds the trained LM head")
        indices = mx.array(lexical_ids)
        embeddings = head.weight[indices]
        if hasattr(head, "scales"):
            biases = getattr(head, "biases", None)
            embeddings = mx.dequantize(
                embeddings,
                head.scales[indices],
                None if biases is None else biases[indices],
                group_size=head.group_size,
                bits=head.bits,
                mode=head.mode,
            )
        from .clef_head import _span_means

        spans = list(pairwise(boundaries))
        return _span_means(spans, len(lexical_ids)) @ embeddings

    def _predict_locked(self, request):
        import mlx.core as mx

        rendered = render_decision(
            self.tokenizer,
            request.state,
            request.questions,
            max_length=self.artifact["max_context"],
            truncate=request.truncate,
        )
        input_ids = mx.array([rendered.token_ids])
        self._begin_execution()
        hidden = self.model.model(input_ids)[0]
        hidden = self.head.hidden_norm(hidden)
        lexical = self._lexical_embeddings(rendered).astype(hidden.dtype)
        question_spans = [row.question_span for row in rendered.questions]
        option_spans = [span for row in rendered.questions for span in row.option_spans]
        counts = [len(row.option_spans) for row in rendered.questions]
        kinds = mx.array(
            [QUESTION_TYPES[row.question["type"]] for row in rendered.questions]
        )
        logits = self.head(
            hidden,
            question_spans,
            option_spans,
            lexical,
            kinds,
            counts,
        )
        mx.eval(logits)
        distributions = []
        start = 0
        for count in counts:
            probabilities = mx.softmax(logits[start : start + count].astype(mx.float32))
            mx.eval(probabilities)
            distributions.append(probabilities.tolist())
            start += count
        answers = format_answers(rendered, distributions)
        return {
            "model": self.model_name,
            "answers": answers,
            "usage": {
                "input_tokens": len(rendered.token_ids),
                "output_tokens": 0,
                "truncated": rendered.state_tokens_dropped > 0,
                "state_tokens_dropped": rendered.state_tokens_dropped,
            },
            "mlx2": self.route_receipt(observed_used=True),
        }

    def predict(self, request):
        input_tokens = 0
        with self._lock:
            started_at = self.decision_metrics.execution_started()
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
            else:
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

    def _begin_execution(self) -> None:
        """Mark the model forward as started: the route is observed used now.

        Counted here rather than at completion so a status read during the
        first in-flight request, or after a failure, agrees with the receipt.
        """
        self._execution_started = True
        with self._counts_lock:
            self._counts["executions"] += 1

    def record_refusal(self) -> None:
        with self._counts_lock:
            self._counts["refusals"] += 1

    def route_receipt(self, *, observed_used: bool) -> dict:
        return {
            "route": "decision",
            "family": self.family,
            "variant": self.artifact["variant"],
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
            "source_revision": "01d6ebaeaa4f2dc2394204798a2032e38d8a2841",
            **(
                {"qualification_receipt_sha256": self._qualification["receipt_sha256"]}
                if self._qualification["qualified"]
                else {}
            ),
        }

    def status(self) -> dict:
        with self._counts_lock:
            counters = dict(self._counts)
        return {
            "ready": True,
            "model": self.model_name,
            "decision_family": self.family,
            "qualification": self._qualification["qualification"],
            "route": self.route_receipt(observed_used=counters["executions"] > 0),
            "tokenizer_integrity": self.pretokenizer_receipt,
            "counters": counters,
        }

    def prometheus_metrics(self) -> str:
        return render_decision_metrics(self)

    def close(self) -> None:
        import mlx.core as mx

        self.model = None
        self.head = None
        mx.clear_cache()


def inspect_decision_model(model_path: str | Path) -> dict:
    path = Path(model_path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    model_type = config.get("model_type")
    if model_type == "clef":
        return inspect_artifact(path)
    if model_type == "decision2":
        from .candidates import inspect_decision2

        return inspect_decision2(path)
    if model_type == "jev_text":
        from .candidates import inspect_jev

        return inspect_jev(path)
    if model_type == "qwen3_5" and (path / "decision_config.json").is_file():
        from .candidates import inspect_pplx

        return inspect_pplx(path)
    raise ValueError(
        f"unsupported decision model type {model_type!r}; expected clef, "
        "decision2, jev_text, or a pplx-decider package"
    )


def load_decision_engine(
    model_path: str | Path, *, served_model_name: str | None = None
):
    artifact = inspect_decision_model(model_path)
    model_type = artifact["config"]["model_type"]
    if model_type == "clef":
        return ClefEngine(model_path, served_model_name=served_model_name)
    if model_type == "decision2":
        from .candidates import Decision2Engine

        return Decision2Engine(model_path, served_model_name=served_model_name)
    if model_type == "jev_text":
        from .candidates import JevEngine

        return JevEngine(model_path, served_model_name=served_model_name)
    if model_type == "qwen3_5":
        from .candidates import PplxDeciderEngine

        return PplxDeciderEngine(model_path, served_model_name=served_model_name)
    raise AssertionError("decision registry returned an unhandled model type")
