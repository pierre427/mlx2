"""Exact relevance-trained MS MARCO cross-encoder; tensor-free inspection."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

from ..contracts import Capability, ModelDescriptor
from .minilm import _read_pinned, expected_weight_schema, validate_weight_header

MODEL_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
MAX_SEQUENCE_LENGTH = 512
MAX_INPUT_BYTES = 1 << 20
ARTIFACT_FILES = {
    "config.json": (794, "380e02c93f431831be65d99a4e7e5f67c133985bf2e77d9d4eba46847190bacc"),
    "tokenizer.json": (711396, "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"),
    "tokenizer_config.json": (1330, "a5c2e5a7b1a29a0702cd28c08a399b5ecc110c263009d17f7e3b415f25905fd8"),
    "special_tokens_map.json": (132, "3c3507f36dff57bce437223db3b3081d1e2b52ec3e56ee55438193ecb2c94dd6"),
    "vocab.txt": (231508, "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"),
    "model.safetensors": (90870598, "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"),
}


def representation_contract():
    return {
        "kind": "task_trained_relevance_cross_encoder",
        "model": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "revision": MODEL_REVISION,
        "training_task": "ms_marco_passage_ranking",
        "backbone": "bidirectional_bert",
        "tokenizer": "uncased_wordpiece_query_document_pair",
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "truncation": "longest_first_right_including_cls_sep",
        "pooling": "trained_cls_dense_tanh",
        "readout": "trained_scalar_linear",
        "score_semantics": "raw_relevance_logit_higher_is_better",
        "score_activation": "identity",
        "probability_calibrated": False,
        "execution_width": 1,
        "implemented": True,
        "qualified": False,
        "retrieval_quality_evaluated": True,
        "retrieval_quality_scope": "bounded_scifact_24_queries_360_candidates_cpu",
        "retrieval_quality_qualified": False,
    }


DESCRIPTOR = ModelDescriptor(
    model_type="bert", family="minilm_reranker", variant="ms-marco-minilm-l6-v2-f32",
    state_planes=frozenset(), capabilities=frozenset({Capability.RERANK}),
    metadata={"execution_kind": "rerank", "qualification": "unqualified",
              "representation": representation_contract()},
)


def weight_schema():
    return {
        **{"bert." + name: value for name, value in expected_weight_schema().items()},
        "classifier.weight": ("F32", [1, 384]),
        "classifier.bias": ("F32", [1]),
    }


def inspect_artifact(model_path):
    path = Path(model_path).expanduser().resolve()
    metadata = {name: _read_pinned(path, name, manifest=ARTIFACT_FILES)
                for name in ARTIFACT_FILES if name != "model.safetensors"}
    weights = path / "model.safetensors"
    size = weights.stat().st_size
    if size != ARTIFACT_FILES["model.safetensors"][0]:
        raise ValueError("MiniLM reranker checkpoint byte size mismatch")
    with weights.open("rb") as stream:
        validate_weight_header(stream, size, schema=weight_schema())
    return {
        "path": str(path), "config": json.loads(metadata["config.json"]),
        "identity": {
            "fingerprint": hashlib.sha256(json.dumps(ARTIFACT_FILES, sort_keys=True).encode()).hexdigest(),
            "revision": MODEL_REVISION,
        },
        "weight_payload_verified": False, "representation": representation_contract(),
    }


class MiniLMRerankerAdapter:
    descriptor = DESCRIPTOR
    default_route = "rerank"

    def __init__(self, model_path, *, execution_policy=None):
        if execution_policy:
            raise ValueError("MiniLM reranking does not accept generation execution policies")
        artifact = inspect_artifact(model_path)
        path = Path(artifact["path"])
        data = _read_pinned(path, "model.safetensors", manifest=ARTIFACT_FILES)
        validate_weight_header(io.BytesIO(data), len(data), schema=weight_schema())
        import mlx.core as mx
        from tokenizers import Tokenizer

        from ..runtime.models.bert import BertArgs, BertSequenceClassifier

        with io.BytesIO(data) as stream:
            weights = mx.load(stream, format="safetensors")
            mx.eval(weights)
        if not all(bool(mx.all(mx.isfinite(value)).item()) for value in weights.values()):
            raise ValueError("MiniLM reranker weights contain nonfinite values")
        self.model = BertSequenceClassifier(BertArgs())
        self.model.load_weights(self.model.sanitize(weights), strict=True)
        self.model.eval()
        mx.eval(self.model.parameters())
        self.tokenizer = Tokenizer.from_str(
            _read_pinned(path, "tokenizer.json", manifest=ARTIFACT_FILES).decode()
        )
        self.tokenizer.enable_truncation(MAX_SEQUENCE_LENGTH, strategy="longest_first", direction="right")
        self.tokenizer.no_padding()
        self.identity = {**artifact["identity"], "weight_payload_verified": True}
        self.config, self.max_context, self.environment = artifact["config"], MAX_SEQUENCE_LENGTH, {}
        self._executions = 0

    def representation_execution_count(self):
        return self._executions

    def representation_contract(self):
        return {**representation_contract(), "selected": True, "observed_used": self._executions > 0}

    def rerank(self, query, documents):
        if not isinstance(query, str) or not query:
            raise ValueError("rerank query must be nonempty text")
        if not isinstance(documents, (list, tuple)) or not 1 <= len(documents) <= 1000:
            raise ValueError("rerank requires 1 to 1000 documents")
        if any(not isinstance(text, str) or not text for text in documents):
            raise ValueError("rerank documents must be nonempty text")
        texts = (query, *documents)
        if any(len(text) > MAX_INPUT_BYTES for text in texts) or sum(
            len(text.encode("utf-8")) for text in texts
        ) > MAX_INPUT_BYTES:
            raise ValueError("rerank inputs exceed the 1 MiB text byte budget")
        if self.model is None:
            raise RuntimeError("MiniLM reranker is closed")
        import mlx.core as mx

        scores, tokens = [], 0
        for document in documents:
            pair = self.tokenizer.encode(query, document, add_special_tokens=True)
            ids = mx.array([pair.ids], dtype=mx.int32)
            mask = mx.array([pair.attention_mask], dtype=mx.bool_)
            types = mx.array([pair.type_ids], dtype=mx.int32)
            self._executions += 1
            score = self.model(ids, mask, types)
            mx.eval(score)
            if not bool(mx.all(mx.isfinite(score)).item()):
                raise ValueError("reranker produced a nonfinite relevance score")
            scores.append(float(score[0].item()))
            tokens += sum(pair.attention_mask)
        return scores, tokens

    def diagnostics(self):
        return {
            "architecture": "bert", "route": "rerank", "qualification": "unqualified",
            "representation": representation_contract(), "generation": "unsupported",
            "embeddings": "unsupported", "apcv2": "not_applicable",
            "kv_cache": "not_applicable", "streaming": "not_applicable",
        }

    def close(self):
        self.model = None
        self.tokenizer = None
