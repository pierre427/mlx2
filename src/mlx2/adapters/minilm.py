"""Pinned, task-trained MiniLM sentence embeddings; inspection is tensor-free."""

from __future__ import annotations

import hashlib
import io
import json
import math
import struct
from pathlib import Path

from ..contracts import Capability, ModelDescriptor

MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MAX_SEQUENCE_LENGTH = 256
DIMENSIONS = 384
MAX_INPUT_BYTES = 1 << 20
MAX_INPUTS = 2048
# Exact released recipe and checkpoint: a similarly shaped generic BERT is
# not evidence of sentence-embedding training. Copies of these bytes work at
# any local path; hub access and remote code are never needed.
ARTIFACT_FILES = {
    "config.json": (612, "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41"),
    "modules.json": (349, "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf"),
    "sentence_bert_config.json": (53, "fc1993fde0a95c24ec6c022539d41cf6e2f7c9721e5415d6fb6897472a9cd4b7"),
    "1_Pooling/config.json": (190, "4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23"),
    "tokenizer.json": (466247, "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"),
    "tokenizer_config.json": (350, "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b"),
    "special_tokens_map.json": (112, "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3"),
    "vocab.txt": (231508, "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"),
    "model.safetensors": (90868376, "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"),
}


def representation_contract():
    return {
        "kind": "task_trained_sentence_embedding",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": MODEL_REVISION,
        "training_objective": "contrastive_sentence_pairs",
        "backbone": "bidirectional_bert",
        "tokenizer": "uncased_wordpiece",
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "truncation": "right_including_cls_sep",
        "pooling": "attention_mask_mean_including_special_tokens",
        "normalization": "l2",
        "dimensions": DIMENSIONS,
        "supported_dimensions": [DIMENSIONS],
        "query_document_prefix": None,
        "execution_width": 1,
        "implemented": True,
        "qualified": False,
        "retrieval_quality_evaluated": True,
        "retrieval_quality_scope": "bounded_scifact_24_queries_360_candidates_cpu",
        "retrieval_quality_qualified": False,
    }


DESCRIPTOR = ModelDescriptor(
    model_type="bert",
    family="minilm",
    variant="all-minilm-l6-v2-f32",
    state_planes=frozenset(),
    capabilities=frozenset({Capability.EMBEDDINGS}),
    metadata={
        "execution_kind": "embeddings",
        "qualification": "unqualified",
        "representation": representation_contract(),
    },
)


def _read_pinned(path, name, *, manifest=ARTIFACT_FILES):
    size, expected = manifest[name]
    with (path / name).open("rb") as stream:
        data = stream.read(size + 1)
    if len(data) != size or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"MiniLM pinned artifact content mismatch: {name}")
    return data


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate MiniLM safetensors key: {key}")
        result[key] = value
    return result


def expected_weight_schema():
    schema = {
        "embeddings.position_ids": ("I64", [1, 512]),
        "embeddings.word_embeddings.weight": ("F32", [30522, 384]),
        "embeddings.position_embeddings.weight": ("F32", [512, 384]),
        "embeddings.token_type_embeddings.weight": ("F32", [2, 384]),
        "embeddings.LayerNorm.weight": ("F32", [384]),
        "embeddings.LayerNorm.bias": ("F32", [384]),
        "pooler.dense.weight": ("F32", [384, 384]),
        "pooler.dense.bias": ("F32", [384]),
    }
    for layer in range(6):
        prefix = f"encoder.layer.{layer}."
        linear = {
            **{f"attention.self.{name}": (384, 384) for name in ("query", "key", "value")},
            "attention.output.dense": (384, 384),
            "intermediate.dense": (1536, 384),
            "output.dense": (384, 1536),
        }
        for name, (output_size, input_size) in linear.items():
            schema[prefix + name + ".weight"] = ("F32", [output_size, input_size])
            schema[prefix + name + ".bias"] = ("F32", [output_size])
        for name in ("attention.output.LayerNorm", "output.LayerNorm"):
            for suffix in ("weight", "bias"):
                schema[prefix + name + "." + suffix] = ("F32", [384])
    return schema


def validate_weight_header(stream, total_bytes, *, schema=None):
    """Exact names, dimensions, dtype and contiguous nonoverlapping payload."""
    prefix = stream.read(8)
    if len(prefix) != 8:
        raise ValueError("truncated MiniLM safetensors prefix")
    length = struct.unpack("<Q", prefix)[0]
    if not 2 <= length <= 1 << 20 or length + 8 > total_bytes:
        raise ValueError("invalid MiniLM safetensors header size")
    header = json.loads(stream.read(length), object_pairs_hook=_unique_object)
    if not isinstance(header, dict):
        raise ValueError("MiniLM safetensors header must be an object")  # noqa: TRY004
    header.pop("__metadata__", None)
    schema = expected_weight_schema() if schema is None else schema
    if header.keys() != schema.keys():
        raise ValueError("MiniLM weight keys differ from the exact encoder/pooler schema")
    intervals = []
    for name, (dtype, shape) in schema.items():
        tensor = header[name]
        if (
            not isinstance(tensor, dict) or tensor.get("dtype") != dtype
            or tensor.get("shape") != shape
            or any(type(value) is not int for value in tensor["shape"])
        ):
            raise ValueError(f"MiniLM weight shape/dtype mismatch: {name}")
        offsets = tensor.get("data_offsets")
        if (
            not isinstance(offsets, list) or len(offsets) != 2
            or any(type(value) is not int for value in offsets)
            or offsets[0] < 0
            or offsets[1] - offsets[0] != math.prod(shape) * (8 if dtype == "I64" else 4)
        ):
            raise ValueError(f"MiniLM weight offsets mismatch: {name}")
        intervals.append(tuple(offsets))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError("MiniLM tensor payload is overlapping or has holes")
        cursor = end
    if cursor + length + 8 != total_bytes:
        raise ValueError("MiniLM tensor payload length mismatch")


def inspect_artifact(model_path):
    path = Path(model_path).expanduser().resolve()
    metadata = {
        name: _read_pinned(path, name)
        for name in ARTIFACT_FILES if name != "model.safetensors"
    }
    weight_path = path / "model.safetensors"
    size = weight_path.stat().st_size
    if size != ARTIFACT_FILES["model.safetensors"][0]:
        raise ValueError("MiniLM checkpoint byte size mismatch")
    with weight_path.open("rb") as stream:
        validate_weight_header(stream, size)
    fingerprint = hashlib.sha256(
        json.dumps(ARTIFACT_FILES, sort_keys=True).encode()
    ).hexdigest()
    return {
        "path": str(path),
        "config": json.loads(metadata["config.json"]),
        "identity": {"fingerprint": fingerprint, "revision": MODEL_REVISION},
        "weight_payload_verified": False,
        "representation": representation_contract(),
    }


class MiniLMAdapter:
    descriptor = DESCRIPTOR
    default_route = "embeddings"

    def __init__(self, model_path, *, execution_policy=None):
        if execution_policy:
            raise ValueError("MiniLM embeddings do not accept generation execution policies")
        artifact = inspect_artifact(model_path)
        path = Path(artifact["path"])
        # Load precisely the verified immutable bytes, not a second read of a
        # mutable checkpoint path after verification.
        data = _read_pinned(path, "model.safetensors")
        validate_weight_header(io.BytesIO(data), len(data))
        import mlx.core as mx
        from tokenizers import Tokenizer

        from ..runtime.models.bert import BertArgs, BertModel, encoder_weights

        with io.BytesIO(data) as stream:
            weights = mx.load(stream, format="safetensors")
            mx.eval(weights)
        if not all(bool(mx.all(mx.isfinite(value)).item()) for value in weights.values()):
            raise ValueError("MiniLM weights contain nonfinite values")
        self.model = BertModel(BertArgs())
        self.model.load_weights(encoder_weights(weights), strict=True)
        self.model.eval()
        mx.eval(self.model.parameters())
        self.tokenizer = Tokenizer.from_str(_read_pinned(path, "tokenizer.json").decode())
        self.tokenizer.enable_truncation(MAX_SEQUENCE_LENGTH, direction="right")
        self.tokenizer.no_padding()
        self.identity = {**artifact["identity"], "weight_payload_verified": True}
        self.config = artifact["config"]
        self.max_context = MAX_SEQUENCE_LENGTH
        self.environment = {}
        self._embedding_executions = 0

    def representation_execution_count(self):
        """Host-only forward-start count; never materialize tensors for status."""
        return self._embedding_executions

    def embedding_execution_count(self):
        """Compatibility alias for callers predating generalized representations."""
        return self.representation_execution_count()

    def representation_contract(self):
        return {
            **representation_contract(),
            "selected": True,
            "observed_used": self._embedding_executions > 0,
        }

    def embed(self, texts, *, dimensions=None):
        if dimensions is not None and (type(dimensions) is not int or dimensions != DIMENSIONS):
            raise ValueError("MiniLM supports exactly 384 dimensions; truncation is not trained")
        if not isinstance(texts, (list, tuple)) or not 1 <= len(texts) <= MAX_INPUTS:
            raise ValueError("MiniLM input must contain 1 to 2048 texts")
        if any(not isinstance(text, str) or not text for text in texts):
            raise ValueError("MiniLM inputs must be nonempty text")
        if any(len(text) > MAX_INPUT_BYTES for text in texts) or sum(
            len(text.encode("utf-8")) for text in texts
        ) > MAX_INPUT_BYTES:
            raise ValueError("MiniLM inputs exceed the 1 MiB text byte budget")
        if self.model is None:
            raise RuntimeError("MiniLM adapter is closed")
        import mlx.core as mx

        from ..runtime.models.bert import normalized_masked_mean

        vectors, token_count = [], 0
        # Independent width-one calls bound activation memory and preserve
        # the exact readout semantics for requests with mixed text lengths.
        for text in texts:
            encoded = self.tokenizer.encode(text, add_special_tokens=True)
            ids = mx.array([encoded.ids], dtype=mx.int32)
            mask = mx.array([encoded.attention_mask], dtype=mx.bool_)
            types = mx.array([encoded.type_ids], dtype=mx.int32)
            self._embedding_executions += 1
            vector = normalized_masked_mean(self.model(ids, mask, types), mask)
            mx.eval(vector)
            vectors.append(vector[0].tolist())
            token_count += sum(encoded.attention_mask)
        return vectors, token_count

    def diagnostics(self):
        return {
            "architecture": "bert", "route": "embeddings",
            "qualification": "unqualified", "representation": representation_contract(),
            "generation": "unsupported", "reranking": "unsupported",
            "apcv2": "not_applicable", "kv_cache": "not_applicable",
            "streaming": "not_applicable",
        }

    def close(self):
        self.model = None
        self.tokenizer = None
