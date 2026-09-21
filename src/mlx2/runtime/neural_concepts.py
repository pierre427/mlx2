"""Learned recurrent concept state and a bounded prefill cross-attention bridge.

The module deliberately keeps persistence and CPU validation independent of
MLX.  Training/export may use MLX, while the serving adapter converts the
validated numpy arrays to device arrays only after all identity checks pass.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .hyper_directory import DirectoryContext, Scope
from .semantic_memory import RELATIONS, SemanticMemory

NEURAL_CONCEPT_SCHEMA = "mlx2-neural-concept-bridge-v1"
NEURAL_STATE_SCHEMA = "mlx2-neural-concept-state-v1"
MAX_ACTIVE_CONCEPTS = 32


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_features(label: str, dimension: int) -> np.ndarray:
    """Stable signed character n-gram features with no vocabulary authority."""
    if not isinstance(label, str) or not label.strip():
        raise ValueError("concept labels must be nonempty text")
    if type(dimension) is not int or not 8 <= dimension <= 1024:
        raise ValueError("feature dimension must be an integer in 8..1024")
    normalized = " ".join(label.casefold().split())
    padded = f"^{normalized}$"
    result = np.zeros((dimension,), dtype=np.float32)
    grams = [padded]
    for width in (2, 3, 4):
        grams.extend(padded[index : index + width] for index in range(len(padded) - width + 1))
    for gram in grams:
        raw = hashlib.blake2b(gram.encode("utf-8"), digest_size=16).digest()
        index = int.from_bytes(raw[:8], "little") % dimension
        result[index] += 1.0 if raw[8] & 1 else -1.0
    norm = float(np.linalg.norm(result))
    return result if norm == 0 else result / norm


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-value))


def _softmax(value: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = value - np.max(value, axis=axis, keepdims=True)
    weights = np.exp(shifted)
    return weights / np.sum(weights, axis=axis, keepdims=True)


@dataclass(frozen=True, slots=True)
class NeuralConceptArtifact:
    root: Path
    manifest: Mapping
    arrays: Mapping[str, np.ndarray]

    @property
    def feature_dim(self) -> int:
        return int(self.manifest["feature_dim"])

    @property
    def state_dim(self) -> int:
        return int(self.manifest["state_dim"])

    @property
    def hidden_dim(self) -> int:
        return int(self.manifest["hidden_dim"])

    @property
    def fingerprint(self) -> str:
        return str(self.manifest["fingerprint"])

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        model_binding: str,
        tokenizer_binding: str,
        runtime_binding: str,
    ) -> NeuralConceptArtifact:
        root = Path(root).expanduser().resolve()
        manifest_path = root / "manifest.json"
        weights_path = root / "weights.npz"
        if root.is_symlink() or not root.is_dir():
            raise ValueError("neural concept artifact root must be a real directory")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != NEURAL_CONCEPT_SCHEMA:
            raise ValueError("unsupported neural concept artifact schema")
        expected = {
            "model": model_binding,
            "tokenizer": tokenizer_binding,
            "runtime": runtime_binding,
        }
        if manifest.get("bindings") != expected:
            raise ValueError("neural concept artifact binding mismatch")
        if manifest.get("weights_sha256") != _sha256(weights_path):
            raise ValueError("neural concept artifact weight digest mismatch")
        fingerprint_payload = {
            key: value for key, value in manifest.items() if key != "fingerprint"
        }
        encoded = json.dumps(
            fingerprint_payload, sort_keys=True, separators=(",", ":")
        ).encode()
        fingerprint = hashlib.sha256(encoded + weights_path.read_bytes()).hexdigest()
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("neural concept artifact fingerprint mismatch")
        feature_dim = int(manifest.get("feature_dim", 0))
        state_dim = int(manifest.get("state_dim", 0))
        hidden_dim = int(manifest.get("hidden_dim", 0))
        rounds = int(manifest.get("message_rounds", 0))
        temperature = float(manifest.get("attention_temperature", 0.0))
        if not (8 <= feature_dim <= 1024 and 8 <= state_dim <= 512):
            raise ValueError("neural concept dimensions are out of bounds")
        if (
            hidden_dim <= 0
            or hidden_dim > 32768
            or not 1 <= rounds <= 4
            or not 0.001 <= temperature <= 1.0
        ):
            raise ValueError("neural concept bridge geometry is out of bounds")
        relation_order = tuple(manifest.get("relations", ()))
        if relation_order != tuple(sorted(RELATIONS)):
            raise ValueError("neural concept relation vocabulary mismatch")
        required = {
            "input_weight": (feature_dim, state_dim),
            "input_bias": (state_dim,),
            "relation_weight": (len(relation_order), state_dim, state_dim),
            "gru_z_weight": (2 * state_dim, state_dim),
            "gru_z_bias": (state_dim,),
            "gru_r_weight": (2 * state_dim, state_dim),
            "gru_r_bias": (state_dim,),
            "gru_h_weight": (2 * state_dim, state_dim),
            "gru_h_bias": (state_dim,),
            "key_projection": (state_dim, hidden_dim),
            "value_projection": (state_dim, hidden_dim),
            "output_gate": (1,),
        }
        with np.load(weights_path, allow_pickle=False) as stored:
            if set(stored.files) != set(required):
                raise ValueError("neural concept artifact tensor set mismatch")
            arrays = {}
            for name, shape in required.items():
                value = np.asarray(stored[name], dtype=np.float32)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"invalid neural concept tensor: {name}")
                arrays[name] = value
        return cls(root=root, manifest=manifest, arrays=arrays)


@dataclass(frozen=True, slots=True)
class EncodedConcept:
    concept_id: str
    key_state: tuple[float, ...]
    value_state: tuple[float, ...]


class RecurrentConceptEncoder:
    """Relation-aware synchronous message passing with a learned GRU update."""

    def __init__(self, artifact: NeuralConceptArtifact):
        self.artifact = artifact
        self.relations = {
            name: index for index, name in enumerate(artifact.manifest["relations"])
        }

    def encode_graph(self, graph: Mapping) -> tuple[EncodedConcept, ...]:
        concepts = graph.get("concepts")
        edges = graph.get("edges")
        if not isinstance(concepts, Mapping) or not isinstance(edges, list):
            raise TypeError("semantic graph must contain concepts and edges")
        identifiers = tuple(sorted(concepts))
        if len(identifiers) > int(self.artifact.manifest.get("max_graph_concepts", 4096)):
            raise ValueError("semantic graph exceeds neural encoder bound")
        if not identifiers:
            return ()
        index = {identifier: position for position, identifier in enumerate(identifiers)}
        features = np.stack(
            [
                label_features(concepts[identifier]["label"], self.artifact.feature_dim)
                for identifier in identifiers
            ]
        )
        arrays = self.artifact.arrays
        base = np.tanh(features @ arrays["input_weight"] + arrays["input_bias"])
        state = base.copy()
        for _ in range(int(self.artifact.manifest["message_rounds"])):
            messages = np.zeros_like(state)
            counts = np.zeros((len(identifiers), 1), dtype=np.float32)
            for edge in edges:
                if edge.get("authority") != "committed":
                    continue
                subject = index.get(edge.get("subject"))
                object_ = index.get(edge.get("object"))
                relation = self.relations.get(edge.get("relation"))
                if subject is None or object_ is None or relation is None:
                    continue
                messages[subject] += state[object_] @ arrays["relation_weight"][relation]
                counts[subject] += 1.0
            messages /= np.maximum(counts, 1.0)
            joined = np.concatenate([messages, state], axis=-1)
            z = _sigmoid(joined @ arrays["gru_z_weight"] + arrays["gru_z_bias"])
            r = _sigmoid(joined @ arrays["gru_r_weight"] + arrays["gru_r_bias"])
            candidate_input = np.concatenate([messages, r * state], axis=-1)
            candidate = np.tanh(
                candidate_input @ arrays["gru_h_weight"] + arrays["gru_h_bias"]
            )
            updated = (1.0 - z) * state + z * candidate
            state = np.where(counts > 0, updated, state)
        return tuple(
            EncodedConcept(
                concept_id=identifier,
                key_state=tuple(float(value) for value in base[position]),
                value_state=tuple(float(value) for value in state[position]),
            )
            for position, identifier in enumerate(identifiers)
        )

    def state_document(self, graph: Mapping, *, semantic_capsule: str) -> dict:
        encoded = self.encode_graph(graph)
        return {
            "schema": NEURAL_STATE_SCHEMA,
            "artifact_fingerprint": self.artifact.fingerprint,
            "semantic_capsule": semantic_capsule,
            "state_dim": self.artifact.state_dim,
            "concepts": [
                {
                    "id": item.concept_id,
                    "key_state": list(item.key_state),
                    "value_state": list(item.value_state),
                }
                for item in encoded
            ],
        }


class ConceptCrossAttention:
    """Numpy correctness oracle for the learned prefill bridge."""

    def __init__(self, artifact: NeuralConceptArtifact):
        self.artifact = artifact

    def project(self, concepts: Sequence[EncodedConcept]):
        if not concepts or len(concepts) > MAX_ACTIVE_CONCEPTS:
            raise ValueError("active concept count must be in 1..32")
        keys = np.asarray([item.key_state for item in concepts], dtype=np.float32)
        values = np.asarray([item.value_state for item in concepts], dtype=np.float32)
        keys = keys @ self.artifact.arrays["key_projection"]
        values = values @ self.artifact.arrays["value_projection"]
        key_norm = np.linalg.norm(keys, axis=-1, keepdims=True)
        keys = keys / np.maximum(key_norm, 1e-6)
        return keys, values

    def apply(self, query_embeddings: np.ndarray, concepts: Sequence[EncodedConcept]):
        queries = np.asarray(query_embeddings, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.artifact.hidden_dim:
            raise ValueError("query embeddings do not match bridge hidden dimension")
        keys, values = self.project(concepts)
        normalized = queries / np.maximum(
            np.linalg.norm(queries, axis=-1, keepdims=True), 1e-6
        )
        logits = normalized @ keys.T
        weights = _softmax(
            logits / float(self.artifact.manifest["attention_temperature"]), axis=-1
        )
        gate = float(_sigmoid(self.artifact.arrays["output_gate"])[0])
        residual = gate * (weights @ values)
        return queries + residual, weights


def validate_state_document(document: Mapping, artifact: NeuralConceptArtifact) -> tuple[EncodedConcept, ...]:
    if document.get("schema") != NEURAL_STATE_SCHEMA:
        raise ValueError("unsupported neural concept state schema")
    if document.get("artifact_fingerprint") != artifact.fingerprint:
        raise ValueError("neural concept state artifact mismatch")
    if document.get("state_dim") != artifact.state_dim:
        raise ValueError("neural concept state dimension mismatch")
    rows = document.get("concepts")
    if not isinstance(rows, list) or len(rows) > int(
        artifact.manifest.get("max_graph_concepts", 4096)
    ):
        raise ValueError("invalid neural concept state rows")
    result = []
    seen = set()
    for row in rows:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("neural concept state IDs must be unique strings")
        seen.add(identifier)
        key = np.asarray(row.get("key_state"), dtype=np.float32)
        value = np.asarray(row.get("value_state"), dtype=np.float32)
        if key.shape != (artifact.state_dim,) or value.shape != (artifact.state_dim,):
            raise ValueError("neural concept state row has wrong shape")
        if not np.isfinite(key).all() or not np.isfinite(value).all():
            raise ValueError("neural concept state row must be finite")
        result.append(
            EncodedConcept(identifier, tuple(map(float, key)), tuple(map(float, value)))
        )
    return tuple(result)


class NeuralConceptMemory:
    """Derived recurrent state bound to one authoritative semantic capsule."""

    def __init__(self, semantic_memory: SemanticMemory, artifact: NeuralConceptArtifact):
        self.semantic_memory = semantic_memory
        self.capsules = semantic_memory.capsules
        self.directory = semantic_memory.directory
        self.artifact = artifact
        self.encoder = RecurrentConceptEncoder(artifact)

    def load(self, context: DirectoryContext) -> tuple[EncodedConcept, ...]:
        resolved = self.directory.resolve(context)
        semantic_digest = resolved.handles.get("semantic-memory")
        neural_digest = resolved.handles.get("neural-concepts")
        if semantic_digest is None or neural_digest is None:
            return ()
        capsule = self.capsules.get(neural_digest)
        if capsule["kind"] != "neural_state":
            raise ValueError("neural-concepts handle points to wrong capsule kind")
        expected = {
            "model": self.semantic_memory.bindings["model_binding"],
            "tokenizer": self.semantic_memory.bindings["tokenizer_binding"],
            "runtime": self.semantic_memory.bindings["runtime_binding"],
        }
        if capsule["bindings"] != expected:
            raise ValueError("neural concept capsule runtime binding mismatch")
        document = capsule["data"]
        if document.get("semantic_capsule") != semantic_digest:
            raise ValueError("neural concept state is stale for semantic graph")
        return validate_state_document(document, self.artifact)

    def rebuild(self, context: DirectoryContext) -> dict:
        graph, semantic_digest, revision = self.semantic_memory.load(context)
        if semantic_digest is None:
            return {"committed": False, "reason": "no-semantic-memory"}
        document = self.encoder.state_document(
            graph, semantic_capsule=semantic_digest
        )
        capsule = self.capsules.put(
            kind="neural_state",
            data=document,
            parents=(semantic_digest,),
            provenance={
                "operation": "recurrent-concept-encode",
                "artifact_fingerprint": self.artifact.fingerprint,
            },
            **self.semantic_memory.bindings,
        )
        layer = self.directory.update(
            Scope.SESSION,
            context,
            expected_revision=revision,
            handles={"neural-concepts": capsule.digest},
        )
        return {
            "committed": True,
            "capsule": capsule.digest,
            "semantic_capsule": semantic_digest,
            "revision": layer["revision"],
            "concepts": len(document["concepts"]),
            "artifact_fingerprint": self.artifact.fingerprint,
        }


__all__ = [
    "MAX_ACTIVE_CONCEPTS",
    "NEURAL_CONCEPT_SCHEMA",
    "NEURAL_STATE_SCHEMA",
    "ConceptCrossAttention",
    "EncodedConcept",
    "NeuralConceptArtifact",
    "NeuralConceptMemory",
    "RecurrentConceptEncoder",
    "label_features",
    "validate_state_document",
]
