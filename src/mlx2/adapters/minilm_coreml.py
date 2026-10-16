"""Pinned MiniLM embeddings executed by a Core ML ``CPU_AND_NE`` bundle.

This route never imports MLX and never selects the GPU: Core ML runs in a
spawned worker with the GPU excluded. The ordinary reference path remains
``adapters.minilm.MiniLMAdapter``; this adapter never falls back to it.
Neural Engine execution is not observed, only permitted and (optionally)
anticipated by the bundle's build-time compute plan.
"""

from __future__ import annotations

import copy
import threading

from ..contracts import Capability, ModelDescriptor
from .minilm import (
    DIMENSIONS,
    MAX_INPUT_BYTES,
    MAX_INPUTS,
    MAX_SEQUENCE_LENGTH,
    representation_contract,
)

DEFAULT_BUCKETS = (32, 64, 128, 256)
VARIANT = "all-minilm-l6-v2-coreml-cpu-and-ne"


def execution_block(buckets) -> dict:
    return {
        "backend": "coreml",
        "compute_units": "cpu_and_ne",
        "gpu": "excluded",
        "precision": "float16",
        "buckets": list(buckets),
        "neural_engine_observed": False,
    }


def descriptor_for(buckets=DEFAULT_BUCKETS) -> ModelDescriptor:
    return ModelDescriptor(
        model_type="bert",
        family="minilm",
        variant=VARIANT,
        state_planes=frozenset(),
        capabilities=frozenset({Capability.EMBEDDINGS}),
        metadata={
            "execution_kind": "embeddings",
            "qualification": "unqualified",
            "representation": {**representation_contract(), "execution": execution_block(buckets)},
        },
    )


DESCRIPTOR = descriptor_for()


def _identity(bundle, *, compiled_payload_verified):
    # The fingerprint binds the served bundle bytes. The manifest's source
    # checkpoint is a builder claim the runtime cannot verify from a compiled
    # graph, so it is reported beside the identity, never as it.
    return {
        "fingerprint": bundle.fingerprint,
        "fingerprint_kind": "coreml_encoder_bundle",
        "path": str(bundle.path),
        "bundle_manifest_sha256": bundle.manifest_sha256,
        "compiled_payload_verified": compiled_payload_verified,
        "source": bundle.source_attestation(),
    }


def _route_representation(bundle):
    from ..runtime.coreml_encoder import SOURCE_RELATIONSHIP, anticipated_placement

    manifest = bundle.manifest
    parity = manifest["build_parity"]
    plan = manifest["compute_plan"]
    return {
        **representation_contract(),
        "execution": {
            **execution_block(bundle.buckets),
            "bundle_fingerprint": bundle.fingerprint,
            "bundle_manifest_sha256": bundle.manifest_sha256,
            "source": bundle.source_attestation(),
            "build_parity": {**{key: parity[key] for key in (
                "reference", "threshold_min_cosine", "min_cosine", "max_abs_error", "passed",
            )}, "attestation": SOURCE_RELATIONSHIP},
            # Build-time MLComputePlan labels, not observed device use. A
            # CPU-planned bucket still serves; it is reported, not refused.
            "anticipated_placement": anticipated_placement(manifest),
            "compute_plan": copy.deepcopy(plan),
            "retrieval_quality_evaluated_on": "ordinary_mlx_reference",
        },
    }


def inspect_artifact(model_path):
    """Manifest, layout and tokenizer-pin inspection; no tokenizer, Core ML or MLX import."""
    from ..runtime.coreml_encoder import inspect_bundle

    bundle = inspect_bundle(model_path)
    return {
        "path": str(bundle.path),
        "identity": _identity(bundle, compiled_payload_verified=False),
        "buckets": list(bundle.buckets),
        "representation": _route_representation(bundle),
    }


def _make_executor(bundle):
    from ..runtime.coreml_encoder import CoreMLEncoderExecutor

    return CoreMLEncoderExecutor(bundle, backend="process")


class MiniLMCoreMLAdapter:
    descriptor = DESCRIPTOR
    default_route = "embeddings"

    def __init__(self, model_path, *, execution_policy=None):
        if execution_policy:
            raise ValueError("MiniLM Core ML embeddings do not accept generation execution policies")
        from ..runtime.coreml_encoder import inspect_bundle

        bundle = inspect_bundle(model_path)
        # Tokenize precisely the verified bytes, not a second read of the path.
        tokenizer_bytes = bundle.read_tokenizer()
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_str(tokenizer_bytes.decode())
        tokenizer.enable_truncation(MAX_SEQUENCE_LENGTH, direction="right")
        tokenizer.no_padding()
        self.executor = _make_executor(bundle)
        self.tokenizer = tokenizer
        self.buckets = bundle.buckets
        self.descriptor = descriptor_for(self.buckets)
        self.identity = _identity(bundle, compiled_payload_verified=True)
        self._representation = _route_representation(bundle)
        self.max_context = MAX_SEQUENCE_LENGTH
        self.environment = {}
        self._embedding_executions = 0
        # One request is atomic with respect to close: close waits for a
        # whole multi-text embed, and an embed after close runs nothing.
        self._lifecycle = threading.RLock()

    def representation_execution_count(self):
        """Host-only forward-start count; never calls Core ML for status."""
        return self._embedding_executions

    def embedding_execution_count(self):
        return self.representation_execution_count()

    def representation_contract(self):
        return {
            **copy.deepcopy(self._representation),
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
        import numpy as np

        with self._lifecycle:
            executor, tokenizer = self.executor, self.tokenizer
            if executor is None or tokenizer is None:
                raise RuntimeError("MiniLM Core ML adapter is closed")
            # Tokenize everything first: a tokenizer fault must not leave a
            # partially executed request.
            rows = []
            for text in texts:
                encoded = tokenizer.encode(text, add_special_tokens=True)
                count = len(encoded.ids)
                if not 2 <= count <= MAX_SEQUENCE_LENGTH or any(value != 1 for value in encoded.attention_mask):
                    raise RuntimeError("MiniLM tokenizer violated the unpadded 256-token recipe")
                bucket = next(length for length in self.buckets if length >= count)
                ids = np.zeros((1, bucket), dtype=np.int32)
                mask = np.zeros((1, bucket), dtype=np.int32)
                ids[0, :count] = encoded.ids
                mask[0, :count] = 1
                rows.append((ids, mask, count))
            vectors, token_count = [], 0
            # Width-one calls, like the ordinary adapter: padding beyond a
            # text's own bucket never changes another text's readout.
            for ids, mask, count in rows:
                self._embedding_executions += 1
                vectors.append(executor.predict(ids, mask))
                token_count += count
            return vectors, token_count

    def diagnostics(self):
        executor = self.executor
        return {
            "architecture": "bert", "route": "embeddings",
            "qualification": "unqualified",
            "representation": copy.deepcopy(self._representation),
            "coreml": executor.status() if executor is not None else {"closed": True},
            "generation": "unsupported", "reranking": "unsupported",
            "apcv2": "not_applicable", "kv_cache": "not_applicable",
            "streaming": "not_applicable",
        }

    def close(self):
        with self._lifecycle:
            executor = self.executor
            if executor is not None:
                executor.close()
            self.executor = None
            self.tokenizer = None
