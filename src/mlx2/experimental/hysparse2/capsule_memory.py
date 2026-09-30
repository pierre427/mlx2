"""Differentiable bounded reads of existing authoritative semantic capsules.

CapsuleStore owns persistence/integrity. This snapshot never writes inference
observations into memory and never promotes semantic proposals.
"""

from dataclasses import dataclass

from mlx2.runtime.semantic_capsules import content_digest
from mlx2.runtime.semantic_memory import SEMANTIC_SCHEMA


@dataclass(frozen=True)
class CapsuleMemory:
    capsule_digest: str
    model_binding: str
    tokenizer_binding: str
    runtime_binding: str
    records: tuple
    tokens: tuple
    top_k: int
    vocab_size: int
    fingerprint: str

    @classmethod
    def from_store(
        cls,
        store,
        digest,
        *,
        model_binding,
        tokenizer_binding,
        runtime_binding,
        encode,
        vocab_size,
        top_k=4,
        max_records=32,
        max_tokens=512,
    ):
        if any(
            not isinstance(v, str) or not v
            for v in (model_binding, tokenizer_binding, runtime_binding)
        ):
            raise ValueError("explicit model/tokenizer/runtime bindings required")
        if (
            any(
                type(v) is not int or v < 1
                for v in (vocab_size, top_k, max_records, max_tokens)
            )
            or max_records > 32
            or max_tokens > 512
        ):
            raise ValueError("invalid bounded memory dimensions")
        capsule = store.get(digest)
        expected = {
            "model": model_binding,
            "tokenizer": tokenizer_binding,
            "runtime": runtime_binding,
        }
        if (
            capsule["kind"] not in {"semantic_base", "semantic_delta"}
            or capsule["bindings"] != expected
        ):
            raise ValueError("semantic capsule kind or bindings differ")
        graph = capsule["data"]
        if graph.get("schema") != SEMANTIC_SCHEMA:
            raise ValueError("unsupported semantic graph")
        concepts = graph["concepts"]
        rows = []
        for edge in sorted(
            graph["edges"], key=lambda e: (e["subject"], e["relation"], e["object"])
        ):
            if edge.get("authority") != "committed":
                continue
            evidence = edge.get("evidence_digest")
            if (
                not isinstance(evidence, str)
                or len(evidence) != 64
                or any(c not in "0123456789abcdef" for c in evidence)
            ):
                raise ValueError("committed edge requires evidence digest")
            text = f"{concepts[edge['subject']]['label']} {edge['relation']} {concepts[edge['object']]['label']}"
            ids = tuple(encode(text))
            if (
                not ids
                or len(ids) > max_tokens
                or any(type(t) is not int or not 0 <= t < vocab_size for t in ids)
            ):
                raise ValueError("capsule tokenization exceeds bounds")
            rows.append((text, evidence, ids))
        if not rows or len(rows) > max_records:
            raise ValueError("capsule needs a bounded nonempty committed edge set")
        top_k = min(top_k, len(rows))
        records = tuple((r[0], r[1]) for r in rows)
        tokens = tuple(r[2] for r in rows)
        fingerprint = content_digest(
            {
                "schema": "mlx2.hysparse2-capsule-read.v1",
                "capsule": digest,
                "bindings": expected,
                "tokens": tokens,
                "top_k": top_k,
                "vocab_size": vocab_size,
            }
        )
        return cls(
            digest,
            model_binding,
            tokenizer_binding,
            runtime_binding,
            records,
            tokens,
            top_k,
            vocab_size,
            fingerprint,
        )

    def binding(self):
        return {
            "capsule_digest": self.capsule_digest,
            "read_fingerprint": self.fingerprint,
            "model": self.model_binding,
            "tokenizer": self.tokenizer_binding,
            "runtime": self.runtime_binding,
        }

    def read(self, ple, query):
        import mlx.core as mx

        width = max(map(len, self.tokens))
        ids = mx.array([list(row) + [0] * (width - len(row)) for row in self.tokens])
        mask = (
            mx.arange(width)[None, :]
            < mx.array([len(row) for row in self.tokens])[:, None]
        )
        rows = ple.embedding(ple.indices(ids))
        pooled = (
            mx.sum(mx.where(mask[..., None], rows, 0), axis=1)
            / mx.sum(mask, axis=1)[:, None]
        )
        keys, values = ple.key(pooled), ple.value(pooled)
        normalized = query * mx.rsqrt(
            mx.mean(mx.square(query.astype(mx.float32)), axis=-1, keepdims=True) + 1e-6
        ).astype(query.dtype)
        scores = (
            normalized.astype(mx.float32) @ keys.astype(mx.float32).T
        ) * query.shape[-1] ** -0.5
        chosen = mx.stop_gradient(
            mx.argpartition(scores, kth=len(self.tokens) - self.top_k, axis=-1)[
                ..., -self.top_k :
            ]
        )
        weights = mx.softmax(mx.take_along_axis(scores, chosen, axis=-1), axis=-1)
        result = mx.sum(
            values[chosen] * weights[..., None].astype(values.dtype), axis=-2
        )
        # Reuse the learned PLE gate; no new untrained parameter tree is added.
        return mx.sigmoid(ple.gate_bias).astype(result.dtype) * result
