"""Automatic serving middleware for capsule-backed semantic memory."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import threading
from typing import Mapping

from .runtime.hyper_directory import DirectoryContext, HyperDirectory
from .runtime.semantic_capsules import CapsuleStore
from .runtime.semantic_memory import SemanticMemory, SemanticProposal


REMEMBER_PATTERN = re.compile(
    r"\bremember\s+(?:that\s+)?(?P<subject>[\w][\w '\-]{0,79}?)\s+"
    r"(?:is|are|=)\s+(?P<object>[^.!?\n]{1,120})",
    re.IGNORECASE,
)


def _safe_scope(prefix: str, value: str) -> str:
    return f"{prefix}-{hashlib.sha256(str(value).encode()).hexdigest()[:24]}"


def _message_text(messages) -> str:
    values = []
    for message in messages or ():
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            values.append(content)
        elif isinstance(content, list):
            values.extend(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
    return "\n".join(values)


@dataclass(frozen=True, slots=True)
class SemanticRequestState:
    context: DirectoryContext
    source_text: str
    directory_revision: int
    directory_fingerprint: str
    retrieved_concepts: int
    retrieved_edges: int
    authenticated_tenant: bool


class SemanticServingMiddleware:
    """Retrieve before generation and commit only after response delivery."""

    def __init__(self, memory: SemanticMemory, *, model_scope: str, retrieval_limit=8):
        self.memory = memory
        self.model_scope = _safe_scope("model", model_scope)
        self.retrieval_limit = int(retrieval_limit)
        if not 1 <= self.retrieval_limit <= 32:
            raise ValueError("semantic retrieval limit must be between 1 and 32")
        self._lock = threading.Lock()
        self._status = {
            "enabled": True,
            "schema": "mlx2-semantic-middleware-v1",
            "prepared": 0,
            "retrieval_hits": 0,
            "post_delivery_commits": 0,
            "deferred_proposals": 0,
            "failures": 0,
            "last_commit": None,
        }

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        model_binding: str,
        tokenizer_binding: str,
        runtime_binding: str,
        retrieval_limit: int = 8,
    ):
        root = Path(root).expanduser().resolve()
        capsules = CapsuleStore(root / "capsules")
        directory = HyperDirectory(root / "directory", capsules)
        memory = SemanticMemory(
            capsules=capsules,
            directory=directory,
            model_binding=model_binding,
            tokenizer_binding=tokenizer_binding,
            runtime_binding=runtime_binding,
        )
        return cls(memory, model_scope=model_binding, retrieval_limit=retrieval_limit)

    def _context(self, tenant_id: str, session_id: str) -> DirectoryContext:
        return DirectoryContext(
            model=self.model_scope,
            tenant=_safe_scope("tenant", tenant_id),
            session=_safe_scope("session", session_id),
        )

    def prepare(self, body: Mapping, *, tenant_id: str, authenticated_tenant: bool):
        session_id = body.get("session_id")
        messages = body.get("messages")
        if not session_id or not isinstance(messages, list) or body.get("n", 1) != 1:
            return dict(body), None
        context = self._context(tenant_id, session_id)
        query = _message_text(messages[-4:])
        resolved = self.memory.directory.resolve(context)
        result = self.memory.retrieve(context, query, limit=self.retrieval_limit)
        preamble = result.preamble()
        prepared = dict(body)
        if preamble:
            prepared["messages"] = [
                {"role": "system", "content": preamble},
                *messages,
            ]
        prepared["_mlx2_semantic_fingerprint"] = resolved.fingerprint
        state = SemanticRequestState(
            context=context,
            source_text=query,
            directory_revision=resolved.layers[-1]["revision"] if resolved.layers else 0,
            directory_fingerprint=resolved.fingerprint,
            retrieved_concepts=len(result.concepts),
            retrieved_edges=len(result.edges),
            authenticated_tenant=bool(authenticated_tenant),
        )
        with self._lock:
            self._status["prepared"] += 1
            if result.concepts:
                self._status["retrieval_hits"] += 1
        return prepared, state

    @staticmethod
    def proposals(state: SemanticRequestState, output_text: str):
        # Deliberately conservative v1 extractor: only explicit "remember"
        # statements can cross the durable commit gate. The response is part
        # of the evidence digest but cannot invent a memory by itself.
        evidence = state.source_text + "\n---assistant---\n" + output_text
        return tuple(
            SemanticProposal(
                match.group("subject").strip(),
                "has_property",
                match.group("object").strip(),
                0.99,
                0.80,
                evidence,
            )
            for match in REMEMBER_PATTERN.finditer(state.source_text)
        )

    def complete(self, state: SemanticRequestState | None, output_text: str) -> dict | None:
        if state is None:
            return None
        try:
            result = self.memory.commit_after_delivery(
                state.context,
                self.proposals(state, output_text),
                response_delivered=True,
                authenticated_tenant=state.authenticated_tenant,
                expected_revision=state.directory_revision,
            )
            with self._lock:
                if result.get("committed"):
                    self._status["post_delivery_commits"] += 1
                    self._status["deferred_proposals"] += result.get(
                        "deferred_proposals", 0
                    )
                self._status["last_commit"] = result
            return result
        except Exception:
            with self._lock:
                self._status["failures"] += 1
                self._status["last_commit"] = {
                    "committed": False,
                    "reason": "post-delivery-commit-failed",
                }
            return self._status["last_commit"]

    def receipt(self, state: SemanticRequestState | None) -> dict | None:
        if state is None:
            return None
        return {
            "engaged": True,
            "directory_fingerprint": state.directory_fingerprint,
            "directory_revision": state.directory_revision,
            "retrieved_concepts": state.retrieved_concepts,
            "retrieved_edges": state.retrieved_edges,
            "durable_commit_eligible": state.authenticated_tenant,
            "commit_boundary": "after-response-delivery",
        }

    def delete_session(self, tenant_id: str, session_id: str) -> bool:
        return self.memory.directory.delete_session(self._context(tenant_id, session_id))

    def status(self) -> dict:
        with self._lock:
            return dict(self._status)


__all__ = ["SemanticRequestState", "SemanticServingMiddleware"]
