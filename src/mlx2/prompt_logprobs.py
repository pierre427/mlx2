"""Opt-in prompt-token log-probabilities (``echo`` + ``logprobs``, ``prompt_logprobs``).

Original mlx2 implementation; no code mined.  The API shapes follow the public
OpenAI legacy Completions contract (``echo`` with integer ``logprobs``,
``max_tokens: 0``) and vLLM's ``prompt_logprobs`` request extension.

What a scored value is
----------------------
Position ``i`` (``i >= 1``) carries ``log p(t_i | t_0..t_{i-1})`` under the
loaded target model: the fp32 log-softmax of the row the target forward
computes at position ``i - 1``.  Position 0 has no context and is ``null``.
These are the model's own distribution.  Sampling transforms and logits
processors (temperature, penalties, bias, grammar) are generation-time laws
and are not applied.

How it runs (model-agnostic)
----------------------------
A scoring request is one task for the serving worker
(``ServingEngine.submit_prompt_scoring``).  Between scheduler steps the worker
runs one bounded position chunk of it, so live lanes keep decoding and new
requests keep being admitted; nothing drains or closes admission.  The task
owns one fresh request-private cache built with ``make_prompt_cache`` and runs
the target's ordinary prompt forward (``prefill_model_forward``:
``model.prefill_forward`` when the model declares one, else
``model.__call__``).  So:

* width is 1 by construction; no other lane shares the forward;
* the APCv2 store is neither read nor written, so every prompt row exists and
  no APCv2 state, revision or hit gate is involved;
* no drafter, MTP head, verifier or speculative route runs; returned values
  never contain draft or residual probabilities;
* no checkpoint-row request is open, so prefill tail-skip and head trims stay
  inert and the forward returns every row (checked; a short row count fails
  closed);
* the chunk stays below every approximate row-threshold path the engine has
  bound (W8A8 int8 prefill), so those kernels never engage; a route whose
  numerics are declared approximate, or whose APCv2 reuse is disabled for a
  fidelity reason, is refused (``scoring_fidelity``).

Memory: only ``chunk_tokens`` rows of logits exist at a time.  Each chunk is
normalized in fp32, reduced to the actual token's log-probability, its rank
and the top-K, and evaluated before the next chunk runs.  A ``[T, vocab]``
array is never formed.

Adapter contract (declared attributes; serving never branches on a model name):
``prompt_logprobs_binding()`` returns ``(forward, cache)`` to replace the
default; ``prompt_logprobs_refusal`` (string or callable) refuses the route;
``prompt_logprobs_fidelity`` (string or callable) returns ``"exact"`` or the
reason the selected numerics are not exact.
"""

from __future__ import annotations

import codecs
import json
import math
import threading
import time
from dataclasses import dataclass, field

SCHEMA = "mlx2.prompt-logprobs.v2"
FLAG = "--prompt-logprobs"
DEFAULT_MAX_PROMPT_TOKENS = 32768
DEFAULT_MAX_TOP = 20
DEFAULT_CHUNK_TOKENS = 256
DEFAULT_TIMEOUT_SECONDS = 600.0
# Request-scoped weight programs change the forward for one request only; the
# scorer runs the base target, so it refuses them rather than mislabel rows.
_REQUEST_WEIGHT_PROGRAMS = (
    "weight_overlay",
    "layer_excision",
    "expert_mask",
    "mlx2_overlay",
    "mlx2_amoeba_tiles",
    "mlx2_amoeba_open",
    "mlx2_amoeba_gate_mode",
    "prefill_handoff",
)
# APCv2 reuse restrictions that concern batch invariance only.  A width-1
# scoring forward on a fresh cache is unaffected by them; every other declared
# restriction is treated as a statement that the route's numerics are not the
# exact reference, and scoring refuses it.
WIDTH_ONLY_APC_REASONS = frozenset({"external_varlen_prefill_not_batch_invariant"})


class PromptLogprobsRefused(ValueError):
    """A prompt-scoring request the server will not serve; carries a receipt.

    A ``ValueError`` so the HTTP layer's request-error path answers it with
    ``http_status``, ``code`` and the ``mlx2`` receipt.
    """

    def __init__(self, message, *, code, http_status=400, reason=None, extra=None):
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.mlx2 = {
            "prompt_logprobs": {
                "schema": SCHEMA,
                "status": "refused",
                "reason": reason or code,
                **(extra or {}),
            }
        }


def busy(message, reason="busy", extra=None):
    return PromptLogprobsRefused(message, code="prompt_logprobs_busy", http_status=503,
                                 reason=reason, extra=extra)


@dataclass(frozen=True)
class PromptLogprobsPolicy:
    """Server-wide switch, caps and access.  Default off."""

    enabled: bool = False
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS
    max_top: int = DEFAULT_MAX_TOP
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    # Tenants that may score under tenant authentication (besides credentials
    # with the ``prompt-logprobs`` scope).  Refused at startup without it.
    tenants: tuple = ()

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("prompt logprobs enabled must be boolean")
        for name, lower, upper in (
            ("max_prompt_tokens", 2, 1 << 22),
            ("max_top", 0, 1024),
            ("chunk_tokens", 2, 8192),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"prompt logprobs {name} must be an integer from {lower} to {upper}")
        timeout = self.timeout_seconds
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 86400:
            raise ValueError("prompt logprobs timeout must be in (0, 86400] seconds")
        tenants = tuple(self.tenants)
        if any(not isinstance(item, str) or not item for item in tenants):
            raise ValueError("prompt logprobs tenants must be nonempty tenant ids")
        object.__setattr__(self, "tenants", tenants)

    def authorized(self, *, tenant_id, tenant_auth, scoped=False):
        """Who may score.

        Without tenant authentication every caller that passed the transport
        gate (loopback bind or ``--api-key``) may score; a tenant list is
        refused at startup there.  With tenant authentication, a credential
        carrying the dedicated ``prompt-logprobs`` scope or a listed tenant may.
        """
        if not tenant_auth:
            return not self.tenants
        return bool(scoped) or tenant_id in self.tenants

    def status(self) -> dict:
        return {
            "schema": SCHEMA,
            "enabled": self.enabled,
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_top": self.max_top,
            "chunk_tokens": self.chunk_tokens,
            "timeout_seconds": float(self.timeout_seconds),
            "access": "tenants" if self.tenants else "default",
            "concurrency": 1,
            "endpoints": {
                "/v1/completions": ["echo+logprobs", "prompt_logprobs"],
                "/v1/chat/completions": ["prompt_logprobs"],
            },
            "execution": "worker_interleaved_width1_ordinary_target_forward",
        }


@dataclass(frozen=True)
class PromptScoringSpec:
    """What one request asked for, extracted before ordinary validation."""

    echo: bool
    echo_top: int | None  # legacy Completions alternatives when echo+logprobs
    prompt_top: int | None  # vLLM ``prompt_logprobs`` alternatives
    generate: bool  # False for max_tokens 0

    @property
    def scoring(self) -> bool:
        return self.echo_top is not None or self.prompt_top is not None

    @property
    def top_k(self) -> int:
        return max(self.echo_top or 0, self.prompt_top or 0)


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def requests_prompt_scoring(body):
    return isinstance(body, dict) and (
        body.get("echo") is not None or body.get("prompt_logprobs") is not None
    )


def extract_prompt_scoring(body, *, chat):
    """Split ``echo``/``prompt_logprobs``/``max_tokens: 0`` off a raw body.

    Returns ``(body, spec)``; ``spec`` is None and ``body`` unchanged when the
    request asks for neither echo nor prompt log-probabilities.  The returned
    body carries no scoring field, so ``validate_request`` sees an ordinary
    request (``max_tokens`` 0 is removed and reported as ``generate=False``).
    A recognised null is an absent field, as ``validate_request`` treats it.
    """
    if not isinstance(body, dict):
        return body, None
    echo = body.get("echo")
    prompt_top = body.get("prompt_logprobs")
    if echo is None and prompt_top is None:
        return body, None
    if echo is not None:
        if chat:
            raise ValueError("echo is a Completions field; use prompt_logprobs on /v1/chat/completions")
        if not isinstance(echo, bool):
            raise ValueError("echo must be boolean")
    if prompt_top is not None:
        prompt_top = _count(prompt_top, "prompt_logprobs")
    echo = bool(echo)
    if not echo and prompt_top is None:
        # ``echo: false`` alone is the default behaviour; the rest of the body
        # (``max_tokens: 0`` included) is validated as an ordinary request.
        return {key: value for key, value in body.items() if key != "echo"}, None
    echo_top = None
    if echo:
        legacy = body.get("logprobs")
        if legacy is not None and not isinstance(legacy, bool):
            echo_top = _count(legacy, "logprobs")
        elif legacy is True or body.get("top_logprobs"):
            echo_top = _count(body.get("top_logprobs") or 0, "top_logprobs")
        if echo_top is not None and (
            body.get("logprobs_start") is not None or body.get("logprobs_end") is not None
        ):
            raise PromptLogprobsRefused(
                "echo cannot be combined with logprobs_start/logprobs_end",
                code="prompt_logprobs_window_unsupported",
            )
    generate = True
    for key in ("max_tokens", "max_completion_tokens"):
        if key in body and body[key] == 0 and not isinstance(body[key], bool):
            generate = False
    if not generate and any(
        key in body and body[key] not in (0, None) for key in ("max_tokens", "max_completion_tokens")
    ):
        raise ValueError("max_tokens and max_completion_tokens disagree")
    stripped = {
        key: value
        for key, value in body.items()
        if key not in {"echo", "prompt_logprobs"}
        and not (not generate and key in {"max_tokens", "max_completion_tokens"})
    }
    if body.get("stream"):
        raise PromptLogprobsRefused(
            "echo and prompt_logprobs are not supported with stream=true",
            code="prompt_logprobs_stream_unsupported",
        )
    if body.get("n") not in (None, 1):
        raise PromptLogprobsRefused(
            "echo and prompt_logprobs require n=1",
            code="prompt_logprobs_parallel_unsupported",
        )
    return stripped, PromptScoringSpec(echo=echo, echo_top=echo_top, prompt_top=prompt_top,
                                       generate=generate)


def validate_exposure(policy, *, loopback_bind, tenant_auth, api_key, allow_unauthenticated):
    """Startup checks: a tenant list needs verified tenants; a public bind needs auth."""
    if not policy.enabled:
        return
    if policy.tenants and not tenant_auth:
        raise ValueError(
            "--prompt-logprobs-tenants requires tenant authentication: without it "
            "X-Tenant-ID is client-set and the list would admit anyone who names a tenant"
        )
    if not loopback_bind and not tenant_auth and not api_key and not allow_unauthenticated:
        raise ValueError(
            "--prompt-logprobs on with a non-loopback bind needs tenant authentication or "
            "an API key (or --prompt-logprobs-allow-unauthenticated to accept the exposure)"
        )


def disabled_refusal(message):
    return PromptLogprobsRefused(
        message,
        code="prompt_logprobs_disabled",
        reason="capability_disabled",
        extra={"flag": FLAG, "enabled": False},
    )


def admit_prompt_scoring(policy, spec, body, *, base_model):
    """Fail closed before any device work for a request the route cannot score."""
    if spec is None:
        return
    policy = policy or PromptLogprobsPolicy()
    if not policy.enabled:
        raise disabled_refusal(
            "prompt logprobs are disabled on this server (start it with --prompt-logprobs on)"
        )
    if not spec.scoring:
        return
    if spec.top_k > policy.max_top:
        raise PromptLogprobsRefused(
            f"prompt logprob alternatives exceed the server cap of {policy.max_top}",
            code="prompt_logprobs_top_cap",
            extra={"requested_top": spec.top_k, "max_top": policy.max_top},
        )
    selected = body.get("model")
    if selected is not None and selected != base_model:
        raise PromptLogprobsRefused(
            "prompt logprobs score the loaded base model only",
            code="prompt_logprobs_model_unsupported",
            extra={"requested_model": selected, "base_model": base_model},
        )
    for key in _REQUEST_WEIGHT_PROGRAMS:
        if key in body:
            raise PromptLogprobsRefused(
                f"prompt logprobs cannot be combined with {key}",
                code="prompt_logprobs_weight_program_unsupported",
                extra={"field": key},
            )
    for message in body.get("messages", ()) or ():
        if isinstance(message, dict) and isinstance(message.get("content"), list):
            raise PromptLogprobsRefused(
                "prompt logprobs are text-only; media parts are not scored",
                code="prompt_logprobs_media_unsupported",
            )


def admit_prompt_length(policy, prompt_tokens, *, max_context=None):
    policy = policy or PromptLogprobsPolicy()
    limit = policy.max_prompt_tokens
    if isinstance(max_context, int) and not isinstance(max_context, bool) and max_context > 0:
        limit = min(limit, max_context)
    if prompt_tokens > limit:
        raise PromptLogprobsRefused(
            f"prompt has {prompt_tokens} tokens; prompt logprobs are capped at {limit}",
            code="prompt_logprobs_length_cap",
            extra={"prompt_tokens": prompt_tokens, "max_prompt_tokens": policy.max_prompt_tokens,
                   "max_context": max_context, "limit": limit},
        )


# --------------------------------------------------------------------------
# Adapter contract


def _declared(value):
    return value() if callable(value) else value


def scoring_fidelity(adapter, *, apc_reuse_disabled_reason=None, numerical_contract=None,
                     microscaling_active=False):
    """Declared numerics of the selected route; refuse unless exact.

    Reads only declared attributes: the adapter's ``prompt_logprobs_fidelity``,
    its ``numeric_profile``/``numerics_revision``, its execution numerics
    contract, and the engine's APCv2 reuse restriction for the route.
    """
    numeric_profile = getattr(adapter, "numeric_profile", None)
    numerics_revision = getattr(adapter, "numerics_revision", None)
    reasons = []
    declared = getattr(adapter, "prompt_logprobs_fidelity", None)
    declared = _declared(declared) if declared is not None else None
    if declared not in (None, "exact"):
        reasons.append(f"adapter_declared:{declared}")
    if isinstance(numeric_profile, str) and "approximate" in numeric_profile.lower():
        reasons.append(f"numeric_profile:{numeric_profile}")
    if numerical_contract is not None and "approximate" in json.dumps(
        numerical_contract, sort_keys=True
    ).lower():
        reasons.append("execution_numerics_contract_approximate")
    if apc_reuse_disabled_reason and apc_reuse_disabled_reason not in WIDTH_ONLY_APC_REASONS:
        reasons.append(f"apcv2_reuse_disabled:{apc_reuse_disabled_reason}")
    if microscaling_active:
        reasons.append("microscaling_prefill_bound")
    receipt = {
        "fidelity": "exact" if not reasons else "refused",
        "declared": declared,
        "numeric_profile": numeric_profile if isinstance(numeric_profile, str) else None,
        "numerics_revision": numerics_revision if isinstance(numerics_revision, str) else None,
        "apcv2_reuse_disabled_reason": apc_reuse_disabled_reason,
    }
    if reasons:
        raise PromptLogprobsRefused(
            "the selected route does not declare exact numerics for prompt scoring: "
            + "; ".join(reasons),
            code="prompt_logprobs_unsupported_route",
            http_status=501,
            reason="route_not_exact",
            extra={**receipt, "reasons": reasons},
        )
    return receipt


def int8_chunk_cap(chunk_tokens, int8_handle):
    """Keep the scoring forward below the bound W8A8 prefill row threshold."""
    policy = getattr(int8_handle, "policy", None) if int8_handle is not None else None
    threshold = getattr(policy, "row_threshold", None)
    if not isinstance(threshold, int) or isinstance(threshold, bool):
        return chunk_tokens, {"active": False}
    capped = min(chunk_tokens, threshold - 1)
    return capped, {
        "active": True,
        "row_threshold": threshold,
        "engaged": False,
        "mechanism": "chunk_below_row_threshold",
        "chunk_tokens": capped,
    }


@dataclass
class PromptScoringBinding:
    forward: object  # (tokens [1, C], cache) -> logits [1, C, vocab]
    cache: object
    kind: str


def prompt_scoring_binding(adapter):
    """Resolve the target forward for scoring, or refuse with a receipt (501)."""
    def refuse(message, reason):
        raise PromptLogprobsRefused(message, code="prompt_logprobs_unsupported_route",
                                    http_status=501, reason=reason)

    refusal = getattr(adapter, "prompt_logprobs_refusal", None)
    refusal = _declared(refusal) if refusal is not None else None
    if refusal:
        refuse(f"the loaded route cannot score prompts: {refusal}", "adapter_declared_refusal")
    require_text_route(adapter)
    declared = getattr(adapter, "prompt_logprobs_binding", None)
    if callable(declared):
        forward, cache = declared()
        return PromptScoringBinding(forward, cache, "adapter_binding")
    model = getattr(adapter, "model", None)
    if model is None or not callable(model):
        refuse("the loaded adapter exposes no in-process target model", "no_target_model")
    from .runtime.generate import prefill_model_forward
    from .runtime.models.cache import make_prompt_cache

    call = prefill_model_forward(model)
    kind = "prefill_forward" if call is not model else "model_call"
    return PromptScoringBinding(lambda tokens, cache: call(tokens, cache=cache),
                                make_prompt_cache(model), kind)


def _cache_entries(value, depth=0):
    if depth > 4 or value is None:
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _cache_entries(item, depth + 1)
        return
    yield value
    inner = getattr(value, "caches", None)
    if isinstance(inner, (list, tuple)):
        for item in inner:
            yield from _cache_entries(item, depth + 1)


def scoring_cache_receipt(cache):
    """Label the private cache actually built; refuse a quantized one.

    A cache class that stores quantized state (``Quantized*`` or ``*Q8*``
    classes, or a ``bits`` field below 16) is approximate KV: the scored
    values would not be the exact reference, so the route is refused.
    """
    names = sorted({type(entry).__name__ for entry in _cache_entries(cache)})
    quantized = [
        type(entry).__name__ for entry in _cache_entries(cache)
        if "quantized" in type(entry).__name__.lower()
        or "q8" in type(entry).__name__.lower()
        or (isinstance(getattr(entry, "bits", None), int) and getattr(entry, "bits") < 16)
    ]
    if quantized:
        raise PromptLogprobsRefused(
            "the route builds a quantized KV cache; prompt scoring needs exact state",
            code="prompt_logprobs_unsupported_route", http_status=501,
            reason="route_not_exact", extra={"reasons": [f"quantized_kv_cache:{n}" for n in sorted(set(quantized))],
                                             "cache_classes": names},
        )
    return {"kind": "fresh_request_private", "classes": names}


def require_text_route(adapter):
    descriptor = getattr(adapter, "descriptor", None)
    if descriptor is not None:
        from .contracts import Capability

        if Capability.TEXT not in descriptor.capabilities:
            raise PromptLogprobsRefused(
                "the loaded adapter is not a text-generation route",
                code="prompt_logprobs_unsupported_route", http_status=501,
                reason="not_autoregressive_text",
            )


# --------------------------------------------------------------------------
# Chunked scorer


@dataclass
class PromptScores:
    """Per scored position ``i`` = 1..T-1 (index ``i - 1`` in each list)."""

    logprobs: list
    ranks: list
    top_ids: list
    top_logprobs: list
    chunks: int
    vocab: int


def chunk_log_softmax(logits, mx):
    """fp32 log-softmax of one ``[rows, vocab]`` chunk (the served normalization)."""
    logits = logits.astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def _nonfinite(where):
    return PromptLogprobsRefused(
        f"target forward produced a non-finite prompt log probability ({where})",
        code="prompt_logprobs_nonfinite", http_status=500, reason="nonfinite",
    )


class PromptScoringRun:
    """Resumable chunked scorer: ``step()`` runs one chunk, True when done.

    Row ``p`` predicts token ``p + 1``.  Every prompt token is fed, so a
    prompt that fits one chunk runs exactly the shape of a plain full-sequence
    forward (bf16 kernels are shape-sensitive at the last ulp); the final
    row's prediction is discarded.  A trailing chunk holding only the final
    token predicts nothing and is skipped.
    """

    def __init__(self, forward, cache, tokens, *, top_k, chunk_tokens, mx):
        self.forward = forward
        self.cache = cache
        self.tokens = [int(token) for token in tokens]
        self.top_k = int(top_k)
        self.chunk_tokens = int(chunk_tokens)
        self.mx = mx
        self.start = 0
        self.scores = PromptScores([], [], [], [], 0, 0)

    @property
    def done(self):
        total = len(self.tokens)
        return total < 2 or self.start >= total - 1

    def step(self):
        if self.done:
            self.release()
            return True
        mx, tokens, total = self.mx, self.tokens, len(self.tokens)
        start = self.start
        stop = min(start + self.chunk_tokens, total)
        rows = stop - start
        scored = rows - 1 if stop == total else rows
        logits = self.forward(mx.array(tokens[start:stop])[None], self.cache)
        if isinstance(logits, (tuple, list)):
            raise PromptLogprobsRefused(
                "target forward returned a tuple, not one logits array",
                code="prompt_logprobs_unsupported_route", http_status=501,
                reason="forward_returns_tuple",
            )
        if logits.ndim != 3 or int(logits.shape[0]) != 1 or int(logits.shape[1]) != rows:
            raise PromptLogprobsRefused(
                f"target forward returned logits shaped {tuple(logits.shape)} for a {rows}-row chunk",
                code="prompt_logprobs_unsupported_route", http_status=501,
                reason="forward_not_full_sequence",
            )
        normalized = chunk_log_softmax(logits[0, :scored], mx)
        targets = mx.array(tokens[start + 1:start + scored + 1])[:, None]
        actual = mx.take_along_axis(normalized, targets, axis=-1)
        ranks = mx.sum(normalized > actual, axis=-1) + 1
        outputs = [actual[:, 0], ranks]
        if self.top_k:
            count = min(self.top_k, int(normalized.shape[-1]))
            ids = mx.argpartition(-normalized, kth=count - 1, axis=-1)[:, :count]
            outputs += [ids, mx.take_along_axis(normalized, ids, axis=-1)]
        mx.eval(*outputs)
        values = outputs[0].tolist()
        if any(math.isnan(v) or v == math.inf for v in values):
            raise _nonfinite("actual token")
        scores = self.scores
        scores.logprobs.extend(values)
        scores.ranks.extend(int(v) for v in outputs[1].tolist())
        if self.top_k:
            for row_ids, row_values in zip(outputs[2].tolist(), outputs[3].tolist()):
                if any(math.isnan(v) or v == math.inf for v in row_values):
                    raise _nonfinite("top-k alternative")
                pairs = sorted(zip(row_ids, row_values), key=lambda item: (-item[1], item[0]))
                scores.top_ids.append([int(i) for i, _ in pairs])
                scores.top_logprobs.append([float(v) for _, v in pairs])
        else:
            scores.top_ids.extend([] for _ in range(scored))
            scores.top_logprobs.extend([] for _ in range(scored))
        scores.chunks += 1
        scores.vocab = int(normalized.shape[-1])
        self.start = stop if stop < total else total
        del logits, normalized, outputs
        if self.done:
            self.release()
            return True
        return False

    def release(self):
        self.cache = None
        self.forward = None


def score_prompt_tokens(forward, cache, tokens, *, top_k, chunk_tokens, mx):
    """Score ``tokens`` to completion (tests and direct callers)."""
    run = PromptScoringRun(forward, cache, tokens, top_k=top_k, chunk_tokens=chunk_tokens, mx=mx)
    while not run.step():
        pass
    return run.scores


def scoring_receipt(run, binding_kind, *, chunk_tokens, top_k, fidelity, int8, seconds):
    return {
        "schema": SCHEMA,
        "status": "completed",
        "implemented": True,
        "qualified": False,
        "selected": True,
        "observed_used": run.scores.chunks > 0,
        "route": "ordinary_target_forward",
        "forward": binding_kind,
        "lane_width": 1,
        "batching": "worker_interleaved_private_cache",
        "apc": {"read": "bypassed", "write": "none", "cached_tokens": 0},
        "speculation": "not_used",
        "draft_or_residual_probabilities": False,
        "distribution": "model_log_softmax_fp32_pre_processor",
        "numerics": fidelity,
        "int8_prefill": int8,
        "prompt_tokens": len(run.tokens),
        "scored_positions": len(run.scores.logprobs),
        "top_k": top_k,
        "chunk_tokens": chunk_tokens,
        "chunks": run.scores.chunks,
        "vocab": run.scores.vocab,
        "scoring_seconds": seconds,
    }


def run_prompt_scoring(adapter, tokens, *, top_k, chunk_tokens, mx=None):
    """Bind the adapter's forward and score to completion; ``(scores, receipt)``."""
    if mx is None:
        import mlx.core as mx
    binding = prompt_scoring_binding(adapter)
    started = time.monotonic()
    run = PromptScoringRun(binding.forward, binding.cache, tokens,
                           top_k=top_k, chunk_tokens=chunk_tokens, mx=mx)
    binding.cache = None
    while not run.step():
        pass
    return run.scores, scoring_receipt(
        run, binding.kind, chunk_tokens=chunk_tokens, top_k=top_k,
        fidelity=None, int8={"active": False}, seconds=time.monotonic() - started,
    )


@dataclass
class PromptScoringTask:
    """One scoring request handed to the serving worker."""

    request: dict
    top_k: int
    policy: PromptLogprobsPolicy
    done: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    created: float = field(default_factory=time.monotonic)
    result: object = None
    error: BaseException | None = None
    run: object = None
    started: bool = False  # claimed by the worker (set under the engine lock)
    prompt_ids: list | None = None
    state: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Response shapes


class TokenText:
    """Memoized per-token text and bytes for one tokenizer (one response)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._text = {}
        self._bytes = {}
        special = set()
        for name in ("all_special_ids",):
            try:
                special.update(int(i) for i in getattr(tokenizer, name, ()) or ())
            except (TypeError, ValueError):
                pass
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            value = getattr(tokenizer, name, None)
            if isinstance(value, int) and not isinstance(value, bool):
                special.add(value)
        self.special = frozenset(special)

    def text(self, token_id):
        token_id = int(token_id)
        value = self._text.get(token_id)
        if value is None:
            value = self.tokenizer.decode([token_id], skip_special_tokens=False)
            self._text[token_id] = value
        return value

    def bytes(self, token_id):
        token_id = int(token_id)
        value = self._bytes.get(token_id)
        if value is None:
            try:
                from .structured_automaton import token_bytes_value

                value = token_bytes_value(self.tokenizer, token_id)
            except (AttributeError, TypeError, ValueError):
                value = self.text(token_id).encode("utf-8")
            self._bytes[token_id] = value
        return value


class EchoText:
    """Each token's contribution to the echoed text, and its character offset.

    Special tokens (BOS, EOS, chat markers) have zero width: their entry is
    their literal spelling but they add nothing to the text.  Other tokens
    contribute the characters their bytes complete through one incremental
    UTF-8 decoder, so a byte-level token that splits a multibyte character
    contributes ``""`` and the token that completes it carries the whole
    character.  Joining the non-special entries reproduces the text.
    """

    def __init__(self, texts):
        self.texts = texts
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.position = 0

    def add(self, token_id):
        offset = self.position
        if int(token_id) in self.texts.special:
            return self.texts.text(token_id), offset
        piece = self.decoder.decode(self.texts.bytes(token_id))
        self.position += len(piece)
        return piece, offset


def vllm_prompt_logprobs(tokenizer, tokens, scores, top, *, texts=None):
    """vLLM shape: ``[None, {"<id>": {logprob, rank, decoded_token}}, ...]``."""
    texts = texts or TokenText(tokenizer)
    result = [None]
    for index, token in enumerate(tokens[1:]):
        row = {
            str(int(token)): {
                "logprob": max(float(scores.logprobs[index]), -9999.0),
                "rank": int(scores.ranks[index]),
                "decoded_token": texts.text(token),
            }
        }
        for rank, (alt, value) in enumerate(
            zip(scores.top_ids[index][:top], scores.top_logprobs[index][:top]), start=1
        ):
            row.setdefault(str(int(alt)), {
                "logprob": max(float(value), -9999.0),
                "rank": rank,
                "decoded_token": texts.text(alt),
            })
        result.append(row)
    return result


def legacy_echo_logprobs(tokenizer, tokens, scores, top, generated=(), *, texts=None):
    """OpenAI legacy Completions ``logprobs`` covering the echoed prompt.

    ``generated`` are the existing generated-token entries (``{"id",
    "logprob", "top_logprobs": [...]}``) appended after the prompt, so one
    object covers the whole echoed text as the legacy contract does.  Offsets
    continue from the prompt tokens' own text, never from ``len(prompt)``.
    """
    texts = texts or TokenText(tokenizer)
    echo = EchoText(texts)
    out = {"tokens": [], "token_ids": [], "token_logprobs": [None], "top_logprobs": [None],
           "text_offset": []}
    for token in tokens:
        piece, offset = echo.add(token)
        out["tokens"].append(piece)
        out["token_ids"].append(int(token))
        out["text_offset"].append(offset)
    out["token_logprobs"].extend(max(float(v), -9999.0) for v in scores.logprobs)
    for index in range(len(tokens) - 1):
        alternatives = {}
        for alt, value in zip(scores.top_ids[index][:top], scores.top_logprobs[index][:top]):
            alternatives.setdefault(texts.text(alt), max(float(value), -9999.0))
        out["top_logprobs"].append(alternatives)
    for entry in generated:
        piece, offset = echo.add(entry["id"])
        out["tokens"].append(piece)
        out["token_ids"].append(int(entry["id"]))
        out["token_logprobs"].append(entry["logprob"])
        alternatives = {}
        for alt in list(entry.get("top_logprobs", ()))[:top]:
            alternatives.setdefault(texts.text(alt["id"]), alt["logprob"])
        out["top_logprobs"].append(alternatives)
        out["text_offset"].append(offset)
    return out
