"""Measured fidelity of approximate KV operations, and the selection gate.

Two halves:

* **Measurement.** :func:`measure_teacher_forced` runs one token stream through
  two caches of the same model: the exact cache and a cache quantized by the
  adapter-declared :class:`~.approximate_kv.KVQuantizationOperation` (the
  serving code path, not a re-implementation).  After a chunked prefill of the
  context it teacher-forces a scored window one token at a time and records,
  per position, ``KL(p_exact || p_quant)``, top-1 agreement, top-5 overlap and
  the exact token's log-prob delta.  Teacher forcing keeps positions
  comparable: it measures per-step distributional damage, not a divergence
  cascade.  Every arm carries a mechanism count (quantized attention planes);
  a quantized arm with 0 planes, or an exact arm with any, is refused.

* **Gate.** :func:`evaluate_fidelity_report` decides whether a measured report
  qualifies an operation for selection.  Thresholds are data
  (:data:`DEFAULT_THRESHOLDS`); a report that is missing contexts, measured on
  the wrong device, bound to another adapter or with a zero mechanism count
  fails closed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

REPORT_SCHEMA = "mlx2.kv-quant-fidelity.v1"
BUNDLE_SCHEMA = "mlx2.kv-quant-fidelity-bundle.v1"


# ----------------------------------------------------------------- measurement


def token_metrics(exact_logits, quant_logits, target=None):
    """Per-position fidelity metrics for ``[..., V]`` logits (float32 math)."""
    import mlx.core as mx

    exact = exact_logits.astype(mx.float32)
    quant = quant_logits.astype(mx.float32)
    log_p = exact - mx.logsumexp(exact, axis=-1, keepdims=True)
    log_q = quant - mx.logsumexp(quant, axis=-1, keepdims=True)
    kl = mx.maximum(mx.sum(mx.exp(log_p) * (log_p - log_q), axis=-1), 0.0)
    agree = mx.argmax(exact, axis=-1) == mx.argmax(quant, axis=-1)
    k = min(5, exact.shape[-1])
    top_p = mx.argpartition(-exact, k - 1, axis=-1)[..., :k]
    top_q = mx.argpartition(-quant, k - 1, axis=-1)[..., :k]
    overlap = mx.sum(
        mx.any(top_p[..., :, None] == top_q[..., None, :], axis=-1), axis=-1
    ) / float(k)
    result = {"kl": kl, "agree": agree, "top5_overlap": overlap}
    if target is not None:
        index = mx.expand_dims(target, -1)
        result["logprob_delta"] = mx.squeeze(
            mx.take_along_axis(log_q, index, axis=-1)
            - mx.take_along_axis(log_p, index, axis=-1),
            -1,
        )
    return result


def attention_plane_bytes(planes) -> int:
    """Bytes held by attention planes (the ones KV quantization touches)."""
    from .approximate_kv import _leaves

    total = 0
    for leaf in _leaves(list(planes or ())):
        if hasattr(leaf, "to_quantized") or hasattr(leaf, "key_bits"):
            total += int(getattr(leaf, "nbytes", 0) or 0)
    return total


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(per_token: Mapping[str, Sequence[float]]) -> dict:
    kl = list(per_token["kl"])
    n = len(kl)
    summary = {
        "scored_tokens": n,
        "kl_mean": sum(kl) / n if n else float("nan"),
        "kl_p99": _percentile(kl, 0.99),
        "kl_max": max(kl) if kl else float("nan"),
        "top1_agreement": sum(per_token["agree"]) / n if n else float("nan"),
        "top5_overlap": sum(per_token["top5_overlap"]) / n if n else float("nan"),
    }
    if per_token.get("logprob_delta"):
        deltas = list(per_token["logprob_delta"])
        summary["logprob_delta_mean"] = sum(deltas) / len(deltas)
    return summary


def measure_teacher_forced(
    model,
    tokens: Sequence[int],
    *,
    context: int,
    score_tokens: int,
    operation,
    prefill_step: int = 2048,
    make_cache=None,
) -> dict:
    """Teacher-forced exact-vs-quantized comparison over one token stream.

    ``tokens`` must hold at least ``context + score_tokens`` ids.  Positions
    ``[0, context)`` are prefilled in ``prefill_step`` chunks on both caches;
    positions ``context .. context + score_tokens - 1`` are then fed one at a
    time and the next-token distributions compared (the target for the
    log-prob delta is the next token of the stream).
    """
    import mlx.core as mx

    from .approximate_kv import LaneKVState, quantized_plane_count

    if context < 1 or score_tokens < 1:
        raise ValueError("context and score_tokens must be positive")
    if len(tokens) < context + score_tokens + 1:
        raise ValueError("token stream shorter than context + score_tokens + 1")
    if make_cache is None:
        from .models.cache import make_prompt_cache

        make_cache = lambda: make_prompt_cache(model)  # noqa: E731
    exact = list(make_cache())
    staged = operation.apply(LaneKVState(operation.revision, tuple(make_cache())))
    quant = list(staged.planes)
    quantized_planes = quantized_plane_count(quant)
    exact_planes = quantized_plane_count(exact)
    if quantized_planes == 0:
        raise RuntimeError("quantized arm has no quantized plane; refusing to report")
    if exact_planes:
        raise RuntimeError("exact arm carries quantized planes; refusing to report")

    stream = mx.array([list(tokens)], dtype=mx.uint32)
    for start in range(0, context, prefill_step):
        chunk = stream[:, start : min(start + prefill_step, context)]
        mx.eval(model(chunk, cache=exact), model(chunk, cache=quant))
    per_token = {"kl": [], "agree": [], "top5_overlap": [], "logprob_delta": []}
    for position in range(context, context + score_tokens):
        step = stream[:, position : position + 1]
        exact_logits = model(step, cache=exact)[0, -1]
        quant_logits = model(step, cache=quant)[0, -1]
        metrics = token_metrics(exact_logits, quant_logits, stream[0, position + 1])
        mx.eval(metrics)
        per_token["kl"].append(float(metrics["kl"].item()))
        per_token["agree"].append(int(metrics["agree"].item()))
        per_token["top5_overlap"].append(float(metrics["top5_overlap"].item()))
        per_token["logprob_delta"].append(float(metrics["logprob_delta"].item()))
    total = context + score_tokens
    exact_bytes = attention_plane_bytes(exact)
    quant_bytes = attention_plane_bytes(quant)
    return {
        "context": int(context),
        **summarize(per_token),
        "quantized_planes": int(quantized_planes),
        "exact_quantized_planes": int(exact_planes),
        # Allocated bytes include the cache's 256-token growth step; the
        # ratio is what the gate uses, per-token bytes are informational.
        "kv_bytes_exact": exact_bytes,
        "kv_bytes_quant": quant_bytes,
        "kv_bytes_per_token_exact": exact_bytes / total,
        "kv_bytes_per_token_quant": quant_bytes / total,
        "kv_bytes_ratio": (quant_bytes / exact_bytes) if exact_bytes else float("nan"),
    }


# ------------------------------------------------------------------------ gate


@dataclass(frozen=True)
class KVQuantFidelityThresholds:
    kl_mean_max: float
    kl_p99_max: float
    top1_agreement_min: float
    bytes_ratio_max: float
    # Needles the quantized arm may miss that the exact arm retrieved.
    needle_losses_max: int = 0
    min_context: int = 4096
    required_contexts: tuple[int, ...] = (4096, 16384, 32768)
    min_scored_tokens: int = 128
    devices: tuple[str, ...] = ("gpu",)

    def as_dict(self) -> dict:
        return {
            "kl_mean_max": self.kl_mean_max,
            "kl_p99_max": self.kl_p99_max,
            "top1_agreement_min": self.top1_agreement_min,
            "bytes_ratio_max": self.bytes_ratio_max,
            "needle_losses_max": self.needle_losses_max,
            "min_context": self.min_context,
            "required_contexts": list(self.required_contexts),
            "min_scored_tokens": self.min_scored_tokens,
            "devices": list(self.devices),
        }


# Priors from peer practice (llama.cpp q8_0 KV is treated as lossless in
# practice; K8/V4 is the commonly accepted aggressive setting).  To be
# confirmed by the first GPU runs; see docs/SERVING.md.
DEFAULT_THRESHOLDS = {
    "kv_q8": KVQuantFidelityThresholds(
        kl_mean_max=0.005,
        kl_p99_max=0.05,
        top1_agreement_min=0.99,
        bytes_ratio_max=0.55,
        needle_losses_max=0,
    ),
    "kv_k8v4": KVQuantFidelityThresholds(
        kl_mean_max=0.02,
        kl_p99_max=0.2,
        top1_agreement_min=0.97,
        bytes_ratio_max=0.42,
        needle_losses_max=1,
    ),
}


def evaluate_fidelity_report(
    report: Mapping,
    *,
    operation: str,
    adapter_fingerprint: str | None = None,
    thresholds: KVQuantFidelityThresholds | None = None,
) -> dict:
    """Return ``{"passed": bool, "failures": [...]}``; never raises on bad data."""
    failures: list[str] = []
    limits = thresholds or DEFAULT_THRESHOLDS.get(operation)
    if limits is None:
        return {"passed": False, "failures": [f"no thresholds for {operation!r}"]}
    if isinstance(report, Mapping) and report.get("schema") == BUNDLE_SCHEMA:
        # One harness run measures several operations; judge the selected one.
        report = (report.get("reports") or {}).get(operation)
    if not isinstance(report, Mapping) or report.get("schema") != REPORT_SCHEMA:
        return {"passed": False, "failures": ["not a kv-quant fidelity report"]}
    if report.get("operation") != operation:
        failures.append(
            f"report measures {report.get('operation')!r}, route selects {operation!r}"
        )
    if adapter_fingerprint is not None and (
        report.get("adapter_fingerprint") != adapter_fingerprint
    ):
        failures.append("report is bound to another adapter artifact")
    if report.get("device") not in limits.devices:
        failures.append(f"report device {report.get('device')!r} not accepted")
    contexts = [c for c in report.get("contexts") or () if isinstance(c, Mapping)]
    measured = {int(c.get("context", 0)) for c in contexts}
    for needed in limits.required_contexts:
        if needed not in measured:
            failures.append(f"missing context {needed}")
    gated = [c for c in contexts if int(c.get("context", 0)) >= limits.min_context]
    if not gated:
        failures.append(f"no measured context >= {limits.min_context}")

    def number(entry, key):
        value = entry.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return None if math.isnan(value) else float(value)

    for entry in gated:
        where = f"context {entry.get('context')}"
        if not entry.get("quantized_planes"):
            failures.append(f"{where}: quantized arm mechanism count is 0")
        if entry.get("exact_quantized_planes"):
            failures.append(f"{where}: exact arm is not exact")
        scored = number(entry, "scored_tokens")
        if scored is None or scored < limits.min_scored_tokens:
            failures.append(f"{where}: fewer than {limits.min_scored_tokens} scored tokens")
        checks = (
            ("kl_mean", "<=", limits.kl_mean_max),
            ("kl_p99", "<=", limits.kl_p99_max),
            ("top1_agreement", ">=", limits.top1_agreement_min),
            ("kv_bytes_ratio", "<=", limits.bytes_ratio_max),
        )
        for key, op, bound in checks:
            value = number(entry, key)
            if value is None:
                failures.append(f"{where}: {key} missing")
            elif (op == "<=" and value > bound) or (op == ">=" and value < bound):
                failures.append(f"{where}: {key} {value:.6g} not {op} {bound}")
        needles = entry.get("needles")
        if needles is not None:
            losses = number(needles, "quant_losses")
            if losses is None:
                failures.append(f"{where}: needle losses missing")
            elif losses > limits.needle_losses_max:
                failures.append(
                    f"{where}: {int(losses)} needle losses > {limits.needle_losses_max}"
                )
            # Losses count needles the exact arm found and the quantized arm
            # missed; if the exact arm found none, 0 losses proves nothing.
            total = number(needles, "total")
            hits = number(needles, "exact_hits")
            if total and not hits:
                failures.append(f"{where}: needle check vacuous (exact arm retrieved 0)")
    return {
        "passed": not failures,
        "failures": failures,
        "operation": operation,
        "thresholds": limits.as_dict(),
    }
