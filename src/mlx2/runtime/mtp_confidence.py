"""Per-position draft confidence for cost-aware self-MTP depth (DSpark-style).

Pieces, all host-side except :func:`draft_position_features`:

* :class:`DraftConfidenceProbe`: carried by a :class:`SelfMTPLane`.  It asks
  the batched self-MTP round to compute per-draft-position features on device
  and fetch them in the same ``mx.eval`` as the accept boundary (no extra
  synchronisation).  Greedy cycles may also draft ``lookahead`` unverified
  positions whose confidences fill the censored tail.
* Confidence models: :class:`Top1ProbConfidence`, the zero-training baseline,
  and :class:`LogisticConfidence`, which is trained offline by
  :func:`fit_logistic_confidence` from logged acceptance records.  Both are
  calibrated with sequential temperature scaling
  (:func:`fit_sequential_temperatures`).
* :class:`MTPAcceptanceLogger`: a bounded, default-off JSONL data-collection
  hook, plus :func:`load_acceptance_log` to read it back.

**What is deliberately absent.**  The DSpark-style *scheduler* that turned
these confidences into a per-cycle depth was measured on GPU (rm10,
`qualification/runs/mtp-confidence-20260919/`) and is a no-go: the confidence
arms ran 14.0/5.7/3.2% below fixed depth on 27B at 1/2/4 lanes, and
`scripts/best_constant_depth.py` finds the adapter cap optimal at every width
on 27B and 35B-A3B, 7-15% ahead of depth 2.  Separability was never the
problem -- holdout top-1 AUC is 0.93-0.94 at the first drafted position -- so
what remains here is the *measurement* apparatus: features, calibrated
confidence models, the offline fit and the log.  There is no selection path
and no cost profile on this tree; anything that picks a depth from these
numbers is an offline tool working from measured inputs.

Source: DSpark, arXiv 2607.05147 (confidence head and STS calibration).  This
is a clean-room adaptation; see ``docs/PROVENANCE.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

_COUNTER_MAX = (1 << 63) - 1
FEATURE_NAMES = ("top1_prob", "entropy", "margin")
CONFIDENCE_SCHEMA = "mlx2.mtp_confidence.v1"
LOG_SCHEMA = "mlx2.mtp_acceptance_log.v1"
_EPS = 1e-6


def _bump(counters: dict[str, int], key: str, amount: int = 1) -> None:
    counters[key] = min(_COUNTER_MAX, int(counters.get(key, 0)) + int(amount))


def _sigmoid(z):
    z = np.clip(np.asarray(z, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(p) - np.log1p(-p)


def _bucket(token: int, buckets: int, salt: int) -> int:
    return int((int(token) * 2654435761 + salt * 40503) % (1 << 32)) % buckets


# --------------------------------------------------------------------------
# Device probe
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DraftConfidenceProbe:
    """Per-lane request for draft features.  ``projection`` is ``(hidden, r)``."""

    lookahead: int = 0
    projection: Any = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.lookahead, bool)
            or not isinstance(self.lookahead, int)
            or not 0 <= self.lookahead <= 16
        ):
            raise ValueError("confidence lookahead must be an integer in [0, 16]")

    @property
    def width(self) -> int:
        extra = 0 if self.projection is None else int(self.projection.shape[-1])
        return len(FEATURE_NAMES) + extra


def draft_position_features(logprobs, hidden=None, projection=None):
    """Return a float32 device vector of draft features for one position.

    ``logprobs`` is the lane's (constrained, temperature-transformed) draft
    log-distribution over the vocabulary.  Only V-sized reductions are
    performed; nothing is synchronised here.
    """
    import mlx.core as mx

    lp = logprobs.astype(mx.float32).reshape(-1)
    top2 = mx.topk(lp, 2) if lp.shape[0] >= 2 else mx.concatenate([lp, lp])
    top1 = mx.max(top2)
    second = mx.min(top2)
    probs = mx.exp(lp)
    finite = mx.where(mx.isfinite(lp), lp, mx.zeros_like(lp))
    entropy = -mx.sum(probs * finite)
    parts = [mx.exp(top1).reshape(1), entropy.reshape(1), (top1 - second).reshape(1)]
    if projection is not None and hidden is not None:
        h = hidden.astype(mx.float32).reshape(1, -1)
        parts.append((h @ projection.astype(mx.float32)).reshape(-1))
    return mx.concatenate(parts)


def random_hidden_projection(hidden_size: int, rank: int, seed: int = 0):
    """Return a fixed Gaussian sketch ``(hidden, rank)`` scaled by 1/sqrt(hidden)."""
    rng = np.random.default_rng(int(seed))
    return (rng.standard_normal((int(hidden_size), int(rank))) / math.sqrt(hidden_size)).astype(np.float32)


# --------------------------------------------------------------------------
# Per-cycle host rows
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DraftConfidenceRow:
    """Host copy of one lane's drafted positions in one cycle.

    ``features[j]`` describes drafted position ``j + 1``.  The first
    ``verify_depth`` positions were verified; later positions are lookahead.
    Labels follow the conditional semantics of ``c_k``: positions before the
    first rejection are 1, the first rejected position is 0, and every later
    position (and every lookahead position) is censored.
    """

    uid: int
    features: tuple[tuple[float, ...], ...]
    tokens: tuple[int, ...]
    prev_token: int
    verify_depth: int
    accepted: int
    relaxed: int = 0

    def labels(self) -> list[Optional[int]]:
        out: list[Optional[int]] = []
        for index in range(len(self.features)):
            if index >= self.verify_depth:
                out.append(None)
            elif index < self.accepted:
                out.append(1)
            elif index == self.accepted:
                out.append(0)
            else:
                out.append(None)
        return out

    def prev_tokens(self) -> list[int]:
        return [int(self.prev_token)] + [int(t) for t in self.tokens[:-1]]


# --------------------------------------------------------------------------
# Confidence models
# --------------------------------------------------------------------------


def _temperatures_array(temperatures, positions: int) -> np.ndarray:
    temps = np.ones(positions, dtype=np.float64)
    if temperatures:
        values = np.asarray(temperatures, dtype=np.float64)
        n = min(len(values), positions)
        temps[:n] = values[:n]
        if positions > len(values):
            temps[len(values):] = values[-1]
    return temps


class ConfidenceModel:
    kind = "abstract"

    def logits(self, row: DraftConfidenceRow) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    temperatures: tuple[float, ...] = ()

    def predict(self, row: DraftConfidenceRow) -> np.ndarray:
        z = self.logits(row)
        if not len(z):
            return np.zeros(0)
        return _sigmoid(z / _temperatures_array(self.temperatures, len(z)))

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "sha256": self.sha256()}

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass
class Top1ProbConfidence(ConfidenceModel):
    """Zero-training baseline: ``c_k = sigmoid(logit(top1_prob_k) / T_k)``."""

    temperatures: tuple[float, ...] = ()
    kind = "top1"

    def logits(self, row: DraftConfidenceRow) -> np.ndarray:
        return _logit([float(f[0]) for f in row.features])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONFIDENCE_SCHEMA,
            "kind": self.kind,
            "temperatures": [float(t) for t in self.temperatures],
        }


def _dense_features(features: Sequence[Sequence[float]]) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < len(FEATURE_NAMES):
        raise ValueError("confidence features must be (positions, >=3)")
    top1 = np.clip(x[:, 0], _EPS, 1.0 - _EPS)
    base = np.stack([top1, np.log(top1), x[:, 1], x[:, 2]], axis=1)
    return np.concatenate([base, x[:, len(FEATURE_NAMES):]], axis=1)


@dataclass
class LogisticConfidence(ConfidenceModel):
    """``c_k = sigmoid((b_k + w.phi + u[prev] + v[token]) / T_k)``.

    ``phi = [top1, log top1, entropy, margin, P^T h]``.  ``u`` and ``v`` are
    hashed rank-1 tables standing in for DSpark's ``W1[x_{k-1}]`` Markov term.
    """

    weights: tuple[float, ...]
    position_bias: tuple[float, ...]
    prev_bucket_bias: tuple[float, ...] = ()
    token_bucket_bias: tuple[float, ...] = ()
    temperatures: tuple[float, ...] = ()
    feature_mean: tuple[float, ...] = ()
    feature_scale: tuple[float, ...] = ()
    projection_rank: int = 0
    kind = "logistic"

    def _phi(self, features) -> np.ndarray:
        phi = _dense_features(features)
        if phi.shape[1] != len(self.weights):
            raise ValueError(
                f"confidence model expects {len(self.weights)} features, got {phi.shape[1]}"
            )
        if self.feature_mean:
            phi = (phi - np.asarray(self.feature_mean)) / np.asarray(self.feature_scale)
        return phi

    def logits(self, row: DraftConfidenceRow) -> np.ndarray:
        if not row.features:
            return np.zeros(0)
        phi = self._phi(row.features)
        z = phi @ np.asarray(self.weights, dtype=np.float64)
        bias = np.asarray(self.position_bias, dtype=np.float64)
        for index in range(len(z)):
            z[index] += bias[min(index, len(bias) - 1)]
        if self.prev_bucket_bias:
            table, n = self.prev_bucket_bias, len(self.prev_bucket_bias)
            for index, prev in enumerate(row.prev_tokens()):
                z[index] += table[_bucket(prev, n, 1)]
        if self.token_bucket_bias:
            table, n = self.token_bucket_bias, len(self.token_bucket_bias)
            for index, token in enumerate(row.tokens):
                z[index] += table[_bucket(token, n, 2)]
        return z

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONFIDENCE_SCHEMA,
            "kind": self.kind,
            "weights": [float(v) for v in self.weights],
            "position_bias": [float(v) for v in self.position_bias],
            "prev_bucket_bias": [float(v) for v in self.prev_bucket_bias],
            "token_bucket_bias": [float(v) for v in self.token_bucket_bias],
            "temperatures": [float(v) for v in self.temperatures],
            "feature_mean": [float(v) for v in self.feature_mean],
            "feature_scale": [float(v) for v in self.feature_scale],
            "projection_rank": int(self.projection_rank),
        }


def confidence_model_from_dict(data: Mapping[str, Any]) -> ConfidenceModel:
    if data.get("schema") != CONFIDENCE_SCHEMA:
        raise ValueError("unsupported confidence model schema")
    kind = data.get("kind")
    temps = tuple(float(v) for v in data.get("temperatures", ()))
    if any(not math.isfinite(t) or t <= 0 for t in temps):
        raise ValueError("confidence temperatures must be finite and positive")
    if kind == "top1":
        return Top1ProbConfidence(temperatures=temps)
    if kind == "logistic":
        model = LogisticConfidence(
            weights=tuple(float(v) for v in data["weights"]),
            position_bias=tuple(float(v) for v in data["position_bias"]),
            prev_bucket_bias=tuple(float(v) for v in data.get("prev_bucket_bias", ())),
            token_bucket_bias=tuple(float(v) for v in data.get("token_bucket_bias", ())),
            temperatures=temps,
            feature_mean=tuple(float(v) for v in data.get("feature_mean", ())),
            feature_scale=tuple(float(v) for v in data.get("feature_scale", ())),
            projection_rank=int(data.get("projection_rank", 0)),
        )
        if not model.position_bias:
            raise ValueError("logistic confidence needs position_bias")
        values = (
            model.weights + model.position_bias + model.prev_bucket_bias
            + model.token_bucket_bias + model.feature_mean + model.feature_scale
        )
        if not all(math.isfinite(v) for v in values):
            raise ValueError("confidence model parameters must be finite")
        return model
    raise ValueError(f"unknown confidence model kind {kind!r}")


def load_confidence_model(value: Any) -> ConfidenceModel:
    """Resolve ``"top1"``, a mapping, a model, or a JSON path."""
    if value is None or value == "top1":
        return Top1ProbConfidence()
    if isinstance(value, ConfidenceModel):
        return value
    if isinstance(value, Mapping):
        return confidence_model_from_dict(value)
    if isinstance(value, (str, os.PathLike)):
        return confidence_model_from_dict(json.loads(Path(value).read_text()))
    raise ValueError("confidence_model must be 'top1', a mapping, or a JSON path")


# --------------------------------------------------------------------------
# Calibration and training (offline, numpy)
# --------------------------------------------------------------------------


def expected_calibration_error(pred, labels, bins: int = 10) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if not len(pred):
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(pred, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            total += mask.sum() * abs(pred[mask].mean() - labels[mask].mean())
    return float(total / len(pred))


def roc_auc(scores, labels) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos, neg = labels.sum(), len(labels) - labels.sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def _survival_targets(rows: Sequence[DraftConfidenceRow], positions: int):
    """Per position k (1-based): rows whose verify depth reached k, with labels."""
    out = []
    for k in range(1, positions + 1):
        members = [r for r in rows if r.verify_depth >= k and len(r.features) >= k]
        out.append((members, [1 if r.accepted >= k else 0 for r in members]))
    return out


def fit_sequential_temperatures(
    model: ConfidenceModel,
    rows: Sequence[DraftConfidenceRow],
    positions: int,
    grid: Optional[Sequence[float]] = None,
    bins: int = 10,
) -> tuple[float, ...]:
    """DSpark STS: per position, a 1-D grid search on ``T_k`` minimising the
    ECE of the *survival* ``prod_{i<=k} c_i``, with earlier ``T`` held fixed.
    The mean NLL breaks ties.
    """
    grid = np.exp(np.linspace(math.log(0.25), math.log(4.0), 49)) if grid is None else np.asarray(grid)
    raw = {id(r): model.logits(r) for r in rows}
    temps: list[float] = []
    for k, (members, labels) in enumerate(_survival_targets(rows, positions), start=1):
        if len(members) < 8:
            temps.append(temps[-1] if temps else 1.0)
            continue
        labels_a = np.asarray(labels, dtype=np.float64)
        prefix = np.ones(len(members))
        for i in range(k - 1):
            prefix *= _sigmoid(np.asarray([raw[id(r)][i] for r in members]) / temps[i])
        zk = np.asarray([raw[id(r)][k - 1] for r in members])
        best, best_t = None, 1.0
        for t in grid:
            survival = np.clip(prefix * _sigmoid(zk / t), _EPS, 1 - _EPS)
            ece = expected_calibration_error(survival, labels_a, bins)
            nll = -np.mean(labels_a * np.log(survival) + (1 - labels_a) * np.log(1 - survival))
            score = (round(ece, 4), nll)
            if best is None or score < best:
                best, best_t = score, float(t)
        temps.append(best_t)
    return tuple(temps)


def _labelled(rows: Sequence[DraftConfidenceRow]):
    feats, pos, prev, tok, y = [], [], [], [], []
    for row in rows:
        prevs = row.prev_tokens()
        for index, label in enumerate(row.labels()):
            if label is None:
                continue
            feats.append(row.features[index])
            pos.append(index)
            prev.append(prevs[index])
            tok.append(row.tokens[index])
            y.append(label)
    return feats, np.asarray(pos), np.asarray(prev), np.asarray(tok), np.asarray(y, dtype=np.float64)


def fit_logistic_confidence(
    rows: Sequence[DraftConfidenceRow],
    *,
    positions: int,
    l2: float = 1e-2,
    buckets: int = 1024,
    bucket_l2: float = 4.0,
    iterations: int = 8,
    calibrate: bool = True,
) -> LogisticConfidence:
    """Fit by alternating a dense Newton step (features and position one-hot)
    with per-bucket Newton steps for the hashed token tables.  Only uncensored
    positions are used.
    """
    feats, pos, prev, tok, y = _labelled(rows)
    if len(y) < 16 or y.min() == y.max():
        raise ValueError("need at least 16 labelled positions with both outcomes")
    phi = _dense_features(feats)
    mean, scale = phi.mean(axis=0), phi.std(axis=0) + 1e-6
    phi = (phi - mean) / scale
    onehot = np.zeros((len(y), positions))
    onehot[np.arange(len(y)), np.minimum(pos, positions - 1)] = 1.0
    x = np.concatenate([phi, onehot], axis=1)
    dim = x.shape[1]
    theta = np.zeros(dim)
    prev_idx = np.asarray([_bucket(t, buckets, 1) for t in prev]) if buckets else None
    tok_idx = np.asarray([_bucket(t, buckets, 2) for t in tok]) if buckets else None
    u = np.zeros(buckets)
    v = np.zeros(buckets)
    reg = np.full(dim, l2)
    reg[phi.shape[1]:] = l2 * 1e-2  # position intercepts: weak prior
    for _ in range(iterations):
        offset = (u[prev_idx] + v[tok_idx]) if buckets else 0.0
        for _newton in range(4):
            p = _sigmoid(x @ theta + offset)
            grad = x.T @ (p - y) + reg * theta
            hess = (x * (p * (1 - p))[:, None]).T @ x + np.diag(reg)
            theta -= np.linalg.solve(hess, grad)
        if buckets:
            base = x @ theta
            for table, idx, other in ((u, prev_idx, lambda: v[tok_idx]), (v, tok_idx, lambda: u[prev_idx])):
                p = _sigmoid(base + table[idx] + other())
                g = np.bincount(idx, weights=y - p, minlength=buckets) - bucket_l2 * table
                h = np.bincount(idx, weights=p * (1 - p), minlength=buckets) + bucket_l2
                table += g / h
    model = LogisticConfidence(
        weights=tuple(float(v_) for v_ in theta[: phi.shape[1]]),
        position_bias=tuple(float(v_) for v_ in theta[phi.shape[1]:]),
        prev_bucket_bias=tuple(float(v_) for v_ in u) if buckets else (),
        token_bucket_bias=tuple(float(v_) for v_ in v) if buckets else (),
        feature_mean=tuple(float(v_) for v_ in mean),
        feature_scale=tuple(float(v_) for v_ in scale),
        projection_rank=int(phi.shape[1] - 4),
    )
    if calibrate:
        model.temperatures = fit_sequential_temperatures(model, rows, positions)
    return model


def evaluate_confidence(
    model: ConfidenceModel, rows: Sequence[DraftConfidenceRow], positions: int
) -> dict[str, Any]:
    """ECE/AUC per position (conditional), plus the expected-length error."""
    report: dict[str, Any] = {"model": model.describe(), "positions": []}
    preds = {id(r): model.predict(r) for r in rows}
    for k in range(positions):
        p, y = [], []
        for r in rows:
            labels = r.labels()
            if k < len(labels) and labels[k] is not None:
                p.append(preds[id(r)][k])
                y.append(labels[k])
        report["positions"].append(
            {
                "position": k + 1,
                "n": len(y),
                "base_rate": float(np.mean(y)) if y else None,
                "mean_pred": float(np.mean(p)) if p else None,
                "ece": expected_calibration_error(p, y) if y else None,
                "auc": roc_auc(p, y) if y else None,
            }
        )
    errors = []
    for r in rows:
        if r.verify_depth <= 0:
            continue
        c = preds[id(r)][: r.verify_depth]
        errors.append(float(np.cumprod(c).sum()) - r.accepted)
    report["expected_length_bias"] = float(np.mean(errors)) if errors else None
    report["expected_length_mae"] = float(np.mean(np.abs(errors))) if errors else None
    return report


def rows_from_proposal(proposal, prev_tokens: Sequence[int]) -> list[DraftConfidenceRow]:
    """Build host rows from a closed ``SelfMTPCycleResult`` carrying features."""
    features = getattr(proposal, "draft_features", ()) or ()
    tokens = getattr(proposal, "draft_feature_tokens", ()) or ()
    if not features:
        return []
    relaxed = proposal.relaxed_accepts or (0,) * len(features)
    rows = []
    for index, (feat, toks) in enumerate(zip(features, tokens)):
        if not feat:
            continue
        rows.append(
            DraftConfidenceRow(
                uid=int(proposal.lane_uids[index]),
                features=tuple(tuple(float(v) for v in f) for f in feat),
                tokens=tuple(int(t) for t in toks),
                prev_token=int(prev_tokens[index]),
                verify_depth=int(proposal.draft_depths[index]),
                accepted=int(proposal.accepted_lengths[index]),
                relaxed=int(relaxed[index]),
            )
        )
    return rows


# --------------------------------------------------------------------------
# Data collection hook
# --------------------------------------------------------------------------


class MTPAcceptanceLogger:
    """Bounded JSONL writer of per-cycle draft rows (default-off)."""

    def __init__(
        self,
        path,
        *,
        max_records: int = 1_000_000,
        tag: str = "",
        lookahead: int = 0,
        projection: Any = None,
    ) -> None:
        if isinstance(max_records, bool) or not isinstance(max_records, int) or max_records < 1:
            raise ValueError("acceptance log max_records must be a positive integer")
        # Probe used when no confidence controller supplies its own.
        self.probe = DraftConfidenceProbe(lookahead=lookahead, projection=projection)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_records = max_records
        self.tag = str(tag)
        self.records = 0
        self.dropped = 0
        self._lock = threading.Lock()
        self._handle = open(self.path, "a", encoding="utf-8")

    def record(self, rows: Sequence[DraftConfidenceRow]) -> int:
        written = 0
        with self._lock:
            for row in rows:
                if self.records >= self.max_records:
                    self.dropped += 1
                    continue
                self._handle.write(
                    json.dumps(
                        {
                            "schema": LOG_SCHEMA,
                            "tag": self.tag,
                            "uid": row.uid,
                            "features": [list(f) for f in row.features],
                            "tokens": list(row.tokens),
                            "prev_token": row.prev_token,
                            "verify_depth": row.verify_depth,
                            "accepted": row.accepted,
                            "relaxed": row.relaxed,
                            "labels": row.labels(),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                self.records += 1
                written += 1
            self._handle.flush()
        return written

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.close()


def load_acceptance_log(path) -> list[DraftConfidenceRow]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get("schema") != LOG_SCHEMA:
                raise ValueError("unsupported acceptance log schema")
            rows.append(
                DraftConfidenceRow(
                    uid=int(data["uid"]),
                    features=tuple(tuple(float(v) for v in f) for f in data["features"]),
                    tokens=tuple(int(t) for t in data["tokens"]),
                    prev_token=int(data["prev_token"]),
                    verify_depth=int(data["verify_depth"]),
                    accepted=int(data["accepted"]),
                    relaxed=int(data.get("relaxed", 0)),
                )
            )
    return rows
