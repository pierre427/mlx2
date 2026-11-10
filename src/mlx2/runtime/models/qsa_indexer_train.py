# SPDX-License-Identifier: MIT
# Objective and teacher ported from higa2515/Qwen3.8-27B-QSA-BF16 research/qsa
# (Apache-2.0: losses.py, teacher.py, train_indexer.py); see
# provenance/qsa-indexers-ripple-65b.json.  No code copied; MLX rewrite.
"""Stage-1 QSA indexer distillation in MLX (research; not a serving path).

The backbone stays frozen.  For one training sequence the caller records, per
full-attention layer, the attention input ``x`` [T, hidden], the main
attention's queries [Hq, T, D] and its cached keys [Hkv, T, D].  Then:

* ``block_teacher``: the dense softmax of those queries over those keys,
  summed over heads, max-pooled over each complete 4-token key block, masked
  to fully observed blocks and L1-normalised per query (paper Eq. 17);
* ``TrainableIndexer.scores``: the indexer's block scores with gradients, the
  exact math ``qsa_sidecar.LayerIndexer`` runs at inference;
* ``kl_loss``: ``KL(teacher || softmax(scores))`` over valid blocks, the mean
  over queries that see at least one block (paper Eq. 18);
* ``capture``: teacher mass inside the indexer's own top-k selection.
"""

from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn

from .qsa_sidecar import BLOCK, _rope


def block_valid(length: int) -> mx.array:
    """[T, nb] blocks fully observed by each query."""
    n_blocks = length // BLOCK
    starts = mx.arange(n_blocks, dtype=mx.int32) * BLOCK
    return (starts[None, :] + (BLOCK - 1)) <= mx.arange(length, dtype=mx.int32)[:, None]


def block_teacher(queries: mx.array, keys: mx.array, scale: float,
                  query_chunk: int = 256) -> mx.array:
    """``queries`` [Hq, T, D], ``keys`` [Hkv, T, D] -> target [T, nb] float32.

    Grouped-query heads share their KV head (``Hq % Hkv == 0``).  Each query
    chunk only reads the keys it can see, so the cost is the causal half.
    """
    (n_q, length, dim) = queries.shape
    n_kv = keys.shape[0]
    group = n_q // n_kv
    if group * n_kv != n_q:
        raise ValueError("query heads must be a multiple of KV heads")
    n_blocks = length // BLOCK
    rows = []
    q = queries.reshape(n_kv, group, length, dim)
    for start in range(0, length, query_chunk):
        stop = min(start + query_chunk, length)
        visible = stop
        part = q[:, :, start:stop]
        k = keys[:, None, :visible]  # [Hkv, 1, S, D]
        logits = (part @ k.swapaxes(-1, -2)).astype(mx.float32) * scale  # [Hkv, G, c, S]
        pos = mx.arange(start, stop, dtype=mx.int32)[:, None]
        causal = mx.arange(visible, dtype=mx.int32)[None, :] <= pos
        logits = mx.where(causal, logits, -mx.inf)
        probs = mx.softmax(logits, axis=-1).sum(axis=(0, 1))  # [c, S]
        nb_vis = visible // BLOCK
        blocks = probs[:, : nb_vis * BLOCK].reshape(stop - start, nb_vis, BLOCK).max(axis=-1)
        if nb_vis < n_blocks:
            blocks = mx.pad(blocks, [(0, 0), (0, n_blocks - nb_vis)])
        mx.eval(blocks)
        rows.append(blocks)
    target = mx.concatenate(rows, axis=0) if len(rows) > 1 else rows[0]
    target = mx.where(block_valid(length), target, 0.0)
    denom = target.sum(axis=-1, keepdims=True)
    return mx.where(denom > 0, target / mx.maximum(denom, 1e-12), target)


class TrainableIndexer(nn.Module):
    """One layer's QSA indexer: H query heads, one shared key head, 4-token
    mean-pooled key blocks, zero-centred RMSNorms, optional partial RoPE."""

    def __init__(self, hidden: int, *, heads: int = 4, head_dim: int = 128,
                 rope_dim: int = 0, rope_base: float = 1e7, eps: float = 1e-6,
                 temperature: dict | None = None):
        super().__init__()
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, head_dim, bias=False)
        self.q_norm = mx.zeros((head_dim,))
        self.k_norm = mx.zeros((head_dim,))
        self.k_block_norm = mx.zeros((head_dim,))
        self._geometry = (heads, head_dim, rope_dim, rope_base, eps)
        # Training-only softmax temperature for the KL (top-k selection is
        # invariant to a positive scale, so inference never reads it).  The
        # scores are RMS-normed dot products: their scale moves only through
        # the zero-centred norm weights, ~lr per step, so a 3e-4 run keeps the
        # initial (much sharper than the teacher) softmax for hundreds of
        # steps.  ``log_temp = lr_mult * u`` gives the scalar its own pace
        # under the shared AdamW.
        self._temp_mult = None
        if temperature:
            self._temp_mult = float(temperature.get("lr_mult", 1.0))
            self.temp_u = mx.array([math.log(temperature["init"]) / self._temp_mult])

    def temperature(self) -> float:
        if self._temp_mult is None:
            return 1.0
        return float(mx.exp(self.temp_u[0] * self._temp_mult).item())

    def loss_scores(self, x: mx.array) -> mx.array:
        """Scores as the KL sees them: raw scores times the learned temperature."""
        if self._temp_mult is None:
            return self.scores(x)
        # Scale before masking: a -inf entry would put 0 * inf into d/dtemp.
        s = self._raw_scores(x) * mx.exp(self.temp_u[0] * self._temp_mult)
        return mx.where(block_valid(x.shape[0]), s, -mx.inf)

    def scores(self, x: mx.array) -> mx.array:
        """``x`` [T, hidden] float32 -> [T, nb] scores (invalid blocks -inf)."""
        return mx.where(block_valid(x.shape[0]), self._raw_scores(x), -mx.inf)

    def _raw_scores(self, x: mx.array) -> mx.array:
        (heads, head_dim, rope_dim, rope_base, eps) = self._geometry
        length = x.shape[0]
        n_blocks = length // BLOCK
        q = self.q_proj(x).reshape(length, heads, head_dim)
        q = mx.fast.rms_norm(q, 1.0 + self.q_norm, eps).transpose(1, 0, 2)
        q = _rope(q, dims=rope_dim, base=rope_base, offset=0)
        k = mx.fast.rms_norm(self.k_proj(x), 1.0 + self.k_norm, eps)
        kb = k[: n_blocks * BLOCK].reshape(n_blocks, BLOCK, head_dim).mean(axis=1)
        kb = mx.fast.rms_norm(kb, 1.0 + self.k_block_norm, eps)
        kb = _rope(kb[None], dims=rope_dim, base=rope_base, offset=0, scale=float(BLOCK))[0]
        return mx.maximum(q @ kb.T, 0.0).sum(axis=0)

    def export(self, prefix: str) -> dict:
        """Tensors under the names ``qsa_sidecar.QSASidecar`` loads."""
        return {
            f"{prefix}._qsa_indexer.q_proj.weight": self.q_proj.weight,
            f"{prefix}._qsa_indexer.k_proj.weight": self.k_proj.weight,
            f"{prefix}._qsa_indexer.q_norm": self.q_norm,
            f"{prefix}._qsa_indexer.k_norm": self.k_norm,
            f"{prefix}._qsa_indexer.k_block_norm": self.k_block_norm,
        }


def fixed_blocks(length: int, sink_blocks: int, recent_blocks: int) -> mx.array:
    """[T, nb] blocks a hybrid selection always takes: the first
    ``sink_blocks`` and each query's newest ``recent_blocks`` complete blocks."""
    n_blocks = length // BLOCK
    j = mx.arange(n_blocks, dtype=mx.int32)[None, :]
    newest = (mx.arange(length, dtype=mx.int32)[:, None] + 1) // BLOCK
    age = newest - 1 - j
    return (j < sink_blocks) | ((age >= 0) & (age < recent_blocks))


def kl_loss(scores: mx.array, target: mx.array, fixed: mx.array | None = None) -> mx.array:
    """Mean over queries with teacher mass of KL(target || softmax(scores)).

    ``fixed`` [T, nb]: blocks a hybrid selection always takes.  The KL is
    then over the remaining valid blocks with the teacher renormalised inside
    them (the reference's stage-2 restriction, paper Eq. 20), and starts at
    the first query that has such a block.
    """
    if fixed is not None:
        allowed = ~fixed & block_valid(scores.shape[0])
        has_free = allowed.any(axis=-1)
        if not bool(has_free.any().item()):
            # Every row would be all -inf: loss 0 with NaN gradients.  Refuse.
            raise ValueError("hybrid KL: no query has a valid block outside the fixed "
                             "sink/recent blocks (sequence too short for this selection)")
        first = int(mx.argmax(has_free).item())
        scores = mx.where(allowed, scores, -mx.inf)[first:]
        target = mx.where(allowed, target, 0.0)[first:]
        denom = target.sum(-1, keepdims=True)
        target = mx.where(denom > 0, target / mx.maximum(denom, 1e-12), 0.0)
        return _kl(scores, target)
    # Queries before the first complete block have no valid score at all; an
    # all ``-inf`` row would put NaN into the logsumexp gradient.
    return _kl(scores[BLOCK - 1:], target[BLOCK - 1:])


def _kl(scores: mx.array, target: mx.array) -> mx.array:
    logp =scores - mx.logsumexp(scores, axis=-1, keepdims=True)
    has = target > 0
    safe_t = mx.where(has, target, 1.0)
    per = mx.where(has, target * (mx.log(safe_t) - mx.where(has, logp, 0.0)), 0.0).sum(-1)
    rows = (target.sum(-1) > 0).astype(mx.float32)
    return (per * rows).sum() / mx.maximum(rows.sum(), 1.0)


def capture(scores: mx.array, target: mx.array, topk: int) -> tuple[float, int]:
    """Mean teacher mass inside the top-``topk`` indexer blocks, over the rows
    that see more than ``topk`` complete blocks (others are dense anyway).
    Returns ``(mean mass, rows counted)``."""
    n_blocks = scores.shape[1]
    if n_blocks <= topk:
        return 1.0, 0
    first = (topk + 1) * BLOCK - 1  # first query with more than topk blocks
    s = scores[first:]
    t = target[first:]
    ids = mx.argpartition(-s, kth=topk - 1, axis=-1)[:, :topk]
    mass = mx.take_along_axis(t, ids, axis=-1).sum(-1)
    return float(mass.mean().item()), int(mass.shape[0])


def teacher_entropy(target: mx.array) -> float:
    rows = target.sum(-1) > 0
    ent = -(mx.where(target > 0, target * mx.log(mx.maximum(target, 1e-30)), 0.0)).sum(-1)
    return float((mx.where(rows, ent, 0.0).sum() / mx.maximum(rows.sum(), 1)).item())


def global_clip(grads: list, max_norm: float) -> tuple[list, float]:
    """Clip a list of grad trees by their joint L2 norm (all layers together)."""
    from mlx.utils import tree_flatten, tree_map

    total = 0.0
    for tree in grads:
        for (_, g) in tree_flatten(tree):
            total += float((g.astype(mx.float32) ** 2).sum().item())
    norm = math.sqrt(total)
    factor = min(1.0, max_norm / (norm + 1e-6))
    if factor < 1.0:
        grads = [tree_map(lambda g: g * factor, tree) for tree in grads]
    return grads, norm


def mass_breakdown(target: mx.array, scores: dict, topk: int, *, sink_blocks: int = 16,
                   hybrid_recent: int = 128, selections: dict | None = None) -> dict:
    """Diagnostic: where does the teacher's block mass sit, per selection rule?

    Rows that see more than ``topk`` complete blocks only.  ``scores`` maps an
    arm name to indexer scores [T, nb].  Returns the mean teacher mass for:
    ``oracle`` (top-k of the teacher itself, the ceiling for any k-block
    rule); ``sink`` (first ``sink_blocks`` blocks); ``recent`` (the newest
    ``topk`` complete blocks, a sliding window of the same budget);
    ``sink_recent`` (sink plus the newest ``topk - sink_blocks``); each
    indexer arm's own top-k; and ``hybrid:<arm>`` (sink + newest
    ``hybrid_recent`` blocks + the arm's top picks among the rest, same k);
    plus mass by distance (newest 16/64/256/512/1024 blocks).

    ``selections`` maps an arm name to the ``{"sink_blocks", "recent_blocks"}``
    rule it was trained with; that arm's ``hybrid:<arm>`` column (and a
    ``hybrid:oracle:<arm>`` ceiling when it differs from the defaults) uses its
    own rule.  A hybrid column is omitted when its fixed blocks leave no
    learned slot in the budget (the sidecar refuses such a rule too).
    """
    (length, n_blocks) = target.shape
    first = (topk + 1) * BLOCK - 1
    if first >= length:
        return {}
    t = target[first:]
    rows = t.shape[0]
    pos = mx.arange(first, length, dtype=mx.int32)[:, None]
    newest = (pos + 1) // BLOCK  # blocks [0, newest) are complete
    j = mx.arange(n_blocks, dtype=mx.int32)[None, :]
    age = newest - 1 - j  # 0 = newest complete block; negative = not complete

    def mass(mask):
        return float((mx.where(mask, t, 0.0).sum(-1)).mean().item())

    def topk_mass(s, k, allowed=None):
        s = mx.where(allowed, s, -mx.inf) if allowed is not None else s
        ids = mx.argpartition(-s, kth=k - 1, axis=-1)[:, :k]
        return mx.take_along_axis(t, ids, axis=-1).sum(-1)

    sink = j < sink_blocks
    out = {"rows": rows, "oracle": float(topk_mass(t, topk).mean().item()),
           "sink": mass(sink), "recent": mass((age >= 0) & (age < topk)),
           "sink_recent": mass(sink | ((age >= 0) & (age < topk - sink_blocks)))}
    for width in (16, 64, 256, 512, 1024):
        out[f"newest_{width}"] = mass((age >= 0) & (age < width))
    def hybrid(s, rule):
        (n_sink, n_recent) = rule
        rest = topk - n_sink - n_recent
        if rest <= 0:
            return None
        fixed = (j < n_sink) | ((age >= 0) & (age < n_recent))
        picked = topk_mass(s, rest, allowed=(~fixed) & (age >= 0))
        return float((mx.where(fixed, t, 0.0).sum(-1) + picked).mean().item())

    default = (sink_blocks, hybrid_recent)
    for (name, s) in scores.items():
        s = s[first:]
        out[name] = float(topk_mass(s, topk).mean().item())
        sel = (selections or {}).get(name)
        rule = (int(sel["sink_blocks"]), int(sel["recent_blocks"])) if sel else default
        value = hybrid(s, rule)
        if value is not None:
            out[f"hybrid:{name}"] = value
        if rule != default:
            value = hybrid(t, rule)
            if value is not None:
                out[f"hybrid:oracle:{name}"] = value
    value = hybrid(t, default)
    if value is not None:
        out["hybrid:oracle"] = value
    return out


INDEXER_KEYS = ("q_proj.weight", "k_proj.weight", "q_norm", "k_norm", "k_block_norm")


def trainer_state(indexers: dict, opts: dict) -> dict:
    """Flat tensors for an exact resume: each layer's temperature parameter
    and AdamW state (``{layer}.temp_u``, ``{layer}.opt.<path>``)."""
    from mlx.utils import tree_flatten

    out = {}
    for (i, idx) in indexers.items():
        if getattr(idx, "temp_u", None) is not None:
            out[f"{i}.temp_u"] = idx.temp_u
        for (key, value) in tree_flatten(opts[i].state):
            out[f"{i}.opt.{key}"] = value
    return out


def load_trainer_checkpoint(directory, indexers: dict, opts: dict, *, step: int) -> dict:
    """Load a ``save`` directory into ``indexers`` (and ``opts`` when the
    directory has ``trainer-state.safetensors``).  Checkpoints written before
    the trainer state existed restore the temperature from ``config.json``
    and start AdamW fresh; the receipt says which happened."""
    import json
    from pathlib import Path

    from mlx.utils import tree_unflatten

    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())["qsa"]
    if int(config["trained_steps"]) != step:
        raise ValueError(f"checkpoint holds step {config['trained_steps']}, resume asked {step}")
    weights = mx.load(str(directory / "qsa-indexers.safetensors"))
    state_file = directory / "trainer-state.safetensors"
    state = mx.load(str(state_file)) if state_file.is_file() else None
    temps = config.get("training_temperature") or {}
    for (i, idx) in indexers.items():
        prefix = f"model.layers.{i}.self_attn._qsa_indexer."
        idx.update({"q_proj": {"weight": weights[prefix + "q_proj.weight"]},
                    "k_proj": {"weight": weights[prefix + "k_proj.weight"]},
                    "q_norm": weights[prefix + "q_norm"],
                    "k_norm": weights[prefix + "k_norm"],
                    "k_block_norm": weights[prefix + "k_block_norm"]})
        if getattr(idx, "temp_u", None) is not None:
            if state is not None and f"{i}.temp_u" in state:
                idx.temp_u = state[f"{i}.temp_u"]
            elif str(i) in temps:
                idx.temp_u = mx.array([math.log(temps[str(i)]) / idx._temp_mult])
            else:
                raise ValueError(f"no training temperature for layer {i}")
        if state is not None:
            prefix = f"{i}.opt."
            # MLX keeps the learning rate inside the optimizer state, so a
            # restore would silently replace the rate the resume asked for.
            # An exact resume continues at the checkpoint's rate: refuse a
            # different one instead of ignoring it.
            wanted = float(opts[i].learning_rate.item())
            opts[i].state = tree_unflatten(
                [(k[len(prefix):], v) for (k, v) in state.items() if k.startswith(prefix)])
            restored = float(opts[i].learning_rate.item())
            if not math.isclose(restored, wanted, rel_tol=1e-6):
                raise ValueError(f"layer {i}: checkpoint learning rate {restored:g} differs "
                                 f"from the resume's {wanted:g}; an exact resume keeps it")
        mx.eval(idx.parameters())
    rates = sorted({float(opt.learning_rate.item()) for opt in opts.values()})
    return {"optimizer_state": "restored" if state is not None else "fresh (not saved)",
            "learning_rate": rates[0] if len(rates) == 1 else rates,
            "temperature": ("trainer-state" if state is not None else "config.json"),
            "indexers_sha256": __import__("hashlib").sha256(
                (directory / "qsa-indexers.safetensors").read_bytes()).hexdigest()}
