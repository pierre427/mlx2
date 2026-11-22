"""TRAC-style persistence early commit for DiffusionGemma: opt-in, unqualified.

Method: arXiv 2611.25643 (TRAC, "stability-aware early commitment"),
paraphrased.  On top of a Fast-dLLM confidence-threshold sampler, a masked
(still unrevealed) position whose confidence is only *medium* may also be
committed early when its prediction is temporally stable: its top-1 token is
unchanged across consecutive denoising steps and, optionally, the approximate
Jensen-Shannon divergence between the two steps' top-K distributions stays
under a bound.  High-confidence commits and the baseline's force-one rule are
untouched, so with no stable medium position a step is bit-identical to the
confidence-threshold baseline.

Parameters are the paper's reported settings (its Table 2 configuration):
``low_threshold`` 0.5 (tau_low), ``stable_steps`` 2 (k: consecutive stable
step-to-step transitions, i.e. the same top-1 on k + 1 consecutive forwards),
``jsd_top_k`` 8 with an OTHER bucket, ``jsd_epsilon`` 0.04.  The paper uses
tau_high 0.8 = its Fast-dLLM threshold; here tau_high is whatever threshold
the confidence-threshold sampler runs with (mlx-vlm default 0.9), so an A/B
against that sampler isolates the persistence rule.  The paper does not state
the JSD log base; natural log is assumed here (max ln 2).  The paper does not
sweep k, K or epsilon.

Status: implemented, NOT qualified, never selected by default.  The paper
evaluates it only on Chinese-English machine translation with LLaDA-8B
(fewer forwards than Fast-dLLM at a ~0.02 COMET macro delta); nothing is
known about DiffusionGemma, other tasks, or its random-token canvas.  It is
a lossy policy change (different commit schedule, different output), not an
exact optimization.  The paper's adaptive canvas length is not implemented.

Cost: every denoising step copies the canvas's top-1, top-8 and confidence
rows to the host and runs a Python JSD loop over the canvas (256 positions
for DiffusionGemma), beside the top-8 partition over the full vocabulary on
device.  Fewer forwards therefore need not mean less time: an A/B must
measure wall time per response, not only forward counts.

This module imports no MLX at module scope.  ``TRACPersistenceState`` and
``simulate_generate`` are host/numpy reference code (CPU tests use them with a
fake denoiser); ``persistence_hook`` installs the policy into the pinned
mlx-vlm confidence-threshold loop for the duration of one call.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

SAMPLER = "trac-persistence"
# sha256 of mlx_vlm/generate/diffusion.py at the pinned SOURCE_REVISION
# (provenance/diffusion-gemma.json).  The hook replaces two module functions
# of exactly that file and refuses any other version.
PINNED_DIFFUSION_SHA256 = "bf808f25b1056d2d5efd364d72c5bedc9ea1c11183a35811e7424c95d5cba868"


@dataclass(frozen=True, slots=True)
class TRACPersistenceConfig:
    low_threshold: float = 0.5
    stable_steps: int = 2
    jsd_top_k: int = 8
    # None disables the JSD check (top-1 persistence only).
    jsd_epsilon: float | None = 0.04
    # Optional token id the early-commit path must never commit (a [MASK] id;
    # DiffusionGemma's canvas is random-token noise and has none).
    mask_token_id: int | None = None

    def __post_init__(self):
        if (isinstance(self.low_threshold, bool) or not isinstance(self.low_threshold, (int, float))
                or not 0.0 < self.low_threshold <= 1.0):
            raise ValueError("TRAC low_threshold must be in (0, 1]")
        if type(self.stable_steps) is not int or not 1 <= self.stable_steps <= 48:
            raise ValueError("TRAC stable_steps must be 1..48")
        if type(self.jsd_top_k) is not int or not 1 <= self.jsd_top_k <= 64:
            raise ValueError("TRAC jsd_top_k must be 1..64")
        if self.jsd_epsilon is not None and (
                isinstance(self.jsd_epsilon, bool) or not isinstance(self.jsd_epsilon, (int, float))
                or not 0.0 <= self.jsd_epsilon <= float(np.log(2.0))):
            raise ValueError("TRAC jsd_epsilon must be None or 0..ln 2")
        if self.mask_token_id is not None and (type(self.mask_token_id) is not int
                                               or self.mask_token_id < 0):
            raise ValueError("TRAC mask_token_id must be a nonnegative integer")

    def as_dict(self) -> dict:
        return {"sampler": SAMPLER, **asdict(self), "jsd_log_base": "e",
                "source": "arXiv 2611.25643", "qualified": False}


def approx_topk_jsd(prev_ids, prev_probs, ids, probs) -> np.ndarray:
    """Per-position JSD (nats) over the union of two top-K sets plus OTHER.

    A token outside a step's own top-K counts as 0 there; each step's OTHER
    bucket holds ``1 - sum(top-K)``.  Shapes: ids/probs ``[L, K]``.
    """
    prev_ids = np.asarray(prev_ids)
    ids = np.asarray(ids)
    prev_probs = np.asarray(prev_probs, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    out = np.empty(ids.shape[0], dtype=np.float64)
    for i in range(ids.shape[0]):
        union = np.union1d(prev_ids[i], ids[i])
        p = np.zeros(union.size + 1)
        q = np.zeros(union.size + 1)
        p[np.searchsorted(union, prev_ids[i])] = prev_probs[i]
        q[np.searchsorted(union, ids[i])] = probs[i]
        p[-1] = max(0.0, 1.0 - prev_probs[i].sum())
        q[-1] = max(0.0, 1.0 - probs[i].sum())
        m = 0.5 * (p + q)

        def kl(a):
            nz = a > 0
            return float(np.sum(a[nz] * np.log(a[nz] / m[nz])))

        out[i] = 0.5 * kl(p) + 0.5 * kl(q)
    return out


class TRACPersistenceState:
    """Per-canvas stability streaks.  Reset at every new canvas (block)."""

    def __init__(self, config: TRACPersistenceConfig, length: int):
        self.config = config
        self.length = int(length)
        self.prev_ids: np.ndarray | None = None
        self.prev_topk: tuple[np.ndarray, np.ndarray] | None = None
        self.streak = np.zeros(self.length, dtype=np.int64)
        self.early_commits = 0

    def observe(self, top1_ids, topk_ids=None, topk_probs=None) -> np.ndarray:
        """Record one forward's prediction; return the current streaks."""
        top1_ids = np.asarray(top1_ids).reshape(-1)
        if top1_ids.size != self.length:
            raise ValueError("TRAC prediction length does not match its canvas")
        stable = np.zeros(self.length, dtype=bool)
        if self.prev_ids is not None:
            stable = top1_ids == self.prev_ids
            if self.config.jsd_epsilon is not None:
                if topk_ids is None or self.prev_topk is None:
                    raise ValueError("TRAC JSD check needs top-K ids and probabilities")
                jsd = approx_topk_jsd(*self.prev_topk, topk_ids, topk_probs)
                stable &= jsd <= self.config.jsd_epsilon
        self.streak = np.where(stable, self.streak + 1, 0)
        self.prev_ids = top1_ids.copy()
        if topk_ids is not None:
            self.prev_topk = (np.asarray(topk_ids).copy(), np.asarray(topk_probs).copy())
        return self.streak

    def early_mask(self, unrevealed, denoiser_ids, top1_ids, top1_prob) -> np.ndarray:
        """Medium-confidence positions eligible for early commit this step.

        Only unrevealed positions; only where the denoiser token is the
        persistent top-1 (a sampled non-argmax token is never committed by
        this path); never the configured mask token.
        """
        unrevealed = np.asarray(unrevealed, dtype=bool).reshape(-1)
        denoiser_ids = np.asarray(denoiser_ids).reshape(-1)
        top1_ids = np.asarray(top1_ids).reshape(-1)
        top1_prob = np.asarray(top1_prob, dtype=np.float64).reshape(-1)
        mask = (unrevealed & (self.streak >= self.config.stable_steps)
                & (top1_prob >= self.config.low_threshold) & (denoiser_ids == top1_ids))
        if self.config.mask_token_id is not None:
            mask &= top1_ids != self.config.mask_token_id
        return mask


def confidence_transfer_mask(confidence, unrevealed, threshold, *, early=None,
                             force_all=False) -> np.ndarray:
    """Numpy mirror of the pinned ``_diffusion_confidence_transfer_mask``.

    ``early`` positions are admitted as if they had met the threshold, so
    the force-one rule fires only when neither path commits.
    """
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    unrevealed = np.asarray(unrevealed, dtype=bool).reshape(-1)
    if force_all:
        return unrevealed.copy()
    if early is not None:
        confidence = np.where(early, 2.0, confidence)
    transfer = unrevealed & (confidence >= threshold)
    if unrevealed.any() and not transfer.any():
        best = int(np.argmax(np.where(unrevealed, confidence, -np.inf)))
        transfer = transfer.copy()
        transfer[best] = True
    return transfer


@dataclass(frozen=True, slots=True)
class SimulatedGeneration:
    tokens: list[int]
    forwards: int
    steps_per_block: list[int]
    commit_step: list[list[int]]  # per block, per position: 1-based step committed (0: never)
    early_commits: int


def simulate_generate(denoise: Callable[[int, int, np.ndarray], np.ndarray], *,
                      max_tokens: int, canvas_length: int, max_denoising_steps: int,
                      threshold: float = 0.9, trac: TRACPersistenceConfig | None = None,
                      noise: Callable[[int], np.ndarray] | None = None) -> SimulatedGeneration:
    """Greedy (temperature 0) reference of the pinned confidence-threshold loop.

    ``denoise(block, step, canvas) -> probs [L, V]`` is one forward.  Mirrors
    the pinned control flow: the last scheduled step only reads the argmax;
    a block ends early once every position is revealed; the block's output is
    the last forward's argmax.  ``trac=None`` is the baseline sampler.
    Stopping-config early exit and EOS handling are not modelled.
    """
    if canvas_length < 1 or max_tokens < 1 or max_denoising_steps < 1:
        raise ValueError("simulate_generate limits must be positive")
    noise = noise or (lambda n: np.zeros(n, dtype=np.int64))
    tokens: list[int] = []
    forwards = 0
    steps_per_block: list[int] = []
    commit_steps: list[list[int]] = []
    early_total = 0
    block = 0
    while len(tokens) < max_tokens:
        length = canvas_length
        canvas = noise(length)
        revealed = np.zeros(length, dtype=bool)
        draft = canvas.copy()
        commit_step = [0] * length
        state = TRACPersistenceState(trac, length) if trac is not None else None
        steps = 0
        argmax = canvas
        for cur_step in reversed(range(1, max_denoising_steps + 1)):
            steps += 1
            forwards += 1
            probs = np.asarray(denoise(block, steps, canvas.copy()), dtype=np.float64)
            if probs.shape[0] != length:
                raise ValueError("fake denoiser returned the wrong canvas length")
            argmax = np.argmax(probs, axis=-1)
            if cur_step == 1:
                break
            denoiser = argmax
            confidence = probs[np.arange(length), denoiser]
            unrevealed = ~revealed
            early = None
            if state is not None:
                k = min(trac.jsd_top_k, probs.shape[1])
                topk_ids = np.argsort(-probs, axis=-1, kind="stable")[:, :k]
                topk_probs = np.take_along_axis(probs, topk_ids, axis=-1)
                state.observe(argmax, topk_ids, topk_probs)
                early = state.early_mask(unrevealed, denoiser, argmax, confidence)
                high = unrevealed & (confidence >= threshold)
                early = early & ~high
            accept = confidence_transfer_mask(confidence, unrevealed, threshold,
                                              early=early, force_all=cur_step == 1)
            if early is not None:
                n_early = int((accept & early).sum())
                state.early_commits += n_early
                early_total += n_early
            draft = np.where(accept, denoiser, draft)
            for i in np.flatnonzero(accept):
                commit_step[i] = steps
            revealed |= accept
            canvas = np.where(revealed, draft, noise(length))
            if revealed.all():
                break
        steps_per_block.append(steps)
        commit_steps.append(commit_step)
        room = max_tokens - len(tokens)
        tokens.extend(int(t) for t in argmax[:room])
        block += 1
    return SimulatedGeneration(tokens, forwards, steps_per_block, commit_steps, early_total)


# --- MLX hook into the pinned mlx-vlm loop (imports MLX lazily) -------------

_HOOK_LOCK = threading.Lock()
# Identity of the call that installed the hook.  The wrappers are module
# globals, so every caller of the pinned loop reaches them while installed;
# only the installing call's context (a new thread starts with an empty
# context) gets TRAC, every other caller falls through to the originals.
_HOOK_OWNER: contextvars.ContextVar[object | None] = contextvars.ContextVar(
    "mlx2_trac_hook_owner", default=None)


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextlib.contextmanager
def persistence_hook(config: TRACPersistenceConfig, *, module=None,
                     expected_sha256: str | None = PINNED_DIFFUSION_SHA256):
    """Run the pinned confidence-threshold loop with TRAC early commits.

    Wraps two module functions of ``mlx_vlm.generate.diffusion`` for one
    call: ``_diffusion_token_probability`` (to see the step's logits and
    denoiser tokens) and ``_diffusion_confidence_transfer_mask`` (to admit
    early commits).  The originals are restored on exit.  Refuses a source
    file whose hash differs from the pinned revision.  Yields a dict counter
    ``{"early_commits": int, "steps": int}``.

    Only the installing call is affected: a concurrent confidence-threshold
    call from another thread or context reaches the wrappers but runs the
    originals unchanged.  A second concurrent TRAC call is refused.
    """
    if module is None:
        import mlx_vlm.generate.diffusion as module  # noqa: PLC0415
    if expected_sha256 is not None and _file_sha256(module.__file__) != expected_sha256:
        raise RuntimeError("TRAC hook refuses an unpinned mlx_vlm diffusion source")
    import mlx.core as mx  # noqa: PLC0415

    original_probability = module._diffusion_token_probability
    original_transfer = module._diffusion_confidence_transfer_mask
    counters = {"early_commits": 0, "steps": 0}
    owner = object()

    def owned() -> bool:
        return _HOOK_OWNER.get() is owner
    holder: dict = {"state": None, "step": None}

    def token_probability(processed_logits, token_ids):
        confidence = original_probability(processed_logits, token_ids)
        if not owned():
            return confidence
        logits = processed_logits.astype(mx.float32)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        k = min(config.jsd_top_k, logp.shape[-1])
        top_ids = mx.argpartition(-logp, kth=k - 1, axis=-1)[..., :k]
        top_p = mx.exp(mx.take_along_axis(logp, top_ids, axis=-1))
        top1 = mx.argmax(logp, axis=-1)
        top1_p = mx.exp(mx.max(logp, axis=-1))
        holder["step"] = (token_ids, top1, top1_p, top_ids, top_p)
        return confidence

    def transfer_mask(confidence, unrevealed_mask, threshold, *, force_all=False):
        if not owned():
            return original_transfer(confidence, unrevealed_mask, threshold,
                                     force_all=force_all)
        step = holder.pop("step", None)
        holder["step"] = None
        if step is None or force_all or confidence.shape[0] != 1:
            return original_transfer(confidence, unrevealed_mask, threshold,
                                     force_all=force_all)
        token_ids, top1, top1_p, top_ids, top_p = step
        mx.eval(unrevealed_mask, token_ids, top1, top1_p, top_ids, top_p, confidence)
        unrevealed = np.array(unrevealed_mask[0]).astype(bool)
        length = unrevealed.size
        state = holder["state"]
        if state is None or state.length != length or unrevealed.all():
            state = holder["state"] = TRACPersistenceState(config, length)
        top1_np = np.array(top1[0])
        state.observe(top1_np, np.array(top_ids[0]), np.array(top_p[0], dtype=np.float64))
        early = state.early_mask(unrevealed, np.array(token_ids[0]), top1_np,
                                 np.array(top1_p[0], dtype=np.float64))
        early &= ~(np.array(confidence[0], dtype=np.float64) >= threshold)
        counters["steps"] += 1
        if not early.any():
            return original_transfer(confidence, unrevealed_mask, threshold,
                                     force_all=force_all)
        state.early_commits += int(early.sum())
        counters["early_commits"] += int(early.sum())
        boosted = mx.where(mx.array(early)[None, :], mx.array(2.0, confidence.dtype), confidence)
        return original_transfer(boosted, unrevealed_mask, threshold, force_all=force_all)

    # Everything that can fail before patching is above; the lock is taken
    # last and released on every path below.
    if not _HOOK_LOCK.acquire(blocking=False):
        raise RuntimeError("TRAC hook is already installed by another call")
    try:
        context_token = _HOOK_OWNER.set(owner)
        try:
            module._diffusion_token_probability = token_probability
            module._diffusion_confidence_transfer_mask = transfer_mask
            yield counters
        finally:
            module._diffusion_token_probability = original_probability
            module._diffusion_confidence_transfer_mask = original_transfer
            _HOOK_OWNER.reset(context_token)
    finally:
        _HOOK_LOCK.release()


__all__: Sequence[str] = (
    "SAMPLER", "TRACPersistenceConfig", "TRACPersistenceState", "approx_topk_jsd",
    "confidence_transfer_mask", "simulate_generate", "SimulatedGeneration",
    "persistence_hook",
)
