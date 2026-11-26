"""Exact block verification decided on the device (lazy MLX graph).

Original mlx2 code.  The algorithm is the host verifier's
``speculative_sampling.verify_block_proposals`` (Sun et al. 2024,
arXiv:2403.10444, Algorithm 2; provenance/block-verification.json), written
as one lazy graph so ``device_round``'s ratio rule can take it without a host
read per position:

* ``P_0 = 1``, ``P_i = min(1, P_{i-1} p_i(x_i) / q_i(x_i))``;
* position ``i < K`` passes with ``h_i = S_i / (S_i + 1 - P_i)``, where
  ``S_i = sum_v (P_i p_{i+1}(v) - q_{i+1}(v))+``; position ``K`` passes with
  ``h_K = P_K``; a zero denominator passes (``h = 1``), as on the host;
* ``tau`` is the LAST passing position (strict ``u < h``);
* the round emits ``x_1..x_tau`` and then the bonus from ``p_{K+1}`` when
  ``tau = K``, else a correction from ``normalized(P_tau p_{tau+1} -
  q_{tau+1})+`` (``p_{tau+1}`` when rounding leaves no mass).

Every candidate correction is drawn lazily and the one at ``tau`` is taken,
the same shape as the token-wise device ratio rule, so the graph has no
data-dependent host branch.  Same output law as the host verifier; the
random draws differ (Pierre, 2026-11-26: sampled device rounds keep the law,
with RNG keyed by seed, position and purpose).

Leading batch dimensions are allowed so tests can draw many independent
trials in one graph; ``device_round`` calls it with one lane.
"""

from __future__ import annotations


def block_decide(mx, logp, logq, drafts, uniforms, correction_keys, bonus_key):
    """Return ``(accepted, last, coverage)``.

    ``logp``: ``(..., K+1, V)`` target log-probabilities (the transformed
    law the lane samples from); ``logq``: ``(..., K, V)`` the drafter's
    normalized log-law (``-inf`` outside its support); ``drafts``:
    ``(..., K)`` int32 drafted tokens; ``uniforms``: ``(..., K)`` in
    ``[0, 1)``; ``correction_keys``: ``K`` PRNG keys, one per candidate
    correction position; ``bonus_key``: one key.  ``accepted`` is ``tau``
    (int32, ``(...)``), ``last`` the token emitted after the accepted
    prefix, ``coverage`` ``P_tau`` (float32).
    """
    count = int(drafts.shape[-1])
    if count < 1:
        raise ValueError("block verification needs at least one drafted token")
    if tuple(logp.shape[-2:-1]) != (count + 1,) or tuple(logq.shape[-2:-1]) != (count,):
        raise ValueError("block verification row counts do not match the drafts")
    if int(logq.shape[-1]) != int(logp.shape[-1]):
        raise ValueError("drafter and target laws differ in width")
    f32 = mx.float32
    logp = logp.astype(f32)
    logq = logq.astype(f32)
    p = mx.exp(logp)
    q = mx.exp(logq)
    idx = drafts.astype(mx.int32)[..., None]
    lp = mx.take_along_axis(logp[..., :count, :], idx, axis=-1)[..., 0]
    lq = mx.take_along_axis(logq, idx, axis=-1)[..., 0]
    # p(x)/q(x) in log space; q(x) > 0 for every drafted token (the drafter
    # sampled it), and a target that forbids x gives ratio 0.
    ratio = mx.where(mx.isfinite(lp), mx.exp(lp - lq), mx.zeros_like(lp))
    one = mx.ones(drafts.shape[:-1], dtype=f32)
    masses = [one]
    mass = one
    passes = []
    for i in range(count):
        mass = mx.minimum(one, mass * ratio[..., i])
        masses.append(mass)
        if i == count - 1:
            passing = mass
        else:
            residual = mx.sum(mx.maximum(mass[..., None] * p[..., i + 1, :] - q[..., i + 1, :], 0.0),
                              axis=-1)
            denominator = residual + 1.0 - mass
            passing = mx.where(denominator <= 0, one, residual / mx.where(denominator <= 0, one, denominator))
        passes.append(uniforms[..., i].astype(f32) < passing)
    positions = mx.arange(1, count + 1, dtype=mx.int32)
    passed = mx.stack(passes, axis=-1).astype(mx.int32)
    accepted = mx.max(passed * positions, axis=-1).astype(mx.int32)
    masses = mx.stack(masses, axis=-1)                       # (..., K+1)
    coverage = mx.take_along_axis(masses, accepted[..., None], axis=-1)[..., 0]
    outcomes = []
    for j in range(count):
        residual = mx.maximum(masses[..., j, None] * p[..., j, :] - q[..., j, :], 0.0)
        total = mx.sum(residual, axis=-1, keepdims=True)
        law = mx.where(total > 0, mx.log(residual), logp[..., j, :])
        outcomes.append(mx.random.categorical(law, key=correction_keys[j]).astype(mx.int32))
    outcomes.append(mx.random.categorical(logp[..., count, :], key=bonus_key).astype(mx.int32))
    outcomes = mx.stack(outcomes, axis=-1)
    last = mx.take_along_axis(outcomes, accepted[..., None], axis=-1)[..., 0]
    return accepted, last, coverage
