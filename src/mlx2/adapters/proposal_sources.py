"""Adapter-owned native MTP source for external proposal arbitration.

Uses the existing target's public MTP protocol; every cache is private and
discarded after proposing. This deliberately trades recomputation for a small,
auditable lifetime contract. No private state is published into APCv2.
"""

from __future__ import annotations


def native_mtp_source(model):
    from ..runtime.models.cache import make_prompt_cache

    required = ("make_mtp_cache", "mtp_step")
    if getattr(model, "mtp", None) is None or any(
        not callable(getattr(model, name, None)) for name in required
    ):
        raise ValueError("target does not provide a native MTP proposal protocol")
    if not (
        callable(getattr(model, "mtp_backbone", None))
        or callable(getattr(model, "model", None))
    ):
        raise ValueError("target has no native MTP backbone protocol")  # noqa: TRY004

    def propose(history, anchor, count):
        import mlx.core as mx

        from ..runtime.hybrid_speculative import _mtp_backbone

        if not history or count < 1:
            raise ValueError(
                "native MTP source requires committed context and positive depth"
            )
        target_cache, head_cache = make_prompt_cache(model), model.make_mtp_cache()
        start, end = (
            getattr(model, "mtp_start_cycle", None),
            getattr(model, "mtp_end_cycle", None),
        )
        step = getattr(model, "mtp_step_full_vocab", model.mtp_step)
        try:
            tokens = mx.array([history], mx.uint32)
            _, hidden = _mtp_backbone(model, tokens, target_cache)
            if len(history) > 1:
                step(hidden[:, :-1], tokens[:, 1:], head_cache)
            if callable(start):
                start(head_cache, share_qsa_indices=False)
            hidden = hidden[:, -1:]
            current, out = int(anchor), []
            for _ in range(count):
                logits, hidden = step(
                    hidden, mx.array([[current]], mx.uint32), head_cache
                )
                if bool(mx.any(mx.isnan(logits) | mx.isinf(logits)).item()):
                    raise ValueError("native MTP proposal logits are nonfinite")
                # Full-vocabulary greedy head defines an exact point-mass q,
                # independent of the target request's sampling temperature.
                current = int(mx.argmax(logits[0, -1]).item())
                out.append(current)
            mx.eval(
                hidden,
                [cache.state for cache in target_cache],
                [cache.state for cache in head_cache],
            )
            return out
        finally:
            if callable(end):
                end(head_cache)

    return propose
