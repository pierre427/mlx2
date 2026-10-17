"""Transactional image/text routed calibration in a dedicated model process.

This is numerical capture, never a serving/performance path. Its GPU caller
must supply the bounded ownership, ordinary reference and host guards. CPU
tests exercise the same image prefill and native MTP math on a tiny target.
"""
from __future__ import annotations

import mlx.core as mx
from mlx import nn

from ..runtime.prefill_checkpoint_rows import prefill_checkpoint_rows
from .capture import RoutedCapture
from .importance import ActivationMoments


class _ObservedBackbone(nn.Module):
    def __init__(self, inner, sink):
        super().__init__()
        self.inner = inner
        self._sink = sink

    @property
    def embeddings(self):
        return self.inner.embeddings

    @property
    def layers(self):
        return self.inner.layers

    def __call__(self, *args, **kwargs):
        hidden = self.inner(*args, **kwargs)
        self._sink["hidden"] = hidden
        return hidden


def capture_sample(model, collector, tokens, *, sample_id, input_sha256,
                   image_pixels=None, capture_mtp=False):
    """Commit one sample's moments only after its complete forward is finite.

    Image inputs use the owned image-prefill path. MTP coverage comes from
    separate text teacher-forced packets; unsupported image/MTP semantics
    are rejected. Vocabulary projection is bounded to one checkpoint row.
    """
    if (tokens.ndim != 2 or tokens.shape[0] != 1 or not 1 <= tokens.shape[1] <= 8192
            or tokens.dtype not in (mx.int32, mx.int64, mx.uint32)
            or sample_id in collector.samples):
        raise ValueError("Calibration requires a new bounded single-conversation token sequence")
    if capture_mtp and (image_pixels is not None or tokens.shape[1] < 2 or not hasattr(model, "mtp")):
        raise ValueError("MTP calibration requires a text-only teacher-forced packet")
    delta = ActivationMoments(collector.specs, collector.binding)
    delta.record_sample(sample_id, modality="text" if image_pixels is None else "image_text",
                        input_sha256=input_sha256)
    cache, sink, original = model.make_cache(), {}, model.backbone
    with RoutedCapture(model, delta):
        try:
            model.backbone = _ObservedBackbone(original, sink)
            with prefill_checkpoint_rows([tokens.shape[1]], width=tokens.shape[1]) as checkpoint:
                output = model.prefill_forward(tokens, cache=cache, image_pixels=image_pixels)
            if checkpoint.applied != 1 or output.ndim != 3 or output.shape[:2] != (1, 1):
                raise ValueError("Calibration target did not apply bounded checkpoint-row projection")
            hidden = sink.get("hidden")
            if hidden is None or hidden.shape[:2] != tokens.shape:
                raise ValueError("Calibration target hidden-state geometry differs")
            finite = mx.all(mx.isfinite(output)) & mx.all(mx.isfinite(hidden))
            mtp_pairs = 0
            if capture_mtp:
                # Do not evaluate the full sequence x vocabulary logits. The
                # returned hidden depends on every trained MTP expert block.
                _, mtp_hidden = model.mtp_step(hidden[:, :-1], tokens[:, 1:], model.make_mtp_cache())
                finite = finite & mx.all(mx.isfinite(mtp_hidden))
                mtp_pairs = tokens.shape[1] - 1
            mx.eval(finite, [entry.state for entry in cache])
            if not bool(finite.item()):
                raise ValueError("Nonfinite calibration forward; sample moments were not committed")
        finally:
            # Restore the original path before RoutedCapture restores expert
            # leaves by their original names. No wrappers escape the sample.
            model.backbone = original
    collector.merge(delta)
    return {"sample_id": sample_id, "modality": delta.samples[sample_id]["modality"],
            "tokens": tokens.shape[1], "mtp_teacher_forced_pairs": mtp_pairs,
            "vocabulary_projection_rows": 1}
