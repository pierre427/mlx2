from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from mlx2.conversion.calibration import capture_sample
from mlx2.conversion.importance import ActivationMoments
from mlx2.runtime.models.nemotron35_super import Model as SuperModel
from mlx2.runtime.models.nemotron_h import Model as LanguageModel
from mlx2.runtime.models.switch_layers import SwitchMLP


def tiny_calibration():
    from test_activation_importance_cpu import binding
    from test_nemotron_external_taps_cpu import tiny_model

    args = replace(tiny_model().args, vocab_size=32, hybrid_override_pattern=["M", "E", "*", "E", "M"],
                   moe_intermediate_size=12, n_routed_experts=3, num_experts_per_tok=2,
                   n_group=1, topk_group=1, norm_topk_prob=True, routed_scaling_factor=1.,
                   num_nextn_predict_layers=1, mtp_hybrid_override_pattern=["*", "E"])

    class Tiny(SuperModel):
        def __init__(self):
            LanguageModel.__init__(self, args)
            self.vision = nn.Linear(3, args.hidden_size)
            self.vision_calls = 0

        def encode_images(self, pixels):
            self.vision_calls += 1
            return self.vision(pixels.mean(axis=(2, 3)))[:, None]

    model = Tiny()
    specs = {}
    for path, module in model.named_modules():
        if isinstance(module, SwitchMLP):
            specs["language_model." + path + ".fc1"] = (3, 8)
            specs["language_model." + path + ".fc2"] = (3, 12)
    return model, ActivationMoments(specs, binding())


def test_image_prefill_and_text_mtp_collect_real_assignment_counts_and_restore_modules():
    model, stats = tiny_calibration()
    original, parameters = model.backbone, dict(tree_flatten(model.parameters()))
    images = [np.ones((1, 3, 32, 32), np.float32)]
    receipt = capture_sample(model, stats, mx.array([[2, 18, 3, 4]]), sample_id="image",
                             input_sha256="d" * 64, image_pixels=images)
    assert receipt["modality"] == "image_text" and receipt["vocabulary_projection_rows"] == 1
    assert model.vision_calls == 1
    for name, counts in stats.counts.items():
        assert int(counts.sum()) == (0 if ".mtp." in name else 8)
    receipt = capture_sample(model, stats, mx.array([[2, 3, 4, 5, 6]]), sample_id="text",
                             input_sha256="e" * 64, capture_mtp=True)
    assert receipt["mtp_teacher_forced_pairs"] == 4
    for name, counts in stats.counts.items():
        assert int(counts.sum()) == (8 if ".mtp." in name else 18)
    assert model.backbone is original
    assert {name: value.shape for name, value in tree_flatten(model.parameters())} == {name: value.shape for name, value in parameters.items()}
    assert stats.coverage()["image_text_samples"] == 1 and stats.coverage()["text_samples"] == 1
    with pytest.raises(ValueError, match="new bounded"):
        capture_sample(model, stats, mx.array([[2, 3]]), sample_id="text", input_sha256="e" * 64)


def test_failed_sample_does_not_poison_global_moments_or_leave_wrappers(monkeypatch):
    model, stats = tiny_calibration()
    original = model.backbone
    monkeypatch.setattr(model, "mtp_step", lambda *_: (_ for _ in ()).throw(RuntimeError("failed MTP")))
    with pytest.raises(RuntimeError, match="failed MTP"):
        capture_sample(model, stats, mx.array([[2, 3, 4]]), sample_id="failed", input_sha256="e" * 64, capture_mtp=True)
    assert model.backbone is original and stats.samples == {}
    assert all(not values.any() for values in stats.sums.values())
    assert all(not values.any() for values in stats.counts.values())
    assert not any("Captured" in type(module).__name__ for _, module in model.named_modules())
    with pytest.raises(ValueError, match="text-only"):
        capture_sample(model, stats, mx.array([[2, 18]]), sample_id="bad-image-mtp",
                       input_sha256="e" * 64, image_pixels=[np.ones((1, 3, 32, 32), np.float32)], capture_mtp=True)
