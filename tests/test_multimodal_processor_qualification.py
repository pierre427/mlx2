"""Static assertions over the retained host-only multimodal processor receipt."""

import json
from pathlib import Path


def test_real_processor_receipt_is_cpu_only_aligned_and_bounded():
    root = Path(__file__).parents[1]
    receipt = json.loads(
        (
            root
            / "qualification/runs/multimodal-processors-20260918/receipt.json"
        ).read_text()
    )
    assert receipt["schema"] == "mlx2.multimodal-processor-qualification.v1"
    assert receipt["passed"] is True
    assert receipt["gpu_used"] is False
    assert receipt["metal_imported"] is False
    assert receipt["torch_default_device"] == "cpu"
    assert receipt["scope"] == "processor_only_no_model_weights"

    gemma = receipt["checks"]["gemma3n_video_processor"]
    assert gemma["passed"] is True
    assert gemma["frames"] == len(gemma["timestamps_seconds"])
    assert gemma["pixel_values_shape"][0] == gemma["frames"]
    assert gemma["image_soft_tokens"] == 256 * gemma["frames"]
    assert gemma["media_token_end"] > gemma["image_soft_tokens"]

    minicpmo = receipt["checks"]["minicpmo_media_processor"]
    assert minicpmo["passed"] is True
    assert minicpmo["vision_slices"] == len(minicpmo["image_bounds"])
    assert minicpmo["vision_slices"] == len(minicpmo["pixel_values_shapes"])
    assert minicpmo["audio_chunks"] == len(minicpmo["audio_bounds"])
    assert minicpmo["audio_chunks"] == len(minicpmo["audio_chunk_samples"])
    assert sum(minicpmo["audio_chunk_samples"]) == 36_000
    assert minicpmo["media_token_end"] == minicpmo["audio_bounds"][-1][1]
