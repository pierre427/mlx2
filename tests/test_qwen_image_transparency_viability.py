from scripts.qwen_image_transparency_probe import probe


def test_qwen_image_edit_transparency_viability_boundary():
    report = probe()
    assert report["rgba_png_preserved"]
    # mlx-serve #647's backend output is already representable by mlx2, but
    # the public edit contract cannot yet ask the backend to produce alpha.
    assert report["edit_api_accepts_transparent"] is False
    assert report["current_verdict"].startswith("request-contract-gap")
