"""The M3 tower/APCv2 gate refuses media reuse across changed pixels or text."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import qualify_media_tower_apcv2 as gate

SPANS = {"changed_tail": (4, 7, 12), "changed_lead": (6, 9, 14), "changed_pixels": (4, 7, 12)}


def _row(cached, prompt=12):
    return {"cached_tokens": cached, "prompt_tokens": prompt}


def test_prefix_only_reuse_passes():
    gate.check_media_reuse(_row(8), _row(6, 14), _row(4), SPANS)


@pytest.mark.parametrize(
    ("tail", "lead", "pixels"),
    [
        (_row(8), _row(6, 14), _row(11)),   # a token-only cache reused the changed pixels
        (_row(8), _row(6, 14), _row(5)),    # a restore inside the media span
        (_row(8), _row(7, 14), _row(4)),    # changed leading text reused media KV
        (_row(6), _row(6, 14), _row(4)),    # the tail branch did not get past the media
        (_row(12), _row(6, 14), _row(4)),   # the tail branch recomputed nothing
    ],
)
def test_reuse_across_changed_media_fails(tail, lead, pixels):
    """The gate only checked that a zero-reuse row was cold, so a cache keyed
    on tokens alone (99 of 100 reused across different pixels) passed."""
    with pytest.raises(AssertionError):
        gate.check_media_reuse(tail, lead, pixels, SPANS)


def test_media_span_reads_the_prepared_prompt():
    adapter = SimpleNamespace(
        identity={"config": {"image_token_id": 9, "video_token_id": None}},
        prepare_multimodal_request=lambda request: {
            "_mlx2_prompt_tokens": [1, 2, 9, 9, 9, 3, 4], "_mlx2_media_token_end": 5},
    )
    assert gate.media_span(adapter, {}) == (2, 5, 7)


def test_the_hashed_source_must_be_the_one_that_runs(tmp_path):
    """Hashes were taken under ROOT/src while mlx2 was imported from sys.path."""
    import mlx2

    with pytest.raises(AssertionError, match="executing mlx2"):
        gate.executed_source_root(tmp_path)
    assert gate.executed_source_root(Path(mlx2.__file__).resolve().parents[1].parent) is not None
