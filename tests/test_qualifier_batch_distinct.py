"""The qualifier's width-4 batch check must be able to reach width 4.

APCv2 serves identical prompts one after another (same-prefix wait), so the
batch check needs one distinct prompt per lane; four copies of one prompt
reached only width 3 and never exceeded a width-3 MTP handoff.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import qualify_serving  # noqa: E402

from mlx2.qualification import QUALIFIER_BATCH_WIDTH  # noqa: E402


def test_batch_check_has_one_distinct_prompt_per_lane():
    subjects = qualify_serving.BATCH_SUBJECTS
    assert len(subjects) == QUALIFIER_BATCH_WIDTH
    assert len(set(subjects)) == len(subjects)


def test_batch_prompts_differ_from_their_first_word():
    # A shared leading phrase is fine (chat template); the subject must differ
    # before the prompt ends so no two lanes are the same prefix.
    prompts = {f"Explain how {s} works in detail." for s in qualify_serving.BATCH_SUBJECTS}
    assert len(prompts) == QUALIFIER_BATCH_WIDTH
