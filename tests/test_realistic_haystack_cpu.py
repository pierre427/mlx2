"""The shared haystack is varied repository text and never repeats itself."""

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import realistic_haystack as hay  # noqa: E402


def test_haystack_is_varied_text_without_answers():
    text = hay.haystack(200_000)
    words = text.split()
    top_share = Counter(words).most_common(1)[0][1] / len(words)
    assert len(set(words)) > 5_000
    assert top_share < 0.1
    assert "NEEDLE-" not in text and "launch code" not in text.lower()


def test_exclude_markers_drop_files_that_name_the_answer(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("readme text\n")
    (tmp_path / "docs" / "a.md").write_text("mentions LONG_READY here\n")
    (tmp_path / "docs" / "b.md").write_text("plain notes\n")
    corpus = hay.artifact_corpus(tmp_path, exclude=("LONG_READY",))
    assert "plain notes" in corpus and "LONG_READY" not in corpus
    assert "LONG_READY" in hay.artifact_corpus(tmp_path)


def test_haystack_refuses_to_wrap_instead_of_repeating(tmp_path):
    (tmp_path / "README.md").write_text("short\n")
    with pytest.raises(ValueError, match="corpus has"):
        hay.haystack(10_000, root=tmp_path)


def test_bury_places_the_fact_once_on_its_own_line():
    text = "".join(f"line {i}\n" for i in range(30))
    out = hay.bury(text, "FACT")
    assert out.count("FACT") == 1 and "\nFACT\n" in out
    assert out.index("FACT") < len(out) // 2
