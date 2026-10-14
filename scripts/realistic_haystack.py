"""Realistic, varied haystack text for long-context and recall probes.

Reasoning models do extremely poorly on repeated-string input: on a prompt of
one phrase repeated thousands of times they spend their answer budget
describing the repetition instead of answering (LFM2.5-24B-A2B, 2026-10-14;
North-Mini-Code collapsed on ~30K copies of ``data``, 2026-09-18).  A probe
that fills its context with repetition measures that pathology, not serving.
Every recall, needle or cache-marker probe takes its filler from here instead:
the repository's own README, docs and runtime sources, which are real,
varied text and fixed per snapshot.

Repetition stays available only as an explicitly named pathology probe in the
harness that needs it (``thermal_ladder.py --filler archival``).
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(
    os.environ.get("MLX2_CAMPAIGN_ROOT", str(Path(__file__).resolve().parents[1]))
).resolve()

# A file that names an answer a probe asks for is left out, so the buried
# fact is the only answer in the prompt.
DEFAULT_EXCLUDE = ("launch code", "NEEDLE-")

_CORPORA: dict[tuple[str, tuple[str, ...]], str] = {}


def corpus_paths(root: Path) -> list[Path]:
    """README, docs/*.md and src/mlx2/runtime/*.py, in a fixed order."""
    paths = [root / "README.md", *sorted((root / "docs").glob("*.md"))]
    paths += sorted((root / "src" / "mlx2" / "runtime").glob("*.py"))
    return [path for path in paths if path.is_file()]


def artifact_corpus(root: Path | str | None = None, exclude: tuple[str, ...] = ()) -> str:
    """The snapshot's shared artifacts as one text, cached per root.

    A file whose text contains any ``exclude`` marker (case-insensitively for
    lowercase markers, exactly otherwise), or a default one, is skipped.
    """
    root = Path(root or ROOT).resolve()
    markers = tuple(dict.fromkeys((*DEFAULT_EXCLUDE, *exclude)))
    key = (str(root), markers)
    if key not in _CORPORA:
        parts = []
        for path in corpus_paths(root):
            body = path.read_text(errors="replace")
            if any(
                (marker in body.lower()) if marker.islower() else (marker in body)
                for marker in markers
            ):
                continue
            parts.append(f"\n--- Shared artifact: {path.relative_to(root)} ---\n{body}\n")
        _CORPORA[key] = "".join(parts)
    return _CORPORA[key]


def haystack(
    chars: int,
    *,
    root: Path | str | None = None,
    exclude: tuple[str, ...] = (),
    offset: int = 0,
) -> str:
    """``chars`` characters of varied corpus text starting at ``offset``.

    Never wraps around: a probe that needs more text than the corpus holds
    fails instead of silently repeating it.
    """
    corpus = artifact_corpus(root, exclude)
    if chars < 0 or offset < 0 or offset + chars > len(corpus):
        raise ValueError(
            f"haystack needs {offset + chars} characters; the corpus has {len(corpus)}"
        )
    return corpus[offset : offset + chars]


def bury(text: str, fact: str, fraction: float = 1 / 3) -> str:
    """Insert ``fact`` on its own line at the newline nearest ``fraction`` in."""
    cut = text.rfind("\n", 0, int(len(text) * fraction)) + 1 if text else 0
    return f"{text[:cut]}\n{fact}\n{text[cut:]}"
