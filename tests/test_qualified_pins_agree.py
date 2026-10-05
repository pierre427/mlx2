"""The three places that name the qualified substrate name the same revisions."""

import re
from pathlib import Path

import pytest

from mlx2.adapters.mlx_vlm_pin import MLX_VLM_REVISION

ROOT = Path(__file__).resolve().parents[1]


def _mlx_vlm_rev(text):
    m = re.search(r"mlx-vlm\.git@([0-9a-f]{40})", text)
    assert m, "no pinned mlx-vlm revision"
    return m.group(1)


def _pinned_requirements():
    """The private pinned requirements, or skip on the public mirror.

    The public mirror carries an unpinned requirements-qualified.txt (no git+
    lines: the lab MLX fork is not reachable from there). A private file with
    git+ pins is always checked.
    """
    path = ROOT / "requirements-qualified.txt"
    text = path.read_text() if path.is_file() else ""
    if "git+" not in text:
        pytest.skip("requirements-qualified.txt carries no git+ pins "
                    "(private pinned file absent, e.g. public mirror)")
    return text


def test_requirements_and_pyproject_pin_the_adapter_revision():
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert _mlx_vlm_rev(pyproject) == MLX_VLM_REVISION
    requirements = _pinned_requirements()
    assert _mlx_vlm_rev(requirements) == MLX_VLM_REVISION


def test_requirements_pin_the_lab_mlx_fork():
    requirements = _pinned_requirements()
    m = re.search(r"^mlx @ git\+ssh://\S+@([0-9a-f]{9,40})$", requirements, re.M)
    assert m, "mlx must be pinned to a fork revision, not left to PyPI"
    assert m.group(1) in requirements.split("mlx @")[0], "the header must name the same revision"
