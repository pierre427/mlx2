"""The places that name the qualified substrates name the same revisions."""

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
    source pin for mlx-vlm). A private file with source pins is always checked.
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


def test_requirements_pin_the_verified_mlx_release():
    requirements = _pinned_requirements()
    m = re.search(r"^mlx==([0-9]+\.[0-9]+\.[0-9]+)$", requirements, re.M)
    assert m, "mlx must be pinned to one exact verified release"
    version = m.group(1)

    pyproject = (ROOT / "pyproject.toml").read_text()
    serve = re.search(r'^serve = \["mlx>=([^,]+),<([^\"]+)"', pyproject, re.M)
    assert serve
    assert version == serve.group(1)
    major, minor, patch = map(int, version.split("."))
    assert serve.group(2) == f"{major}.{minor}.{patch + 1}"

    from mlx2.runtime.mlx_build import VERIFIED_MLX_BUILDS

    assert version in VERIFIED_MLX_BUILDS
