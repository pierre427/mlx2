"""The three places that name the qualified substrate name the same revisions."""

import re
from pathlib import Path

from mlx2.adapters.mlx_vlm_pin import MLX_VLM_REVISION

ROOT = Path(__file__).resolve().parents[1]


def _mlx_vlm_rev(text):
    m = re.search(r"mlx-vlm\.git@([0-9a-f]{40})", text)
    assert m, "no pinned mlx-vlm revision"
    return m.group(1)


def test_requirements_and_pyproject_pin_the_adapter_revision():
    requirements = (ROOT / "requirements-qualified.txt").read_text()
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert _mlx_vlm_rev(requirements) == MLX_VLM_REVISION
    assert _mlx_vlm_rev(pyproject) == MLX_VLM_REVISION


def test_requirements_pin_the_lab_mlx_fork():
    requirements = (ROOT / "requirements-qualified.txt").read_text()
    m = re.search(r"^mlx @ git\+ssh://\S+@([0-9a-f]{9,40})$", requirements, re.M)
    assert m, "mlx must be pinned to a fork revision, not left to PyPI"
    assert m.group(1) in requirements.split("mlx @")[0], "the header must name the same revision"
