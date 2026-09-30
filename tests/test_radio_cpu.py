"""RADIO recipe and loading regressions; explicitly CPU-only MLX execution."""

import pytest

from mlx2.adapters.radio_config import RadioConfig


def recipe(**args):
    return {
        "args": {
            "model": "vit_base_patch16_224",
            "cls_token_per_teacher": True,
            "teachers": [{"name": "a"}, {"name": "b", "use_summary": False}],
            "register_multiple": 8,
            **args,
        },
        "patch_size": 16,
        "max_resolution": 32,
        "preferred_resolution": [32, 32],
    }


def test_recipe_explicit_v3_registers():
    cfg = RadioConfig.from_dict(recipe(register_multiple=0, cpe_num_registers=6))
    assert (cfg.num_cls_tokens, cfg.num_registers, cfg.summary_idxs) == (2, 6, (0,))


def test_multiple_includes_full_register_group():
    cfg = RadioConfig.from_dict(recipe(register_multiple=2))
    assert cfg.num_registers == 2


@pytest.mark.parametrize(
    "change",
    [
        {"args": {"model": "unknown"}},
        {"feature_normalizer_config": {"dim": 1}},
        {"vitdet_window_size": 16},
        {"adaptor_names": ["clip"]},
        {"patch_size": 14},
        {"preferred_resolution": [31, 32]},
    ],
)
def test_unsupported_variants_fail_closed(change):
    with pytest.raises(ValueError):
        RadioConfig.from_dict({**recipe(), **change})


@pytest.mark.parametrize(
    "version,aligned",
    [("radio_v2.5-b", True), ("c-radio_v3-h", True), ("c-radio_v4-h", False)],
)
def test_generation_interpolation(version, aligned):
    assert (
        RadioConfig.from_dict({**recipe(), "version": version}).align_corners is aligned
    )


def test_encoder_runtime_in_isolated_cpu_process():
    # Artifact-inspection tests require a process that has never imported MLX.
    # Keep numerical encoder tests out of that process rather than unloading an
    # extension module or weakening the existing registry invariant.
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(root / "tests/fixtures/radio_runtime_checks.py"),
            "-q",
        ],
        cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root / "src")),
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
