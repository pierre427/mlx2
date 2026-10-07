"""Flash-Next selects the QSA stage-one selector through its policy only.

configure_environment() clears inherited MLX_QWEN* variables so a lab
experiment cannot silently change the serving profile.  The 2026-10-07 GVR
A/B set MLX_QWEN4_QSA_STAGE1_DIRECT_SELECTOR=gvr in the server environment;
the adapter cleared it and both arms ran the radix selector.  The policy key
is the supported route.
"""

from __future__ import annotations

import os

import pytest

from mlx2.adapters.flash_next import configure_environment
from mlx2.adapters.flash_next_policy import FlashNextPolicy

KEY = "MLX_QWEN4_QSA_STAGE1_DIRECT_SELECTOR"


@pytest.fixture
def restore_environ():
    saved = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_an_inherited_selector_variable_is_cleared(tmp_path, restore_environ):
    os.environ[KEY] = "gvr"
    profile = configure_environment(tmp_path, FlashNextPolicy())
    assert KEY not in os.environ
    assert KEY not in profile


@pytest.mark.parametrize("mode", ["direct8", "direct4", "gvr"])
def test_the_policy_key_selects_the_selector(tmp_path, restore_environ, mode):
    policy = FlashNextPolicy.from_mapping({"qsa_stage1_direct_selector": mode})
    profile = configure_environment(tmp_path, policy)
    assert os.environ[KEY] == mode
    assert profile[KEY] == mode


def test_off_leaves_environment_and_receipt_unchanged():
    policy = FlashNextPolicy.from_mapping({"qsa_stage1_direct_selector": "off"})
    assert KEY not in policy.environment()
    assert policy.environment() == FlashNextPolicy().environment()


def test_unknown_selector_fails_closed():
    with pytest.raises(ValueError, match="qsa_stage1_direct_selector"):
        FlashNextPolicy.from_mapping({"qsa_stage1_direct_selector": "fast"})


def test_selection_instrumentation_is_policy_driven(tmp_path, restore_environ):
    policy = FlashNextPolicy.from_mapping(
        {"qsa_stage1_select_timing": True, "qsa_stage1_gvr_count_paths": True}
    )
    configure_environment(tmp_path, policy)
    assert os.environ["MLX_QWEN4_QSA_STAGE1_SELECT_TIMING"] == "1"
    assert os.environ["MLX_QWEN4_QSA_STAGE1_GVR_COUNT_PATHS"] == "1"
