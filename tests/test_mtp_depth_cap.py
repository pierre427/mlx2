"""The self-MTP depth cap: unchanged by default, raised only on opt-in."""

import pytest

from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.adapters.mtp_depth_cap import (
    DEFAULT_SELF_MTP_DEPTH_CAP,
    MAX_SELF_MTP_DEPTH_CAP,
    self_mtp_depth_cap,
    validate_self_mtp_num_draft,
)

ADAPTERS = ("qwen38_27b", "qwen36_35b", "xing", "flash_next_policy")


def test_default_cap_is_three_and_message_is_unchanged(monkeypatch):
    monkeypatch.delenv("MLX2_MTP_DEPTH_CAP", raising=False)
    assert self_mtp_depth_cap() == DEFAULT_SELF_MTP_DEPTH_CAP == 3
    for depth in (1, 2, 3):
        assert validate_self_mtp_num_draft(depth) == depth
    for depth in (0, 4, 5, 16, -1):
        with pytest.raises(ValueError, match=r"^num_draft must be 1, 2, or 3$"):
            validate_self_mtp_num_draft(depth)


@pytest.mark.parametrize("value", [True, False, 2.0, "2", None])
def test_default_cap_still_rejects_non_integers(monkeypatch, value):
    monkeypatch.delenv("MLX2_MTP_DEPTH_CAP", raising=False)
    with pytest.raises(ValueError, match="num_draft must be 1, 2, or 3"):
        validate_self_mtp_num_draft(value)


def test_opt_in_raises_the_cap_and_validates_its_own_bound(monkeypatch):
    monkeypatch.setenv("MLX2_MTP_DEPTH_CAP", "6")
    assert self_mtp_depth_cap() == 6
    for depth in (1, 3, 4, 5, 6):
        assert validate_self_mtp_num_draft(depth) == depth
    for depth in (0, 7, True, 4.0):
        with pytest.raises(ValueError, match=r"between 1 and 6 \(MLX2_MTP_DEPTH_CAP\)"):
            validate_self_mtp_num_draft(depth)


def test_opt_in_cannot_be_lowered_below_one_or_raised_without_bound(monkeypatch):
    monkeypatch.setenv("MLX2_MTP_DEPTH_CAP", "1")
    assert self_mtp_depth_cap() == 1
    assert validate_self_mtp_num_draft(1) == 1
    with pytest.raises(ValueError, match=r"between 1 and 1 \(MLX2_MTP_DEPTH_CAP\)"):
        validate_self_mtp_num_draft(2)
    for bad in ("0", "17", "-3", "three", "3.5", " "):
        monkeypatch.setenv("MLX2_MTP_DEPTH_CAP", bad)
        with pytest.raises(ValueError, match="MLX2_MTP_DEPTH_CAP must be an integer"):
            self_mtp_depth_cap()
    monkeypatch.setenv("MLX2_MTP_DEPTH_CAP", str(MAX_SELF_MTP_DEPTH_CAP))
    assert self_mtp_depth_cap() == MAX_SELF_MTP_DEPTH_CAP


def test_empty_opt_in_is_the_default(monkeypatch):
    monkeypatch.setenv("MLX2_MTP_DEPTH_CAP", "")
    assert self_mtp_depth_cap() == DEFAULT_SELF_MTP_DEPTH_CAP


def test_flash_next_policy_follows_the_cap(monkeypatch):
    monkeypatch.delenv("MLX2_MTP_DEPTH_CAP", raising=False)
    assert FlashNextPolicy(num_draft=3).num_draft == 3
    with pytest.raises(ValueError, match="num_draft must be 1, 2, or 3"):
        FlashNextPolicy(num_draft=4)
    monkeypatch.setenv("MLX2_MTP_DEPTH_CAP", "5")
    assert FlashNextPolicy(num_draft=5).num_draft == 5
    with pytest.raises(ValueError, match=r"between 1 and 5"):
        FlashNextPolicy(num_draft=6)


@pytest.mark.parametrize("module", ADAPTERS)
def test_no_adapter_keeps_a_private_copy_of_the_cap(module):
    """The four self-MTP adapters must route through the shared guard.

    A second inline ``1 <= num_draft <= 3`` would silently re-impose the cap on
    one route while the opt-in appeared to work everywhere else.
    """
    import importlib
    import pathlib

    source = pathlib.Path(
        importlib.import_module(f"mlx2.adapters.{module}").__file__
    ).read_text()
    assert "validate_self_mtp_num_draft" in source
    assert "num_draft must be 1, 2, or 3" not in source
    assert "num_draft <= 3" not in source


@pytest.mark.parametrize("num_draft", [4, 5])
def test_serving_refuses_at_startup_a_depth_lane_admission_cannot_cost(monkeypatch, num_draft):
    """An opted-in depth past the calibrated verify transients fails the load.

    Lane admission costs a self-MTP lane with ``TRANSIENT_SCALE[num_draft]``;
    a depth it has no calibration for used to start the server and then
    refuse every request with 400.
    """
    from mlx2 import memory, serving
    from mlx2.runtime import os_memory
    from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController
    from test_apc_hits_hybrid_gdn_self_mtp import make_adapter, run, tiny_qwen38_mtp

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "src"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    assert max(SelfMTPLaneAdmissionController.TRANSIENT_SCALE) == 4
    depth = validate_self_mtp_num_draft(num_draft, environ={"MLX2_MTP_DEPTH_CAP": "8"})
    model, vocab = tiny_qwen38_mtp()

    class Adapter(make_adapter(model, vocab)):
        def execution_config(self, *, max_lanes, prefill_step):
            config = super().execution_config(max_lanes=max_lanes, prefill_step=prefill_step)
            return {**config, "num_draft": depth}

    engine = serving.ServingEngine(
        "tiny", adapter_factory=Adapter, qualification_mode=True, mtp=True,
        max_lanes=1, prefill_step=16,
    )
    try:
        if num_draft == 4:
            assert engine.ready.wait(60), engine.error
            tokens, _receipt, _job = run(engine, range(1, 20), max_tokens=4)
            assert len(tokens) == 4
        else:
            engine.thread.join(60)
            assert not engine.ready.is_set()
            assert engine.error == (
                "ValueError: self-MTP num_draft 5 has no calibrated lane-admission "
                "verify transient; calibrated depths are 1 to 4"
            )
    finally:
        engine.close()
