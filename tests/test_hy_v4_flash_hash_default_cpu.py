"""``engram_hash: auto`` is the default: device hashing when exact, else host (CPU)."""

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from test_hy_v4_flash_cpu import build, checkpoint, tiny_config  # noqa: E402

from mlx2.adapters import hy_v4_flash as A  # noqa: E402
from mlx2.runtime.models import hy_v4_flash_device_hash as D  # noqa: E402


@pytest.fixture()
def model():
    config = tiny_config()
    m, _ = build(config, checkpoint(config))
    return m


def test_default_is_auto_and_policies_parse():
    assert A.ENGRAM_HASH_DEFAULT == "auto"
    for mode in ("auto", "host", "device"):
        A._parse_policy({"engram_hash": mode})
    with pytest.raises(ValueError, match="engram_hash"):
        A._parse_policy({"engram_hash": "gpu"})


def test_auto_selects_device_when_the_geometry_is_exact(model):
    model.engram_hash, model.engram_hash_requested, model.engram_hash_auto_reason = "host", "auto", None
    assert A.resolve_engram_hash(model) == "device"
    assert model.engram_hash == "device" and model.engram_hash_auto_reason is None


def test_auto_falls_back_to_host_with_the_reason(model, monkeypatch):
    def refuse(*args, **kwargs):
        raise ValueError("Engram multiplier bound exceeds the exact device range")

    monkeypatch.setattr(D, "DeviceEngramHasher", refuse)
    model.engram_hash, model.engram_hash_requested, model.engram_hash_auto_reason = "host", "auto", None
    model._device_hasher = None
    assert A.resolve_engram_hash(model) == "host"
    assert "exact device range" in model.engram_hash_auto_reason


def test_explicit_modes_are_left_alone(model):
    for mode in ("host", "device"):
        model.engram_hash, model.engram_hash_requested = mode, mode
        assert A.resolve_engram_hash(model) == mode
