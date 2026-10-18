"""Nemotron prefill levers: the NAX gather policy, the int8 selector, the phase."""
import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mlx2.adapters.nemotron35_super import (
    NEMOTRON_INT8_PREFILL_SCOPES,
    Nemotron35SuperAdapter,
    _moe_nax_gather_report,
    _split_moe_nax_gather,
    nemotron_int8_prefill_select,
)
from mlx2.adapters.nemotron35_super_vl import Nemotron35SuperVLAdapter
from mlx2.runtime import int8_prefill
from mlx2.runtime.models import moe_nax_gather as nax


def test_nax_policy_defaults_off_and_refuses_fused():
    assert _split_moe_nax_gather(None) == (None, "off")
    assert _split_moe_nax_gather({"num_draft": 2}) == ({"num_draft": 2}, "off")
    original = {"moe_nax_gather": "gather", "num_draft": 2}
    assert _split_moe_nax_gather(original) == ({"num_draft": 2}, "gather")
    assert "moe_nax_gather" in original  # the caller's policy is not mutated
    with pytest.raises(ValueError, match="off or gather"):
        _split_moe_nax_gather({"moe_nax_gather": "fused"})


def test_nax_report_carries_engagement_keys_when_selected():
    assert _moe_nax_gather_report("off")["selected"] is False
    report = _moe_nax_gather_report("gather")
    assert report["selected"] is True and report["qualified"] is False
    assert "calls" in report and "fallbacks" in report  # read by the qualifier


@pytest.mark.parametrize("adapter", [Nemotron35SuperAdapter, Nemotron35SuperVLAdapter])
def test_int8_prefill_declared_on_both_adapters(adapter):
    assert int8_prefill.adapter_scopes(adapter) == frozenset(NEMOTRON_INT8_PREFILL_SCOPES)
    assert adapter.int8_prefill_select is not None


@pytest.mark.parametrize(
    "path, mlp, all_",
    [
        ("backbone.layers.1.mixer.shared_experts.up_proj", True, True),
        ("backbone.layers.1.mixer.shared_experts.down_proj", True, True),
        ("backbone.layers.0.mixer.in_proj", False, True),
        ("backbone.layers.0.mixer.out_proj", False, True),
        ("backbone.layers.7.mixer.q_proj", False, True),
        ("backbone.layers.7.mixer.k_proj", False, True),
        ("backbone.layers.1.mixer.switch_mlp.fc1", False, False),
        ("backbone.layers.1.mixer.fc1_latent_proj", False, False),
        ("backbone.layers.1.mixer.gate", False, False),
        ("lm_head", False, False),
        ("mtp.layers.0.mixer.shared_experts.up_proj", False, False),
        ("vision_model.blocks.0.attn.qkv", False, False),
    ],
)
def test_int8_selector_keeps_routed_heads_and_media_stock(path, mlp, all_):
    assert nemotron_int8_prefill_select("mlp")(path, None) is mlp
    assert nemotron_int8_prefill_select("all")(path, None) is all_


def test_trunk_forward_opens_the_prefill_phase(monkeypatch):
    from tests.test_nemotron_external_taps_cpu import tiny_model

    model = tiny_model()
    model.eval()
    seen = []
    backbone = model.backbone
    original = type(backbone)._forward

    def spy(self, *args):
        seen.append(nax._PREFILL.get())
        return original(self, *args)

    monkeypatch.setattr(type(backbone), "_forward", spy)
    old = nax.set_mode("gather")
    try:
        cache = model.make_cache()
        model(mx.array([[1, 2, 3, 4]]), cache=cache)
        model(mx.array([[5]]), cache=cache)
    finally:
        nax.set_mode(old)
    assert seen == [True, False]
    # Off: the scope stays undecided and the call sites never admit NAX.
    seen.clear()
    model(mx.array([[1, 2]]), cache=model.make_cache())
    assert seen == [None]
