"""An adapter's execution policy is part of the qualification route identity
(sweep P1).

Flash-Next options that are neither environment switches nor batch config
(row_exact_verify, mtp_draft_vocab, tensorfold_qmv_rows) left the serving
settings identical to the default route's, so a default receipt qualified a
row-exact-verify route that changes MTP verify numerics.
"""

import pytest

from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.qualification import APPROVED_QUALIFICATION_HARNESS, load_qualified_route

ROW_EXACT = FlashNextPolicy(row_exact_verify=True, row_exact_window_kernels=False,
                            moe_window_row_exact=False)


def _settings(monkeypatch, policy):
    from route_harness import make_engine, patch_host, tiny_qwen38_mtp

    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()

    class PolicyMixin:
        pass

    PolicyMixin.policy = policy
    engine = make_engine(model, vocab, mtp=True, adapter_mixin=PolicyMixin)
    try:
        return engine, engine.status()["settings"]
    except BaseException:
        engine.close()
        raise


def test_policy_options_enter_the_settings(monkeypatch):
    engine, settings = _settings(monkeypatch, ROW_EXACT)
    try:
        assert settings["adapter_policy"] == ROW_EXACT.as_dict()
        assert settings["adapter_policy"]["row_exact_verify"] is True
    finally:
        engine.close()


def test_a_default_receipt_does_not_qualify_a_row_exact_route(monkeypatch, tmp_path):
    import json

    engine, default_settings = _settings(monkeypatch, FlashNextPolicy())
    engine.close()
    engine, settings = _settings(monkeypatch, ROW_EXACT)
    try:
        descriptor = engine.adapter.descriptor
    finally:
        engine.close()
    assert default_settings != settings
    receipt = tmp_path / "default.json"
    receipt.write_text(json.dumps({
        "qualification_harness": APPROVED_QUALIFICATION_HARNESS,
        "runtime": "r", "artifact": "a", "settings": default_settings,
        "passed": True, "checks": {},
    }))
    with pytest.raises(ValueError, match="does not match serving settings"):
        load_qualified_route(receipt, runtime="r", artifact="a", settings=settings,
                             descriptor=descriptor, name="tiny")
