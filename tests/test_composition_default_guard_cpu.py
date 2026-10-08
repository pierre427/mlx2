"""The external-route composition default must respect backend exactness.

qualify-1007-extra: Laguna XS 2.1 + its DFlash drafter died at startup with
"proposal composition requires a backend with exact proposal-law support"
because the shared binder attached the default composition to any chain
drafter exposing ``draft_distributions``, without checking that the drafter
publishes an exact proposal law.  Strictly CPU, no model weights.
"""

import pytest

from mlx2.adapters.external_draft_policy import (
    COMPOSITION_DEFAULT_SKIPPED_NO_EXACT_LAW,
    ExternalDraftAdapterMixin,
)
from mlx2.contracts import Capability, ModelDescriptor
from mlx2.runtime.proposal_composition import ComposedDraftModel

DESCRIPTOR = ModelDescriptor(
    model_type="fake",
    family="fake",
    variant="tiny",
    cache_layout="fake",
    capabilities=frozenset({Capability.TEXT}),
    state_planes=frozenset(),
)


class _LawlessChainDrafter:
    """Chain drafter shaped like LagunaDFlashDraftModel: it exposes
    ``draft_distributions`` but declares no exact ``proposal_distribution``."""

    def draft_distributions(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("not called at bind time")


class _ExactChainDrafter(_LawlessChainDrafter):
    proposal_distribution = "stochastic_exact_law"


def _adapter(policy):
    adapter = ExternalDraftAdapterMixin()
    adapter.model = object()
    adapter.identity = {"fingerprint": "target-revision"}
    adapter.layout = "target-layout"
    adapter.external_policy = dict(policy)
    return adapter


def _bind(adapter, drafter):
    adapter._bind_external_drafter(
        {"fingerprint": "head-revision"}, lambda record, target: drafter, DESCRIPTOR
    )


def test_default_composition_skipped_for_backend_without_exact_law():
    drafter = _LawlessChainDrafter()
    adapter = _adapter({"draft_model": "laguna-dflash", "num_draft": 3})
    _bind(adapter, drafter)  # raised ValueError at startup before the fix
    assert adapter.draft_model is drafter
    assert "proposal_composition" not in adapter.external_policy
    assert adapter.skipped_route_defaults == {
        "proposal_composition": COMPOSITION_DEFAULT_SKIPPED_NO_EXACT_LAW
    }


def test_explicit_composition_on_lawless_backend_still_fails_closed():
    adapter = _adapter(
        {
            "draft_model": "laguna-dflash",
            "num_draft": 3,
            "proposal_composition": {"prompt_lookup": True},
        }
    )
    with pytest.raises(ValueError, match="exact proposal-law support"):
        _bind(adapter, _LawlessChainDrafter())


def test_default_composition_still_attaches_for_exact_backend():
    adapter = _adapter({"draft_model": "dflash2", "num_draft": 3})
    _bind(adapter, _ExactChainDrafter())
    assert isinstance(adapter.draft_model, ComposedDraftModel)
    assert adapter.external_policy["proposal_composition"]["prompt_lookup"] is True
    assert not getattr(adapter, "skipped_route_defaults", None)


def test_adapter_skipped_default_reaches_route_receipt(monkeypatch):
    from route_harness import make_engine, patch_host, tiny_qwen38_mtp

    class Mixin:
        skipped_route_defaults = {
            "proposal_composition": COMPOSITION_DEFAULT_SKIPPED_NO_EXACT_LAW
        }

    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    engine = make_engine(
        model, vocab, mtp=True, adapter_mixin=Mixin,
        skipped_route_defaults={"mtp_ordinary_handoff": "why"},
    )
    try:
        assert engine.status()["settings"]["skipped_route_defaults"] == {
            "mtp_ordinary_handoff": "why",
            "proposal_composition": COMPOSITION_DEFAULT_SKIPPED_NO_EXACT_LAW,
        }
    finally:
        engine.close()
