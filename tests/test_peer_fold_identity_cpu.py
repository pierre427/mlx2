"""FOLD cache identity survives the actual external-drafter policy wrappers."""

import pytest
from test_external_dflash2_cpu import tiny

from mlx2.adapters.external_draft_policy import ExternalDraftAdapterMixin
from mlx2.contracts import Capability, ModelDescriptor


def _bind(mode, fold):
    target, draft = tiny()
    adapter = ExternalDraftAdapterMixin()
    adapter.model = target
    adapter.identity = {"fingerprint": "a" * 64}
    adapter.layout = "target-layout"
    adapter.external_policy = {"draft_model": "tiny", "num_draft": 2}
    if fold is not None:
        adapter.external_policy["fold"] = fold
    if mode == "disabled":
        adapter.external_policy["proposal_composition"] = False
    elif mode == "explicit":
        adapter.external_policy["proposal_composition"] = {"lookback": 64}
    elif mode == "continuation":
        adapter.external_policy["continuation_pool"] = {}
    descriptor = ModelDescriptor(
        model_type="muse", family="muse", variant="tiny", cache_layout="tiny",
        capabilities=frozenset({Capability.TEXT}), state_planes=frozenset(),
    )
    adapter._bind_external_drafter(
        {"fingerprint": "b" * 64}, lambda _record, _target: draft, descriptor,
    )
    if fold is not None and fold is not False:
        assert draft.fold_policy.as_dict()["window"] == fold["window"]
    return adapter


@pytest.mark.parametrize("mode", ["default", "explicit", "disabled", "continuation"])
def test_fold_policy_changes_apc_identity_through_every_wrapper(mode):
    first = _bind(mode, {"window": 3, "alpha": 2, "sliding_layers": "fold"})
    equivalent = _bind(mode, {"sliding_layers": "fold", "alpha": 2.0, "window": 3})
    wider = _bind(mode, {"window": 4, "alpha": 2, "sliding_layers": "fold"})
    sharper = _bind(mode, {"window": 3, "alpha": 3, "sliding_layers": "fold"})
    ordinary = _bind(mode, None)
    assert first.identity == equivalent.identity
    assert first.layout == equivalent.layout
    assert len({a.identity["fingerprint"] for a in (first, wider, sharper, ordinary)}) == 4
    assert len({a.layout for a in (first, wider, sharper, ordinary)}) == 4
    assert {a.identity["target_fingerprint"] for a in (first, wider, sharper, ordinary)} == {"a" * 64}
    assert {a.identity["draft_fingerprint"] for a in (first, wider, sharper, ordinary)} == {"b" * 64}


def test_fold_false_preserves_the_ordinary_route_identity():
    disabled, ordinary = _bind("default", False), _bind("default", None)
    assert disabled.identity == ordinary.identity
    assert disabled.layout == ordinary.layout
