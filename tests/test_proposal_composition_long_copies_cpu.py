"""Opt-in long prompt-lookup copies and expected-token ranking (recon 2026-11-22)."""
import pytest

from mlx2.adapters.external_draft_policy import DEFAULT_EXTERNAL_PROPOSAL_COMPOSITION
from mlx2.runtime.proposal_composition import ComposedDraftModel, ProposalCompositionPolicy


def test_default_policy_identity_has_no_new_keys():
    d = ProposalCompositionPolicy.from_value(dict(DEFAULT_EXTERNAL_PROPOSAL_COMPOSITION)).as_dict()
    assert "prompt_lookup_max_span" not in d and "ranking" not in d
    d = ProposalCompositionPolicy.from_value({"prompt_lookup_max_span": 15,
                                              "ranking": "expected_tokens"}).as_dict()
    assert d["prompt_lookup_max_span"] == 15 and d["ranking"] == "expected_tokens"


@pytest.mark.parametrize("bad", [{"prompt_lookup_max_span": -1}, {"prompt_lookup_max_span": 65},
                                 {"prompt_lookup_max_span": True}, {"ranking": "span"},
                                 {"prompt_lookup": False, "native_mtp": True,
                                  "prompt_lookup_max_span": 8}])
def test_invalid_long_copy_policies(bad):
    with pytest.raises((ValueError, TypeError)):
        ProposalCompositionPolicy.from_value(bad)


class _Backend:
    proposal_distribution = "deterministic_point_mass"

    class config:
        block_size = 8


def test_ranking_expected_tokens_prefers_a_long_copy():
    score = {"external": (996, 1000)}  # ~0.996 committed acceptance
    row = [("prompt_lookup", list(range(15)), "pld_3"), ("external", None, "external")]
    plain = ComposedDraftModel(_Backend(), {"prompt_lookup_max_span": 15})
    assert plain._select_candidate(row, 7, score)[0] == "external"
    ranked = ComposedDraftModel(_Backend(), {"prompt_lookup_max_span": 15,
                                             "ranking": "expected_tokens"})
    assert ranked.maximum_proposal_length == 15
    assert ranked._select_candidate(row, 7, score)[0] == "prompt_lookup"  # 0.5x15 > 0.996x7
    short = [("prompt_lookup", list(range(7)), "pld_3"), ("external", None, "external")]
    assert ranked._select_candidate(short, 7, score)[0] == "external"


def test_backend_depth_alone_without_long_copies():
    assert ComposedDraftModel(_Backend(), {"lookback": 4096}).maximum_proposal_length == 7


def test_inkling2_admits_num_draft_up_to_copy_span_only_with_long_copies():
    """Live-window regression (2ab342726 dspark_copy_long): the adapter's
    load-time check refused num_draft 15 before the drafter cap was reached."""
    from types import SimpleNamespace

    from mlx2.adapters.inkling2 import Inkling2Adapter

    record = {"args": SimpleNamespace(block_size=8, minimum_proposal_length=1)}
    adapter = Inkling2Adapter.__new__(Inkling2Adapter)
    adapter.external_policy = {"draft_model": "dspark", "num_draft": 15,
                               "proposal_composition": {"prompt_lookup": True,
                                                        "prompt_lookup_max_span": 15}}
    adapter._check_num_draft(record)
    assert adapter.external_policy["num_draft"] == 15
    adapter.external_policy = {"draft_model": "dspark", "num_draft": 16,
                               "proposal_composition": {"prompt_lookup": True,
                                                        "prompt_lookup_max_span": 15}}
    with pytest.raises(ValueError, match="block size"):
        adapter._check_num_draft(record)
    adapter.external_policy = {"draft_model": "dspark", "num_draft": 8}
    with pytest.raises(ValueError, match="block size"):
        adapter._check_num_draft(record)
