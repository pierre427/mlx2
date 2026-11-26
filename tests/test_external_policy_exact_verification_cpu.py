"""The generic external-draft policy exposes exact_verification (token|block)."""
from types import SimpleNamespace

import pytest

from mlx2.adapters.external_draft_policy import ExternalDraftAdapterMixin

parse = ExternalDraftAdapterMixin._parse_external_policy


def test_block_is_accepted_and_validated():
    host = SimpleNamespace()
    assert parse(host, {"draft_model": "x", "exact_verification": "block",
                        "device_round": {"lazy_draft": True, "decide": True, "sampled_drafts": True}},
                 family="T")
    assert parse(SimpleNamespace(), {"draft_model": "x", "exact_verification": "token"}, family="T")
    with pytest.raises(ValueError, match="'token' or 'block'"):
        parse(SimpleNamespace(), {"draft_model": "x", "exact_verification": "tree"}, family="T")
    for key, value in (("adaptive_verification", {}), ("continuation_pool", {}),
                       ("progressive_verification_tile", 4)):
        with pytest.raises(ValueError, match="linear chain"):
            parse(SimpleNamespace(), {"draft_model": "x", "exact_verification": "block", key: value},
                  family="T")
