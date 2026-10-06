"""Host-only gates for the fail-closed exact-prefix serving boundary."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from mlx2.runtime.exact_prefix_serving import (
    ExactPrefixServingUnsupported,
    open_exact_prefix_serving_session,
)
from mlx2.runtime.exact_prefix_state_binding import (
    validate_exact_prefix_state_binding,
)


def _contract(**changes):
    contract = {
        "schema": "mlx2.exact-prefix-cascade-contract.v1",
        "verification_order": "longest_first",
        "invalid_sibling_pruning": True,
        "accepted_prefix_state": "b1_tokenwise_hybrid_transaction",
        "shared_prefix_reuse": "committed_target_cache_clone",
        "common_tokens_recomputed": False,
        "request_private_only": True,
        "apcv2_publication": False,
        "mtp_cache_reuse": False,
        "twotower_reuse": False,
        "implemented": True,
        "serving_route_implemented": False,
        "http_request_selection_implemented": False,
        "state_binding": (
            "artifact_source_layout_request_owner_checkpoint_planes_b1"
        ),
        "target_observation": "unimplemented_sampler_processors_rng_history",
        "qualified": False,
        "selected": False,
        "observed_used": False,
    }
    contract.update(changes)
    return contract


def _binding(owner_request_id, **changes):
    binding = {
        "artifact_fingerprint": "a" * 64,
        "source_revision": "f6415871606005626a2ac36c0c5630d5b5a8568e",
        "cache_layout": "test-hybrid-cache",
        "request_id": owner_request_id,
        "ownership": "request_private",
        "checkpoint_kind": "live_authoritative_exact",
        "execution_domain": "ordinary_target_b1",
        "checkpoint_position": 3,
        "state_planes": ["attention_kv", "recurrent"],
        "batch_size": 1,
        "mtp_state": "absent",
        "twotower_state": "absent",
    }
    binding.update(changes)
    return binding


class _Adapter:
    exact_prefix_cascade_contract = staticmethod(_contract)


def test_every_internal_open_refuses_missing_target_observation_handoff():
    with pytest.raises(
        ExactPrefixServingUnsupported,
        match="sampler, processors, RNG and history",
    ):
        open_exact_prefix_serving_session(
            _Adapter(),
            (),
            ((1, 2),),
            request_id="request",
            maximum=2,
            state_binding=_binding("request"),
        )


def test_adapter_cannot_claim_a_served_route_without_the_generic_contract():
    class Overclaim:
        exact_prefix_cascade_contract = staticmethod(
            lambda: _contract(serving_route_implemented=True)
        )

    with pytest.raises(ExactPrefixServingUnsupported, match="serving_route_implemented"):
        open_exact_prefix_serving_session(
            Overclaim(),
            (),
            ((1,),),
            request_id="overclaim",
            maximum=1,
            state_binding=_binding("overclaim"),
        )

    with pytest.raises(ExactPrefixServingUnsupported, match="does not declare"):
        open_exact_prefix_serving_session(
            object(),
            (),
            ((1,),),
            request_id="missing",
            maximum=1,
            state_binding=_binding("missing"),
        )


def test_serving_engine_seam_remains_explicit_but_is_documented_fail_closed():
    source = Path("src/mlx2/serving.py").read_text()
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "open_exact_prefix_cascade"
    )
    argument_names = {argument.arg for argument in method.args.kwonlyargs}
    rendered = ast.unparse(method)
    assert {"request_id", "maximum", "state_binding"} <= argument_names
    assert "state_binding=state_binding" in rendered
    assert "mtp_active=bool(getattr(self, 'mtp', False))" in rendered
    assert "always refuses" in ast.get_docstring(method)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"artifact_fingerprint": "b" * 64}, "artifact fingerprint"),
        ({"source_revision": "0" * 40}, "source revision"),
        ({"request_id": "another-request"}, "request owner"),
        ({"cache_layout": "foreign-layout"}, "cache layout"),
        ({"checkpoint_position": 4}, "checkpoint position"),
    ],
)
def test_direct_primitive_state_binding_still_refuses_foreign_state(change, message):
    with pytest.raises(ValueError, match=message):
        validate_exact_prefix_state_binding(
            _binding("request", **change),
            expected_artifact_fingerprint="a" * 64,
            expected_source_revision=(
                "f6415871606005626a2ac36c0c5630d5b5a8568e"
            ),
            expected_cache_layout="test-hybrid-cache",
            expected_request_id="request",
            expected_state_planes=frozenset({"attention_kv", "recurrent"}),
            expected_execution_domain="ordinary_target_b1",
            live_checkpoint_position=3,
        )


def test_adapter_contract_preserves_direct_primitive_and_withdraws_route_claim():
    source = Path("src/mlx2/adapters/nemotron3_super.py").read_text()
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "exact_prefix_reuse_geometry"
    )
    rendered = ast.unparse(method)
    assert rendered.count("validate_exact_prefix_state_binding(") == 2
    assert "live_checkpoint_position=geometry.receipt()['position']" in rendered
    assert '"serving_route_implemented": False' in source
    assert '"http_request_selection_implemented": False' in source
    assert "unimplemented_sampler_processors_rng_history" in source

    geometry = Path("src/mlx2/runtime/nemotron_prefix_reuse.py").read_text()
    assert geometry.count("shared-prefix state must have exactly one row") == 2
    assert "cache includes an MTP, merged, or foreign state plane" in geometry
    assert "not TwoTower" in geometry


def test_generic_boundary_has_no_model_name_dispatch_or_mlx_import():
    source = Path("src/mlx2/runtime/exact_prefix_serving.py").read_text()
    assert "model_type" not in source
    assert "nemotron" not in source.lower()
    code = (
        "import sys; "
        "from mlx2.runtime.exact_prefix_serving import "
        "ExactPrefixServingUnsupported, open_exact_prefix_serving_session; "
        "from mlx2.runtime.exact_prefix_state_binding import "
        "validate_exact_prefix_state_binding; "
        "assert 'mlx' not in sys.modules; "
        "assert not any(name.startswith('mlx.') for name in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
