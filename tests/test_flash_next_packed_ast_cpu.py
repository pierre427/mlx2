"""Scheduler ownership tests. CPU numerical stubs never qualify Metal."""
import mlx.core as mx
import pytest
from test_apc_hits_hybrid_gdn_self_mtp import tiny_qwen4_mtp
from test_native_mtp_ast_cpu import drain, reference

from mlx2.adapters.flash_next_tree import FlashNextTargetTreeExecutor, install_ast_target_contract
from mlx2.adapters.proposal_path_sources import build_continuation_drafter
from mlx2.runtime.ast_planner import AlternativeExecutionBackend, NumericalEquivalenceOracle
from mlx2.runtime.drafters.native_mtp import NativeMTPDraftModel
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.proposal_providers import ContinuationPoolPolicy

PIN = "a" * 64


@pytest.fixture(autouse=True)
def cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


class CPUProvider(FlashNextTargetTreeExecutor):
    def __init__(self, model):
        super().__init__(model, cpu_reference=True)
        self.calls = []

    def ast_execution_backend(self, profile):
        self.ast_input_widths = (16,)
        return AlternativeExecutionBackend(
            "target_tree_single_session", profile.binding, profile.device, profile.build,
            self.source_revision_token, (16,), profile.row_costs,
            oracle=NumericalEquivalenceOracle(profile.binding, profile.device, profile.build,
                                              self.source_revision_token, "CPU_TEST_ONLY", True),
        )

    def forward(self, model, inputs, parents, cache, layers, *, cached=True):
        self.calls.append((tuple(inputs), tuple(parents)))
        assert len(inputs) == 16
        return super().forward(model, inputs, parents, cache, layers, cached=cached)


def setup(pattern):
    model, _ = tiny_qwen4_mtp()
    model.select_external_feature_source("mtp_seed")
    object.__setattr__(model, "_mlx2_flash_tree_law", (("cpu_test_only", True),))
    install_ast_target_contract(model)
    draft = NativeMTPDraftModel(model, num_draft=max(pattern), revision=PIN)
    policy = {"sources": ["external"], "limit": len(pattern), "ast_planner": True,
              "packed_pattern": list(pattern)}
    wrapper = build_continuation_drafter(model, draft, policy, target_revision=PIN,
        draft_revision=PIN, tokenizer_revision=PIN, session_revision=PIN)
    provider = CPUProvider(model)
    batch = ExternalDraftBatchGenerator(model, draft_model=wrapper, binding=PIN,
        num_draft=max(pattern), completion_batch_size=1,
        continuation_pool=wrapper.policy, continuation_pool_ast_bridge=True,
        progressive_verification_tile=2, progressive_alternative_branches=len(pattern),
        progressive_alternative_strategy="confidence_tree", cost_per_commit_handoff=False,
        ast_target_tree_executor_provider=provider)
    return model, draft, wrapper, provider, batch


@pytest.mark.parametrize("pattern", [(5, 5, 5), (3, 3, 3, 3, 3)])
def test_actual_scheduler_uses_full_packed_tiles_and_matches_ordinary(pattern, monkeypatch):
    model, draft, wrapper, provider, batch = setup(pattern)
    prompt = [1, 7, 3, 9, 2, 8, 4, 6, 5, 11, 13, 2]
    expected = reference(model, prompt, 24)
    # No legacy recomputation by a continuation source is permitted.
    def forbidden(*args, **kwargs):
        raise AssertionError("legacy short-fork source invoked")
    monkeypatch.setattr(draft, "propose_complete_paths", forbidden)
    try:
        batch.insert([prompt], max_tokens=[24], sampling_configs=[{"sampling_temp": 0}])
        output, final = drain(batch)
        assert output[0] == expected
        assert provider.calls and all(len(tokens) == 16 for tokens, _ in provider.calls)
        assert not model._progressive_row_exact
        assert not wrapper.proposal_pool._pending
        assert draft.stats["draft_calls"] == len(provider.calls)
        assert final[0].speculative_receipt["progressive_verification"]["observed_used"]
    finally:
        batch.close()


@pytest.mark.parametrize("value", [[], [16], [5, 5, 6], [True, 2], [3] * 6])
def test_bad_patterns_fail_before_source_mutation(value):
    with pytest.raises(ValueError, match="packed_pattern"):
        ContinuationPoolPolicy.from_value({"sources": ["external"], "ast_planner": True,
                                           "limit": len(value), "packed_pattern": value})


def test_default_policy_identity_unchanged_and_tree_has_no_serial_fallback():
    assert "packed_pattern" not in ContinuationPoolPolicy().as_dict()
    model, _ = tiny_qwen4_mtp()
    model.select_external_feature_source("mtp_seed")
    with pytest.raises(ValueError, match="cold-bound"):
        install_ast_target_contract(model)
    object.__setattr__(model, "_mlx2_flash_tree_law", (("cpu_test_only", True),))
    install_ast_target_contract(model)
    with pytest.raises(ValueError, match="no serial"):
        model.progressive_external_target_rows(mx.array([[1, 2]]), model.make_cache(), (1,))


def test_unvisited_stop_candidates_do_not_switch_to_serial_verification():
    model, draft, wrapper, provider, batch = setup((3, 3, 3, 3, 3))
    prompt = [1, 7, 3, 9, 2, 8, 4, 6, 5]
    expected = reference(model, prompt, 10)
    # Future proposal stops may not refuse an admitted tree when the actual
    # target-approved continuation does not stop.
    batch.stops = set(range(64)) - set(expected)
    try:
        batch.insert([prompt], max_tokens=[10], sampling_configs=[{"sampling_temp": 0}])
        output, final = drain(batch)
        assert output[0] == expected
        assert provider.calls
        assert not wrapper.proposal_pool._pending
    finally:
        batch.close()


def test_oracle_absent_stale_and_foreign_law_decline(tmp_path):
    import json
    from types import SimpleNamespace

    model, _ = tiny_qwen4_mtp()
    model.select_external_feature_source("mtp_seed")
    object.__setattr__(model, "_mlx2_flash_tree_law", (("cpu_test_only", True),))
    provider = FlashNextTargetTreeExecutor(model)
    profile = SimpleNamespace(binding="session", device="cpu_test", build="build",
                              row_costs=(1.0,) * 16, launch_cost=0.0,
                              context_row_cost=0.0, context_windows=())
    assert provider.ast_execution_backend(profile) is None
    binding = provider.oracle_binding(artifact=PIN, capture_layers=(1,), target_binding={}, profile=profile)
    doc = {"schema": "mlx2.flash-next-tree-oracle.v1", "binding": binding,
           "input_widths": [16], "receipt_sha256": "b" * 64,
           "checks": dict.fromkeys(("logits", "features", "prepared_state", "continuation", "isolation", "commit_abort"), True)}
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps(doc))
    provider.with_ast_oracle(path, artifact=PIN, capture_layers=(1,), target_binding={})
    assert provider.ast_execution_backend(profile).input_widths == (16,)
    profile.build = "foreign"
    with pytest.raises(ValueError, match="live source"):
        provider.ast_execution_backend(profile)
    object.__setattr__(model, "_mlx2_flash_tree_law", (("foreign", True),))
    with pytest.raises(ValueError, match="law changed"):
        provider.validate_revision()


@pytest.mark.parametrize("pattern", [(5, 5, 5), (3, 3, 3, 3, 3)])
def test_actual_flash_policy_split_and_normalization(pattern):
    from mlx2.adapters.flash_next import split_native_mtp_external_policy
    from mlx2.adapters.external_draft_policy import normalize_progressive_external_policy

    policy = {"draft_model": "/model", "num_draft": max(pattern), "row_exact_verify": False,
              "progressive_verification_tile": 2, "progressive_alternative_branches": len(pattern),
              "progressive_alternative_strategy": "confidence_tree", "cost_per_commit_handoff": False,
              "continuation_pool": {"sources": ["external"], "limit": len(pattern),
                                    "ast_planner": True, "packed_pattern": list(pattern)},
              "ast_target_tree_oracle": "/oracle.json"}
    native, external = split_native_mtp_external_policy(policy)
    assert native == {"row_exact_verify": False}
    result = normalize_progressive_external_policy(external, num_draft=max(pattern))
    assert result["progressive_alternative_branches"] == len(pattern)


def test_http_receipt_boundary_accepts_null_ordinary_but_requires_real_ast():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/probe_flash_next_packed_ast.py"
    spec = importlib.util.spec_from_file_location("packed_http_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.require_route({"mlx2": {"speculation": None}}, "ordinary")
    with pytest.raises(RuntimeError, match="did not engage"):
        module.require_route({"mlx2": {"speculation": None}}, "ast")
    module.require_route({"mlx2": {"speculation": {"ast_tree_tiles": {
        "observed_used": True, "target_rows": 16}}}}, "ast")
