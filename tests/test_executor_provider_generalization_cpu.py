"""CPU contracts for adapter-owned target and continuation executors."""

from pathlib import Path
from types import SimpleNamespace
import pytest


ROOT = Path(__file__).parents[1]


def test_shared_external_scheduler_has_no_qwen38_executor_imports():
    source = (ROOT / "src/mlx2/runtime/external_speculative.py").read_text()
    assert "qwen38_tensorfold_source" not in source
    assert "from .qwen38_tensorfold import" not in source
    assert "target_tree_executor_provider" in source


def test_qwen38_target_provider_detects_source_revision_drift(monkeypatch):
    from mlx2.adapters import qwen38_tensorfold_source as source
    from mlx2.adapters.qwen38_target_tree_executor import Qwen38TargetTreeExecutor
    from mlx2.runtime import qwen38_tensorfold

    monkeypatch.setattr(
        source, "qualification_source_identity", lambda: {"revision": "one"}
    )
    monkeypatch.setattr(qwen38_tensorfold, "prepare_validated_source", lambda token: None)
    provider = Qwen38TargetTreeExecutor.create()
    from mlx2.runtime.target_tree_executor import TargetTreeExecutorProvider
    assert isinstance(provider, TargetTreeExecutorProvider)
    monkeypatch.setattr(
        source, "qualification_source_identity", lambda: {"revision": "two"}
    )
    # Hot-path cohort validation consumes only the immutable construction token.
    provider.validate_cohort(1, 1)
    with pytest.raises(RuntimeError, match="source revision changed"):
        provider.validate_revision()


def test_prepared_tensorfold_token_skips_hot_path_hash_validation(monkeypatch):
    from mlx2.adapters import qwen38_tensorfold_source as source
    from mlx2.adapters.qwen38_target_tree_executor import Qwen38TargetTreeExecutor
    from mlx2.runtime import qwen38_tensorfold

    provider = Qwen38TargetTreeExecutor.create()
    monkeypatch.setattr(
        source,
        "validate_source",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("hot path rehashed source")
        ),
    )
    monkeypatch.setattr(
        qwen38_tensorfold,
        "_install_owned_gdn",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("hot path reinstalled owned kernels")
        ),
    )
    module, hit = qwen38_tensorfold._modules(
        validated_token=provider.source_revision_token
    )
    assert module is not None and hit is True
    provider.validate_cohort(4, 4)


def test_target_tree_metric_failure_aborts_open_transaction():
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    class Provider:
        def dispatch_metrics(self, width, transaction, *, cached):
            raise RuntimeError("synthetic metric failure")

    class Transaction:
        closed = False
        provider_protocol = "bound-provider-v1"
        provider_revision_token = "revision-a"

        def commit_paths(self, paths):
            return paths

        def abort(self):
            self.closed = True

    owner = object.__new__(ExternalDraftBatchGenerator)
    provider = Provider()
    provider.protocol = "bound-provider-v1"
    provider.source_revision_token = "revision-a"
    owner.target_tree_executor_provider = provider
    transaction = Transaction()
    with pytest.raises(RuntimeError, match="synthetic metric failure"):
        owner._finish_target_tree_dispatch(
            [object()], (object(), object(), transaction), cached=False
        )
    assert transaction.closed


def test_malformed_target_tree_closed_state_aborts_before_type_error():
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    class Provider:
        protocol = "bound-provider-v1"
        source_revision_token = "revision-a"

    class Transaction:
        closed = 0
        provider_protocol = "bound-provider-v1"
        provider_revision_token = "revision-a"

        def commit_paths(self, paths):
            return paths

        def abort(self):
            self.closed = True

    owner = object.__new__(ExternalDraftBatchGenerator)
    owner.target_tree_executor_provider = Provider()
    transaction = Transaction()
    with pytest.raises(
        TypeError, match="target-tree transaction closed state must be boolean"
    ):
        owner._finish_target_tree_dispatch(
            [object()], (object(), object(), transaction), cached=False
        )
    assert transaction.closed is True


def test_invalid_target_tree_protocol_with_truthy_closed_state_still_aborts():
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    class InvalidTransaction:
        closed = "open"

        def commit_paths(self, paths):
            return paths

        def abort(self):
            self.closed = True

    owner = object.__new__(ExternalDraftBatchGenerator)
    # The finish reads the bound provider first (dd81dfc6a); the invalid
    # transaction must still abort before any provider protocol check.
    owner.target_tree_executor_provider = None
    transaction = InvalidTransaction()
    with pytest.raises(TypeError, match="invalid transaction"):
        owner._finish_target_tree_dispatch(
            [object()], (object(), object(), transaction), cached=False
        )
    assert transaction.closed is True


def test_target_tree_metric_failure_aborts_after_closed_state_corruption():
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    class Provider:
        protocol = "bound-provider-v1"
        source_revision_token = "revision-a"

        def dispatch_metrics(self, width, transaction, *, cached):
            transaction.closed = "open"
            raise RuntimeError("synthetic corrupting metric failure")

    class Transaction:
        closed = False
        provider_protocol = "bound-provider-v1"
        provider_revision_token = "revision-a"

        def commit_paths(self, paths):
            return paths

        def abort(self):
            self.closed = True

    owner = object.__new__(ExternalDraftBatchGenerator)
    owner.target_tree_executor_provider = Provider()
    transaction = Transaction()
    with pytest.raises(RuntimeError, match="corrupting metric failure"):
        owner._finish_target_tree_dispatch(
            [object()], (object(), object(), transaction), cached=False
        )
    assert transaction.closed is True


def test_shared_scheduler_applies_validated_target_tree_metrics_atomically():
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from mlx2.runtime.target_tree_executor import TargetTreeDispatchMetrics

    owner = object.__new__(ExternalDraftBatchGenerator)
    owner.scheduler_stats = {
        "external_target_tree_target_width_2_rounds": 0,
        "external_target_tree_target_rounds": 0,
        "external_target_tree_physical_target_forwards": 0,
        "external_target_tree_cohort_rounds": 0,
        "external_target_tree_cohort_lanes": 0,
        "external_target_tree_cohort_max_width": 0,
        "external_target_tree_packed_target_rounds": 0,
        "external_target_tree_packed_target_lanes": 0,
        "external_target_tree_executor_cache_hits": 0,
    }
    lanes = [
        SimpleNamespace(
            target_tree_width_hist={},
            target_tree_width_trace=[],
            target_tree_width_trace_overflow=0,
        )
        for _ in range(2)
    ]
    owner._apply_target_tree_dispatch_metrics(
        lanes,
        TargetTreeDispatchMetrics(
            width=2,
            physical_packed=True,
            executor_cached=True,
            cached=True,
            provider_protocol="test-provider-v1",
        ),
    )
    assert owner.scheduler_stats["external_target_tree_target_width_2_rounds"] == 1
    assert owner.scheduler_stats["external_target_tree_packed_target_lanes"] == 2
    assert owner.scheduler_stats["external_target_tree_executor_cache_hits"] == 1
    assert all(lane.target_tree_width_trace == [2] for lane in lanes)


def test_target_tree_dispatch_rejects_cache_or_provider_misbinding_and_aborts():
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from mlx2.runtime.target_tree_executor import TargetTreeDispatchMetrics

    class Transaction:
        closed = False
        provider_protocol = "bound-provider-v1"
        provider_revision_token = "revision-a"

        def commit_paths(self, paths):
            return paths

        def abort(self):
            self.closed = True

    class Provider:
        protocol = "bound-provider-v1"
        source_revision_token = "revision-a"

        def dispatch_metrics(self, width, transaction, *, cached):
            return TargetTreeDispatchMetrics(
                width=width,
                physical_packed=False,
                executor_cached=True,
                cached=False,
                provider_protocol="other-provider-v1",
            )

    owner = object.__new__(ExternalDraftBatchGenerator)
    owner.target_tree_executor_provider = Provider()
    transaction = Transaction()
    with pytest.raises(RuntimeError, match="provider identity mismatch"):
        owner._finish_target_tree_dispatch(
            [object()], (object(), object(), transaction), cached=True
        )
    assert transaction.closed


def test_native_provider_is_candidate_and_revision_bound(monkeypatch):
    from mlx2.adapters.native_continuation import bind_qwen_native_continuation_provider
    from mlx2.runtime import paged_native_continuation

    model = object()
    candidate = type("CandidateA", (), {})()
    candidate.model = model
    provider = bind_qwen_native_continuation_provider(
        candidate, "revision-a", model, "native-test"
    )
    provider.validate_candidate(candidate)
    other = type("CandidateB", (), {})()
    other.model = model
    other._native_continuation_provider = provider
    with pytest.raises(TypeError, match="not bound"):
        provider.validate_candidate(other)

    class HostileClaimant:
        state_planes = ("kv", "gdn")
        bootstrap_generation = 0
        supports_singleton = True
        owns_physical_dispatch_proof = True
        native_layer_count = 2
        logical_layer_count = 3
        backend = object()

        def __init__(self):
            self.model = model
            self._native_continuation_provider = provider

        def packed_lane(self, *_args, **_kwargs):
            return None

        def forward_staged(self, *_args, **_kwargs):
            return None

    with pytest.raises(TypeError, match="not bound"):
        provider.validate_candidate(HostileClaimant())

    monkeypatch.setattr(
        paged_native_continuation,
        "NativeQwen3Continuation",
        lambda **kwargs: kwargs,
    )
    with pytest.raises(ValueError, match="revision differs"):
        provider.create_continuation(
            candidate=candidate, revision="revision-b"
        )


def test_shared_generate_and_serving_do_not_name_qwen_candidate_types():
    for path in (
        ROOT / "src/mlx2/runtime/generate.py",
        ROOT / "src/mlx2/serving.py",
    ):
        source = path.read_text()
        assert "Qwen3PackedCandidate" not in source
        assert "NativeQwen3Continuation" not in source
