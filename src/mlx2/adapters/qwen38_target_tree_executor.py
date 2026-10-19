"""Revision-bound Qwen3.8 target-tree execution provider."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from ..runtime.target_tree_executor import TargetTreeDispatchMetrics


@dataclass(frozen=True)
class Qwen38TargetTreeExecutor:
    """Own Qwen3.8 source validation, execution and route evidence."""

    source_identity: tuple[tuple[str, str], ...]
    source_revision_token: str
    protocol: str = "qwen38_tensorfold_v1"
    maximum_cohort_width: int = 4
    ast_numerical_oracle: object = None

    @classmethod
    def create(cls):
        from .qwen38_tensorfold_source import qualification_source_identity

        identity = qualification_source_identity()
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        token = hashlib.sha256(encoded.encode()).hexdigest()
        public_identity = {
            key: value for key, value in identity.items() if key != "module_sha256"
        }
        provider = cls(
            tuple(sorted((str(k), str(v)) for k, v in public_identity.items())),
            token,
        )
        from ..runtime.qwen38_tensorfold import prepare_validated_source

        prepare_validated_source(token)
        return provider

    def validate_revision(self) -> None:
        """Cold-path revalidation of the immutable construction-time token."""
        from .qwen38_tensorfold_source import qualification_source_identity

        identity = qualification_source_identity()
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        current = hashlib.sha256(encoded.encode()).hexdigest()
        if current != self.source_revision_token:
            raise RuntimeError("target-tree executor source revision changed")

    def validate_cohort(self, width: int, configured_limit: int) -> None:
        if (
            type(width) is not int
            or not 1 <= width <= self.maximum_cohort_width
            or width > configured_limit
        ):
            raise ValueError("target-tree executor cohort exceeds its bound")

    def forward(self, model, inputs, parents, cache, layers, *, cached: bool):
        self.validate_cohort(1, 1)
        from ..runtime.qwen38_tensorfold import forward

        result = forward(
            model, inputs, parents, cache, layers, cached=cached,
            validated_token=self.source_revision_token,
        )
        return self._bind_transaction(result)

    def forward_many(
        self, model, inputs, parents, caches, layers, *, cached: bool,
        configured_limit: int,
    ):
        self.validate_cohort(len(caches), configured_limit)
        from ..runtime.qwen38_tensorfold import forward_many

        result = forward_many(
            model, inputs, parents, caches, layers, cached=cached,
            validated_token=self.source_revision_token,
        )
        return self._bind_transaction(result)

    def _bind_transaction(self, result):
        if not isinstance(result, tuple) or len(result) != 3:
            raise TypeError("Qwen3.8 target-tree executor returned an invalid result")
        transaction = result[2]
        transaction.provider_protocol = self.protocol
        transaction.provider_revision_token = self.source_revision_token
        return result

    def dispatch_metrics(self, width, transaction, *, cached: bool):
        self.validate_cohort(width, self.maximum_cohort_width)
        return TargetTreeDispatchMetrics(
            width=width,
            physical_packed=bool(getattr(transaction, "physical_packed", False)),
            executor_cached=bool(getattr(transaction, "executor_cached", False)),
            cached=bool(cached),
            provider_protocol=self.protocol,
        )

    def ast_execution_backend(self, profile):
        """Publish the phase-1 chain backend only after an exact oracle.

        The current adapter carries no pre-approved AST oracle, so normal
        construction returns a refusal. Qualification code may construct a
        provider with an explicit ``NumericalEquivalenceOracle``; its complete
        profile/device/build/source binding is rechecked by the generic
        execution profile before planning.
        """

        oracle = self.ast_numerical_oracle
        if oracle is None:
            return None
        from ..runtime.ast_planner import AlternativeExecutionBackend

        widths = tuple(
            width
            for width in profile.terminal_widths
            if width <= len(profile.row_costs)
        )
        return AlternativeExecutionBackend(
            backend="target_tree_single_session",
            profile_binding=profile.binding,
            device=profile.device,
            build=profile.build,
            source_revision_token=self.source_revision_token,
            input_widths=widths,
            # TensorFold's chain launch reads target weights once. Context
            # traffic remains the profile's declared exact attention cost.
            row_costs=tuple(1.0 for _ in profile.row_costs),
            launch_cost=0.0,
            context_row_cost=profile.context_row_cost,
            context_windows=profile.context_windows,
            cost_authority="tensorfold_single_launch_weight_pass_v1",
            maximum_sessions=1,
            oracle=oracle,
        )

    def receipt(self) -> dict:
        from ..runtime.qwen38_tensorfold import gdn_backend_stats, glue_backend_stats

        return {
            "provider": self.protocol,
            "source_identity": dict(self.source_identity),
            "source_revision_token": self.source_revision_token,
            "ast_target_tree": {
                "implemented": True,
                "qualified": False,
                "selected": self.ast_numerical_oracle is not None,
                "maximum_sessions": 1,
                "sibling_cache_sharing": False,
                "numerical_oracle": (
                    None
                    if self.ast_numerical_oracle is None
                    else self.ast_numerical_oracle.evidence
                ),
            },
            "gdn_backend": gdn_backend_stats(),
            "glue_backend": glue_backend_stats(),
        }
