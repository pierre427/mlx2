"""Revision-bound Qwen3.8 target-tree execution provider."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess

from ..runtime.target_tree_executor import TargetTreeDispatchMetrics

AST_CHAIN_ORACLE_SCHEMA = "mlx2.qwen38-ast-chain-oracle.v1"
AST_CHAIN_ORACLE_CHECKS = (
    "full_logits",
    "features",
    "prepared_state",
    "prefix_commit",
    "continuation",
    "abort",
    "lane_isolation",
    "ordinary_reference",
)
AST_FAST_TREE_ORACLE_SCHEMA = "mlx2.qwen38-ast-fast-tree-oracle.v1"
AST_FAST_COHORT_ORACLE_SCHEMA = "mlx2.qwen38-ast-fast-tree-oracle.v2"
AST_FAST_TREE_ORACLE_CHECKS = (
    *AST_CHAIN_ORACLE_CHECKS,
    "branch_paths",
    "shared_prefix",
)
AST_FAST_COHORT_ORACLE_CHECKS = (
    *AST_FAST_TREE_ORACLE_CHECKS,
    "heterogeneous_parents",
    "nonzero_lane_offsets",
    "mixed_path_commit",
    "zero_partial_full_commits",
)
_AST_COHORT_WIDTHS = (
    (2, tuple(range(1, 9))),
    (3, tuple(range(1, 6))),
    (4, tuple(range(1, 5))),
)


def ast_fast_cohort_domain():
    """Exact offline evidence domain; no permission is minted by this helper."""
    return {
        "layout": "packed_independent_private_trees",
        "topology_law": "parent_ordered_rooted_tree_v1",
        "maximum_total_rows": 16,
        "input_widths_by_sessions": {
            str(sessions): list(widths) for sessions, widths in _AST_COHORT_WIDTHS
        },
        "checks": {check: True for check in AST_FAST_COHORT_ORACLE_CHECKS},
    }


def _rooted_tree(inputs, parents):
    return (
        1 <= len(inputs) <= 16
        and len(parents) == len(inputs)
        and type(parents[0]) is int
        and parents[0] == -1
        and all(
            type(parent) is int and 0 <= parent < index
            for index, parent in enumerate(parents[1:], 1)
        )
    )


_AST_SOURCE_PATHS = (
    "src/mlx2/runtime/ast_private_state.py",
    "src/mlx2/runtime/cow_cache.py",
    "src/mlx2/runtime/drafters/fold.py",
    "src/mlx2/adapters/qwen38_target_tree_executor.py",
    "src/mlx2/adapters/qwen38_tensorfold_source.py",
    "src/mlx2/adapters/qwen38_hybrid_metal_probe.py",
    "src/mlx2/runtime/qwen38_tensorfold.py",
    "src/mlx2/runtime/target_tree_executor.py",
    "src/mlx2/runtime/models/qwen38_tree_gdn.py",
    "src/mlx2/adapters/qwen38_fast_ast.py",
    "src/mlx2/runtime/lane/installer.py",
    "src/mlx2/runtime/lane/policy.py",
    "src/mlx2/runtime/lane/matmul.py",
)
_ROOT = Path(__file__).resolve().parents[3]


def _ast_source_binding(target_binding):
    from .qwen38_hybrid_metal_probe import _SOURCE_PATHS as target_sources

    paths = tuple(
        sorted(
            set(
                (
                    *target_binding["source_hashes"],
                    *target_sources,
                    *_AST_SOURCE_PATHS,
                )
            )
        )
    )
    revision = subprocess.check_output(
        ["git", "-C", str(_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.run(
        ["git", "-C", str(_ROOT), "diff", "--quiet", revision, "--", *paths],
        check=False,
    ).returncode:
        raise ValueError("AST TensorFold oracle requires clean bound source")
    return revision, {
        path: hashlib.sha256((_ROOT / path).read_bytes()).hexdigest() for path in paths
    }


@dataclass(frozen=True)
class Qwen38TargetTreeExecutor:
    """Own Qwen3.8 source validation, execution and route evidence."""

    source_identity: tuple[tuple[str, str], ...]
    source_revision_token: str
    protocol: str = "qwen38_tensorfold_v1"
    maximum_cohort_width: int = 4
    ast_numerical_oracle: object = None
    ast_oracle_document: str | None = None
    ast_oracle_context: str | None = None
    ast_oracle_path: str | None = None
    ast_input_widths: tuple[int, ...] = ()
    ast_capture_layers: tuple[int, ...] = ()
    ast_chain: bool = False
    ast_fast_tree: bool = False
    ast_fast_model: object = None
    ast_fast_route: object = None
    ast_cohort_probe: bool = False
    ast_cohort_domain: tuple = ()

    @classmethod
    def create(cls, *, ast_chain=False, ast_fast_tree=False, ast_cohort_probe=False):
        from .qwen38_tensorfold_source import qualification_source_identity

        if (
            type(ast_chain) is not bool
            or type(ast_fast_tree) is not bool
            or type(ast_cohort_probe) is not bool
            or (ast_chain and ast_fast_tree)
            or (ast_cohort_probe and not ast_fast_tree)
        ):
            raise ValueError("AST projection selection must be one explicit boolean")
        identity = qualification_source_identity()
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        token = hashlib.sha256(encoded.encode()).hexdigest()
        public_identity = {
            key: value for key, value in identity.items() if key != "module_sha256"
        }
        provider = cls(
            tuple(sorted((str(k), str(v)) for k, v in public_identity.items())),
            token,
            ast_chain=ast_chain,
            ast_fast_tree=ast_fast_tree,
            ast_cohort_probe=ast_cohort_probe,
        )
        from ..runtime.qwen38_tensorfold import prepare_validated_source

        prepare_validated_source(token)
        return provider

    def _ast_context(self, *, model, target_binding, artifact, capture_layers):
        from .qwen38_hybrid_metal_probe import qwen38_hybrid_model_manifest_sha256
        from ..runtime.progressive_external_verify import (
            PROGRESSIVE_TARGET_PROTOCOL_VERSION,
            progressive_target_protocol_binding,
        )

        if (
            not (self.ast_chain or self.ast_fast_tree)
            or target_binding.get("fidelity") != "exact"
            or target_binding.get("target_protocol")
            != PROGRESSIVE_TARGET_PROTOCOL_VERSION
            or target_binding != progressive_target_protocol_binding(model, "exact")
            or not isinstance(artifact, str)
            or not artifact
        ):
            raise ValueError("AST TensorFold requires the exact live target binding")
        layers = tuple(capture_layers)
        if not layers or any(type(layer) is not int or layer < 0 for layer in layers):
            raise ValueError("AST TensorFold requires exact capture layers")
        revision, hashes = _ast_source_binding(target_binding)
        self.validate_revision()
        context = {
            "source_revision": revision,
            "source_hashes": hashes,
            "artifact": artifact,
            "executable_model_manifest": qwen38_hybrid_model_manifest_sha256(model),
            "target_protocol_binding": target_binding,
            "capture_layers": list(layers),
            "provider_protocol": self.protocol,
            "provider_revision_token": self.source_revision_token,
            "fidelity": "exact",
            "layout": (
                "independent_private_b1_tree"
                if self.ast_fast_tree
                else "independent_private_b1_chain"
            ),
            "projection_law": (
                "lane_mpp_exact"
                if self.ast_fast_tree
                else "row_exact_valid_rows_window_v1"
            ),
        }
        if self.ast_fast_tree:
            route = getattr(model, "_mlx2_qwen38_fast_ast", None)
            if route is None:
                raise ValueError("AST TensorFold fast numerical law is not selected")
            context["lane_numerical_identity"] = route.identity()
            context["topology_law"] = "parent_ordered_rooted_tree_v1"
            context["maximum_tree_rows"] = 16
        return context

    def ast_oracle_binding(
        self, profile, *, model, target_binding, artifact, capture_layers
    ):
        """Cold offline binding for an actual model construction and profile."""
        return {
            **self._ast_context(
                model=model,
                target_binding=target_binding,
                artifact=artifact,
                capture_layers=capture_layers,
            ),
            "profile_binding": profile.binding,
            "device": profile.device,
            "build": profile.build,
        }

    def with_ast_oracle(self, path, *, model, target_binding, artifact, capture_layers):
        """Load an affirmative offline oracle; never run request-time canaries."""
        path = Path(path).expanduser().resolve()
        document = json.loads(path.read_text())
        cohort = (
            self.ast_fast_tree
            and isinstance(document, dict)
            and (document.get("schema") == AST_FAST_COHORT_ORACLE_SCHEMA)
        )
        schema = (
            AST_FAST_COHORT_ORACLE_SCHEMA
            if cohort
            else AST_FAST_TREE_ORACLE_SCHEMA
            if self.ast_fast_tree
            else AST_CHAIN_ORACLE_SCHEMA
        )
        fields = {"schema", "bindings", "input_widths", "checks", "target_timings"}
        if cohort:
            fields.add("cohort_domain")
        checks = (
            AST_FAST_TREE_ORACLE_CHECKS
            if self.ast_fast_tree
            else AST_CHAIN_ORACLE_CHECKS
        )
        if (
            not isinstance(document, dict)
            or not {"schema", "bindings", "input_widths", "checks"} <= set(document)
            or set(document) - fields
            or document["schema"] != schema
            or not isinstance(document["bindings"], list)
            or not document["bindings"]
            or any(not isinstance(binding, dict) for binding in document["bindings"])
            or not isinstance(document["checks"], dict)
            or set(document["checks"]) != set(checks)
            or any(document["checks"][key] is not True for key in checks)
        ):
            raise ValueError("AST TensorFold oracle is incomplete or did not pass")
        if cohort and json.dumps(
            document.get("cohort_domain"), sort_keys=True
        ) != json.dumps(ast_fast_cohort_domain(), sort_keys=True):
            raise ValueError(
                "AST TensorFold cohort oracle domain is incomplete or invalid"
            )
        widths = document["input_widths"]
        if (
            not isinstance(widths, list)
            or not widths
            or any(type(width) is not int or not 1 <= width <= 16 for width in widths)
            or widths != sorted(set(widths))
        ):
            raise ValueError("AST TensorFold oracle input widths are invalid")
        if cohort and widths != list(range(1, 17)):
            raise ValueError(
                "AST TensorFold cohort oracle requires the complete B1 domain"
            )
        timings = document.get("target_timings", [])
        if (
            not isinstance(timings, list)
            or any(
                not isinstance(item, dict)
                or set(item)
                != {
                    "input_width",
                    "context_length",
                    "ordinary_seconds",
                    "candidate_seconds",
                }
                or type(item["input_width"]) is not int
                or item["input_width"] not in widths
                or type(item["context_length"]) is not int
                or item["context_length"] < 0
                or any(
                    type(item[key]) not in (int, float)
                    or not math.isfinite(item[key])
                    or item[key] <= 0
                    for key in ("ordinary_seconds", "candidate_seconds")
                )
                for item in timings
            )
            or len({item["input_width"] for item in timings}) != len(timings)
        ):
            raise ValueError("AST TensorFold target timing evidence is invalid")
        context = self._ast_context(
            model=model,
            target_binding=target_binding,
            artifact=artifact,
            capture_layers=capture_layers,
        )
        return replace(
            self,
            ast_oracle_document=json.dumps(document, sort_keys=True),
            ast_oracle_context=json.dumps(context, sort_keys=True),
            ast_oracle_path=str(path),
            ast_input_widths=tuple(widths),
            ast_capture_layers=tuple(capture_layers),
            ast_fast_model=model if self.ast_fast_tree else None,
            ast_fast_route=(
                getattr(model, "_mlx2_qwen38_fast_ast", None)
                if self.ast_fast_tree
                else None
            ),
            ast_cohort_probe=False,
            ast_cohort_domain=_AST_COHORT_WIDTHS if cohort else (),
        )

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
        if self.ast_fast_tree:
            if not _rooted_tree(inputs, parents):
                raise ValueError(
                    "AST TensorFold requires a bounded shared-root B1 tree"
                )
            route = getattr(model, "_mlx2_qwen38_fast_ast", None)
            if route is None:
                raise ValueError("AST TensorFold fast numerical law is not selected")
            if self.ast_oracle_document is not None and (
                model is not self.ast_fast_model or route is not self.ast_fast_route
            ):
                raise ValueError("AST TensorFold fast model binding changed")
            route.require_bound()
        if self.ast_chain and (
            not 1 <= len(inputs) <= 16
            or (
                not self.ast_fast_tree
                and list(parents) != list(range(-1, len(inputs) - 1))
            )
        ):
            raise ValueError("AST TensorFold requires a bounded B1 linear chain")
        if self.ast_oracle_document is not None and (
            len(inputs) not in self.ast_input_widths
            or (
                not self.ast_fast_tree
                and list(parents) != list(range(-1, len(inputs) - 1))
            )
            or tuple(layers) != self.ast_capture_layers
        ):
            raise ValueError("AST TensorFold launch is outside its qualified geometry")
        from ..runtime.qwen38_tensorfold import forward

        def execute():
            return forward(
                model,
                inputs,
                parents,
                cache,
                layers,
                cached=cached,
                validated_token=self.source_revision_token,
            )

        if self.ast_chain:
            from ..runtime import row_exact_verify as REV

            record = REV.Window(len(inputs))
            # The TensorFold stateful traversal remains unchanged. Its shared
            # projection modules must obey the selected ordinary target law.
            # A flat B1 chain must not inherit a caller's ragged BxL mask.
            with REV.valid_rows((len(inputs),), len(inputs)), REV.window(record):
                result = execute()
            result = self._bind_transaction(result)
            if not record.exact:
                result[2].abort()
                raise RuntimeError(
                    f"AST TensorFold row-exact projection declined: {record.failures}"
                )
            result[2].ast_projection_window = {
                "rows": record.rows,
                "stages": record.stages,
                "failures": record.failures,
            }
            return result
        result = execute()
        return self._bind_transaction(result)

    def forward_many(
        self,
        model,
        inputs,
        parents,
        caches,
        layers,
        *,
        cached: bool,
        configured_limit: int,
    ):
        if self.ast_chain:
            raise ValueError("AST TensorFold chain provider is B1-only")
        self.validate_cohort(len(caches), configured_limit)
        if self.ast_fast_tree:
            width = len(caches)
            if not self.ast_cohort_probe and self.ast_oracle_document is None:
                raise ValueError(
                    "AST TensorFold fast provider is B1-only without cohort evidence"
                )
            domain = (
                dict(_AST_COHORT_WIDTHS)
                if self.ast_cohort_probe
                else dict(self.ast_cohort_domain)
            )
            if width not in domain:
                raise ValueError(
                    "AST TensorFold fast provider is B1-only without cohort evidence"
                )
            if (
                cached is not True
                or len(inputs) != width
                or len(parents) != width
                or len({len(row) for row in inputs}) != 1
                or len(inputs[0]) not in domain[width]
                or width * len(inputs[0]) > 16
                or any(
                    not _rooted_tree(row, tree)
                    for row, tree in zip(inputs, parents, strict=True)
                )
                or len({id(cache) for cache in caches}) != width
                or len({id(entry) for cache in caches for entry in cache})
                != sum(len(cache) for cache in caches)
            ):
                raise ValueError(
                    "AST TensorFold cohort is outside its qualified geometry"
                )
            route = getattr(model, "_mlx2_qwen38_fast_ast", None)
            if route is None:
                raise ValueError("AST TensorFold fast numerical law is not selected")
            if self.ast_oracle_document is not None and (
                model is not self.ast_fast_model
                or route is not self.ast_fast_route
                or tuple(layers) != self.ast_capture_layers
            ):
                raise ValueError("AST TensorFold fast cohort binding changed")
            route.require_bound()
        from ..runtime.qwen38_tensorfold import forward_many

        result = forward_many(
            model,
            inputs,
            parents,
            caches,
            layers,
            cached=cached,
            validated_token=self.source_revision_token,
        )
        return self._bind_transaction(result)

    @property
    def ast_tree_max_sessions(self):
        """Admitted physical request count; offline probes never advertise it."""
        return (
            max(sessions for sessions, _widths in self.ast_cohort_domain)
            if self.ast_fast_tree
            and self.ast_oracle_document is not None
            and self.ast_cohort_domain
            else 1
        )

    def ast_tree_cohort_input_widths(self, sessions):
        if type(sessions) is not int or self.ast_tree_max_sessions == 1:
            return ()
        return dict(self.ast_cohort_domain).get(sessions, ())

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
        if self.ast_oracle_document is not None:
            from ..runtime.ast_planner import NumericalEquivalenceOracle

            expected = {
                **json.loads(self.ast_oracle_context),
                "profile_binding": profile.binding,
                "device": profile.device,
                "build": profile.build,
            }
            document = json.loads(self.ast_oracle_document)
            if expected not in document["bindings"]:
                raise ValueError(
                    "AST TensorFold oracle does not match the live profile"
                )
            oracle = NumericalEquivalenceOracle(
                profile.binding,
                profile.device,
                profile.build,
                self.source_revision_token,
                self.ast_oracle_path,
                True,
            )
        if oracle is None:
            return None
        from ..runtime.ast_planner import AlternativeExecutionBackend

        offered_widths = (
            self.ast_input_widths
            if self.ast_fast_tree
            else sorted(set((*profile.tile_widths, *profile.terminal_widths)))
        )
        widths = tuple(
            width
            for width in offered_widths
            if width <= len(profile.row_costs)
            and (not self.ast_input_widths or width in self.ast_input_widths)
        )
        if not widths:
            raise ValueError("AST TensorFold oracle admits no live input widths")
        # Until actual paired target timings exist, the projection law has the
        # same conservative prior as the ordinary exact target. Do not invent
        # a single weight pass for a row-exact QMV projection.
        row_costs = list(profile.row_costs)
        fitted = False
        if self.ast_oracle_document is not None:
            for sample in document.get("target_timings", []):
                width = sample["input_width"]
                if width not in widths:
                    continue
                context = sample["context_length"]
                ordinary = profile.cost((width,), (context,))
                unchanged = ordinary - profile.row_costs[width - 1]
                relative_total = (
                    ordinary * sample["candidate_seconds"] / sample["ordinary_seconds"]
                )
                row_prior = relative_total - unchanged
                if math.isfinite(row_prior) and row_prior > 0:
                    row_costs[width - 1] = row_prior
                    fitted = True
        return AlternativeExecutionBackend(
            backend="target_tree_single_session",
            profile_binding=profile.binding,
            device=profile.device,
            build=profile.build,
            source_revision_token=self.source_revision_token,
            input_widths=widths,
            row_costs=tuple(row_costs),
            launch_cost=profile.launch_cost,
            context_row_cost=profile.context_row_cost,
            context_windows=profile.context_windows,
            cost_authority=(
                "offline_paired_target_successor_timing_at_probe_context_v1"
                if fitted
                else "ordinary_exact_projection_prior_v1"
            ),
            timing_boundary="target_successor",
            maximum_sessions=1,
            oracle=oracle,
        )

    @property
    def ast_tree_tile_rows(self):
        """A true-tree capability, never inferred from the protocol's name."""
        return (
            min(16, max(self.ast_input_widths))
            if self.ast_fast_tree
            and self.ast_oracle_document is not None
            and self.ast_input_widths
            else 0
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
                "selected": (
                    self.ast_numerical_oracle is not None
                    or self.ast_oracle_document is not None
                ),
                "input_widths": list(self.ast_input_widths),
                "maximum_sessions": self.ast_tree_max_sessions,
                "cohort_input_widths": {
                    str(sessions): list(widths)
                    for sessions, widths in self.ast_cohort_domain
                },
                "offline_cohort_probe": self.ast_cohort_probe,
                "projection_law": (
                    "lane_mpp_exact"
                    if self.ast_fast_tree
                    else "row_exact_valid_rows_window_v1"
                    if self.ast_chain
                    else "historical"
                ),
                "tree_tile_rows": self.ast_tree_tile_rows,
                "sibling_cache_sharing": False,
                "target_timings": (
                    json.loads(self.ast_oracle_document).get("target_timings", [])
                    if self.ast_oracle_document is not None
                    else []
                ),
                "numerical_oracle": (
                    self.ast_oracle_path
                    if self.ast_oracle_document is not None
                    else None
                    if self.ast_numerical_oracle is None
                    else self.ast_numerical_oracle.evidence
                ),
            },
            "gdn_backend": gdn_backend_stats(),
            "glue_backend": glue_backend_stats(),
        }
