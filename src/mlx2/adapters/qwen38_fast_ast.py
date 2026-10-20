"""Explicit MPP TensorFold law and its ordinary B1 fallback.

The fast target is not the native REV projection law.  Only the existing
grouped MPP exact installer selects its arithmetic, once before any cache.
"""

from __future__ import annotations

import copy
import hashlib
import json

LAW = "lane_mpp_exact"
MAX_ROWS = 16


class Qwen38FastAST:
    def __init__(self, model):
        from ..runtime.progressive_external_verify import (
            PROGRESSIVE_TARGET_PROTOCOL_VERSION,
        )

        if getattr(model, "_mlx2_qwen_hybrid_row_exact", None) is not None:
            raise ValueError("fast TensorFold cannot share the native REV target")
        self.model = model
        self._identity = None
        object.__setattr__(model, "_mlx2_qwen38_fast_ast", self)
        object.__setattr__(
            model,
            "progressive_external_verify_protocol",
            PROGRESSIVE_TARGET_PROTOCOL_VERSION,
        )
        object.__setattr__(model, "progressive_external_target_rows", self.execute)
        object.__setattr__(
            model, "progressive_external_exact_tile_widths", tuple(range(1, MAX_ROWS))
        )
        object.__setattr__(
            model,
            "progressive_external_exact_terminal_input_widths",
            tuple(range(2, MAX_ROWS + 1)),
        )
        object.__setattr__(model, "progressive_external_exact_max_lanes", 1)
        object.__setattr__(model, "progressive_external_private_request_dispatch", True)
        object.__setattr__(model, "progressive_external_compact_exact_rows", False)
        object.__setattr__(
            model, "progressive_external_alternative_row_capacity", MAX_ROWS
        )

    @staticmethod
    def validate_policy(policy):
        minimum = policy.get("min_rows", {})
        if (
            policy.get("mode") != "exact"
            or policy.get("grouping") is not True
            or type(policy.get("max_rows")) is not int
            or policy["max_rows"] < MAX_ROWS
            or not isinstance(minimum, dict)
            or not minimum
        ):
            raise ValueError("fast TensorFold requires grouped exact lane matmul")
        # apply_policy replaces all thresholds with one in explicit exact mode.

    def bind(self, receipt):
        """Cold immutable numerical identity after the real lane installer."""
        from ..runtime.lane import installer

        minimum = receipt.get("min_rows")
        minimums = list(minimum.values()) if isinstance(minimum, dict) else [minimum]
        if (
            receipt.get("backend") != "mpp"
            or receipt.get("available") is not True
            or not receipt.get("covered")
            or not receipt.get("groups")
            or not isinstance(receipt.get("law_id"), str)
            or not receipt["law_id"].endswith("+grouped")
            or not minimums
            or any(type(value) is not int or value != 1 for value in minimums)
            or type(receipt.get("max_rows")) is not int
            or receipt["max_rows"] < MAX_ROWS
        ):
            raise ValueError(
                "fast TensorFold requires an installed grouped MPP exact law"
            )
        modules = []
        groups = {}
        for name, module in self.model.named_modules():
            prepared = installer.installed_lane(module)
            if prepared is None:
                continue
            if (
                prepared.backend != "mpp"
                or getattr(module, "_lane_min_rows", None) != 1
                or getattr(module, "_lane_max_rows", 0) < MAX_ROWS
            ):
                raise ValueError(
                    "fast TensorFold found a foreign installed projection law"
                )
            group = module.__dict__.get("_lane_group")
            if group is not None:
                groups.setdefault(id(group), []).append(name)
            modules.append(
                (
                    name,
                    prepared.format,
                    tuple(module.weight.shape),
                    int(module._lane_max_rows),
                )
            )
        if not modules or not groups:
            raise ValueError("fast TensorFold has no installed grouped projections")
        identity = {
            "projection_law": LAW,
            "lane_law_id": receipt["law_id"],
            "backend": "mpp",
            "maximum_tree_rows": MAX_ROWS,
            "modules": modules,
            "groups": sorted(sorted(names) for names in groups.values()),
        }
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        sealed = json.loads(canonical)
        sealed["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
        if self._identity is not None and self._identity != sealed:
            raise ValueError(
                "fast TensorFold numerical law cannot change in a live model"
            )
        self._identity = sealed
        self.require_bound()
        return self.identity()

    def require_bound(self):
        from ..runtime import row_exact_verify as REV
        from ..runtime.lane import installer

        if (
            self._identity is None
            or getattr(self.model, "_mlx2_qwen38_fast_ast", None) is not self
            or not installer.ENABLED[0]
            or not installer.GROUPING[0]
            or REV.current() is not None
        ):
            raise ValueError("fast TensorFold numerical law is absent or changed")

    def identity(self):
        self.require_bound()
        return copy.deepcopy(self._identity)

    def execute(self, tokens, caches, capture_layers):
        """Exact ordinary fallback, retaining the installed fast projection law.

        A literal B1 step is the reference. Recurrent rollback records append
        across these steps under the caller's existing HybridVerifyRows epoch.
        No REV context, replacement projection or independent cache is created.
        """
        import mlx.core as mx

        from ..runtime.progressive_external_verify import (
            PROGRESSIVE_TARGET_PROTOCOL_VERSION,
            ProgressiveTargetRows,
        )

        self.require_bound()
        if len(tokens.shape) != 2 or int(tokens.shape[0]) != 1:
            raise ValueError("fast TensorFold ordinary fallback is B1-only")
        width = int(tokens.shape[1])
        if not 1 <= width <= MAX_ROWS:
            raise ValueError("fast TensorFold ordinary fallback exceeds its row bound")
        logits, features = [], []
        for position in range(width):
            output, taps = self.model.forward_with_taps(
                tokens[:, position : position + 1], caches, capture_layers
            )
            logits.append(output)
            features.append(taps)
        return ProgressiveTargetRows(
            mx.concatenate(logits, axis=1),
            mx.concatenate(features, axis=1),
            PROGRESSIVE_TARGET_PROTOCOL_VERSION,
            True,
            {
                "schema": "mlx2.qwen38-fast-lane-ordinary.v1",
                "algorithm": "ordinary_serial_b1_same_mpp_law",
                "rows": width,
                "physical_width": 1,
                "token_width": width,
            },
        )
