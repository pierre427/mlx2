# SPDX-License-Identifier: Apache-2.0
"""Adapter-owned exact native self-MTP row limits.

The native self-MTP scheduler may choose how many tokens to propose, but only
the model adapter knows the widest verification window whose complete tensor
path is exact and the widest window every state plane can roll back exactly.
Adapters opt in by declaring both
``max_exact_self_mtp_verification_rows`` and
``max_exact_self_mtp_rollback_rows`` on their concrete class. A missing
declaration keeps the historical runtime unchanged; a partial or malformed
declaration fails closed.

This contract intentionally does not describe external-draft routes. Those
routes own separate target/draft geometry and policy receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

EXACT_SELF_MTP_ROWS_SCHEMA = "mlx2.exact-self-mtp-rows.v1"
_FIELDS = (
    "max_exact_self_mtp_verification_rows",
    "max_exact_self_mtp_rollback_rows",
)


@dataclass(frozen=True)
class ExactSelfMTPRows:
    max_exact_self_mtp_verification_rows: int
    max_exact_self_mtp_rollback_rows: int

    def __post_init__(self) -> None:
        for name in _FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(
                    f"adapter {name} must be an integer >= 2, got {value!r}"
                )

    @property
    def effective_max_self_mtp_rows(self) -> int:
        return min(
            self.max_exact_self_mtp_verification_rows,
            self.max_exact_self_mtp_rollback_rows,
        )

    @property
    def effective_max_self_mtp_proposer_depth(self) -> int:
        # One verify row is the pending target token; the rest are proposals.
        return self.effective_max_self_mtp_rows - 1

    def receipt(
        self,
        *,
        requested_self_mtp_num_draft: int,
        effective_self_mtp_num_draft: int,
        requested_self_mtp_copy_max_span: int | None,
        effective_self_mtp_copy_max_span: int | None,
    ) -> dict:
        return {
            "schema": EXACT_SELF_MTP_ROWS_SCHEMA,
            "max_exact_self_mtp_verification_rows": (
                self.max_exact_self_mtp_verification_rows
            ),
            "max_exact_self_mtp_rollback_rows": (
                self.max_exact_self_mtp_rollback_rows
            ),
            "effective_max_self_mtp_rows": self.effective_max_self_mtp_rows,
            "effective_max_self_mtp_proposer_depth": (
                self.effective_max_self_mtp_proposer_depth
            ),
            "requested_self_mtp_num_draft": int(requested_self_mtp_num_draft),
            "effective_self_mtp_num_draft": int(effective_self_mtp_num_draft),
            "requested_self_mtp_copy_max_span": requested_self_mtp_copy_max_span,
            "effective_self_mtp_copy_max_span": effective_self_mtp_copy_max_span,
            "clamped": (
                int(requested_self_mtp_num_draft)
                != int(effective_self_mtp_num_draft)
                or requested_self_mtp_copy_max_span
                != effective_self_mtp_copy_max_span
            ),
        }


def declared_exact_self_mtp_rows(adapter: Any) -> ExactSelfMTPRows | None:
    """Read a concrete adapter's paired native self-MTP declaration.

    The concrete class must own the declaration. This prevents a sibling model
    family from silently inheriting limits established for another artifact
    geometry.
    """

    declarations = vars(type(adapter))
    present = tuple(name in declarations for name in _FIELDS)
    if not any(present):
        return None
    if not all(present):
        missing = [name for name, found in zip(_FIELDS, present) if not found]
        raise ValueError(
            "adapter exact self-MTP row capability must declare both "
            f"{_FIELDS[0]} and {_FIELDS[1]}; missing {missing}"
        )
    return ExactSelfMTPRows(**{name: declarations[name] for name in _FIELDS})


def constrain_self_mtp_proposers(
    capabilities: ExactSelfMTPRows | None,
    *,
    self_mtp_num_draft: int,
    self_mtp_copy_draft_policy: Any,
) -> tuple[int, Any, dict | None]:
    """Apply a declared row contract to native self-MTP proposer widths."""

    if capabilities is None:
        return self_mtp_num_draft, self_mtp_copy_draft_policy, None
    if (
        isinstance(self_mtp_num_draft, bool)
        or not isinstance(self_mtp_num_draft, int)
        or self_mtp_num_draft < 1
    ):
        raise ValueError("self-MTP num_draft must be an integer >= 1")
    maximum = capabilities.effective_max_self_mtp_proposer_depth
    effective_self_mtp_num_draft = min(self_mtp_num_draft, maximum)
    requested_self_mtp_copy_max_span = (
        self_mtp_copy_draft_policy.span_ceiling
        if self_mtp_copy_draft_policy.enabled
        else None
    )
    effective_self_mtp_copy_policy = (
        self_mtp_copy_draft_policy.clamped_to_self_mtp_proposer_depth(maximum)
        if self_mtp_copy_draft_policy.enabled
        else self_mtp_copy_draft_policy
    )
    effective_self_mtp_copy_max_span = (
        effective_self_mtp_copy_policy.span_ceiling
        if effective_self_mtp_copy_policy.enabled
        else None
    )
    return (
        effective_self_mtp_num_draft,
        effective_self_mtp_copy_policy,
        capabilities.receipt(
            requested_self_mtp_num_draft=self_mtp_num_draft,
            effective_self_mtp_num_draft=effective_self_mtp_num_draft,
            requested_self_mtp_copy_max_span=requested_self_mtp_copy_max_span,
            effective_self_mtp_copy_max_span=effective_self_mtp_copy_max_span,
        ),
    )


def constrain_self_mtp_draft_loop(
    capabilities: ExactSelfMTPRows | None,
    receipt: dict | None,
    config: Any,
) -> dict | None:
    """Hold a self-MTP draft loop inside the declared exact row window.

    ``constrain_self_mtp_proposers`` sizes ``num_draft`` and the copy span;
    a ``draft_loop`` in the execution policy lets a lane draft past
    ``num_draft`` to the loop's ceiling, and the verify forward then carries
    ceiling + 1 rows.  Fails closed when that exceeds the window (the tensor
    path and state rollback past it are not exact), otherwise returns the
    receipt extended with the ceiling the route really verifies.
    """
    if capabilities is None:
        return receipt
    from ..runtime.draft_loop import draft_depth_ceiling

    ceiling = draft_depth_ceiling(config)
    verify_rows = ceiling + 1
    if verify_rows > capabilities.effective_max_self_mtp_rows:
        raise ValueError(
            f"self-MTP draft_loop {dict((config or {}).get('draft_loop') or {})} "
            f"lets a lane at num_draft {int((config or {}).get('num_draft') or 0)} "
            f"draft {ceiling} tokens ({verify_rows} verify rows), past the adapter's "
            f"exact self-MTP window of {capabilities.effective_max_self_mtp_rows} "
            "rows; lower the loop's last boundary or declare a wider exact window"
        )
    if receipt is None:
        return None
    # A copy round verifies the copied span (already clamped to the window
    # by constrain_self_mtp_proposers) in place of the head drafts, so the
    # rows the route may verify are the wider of the two.
    copy_span = int(receipt.get("effective_self_mtp_copy_max_span") or 0)
    return {
        **receipt,
        "effective_self_mtp_draft_ceiling": int(ceiling),
        "effective_self_mtp_verify_rows": int(max(ceiling, copy_span) + 1),
    }


__all__ = [
    "EXACT_SELF_MTP_ROWS_SCHEMA",
    "ExactSelfMTPRows",
    "constrain_self_mtp_draft_loop",
    "constrain_self_mtp_proposers",
    "declared_exact_self_mtp_rows",
]
