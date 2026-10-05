"""Default-off live-row compaction for padded prefill MLP-family blocks.

The scheduler may retain a rectangular token slab while an adapter-owned model
publishes the true row lengths for that one forward. Adapter-selected dense
MLPs or complete sparse-MoE blocks gather the live token rows, execute their
existing math unchanged, and scatter the results back to the rectangular
residual shape. Attention, recurrence, decode and cache ownership are outside
this mechanism.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import ClassVar

_SCOPE = ContextVar("mlx2_varlen_dense_mlp_scope", default=None)


@dataclass(frozen=True)
class _VarlenMLPPolicy:
    field_name: ClassVar[str] = "varlen_mlp"
    enabled: bool = False
    minimum_padding_rows: int = 1
    minimum_padding_fraction: float = 0.0

    @classmethod
    def from_value(cls, value):
        if value is None or value is False:
            return cls()
        if value is True:
            return cls(enabled=True)
        if not isinstance(value, Mapping):
            raise TypeError(f"{cls.field_name} must be boolean or an object")
        unknown = set(value) - {
            "enabled",
            "minimum_padding_rows",
            "minimum_padding_fraction",
        }
        if unknown:
            raise ValueError(
                f"unknown {cls.field_name} settings: {sorted(unknown)}"
            )
        enabled = value.get("enabled", True)
        rows = value.get("minimum_padding_rows", 1)
        fraction = value.get("minimum_padding_fraction", 0.0)
        if type(enabled) is not bool:
            raise TypeError(f"{cls.field_name} enabled must be boolean")
        if type(rows) is not int or rows < 1:
            raise ValueError("minimum_padding_rows must be a positive integer")
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not 0.0 <= float(fraction) <= 1.0
        ):
            raise ValueError("minimum_padding_fraction must be between 0 and 1")
        return cls(enabled, rows, float(fraction))

    def as_dict(self):
        return asdict(self)


class VarlenDenseMLPPolicy(_VarlenMLPPolicy):
    field_name = "varlen_dense_mlp"


class VarlenSparseMoEPolicy(_VarlenMLPPolicy):
    field_name = "varlen_sparse_moe"


@dataclass(frozen=True)
class _LiveRows:
    batch: int
    width: int
    live_rows: int
    padding_rows: int
    indices: object
    counters: Counter
    operation: str
    counter_prefix: str


@dataclass(frozen=True)
class CompactedRows:
    values: object
    _scope: _LiveRows

    def restore(self, values):
        """Scatter live MLP outputs into the original rectangular shape."""
        import mlx.core as mx

        if values.ndim != 3 or values.shape[:2] != (1, self._scope.live_rows):
            raise ValueError("compacted MLP output geometry differs")
        features = int(values.shape[-1])
        restored = mx.zeros(
            (self._scope.batch * self._scope.width, features), dtype=values.dtype
        )
        restored[self._scope.indices] = values.reshape(self._scope.live_rows, features)
        self._scope.counters[f"{self._scope.counter_prefix}_scatter_calls"] += 1
        return restored.reshape(self._scope.batch, self._scope.width, features)


def install(model, policy: _VarlenMLPPolicy):
    variants = {
        VarlenDenseMLPPolicy: (
            "mlx2.varlen-dense-mlp.v1",
            "gather-live-existing-mlp-scatter-zero-padding",
            "dense_mlp",
            "mlp",
            "_varlen_dense_mlp",
        ),
        VarlenSparseMoEPolicy: (
            "mlx2.varlen-sparse-moe.v1",
            "gather-live-existing-sparse-moe-scatter-zero-padding",
            "sparse_moe",
            "moe",
            "_varlen_sparse_moe",
        ),
    }
    variant = variants.get(type(policy))
    if variant is None:
        raise TypeError("validated varlen MLP policy required")
    if not policy.enabled:
        return None
    if not callable(getattr(model, "prefill_row_context", None)):
        raise TypeError("model does not declare a prefill row-length context")
    schema, law, operation, counter_prefix, model_attribute = variant
    handle = {
        "schema": schema,
        "law": law,
        "operation": operation,
        "counter_prefix": counter_prefix,
        "policy": policy.as_dict(),
        "counters": Counter(),
    }
    object.__setattr__(model, model_attribute, handle)
    return handle


def identity(handle):
    if handle is None:
        return None
    return {
        "schema": handle["schema"],
        "law": handle["law"],
        "policy": dict(handle["policy"]),
    }


def status(handle):
    if handle is None:
        return None
    counters = dict(handle["counters"])
    return {
        **identity(handle),
        "selected": True,
        "observed_used": counters.get(
            f"{handle['counter_prefix']}_compaction_calls", 0
        )
        > 0,
        "qualified": False,
        "counters": counters,
    }


@contextmanager
def _active_scope(lengths, width, handle):
    if handle is None:
        yield
        return
    if type(width) is not int or width < 1:
        raise ValueError("prefill row width must be a positive integer")
    if not isinstance(lengths, (list, tuple)) or not lengths:
        raise ValueError("prefill row lengths must be a nonempty host sequence")
    lengths = tuple(lengths)
    if any(type(value) is not int or not 0 <= value <= width for value in lengths):
        raise ValueError("prefill row lengths must be integers within the slab")
    counters = handle["counters"]
    batch = len(lengths)
    rectangular_rows = batch * width
    live_rows = sum(lengths)
    padding_rows = rectangular_rows - live_rows
    counters["prefill_scopes"] += 1
    counters["logical_token_rows"] += live_rows
    counters["rectangular_token_rows"] += rectangular_rows
    counters["padding_token_rows"] += padding_rows
    policy = handle["policy"]
    fraction = padding_rows / rectangular_rows
    if padding_rows == 0:
        counters["unpadded_declines"] += 1
        yield
        return
    if (
        padding_rows < policy["minimum_padding_rows"]
        or fraction < policy["minimum_padding_fraction"]
    ):
        counters["policy_declines"] += 1
        yield
        return
    if live_rows == 0:
        raise RuntimeError("a prefill slab cannot contain only padding")
    import mlx.core as mx

    indices = mx.array(
        [
            row * width + column
            for row, length in enumerate(lengths)
            for column in range(length)
        ],
        dtype=mx.int32,
    )
    counters["selected_scopes"] += 1
    token = _SCOPE.set(
        _LiveRows(
            batch,
            width,
            live_rows,
            padding_rows,
            indices,
            counters,
            handle["operation"],
            handle["counter_prefix"],
        )
    )
    try:
        yield
    finally:
        _SCOPE.reset(token)


def prefill_row_context(model, lengths, *, width):
    handles = tuple(
        handle
        for handle in (
            getattr(model, "_varlen_dense_mlp", None),
            getattr(model, "_varlen_sparse_moe", None),
        )
        if handle is not None
    )
    if len(handles) > 1:
        raise RuntimeError("multiple varlen MLP-family policies are installed")
    handle = handles[0] if handles else None
    if handle is None:
        return nullcontext()
    return _active_scope(lengths, width, handle)


def compact_rows(values, *, operation="dense_mlp"):
    """Return live rows when ``values`` matches the active padded slab."""
    scope = _SCOPE.get()
    if scope is None or scope.operation != operation:
        return None
    if values.ndim != 3 or values.shape[:2] != (scope.batch, scope.width):
        return None
    features = int(values.shape[-1])
    packed = values.reshape(scope.batch * scope.width, features)[scope.indices]
    prefix = scope.counter_prefix
    scope.counters[f"{prefix}_compaction_calls"] += 1
    scope.counters[f"{prefix}_live_rows"] += scope.live_rows
    scope.counters[f"{prefix}_padding_rows_skipped"] += scope.padding_rows
    return CompactedRows(packed.reshape(1, scope.live_rows, features), scope)


__all__ = [
    "CompactedRows",
    "VarlenDenseMLPPolicy",
    "VarlenSparseMoEPolicy",
    "compact_rows",
    "identity",
    "install",
    "prefill_row_context",
    "status",
]
