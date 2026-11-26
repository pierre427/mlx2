"""Shared spec/fast-path dispatch for technology reference implementations."""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field, fields, replace
from typing import Any, Callable, Mapping


def spec_from_config(spec_cls, config: Any, mapping: Mapping[str, Any], **overrides):
    """Build ``spec_cls`` from a config object or dict.

    ``mapping`` maps a spec field to either a config key (str), a tuple of
    alternative keys tried in order, or a callable ``config -> value``.
    Missing keys leave the spec default in place; ``overrides`` win over
    everything and must name real spec fields.
    """
    names = {f.name for f in fields(spec_cls)}
    unknown = set(overrides) - names
    if unknown:
        raise TypeError(f"{spec_cls.__name__} has no field(s) {sorted(unknown)}")

    def get(key):
        if isinstance(config, Mapping):
            return config.get(key, _MISSING)
        return getattr(config, key, _MISSING)

    values: dict[str, Any] = {}
    for name, source in mapping.items():
        if name not in names:
            raise TypeError(f"{spec_cls.__name__} has no field {name!r}")
        if callable(source):
            value = source(config)
        else:
            keys = (source,) if isinstance(source, str) else tuple(source)
            value = _MISSING
            for key in keys:
                value = get(key)
                if value is not _MISSING and value is not None:
                    break
        if value is not _MISSING and value is not None:
            values[name] = value
    values.update(overrides)
    return spec_cls(**values)


_MISSING = object()


@dataclass(frozen=True)
class FastPath:
    """A registered fast implementation of a technology operation.

    ``admits(spec, **shape) -> str | None`` returns ``None`` when the path can
    run and a refusal reason otherwise.
    """

    name: str
    admits: Callable[..., str | None]
    run: Callable[..., Any]


@dataclass
class Receipt:
    operation: str
    selected: str
    refusals: dict[str, str] = field(default_factory=dict)


_COUNTS_LOCK = threading.Lock()
# One mutable cell per counter key, so the hot path increments a cached cell
# instead of hashing a key tuple into a Counter.
_SELECTIONS: dict[tuple[str, str, str], list[int]] = {}  # (technology, operation, path)
_REFUSALS: dict[tuple[str, str, str, str], list[int]] = {}  # (+ code)
_OUTCOMES: dict = {}  # (technology, operation, selected, refusals) -> cells
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_CODE_RE = re.compile(r"[^a-z0-9_]+")
_CODE_CACHE: dict[str, str] = {}


def refusal_code(reason: str) -> str:
    """Bounded metric label for a refusal: the text before the first ':'.

    Fast paths write refusals as ``"<snake_case_code>: <free detail>"``; only
    the code reaches Prometheus labels, the detail stays in receipts.
    """
    prefix = str(reason).partition(":")[0]
    code = _CODE_CACHE.get(prefix)
    if code is None:
        code = _CODE_RE.sub("_", prefix.strip().lower()).strip("_")[:48] or "unspecified"
        if len(_CODE_CACHE) < 4096:  # codes are a small fixed set per path
            _CODE_CACHE[prefix] = code
    return code


def _cells(technology, operation, selected, refusals):
    sel = _SELECTIONS.setdefault((technology, operation, selected), [0])
    refused = tuple(
        _REFUSALS.setdefault((technology, operation, path, refusal_code(reason)), [0])
        for path, reason in refusals
    )
    return (sel,) + refused


def _record(technology: str, operation: str, receipt: "Receipt") -> None:
    # Steady-state decode repeats the same outcome (selection and refusal
    # texts), so its counter cells are resolved once and reused.
    refusals = receipt.refusals
    outcome = (technology, operation, receipt.selected,
               tuple(refusals.items()) if refusals else ())
    cells = _OUTCOMES.get(outcome)
    with _COUNTS_LOCK:
        if cells is None:
            cells = _cells(*outcome)
            if len(_OUTCOMES) < 8192:
                _OUTCOMES[outcome] = cells
        for cell in cells:
            cell[0] += 1


def counters_snapshot() -> dict:
    """Process-wide dispatch counts, shaped for route diagnostics."""
    with _COUNTS_LOCK:
        selections = sorted((k, c[0]) for k, c in _SELECTIONS.items() if c[0])
        refusals = sorted((k, c[0]) for k, c in _REFUSALS.items() if c[0])
    return {
        "selections": [
            {"technology": t, "operation": o, "path": p, "count": n}
            for (t, o, p), n in selections
        ],
        "refusals": [
            {"technology": t, "operation": o, "path": p, "code": c, "count": n}
            for (t, o, p, c), n in refusals
        ],
    }


def reset_counters() -> None:
    with _COUNTS_LOCK:
        for cell in _SELECTIONS.values():
            cell[0] = 0
        for cell in _REFUSALS.values():
            cell[0] = 0


class KernelRegistry:
    """Ordered fast paths for one operation, with the reference last.

    ``operation`` is ``"<technology>.<op>"`` in snake_case (for example
    ``"mla.decode"``); the two halves become Prometheus labels.
    """

    def __init__(self, operation: str, reference: Callable[..., Any]):
        technology, _, op = operation.partition(".")
        if not (_NAME_RE.match(technology) and _NAME_RE.match(op)):
            raise ValueError(f"operation must be '<technology>.<op>' in snake_case, got {operation!r}")
        self.technology = technology
        self.op = op
        self.operation = operation
        self.reference = reference
        self._paths: list[FastPath] = []

    def register(self, path: FastPath) -> FastPath:
        if any(p.name == path.name for p in self._paths):
            raise ValueError(f"{self.operation}: fast path {path.name!r} already registered")
        self._paths.append(path)
        return path

    def select(self, spec, *, policy: str | None = None, **shape):
        fn, receipt = self._select(spec, policy=policy, **shape)
        _record(self.technology, self.op, receipt)
        return fn, receipt

    def _select(self, spec, *, policy: str | None = None, **shape):
        """Return ``(callable, receipt)``.

        ``policy`` names a required path: ``"reference"`` forces the
        reference; a fast-path name fails closed if that path refuses.
        ``None`` takes the first admitting fast path, else the reference.
        """
        refusals: dict[str, str] = {}
        if policy == "reference":
            return self.reference, Receipt(self.operation, "reference", refusals)
        for path in self._paths:
            if policy is not None and path.name != policy:
                continue
            reason = path.admits(spec, **shape)
            if reason is None:
                return path.run, Receipt(self.operation, path.name, refusals)
            refusals[path.name] = reason
            if policy is not None:
                _record(self.technology, self.op, Receipt(self.operation, "failed_closed", refusals))
                raise RuntimeError(
                    f"{self.operation}: required fast path {policy!r} refused: {reason}"
                )
        if policy is not None:
            # Operator input must never become an unbounded metric label.
            _record(self.technology, self.op, Receipt(
                self.operation, "failed_closed", {"unknown": "unknown_path"}))
            raise RuntimeError(f"{self.operation}: unknown fast path {policy!r}")
        return self.reference, Receipt(self.operation, "reference", refusals)


__all__ = [
    "FastPath", "KernelRegistry", "Receipt", "counters_snapshot", "refusal_code",
    "replace", "reset_counters", "spec_from_config",
]
